//! Provider, to be: the items around the player it could pick up, and their taking, for a Monaka
//! ready-made that picks things up by walking to them (today
//! [`auto_pickup`](crate::cooking::auto_pickup)).
//!
//! After each of the player's thinks, the entities around it are read from the gather asked for at
//! an earlier think, and the next gather asked for (`engine::Pickups`: the game gathers so for the
//! player only in multiplayer): [`items`], each item on the ground the player could use now, with
//! its bounds and what its inventory holds. An item is taken by the use button's own use
//! ([`take`]), as if it were in focus.

use crate::providers::engine::{self, AbsBoundsFn, Entities, FitsFn, GatheredFn, InventoryFn, Pickups, TypeFn, TypeTestFn, UsableFn, UseFn, GatherFn, PlayerThinkFn};
use monaka_hook::module::Module;
use monaka_hook::{Hooks, InFlight, Original, mem};
use monaka_producer::game::Lack;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering::*};

pub static THINK_ORIGINAL: Original<PlayerThinkFn> = Original::new();

/// The key of the last gather asked for (0: none).
static KEY: AtomicU64 = AtomicU64::new(0);
/// Gathers read, and the entities they held.
static READ: AtomicU64 = AtomicU64::new(0);
static GATHERED_SEEN: AtomicU64 = AtomicU64::new(0);
/// The gather's label in the clip world's list (a C string).
const LABEL: &[u8] = b"monaka: pickups\0";

/// The executable's base, this build's pickup facts and its entities, from the install.
static FACTS: OnceLock<(usize, &'static Pickups, Entities)> = OnceLock::new();

/// The soldiers' armour pieces (`idProp_AIArmor`) found in the entity table: (number, entity),
/// looked for again every [`ARMOR_SCAN_THINKS`] thinks.
static ARMOR_PIECES: std::sync::Mutex<Vec<(usize, usize)>> = std::sync::Mutex::new(Vec::new());
static ARMOR_SCANS: AtomicU64 = AtomicU64::new(0);
static ARMOR_LOGGED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
const ARMOR_SCAN_THINKS: u64 = 30;
/// Half the size of an armour piece's box where its physics gives no bounds (units).
const ARMOR_PIECE_HALF: f32 = 8.0;

/// The most gathered entities read in one think, and the most items in one inventory.
const GATHERED: usize = 1024;
const MOST_ITEMS: i32 = 64;

/// An item on the ground the player could use now.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Item {
    pub entity: usize,
    /// Its absolute bounds: min x, y, z, then max.
    pub bounds: [f32; 6],
    pub holds: Holds,
}

/// What an item's inventory holds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Holds {
    /// Ammo and guns only (what the use prompt shows as ammo: a gun the player has gives its ammo),
    /// none of them a gun carried in both hands; whether any of it fits in the player's inventory.
    Ammo { fits: bool },
    /// Health only: how much it gives in all.
    Health { amount: f32 },
    /// Armour only: the most armour its items allow (0 when not read).
    Armor { max: f32 },
    /// Anything else (a carried gun, a note, a mix), or nothing.
    Other,
}

/// The player's health and armour, each `(current, most)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vitals {
    pub health: (f32, f32),
    pub armor: (f32, f32),
}

/// The player's (`idPlayer`) health and armour, if they read as such.
pub fn vitals(player: usize) -> Option<Vitals> {
    let at = player + engine::PLAYER_HEALTH_COMPONENTS;
    let [health, health_max, armor, armor_max] = mem::read::<[f32; 4]>(at)?;
    let sane = |v: f32| v.is_finite() && (0.0..=10_000.0).contains(&v);
    ([health, health_max, armor, armor_max].into_iter().all(sane) && health_max > 0.0).then_some(Vitals { health: (health, health_max), armor: (armor, armor_max) })
}

/// Hooks the player's think: after each, [`auto_pickup::think`](crate::cooking::auto_pickup::think).
pub fn install(hooks: &mut Hooks, exe: &Module, build: &'static engine::Build, pickups: &'static Pickups) -> Result<(), Lack> {
    let _ = FACTS.set((exe.base(), pickups, build.entities));
    hooks.site(&THINK_ORIGINAL, exe, &pickups.think, think)?;
    Ok(())
}

unsafe extern "system" fn think(player: usize) {
    let flight = InFlight::enter();
    // SAFETY: forwards the game's own call.
    unsafe { THINK_ORIGINAL.get(&flight)(player) };
    crate::cooking::auto_pickup::think(player);
}

/// A function of this build at `address` (absolute).
///
/// # Safety
/// `address` must start a function of type `F`.
unsafe fn function<F: Copy>(address: usize) -> F {
    debug_assert_eq!(std::mem::size_of::<F>(), std::mem::size_of::<usize>());
    // SAFETY: the caller's claim; `F` is a function pointer type.
    unsafe { std::mem::transmute_copy(&address) }
}

/// An entity's virtual method at `slot` (an offset into its vtable).
///
/// # Safety
/// `entity` must be a live entity whose vtable has a method of type `F` at `slot`.
unsafe fn method<F: Copy>(entity: usize, slot: usize) -> Option<F> {
    let vtable = mem::read::<usize>(entity).filter(|&v| v != 0)?;
    let address = mem::read::<usize>(vtable + slot).filter(|&a| a != 0)?;
    // SAFETY: the caller's claim.
    Some(unsafe { function(address) })
}

/// Gathers read and the entities they held, for the summary.
pub fn counts() -> (u64, u64) {
    (READ.load(Relaxed), GATHERED_SEEN.load(Relaxed))
}

/// Each item on the ground the player (`idPlayer`, the one whose think it is) could use now, among
/// the entities the last gather found; then the next gather is asked for, of the player's bounds
/// (`own`) grown by `around`. On the game thread, after the player's think.
pub fn items(player: usize, own: &[f32; 6], around: f32, mut each: impl FnMut(Item)) {
    let Some(&(base, pickups, entities)) = FACTS.get() else { return };
    let Some(clip) = mem::read::<usize>(base + pickups.client_game).filter(|&c| c != 0).map(|c| c + pickups.clip) else { return };
    let key = KEY.load(Relaxed);
    if let (Some(oldest), Some(newest)) = (mem::read::<u64>(clip + pickups.keys), mem::read::<u64>(clip + pickups.keys + 8))
        // A gather is read only while its key is in the clip world's window, as the game reads it.
        && key != 0
        && (oldest..=newest).contains(&key)
        && let Some(game) = mem::read::<usize>(base + entities.game_local).filter(|&g| g != 0)
    {
        read(base, pickups, entities, clip, game, key, player, &mut each);
    }
    if let Some(game) = mem::read::<usize>(base + entities.game_local).filter(|&g| g != 0) {
        armor_pieces(base, pickups, entities, game, player, &mut each);
    }
    let bounds: [f32; 6] = std::array::from_fn(|k| if k < 3 { own[k] - around } else { own[k] + around });
    let number = mem::read::<u16>(player + pickups.entity_number).unwrap_or(0);
    let mut asked = 0u64;
    // SAFETY: `engine::Pickups::gather` of this build (its fingerprint checked), as the player's
    // usables update asks it, on the game thread inside the player's think; the label is static.
    let _ = unsafe { function::<GatherFn>(base + pickups.gather)(clip, &mut asked, &bounds, pickups.contents, number, LABEL.as_ptr()) };
    KEY.store(asked, Relaxed);
}

/// The items among the entities of the gather `key`.
#[allow(clippy::too_many_arguments)]
fn read(base: usize, pickups: &Pickups, entities: Entities, clip: usize, game: usize, key: u64, player: usize, each: &mut impl FnMut(Item)) {
    let mut numbers = [0i32; GATHERED];
    // SAFETY: `engine::Pickups::gathered` of this build (its fingerprint checked), given a key in
    // the clip world's window and room for `GATHERED` numbers, on the game thread.
    let count = unsafe { function::<GatheredFn>(base + pickups.gathered)(clip, &key, numbers.as_mut_ptr(), GATHERED as i32) } as usize;
    READ.fetch_add(1, Relaxed);
    GATHERED_SEEN.fetch_add(count.min(GATHERED) as u64, Relaxed);
    for &number in &numbers[..count.min(GATHERED)] {
        if !(0..1 << entities.bits).contains(&number) {
            continue;
        }
        let Some(entity) = mem::read::<usize>(game + entities.entities + number as usize * 8).filter(|&e| e != 0 && e != player) else { continue };
        if let Some(item) = item(base, pickups, entity, player) {
            each(item);
        }
    }
}

/// The soldiers' armour pieces (`idProp_AIArmor`, outside the gather) the player could use now, as
/// armour items; the entity table scanned for them every [`ARMOR_SCAN_THINKS`] thinks.
fn armor_pieces(base: usize, pickups: &Pickups, entities: Entities, game: usize, player: usize, each: &mut impl FnMut(Item)) {
    let Some(vtable) = pickups.ai_armor_vtable.map(|v| base + v) else { return };
    let Ok(mut pieces) = ARMOR_PIECES.try_lock() else { return };
    let at = |number: usize| mem::read::<usize>(game + entities.entities + number * 8).filter(|&e| e != 0);
    if ARMOR_SCANS.fetch_add(1, Relaxed) % ARMOR_SCAN_THINKS == 0 {
        pieces.clear();
        pieces.extend((0..1usize << entities.bits).filter_map(|n| at(n).filter(|&e| mem::read::<usize>(e) == Some(vtable)).map(|e| (n, e))));
    }
    // Each still in its place in the table, as an armour piece.
    pieces.retain(|&(n, e)| at(n) == Some(e) && mem::read::<usize>(e) == Some(vtable));
    for &(_, entity) in pieces.iter() {
        // SAFETY: this build's vtable slots (its fingerprint checked), on a live entity from the
        // game's table, on the game thread inside the player's think.
        let usable = unsafe {
            let ever = method::<UsableFn>(entity, pickups.ever_usable);
            let now = method::<UsableFn>(entity, pickups.usable_now);
            ever.zip(now).is_some_and(|(ever, now)| ever(entity, player) != 0 && now(entity, player) != 0)
        };
        if !ARMOR_LOGGED.swap(true, Relaxed) {
            monaka_producer::log!("auto pickup: soldiers' armor pieces found in the entity table ({}), usable now: {usable}", pieces.len());
        }
        if !usable {
            continue;
        }
        let bounds = bounds(entity).or_else(|| {
            let o = mem::read::<[f32; 3]>(entity + pickups.ai_armor_origin).filter(|o| o.iter().all(|v| v.is_finite()))?;
            Some(std::array::from_fn(|k| if k < 3 { o[k] - ARMOR_PIECE_HALF } else { o[k - 3] + ARMOR_PIECE_HALF }))
        });
        if let Some(bounds) = bounds {
            each(Item { entity, bounds, holds: Holds::Armor { max: 0.0 } });
        }
    }
}

/// `entity` as an item on the ground the player could use now, asked as the usables update asks.
fn item(base: usize, pickups: &Pickups, entity: usize, player: usize) -> Option<Item> {
    // SAFETY (each call below): this build's functions and vtable slots (its fingerprint checked),
    // on a live entity from the game's table, on the game thread inside the player's think.
    let is_moveable = unsafe { function::<TypeTestFn>(base + pickups.is_moveable) };
    // Armour pieces are read apart ([`armor_pieces`]).
    if unsafe { is_moveable(entity) } == 0 || pickups.ai_armor_vtable.is_some_and(|v| mem::read::<usize>(entity) == Some(base + v)) {
        return None;
    }
    let ever = unsafe { method::<UsableFn>(entity, pickups.ever_usable)? };
    let now = unsafe { method::<UsableFn>(entity, pickups.usable_now)? };
    if unsafe { ever(entity, player) } == 0 || unsafe { now(entity, player) } == 0 {
        return None;
    }
    let bounds = bounds(entity)?;
    let (items, count) = inventory(pickups, entity)?;
    let is_ammo = unsafe { function::<TypeTestFn>(base + pickups.is_ammo) };
    let is_weapon = unsafe { function::<TypeTestFn>(base + pickups.is_weapon) };
    let (mut ammo, mut any_fits) = (0, false);
    let (mut healths, mut heals, mut armors, mut armor_max) = (0, 0.0f32, 0, 0.0f32);
    for i in 0..count {
        let Some(it) = mem::read::<usize>(items + i * 8).filter(|&it| it != 0) else { continue };
        let decl = || mem::read::<usize>(it + engine::INVENTORY_ITEM_DECL).filter(|&d| d != 0);
        match type_name(it).as_deref() {
            Some(engine::HEALTH_ITEM_TYPE) => {
                let amount = decl().and_then(|d| mem::read::<f32>(d + engine::HEALTH_ITEM_AMOUNT)).filter(|a| a.is_finite() && *a > 0.0);
                let times = mem::read::<i32>(it + engine::INVENTORY_ITEM_COUNT).unwrap_or(1).clamp(1, 16);
                if let Some(amount) = amount {
                    healths += 1;
                    heals += amount * times as f32;
                }
                continue;
            }
            Some(engine::ARMOR_ITEM_TYPE) => {
                armors += 1;
                let most = decl().and_then(|d| mem::read::<f32>(d + engine::ARMOR_ITEM_MAX)).filter(|m| m.is_finite() && *m > 0.0);
                armor_max = armor_max.max(most.unwrap_or(0.0));
                continue;
            }
            _ => {}
        }
        let gun = unsafe { is_weapon(it) } != 0;
        let carried = gun && mem::read::<usize>(it + engine::INVENTORY_ITEM_DECL).filter(|&d| d != 0).and_then(|d| mem::read::<u8>(d + pickups.carried_gun)).is_none_or(|c| c != 0);
        if (unsafe { is_ammo(it) } != 0 || gun) && !carried {
            ammo += 1;
            let Some(fits) = (unsafe { method::<FitsFn>(it, pickups.fits) }) else { continue };
            any_fits |= unsafe { fits(it, player, 0) } != 0;
        }
    }
    let holds = match () {
        _ if count > 0 && ammo == count => Holds::Ammo { fits: any_fits },
        _ if count > 0 && healths == count => Holds::Health { amount: heals },
        _ if count > 0 && armors == count => Holds::Armor { max: armor_max },
        _ => Holds::Other,
    };
    Some(Item { entity, bounds, holds })
}

/// An entity's inventory: its items (pointers) and their count.
fn inventory(pickups: &Pickups, entity: usize) -> Option<(usize, usize)> {
    // SAFETY: this build's vtable slot (its fingerprint checked), on an `idProp_Moveable` from the
    // game's table, on the game thread.
    let inventory = unsafe { method::<InventoryFn>(entity, pickups.inventory)? };
    let list = Some(unsafe { inventory(entity) }).filter(|&l| l != 0)?;
    let items = mem::read::<usize>(list)?;
    let count = mem::read::<i32>(list + engine::LIST_COUNT)?.clamp(0, MOST_ITEMS) as usize;
    Some((items, count))
}

/// An item's type and its inventory's by type, for the summary (`idProp_Moveable of idHealthItem`).
pub fn described(entity: usize) -> String {
    let own = type_name(entity).unwrap_or_else(|| "?".to_owned());
    let Some((items, count)) = FACTS.get().and_then(|&(_, pickups, _)| inventory(pickups, entity)) else { return own };
    let held: Vec<String> = (0..count).filter_map(|i| mem::read::<usize>(items + i * 8).filter(|&it| it != 0)).map(|it| type_name(it).unwrap_or_else(|| "?".to_owned())).collect();
    if held.is_empty() { own } else { format!("{own} of {}", held.join(" + ")) }
}

/// An entity's absolute bounds (the player's too).
pub fn bounds(entity: usize) -> Option<[f32; 6]> {
    let &(_, pickups, _) = FACTS.get()?;
    let physics = mem::read::<usize>(entity + pickups.physics).filter(|&p| p != 0)?;
    // SAFETY: this build's physics vtable slot (its fingerprint checked), on a live entity's
    // physics, on the game thread.
    let abs_bounds = unsafe { method::<AbsBoundsFn>(physics, pickups.abs_bounds)? };
    let at = unsafe { abs_bounds(physics, -1) };
    mem::read::<[f32; 6]>(at as usize)
}

/// Takes `entity` (an [`Item`] just found) by the use button's use: whether the game took it.
pub fn take(entity: usize, player: usize) -> bool {
    let Some(&(_, pickups, _)) = FACTS.get() else { return false };
    // SAFETY: this build's vtable slot (its fingerprint checked), on an item [`items`] just found, by
    // the player whose think it is, on the game thread as the use button's use runs.
    let Some(use_it) = (unsafe { method::<UseFn>(entity, pickups.use_it) }) else { return false };
    unsafe { use_it(entity, player, 0) != 0 }
}

/// An entity's type name (`idProp_Moveable`).
pub fn type_name(entity: usize) -> Option<String> {
    // SAFETY: every entity's first virtual method gives its type, on a live entity.
    let get = unsafe { method::<TypeFn>(entity, 0)? };
    let info = Some(unsafe { get(entity) }).filter(|&i| i != 0)?;
    mem::read_c_string(mem::read::<usize>(info + engine::TYPE_NAME)?, 64)
}

/// What the use button can reach from far (`focus`): an interaction the player can use now (a
/// scene's "press X", as on the London Monitor once it kneels) up to [`FAR_USE_METRES`] off and
/// within [`FAR_USE_DEGREES`] of where the player looks, however small its own spot.
pub const FAR_USE_METRES: f32 = 50.0;
pub const FAR_USE_DEGREES: f32 = 35.0;
/// The entity types reached from far: the game's interactions (not its doors, switches or items).
pub const FAR_USE_TYPES: &[&str] = &["idProp_PlayerInteraction", "idVolume_PlayerInteraction", "idAnimatedInteractable", "idTrigger_AIInteraction", "idAIInteraction", "idInteractiveMarker", "idRobot"];
/// The interactions in the entity table: (number, entity), looked for again every
/// [`ARMOR_SCAN_THINKS`] updates.
static FAR: std::sync::Mutex<Vec<(usize, usize)>> = std::sync::Mutex::new(Vec::new());
static FAR_SCANS: AtomicU64 = AtomicU64::new(0);
static FAR_LOGGED: AtomicU64 = AtomicU64::new(0);

/// Whether a thing at `at` is within reach from far of a look from `origin` along unit `forward`
/// (`units` a metre).
pub fn far_reached(origin: [f32; 3], forward: [f32; 3], at: [f32; 3], units: f32) -> Option<f32> {
    let to: [f32; 3] = std::array::from_fn(|k| at[k] - origin[k]);
    let distance = to.iter().map(|v| v * v).sum::<f32>().sqrt();
    if !(distance <= FAR_USE_METRES * units) {
        return None;
    }
    if distance < 1.0 {
        return Some(0.0);
    }
    let cosine = (0..3).map(|k| to[k] * forward[k]).sum::<f32>() / distance;
    (cosine >= FAR_USE_DEGREES.to_radians().cos()).then(|| cosine.clamp(-1.0, 1.0).acos())
}

/// The interaction the player (`idPlayer`) can use now nearest where it looks, from `origin` along
/// unit `forward` ([`far_reached`]), as an entity handle (spawn id over number). On the game thread.
pub fn far_usable(player: usize, origin: [f32; 3], forward: [f32; 3], units: f32) -> Option<u32> {
    let &(base, pickups, entities) = FACTS.get()?;
    let game = mem::read::<usize>(base + entities.game_local).filter(|&g| g != 0)?;
    let at = |n: usize| mem::read::<usize>(game + entities.entities + n * 8).filter(|&e| e != 0);
    let mut far = FAR.try_lock().ok()?;
    let scan = FAR_SCANS.fetch_add(1, Relaxed) % ARMOR_SCAN_THINKS == 0;
    if scan {
        far.clear();
        for n in 0..1usize << entities.bits {
            let Some(e) = at(n) else { continue };
            let Some(name) = type_name(e) else { continue };
            if FAR_USE_TYPES.contains(&name.as_str()) {
                far.push((n, e));
            }
        }
    }
    far.retain(|&(n, e)| at(n) == Some(e));
    let mut best: Option<(f32, usize, usize)> = None;
    for &(n, entity) in far.iter() {
        let Some(b) = bounds(entity) else { continue };
        let middle = std::array::from_fn(|k| 0.5 * (b[k] + b[k + 3]));
        let Some(angle) = far_reached(origin, forward, middle, units) else { continue };
        // SAFETY: this build's vtable slots (its fingerprint checked), on a live entity from the
        // game's table, on the game thread in the player's focus update.
        let (ever, now) = unsafe {
            let ever = method::<UsableFn>(entity, pickups.ever_usable).is_some_and(|f| f(entity, player) != 0);
            let now = method::<UsableFn>(entity, pickups.usable_now).is_some_and(|f| f(entity, player) != 0);
            (ever, now)
        };
        if scan && ever && FAR_LOGGED.fetch_add(1, Relaxed) < 12 {
            monaka_producer::log!("far use: {} {:.0} m off, {:.0} degrees from the look, usable now {now}", type_name(entity).unwrap_or_default(), (0..3).map(|k| (middle[k] - origin[k]).powi(2)).sum::<f32>().sqrt() / units, angle.to_degrees());
        }
        if ever && now && best.is_none_or(|(a, ..)| angle < a) {
            best = Some((angle, n, entity));
        }
    }
    let (_, n, _) = best?;
    handle(game, entities, n)
}

/// Whether the entity handle `handle` is a live entity's.
pub fn handle_live(live: u32) -> bool {
    let Some(&(base, _, entities)) = FACTS.get() else { return false };
    let Some(game) = mem::read::<usize>(base + entities.game_local).filter(|&g| g != 0) else { return false };
    let n = (live & ((1 << entities.bits) - 1)) as usize;
    handle(game, entities, n) == Some(live) && mem::read::<usize>(game + entities.entities + n * 8).is_some_and(|e| e != 0)
}

/// The handle of entity number `n` now.
fn handle(game: usize, entities: Entities, n: usize) -> Option<u32> {
    let spawn = mem::read::<u32>(game + entities.spawn_ids + n * 4)?;
    Some((spawn << entities.bits) | n as u32)
}
