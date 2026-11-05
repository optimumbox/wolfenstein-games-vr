//! Provider: melee hits (`monaka_body::melee_hit::Hits`): the hands' melee trace swept along the
//! weapon in the right hand.
//!
//! The game's own melee hit is a trace the hands own (`engine::Melee`): while it is open, each
//! update sweeps a box from where a point was to where it is, and a second one along a blade (from
//! a second point to the first), and damages what they touch, each thing once, by the weapon's
//! melee damage, in the direction the point moved, at the joint it touched. The game opens it from
//! its attack animations; for the knife only as a line straight out of the eye. Here, with a
//! melee weapon in the right hand (the knife; The Old Blood's pipes):
//!
//! - each update feeds the controllers to Monaka's swing detector ([`monaka_body::melee`]); a
//!   swing of the right controller presses nothing (the game's attack would play its canned
//!   animation) and opens the trace at the blade's tip ([`open`]) for as long as the swing lasts:
//!   until the controller's tip (against the player, not the world) slows below the detector's
//!   settle speed;
//! - while it is open, each update sweeps it once ([`sweep`]): the tip from where it was to where
//!   it is, and the blade from the hand to the tip, the hand where the controller holds the weapon
//!   now ([`held_item::gun_target`]), the blade along the weapon's blade axis ([`held_item::blade_row`]),
//!   `melee_hit::blade` long. The game's own update is skipped, so nothing else sweeps it;
//! - while the weapon is held, the right trigger is let go before the game reads the pad
//!   ([`let_go_of_trigger`]), and the game's own starts and ends of the trace do nothing
//!   ([`held_back`]): only swings attack. The quick melee (the melee button, which draws the knife
//!   or the pipe for one attack) stays the game's own.
//!
//! The trace is kept open two updates at a time, so it shuts by itself soon after the producer
//! stops; the component going off shuts it at once. Each swing opens it afresh, so each swing
//! hits each thing once.

use crate::providers::engine::{self, AnimEventFn, HandsLayout, Melee, MeleeCoreFn, MeleeUpdateFn, StartPointFn};
use crate::providers::held_item;
use crate::{Facts, Wolf};
use monaka_body::held::Kind;
use monaka_body::melee;
use monaka_body::melee_hit::{self, Hits};
use monaka_core::math::Vec3;
use monaka_contract::gamepad::Gamepad;
use monaka_contract::protocol::{LEFT_HAND, RIGHT_HAND};
use monaka_hook::module::Module;
use monaka_hook::{InFlight, Original, mem};
use monaka_producer::game::{Lack, Start, lines};
use monaka_producer::log;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::*};

/// The trace's settings: what it may damage in all (the game's), continuous (it stays open after a
/// hit, each thing hit once), and its box (1: 8 units a side) swept along the tip's path and the
/// blade.
const DAMAGE_CAP: f32 = 100.0;
const CONTINUOUS: u8 = 3;
const BOX: u8 = 1;

struct Setup {
    exe: Module,
    melee: &'static Melee,
    layout: &'static HandsLayout,
    swings: melee::Settings,
}

static SETUP: OnceLock<Setup> = OnceLock::new();

static UPDATE_ORIGINAL: Original<MeleeUpdateFn> = Original::new();
static START_JOINT_ORIGINAL: Original<AnimEventFn> = Original::new();
static START_TAG_ORIGINAL: Original<AnimEventFn> = Original::new();
static START_SIMPLE_ORIGINAL: Original<AnimEventFn> = Original::new();
static END_ORIGINAL: Original<AnimEventFn> = Original::new();

/// A swing's trace is open (until its last queries are collected).
static OPEN: AtomicBool = AtomicBool::new(false);
/// The right hand holds its melee weapon and swings attack (as of the latest update).
static SWINGS_ATTACK: AtomicBool = AtomicBool::new(false);
/// The swing the trace was last opened for (its tick).
static OPENED_FOR: AtomicU64 = AtomicU64::new(0);

static OPENED: AtomicU64 = AtomicU64::new(0);
static SWEEPS: AtomicU64 = AtomicU64::new(0);
static HITS: AtomicU64 = AtomicU64::new(0);
static HELD_BACK: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];
static NO_DAMAGE: AtomicU64 = AtomicU64::new(0);

/// The melee trace swept along the right hand's weapon. The game starts its attack as the trigger
/// goes down.
impl Hits for Wolf {
    const ATTACKS_ON_RELEASE: bool = false;

    fn install(start: &mut Start<'_, Self>) -> Result<(), Lack> {
        let Facts { exe, build } = *start.build;
        let (Some(melee), Some(guns)) = (build.melee.as_ref(), build.guns.as_ref()) else {
            return Err(Lack::Missing(format!("{}'s melee trace is not mapped", build.game)));
        };
        let _ = SETUP.set(Setup { exe, melee, layout: &guns.layout, swings: melee::Settings::from_options(start.options, RIGHT_HAND) });
        start.hooks.site(&UPDATE_ORIGINAL, &exe, &melee.update, update)?;
        // SAFETY: each target is checked against its exact prologue; the detours have their
        // signatures.
        unsafe {
            start.hooks.inline(&START_JOINT_ORIGINAL, "melee start (joint)", exe.at(melee.start_joint), &engine::START_JOINT_PROLOGUE, start_joint)?;
            start.hooks.inline(&START_TAG_ORIGINAL, "melee start (tag)", exe.at(melee.start_tag), &engine::START_TAG_PROLOGUE, start_tag)?;
            start.hooks.inline(&START_SIMPLE_ORIGINAL, "melee start (eye)", exe.at(melee.start_simple), &engine::START_TAG_PROLOGUE, start_simple)?;
            start.hooks.inline(&END_ORIGINAL, "melee end", exe.at(melee.end), &engine::START_JOINT_PROLOGUE, end)?;
        }
        melee::set_presses(false);
        monaka_pad::rewrite(Some(crate::cooking::body_forward::pad));
        log!("physical melee: a swing of the right controller with the knife or the pipe sweeps the game's melee trace along it; the trigger does nothing");
        Ok(())
    }

    fn summary() -> String {
        lines(&[summary(), melee::summary()])
    }
}

/// The stop's first phase: the trigger reaches the game again.
pub fn quiet() {
    SWINGS_ATTACK.store(false, Relaxed);
    monaka_pad::rewrite(None);
}

/// The pad the game reads, last: the right trigger let go while the right hand holds its melee
/// weapon and swings attack (the trigger would play the game's canned attack).
pub(crate) fn let_go_of_trigger(pad: &mut Gamepad) {
    if SWINGS_ATTACK.load(Relaxed) {
        pad.right_trigger = 0;
    }
}

/// Whether the hooks are in, the last of them too (the melee probe then leaves the animation
/// events to them).
pub fn hooked() -> bool {
    END_ORIGINAL.is_set()
}

pub fn summary() -> String {
    format!(
        "physical melee: {} swings opened the trace, {} sweeps, {} hit; the game's own attack held back {} starts, {} ends; {} swings without the weapon's melee damage",
        OPENED.load(Relaxed),
        SWEEPS.load(Relaxed),
        HITS.load(Relaxed),
        HELD_BACK[0].load(Relaxed),
        HELD_BACK[1].load(Relaxed),
        NO_DAMAGE.load(Relaxed)
    )
}

/// The quick melee is under way (the melee button's attack, the game's own).
fn quick_melee(setup: &Setup, hands: usize) -> bool {
    mem::read::<u8>(hands + setup.melee.in_swing).is_some_and(|b| b != 0)
}

/// What the right hand swings with: its melee weapon, outside the quick melee and while the view is
/// the player's own; else a gun (no swings). In a cutscene or a grab another camera has the view,
/// and the game's own attack (a scene's "fight back" prompt) must reach it; while hotwiring the
/// controllers move the wires.
fn swung(setup: &Setup, hands: usize) -> Kind {
    let kind = if quick_melee(setup, hands) || crate::providers::camera_taken::controlled() || crate::extras::hotwire::active() { Kind::Gun } else { held_item::right_kind(setup.layout, hands) };
    note_change(kind);
    kind
}

/// The kind last swung with, and how many changes were logged.
static LAST_KIND: AtomicU64 = AtomicU64::new(u64::MAX);
static CHANGES_LOGGED: AtomicU64 = AtomicU64::new(0);
const CHANGES_LOG_MOST: u64 = 300;

/// At a change of what the right hand holds (the knife put away for a ladder, taken out again in
/// the water), the view's pitch and height, and the head's share of the pitch: what the game's
/// ladder and swimming read to go up or down.
fn note_change(kind: Kind) {
    let code = kind as u64;
    if LAST_KIND.swap(code, Relaxed) == code || CHANGES_LOGGED.fetch_add(1, Relaxed) >= CHANGES_LOG_MOST {
        return;
    }
    let (Some(view), Some(up)) = (crate::providers::eye_site::render_view(), crate::providers::eye_site::render_view_up()) else { return };
    let head = monaka_body::look::baked::<Wolf>(None).map_or(f32::NAN, |share| share.pitch.to_degrees());
    log!("melee: now {kind:?}; the view pitched {up:.1} degrees up (the head's share {head:.1}), at height {:.1}", view[2]);
}

/// The right hand's weapon where its controller holds it now: the hand, and the blade's tip
/// (world units).
fn blade(hands: usize) -> Option<(Vec3, Vec3)> {
    let guns = held_item::setup()?;
    let (hand, axis) = held_item::gun_target(guns, hands, RIGHT_HAND)?;
    let reach = melee_hit::blade(Kind::Melee)? * guns.units_per_metre();
    let row = 3 * held_item::blade_row(guns);
    Some((hand, [0, 1, 2].map(|k| hand[k] + axis[row + k] * reach)))
}

/// The right hand's weapon's melee damage and impact effect (its melee projectile's).
fn damage(setup: &Setup, hands: usize) -> Option<(usize, usize)> {
    let decl = held_item::right_decl(setup.layout, hands)?;
    let projectile = mem::read::<usize>(decl + setup.melee.pistol_whip).filter(|&p| p != 0)?;
    let damage = mem::read::<usize>(projectile + setup.melee.projectile_damage).filter(|&d| d != 0)?;
    Some((damage, mem::read::<usize>(projectile + engine::PROJECTILE_IMPACT_EFFECT).unwrap_or(0)))
}

/// Opens the trace at `tip` with the right hand's weapon's melee damage, a box swept along the
/// tip's path and along the blade. Whether it opened.
fn open(setup: &Setup, hands: usize, trace: usize, tip: Vec3) -> bool {
    let Some((damage, effect)) = damage(setup, hands) else {
        if NO_DAMAGE.fetch_add(1, Relaxed) < 3 {
            log!("physical melee: the right hand's weapon has no melee damage to deal");
        }
        return false;
    };
    // SAFETY: the game's own start of its own trace, on the game thread (inside the hands' think).
    unsafe {
        let start = std::mem::transmute::<usize, StartPointFn>(setup.exe.at(setup.melee.start_point));
        start(trace, tip.as_ptr(), 2, DAMAGE_CAP, CONTINUOUS, BOX, damage, effect);
    }
    let fields = &setup.melee.fields;
    // The blade from the hand (second point) to the tip; the damage at the weapon's own scale; a
    // box gathered around the tip too.
    mem::write::<i16>(trace + engine::MELEE_JOINT2, 0);
    mem::write::<i32>(trace + engine::MELEE_DAMAGE_SCALE, 1);
    mem::write::<u8>(trace + fields.one_last_update + 1, 0);
    let n = OPENED.fetch_add(1, Relaxed);
    if log::seldom(n, 8) {
        log!("physical melee: swing {n} opened the trace at {tip:.1?}");
    }
    true
}

/// The game is the server (its core damages at once; in single player it is).
fn is_server(setup: &Setup) -> bool {
    mem::read::<usize>(setup.exe.at(setup.melee.game_local)).filter(|&g| g != 0).and_then(|g| mem::read::<u8>(g + setup.melee.is_server)).is_none_or(|b| b != 0)
}

/// One sweep of the trace: the tip from where it was to `tip`, the blade from `hand` to `tip`.
/// Whether it hit.
fn sweep(setup: &Setup, hands: usize, trace: usize, hand: Vec3, tip: Vec3) -> Option<bool> {
    let player = mem::read::<usize>(hands + engine::HANDS_PLAYER).filter(|&p| p != 0)?;
    let (core, attacker) = if is_server(setup) {
        (setup.melee.core_server, mem::read::<usize>(player + setup.melee.player_entity).filter(|&e| e != 0)?)
    } else {
        (setup.melee.core_client, player)
    };
    SWEEPS.fetch_add(1, Relaxed);
    // SAFETY: the game's own sweep of its own trace, on the game thread (inside the hands' think),
    // with the attacker its own update passes.
    let hit = unsafe { std::mem::transmute::<usize, MeleeCoreFn>(setup.exe.at(core))(trace, attacker, tip.as_ptr(), hand.as_ptr(), 0) } as u8 != 0;
    if hit {
        let n = HITS.fetch_add(1, Relaxed);
        if log::seldom(n, 8) {
            let what = mem::read::<usize>(trace + setup.melee.fields.hit_entity).unwrap_or(0);
            log!("physical melee: hit {n} ({what:#x}) with the blade from {hand:.1?} to {tip:.1?}");
        }
    }
    Some(hit)
}

/// Shuts a swing's trace, if one is open (no updates left, nothing more collected).
fn shut(setup: &Setup, trace: usize) {
    if OPEN.swap(false, Relaxed) {
        mem::write::<i32>(trace + setup.melee.fields.num_samples, 0);
        mem::write::<u8>(trace + setup.melee.fields.one_last_update, 0);
    }
}

/// One update with the controllers: the swings fed, the trace opened by a new swing and swept
/// while it is open. Whether it was swept (the game's own update is then skipped).
fn swept(setup: &Setup, hands: usize) -> bool {
    let kind = swung(setup, hands);
    SWINGS_ATTACK.store(kind == Kind::Melee && setup.swings.swing, Relaxed);
    melee::update(kind, [LEFT_HAND, RIGHT_HAND].map(monaka_body::controllers::palm), &setup.swings);
    let trace = hands + setup.melee.trace;
    if kind != Kind::Melee {
        shut(setup, trace);
        return false;
    }
    let swinging = melee::swinging(RIGHT_HAND);
    // A swing not yet opened for: the one under way (one that is already over opens nothing).
    let new = melee::swung_at().filter(|&at| at != OPENED_FOR.load(Relaxed));
    if let Some(at) = new.filter(|_| !swinging) {
        OPENED_FOR.store(at, Relaxed);
    }
    if !swinging && !OPEN.load(Relaxed) {
        return false;
    }
    let Some((hand, tip)) = blade(hands) else {
        shut(setup, trace);
        return false;
    };
    if let Some(at) = new.filter(|_| swinging) {
        OPENED_FOR.store(at, Relaxed);
        if open(setup, hands, trace, tip) {
            OPEN.store(true, Relaxed);
        }
    }
    if !OPEN.load(Relaxed) {
        return false;
    }
    let fields = &setup.melee.fields;
    // Two updates at a time while the swing lasts. Then the last one runs out: its sweep, and an
    // update that only collects what that sweep found.
    if swinging {
        mem::write::<i32>(trace + fields.num_samples, 2);
    }
    sweep(setup, hands, trace, hand, tip);
    let samples = mem::read::<i32>(trace + fields.num_samples).unwrap_or(0);
    let collecting = mem::read::<u8>(trace + fields.one_last_update).is_some_and(|b| b != 0);
    if samples <= 0 && !collecting {
        OPEN.store(false, Relaxed);
    }
    true
}

unsafe extern "system" fn update(hands: usize) {
    let flight = InFlight::enter();
    if let Some(setup) = SETUP.get() {
        if !crate::on(crate::MELEE_AT) {
            shut(setup, hands + setup.melee.trace);
        } else if swept(setup, hands) {
            return;
        }
    }
    // SAFETY: forwards the game's own call.
    unsafe { UPDATE_ORIGINAL.get(&flight)(hands) }
}

/// Whether the game's own start or end of the trace waits: while a swing's trace is open, and
/// (starts only) while the right hand holds its melee weapon outside the quick melee and swings
/// attack.
fn held_back(hands: usize, ends: bool) -> bool {
    let Some(setup) = SETUP.get().filter(|_| crate::on(crate::MELEE_AT)) else { return false };
    OPEN.load(Relaxed) || (!ends && setup.swings.swing && swung(setup, hands) == Kind::Melee)
}

fn event(original: &Original<AnimEventFn>, ends: bool, hands: usize, args: [usize; 5]) -> usize {
    let flight = InFlight::enter();
    if held_back(hands, ends) {
        HELD_BACK[ends as usize].fetch_add(1, Relaxed);
        return 0;
    }
    // SAFETY: forwards the game's own call.
    unsafe { original.get(&flight)(hands, args[0], args[1], args[2], args[3], args[4]) }
}

unsafe extern "system" fn start_joint(hands: usize, a: usize, b: usize, c: usize, d: usize, e: usize) -> usize {
    event(&START_JOINT_ORIGINAL, false, hands, [a, b, c, d, e])
}

unsafe extern "system" fn start_tag(hands: usize, a: usize, b: usize, c: usize, d: usize, e: usize) -> usize {
    event(&START_TAG_ORIGINAL, false, hands, [a, b, c, d, e])
}

unsafe extern "system" fn start_simple(hands: usize, a: usize, b: usize, c: usize, d: usize, e: usize) -> usize {
    event(&START_SIMPLE_ORIGINAL, false, hands, [a, b, c, d, e])
}

unsafe extern "system" fn end(hands: usize, a: usize, b: usize, c: usize, d: usize, e: usize) -> usize {
    event(&END_ORIGINAL, true, hands, [a, b, c, d, e])
}
