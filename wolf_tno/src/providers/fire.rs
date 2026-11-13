//! Provider: the fire hook (`monaka_body::launch::Fires`), where the player's guns fire. Every shot
//! is queued as fire parms (`engine::Guns::fire_queue`): its start and its axis rows (forward, left,
//! up), the game's spread applied after. A shot from a hand's gun (its weapon the one that hand's
//! gun was placed with, [`held_item::placed`]) is asked of the gun ready-made
//! (`monaka_body::launch::shot`, with that hand), and its start and forward axis are written as it
//! answers, the gun's own left and up beside it.

use crate::providers::engine::{self, FireQueueFn};
use crate::providers::held_item::{self, Placed};
use crate::{Facts, Wolf};
use monaka_body::launch::{self, Fired, Fires, Flight, Shot, Weapon};
use monaka_contract::protocol::{LEFT_HAND, RIGHT_HAND};
use monaka_hook::{InFlight, Original, mem};
use monaka_producer::game::{Lack, Start};
use monaka_producer::log;
use std::sync::atomic::{AtomicU64, Ordering::*};

/// The guns fire through the fire queue; a shot keeps the game's speed.
impl Fires for Wolf {
    const SPEED: bool = false;

    fn install(start: &mut Start<'_, Self>) -> Result<(), Lack> {
        if FIRE_ORIGINAL.is_set() {
            return Ok(());
        }
        let Facts { exe, build } = *start.build;
        let Some(facts) = build.guns.as_ref() else {
            return Err(Lack::Missing(format!("where {} fires its guns is not mapped", build.game)));
        };
        start.hooks.site(&FIRE_ORIGINAL, &exe, &facts.fire_queue, fire_queue)?;
        full_recharge(exe.base());
        Ok(())
    }
}

/// The laser rifle recharges itself to a full battery, not the game's share of it
/// (`engine::BATTERY_RECHARGE_SHARE`), where this build's bytes are as expected.
fn full_recharge(base: usize) {
    let (at, bytes, one) = engine::BATTERY_RECHARGE_SHARE;
    let (code, one) = (base + at, base + one);
    let mut now = [0u8; 8];
    let ok = mem::read_bytes(code, &mut now) && now == bytes && mem::read::<f32>(one) == Some(1.0);
    let disp = (one as i64 - (code + 8) as i64) as i32;
    let done = ok && mem::write_bytes(code + 4, &disp.to_le_bytes());
    let share = mem::read::<f32>((code as i64 + 8 + i64::from(i32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]))) as usize);
    log!(
        "guns: the laser rifle's own recharge {}",
        if done { format!("fills its battery (it stopped at {:.0}% of it)", share.unwrap_or(f32::NAN) * 100.0) } else { "left as the game has it (not this build's code)".to_owned() }
    );
}

pub static FIRE_ORIGINAL: Original<FireQueueFn> = Original::new();
static SHOTS: AtomicU64 = AtomicU64::new(0);
/// Shots from a weapon no hand's gun was placed with (a thrown knife), logged the first few.
static OTHER_SHOTS: AtomicU64 = AtomicU64::new(0);
static AIMED: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];

pub fn summary() -> String {
    format!("{} shots, aimed right {} left {}", SHOTS.load(Relaxed), AIMED[RIGHT_HAND].load(Relaxed), AIMED[LEFT_HAND].load(Relaxed))
}

/// The hand whose gun was placed with `weapon`, and that gun: the left hand's first.
fn gun_of(placed: [Option<Placed>; 2], weapon: usize) -> Option<(usize, Placed)> {
    placed.iter().enumerate().find_map(|(side, p)| p.filter(|p| p.weapon == weapon).map(|p| (side, p)))
}

/// The shot queued at `parms` from the gun of `side` (`gun`): the game's start and forward axis
/// asked of `shot`, and its answer written (the start, then the axis: the answer's forward, the
/// gun's left and up). Whether both were written.
fn aim(parms: usize, side: usize, gun: &Placed, shot: impl FnOnce(&Fired) -> Option<Shot>) -> bool {
    let (Some(start), Some(axis)) = (mem::read::<[f32; 3]>(parms + engine::FIRE_START), mem::read::<[f32; 9]>(parms + engine::FIRE_AXIS)) else {
        return false;
    };
    // Where the game starts it: the gun's muzzle as it is drawn.
    held_item::learn_muzzle(gun, start);
    let forward = [axis[0], axis[1], axis[2]];
    let fired = Fired { weapon: Weapon::Gun, game: Flight { from: start, velocity: forward }, aim: Some(forward), side: Some(side) };
    let Some(shot) = shot(&fired) else { return false };
    let [f0, f1, f2] = shot.direction;
    let aimed = held_item::aimed_axis(gun);
    let axis = [f0, f1, f2, aimed[3], aimed[4], aimed[5], aimed[6], aimed[7], aimed[8]];
    mem::write(parms + engine::FIRE_START, shot.from) && mem::write(parms + engine::FIRE_AXIS, axis)
}

pub unsafe extern "system" fn fire_queue(manager: usize, parms: usize) -> u32 {
    let flight = InFlight::enter();
    if crate::on(crate::GUN_AT)
        && parms != 0
        && let Some(weapon) = mem::read::<usize>(parms + engine::FIRE_WEAPON).filter(|&w| w != 0)
    {
        SHOTS.fetch_add(1, Relaxed);
        let from_hand = gun_of(held_item::placed(), weapon);
        // A knife thrown by a swing leaves from the hand along it (`cooking::knife_throw`): a shot
        // from the knife the right hand holds (placed on its controller), or from no hand's gun.
        let knife = from_hand.is_none_or(|(side, _)| side == RIGHT_HAND && held_item::right_throws_knives());
        let thrown = if knife { crate::cooking::knife_throw::thrown_now() } else { None };
        match (thrown, from_hand) {
            (Some((from, axis)), _) => {
                mem::write(parms + engine::FIRE_START, from);
                mem::write(parms + engine::FIRE_AXIS, axis);
            }
            (None, Some((side, gun))) => {
                if aim(parms, side, &gun, launch::shot) && AIMED[side].fetch_add(1, Relaxed) < 3 {
                    log!("guns: {} gun shot from {:?} along {:?}", if side == RIGHT_HAND { "right" } else { "left" }, gun.origin, &gun.axis[..3]);
                }
            }
            (None, None) => {
                if OTHER_SHOTS.fetch_add(1, Relaxed) < 10 {
                    let name = mem::read::<usize>(weapon + engine::INVENTORY_ITEM_DECL).filter(|&d| d != 0).and_then(held_item::decl_name);
                    log!("guns: a shot from {} (weapon {weapon:#x}), not a hand's gun: left as the game aims it", name.as_deref().unwrap_or("an unnamed weapon"));
                }
            }
        }
    }
    // SAFETY: forwards the game's own call.
    unsafe { FIRE_ORIGINAL.get(&flight)(manager, parms) }
}
