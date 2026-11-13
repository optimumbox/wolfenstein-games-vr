//! Cooking, waiting for a Wolfenstein provider of Monaka's throw by hand (`monaka_body::throw`):
//! the knife thrown by swinging it.
//!
//! The game throws a knife from the knife held in hand by the right stick's press, firing it as a
//! shot of its own through the fire queue (`providers::fire`). With the knife in the right hand the
//! trigger does nothing else (swings attack: `providers::melee`), so holding the trigger readies a
//! throw and letting go throws ([`Trigger`]): the right stick is pressed for the game a moment. The
//! knife that leaves then goes from the hand along the swing ([`swing`]: the way the knife moved
//! fastest over the last moments before the let-go), at the game's own speed; with no swing to read
//! (the hand still) it goes as the game aims it.

use monaka_contract::gamepad::{Gamepad, button};
use monaka_core::math::{Vec3, cross, length, scale, sub, unit};
use monaka_core::sync::Locked;
use monaka_producer::log;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering::*};
use std::time::Instant;

/// The trigger past this readies a throw; back under the second, it lets go (0..=255).
pub const READY: u8 = 128;
pub const LET_GO: u8 = 64;
/// How long the right stick is pressed for the game at a let-go (ms).
pub const TAP_MS: u64 = 120;
/// A knife leaving within this long of a let-go is the one thrown (ms).
pub const THROW_WINDOW_MS: u64 = 1500;
/// The swing is read over this long before the let-go (ms), and counts from this fast (m/s).
pub const SWING_MS: u64 = 300;
pub const SWING_FAST: f32 = 1.5;

/// The trigger's state: ready (held past [`READY`]) or not.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Trigger {
    ready: bool,
}

impl Trigger {
    /// The trigger at `pulled`: whether this is a let-go (it was ready and is now back under
    /// [`LET_GO`]).
    pub fn step(&mut self, pulled: u8) -> bool {
        if pulled >= READY {
            self.ready = true;
            return false;
        }
        let let_go = self.ready && pulled < LET_GO;
        if let_go {
            self.ready = false;
        }
        let_go
    }
}

/// One place of the knife in hand (game units) and when (ms on [`now_ms`]'s clock).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sample {
    pub at: u64,
    pub position: Vec3,
}

/// The swing in `samples` (oldest first) over the [`SWING_MS`] up to `until`: the fastest of the
/// knife's moves between every other sample (game units a second) and its way (unit). None below
/// [`SWING_FAST`] (in `units_per_metre`).
pub fn swing(samples: &[Sample], until: u64, units_per_metre: f32) -> Option<(f32, Vec3)> {
    let recent: Vec<&Sample> = samples.iter().filter(|s| s.at <= until && until - s.at <= SWING_MS).collect();
    let mut best: Option<(f32, Vec3)> = None;
    for pair in recent.windows(3) {
        let (a, b) = (pair[0], pair[2]);
        let seconds = (b.at.saturating_sub(a.at)) as f32 / 1000.0;
        if seconds <= 0.0 {
            continue;
        }
        let moved = sub(b.position, a.position);
        let speed = length(moved) / seconds;
        if best.is_none_or(|(s, _)| speed > s) {
            best = unit(moved).map(|way| (speed, way));
        }
    }
    best.filter(|(speed, _)| *speed >= SWING_FAST * units_per_metre)
}

/// Fire axis rows (forward, left, up; id: z up) for a shot along `forward` (unit).
pub fn fire_axis(forward: Vec3) -> Option<[f32; 9]> {
    let left = unit(cross([0.0, 0.0, 1.0], forward)).or_else(|| unit(cross([1.0, 0.0, 0.0], forward)))?;
    let up = cross(forward, left);
    Some([forward[0], forward[1], forward[2], left[0], left[1], left[2], up[0], up[1], up[2]])
}

/// Milliseconds since the first ask (a monotonic clock fine enough for a swing).
pub fn now_ms() -> u64 {
    static START: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
    START.get_or_init(Instant::now).elapsed().as_millis() as u64
}

/// The knife's recent places, the trigger, the last let-go (with where the knife pointed then:
/// origin, aim axis rows) and the tap's end.
struct State {
    samples: Vec<Sample>,
    trigger: Trigger,
    let_go: Option<(u64, Option<(Vec3, [f32; 9])>)>,
    tap_until: u64,
}

/// The aim of a knife placed with axis rows `axis` (the game's: blade along its up): forward along
/// the blade, left kept, up the placement's back.
pub fn knife_axis(axis: &[f32; 9]) -> [f32; 9] {
    let [f0, f1, f2, l0, l1, l2, u0, u1, u2] = *axis;
    [u0, u1, u2, l0, l1, l2, -f0, -f1, -f2]
}

/// The knife in the right hand's aim now (origin, axis rows: [`knife_axis`]), as last placed.
pub fn knife_aim() -> Option<(Vec3, [f32; 9])> {
    let placed = crate::providers::held_item::placed()[monaka_contract::protocol::RIGHT_HAND]?;
    Some((placed.origin, knife_axis(&placed.axis)))
}

/// Whether a throw is readied: the knife in the right hand, the trigger held (the red dot shows
/// where it goes).
pub fn readied() -> bool {
    crate::providers::held_item::right_throws_knives() && STATE.try_lock().is_ok_and(|s| s.trigger.ready)
}

static STATE: Mutex<State> = Mutex::new(State { samples: Vec::new(), trigger: Trigger { ready: false }, let_go: None, tap_until: 0 });
static LET_GOS: AtomicU64 = AtomicU64::new(0);
static THROWN: AtomicU64 = AtomicU64::new(0);
const KEPT: usize = 64;

/// The knife in the right hand was placed at `position` (game units), from the item placer.
pub fn note_knife(position: Vec3) {
    let Ok(mut state) = STATE.try_lock() else { return };
    state.samples.push(Sample { at: now_ms(), position });
    if state.samples.len() > KEPT {
        state.samples.remove(0);
    }
}

/// The pad the game reads, before the trigger is let go for the swings: with the knife in the right
/// hand, a trigger let go presses the right stick for [`TAP_MS`] and keeps the swing for the knife
/// that leaves.
pub fn pad(pad: &mut Gamepad) {
    let knife = crate::providers::held_item::right_throws_knives();
    let now = now_ms();
    let mut state = STATE.locked();
    if knife && state.trigger.step(pad.right_trigger) {
        let units = crate::providers::held_item::setup().map_or(1.0, |s| s.units_per_metre());
        let swung = swing(&state.samples, now, units);
        state.let_go = Some((now, knife_aim()));
        state.tap_until = now + TAP_MS;
        let n = LET_GOS.fetch_add(1, Relaxed) + 1;
        if n <= 20 {
            match swung {
                Some((speed, _)) => log!("knife throw: let go {n}, swung at {:.1} m/s: thrown where the knife points", speed / units),
                None => log!("knife throw: let go {n}: thrown where the knife points"),
            }
        }
    }
    if !knife {
        state.trigger = Trigger::default();
    }
    if now < state.tap_until {
        pad.buttons |= button::RIGHT_THUMB;
    }
}

/// A shot from a weapon no hand's gun was placed with, leaving now: if a knife throw was let go
/// lately, where it leaves (the knife at the let-go) and its fire axis (where the blade pointed
/// then: the red dot's line).
pub fn thrown_now() -> Option<(Vec3, [f32; 9])> {
    let mut state = STATE.locked();
    let (at, aim) = state.let_go?;
    state.let_go = None;
    if now_ms().saturating_sub(at) > THROW_WINDOW_MS {
        return None;
    }
    let (from, axis) = aim?;
    let way = [axis[0], axis[1], axis[2]];
    let fire = fire_axis(way)?;
    if THROWN.fetch_add(1, Relaxed) < 20 {
        log!("knife throw: the knife leaves the hand where it pointed {:.2?}", scale(way, 1.0));
    }
    Some((from, fire))
}

#[cfg(test)]
mod tests;
