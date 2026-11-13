//! Provider part, to be: the player on a ladder (the game's contextual ladder, in the player's
//! behaviours: `engine::Ladder`).
//!
//! On a ladder the game reads the view's angles and the left stick: a head turned away, or the
//! stick turned with it, pushes the player off sideways or back. While the player is on one
//! ([`on`]: mounting, idle or climbing; not dismounting), the look holds the head's share
//! (`look::holds`), so the head looks around freely, and the stick goes to the game straight up
//! and down only ([`pad`]): forward climbs up, back down, and nothing pushes the player off. The
//! jump button (A) still lets go; the top and the bottom still step off.

use crate::providers::camera_taken;
use crate::providers::engine::Ladder;
use monaka_contract::gamepad::Gamepad;
use monaka_hook::mem;
use monaka_producer::log;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicI32, AtomicU64, Ordering::*};

static FACTS: OnceLock<Ladder> = OnceLock::new();
/// The ladder state last seen (-1: none read), and changes logged.
static LAST: AtomicI32 = AtomicI32::new(-1);
static LOGGED: AtomicU64 = AtomicU64::new(0);

/// The states the player is on the ladder in: mounting (1 to 4), idle (5, 6), climbing (7 to
/// 10), sliding down (11, 12). 0 is none, 13 on dismounting.
pub fn holding(state: i32) -> bool {
    (1..=12).contains(&state)
}

pub fn install(ladder: Option<Ladder>) {
    if let Some(ladder) = ladder {
        let _ = FACTS.set(ladder);
    } else {
        log!("ladder: not mapped for this build: ladders as the game has them");
    }
}

/// The player's ladder state now, if it reads as one.
pub fn state() -> Option<i32> {
    let facts = FACTS.get()?;
    let (_, player) = camera_taken::game_player()?;
    mem::read::<i32>(player + facts.state).filter(|s| (0..=0x11).contains(s))
}

/// Whether the player is on a ladder now (not dismounting).
pub fn on() -> bool {
    state().is_some_and(holding)
}

/// Whether the game reads the ladder's stick turned over (up the ladder by pulling back).
fn inverted() -> bool {
    let (Some(facts), Some((_, player))) = (FACTS.get(), camera_taken::game_player()) else { return false };
    mem::read::<u8>(player + facts.invert_input) == Some(1)
}

/// The stick for the game on a ladder, pushed `(x, y)`: up and down only, the way the game climbs
/// (`inverted`: its stick turned over).
pub fn climb_stick((_, y): (i16, i16), inverted: bool) -> (i16, i16) {
    (0, if inverted { y.saturating_neg() } else { y })
}

/// The pad the game reads, last: on a ladder, the left stick as pushed (`pushed`, before any turn
/// by the head or the room-scale push) up and down only.
pub fn pad(pad: &mut Gamepad, pushed: (i16, i16)) {
    let Some(state) = state() else { return };
    let inverted = inverted();
    if LAST.swap(state, Relaxed) != state && LOGGED.fetch_add(1, Relaxed) < 60 {
        log!(
            "ladder: state {state} ({}){}",
            if holding(state) { "on it: the head's turns held out, the stick up and down only" } else { "off it" },
            if inverted { ", its stick turned over" } else { "" }
        );
    }
    if holding(state) {
        (pad.thumb_lx, pad.thumb_ly) = climb_stick(pushed, inverted);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn on_a_ladder_the_stick_only_climbs() {
        assert!(holding(5) && holding(10) && holding(1));
        assert!(!holding(0) && !holding(13) && !holding(16));
        assert_eq!(climb_stick((20000, 30000), false), (0, 30000));
        assert_eq!(climb_stick((-5, -30000), true), (0, 30000));
        assert_eq!(climb_stick((0, i16::MIN), true), (0, i16::MAX));
    }
}
