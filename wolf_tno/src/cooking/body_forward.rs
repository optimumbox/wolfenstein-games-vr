//! Cooking, waiting for a Monaka ready-made for the run's heading (part of the look): the left
//! stick on the head's heading, the body's, or halfway between ([`MOVE_HEADING`]).
//!
//! With head aim the game camera carries the head's yaw (`monaka_body::look`'s share in the
//! command angles), and the game moves the player relative to its camera: the run goes where the
//! head looks. Fully on the body's heading (the right stick's turn) a glance aside felt like
//! moving sideways; fully on the head's, the head steers the run, which is what physical turning
//! needs (as KF2VR's head-relative movement, its default). Before the game reads the pad, the
//! left stick is turned back by the chosen share of the head's turn ([`head_share_out`]). The
//! view, the aim and the shots stay the head's.

use crate::Wolf;
use monaka_body::look;
use monaka_contract::gamepad::Gamepad;
use monaka_contract::manifest::{Opt, Shown, choice, opt};
use monaka_core::options::Options;
use monaka_producer::game::{Component, Lack, Start};
use std::sync::atomic::{AtomicU32, Ordering};

/// Where the left stick moves the player (`move_heading`): where the head looks (the default,
/// physical turns steer), halfway, or where the body faces (only the right stick steers).
pub const MOVE_HEADING: Opt = opt(
    "move_heading",
    "Movement direction",
    "Head: forward is where you look, so physical turns steer. Body: only the right stick turns.",
    choice(&[("head", "Head"), ("halfway", "Halfway"), ("body", "Body")], "head"),
)
.on(Shown::Controls);

/// The run's heading as `move_heading` asks, read as the run starts.
pub const MOVE_HEADING_COMPONENT: Component<Wolf> = Component {
    name: "move heading",
    options: &[MOVE_HEADING],
    settings: |options, _| {
        share_out(options);
    },
    install,
    ..Component::NONE
};

/// The shares of the head's turn taken out of the run's heading, by `move_heading` id.
const SHARES: [(&str, f32); 3] = [("head", 0.0), ("halfway", 0.5), ("body", 1.0)];

/// The run's share taken out, as `f32` bits (0.0 until read: the head's heading).
static SHARE_OUT: AtomicU32 = AtomicU32::new(0);

fn install(start: &mut Start<'_, Wolf>) -> Result<(), Lack> {
    SHARE_OUT.store(share_out(start.options).to_bits(), Ordering::Relaxed);
    Ok(())
}

/// How much of the head's turn `options` take out of the run's heading (0: where the head looks,
/// 1: where the body faces).
pub fn share_out(options: &Options) -> f32 {
    options.choice("move_heading", &SHARES, 0.0)
}

/// The run's share taken out ([`MOVE_HEADING_COMPONENT`]).
pub fn head_share_out() -> f32 {
    f32::from_bits(SHARE_OUT.load(Ordering::Relaxed))
}

/// The pad the game reads, last (`monaka_pad::rewrite`): the knife thrown by the trigger
/// (`cooking::knife_throw`), the left stick onto the chosen heading, the right trigger let go
/// while swings attack (`providers::melee`), the left one while it would only zoom
/// (`cooking::sights`), and Y held opening the journal (`cooking::journal`).
pub fn pad(pad: &mut Gamepad) {
    crate::cooking::sights::no_zoom(pad);
    crate::cooking::journal::pad(pad);
    // The knife's throw reads the trigger before the swings let go of it.
    crate::cooking::knife_throw::pad(pad);
    crate::providers::melee::let_go_of_trigger(pad);
    let out = head_share_out();
    if out == 0.0 {
        return;
    }
    if let Some(share) = look::baked::<Wolf>(None) {
        (pad.thumb_lx, pad.thumb_ly) = turned_stick(pad.thumb_lx, pad.thumb_ly, share.yaw * out);
    }
}

/// The stick `(x, y)` (x right, y forward) given on a heading `yaw` radians left of the one wanted,
/// turned onto it.
pub fn turned_stick(x: i16, y: i16, yaw: f32) -> (i16, i16) {
    let (s, c) = (-yaw).sin_cos();
    let (x, y) = (f32::from(x), f32::from(y));
    let axis = |v: f32| v.round().clamp(-32768.0, 32767.0) as i16;
    (axis(x * c - y * s), axis(x * s + y * c))
}

#[cfg(test)]
mod tests;
