//! Cooking, waiting for a Monaka aim-down-sights feature, which would own the dominant eye's shift
//! and the sight lock for every game with sights, reading the held item provider (the held gun's
//! sight line) and whether the game is aiming (a situation or an action; here the eye site reads it
//! from the game camera's zoom against the resting field of view,
//! [`eye_site`](crate::providers::eye_site)).
//!
//! The eyes on the sights: the dominant eye sits where the game camera is, so it looks straight
//! down the weapon's sights ([`put_dominant_eye`]), and while aiming down sights the head's
//! position slides onto the game camera, so the sights line up whatever `head_position` says
//! ([`sighting`], [`head_position`]).
//!
//! No aiming down sights from the pad, though: in VR the gun is aimed where it points, and the
//! game's zoom only narrows the view. On foot with one gun the left trigger (the game's zoom) is
//! let go before the game reads it ([`no_zoom`]).

use crate::eyes::DominantEye;
use crate::providers::{camera_taken, held_item};
use monaka_contract::gamepad::Gamepad;
use monaka_core::options::Shared;
use monaka_contract::protocol::HeadPose;
use std::sync::atomic::{AtomicBool, Ordering::Relaxed};

/// The laser rifle's laser sight is on (the right stick's push up turns it on and off).
static LASER_ON: AtomicBool = AtomicBool::new(false);
/// The right stick is pushed up and has not come back yet.
static LASER_PUSHED: AtomicBool = AtomicBool::new(false);
/// The right stick's push up past this share of its range turns the laser sight on or off; back
/// under the second, it can again.
const LASER_PUSH: f32 = 0.7;
const LASER_REST: f32 = 0.3;

/// Whether the laser rifle's laser sight is on.
pub fn laser_switched_on() -> bool {
    LASER_ON.load(Relaxed)
}

/// Whether a push of the right stick to `y` (up positive, full 32767) with it `pushed` already turns
/// the laser sight on or off, and whether it is pushed after it.
pub fn laser_push(y: i16, pushed: bool) -> (bool, bool) {
    let share = f32::from(y) / 32767.0;
    match pushed {
        false if share > LASER_PUSH => (true, true),
        true if share < LASER_REST => (false, false),
        _ => (false, pushed),
    }
}

/// In the pad the game reads, with the laser rifle in the right hand: the right stick's push up
/// turns its laser sight on and off, and goes no further (it would look up). The stick is read as
/// the controller holds it: the pad the game reads has its up and down taken (the head looks).
pub fn laser_switch(pad: &mut Gamepad) {
    if !held_item::right_laser_switched() || camera_taken::controlled() || camera_taken::wheel() {
        LASER_PUSHED.store(false, Relaxed);
        return;
    }
    let y = monaka_pad::right_stick(250).map_or(pad.thumb_ry, |[_, y]| (y * 32767.0).round() as i16);
    let (switch, pushed) = laser_push(y, LASER_PUSHED.load(Relaxed));
    LASER_PUSHED.store(pushed, Relaxed);
    if switch {
        let on = !LASER_ON.fetch_xor(true, Relaxed);
        monaka_producer::log!("sights: the laser rifle's laser sight turned {}", if on { "on" } else { "off" });
    }
    if pad.thumb_ry > 0 {
        pad.thumb_ry = 0;
    }
}

/// In the pad the game reads: the left trigger let go while it would only zoom ([`zooms`]).
pub fn no_zoom(pad: &mut Gamepad) {
    pad.left_trigger = left_pulled(pad.left_trigger, held_item::dual());
    if zooms(held_item::dual(), camera_taken::controlled() || camera_taken::mounted()) {
        pad.left_trigger = 0;
    }
}

/// The left trigger rests a little pulled (a finger on it reads ~36 of 255), which the game takes
/// for a pull: with two guns it fires the left one on its own. With two guns the trigger reads
/// pulled only past [`LEFT_PULLED`], as pulled from there.
pub fn left_pulled(trigger: u8, dual: bool) -> u8 {
    if !dual {
        return trigger;
    }
    if trigger <= LEFT_PULLED {
        return 0;
    }
    let share = f32::from(trigger - LEFT_PULLED) / f32::from(255 - LEFT_PULLED);
    (share * 255.0).round() as u8
}

/// With two guns, the left trigger's rest (of 255): below it, not pulled.
pub const LEFT_PULLED: u8 = 90;

/// Whether the left trigger is the game's zoom: on foot with one gun. With two guns it fires the
/// left one; on a mounted gun, in the mech or a car (another camera on the view) it is that
/// machine's (firing, reversing).
pub fn zooms(dual: bool, other_camera: bool) -> bool {
    !dual && !other_camera
}

/// Where the head's position puts the view (metres): the head model's, slid onto the game camera
/// as far as the view is `sighted` ([`sighting`]).
pub fn head_position(shared: &Shared, pose: &HeadPose, sighted: f32) -> [f32; 3] {
    let position = shared.head_position.position(pose);
    if sighted > 0.0 { position.map(|v| v * (1.0 - sighted)) } else { position }
}

/// Degrees of zoom below the resting field of view that count as fully aiming down sights, and
/// the margin below which it is still rest (The Old Blood's hip view is 79.999985 of 80).
const SIGHT_ZOOM: f32 = 5.0;
const SIGHT_MARGIN: f32 = 0.5;

/// How far into aiming down sights the game camera is, from its field of view and the resting
/// one: 0 at rest (or wider, sprinting), 1 once zoomed in by [`SIGHT_ZOOM`] degrees or more. The
/// engine zooms over a few frames, so the eyes slide onto the camera rather than jump.
pub fn sighting(fov: f32, rest: f32) -> f32 {
    if !(fov.is_finite() && rest.is_finite() && rest > 1.0) {
        return 0.0;
    }
    ((rest - fov - SIGHT_MARGIN) / (SIGHT_ZOOM - SIGHT_MARGIN)).clamp(0.0, 1.0)
}

/// Moves the view sideways so the dominant eye sits where the game camera is: the engine puts
/// the left eye `separation` units along the view's left axis and the right eye as far the other
/// way.
pub fn put_dominant_eye(view: &mut [f32; 12], dominant: DominantEye, separation: f32) {
    let side = match dominant {
        DominantEye::None => return,
        DominantEye::Right => 1.0,
        DominantEye::Left => -1.0,
    };
    for k in 0..3 {
        view[k] += side * separation * view[6 + k];
    }
}

#[cfg(test)]
mod tests;
