//! Cooking, waiting for a step hook in the engine's player physics: room-scale following
//! (`room_scale`, `monaka_body::roomscale`) by the game's own movement input. The character walks
//! to where the player stands in the room, so the eyes and arms never leave the body.
//!
//! The Wolfensteins move the player by the command's movement, from the pad's left stick, so the
//! following velocity goes in as a push of the left stick ([`pad`]) along the game camera's
//! heading, and the game walks the body with its own collision (walls, props and enemies stop it).
//! Each view build is a step ([`step`]): the body's move since the last (the game camera's place)
//! goes whole into the following's anchor while the push walked it, braking included
//! (`roomscale::moved`), and the next push is asked for. The eyes are built on the game camera moved
//! back by that anchor (`roomscale::follow_view`, in `providers::eye_site`), so the view moves with
//! the head alone.
//!
//! Waits while the player walks with the stick themselves, while the player's own camera is not the
//! view (a cutscene, another camera), on a mounted gun, in the mech, while a menu or the weapon
//! wheel holds the player, and while the view turns (a turn in place swings the eyes about the neck:
//! not a step).
//!
//! The game walks only on a push past its own dead zone, and then at a walk, so following pushes
//! only once the gap is a step's ([`START_SPEED`]) and keeps on until it has nearly closed
//! ([`STOP_SPEED`]): small gaps pushed at once lurch the body instead of walking it.

use crate::Wolf;
use crate::providers::camera_taken;
use monaka_body::feature::Steps;
use monaka_body::look;
use monaka_body::roomscale::{self, Wait};
use monaka_contract::gamepad::Gamepad;
use monaka_core::camera::Mat34;
use monaka_core::sync::Locked;
use monaka_producer::game::{Component, Lack, Start};
use monaka_producer::log;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering::*};
use std::time::{Duration, Instant};

/// Room-scale following (`room_scale`, on by default with the hand rig and a tracked head).
pub const ROOM_SCALE: Component<Wolf> = monaka_body::feature::room_scale::<Wolf>();

impl Steps for Wolf {
    /// The step is the view build ([`step`]); nothing to hook.
    fn install(_: &mut Start<'_, Self>) -> Result<(), Lack> {
        Ok(())
    }

    fn summary() -> String {
        let s = STEP.locked();
        format!("room-scale pushes: {} steps pushing, {:.2} m walked by them, {} steps held while the view turned", s.pushing, s.walked, s.turns_held)
    }
}

/// The left stick's dead zone as the game reads it (XInput's own, `XINPUT_GAMEPAD_LEFT_THUMB_DEADZONE`).
const STICK_DEAD_ZONE: f32 = 7849.0 / 32767.0;
/// The speed (m/s) the game walks the player at with the stick fully pushed: a push asks for its
/// share of it, past the dead zone.
const FULL_SPEED: f32 = 4.0;
/// A move this long after the last push still came from it (the game braking).
const BRAKING: Duration = Duration::from_millis(300);
/// The player's own push of the left stick holds following off this long after.
const OWN_STICK_HOLD_MS: u64 = 300;
/// A push not asked for again this long is let go (no view builds: a load, a pause).
const PUSH_FRESH: Duration = Duration::from_millis(150);
/// The body jumping this far between view builds was put somewhere (a load, a checkpoint), not
/// walked (metres).
const JUMP: f32 = 0.5;
/// How often a line on the pushes is logged while they walk (seconds).
const LOG_EVERY: f32 = 10.0;
/// Following starts pushing at this speed asked (m/s: a gap of about 12 cm at the default
/// response) and stops under [`STOP_SPEED`] (about 4 cm).
const START_SPEED: f32 = 0.75;
const STOP_SPEED: f32 = 0.1;
/// The view turning faster than this (radians a second) is a turn, not a step: following waits
/// until [`TURN_SETTLE`] after it slows.
const TURNING: f32 = 60.0 * std::f32::consts::PI / 180.0;
const TURN_SETTLE: Duration = Duration::from_millis(250);

struct Step {
    /// The body's place (the game camera's, metres) at the last view build, and when.
    body: Option<([f32; 3], Instant)>,
    /// The stick push asked for (x right, y forward, -1..=1), when, and when the last non-zero one was.
    push: ([f32; 2], Option<Instant>),
    pushed: Option<Instant>,
    pushing: u64,
    walked: f32,
    /// Since the last line: speed asked and walked (m/s summed), the steps, and when.
    asked_sum: f32,
    walked_sum: f32,
    samples: u32,
    logged: Option<Instant>,
    /// The camera's heading at the last view build (radians), when it last turned fast, and
    /// whether a push is under way (between [`START_SPEED`] and [`STOP_SPEED`]).
    heading: Option<f32>,
    turned: Option<Instant>,
    following: bool,
    turns_held: u64,
}

static STEP: Mutex<Step> = Mutex::new(Step {
    body: None,
    push: ([0.0; 2], None),
    pushed: None,
    pushing: 0,
    walked: 0.0,
    asked_sum: 0.0,
    walked_sum: 0.0,
    samples: 0,
    logged: None,
    heading: None,
    turned: None,
    following: false,
    turns_held: 0,
});

/// When the player last pushed the left stick themselves (`monaka_channel::tick`).
static OWN_STICK: AtomicU64 = AtomicU64::new(0);

/// Why following waits now, if it does.
fn waiting() -> Option<Wait> {
    if look::baked::<Wolf>(None).is_none() || camera_taken::controlled() || camera_taken::camera_away() {
        return Some(Wait::Cutscene);
    }
    if camera_taken::holds() || camera_taken::mounted() {
        return Some(Wait::Attached);
    }
    let own = OWN_STICK.load(Relaxed);
    if own != 0 && monaka_channel::tick().saturating_sub(own) < OWN_STICK_HOLD_MS {
        // The player walks with the stick: that move is theirs, not following's.
        return Some(Wait::Animated);
    }
    None
}

/// A view build of the player's own camera `camera` (canonical, metres): the body's move since the
/// last credited while the push walked it, and the next push asked for.
pub fn step(camera: &Mat34) {
    if !roomscale::active() {
        return;
    }
    let now = Instant::now();
    let body = [camera[3], camera[7], camera[11]];
    let mut waiting = waiting();
    let mut s = STEP.locked();
    // A turn of the view (the head turning the body, the stick): the eyes swing about the neck.
    let heading = camera[2].atan2(camera[10]);
    if let (Some(last), Some((_, at))) = (s.heading, s.body) {
        let dt = now.duration_since(at).as_secs_f32();
        let turn = (heading - last + std::f32::consts::PI).rem_euclid(std::f32::consts::TAU) - std::f32::consts::PI;
        if dt > 0.0 && turn.abs() / dt > TURNING {
            s.turned = Some(now);
        }
    }
    s.heading = Some(heading);
    if waiting.is_none() && s.turned.is_some_and(|at| now.duration_since(at) < TURN_SETTLE) {
        waiting = Some(Wait::Animated);
        s.turns_held += 1;
    }
    if let Some((before, at)) = s.body {
        let moved = [body[0] - before[0], 0.0, body[2] - before[2]];
        let distance = moved[0].hypot(moved[2]);
        let dt = now.duration_since(at).as_secs_f32();
        let walking = s.pushed.is_some_and(|pushed| now.duration_since(pushed) < BRAKING);
        if walking && distance < JUMP && waiting.is_none_or(|w| w != Wait::Animated) {
            roomscale::moved(moved);
            s.walked += distance;
            if dt > 0.0 && s.push.0 != [0.0; 2] {
                s.walked_sum += distance / dt;
                s.samples += 1;
            }
        }
    }
    s.body = Some((body, now));
    let follow = roomscale::follow(waiting);
    let speed = follow[0].hypot(follow[1]);
    s.following = match s.following {
        true => speed >= STOP_SPEED,
        false => speed >= START_SPEED,
    };
    let push = if s.following { push_for(follow, camera) } else { [0.0; 2] };
    if push != [0.0; 2] {
        s.pushed = Some(now);
        s.pushing += 1;
        s.asked_sum += follow[0].hypot(follow[1]);
    }
    s.push = (push, Some(now));
    if s.samples > 0 && s.logged.is_none_or(|at| now.duration_since(at).as_secs_f32() >= LOG_EVERY) {
        let n = s.samples as f32;
        log!("room scale: the pushes asked {:.2} m/s on average and walked the body {:.2} m/s ({} steps)", s.asked_sum / n, s.walked_sum / n, s.samples);
        (s.asked_sum, s.walked_sum, s.samples, s.logged) = (0.0, 0.0, 0, Some(now));
    }
}

/// The left stick push (x right, y forward, -1..=1) that walks the body at `follow` (world x and z,
/// m/s) for the game camera `camera` (columns right, up, back): along the camera's heading on the
/// ground, its share of [`FULL_SPEED`] past the dead zone.
pub fn push_for(follow: [f32; 2], camera: &Mat34) -> [f32; 2] {
    let speed = follow[0].hypot(follow[1]);
    if speed < 1e-3 || !speed.is_finite() {
        return [0.0; 2];
    }
    let flat = |x: f32, z: f32| {
        let length = x.hypot(z);
        (length > 1e-4).then(|| [x / length, z / length])
    };
    let (Some(right), Some(back)) = (flat(camera[0], camera[8]), flat(camera[2], camera[10])) else { return [0.0; 2] };
    let [x, y] = [(follow[0] * right[0] + follow[1] * right[1]) / speed, -(follow[0] * back[0] + follow[1] * back[1]) / speed];
    let deflection = STICK_DEAD_ZONE + (1.0 - STICK_DEAD_ZONE) * (speed / FULL_SPEED).min(1.0);
    [x * deflection, y * deflection]
}

/// The pad the game reads, last: with the player's own left stick at rest, the push following asks
/// for; pushed, it is theirs (and following waits).
pub fn pad(pad: &mut Gamepad) {
    if !roomscale::active() {
        return;
    }
    let own = f32::from(pad.thumb_lx).hypot(f32::from(pad.thumb_ly)) / 32767.0;
    if own > STICK_DEAD_ZONE {
        OWN_STICK.store(monaka_channel::tick(), Relaxed);
        return;
    }
    let (push, at) = STEP.locked().push;
    if push == [0.0; 2] || at.is_none_or(|at| at.elapsed() > PUSH_FRESH) {
        return;
    }
    let axis = |v: f32| (v.clamp(-1.0, 1.0) * 32767.0).round() as i16;
    (pad.thumb_lx, pad.thumb_ly) = (axis(push[0]), axis(push[1]));
}

#[cfg(test)]
mod tests;
