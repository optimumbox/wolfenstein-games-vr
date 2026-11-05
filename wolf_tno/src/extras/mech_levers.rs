//! The mech's aim by its levers: in The New Order's mech the driver pulls both levers back to aim
//! up and pushes them forward to aim down; with this on (`mech_levers`, off by default) the player
//! does the same with both controllers.
//!
//! How far the controllers' midpoint is ahead of the head, along the head's level heading, is
//! measured against where it was a second after the player got in ([`SETTLE_MS`]): the hands'
//! rest on the levers. Measured from the head, leaning and recentring leave it alone. [`THROW`]
//! either way reaches the mech's look limits ([`pitch_for`]); the aim eases there
//! ([`EASE`]). The mech's own update is handed the pitch change that takes it there
//! (`engine::MechAim`), so the driver's arms pull the levers as they do for the stick, and the
//! limits are the mech's own. The right stick still turns it left and right.

use crate::Wolf;
use crate::cooking::head;
use crate::providers::camera_taken;
use crate::providers::engine::{MechAim, MechViewAnglesFn};
use monaka_body::hands::{self, At, In, Point};
use monaka_contract::manifest::{Kind, Opt, Shown, opt};
use monaka_contract::protocol::{HeadPose, LEFT_HAND, RIGHT_HAND};
use monaka_core::math::{Vec3, forward_axis};
use monaka_core::sync::Locked;
use monaka_hook::{InFlight, Original, mem};
use monaka_producer::game::{Component, Lack, OrMissing, Start, Switch};
use monaka_producer::log;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::*};
use std::sync::{Mutex, OnceLock};

pub const MECH_LEVERS_OPTION: Opt = opt(
    "mech_levers",
    "Aim the mech with your hands",
    "Pull both controllers back to aim up, push them forward to aim down. Off: the right controller aims it.",
    Kind::Switch { default: false },
)
.on(Shown::Game)
.dev();

/// The mech aimed up and down by the controllers (`mech_levers`, off by default).
pub const MECH_LEVERS: Component<Wolf> = Component {
    name: "mech_levers",
    games: &["wolf_tno"],
    options: &[MECH_LEVERS_OPTION],
    settings: |options, manifest| {
        SWITCH.on(options, manifest);
    },
    install,
    summary: Some(summary),
    ..Component::NONE
};

/// The mech aimed by the right stick or the right controller (`mech_levers=0`). Read at the
/// install, not as the component's switch: the mech's hook is placed either way, to say when the
/// player is in the mech (`turret_aim`).
pub const SWITCH: Switch = Switch::new("mech_levers", false, "the right stick aims the mech");

/// Whether the levers aim the mech (`mech_levers`), read at the install.
static LEVERS_ON: AtomicBool = AtomicBool::new(false);
/// When the mech's view angles were last updated (tick, ms; 0: never): the player is in the mech.
static MECH_SEEN: AtomicU64 = AtomicU64::new(0);
/// The mech counts as driven this long after its last update (ms).
const SEEN_MS: u64 = 250;

/// Whether the player is in the mech now (its own update ran just now).
pub fn in_mech() -> bool {
    let seen = MECH_SEEN.load(Relaxed);
    seen != 0 && monaka_channel::tick().saturating_sub(seen) <= SEEN_MS
}

/// Whether the levers aim the mech.
pub fn levers_on() -> bool {
    LEVERS_ON.load(Relaxed)
}

/// How long after getting in the hands' rest is taken (ms): time to put them on the levers.
pub const SETTLE_MS: u64 = 1000;
/// How far the hands move from their rest to reach the mech's look limit (metres).
pub const THROW: f32 = 0.15;
/// How much of the way to the hands' pitch the aim goes each command.
pub const EASE: f32 = 0.35;

pub static VIEW_ANGLES_ORIGINAL: Original<MechViewAnglesFn> = Original::new();
static FACTS: OnceLock<MechAim> = OnceLock::new();

/// The levers as the game thread holds them.
struct Levers {
    /// The time in the mech they are for (`camera_taken::drives`), and when it began (tick, ms).
    drive: u64,
    began: u64,
    /// How far ahead of the head the hands rest (metres), once taken.
    rest: Option<f32>,
    /// The pitch the aim is easing toward.
    aim: Option<f32>,
}

static LEVERS: Mutex<Levers> = Mutex::new(Levers { drive: 0, began: 0, rest: None, aim: None });
static AIMED: AtomicU64 = AtomicU64::new(0);
static RESTS: AtomicU64 = AtomicU64::new(0);

fn install(start: &mut Start<'_, Wolf>) -> Result<(), Lack> {
    let crate::Facts { exe, build } = *start.build;
    let facts = build.mech_aim.as_ref().or_missing(&format!("{} has no mech mapped", build.game))?;
    let _ = FACTS.set(*facts);
    let on = SWITCH.on(start.options, start.manifest);
    LEVERS_ON.store(on, Relaxed);
    log!("mech_levers: {}", if on { "the hands aim the mech up and down" } else { "off (the right controller aims the mech)" });
    start.hooks.site(&VIEW_ANGLES_ORIGINAL, &exe, &facts.view_angles, view_angles)?;
    Ok(())
}

fn summary() -> String {
    format!("mech levers: the hands' rest taken {} times, {} commands aimed by the hands", RESTS.load(Relaxed), AIMED.load(Relaxed))
}

/// How far the midpoint of `hands` (tracking space, metres) is ahead of the head `head`, along
/// the head's level heading. None with the head looking straight up or down.
pub fn reach(head: &HeadPose, hands: [Vec3; 2]) -> Option<f32> {
    let forward = forward_axis(head.orientation);
    let level = forward[0].hypot(forward[2]);
    if level < 1e-3 {
        return None;
    }
    let mid: Vec3 = std::array::from_fn(|k| 0.5 * (hands[0][k] + hands[1][k]) - head.position[k]);
    Some((mid[0] * forward[0] + mid[2] * forward[2]) / level)
}

/// The mech's pitch (degrees, down positive) for the hands `push` metres ahead of their rest:
/// [`THROW`] forward reaches `max` (down), as far back reaches `min` (up), level at rest.
pub fn pitch_for(push: f32, min: f32, max: f32) -> f32 {
    let t = (push / THROW).clamp(-1.0, 1.0);
    if t >= 0.0 { t * max.max(0.0) } else { -t * min.min(0.0) }
}

/// Both controllers' positions in the tracking space, if both are tracked.
fn both_hands() -> Option<[Vec3; 2]> {
    let at = |side| hands::pose::<Wolf>(side, Point::Controller, At::Latest, In::Tracking).filter(|p| p.valid != 0).map(|p| p.position);
    Some([at(LEFT_HAND)?, at(RIGHT_HAND)?])
}

/// The pitch change that takes `mech` toward the hands' pitch, or None for the game's own.
fn lever_change(mech: usize) -> Option<f32> {
    if !levers_on() {
        return None;
    }
    let facts = FACTS.get()?;
    let reach = reach(&head::HEAD.current()?, both_hands()?)?;
    let now = monaka_channel::tick();
    let mut levers = LEVERS.locked();
    let drive = camera_taken::drives();
    if levers.drive != drive {
        *levers = Levers { drive, began: now, rest: None, aim: None };
    }
    let rest = match levers.rest {
        Some(rest) => rest,
        None if now.saturating_sub(levers.began) >= SETTLE_MS => {
            levers.rest = Some(reach);
            if RESTS.fetch_add(1, Relaxed) < 20 {
                log!("mech levers: the hands rest {:.0} cm ahead of the head: push them forward to aim down, pull them back to aim up", reach * 100.0);
            }
            reach
        }
        None => return None,
    };
    let (pitch, min, max, scale) = (
        mem::read_finite::<1>(mech + facts.pitch)?[0],
        mem::read_finite::<1>(mech + facts.min_pitch)?[0],
        mem::read_finite::<1>(mech + facts.max_pitch)?[0],
        mem::read_finite::<1>(mech + facts.pitch_scale)?[0],
    );
    if scale.abs() < 1e-3 {
        return None;
    }
    let wanted = pitch_for(reach - rest, min, max);
    let aim = levers.aim.unwrap_or(pitch);
    let aim = aim + (wanted - aim) * EASE;
    levers.aim = Some(aim);
    AIMED.fetch_add(1, Relaxed);
    Some((aim - pitch) / scale)
}

/// The mech's view angles update: the pitch change the hands ask for in place of the
/// commands'; the yaw change as the commands give it.
unsafe extern "system" fn view_angles(mech: usize, pitch: f32, yaw: f32) {
    let flight = InFlight::enter();
    MECH_SEEN.store(monaka_channel::tick().max(1), Relaxed);
    let pitch = lever_change(mech).unwrap_or(pitch);
    // SAFETY: forwards the game's own call, the pitch change ours while the hands aim.
    unsafe { VIEW_ANGLES_ORIGINAL.get(&flight)(mech, pitch, yaw) };
}

#[cfg(test)]
mod tests;
