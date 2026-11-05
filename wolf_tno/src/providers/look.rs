//! Provider: the look (`monaka_body::look::Looks`), delivered through the game's input command
//! (`Delivery::Command`): where each command is built (the usercmd builder, hooked here), the head
//! the same-frame loop holds, levelling only while the game's pad is on, and another camera taking
//! the view ([`camera_taken`]).
//!
//! After the game's usercmd builder has added the keyboard, mouse and stick look input into the
//! player's accumulated view angles, the look takes them (`monaka_body::look::command::built`)
//! and what it returns is written back, with the command's angles repacked from it
//! (`eng_idt::usercmd`, at `engine::COMMANDS`), so movement, weapons and the game camera all
//! follow.

use crate::providers::camera_taken;
use crate::providers::engine::{self, UsercmdBuilderFn};
use crate::{Facts, Wolf};
use monaka_body::look::{Delivery, Looks, Pitch, StandDown, command};
use monaka_core::situation::Situation;
use monaka_hook::module::Module;
use monaka_hook::{InFlight, Original, mem};
use monaka_producer::Rejection;
use monaka_producer::game::{Lack, Start};
use monaka_stereo::same_frame::PAIRS;
use monaka_body::look::Share;
use monaka_core::sync::Locked;
use std::sync::{Mutex, OnceLock};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering::*};

/// The share is held while the game holds it ([`camera_taken::holds`]: a mounted gun, a machine
/// driven, the wheel) and through a short take of the view by another camera ([`TAKEOVER`]: a
/// ladder's mount from the water), not under a control camera ([`camera_taken::grabbed`]) unless
/// the view points up as it begins (the ladder's mount is one; a crate's latch cut looks down),
/// nor past [`HOLD_FOR`] (a scene): those are as they always were.
fn holds() -> bool {
    camera_taken::holds() || TAKEOVER.locked().holding()
}

/// A take of the view the share is held through: another camera's, or a control camera's (a
/// grab) with the view pointing [`CLIMB_UP`] or more up (the ladder's mount from the water).
fn held_take() -> bool {
    !camera_taken::grabbed() || crate::providers::eye_site::render_view_up().is_some_and(|up| up >= CLIMB_UP)
}

/// A grab beginning with the view this far up (degrees) is a ladder's mount (19 to 49 seen); a
/// crate's latch cut begins level or looking down.
pub const CLIMB_UP: f32 = 10.0;

/// The Wolfensteins look by the command: the head the same-frame loop holds, the player's own pitch
/// kept level only while the game's pad is on; another camera has the view while the game's view
/// camera getter says so; the share held as it was on a mounted gun, in a driven machine and while
/// the weapon wheel is open ([`camera_taken::holds`]).
impl Looks for Wolf {
    const DELIVERY: Delivery<Self> = Delivery::Command { head: || PAIRS.head().current(), levels: || crate::on(crate::GAMEPAD_AT), holds: holds, pitch: Pitch::Own };
    const REPORTS: &'static [Situation] = &[Situation::CameraTaken];

    fn install(start: &mut Start<'_, Self>) -> Result<(), Lack> {
        let Facts { exe, build } = *start.build;
        install(&exe, build).map_err(Rejection::revision)?;
        camera_taken::install(&exe, build);
        start.hooks.site(&USERCMD_BUILDER_ORIGINAL, &exe, &build.usercmd_builder, usercmd_builder)?;
        start.hooks.site(&camera_taken::VIEW_CAMERA_ORIGINAL, &exe, &build.view_camera, camera_taken::view_camera)?;
        crate::providers::focus::install(start.hooks, &exe, build)?;
        // The run goes halfway between where the head looks and where the body faces (or the
        // body turns with the player).
        crate::cooking::body_forward::configure(start.options, start.manifest);
        monaka_pad::rewrite(Some(crate::cooking::body_forward::pad));
        Ok(())
    }

    fn stand_down(_: usize) -> StandDown {
        StandDown { now: camera_taken::controlled(), changed: false }
    }
}

pub static USERCMD_BUILDER_ORIGINAL: Original<UsercmdBuilderFn> = Original::new();
static SCALE: OnceLock<f32> = OnceLock::new();
/// The thread that builds commands (the view builder compares its own, for the log).
pub static COMMAND_THREAD: AtomicU32 = AtomicU32::new(0);
static COMMANDS: AtomicU64 = AtomicU64::new(0);

/// The command's angle scale, checked where this build keeps it.
fn install(exe: &Module, build: &engine::Build) -> Result<(), String> {
    let scale = mem::read::<f32>(exe.at(build.angle_to_short)).filter(|s| (s - eng_idt::usercmd::ANGLE_SCALE).abs() < 1e-3);
    let scale = scale.ok_or("the angle-to-short scale is not where this build keeps it")?;
    let _ = SCALE.set(scale);
    Ok(())
}

/// Commands built so far.
pub fn commands() -> u64 {
    COMMANDS.load(Relaxed)
}

pub unsafe extern "system" fn usercmd_builder(generator: usize, out: usize, player: usize, settings: usize, a: usize, b: usize) -> usize {
    let flight = InFlight::enter();
    // SAFETY: forwards the game's own call.
    let result = unsafe { USERCMD_BUILDER_ORIGINAL.get(&flight)(generator, out, player, settings, a, b) };
    COMMANDS.fetch_add(1, Relaxed);
    COMMAND_THREAD.store(monaka_hook::thread_id(), Relaxed);
    if let Some(&scale) = SCALE.get() {
        // A snap turn goes in as the player's own turn (yaw positive left), under the head's share.
        let snapped = crate::cooking::body_forward::take_snapped();
        engine::COMMANDS.rewrite(generator, out, player as u32 as usize, scale, |pitch, yaw| {
            let yaw = yaw - snapped;
            let taken = camera_taken::controlled() && !camera_taken::mounted();
            let (pitch, yaw) = TAKEOVER.locked().command(pitch, yaw, taken, taken && held_take(), monaka_body::look::baked::<Wolf>(None));
            let built = command::built(pitch, yaw).or((snapped != 0.0).then_some((pitch, yaw)));
            // On a mounted gun, turned toward where the right controller points.
            let (p, y) = built.unwrap_or((pitch, yaw));
            let out = crate::cooking::turret_aim::steer(p, y).or(built);
            TAKEOVER.locked().sent(out.unwrap_or((p, y)));
            out
        });
    }
    result
}


/// Another camera taking the view (a ladder's mount, a grab, a scene): the head's share stays in
/// the player's angles as it was, so the game sees no turn the size of the head's at the take (a
/// ladder taken from the water let go of at once with the head's pitch pulled out). If the game
/// set the angles meanwhile (a scene ending where it faces), they hold no share at the end: the
/// held one is put back in for the look to take out, and the head's goes on them.
pub static TAKEOVER: Mutex<Takeover> = Mutex::new(Takeover::new());

/// The angles the game set at the end of a takeover differ from the last sent by more than this
/// (degrees).
pub const SET_BY_GAME: f32 = 2.0;

/// A take of the view longer than this is a scene's, not a ladder's mount (1.7 s): the share is
/// let go of as it always was.
pub const HOLD_FOR: std::time::Duration = std::time::Duration::from_secs(3);

pub struct Takeover {
    /// The share the angles held before the take, while not taken.
    before: Option<Share>,
    taken: bool,
    /// The angles last sent (pitch, yaw).
    last: Option<(f32, f32)>,
    /// Takeovers ended with the angles as left, and with the game's own.
    kept: u64,
    set: u64,
    /// When the take began, while taken.
    since: Option<std::time::Instant>,
    /// The share held through this take (decided as it began).
    hold: bool,
    /// Grabs held through (a ladder's mount).
    grabs: u64,
}

impl Takeover {
    pub const fn new() -> Self {
        Self { before: None, taken: false, last: None, kept: 0, set: 0, since: None, hold: false, grabs: 0 }
    }

    /// A command's angles as built (pitch, yaw) with another camera on the view (`taken`) or not,
    /// whether the share is held through it if it begins now (`hold`), the look's share now
    /// (`share`): as the look should take them.
    pub fn command(&mut self, pitch: f32, yaw: f32, taken: bool, hold: bool, share: Option<Share>) -> (f32, f32) {
        let was = std::mem::replace(&mut self.taken, taken);
        if taken {
            if !was {
                self.since = Some(std::time::Instant::now());
                self.hold = hold;
                if hold && camera_taken::grabbed() {
                    self.grabs += 1;
                    if self.grabs <= 20 {
                        monaka_producer::log!("look: a grab begins with the view pointing up (a ladder's mount): the head's share held through it");
                    }
                }
            }
            return (pitch, yaw);
        }
        let short = self.since.take().is_some_and(|since| since.elapsed() < HOLD_FOR);
        let held_now = std::mem::take(&mut self.hold);
        let mut out = (pitch, yaw);
        if was && short && held_now && let Some(held) = self.before {
            let set = self.last.is_some_and(|(p, y)| (pitch - p).abs() > SET_BY_GAME || crate::cooking::turret_aim::wrapped(yaw, y).abs() > SET_BY_GAME);
            if set {
                out = command::reshare(pitch, yaw, Share::default(), held);
                self.set += 1;
            } else {
                self.kept += 1;
            }
            if self.kept + self.set <= 40 {
                monaka_producer::log!(
                    "look: another camera let go of the view; the angles {} (the head's share held {:.1} yaw, {:.1} pitch degrees)",
                    if set { "set by the game: the share goes on them" } else { "as left: the share stays in them" },
                    held.yaw.to_degrees(),
                    held.pitch.to_degrees()
                );
            }
        }
        if let Some(share) = share {
            self.before = Some(share);
        }
        out
    }

    /// Whether the share is held now: taken, for no longer than [`HOLD_FOR`].
    pub fn holding(&self) -> bool {
        self.taken && self.hold && self.since.is_some_and(|since| since.elapsed() < HOLD_FOR)
    }

    pub fn sent(&mut self, angles: (f32, f32)) {
        self.last = Some(angles);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_takeover_keeps_the_share_and_puts_it_back_on_angles_the_game_set() {
        let share = Share { yaw: 0.5, pitch: 0.3 };
        let mut t = Takeover::new();
        assert_eq!(t.command(-10.0, 40.0, false, false, Some(share)), (-10.0, 40.0));
        t.sent((-10.0, 40.0));
        // Taken: the angles go through as they are.
        assert_eq!(t.command(-10.0, 40.0, true, true, None), (-10.0, 40.0));
        assert!(t.holding());
        t.sent((-10.0, 40.0));
        // Let go with the angles as left: the share is still in them.
        assert_eq!(t.command(-10.0, 40.0, false, false, Some(share)), (-10.0, 40.0));
        t.sent((-10.0, 40.0));
        // A take not held (a crate's latch cut): nothing put back at its end.
        t.command(-10.0, 40.0, true, false, None);
        assert!(!t.holding());
        assert_eq!(t.command(5.0, 90.0, false, false, None), (5.0, 90.0));
        t.sent((-10.0, 40.0));
        // Let go with the angles the game set: the held share is put in for the look to take out.
        t.command(-10.0, 40.0, true, true, None);
        let (pitch, yaw) = t.command(5.0, 90.0, false, false, None);
        let (held_yaw, held_pitch) = (share.yaw.to_degrees(), -share.pitch.to_degrees());
        assert!((pitch - (5.0 + held_pitch)).abs() < 1e-3 && (yaw - (90.0 + held_yaw)).abs() < 1e-3, "{pitch} {yaw}");
    }
}
