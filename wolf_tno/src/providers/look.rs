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
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering::*};

/// The Wolfensteins look by the command: the head the same-frame loop holds, the player's own pitch
/// kept level only while the game's pad is on; another camera has the view while the game's view
/// camera getter says so; the share held as it was on a mounted gun, in a driven machine and while
/// the weapon wheel is open ([`camera_taken::holds`]).
impl Looks for Wolf {
    const DELIVERY: Delivery<Self> = Delivery::Command { head: || PAIRS.head().current(), levels: || crate::on(crate::GAMEPAD_AT), holds: camera_taken::holds, pitch: Pitch::Own };
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
            command::built(pitch, yaw).or((snapped != 0.0).then_some((pitch, yaw)))
        });
    }
    result
}

