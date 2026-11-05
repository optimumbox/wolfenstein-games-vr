//! Waits for: a Monaka ready-made frame rate (a game's rate following what the PC holds), with a
//! provider for the game's rate and its frame timing.
//!
//! The engine runs one tick a frame, paced to real time at its rate (60), so the rate is the frame
//! rate's cap; a frame rate below the rate slows the whole game down. The player picks a ceiling
//! ([`FRAME_RATE`]); the rate follows what the PC holds, between 60 and that ceiling, so the game
//! never runs slow. Each tick is a whole number of milliseconds, so the rate moves only between
//! the rates whose ticks keep the game's own speed ([`STEPS`]).
//!
//! Every [`WINDOW`] of frames on the game thread ([`frame`], from the view builder), each frame's
//! own time (its period less the frame loop's wait for the tick) says what the PC could hold: the
//! rate drops at once below it, and rises only after [`RISE_WINDOWS`] windows with room to spare
//! ([`Steps`]). That measure sees only the game thread: with room to spare it promises far more
//! than the PC holds (the frame's rendering is on another thread). So whether the frames keep up
//! is the frame rate achieved ([`BEHIND`]): behind, the rate drops to what was achieved
//! ([`CAP_MARGIN`]) and stays under it for [`BACKOFF`], doubled with each drop, before a higher
//! rate is tried.
//!
//! On the headset's refresh the game thread's promise is not the whole story either: when the
//! window's present holds the render thread (a swap blocked for most of a refresh), the game thread
//! waits on it and promises no more than the refresh itself, though the frames keep up with it. A
//! rate on the refresh whose frames are already achieved ([`achieved_step`]) is taken too, or the
//! rate stays at 60, the frames are never paced, and the world moves 60 ticks over 90 frames.

use crate::Wolf;
use crate::providers::settings::{self, GameRate};
use monaka_contract::manifest::{Opt, Shown, choice, opt};
use monaka_core::sync::Locked;
use monaka_producer::game::{Component, Lack, Start};
use monaka_producer::ledger::{GameThreadLedger, Setting};
use monaka_producer::log;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::*};
use std::time::{Duration, Instant};

/// The ceiling of the game's rate.
pub const FRAME_RATE: Opt = opt(
    "frame_rate",
    "Frame rate",
    "Frame rates above 60 may result in unstable or buggy gameplay.",
    choice(&[("60", "60 (native)"), ("80", "80"), ("90", "90"), ("96", "96"), ("120", "120"), ("160", "160"), ("240", "240")], "60"),
)
.on(Shown::Game);

/// The game's rate following what the PC holds, up to the player's ceiling.
pub const FRAME_RATE_COMPONENT: Component<Wolf> =
    Component {
        name: "frame rate",
        options: &[FRAME_RATE],
        settings: |options, _| {
            ceiling(options);
        },
        install,
        quiet,
        summary: Some(summary),
        ..Component::NONE
    };

/// The rates the game may take: those whose whole-millisecond tick keeps the game's clock at its
/// own 0.96 of real time (16 ms at 60, 4 ms at 240), and 90 for 90 Hz headsets, whose 11 ms tick
/// runs the clock at 0.99 (3% faster than the game's own; no rate near 90 keeps 0.96).
pub const STEPS: [i32; 9] = [60, 64, 80, 90, 96, 120, 160, 192, 240];
/// The game's own rate, and the lowest the rate goes.
pub const OWN: i32 = 60;
/// A rate is taken only with this much of a frame's time to spare.
pub const MARGIN: f64 = 0.9;
/// Under a measured ceiling (what the PC held while behind) a rate needs only this much to spare.
pub const CAP_MARGIN: f64 = 0.97;
/// The frames are behind the rate when they reach less than this share of it, this many measures
/// in a row (one slow measure is a hitch: a load, a new upscaler, a shader).
pub const BEHIND: f64 = 0.97;
pub const BEHIND_WINDOWS: u32 = 2;
/// Paced to the headset's refresh, a frame late misses a whole refresh: a hitch of two in a measure
/// is already 4% behind, while a PC that cannot hold the rate misses most refreshes. So there the
/// frames are behind only under this share, this many measures in a row (a 60 on a 90 Hz headset
/// shows its frames for one refresh, then two: a judder worse than the hitches).
pub const BEHIND_PACED: f64 = 0.85;
pub const BEHIND_PACED_WINDOWS: u32 = 4;
/// How long a measure runs, and how many in a row with room to spare before the rate rises.
pub const WINDOW: Duration = Duration::from_millis(500);
pub const RISE_WINDOWS: u32 = 4;
/// After a drop the rate stays under the measured ceiling this long, doubled with each drop within
/// [`BACKOFF_MAX`] of the last, at most [`BACKOFF_MAX`].
pub const BACKOFF: Duration = Duration::from_secs(30);
pub const BACKOFF_MAX: Duration = Duration::from_secs(300);
/// Frames further apart than this are a pause or a load, not a measure.
const GAP: Duration = Duration::from_millis(250);

/// The highest step at most `ceiling` that a PC holding `holds` frames a second keeps with
/// [`MARGIN`] to spare; at least [`OWN`].
pub fn fitting(holds: f64, ceiling: i32) -> i32 {
    STEPS.iter().copied().filter(|&step| step <= ceiling && f64::from(step) <= holds * MARGIN && paced(step)).max().unwrap_or(OWN).max(OWN)
}

/// The highest step at most `ceiling` that `holds` frames a second keeps with [`CAP_MARGIN`].
fn under_cap(holds: f64, ceiling: i32) -> i32 {
    STEPS.iter().copied().filter(|&step| step <= ceiling && f64::from(step) <= holds * CAP_MARGIN && paced(step)).max().unwrap_or(OWN).max(OWN)
}

/// The highest step at most `ceiling`, on the headset's refresh, whose frames `achieved` already
/// keeps up with (as a measure not [`BEHIND`] it would); at least [`OWN`].
pub fn achieved_step(achieved: f64, ceiling: i32) -> i32 {
    STEPS.iter().copied().filter(|&step| step <= ceiling && f64::from(step) * BEHIND <= achieved && paced(step)).max().unwrap_or(OWN).max(OWN)
}

/// Whether `step` keeps the frames on the headset's refresh while they are paced to it (every step
/// when they are not): a rate between two fractions of the refresh would drift against it.
fn paced(step: i32) -> bool {
    crate::cooking::frame_pacing::allows(step)
}

/// The rate as measures come: down at once when the frames fall behind it (to what they held then,
/// [`CAP_MARGIN`]), up only after [`RISE_WINDOWS`] measures in a row with room for more, and not
/// above what they held while behind until that ceiling's wait is over ([`BACKOFF`]).
#[derive(Clone, Copy, Debug)]
pub struct Steps {
    pub current: i32,
    rises: u32,
    /// Measures in a row behind the rate.
    behind: u32,
    /// The last time the frames fell behind: what they held, when, and the drops in a row.
    cap: Option<(f64, Instant, u32)>,
}

impl Steps {
    pub const fn new(current: i32) -> Self {
        Self { current, rises: 0, behind: 0, cap: None }
    }

    /// The rate after a measure at `now`: the game thread's promise (`holds` frames a second), the
    /// frames `achieved` a second, up to `ceiling`.
    pub fn next(&mut self, holds: f64, achieved: f64, ceiling: i32, now: Instant) -> i32 {
        self.step(holds, achieved, ceiling, now, crate::cooking::frame_pacing::headset_period().is_some())
    }

    /// [`Steps::next`], the frames paced to the headset's refresh or not (`on_grid`).
    fn step(&mut self, holds: f64, achieved: f64, ceiling: i32, now: Instant, on_grid: bool) -> i32 {
        // A rate off the headset's refresh (taken before the headset's timing came) is behind by
        // its nature: straight to the best rate on it, nothing learnt.
        if self.current != OWN && !paced(self.current) {
            (self.rises, self.behind) = (0, 0);
            self.current = fitting(holds, ceiling);
            return self.current;
        }
        // Behind at the game's own rate there is nowhere lower to go: nothing is learnt.
        let (share, windows) = if on_grid { (BEHIND_PACED, BEHIND_PACED_WINDOWS) } else { (BEHIND, BEHIND_WINDOWS) };
        let behind = achieved < f64::from(self.current) * share && self.current > OWN;
        self.behind = if behind { self.behind + 1 } else { 0 };
        if behind && self.behind < windows {
            self.rises = 0;
            return self.current;
        }
        if behind {
            // Behind: what was achieved is the PC's own, whatever the game thread promised.
            self.behind = 0;
            let drops = match self.cap {
                Some((_, at, drops)) if now.saturating_duration_since(at) < BACKOFF_MAX => drops + 1,
                _ => 1,
            };
            self.cap = Some((achieved, now, drops));
            self.rises = 0;
            self.current = under_cap(achieved, ceiling).min(self.current);
            return self.current;
        }
        let mut target = fitting(holds, ceiling);
        if on_grid {
            target = target.max(achieved_step(achieved, ceiling));
        }
        if let Some((held, at, drops)) = self.cap {
            let wait = BACKOFF.saturating_mul(1 << (drops - 1).min(8)).min(BACKOFF_MAX);
            if now.saturating_duration_since(at) < wait {
                target = target.min(under_cap(held, ceiling));
            }
        }
        if target <= self.current {
            self.rises = 0;
            return self.current;
        }
        self.rises += 1;
        if self.rises >= RISE_WINDOWS {
            self.rises = 0;
            self.current = target;
        }
        self.current
    }
}

/// A measure under way: since when, the frames in it, their waits.
struct Measure {
    start: Instant,
    last: Instant,
    frames: u32,
    waited_us: i64,
}

struct Adapt {
    setting: GameRate,
    spinning: usize,
    ceiling: i32,
    own: i32,
    steps: Steps,
    measure: Option<Measure>,
}

static ADAPT: Mutex<Option<Adapt>> = Mutex::new(None);
/// Set while the rate follows the PC (from the game thread's first turn to the stop).
static FOLLOWING: AtomicBool = AtomicBool::new(false);
static GAME_RATE: GameThreadLedger = GameThreadLedger::new();
static CHANGES: AtomicU64 = AtomicU64::new(0);
/// How long the stop waits for the game thread to give the rate back (it builds no view while
/// paused or loading).
const GIVE_BACK_MS: u64 = 1000;
/// Rate changes logged at most.
const LOGGED: u64 = 200;
/// Measures leaving the rate as it was; one in [`HELD_EVERY`] logged while under the ceiling (every
/// 10 s), at most [`HELD_LOGGED`] times.
static HELD: AtomicU64 = AtomicU64::new(0);
const HELD_EVERY: u64 = 20;
const HELD_LOGGED: u64 = 30;

/// The game's rate (ticks, and so frames, a second) at most while VR runs ([`FRAME_RATE`]; 60 is
/// the game's own).
pub(crate) fn ceiling(options: &monaka_core::options::Options) -> i32 {
    options.choice("frame_rate", &[("80", 80), ("90", 90), ("96", 96), ("120", 120), ("160", 160), ("240", 240)], 60)
}

fn install(start: &mut Start<'_, Wolf>) -> Result<(), Lack> {
    let ceiling = ceiling(start.options);
    let setting = GameRate::of(start.build);
    let Some(own) = setting.read() else {
        log!("the game's rate is not where this build keeps it; it stays as it is");
        return Ok(());
    };
    if ceiling <= own {
        return Ok(());
    }
    let spinning = start.build.exe.at(start.build.build.time_spinning);
    *ADAPT.locked() = Some(Adapt { setting, spinning, ceiling, own, steps: Steps::new(own), measure: None });
    GAME_RATE.ask();
    Ok(())
}

/// Once per frame, on the game thread: takes the rate on (keeping the game's own for the stop),
/// gives it back when the stop asks, and between the two follows what the PC holds.
pub fn frame() {
    if GAME_RATE.due() {
        GAME_RATE.on_game_thread(|ledger| {
            if let Some(adapt) = ADAPT.locked().as_ref() {
                ledger.keep(adapt.setting);
                FOLLOWING.store(true, Release);
                log!("the game's rate follows what the PC holds, up to {}", adapt.ceiling);
            }
        });
    }
    if !FOLLOWING.load(Acquire) {
        return;
    }
    let now = Instant::now();
    let mut guard = ADAPT.locked();
    let Some(adapt) = guard.as_mut() else { return };
    // The frame loop's wait, or (its own wait off) the frame pacing's.
    let waited = settings::spinning(adapt.spinning).unwrap_or(0) + crate::cooking::frame_pacing::take_waited_us() as i32;
    let measure = adapt.measure.get_or_insert(Measure { start: now, last: now, frames: 0, waited_us: 0 });
    if now.duration_since(measure.last) > GAP {
        // A pause or a load: start again.
        *measure = Measure { start: now, last: now, frames: 0, waited_us: 0 };
        return;
    }
    measure.last = now;
    measure.frames += 1;
    measure.waited_us += i64::from(waited);
    let span = now.duration_since(measure.start);
    if span < WINDOW || measure.frames < 2 {
        return;
    }
    // Each frame's own time: the measure's span shared out, less the frame loop's wait.
    let own_us = (span.as_micros() as f64 - measure.waited_us as f64) / f64::from(measure.frames);
    let measure_frames = measure.frames;
    adapt.measure = None;
    if own_us <= 0.0 {
        return;
    }
    let holds = 1e6 / own_us;
    let achieved = f64::from(measure_frames) / span.as_secs_f64();
    let before = adapt.steps;
    let rate = if crate::research::options().fixed_rate { adapt.ceiling } else { adapt.steps.next(holds, achieved, adapt.ceiling, now) };
    if rate == before.current {
        let held = HELD.fetch_add(1, Relaxed);
        if rate < adapt.ceiling && held % HELD_EVERY == 0 && held / HELD_EVERY < HELD_LOGGED {
            log!("the game's rate stays {rate} ({achieved:.0} frames a second, the game thread promising {holds:.0})");
        }
        return;
    }
    if !adapt.setting.write(&rate) {
        // Not taken: the rate stays as the game has it.
        adapt.steps = before;
        return;
    }
    if CHANGES.fetch_add(1, Relaxed) < LOGGED {
        log!("the game's rate {} -> {rate} ({achieved:.0} frames a second, the game thread promising {holds:.0})", before.current);
    }
}

fn quiet() {
    FOLLOWING.store(false, Release);
    let Some(adapt) = ADAPT.locked().take() else { return };
    if !GAME_RATE.restore(GIVE_BACK_MS) {
        // No view built meanwhile (a menu, a load): the game's own rate goes back from here.
        log!("no frame to give the game's rate back on; set back to {} from the stop", adapt.own);
        adapt.setting.write(&adapt.own);
    }
}

fn summary() -> String {
    format!("the game's rate changed {} times", CHANGES.load(Relaxed))
}

#[cfg(test)]
mod tests;
