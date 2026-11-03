//! Cooking, waiting for a Monaka ready-made for frames in step with the headset (a game's frame
//! start held to the headset's refresh), with a provider for the game's frame start.
//!
//! The engine paces its frames itself: each frame first waits out the rest of its tick on its own
//! millisecond clock (`engine::Build::frame`, while `com_syncToTime` is on), counting from when it
//! last woke, so a frame that runs over pushes every later one back. That clock is not the
//! headset's: at 60 the game makes 59.7 frames a second on a 120 Hz headset, and every few seconds
//! one frame stays up for a third refresh while everything moving stands still for it.
//!
//! With the headset's refresh from the viewer (`monaka_contract::protocol::DisplayTiming`: how long
//! a headset frame is, and when the viewer last woke for one), the engine's wait is turned off and
//! each frame starts on the headset's grid instead, every [`refreshes_per_frame`] refreshes, so
//! each headset frame gets a new pair at the same place in it. The world still advances one tick a
//! frame, so at a rate that is the refresh or a fraction of it the game keeps its own speed; at any
//! other rate (or with no refresh sent: an older viewer, the headset idle) the engine paces as
//! before. Where in a refresh the finished pairs land is kept halfway between the viewer's wakes
//! ([`Phase`]), so none lands on the edge of one and is taken a refresh early or late.
//!
//! SteamVR halves the refresh it reports while it reprojects (a 90 Hz headset said to be 45 Hz for a
//! moment): the grid is the shortest refresh the headset has reported ([`steady`]), which those
//! moments are whole multiples of, so neither the pacing nor the game's rate follow them.

use crate::Wolf;
use crate::providers::engine::FrameFn;
use crate::providers::settings::GameRate;
use eng_idt::{Cvar, IntCvar};
use monaka_channel::pose::now_ns;
use monaka_contract::protocol::DisplayTiming;
use monaka_core::sync::Locked;
use monaka_hook::module::Module;
use monaka_hook::{InFlight, Original};
use monaka_producer::game::{Component, Lack, Start, place};
use monaka_producer::ledger::Setting;
use monaka_producer::log;
use monaka_stereo::same_frame::PAIRS;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::*};
use std::sync::{Mutex, OnceLock};

/// Each frame started on the headset's refresh.
pub const FRAME_PACING_COMPONENT: Component<Wolf> = Component {
    name: "frame pacing",
    install,
    quiet,
    summary: Some(summary),
    ..Component::NONE
};

pub(crate) const FRAME_PACING_AT: usize = place::<Wolf>(&FRAME_PACING_COMPONENT);

pub static FRAME_ORIGINAL: Original<FrameFn> = Original::new();

/// A refresh older than this (nanoseconds) is no longer the headset's: the viewer stopped drawing.
const STALE_NS: u64 = 100_000_000;
/// A rate is a fraction of the refresh when within this share of one.
const FITS: f64 = 0.005;
/// How far the landing point moves toward halfway each pair (a share of how far off it is).
const PHASE_GAIN: f64 = 0.05;
/// The most refreshes a frame is held for.
const MOST_REFRESHES: u64 = 4;
/// The shortest refresh the headset has reported (nanoseconds; `u64::MAX`: none yet), and the
/// shortest taken as one (a shorter report is not a headset's).
static SHORTEST_NS: AtomicU64 = AtomicU64::new(u64::MAX);
const SHORTEST_TAKEN_NS: u64 = 2_000_000;
/// How often the pacing's tallies since the last are logged (nanoseconds), and how many times.
const WINDOW_NS: u64 = 10_000_000_000;
const WINDOWS_LOGGED: u64 = 60;

/// The executable, `com_syncToTime` and the game's rate, from the install.
static FACTS: OnceLock<(Module, &'static Cvar, GameRate)> = OnceLock::new();
/// Whether the engine's own wait is off now (the frames start on the headset's grid).
static PACING: AtomicBool = AtomicBool::new(false);
/// The latest pair's publish (nanoseconds; 0: none since pacing began).
static PUBLISHED: AtomicU64 = AtomicU64::new(0);
/// How long the last frame waited for its slot (microseconds), for the frame rate's measure.
static WAITED_US: AtomicU64 = AtomicU64::new(0);
static STATE: Mutex<State> = Mutex::new(State::new());

fn install(start: &mut Start<'_, Wolf>) -> Result<(), Lack> {
    let crate::Facts { exe, build } = *start.build;
    if !build.sync_to_time.present(&exe) {
        return Err(Lack::Missing("com_syncToTime is not where this build keeps it".to_owned()));
    }
    // What the engine had goes back at the stop, whatever the pacing left.
    if start.ledger.keep(IntCvar { exe, cvar: &build.sync_to_time }).is_none() {
        return Err(Lack::Missing("com_syncToTime could not be kept".to_owned()));
    }
    let _ = FACTS.set((exe, &build.sync_to_time, GameRate::of(start.build)));
    start.hooks.site(&FRAME_ORIGINAL, &exe, &build.frame, frame)?;
    log!("frame pacing: each frame starts on the headset's refresh while the viewer sends it");
    Ok(())
}

fn quiet() {
    set_pacing(false);
}

/// The headset's refresh period (nanoseconds) as the viewer last said it, while frames may be paced
/// to it (whether or not the game's rate lets them be yet).
pub fn headset_period() -> Option<u64> {
    let timing = crate::on(FRAME_PACING_AT).then(|| PAIRS.head().display()).flatten()?;
    (now_ns().saturating_sub(timing.woke_ns) < STALE_NS && timing.period_ns > 0).then(|| steady(timing).period_ns)
}

/// `timing` on the headset's own refresh: the shortest it has reported, not a reprojecting
/// SteamVR's multiple of it.
fn steady(timing: DisplayTiming) -> DisplayTiming {
    let period = timing.period_ns;
    if period < SHORTEST_TAKEN_NS {
        return timing;
    }
    let shortest = SHORTEST_NS.fetch_min(period, Relaxed).min(period);
    DisplayTiming { period_ns: shortest, ..timing }
}

/// Whether the game's rate `hz` keeps frames on the headset's refresh (any rate when they cannot be
/// paced): the frame rate takes only such rates. Pacing starts only at such a rate, so this asks the
/// headset, not the pacing (96 on a 90 Hz headset is never paced, and shows 90 of its frames).
pub fn allows(hz: i32) -> bool {
    headset_period().is_none_or(|period| refreshes_per_frame(period, hz).is_some())
}

/// How long the last frame waited for its slot (microseconds), taken once.
pub fn take_waited_us() -> u64 {
    WAITED_US.swap(0, Relaxed)
}

/// A pair is being published now (the present, on the render thread).
pub fn published() {
    if PACING.load(Acquire) {
        PUBLISHED.store(now_ns(), Relaxed);
    }
}

/// How many headset refreshes of `period` nanoseconds a frame at `hz` takes, when that is whole.
pub fn refreshes_per_frame(period: u64, hz: i32) -> Option<u64> {
    if period == 0 || hz <= 0 {
        return None;
    }
    let refreshes = 1e9 / period as f64 / f64::from(hz);
    let whole = refreshes.round();
    (whole >= 1.0 && whole <= MOST_REFRESHES as f64 && (refreshes - whole).abs() <= FITS * whole).then_some(whole as u64)
}

/// `value` brought into `[-half, half)` of a cycle of `period`.
fn wrapped(value: i64, period: u64) -> i64 {
    let p = period as i64;
    (value + p / 2).rem_euclid(p) - p / 2
}

/// The slot the next frame starts at on the headset's grid (`timing`, its points moved by
/// `phase`): the grid point nearest `refreshes` refreshes after the `last` slot; or, when that is
/// past `now` by half a refresh or more (or there is no last slot), the first point from `now` on.
/// Whether it came late (the frame before ran over).
pub fn next_slot(timing: &DisplayTiming, phase: u64, last: Option<u64>, refreshes: u64, now: u64) -> (u64, bool) {
    let period = timing.period_ns;
    let origin = timing.woke_ns + phase % period;
    let point_at_or_after = |t: u64| {
        let n = t.saturating_sub(origin).div_ceil(period);
        origin + n * period
    };
    let Some(last) = last else { return (point_at_or_after(now), false) };
    let wanted = last + refreshes * period;
    // The grid point nearest the slot wanted (the grid may have slid a little against it).
    let nearest = point_at_or_after(wanted.saturating_sub(period / 2));
    if nearest + period / 2 <= now {
        return (point_at_or_after(now), true);
    }
    (nearest, false)
}

/// Where the paced frames' pairs land: an offset of the grid's points, moved each pair so its
/// publish lands halfway between two of the viewer's wakes.
pub struct Phase;

impl Phase {
    /// The offset after a pair published at `published`, from `phase`, on the grid `timing`.
    pub fn after(timing: &DisplayTiming, phase: u64, published: u64) -> u64 {
        let period = timing.period_ns;
        let half = period as i64 / 2;
        let off = wrapped(published as i64 - timing.woke_ns as i64 - half, period);
        let moved = phase as i64 - (off as f64 * PHASE_GAIN) as i64;
        moved.rem_euclid(period as i64) as u64
    }

    /// How far a pair published at `published` lands from halfway (nanoseconds, signed).
    pub fn off(timing: &DisplayTiming, published: u64) -> i64 {
        wrapped(published as i64 - timing.woke_ns as i64 - timing.period_ns as i64 / 2, timing.period_ns)
    }
}

/// The pacing's own state and tallies.
struct State {
    period: u64,
    phase: u64,
    last: Option<u64>,
    refreshes: u64,
    frames: u64,
    late: u64,
    /// Starts more than a tenth of a millisecond off their slot, and the most.
    missed_slot: u64,
    worst_us: u64,
    /// Pairs measured for where they landed, how far off halfway (sum, microseconds), the most.
    landed: u64,
    off_sum_us: u64,
    off_worst_us: u64,
    began: u64,
    /// The window logged every [`WINDOW_NS`]: since when, and its tallies.
    window: Window,
    windows_logged: u64,
    /// When the last frame started after its wait (nanoseconds; 0: none).
    last_started: u64,
}

/// The pacing's tallies over one logged window.
#[derive(Clone, Copy)]
struct Window {
    since: u64,
    frames: u64,
    late: u64,
    missed_slot: u64,
    worst_us: u64,
    /// The game thread's longest frame, its wait for the slot left out (microseconds).
    work_worst_us: u64,
    /// Frames whose own work took longer than a refresh.
    work_over: u64,
}

impl Window {
    const fn new(since: u64) -> Self {
        Self { since, frames: 0, late: 0, missed_slot: 0, worst_us: 0, work_worst_us: 0, work_over: 0 }
    }
}

impl State {
    const fn new() -> Self {
        Self { period: 0, phase: 0, last: None, refreshes: 0, frames: 0, late: 0, missed_slot: 0, worst_us: 0, landed: 0, off_sum_us: 0, off_worst_us: 0, began: 0, window: Window::new(0), windows_logged: 0, last_started: 0 }
    }
}

/// The engine's own wait off (`pacing`) or back on.
fn set_pacing(pacing: bool) {
    if PACING.swap(pacing, AcqRel) == pacing {
        return;
    }
    if let Some((exe, cvar, _)) = FACTS.get() {
        cvar.set_int(exe, i32::from(!pacing));
    }
    let mut state = STATE.locked();
    // The paced stretch's tallies, however short, then a fresh window for the next.
    if !pacing {
        let (now, period) = (now_ns(), state.period);
        log_window(&mut state, now, period, true);
        state.window = Window::new(0);
    }
    state.last = None;
    state.last_started = 0;
    PUBLISHED.store(0, Relaxed);
    if state.began < 20 {
        state.began += 1;
        match pacing {
            true => log!("frame pacing: frames start on the headset's refresh ({:.1} Hz, a frame every {} refreshes)", 1e9 / state.period.max(1) as f64, state.refreshes),
            false => log!("frame pacing: the game keeps its own time again"),
        }
    }
}

/// Waits (sleeping, then spinning the last stretch) until `slot` on the performance counter.
fn wait_until(slot: u64) {
    loop {
        let now = now_ns();
        if now >= slot {
            return;
        }
        let left = slot - now;
        if left > 2_000_000 {
            std::thread::sleep(std::time::Duration::from_nanos(left - 1_500_000));
        } else {
            std::thread::yield_now();
        }
    }
}

/// At each frame's start, on the game thread: the frame held to its slot on the headset's grid,
/// or the engine's own wait left to pace it.
fn pace() {
    let Some((_, _, rate)) = FACTS.get() else { return };
    let timing = crate::on(FRAME_PACING_AT).then(|| PAIRS.head().display()).flatten();
    let now = now_ns();
    let fresh = timing.filter(|t| now.saturating_sub(t.woke_ns) < STALE_NS && t.period_ns > 0).map(steady);
    let refreshes = fresh.zip(rate.read()).and_then(|(t, hz)| refreshes_per_frame(t.period_ns, hz));
    // A menu runs on the engine's own clock: held to the headset's grid, it answers seconds late.
    let (Some(timing), Some(refreshes), false) = (fresh, refreshes, crate::providers::eye_site::in_menu()) else {
        set_pacing(false);
        return;
    };
    {
        let mut state = STATE.locked();
        state.period = timing.period_ns;
        state.refreshes = refreshes;
    }
    set_pacing(true);
    let mut state = STATE.locked();
    let published = PUBLISHED.swap(0, Relaxed);
    if published != 0 {
        let off = Phase::off(&timing, published).unsigned_abs() / 1000;
        state.landed += 1;
        state.off_sum_us += off;
        state.off_worst_us = state.off_worst_us.max(off);
        state.phase = Phase::after(&timing, state.phase, published);
    }
    let (slot, late) = next_slot(&timing, state.phase, state.last, refreshes, now);
    if late {
        state.late += 1;
        state.window.late += 1;
    }
    // The frame before's own work: from its start to now.
    if state.last_started != 0 && state.last.is_some() {
        let work_us = now.saturating_sub(state.last_started) / 1000;
        state.window.work_worst_us = state.window.work_worst_us.max(work_us);
        state.window.work_over += u64::from(work_us * 1000 > timing.period_ns);
    }
    state.last = Some(slot);
    state.frames += 1;
    state.window.frames += 1;
    log_window(&mut state, now, timing.period_ns, false);
    drop(state);
    // A slot further off than a frame's refreshes is not this grid's: no wait.
    if slot > now && slot - now <= (refreshes + 1) * timing.period_ns {
        wait_until(slot);
    }
    let started = now_ns();
    WAITED_US.store(started.saturating_sub(now) / 1000, Relaxed);
    let off_slot = started.abs_diff(slot) / 1000;
    let mut state = STATE.locked();
    if off_slot > 100 {
        state.missed_slot += 1;
        state.window.missed_slot += 1;
    }
    state.worst_us = state.worst_us.max(off_slot);
    state.window.worst_us = state.window.worst_us.max(off_slot);
    state.last_started = started;
}

/// Every [`WINDOW_NS`] (or at once when `ending`, the pacing stopping), the window's tallies
/// logged and started afresh.
fn log_window(state: &mut State, now: u64, period: u64, ending: bool) {
    if state.window.since == 0 {
        if !ending {
            state.window = Window::new(now);
        }
        return;
    }
    if now.saturating_sub(state.window.since) < WINDOW_NS && !(ending && state.window.frames > 0) {
        return;
    }
    let w = state.window;
    let seconds = now.saturating_sub(w.since) as f64 / 1e9;
    if state.windows_logged < WINDOWS_LOGGED {
        state.windows_logged += 1;
        log!(
            "frame pacing over {seconds:.0} s: {:.1} frames a second on a {:.1} Hz grid; {} late, {} over 0.1 ms off their slot (the most {} us); the game thread's own work over a refresh {} times, at most {:.2} ms",
            w.frames as f64 / seconds,
            1e9 / period.max(1) as f64,
            w.late,
            w.missed_slot,
            w.worst_us,
            w.work_over,
            w.work_worst_us as f64 / 1000.0,
        );
    }
    state.window = Window::new(now);
}

unsafe extern "system" fn frame(common: usize) -> usize {
    let flight = InFlight::enter();
    pace();
    let original = FRAME_ORIGINAL.get(&flight);
    // The game's frame can hold a whole load (seconds): let go before it, so a stop meanwhile does
    // not wait for it to end (nothing runs here after it).
    drop(flight);
    // SAFETY: forwards the game's own call.
    unsafe { original(common) }
}

fn summary() -> String {
    let s = STATE.locked();
    let mean_off = s.off_sum_us.checked_div(s.landed).unwrap_or(0);
    format!(
        "frame pacing: {} frames started on the headset's refresh ({} late: the frame before ran over; {} started over 0.1 ms off their slot, the most {} us); pairs landed {} us from halfway between the viewer's wakes on average, at most {} us",
        s.frames, s.late, s.missed_slot, s.worst_us, mean_off, s.off_worst_us
    )
}

#[cfg(test)]
mod tests;
