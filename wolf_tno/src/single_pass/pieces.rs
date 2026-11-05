//! The game's GPU work in pieces: timed on the GPU draw by draw, and flushed so no piece runs long.
//!
//! The viewer's few hundred microseconds a refresh go to the GPU between the game's command
//! buffers, not inside one: a buffer the game's driver sends whole runs to its end first. With
//! the larger texture cache ("VT cache" high) one stretch of each frame went as a single 2.0-2.9 ms
//! buffer (none longer than 0.8 ms on low), so the viewer's frame waited behind it, reached
//! SteamVR late and was throttled to half rate. Each piece of the game's work here is flushed once
//! its expected GPU time would pass [`BUDGET_NS`], so the driver sends shorter buffers and the
//! viewer's work goes between them.
//!
//! The expectation is learnt from the GPU itself: a timestamp is written before each of the game's
//! draws and at the swap, read back a few frames later, and the time from one stamp to the next
//! (the draw's work and whatever came before the next one, idle time included) kept per program.
//! The flushing is off ([`SPLIT`]): with it the viewer's GPU time as SteamVR saw it went from about
//! 2 ms to 9-18 ms and SteamVR throttled it more (each flush another switch between the game's
//! work and the viewer's), where the same cache without it ran at full rate.
//!
//! Only one frame in [`SAMPLE`] is stamped: a stamp before every draw (about two thousand a frame)
//! and their reading back cost the render thread enough to make the game miss refreshes. The
//! flushes go on every frame from what the sampled frames taught.
//!
//! Every [`WINDOW`] the costliest programs, the longest single stretch and the flushes added are
//! logged.

use super::gl_api;
use monaka_producer::log;
use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering::Relaxed};
use std::time::{Duration, Instant};

/// A piece's expected GPU time before the game's commands are flushed.
pub const BUDGET_NS: f64 = 500_000.0;
/// Whether the GPU pieces are timed at all: off, as a timed frame (two thousand stamps written and
/// read back) costs the render thread enough to drop refreshes. On for a run that needs them.
const TIMING: bool = false;
/// Whether the game's commands are flushed into pieces at all (else only timed).
const SPLIT: bool = false;
/// A draw expected to cost less than this is not flushed ahead of on its own (a flush between two
/// cheap draws buys nothing).
const ALONE_NS: f64 = 50_000.0;
/// Frames of stamps in flight before they are read.
const RING: usize = 4;
/// Stamps a frame at most; the rest of a longer frame goes untimed.
const STAMPS: usize = 2048;
const WINDOW: Duration = Duration::from_secs(10);
/// How much of a program's expected cost each new frame's measure takes.
const LEARN: f64 = 0.2;
/// One frame in this many is stamped (and later read); the rest only flush.
const SAMPLE: u64 = 16;

const GL_TIMESTAMP: u32 = 0x8E28;
const GL_VIEWPORT: u32 = 0x0BA2;
const GL_QUERY_RESULT: u32 = 0x8866;
const GL_QUERY_RESULT_AVAILABLE: u32 = 0x8867;

type GenQueriesFn = unsafe extern "system" fn(i32, *mut u32);
type QueryCounterFn = unsafe extern "system" fn(u32, u32);
type GetQueryObjectUi64Fn = unsafe extern "system" fn(u32, u32, *mut u64);
type GetQueryObjectIvFn = unsafe extern "system" fn(u32, u32, *mut i32);

/// What a stamp follows: the stretch from it to the next stamp is put down to it.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Mark {
    /// A draw of this program (the draw's work, and what comes before the next draw).
    Draw(u32),
    /// The swap: the frame's capture and its hand-off to the headset (DLSS, the copies).
    Swap,
    /// The hand-off done: the next frame's start until its first draw.
    Handed,
    /// One of the game's GL calls that moves data on the GPU without drawing (an upload, a copy,
    /// a read back, a buffer mapped, a fence), by name.
    Call(&'static str),
}

struct Frame {
    queries: Vec<u32>,
    marks: Vec<Mark>,
}

struct Gl {
    query_counter: QueryCounterFn,
    result_u64: GetQueryObjectUi64Fn,
    result_iv: GetQueryObjectIvFn,
}

#[derive(Default, Clone, Copy)]
struct Tally {
    ns: f64,
    count: u64,
    longest: f64,
}

struct Pieces {
    gl: Gl,
    frames: Vec<Frame>,
    at: usize,
    /// Each program's expected GPU time a draw (nanoseconds), learnt from the stamps.
    expected: HashMap<Mark, f64>,
    /// The expected GPU time sent since the last flush.
    piece: f64,
    /// Swaps seen, to pick the frames stamped.
    swaps: u64,
    /// The frame under way is stamped.
    timing: bool,
    window: Window,
}

struct Window {
    start: Instant,
    frames: u64,
    by_mark: HashMap<Mark, Tally>,
    /// Each frame's whole time (first stamp to last) and its longest stretch with what it followed.
    frame_ns: f64,
    longest: (f64, Option<Mark>),
    over_ms: u64,
    flushes: u64,
    /// The viewport each program drew at (its first timed draw in the window).
    sizes: HashMap<u32, (i32, i32)>,
    /// Swaps in the window (every frame, timed or not): the flushes' divisor.
    swaps: u64,
}

impl Window {
    fn new() -> Self {
        Self { start: Instant::now(), frames: 0, by_mark: HashMap::new(), frame_ns: 0.0, longest: (0.0, None), over_ms: 0, flushes: 0, sizes: HashMap::new(), swaps: 0 }
    }
}

static PIECES: Mutex<Option<Pieces>> = Mutex::new(None);
/// Stamps and flushes happen (pairs published, the GL calls found).
static ON: AtomicBool = AtomicBool::new(false);
/// The program the game last made current.
static PROGRAM: AtomicU32 = AtomicU32::new(0);
/// The render thread (the one that swaps): the game's other threads' GL calls go to other
/// contexts, where our queries are not, and are not stamped.
static RENDER_THREAD: AtomicU32 = AtomicU32::new(0);

fn on_render_thread() -> bool {
    // SAFETY: a plain query of the calling thread's id.
    RENDER_THREAD.load(Relaxed) == unsafe { windows::Win32::System::Threading::GetCurrentThreadId() }
}

/// Set once the GL calls were looked for and not all found.
static MISSING: AtomicBool = AtomicBool::new(false);

/// The game made `program` current (from its `glUseProgram`).
pub fn program(program: u32) {
    PROGRAM.store(program, Relaxed);
}

/// Before one of the game's draws goes to the driver (render thread): flushed first if it would
/// make the piece too long, then stamped.
pub fn before_draw() {
    if !ON.load(Relaxed) || !on_render_thread() {
        return;
    }
    let program = PROGRAM.load(Relaxed);
    let mark = Mark::Draw(program);
    let Ok(mut held) = PIECES.lock() else { return };
    let Some(p) = held.as_mut() else { return };
    let cost = p.expected.get(&mark).copied().unwrap_or(0.0);
    if SPLIT && splits(p.piece, cost) {
        // SAFETY: the game's context is current on its render thread.
        unsafe { windows::Win32::Graphics::OpenGL::glFlush() };
        p.piece = 0.0;
        p.window.flushes += 1;
    }
    p.piece += cost;
    if p.timing && !p.window.sizes.contains_key(&program) {
        let mut viewport = [0i32; 4];
        // SAFETY: four integers into a local, on the render thread with the game's context current.
        unsafe { windows::Win32::Graphics::OpenGL::glGetIntegerv(GL_VIEWPORT, viewport.as_mut_ptr()) };
        p.window.sizes.insert(program, (viewport[2], viewport[3]));
    }
    p.stamp(mark);
}

/// Whether the frame after swap number `swaps` is stamped.
pub fn sampled(swaps: u64) -> bool {
    swaps % SAMPLE == 0
}

/// Whether to flush ahead of a draw expected to cost `cost` with `piece` already sent unflushed.
pub fn splits(piece: f64, cost: f64) -> bool {
    piece > 0.0 && piece + cost > BUDGET_NS && (cost >= ALONE_NS || piece >= BUDGET_NS)
}

/// At the swap, before the frame is captured and handed off (render thread): `publishing`, whether
/// pairs go out (nothing is stamped or flushed otherwise). The frame's stamps go down, the oldest
/// frame's are read, and the window's summary is logged when due, with `name` for converted
/// programs.
pub fn swap(publishing: bool, name: impl Fn(u32) -> Option<&'static str>) {
    if !TIMING || !publishing {
        ON.store(false, Relaxed);
        return;
    }
    // SAFETY: a plain query of the calling thread's id.
    RENDER_THREAD.store(unsafe { windows::Win32::System::Threading::GetCurrentThreadId() }, Relaxed);
    let Ok(mut held) = PIECES.lock() else { return };
    if held.is_none() {
        if MISSING.load(Relaxed) {
            return;
        }
        match Pieces::new() {
            Some(made) => *held = Some(made),
            None => {
                MISSING.store(true, Relaxed);
                log!("GPU pieces: off: the driver has no timestamp queries");
                return;
            }
        }
    }
    let p = held.as_mut().expect("made above");
    ON.store(true, Relaxed);
    p.stamp(Mark::Swap);
    // The piece ends at the swap: the hand-off flushes the game's commands itself.
    p.piece = 0.0;
    p.at = (p.at + 1) % RING;
    p.read_oldest();
    p.swaps += 1;
    p.window.swaps += 1;
    p.timing = sampled(p.swaps);
    // The new frame's stamps begin at the same swap, so its hand-off is timed with it.
    p.stamp(Mark::Swap);
    if p.window.start.elapsed() >= WINDOW {
        let window = std::mem::replace(&mut p.window, Window::new());
        if let Some(line) = describe(&window, &name) {
            log!("{line}");
        }
    }
}

/// Before one of the game's GL calls that is not a draw (render thread): stamped, on timed frames.
pub fn call(name: &'static str) {
    if !ON.load(Relaxed) || !on_render_thread() {
        return;
    }
    let Ok(mut held) = PIECES.lock() else { return };
    if let Some(p) = held.as_mut() {
        p.stamp(Mark::Call(name));
    }
}

/// The hand-off done (render thread): what follows is the next frame's start.
pub fn handed() {
    if !ON.load(Relaxed) {
        return;
    }
    let Ok(mut held) = PIECES.lock() else { return };
    if let Some(p) = held.as_mut() {
        p.stamp(Mark::Handed);
    }
}

impl Pieces {
    fn new() -> Option<Self> {
        let load = |name: &str| gl_api::address(name);
        // SAFETY: the driver's function of that name, of the type its specification gives.
        let gen_queries = unsafe { std::mem::transmute::<usize, GenQueriesFn>(load("glGenQueries")?) };
        let gl = Gl {
            // SAFETY (each): the driver's function of that name, of the type its specification gives.
            query_counter: unsafe { std::mem::transmute::<usize, QueryCounterFn>(load("glQueryCounter")?) },
            result_u64: unsafe { std::mem::transmute::<usize, GetQueryObjectUi64Fn>(load("glGetQueryObjectui64v")?) },
            result_iv: unsafe { std::mem::transmute::<usize, GetQueryObjectIvFn>(load("glGetQueryObjectiv")?) },
        };
        let frames = (0..RING)
            .map(|_| {
                let mut queries = vec![0u32; STAMPS];
                // SAFETY: fills `queries` with new query names on the current context.
                unsafe { gen_queries(STAMPS as i32, queries.as_mut_ptr()) };
                Frame { queries, marks: Vec::with_capacity(STAMPS) }
            })
            .collect();
        Some(Self { gl, frames, at: 0, expected: HashMap::new(), piece: 0.0, swaps: 0, timing: false, window: Window::new() })
    }

    fn stamp(&mut self, mark: Mark) {
        if !self.timing {
            return;
        }
        let frame = &mut self.frames[self.at];
        let Some(&query) = frame.queries.get(frame.marks.len()) else { return };
        // SAFETY: one of our query names, given the GPU's time when the commands before it are done.
        unsafe { (self.gl.query_counter)(query, GL_TIMESTAMP) };
        frame.marks.push(mark);
    }

    /// The frame stamped [`RING`] - 1 swaps ago (the one about to be stamped again), read if its
    /// last stamp is in; else dropped.
    fn read_oldest(&mut self) {
        let frame = &mut self.frames[self.at];
        let marks = std::mem::take(&mut frame.marks);
        let (Some(&last), true) = (marks.len().checked_sub(1).and_then(|i| frame.queries.get(i)), marks.len() > 1) else { return };
        let mut available = 0;
        // SAFETY: our query name and a local.
        unsafe { (self.gl.result_iv)(last, GL_QUERY_RESULT_AVAILABLE, &mut available) };
        if available == 0 {
            return;
        }
        let mut times = vec![0u64; marks.len()];
        for (query, time) in frame.queries.iter().zip(times.iter_mut()) {
            // SAFETY: our query names (all in, the last one being), into locals.
            unsafe { (self.gl.result_u64)(*query, GL_QUERY_RESULT, time) };
        }
        let stretches = stretches(&marks, &times);
        let mut frame_cost: HashMap<Mark, (f64, u64)> = HashMap::new();
        let w = &mut self.window;
        w.frames += 1;
        w.frame_ns += times[times.len() - 1].saturating_sub(times[0]) as f64;
        for &(mark, ns) in &stretches {
            let t = w.by_mark.entry(mark).or_default();
            t.ns += ns;
            t.count += 1;
            t.longest = t.longest.max(ns);
            if ns > w.longest.0 {
                w.longest = (ns, Some(mark));
            }
            w.over_ms += u64::from(ns > 1_000_000.0);
            let c = frame_cost.entry(mark).or_default();
            c.0 += ns;
            c.1 += 1;
        }
        for (mark, (ns, n)) in frame_cost {
            let measured = ns / n as f64;
            let e = self.expected.entry(mark).or_insert(measured);
            *e += (measured - *e) * LEARN;
        }
    }
}

/// Each stamp's stretch to the next: what it followed and its GPU time (nanoseconds).
pub fn stretches(marks: &[Mark], times: &[u64]) -> Vec<(Mark, f64)> {
    marks.iter().zip(times.windows(2)).map(|(&mark, t)| (mark, t[1].saturating_sub(t[0]) as f64)).collect()
}

fn label(mark: Mark, name: &impl Fn(u32) -> Option<&'static str>) -> String {
    match mark {
        Mark::Draw(program) => match name(program) {
            Some(n) => format!("{n} ({program})"),
            None => format!("program {program}"),
        },
        Mark::Swap => "the swap's capture and hand-off".to_owned(),
        Mark::Handed => "the frame's start before its first draw".to_owned(),
        Mark::Call(name) => name.to_owned(),
    }
}

/// The window's summary line; none without a frame read.
fn describe(w: &Window, name: &impl Fn(u32) -> Option<&'static str>) -> Option<String> {
    if w.frames == 0 {
        return None;
    }
    let frames = w.frames as f64;
    let ms = |ns: f64| ns / 1e6;
    let mut costly: Vec<(&Mark, &Tally)> = w.by_mark.iter().collect();
    costly.sort_by(|a, b| b.1.ns.total_cmp(&a.1.ns));
    let top: Vec<String> = costly
        .iter()
        .take(8)
        .map(|(mark, t)| {
            let size = match mark {
                Mark::Draw(program) => w.sizes.get(program).map_or_else(String::new, |(x, y)| format!(" at {x}x{y}")),
                _ => String::new(),
            };
            format!("{}{size} {:.2} ms a frame ({:.1} a frame, the longest {:.2})", label(**mark, name), ms(t.ns / frames), t.count as f64 / frames, ms(t.longest))
        })
        .collect();
    let longest = w.longest.1.map_or_else(String::new, |m| format!(" after {}", label(m, name)));
    Some(format!(
        "GPU pieces over {:.0} s: {} frames timed, the game's GPU work {:.2} ms a frame from swap to swap; stretches over 1 ms {:.2} a frame, the longest {:.2} ms{longest}; flushes added {:.1} a frame (pieces of {:.1} ms); the costliest: {}",
        w.start.elapsed().as_secs_f64(),
        w.frames,
        ms(w.frame_ns / frames),
        w.over_ms as f64 / frames,
        ms(w.longest.0),
        w.flushes as f64 / w.swaps.max(1) as f64,
        ms(BUDGET_NS),
        top.join("; ")
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_draw_that_would_overrun_the_piece_is_flushed_ahead_of() {
        // Nothing sent yet: never a flush.
        assert!(!splits(0.0, 2_400_000.0));
        // A long draw after some work: flushed ahead of, so it goes alone.
        assert!(splits(100_000.0, 2_400_000.0));
        // Cheap draws add up to the budget, then flush.
        assert!(!splits(300_000.0, 100_000.0));
        assert!(splits(450_000.0, 100_000.0));
        // A draw too cheap to matter, with the piece under budget: no flush.
        assert!(!splits(480_000.0, 30_000.0));
        // ...but a piece already at the budget is flushed whatever follows.
        assert!(splits(500_000.0, 10_000.0));
    }

    #[test]
    fn one_frame_in_sixteen_is_stamped() {
        let stamped: Vec<u64> = (1..=48).filter(|&n| sampled(n)).collect();
        assert_eq!(stamped, vec![16, 32, 48]);
    }

    #[test]
    fn each_stamp_owns_the_time_to_the_next() {
        let marks = [Mark::Handed, Mark::Draw(7), Mark::Draw(9), Mark::Swap];
        let times = [1_000, 1_500, 4_000, 4_100];
        assert_eq!(stretches(&marks, &times), vec![(Mark::Handed, 500.0), (Mark::Draw(7), 2_500.0), (Mark::Draw(9), 100.0)]);
    }

    #[test]
    fn the_summary_names_the_costliest_and_the_longest() {
        let mut w = Window::new();
        w.frames = 2;
        w.frame_ns = 2.0 * 9_000_000.0;
        w.by_mark.insert(Mark::Draw(7), Tally { ns: 4_800_000.0, count: 2, longest: 2_500_000.0 });
        w.by_mark.insert(Mark::Handed, Tally { ns: 400_000.0, count: 2, longest: 300_000.0 });
        w.longest = (2_500_000.0, Some(Mark::Draw(7)));
        w.over_ms = 2;
        w.flushes = 10;
        w.sizes.insert(7, (8192, 4096));
        w.swaps = 2;
        let line = describe(&w, &|p| (p == 7).then_some("landuniquebump")).unwrap();
        assert!(line.contains("9.00 ms a frame from swap to swap"), "{line}");
        assert!(line.contains("the longest 2.50 ms after landuniquebump (7)"), "{line}");
        assert!(line.contains("flushes added 5.0 a frame"), "{line}");
        assert!(line.contains("the costliest: landuniquebump (7) at 8192x4096 2.40 ms a frame"), "{line}");
    }
}
