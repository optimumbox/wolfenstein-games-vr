//! The draws: each scene draw of the left eye's view also draws the right eye, and the right eye's
//! view skips the draws already made.
//!
//! The engine draws its two views one after the other into the halves of its eye targets (window
//! wide: the left half is view 1, the right half view 2). A draw is a scene draw of view 1 when its
//! viewport lies in the left half of an eye target, its program is converted ([`super::programs`])
//! and its position rows are a perspective's; the converted program then sends a second copy of
//! each vertex, carried by the eye matrix ([`super::eye_matrix`]), to viewport 1 (the same rect in
//! the right half). The same draw in view 2 (right half) is skipped. Both views' rows are recorded
//! as they go: the matrix for the next frame is measured from them at `SwapBuffers`, and without
//! one each view draws for itself, as before.
//!
//! Clears follow: in view 1 a clear scissored to the left half is made again in the right half;
//! in view 2 a clear of the right half of a target view 1 drew both eyes into is skipped.
//!
//! The engine's extension functions are called through its own function table, whose slots are
//! found in the executable's data by the driver's and OpenGL32's addresses ([`install`]).

use super::eye_matrix::{self, Rows};
use super::gl_api::{self, Gl};
use super::programs::{Conversion, Converted};
use monaka_core::sync::Locked;
use monaka_hook::module::Module;
use monaka_hook::{Hooks, InFlight, Original};
use monaka_producer::log;
use std::collections::{HashMap, HashSet};
use std::ffi::c_void;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering::*};
use std::time::{Duration, Instant};
use windows::Win32::Graphics::OpenGL::{glGetIntegerv, glIsEnabled};

/// The draws recorded per view and frame, for the eye matrix.
const RECORDED: usize = 96;
/// How often the counters are logged.
const LOG_EVERY: Duration = Duration::from_secs(10);

#[derive(Clone, Copy, PartialEq, Debug)]
enum Phase {
    /// No view drawn yet this frame.
    Before,
    /// View 1 (left half).
    Left,
    /// View 2 (right half), to the frame's end.
    Right,
}

/// A converted program as drawn: its locations, its position rows as last uploaded, and its
/// stereo uniform as last set.
struct Program {
    converted: Converted,
    rows: Rows,
    switch: bool,
    matrix: u64,
}

#[derive(Default)]
struct Counts {
    stereo: u64,
    skipped: u64,
    mirrored: u64,
    clears_skipped: u64,
    frames_on: u64,
    frames_off: u64,
}

struct State {
    gl: Option<Gl>,
    conversion: Conversion,
    started: bool,
    /// The hooks were tried this run (not again after they could not go in).
    tried: bool,
    /// Detours act (the hooks in, the stop not begun).
    live: bool,
    programs: HashMap<u32, Program>,
    current: u32,
    /// The eye targets' size (the window's).
    size: (i32, i32),
    viewport: [i32; 4],
    phase: Phase,
    /// Viewport and scissor 1 need setting before the next stereo draw.
    index_one: bool,
    widths: HashMap<u32, Option<i32>>,
    /// This frame's eye matrix (measured last frame) and its number.
    matrix: Option<Rows>,
    matrix_number: u64,
    left: Vec<(u32, Rows)>,
    right: Vec<(u32, Rows)>,
    both_eyes: HashSet<u32>,
    counts: Counts,
    agreeing: (usize, usize),
    logged: Option<Instant>,
}

static STATE: Mutex<Option<State>> = Mutex::new(None);
static HOOKS: Mutex<Option<Hooks>> = Mutex::new(None);
/// Set by the render thread once every converted program's switch is off after the stop began.
static LET_GO: AtomicBool = AtomicBool::new(false);

fn state() -> State {
    State {
        gl: None,
        conversion: Conversion::new(),
        started: false,
        tried: false,
        live: false,
        programs: HashMap::new(),
        current: 0,
        size: (0, 0),
        viewport: [0; 4],
        phase: Phase::Before,
        index_one: true,
        widths: HashMap::new(),
        matrix: None,
        matrix_number: 0,
        left: Vec::new(),
        right: Vec::new(),
        both_eyes: HashSet::new(),
        counts: Counts::default(),
        agreeing: (0, 0),
        logged: None,
    }
}

// ---- The frame ----

/// At each `SwapBuffers` (render thread, the game's context current): the conversion's next step,
/// then the eye matrix for the next frame from this one's draws.
pub fn frame(on: bool) {
    let mut guard = STATE.locked();
    if guard.is_none() {
        if !on {
            return;
        }
        *guard = Some(state());
    }
    let state = guard.as_mut().expect("made above");
    if !on {
        stop(state);
        return;
    }
    // Once views are being built: the engine's programs are made by then.
    if !state.started {
        if crate::providers::eye_site::builds() == 0 {
            return;
        }
        state.started = true;
        match Gl::load() {
            Ok(gl) => state.gl = Some(gl),
            Err(missing) => {
                log!("single pass: off: the driver has no {missing}");
                return;
            }
        }
    }
    let Some(gl) = state.gl.as_ref() else { return };
    if !state.conversion.finished() {
        if let Some(converted) = state.conversion.step(gl) {
            state.programs = converted.into_iter().map(|c| (c.program, Program { converted: c, rows: [[0.0; 4]; 4], switch: false, matrix: 0 })).collect();
        }
        return;
    }
    if !state.live {
        // Once per run (a run after an earlier one's stop finds the programs converted already).
        if state.tried || state.programs.is_empty() || HOOKS.locked().is_some() {
            return;
        }
        state.tried = true;
        state.live = install();
        if !state.live {
            return;
        }
    }
    let size = monaka_stereo::same_frame::gl::frame_size().unwrap_or((0, 0));
    state.size = (size.0 as i32, size.1 as i32);
    let pairs: Vec<(Rows, Rows)> = state.left.iter().zip(&state.right).filter(|(l, r)| l.0 == r.0).map(|(l, r)| (l.1, r.1)).collect();
    let measured = eye_matrix::measure(&pairs);
    if state.matrix.is_some() {
        state.counts.frames_on += 1;
    } else {
        state.counts.frames_off += 1;
    }
    state.matrix = measured.map(|m| m.matrix);
    state.matrix_number += 1;
    state.agreeing = measured.map_or((0, pairs.len()), |m| (m.agreeing, m.looked_at));
    state.left.clear();
    state.right.clear();
    state.both_eyes.clear();
    state.widths.clear();
    state.phase = Phase::Before;
    if state.logged.is_none_or(|at| at.elapsed() >= LOG_EVERY) {
        state.logged = Some(Instant::now());
        let c = &state.counts;
        log!(
            "single pass: {} frames both eyes in one pass, {} each eye its own; {} draws for both eyes, {} skipped; {} clears mirrored, {} skipped; eye matrix from {} of {} pairs",
            c.frames_on,
            c.frames_off,
            c.stereo,
            c.skipped,
            c.mirrored,
            c.clears_skipped,
            state.agreeing.0,
            state.agreeing.1
        );
    }
}

/// The stop on the render thread: every switch off, then the detours stand aside.
fn stop(state: &mut State) {
    if let Some(gl) = state.gl.as_ref() {
        for program in state.programs.values_mut() {
            set_switch(gl, program, false);
        }
    }
    state.live = false;
    state.tried = false;
    LET_GO.store(true, Release);
}

/// The stop's give-back: waits a little for the render thread to turn every switch off.
pub fn give_back() {
    let until = Instant::now() + Duration::from_millis(500);
    while !LET_GO.load(Acquire) && HOOKS.locked().is_some() && Instant::now() < until {
        std::thread::sleep(Duration::from_millis(5));
    }
}

/// Whether the converted programs draw as the engine's again (their switches off), the hooks out.
pub fn let_go() -> Result<(), monaka_producer::Rejection> {
    let mut hooks = HOOKS.locked();
    if hooks.is_some() && !LET_GO.load(Acquire) {
        return Err(monaka_producer::Rejection::not_ready("single pass: the render thread has not turned its programs back yet"));
    }
    if let Some(mut hooks) = hooks.take()
        && let Err(e) = hooks.disable()
    {
        log!("single pass: {e}");
    }
    InFlight::wait_idle(500);
    Ok(())
}

pub fn summary() -> String {
    let guard = STATE.locked();
    let Some(state) = guard.as_ref() else { return String::new() };
    let c = &state.counts;
    format!(
        "single pass: {} programs converted; {} frames in one pass, {} not; {} draws for both eyes, {} skipped",
        state.programs.len(),
        c.frames_on,
        c.frames_off,
        c.stereo,
        c.skipped
    )
}

// ---- The hooks ----

type UseProgramFn = unsafe extern "system" fn(u32);
type Uniform4fvFn = unsafe extern "system" fn(i32, i32, *const f32);
type ProgramUniform4fvFn = unsafe extern "system" fn(u32, i32, i32, *const f32);
type RectFn = unsafe extern "system" fn(i32, i32, i32, i32);
type ClearFn = unsafe extern "system" fn(u32);
type DrawArraysFn = unsafe extern "system" fn(u32, i32, i32);
type DrawElementsFn = unsafe extern "system" fn(u32, i32, u32, *const c_void);
type DrawElementsBaseVertexFn = unsafe extern "system" fn(u32, i32, u32, *const c_void, i32);
type DrawRangeElementsFn = unsafe extern "system" fn(u32, u32, u32, i32, u32, *const c_void);
type DrawRangeElementsBaseVertexFn = unsafe extern "system" fn(u32, u32, u32, i32, u32, *const c_void, i32);
type DrawElementsInstancedFn = unsafe extern "system" fn(u32, i32, u32, *const c_void, i32);
type DrawElementsInstancedBaseVertexFn = unsafe extern "system" fn(u32, i32, u32, *const c_void, i32, i32);
type DrawArraysInstancedFn = unsafe extern "system" fn(u32, i32, i32, i32);

static USE_PROGRAM: Original<UseProgramFn> = Original::new();
static UNIFORM_4FV: Original<Uniform4fvFn> = Original::new();
static PROGRAM_UNIFORM_4FV: Original<ProgramUniform4fvFn> = Original::new();
static VIEWPORT: Original<RectFn> = Original::new();
static SCISSOR: Original<RectFn> = Original::new();
static CLEAR: Original<ClearFn> = Original::new();
static DRAW_ARRAYS: Original<DrawArraysFn> = Original::new();
static DRAW_ELEMENTS: Original<DrawElementsFn> = Original::new();
static DRAW_ELEMENTS_BASE_VERTEX: Original<DrawElementsBaseVertexFn> = Original::new();
static DRAW_RANGE_ELEMENTS: Original<DrawRangeElementsFn> = Original::new();
static DRAW_RANGE_ELEMENTS_BASE_VERTEX: Original<DrawRangeElementsBaseVertexFn> = Original::new();
static DRAW_ELEMENTS_INSTANCED: Original<DrawElementsInstancedFn> = Original::new();
static DRAW_ELEMENTS_INSTANCED_BASE_VERTEX: Original<DrawElementsInstancedBaseVertexFn> = Original::new();
static DRAW_ARRAYS_INSTANCED: Original<DrawArraysInstancedFn> = Original::new();

/// The slots in the executable's data holding the driver's `address`.
fn slots(exe: &Module, address: usize) -> Vec<usize> {
    let Some(code) = exe.kept_code() else { return Vec::new() };
    let mut out = Vec::new();
    for section in code.sections().iter().filter(|s| !s.executable) {
        for (index, chunk) in section.bytes.chunks_exact(8).enumerate() {
            if usize::from_le_bytes(chunk.try_into().expect("8 bytes")) == address {
                out.push(exe.base() + section.rva as usize + index * 8);
            }
        }
    }
    out
}

/// Every hook in (render thread, the context current); whether they went in. Each function is
/// hooked wherever the executable's data holds it: the driver's address (from its own loader's
/// table) or OpenGL32's export (an import slot, or a table filled by `GetProcAddress`).
fn install() -> bool {
    let Some(exe) = Module::find(&monaka_hook::module::host_name()) else {
        log!("single pass: off: the executable is not found");
        return false;
    };
    let opengl = Module::find("opengl32.dll");
    let mut hooks = Hooks::default();
    let mut found = Vec::new();
    let mut draws = 0;
    macro_rules! hook {
        ($original:expr, $name:literal, $detour:expr, $required:expr) => {{
            let mut at: Vec<usize> = [gl_api::address($name), opengl.as_ref().and_then(|m| m.export($name))]
                .into_iter()
                .flatten()
                .flat_map(|address| slots(&exe, address))
                .collect();
            at.sort_unstable();
            at.dedup();
            found.push(format!("{} {}", $name, at.len()));
            if at.is_empty() && $required {
                log!("single pass: off: the engine's {} is not in its data ({})", $name, found.join(", "));
                return false;
            }
            for slot in &at {
                // SAFETY: the slot holds the GL function of that name, called through it.
                if let Err(e) = unsafe { hooks.pointer(&$original, $name, *slot, $detour) } {
                    log!("single pass: off: {e}");
                    return false;
                }
            }
            at.len()
        }};
    }
    hook!(USE_PROGRAM, "glUseProgram", use_program as UseProgramFn, true);
    hook!(UNIFORM_4FV, "glUniform4fv", uniform_4fv as Uniform4fvFn, true);
    hook!(PROGRAM_UNIFORM_4FV, "glProgramUniform4fv", program_uniform_4fv as ProgramUniform4fvFn, false);
    hook!(VIEWPORT, "glViewport", viewport as RectFn, true);
    hook!(SCISSOR, "glScissor", scissor as RectFn, false);
    hook!(CLEAR, "glClear", clear as ClearFn, false);
    draws += hook!(DRAW_ELEMENTS, "glDrawElements", draw_elements as DrawElementsFn, false);
    draws += hook!(DRAW_ARRAYS, "glDrawArrays", draw_arrays as DrawArraysFn, false);
    draws += hook!(DRAW_ELEMENTS_BASE_VERTEX, "glDrawElementsBaseVertex", draw_elements_base_vertex as DrawElementsBaseVertexFn, false);
    draws += hook!(DRAW_RANGE_ELEMENTS, "glDrawRangeElements", draw_range_elements as DrawRangeElementsFn, false);
    draws += hook!(DRAW_RANGE_ELEMENTS_BASE_VERTEX, "glDrawRangeElementsBaseVertex", draw_range_elements_base_vertex as DrawRangeElementsBaseVertexFn, false);
    draws += hook!(DRAW_ELEMENTS_INSTANCED, "glDrawElementsInstanced", draw_elements_instanced as DrawElementsInstancedFn, false);
    draws += hook!(DRAW_ELEMENTS_INSTANCED_BASE_VERTEX, "glDrawElementsInstancedBaseVertex", draw_elements_instanced_base_vertex as DrawElementsInstancedBaseVertexFn, false);
    draws += hook!(DRAW_ARRAYS_INSTANCED, "glDrawArraysInstanced", draw_arrays_instanced as DrawArraysInstancedFn, false);
    log!("single pass: the engine's GL functions found: {}", found.join(", "));
    if draws == 0 {
        log!("single pass: off: none of the engine's draw functions found");
        return false;
    }
    if let Err(e) = hooks.enable() {
        let _ = hooks.disable();
        log!("single pass: off: {e}");
        return false;
    }
    LET_GO.store(false, Release);
    *HOOKS.locked() = Some(hooks);
    true
}

// ---- The detours' work ----

fn set_switch(gl: &Gl, program: &mut Program, on: bool) {
    if program.switch != on {
        let value = [f32::from(u8::from(on)), 0.0, 0.0, 0.0];
        // SAFETY: a plain GL call on the engine's program, with one vec4.
        unsafe { (gl.program_uniform_4fv)(program.converted.program, program.converted.stereo[4], 1, value.as_ptr()) };
        program.switch = on;
    }
}

/// Uniform rows uploaded to `program` from `location` on: the position rows among them kept.
fn uploaded(state: &mut State, program: u32, location: i32, count: i32, values: *const f32) {
    let Some(program) = state.programs.get_mut(&program) else { return };
    for (row, &at) in program.converted.rows.iter().enumerate() {
        let offset = at - location;
        if offset >= 0 && offset < count && !values.is_null() {
            // SAFETY: the engine's buffer holds `count` vec4s.
            program.rows[row] = unsafe { *values.add(offset as usize * 4).cast::<[f32; 4]>() };
        }
    }
}

fn draw_framebuffer() -> u32 {
    let mut framebuffer = 0;
    // SAFETY: a state query into a local.
    unsafe { glGetIntegerv(gl_api::GL_DRAW_FRAMEBUFFER_BINDING, &mut framebuffer) };
    framebuffer as u32
}

/// Whether the bound draw framebuffer is an eye target (window wide), and its name.
fn eye_target(state: &mut State) -> Option<u32> {
    let framebuffer = draw_framebuffer();
    let width = match state.widths.get(&framebuffer) {
        Some(&width) => width,
        None => {
            let width = if framebuffer == 0 { Some(state.size.0) } else { state.gl.as_ref()?.framebuffer_width(framebuffer) };
            state.widths.insert(framebuffer, width);
            width
        }
    };
    (width == Some(state.size.0) && state.size.0 > 0).then_some(framebuffer)
}

#[derive(Clone, Copy, PartialEq)]
enum Half {
    Left,
    Right,
    Neither,
}

fn half(state: &State, x: i32, w: i32) -> Half {
    let middle = state.size.0 / 2;
    if w <= 0 || middle <= 0 {
        Half::Neither
    } else if x + w <= middle {
        Half::Left
    } else if x >= middle {
        Half::Right
    } else {
        Half::Neither
    }
}

/// Whether the draw about to be made goes ahead (one of view 2's already drawn is skipped).
fn draw() -> bool {
    let mut guard = STATE.locked();
    let Some(state) = guard.as_mut() else { return true };
    if !state.live || !state.programs.contains_key(&state.current) {
        return true;
    }
    let current = state.current;
    let rows = state.programs[&current].rows;
    let perspective = rows[3][..3].iter().any(|v| v.abs() > 1e-6);
    let side = half(state, state.viewport[0], state.viewport[2]);
    let target = if perspective && side != Half::Neither { eye_target(state) } else { None };
    let Some(gl) = state.gl.as_ref() else { return true };
    let program = state.programs.get_mut(&current).expect("looked up above");
    match (target, side, state.phase) {
        (Some(framebuffer), Half::Left, Phase::Before | Phase::Left) => {
            if state.left.len() < RECORDED {
                state.left.push((current, rows));
            }
            let Some(matrix) = state.matrix else {
                set_switch(gl, program, false);
                return true;
            };
            if program.matrix != state.matrix_number {
                for (row, values) in matrix.iter().enumerate() {
                    // SAFETY: a plain GL call on the engine's program, with one vec4.
                    unsafe { (gl.program_uniform_4fv)(current, program.converted.stereo[row], 1, values.as_ptr()) };
                }
                program.matrix = state.matrix_number;
            }
            set_switch(gl, program, true);
            if state.index_one {
                let [x, y, w, h] = state.viewport;
                let middle = state.size.0 / 2;
                // SAFETY: plain GL calls; viewport and scissor 1 serve only the second copy.
                unsafe {
                    (gl.viewport_indexed_f)(1, (x + middle) as f32, y as f32, w as f32, h as f32);
                    (gl.scissor_indexed)(1, middle, 0, middle, state.size.1);
                }
                state.index_one = false;
            }
            state.both_eyes.insert(framebuffer);
            state.counts.stereo += 1;
            true
        }
        (Some(_), Half::Right, Phase::Right) => {
            if state.right.len() < RECORDED {
                state.right.push((current, rows));
            }
            set_switch(gl, program, false);
            if state.matrix.is_some() {
                state.counts.skipped += 1;
                return false;
            }
            true
        }
        _ => {
            set_switch(gl, program, false);
            true
        }
    }
}

/// What to do with a clear: make it as it is, make it again in the right half, or skip it.
enum ClearPlan {
    AsIs,
    Mirror([i32; 4]),
    Skip,
}

fn clear_plan() -> ClearPlan {
    let mut guard = STATE.locked();
    let Some(state) = guard.as_mut() else { return ClearPlan::AsIs };
    if !state.live || state.matrix.is_none() {
        return ClearPlan::AsIs;
    }
    // SAFETY: a state query.
    if unsafe { glIsEnabled(gl_api::GL_SCISSOR_TEST) } == 0 {
        return ClearPlan::AsIs;
    }
    let mut scissor = [0i32; 4];
    // SAFETY: a state query into four ints.
    unsafe { glGetIntegerv(gl_api::GL_SCISSOR_BOX, scissor.as_mut_ptr()) };
    let side = half(state, scissor[0], scissor[2]);
    if side == Half::Neither {
        return ClearPlan::AsIs;
    }
    let Some(framebuffer) = eye_target(state) else { return ClearPlan::AsIs };
    match (side, state.phase) {
        (Half::Left, Phase::Before | Phase::Left) => {
            state.counts.mirrored += 1;
            ClearPlan::Mirror(scissor)
        }
        (Half::Right, Phase::Right) if state.both_eyes.contains(&framebuffer) => {
            state.counts.clears_skipped += 1;
            ClearPlan::Skip
        }
        _ => ClearPlan::AsIs,
    }
}

// ---- The detours ----

unsafe extern "system" fn use_program(program: u32) {
    let flight = InFlight::enter();
    if let Some(state) = STATE.locked().as_mut() {
        state.current = program;
    }
    // SAFETY: forwards the game's own call.
    unsafe { USE_PROGRAM.get(&flight)(program) }
}

unsafe extern "system" fn uniform_4fv(location: i32, count: i32, values: *const f32) {
    let flight = InFlight::enter();
    if let Some(state) = STATE.locked().as_mut() {
        let current = state.current;
        uploaded(state, current, location, count, values);
    }
    // SAFETY: forwards the game's own call.
    unsafe { UNIFORM_4FV.get(&flight)(location, count, values) }
}

unsafe extern "system" fn program_uniform_4fv(program: u32, location: i32, count: i32, values: *const f32) {
    let flight = InFlight::enter();
    if let Some(state) = STATE.locked().as_mut() {
        uploaded(state, program, location, count, values);
    }
    // SAFETY: forwards the game's own call.
    unsafe { PROGRAM_UNIFORM_4FV.get(&flight)(program, location, count, values) }
}

unsafe extern "system" fn viewport(x: i32, y: i32, w: i32, h: i32) {
    let flight = InFlight::enter();
    if let Some(state) = STATE.locked().as_mut() {
        state.viewport = [x, y, w, h];
        // glViewport sets every viewport: viewport 1 follows before the next stereo draw.
        state.index_one = true;
        match half(state, x, w) {
            Half::Left if state.phase == Phase::Before => state.phase = Phase::Left,
            Half::Right if state.phase != Phase::Before => state.phase = Phase::Right,
            _ => {}
        }
    }
    // SAFETY: forwards the game's own call.
    unsafe { VIEWPORT.get(&flight)(x, y, w, h) }
}

unsafe extern "system" fn scissor(x: i32, y: i32, w: i32, h: i32) {
    let flight = InFlight::enter();
    if let Some(state) = STATE.locked().as_mut() {
        state.index_one = true;
    }
    // SAFETY: forwards the game's own call.
    unsafe { SCISSOR.get(&flight)(x, y, w, h) }
}

unsafe extern "system" fn clear(mask: u32) {
    let flight = InFlight::enter();
    let original = CLEAR.get(&flight);
    match clear_plan() {
        ClearPlan::AsIs => unsafe { original(mask) },
        ClearPlan::Skip => {}
        ClearPlan::Mirror([x, y, w, h]) => {
            let scissor = SCISSOR.get(&flight);
            let middle = STATE.locked().as_ref().map_or(0, |s| s.size.0 / 2);
            // SAFETY: the game's own clear, made in its rect and again in the right half, its
            // scissor put back.
            unsafe {
                original(mask);
                scissor(x + middle, y, w, h);
                original(mask);
                scissor(x, y, w, h);
            }
            if let Some(state) = STATE.locked().as_mut() {
                state.index_one = true;
            }
        }
    }
}

macro_rules! draw_detour {
    ($name:ident, $original:ident, $type:ty, ($($arg:ident: $ty:ty),*)) => {
        unsafe extern "system" fn $name($($arg: $ty),*) {
            let flight = InFlight::enter();
            if draw() {
                // SAFETY: forwards the game's own call.
                unsafe { $original.get(&flight)($($arg),*) }
            }
        }
    };
}

draw_detour!(draw_arrays, DRAW_ARRAYS, DrawArraysFn, (mode: u32, first: i32, count: i32));
draw_detour!(draw_elements, DRAW_ELEMENTS, DrawElementsFn, (mode: u32, count: i32, kind: u32, indices: *const c_void));
draw_detour!(draw_elements_base_vertex, DRAW_ELEMENTS_BASE_VERTEX, DrawElementsBaseVertexFn, (mode: u32, count: i32, kind: u32, indices: *const c_void, base: i32));
draw_detour!(draw_range_elements, DRAW_RANGE_ELEMENTS, DrawRangeElementsFn, (mode: u32, start: u32, end: u32, count: i32, kind: u32, indices: *const c_void));
draw_detour!(
    draw_range_elements_base_vertex,
    DRAW_RANGE_ELEMENTS_BASE_VERTEX,
    DrawRangeElementsBaseVertexFn,
    (mode: u32, start: u32, end: u32, count: i32, kind: u32, indices: *const c_void, base: i32)
);
draw_detour!(draw_elements_instanced, DRAW_ELEMENTS_INSTANCED, DrawElementsInstancedFn, (mode: u32, count: i32, kind: u32, indices: *const c_void, instances: i32));
draw_detour!(
    draw_elements_instanced_base_vertex,
    DRAW_ELEMENTS_INSTANCED_BASE_VERTEX,
    DrawElementsInstancedBaseVertexFn,
    (mode: u32, count: i32, kind: u32, indices: *const c_void, instances: i32, base: i32)
);
draw_detour!(draw_arrays_instanced, DRAW_ARRAYS_INSTANCED, DrawArraysInstancedFn, (mode: u32, first: i32, count: i32, instances: i32));
