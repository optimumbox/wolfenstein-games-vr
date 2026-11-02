//! The draws: the scene draws of the left eye's pass also draw the right eye, and the right eye's
//! pass skips them.
//!
//! The engine draws a frame as two passes, the left eye's then the right's, each into the same
//! eye-sized textures (a colour, depth and stencil clear begins each). Those textures are stood in
//! for by two-layer arrays ([`super::layers`]): the engine's framebuffer and texture binds are
//! redirected to ours, layer 0 in the left eye's pass and layer 1 in the right's.
//!
//! Each pass begins with its scene: draws with a converted program ([`super::programs`]) and
//! perspective position rows into framebuffers whose attachments are all eye textures. While the
//! left eye's pass is in that stretch its framebuffers are the layered ones and the converted
//! programs send a second copy of each vertex, carried by the eye matrix ([`super::eye_matrix`]),
//! into layer 1. The right eye's pass skips the same stretch, its draws and clears, finding its
//! layer drawn. Each pass's stretch ends at its first other draw into an eye texture (lighting,
//! post-processing, which take per-eye inputs): from there both passes draw as the engine does,
//! in its order. Both passes' rows are recorded in the stretch: the matrix for the next frame is
//! measured from them at `SwapBuffers`, and without one each pass draws for itself, as before.
//!
//! The stand-ins are made once the scene's framebuffers have been seen for a while
//! ([`DISCOVERY_FRAMES`]); at each `SwapBuffers` the frame's last layer is copied back into the
//! engine's textures, for what reads them at the present (the frame's depth).
//!
//! The engine's extension functions are called through its own function table, whose slots are
//! found in the executable's data by the driver's and OpenGL32's addresses ([`install`]).

use super::eye_matrix::{self, Rows};
use super::gl_api::{self, Gl};
use super::layers::Layers;
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
use windows::Win32::Graphics::OpenGL::glGetIntegerv;

/// The draws recorded per pass and frame, for the eye matrix.
const RECORDED: usize = 96;
/// How often the counters are logged.
const LOG_EVERY: Duration = Duration::from_secs(10);
/// The live frames whose scene framebuffers are gathered before the stand-ins are made.
const DISCOVERY_FRAMES: u64 = 60;

const GL_FRAMEBUFFER: u32 = 0x8D40;
const GL_READ_FRAMEBUFFER: u32 = 0x8CA8;
const GL_DRAW_FRAMEBUFFER: u32 = 0x8CA9;
const GL_READ_FRAMEBUFFER_BINDING: u32 = 0x8CAA;
const GL_TEXTURE0: u32 = 0x84C0;
const GL_COLOR_BUFFER_BIT: u32 = 0x4000;
const GL_DEPTH_BUFFER_BIT: u32 = 0x100;

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
    clears_skipped: u64,
    frames_on: u64,
    frames_off: u64,
    /// Frames with other than two passes.
    odd_frames: u64,
    /// The draws seen, then those passing each test in turn: a converted program, perspective
    /// rows, a framebuffer stood in for, the scene's stretch.
    draws: u64,
    converted: u64,
    perspective: u64,
    stood_in: u64,
    in_stretch: u64,
    /// Occlusion queries begun in each pass, their results read back as visible (their draws
    /// skipped), and conditional renders on them made unconditional.
    queries: [u64; 3],
    blind_results: u64,
    blind_conditions: u64,
}

/// The live frames whose passes are written to the log: each run of draws into one framebuffer
/// and viewport as a line, with the clears between them.
const TRACED_FRAMES: std::ops::Range<u64> = 600..602;
const TRACED_LINES: usize = 600;

/// A traced frame: its lines and the run of draws being counted.
struct Trace {
    lines: Vec<String>,
    run: Option<(u32, [i32; 4])>,
    draws: u32,
    scene: u32,
    /// The textures the run's draws were given ("t1@u3").
    binds: Vec<String>,
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
    /// The window's size (both eyes side by side): an eye is half its width.
    size: (i32, i32),
    viewport: [i32; 4],
    /// The pass this frame is in: 0 before the first, 1 the left eye's, 2 the right's.
    pass: u32,
    /// The pass's scene stretch is over.
    cut: bool,
    /// The framebuffers and eye textures the engine has bound (its names).
    draw_framebuffer: u32,
    read_framebuffer: u32,
    units: HashMap<u32, (u32, u32)>,
    layers: Option<Layers>,
    /// The stand-ins could not be made (not tried again this run).
    layers_failed: bool,
    /// The scene's framebuffers seen before the stand-ins are made, and framebuffers' widths.
    seen: HashSet<u32>,
    widths: HashMap<u32, Option<i32>>,
    /// This frame's eye matrix (measured last frame) and its number.
    matrix: Option<Rows>,
    matrix_number: u64,
    left: Vec<(u32, Rows)>,
    right: Vec<(u32, Rows)>,
    counts: Counts,
    agreeing: (usize, usize),
    logged: Option<Instant>,
    live_frames: u64,
    trace: Option<Trace>,
    /// The occlusion query being counted, the queries some of whose draws were skipped, and
    /// whether the current conditional render was left out.
    query: Option<u32>,
    blind: HashSet<u32>,
    unconditional: bool,
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
        pass: 0,
        cut: false,
        draw_framebuffer: 0,
        read_framebuffer: 0,
        units: HashMap::new(),
        layers: None,
        layers_failed: false,
        seen: HashSet::new(),
        widths: HashMap::new(),
        matrix: None,
        matrix_number: 0,
        left: Vec::new(),
        right: Vec::new(),
        counts: Counts::default(),
        agreeing: (0, 0),
        logged: None,
        live_frames: 0,
        trace: None,
        query: None,
        blind: HashSet::new(),
        unconditional: false,
    }
}

impl State {
    /// The layer the pass draws into.
    fn layer(&self) -> usize {
        usize::from(self.pass >= 2)
    }

    /// The framebuffer drawn into for the engine's `framebuffer`.
    fn mapped(&self, framebuffer: u32) -> u32 {
        let Some(stand) = self.layers.as_ref().filter(|_| self.live).and_then(|l| l.stand(framebuffer)) else { return framebuffer };
        match stand.layered {
            Some(layered) if self.pass == 1 && !self.cut => layered,
            _ => stand.layer[self.layer()],
        }
    }

    /// The texture given for the engine's `texture`.
    fn texture(&self, texture: u32) -> u32 {
        self.layers.as_ref().filter(|_| self.live).and_then(|l| l.view(texture, self.layer())).unwrap_or(texture)
    }

    /// The engine's bound framebuffers and eye textures bound again as the pass has them.
    fn rebind(&self) {
        let Some(gl) = self.gl.as_ref() else { return };
        // SAFETY: plain binds of the engine's objects or ours standing in for them.
        unsafe {
            (gl.bind_framebuffer)(GL_DRAW_FRAMEBUFFER, self.mapped(self.draw_framebuffer));
            (gl.bind_framebuffer)(GL_READ_FRAMEBUFFER, self.mapped(self.read_framebuffer));
            for (&unit, &(target, texture)) in &self.units {
                (gl.bind_multi_texture)(GL_TEXTURE0 + unit, target, self.texture(texture));
            }
        }
    }

    /// Whether `framebuffer` draws into an eye texture (half the window wide).
    fn eye_sized(&mut self, framebuffer: u32) -> bool {
        let width = match self.widths.get(&framebuffer) {
            Some(&width) => width,
            None => {
                let width = self.gl.as_ref().and_then(|gl| gl.framebuffer_width(framebuffer));
                self.widths.insert(framebuffer, width);
                width
            }
        };
        width == Some(self.size.0 / 2) && self.size.0 > 0
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
        let (mut draw, mut read) = (0, 0);
        // SAFETY: state queries into locals (the engine's own bindings: nothing stands in yet).
        unsafe {
            glGetIntegerv(gl_api::GL_DRAW_FRAMEBUFFER_BINDING, &mut draw);
            glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &mut read);
        }
        state.draw_framebuffer = draw as u32;
        state.read_framebuffer = read as u32;
        state.live = install();
        if !state.live {
            return;
        }
    }
    state.live_frames += 1;
    if let Some(mut trace) = state.trace.take() {
        end_run(state.gl.as_ref(), &mut trace);
        let lines = trace.lines;
        log!("single pass: frame {} traced:\n  {}", state.live_frames - 1, lines.join("\n  "));
    }
    if TRACED_FRAMES.contains(&state.live_frames) {
        state.trace = Some(Trace { lines: Vec::new(), run: None, draws: 0, scene: 0, binds: Vec::new() });
    }
    let passes = state.pass;
    if let (Some(layers), Some(gl)) = (state.layers.as_ref(), state.gl.as_ref()) {
        layers.copy_back(gl, state.layer());
    }
    let size = monaka_stereo::same_frame::gl::frame_size().unwrap_or((0, 0));
    state.size = (size.0 as i32, size.1 as i32);
    let pairs: Vec<(Rows, Rows)> = state.left.iter().zip(&state.right).filter(|(l, r)| l.0 == r.0).map(|(l, r)| (l.1, r.1)).collect();
    let measured = if passes == 2 { eye_matrix::measure(&pairs) } else { None };
    if state.matrix.is_some() {
        state.counts.frames_on += 1;
    } else {
        state.counts.frames_off += 1;
    }
    state.counts.odd_frames += u64::from(passes != 2);
    state.matrix = measured.map(|m| m.matrix);
    state.matrix_number += 1;
    state.agreeing = measured.map_or((0, pairs.len()), |m| (m.agreeing, m.looked_at));
    state.left.clear();
    state.right.clear();
    state.pass = 0;
    state.cut = false;
    if state.layers.is_none() && !state.layers_failed && passes == 2 && state.live_frames >= DISCOVERY_FRAMES && !state.seen.is_empty() {
        let seen: Vec<u32> = state.seen.iter().copied().collect();
        let eye = (state.size.0 / 2, state.size.1);
        match Layers::build(state.gl.as_ref().expect("loaded above"), &seen, eye) {
            Ok(layers) => {
                let (textures, framebuffers, layered) = layers.counts();
                log!("single pass: {textures} eye textures stood in for by layered arrays; {framebuffers} framebuffers drawing into them, {layered} with every attachment layered (scene framebuffers {seen:?})");
                state.layers = Some(layers);
            }
            Err(e) => {
                log!("single pass: off: {e}");
                state.layers_failed = true;
            }
        }
    }
    state.rebind();
    if state.logged.is_none_or(|at| at.elapsed() >= LOG_EVERY) {
        state.logged = Some(Instant::now());
        let c = &state.counts;
        log!(
            "single pass: {} frames both eyes in one pass, {} each eye its own ({} without two passes); {} draws for both eyes, {} skipped, {} clears skipped; eye matrix from {} of {} pairs",
            c.frames_on,
            c.frames_off,
            c.odd_frames,
            c.stereo,
            c.skipped,
            c.clears_skipped,
            state.agreeing.0,
            state.agreeing.1
        );
        log!(
            "single pass: {} draws, {} with a converted program, {} perspective, {} into a stood-in framebuffer, {} in the scene's stretch; occlusion queries {:?} by pass, {} read as visible, {} conditional renders made unconditional",
            c.draws,
            c.converted,
            c.perspective,
            c.stood_in,
            c.in_stretch,
            c.queries,
            c.blind_results,
            c.blind_conditions
        );
    }
}

/// The stop on the render thread: every switch off, the engine's own objects bound again, then the
/// detours stand aside.
fn stop(state: &mut State) {
    if let Some(gl) = state.gl.as_ref() {
        for program in state.programs.values_mut() {
            set_switch(gl, program, false);
        }
        if let Some(layers) = state.layers.as_ref() {
            layers.copy_back(gl, state.layer());
        }
    }
    state.live = false;
    state.rebind();
    if let (Some(mut layers), Some(gl)) = (state.layers.take(), state.gl.as_ref()) {
        layers.release(gl);
    }
    state.layers_failed = false;
    state.seen.clear();
    state.units.clear();
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
type BindFramebufferFn = unsafe extern "system" fn(u32, u32);
type BindTextureFn = unsafe extern "system" fn(u32, u32);
type BindTexturesFn = unsafe extern "system" fn(u32, i32, *const u32);
type BindMultiTextureFn = unsafe extern "system" fn(u32, u32, u32);
type BeginQueryFn = unsafe extern "system" fn(u32, u32);
type EndQueryFn = unsafe extern "system" fn(u32);
type GetQueryObjectIvFn = unsafe extern "system" fn(u32, u32, *mut i32);
type GetQueryObjectUivFn = unsafe extern "system" fn(u32, u32, *mut u32);
type GetQueryObjectI64vFn = unsafe extern "system" fn(u32, u32, *mut i64);
type GetQueryObjectUi64vFn = unsafe extern "system" fn(u32, u32, *mut u64);
type BeginConditionalRenderFn = unsafe extern "system" fn(u32, u32);
type EndConditionalRenderFn = unsafe extern "system" fn();
type DrawArraysFn = unsafe extern "system" fn(u32, i32, i32);
type DrawElementsFn = unsafe extern "system" fn(u32, i32, u32, *const c_void);
type DrawElementsBaseVertexFn = unsafe extern "system" fn(u32, i32, u32, *const c_void, i32);
type DrawRangeElementsFn = unsafe extern "system" fn(u32, u32, u32, i32, u32, *const c_void);
type DrawRangeElementsBaseVertexFn = unsafe extern "system" fn(u32, u32, u32, i32, u32, *const c_void, i32);
type DrawElementsInstancedFn = unsafe extern "system" fn(u32, i32, u32, *const c_void, i32);
type DrawElementsInstancedBaseVertexFn = unsafe extern "system" fn(u32, i32, u32, *const c_void, i32, i32);
type DrawArraysInstancedFn = unsafe extern "system" fn(u32, i32, i32, i32);
type DrawElementsIndirectFn = unsafe extern "system" fn(u32, u32, *const c_void);
type MultiDrawElementsIndirectFn = unsafe extern "system" fn(u32, u32, *const c_void, i32, i32);
type DrawArraysIndirectFn = unsafe extern "system" fn(u32, *const c_void);
type MultiDrawArraysIndirectFn = unsafe extern "system" fn(u32, *const c_void, i32, i32);
type MultiDrawElementsFn = unsafe extern "system" fn(u32, *const i32, u32, *const *const c_void, i32);
type MultiDrawElementsBaseVertexFn = unsafe extern "system" fn(u32, *const i32, u32, *const *const c_void, i32, *const i32);
type DrawElementsInstancedBaseInstanceFn = unsafe extern "system" fn(u32, i32, u32, *const c_void, i32, u32);
type DrawElementsInstancedBaseVertexBaseInstanceFn = unsafe extern "system" fn(u32, i32, u32, *const c_void, i32, i32, u32);
type DrawArraysInstancedBaseInstanceFn = unsafe extern "system" fn(u32, i32, i32, i32, u32);
type MultiDrawElementsIndirectBindlessFn = unsafe extern "system" fn(u32, u32, *const c_void, i32, i32, i32);

static USE_PROGRAM: Original<UseProgramFn> = Original::new();
static UNIFORM_4FV: Original<Uniform4fvFn> = Original::new();
static PROGRAM_UNIFORM_4FV: Original<ProgramUniform4fvFn> = Original::new();
static VIEWPORT: Original<RectFn> = Original::new();
static CLEAR: Original<ClearFn> = Original::new();
static BIND_FRAMEBUFFER: Original<BindFramebufferFn> = Original::new();
static BIND_FRAMEBUFFER_EXT: Original<BindFramebufferFn> = Original::new();
static BIND_TEXTURE: Original<BindTextureFn> = Original::new();
static BIND_TEXTURES: Original<BindTexturesFn> = Original::new();
static BIND_MULTI_TEXTURE: Original<BindMultiTextureFn> = Original::new();
static BEGIN_QUERY: Original<BeginQueryFn> = Original::new();
static END_QUERY: Original<EndQueryFn> = Original::new();
static GET_QUERY_OBJECT_IV: Original<GetQueryObjectIvFn> = Original::new();
static GET_QUERY_OBJECT_UIV: Original<GetQueryObjectUivFn> = Original::new();
static GET_QUERY_OBJECT_I64V: Original<GetQueryObjectI64vFn> = Original::new();
static GET_QUERY_OBJECT_UI64V: Original<GetQueryObjectUi64vFn> = Original::new();
static BEGIN_CONDITIONAL_RENDER: Original<BeginConditionalRenderFn> = Original::new();
static END_CONDITIONAL_RENDER: Original<EndConditionalRenderFn> = Original::new();
static DRAW_ARRAYS: Original<DrawArraysFn> = Original::new();
static DRAW_ELEMENTS: Original<DrawElementsFn> = Original::new();
static DRAW_ELEMENTS_BASE_VERTEX: Original<DrawElementsBaseVertexFn> = Original::new();
static DRAW_RANGE_ELEMENTS: Original<DrawRangeElementsFn> = Original::new();
static DRAW_RANGE_ELEMENTS_BASE_VERTEX: Original<DrawRangeElementsBaseVertexFn> = Original::new();
static DRAW_ELEMENTS_INSTANCED: Original<DrawElementsInstancedFn> = Original::new();
static DRAW_ELEMENTS_INSTANCED_BASE_VERTEX: Original<DrawElementsInstancedBaseVertexFn> = Original::new();
static DRAW_ARRAYS_INSTANCED: Original<DrawArraysInstancedFn> = Original::new();
static DRAW_ELEMENTS_INDIRECT: Original<DrawElementsIndirectFn> = Original::new();
static MULTI_DRAW_ELEMENTS_INDIRECT: Original<MultiDrawElementsIndirectFn> = Original::new();
static DRAW_ARRAYS_INDIRECT: Original<DrawArraysIndirectFn> = Original::new();
static MULTI_DRAW_ARRAYS_INDIRECT: Original<MultiDrawArraysIndirectFn> = Original::new();
static MULTI_DRAW_ELEMENTS: Original<MultiDrawElementsFn> = Original::new();
static MULTI_DRAW_ELEMENTS_BASE_VERTEX: Original<MultiDrawElementsBaseVertexFn> = Original::new();
static DRAW_ELEMENTS_INSTANCED_BASE_INSTANCE: Original<DrawElementsInstancedBaseInstanceFn> = Original::new();
static DRAW_ELEMENTS_INSTANCED_BASE_VERTEX_BASE_INSTANCE: Original<DrawElementsInstancedBaseVertexBaseInstanceFn> = Original::new();
static DRAW_ARRAYS_INSTANCED_BASE_INSTANCE: Original<DrawArraysInstancedBaseInstanceFn> = Original::new();
static MULTI_DRAW_ELEMENTS_INDIRECT_BINDLESS: Original<MultiDrawElementsIndirectBindlessFn> = Original::new();

/// Where the executable's data holds `address`.
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
    hook!(VIEWPORT, "glViewport", viewport as RectFn, false);
    hook!(CLEAR, "glClear", clear as ClearFn, true);
    let binds = hook!(BIND_FRAMEBUFFER, "glBindFramebuffer", bind_framebuffer as BindFramebufferFn, false)
        + hook!(BIND_FRAMEBUFFER_EXT, "glBindFramebufferEXT", bind_framebuffer_ext as BindFramebufferFn, false);
    if binds == 0 {
        log!("single pass: off: the engine's glBindFramebuffer is not in its data ({})", found.join(", "));
        return false;
    }
    hook!(BIND_TEXTURE, "glBindTexture", bind_texture as BindTextureFn, false);
    hook!(BIND_TEXTURES, "glBindTextures", bind_textures as BindTexturesFn, false);
    hook!(BIND_MULTI_TEXTURE, "glBindMultiTextureEXT", bind_multi_texture as BindMultiTextureFn, false);
    hook!(BEGIN_QUERY, "glBeginQuery", begin_query as BeginQueryFn, false);
    hook!(END_QUERY, "glEndQuery", end_query as EndQueryFn, false);
    hook!(GET_QUERY_OBJECT_IV, "glGetQueryObjectiv", get_query_object_iv as GetQueryObjectIvFn, false);
    hook!(GET_QUERY_OBJECT_UIV, "glGetQueryObjectuiv", get_query_object_uiv as GetQueryObjectUivFn, false);
    hook!(GET_QUERY_OBJECT_I64V, "glGetQueryObjecti64v", get_query_object_i64v as GetQueryObjectI64vFn, false);
    hook!(GET_QUERY_OBJECT_UI64V, "glGetQueryObjectui64v", get_query_object_ui64v as GetQueryObjectUi64vFn, false);
    hook!(BEGIN_CONDITIONAL_RENDER, "glBeginConditionalRender", begin_conditional_render as BeginConditionalRenderFn, false);
    hook!(END_CONDITIONAL_RENDER, "glEndConditionalRender", end_conditional_render as EndConditionalRenderFn, false);
    draws += hook!(DRAW_ELEMENTS, "glDrawElements", draw_elements as DrawElementsFn, false);
    draws += hook!(DRAW_ARRAYS, "glDrawArrays", draw_arrays as DrawArraysFn, false);
    draws += hook!(DRAW_ELEMENTS_BASE_VERTEX, "glDrawElementsBaseVertex", draw_elements_base_vertex as DrawElementsBaseVertexFn, false);
    draws += hook!(DRAW_RANGE_ELEMENTS, "glDrawRangeElements", draw_range_elements as DrawRangeElementsFn, false);
    draws += hook!(DRAW_RANGE_ELEMENTS_BASE_VERTEX, "glDrawRangeElementsBaseVertex", draw_range_elements_base_vertex as DrawRangeElementsBaseVertexFn, false);
    draws += hook!(DRAW_ELEMENTS_INSTANCED, "glDrawElementsInstanced", draw_elements_instanced as DrawElementsInstancedFn, false);
    draws += hook!(DRAW_ELEMENTS_INSTANCED_BASE_VERTEX, "glDrawElementsInstancedBaseVertex", draw_elements_instanced_base_vertex as DrawElementsInstancedBaseVertexFn, false);
    draws += hook!(DRAW_ARRAYS_INSTANCED, "glDrawArraysInstanced", draw_arrays_instanced as DrawArraysInstancedFn, false);
    draws += hook!(DRAW_ELEMENTS_INDIRECT, "glDrawElementsIndirect", draw_elements_indirect as DrawElementsIndirectFn, false);
    draws += hook!(MULTI_DRAW_ELEMENTS_INDIRECT, "glMultiDrawElementsIndirect", multi_draw_elements_indirect as MultiDrawElementsIndirectFn, false);
    draws += hook!(DRAW_ARRAYS_INDIRECT, "glDrawArraysIndirect", draw_arrays_indirect as DrawArraysIndirectFn, false);
    draws += hook!(MULTI_DRAW_ARRAYS_INDIRECT, "glMultiDrawArraysIndirect", multi_draw_arrays_indirect as MultiDrawArraysIndirectFn, false);
    draws += hook!(MULTI_DRAW_ELEMENTS, "glMultiDrawElements", multi_draw_elements as MultiDrawElementsFn, false);
    draws += hook!(MULTI_DRAW_ELEMENTS_BASE_VERTEX, "glMultiDrawElementsBaseVertex", multi_draw_elements_base_vertex as MultiDrawElementsBaseVertexFn, false);
    draws += hook!(DRAW_ELEMENTS_INSTANCED_BASE_INSTANCE, "glDrawElementsInstancedBaseInstance", draw_elements_instanced_base_instance as DrawElementsInstancedBaseInstanceFn, false);
    draws += hook!(DRAW_ELEMENTS_INSTANCED_BASE_VERTEX_BASE_INSTANCE, "glDrawElementsInstancedBaseVertexBaseInstance", draw_elements_instanced_base_vertex_base_instance as DrawElementsInstancedBaseVertexBaseInstanceFn, false);
    draws += hook!(DRAW_ARRAYS_INSTANCED_BASE_INSTANCE, "glDrawArraysInstancedBaseInstance", draw_arrays_instanced_base_instance as DrawArraysInstancedBaseInstanceFn, false);
    draws += hook!(MULTI_DRAW_ELEMENTS_INDIRECT_BINDLESS, "glMultiDrawElementsIndirectBindlessNV", multi_draw_elements_indirect_bindless as MultiDrawElementsIndirectBindlessFn, false);
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

/// Whether the draw about to be made goes ahead (one of the right eye's pass already drawn is
/// skipped).
fn draw() -> bool {
    let mut guard = STATE.locked();
    let Some(state) = guard.as_mut() else { return true };
    if !state.live {
        return true;
    }
    state.counts.draws += 1;
    let current = state.current;
    let rows = state.programs.get(&current).map(|p| p.rows);
    let perspective = rows.is_some_and(|rows| rows[3][..3].iter().any(|v| v.abs() > 1e-6));
    state.counts.converted += u64::from(rows.is_some());
    state.counts.perspective += u64::from(perspective);
    traced_draw(state, perspective);
    let framebuffer = state.draw_framebuffer;
    if state.layers.is_none() {
        if perspective && state.pass >= 1 && state.eye_sized(framebuffer) {
            state.seen.insert(framebuffer);
        }
    }
    let stand = state.layers.as_ref().and_then(|l| l.stand(framebuffer));
    state.counts.stood_in += u64::from(stand.is_some());
    let mut stretch = stand.is_some() && !state.cut && (state.pass == 1 || state.pass == 2);
    if stretch && !(perspective && stand.is_some_and(|s| s.layered.is_some())) {
        // The scene's stretch ends: the left eye's pass draws into its own layer from here.
        state.cut = true;
        stretch = false;
        if state.pass == 1 {
            state.rebind();
        }
    }
    state.counts.in_stretch += u64::from(stretch);
    let Some(rows) = rows else { return true };
    let Some(gl) = state.gl.as_ref() else { return true };
    let program = state.programs.get_mut(&current).expect("looked up above");
    match (stretch, state.pass) {
        (true, 1) => {
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
            state.counts.stereo += 1;
            true
        }
        (true, _) => {
            if state.right.len() < RECORDED {
                state.right.push((current, rows));
            }
            set_switch(gl, program, false);
            if state.matrix.is_some() {
                state.counts.skipped += 1;
                // Its occlusion query would count none of its samples: the engine would hide
                // the object from the right eye next frame.
                if let Some(query) = state.query {
                    state.blind.insert(query);
                }
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

/// Whether a clear goes ahead: a colour and depth clear begins a pass; in the right eye's pass a
/// clear of the scene's stretch, already made in both layers, is skipped.
fn clear_goes_ahead(mask: u32) -> bool {
    let mut guard = STATE.locked();
    let Some(state) = guard.as_mut() else { return true };
    if !state.live {
        return true;
    }
    if mask & (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT) == GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT {
        state.pass += 1;
        state.cut = false;
        state.rebind();
    }
    let layered = state.layers.as_ref().and_then(|l| l.stand(state.draw_framebuffer)).is_some_and(|s| s.layered.is_some());
    if state.pass == 2 && !state.cut && layered && state.matrix.is_some() {
        state.counts.clears_skipped += 1;
        return false;
    }
    true
}

// ---- The trace ----

/// What framebuffer `framebuffer` draws into: its colour and depth attachments, with sizes.
fn describe(gl: Option<&Gl>, framebuffer: u32) -> String {
    let Some(gl) = gl else { return format!("fb {framebuffer}") };
    if framebuffer == 0 {
        return "the window".to_owned();
    }
    let mut parts = Vec::new();
    for (label, attachment) in [("colour", gl_api::GL_COLOR_ATTACHMENT0), ("colour1", gl_api::GL_COLOR_ATTACHMENT0 + 1), ("depth", gl_api::GL_DEPTH_ATTACHMENT)] {
        let (mut kind, mut name, mut width, mut height) = (0, 0, 0, 0);
        // SAFETY: plain queries into locals.
        unsafe {
            (gl.get_named_framebuffer_attachment_parameter_iv)(framebuffer, attachment, gl_api::GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE, &mut kind);
            (gl.get_named_framebuffer_attachment_parameter_iv)(framebuffer, attachment, gl_api::GL_FRAMEBUFFER_ATTACHMENT_OBJECT_NAME, &mut name);
            if kind as u32 == gl_api::GL_TEXTURE {
                (gl.get_texture_level_parameter_iv)(name as u32, 0, gl_api::GL_TEXTURE_WIDTH, &mut width);
                (gl.get_texture_level_parameter_iv)(name as u32, 0, 0x1001, &mut height);
            }
        }
        if kind != 0 {
            let what = if kind as u32 == gl_api::GL_TEXTURE { "tex" } else { "rb" };
            parts.push(format!("{label} {what} {name} {width}x{height}"));
        }
    }
    format!("fb {framebuffer} ({})", parts.join(", "))
}

fn end_run(gl: Option<&Gl>, trace: &mut Trace) {
    if let Some((framebuffer, viewport)) = trace.run.take()
        && trace.lines.len() < TRACED_LINES
    {
        let binds = if trace.binds.is_empty() { String::new() } else { format!(" given {}", trace.binds.join(" ")) };
        trace.lines.push(format!("{} viewport {viewport:?}: {} draws, {} perspective{binds}", describe(gl, framebuffer), trace.draws, trace.scene));
    }
    trace.binds.clear();
    trace.draws = 0;
    trace.scene = 0;
}

fn traced_draw(state: &mut State, scene: bool) {
    let Some(mut trace) = state.trace.take() else { return };
    let run = (state.draw_framebuffer, state.viewport);
    if trace.run != Some(run) {
        end_run(state.gl.as_ref(), &mut trace);
        trace.run = Some(run);
    }
    trace.draws += 1;
    trace.scene += u32::from(scene);
    state.trace = Some(trace);
}

/// A clear between the runs of draws.
fn traced(state: &mut State, text: impl FnOnce() -> String) {
    let Some(mut trace) = state.trace.take() else { return };
    end_run(state.gl.as_ref(), &mut trace);
    if trace.lines.len() < TRACED_LINES {
        trace.lines.push(text());
    }
    state.trace = Some(trace);
}

/// A texture given to the coming draws, noted on the traced run (once each).
fn given(state: &mut State, text: impl FnOnce() -> String) {
    let Some(trace) = state.trace.as_mut() else { return };
    let text = text();
    if !trace.binds.contains(&text) && trace.binds.len() < 48 {
        trace.binds.push(text);
    }
}

fn active_unit() -> u32 {
    let mut unit = 0;
    // SAFETY: a state query into a local (GL_ACTIVE_TEXTURE).
    unsafe { glGetIntegerv(0x84E0, &mut unit) };
    (unit as u32).wrapping_sub(GL_TEXTURE0)
}

/// The texture to bind to `unit` for the engine's `texture` (kept as an eye texture's binding).
fn unit_texture(unit: u32, target: u32, texture: u32) -> u32 {
    let mut guard = STATE.locked();
    let Some(state) = guard.as_mut().filter(|s| s.live) else { return texture };
    given(state, || format!("t{texture}@u{unit}"));
    let given = state.texture(texture);
    if given == texture {
        state.units.remove(&unit);
    } else {
        state.units.insert(unit, (target, texture));
    }
    given
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
    }
    // SAFETY: forwards the game's own call.
    unsafe { VIEWPORT.get(&flight)(x, y, w, h) }
}

unsafe extern "system" fn clear(mask: u32) {
    let flight = InFlight::enter();
    if let Some(state) = STATE.locked().as_mut() {
        let framebuffer = state.draw_framebuffer;
        traced(state, || format!("clear {mask:#x} fb {framebuffer}"));
    }
    if clear_goes_ahead(mask) {
        // SAFETY: forwards the game's own call.
        unsafe { CLEAR.get(&flight)(mask) }
    }
}

/// The engine's framebuffer bind, made with the framebuffer standing in for it.
fn bound(target: u32, framebuffer: u32) -> u32 {
    let mut guard = STATE.locked();
    let Some(state) = guard.as_mut().filter(|s| s.live) else { return framebuffer };
    if target == GL_FRAMEBUFFER || target == GL_DRAW_FRAMEBUFFER {
        state.draw_framebuffer = framebuffer;
    }
    if target == GL_FRAMEBUFFER || target == GL_READ_FRAMEBUFFER {
        state.read_framebuffer = framebuffer;
    }
    state.mapped(framebuffer)
}

unsafe extern "system" fn bind_framebuffer(target: u32, framebuffer: u32) {
    let flight = InFlight::enter();
    let framebuffer = bound(target, framebuffer);
    // SAFETY: forwards the game's own call (with our stand-in).
    unsafe { BIND_FRAMEBUFFER.get(&flight)(target, framebuffer) }
}

unsafe extern "system" fn bind_framebuffer_ext(target: u32, framebuffer: u32) {
    let flight = InFlight::enter();
    let framebuffer = bound(target, framebuffer);
    // SAFETY: forwards the game's own call (with our stand-in).
    unsafe { BIND_FRAMEBUFFER_EXT.get(&flight)(target, framebuffer) }
}

unsafe extern "system" fn bind_texture(target: u32, texture: u32) {
    let flight = InFlight::enter();
    let texture = unit_texture(active_unit(), target, texture);
    // SAFETY: forwards the game's own call (with our stand-in).
    unsafe { BIND_TEXTURE.get(&flight)(target, texture) }
}

unsafe extern "system" fn bind_multi_texture(unit: u32, target: u32, texture: u32) {
    let flight = InFlight::enter();
    let texture = unit_texture(unit.wrapping_sub(GL_TEXTURE0), target, texture);
    // SAFETY: forwards the game's own call (with our stand-in).
    unsafe { BIND_MULTI_TEXTURE.get(&flight)(unit, target, texture) }
}

unsafe extern "system" fn bind_textures(first: u32, count: i32, textures: *const u32) {
    let flight = InFlight::enter();
    let count_usize = count.max(0) as usize;
    let given: Option<Vec<u32>> = (!textures.is_null()).then(|| {
        // SAFETY: the game's array of `count` names.
        let names = unsafe { std::slice::from_raw_parts(textures, count_usize) };
        // The target is the engine's own for its eye textures: 2D.
        names.iter().enumerate().map(|(i, &t)| unit_texture(first + i as u32, 0x0DE1, t)).collect()
    });
    if given.is_none() {
        for i in 0..count_usize {
            unit_texture(first + i as u32, 0x0DE1, 0);
        }
    }
    let pointer = given.as_ref().map_or(textures, |g| g.as_ptr());
    // SAFETY: forwards the game's own call (with our stand-ins).
    unsafe { BIND_TEXTURES.get(&flight)(first, count, pointer) }
}

/// The occlusion queries' targets: samples passed, any passed, any passed (conservative).
fn occlusion(target: u32) -> bool {
    matches!(target, 0x8914 | 0x8C2F | 0x8D6A)
}

unsafe extern "system" fn begin_query(target: u32, id: u32) {
    let flight = InFlight::enter();
    if occlusion(target)
        && let Some(state) = STATE.locked().as_mut().filter(|s| s.live)
    {
        state.query = Some(id);
        state.blind.remove(&id);
        state.counts.queries[state.pass.min(2) as usize] += 1;
    }
    // SAFETY: forwards the game's own call.
    unsafe { BEGIN_QUERY.get(&flight)(target, id) }
}

unsafe extern "system" fn end_query(target: u32) {
    let flight = InFlight::enter();
    if occlusion(target)
        && let Some(state) = STATE.locked().as_mut()
    {
        state.query = None;
    }
    // SAFETY: forwards the game's own call.
    unsafe { END_QUERY.get(&flight)(target) }
}

/// Whether a query's result read back is to be made visible (some of its draws skipped).
fn blind_result(id: u32, pname: u32) -> bool {
    // GL_QUERY_RESULT, GL_QUERY_RESULT_NO_WAIT.
    if pname != 0x8866 && pname != 0x9194 {
        return false;
    }
    let mut guard = STATE.locked();
    let Some(state) = guard.as_mut().filter(|s| s.live && s.blind.contains(&id)) else { return false };
    state.counts.blind_results += 1;
    true
}

macro_rules! query_detour {
    ($name:ident, $original:ident, $type:ty) => {
        unsafe extern "system" fn $name(id: u32, pname: u32, params: *mut $type) {
            let flight = InFlight::enter();
            // SAFETY: forwards the game's own call.
            unsafe { $original.get(&flight)(id, pname, params) };
            if !params.is_null() && blind_result(id, pname) {
                // SAFETY: the game's own result, written into its pointer above.
                unsafe {
                    if *params == 0 {
                        *params = 1;
                    }
                }
            }
        }
    };
}

query_detour!(get_query_object_iv, GET_QUERY_OBJECT_IV, i32);
query_detour!(get_query_object_uiv, GET_QUERY_OBJECT_UIV, u32);
query_detour!(get_query_object_i64v, GET_QUERY_OBJECT_I64V, i64);
query_detour!(get_query_object_ui64v, GET_QUERY_OBJECT_UI64V, u64);

unsafe extern "system" fn begin_conditional_render(id: u32, mode: u32) {
    let flight = InFlight::enter();
    let leave_out = match STATE.locked().as_mut() {
        Some(state) => {
            state.unconditional = state.live && state.blind.contains(&id);
            state.counts.blind_conditions += u64::from(state.unconditional);
            state.unconditional
        }
        None => false,
    };
    if !leave_out {
        // SAFETY: forwards the game's own call.
        unsafe { BEGIN_CONDITIONAL_RENDER.get(&flight)(id, mode) }
    }
}

unsafe extern "system" fn end_conditional_render() {
    let flight = InFlight::enter();
    let left_out = STATE.locked().as_mut().is_some_and(|s| std::mem::take(&mut s.unconditional));
    if !left_out {
        // SAFETY: forwards the game's own call.
        unsafe { END_CONDITIONAL_RENDER.get(&flight)() }
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
draw_detour!(draw_elements_indirect, DRAW_ELEMENTS_INDIRECT, DrawElementsIndirectFn, (mode: u32, kind: u32, indirect: *const c_void));
draw_detour!(multi_draw_elements_indirect, MULTI_DRAW_ELEMENTS_INDIRECT, MultiDrawElementsIndirectFn, (mode: u32, kind: u32, indirect: *const c_void, draws: i32, stride: i32));
draw_detour!(draw_arrays_indirect, DRAW_ARRAYS_INDIRECT, DrawArraysIndirectFn, (mode: u32, indirect: *const c_void));
draw_detour!(multi_draw_arrays_indirect, MULTI_DRAW_ARRAYS_INDIRECT, MultiDrawArraysIndirectFn, (mode: u32, indirect: *const c_void, draws: i32, stride: i32));
draw_detour!(multi_draw_elements, MULTI_DRAW_ELEMENTS, MultiDrawElementsFn, (mode: u32, counts: *const i32, kind: u32, indices: *const *const c_void, draws: i32));
draw_detour!(multi_draw_elements_base_vertex, MULTI_DRAW_ELEMENTS_BASE_VERTEX, MultiDrawElementsBaseVertexFn, (mode: u32, counts: *const i32, kind: u32, indices: *const *const c_void, draws: i32, bases: *const i32));
draw_detour!(draw_elements_instanced_base_instance, DRAW_ELEMENTS_INSTANCED_BASE_INSTANCE, DrawElementsInstancedBaseInstanceFn, (mode: u32, count: i32, kind: u32, indices: *const c_void, instances: i32, first: u32));
draw_detour!(draw_elements_instanced_base_vertex_base_instance, DRAW_ELEMENTS_INSTANCED_BASE_VERTEX_BASE_INSTANCE, DrawElementsInstancedBaseVertexBaseInstanceFn, (mode: u32, count: i32, kind: u32, indices: *const c_void, instances: i32, base: i32, first: u32));
draw_detour!(draw_arrays_instanced_base_instance, DRAW_ARRAYS_INSTANCED_BASE_INSTANCE, DrawArraysInstancedBaseInstanceFn, (mode: u32, first: i32, count: i32, instances: i32, base: u32));
draw_detour!(multi_draw_elements_indirect_bindless, MULTI_DRAW_ELEMENTS_INDIRECT_BINDLESS, MultiDrawElementsIndirectBindlessFn, (mode: u32, kind: u32, indirect: *const c_void, draws: i32, stride: i32, buffers: i32));
