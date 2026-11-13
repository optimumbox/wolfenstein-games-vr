//! Decals (blood, bullet holes) with each eye drawn in its own pass. A decal's fragment shader
//! takes its pixel back to the world with four `_fa_` rows ([`convert::unprojection_row`]): the
//! inverse of the view-projection, window to world. The engine gives them the left eye's in both
//! passes, so the right eye's pixels land outside every decal's box and nothing is drawn there.
//! (Single pass gives the right eye's from the eye matrix: [`super::draws`].)
//!
//! Here each decal program's rows are made from its own position rows (`_va_`, the box's
//! view-projection for the eye being drawn): `(T · VP)⁻¹` ([`eye_matrix::window_unprojection`]).
//! That holds only if the box is drawn in world space, so it is checked against the engine first:
//! a program whose engine rows match its own in some of its uploads (the left eye's) is given its
//! own each time; one that never matches is left as the engine draws it.

use super::convert;
use super::archive::Renderprog;
use super::eye_matrix::{self, Rows};
use super::gl_api::{self, Gl};
use monaka_hook::module::Module;
use monaka_hook::{Hooks, InFlight, Original};
use monaka_producer::log;
use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

/// A decal program: the engine's program, the locations of its four position rows and of the
/// first of its four unprojection rows.
#[derive(Clone, Copy, Debug)]
pub struct Decal {
    pub engine: u32,
    pub position: [i32; 4],
    pub unprojection: i32,
}

/// [`Decal`] for `engine` linked from `source`, if it unprojects its pixel.
pub fn decal(gl: &Gl, engine: u32, source: &Renderprog) -> Option<Decal> {
    let row = convert::unprojection_row(&source.fragment)?;
    let rows = convert::position_rows(&source.vertex)?;
    // SAFETY: plain queries with NUL-terminated names.
    let location = |name: String| unsafe { (gl.get_uniform_location)(engine, gl_api::c_name(&name).as_ptr()) };
    let unprojection = location(format!("_fa_[{row}]"));
    let position = rows.map(|r| location(format!("_va_[{r}]")));
    (unprojection >= 0 && position.iter().all(|&l| l >= 0)).then_some(Decal { engine, position, unprojection })
}

/// The decal programs found ([`found`]), until the hooks take them.
static FOUND: Mutex<Option<Vec<Decal>>> = Mutex::new(None);

/// The decal programs the conversion matched (render thread).
pub fn found(decals: Vec<Decal>) {
    log!("decals: {} decal programs found", decals.len());
    if let Ok(mut found) = FOUND.lock() {
        *found = Some(decals);
    }
}

/// How far (each column, relative to its largest entry) the engine's rows may be from a program's
/// own and still be the same eye's: the jitter moves them by far less, the other eye's frustum by
/// far more.
const SAME: f64 = 0.02;
/// The matching uploads a program needs before it is given its own rows, and the share of its
/// uploads they must be (two-pass: about half match, the left eye's).
const TRUSTED_AFTER: u32 = 16;
const TRUSTED_SHARE: u32 = 5;

struct Program {
    decal: Decal,
    position: Rows,
    position_seen: [bool; 4],
    engine: Rows,
    engine_seen: [bool; 4],
    /// The engine's rows over the program's own, from the last matching upload.
    scale: f32,
    compared: u32,
    matched: u32,
    /// The rows in the program are ours, not the engine's last.
    given: bool,
}

struct State {
    programs: HashMap<u32, Program>,
    current: u32,
    write: unsafe extern "system" fn(u32, i32, i32, *const f32),
}

static STATE: Mutex<Option<State>> = Mutex::new(None);
static HOOKS: Mutex<Option<Hooks>> = Mutex::new(None);
/// Uploads compared, matched, and given the eye's own rows, since the last report.
static COMPARED: AtomicU64 = AtomicU64::new(0);
static MATCHED: AtomicU64 = AtomicU64::new(0);
static GIVEN: AtomicU64 = AtomicU64::new(0);
static FRAMES: AtomicU64 = AtomicU64::new(0);
const REPORT_EVERY: u64 = 900;

type UseProgramFn = unsafe extern "system" fn(u32);
type Uniform4fvFn = unsafe extern "system" fn(i32, i32, *const f32);
type ProgramUniform4fvFn = unsafe extern "system" fn(u32, i32, i32, *const f32);
static USE_PROGRAM: Original<UseProgramFn> = Original::new();
static UNIFORM_4FV: Original<Uniform4fvFn> = Original::new();
static PROGRAM_UNIFORM_4FV: Original<ProgramUniform4fvFn> = Original::new();

/// At each `SwapBuffers` (render thread, the game's context current): the hooks in once the decal
/// programs are found, out when `on` is not (single pass hooks the same functions).
pub fn frame(on: bool) {
    let installed = HOOKS.lock().is_ok_and(|h| h.is_some());
    if !on {
        if installed {
            let _ = let_go();
        }
        return;
    }
    if installed {
        let frames = FRAMES.fetch_add(1, Relaxed) + 1;
        if frames % REPORT_EVERY == 0 {
            report();
        }
        return;
    }
    let Some(decals) = FOUND.lock().ok().and_then(|mut f| f.take()) else { return };
    if decals.is_empty() {
        return;
    }
    let Ok(gl) = Gl::load() else { return };
    let programs = decals
        .into_iter()
        .map(|decal| {
            let program = Program { decal, position: [[0.0; 4]; 4], position_seen: [false; 4], engine: [[0.0; 4]; 4], engine_seen: [false; 4], scale: 1.0, compared: 0, matched: 0, given: false };
            (decal.engine, program)
        })
        .collect();
    *STATE.lock().unwrap_or_else(|e| e.into_inner()) = Some(State { programs, current: 0, write: gl.program_uniform_4fv });
    if !install() {
        *STATE.lock().unwrap_or_else(|e| e.into_inner()) = None;
    }
}

fn install() -> bool {
    let Some(exe) = Module::find(&monaka_hook::module::host_name()) else { return false };
    let opengl = Module::find("opengl32.dll");
    let mut hooks = Hooks::default();
    let mut found = Vec::new();
    macro_rules! hook {
        ($original:expr, $name:literal, $detour:expr) => {{
            let mut at: Vec<usize> = [gl_api::address($name), opengl.as_ref().and_then(|m| m.export($name))]
                .into_iter()
                .flatten()
                .flat_map(|address| super::draws::slots(&exe, address))
                .collect();
            at.sort_unstable();
            at.dedup();
            found.push(format!("{} {}", $name, at.len()));
            for slot in &at {
                // SAFETY: the slot holds the GL function of that name, called through it.
                if let Err(e) = unsafe { hooks.pointer(&$original, $name, *slot, $detour) } {
                    log!("decals: left as the engine draws them: {e}");
                    return false;
                }
            }
            at.len()
        }};
    }
    let programs = hook!(USE_PROGRAM, "glUseProgram", use_program as UseProgramFn);
    let uploads = hook!(UNIFORM_4FV, "glUniform4fv", uniform_4fv as Uniform4fvFn) + hook!(PROGRAM_UNIFORM_4FV, "glProgramUniform4fv", program_uniform_4fv as ProgramUniform4fvFn);
    if programs == 0 || uploads == 0 {
        log!("decals: left as the engine draws them: its GL functions are not in its data ({})", found.join(", "));
        return false;
    }
    if let Err(e) = hooks.enable() {
        let _ = hooks.disable();
        log!("decals: left as the engine draws them: {e}");
        return false;
    }
    log!("decals: each eye's decals given that eye's unprojection where the engine's is checked to be the view's ({})", found.join(", "));
    *HOOKS.lock().unwrap_or_else(|e| e.into_inner()) = Some(hooks);
    true
}

/// The hooks out (the stop, or single pass on).
pub fn let_go() -> Result<(), monaka_producer::Rejection> {
    let hooks = HOOKS.lock().unwrap_or_else(|e| e.into_inner()).take();
    if let Some(mut hooks) = hooks
        && let Err(e) = hooks.disable()
    {
        log!("decals: {e}");
    }
    InFlight::wait_idle(500);
    *STATE.lock().unwrap_or_else(|e| e.into_inner()) = None;
    Ok(())
}

fn report() {
    let (compared, matched, given) = (COMPARED.swap(0, Relaxed), MATCHED.swap(0, Relaxed), GIVEN.swap(0, Relaxed));
    let (trusted, programs) = STATE.lock().ok().and_then(|s| s.as_ref().map(|s| (s.programs.values().filter(|p| trusted(p)).count(), s.programs.len()))).unwrap_or((0, 0));
    log!("decals: {compared} uploads checked, {matched} matched the view's own, {given} given the eye's own rows; {trusted} of {programs} programs checked");
}

fn trusted(p: &Program) -> bool {
    p.matched >= TRUSTED_AFTER && p.matched * TRUSTED_SHARE >= p.compared
}

/// Rows uploaded to `program` from `location` on: the rows to write after the engine's upload, if
/// any.
fn uploaded(state: &mut State, program: u32, location: i32, count: i32, values: *const f32) -> Option<(i32, Rows)> {
    let p = state.programs.get_mut(&program)?;
    if values.is_null() {
        return None;
    }
    let read = |at: i32| {
        let offset = at - location;
        // SAFETY: the engine's buffer holds `count` vec4s.
        (offset >= 0 && offset < count).then(|| unsafe { *values.add(offset as usize * 4).cast::<[f32; 4]>() })
    };
    let mut touched = false;
    for row in 0..4 {
        if let Some(v) = read(p.decal.position[row]) {
            p.position[row] = v;
            p.position_seen[row] = true;
            touched = true;
        }
        if let Some(v) = read(p.decal.unprojection + row as i32) {
            p.engine[row] = v;
            p.engine_seen[row] = true;
            p.given = false;
            touched = true;
        }
    }
    if !touched || !p.position_seen.iter().chain(&p.engine_seen).all(|&s| s) {
        return None;
    }
    let own = eye_matrix::window_unprojection(&p.position)?;
    let (scale, apart) = eye_matrix::scaled_apart(&p.engine, &own);
    p.compared += 1;
    COMPARED.fetch_add(1, Relaxed);
    let same = apart < SAME;
    if same {
        p.matched += 1;
        p.scale = scale;
        MATCHED.fetch_add(1, Relaxed);
    }
    if p.compared >= 1 << 16 {
        p.compared /= 2;
        p.matched /= 2;
    }
    if !trusted(p) || (same && !p.given) {
        return None;
    }
    p.given = !same;
    if !same {
        GIVEN.fetch_add(1, Relaxed);
    }
    Some((p.decal.unprojection, own.map(|row| row.map(|v| v * p.scale))))
}

fn after_upload(program: u32, location: i32, count: i32, values: *const f32) {
    let mut guard = STATE.lock().unwrap_or_else(|e| e.into_inner());
    let Some(state) = guard.as_mut() else { return };
    if let Some((at, rows)) = uploaded(state, program, location, count, values) {
        // SAFETY: four vec4 rows into the engine's program, at its own array's location.
        unsafe { (state.write)(program, at, 4, rows.as_ptr().cast()) };
    }
}

unsafe extern "system" fn use_program(program: u32) {
    let flight = InFlight::enter();
    if let Some(state) = STATE.lock().unwrap_or_else(|e| e.into_inner()).as_mut() {
        state.current = program;
    }
    // SAFETY: forwards the game's own call.
    unsafe { USE_PROGRAM.get(&flight)(program) }
}

unsafe extern "system" fn uniform_4fv(location: i32, count: i32, values: *const f32) {
    let flight = InFlight::enter();
    // SAFETY: forwards the game's own call.
    unsafe { UNIFORM_4FV.get(&flight)(location, count, values) };
    let current = STATE.lock().ok().and_then(|s| s.as_ref().map(|s| s.current)).unwrap_or(0);
    after_upload(current, location, count, values);
}

unsafe extern "system" fn program_uniform_4fv(program: u32, location: i32, count: i32, values: *const f32) {
    let flight = InFlight::enter();
    // SAFETY: forwards the game's own call.
    unsafe { PROGRAM_UNIFORM_4FV.get(&flight)(program, location, count, values) };
    after_upload(program, location, count, values);
}
