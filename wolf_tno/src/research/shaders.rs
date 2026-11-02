//! Probe (`probe_shaders`): every shader the game has given the driver, read back out of it and
//! written beside the run's log (`shaders\`), the first step of single-pass stereo (each scene
//! shader converted to draw both eyes). Monaka attaches at the game's menu, after the engine made
//! its programs, so nothing is hooked: on the render thread, with the game's context current, the
//! GL names are swept (`glIsProgram`, `glIsProgramARB`) and each program gives what the driver
//! keeps of it: the GLSL source of its attached shaders, the text of an ARB/NV assembly program,
//! and the program binary where no source is attached (a separable program, or one loaded from a
//! binary). Swept once gameplay starts, then again now and then for programs made at level loads;
//! each piece is written once (by its hash). `shaders\index.txt` lists them.

use monaka_core::sync::Locked;
use monaka_producer::{Session, log};
use std::collections::BTreeSet;
use std::ffi::c_void;
use std::fmt::Write as _;
use std::path::PathBuf;
use std::sync::Mutex;
use windows::Win32::Graphics::OpenGL::{wglGetCurrentContext, wglGetProcAddress};
use windows::core::PCSTR;

/// The first sweep this many frames after views began, then one every [`EVERY`] frames.
const FIRST: u64 = 60;
const EVERY: u64 = 1800;
/// Names swept: up to this many past the highest found so far (and at least [`LEAST`]).
const PAST_HIGHEST: u32 = 4096;
const LEAST: u32 = 16384;

const GL_SHADER_TYPE: u32 = 0x8B4F;
const GL_SHADER_SOURCE_LENGTH: u32 = 0x8B88;
const GL_ATTACHED_SHADERS: u32 = 0x8B85;
const GL_PROGRAM_BINARY_LENGTH: u32 = 0x8741;
const GL_PROGRAM_LENGTH_ARB: u32 = 0x8627;
const GL_PROGRAM_STRING_ARB: u32 = 0x8628;
const GL_PROGRAM_BINDING_ARB: u32 = 0x8677;
/// The assembly program targets, with the file extension each is written under.
const ARB_TARGETS: [(u32, &str); 6] = [(0x8620, "vp"), (0x8804, "fp"), (0x8C26, "gp"), (0x891E, "tcp"), (0x891F, "tep"), (0x90FB, "cp")];

type IsFn = unsafe extern "system" fn(u32) -> u8;
type GetAttachedFn = unsafe extern "system" fn(u32, i32, *mut i32, *mut u32);
type GetIvFn = unsafe extern "system" fn(u32, u32, *mut i32);
type GetSourceFn = unsafe extern "system" fn(u32, i32, *mut i32, *mut u8);
type GetBinaryFn = unsafe extern "system" fn(u32, i32, *mut i32, *mut u32, *mut c_void);
type BindArbFn = unsafe extern "system" fn(u32, u32);
type GetStringArbFn = unsafe extern "system" fn(u32, u32, *mut c_void);

/// The driver's entry points the sweep reads with.
struct Gl {
    is_program: IsFn,
    get_attached_shaders: GetAttachedFn,
    get_shader_iv: GetIvFn,
    get_shader_source: GetSourceFn,
    get_program_iv: GetIvFn,
    get_program_binary: Option<GetBinaryFn>,
    arb: Option<Arb>,
}

/// The assembly programs' entry points (`GL_ARB_vertex_program`).
struct Arb {
    is_program: IsFn,
    bind_program: BindArbFn,
    get_program_iv: GetIvFn,
    get_program_string: GetStringArbFn,
}

fn load<F: Copy>(name: &str) -> Option<F> {
    let text = format!("{name}\0");
    // SAFETY: asks the current context's driver for a function by NUL-terminated name.
    let function = unsafe { wglGetProcAddress(PCSTR(text.as_ptr())) }?;
    if (function as usize) < 4 {
        return None;
    }
    // SAFETY: the driver's function, of the type its specification gives.
    Some(unsafe { std::mem::transmute_copy(&function) })
}

impl Gl {
    fn load() -> Option<Self> {
        // SAFETY: queries this thread's current context.
        if unsafe { wglGetCurrentContext() }.is_invalid() {
            return None;
        }
        Some(Self {
            is_program: load("glIsProgram")?,
            get_attached_shaders: load("glGetAttachedShaders")?,
            get_shader_iv: load("glGetShaderiv")?,
            get_shader_source: load("glGetShaderSource")?,
            get_program_iv: load("glGetProgramiv")?,
            get_program_binary: load("glGetProgramBinary"),
            arb: (|| {
                Some(Arb {
                    is_program: load("glIsProgramARB")?,
                    bind_program: load("glBindProgramARB")?,
                    get_program_iv: load("glGetProgramivARB")?,
                    get_program_string: load("glGetProgramStringARB")?,
                })
            })(),
        })
    }

    /// Clears the error flags the sweep's own calls may have raised.
    fn clear_errors(&self) -> bool {
        let mut any = false;
        for _ in 0..8 {
            // SAFETY: a plain GL call (opengl32's own export) on the current context.
            if unsafe { windows::Win32::Graphics::OpenGL::glGetError() } == 0 {
                break;
            }
            any = true;
        }
        any
    }
}

/// What the probe has seen: the pieces written (by hash), the highest name found, the sweeps.
struct Seen {
    written: BTreeSet<u64>,
    highest: u32,
    sweeps: u32,
    index: String,
    started: Option<u64>,
}

static SEEN: Mutex<Seen> = Mutex::new(Seen { written: BTreeSet::new(), highest: 0, sweeps: 0, index: String::new(), started: None });

/// At each SwapBuffers (render thread, the game's context current), `frame` its count: a sweep
/// when one is due.
pub fn frame(frame: u64) {
    if !super::options().shaders {
        return;
    }
    let mut seen = SEEN.locked();
    if seen.started.is_none() && crate::providers::eye_site::builds() > 0 {
        seen.started = Some(frame);
    }
    let Some(started) = seen.started else { return };
    let since = frame.saturating_sub(started);
    if since < FIRST || (since - FIRST) % EVERY != 0 {
        return;
    }
    let Some(gl) = Gl::load() else {
        return log!("probe_shaders: no GL context on the render thread");
    };
    let Some(folder) = Session::get().map(|s| s.path("shaders")) else { return };
    if let Err(e) = std::fs::create_dir_all(&folder) {
        return log!("probe_shaders: {}: {e}", folder.display());
    }
    sweep(&gl, &folder, &mut seen);
}

fn fnv(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &b| (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3))
}

/// The extension a GLSL shader of `kind` is written under.
fn glsl_extension(kind: i32) -> &'static str {
    match kind as u32 {
        0x8B31 => "vert",
        0x8B30 => "frag",
        0x8DD9 => "geom",
        0x8E88 => "tesc",
        0x8E87 => "tese",
        0x91B9 => "comp",
        _ => "glsl",
    }
}

/// Writes `bytes` as `<hash>.<extension>` once; a line for the index either way.
fn keep(folder: &PathBuf, seen: &mut Seen, name: u32, what: &str, extension: &str, bytes: &[u8]) -> bool {
    let hash = fnv(bytes);
    let file = format!("{hash:016x}.{extension}");
    let _ = writeln!(seen.index, "{name}\t{what}\t{}\t{file}", bytes.len());
    if !seen.written.insert(hash) {
        return false;
    }
    if let Err(e) = std::fs::write(folder.join(&file), bytes) {
        log!("probe_shaders: {file}: {e}");
    }
    true
}

fn sweep(gl: &Gl, folder: &PathBuf, seen: &mut Seen) {
    gl.clear_errors();
    let last = seen.highest.saturating_add(PAST_HIGHEST).max(LEAST);
    let (mut programs, mut arb_programs, mut new) = (0u32, 0u32, 0u32);
    seen.index.clear();
    for name in 1..=last {
        // SAFETY: plain queries on the current context.
        if unsafe { (gl.is_program)(name) } != 0 {
            programs += 1;
            seen.highest = seen.highest.max(name);
            new += glsl_program(gl, folder, seen, name);
        }
        if let Some(arb) = &gl.arb {
            // SAFETY: as above.
            if unsafe { (arb.is_program)(name) } != 0 {
                arb_programs += 1;
                seen.highest = seen.highest.max(name);
                new += arb_program(gl, arb, folder, seen, name);
            }
        }
    }
    gl.clear_errors();
    seen.sweeps += 1;
    let index = std::mem::take(&mut seen.index);
    let _ = std::fs::write(folder.join(format!("index-{}.txt", seen.sweeps)), &index);
    log!("probe_shaders: sweep {}: {programs} GLSL programs, {arb_programs} assembly programs, {new} new pieces written to {}", seen.sweeps, folder.display());
}

/// A GLSL program's attached shaders' sources, or its binary when none is attached. New pieces.
fn glsl_program(gl: &Gl, folder: &PathBuf, seen: &mut Seen, program: u32) -> u32 {
    let mut attached = 0;
    // SAFETY: plain queries on the current context, into locals and buffers of the sizes given.
    unsafe { (gl.get_program_iv)(program, GL_ATTACHED_SHADERS, &mut attached) };
    let mut new = 0;
    if attached > 0 {
        let mut shaders = vec![0u32; attached as usize];
        let mut count = 0;
        unsafe { (gl.get_attached_shaders)(program, attached, &mut count, shaders.as_mut_ptr()) };
        for &shader in &shaders[..count.clamp(0, attached) as usize] {
            let (mut kind, mut length) = (0, 0);
            unsafe {
                (gl.get_shader_iv)(shader, GL_SHADER_TYPE, &mut kind);
                (gl.get_shader_iv)(shader, GL_SHADER_SOURCE_LENGTH, &mut length);
            }
            if length <= 1 {
                continue;
            }
            let mut source = vec![0u8; length as usize];
            let mut written = 0;
            unsafe { (gl.get_shader_source)(shader, length, &mut written, source.as_mut_ptr()) };
            source.truncate(written.clamp(0, length) as usize);
            let extension = glsl_extension(kind);
            new += u32::from(keep(folder, seen, program, &format!("glsl {extension} shader {shader}"), extension, &source));
        }
        return new;
    }
    let Some(get_binary) = gl.get_program_binary else { return 0 };
    let mut length = 0;
    unsafe { (gl.get_program_iv)(program, GL_PROGRAM_BINARY_LENGTH, &mut length) };
    if length <= 0 {
        return 0;
    }
    let mut binary = vec![0u8; length as usize];
    let (mut written, mut format) = (0, 0u32);
    unsafe { get_binary(program, length, &mut written, &mut format, binary.as_mut_ptr().cast()) };
    binary.truncate(written.clamp(0, length) as usize);
    u32::from(keep(folder, seen, program, &format!("binary format {format:#x}"), "bin", &binary))
}

/// An assembly program's text, found by binding it to each target until one takes it (the
/// binding put back after). New pieces.
fn arb_program(gl: &Gl, arb: &Arb, folder: &PathBuf, seen: &mut Seen, program: u32) -> u32 {
    for (target, extension) in ARB_TARGETS {
        let mut bound = 0;
        // SAFETY: plain GL calls on the current context; the binding is restored before returning.
        unsafe { (arb.get_program_iv)(target, GL_PROGRAM_BINDING_ARB, &mut bound) };
        if gl.clear_errors() {
            continue;
        }
        unsafe { (arb.bind_program)(target, program) };
        if gl.clear_errors() {
            continue;
        }
        let mut length = 0;
        unsafe { (arb.get_program_iv)(target, GL_PROGRAM_LENGTH_ARB, &mut length) };
        let mut text = vec![0u8; length.max(0) as usize];
        if length > 0 {
            unsafe { (arb.get_program_string)(target, GL_PROGRAM_STRING_ARB, text.as_mut_ptr().cast()) };
        }
        unsafe { (arb.bind_program)(target, bound as u32) };
        gl.clear_errors();
        if text.is_empty() {
            return 0;
        }
        return u32::from(keep(folder, seen, program, &format!("assembly {extension}"), extension, &text));
    }
    0
}
