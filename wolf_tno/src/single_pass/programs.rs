//! The engine's programs converted in place, on the render thread over a few frames:
//!
//! 1. The render programs are read from the game's archive (on a thread of their own).
//! 2. Each is compiled as it is, in parallel (`KHR_parallel_shader_compile`), for its assembly.
//! 3. The engine's programs are swept; each one's binary assembly names the render program it was
//!    made from ([`gl_api::assembly_key`]; where several share it, the one with the same uniform
//!    names). Those whose vertex shader writes the position as matrix rows
//!    ([`convert::position_rows`]) are converted with the engine program's own uniform, attribute
//!    and output locations, and linked in parallel.
//! 4. Each converted binary goes into the engine's program (`glProgramBinary`), its uniform values
//!    and block bindings put back as they were; a program that does not take it gets its own
//!    binary back.
//!
//! A converted program draws exactly as before until its stereo switch is set ([`super::draws`]).

use super::archive::{self, Renderprog};
use super::convert;
use super::gl_api::{self, ActiveUniform, Gl};
use monaka_producer::log;
use std::collections::HashMap;
use std::sync::mpsc::{Receiver, channel};
use std::time::{Duration, Instant};

/// A converted engine program, as the draw hooks use it.
#[derive(Clone, Copy, Debug)]
pub struct Converted {
    pub program: u32,
    /// The locations of the `_va_` rows that give the position (x, y, z, w).
    pub rows: [i32; 4],
    /// The locations of the stereo uniform's rows: the eye matrix's four, then the switch.
    pub stereo: [i32; convert::STEREO_ROWS],
}

/// The engine's program names swept.
const LAST_NAME: u32 = 16384;
/// Programs moved into the engine's per frame (each a binary load and a uniform copy).
const SWAPS_PER_FRAME: usize = 48;
/// The longest the driver is waited for, per batch of compiles.
const COMPILE_PATIENCE: Duration = Duration::from_secs(60);

enum Stage {
    Idle,
    Reading(Receiver<Result<Vec<Renderprog>, String>>),
    Compiling { sources: Vec<Renderprog>, scratch: Vec<u32>, since: Instant },
    Converting { jobs: Vec<Job>, since: Instant },
    Swapping { jobs: Vec<Job>, done: usize },
    Done,
    Failed,
}

/// One engine program to convert: its render program and the converted scratch program.
struct Job {
    engine: u32,
    source: usize,
    rows: [usize; 4],
    scratch: u32,
}

pub struct Conversion {
    stage: Stage,
    sources: Vec<Renderprog>,
    converted: Vec<Converted>,
    failed: usize,
}

impl Conversion {
    pub const fn new() -> Self {
        Self { stage: Stage::Idle, sources: Vec::new(), converted: Vec::new(), failed: 0 }
    }

    /// Whether it ended (converted or given up).
    pub fn finished(&self) -> bool {
        matches!(self.stage, Stage::Done | Stage::Failed)
    }

    /// One frame's step; the programs converted when it finishes this frame.
    pub fn step(&mut self, gl: &Gl) -> Option<Vec<Converted>> {
        let stage = std::mem::replace(&mut self.stage, Stage::Failed);
        self.stage = match stage {
            Stage::Idle => start_reading(),
            Stage::Reading(receiver) => match receiver.try_recv() {
                Ok(Ok(sources)) => {
                    log!("single pass: {} render programs read from the game's archive; compiling them for their assembly", sources.len());
                    parallel_compiles();
                    let scratch = sources.iter().map(|s| gl.program(&s.vertex, &s.fragment, &[], &outputs(s, |_| None))).collect();
                    Stage::Compiling { sources, scratch, since: Instant::now() }
                }
                Ok(Err(e)) => {
                    log!("single pass: off: the render programs could not be read: {e}");
                    Stage::Failed
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => Stage::Reading(receiver),
                Err(_) => Stage::Failed,
            },
            Stage::Compiling { sources, scratch, since } => {
                if !scratch.iter().all(|&p| gl.program_done(p)) && since.elapsed() < COMPILE_PATIENCE {
                    Stage::Compiling { sources, scratch, since }
                } else {
                    self.sources = sources;
                    match_and_convert(gl, &self.sources, &scratch)
                }
            }
            Stage::Converting { jobs, since } => {
                if !jobs.iter().all(|j| gl.program_done(j.scratch)) && since.elapsed() < COMPILE_PATIENCE {
                    Stage::Converting { jobs, since }
                } else {
                    Stage::Swapping { jobs, done: 0 }
                }
            }
            Stage::Swapping { jobs, done } => {
                let end = (done + SWAPS_PER_FRAME).min(jobs.len());
                for job in &jobs[done..end] {
                    match swap(gl, job, &self.sources[job.source]) {
                        Ok(converted) => self.converted.push(converted),
                        Err(why) => {
                            self.failed += 1;
                            if self.failed <= 10 {
                                log!("single pass: program {} ({}) stays as it was: {why}", job.engine, self.sources[job.source].name);
                            }
                        }
                    }
                    // SAFETY: our own scratch program.
                    unsafe { (gl.delete_program)(job.scratch) };
                }
                if end < jobs.len() {
                    Stage::Swapping { jobs, done: end }
                } else {
                    log!("single pass: {} engine programs converted, {} left as they were", self.converted.len(), self.failed);
                    self.stage = Stage::Done;
                    return Some(self.converted.clone());
                }
            }
            Stage::Done => Stage::Done,
            Stage::Failed => Stage::Failed,
        };
        None
    }
}

fn start_reading() -> Stage {
    let Some(base) = std::env::current_exe().ok().and_then(|exe| Some(exe.parent()?.join("base"))) else {
        log!("single pass: off: the game's folder is not known");
        return Stage::Failed;
    };
    let (sender, receiver) = channel();
    std::thread::spawn(move || {
        let _ = sender.send(archive::renderprogs(&base));
    });
    Stage::Reading(receiver)
}

/// The driver compiles on threads of its own from here on (`KHR_parallel_shader_compile`), so a
/// batch of compiles costs the render thread only their submission.
fn parallel_compiles() {
    type MaxThreadsFn = unsafe extern "system" fn(u32);
    let address = gl_api::address("glMaxShaderCompilerThreadsKHR").or_else(|| gl_api::address("glMaxShaderCompilerThreadsARB"));
    match address {
        // SAFETY: the driver's function of that type.
        Some(address) => unsafe { std::mem::transmute::<usize, MaxThreadsFn>(address)(0xFFFF_FFFF) },
        None => log!("single pass: the driver compiles on the render thread (no parallel compiles)"),
    }
}

/// The fragment shader's outputs with their locations (`location` of each name; in order where
/// it knows none).
fn outputs(source: &Renderprog, location: impl Fn(&str) -> Option<u32>) -> Vec<(String, u32)> {
    convert::outputs(&source.fragment).into_iter().enumerate().map(|(i, name)| (name.clone(), location(&name).unwrap_or(i as u32))).collect()
}

fn uniform_names(uniforms: &[ActiveUniform]) -> Vec<String> {
    let mut names: Vec<String> = uniforms.iter().map(|u| u.name.clone()).collect();
    names.sort();
    names
}

/// The sources' assembly matched to the engine's programs, and a converted scratch program
/// started for each engine program that can draw both eyes.
fn match_and_convert(gl: &Gl, sources: &[Renderprog], scratch: &[u32]) -> Stage {
    let mut by_key: HashMap<String, Vec<usize>> = HashMap::new();
    let mut names = Vec::with_capacity(sources.len());
    for (index, &program) in scratch.iter().enumerate() {
        names.push(uniform_names(&gl.uniforms(program)));
        if gl.linked(program).is_ok()
            && let Some(key) = gl.binary(program).and_then(|(_, b)| gl_api::assembly_key(&b))
        {
            by_key.entry(key).or_default().push(index);
        }
        // SAFETY: our own scratch program.
        unsafe { (gl.delete_program)(program) };
    }
    let (mut engine_programs, mut matched, mut flat) = (0, 0, 0);
    let mut jobs = Vec::new();
    for engine in 1..=LAST_NAME {
        // SAFETY: a plain query.
        if unsafe { (gl.is_program)(engine) } == 0 {
            continue;
        }
        engine_programs += 1;
        let Some(key) = gl.binary(engine).and_then(|(_, b)| gl_api::assembly_key(&b)) else { continue };
        let Some(candidates) = by_key.get(&key) else { continue };
        let uniforms = gl.uniforms(engine);
        let own = uniform_names(&uniforms);
        let Some(&source) = candidates.iter().find(|&&c| names[c] == own).or_else(|| (candidates.len() == 1).then(|| &candidates[0])) else { continue };
        matched += 1;
        let program = &sources[source];
        let Some(rows) = convert::position_rows(&program.vertex) else {
            flat += 1;
            continue;
        };
        let Some(vertex) = convert::stereo_vertex(&program.vertex) else { continue };
        let located = |u: &convert::Uniform| uniforms.iter().find(|a| a.name == u.name).and_then(|a| a.locations.first().copied()).filter(|&l| l >= 0);
        let vertex = convert::with_locations(&vertex, &located);
        let fragment = convert::with_locations(&program.fragment, &located);
        // SAFETY: plain queries with NUL-terminated names.
        let outputs = outputs(program, |name| u32::try_from(unsafe { (gl.get_frag_data_location)(engine, gl_api::c_name(name).as_ptr()) }).ok());
        let scratch = gl.program(&vertex, &fragment, &gl.attributes(engine), &outputs);
        jobs.push(Job { engine, source, rows, scratch });
    }
    log!(
        "single pass: {engine_programs} engine programs, {matched} matched to their render program ({flat} full-screen or 2D, left as they are); converting {}",
        jobs.len()
    );
    if jobs.is_empty() {
        return Stage::Failed;
    }
    Stage::Converting { jobs, since: Instant::now() }
}

/// The converted binary moved into the engine's program, its uniform values and block bindings
/// as they were.
fn swap(gl: &Gl, job: &Job, source: &Renderprog) -> Result<Converted, String> {
    gl.linked(job.scratch).map_err(|log| format!("the converted program does not link: {}", log.lines().next().unwrap_or("")))?;
    let (format, binary) = gl.binary(job.scratch).ok_or("no binary of the converted program")?;
    let engine = job.engine;
    let before = gl.uniforms(engine);
    let values: Vec<_> = before.iter().map(|u| gl.values(engine, u)).collect();
    let blocks = gl.block_bindings(engine);
    let original = gl.binary(engine).ok_or("no binary of the engine's program")?;
    let put_back = |why: String| -> String {
        // SAFETY: the program's own binary, as the driver gave it.
        unsafe { (gl.program_binary)(engine, original.0, original.1.as_ptr().cast(), original.1.len() as i32) };
        restore(gl, engine, &before, &values, &blocks);
        why
    };
    // SAFETY: a binary the driver made in this context, of its format.
    unsafe { (gl.program_binary)(engine, format, binary.as_ptr().cast(), binary.len() as i32) };
    gl.linked(engine).map_err(|log| put_back(format!("the binary was refused: {}", log.lines().next().unwrap_or(""))))?;
    let after = gl.uniforms(engine);
    for uniform in &before {
        if !after.iter().any(|a| a.name == uniform.name && a.locations == uniform.locations) {
            return Err(put_back(format!("{} moved", uniform.name)));
        }
    }
    restore(gl, engine, &before, &values, &blocks);
    // SAFETY: plain queries with NUL-terminated names.
    let location = |name: String| unsafe { (gl.get_uniform_location)(engine, gl_api::c_name(&name).as_ptr()) };
    let rows = job.rows.map(|row| location(format!("_va_[{row}]")));
    let stereo = std::array::from_fn(|row| location(format!("{}[{row}]", convert::STEREO_UNIFORM)));
    if rows.iter().chain(&stereo).any(|&l| l < 0) {
        return Err(put_back(format!("{} has no _va_ rows or stereo uniform", source.name)));
    }
    Ok(Converted { program: engine, rows, stereo })
}

fn restore(gl: &Gl, program: u32, uniforms: &[ActiveUniform], values: &[Option<Vec<gl_api::Values>>], blocks: &[u32]) {
    for (uniform, values) in uniforms.iter().zip(values) {
        if let Some(values) = values {
            gl.set_values(program, uniform, values);
        }
    }
    for (block, &binding) in blocks.iter().enumerate() {
        // SAFETY: a plain GL call on our program's block.
        unsafe { (gl.uniform_block_binding)(program, block as u32, binding) };
    }
}
