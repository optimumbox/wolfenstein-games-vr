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
//!
//! The post-process and the depth-of-field passes' size programs are swapped the same way for
//! their fragment shader without depth of field or film grain ([`convert::without_blur_or_grain`]).

use super::archive::{self, Renderprog};
use super::convert;
use super::gl_api::{self, ActiveUniform, Gl};
use monaka_producer::log;
use std::collections::HashMap;
use std::sync::mpsc::{Receiver, channel};
use std::sync::{LazyLock, Mutex};
use std::time::{Duration, Instant};

/// A converted engine program, as the draw hooks use it.
#[derive(Clone, Copy, Debug)]
pub struct Converted {
    pub program: u32,
    /// The locations of the `_va_` rows that give the position (x, y, z, w).
    pub rows: [i32; 4],
    /// The locations of the stereo uniform's rows: the eye matrix's four, then the switch.
    pub stereo: [i32; convert::STEREO_ROWS],
    /// Its fragment shader reads the scene's maps at its own pixel ([`convert::reads_screen`]).
    pub reads_screen: bool,
    /// Its render program's name.
    pub name: &'static str,
    /// Its fragment shader reads the scene's maps layered ([`convert::layered_fragment`]).
    pub screen: Option<&'static Screen>,
    /// Its fragment shader unprojects its pixel with four `_fa_` rows ([`convert::unprojection_row`]):
    /// the first one's location, and the unit its scene depth (`samp_viewdepthmap`) is read from.
    pub unproject: Option<(i32, Option<u32>)>,
}

/// [`Converted::unproject`] for `engine`, linked from `source`.
fn unproject(gl: &Gl, engine: u32, source: &Renderprog) -> Option<(i32, Option<u32>)> {
    let row = convert::unprojection_row(&source.fragment)?;
    // SAFETY: plain queries with NUL-terminated names, into a local.
    unsafe {
        let at = (gl.get_uniform_location)(engine, gl_api::c_name(&format!("_fa_[{row}]")).as_ptr());
        if at < 0 {
            return None;
        }
        let sampler = (gl.get_uniform_location)(engine, gl_api::c_name("samp_viewdepthmap").as_ptr());
        let mut unit = -1;
        if sampler >= 0 {
            (gl.get_uniform_iv)(engine, sampler, &mut unit);
        }
        Some((at, u32::try_from(unit).ok()))
    }
}

/// A converted program whose fragment shader reads the scene's maps at the layer it draws into
/// and its rows for the other eye's copy from [`convert::RIGHT_ROWS`].
#[derive(Debug)]
pub struct Screen {
    /// Each layered map: the engine's sampler's location, and the unit its array sampler reads.
    pub maps: Vec<(i32, u32)>,
    /// The first location and count of the vertex rows (`_va_`) and the fragment rows (`_fa_`),
    /// and the right rows' first location (each array's locations in a row).
    pub vertex_rows: (i32, usize),
    pub fragment_rows: (i32, usize),
    /// None for maps of each eye's own ([`convert::eye_maps_fragment`]): the rows are the same for
    /// both eyes, none learnt or given (the vertex and fragment rows then (0, 0)).
    pub right_rows: Option<i32>,
}

/// The engine's program names swept.
const LAST_NAME: u32 = 16384;
/// Programs moved into the engine's per frame (each a binary load and a uniform copy).
const SWAPS_PER_FRAME: usize = 48;
/// The longest the driver is waited for, per batch of compiles.
const COMPILE_PATIENCE: Duration = Duration::from_secs(60);
/// Whether programs reading the scene's maps are converted to read them at the layer they draw
/// into ([`convert::layered_fragment`]). Off: their draws are in the passes' rests, which each eye
/// draws anyway, and learning their rows asked the driver for uniforms on the render thread.
pub const LAYERED_SCREEN: bool = false;

enum Stage {
    Idle,
    Reading(Receiver<Result<Vec<Renderprog>, String>>),
    Compiling { sources: Vec<Renderprog>, scratch: Vec<u32>, since: Instant },
    Converting { jobs: Vec<Job>, since: Instant },
    Swapping { jobs: Vec<Job>, done: usize },
    Done,
    Failed,
}

/// One engine program to convert: its render program and the converted scratch program, with
/// the screen maps its fragment shader reads layered and what links it with the plain one
/// should the layered one not link.
struct Job {
    engine: u32,
    source: usize,
    rows: [usize; 4],
    scratch: u32,
    maps: Vec<String>,
    plain: Option<Relink>,
    /// Swapped for its fragment shader alone (no stereo rows): [`convert::without_blur_or_grain`].
    patched: bool,
}

struct Relink {
    vertex: String,
    fragment: String,
    attributes: Vec<(String, u32)>,
    outputs: Vec<(String, u32)>,
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
                    let stage = match_and_convert(gl, &self.sources, &scratch, &mut self.converted);
                    if matches!(stage, Stage::Failed) && !self.converted.is_empty() {
                        self.stage = Stage::Done;
                        return Some(self.converted.clone());
                    }
                    stage
                }
            }
            Stage::Converting { mut jobs, since } => {
                if !jobs.iter().all(|j| gl.program_done(j.scratch)) && since.elapsed() < COMPILE_PATIENCE {
                    Stage::Converting { jobs, since }
                } else if relink_plain(gl, &mut jobs, &self.sources) {
                    Stage::Converting { jobs, since: Instant::now() }
                } else {
                    Stage::Swapping { jobs, done: 0 }
                }
            }
            Stage::Swapping { jobs, done } => {
                let end = (done + SWAPS_PER_FRAME).min(jobs.len());
                for job in &jobs[done..end] {
                    match swap(gl, job, &self.sources[job.source]) {
                        Ok(Some(converted)) => self.converted.push(converted),
                        Ok(None) => log!("single pass: program {} ({}) drawn without depth of field or film grain", job.engine, self.sources[job.source].name),
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

/// The jobs whose layered fragment shader did not link linked again with the plain one; whether
/// any was.
fn relink_plain(gl: &Gl, jobs: &mut [Job], sources: &[Renderprog]) -> bool {
    let mut any = false;
    for job in jobs.iter_mut() {
        let Some(plain) = job.plain.take() else { continue };
        let Err(why) = gl.linked(job.scratch) else { continue };
        if !any {
            log!("single pass: {} does not link reading the scene's maps layered ({}): each eye draws it", sources[job.source].name, why.lines().next().unwrap_or(""));
        }
        any = true;
        // SAFETY: our own scratch program.
        unsafe { (gl.delete_program)(job.scratch) };
        job.scratch = gl.program(&plain.vertex, &plain.fragment, &plain.attributes, &plain.outputs);
        job.maps.clear();
    }
    any
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

/// Each matched engine program's render program name, converted or not (for the logs).
static NAMES: LazyLock<Mutex<HashMap<u32, &'static str>>> = LazyLock::new(|| Mutex::new(HashMap::new()));

fn remember(engine: u32, name: &str) {
    let Ok(mut names) = NAMES.lock() else { return };
    if names.get(&engine).is_none_or(|known| *known != name) {
        names.insert(engine, Box::leak(name.to_owned().into_boxed_str()));
    }
}

/// The render program name of the engine program `engine`, if it was matched to one.
pub fn engine_name(engine: u32) -> Option<&'static str> {
    NAMES.lock().ok()?.get(&engine).copied()
}

/// The sources' assembly matched to the engine's programs, and a converted scratch program
/// started for each engine program that can draw both eyes.
fn match_and_convert(gl: &Gl, sources: &[Renderprog], scratch: &[u32], already: &mut Vec<Converted>) -> Stage {
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
    let (mut engine_programs, mut matched, mut flat, mut no_binary) = (0, 0, 0, 0);
    let mut jobs = Vec::new();
    for engine in 1..=LAST_NAME {
        // SAFETY: a plain query.
        if unsafe { (gl.is_program)(engine) } == 0 {
            continue;
        }
        engine_programs += 1;
        // Converted by an earlier run in this process: known by its stereo uniform, its source by
        // its other uniforms' names.
        if let Some((converted, source)) = converted_before(gl, engine, sources, &names) {
            // Converted before its fragment shader read the scene's maps layered: converted again.
            let layered = gl.uniforms(engine).iter().any(|u| u.name.starts_with(convert::LAYERED_PREFIX));
            let fragment = &sources[source].fragment;
            let wanted = layered_maps(fragment, fragment).is_some();
            if layered == wanted {
                already.push(converted);
            } else if !has_binary(gl, engine) {
                // Its binary cannot be had to put back: it would stay as it is after a compile.
                no_binary += 1;
            } else if let Some(job) = job(gl, engine, &gl.uniforms(engine), source, &sources[source]) {
                jobs.push(job);
            }
            continue;
        }
        let Some(key) = gl.binary(engine).and_then(|(_, b)| gl_api::assembly_key(&b)) else { continue };
        let Some(candidates) = by_key.get(&key) else { continue };
        let uniforms = gl.uniforms(engine);
        let own = uniform_names(&uniforms);
        let Some(&source) = candidates.iter().find(|&&c| names[c] == own).or_else(|| (candidates.len() == 1).then(|| &candidates[0])) else { continue };
        matched += 1;
        let program = &sources[source];
        remember(engine, &program.name);
        if let Some(fragment) = convert::without_blur_or_grain(&program.name, &program.fragment) {
            jobs.push(patch(gl, engine, &uniforms, source, program, &fragment));
            continue;
        }
        if convert::position_rows(&program.vertex).is_none() {
            flat += 1;
            continue;
        }
        if let Some(job) = job(gl, engine, &uniforms, source, program) {
            jobs.push(job);
        }
    }
    log!(
        "single pass: {engine_programs} engine programs, {} converted before ({no_binary} more left as they are, no binary of theirs to be had), {matched} matched to their render program ({flat} full-screen or 2D, left as they are); converting {}",
        already.len(),
        jobs.len()
    );
    if jobs.is_empty() {
        return Stage::Failed;
    }
    Stage::Converting { jobs, since: Instant::now() }
}

/// Whether the driver gives `program`'s binary (the swap puts it back if the converted one fails).
fn has_binary(gl: &Gl, program: u32) -> bool {
    let mut length = 0;
    // SAFETY: a plain query into a local.
    unsafe { (gl.get_program_iv)(program, gl_api::GL_PROGRAM_BINARY_LENGTH, &mut length) };
    length > 0
}

/// The converted scratch program for `engine` from its render program `program` (`sources[source]`),
/// with the engine program's own uniform, attribute and output locations; its fragment shader
/// reads the scene's maps layered where it can.
fn job(gl: &Gl, engine: u32, uniforms: &[ActiveUniform], source: usize, program: &Renderprog) -> Option<Job> {
    let rows = convert::position_rows(&program.vertex)?;
    let vertex = convert::stereo_vertex(&program.vertex)?;
    // Shine and reflections lit for the eye each copy is ([`convert::eye_rays`]).
    let lit = convert::eye_rays(&program.vertex, &vertex, &program.fragment);
    let (vertex, lit_fragment) = match lit {
        Some((vertex, fragment)) => (vertex, Some(fragment)),
        None => (vertex, None),
    };
    let fragment_source = lit_fragment.as_deref().unwrap_or(&program.fragment);
    let located = |u: &convert::Uniform| uniforms.iter().find(|a| a.name == u.name).and_then(|a| a.locations.first().copied()).filter(|&l| l >= 0);
    let vertex = convert::with_locations(&vertex, &located);
    let plain = convert::with_locations(fragment_source, &located);
    let layered = layered_maps(&program.fragment, fragment_source);
    // SAFETY: plain queries with NUL-terminated names.
    let outputs = outputs(program, |name| u32::try_from(unsafe { (gl.get_frag_data_location)(engine, gl_api::c_name(name).as_ptr()) }).ok());
    let attributes = gl.attributes(engine);
    Some(match layered {
        Some(layered) => {
            let fragment = convert::with_locations(&layered.source, &located);
            let scratch = gl.program(&vertex, &fragment, &attributes, &outputs);
            Job { engine, source, rows, scratch, maps: layered.maps, plain: Some(Relink { vertex, fragment: plain, attributes, outputs }), patched: false }
        }
        None => Job { engine, source, rows, scratch: gl.program(&vertex, &plain, &attributes, &outputs), maps: Vec::new(), plain: None, patched: false },
    })
}

/// The scratch program for `engine` with its render program's vertex shader and `fragment` in
/// place of its own, with the engine program's own locations.
fn patch(gl: &Gl, engine: u32, uniforms: &[ActiveUniform], source: usize, program: &Renderprog, fragment: &str) -> Job {
    let located = |u: &convert::Uniform| uniforms.iter().find(|a| a.name == u.name).and_then(|a| a.locations.first().copied()).filter(|&l| l >= 0);
    let vertex = convert::with_locations(&program.vertex, &located);
    let fragment = convert::with_locations(fragment, &located);
    // SAFETY: plain queries with NUL-terminated names.
    let outputs = outputs(program, |name| u32::try_from(unsafe { (gl.get_frag_data_location)(engine, gl_api::c_name(name).as_ptr()) }).ok());
    let attributes = gl.attributes(engine);
    let scratch = gl.program(&vertex, &fragment, &attributes, &outputs);
    Job { engine, source, rows: [0; 4], scratch, maps: Vec::new(), plain: None, patched: true }
}

/// An engine program an earlier run converted: its locations, from the render program with the
/// same uniforms (its own less ours), and that program's index.
fn converted_before(gl: &Gl, engine: u32, sources: &[Renderprog], names: &[Vec<String>]) -> Option<(Converted, usize)> {
    let stereo = locations(gl, engine, &[convert::STEREO_UNIFORM], convert::STEREO_ROWS)?;
    let own: Vec<String> = uniform_names(&gl.uniforms(engine)).into_iter().filter(|n| !n.starts_with("monaka_")).collect();
    let (rows, index) = (0..sources.len()).filter(|&i| names[i] == own).find_map(|i| Some((convert::position_rows(&sources[i].vertex)?, i)))?;
    let source = &sources[index];
    let rows = locations(gl, engine, &rows.map(|r| format!("_va_[{r}]")).iter().map(String::as_str).collect::<Vec<_>>(), 0)?;
    let maps = layered_maps(&source.fragment, &source.fragment).map(|l| l.maps).unwrap_or_default();
    let converted = Converted {
        program: engine,
        rows: rows.try_into().ok()?,
        stereo: stereo.try_into().ok()?,
        reads_screen: convert::reads_screen(&source.fragment),
        name: leak(&source.name),
        screen: screen(gl, engine, &maps),
        unproject: unproject(gl, engine, source),
    };
    Some((converted, index))
}

/// A fragment shader reading maps at the layer it draws into: the scene's maps with the other
/// eye's rows ([`convert::layered_fragment`], with [`LAYERED_SCREEN`]); otherwise the maps each
/// eye draws its own of ([`convert::eye_maps_fragment`]), its rows the same for both eyes. Made
/// from `source` (the render program's fragment shader, or a version of it lit per eye).
fn layered_maps(engine: &str, source: &str) -> Option<convert::LayeredFragment> {
    if convert::reads_screen(engine) {
        LAYERED_SCREEN.then(|| convert::layered_fragment(source)).flatten()
    } else {
        convert::eye_maps_fragment(source)
    }
}

/// The layered maps' array samplers given units of their own (the last ones, which the engine's
/// samplers do not use), and the rows' locations; none when the program does not read `maps`
/// layered.
fn screen(gl: &Gl, program: u32, maps: &[String]) -> Option<&'static Screen> {
    if maps.is_empty() {
        return None;
    }
    let uniforms = gl.uniforms(program);
    let in_a_row = |name: &str| {
        let uniform = uniforms.iter().find(|u| u.name == name)?;
        let first = *uniform.locations.first()?;
        uniform.locations.iter().enumerate().all(|(i, &l)| l == first + i as i32).then_some((first, uniform.locations.len()))
    };
    // Without the other eye's rows (maps of each eye's own), no rows are learnt or given.
    let right = in_a_row(convert::RIGHT_ROWS);
    let (vertex_rows, fragment_rows) = if right.is_some() { (in_a_row("_va_")?, in_a_row(convert::FRAGMENT_ROWS)?) } else { ((0, 0), (0, 0)) };
    if right.is_some_and(|(_, count)| count != fragment_rows.1) {
        return None;
    }
    let right_rows = right.map(|(first, _)| first);
    let mut most = 0;
    // SAFETY: a state query into a local (GL_MAX_TEXTURE_IMAGE_UNITS).
    unsafe { windows::Win32::Graphics::OpenGL::glGetIntegerv(0x8872, &mut most) };
    let engines: Vec<i32> = uniforms
        .iter()
        .filter(|u| !u.name.starts_with("monaka_") && gl_api::components(u.kind) == Some((1, true)))
        .filter_map(|u| gl.values(program, u))
        .flatten()
        .filter_map(|v| match v {
            gl_api::Values::Ints(v) => v.first().copied(),
            gl_api::Values::Floats(_) => None,
        })
        .collect();
    let mut out = Vec::new();
    for (i, map) in maps.iter().enumerate() {
        let unit = most - 1 - i as i32;
        if unit < 0 || engines.contains(&unit) {
            return None;
        }
        let ours = format!("{}{map}", convert::LAYERED_PREFIX);
        let (Some(engine), Some(layered)) = (uniforms.iter().find(|u| &u.name == map), uniforms.iter().find(|u| u.name == ours)) else { continue };
        let (Some(&engine), Some(&layered)) = (engine.locations.first(), layered.locations.first()) else { continue };
        // SAFETY: a plain GL call on the engine's program, with one int.
        unsafe { (gl.program_uniform_1iv)(program, layered, 1, &unit) };
        out.push((engine, unit as u32));
    }
    (!out.is_empty()).then(|| &*Box::leak(Box::new(Screen { maps: out, vertex_rows, fragment_rows, right_rows })))
}

/// A render program's name kept for the run (a few hundred, made once).
fn leak(name: &str) -> &'static str {
    Box::leak(name.to_owned().into_boxed_str())
}

/// The locations of `names` in `program` (with `rows` > 0: the rows of the one array named), none
/// when one is missing.
fn locations(gl: &Gl, program: u32, names: &[&str], rows: usize) -> Option<Vec<i32>> {
    let names: Vec<String> = if rows > 0 { (0..rows).map(|r| format!("{}[{r}]", names[0])).collect() } else { names.iter().map(|n| (*n).to_owned()).collect() };
    names
        .iter()
        // SAFETY: plain queries with NUL-terminated names.
        .map(|name| Some(unsafe { (gl.get_uniform_location)(program, gl_api::c_name(name).as_ptr()) }).filter(|&l| l >= 0))
        .collect()
}

/// The converted binary moved into the engine's program, its uniform values and block bindings
/// as they were; none for a program patched alone (it draws no eye but its own).
fn swap(gl: &Gl, job: &Job, source: &Renderprog) -> Result<Option<Converted>, String> {
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
    if job.patched {
        return Ok(None);
    }
    // SAFETY: plain queries with NUL-terminated names.
    let location = |name: String| unsafe { (gl.get_uniform_location)(engine, gl_api::c_name(&name).as_ptr()) };
    let rows = job.rows.map(|row| location(format!("_va_[{row}]")));
    let stereo = std::array::from_fn(|row| location(format!("{}[{row}]", convert::STEREO_UNIFORM)));
    if rows.iter().chain(&stereo).any(|&l| l < 0) {
        return Err(put_back(format!("{} has no _va_ rows or stereo uniform", source.name)));
    }
    Ok(Some(Converted { program: engine, rows, stereo, reads_screen: convert::reads_screen(&source.fragment), name: leak(&source.name), screen: screen(gl, engine, &job.maps), unproject: unproject(gl, engine, source) }))
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
