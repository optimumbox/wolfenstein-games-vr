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
//! The engine's GPU culling tests (bounding boxes drawn with colour and depth writes off, their
//! results read back for the next frame) are draws of the stretch too. They are not skipped in
//! the right eye's pass: its culling is its own, and skipped it would cull nearly all of the next
//! frame's right eye ([`Writes`]).
//!
//! Each pass culls for its own eye: a thing only the right eye sees (at its outer edge, past
//! something that hides it from the left eye) is drawn by the right eye's pass alone. So the left
//! eye's stretch draws are kept by program and geometry (the draw call's own arguments, [`Carried`])
//! with their rows carried by the matrix, and a right eye's stretch draw not among them is drawn
//! into its layer rather than skipped.
//!
//! Both copies of a carried draw share the left eye's viewport, and so its scissor: a rectangle
//! the engine fits to what the left eye sees would clip the right eye's copy. A carried draw made
//! with a scissor smaller than its viewport is drawn with the scissor test off (the depth test
//! hides what the rectangle would have), and the test is put back right after it.
//!
//! A draw of the stretch with another depth projection than the world's (the hands and their
//! weapon, their depth squeezed in front of the world) is not carried by the eye matrix, which
//! fits the world's: each pass draws it into its own layer ([`depth_ratio`]).
//!
//! The detours ask the driver nothing per call: a state query waits for the driver's own thread.
//! The active texture unit and what draws write are kept from the engine's own calls ([`Shadow`]),
//! checked against queries now and then.
//!
//! Some programs place their vertices by more than the position rows (glows and sprites turned to
//! the camera, among them): the eye matrix puts their copy where the right eye's own pass would
//! not. Every [`CHECK_EVERY`] frames each of the right eye's stretch draws is looked for among the
//! left eye's draws of its program, carried by the matrix; a program whose draws mostly are not
//! found is drawn by each eye from then on, like the hands ([`Misfits`]). So is every program whose
//! fragment shader reads the scene's depth, normal or colour maps at its own pixel (decals, lights,
//! soft particles: [`convert::reads_screen`](super::convert::reads_screen)): the copy would read the
//! left eye's maps and place itself by the left eye's pixel.
//!
//! The scene's stretch is mostly the engine's occlusion queries: each object's depth drawn, writing
//! nothing, inside a samples-passed query. The left eye's, drawn for both eyes, counts the samples of
//! either; so each of the right eye's queries whose draw is the copy of one of the left's (found by
//! the eye matrix) is not made, and its result is read from that left query ([`QUERIES_SHARED`]).
//!
//! The stand-ins are made once the scene's framebuffers have been seen for a while
//! ([`DISCOVERY_FRAMES`]); at each `SwapBuffers` the frame's last layer is copied back into the
//! engine's textures, for what reads them at the present (the frame's depth).
//!
//! The engine's extension functions are called through its own function table, whose slots are
//! found in the executable's data by the driver's and OpenGL32's addresses ([`install`]).

use super::eye_matrix::{self, FitStep, Rows};
use super::gl_api::{self, Gl};
use super::layers::Layers;
use super::programs::{Conversion, Converted, Screen};
use super::screen::{self, Drawn, Learner, Row};
use monaka_core::sync::Locked;
use monaka_hook::module::Module;
use monaka_hook::{Hooks, InFlight, Original};
use monaka_producer::log;
use std::collections::{HashMap, HashSet};
use std::ffi::c_void;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering::*};
use std::time::{Duration, Instant};
use windows::Win32::Graphics::OpenGL::{glGetBooleanv, glGetIntegerv, glIsEnabled};

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
    /// For a program reading the scene's maps layered: its vertex and fragment rows as uploaded
    /// (read from the driver at its first draw), the right rows last given, and the units its
    /// maps' samplers read (looked at again on the checking frames).
    vertex: Vec<Row>,
    fragment: Vec<Row>,
    right_given: Vec<Row>,
    map_units: Vec<u32>,
}

impl Program {
    fn new(converted: Converted) -> Self {
        Self { converted, rows: [[0.0; 4]; 4], switch: false, matrix: 0, vertex: Vec::new(), fragment: Vec::new(), right_given: Vec::new(), map_units: Vec::new() }
    }
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
    /// The right eye's stretch draws writing neither colour nor depth (its culling's), drawn.
    unwritten: u64,
    /// Of those: inside a query, with the stencil test on, with the rasterizer discarding.
    unwritten_query: u64,
    unwritten_stencil: u64,
    unwritten_discard: u64,
    unwritten_depth_test: u64,
    /// The right eye's queries answered by the left eye's.
    queries_shared: u64,
    /// The right eye's queries only another thing's left query had the rows of: made by the right
    /// eye's pass.
    queries_rows_only: u64,
    /// Checked ones (on the shadow's check frames) writing colour to a draw buffer past the first.
    unwritten_checked: u64,
    unwritten_other_buffers: u64,
    /// The stretch's draws with another depth projection than the world's, drawn by each pass.
    own_projection: u64,
    /// The stretch's draws of programs the eye matrix does not carry, or that read the scene's maps
    /// at their own pixel, drawn by each pass.
    misfit_draws: u64,
    /// Draws of programs reading the scene's maps made for both eyes.
    screen_shared: u64,
    /// The left eye's carried draws made with the scissor test off, their scissor smaller than the
    /// viewport.
    scissor_opened: u64,
    /// The right eye's stretch draws the left eye's pass did not make (culled for the left eye),
    /// drawn by the right eye's pass.
    right_only: u64,
    /// Frames whose matrix came from pairing the passes' draws by the last matrix.
    rematched: u64,
    /// The right eye's stretch draws found among the left eye's by shape alone ([`Carried`]).
    by_shape: u64,
    /// The right eye's stretch draws of a mesh the left eye's drew a little elsewhere (moved
    /// between the passes), skipped.
    moved: u64,
    /// The right eye's draws drawn though the left eye's drew the same mesh (farther off).
    right_only_same_geometry: u64,
    /// Frames whose measured matrix was far from the last, and those that kept the last one.
    matrix_jumps: u64,
    matrix_held: u64,
    /// Pairs added to the averaged eye matrix.
    fitted: u64,
}

/// The left eye's stretch draws of a frame made for both eyes, by program and geometry (a key of
/// the draw call's arguments: the same mesh drawn by both passes makes the same call), with their
/// rows carried into the right eye.
///
/// A mesh streamed each pass (skinned characters' decals, among them) lands at another offset of
/// the engine's buffers in each pass: its call differs only there. So each is kept by its shape
/// too (the call and its counts, not its offsets), and a right eye's draw not found by geometry is
/// looked for by shape, each of the left eye's draws standing for one such.
#[derive(Default)]
struct Carried {
    by: HashMap<(u32, u64), Vec<Rows>>,
    by_shape: HashMap<(u32, u64), Vec<Rows>>,
}

/// How a right eye's draw was found among the left eye's carried draws.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Found {
    Geometry,
    Shape,
}

impl Carried {
    fn keep(&mut self, program: u32, geometry: u64, shape: u64, carried: Rows) {
        self.by.entry((program, geometry)).or_default().push(carried);
        self.by_shape.entry((program, shape)).or_default().push(carried);
    }

    /// Whether the right eye's draw of `program` with `geometry`, `shape` and `rows` is one of them.
    fn find(&mut self, program: u32, geometry: u64, shape: u64, rows: &Rows) -> Option<Found> {
        let agrees = |c: &Rows| eye_matrix::apart(c, rows) <= eye_matrix::AGREES * 10.0;
        if self.by.get(&(program, geometry)).is_some_and(|carried| carried.iter().any(agrees)) {
            return Some(Found::Geometry);
        }
        let shaped = self.by_shape.get_mut(&(program, shape))?;
        let at = shaped.iter().position(agrees)?;
        shaped.swap_remove(at);
        Some(Found::Shape)
    }

    /// How far the right eye's draw with `rows` is from the nearest of the left eye's carried
    /// draws of the same program and geometry (none: no such draw).
    fn nearest(&self, program: u32, geometry: u64, rows: &Rows) -> Option<f64> {
        self.by.get(&(program, geometry))?.iter().map(|c| eye_matrix::apart(c, rows)).min_by(f64::total_cmp)
    }

    fn clear(&mut self) {
        self.by.clear();
        self.by_shape.clear();
    }
}

/// How far (relative) the right eye's draw of a mesh may be from the left eye's carried draw of it
/// and still be the same thing, moved between the passes (swaying, turned to the camera, animated
/// on): a repeat of the mesh elsewhere is metres away, far beyond it.
const MOVED_BETWEEN: f64 = 0.02;


/// The frames between checks of the right eye's draws against the left's carried by the matrix.
const CHECK_EVERY: u64 = 15;
/// Checked draws needed before a program is judged, and the share not found that makes it a misfit:
/// most of them, as a draw the left eye culled is not found either (the world's depth pass, half
/// not found, is carried well; the glows' programs miss all but a few).
const JUDGED_AFTER: u64 = 12;
const MISFIT_SHARE: f64 = 0.8;
/// The right eye's draws of a program seen before judging whether its geometry is its own.
/// Judged on many draws, nearly all missed: a program misjudged so draws the world for each eye
/// (the world's own program was, from its first 24 draws while a level came in), and the right
/// eye's own draws of it fail the depth the left eye's copies laid (black floors), at twice the cost.
const CARRIED_JUDGED_AFTER: u64 = 600;
const CARRIED_MISFIT_SHARE: f64 = 0.97;
/// Frames with the program's draws needed too: the world's program makes 600 draws in a dozen
/// frames, as few as a level's first frames (the eye matrix still settling) miss.
const CARRIED_JUDGED_FRAMES: u64 = 180;
/// A frame counts towards the judgments only when the eye matrix carried this share of its right
/// eye's draws: in a frame it does not, every program misses alike.
const FRAME_FOUND_SHARE: f64 = 0.33;

/// The programs the eye matrix does not carry: each one's checked draws and those not found, and
/// the programs judged misfits (drawn by each pass).
#[derive(Default)]
struct Misfits {
    checked: HashMap<u32, (u64, u64)>,
    misfits: HashSet<u32>,
    /// A checked frame's left-eye stretch rows by program.
    left: HashMap<u32, Vec<Rows>>,
    /// The right eye's stretch draws by program, and those not among the left eye's ([`Carried`]),
    /// with the frames they came from; counted from frames the matrix carried ([`Self::end_frame`]).
    carried: HashMap<u32, (u64, u64, u64)>,
    /// The frame's own count, by program, before it is known whether the frame counts.
    frame: HashMap<u32, (u64, u64)>,
    frame_names: HashMap<u32, &'static str>,
}

impl Misfits {
    /// The right eye's draw of `program`, `found` among the left eye's draws or not. A program whose
    /// draws mostly are not found draws other geometry in each pass (sprites and particles built
    /// for each eye's view), not what one eye culled: drawn by each eye from then on, or the right
    /// eye would show both its own and the left eye's copy.
    ///
    /// A draw found a little elsewhere (moved between the passes: animated, swaying) counts as found.
    /// The count waits for the frame's end: a frame whose draws the matrix mostly did not carry
    /// (a level coming in, the matrix settling) is left out, or it would judge the world's own
    /// programs (black floors, doubled characters).
    fn carried(&mut self, program: u32, name: &'static str, found: bool) {
        let (seen, missed) = self.frame.entry(program).or_default();
        *seen += 1;
        *missed += u64::from(!found);
        self.frame_names.entry(program).or_insert(name);
    }

    /// The frame's counts kept when the matrix carried enough of its draws, and the programs judged.
    fn end_frame(&mut self) {
        let (seen, missed) = self.frame.values().fold((0, 0), |(s, m), (fs, fm)| (s + fs, m + fm));
        let counts = seen > 0 && (seen - missed) as f64 >= seen as f64 * FRAME_FOUND_SHARE;
        for (program, (frame_seen, frame_missed)) in self.frame.drain() {
            if !counts {
                continue;
            }
            let (seen, missed, frames) = self.carried.entry(program).or_default();
            *seen += frame_seen;
            *missed += frame_missed;
            *frames += 1;
            if *seen >= CARRIED_JUDGED_AFTER && *frames >= CARRIED_JUDGED_FRAMES && *missed as f64 > *seen as f64 * CARRIED_MISFIT_SHARE && self.misfits.insert(program) {
                let name = self.frame_names.get(&program).copied().unwrap_or("?");
                log!("single pass: program {program} ({name}) draws other geometry in each eye's pass ({missed} of {seen} of the right eye's draws over {frames} frames not the left's): each eye draws it");
            }
        }
    }

    /// The right eye's draw of `program` with `rows`, against the left eye's carried by `matrix`.
    fn check(&mut self, program: u32, name: &str, rows: &Rows, matrix: &Rows) {
        let found = self.left.get(&program).is_some_and(|left| left.iter().any(|l| eye_matrix::misfit(matrix, l, rows) <= eye_matrix::AGREES * 10.0));
        let (checked, missed) = self.checked.entry(program).or_default();
        *checked += 1;
        *missed += u64::from(!found);
        if *checked >= JUDGED_AFTER && *missed as f64 > *checked as f64 * MISFIT_SHARE && self.misfits.insert(program) {
            log!("single pass: program {program} ({name}) is not carried by the eye matrix ({missed} of {checked} draws not found): each eye draws it");
        }
    }
}

/// The render thread's time per frame by part, from the swap: before the left eye's pass, its
/// stretch, the rest of it, the right eye's stretch, the rest of it (to the next swap); and the
/// draws in each, with the time in the draw detour's own work.
#[derive(Default)]
struct Profile {
    part: usize,
    since: Option<Instant>,
    times: [Duration; 5],
    draws: [u64; 5],
    ours: Duration,
    frames: u64,
    /// The frame's last draw, and the time from it to the swap.
    last_draw: Option<Instant>,
    after_last: Duration,
    /// The rests' time by the framebuffer drawn into ("L28", "R0"): from the draw before (or the
    /// stretch's end) to each draw, and the draws.
    by_target: HashMap<String, (Duration, u64)>,
}

impl Profile {
    fn enter(&mut self, part: usize) {
        let now = Instant::now();
        if let Some(since) = self.since {
            self.times[self.part] += now - since;
        }
        self.since = Some(now);
        self.part = part;
    }

    fn report(&mut self) -> String {
        let frames = self.frames.max(1) as f64;
        let ms = |d: Duration| d.as_secs_f64() * 1000.0 / frames;
        let parts = ["before the left pass", "left stretch", "left rest", "right stretch", "right rest"];
        let text = parts
            .iter()
            .enumerate()
            .map(|(i, name)| format!("{name} {:.2} ms ({:.0} draws)", ms(self.times[i]), self.draws[i] as f64 / frames))
            .collect::<Vec<_>>()
            .join(", ");
        let waits = WAITED
            .iter()
            .zip(WAITS)
            .filter_map(|((time, calls), name)| {
                let (time, calls) = (time.swap(0, Relaxed), calls.swap(0, Relaxed));
                (calls > 0).then(|| format!("{name} {:.2} ms ({:.0})", time as f64 / 1e6 / frames, calls as f64 / frames))
            })
            .collect::<Vec<_>>()
            .join(", ");
        let mut targets: Vec<_> = self.by_target.drain().collect();
        targets.sort_by(|a, b| b.1.0.cmp(&a.1.0));
        let targets = targets.iter().take(14).map(|(name, (time, draws))| format!("{name} {:.2} ms ({:.0})", ms(*time), *draws as f64 / frames)).collect::<Vec<_>>().join(", ");
        let text = format!("{text}; the rests by framebuffer: {targets}; the draw detour's own work {:.2} ms; from the last draw to the swap {:.2} ms; in the engine's waits and reads: {waits}", ms(self.ours), ms(self.after_last));
        *self = Self { part: self.part, since: self.since, ..Self::default() };
        text
    }
}

/// The live frames whose passes are written to the log: each run of draws into one framebuffer
/// and viewport as a line, with the clears between them.
const TRACED_FRAMES: std::ops::Range<u64> = 600..602;
/// Whether the render thread is profiled (by part, by framebuffer, the engine's waits timed) and
/// frames [`TRACED_FRAMES`] traced: research only, it costs the render thread time per draw.
const PROFILING: bool = false;
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
    /// The engine's scissor rectangle (none until it sets one).
    scissor: Option<[i32; 4]>,
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
    /// The world's depth projection (the stretch's usual [`depth_ratio`], measured last frame),
    /// and this frame's left-eye stretch ratios.
    world_depth: Option<f32>,
    depth_ratios: Vec<f32>,
    left: Vec<(u32, Rows)>,
    right: Vec<(u32, Rows)>,
    counts: Counts,
    /// The right eye's draws writing neither colour nor depth, by program.
    unwritten_by: HashMap<u32, u64>,
    /// The right eye's stretch draws each eye draws itself (its own projection, or a misfit), by
    /// program.
    own_by: HashMap<u32, u64>,
    /// The left eye's queries made around stretch draws for both eyes this frame (id, rows, the
    /// draw's geometry, answered for the right eye yet), and where the right eye's pass looks next.
    left_queries: Vec<LeftQuery>,
    query_cursor: usize,
    /// This frame's left-eye stretch draws made for both eyes ([`Carried`]).
    carried: Carried,
    /// The right eye's stretch draws the left eye's did not make, by program.
    right_only_by: HashMap<u32, u64>,
    /// The right eye's stretch draws found by shape alone, by program.
    by_shape_by: HashMap<u32, u64>,
    /// The last matrix measured, to pair the passes' draws by when their order does not.
    last_matrix: Option<Rows>,
    /// The eye matrix averaged over many frames ([`eye_matrix::Fit`]).
    fit: eye_matrix::Fit,
    /// The farthest a frame's own measured matrix has been from the averaged one.
    largest_jump: f64,
    /// The right eye's draws skipped as moved between the passes, by program, and the farthest.
    moved_by: HashMap<u32, (u64, f64)>,
    misfits: Misfits,
    /// The right eye's rows of the programs reading the scene's maps, learnt.
    screen: Learner,
    /// The arrays bound to the units of the layered maps' samplers.
    map_arrays: HashMap<u32, u32>,
    /// The rows kept from the engine's uploads disagreed with the driver's: read from it always.
    rows_untrusted: bool,
    profile: Profile,
    agreeing: (usize, usize),
    logged: Option<Instant>,
    live_frames: u64,
    trace: Option<Trace>,
    /// Every draw of a few frames, asked for by a capture of what the headset shows ([`Inspect`]).
    inspect: Option<Inspect>,
    /// How the last draw went, for [`Inspect`].
    last_path: &'static str,
    /// The run's captures folder and the next capture's number looked for in it.
    captures: Option<(std::path::PathBuf, u32)>,
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
        scissor: None,
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
        world_depth: None,
        depth_ratios: Vec::new(),
        left: Vec::new(),
        right: Vec::new(),
        counts: Counts::default(),
        unwritten_by: HashMap::new(),
        own_by: HashMap::new(),
        left_queries: Vec::new(),
        query_cursor: 0,
        carried: Carried::default(),
        right_only_by: HashMap::new(),
        by_shape_by: HashMap::new(),
        last_matrix: None,
        fit: eye_matrix::Fit::default(),
        largest_jump: 0.0,
        moved_by: HashMap::new(),
        misfits: Misfits::default(),
        screen: Learner::default(),
        map_arrays: HashMap::new(),
        rows_untrusted: false,
        profile: Profile::default(),
        agreeing: (0, 0),
        logged: None,
        live_frames: 0,
        trace: None,
        inspect: None,
        last_path: "",
        captures: None,
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
            let reading = converted.iter().filter(|c| c.reads_screen).count();
            log!("single pass: {reading} of {} converted programs read the scene's maps at their own pixel: each eye draws them", converted.len());
            let layered = converted.iter().filter(|c| c.screen.is_some()).count();
            log!("single pass: {layered} of them read the maps at the layer they draw into, for both eyes in one draw once their right eye's rows are learnt");
            state.programs = converted.into_iter().map(|c| (c.program, Program::new(c))).collect();
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
    inspect_frame(state);
    if PROFILING && TRACED_FRAMES.contains(&state.live_frames) {
        state.trace = Some(Trace { lines: Vec::new(), run: None, draws: 0, scene: 0, binds: Vec::new() });
    }
    let passes = state.pass;
    if let (Some(layers), Some(gl)) = (state.layers.as_ref(), state.gl.as_ref()) {
        layers.copy_back(gl, state.layer());
    }
    let size = monaka_stereo::same_frame::gl::frame_size().unwrap_or((0, 0));
    state.size = (size.0 as i32, size.1 as i32);
    let pairs: Vec<(Rows, Rows)> = state.left.iter().zip(&state.right).filter(|(l, r)| l.0 == r.0).map(|(l, r)| (l.1, r.1)).collect();
    let mut measured = if passes == 2 { eye_matrix::measure(&pairs) } else { None };
    let mut paired = pairs;
    if passes == 2 && measured.is_none() && let Some(last) = state.fit.matrix().or(state.last_matrix) {
        paired = eye_matrix::pairs_with(&last, &state.left, &state.right);
        measured = eye_matrix::measure(&paired);
        state.counts.rematched += u64::from(measured.is_some());
    }
    if passes == 2 {
        match state.fit.add(&paired, measured.as_ref().map(|m| &m.matrix)) {
            FitStep::Added(n) => state.counts.fitted += n as u64,
            FitStep::Held => state.counts.matrix_held += 1,
            FitStep::Reset => state.counts.matrix_jumps += 1,
            FitStep::Nothing => {}
        }
    }
    let fitted = state.fit.matrix().filter(|_| passes == 2);
    if let (Some(m), Some(fit)) = (measured.as_ref(), fitted.as_ref()) {
        state.largest_jump = state.largest_jump.max(eye_matrix::apart(&m.matrix, fit));
    }
    if fitted.is_some() {
        state.last_matrix = fitted;
    } else if let Some(m) = &measured {
        state.last_matrix = Some(m.matrix);
    }
    if state.matrix.is_some() {
        state.counts.frames_on += 1;
    } else {
        state.counts.frames_off += 1;
    }
    state.counts.odd_frames += u64::from(passes != 2);
    state.screen.end_frame(state.matrix.as_ref());
    // The average whenever there is one: the matrix does not change, so a frame whose own pairs
    // gave none still has both eyes in one pass (each eye on its own for a frame is a blink).
    state.matrix = fitted.or(measured.map(|m| m.matrix));
    state.matrix_number += 1;
    state.agreeing = measured.map_or((0, paired.len()), |m| (m.agreeing, m.looked_at));
    state.left.clear();
    state.misfits.left.clear();
    state.misfits.end_frame();
    state.left_queries.clear();
    state.query_cursor = 0;
    state.carried.clear();
    SHARING.store(false, Relaxed);
    state.world_depth = median(&mut state.depth_ratios);
    state.depth_ratios.clear();
    state.right.clear();
    state.pass = 0;
    PASS.store(0, Relaxed);
    state.cut = false;
    if passes == 2 {
        state.profile.frames += 1;
    }
    if let Some(last) = state.profile.last_draw.take() {
        state.profile.after_last += last.elapsed();
    }
    state.profile.enter(0);
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
        if PROFILING {
            log!("single pass: render thread per frame: {}", state.profile.report());
        }
        let frames = c.frames_on.max(1) as f64;
        let queries = QUERIES.iter().map(|(target, by_pass)| {
            let by_pass: Vec<String> = by_pass.iter().map(|n| format!("{:.1}", n.load(Relaxed) as f64 / frames)).collect();
            format!("{target:#x} {}", by_pass.join("/"))
        });
        log!(
            "single pass: of the right eye's draws writing neither colour nor depth: {} in a query, {} with the stencil test on, {} discarding, {} with the depth test on, {} of {} checked writing colour past the first draw buffer; by program: {}; conditional renders per frame {:.1}; the right eye's queries answered by the left eye's {:.1}, made by its own pass as only another thing's left query had their rows {:.1}; queries begun per frame (before/left/right pass): {}",
            c.unwritten_query,
            c.unwritten_stencil,
            c.unwritten_discard,
            c.unwritten_depth_test,
            c.unwritten_other_buffers,
            c.unwritten_checked,
            {
                let mut by: Vec<_> = state.unwritten_by.iter().map(|(p, n)| (*n, state.programs.get(p).map_or("?", |p| p.converted.name), *p)).collect();
                by.sort_unstable_by(|a, b| b.0.cmp(&a.0));
                by.iter().take(16).map(|(n, name, p)| format!("{name} ({p}) {:.1}", *n as f64 / frames)).collect::<Vec<_>>().join(", ")
            },
            CONDITIONAL.load(Relaxed) as f64 / frames,
            c.queries_shared as f64 / frames,
            c.queries_rows_only as f64 / frames,
            queries.collect::<Vec<_>>().join(", ")
        );
        log!("single pass: the right eye's stretch draws the left eye's pass did not make (culled for the left eye), drawn by the right eye's: {} ({:.1} a frame; by program: {}); the eye matrix paired by the last one in {} frames", c.right_only, c.right_only as f64 / frames, {
            let mut by: Vec<_> = state.right_only_by.iter().map(|(p, n)| (*n, state.programs.get(p).map_or("?", |p| p.converted.name), *p)).collect();
            by.sort_unstable_by(|a, b| b.0.cmp(&a.0));
            by.iter().take(12).map(|(n, name, p)| format!("{name} ({p}) {:.1}", *n as f64 / frames)).collect::<Vec<_>>().join(", ")
        }, c.rematched);
        log!(
            "single pass: the right eye's stretch draws of a mesh the left eye's drew a little elsewhere (moved between the passes), skipped: {} ({:.1} a frame; by program, farthest: {}); right eye's own draws of a mesh the left eye's drew farther off: {}; the averaged eye matrix: started afresh {} times, kept while a frame measured far from it in {}, a frame's own measure from it at most {:.2e}, pairs averaged {}",
            c.moved,
            c.moved as f64 / frames,
            {
                let mut by: Vec<_> = state.moved_by.iter().map(|(p, (n, far))| (*n, *far, state.programs.get(p).map_or("?", |p| p.converted.name), *p)).collect();
                by.sort_unstable_by(|a, b| b.0.cmp(&a.0));
                by.iter().take(12).map(|(n, far, name, p)| format!("{name} ({p}) {:.1} {far:.3}", *n as f64 / frames)).collect::<Vec<_>>().join(", ")
            },
            c.right_only_same_geometry,
            c.matrix_jumps,
            c.matrix_held,
            state.largest_jump,
            c.fitted
        );
        log!("single pass: the right eye's stretch draws found among the left eye's by shape alone (streamed at another offset), skipped: {} ({:.1} a frame; by program: {})", c.by_shape, c.by_shape as f64 / frames, {
            let mut by: Vec<_> = state.by_shape_by.iter().map(|(p, n)| (*n, state.programs.get(p).map_or("?", |p| p.converted.name), *p)).collect();
            by.sort_unstable_by(|a, b| b.0.cmp(&a.0));
            by.iter().take(12).map(|(n, name, p)| format!("{name} ({p}) {:.1}", *n as f64 / frames)).collect::<Vec<_>>().join(", ")
        });
        let (shared, own) = state.screen.counts();
        log!("single pass: programs reading the scene's maps: {shared} drawn for both eyes, {own} by each eye, the rest being learnt");
        log!("single pass: the right eye's stretch draws each eye draws itself, per frame by program: {}", {
            let mut by: Vec<_> = state.own_by.iter().map(|(p, n)| (*n, state.programs.get(p).map_or("?", |p| p.converted.name), *p)).collect();
            by.sort_unstable_by(|a, b| b.0.cmp(&a.0));
            by.iter().take(24).map(|(n, name, p)| format!("{name} ({p}) {:.1}", *n as f64 / frames)).collect::<Vec<_>>().join(", ")
        });
        log!(
            "single pass: {} draws, {} with a converted program, {} perspective, {} into a stood-in framebuffer, {} in the scene's stretch; {} of the right eye's writing neither colour nor depth, drawn; {} with their own depth projection, drawn by each eye (world's {:?}); {} of programs reading the scene's maps or not carried by the eye matrix ({} judged), drawn by each eye; {} of programs reading the scene's maps made for both eyes; {} carried with the scissor test off (their scissor smaller than the viewport)",
            c.draws,
            c.converted,
            c.perspective,
            c.stood_in,
            c.in_stretch,
            c.unwritten,
            c.own_projection,
            state.world_depth,
            c.misfit_draws,
            state.misfits.misfits.len(),
            c.screen_shared,
            c.scissor_opened
        );
    }
}

/// The stop on the render thread: every switch off, the engine's own objects bound again, then the
/// detours stand aside.
fn stop(state: &mut State) {
    SHARING.store(false, Relaxed);
    forward_pending();
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
    state.map_arrays.clear();
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
type CapFn = unsafe extern "system" fn(u32);
type VoidFn = unsafe extern "system" fn();
type ReadPixelsFn = unsafe extern "system" fn(i32, i32, i32, i32, u32, u32, *mut c_void);
type ClientWaitSyncFn = unsafe extern "system" fn(*const c_void, u32, u64) -> u32;
type WaitSyncFn = unsafe extern "system" fn(*const c_void, u32, u64);
type GetSyncivFn = unsafe extern "system" fn(*const c_void, u32, i32, *mut i32, *mut i32);
type GetBufferSubDataFn = unsafe extern "system" fn(u32, isize, isize, *mut c_void);
type MapBufferRangeFn = unsafe extern "system" fn(u32, isize, isize, u32) -> *mut c_void;
type MapBufferFn = unsafe extern "system" fn(u32, u32) -> *mut c_void;
type GetvFn = unsafe extern "system" fn(u32, *mut c_void);
type GetTexImageFn = unsafe extern "system" fn(u32, i32, u32, u32, *mut c_void);
type GetQueryObjectFn = unsafe extern "system" fn(u32, u32, *mut c_void);
type ColorMaskFn = unsafe extern "system" fn(u8, u8, u8, u8);
type DepthMaskFn = unsafe extern "system" fn(u8);
type BeginQueryFn = unsafe extern "system" fn(u32, u32);

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
static ACTIVE_TEXTURE: Original<CapFn> = Original::new();
static READ_PIXELS: Original<ReadPixelsFn> = Original::new();
static FINISH: Original<VoidFn> = Original::new();
static FLUSH: Original<VoidFn> = Original::new();
static CLIENT_WAIT_SYNC: Original<ClientWaitSyncFn> = Original::new();
static WAIT_SYNC: Original<WaitSyncFn> = Original::new();
static GET_SYNC_IV: Original<GetSyncivFn> = Original::new();
static GET_BUFFER_SUB_DATA: Original<GetBufferSubDataFn> = Original::new();
static MAP_BUFFER_RANGE: Original<MapBufferRangeFn> = Original::new();
static MAP_BUFFER: Original<MapBufferFn> = Original::new();
static GET_INTEGER_V: Original<GetvFn> = Original::new();
static GET_BOOLEAN_V: Original<GetvFn> = Original::new();
static GET_FLOAT_V: Original<GetvFn> = Original::new();
static GET_TEX_IMAGE: Original<GetTexImageFn> = Original::new();
static GET_QUERY_OBJECT_UIV: Original<GetQueryObjectFn> = Original::new();
static GET_QUERY_OBJECT_UI64V: Original<GetQueryObjectFn> = Original::new();
static GET_QUERY_OBJECT_IV: Original<GetQueryObjectFn> = Original::new();
static GET_QUERY_OBJECT_UIV_ARB: Original<GetQueryObjectFn> = Original::new();
static GET_QUERY_OBJECT_IV_ARB: Original<GetQueryObjectFn> = Original::new();
static GET_QUERY_OBJECT_I64V: Original<GetQueryObjectFn> = Original::new();

/// The engine's calls that may wait for the driver or the GPU, timed: their names, and their time
/// (nanoseconds) and calls since the last report.
const WAITS: [&str; 19] = [
    "glReadPixels",
    "glFinish",
    "glFlush",
    "glClientWaitSync",
    "glWaitSync",
    "glGetSynciv",
    "glGetBufferSubData",
    "glMapBufferRange",
    "glMapBuffer",
    "glGetIntegerv",
    "glGetBooleanv",
    "glGetFloatv",
    "glGetTexImage",
    "glGetQueryObjectuiv",
    "glGetQueryObjectui64v",
    "glGetQueryObjectiv",
    "glGetQueryObjectuivARB",
    "glGetQueryObjectivARB",
    "glGetQueryObjecti64v",
];
static WAITED: [(AtomicU64, AtomicU64); 19] = [const { (AtomicU64::new(0), AtomicU64::new(0)) }; 19];

/// `call` timed as the engine's wait `index`.
fn timed<T>(index: usize, call: impl FnOnce() -> T) -> T {
    if !PROFILING {
        return call();
    }
    let started = Instant::now();
    let out = call();
    WAITED[index].0.fetch_add(started.elapsed().as_nanos() as u64, Relaxed);
    WAITED[index].1.fetch_add(1, Relaxed);
    out
}
static COLOR_MASK: Original<ColorMaskFn> = Original::new();
static DEPTH_MASK: Original<DepthMaskFn> = Original::new();
static BEGIN_QUERY: Original<BeginQueryFn> = Original::new();
static END_QUERY: Original<CapFn> = Original::new();
static BEGIN_QUERY_ARB: Original<BeginQueryFn> = Original::new();
static END_QUERY_ARB: Original<CapFn> = Original::new();
static BEGIN_OCCLUSION_NV: Original<CapFn> = Original::new();
static END_OCCLUSION_NV: Original<VoidFn> = Original::new();
static BEGIN_CONDITIONAL: Original<BeginQueryFn> = Original::new();
static BEGIN_CONDITIONAL_NV: Original<BeginQueryFn> = Original::new();
static ENABLE: Original<CapFn> = Original::new();
static DISABLE: Original<CapFn> = Original::new();
static SCISSOR: Original<RectFn> = Original::new();
/// The scissor test is off for the draw being made, to be put back after it.
static SCISSOR_OPENED: AtomicBool = AtomicBool::new(false);

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
    hook!(SCISSOR, "glScissor", scissor as RectFn, false);
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
    let unit = hook!(ACTIVE_TEXTURE, "glActiveTexture", active_texture as CapFn, false);
    let state_kept = [
        hook!(COLOR_MASK, "glColorMask", color_mask as ColorMaskFn, false),
        hook!(DEPTH_MASK, "glDepthMask", depth_mask as DepthMaskFn, false),
        hook!(ENABLE, "glEnable", enable as CapFn, false),
        hook!(DISABLE, "glDisable", disable as CapFn, false),
    ];
    hook!(BEGIN_QUERY, "glBeginQuery", begin_query as BeginQueryFn, false);
    hook!(END_QUERY, "glEndQuery", end_query as CapFn, false);
    hook!(BEGIN_QUERY_ARB, "glBeginQueryARB", begin_query_arb as BeginQueryFn, false);
    hook!(END_QUERY_ARB, "glEndQueryARB", end_query_arb as CapFn, false);
    hook!(BEGIN_OCCLUSION_NV, "glBeginOcclusionQueryNV", begin_occlusion_nv as CapFn, false);
    hook!(END_OCCLUSION_NV, "glEndOcclusionQueryNV", end_occlusion_nv as VoidFn, false);
    hook!(BEGIN_CONDITIONAL, "glBeginConditionalRender", begin_conditional as BeginQueryFn, false);
    hook!(BEGIN_CONDITIONAL_NV, "glBeginConditionalRenderNV", begin_conditional_nv as BeginQueryFn, false);
    if PROFILING {
        hook!(READ_PIXELS, "glReadPixels", read_pixels as ReadPixelsFn, false);
        hook!(FINISH, "glFinish", finish as VoidFn, false);
        hook!(FLUSH, "glFlush", flush as VoidFn, false);
        hook!(CLIENT_WAIT_SYNC, "glClientWaitSync", client_wait_sync as ClientWaitSyncFn, false);
        hook!(WAIT_SYNC, "glWaitSync", wait_sync as WaitSyncFn, false);
        hook!(GET_SYNC_IV, "glGetSynciv", get_synciv as GetSyncivFn, false);
        hook!(GET_BUFFER_SUB_DATA, "glGetBufferSubData", get_buffer_sub_data as GetBufferSubDataFn, false);
        hook!(MAP_BUFFER_RANGE, "glMapBufferRange", map_buffer_range as MapBufferRangeFn, false);
        hook!(MAP_BUFFER, "glMapBuffer", map_buffer as MapBufferFn, false);
        hook!(GET_INTEGER_V, "glGetIntegerv", get_integer_v as GetvFn, false);
        hook!(GET_BOOLEAN_V, "glGetBooleanv", get_boolean_v as GetvFn, false);
        hook!(GET_FLOAT_V, "glGetFloatv", get_float_v as GetvFn, false);
        hook!(GET_TEX_IMAGE, "glGetTexImage", get_tex_image as GetTexImageFn, false);
    }
    hook!(GET_QUERY_OBJECT_UIV, "glGetQueryObjectuiv", get_query_object_uiv as GetQueryObjectFn, false);
    hook!(GET_QUERY_OBJECT_UI64V, "glGetQueryObjectui64v", get_query_object_ui64v as GetQueryObjectFn, false);
    hook!(GET_QUERY_OBJECT_IV, "glGetQueryObjectiv", get_query_object_iv as GetQueryObjectFn, false);
    hook!(GET_QUERY_OBJECT_UIV_ARB, "glGetQueryObjectuivARB", get_query_object_uiv_arb as GetQueryObjectFn, false);
    hook!(GET_QUERY_OBJECT_IV_ARB, "glGetQueryObjectivARB", get_query_object_iv_arb as GetQueryObjectFn, false);
    hook!(GET_QUERY_OBJECT_I64V, "glGetQueryObjecti64v", get_query_object_i64v as GetQueryObjectFn, false);
    Shadow::take();
    UNIT_TRUSTED.store(unit > 0, Relaxed);
    SHADOW_TRUSTED.store(state_kept.iter().all(|&n| n > 0), Relaxed);
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
    if let Some(screen) = program.converted.screen {
        kept(&mut program.vertex, screen.vertex_rows, location, count, values);
        kept(&mut program.fragment, screen.fragment_rows, location, count, values);
    }
}

/// An upload of `count` rows at `location` kept in `rows` (the array at `first`, once read).
fn kept(rows: &mut [Row], (first, n): (i32, usize), location: i32, count: i32, values: *const f32) {
    if rows.len() != n || values.is_null() {
        return;
    }
    for i in 0..count.max(0) {
        let at = location + i - first;
        if (0..n as i32).contains(&at) {
            // SAFETY: the engine's buffer holds `count` vec4s.
            rows[at as usize] = unsafe { *values.add(i as usize * 4).cast::<Row>() };
        }
    }
}

/// `n` rows of `program` from `first`, as the driver has them.
fn driver_rows(gl: &Gl, program: u32, (first, n): (i32, usize)) -> Vec<Row> {
    (0..n as i32)
        .map(|i| {
            let mut row = [0.0; 4];
            // SAFETY: a plain query into a vec4.
            unsafe { (gl.get_uniform_fv)(program, first + i, row.as_mut_ptr()) };
            row
        })
        .collect()
}

/// A program reading the scene's maps about to draw in the stretch: its rows brought up to date
/// (read from the driver the first time, on the checking frames to check them, and always once
/// they disagreed), the draw recorded for learning, and the arrays its maps read when it is drawn
/// for both eyes; none when each eye draws it.
fn screen_draw(state: &mut State, current: u32, screen: &Screen, rows: &Rows, checking: bool) -> Option<Vec<(u32, u32)>> {
    let gl = state.gl.as_ref()?;
    let program = state.programs.get_mut(&current)?;
    let wants = state.screen.wants(current, checking);
    let shared = state.screen.shared(current).is_some();
    if !wants && !shared {
        return None;
    }
    if program.vertex.len() != screen.vertex_rows.1 || state.rows_untrusted {
        program.vertex = driver_rows(gl, current, screen.vertex_rows);
        program.fragment = driver_rows(gl, current, screen.fragment_rows);
    } else if checking {
        let (vertex, fragment) = (driver_rows(gl, current, screen.vertex_rows), driver_rows(gl, current, screen.fragment_rows));
        if vertex != program.vertex || fragment != program.fragment {
            log!("single pass: the rows kept from the engine's uploads disagree with the driver's ({}): read from the driver from here", program.converted.name);
            state.rows_untrusted = true;
            program.vertex = vertex;
            program.fragment = fragment;
        }
    }
    if wants && state.matrix.is_some() && (state.pass == 1 || state.pass == 2) {
        let position: Vec<usize> = program.converted.rows.iter().map(|&at| (at - screen.vertex_rows.0) as usize).collect();
        let vertex = program.vertex.iter().enumerate().filter(|(i, _)| !position.contains(i)).map(|(_, row)| *row).collect();
        let drawn = Drawn { program: current, name: program.converted.name, position: *rows, vertex, fragment: program.fragment.clone() };
        let list = if state.pass == 1 { &mut state.screen.left } else { &mut state.screen.right };
        if list.len() < SCREEN_RECORDED {
            list.push(drawn);
        }
    }
    if !shared {
        return None;
    }
    if program.map_units.len() != screen.maps.len() || checking {
        program.map_units = screen
            .maps
            .iter()
            .map(|&(location, _)| {
                let mut unit = 0;
                // SAFETY: a plain query into an int.
                unsafe { (gl.get_uniform_iv)(current, location, &mut unit) };
                unit as u32
            })
            .collect();
    }
    let layers = state.layers.as_ref()?;
    screen.maps.iter().zip(&program.map_units).map(|(&(_, ours), unit)| Some((ours, layers.array(state.units.get(unit)?.1)?))).collect()
}

/// The left eye's draw of a program reading the scene's maps made for both eyes: its maps'
/// arrays bound to their samplers' units, and the right eye's rows given (worked out from the
/// left eye's).
fn give_right_eye(gl: &Gl, program: &mut Program, learner: &Learner, map_arrays: &mut HashMap<u32, u32>, arrays: &[(u32, u32)]) {
    for &(unit, array) in arrays {
        if map_arrays.get(&unit) != Some(&array) {
            // SAFETY: a plain bind of our array to a unit the engine's samplers do not use.
            unsafe { (gl.bind_multi_texture)(GL_TEXTURE0 + unit, GL_TEXTURE_2D_ARRAY, array) };
            map_arrays.insert(unit, array);
        }
    }
    let (Some(screen), Some((carried, k))) = (program.converted.screen, learner.shared(program.converted.program)) else { return };
    let right: Vec<Row> = program.fragment.iter().zip(carried).map(|(row, &carried)| if carried { screen::carry(row, k) } else { *row }).collect();
    if right != program.right_given {
        // SAFETY: a plain GL call on the engine's program, with as many vec4s as its right rows.
        unsafe { (gl.program_uniform_4fv)(program.converted.program, screen.right_rows, right.len() as i32, right.as_ptr().cast()) };
        program.right_given = right;
    }
}

/// Whether the draw about to be made goes ahead (one of the right eye's pass already drawn is
/// skipped).
fn draw(geometry: u64, shape: u64) -> bool {
    let goes_ahead = draw_locked(geometry, shape);
    match PENDING.load(Relaxed) {
        // A query of the right eye answered by the left eye's: nothing in it is drawn.
        PENDING_SHARED => false,
        PENDING_OPEN => {
            forward_pending();
            goes_ahead
        }
        _ => goes_ahead,
    }
}

/// A left eye's query made around a stretch draw for both eyes.
#[derive(Clone, Copy, Debug)]
struct LeftQuery {
    id: u32,
    rows: Rows,
    geometry: u64,
    /// Already the answer to one of the right eye's queries.
    taken: bool,
}

/// What the right eye's query was matched to among the left eye's.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum QueryMatch {
    /// The left eye's query of the same thing: its answer is the right eye's.
    Same(u32),
    /// Only a left query of something else with the same rows (the world's bounding boxes share the
    /// view's rows, each box its own geometry): not answered by it.
    RowsOnly,
    None,
}

/// The left eye's query whose draw the right eye's draw with `rows` and `geometry` copies, looked
/// for from `cursor` on (the passes make their queries in much the same order), then from the
/// start; each left query answers one right query. Rows alone do not tell the queries apart (the
/// world's bounding boxes all share the view's rows): answered by another box's query, the right
/// eye's pass culled what it sees and drew what it does not, a wall here and a floor there.
fn left_query(left: &mut [LeftQuery], cursor: &mut usize, matrix: &Rows, rows: &Rows, geometry: u64) -> QueryMatch {
    let start = (*cursor).min(left.len());
    let mut rows_only = false;
    for i in (start..left.len()).chain(0..start) {
        let query = &mut left[i];
        if query.taken || eye_matrix::misfit(matrix, &query.rows, rows) > eye_matrix::AGREES * 10.0 {
            continue;
        }
        if query.geometry != geometry {
            rows_only = true;
            continue;
        }
        query.taken = true;
        *cursor = i + 1;
        return QueryMatch::Same(query.id);
    }
    if rows_only { QueryMatch::RowsOnly } else { QueryMatch::None }
}

fn draw_locked(geometry: u64, shape: u64) -> bool {
    let mut guard = STATE.locked();
    let Some(state) = guard.as_mut() else { return true };
    if !state.live {
        return true;
    }
    if !PROFILING {
        return draw_in(state, geometry, shape);
    }
    let started = Instant::now();
    if matches!(state.profile.part, 2 | 4) {
        let since = state.profile.last_draw.into_iter().chain(state.profile.since).max();
        if let Some(since) = since {
            let name = format!("{}{}", if state.profile.part == 2 { 'L' } else { 'R' }, state.draw_framebuffer);
            let entry = state.profile.by_target.entry(name).or_default();
            entry.0 += started - since;
            entry.1 += 1;
        }
    }
    let goes_ahead = draw_in(state, geometry, shape);
    state.profile.draws[state.profile.part] += 1;
    let now = Instant::now();
    state.profile.ours += now - started;
    state.profile.last_draw = Some(now);
    goes_ahead
}

fn draw_in(state: &mut State, geometry: u64, shape: u64) -> bool {
    if state.inspect.is_none() {
        return draw_decided(state, geometry, shape);
    }
    state.last_path = "?";
    let pass = state.pass;
    let goes_ahead = draw_decided(state, geometry, shape);
    inspected_draw(state, pass, geometry, shape, goes_ahead);
    goes_ahead
}

fn draw_decided(state: &mut State, geometry: u64, shape: u64) -> bool {
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
        state.profile.enter(if state.pass >= 2 { 4 } else { 2 });
        if state.pass == 1 {
            state.rebind();
        }
    }
    state.counts.in_stretch += u64::from(stretch);
    let Some(rows) = rows else {
        state.last_path = "not converted";
        return true;
    };
    let ratio = depth_ratio(&rows);
    if stretch && state.pass == 1 && state.depth_ratios.len() < DEPTH_RATIOS {
        state.depth_ratios.extend(ratio);
    }
    let checking = state.live_frames % CHECK_EVERY == 0;
    let screen = state.programs.get(&current).and_then(|p| p.converted.screen);
    let screen_arrays = screen.filter(|_| stretch).and_then(|screen| screen_draw(state, current, screen, &rows, checking));
    let Some(gl) = state.gl.as_ref() else { return true };
    let misfit = stretch && (state.misfits.misfits.contains(&current) || state.programs.get(&current).is_some_and(|p| p.converted.reads_screen) && screen_arrays.is_none());
    let own_projection = misfit || stretch && state.world_depth.zip(ratio).is_some_and(|(world, ratio)| (ratio - world).abs() > 1e-3 * world.abs().max(1e-6));
    if PENDING.load(Relaxed) == PENDING_OPEN && stretch && state.pass == 2 && !own_projection && let Some(matrix) = state.matrix.as_ref() {
        let matched = left_query(&mut state.left_queries, &mut state.query_cursor, matrix, &rows, geometry);
        state.counts.queries_rows_only += u64::from(matched == QueryMatch::RowsOnly);
        if let QueryMatch::Same(left) = matched {
            // Found: counted for the program's judgment too, or only the draws not found would be.
            if checking {
                let name = state.programs.get(&current).map_or("?", |p| p.converted.name);
                state.misfits.check(current, name, &rows, matrix);
            }
            let right = PENDING_ID.load(Relaxed);
            QUERIES_SHARED.lock().unwrap_or_else(|e| e.into_inner()).insert(right, left);
            PENDING.store(PENDING_SHARED, Relaxed);
            state.counts.queries_shared += 1;
            state.last_path = "right: its query answered by the left eye's, not made";
            return false;
        }
    }
    let program = state.programs.get_mut(&current).expect("looked up above");
    if own_projection {
        state.counts.misfit_draws += u64::from(misfit);
        // Drawn into the pass's own layer (the switch off: layer 0 of the layered framebuffer in
        // the left eye's pass, the right eye's layer in its own).
        set_switch(gl, program, false);
        state.counts.own_projection += 1;
        state.last_path = if misfit { "each eye its own (a misfit)" } else { "each eye its own (its own depth projection)" };
        if state.pass == 2 {
            *state.own_by.entry(current).or_default() += 1;
        }
        return true;
    }
    match (stretch, state.pass) {
        (true, 1) => {
            if state.left.len() < RECORDED {
                state.left.push((current, rows));
            }
            if checking {
                state.misfits.left.entry(current).or_default().push(rows);
            }
            let Some(matrix) = state.matrix else {
                set_switch(gl, program, false);
                state.last_path = "left: no eye matrix, its own";
                return true;
            };
            if program.matrix != state.matrix_number {
                for (row, values) in matrix.iter().enumerate() {
                    // SAFETY: a plain GL call on the engine's program, with one vec4.
                    unsafe { (gl.program_uniform_4fv)(current, program.converted.stereo[row], 1, values.as_ptr()) };
                }
                program.matrix = state.matrix_number;
            }
            if let Some(arrays) = &screen_arrays {
                give_right_eye(gl, program, &state.screen, &mut state.map_arrays, arrays);
                state.counts.screen_shared += 1;
            }
            set_switch(gl, program, true);
            if scissor_clips(state) {
                open_scissor();
                state.counts.scissor_opened += 1;
            }
            state.counts.stereo += 1;
            state.last_path = "left: carried into the right eye";
            state.carried.keep(current, geometry, shape, eye_matrix::carry(&matrix, &rows));
            let query = ACTIVE_QUERY_ID.load(Relaxed);
            if query != 0 && state.left_queries.last().is_none_or(|q| q.id != query) {
                state.left_queries.push(LeftQuery { id: query, rows, geometry, taken: false });
            }
            true
        }
        (true, _) => {
            if state.right.len() < RECORDED {
                state.right.push((current, rows));
            }
            if checking && let Some(matrix) = state.matrix.as_ref() {
                let name = state.programs.get(&current).map_or("?", |p| p.converted.name);
                state.misfits.check(current, name, &rows, matrix);
            }
            let program = state.programs.get_mut(&current).expect("looked up above");
            set_switch(gl, program, false);
            let check = state.live_frames % SHADOW_CHECK_EVERY == 0;
            if !Shadow::writes(check).any() {
                state.counts.unwritten += 1;
                let bits = SHADOW.load(Relaxed);
                state.counts.unwritten_query += u64::from(ACTIVE_QUERY.load(Relaxed) != 0);
                state.counts.unwritten_stencil += u64::from(bits & STENCIL_TEST != 0);
                state.counts.unwritten_discard += u64::from(bits & DISCARD != 0);
                state.counts.unwritten_depth_test += u64::from(bits & DEPTH_TEST != 0);
                *state.unwritten_by.entry(current).or_default() += 1;
                if check {
                    state.counts.unwritten_checked += 1;
                    state.counts.unwritten_other_buffers += u64::from(other_buffers_written());
                }
                state.last_path = "right: writes neither colour nor depth, drawn";
                return true;
            }
            if state.matrix.is_some() {
                let found = state.carried.find(current, geometry, shape, &rows);
                let name = state.programs.get(&current).map_or("?", |p| p.converted.name);
                let moved = found.is_none() && state.carried.nearest(current, geometry, &rows).is_some_and(|near| near <= MOVED_BETWEEN);
                state.misfits.carried(current, name, found.is_some() || moved);
                if found == Some(Found::Shape) {
                    state.counts.by_shape += 1;
                    *state.by_shape_by.entry(current).or_default() += 1;
                }
                if found.is_some() {
                    state.counts.skipped += 1;
                    state.last_path = if found == Some(Found::Shape) { "right: found carried by shape, skipped" } else { "right: found carried, skipped" };
                    return false;
                }
                let nearest = state.carried.nearest(current, geometry, &rows);
                if let Some(near) = nearest.filter(|&near| near <= MOVED_BETWEEN) {
                    // The same mesh the left eye's pass drew, moved a little between the passes:
                    // drawn too it would show twice in the right eye.
                    state.counts.moved += 1;
                    let entry = state.moved_by.entry(current).or_default();
                    entry.0 += 1;
                    entry.1 = entry.1.max(near);
                    state.last_path = "right: moved between the passes, skipped";
                    return false;
                }
                state.counts.right_only_same_geometry += u64::from(nearest.is_some());
                // Culled for the left eye: the right eye's pass draws it into its own layer.
                state.counts.right_only += 1;
                *state.right_only_by.entry(current).or_default() += 1;
                state.last_path = "right: not among the left eye's, drawn";
            } else {
                state.last_path = "right: no eye matrix, drawn";
            }
            true
        }
        _ => {
            set_switch(gl, program, false);
            state.last_path = "outside the stretch, each eye its own";
            true
        }
    }
}

// ---- Inspected frames ----

/// Every draw of [`INSPECTED_FRAMES`] frames, both passes, written into a capture's folder as
/// `draws.txt` when the viewer makes a capture of what the headset shows (a new `capture-N` folder
/// in the run's `captures` folder, looked for every [`CAPTURE_LOOK_EVERY`] frames): how each draw
/// was drawn (carried into the right eye, skipped as carried, each eye its own...), its program,
/// framebuffer, depth projection, GL state and where it lands on screen, so a fault seen in the
/// capture is found among the draws.
struct Inspect {
    folder: std::path::PathBuf,
    frames: u32,
    lines: Vec<String>,
}

const INSPECTED_FRAMES: u32 = 3;
const INSPECTED_LINES: usize = 40_000;
const CAPTURE_LOOK_EVERY: u64 = 6;

/// The run's captures folder, set by the presenter's install: captures there are inspected.
static CAPTURES: Mutex<Option<std::path::PathBuf>> = Mutex::new(None);

/// Draws of the frames after each capture of what the headset shows into `folder` are written
/// beside it ([`Inspect`]).
pub fn inspect_captures(folder: std::path::PathBuf) {
    *CAPTURES.locked() = Some(folder);
}

/// At each frame's end: an inspection's frame ends (written once the last is), or a new capture's
/// folder begins one.
fn inspect_frame(state: &mut State) {
    if let Some(mut inspect) = state.inspect.take() {
        inspect.lines.push(format!("---- frame {} ends ({} passes) ----", state.live_frames - 1, state.pass));
        inspect.frames -= 1;
        if inspect.frames > 0 {
            state.inspect = Some(inspect);
        } else {
            let path = inspect.folder.join("draws.txt");
            let text = inspect.lines.join("\n") + "\n";
            std::thread::spawn(move || {
                if let Err(e) = std::fs::write(&path, text) {
                    log!("single pass: {} not written: {e}", path.display());
                }
            });
            log!("single pass: the draws of {INSPECTED_FRAMES} frames written beside the capture");
        }
        return;
    }
    if state.live_frames % CAPTURE_LOOK_EVERY != 0 {
        return;
    }
    if state.captures.is_none() {
        let Some(folder) = CAPTURES.locked().clone() else { return };
        let next = (1..).find(|n| !folder.join(format!("capture-{n}")).exists()).unwrap_or(1);
        state.captures = Some((folder, next));
        return;
    }
    let Some((folder, next)) = state.captures.as_mut() else { return };
    let capture = folder.join(format!("capture-{next}"));
    if !capture.exists() {
        return;
    }
    *next += 1;
    let lines = vec![
        format!("The draws of {INSPECTED_FRAMES} frames from frame {} on, both passes (1 the left eye's, 2 the right's), as the single pass drew them.", state.live_frames),
        "Each line: pass, framebuffer, program, how it was drawn, depth projection ratio (world's in the header), GL state (depth test function, depth writes, polygon offset, blend, cull, the scissor while its test is on: \"opened\" when turned off for the right eye's copy), where the model's origin lands (NDC x y z), for the left eye's carried draws where their copy lands in the right eye, and for the right eye's own the farthest-apart of the left eye's carried copies of the same mesh. Draws of decals, particles and unconverted programs add, after bars: GL errors raised since the draw before, the framebuffer bound and its attachments, the stencil test, the colour mask and cull face, each sampler's unit and texture (ours: a layer's view or an array), and the fragment rows.".into(),
        format!("world depth ratio {:?}, eye matrix {:?}", state.world_depth, state.matrix),
    ];
    state.inspect = Some(Inspect { folder: capture, frames: INSPECTED_FRAMES, lines });
}

/// Where `rows` put the model's origin, in NDC.
fn origin(rows: &Rows) -> String {
    let w = rows[3][3];
    if w.abs() < 1e-9 {
        return "(w 0)".into();
    }
    format!("({:.4} {:.4} {:.5})", rows[0][3] / w, rows[1][3] / w, rows[2][3] / w)
}

fn inspected_draw(state: &mut State, pass: u32, geometry: u64, shape: u64, goes_ahead: bool) {
    let Some(inspect) = state.inspect.as_ref() else { return };
    if inspect.lines.len() >= INSPECTED_LINES {
        return;
    }
    let current = state.current;
    let program = state.programs.get(&current);
    let name = program.map_or("-", |p| p.converted.name);
    let rows = program.map(|p| p.rows);
    let (mut depth_func, mut depth_write, mut offset, mut blend, mut cull, mut depth_test) = (0, 0, 0, 0, 0, 0);
    // SAFETY: state queries into locals (only while inspecting: each waits for the driver).
    unsafe {
        glGetIntegerv(0x0B74, &mut depth_func);
        glGetIntegerv(0x0B72, &mut depth_write);
        glGetIntegerv(0x8037, &mut offset);
        glGetIntegerv(0x0BE2, &mut blend);
        glGetIntegerv(0x0B44, &mut cull);
        glGetIntegerv(0x0B71, &mut depth_test);
    }
    let func = match depth_func {
        0x0200 => "never",
        0x0201 => "less",
        0x0202 => "equal",
        0x0203 => "lequal",
        0x0204 => "greater",
        0x0205 => "notequal",
        0x0206 => "gequal",
        0x0207 => "always",
        _ => "?",
    };
    let mut line = format!(
        "{pass} fb {} {name}({current}) {}{}: ratio {:?} depth {} write {} offset {} blend {} cull {} geometry {geometry:x} shape {shape:x}",
        state.draw_framebuffer,
        state.last_path,
        if goes_ahead { "" } else { " [not drawn]" },
        rows.as_ref().and_then(depth_ratio),
        if depth_test != 0 { func } else { "off" },
        depth_write,
        offset,
        blend,
        cull,
    );
    if SHADOW.load(Relaxed) & SCISSOR_TEST != 0 {
        line += &format!(" scissor {:?}{}", state.scissor, if SCISSOR_OPENED.load(Relaxed) { " opened" } else { "" });
    }
    if let Some(rows) = rows.as_ref() {
        line += &format!(" at {}", origin(rows));
        if pass == 1 && state.last_path.starts_with("left: carried") && let Some(matrix) = state.matrix.as_ref() {
            line += &format!(" copy at {}", origin(&eye_matrix::carry(matrix, rows)));
        }
        if pass >= 2 && let Some(near) = state.carried.nearest(current, geometry, rows) {
            line += &format!(" nearest carried {near:.2e}");
        }
    }
    if name == "-" || name.contains("decal") || name.contains("particle") {
        line += &inspected_detail(state, current);
    }
    if let Some(inspect) = state.inspect.as_mut() {
        inspect.lines.push(line);
    }
}

/// More of how a draw reading the scene's maps (decals, soft particles) or an unconverted one was
/// made, for the inspection: the GL errors raised since the draw before, the framebuffer bound and
/// its attachments, the stencil test, the colour mask, the cull face, each sampler's unit and the
/// texture bound there (ours: a layer's view or an array standing in for an engine eye texture),
/// and the program's fragment rows (`_fa_`).
fn inspected_detail(state: &State, program: u32) -> String {
    let Some(gl) = state.gl.as_ref() else { return String::new() };
    let mut out = String::new();
    let mut errors = Vec::new();
    // SAFETY: state queries into locals (only while inspecting).
    unsafe {
        for _ in 0..4 {
            let error = windows::Win32::Graphics::OpenGL::glGetError();
            if error == 0 {
                break;
            }
            errors.push(format!("{error:#x}"));
        }
        if !errors.is_empty() {
            out += &format!(" | GL errors before {}", errors.join(" "));
        }
        let mut bound = 0;
        glGetIntegerv(0x8CA6, &mut bound);
        out += &format!(" | drawn into {}", describe(Some(gl), bound as u32));
        let (mut func, mut reference, mut mask, mut write) = (0, 0, 0, 0);
        if glIsEnabled(0x0B90) != 0 {
            glGetIntegerv(0x0B92, &mut func);
            glGetIntegerv(0x0B97, &mut reference);
            glGetIntegerv(0x0B93, &mut mask);
            glGetIntegerv(0x0B98, &mut write);
            out += &format!(" | stencil func {func:#x} ref {reference} mask {mask:#x} write {write:#x}");
        } else {
            out += " | stencil off";
        }
        let bounds = glIsEnabled(0x8890) != 0;
        if bounds {
            out += " | depth bounds test on";
        }
        let mut colour = [0u8; 4];
        glGetBooleanv(0x0C23, colour.as_mut_ptr());
        let (mut face, mut front) = (0, 0);
        glGetIntegerv(0x0B45, &mut face);
        glGetIntegerv(0x0B46, &mut front);
        out += &format!(" | colour mask {colour:?} cull face {face:#x} front {front:#x}");
    }
    type IndexedFn = unsafe extern "system" fn(u32, u32, *mut i32);
    let indexed = gl_api::address("glGetIntegerIndexedvEXT").map(|a| {
        // SAFETY: EXT_direct_state_access's query, with its signature.
        unsafe { std::mem::transmute::<usize, IndexedFn>(a) }
    });
    let mut samplers = Vec::new();
    let mut rows = String::new();
    for uniform in gl.uniforms(program) {
        let Some(values) = gl.values(program, &uniform) else { continue };
        if (0x8B5D..=0x8B64).contains(&uniform.kind) || (0x8DC0..=0x8DD8).contains(&uniform.kind) || (0x9108..=0x910A).contains(&uniform.kind) {
            let Some(gl_api::Values::Ints(unit)) = values.first() else { continue };
            let unit = unit[0].max(0) as u32;
            let binding = if uniform.kind == 0x8DC1 { 0x8C1D } else { 0x8069 };
            let mut texture = -1;
            if let Some(indexed) = indexed {
                // SAFETY: a state query into a local.
                unsafe { indexed(binding, unit, &mut texture) };
            }
            let ours = state.layers.as_ref().and_then(|l| l.ours(texture.max(0) as u32));
            let what = match ours {
                Some((engine, Some(layer))) => format!(" (layer {layer} of {engine})"),
                Some((engine, None)) => format!(" (array of {engine})"),
                None => String::new(),
            };
            let engine = state.units.get(&unit).map_or(String::new(), |&(_, t)| format!(" engine {t}"));
            samplers.push(format!("{}=u{unit} t{texture}{what}{engine}", uniform.name));
        } else if uniform.name == "_fa_" || uniform.name.starts_with("monaka_f") {
            let text: Vec<String> = values
                .iter()
                .take(8)
                .map(|v| match v {
                    gl_api::Values::Floats(f) => format!("[{}]", f.iter().map(|x| format!("{x:.4}")).collect::<Vec<_>>().join(" ")),
                    gl_api::Values::Ints(i) => format!("{i:?}"),
                })
                .collect();
            rows += &format!(" | {} {}", uniform.name, text.join(" "));
        }
    }
    if !samplers.is_empty() {
        out += &format!(" | samplers {}", samplers.join(", "));
    }
    out + &rows
}

/// The left eye's stretch ratios kept a frame.
const DEPTH_RATIOS: usize = 512;
/// The draws of programs reading the scene's maps recorded per pass and frame, for learning.
const SCREEN_RECORDED: usize = 1024;
const GL_TEXTURE_2D_ARRAY: u32 = 0x8C1A;

/// A draw's depth projection: its z row against its w row (x, y, z parts). A perspective
/// projection's z row is its w row times a constant (the view's own z) plus a constant, so draws
/// with the same projection share it whatever their model and view.
fn depth_ratio(rows: &Rows) -> Option<f32> {
    let (z, w) = (&rows[2][..3], &rows[3][..3]);
    let ww: f32 = w.iter().map(|v| v * v).sum();
    (ww > 1e-12).then(|| z.iter().zip(w).map(|(z, w)| z * w).sum::<f32>() / ww)
}

fn median(values: &mut [f32]) -> Option<f32> {
    values.sort_unstable_by(f32::total_cmp);
    values.get(values.len() / 2).copied()
}

/// What the draw about to be made writes into its framebuffer. One writing neither colour nor
/// depth leaves the eye textures as they are: its effect is elsewhere (the engine's GPU culling
/// tests), and the right eye's pass makes its own.
#[derive(Clone, Copy, PartialEq, Debug)]
struct Writes {
    colour: bool,
    depth: bool,
    discard: bool,
}

/// The frames between checks of the shadowed state against queries.
const SHADOW_CHECK_EVERY: u64 = 97;

/// The GL state the detours need, kept from the engine's own calls (`glActiveTexture`,
/// `glColorMask`, `glDepthMask`, `glEnable`, `glDisable`): bits of [`SHADOW`]. Checked against
/// queries now and then; once it disagrees (state set some way not hooked) it is queried always.
struct Shadow;

static SHADOW: AtomicU32 = AtomicU32::new(0);
/// The active texture unit (its offset from `GL_TEXTURE0`).
static ACTIVE_UNIT: AtomicU32 = AtomicU32::new(0);
/// The shadow is trusted: every function it is kept from hooked, no check disagreeing.
static SHADOW_TRUSTED: AtomicBool = AtomicBool::new(false);
static UNIT_TRUSTED: AtomicBool = AtomicBool::new(false);

const COLOUR_BITS: u32 = 0xF;
const DEPTH_WRITE: u32 = 0x10;
const DEPTH_TEST: u32 = 0x20;
const DISCARD: u32 = 0x40;
const STENCIL_TEST: u32 = 0x80;
const SCISSOR_TEST: u32 = 0x100;
const GL_DEPTH_TEST: u32 = 0x0B71;
const GL_SCISSOR_TEST: u32 = 0x0C11;
const GL_STENCIL_TEST: u32 = 0x0B90;

/// The query begun and not yet ended (its target; 0 for none), the pass the frame is in (0 before
/// the left eye's, 1, 2), and the queries begun per target and pass.
static ACTIVE_QUERY: AtomicU32 = AtomicU32::new(0);
static ACTIVE_QUERY_ID: AtomicU32 = AtomicU32::new(0);

/// The right eye's pass may answer its queries by the left eye's ([`SHARING`]): a samples-passed
/// query it begins is held back ([`PENDING_OPEN`]) until its first draw, which either is a copy of a
/// left query's draw ([`PENDING_SHARED`]: the query is not made, nor anything in it drawn) or not
/// (the query is begun then, as the engine began it).
static SHARING: AtomicBool = AtomicBool::new(false);
static PENDING: AtomicU32 = AtomicU32::new(PENDING_NONE);
static PENDING_ID: AtomicU32 = AtomicU32::new(0);
static PENDING_TARGET: AtomicU32 = AtomicU32::new(0);
static PENDING_ARB: AtomicBool = AtomicBool::new(false);
const PENDING_NONE: u32 = 0;
const PENDING_OPEN: u32 = 1;
const PENDING_SHARED: u32 = 2;
const GL_SAMPLES_PASSED: u32 = 0x8914;
/// The right eye's queries answered by the left eye's: right id to left id.
static QUERIES_SHARED: std::sync::LazyLock<Mutex<HashMap<u32, u32>>> = std::sync::LazyLock::new(|| Mutex::new(HashMap::new()));

/// The held-back query begun as the engine began it.
fn forward_pending() {
    if PENDING.load(Relaxed) != PENDING_OPEN {
        return;
    }
    PENDING.store(PENDING_NONE, Relaxed);
    let flight = InFlight::enter();
    let (target, id) = (PENDING_TARGET.load(Relaxed), PENDING_ID.load(Relaxed));
    // SAFETY: the game's own call, made late.
    unsafe {
        if PENDING_ARB.load(Relaxed) {
            BEGIN_QUERY_ARB.get(&flight)(target, id);
        } else {
            BEGIN_QUERY.get(&flight)(target, id);
        }
    }
}

/// Whether the engine's query begin is held back.
fn hold_back(target: u32, id: u32, arb: bool) -> bool {
    QUERIES_SHARED.lock().unwrap_or_else(|e| e.into_inner()).remove(&id);
    if target != GL_SAMPLES_PASSED || !SHARING.load(Relaxed) || PENDING.load(Relaxed) != PENDING_NONE {
        return false;
    }
    PENDING_TARGET.store(target, Relaxed);
    PENDING_ID.store(id, Relaxed);
    PENDING_ARB.store(arb, Relaxed);
    PENDING.store(PENDING_OPEN, Relaxed);
    true
}

/// Whether the engine's query end is made (not for a query answered by the left eye's).
fn end_goes_ahead() -> bool {
    match PENDING.load(Relaxed) {
        PENDING_SHARED => {
            PENDING.store(PENDING_NONE, Relaxed);
            false
        }
        PENDING_OPEN => {
            // Nothing drawn in it: made as the engine made it.
            forward_pending();
            true
        }
        _ => true,
    }
}

/// The query the engine reads: the left eye's for one of the right eye's answered by it.
fn answering(id: u32) -> u32 {
    QUERIES_SHARED.lock().unwrap_or_else(|e| e.into_inner()).get(&id).copied().unwrap_or(id)
}
/// Conditional renders begun.
static CONDITIONAL: AtomicU64 = AtomicU64::new(0);
static PASS: AtomicU32 = AtomicU32::new(0);
static QUERIES: [(u32, [AtomicU64; 3]); 5] = [
    (0x8914, [const { AtomicU64::new(0) }; 3]),
    (0x8C2F, [const { AtomicU64::new(0) }; 3]),
    (0x8D6A, [const { AtomicU64::new(0) }; 3]),
    (0x88BF, [const { AtomicU64::new(0) }; 3]),
    (0x8C87, [const { AtomicU64::new(0) }; 3]),
];
const GL_RASTERIZER_DISCARD: u32 = 0x8C89;

impl Shadow {
    fn set(bits: u32, on: bool) {
        if on {
            SHADOW.fetch_or(bits, Relaxed);
        } else {
            SHADOW.fetch_and(!bits, Relaxed);
        }
    }

    fn capability(cap: u32) -> u32 {
        match cap {
            GL_DEPTH_TEST => DEPTH_TEST,
            GL_RASTERIZER_DISCARD => DISCARD,
            GL_STENCIL_TEST => STENCIL_TEST,
            GL_SCISSOR_TEST => SCISSOR_TEST,
            _ => 0,
        }
    }

    /// The shadow taken from the driver (at the install, the context current).
    fn take() {
        let mut bits = 0;
        let mut colour = [0u8; 4];
        let mut depth_write = 0u8;
        let mut unit = 0;
        // SAFETY: state queries into locals.
        unsafe {
            glGetBooleanv(0x0C23, colour.as_mut_ptr());
            glGetBooleanv(0x0B72, &mut depth_write);
            bits |= colour.iter().enumerate().fold(0, |b, (i, &c)| b | (u32::from(c != 0) << i));
            bits |= if depth_write != 0 { DEPTH_WRITE } else { 0 };
            bits |= if glIsEnabled(GL_DEPTH_TEST) != 0 { DEPTH_TEST } else { 0 };
            bits |= if glIsEnabled(GL_RASTERIZER_DISCARD) != 0 { DISCARD } else { 0 };
            bits |= if glIsEnabled(GL_STENCIL_TEST) != 0 { STENCIL_TEST } else { 0 };
            bits |= if glIsEnabled(GL_SCISSOR_TEST) != 0 { SCISSOR_TEST } else { 0 };
            glGetIntegerv(0x84E0, &mut unit);
        }
        SHADOW.store(bits, Relaxed);
        ACTIVE_UNIT.store((unit as u32).wrapping_sub(GL_TEXTURE0), Relaxed);
    }

    fn shadowed() -> Writes {
        let bits = SHADOW.load(Relaxed);
        Writes { colour: bits & COLOUR_BITS != 0, depth: bits & DEPTH_WRITE != 0 && bits & DEPTH_TEST != 0, discard: bits & DISCARD != 0 }
    }

    /// What the draw writes: the shadow's, or queried when it is not trusted or `check`ed.
    fn writes(check: bool) -> Writes {
        if !SHADOW_TRUSTED.load(Relaxed) {
            return Writes::now();
        }
        let shadowed = Self::shadowed();
        if check {
            let queried = Writes::now();
            if queried != shadowed {
                SHADOW_TRUSTED.store(false, Relaxed);
                log!("single pass: the kept draw state disagrees with the driver's ({shadowed:?}, queried {queried:?}): queried from now on");
            }
            return queried;
        }
        shadowed
    }

    /// The active texture unit: the shadow's, or queried when it is not trusted.
    fn unit() -> u32 {
        if UNIT_TRUSTED.load(Relaxed) { ACTIVE_UNIT.load(Relaxed) } else { active_unit() }
    }
}

impl Writes {
    fn now() -> Self {
        let mut colour = [0u8; 4];
        let mut depth_write = 0u8;
        // SAFETY: state queries into locals (GL_COLOR_WRITEMASK, GL_DEPTH_WRITEMASK, GL_DEPTH_TEST,
        // GL_RASTERIZER_DISCARD).
        unsafe {
            glGetBooleanv(0x0C23, colour.as_mut_ptr());
            glGetBooleanv(0x0B72, &mut depth_write);
            Self {
                colour: colour.iter().any(|&c| c != 0),
                depth: depth_write != 0 && glIsEnabled(0x0B71) != 0,
                discard: glIsEnabled(0x8C89) != 0,
            }
        }
    }

    /// Whether it writes colour or depth.
    fn any(&self) -> bool {
        !self.discard && (self.colour || self.depth)
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
        PASS.store(state.pass.min(2) as u32, Relaxed);
        SHARING.store(state.pass == 2 && state.matrix.is_some() && !state.left_queries.is_empty(), Relaxed);
        state.cut = false;
        state.profile.enter(if state.pass >= 2 { 3 } else { 1 });
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

unsafe extern "system" fn scissor(x: i32, y: i32, w: i32, h: i32) {
    let flight = InFlight::enter();
    if let Some(state) = STATE.locked().as_mut() {
        state.scissor = Some([x, y, w, h]);
    }
    // SAFETY: forwards the game's own call.
    unsafe { SCISSOR.get(&flight)(x, y, w, h) }
}

/// Whether the scissor test is on (as the engine's own calls left it) with a rectangle not
/// covering the viewport: the right eye's copy of a carried draw would be clipped by it.
fn scissor_clips(state: &State) -> bool {
    let Some(s) = state.scissor else { return false };
    let v = state.viewport;
    SHADOW_TRUSTED.load(Relaxed)
        && SHADOW.load(Relaxed) & SCISSOR_TEST != 0
        && !(s[0] <= v[0] && s[1] <= v[1] && s[0] + s[2] >= v[0] + v[2] && s[1] + s[3] >= v[1] + v[3])
}

/// The scissor test off for the draw about to be made ([`close_scissor`] puts it back).
fn open_scissor() {
    let flight = InFlight::enter();
    SCISSOR_OPENED.store(true, Relaxed);
    // SAFETY: a plain GL call on the engine's context, undone after the draw.
    unsafe { DISABLE.get(&flight)(GL_SCISSOR_TEST) }
}

/// The scissor test put back after a draw made with it off.
fn close_scissor(flight: &InFlight) {
    if SCISSOR_OPENED.swap(false, Relaxed) {
        // SAFETY: a plain GL call on the engine's context, putting back its state.
        unsafe { ENABLE.get(flight)(GL_SCISSOR_TEST) }
    }
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

macro_rules! timed_detour {
    ($name:ident, $original:ident, $index:expr, ($($arg:ident: $ty:ty),*) $(-> $ret:ty)?) => {
        unsafe extern "system" fn $name($($arg: $ty),*) $(-> $ret)? {
            let flight = InFlight::enter();
            // SAFETY: forwards the game's own call.
            timed($index, || unsafe { $original.get(&flight)($($arg),*) })
        }
    };
}

timed_detour!(read_pixels, READ_PIXELS, 0, (x: i32, y: i32, w: i32, h: i32, format: u32, kind: u32, out: *mut c_void));
timed_detour!(finish, FINISH, 1, ());
timed_detour!(flush, FLUSH, 2, ());
timed_detour!(client_wait_sync, CLIENT_WAIT_SYNC, 3, (sync: *const c_void, flags: u32, timeout: u64) -> u32);
timed_detour!(wait_sync, WAIT_SYNC, 4, (sync: *const c_void, flags: u32, timeout: u64));
timed_detour!(get_synciv, GET_SYNC_IV, 5, (sync: *const c_void, pname: u32, size: i32, length: *mut i32, values: *mut i32));
timed_detour!(get_buffer_sub_data, GET_BUFFER_SUB_DATA, 6, (target: u32, offset: isize, size: isize, out: *mut c_void));
timed_detour!(map_buffer_range, MAP_BUFFER_RANGE, 7, (target: u32, offset: isize, size: isize, access: u32) -> *mut c_void);
timed_detour!(map_buffer, MAP_BUFFER, 8, (target: u32, access: u32) -> *mut c_void);
timed_detour!(get_integer_v, GET_INTEGER_V, 9, (pname: u32, out: *mut c_void));
timed_detour!(get_boolean_v, GET_BOOLEAN_V, 10, (pname: u32, out: *mut c_void));
timed_detour!(get_float_v, GET_FLOAT_V, 11, (pname: u32, out: *mut c_void));
timed_detour!(get_tex_image, GET_TEX_IMAGE, 12, (target: u32, level: i32, format: u32, kind: u32, out: *mut c_void));
macro_rules! query_detour {
    ($name:ident, $original:ident, $index:expr) => {
        unsafe extern "system" fn $name(id: u32, pname: u32, out: *mut c_void) {
            let flight = InFlight::enter();
            let id = answering(id);
            // SAFETY: forwards the game's own call (for the query answering it).
            timed($index, || unsafe { $original.get(&flight)(id, pname, out) })
        }
    };
}

query_detour!(get_query_object_uiv, GET_QUERY_OBJECT_UIV, 13);
query_detour!(get_query_object_ui64v, GET_QUERY_OBJECT_UI64V, 14);
query_detour!(get_query_object_iv, GET_QUERY_OBJECT_IV, 15);
query_detour!(get_query_object_uiv_arb, GET_QUERY_OBJECT_UIV_ARB, 16);
query_detour!(get_query_object_iv_arb, GET_QUERY_OBJECT_IV_ARB, 17);
query_detour!(get_query_object_i64v, GET_QUERY_OBJECT_I64V, 18);

unsafe extern "system" fn active_texture(unit: u32) {
    let flight = InFlight::enter();
    ACTIVE_UNIT.store(unit.wrapping_sub(GL_TEXTURE0), Relaxed);
    // SAFETY: forwards the game's own call.
    unsafe { ACTIVE_TEXTURE.get(&flight)(unit) }
}

unsafe extern "system" fn color_mask(r: u8, g: u8, b: u8, a: u8) {
    let flight = InFlight::enter();
    let bits = [r, g, b, a].iter().enumerate().fold(0, |bits, (i, &c)| bits | (u32::from(c != 0) << i));
    Shadow::set(COLOUR_BITS & !bits, false);
    Shadow::set(bits, true);
    // SAFETY: forwards the game's own call.
    unsafe { COLOR_MASK.get(&flight)(r, g, b, a) }
}

unsafe extern "system" fn depth_mask(on: u8) {
    let flight = InFlight::enter();
    Shadow::set(DEPTH_WRITE, on != 0);
    // SAFETY: forwards the game's own call.
    unsafe { DEPTH_MASK.get(&flight)(on) }
}

unsafe extern "system" fn begin_query(target: u32, id: u32) {
    let flight = InFlight::enter();
    query_begun(target, id);
    if hold_back(target, id, false) {
        return;
    }
    // SAFETY: forwards the game's own call.
    unsafe { BEGIN_QUERY.get(&flight)(target, id) }
}

/// Whether any draw buffer past the first has a colour write mask on (`glGetBooleani_v`).
fn other_buffers_written() -> bool {
    type GetBooleaniFn = unsafe extern "system" fn(u32, u32, *mut u8);
    static GET: std::sync::OnceLock<Option<GetBooleaniFn>> = std::sync::OnceLock::new();
    // SAFETY: the driver's glGetBooleani_v, of that signature.
    let Some(get) = *GET.get_or_init(|| gl_api::address("glGetBooleani_v").map(|a| unsafe { std::mem::transmute::<usize, GetBooleaniFn>(a) })) else { return false };
    (1..8).any(|buffer| {
        let mut mask = [0u8; 4];
        // SAFETY: GL_COLOR_WRITEMASK for one draw buffer, four booleans.
        unsafe { get(0x0C23, buffer, mask.as_mut_ptr()) };
        mask.iter().any(|&m| m != 0)
    })
}

fn query_begun(target: u32, id: u32) {
    ACTIVE_QUERY.store(target, Relaxed);
    ACTIVE_QUERY_ID.store(id, Relaxed);
    if let Some((_, by_pass)) = QUERIES.iter().find(|(t, _)| *t == target) {
        by_pass[PASS.load(Relaxed) as usize].fetch_add(1, Relaxed);
    }
}

unsafe extern "system" fn begin_query_arb(target: u32, id: u32) {
    let flight = InFlight::enter();
    query_begun(target, id);
    if hold_back(target, id, true) {
        return;
    }
    // SAFETY: forwards the game's own call.
    unsafe { BEGIN_QUERY_ARB.get(&flight)(target, id) }
}

unsafe extern "system" fn end_query_arb(target: u32) {
    let flight = InFlight::enter();
    ACTIVE_QUERY.store(0, Relaxed);
    ACTIVE_QUERY_ID.store(0, Relaxed);
    if !end_goes_ahead() {
        return;
    }
    // SAFETY: forwards the game's own call.
    unsafe { END_QUERY_ARB.get(&flight)(target) }
}

unsafe extern "system" fn begin_occlusion_nv(id: u32) {
    let flight = InFlight::enter();
    query_begun(0x8914, id);
    // SAFETY: forwards the game's own call.
    unsafe { BEGIN_OCCLUSION_NV.get(&flight)(id) }
}

unsafe extern "system" fn end_occlusion_nv() {
    let flight = InFlight::enter();
    ACTIVE_QUERY.store(0, Relaxed);
    // SAFETY: forwards the game's own call.
    unsafe { END_OCCLUSION_NV.get(&flight)() }
}

unsafe extern "system" fn begin_conditional(id: u32, mode: u32) {
    let flight = InFlight::enter();
    CONDITIONAL.fetch_add(1, Relaxed);
    // SAFETY: forwards the game's own call.
    unsafe { BEGIN_CONDITIONAL.get(&flight)(id, mode) }
}

unsafe extern "system" fn begin_conditional_nv(id: u32, mode: u32) {
    let flight = InFlight::enter();
    CONDITIONAL.fetch_add(1, Relaxed);
    // SAFETY: forwards the game's own call.
    unsafe { BEGIN_CONDITIONAL_NV.get(&flight)(id, mode) }
}

unsafe extern "system" fn end_query(target: u32) {
    let flight = InFlight::enter();
    ACTIVE_QUERY.store(0, Relaxed);
    ACTIVE_QUERY_ID.store(0, Relaxed);
    if !end_goes_ahead() {
        return;
    }
    // SAFETY: forwards the game's own call.
    unsafe { END_QUERY.get(&flight)(target) }
}

unsafe extern "system" fn enable(cap: u32) {
    let flight = InFlight::enter();
    Shadow::set(Shadow::capability(cap), true);
    // SAFETY: forwards the game's own call.
    unsafe { ENABLE.get(&flight)(cap) }
}

unsafe extern "system" fn disable(cap: u32) {
    let flight = InFlight::enter();
    Shadow::set(Shadow::capability(cap), false);
    // SAFETY: forwards the game's own call.
    unsafe { DISABLE.get(&flight)(cap) }
}

unsafe extern "system" fn bind_texture(target: u32, texture: u32) {
    let flight = InFlight::enter();
    let texture = unit_texture(Shadow::unit(), target, texture);
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

macro_rules! draw_detour {
    ($name:ident, $original:ident, $type:ty, ($($arg:ident: $ty:ty),*), [$($counted:ident),*]) => {
        unsafe extern "system" fn $name($($arg: $ty),*) {
            let flight = InFlight::enter();
            // The draw's geometry: its call and arguments, the same for a mesh in both passes; and
            // its shape: the call and its counts, the same for a mesh streamed each pass.
            let call = stringify!($name).bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3));
            let mut geometry = call;
            $(geometry = (geometry ^ $arg as usize as u64).wrapping_mul(0x0000_0100_0000_01b3);)*
            let mut shape = call;
            $(shape = (shape ^ $counted as usize as u64).wrapping_mul(0x0000_0100_0000_01b3);)*
            if draw(geometry, shape) {
                // SAFETY: forwards the game's own call.
                unsafe { $original.get(&flight)($($arg),*) }
            }
            close_scissor(&flight);
        }
    };
}

draw_detour!(draw_arrays, DRAW_ARRAYS, DrawArraysFn, (mode: u32, first: i32, count: i32), [mode, count]);
draw_detour!(draw_elements, DRAW_ELEMENTS, DrawElementsFn, (mode: u32, count: i32, kind: u32, indices: *const c_void), [mode, count, kind]);
draw_detour!(draw_elements_base_vertex, DRAW_ELEMENTS_BASE_VERTEX, DrawElementsBaseVertexFn, (mode: u32, count: i32, kind: u32, indices: *const c_void, base: i32), [mode, count, kind]);
draw_detour!(draw_range_elements, DRAW_RANGE_ELEMENTS, DrawRangeElementsFn, (mode: u32, start: u32, end: u32, count: i32, kind: u32, indices: *const c_void), [mode, count, kind]);
draw_detour!(
    draw_range_elements_base_vertex,
    DRAW_RANGE_ELEMENTS_BASE_VERTEX,
    DrawRangeElementsBaseVertexFn,
    (mode: u32, start: u32, end: u32, count: i32, kind: u32, indices: *const c_void, base: i32), [mode, count, kind]
);
draw_detour!(draw_elements_instanced, DRAW_ELEMENTS_INSTANCED, DrawElementsInstancedFn, (mode: u32, count: i32, kind: u32, indices: *const c_void, instances: i32), [mode, count, kind, instances]);
draw_detour!(
    draw_elements_instanced_base_vertex,
    DRAW_ELEMENTS_INSTANCED_BASE_VERTEX,
    DrawElementsInstancedBaseVertexFn,
    (mode: u32, count: i32, kind: u32, indices: *const c_void, instances: i32, base: i32), [mode, count, kind, instances]
);
draw_detour!(draw_arrays_instanced, DRAW_ARRAYS_INSTANCED, DrawArraysInstancedFn, (mode: u32, first: i32, count: i32, instances: i32), [mode, count, instances]);
draw_detour!(draw_elements_indirect, DRAW_ELEMENTS_INDIRECT, DrawElementsIndirectFn, (mode: u32, kind: u32, indirect: *const c_void), [mode, kind, indirect]);
draw_detour!(multi_draw_elements_indirect, MULTI_DRAW_ELEMENTS_INDIRECT, MultiDrawElementsIndirectFn, (mode: u32, kind: u32, indirect: *const c_void, draws: i32, stride: i32), [mode, kind, indirect, draws, stride]);
draw_detour!(draw_arrays_indirect, DRAW_ARRAYS_INDIRECT, DrawArraysIndirectFn, (mode: u32, indirect: *const c_void), [mode, indirect]);
draw_detour!(multi_draw_arrays_indirect, MULTI_DRAW_ARRAYS_INDIRECT, MultiDrawArraysIndirectFn, (mode: u32, indirect: *const c_void, draws: i32, stride: i32), [mode, indirect, draws, stride]);
draw_detour!(multi_draw_elements, MULTI_DRAW_ELEMENTS, MultiDrawElementsFn, (mode: u32, counts: *const i32, kind: u32, indices: *const *const c_void, draws: i32), [mode, counts, kind, indices, draws]);
draw_detour!(multi_draw_elements_base_vertex, MULTI_DRAW_ELEMENTS_BASE_VERTEX, MultiDrawElementsBaseVertexFn, (mode: u32, counts: *const i32, kind: u32, indices: *const *const c_void, draws: i32, bases: *const i32), [mode, counts, kind, indices, draws, bases]);
draw_detour!(draw_elements_instanced_base_instance, DRAW_ELEMENTS_INSTANCED_BASE_INSTANCE, DrawElementsInstancedBaseInstanceFn, (mode: u32, count: i32, kind: u32, indices: *const c_void, instances: i32, first: u32), [mode, count, kind, instances]);
draw_detour!(draw_elements_instanced_base_vertex_base_instance, DRAW_ELEMENTS_INSTANCED_BASE_VERTEX_BASE_INSTANCE, DrawElementsInstancedBaseVertexBaseInstanceFn, (mode: u32, count: i32, kind: u32, indices: *const c_void, instances: i32, base: i32, first: u32), [mode, count, kind, instances]);
draw_detour!(draw_arrays_instanced_base_instance, DRAW_ARRAYS_INSTANCED_BASE_INSTANCE, DrawArraysInstancedBaseInstanceFn, (mode: u32, first: i32, count: i32, instances: i32, base: u32), [mode, count, instances]);
draw_detour!(multi_draw_elements_indirect_bindless, MULTI_DRAW_ELEMENTS_INDIRECT_BINDLESS, MultiDrawElementsIndirectBindlessFn, (mode: u32, kind: u32, indirect: *const c_void, draws: i32, stride: i32, buffers: i32), [mode, kind, indirect, draws, stride, buffers]);

#[cfg(test)]
mod tests {
    use super::*;

    fn rows(x: f32) -> Rows {
        [[1.0, 0.0, 0.0, x], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
    }

    #[test]
    fn a_mesh_streamed_at_another_offset_is_found_by_shape_once() {
        let mut carried = Carried::default();
        carried.keep(7, 100, 5, rows(0.2));
        assert_eq!(carried.find(7, 100, 5, &rows(0.2)), Some(Found::Geometry));
        carried.keep(7, 200, 9, rows(0.4));
        // The same shape at another offset: found once, as the left eye's draw stands for one.
        assert_eq!(carried.find(7, 201, 9, &rows(0.4)), Some(Found::Shape));
        assert_eq!(carried.find(7, 202, 9, &rows(0.4)), None);
        // Another place, or another program: not found.
        carried.keep(7, 300, 11, rows(0.4));
        assert_eq!(carried.find(7, 301, 11, &rows(0.9)), None);
        assert_eq!(carried.find(8, 301, 11, &rows(0.4)), None);
    }

    #[test]
    fn a_mesh_moved_a_little_between_the_passes_is_near_and_a_repeat_elsewhere_is_not() {
        let mut carried = Carried::default();
        carried.keep(7, 100, 5, rows(0.20));
        // Swayed between the passes: the same geometry, rows a little apart.
        let near = carried.nearest(7, 100, &rows(0.205)).unwrap();
        assert!(near > eye_matrix::AGREES * 10.0 && near <= MOVED_BETWEEN, "{near}");
        // The same mesh repeated metres away, or other geometry: not taken for it.
        assert!(carried.nearest(7, 100, &rows(3.0)).unwrap() > MOVED_BETWEEN);
        assert_eq!(carried.nearest(7, 101, &rows(0.205)), None);
    }
}

#[cfg(test)]
mod query_tests {
    use super::*;

    const IDENTITY: Rows = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]];

    #[test]
    fn a_right_query_is_answered_only_by_the_left_query_of_the_same_box() {
        // Three world boxes sharing the view's rows, each its own geometry.
        let mut left: Vec<LeftQuery> = [(10, 1), (11, 2), (12, 3)].map(|(id, geometry)| LeftQuery { id, rows: IDENTITY, geometry, taken: false }).to_vec();
        let mut cursor = 0;
        // The right eye's pass skips box 1 (culled for it): box 2 is answered by box 2's query.
        assert_eq!(left_query(&mut left, &mut cursor, &IDENTITY, &IDENTITY, 2), QueryMatch::Same(11));
        assert_eq!(left_query(&mut left, &mut cursor, &IDENTITY, &IDENTITY, 3), QueryMatch::Same(12));
        // A box the left eye's pass made no query for: rows alike, not answered.
        assert_eq!(left_query(&mut left, &mut cursor, &IDENTITY, &IDENTITY, 4), QueryMatch::RowsOnly);
        // Each left query answers once.
        assert_eq!(left_query(&mut left, &mut cursor, &IDENTITY, &IDENTITY, 2), QueryMatch::RowsOnly);
    }
}
