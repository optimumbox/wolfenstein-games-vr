//! The frame's own motion for DLSS: what the camera's motion (worked out from the depth) misses.
//!
//! - Skinned characters (faces talking, bodies walking): each draw of a converted character program
//!   ([`convert::motion`]) is drawn again into a motion target of ours, its program switched to
//!   write where each pixel was last frame. The joints it was drawn with are copied out of the
//!   engine's `matrices_ubo` for the next frame, with its `_va_` rows (the morph weights and where
//!   the model stood); the next frame's draw of the same mesh (the same program and geometry, the
//!   nearest model of them) reads them back. The motion is in UV units, as the depth pass's, with
//!   the window depth it was drawn at: DLSS's motion pass adds it where that is still what is seen.
//! - Particles (smoke, fire, sparks, explosions): each draw is drawn again writing how much of the
//!   pixel it covers into the target's blue channel (the largest share kept), which DLSS takes
//!   partly from the frame rather than its history: no trails smeared over what is behind them.
//!
//! The target is a two-layer array like the eye textures (layer 0 the left eye), tested against
//! the scene depth; at the present both layers go side by side, as the colour, to the upscaler
//! ([`pair`]). Without it (a frame not drawn layered) DLSS has the camera's motion alone.

use super::convert::motion::{self as shaders, Kind};
use super::gl_api::{self, Gl};
use monaka_producer::log;
use monaka_stereo::same_frame::gl::GlMotion;
use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU32, Ordering::Relaxed};
use std::time::{Duration, Instant};
use windows::Win32::Graphics::OpenGL::{glDepthFunc, glDepthMask, glGetBooleanv, glGetError, glGetIntegerv, glPopAttrib, glPushAttrib, glReadBuffer, glReadPixels};

const GL_TEXTURE_2D_ARRAY: u32 = 0x8C1A;
const GL_RGBA32F: u32 = 0x8814;
const GL_RGBA: u32 = 0x1908;
const GL_DRAW_FRAMEBUFFER: u32 = 0x8CA9;
const GL_FRAMEBUFFER_COMPLETE: u32 = 0x8CD5;
const GL_UNIFORM_BUFFER_BINDING: u32 = 0x8A28;
const GL_UNIFORM_BUFFER_START: u32 = 0x8A29;
const GL_UNIFORM_BUFFER_SIZE: u32 = 0x8A2A;
const GL_SHADER_STORAGE_BUFFER: u32 = 0x90D2;
const GL_BUFFER_SIZE: u32 = 0x8764;
const GL_BUFFER_MAPPED: u32 = 0x88BC;
const GL_BUFFER_ACCESS_FLAGS: u32 = 0x911F;
const GL_MAP_PERSISTENT_BIT: i32 = 0x0040;
/// The character draws whose GL errors are looked at (each step's), for the log.
const CHECKED_DRAWS: u64 = 4000;
const GL_BLEND: u32 = 0x0BE2;
const GL_COLOR_WRITEMASK: u32 = 0x0C23;
const GL_DEPTH_WRITEMASK: u32 = 0x0B72;
const GL_DEPTH_FUNC: u32 = 0x0B74;
const GL_LESS: u32 = 0x0201;
const GL_LEQUAL: u32 = 0x0203;
const GL_BLEND_SRC_RGB: u32 = 0x80C9;
const GL_BLEND_DST_RGB: u32 = 0x80C8;
const GL_BLEND_SRC_ALPHA: u32 = 0x80CB;
const GL_BLEND_DST_ALPHA: u32 = 0x80CA;
const GL_BLEND_EQUATION_RGB: u32 = 0x8009;
const GL_BLEND_EQUATION_ALPHA: u32 = 0x883D;
const GL_SRC_ALPHA: u32 = 0x0302;
const GL_ONE: u32 = 1;
const GL_MAX: u32 = 0x8008;
/// The joints of one draw: `matrices[768]`.
const JOINTS_BYTES: usize = 768 * 16;
/// Each frame's joint copies (one buffer per frame, two in turn).
const POOL_BYTES: usize = 16 << 20;
/// A copy's offset alignment (the storage buffer's, at most this on any driver of the engine's).
const ALIGN: usize = 256;
/// How far (world units) a mesh may have moved between frames and still be taken for the same one.
const SAME_MODEL: f32 = 48.0;
/// The target cleared to: no motion, no particles, a depth nothing is drawn at.
const CLEAR: [f32; 4] = [0.0, 0.0, 0.0, -1.0];
const LOG_EVERY: Duration = Duration::from_secs(30);

type CopyNamedBufferSubData = unsafe extern "system" fn(u32, u32, isize, isize, isize);
type CreateBuffers = unsafe extern "system" fn(i32, *mut u32);
type NamedBufferStorage = unsafe extern "system" fn(u32, isize, *const core::ffi::c_void, u32);
type DeleteBuffers = unsafe extern "system" fn(i32, *const u32);
type BindBufferRange = unsafe extern "system" fn(u32, u32, u32, isize, isize);
type GetNamedBufferParameteriv = unsafe extern "system" fn(u32, u32, *mut i32);
type ClearTexImage = unsafe extern "system" fn(u32, i32, u32, u32, *const core::ffi::c_void);
type GetBooleaniv = unsafe extern "system" fn(u32, u32, *mut u8);
type IsEnabledi = unsafe extern "system" fn(u32, u32) -> u8;
type Enablei = unsafe extern "system" fn(u32, u32);
type ColorMaski = unsafe extern "system" fn(u32, u8, u8, u8, u8);
type BlendFuncSeparatei = unsafe extern "system" fn(u32, u32, u32, u32, u32);
type BlendEquationSeparatei = unsafe extern "system" fn(u32, u32, u32);
type GetUniformBlockIndex = unsafe extern "system" fn(u32, *const u8) -> u32;
type GetTextureImage = unsafe extern "system" fn(u32, i32, u32, u32, i32, *mut core::ffi::c_void);

/// The entry points this pass uses beyond [`Gl`]'s.
struct Extra {
    copy_named_buffer_sub_data: CopyNamedBufferSubData,
    create_buffers: CreateBuffers,
    named_buffer_storage: NamedBufferStorage,
    delete_buffers: DeleteBuffers,
    bind_buffer_range: BindBufferRange,
    get_named_buffer_parameter_iv: GetNamedBufferParameteriv,
    clear_tex_image: ClearTexImage,
    get_boolean_i_v: GetBooleaniv,
    is_enabled_i: IsEnabledi,
    enable_i: Enablei,
    disable_i: Enablei,
    color_mask_i: ColorMaski,
    blend_func_separate_i: BlendFuncSeparatei,
    blend_equation_separate_i: BlendEquationSeparatei,
    get_uniform_block_index: GetUniformBlockIndex,
    get_texture_image: GetTextureImage,
}

fn entry<F: Copy>(name: &str) -> Result<F, String> {
    let address = gl_api::address(name).ok_or_else(|| name.to_owned())?;
    // SAFETY: the driver's function of that name, of the type its specification gives.
    Ok(unsafe { std::mem::transmute_copy(&address) })
}

impl Extra {
    fn load() -> Result<Self, String> {
        Ok(Self {
            copy_named_buffer_sub_data: entry("glCopyNamedBufferSubData")?,
            create_buffers: entry("glCreateBuffers")?,
            named_buffer_storage: entry("glNamedBufferStorage")?,
            delete_buffers: entry("glDeleteBuffers")?,
            bind_buffer_range: entry("glBindBufferRange")?,
            get_named_buffer_parameter_iv: entry("glGetNamedBufferParameteriv")?,
            clear_tex_image: entry("glClearTexImage")?,
            get_boolean_i_v: entry("glGetBooleani_v")?,
            is_enabled_i: entry("glIsEnabledi")?,
            enable_i: entry("glEnablei")?,
            disable_i: entry("glDisablei")?,
            color_mask_i: entry("glColorMaski")?,
            blend_func_separate_i: entry("glBlendFuncSeparatei")?,
            blend_equation_separate_i: entry("glBlendEquationSeparatei")?,
            get_uniform_block_index: entry("glGetUniformBlockIndex")?,
            get_texture_image: entry("glGetTextureImage")?,
        })
    }
}

/// Which of the target's framebuffers a draw goes into: both layers (the left eye's pass drawing
/// both eyes) or one eye's layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Target {
    Both,
    Eye(usize),
}

/// A draw of a converted program the engine just made, to draw again for the motion.
#[derive(Clone, Copy, Debug)]
pub struct Draw {
    pub program: u32,
    /// Its render program's name, for the log.
    pub name: &'static str,
    pub kind: Kind,
    pub target: Target,
    /// The scene depth's array (two layers) and the eye size.
    pub depth: u32,
    pub size: (i32, i32),
    pub geometry: u64,
    /// The framebuffer bound for the engine's draw (ours standing in for its own).
    pub bound: u32,
}

/// A character draw as it was: where its model stood, its joints' copy, its rows and switches.
struct Snapshot {
    origin: [f32; 3],
    joints: (usize, usize),
    va: Vec<[f32; 4]>,
    skinning: [f32; 4],
}

/// A converted program's locations for the motion draw.
struct Locations {
    switch: i32,
    /// For a character: each engine `_va_` row's, ours, and the joints block's binding.
    va: Vec<i32>,
    skinning: i32,
    previous_va: i32,
    previous_rows: i32,
    previous_skinning: i32,
    binding: u32,
}

/// The target and its framebuffers (both layers, each layer), and both eyes side by side.
struct Objects {
    array: u32,
    framebuffers: [u32; 3],
    pair: u32,
    pair_framebuffer: u32,
    depth: u32,
    size: (i32, i32),
}

#[derive(Default)]
struct Counts {
    /// Character draws skipped: their joints in a buffer mapped by the engine (not copyable).
    mapped: u64,
    characters: u64,
    followed: u64,
    particles: u64,
    frames: u64,
}

struct Motion {
    gl: Gl,
    extra: Extra,
    /// The context keeps the attribute stack (not a core profile): the engine's state is saved
    /// and put back around a motion draw by `glPushAttrib`, asking the driver nothing. A state
    /// query waits for the driver's own thread to catch up: one a particle (rain: hundreds a
    /// frame) held the render thread as long as the game's whole frame.
    push: bool,
    objects: Option<Objects>,
    pools: [u32; 2],
    turn: usize,
    cursor: usize,
    /// By program, geometry and eye (the layer drawn into; a draw for both eyes is the left's):
    /// each eye's projection of a mesh is its own, and so may its rows placing the model be.
    previous: HashMap<(u32, u64, usize), Vec<Snapshot>>,
    current: HashMap<(u32, u64, usize), Vec<Snapshot>>,
    locations: HashMap<u32, Option<Locations>>,
    pending: Option<Draw>,
    counts: Counts,
    /// By render program: character draws followed from the last frame, their models' moves
    /// between the frames (summed, largest; world units), and draws not followed.
    moves: HashMap<&'static str, (u64, f32, f32, u64)>,
    logged: Option<Instant>,
    /// Character draws looked at for GL errors so far, and the errors seen (step, code, times).
    checked: u64,
    errors: Vec<(&'static str, u32, u64)>,
    errors_logged: usize,
    /// A capture's folder the next frame's motion is written into ([`dump_into`]), with that
    /// frame's character draws described.
    dumping: Option<(std::path::PathBuf, Vec<String>)>,
    /// Frames ended, and by buffer whether it was mapped (mapped, access flags, whole size) when
    /// last asked, and the frame it was asked in.
    frame: u64,
    buffers: HashMap<u32, (i32, i32, i32, u64)>,
}

// ---- What the engine set, kept as it sets it ----
//
// A character's motion draw needs the joints' uniform buffer binding, whether that buffer is mapped,
// and the model's `_va_` rows. Asked of the driver, each question waits for the driver's own thread
// to catch up with everything the engine sent (eight or more a character: 45 us each, 2 ms a
// frame). So they are kept from the engine's own calls (`glBindBufferRange`/`Base`,
// `glUniform4fv`/`glProgramUniform4fv`), asked only the first time and on the checked draws
// ([`CHECKED_DRAWS`]), where a disagreement turns the kept values off for the run.

/// Uniform buffer binding points kept.
const BINDINGS: usize = 32;
/// The engine's uniform buffer bindings (buffer, start, size; size 0 the whole buffer), kept from
/// its binds once they are hooked.
static UBO: [(AtomicU32, AtomicI64, AtomicI64); BINDINGS] = [const { (AtomicU32::new(0), AtomicI64::new(0), AtomicI64::new(0)) }; BINDINGS];
static UBO_HOOKED: AtomicBool = AtomicBool::new(false);
/// The kept values disagreed with the driver's: asked of it from then on.
static KEPT_OFF: AtomicBool = AtomicBool::new(false);
/// The frames between looks at whether the joints' buffer is mapped.
const MAPPED_EVERY: u64 = 90;
/// By motion program: the locations of its `_va_` rows and skinning switch, with the values the
/// engine last uploaded to each.
static UNIFORMS: Mutex<Option<HashMap<u32, Vec<(i32, Option<[f32; 4]>)>>>> = Mutex::new(None);

/// The engine's binds of uniform buffers are hooked: their bindings are kept from them.
pub fn uniform_buffers_hooked() {
    UBO_HOOKED.store(true, Relaxed);
}

/// The engine bound `buffer` (from `start`, `size` bytes; 0 the whole) at uniform buffer `index`.
pub fn uniform_buffer_bound(index: u32, buffer: u32, start: isize, size: isize) {
    if let Some((b, s, n)) = UBO.get(index as usize) {
        b.store(buffer, Relaxed);
        s.store(start as i64, Relaxed);
        n.store(size as i64, Relaxed);
    }
}

/// The engine uploaded `count` vec4s at `location` of `program` (a motion program's).
pub fn uploaded(program: u32, location: i32, count: i32, values: *const f32) {
    if values.is_null() {
        return;
    }
    let Ok(mut held) = UNIFORMS.lock() else { return };
    let Some(watched) = held.as_mut().and_then(|m| m.get_mut(&program)) else { return };
    for (at, value) in watched.iter_mut() {
        let offset = *at - location;
        if offset >= 0 && offset < count {
            // SAFETY: the engine's buffer holds `count` vec4s.
            *value = Some(unsafe { *values.add(offset as usize * 4).cast::<[f32; 4]>() });
        }
    }
}

fn kept_off(what: &str) {
    if !KEPT_OFF.swap(true, Relaxed) {
        log!("DLSS motion: the {what} kept from the engine's calls disagreed with the driver's: asked of the driver from now on");
    }
}

/// A capture's folder for the next frame's motion ([`dump_into`]).
static DUMP: Mutex<Option<std::path::PathBuf>> = Mutex::new(None);

/// The next whole frame's motion written into `folder` (a capture's): `motion.bmp` (half size,
/// both eyes; red and green the motion's size across and down, blue the particles' share, magenta
/// a value not a number, white one infinite, black nothing drawn) and `motion.txt` (its figures and
/// each character draw).
pub fn dump_into(folder: std::path::PathBuf) {
    if let Ok(mut held) = DUMP.lock() {
        *held = Some(folder);
    }
}

/// A draw is pending ([`pending`]): the draw after it looks.
static PENDING: AtomicBool = AtomicBool::new(false);

static MOTION: Mutex<Option<Result<Motion, ()>>> = Mutex::new(None);
static PAIR: Mutex<Option<GlMotion>> = Mutex::new(None);

/// The motion for the frame being presented, both eyes side by side ([`GlMotion`]): what the
/// upscaler adds to the camera's motion.
pub fn pair() -> Option<GlMotion> {
    *PAIR.lock().ok()?
}

fn with<T>(f: impl FnOnce(&mut Motion) -> Option<T>) -> Option<T> {
    let mut guard = MOTION.lock().ok()?;
    if guard.is_none() {
        *guard = Some(Motion::new().map_err(|why| log!("DLSS motion: off: {why}")));
    }
    f(guard.as_mut()?.as_mut().ok()?)
}

/// A draw the engine is about to make that draws for the motion: drawn again after it ([`after_draw`]).
/// DLSS is upscaling this run ([`wanted`]): without it nothing reads the motion, and no draw is
/// made again for it.
static WANTED: AtomicBool = AtomicBool::new(false);

/// DLSS upscales this run: the characters' and particles' motion is drawn for it.
pub fn wanted() {
    WANTED.store(true, Relaxed);
}

/// Whether the motion is drawn at all (DLSS upscaling).
pub fn is_wanted() -> bool {
    WANTED.load(Relaxed)
}

pub fn pending(draw: Option<Draw>) {
    if draw.is_none() && !PENDING.load(Relaxed) {
        return;
    }
    PENDING.store(draw.is_some(), Relaxed);
    with(|m| {
        m.pending = draw;
        Some(())
    });
}

/// After the engine's draw: the pending one drawn again for the motion by `redraw` (the engine's
/// own call, once more), with the program switched and our target bound.
pub fn after_draw(redraw: impl FnOnce()) {
    if !PENDING.swap(false, Relaxed) {
        return;
    }
    with(|m| {
        let draw = m.pending.take()?;
        m.run(draw, redraw);
        Some(())
    });
}

/// At the present: both layers side by side for the upscaler (when both eyes were drawn), the
/// target cleared and this frame's snapshots made the previous ones.
pub fn frame_end(both_eyes: bool) {
    let pair = with(|m| m.end(both_eyes));
    if let Ok(mut held) = PAIR.lock() {
        *held = pair;
    }
}

/// Every object of ours deleted (the context current); made again at the next draw.
pub fn stop() {
    if let Ok(mut guard) = MOTION.lock()
        && let Some(Ok(m)) = guard.take()
    {
        m.release();
    }
    if let Ok(mut held) = PAIR.lock() {
        *held = None;
    }
}

impl Motion {
    fn new() -> Result<Self, String> {
        let gl = Gl::load().map_err(str::to_owned)?;
        let extra = Extra::load().map_err(|missing| format!("the driver has no {missing}"))?;
        let mut pools = [0u32; 2];
        // SAFETY: new buffers of ours, storage only the GPU writes.
        unsafe {
            (extra.create_buffers)(2, pools.as_mut_ptr());
            for &pool in &pools {
                (extra.named_buffer_storage)(pool, POOL_BYTES as isize, std::ptr::null(), 0);
            }
        }
        let mut profile = 0;
        // SAFETY: a plain query, once.
        unsafe { glGetIntegerv(GL_CONTEXT_PROFILE_MASK, &mut profile) };
        let push = profile & GL_CONTEXT_CORE_PROFILE_BIT == 0;
        log!(
            "DLSS motion: the engine's state around each motion draw {} (context profile mask {profile:#x})",
            if push { "saved and put back on the attribute stack, nothing queried" } else { "queried and set again (a core profile)" }
        );
        Ok(Self {
            gl,
            extra,
            push,
            objects: None,
            pools,
            turn: 0,
            cursor: 0,
            previous: HashMap::new(),
            current: HashMap::new(),
            locations: HashMap::new(),
            pending: None,
            counts: Counts::default(),
            moves: HashMap::new(),
            logged: None,
            checked: 0,
            errors: Vec::new(),
            errors_logged: 0,
            dumping: None,
            frame: 0,
            buffers: HashMap::new(),
        })
    }

    fn release(self) {
        if let Some(objects) = &self.objects {
            Self::release_objects(&self.gl, objects);
        }
        // SAFETY: our own buffers.
        unsafe { (self.extra.delete_buffers)(2, self.pools.as_ptr()) };
    }

    fn release_objects(gl: &Gl, o: &Objects) {
        let framebuffers = [o.framebuffers[0], o.framebuffers[1], o.framebuffers[2], o.pair_framebuffer];
        // SAFETY: our own objects.
        unsafe {
            (gl.delete_framebuffers)(4, framebuffers.as_ptr());
            (gl.delete_textures)(2, [o.array, o.pair].as_ptr());
        }
    }

    /// The target for the scene depth array `depth` at eye size `size`, made (again) as needed.
    fn objects(&mut self, depth: u32, size: (i32, i32)) -> Option<&Objects> {
        if self.objects.as_ref().is_some_and(|o| o.depth != depth || o.size != size) {
            let old = self.objects.take().expect("checked");
            Self::release_objects(&self.gl, &old);
        }
        if self.objects.is_none() {
            let gl = &self.gl;
            let (mut array, mut pair) = (0, 0);
            let mut framebuffers = [0u32; 3];
            let mut pair_framebuffer = 0;
            let mut complete = true;
            // SAFETY: new objects of ours; the scene depth array attached, not written.
            unsafe {
                (gl.create_textures)(GL_TEXTURE_2D_ARRAY, 1, &mut array);
                (gl.texture_storage_3d)(array, 1, GL_RGBA32F, size.0, size.1, 2);
                (gl.create_textures)(GL_TEXTURE_2D_ARRAY, 1, &mut pair);
                (gl.texture_storage_3d)(pair, 1, GL_RGBA32F, size.0 * 2, size.1, 1);
                (gl.create_framebuffers)(3, framebuffers.as_mut_ptr());
                (gl.create_framebuffers)(1, &mut pair_framebuffer);
                (gl.named_framebuffer_texture)(framebuffers[0], gl_api::GL_COLOR_ATTACHMENT0, array, 0);
                (gl.named_framebuffer_texture)(framebuffers[0], gl_api::GL_DEPTH_ATTACHMENT, depth, 0);
                for layer in 0..2 {
                    (gl.named_framebuffer_texture_layer)(framebuffers[1 + layer], gl_api::GL_COLOR_ATTACHMENT0, array, 0, layer as i32);
                    (gl.named_framebuffer_texture_layer)(framebuffers[1 + layer], gl_api::GL_DEPTH_ATTACHMENT, depth, 0, layer as i32);
                }
                (gl.named_framebuffer_texture_layer)(pair_framebuffer, gl_api::GL_COLOR_ATTACHMENT0, pair, 0, 0);
                for &framebuffer in framebuffers.iter().chain([&pair_framebuffer]) {
                    (gl.named_framebuffer_draw_buffers)(framebuffer, 1, &gl_api::GL_COLOR_ATTACHMENT0);
                    (gl.named_framebuffer_read_buffer)(framebuffer, gl_api::GL_COLOR_ATTACHMENT0);
                    complete &= (gl.check_named_framebuffer_status)(framebuffer, GL_DRAW_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE;
                }
                (self.extra.clear_tex_image)(array, 0, GL_RGBA, gl_api::GL_FLOAT, CLEAR.as_ptr().cast());
            }
            let objects = Objects { array, framebuffers, pair, pair_framebuffer, depth, size };
            if !complete {
                log!("DLSS motion: off: its framebuffers on the scene depth are incomplete");
                Self::release_objects(gl, &objects);
                return None;
            }
            log!("DLSS motion: characters' and particles' own motion drawn for DLSS, {}x{} per eye", size.0, size.1);
            self.objects = Some(objects);
        }
        self.objects.as_ref()
    }

    fn locations(&mut self, program: u32, kind: Kind) -> Option<&Locations> {
        let (gl, extra) = (&self.gl, &self.extra);
        self.locations
            .entry(program)
            .or_insert_with(|| {
                // SAFETY: plain queries with NUL-terminated names.
                let at = |name: &str| unsafe { (gl.get_uniform_location)(program, gl_api::c_name(name).as_ptr()) };
                let switch = at(shaders::SWITCH);
                if switch < 0 {
                    return None;
                }
                let Kind::Character { va, .. } = kind else {
                    return Some(Locations { switch, va: Vec::new(), skinning: -1, previous_va: -1, previous_rows: -1, previous_skinning: -1, binding: 0 });
                };
                let rows: Vec<i32> = (0..va).map(|i| at(&format!("_va_[{i}]"))).collect();
                let (skinning, previous_va, previous_rows, previous_skinning) = (at("hardwareSkinning"), at(&format!("{}[0]", shaders::PREVIOUS_VA)), at(&format!("{}[0]", shaders::PREVIOUS_ROWS)), at(shaders::PREVIOUS_SKINNING));
                if rows.iter().chain([&skinning, &previous_va, &previous_rows, &previous_skinning]).any(|&l| l < 0) {
                    return None;
                }
                if let Ok(mut held) = UNIFORMS.lock() {
                    held.get_or_insert_with(HashMap::new).insert(program, rows.iter().chain([&skinning]).map(|&l| (l, None)).collect());
                }
                // SAFETY: a plain query with a NUL-terminated name.
                let block = unsafe { (extra.get_uniform_block_index)(program, gl_api::c_name("matrices_ubo").as_ptr()) };
                if block == u32::MAX {
                    return None;
                }
                let mut binding = 0;
                // SAFETY: a plain query of the joints' block into a local.
                unsafe { (gl.get_active_uniform_block_iv)(program, block, gl_api::GL_UNIFORM_BLOCK_BINDING, &mut binding) };
                Some(Locations { switch, va: rows, skinning, previous_va, previous_rows, previous_skinning, binding: binding as u32 })
            })
            .as_ref()
    }

    fn run(&mut self, draw: Draw, redraw: impl FnOnce()) {
        let Some(objects) = self.objects(draw.depth, draw.size) else { return };
        let framebuffer = match draw.target {
            Target::Both => objects.framebuffers[0],
            Target::Eye(layer) => objects.framebuffers[1 + layer.min(1)],
        };
        if self.locations(draw.program, draw.kind).is_none() {
            return;
        }
        match draw.kind {
            Kind::Character { rows, world, .. } => self.character(draw, rows, world, framebuffer, redraw),
            Kind::Particle => self.particle(draw, framebuffer, redraw),
        }
    }

    fn character(&mut self, draw: Draw, rows: [usize; 4], world: [usize; 3], framebuffer: u32, redraw: impl FnOnce()) {
        let (gl, extra) = (&self.gl, &self.extra);
        let Some(Some(at)) = self.locations.get(&draw.program) else { return };
        // The joints the engine just drew with, copied for the next frame.
        let checking = self.checked < CHECKED_DRAWS;
        let keeping = UBO_HOOKED.load(Relaxed) && !KEPT_OFF.load(Relaxed);
        let kept = UBO.get(at.binding as usize).filter(|_| keeping).map(|(b, s, n)| (b.load(Relaxed) as i32, s.load(Relaxed), n.load(Relaxed)));
        let (mut buffer, mut start, mut size) = match kept {
            Some((b, s, n)) if b > 0 => (b, s, n),
            _ => (0, 0, 0),
        };
        if buffer <= 0 || checking {
            let (mut b, mut s, mut n) = (0i32, 0i32, 0i32);
            // SAFETY: state queries into locals.
            unsafe {
                (gl.get_integer_i_v)(GL_UNIFORM_BUFFER_BINDING, at.binding, &mut b);
                (gl.get_integer_i_v)(GL_UNIFORM_BUFFER_START, at.binding, &mut s);
                (gl.get_integer_i_v)(GL_UNIFORM_BUFFER_SIZE, at.binding, &mut n);
            }
            let (s, n) = (i64::from(s), i64::from(n));
            if buffer > 0 && (buffer, start, size) != (b, s, n) {
                kept_off("joints' buffer binding");
            }
            (buffer, start, size) = (b, s, n);
        }
        if buffer <= 0 || start < 0 {
            return;
        }
        let frame = self.frame;
        let known = self.buffers.get(&(buffer as u32)).copied().filter(|&(_, _, _, at)| !checking && frame.saturating_sub(at) < MAPPED_EVERY);
        let (mapped, access, whole) = match known {
            Some((mapped, access, whole, _)) => (mapped, access, whole),
            None => {
                let (mut mapped, mut access, mut whole) = (0, 0, 0);
                // SAFETY: plain queries into locals.
                unsafe {
                    (extra.get_named_buffer_parameter_iv)(buffer as u32, GL_BUFFER_MAPPED, &mut mapped);
                    (extra.get_named_buffer_parameter_iv)(buffer as u32, GL_BUFFER_ACCESS_FLAGS, &mut access);
                    (extra.get_named_buffer_parameter_iv)(buffer as u32, GL_BUFFER_SIZE, &mut whole);
                }
                self.buffers.insert(buffer as u32, (mapped, access, whole, frame));
                (mapped, access, whole)
            }
        };
        if mapped != 0 && access & GL_MAP_PERSISTENT_BIT == 0 {
            self.counts.mapped += 1;
            return;
        }
        if checking {
            self.checked += 1;
            // What the engine raised before (it does not look): cleared, so ours are told apart.
            // SAFETY: a plain query.
            unsafe { glGetError() };
        }
        let mut size = size as usize;
        if size == 0 {
            size = (whole as usize).saturating_sub(start as usize);
        }
        let size = size.min(JOINTS_BYTES);
        let offset = self.cursor.next_multiple_of(ALIGN);
        if size == 0 || offset + size > POOL_BYTES {
            return;
        }
        let pool = self.pools[self.turn];
        // SAFETY: a copy from the engine's bound joints into our buffer, in range of both.
        unsafe { (extra.copy_named_buffer_sub_data)(buffer as u32, pool, start as isize, offset as isize, size as isize) };
        if checking && let Some(code) = error() {
            note(&mut self.errors, "copying the joints", code);
            return;
        }
        self.cursor = offset + size;
        let mut known: Vec<(i32, Option<[f32; 4]>)> = if KEPT_OFF.load(Relaxed) {
            Vec::new()
        } else {
            UNIFORMS.lock().ok().and_then(|held| held.as_ref().and_then(|m| m.get(&draw.program).cloned())).unwrap_or_default()
        };
        let mut asked = false;
        let mut read = |location: i32| {
            let kept = known.iter().find(|(l, _)| *l == location).and_then(|(_, v)| *v);
            if let Some(v) = kept.filter(|_| !checking) {
                return v;
            }
            let mut v = [0.0f32; 4];
            // SAFETY: a vec4 uniform's value into a local of its size.
            unsafe { (gl.get_uniform_fv)(draw.program, location, v.as_mut_ptr()) };
            if kept.is_some_and(|k| k != v) {
                kept_off("model's rows");
            }
            if let Some(slot) = known.iter_mut().find(|(l, _)| *l == location) {
                slot.1 = Some(v);
                asked = true;
            }
            v
        };
        let va: Vec<[f32; 4]> = at.va.iter().map(|&l| read(l)).collect();
        let skinning = read(at.skinning);
        if asked && let Ok(mut held) = UNIFORMS.lock()
            && let Some(watched) = held.as_mut().and_then(|m| m.get_mut(&draw.program))
        {
            for (slot, (_, v)) in watched.iter_mut().zip(&known) {
                if slot.1.is_none() {
                    slot.1 = *v;
                }
            }
        }
        let origin = [va[world[0]][3], va[world[1]][3], va[world[2]][3]];
        let now = Snapshot { origin, joints: (offset, size), va, skinning };
        let eye = match draw.target {
            Target::Both => 0,
            Target::Eye(layer) => layer.min(1),
        };
        let key = (draw.program, draw.geometry, eye);
        let before = self.previous.get(&key).and_then(|seen| {
            seen.iter()
                .map(|s| (s, (0..3).map(|i| (s.origin[i] - origin[i]).powi(2)).sum::<f32>().sqrt()))
                .filter(|(_, d)| *d <= SAME_MODEL)
                .min_by(|a, b| a.1.total_cmp(&b.1))
                .map(|(s, _)| s)
        });
        self.counts.characters += 1;
        self.counts.followed += u64::from(before.is_some());
        let moved = before.map(|s| (0..3).map(|i| (s.origin[i] - origin[i]).powi(2)).sum::<f32>().sqrt());
        let entry = self.moves.entry(draw.name).or_default();
        match moved {
            Some(d) => {
                entry.0 += 1;
                entry.1 += d;
                entry.2 = entry.2.max(d);
            }
            None => entry.3 += 1,
        }
        let (was, joints_pool) = match before {
            Some(s) => (s, self.pools[1 - self.turn]),
            None => (&now, pool),
        };
        let projection: [[f32; 4]; 4] = rows.map(|r| now.va[r]);
        let placed = |s: &Snapshot| -> [[f32; 4]; 4] { [s.va[world[0]], s.va[world[1]], s.va[world[2]], [0.0, 0.0, 0.0, 1.0]] };
        // This frame's projection of where the model stood last frame: the projection, back to the
        // model as it stands now, out to the world as it stood.
        let previous_rows = inverse_affine(&placed(&now)).map_or(projection, |back| multiply(&multiply(&projection, &back), &placed(was)));
        let switch_on = [1.0f32, 0.0, 0.0, 0.0];
        let off = [0.0f32; 4];
        // SAFETY: our uniforms on the engine's converted program (its own values untouched), and
        // our joints buffer bound where the program reads it.
        unsafe {
            (gl.program_uniform_4fv)(draw.program, at.previous_va, was.va.len() as i32, was.va.as_ptr().cast());
            (gl.program_uniform_4fv)(draw.program, at.previous_rows, 4, previous_rows.as_ptr().cast());
            (gl.program_uniform_4fv)(draw.program, at.previous_skinning, 1, was.skinning.as_ptr());
            (gl.program_uniform_4fv)(draw.program, at.switch, 1, switch_on.as_ptr());
            (extra.bind_buffer_range)(GL_SHADER_STORAGE_BUFFER, shaders::JOINTS_BINDING, joints_pool, was.joints.0 as isize, was.joints.1 as isize);
        }
        let setup = if checking { error() } else { None };
        let saved = Saved::take(gl, extra, false, self.push.then_some(draw.bound));
        // SAFETY: state for our draw into our target, put back after.
        unsafe {
            (gl.bind_framebuffer)(GL_DRAW_FRAMEBUFFER, framebuffer);
            (extra.disable_i)(GL_BLEND, 0);
            (extra.color_mask_i)(0, 1, 1, 0, 1);
            glDepthMask(0);
            // Unknown when pushed: the engine's characters draw with LESS or LEQUAL (or EQUAL over
            // their depth), all of which LEQUAL passes against the depth just drawn.
            if saved.pushed || saved.depth_func == GL_LESS {
                glDepthFunc(GL_LEQUAL);
            }
        }
        let state = if checking { error() } else { None };
        redraw();
        let drawn = if checking { error() } else { None };
        saved.put_back(gl, extra);
        // SAFETY: our switch off again.
        unsafe { (gl.program_uniform_4fv)(draw.program, at.switch, 1, off.as_ptr()) };
        let after = if checking { error() } else { None };
        for (step, code) in [("setting the uniforms and joints", setup), ("binding our target", state), ("drawing again", drawn), ("putting the state back", after)] {
            if let Some(code) = code {
                note(&mut self.errors, step, code);
            }
        }
        if let Some((_, lines)) = self.dumping.as_mut() {
            let apart = (0..4).flat_map(|r| (0..4).map(move |c| (r, c))).map(|(r, c)| (previous_rows[r][c] - projection[r][c]).abs()).fold(0.0f32, f32::max);
            lines.push(format!(
                "{} (program {}) into {:?}: followed {} moved {:?} origin {:?}; last frame's projection apart from this one's by at most {apart:.5}; skinning {:?} then {:?}; joints {} bytes",
                draw.name,
                draw.program,
                draw.target,
                before.is_some(),
                moved,
                now.origin,
                now.skinning,
                was.skinning,
                was.joints.1
            ));
        }
        self.current.entry(key).or_default().push(now);
    }

    fn particle(&mut self, draw: Draw, framebuffer: u32, redraw: impl FnOnce()) {
        let (gl, extra) = (&self.gl, &self.extra);
        let Some(Some(at)) = self.locations.get(&draw.program) else { return };
        let share = if self.push {
            // Unasked: additive particles (their programs named so) by their colour, the rest
            // blended by their alpha.
            if draw.name.contains("add") { 3.0f32 } else { 2.0 }
        } else {
            // SAFETY: a plain state query.
            if unsafe { (extra.is_enabled_i)(GL_BLEND, 0) } == 0 {
                return;
            }
            0.0
        };
        let saved = Saved::take(gl, extra, true, self.push.then_some(draw.bound));
        let share = if saved.pushed { share } else if saved.blend[0] == GL_SRC_ALPHA { 2.0f32 } else { 3.0 };
        let switch = [share, 0.0, 0.0, 0.0];
        let off = [0.0f32; 4];
        // SAFETY: our switch on the engine's converted program, and state for our draw into our
        // target, put back after.
        unsafe {
            (gl.program_uniform_4fv)(draw.program, at.switch, 1, switch.as_ptr());
            (gl.bind_framebuffer)(GL_DRAW_FRAMEBUFFER, framebuffer);
            (extra.enable_i)(GL_BLEND, 0);
            (extra.blend_func_separate_i)(0, GL_ONE, GL_ONE, GL_ONE, GL_ONE);
            (extra.blend_equation_separate_i)(0, GL_MAX, GL_MAX);
            (extra.color_mask_i)(0, 0, 0, 1, 0);
            glDepthMask(0);
        }
        redraw();
        saved.put_back(gl, extra);
        // SAFETY: our switch off again.
        unsafe { (gl.program_uniform_4fv)(draw.program, at.switch, 1, off.as_ptr()) };
        self.counts.particles += 1;
    }

    fn end(&mut self, both_eyes: bool) -> Option<GlMotion> {
        self.pending = None;
        let dumped = if both_eyes { self.dumping.take() } else { None };
        if self.dumping.is_none() && let Some(folder) = DUMP.lock().ok().and_then(|mut d| d.take()) {
            self.dumping = Some((folder, Vec::new()));
        }
        self.previous = std::mem::take(&mut self.current);
        self.turn = 1 - self.turn;
        self.cursor = 0;
        let objects = self.objects.as_ref()?;
        let (w, h) = objects.size;
        let gl = &self.gl;
        // SAFETY: copies between our own RGBA32F textures (each layer into its half), then the
        // target cleared for the next frame.
        unsafe {
            if both_eyes {
                for layer in 0..2 {
                    (gl.copy_image_sub_data)(objects.array, GL_TEXTURE_2D_ARRAY, 0, 0, 0, layer, objects.pair, GL_TEXTURE_2D_ARRAY, 0, layer * w, 0, 0, w, h, 1);
                }
            }
            (self.extra.clear_tex_image)(objects.array, 0, GL_RGBA, gl_api::GL_FLOAT, CLEAR.as_ptr().cast());
        }
        if let Some((folder, lines)) = dumped {
            let mut pixels = vec![0.0f32; (w * 2 * h * 4) as usize];
            // SAFETY: our own texture read back into a buffer of its size.
            unsafe { (self.extra.get_texture_image)(objects.pair, 0, GL_RGBA, gl_api::GL_FLOAT, (pixels.len() * 4) as i32, pixels.as_mut_ptr().cast()) };
            // The frame DLSS is given (the back buffer, both eyes side by side), before it upscales.
            let mut colour = vec![0u8; (w * 2 * h * 4) as usize];
            let (mut read, mut buffer) = (0, 0);
            // SAFETY: the read binding and buffer saved, the back buffer read into a buffer of its
            // size, both put back.
            unsafe {
                glGetIntegerv(0x8CAA, &mut read);
                glGetIntegerv(0x0C02, &mut buffer);
                (gl.bind_framebuffer)(0x8CA8, 0);
                glReadBuffer(0x0405);
                glReadPixels(0, 0, w * 2, h, GL_RGBA, 0x1401, colour.as_mut_ptr().cast());
                (gl.bind_framebuffer)(0x8CA8, read as u32);
                glReadBuffer(buffer as u32);
            }
            std::thread::spawn(move || {
                write_dump(&folder, &pixels, (w * 2) as usize, h as usize, &lines);
                write_colour(&folder, &colour, (w * 2) as usize, h as usize);
            });
            log!("DLSS motion: the frame's motion written beside the capture");
        }
        self.counts.frames += 1;
        self.frame += 1;
        let now = Instant::now();
        if self.logged.is_none_or(|at| now - at >= LOG_EVERY) && self.counts.frames > 0 {
            let c = &self.counts;
            let per = |n: u64| n as f64 / c.frames as f64;
            log!(
                "DLSS motion: per frame {:.1} character draws ({:.0}% followed from the last frame), {:.1} particle draws",
                per(c.characters),
                if c.characters > 0 { 100.0 * c.followed as f64 / c.characters as f64 } else { 0.0 },
                per(c.particles)
            );
            let mut moves: Vec<_> = self.moves.drain().collect();
            moves.sort_unstable_by(|a, b| (b.1.0 + b.1.3).cmp(&(a.1.0 + a.1.3)));
            let moves: Vec<String> = moves.iter().take(10).map(|(name, (n, sum, most, lost))| format!("{name} {:.1}/frame mean {:.3} most {:.3} unfollowed {:.1}/frame", *n as f64 / c.frames as f64, if *n > 0 { sum / *n as f32 } else { 0.0 }, most, *lost as f64 / c.frames as f64)).collect();
            log!("DLSS motion: characters' models moved between frames (world units), by program: {}", moves.join("; "));
            if c.mapped > 0 {
                log!("DLSS motion: {:.1} character draws per frame skipped, their joints in a buffer the engine had mapped", per(c.mapped));
            }
            if self.errors.len() > self.errors_logged {
                let errors: Vec<String> = self.errors.iter().map(|(step, code, n)| format!("{code:#x} {step} ({n}x)")).collect();
                log!("DLSS motion: GL errors in the character draws (of the first {}): {}", self.checked, errors.join(", "));
                self.errors_logged = self.errors.len();
            }
            self.counts = Counts::default();
            self.logged = Some(now);
        }
        both_eyes.then(|| GlMotion { framebuffer: objects.pair_framebuffer, width: (w * 2) as u32, height: h as u32 })
    }
}

/// `colour.bmp`: the frame DLSS was given (RGBA bytes, bottom row first), whole.
fn write_colour(folder: &std::path::Path, pixels: &[u8], width: usize, height: usize) {
    let row = (width * 3).next_multiple_of(4);
    let size = 54 + row * height;
    let mut bmp = Vec::with_capacity(size);
    bmp.extend_from_slice(b"BM");
    bmp.extend_from_slice(&(size as u32).to_le_bytes());
    bmp.extend_from_slice(&[0; 4]);
    bmp.extend_from_slice(&54u32.to_le_bytes());
    bmp.extend_from_slice(&40u32.to_le_bytes());
    bmp.extend_from_slice(&(width as i32).to_le_bytes());
    bmp.extend_from_slice(&(height as i32).to_le_bytes());
    bmp.extend_from_slice(&1u16.to_le_bytes());
    bmp.extend_from_slice(&24u16.to_le_bytes());
    bmp.extend_from_slice(&[0; 24]);
    for y in 0..height {
        for p in pixels[y * width * 4..][..width * 4].chunks_exact(4) {
            bmp.extend_from_slice(&[p[2], p[1], p[0]]);
        }
        bmp.resize(54 + row * (y + 1), 0);
    }
    if let Err(e) = std::fs::write(folder.join("colour.bmp"), bmp) {
        log!("DLSS motion: colour.bmp not written: {e}");
    }
}

/// [`dump_into`]'s files, from the pair's pixels (RGBA floats, bottom row first).
fn write_dump(folder: &std::path::Path, pixels: &[f32], width: usize, height: usize, lines: &[String]) {
    let (half_w, half_h) = (width / 2, height / 2);
    let row = (half_w * 3).next_multiple_of(4);
    let mut bmp = Vec::with_capacity(54 + row * half_h);
    let size = (54 + row * half_h) as u32;
    bmp.extend_from_slice(b"BM");
    bmp.extend_from_slice(&size.to_le_bytes());
    bmp.extend_from_slice(&[0; 4]);
    bmp.extend_from_slice(&54u32.to_le_bytes());
    bmp.extend_from_slice(&40u32.to_le_bytes());
    bmp.extend_from_slice(&(half_w as i32).to_le_bytes());
    bmp.extend_from_slice(&(half_h as i32).to_le_bytes());
    bmp.extend_from_slice(&1u16.to_le_bytes());
    bmp.extend_from_slice(&24u16.to_le_bytes());
    bmp.extend_from_slice(&[0; 24]);
    // Figures per eye: drawn, not a number, infinite, the largest motion, how many over 0.01 UV.
    let mut figures = [(0u64, 0u64, 0u64, 0.0f32, 0u64, 0u64); 2];
    for y in 0..height {
        for x in 0..width {
            let p = &pixels[(y * width + x) * 4..][..4];
            let f = &mut figures[usize::from(x >= width / 2)];
            if p.iter().any(|v| v.is_nan()) {
                f.1 += 1;
            } else if p.iter().any(|v| v.is_infinite()) {
                f.2 += 1;
            } else if p[3] >= 0.0 {
                f.0 += 1;
                let m = p[0].abs().max(p[1].abs());
                f.3 = f.3.max(m);
                f.4 += u64::from(m > 0.01);
            }
            f.5 += u64::from(p[2] > 0.0);
        }
    }
    for y in 0..half_h {
        for x in 0..half_w {
            let p = &pixels[(y * 2 * width + x * 2) * 4..][..4];
            let level = |v: f32| (v.abs() * 2550.0).sqrt().min(255.0) as u8;
            let [r, g, b] = if p.iter().any(|v| v.is_nan()) {
                [255, 0, 255]
            } else if p.iter().any(|v| v.is_infinite()) {
                [255, 255, 255]
            } else if p[3] < 0.0 {
                [0, 0, (p[2].clamp(0.0, 1.0) * 255.0) as u8]
            } else {
                [level(p[0]).max(24), level(p[1]).max(24), (p[2].clamp(0.0, 1.0) * 255.0) as u8]
            };
            bmp.extend_from_slice(&[b, g, r]);
        }
        bmp.resize(54 + row * (y + 1), 0);
    }
    let mut text = String::from("The frame's motion as DLSS was given it (both eyes, left first): per eye, pixels drawn by a character, not a number, infinite, the largest motion (UV), pixels moving over 0.01 UV, pixels with particles.\n");
    for (eye, f) in ["left", "right"].iter().zip(figures) {
        text.push_str(&format!("{eye}: drawn {} nan {} infinite {} largest {:.4} over 0.01: {} particles {}\n", f.0, f.1, f.2, f.3, f.4, f.5));
    }
    text.push_str("\nThe character draws:\n");
    for line in lines {
        text.push_str(line);
        text.push('\n');
    }
    for (name, bytes) in [("motion.bmp", bmp), ("motion.txt", text.into_bytes())] {
        if let Err(e) = std::fs::write(folder.join(name), bytes) {
            log!("DLSS motion: {name} not written: {e}");
        }
    }
}

/// The GL error raised since the last look, if any.
fn error() -> Option<u32> {
    // SAFETY: a plain query.
    let code = unsafe { glGetError() };
    (code != 0).then_some(code)
}

fn note(errors: &mut Vec<(&'static str, u32, u64)>, step: &'static str, code: u32) {
    match errors.iter_mut().find(|(s, c, _)| *s == step && *c == code) {
        Some(seen) => seen.2 += 1,
        None => errors.push((step, code, 1)),
    }
}

const GL_CONTEXT_PROFILE_MASK: u32 = 0x9126;
const GL_CONTEXT_CORE_PROFILE_BIT: i32 = 0x1;
const GL_COLOR_BUFFER_BIT: u32 = 0x4000;
const GL_DEPTH_BUFFER_BIT: u32 = 0x0100;

/// The engine's state a motion draw changes, as it was.
struct Saved {
    /// On the attribute stack (blending, the masks, the depth test): the rest unknown.
    pushed: bool,
    framebuffer: i32,
    blending: bool,
    mask: [u8; 4],
    depth_mask: u8,
    depth_func: u32,
    /// With `blend`: the first draw buffer's factors (source, destination: colour, alpha) and
    /// equations; else empty.
    blend: [u32; 4],
    equations: [u32; 2],
}

impl Saved {
    /// With `pushed` (the framebuffer bound): the state pushed, nothing queried.
    fn take(gl: &Gl, extra: &Extra, blend: bool, pushed: Option<u32>) -> Self {
        if let Some(framebuffer) = pushed {
            // SAFETY: the colour (blending, colour masks) and depth (test, function, mask)
            // attributes pushed, popped in `put_back`.
            unsafe { glPushAttrib(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT) };
            return Self { pushed: true, framebuffer: framebuffer as i32, blending: false, mask: [1; 4], depth_mask: 1, depth_func: GL_LESS, blend: [0; 4], equations: [0; 2] };
        }
        let mut s = Self { pushed: false, framebuffer: 0, blending: false, mask: [1; 4], depth_mask: 1, depth_func: GL_LESS, blend: [0; 4], equations: [0; 2] };
        let mut func = 0;
        // SAFETY: state queries into locals.
        unsafe {
            glGetIntegerv(gl_api::GL_DRAW_FRAMEBUFFER_BINDING, &mut s.framebuffer);
            s.blending = (extra.is_enabled_i)(GL_BLEND, 0) != 0;
            (extra.get_boolean_i_v)(GL_COLOR_WRITEMASK, 0, s.mask.as_mut_ptr());
            glGetBooleanv(GL_DEPTH_WRITEMASK, &mut s.depth_mask);
            glGetIntegerv(GL_DEPTH_FUNC, &mut func);
            if blend {
                for (i, pname) in [GL_BLEND_SRC_RGB, GL_BLEND_DST_RGB, GL_BLEND_SRC_ALPHA, GL_BLEND_DST_ALPHA].into_iter().enumerate() {
                    let mut v = 0;
                    (gl.get_integer_i_v)(pname, 0, &mut v);
                    s.blend[i] = v as u32;
                }
                for (i, pname) in [GL_BLEND_EQUATION_RGB, GL_BLEND_EQUATION_ALPHA].into_iter().enumerate() {
                    let mut v = 0;
                    (gl.get_integer_i_v)(pname, 0, &mut v);
                    s.equations[i] = v as u32;
                }
            }
        }
        s.depth_func = func as u32;
        s
    }

    fn put_back(&self, gl: &Gl, extra: &Extra) {
        if self.pushed {
            // SAFETY: the binding as it was and the attributes pushed in `take`.
            unsafe {
                (gl.bind_framebuffer)(GL_DRAW_FRAMEBUFFER, self.framebuffer as u32);
                glPopAttrib();
            }
            return;
        }
        // SAFETY: the state read before, set again.
        unsafe {
            (gl.bind_framebuffer)(GL_DRAW_FRAMEBUFFER, self.framebuffer as u32);
            if self.blending {
                (extra.enable_i)(GL_BLEND, 0);
            } else {
                (extra.disable_i)(GL_BLEND, 0);
            }
            if self.blend != [0; 4] {
                (extra.blend_func_separate_i)(0, self.blend[0], self.blend[1], self.blend[2], self.blend[3]);
                (extra.blend_equation_separate_i)(0, self.equations[0], self.equations[1]);
            }
            (extra.color_mask_i)(0, self.mask[0], self.mask[1], self.mask[2], self.mask[3]);
            glDepthMask(self.depth_mask);
            glDepthFunc(self.depth_func);
        }
    }
}

type Rows = [[f32; 4]; 4];

fn multiply(a: &Rows, b: &Rows) -> Rows {
    std::array::from_fn(|r| std::array::from_fn(|c| (0..4).map(|k| a[r][k] * b[k][c]).sum()))
}

/// The inverse of an affine `m` (its last row 0 0 0 1); none if singular.
fn inverse_affine(m: &Rows) -> Option<Rows> {
    let [a, b, c] = [m[0], m[1], m[2]];
    let det = a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) + a[2] * (b[0] * c[1] - b[1] * c[0]);
    if !det.is_finite() || det.abs() < 1e-12 {
        return None;
    }
    let i = 1.0 / det;
    let r = [
        [(b[1] * c[2] - b[2] * c[1]) * i, (a[2] * c[1] - a[1] * c[2]) * i, (a[1] * b[2] - a[2] * b[1]) * i],
        [(b[2] * c[0] - b[0] * c[2]) * i, (a[0] * c[2] - a[2] * c[0]) * i, (a[2] * b[0] - a[0] * b[2]) * i],
        [(b[0] * c[1] - b[1] * c[0]) * i, (a[1] * c[0] - a[0] * c[1]) * i, (a[0] * b[1] - a[1] * b[0]) * i],
    ];
    let t = [a[3], b[3], c[3]];
    let row = |k: usize| [r[k][0], r[k][1], r[k][2], -(r[k][0] * t[0] + r[k][1] * t[1] + r[k][2] * t[2])];
    Some([row(0), row(1), row(2), [0.0, 0.0, 0.0, 1.0]])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_model_that_did_not_move_keeps_its_projection() {
        let projection: Rows = [[1.0, 0.0, 0.5, 2.0], [0.0, 1.2, 0.0, 0.0], [0.0, 0.0, 1.0, -1.0], [0.0, 0.0, 1.0, 0.0]];
        let placed: Rows = [[0.0, -1.0, 0.0, 10.0], [1.0, 0.0, 0.0, -4.0], [0.0, 0.0, 1.0, 3.0], [0.0, 0.0, 0.0, 1.0]];
        let back = inverse_affine(&placed).unwrap();
        let same = multiply(&multiply(&projection, &back), &placed);
        for r in 0..4 {
            for c in 0..4 {
                assert!((same[r][c] - projection[r][c]).abs() < 1e-5, "{same:?}");
            }
        }
    }

    #[test]
    fn a_model_moved_is_projected_where_it_stood() {
        // The world is the model's space shifted; the model stood 5 units further along x.
        let projection: Rows = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]];
        let now: Rows = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]];
        let was: Rows = [[1.0, 0.0, 0.0, 5.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]];
        let rows = multiply(&multiply(&projection, &inverse_affine(&now).unwrap()), &was);
        assert_eq!(rows[0][3], 5.0);
    }
}
