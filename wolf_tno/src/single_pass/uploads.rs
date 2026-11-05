//! What the render thread spends on texture and buffer uploads and on waiting for the GPU, frame
//! by frame: the game streams its virtual texture's pages (and meshes) in as the player moves, and
//! a frame that took far longer than a refresh is logged with what it uploaded and waited for, so
//! a hitch shows whether streaming made it (many pages, long uploads or waits) or something else
//! did (the render thread idle in it: the game thread's own work). A 10 s summary goes with the
//! others.

use monaka_hook::{InFlight, Original};
use monaka_producer::log;
use std::ffi::c_void;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::time::{Duration, Instant};

pub type TexSubImage2dFn = unsafe extern "system" fn(u32, i32, i32, i32, i32, i32, u32, u32, *const c_void);
pub type CompressedTexSubImage2dFn = unsafe extern "system" fn(u32, i32, i32, i32, i32, i32, u32, i32, *const c_void);
pub type TexSubImage3dFn = unsafe extern "system" fn(u32, i32, i32, i32, i32, i32, i32, i32, u32, u32, *const c_void);
pub type CompressedTexSubImage3dFn = unsafe extern "system" fn(u32, i32, i32, i32, i32, i32, i32, i32, u32, i32, *const c_void);
pub type BufferSubDataFn = unsafe extern "system" fn(u32, isize, isize, *const c_void);
pub type TextureSubImage2dFn = unsafe extern "system" fn(u32, u32, i32, i32, i32, i32, i32, u32, u32, *const c_void);
pub type CompressedTextureSubImage2dFn = unsafe extern "system" fn(u32, u32, i32, i32, i32, i32, i32, u32, i32, *const c_void);
pub type NamedBufferSubDataFn = unsafe extern "system" fn(u32, isize, isize, *const c_void);
pub type ClientWaitSyncFn = unsafe extern "system" fn(*const c_void, u32, u64) -> u32;

pub static TEX_SUB_IMAGE_2D: Original<TexSubImage2dFn> = Original::new();
pub static COMPRESSED_TEX_SUB_IMAGE_2D: Original<CompressedTexSubImage2dFn> = Original::new();
pub static TEX_SUB_IMAGE_3D: Original<TexSubImage3dFn> = Original::new();
pub static COMPRESSED_TEX_SUB_IMAGE_3D: Original<CompressedTexSubImage3dFn> = Original::new();
pub static BUFFER_SUB_DATA: Original<BufferSubDataFn> = Original::new();
pub static TEXTURE_SUB_IMAGE_2D: Original<TextureSubImage2dFn> = Original::new();
pub static COMPRESSED_TEXTURE_SUB_IMAGE_2D: Original<CompressedTextureSubImage2dFn> = Original::new();
pub static NAMED_BUFFER_SUB_DATA: Original<NamedBufferSubDataFn> = Original::new();
pub static CLIENT_WAIT_SYNC: Original<ClientWaitSyncFn> = Original::new();

/// One frame's (or a window's) uploads and waits.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Tally {
    /// Texture uploads: calls, texels (uncompressed), bytes (compressed), time inside them.
    pub textures: u64,
    pub texels: u64,
    pub compressed_bytes: u64,
    pub texture_ns: u64,
    /// Buffer uploads: calls, bytes, time.
    pub buffers: u64,
    pub buffer_bytes: u64,
    pub buffer_ns: u64,
    /// Waits for a fence: calls, time.
    pub waits: u64,
    pub wait_ns: u64,
}

const FIELDS: usize = 9;
static NOW: [AtomicU64; FIELDS] = [const { AtomicU64::new(0) }; FIELDS];

impl Tally {
    fn take() -> Self {
        let v: [u64; FIELDS] = std::array::from_fn(|i| NOW[i].swap(0, Relaxed));
        Self { textures: v[0], texels: v[1], compressed_bytes: v[2], texture_ns: v[3], buffers: v[4], buffer_bytes: v[5], buffer_ns: v[6], waits: v[7], wait_ns: v[8] }
    }

    fn add(&mut self, o: &Self) {
        self.textures += o.textures;
        self.texels += o.texels;
        self.compressed_bytes += o.compressed_bytes;
        self.texture_ns += o.texture_ns;
        self.buffers += o.buffers;
        self.buffer_bytes += o.buffer_bytes;
        self.buffer_ns += o.buffer_ns;
        self.waits += o.waits;
        self.wait_ns += o.wait_ns;
    }

    /// The tally as a log's words.
    pub fn describe(&self) -> String {
        let ms = |ns: u64| ns as f64 / 1e6;
        format!(
            "{} texture uploads ({:.1} M texels, {:.1} MB compressed) {:.2} ms; {} buffer uploads ({:.1} MB) {:.2} ms; {} fence waits {:.2} ms",
            self.textures,
            self.texels as f64 / 1e6,
            self.compressed_bytes as f64 / 1e6,
            ms(self.texture_ns),
            self.buffers,
            self.buffer_bytes as f64 / 1e6,
            ms(self.buffer_ns),
            self.waits,
            ms(self.wait_ns)
        )
    }
}

fn count(texture: bool, units: u64, compressed: bool, started: Instant) {
    let ns = started.elapsed().as_nanos() as u64;
    if texture {
        NOW[0].fetch_add(1, Relaxed);
        NOW[if compressed { 2 } else { 1 }].fetch_add(units, Relaxed);
        NOW[3].fetch_add(ns, Relaxed);
    } else {
        NOW[4].fetch_add(1, Relaxed);
        NOW[5].fetch_add(units, Relaxed);
        NOW[6].fetch_add(ns, Relaxed);
    }
}

fn texels(w: i32, h: i32, d: i32) -> u64 {
    u64::try_from(i64::from(w.max(0)) * i64::from(h.max(0)) * i64::from(d.max(0))).unwrap_or(0)
}

/// A frame this much longer than the refresh is logged.
const SLOW: Duration = Duration::from_millis(25);
/// Slow frames logged at most, a run.
const SLOW_LOGGED: u32 = 60;
const WINDOW: Duration = Duration::from_secs(10);

struct Frames {
    last: Option<Instant>,
    window_start: Option<Instant>,
    window: Tally,
    worst: (Duration, Tally),
    frames: u64,
    slow: u64,
    logged: u32,
}

const ZERO: Tally = Tally { textures: 0, texels: 0, compressed_bytes: 0, texture_ns: 0, buffers: 0, buffer_bytes: 0, buffer_ns: 0, waits: 0, wait_ns: 0 };

static FRAMES: Mutex<Frames> = Mutex::new(Frames { last: None, window_start: None, window: ZERO, worst: (Duration::ZERO, ZERO), frames: 0, slow: 0, logged: 0 });

/// At each of the game's frames (render thread): the frame's tally, logged if it was slow, and the
/// window's summary every [`WINDOW`].
pub fn end_frame() {
    crate::providers::settings::watch_streaming();
    let Ok(mut f) = FRAMES.lock() else { return };
    let now = Instant::now();
    let tally = Tally::take();
    let Some(last) = f.last.replace(now) else { return };
    let took = now - last;
    f.frames += 1;
    f.window.add(&tally);
    if took > f.worst.0 {
        f.worst = (took, tally);
    }
    if took > SLOW {
        f.slow += 1;
        if f.logged < SLOW_LOGGED {
            f.logged += 1;
            log!("streaming: a frame of {:.1} ms on the render thread: {}", took.as_secs_f64() * 1e3, tally.describe());
        }
    }
    let start = *f.window_start.get_or_insert(now);
    if now - start >= WINDOW {
        let frames = f.frames.max(1) as f64;
        let w = f.window;
        log!(
            "streaming over 10 s: per frame {:.1} texture uploads ({:.2} ms), {:.1} buffer uploads ({:.2} ms), {:.1} fence waits ({:.2} ms); {} frames over {} ms; the longest frame {:.1} ms: {}",
            w.textures as f64 / frames,
            w.texture_ns as f64 / 1e6 / frames,
            w.buffers as f64 / frames,
            w.buffer_ns as f64 / 1e6 / frames,
            w.waits as f64 / frames,
            w.wait_ns as f64 / 1e6 / frames,
            f.slow,
            SLOW.as_millis(),
            f.worst.0.as_secs_f64() * 1e3,
            f.worst.1.describe()
        );
        log_video_memory();
        f.window_start = Some(now);
        f.window = Tally::default();
        f.worst = (Duration::ZERO, Tally::default());
        f.frames = 0;
        f.slow = 0;
    }
}

/// The GPU's memory as the driver tells the game's context (GL_NVX_gpu_memory_info): what is free,
/// and what the driver evicted to system memory to make room since the last report. Evictions show
/// the game's textures no longer fitting beside everything else on the GPU.
fn log_video_memory() {
    const AVAILABLE: u32 = 0x9049;
    const EVICTIONS: u32 = 0x904A;
    const EVICTED: u32 = 0x904B;
    static LAST: Mutex<(i32, i32)> = Mutex::new((0, 0));
    let (mut available, mut evictions, mut evicted) = (-1, -1, -1);
    // SAFETY: plain queries on the game's current context, on its render thread.
    unsafe {
        windows::Win32::Graphics::OpenGL::glGetIntegerv(AVAILABLE, &mut available);
        windows::Win32::Graphics::OpenGL::glGetIntegerv(EVICTIONS, &mut evictions);
        windows::Win32::Graphics::OpenGL::glGetIntegerv(EVICTED, &mut evicted);
    }
    if available < 0 {
        return;
    }
    let Ok(mut last) = LAST.lock() else { return };
    let (since, since_kb) = (evictions - last.0, evicted - last.1);
    *last = (evictions, evicted);
    log!("video memory: {} MB free; {} evictions ({} MB) since the last report", available / 1024, since, since_kb / 1024);
}

pub unsafe extern "system" fn tex_sub_image_2d(target: u32, level: i32, x: i32, y: i32, w: i32, h: i32, format: u32, kind: u32, pixels: *const c_void) {
    let flight = InFlight::enter();
    let started = Instant::now();
    super::pieces::call("glTexSubImage2D");
    // SAFETY: forwards the game's own call.
    unsafe { TEX_SUB_IMAGE_2D.get(&flight)(target, level, x, y, w, h, format, kind, pixels) };
    count(true, texels(w, h, 1), false, started);
}

pub unsafe extern "system" fn compressed_tex_sub_image_2d(target: u32, level: i32, x: i32, y: i32, w: i32, h: i32, format: u32, size: i32, data: *const c_void) {
    let flight = InFlight::enter();
    let started = Instant::now();
    super::pieces::call("glCompressedTexSubImage2D");
    // SAFETY: forwards the game's own call.
    unsafe { COMPRESSED_TEX_SUB_IMAGE_2D.get(&flight)(target, level, x, y, w, h, format, size, data) };
    count(true, u64::try_from(size).unwrap_or(0), true, started);
}

pub unsafe extern "system" fn tex_sub_image_3d(target: u32, level: i32, x: i32, y: i32, z: i32, w: i32, h: i32, d: i32, format: u32, kind: u32, pixels: *const c_void) {
    let flight = InFlight::enter();
    let started = Instant::now();
    super::pieces::call("glTexSubImage3D");
    // SAFETY: forwards the game's own call.
    unsafe { TEX_SUB_IMAGE_3D.get(&flight)(target, level, x, y, z, w, h, d, format, kind, pixels) };
    count(true, texels(w, h, d), false, started);
}

pub unsafe extern "system" fn compressed_tex_sub_image_3d(target: u32, level: i32, x: i32, y: i32, z: i32, w: i32, h: i32, d: i32, format: u32, size: i32, data: *const c_void) {
    let flight = InFlight::enter();
    let started = Instant::now();
    super::pieces::call("glCompressedTexSubImage3D");
    // SAFETY: forwards the game's own call.
    unsafe { COMPRESSED_TEX_SUB_IMAGE_3D.get(&flight)(target, level, x, y, z, w, h, d, format, size, data) };
    count(true, u64::try_from(size).unwrap_or(0), true, started);
}

pub unsafe extern "system" fn buffer_sub_data(target: u32, offset: isize, size: isize, data: *const c_void) {
    let flight = InFlight::enter();
    let started = Instant::now();
    super::pieces::call("glBufferSubData");
    // SAFETY: forwards the game's own call.
    unsafe { BUFFER_SUB_DATA.get(&flight)(target, offset, size, data) };
    count(false, u64::try_from(size).unwrap_or(0), false, started);
}

pub unsafe extern "system" fn texture_sub_image_2d(texture: u32, target: u32, level: i32, x: i32, y: i32, w: i32, h: i32, format: u32, kind: u32, pixels: *const c_void) {
    let flight = InFlight::enter();
    let started = Instant::now();
    super::pieces::call("glTextureSubImage2DEXT");
    // SAFETY: forwards the game's own call.
    unsafe { TEXTURE_SUB_IMAGE_2D.get(&flight)(texture, target, level, x, y, w, h, format, kind, pixels) };
    count(true, texels(w, h, 1), false, started);
}

pub unsafe extern "system" fn compressed_texture_sub_image_2d(texture: u32, target: u32, level: i32, x: i32, y: i32, w: i32, h: i32, format: u32, size: i32, data: *const c_void) {
    let flight = InFlight::enter();
    let started = Instant::now();
    super::pieces::call("glCompressedTextureSubImage2DEXT");
    // SAFETY: forwards the game's own call.
    unsafe { COMPRESSED_TEXTURE_SUB_IMAGE_2D.get(&flight)(texture, target, level, x, y, w, h, format, size, data) };
    count(true, u64::try_from(size).unwrap_or(0), true, started);
}

pub unsafe extern "system" fn named_buffer_sub_data(buffer: u32, offset: isize, size: isize, data: *const c_void) {
    let flight = InFlight::enter();
    let started = Instant::now();
    super::pieces::call("glNamedBufferSubDataEXT");
    // SAFETY: forwards the game's own call.
    unsafe { NAMED_BUFFER_SUB_DATA.get(&flight)(buffer, offset, size, data) };
    count(false, u64::try_from(size).unwrap_or(0), false, started);
}

pub unsafe extern "system" fn client_wait_sync(sync: *const c_void, flags: u32, timeout: u64) -> u32 {
    let flight = InFlight::enter();
    let started = Instant::now();
    super::pieces::call("glClientWaitSync");
    // SAFETY: forwards the game's own call.
    let result = unsafe { CLIENT_WAIT_SYNC.get(&flight)(sync, flags, timeout) };
    NOW[7].fetch_add(1, Relaxed);
    NOW[8].fetch_add(started.elapsed().as_nanos() as u64, Relaxed);
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_tally_reads_as_its_counts() {
        let t = Tally { textures: 3, texels: 2_000_000, compressed_bytes: 500_000, texture_ns: 1_500_000, buffers: 1, buffer_bytes: 1_000_000, buffer_ns: 250_000, waits: 2, wait_ns: 40_000_000 };
        assert_eq!(t.describe(), "3 texture uploads (2.0 M texels, 0.5 MB compressed) 1.50 ms; 1 buffer uploads (1.0 MB) 0.25 ms; 2 fence waits 40.00 ms");
        assert_eq!(texels(-1, 4, 1), 0);
        assert_eq!(texels(128, 128, 2), 32768);
    }
}
