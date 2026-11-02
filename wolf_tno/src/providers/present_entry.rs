//! Provider: the present entry (`monaka_stereo::same_frame::PresentsPairs`), where a frame leaves
//! the game: GDI32's `SwapBuffers`, which the game calls through its import table from the render
//! thread with its GL context current ([`swap_buffers`]). Each frame goes to the same-frame loop's
//! OpenGL path (`monaka_stereo::same_frame::gl`): with the engine's side-by-side layout each half
//! is one eye, else the whole frame goes to both eyes; with frame warping ([`warp`]) one eye a frame,
//! the other warped from it, with FSR upscaling and frame generation when chosen. A full-screen video
//! (`engine::Build::video`), drawn over the whole window rather than per eye, is reported as a
//! situation (`Situation::Video`), so the loop sends those frames whole to the flat screen.

use crate::{Facts, Wolf};
use monaka_hook::module::Module;
use monaka_hook::{InFlight, Original, mem};
use monaka_producer::game::{Lack, Start};
use monaka_producer::log;
use monaka_producer::situation::{self, Situation};
use crate::providers::eye_site;
use monaka_stereo::afw::Rules;
use monaka_stereo::afw::d3d11::FrameGeneration;
use monaka_stereo::same_frame::gl::{GlFsr, GlWarp};
use monaka_core::sync::Locked;
use monaka_stereo::same_frame::timing::Stage;
use monaka_stereo::same_frame::{Output, PAIRS, PairLoop, PresentsPairs, gl};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering::*};
use windows::Win32::Graphics::Gdi::HDC;
use windows::core::BOOL;

pub type SwapBuffersFn = unsafe extern "system" fn(HDC) -> BOOL;
pub static SWAP_BUFFERS_ORIGINAL: Original<SwapBuffersFn> = Original::new();

/// The Wolfensteins' pairs go out by the same-frame loop's OpenGL path: both eyes in each frame
/// with stereo (`stereo`), the head followed with stereo and head tracking, the eye site's counts
/// in the rate line.
impl PresentsPairs for Wolf {
    const LOOP: PairLoop<Self> = PairLoop {
        follows_head: |start| crate::eyes::stereo_on(start.options) && start.shared.head_tracking,
        rate: Some(rate),
        warp,
        // Each eye's depth with its pair: a pair shown again moves for the eyes, not only turns.
        depth: |_| Some(gl::GlPairDepth { depth: eye_site::scene_depth, near: eye_site::near_metres }),
        // The render thread presents each frame at the start of drawing the next.
        presents_behind_drawing: true,
        // Nothing goes out until the window is the eye size asked for (`render_size`).
        ready: crate::providers::window_size::ready,
        ..PairLoop::new(output, entry)
    };
}

fn output(start: &Start<'_, Wolf>) -> Output {
    Output::OpenGl { side_by_side: crate::eyes::stereo_on(start.options) }
}

/// With frame warping ([`eye_site::warps`]): the eye in the window's left half (the one-eye layout
/// draws it there, at a side-by-side eye's size), its shot and the scene's depth from the eye site;
/// gaps filled from each eye's own frame while it is at most [`eye_site::REFRESH`] frames old. FSR
/// upscaling and frame generation load AMD's DLLs from where the launcher found them (`fsr_dlls`).
fn warp(start: &Start<'_, Wolf>) -> Option<GlWarp> {
    let config = crate::eyes(start);
    if !eye_site::warps(&config, start.shared, start.build.build) {
        return None;
    }
    let mut leaked: Option<&'static std::path::Path> = None;
    let mut dlls = || *leaked.get_or_insert_with(|| Box::leak(start.options.text("fsr_dlls").map_or_else(|| start.session.dir().to_path_buf(), std::path::PathBuf::from).into_boxed_path()));
    let fsr = eye_site::fsr_scale(&config, start.shared, start.build.build).and_then(|_| {
        let settings = config.fsr?;
        Some(GlFsr {
            display: [0.5, 1.0],
            sharpness: settings.sharpness,
            jitter_sign: settings.jitter_sign,
            // The engine's depth: standard, with an infinite far plane.
            depth_flags: monaka_stereo::upscaler::fsr11::STANDARD_INFINITE,
            dlls: dlls(),
        })
    });
    let framegen = config.framegen.then(|| FrameGeneration {
        dlls: dlls(),
        jitter_sign: config.fsr.map_or(1.0, |f| f.jitter_sign),
        // The engine's depth, as for FSR.
        depth_flags: FrameGeneration::STANDARD_INFINITE,
        units_per_metre: config.units_per_metre,
    });
    let region = eye_site::warp_region(eye_site::fsr_scale(&config, start.shared, start.build.build).unwrap_or(1.0));
    Some(GlWarp {
        shot: eye_site::warp_shot,
        depth: eye_site::scene_depth,
        region,
        fsr,
        framegen,
        // The eye's own last real frame only fills gaps: up to a refresh old, it shows the hands
        // where they were (a ghost hand where it was preferred over the warp).
        rules: Rules::WHOLE.every(eye_site::REFRESH).fill_first(false),
        // Command line only: `afw_sources=1` shows where each warped pixel came from.
        sources: start.options.text("afw_sources").and_then(|t| t.trim().parse().ok()).unwrap_or(0),
    })
}

/// The rate line's addition: the view builds and how they were drawn.
fn rate() {
    log!("{}", crate::providers::eye_site::summary());
}

/// Where the engine keeps its open video (0: not watched), and whether both eyes are in the frame
/// (whole frames to both eyes play a video as they are).
static VIDEO: AtomicUsize = AtomicUsize::new(0);
static SIDE_BY_SIDE: AtomicBool = AtomicBool::new(false);
/// Whether a video plays goes into the run's situations each frame, holding this long (ms).
const VIDEO_FRESH_MS: u64 = 500;

/// Hooks `SwapBuffers` in the game's import table, and watches its open video.
fn entry(start: &mut Start<'_, Wolf>) -> Result<(), Lack> {
    let flight = monaka_hook::InFlight::enter();
    let Facts { exe, build } = *start.build;
    // SAFETY: the import slot holds GDI32's SwapBuffers, which the detour's type matches.
    unsafe { start.hooks.import(&SWAP_BUFFERS_ORIGINAL, "SwapBuffers", &exe, "GDI32.dll", "SwapBuffers", swap_buffers)? };
    let system = Module::find("gdi32.dll").and_then(|gdi| gdi.export("SwapBuffers"));
    let imported = SWAP_BUFFERS_ORIGINAL.get(&flight) as usize;
    if system != Some(imported) {
        log!("note: the import slot holds {imported:#x}, not GDI32's SwapBuffers {system:x?} (another hook?); chaining to it");
    }
    SIDE_BY_SIDE.store(crate::eyes::stereo_on(start.options), Release);
    VIDEO.store(exe.at(build.video), Release);
    Ok(())
}

/// Whether a full-screen video plays over both eyes now (none watched: not known).
fn video_playing() -> Option<bool> {
    let video = VIDEO.load(Acquire);
    (video != 0).then(|| SIDE_BY_SIDE.load(Acquire) && mem::read::<usize>(video).is_some_and(|open| open != 0))
}

pub unsafe extern "system" fn swap_buffers(dc: HDC) -> BOOL {
    let flight = InFlight::enter();
    crate::research::profile::phase(crate::research::profile::Phase::Monaka);
    if let Some(returned) = *LAST_SWAP.locked() {
        PAIRS.time(Stage::GameRender, returned.elapsed());
    }
    let cpu = render_thread_cpu();
    if let Some(last) = LAST_CPU.locked().replace(cpu) {
        PAIRS.time(Stage::RenderCpu, cpu.saturating_sub(last));
    }
    // The swapchain's size is not known here: the Stats panel leaves it out.
    monaka_producer::debug::note_present(|| [0, 0]);
    // Monaka's own work before the present is timed while the Stats panel shows.
    let started = monaka_producer::debug::timing().then(std::time::Instant::now);
    if PAIRS.publishing() {
        crate::research::gl::frame(gl::frames() + 1);
        crate::research::profile::frame();
        if let Some(playing) = video_playing() {
            situation::report(Situation::Video, playing, VIDEO_FRESH_MS);
        }
    }
    crate::providers::window_size::seen(dc);
    gl::present(dc);
    crate::cooking::frame_pacing::published();
    if let Some(at) = started {
        use monaka_contract::debug_draw::Cost;
        let took = at.elapsed();
        monaka_producer::debug::cost(Cost::BeforePresent, took);
        monaka_producer::debug::cost(Cost::AtPresent, took);
    }
    let swapping = std::time::Instant::now();
    crate::research::profile::phase(crate::research::profile::Phase::Swap);
    // SAFETY: forwards the game's own call.
    let result = unsafe { SWAP_BUFFERS_ORIGINAL.get(&flight)(dc) };
    crate::research::profile::phase(crate::research::profile::Phase::Game);
    PAIRS.time(Stage::GameSwap, swapping.elapsed());
    *LAST_SWAP.locked() = Some(std::time::Instant::now());
    result
}

/// The render thread's CPU time at the last `SwapBuffers`.
static LAST_CPU: std::sync::Mutex<Option<std::time::Duration>> = std::sync::Mutex::new(None);

/// The calling thread's CPU time (user and kernel) so far.
fn render_thread_cpu() -> std::time::Duration {
    use windows::Win32::Foundation::FILETIME;
    use windows::Win32::System::Threading::{GetCurrentThread, GetThreadTimes};
    let (mut created, mut exited, mut kernel, mut user) = (FILETIME::default(), FILETIME::default(), FILETIME::default(), FILETIME::default());
    // SAFETY: the current thread's pseudo handle and four locals.
    if unsafe { GetThreadTimes(GetCurrentThread(), &mut created, &mut exited, &mut kernel, &mut user) }.is_err() {
        return std::time::Duration::ZERO;
    }
    let ticks = |t: FILETIME| u64::from(t.dwHighDateTime) << 32 | u64::from(t.dwLowDateTime);
    std::time::Duration::from_nanos((ticks(kernel) + ticks(user)) * 100)
}

/// When the game's last `SwapBuffers` returned (its render thread's time between swaps).
static LAST_SWAP: std::sync::Mutex<Option<std::time::Instant>> = std::sync::Mutex::new(None);
