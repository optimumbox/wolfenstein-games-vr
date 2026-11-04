//! Provider: the present entry (`monaka_stereo::same_frame::PresentsPairs`), where a frame leaves
//! the game: GDI32's `SwapBuffers`, which the game calls through its import table from the render
//! thread with its GL context current ([`swap_buffers`]). Each frame goes to the same-frame loop's
//! OpenGL path (`monaka_stereo::same_frame::gl`): with the engine's side-by-side layout each half
//! is one eye, else the whole frame goes to both eyes; with frame warping ([`warp`]) one eye a frame,
//! the other warped from it, with FSR upscaling and frame generation when chosen; with DLSS
//! ([`dlss`]) both eyes drawn smaller and each upscaled to the eye size. A full-screen video
//! (`engine::Build::video`), drawn over the whole window rather than per eye, is reported as a
//! situation (`Situation::Video`), so the loop sends those frames whole to the flat screen.
//!
//! While pairs go out the window's vsync is off ([`vsync`]): the frames start on the headset's
//! refresh (`cooking::frame_pacing`), and a present held for the desktop monitor's (144 Hz against
//! a 90 Hz headset) beats against it, costing frames. The game's own interval goes back after.

use crate::{Facts, Wolf};
use monaka_hook::module::Module;
use monaka_hook::{InFlight, Original, mem};
use monaka_producer::game::{Lack, Start};
use monaka_producer::log;
use monaka_producer::situation::{self, Situation};
use crate::providers::eye_site;
use monaka_stereo::afw::Rules;
use monaka_stereo::afw::d3d11::FrameGeneration;
use monaka_stereo::same_frame::gl::{GlDlss, GlFsr, GlFsrDlss, GlWarp};
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
        dlss,
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
    // The engine's depth: standard, with an infinite far plane.
    let depth_flags = monaka_stereo::upscaler::fsr11::STANDARD_INFINITE;
    let leak = |p: std::path::PathBuf| -> &'static std::path::Path { Box::leak(p.into_boxed_path()) };
    // FSR when chosen, else DLSS (as chosen, with its sharpening) upscales each eye.
    let fsr = eye_site::fsr_scale(&config, start.shared, start.build.build).and_then(|_| match (config.fsr, config.dlss) {
        (Some(settings), _) => Some(GlFsr { display: [0.5, 1.0], sharpness: settings.sharpness, jitter_sign: settings.jitter_sign, depth_flags, dlls: dlls(), dlss: None }),
        (None, Some(settings)) => {
            let found = dlss_folder(start);
            if found.is_none() {
                log!("DLSS: {DLSS_DLL} was not found beside the pack; each eye is resampled");
            }
            Some(GlFsr {
                display: [0.5, 1.0],
                sharpness: settings.sharpness,
                jitter_sign: settings.jitter_sign,
                depth_flags,
                dlls: leak(found.unwrap_or_default()),
                dlss: Some(GlFsrDlss { quality: settings.quality, data: leak(start.session.dir().join("ngx")) }),
            })
        }
        (None, None) => None,
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
        side_by_side: eye_site::drawn_side_by_side,
        region,
        fsr,
        framegen,
        // The eye's own last real frame only fills gaps: up to a refresh old, it shows the hands
        // where they were (a ghost hand where it was preferred over the warp), so never within the
        // hands' reach (the gun's ghost beside it): those gaps show what is behind, not even with
        // its held things moved with the play space (the hand moved too since that frame).
        rules: Rules::WHOLE.every(eye_site::REFRESH).fill_first(false).held_reach(HELD_REACH_CM).held_fill(false),
        // Command line only: `afw_sources=1` shows where each warped pixel came from.
        sources: start.options.text("afw_sources").and_then(|t| t.trim().parse().ok()).unwrap_or(0),
        // Command line only: `afw_snapshots=N` writes N warped pairs beside the log, and the warp's
        // inputs once (0 by default: each stalls the GPU to read the images back).
        snapshots: start.options.text("afw_snapshots").and_then(|t| t.trim().parse().ok()).unwrap_or(0),
    })
}

/// How far from the eyes the hands and what they hold reach (centimetres): frame warping never fills
/// a gap there from the eye's own last frame.
const HELD_REACH_CM: u16 = 90;

/// With DLSS chosen, both eyes drawn each frame and an eye size asked for: the window made the
/// render size, each eye's projection jittered, and DLSS bringing each eye to the eye size. NVIDIA's
/// `nvngx_dlss.dll` is looked for beside the pack's DLL (or where `dlss_dll` says).
fn dlss(start: &Start<'_, Wolf>) -> Option<GlDlss> {
    let config = crate::eyes(start);
    let settings = config.dlss?;
    if !config.stereo || eye_site::warps(&config, start.shared, start.build.build) {
        return None;
    }
    let Some(display) = start.options.size("eye_size") else {
        log!("DLSS is off: no eye size asked for");
        return None;
    };
    let Some(dlls) = dlss_folder(start) else {
        log!("DLSS is off: {DLSS_DLL} was not found beside the pack");
        return None;
    };
    log!("DLSS model from {}", dlls.display());
    eye_site::arm_dlss(settings.quality);
    crate::providers::window_size::arm_dlss(settings.quality, display);
    let leak = |p: std::path::PathBuf| -> &'static std::path::Path { Box::leak(p.into_boxed_path()) };
    Some(GlDlss {
        quality: settings.quality,
        display,
        frame: eye_site::dlss_frame,
        depth: eye_site::scene_depth,
        dlls: leak(dlls),
        data: leak(start.session.dir().join("ngx")),
        jitter_sign: settings.jitter_sign,
        sharpness: settings.sharpness,
    })
}

const DLSS_DLL: &str = "nvngx_dlss.dll";

/// The folder holding NVIDIA's DLSS model: the `dlss_dll` option's, else the installed pack's. The
/// launcher runs a copy of the producer from the run folder, so the pack's own folder is read from
/// the run log's `producer:` line, then the copy's folder and Monaka's usual pack folder are tried.
fn dlss_folder(start: &Start<'_, Wolf>) -> Option<std::path::PathBuf> {
    use std::path::{Path, PathBuf};
    let mut tried: Vec<PathBuf> = start.options.text("dlss_dll").map(PathBuf::from).into_iter().collect();
    let run_log = std::fs::read_to_string(start.session.dir().join("monaka.log")).unwrap_or_default();
    tried.extend(producer_folder(&run_log));
    tried.extend(Module::containing(dlss as *const () as usize).and_then(|m| m.path()).and_then(|p| p.parent().map(Path::to_path_buf)));
    tried.extend(std::env::var_os("LOCALAPPDATA").map(|l| PathBuf::from(l).join(r"monaka\games\wolfenstein-games-vr")));
    tried.into_iter().find(|dir| dir.join(DLSS_DLL).is_file())
}

/// The producer's folder from the run log's header (`producer: <path> sha256 ...`), with
/// `%USERPROFILE%` put back.
fn producer_folder(run_log: &str) -> Option<std::path::PathBuf> {
    let line = run_log.lines().take(20).find_map(|l| l.split_once(" producer: ").map(|(_, rest)| rest))?;
    let path = line.rsplit_once(" sha256 ").map_or(line, |(path, _)| path).trim();
    let path = match (path.strip_prefix("%USERPROFILE%"), std::env::var("USERPROFILE")) {
        (Some(rest), Ok(profile)) => format!("{profile}{rest}"),
        _ => path.to_owned(),
    };
    std::path::Path::new(&path).parent().map(std::path::Path::to_path_buf)
}

#[cfg(test)]
#[test]
fn the_producer_folder_comes_from_the_run_log_header() {
    let log = "    0.005 game: x\n    0.005 producer: C:\\Packs\\wolf pack\\monaka_wolf_tno.dll sha256 F247\n";
    assert_eq!(producer_folder(log), Some(std::path::PathBuf::from(r"C:\Packs\wolf pack")));
    assert_eq!(producer_folder("    0.005 game: x\n"), None);
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
        crate::research::shaders::frame(gl::frames() + 1);
        crate::research::profile::frame();
        if let Some(playing) = video_playing() {
            situation::report(Situation::Video, playing, VIDEO_FRESH_MS);
        }
    }
    crate::providers::window_size::seen(dc);
    crate::single_pass::draws::frame(crate::on(crate::SINGLE_PASS_AT));
    vsync::follow(PAIRS.publishing());
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

/// The window's swap interval (`WGL_EXT_swap_control`), held at 0 while pairs go out, on the render
/// thread with the game's context current.
mod vsync {
    use monaka_producer::log;
    use std::sync::Mutex;
    use windows::Win32::Graphics::OpenGL::wglGetProcAddress;
    use windows::core::{BOOL, s};

    type SwapIntervalFn = unsafe extern "system" fn(i32) -> BOOL;
    type GetSwapIntervalFn = unsafe extern "system" fn() -> i32;

    /// Times the engine set its own interval again that are logged.
    const LOGGED: u64 = 5;

    struct Vsync {
        set: SwapIntervalFn,
        get: GetSwapIntervalFn,
        /// The game's interval before it was turned off; none while the game's own holds.
        game: Option<i32>,
        /// Times the interval was found back on and turned off again.
        again: u64,
    }

    /// Loaded at the first present (none: not yet; `Err`: the driver has no swap control).
    static STATE: Mutex<Option<Result<Vsync, ()>>> = Mutex::new(None);

    fn load() -> Result<Vsync, ()> {
        // SAFETY: asks the current context's driver for its WGL_EXT_swap_control entry points.
        let (set, get) = unsafe { (wglGetProcAddress(s!("wglSwapIntervalEXT")), wglGetProcAddress(s!("wglGetSwapIntervalEXT"))) };
        match (set, get) {
            (Some(set), Some(get)) if (set as usize) >= 4 && (get as usize) >= 4 => {
                // SAFETY: the driver's functions with the signatures WGL_EXT_swap_control gives.
                Ok(unsafe { Vsync { set: std::mem::transmute::<_, SwapIntervalFn>(set), get: std::mem::transmute::<_, GetSwapIntervalFn>(get), game: None, again: 0 } })
            }
            _ => {
                log!("vsync: the driver has no WGL_EXT_swap_control; the game's own stays");
                Err(())
            }
        }
    }

    /// At each present: vsync off while `publishing` (turned off again whenever the engine set it
    /// back), the game's own interval back once not.
    pub fn follow(publishing: bool) {
        let mut state = STATE.lock().unwrap_or_else(|e| e.into_inner());
        if state.is_none() && !publishing {
            return;
        }
        let Ok(vsync) = state.get_or_insert_with(load) else { return };
        // SAFETY: WGL_EXT_swap_control's calls on this thread's current context (the game's).
        unsafe {
            let now = (vsync.get)();
            match (publishing, vsync.game) {
                (true, _) if now == 0 => {}
                (true, None) => {
                    vsync.game = Some(now);
                    let _ = (vsync.set)(0);
                    log!("vsync: off while pairs go out (the game's interval was {now})");
                }
                (true, Some(_)) => {
                    vsync.again += 1;
                    let _ = (vsync.set)(0);
                    if vsync.again <= LOGGED {
                        log!("vsync: the game set an interval of {now} again; off once more");
                    }
                }
                (false, Some(game)) => {
                    let _ = (vsync.set)(game);
                    vsync.game = None;
                    log!("vsync: the game's interval of {game} back");
                }
                (false, None) => {}
            }
        }
    }
}
