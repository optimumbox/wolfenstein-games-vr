//! Provider: the eye site (`monaka_stereo::eye_site::EyeSite`), the Wolfensteins' side of the
//! same-frame loop (`monaka_stereo::same_frame`, its OpenGL path, which installs the site): reading
//! the game camera and writing an eye into it by the recipe (`monaka_core::camera::EyeView::of`),
//! with the id Tech view helpers (`eng_idt::view`) to read and write the render views. With it, the
//! engine's side-by-side layout ([`EYES`]: both eyes drawn each frame, the eye separation and the
//! projection answered here). The dominant eye and the sight lock it applies are cooking
//! ([`sights`]); the head share it swaps is the look's (`monaka_body::look::baked`).
//!
//! Head tracking at the view builder: the player's render view is turned and moved by the loop's
//! head pose for the length of the call (the engine then offsets each eye along the turned axis),
//! each eye's projection is replaced by the headset's own frustum for that eye, and the frame's
//! ticket goes to the loop (`PAIRS.built`), so the pair is published with the pose it was drawn
//! from.
//!
//! Frame warping (AFW, [`warps`]): a layout of our own draws one view into the left half of the
//! window ([`one_eye_layout`]). Its view's eye sign is set each frame to the eye drawn (the dominant
//! eye, the other every [`REFRESH`]th frame: [`warp_eye`]), so the engine offsets the camera onto
//! that eye and asks for its projection as it does for each side-by-side eye (with a sign of 0 it
//! does neither). The view builder keeps how the frame was rendered beside the loop's ticket
//! ([`warp_shot`] takes it at the frame's `SwapBuffers`); the loop warps the other eye from it with
//! the scene's depth ([`scene_depth`]).
//!
//! Its probe (`probe_view`) is in `research::view`.

use monaka_stereo::providers::EyeGeometry;
use crate::eyes::{DominantEye, EyeSettings};
use crate::cooking::sights;
use crate::providers::engine::{self, Build, HandsDepthHackFn, HandsFovScaleFn, ProjectionFn, ViewBuilderFn};
use eng_idt::MakeFrustumFn;
use crate::{Facts, Wolf};
use eng_idt::view::{Players, Saved};
use monaka_body::look::{self, Share, command};
use monaka_core::camera::{self, Base, EyeView, Frustum, HeadPosition, Mat34};
use monaka_core::convention::CameraConvention;
use monaka_core::feature::Feature;
use monaka_contract::manifest::{DOMINANT_EYE, EYE_SEPARATION, HEAD_MOVEMENT};
use monaka_core::options::Shared;
use monaka_contract::protocol::HeadPose;
use monaka_core::rendering::StereoMethod;
use monaka_core::ticket::{Tickets, View as Ticket};
use monaka_stereo::afw::{Rendered, Shot};
use monaka_stereo::driver::Pending;
use monaka_stereo::same_frame::gl::{self as gl_frames, GlDepth, GlDlssFrame, GlShot};
use monaka_stereo::upscaler::fsr11::jittered_at;
use monaka_hook::module::Module;
use monaka_hook::{InFlight, Original, mem};
use monaka_producer::game::{Component, Lack, Need, On, Start};
use monaka_producer::ledger::{Change, Setting};
use monaka_producer::{Rejection, log};
use monaka_stereo::eye_site::{EyeSite, Tally};
use monaka_stereo::same_frame::PAIRS;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering::*};
use std::sync::{Mutex, OnceLock};
use monaka_hook::thread_id;
use monaka_core::sync::Locked;

/// Stereo by the engine's own side-by-side layout, with our eye separation, on the same-frame loop
/// ([`crate::PRESENT`]); with head tracking, the player's render view turned by the head at the view
/// builder and each eye given the headset's frustum. The effects the engine draws for the last view
/// only go off (god rays, lens flares, screen-space reflections) unless an option keeps them.
pub const EYES: Component<Wolf> = Component {
    name: "eyes",
    features: &[Feature::Stereo],
    options: &[crate::eyes::STEREO, crate::eyes::FSR, crate::eyes::FSR_SCALE, crate::eyes::FSR_SHARPNESS, crate::eyes::FRAMEGEN, crate::eyes::DLSS, crate::eyes::DLSS_SHARPNESS, DOMINANT_EYE, HEAD_MOVEMENT, EYE_SEPARATION],
    needs: &[Need::On(&crate::PRESENT)],
    settings: |options, manifest| {
        EyeSettings::from_options(options, manifest);
    },
    on: |run| On::when(crate::eyes::stereo_on(run.options), "stereo is off: the whole frame goes to both eyes"),
    install: install_eyes,
    summary: Some(summary),
    ..Component::NONE
};

/// Head tracking was installed for this run: while the eyes are on, the view follows the head.
static HEAD_TRACKED: AtomicBool = AtomicBool::new(false);

fn install_eyes(start: &mut Start<'_, Wolf>) -> Result<(), Lack> {
    let config = crate::eyes(start);
    let shared = *start.shared;
    let Facts { exe, build } = *start.build;
    enter_stereo(start, &config)?;
    // With one view a frame the engine draws its last-view effects into the eye drawn: they stay.
    // The hands' depth is left true: the other eye is warped from it (in the headset the hands are
    // where the controllers are, so the world clipping them is what the player expects).
    if warps(&config, &shared, build) {
        eng_idt::set_float(start.ledger, &exe, &build.hands_depth_hack, 1.0, "the hands' squeezed depth warps them too far");
        return Ok(());
    }
    if !config.god_rays {
        let (cvar, off) = &build.god_rays_off;
        eng_idt::set_int(start.ledger, &exe, cvar, *off, "god rays stay in the right eye only");
    }
    if !config.flares {
        eng_idt::set_int(start.ledger, &exe, &build.skip_flares, 1, "lens flares stay in the right eye only");
        eng_idt::set_int(start.ledger, &exe, &build.haze_off, 1, "the ghosts bright lights leave fall in a different place in each eye");
    }
    if !config.dim_shadows {
        eng_idt::set_int(start.ledger, &exe, &build.dim_shadows_off, 1, "one eye gets a large shadow the other does not");
    }
    if !config.reflections {
        eng_idt::set_int(start.ledger, &exe, &build.reflections, 0, "screen-space reflections are traced with the wrong eye's camera");
    }
    if !shared.head_tracking {
        log!("head tracking off: the engine's own camera is shown");
    }
    Ok(())
}

/// The eye site as the same-frame loop reads it: each eye drawn with the headset's frustum where
/// the run asks for it (else the game's projection), the pose's eye distance between them.
impl EyeSite for Wolf {
    fn install(start: &mut Start<'_, Self>) -> Result<(), Lack> {
        install_site(start)
    }

    fn tally() -> &'static Tally {
        &BUILDS
    }
}

impl EyeGeometry for Wolf {
    fn frustum(pose: &HeadPose, eye: usize) -> Option<Frustum> {
        SETUP.get().filter(|setup| setup.shared.headset_fov).and_then(|_| Frustum::from_fov(pose.fov[eye.min(1)]))
    }

    fn separation(pose: &HeadPose) -> Option<f32> {
        Some(SETUP.get()?.shared.separation_for(pose))
    }
}

/// The site's hooks (installed by the same-frame loop): with stereo and head tracking, the view
/// builder and the projection.
fn install_site(start: &mut Start<'_, Wolf>) -> Result<(), Lack> {
    if !monaka_stereo::eye_site::stereo_on(start) || !start.shared.head_tracking {
        return Ok(());
    }
    let Facts { exe, build } = *start.build;
    install(crate::eyes(start), *start.shared, exe, build).map_err(Rejection::revision)?;
    start.hooks.site(&VIEW_BUILDER_ORIGINAL, &exe, &build.view_builder, view_builder)?;
    start.hooks.site(&PROJECTION_ORIGINAL, &exe, &build.projection, projection)?;
    start.hooks.site(&HANDS_FOV_SCALE_ORIGINAL, &exe, &build.hands_fov_scale, hands_fov_scale)?;
    start.hooks.site(&HANDS_DEPTH_HACK_ORIGINAL, &exe, &build.hands_depth_hack_get, hands_depth_hack)?;
    HEAD_TRACKED.store(true, Release);
    Ok(())
}

/// The engine's stereo layout: the pointer it draws the views by (one of the engine's layouts, or
/// [`one_eye_layout`]).
struct LayoutSetting {
    exe: Module,
    build: &'static Build,
}

impl Setting for LayoutSetting {
    type Value = usize;
    fn name(&self) -> String {
        "layout".to_owned()
    }
    fn read(&self) -> Option<usize> {
        mem::read::<usize>(self.exe.at(self.build.layout_global))
    }
    fn write(&self, layout: &usize) -> bool {
        mem::write(self.exe.at(self.build.layout_global), *layout)
    }
}

/// A layout record.
type Record = [u8; engine::LAYOUT_RECORD];

/// The one-eye layout's first view (its eye sign set each frame), once made.
static ONE_EYE: OnceLock<usize> = OnceLock::new();
/// The thread that builds the views (the game thread), once one was built.
static GAME_THREAD: AtomicU32 = AtomicU32::new(0);

/// The thread that builds the views (the game thread; 0 before the first build).
pub fn game_thread() -> u32 {
    GAME_THREAD.load(Relaxed)
}

/// Player 0's render view at the last view build, as the game made it (origin, then axis rows).
static RENDER_VIEW: Mutex<Option<[f32; 12]>> = Mutex::new(None);

/// Player 0's render view at the last view build, before the head's turn: what the eyes are built
/// from.
pub fn render_view() -> Option<[f32; 12]> {
    *RENDER_VIEW.locked()
}

/// How far along the game camera's forward axis a driven machine's aim is shown (metres): far
/// enough that both eyes agree on it for what the machine shoots at.
pub const AIM_DISTANCE: f32 = 20.0;

/// The frame's head pose, as a key: the render thread draws the HUD of the frame it is drawing,
/// a build or more after it was built.
type PoseKey = [u32; 7];

fn pose_key(pose: &HeadPose) -> PoseKey {
    let mut key = [0; 7];
    for (k, v) in pose.orientation.iter().chain(&pose.position).enumerate() {
        key[k] = v.to_bits();
    }
    key
}

/// Where the driven machine aimed in each eye at the last few view builds, by the frame's pose.
static AIMS: Mutex<[Option<(PoseKey, [[f32; 2]; 2])>; 4]> = Mutex::new([None; 4]);
static AIM_NEXT: AtomicUsize = AtomicUsize::new(0);

/// Where the game camera `own` aims in each eye (tangents: x right, y up from the eye's straight
/// ahead), the point `distance` units along its forward axis seen from the eyes the engine puts
/// `separation` units either side of the drawn view `drawn` (the left eye along its left axis;
/// both origin, then axis rows: forward, left, up). None if the point is not ahead of an eye.
pub fn aim_tangents(own: &[f32; 12], drawn: &[f32; 12], separation: f32, distance: f32) -> Option<[[f32; 2]; 2]> {
    let dot = |a: [f32; 3], b: &[f32]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let (forward, left, up) = (&drawn[3..6], &drawn[6..9], &drawn[9..12]);
    let aim: [f32; 3] = std::array::from_fn(|k| own[k] + distance * own[3 + k]);
    let mut tangents = [[0.0; 2]; 2];
    for (eye, side) in [1.0f32, -1.0].into_iter().enumerate() {
        let to: [f32; 3] = std::array::from_fn(|k| aim[k] - (drawn[k] + side * separation * left[k]));
        let ahead = dot(to, forward);
        if !(ahead > f32::EPSILON) {
            return None;
        }
        tangents[eye] = [-dot(to, left) / ahead, dot(to, up) / ahead];
    }
    tangents.iter().flatten().all(|v| v.is_finite()).then_some(tangents)
}

/// Where the driven machine aimed in each eye at the view build of the frame with head pose
/// `pose` ([`aim_tangents`]), if that build is still kept and a machine was driven then.
pub fn aim_for(pose: &HeadPose) -> Option<[[f32; 2]; 2]> {
    let key = pose_key(pose);
    AIMS.locked().iter().flatten().find(|(k, _)| *k == key).map(|&(_, tangents)| tangents)
}

fn keep_aim(pose: &HeadPose, tangents: Option<[[f32; 2]; 2]>) {
    let at = AIM_NEXT.fetch_add(1, Relaxed) % 4;
    AIMS.locked()[at] = tangents.map(|t| (pose_key(pose), t));
}
/// FSR's scale for the run (none: no FSR), set before the layout is made.
static SETUP_SCALE: OnceLock<Option<f32>> = OnceLock::new();

/// A layout of our own for frame warping: the side-by-side layout's header and left view, and the
/// single layout's unused second view: the engine draws one view, into the left half of the window,
/// the same size as a side-by-side eye; its eye sign says which eye ([`set_layout_eye`]). Made once
/// and kept for the process (the engine holds the pointer until the restore puts its own back). Its
/// first view's address, as the layout pointer holds it.
fn one_eye_layout(exe: &Module, build: &Build) -> Option<usize> {
    if let Some(layout) = ONE_EYE.get() {
        return Some(*layout);
    }
    let record = |at: usize| mem::read::<Record>(exe.at(at));
    let header = record(build.layout_left_right.0 - engine::LAYOUT_HEADER)?;
    let left = record(build.layout_left_right.0)?;
    let unused = record(build.layout_single.0 + engine::LAYOUT_RECORD)?;
    let mut view = left;
    if let Some(scale) = SETUP_SCALE.get().copied().flatten() {
        // The view's width and height fractions (after name, player, x, y): smaller by the scale.
        for at in [0x14, 0x18] {
            let fraction = f32::from_le_bytes(view[at..at + 4].try_into().ok()?) / scale;
            view[at..at + 4].copy_from_slice(&fraction.to_le_bytes());
        }
    }
    let records: &'static mut [Record; 4] = Box::leak(Box::new(one_eye_records(header, view, unused)));
    let first_view = records[1].as_ptr() as usize;
    Some(*ONE_EYE.get_or_init(|| first_view))
}

/// The one-eye layout's view draws eye `eye` (0 left): its eye sign, -1 left and 1 right, as the
/// side-by-side layout's views have it.
fn set_layout_eye(eye: usize) {
    if let Some(view) = ONE_EYE.get() {
        let sign: f32 = if eye == 0 { -1.0 } else { 1.0 };
        mem::write(view + engine::VIEW_EYE_SIGN, sign);
    }
}

/// The one-eye layout's records from the side-by-side layout's `header` and `left` view and the
/// single layout's `unused` view: the header, the left view, the unused view, and the header again
/// to end it. The engine walks a layout's records while each one's name is empty (every view
/// record's is "") and stops at the first with a name (the next layout's header): the end must be
/// a record whose name points at a real, non-empty name.
fn one_eye_records(header: Record, left: Record, unused: Record) -> [Record; 4] {
    [header, left, unused, header]
}

/// Whether this run warps frames (AFW): chosen, with stereo, head tracking and the headset's own
/// frustum on (the eye's projection is made here), on a build whose scene depth is mapped.
pub fn warps(config: &EyeSettings, shared: &Shared, build: &Build) -> bool {
    config.stereo && config.stereo_method == StereoMethod::Afw && shared.head_tracking && shared.headset_fov && build.scene_depth.is_some()
}

/// The FSR scale (eye size over render size per side) this run upscales by: with frame warping and
/// FSR chosen.
pub fn fsr_scale(config: &EyeSettings, shared: &Shared, build: &Build) -> Option<f32> {
    config.fsr.filter(|_| warps(config, shared, build)).map(|fsr| fsr.scale)
}

/// Where the eye drawn with frame warping lands in the window (x, y from the top-left, width,
/// height: fractions), its view at `1 / scale` of a side-by-side eye's size (1: the left half).
/// A smaller view keeps to the window's bottom-left: GL counts its viewport from the bottom.
pub fn warp_region(scale: f32) -> [f32; 4] {
    let (w, h) = (0.5 / scale, 1.0 / scale);
    [0.0, 1.0 - h, w, h]
}

/// Switches the engine to side-by-side stereo with our eye separation, remembering what it had
/// (on the run's ledger, put back at the stop).
fn enter_stereo(start: &mut Start<'_, Wolf>, config: &EyeSettings) -> Result<(), Lack> {
    let shared = *start.shared;
    let Facts { exe, build } = *start.build;
    let known = [build.layout_single, build.layout_left_right];
    for layout in known {
        if !engine::layout_present(&exe, layout) {
            return Err(Lack::Missing(format!("layout {} is not where this build keeps it", layout.1)));
        }
    }
    for cvar in [&build.stereo_separation, &build.stereo_screen_separation] {
        if !cvar.present(&exe) {
            return Err(Lack::Missing(format!("cvar {} is not where this build keeps it", cvar.name)));
        }
    }
    let before = engine::current_layout(&exe, build, &known).ok_or_else(|| Rejection::not_ready("the engine is in another layout (split screen?)"))?;
    log!("layout was {}", before.1);
    let warping = warps(config, &shared, build);
    if config.stereo_method == StereoMethod::Afw && !warping {
        log!("frame warping needs head tracking, the headset's own frustum and this build's scene depth; side by side instead");
    }
    // Kept, not set: the view builder writes the separation every frame.
    let separation_before = start.ledger.keep(eng_idt::FloatCvar { exe, cvar: &build.stereo_separation });
    let screen_before = start.ledger.keep(eng_idt::FloatCvar { exe, cvar: &build.stereo_screen_separation });
    log!("stereo separation was {separation_before:?}, screen separation {screen_before:?}");
    let separation = config.engine_separation(&shared, &HeadPose::default());
    if !build.stereo_separation.set_float(&exe, separation) || !build.stereo_screen_separation.set_float(&exe, config.screen_separation) {
        return Err(Rejection::unsupported("the stereo cvars are not writable").into());
    }
    let _ = SETUP_SCALE.set(fsr_scale(config, &shared, build));
    let layout = match warping {
        true => one_eye_layout(&exe, build).ok_or_else(|| Rejection::unsupported("the layouts could not be read to make the one-eye layout"))?,
        false => exe.at(build.layout_left_right.0),
    };
    if start.ledger.set(LayoutSetting { exe, build }, layout) == Change::Failed {
        return Err(Rejection::unsupported("the layout pointer is not writable").into());
    }
    WARPING.store(warping, Release);
    match warping {
        true => log!(
            "frame warping on: one eye a frame ({}), {separation} units from the centre to each eye",
            if config.afw_dominant { "the dominant eye, the other every 8th" } else { "the eyes taking turns" }
        ),
        false => log!("side-by-side stereo on: {separation} units from the centre to each eye, screen separation {}", config.screen_separation),
    }
    Ok(())
}

pub static VIEW_BUILDER_ORIGINAL: Original<ViewBuilderFn> = Original::new();
pub static PROJECTION_ORIGINAL: Original<ProjectionFn> = Original::new();
pub static HANDS_FOV_SCALE_ORIGINAL: Original<HandsFovScaleFn> = Original::new();
pub static HANDS_DEPTH_HACK_ORIGINAL: Original<HandsDepthHackFn> = Original::new();

/// With frame warping the hands' depth is left true (none of the squeeze that keeps the world from
/// clipping them): their items take it from here each frame, and the hands' own model is given it
/// here too (it takes the cvar only when shown), so the other eye warps them by where they are.
pub unsafe extern "system" fn hands_depth_hack(hands: usize) -> f32 {
    let flight = InFlight::enter();
    if WARPING.load(Acquire) && crate::on(crate::EYES_AT) {
        if let Some(entity) = mem::read::<usize>(hands + engine::HANDS_PRESENTABLE)
            .filter(|p| *p != 0)
            .and_then(|presentable| mem::read::<usize>(presentable + engine::PRESENTABLE_RENDER_ENTITY))
            .filter(|p| *p != 0)
        {
            mem::write(entity + engine::RENDER_ENTITY_DEPTH_HACK, 1.0f32);
        }
        return 1.0;
    }
    // SAFETY: forwards the game's own call.
    unsafe { HANDS_DEPTH_HACK_ORIGINAL.get(&flight)(hands) }
}

/// While the eyes are on, the hands and their weapon are drawn with the eye's own field of view
/// (scale 1): they are at the controllers, and the scene's depth has them there (the depth pass
/// draws them unscaled), so the eye warped from a frame puts them where the real eye does.
pub unsafe extern "system" fn hands_fov_scale(hands: usize) -> f32 {
    let flight = InFlight::enter();
    if crate::on(crate::EYES_AT) {
        return 1.0;
    }
    // SAFETY: forwards the game's own call.
    unsafe { HANDS_FOV_SCALE_ORIGINAL.get(&flight)(hands) }
}


struct Setup {
    config: EyeSettings,
    shared: Shared,
    exe: Module,
    build: &'static Build,
    /// Sight lock on and the resting field of view readable.
    sight_lock: bool,
    make_frustum: MakeFrustumFn,
    /// The scene depth's destination pointer (absolute), where mapped.
    scene_depth: Option<usize>,
}

static SETUP: OnceLock<Setup> = OnceLock::new();
/// The game thread while it is inside the view builder (0 otherwise), and the projections built
/// in that call so far (0: left eye, 1: right).
static BUILDING_THREAD: AtomicU32 = AtomicU32::new(0);
static EYE_CALLS: AtomicU32 = AtomicU32::new(0);
/// The pose of the frame being built.
static FRAME_POSE: Mutex<Option<HeadPose>> = Mutex::new(None);
/// View builds and the player views turned at them (the engine draws both eyes from each).
static BUILDS: Tally = Tally::new("view builds");
static PROJECTED: AtomicU64 = AtomicU64::new(0);
static LAST_COMMANDS: AtomicU64 = AtomicU64::new(0);
/// Views built with the eyes at least partly locked to the game camera (aiming down sights).
static SIGHTED: AtomicU64 = AtomicU64::new(0);
/// Frame warping: one eye drawn a frame ([`warps`]).
static WARPING: AtomicBool = AtomicBool::new(false);
/// The eye this view build draws (0 left), with frame warping.
static FRAME_EYE: AtomicUsize = AtomicUsize::new(0);
/// The projection the engine made for this build's eye (its frustum maker's matrix, game units),
/// its near distance, the eye's frustum before any jitter and the jitter (render pixels).
static FRAME_PROJECTION: Mutex<Option<EyeProjection>> = Mutex::new(None);
/// An eye's projection: the engine's matrix, its near distance, the frustum before any jitter and
/// the jitter.
type EyeProjection = ([f32; 16], f32, Frustum, [f32; 2]);
/// Each eye's next FSR jitter phase.
static JITTER_PHASE: [AtomicU32; 2] = [AtomicU32::new(0), AtomicU32::new(0)];
/// DLSS with both eyes drawn each frame ([`arm_dlss`]): its quality, once armed.
static DLSS: OnceLock<monaka_dlss11::Quality> = OnceLock::new();
/// The DLSS jitter phase of this view build (both eyes share it).
static DLSS_PHASE: AtomicU32 = AtomicU32::new(0);
/// The projections the engine made for this build's eyes with DLSS (as [`FRAME_PROJECTION`]).
static DLSS_PROJECTIONS: Mutex<[Option<EyeProjection>; 2]> = Mutex::new([None; 2]);
/// Each frame built with DLSS: its pose and how both eyes were drawn, in step with the loop's
/// tickets (as [`WARP_TICKETS`]).
static DLSS_TICKETS: Mutex<Tickets<Option<GlDlssFrame>>> = Mutex::new(Tickets::new(4));
static DLSS_SHOWING: Mutex<Option<monaka_core::ticket::Ticket<Option<GlDlssFrame>>>> = Mutex::new(None);
/// Frames built with DLSS without how both eyes were drawn (logged the first few).
static DLSS_UNRENDERED: AtomicU64 = AtomicU64::new(0);
/// Each frame built with frame warping: its eye, pose and how it was rendered, in step with the
/// loop's tickets (one each per built frame, taken at the same `SwapBuffers`).
static WARP_TICKETS: Mutex<Tickets<WarpFacts>> = Mutex::new(Tickets::new(4));
/// How a frame built with frame warping was rendered, and its jitter (none: not one to warp).
type WarpFacts = Option<(Rendered, [f32; 2])>;
/// The ticket taken at the last `SwapBuffers`: the frame the next one shows ([`warp_shot`]).
static SHOWING: Mutex<Option<monaka_core::ticket::Ticket<WarpFacts>>> = Mutex::new(None);
/// With frame warping the eye other than the dominant one is drawn every this many frames (its
/// gaps are filled from its own frame for that long: `monaka_stereo::afw::Rules::every`).
pub const REFRESH: u64 = 8;
/// Frames that went to the warp without how they were rendered (logged the first few).
static UNRENDERED: AtomicU64 = AtomicU64::new(0);
pub fn install(config: EyeSettings, shared: Shared, exe: Module, build: &'static Build) -> Result<(), String> {
    let make_frustum = build.make_frustum.function(&exe).map_err(|_| "the frustum maker is not where this build keeps it".to_string())?;
    let sight_lock = config.sight_lock && build.fov.present(&exe);
    if config.sight_lock && !sight_lock {
        log!("{} is not where this build keeps it; no sight lock", build.fov.name);
    }
    let scene_depth = build.scene_depth.as_ref().map(|depth| exe.at(depth.destination));
    SETUP.set(Setup { config, shared, exe, build, sight_lock, make_frustum, scene_depth }).map_err(|_| "installed twice".to_string())
}

/// View builds so far (0 while the game is paused in a menu).
pub fn builds() -> u64 {
    BUILDS.calls.load(Relaxed)
}

pub fn summary() -> String {
    format!("{}, {} aiming down sights, {} eye projections
{}", BUILDS.summary(), SIGHTED.load(Relaxed), PROJECTED.load(Relaxed), crate::providers::head_conformance::summary())
}

/// A player's render view as the engine keeps it: origin and axis rows (`[origin, forward, left,
/// up]`, world units) and field of view (degrees).
#[derive(Clone, Copy, Debug, PartialEq)]
struct View {
    origin_axis: [f32; 12],
    fov: [f32; 2],
}

impl View {
    fn read(view: usize) -> Option<Self> {
        Some(Self { origin_axis: mem::read(view + engine::VIEW_ORIGIN_AXIS)?, fov: mem::read(view + engine::VIEW_FOV)? })
    }
}

impl eng_idt::view::RenderView for View {
    fn write(&self, view: usize) -> bool {
        mem::write(view + engine::VIEW_ORIGIN_AXIS, self.origin_axis) && mem::write(view + engine::VIEW_FOV, self.fov)
    }
}

/// A render view's origin and axis rows (`[origin, forward, left, up]`) as a canonical camera.
fn camera_of(origin_axis: &[f32; 12], convention: &CameraConvention) -> Option<Mat34> {
    let (origin, axis) = origin_axis.split_first_chunk::<3>()?;
    eng_idt::view::camera(convention, origin, axis.try_into().ok()?)
}

/// The view the engine splits into the eyes (it moves each eye along the view's left axis by the
/// separation itself) for the player's render view `own`: the recipe's view centre ([`EyeView`]:
/// turned and moved by the head `pose`, its position at `position` (metres, as
/// [`sights::head_position`] places it), the head's share `baked` in the game camera swapped for its
/// current turn), the dominant eye moved onto it ([`sights::put_dominant_eye`]), and the field of
/// view that contains both eyes' frusta with `headset_fov` (what the engine culls with), else the
/// game's.
#[allow(clippy::too_many_arguments)]
fn turned_view(
    own: &View,
    pose: &HeadPose,
    position: [f32; 3],
    config: &EyeSettings,
    shared: &Shared,
    convention: &CameraConvention,
    headset_fov: bool,
    baked: Option<Share>,
) -> Option<View> {
    let camera = camera_of(&own.origin_axis, convention)?;
    // The position goes in as placed here (`Full` keeps it); only the view centre is taken, so
    // the eye asked for does not matter.
    let placed = HeadPose { position, ..*pose };
    let shared_here = Shared { head_position: HeadPosition::Full, headset_fov, ..*shared };
    let base = Base::game_or_swap(baked.map(|share| [share.yaw, share.pitch]));
    let centre = EyeView::of(&camera, base, &placed, 0, &shared_here, false, own.fov.map(f32::to_radians))?.centre;
    let (origin, axis) = eng_idt::view::origin_axis(convention, &centre)?;
    let fov = if headset_fov {
        let [h, v] = Frustum::both_eyes(&pose.fov)?.half_extent();
        [2.0 * h.atan().to_degrees(), 2.0 * v.atan().to_degrees()]
    } else {
        own.fov
    };
    if !eng_idt::view::plausible(convention, &origin, &axis, fov.map(f32::to_radians)) {
        return None;
    }
    let mut origin_axis = [0.0f32; 12];
    origin_axis[..3].copy_from_slice(&origin);
    origin_axis[3..].copy_from_slice(&axis);
    sights::put_dominant_eye(&mut origin_axis, config.dominant_eye, config.engine_separation(shared, pose));
    Some(View { origin_axis, fov })
}

/// Where the headset's tracking space sits in the world (canonical, metres) for the game camera
/// `origin_axis` (the player's first-person view: origin, then axis rows): the view the eyes are
/// built from (levelled as the eyes' is, the head's turn swapped in, the dominant eye on the camera)
/// with the head's own pose taken back out. A controller's pose lands in the world through it where
/// the eyes see it; with head position Lean or Off the hands follow the drawn head, not the real one.
pub fn tracking_space(origin_axis: &[f32; 12], pose: &HeadPose) -> Option<Mat34> {
    let setup = SETUP.get()?;
    let convention = eng_idt::camera_convention(setup.config.units_per_metre);
    // The field of view only has to look like one: it is not used.
    let own = eyes_camera(View { origin_axis: *origin_axis, fov: [80.0, 50.0] });
    let drawn = turned_view(&own, pose, eyes_position(&setup.shared, pose, 0.0), &setup.config, &setup.shared, &convention, false, look::baked::<Wolf>(None))?;
    let eye = camera_of(&drawn.origin_axis, &convention)?;
    // The hands are tracked against the real head, which the drawn view stands for.
    Some(camera::tracking_space(&eye, pose.orientation, pose.position))
}

/// View builds this far apart have a gap between them (a video, a level load, a pause).
const BUILD_GAP: std::time::Duration = std::time::Duration::from_millis(500);
/// When the last view build was, and the times the view slots were made fresh after a gap.
static LAST_BUILD: Mutex<Option<std::time::Instant>> = Mutex::new(None);
static FRESHENED: AtomicU64 = AtomicU64::new(0);

/// Before the view builder `manager` fills its view slots after a gap in view builds (and at the
/// first): every slot after the first has its environment blend lists emptied. The game's own
/// layout fills only the first slot; a level load frees what the others' lists hold without
/// emptying them, and the builder's next copy into them wrote into freed memory (the game
/// crashed). Emptied, they are made anew; their old storage is left, being gone already.
fn fresh_view_slots(manager: usize, slots: &engine::ScreenViews) {
    let now = std::time::Instant::now();
    let gap = LAST_BUILD.locked().replace(now).is_none_or(|last| now.duration_since(last) > BUILD_GAP);
    if !gap {
        return;
    }
    let list = manager + slots.list;
    let Some(storage) = mem::read::<usize>(list).filter(|&p| p != 0) else { return };
    let capacity = mem::read::<i32>(list + 0xc).unwrap_or(0).clamp(0, 16) as usize;
    for slot in 1..capacity {
        let env = storage + slot * slots.stride + slots.env_lists;
        for k in 0..engine::ENV_LISTS {
            let at = env + k * engine::ENV_LIST_SIZE;
            mem::write(at, 0usize);
            mem::write(at + 8, [0i32; 2]);
        }
    }
    if FRESHENED.fetch_add(1, Relaxed) < 8 {
        log!("view slots made fresh after a gap in view builds ({} slots after the first)", capacity.saturating_sub(1));
    }
}

pub unsafe extern "system" fn view_builder(this: usize, views: usize, a: usize, b: usize, c: usize, d: usize) -> usize {
    let flight = InFlight::enter();
    let build = BUILDS.call();
    GAME_THREAD.store(thread_id(), Relaxed);
    crate::providers::pad_connected::frame();
    crate::cooking::frame_rate::frame();
    crate::research::profile::game_thread();
    crate::providers::held_item::watch_build(build);
    crate::providers::settings::freeze_frame();
    if build < 6 || build.is_multiple_of(600) {
        let commands = crate::providers::look::commands();
        let previous = LAST_COMMANDS.swap(commands, Relaxed);
        log!(
            "view build {build} on thread {} (commands on {}), {} commands since the last logged build",
            thread_id(),
            crate::providers::look::COMMAND_THREAD.load(Relaxed),
            commands - previous
        );
    }
    let setup = SETUP.get();
    // One head sample a frame: the one the frame's command took (the game camera carries its
    // share, and the engine draws the world from that camera's turn), so the picture and the pose
    // it goes out with are the same moment's; a newer sample here would label the frame a few
    // milliseconds ahead of what it shows (a copy ahead in a fast turn). With no head in the command
    // (another camera's view, a pause), the newest sample.
    let pose = match setup {
        Some(_) if HEAD_TRACKED.load(Acquire) && crate::on(crate::EYES_AT) => look::command::head_baked().or_else(|| PAIRS.head().current()),
        _ => None,
    };
    let mut saved = Saved::<View, { engine::PLAYER_VIEWS }>::new();
    let warping = WARPING.load(Acquire);
    let eye = warp_eye(build, setup.filter(|s| s.config.afw_dominant).map(|s| s.config.dominant_eye));
    let mut eye_view = None;
    let dlss = DLSS.get().copied().filter(|_| !warping);
    let mut dlss_eyes = [None; 2];
    if let Some(quality) = dlss {
        DLSS_PHASE.store(DLSS_PHASE.load(Relaxed).wrapping_add(1) % monaka_dlss11::phases(quality), Relaxed);
    }
    if let (Some(setup), Some(pose)) = (setup, pose.as_ref()) {
        let convention = eng_idt::camera_convention(setup.config.units_per_metre);
        let players = Players { first: views, stride: setup.build.player_view_stride, active: engine::PLAYER_ACTIVE, render_view: engine::PLAYER_RENDER_VIEW };
        for (player, view) in players.active(engine::PLAYER_VIEWS) {
            let Some(own) = View::read(view) else { continue };
            if player == 0 {
                *RENDER_VIEW.locked() = Some(own.origin_axis);
            }
            let baked = look::baked::<Wolf>(None);
            crate::research::view::seen(build, player, own.fov, &own.origin_axis, baked);
            if player == 0
                && let Some(share) = baked
            {
                // The forward row's height: the game camera's pitch.
                command::note_camera_pitch(own.origin_axis[5].clamp(-1.0, 1.0).asin().to_degrees(), share);
            }
            if baked.is_some() {
                BUILDS.aimed();
            }
            let sighted = match setup.sight_lock {
                true => setup.build.fov.float(&setup.exe).map_or(0.0, |rest| sights::sighting(own.fov[0], rest)),
                false => 0.0,
            };
            if sighted > 0.0 {
                SIGHTED.fetch_add(1, Relaxed);
            }
            let cutscene = levels_camera();
            let position = eyes_position(&setup.shared, pose, sighted);
            let camera = eyes_camera(own);
            let turned = turned_view(&camera, pose, position, &setup.config, &setup.shared, &convention, setup.shared.headset_fov, baked).map(|mut turned| {
                crate::research::view::with_extra_yaw(&mut turned.origin_axis);
                turned
            });
            match turned {
                Some(turned) if saved.write(player, view, &turned, own) => {
                    BUILDS.drawn();
                    if player == 0 {
                        // Where the engine puts the eye drawn: the view moved along its left axis.
                        let separation = setup.config.engine_separation(&setup.shared, pose);
                        eye_view = Some(at_eye(turned, eye, separation).origin_axis);
                        if dlss.is_some() {
                            dlss_eyes = [0, 1].map(|e| Some(at_eye(turned, e, separation).origin_axis));
                        }
                        // Where a driven machine aims, for its crosshair (`hud_pieces`).
                        let aim = crate::providers::camera_taken::driving()
                            .then(|| aim_tangents(&own.origin_axis, &turned.origin_axis, separation, AIM_DISTANCE * setup.config.units_per_metre))
                            .flatten();
                        keep_aim(pose, aim);
                        // The head's shift and turn from the camera the eyes are built on, for
                        // what is worn on the view to follow (and the arms, on the player's own).
                        let own_camera = !crate::providers::camera_taken::controlled() && !crate::providers::camera_taken::camera_away();
                        let shift: [f32; 3] = std::array::from_fn(|k| turned.origin_axis[k] - own.origin_axis[k]);
                        crate::providers::follow_head::set_shift(Some(shift), own_camera);
                        let rows = |v: &View| std::array::from_fn::<f32, 9, _>(|k| v.origin_axis[3 + k]);
                        crate::providers::follow_head::set_turn(Some((rows(&own), rows(&turned))));
                        // Whether the drawn view follows the real head: the tracking space it
                        // gives, while the player's own camera has the view.
                        let space = (!cutscene && !crate::providers::camera_taken::controlled())
                            .then(|| camera_of(&turned.origin_axis, &convention))
                            .flatten()
                            .map(|drawn| camera::tracking_space(&drawn, pose.orientation, pose.position))
                            .zip(camera_of(&own.origin_axis, &convention));
                        crate::research::view::trace(build, pose, baked, &own.origin_axis, &turned.origin_axis);
                        crate::providers::head_conformance::note(space, crate::providers::head_conformance::tilt(pose.orientation), crate::providers::look::commands());
                    }
                }
                _ => BUILDS.refused(build, format_args!("player {player}: fov {:?}, origin and axis {:?}", own.fov, own.origin_axis)),
            }
        }
        setup.build.stereo_separation.set_float(&setup.exe, setup.config.engine_separation(&setup.shared, pose));
    }
    let turned = saved.any();
    *FRAME_POSE.locked() = if turned { pose } else { None };
    FRAME_EYE.store(eye, Relaxed);
    if warping {
        set_layout_eye(eye);
    }
    if let Some(slots) = setup.and_then(|s| s.build.screen_views.as_ref()) {
        fresh_view_slots(this, slots);
    }
    *FRAME_PROJECTION.locked() = None;
    *DLSS_PROJECTIONS.locked() = [None; 2];
    EYE_CALLS.store(0, Relaxed);
    BUILDING_THREAD.store(thread_id(), Release);
    // SAFETY: forwards the game's own call.
    let result = unsafe { VIEW_BUILDER_ORIGINAL.get(&flight)(this, views, a, b, c, d) };
    BUILDING_THREAD.store(0, Release);
    saved.put_back();
    if turned && let Some(pose) = pose {
        match warping {
            true => {
                let projection = FRAME_PROJECTION.locked().take();
                let units = setup.map_or(1.0, |s| s.config.units_per_metre);
                let rendered = eye_view.zip(projection).and_then(|(origin_axis, (gl, near, frustum, jitter))| {
                    let convention = eng_idt::camera_convention(units);
                    let mut rendered = warp_rendered(&camera_of(&origin_axis, &convention)?, &gl, near, units)?;
                    // The label keeps the projection before the jitter (an upscaler's output is unjittered).
                    frustum.apply_to_projection(&mut rendered.projection);
                    Some((rendered, jitter))
                });
                WARP_TICKETS.locked().issue(Ticket::Eye(eye), pose, rendered);
                PAIRS.built(Ticket::Eye(eye), pose);
            }
            false => {
                if dlss.is_some() {
                    let units = setup.map_or(1.0, |s| s.config.units_per_metre);
                    let projections = *DLSS_PROJECTIONS.locked();
                    let frame = dlss_frame_of(dlss_eyes, projections, units);
                    DLSS_TICKETS.locked().issue(Ticket::Both, pose, frame);
                }
                PAIRS.built(Ticket::Both, pose)
            }
        }
    }
    result
}

/// Both eyes as drawn for DLSS: each eye's place (`origins`, the engine's origin and axis rows) and
/// the projection the engine made for it; none unless both are there with the same jitter.
fn dlss_frame_of(origins: [Option<[f32; 12]>; 2], projections: [Option<EyeProjection>; 2], units: f32) -> Option<GlDlssFrame> {
    let convention = eng_idt::camera_convention(units);
    let mut jitter = None;
    let mut views = [None; 2];
    for eye in 0..2 {
        let (gl, near, frustum, eye_jitter) = projections[eye]?;
        if jitter.is_some_and(|j| j != eye_jitter) {
            return None;
        }
        jitter = Some(eye_jitter);
        let mut rendered = warp_rendered(&camera_of(&origins[eye]?, &convention)?, &gl, near, units)?;
        // The projection before the jitter: DLSS's output is unjittered.
        frustum.apply_to_projection(&mut rendered.projection);
        views[eye] = Some(monaka_warp::EyeView { camera: rendered.camera, projection: rendered.projection });
    }
    Some(GlDlssFrame { views: [views[0]?, views[1]?], jitter: jitter? })
}

/// DLSS for this run, with both eyes drawn each frame: each eye's projection jittered, and how both
/// were drawn kept for each frame ([`dlss_frame`]).
pub fn arm_dlss(quality: monaka_dlss11::Quality) {
    let _ = DLSS.set(quality);
}

/// At the frame's `SwapBuffers`: how both eyes of the frame being presented were drawn (its ticket
/// taken in step with the loop's, as [`warp_shot`]), or none for a frame without that.
pub fn dlss_frame() -> Option<GlDlssFrame> {
    let newest = DLSS_TICKETS.locked().show().ticket;
    let ticket = std::mem::replace(&mut *DLSS_SHOWING.locked(), newest)?;
    let frame = ticket.extra;
    if frame.is_none() && DLSS_UNRENDERED.fetch_add(1, Relaxed) < 5 {
        log!("DLSS: a frame without both eyes' cameras and projections; resampled");
    }
    frame
}

/// The eye frame warping draws at view build `n`: the eyes take turns, or (`dominant`, see
/// `EyeSettings::afw_dominant`) the dominant eye with the other every [`REFRESH`]th build.
pub fn warp_eye(n: u64, dominant: Option<DominantEye>) -> usize {
    let dominant = match dominant {
        Some(DominantEye::Left) => 0,
        Some(DominantEye::Right) => 1,
        _ => return (n % 2) as usize,
    };
    if n % REFRESH == REFRESH - 1 { 1 - dominant } else { dominant }
}

/// Whether the eyes are built on the game camera levelled: while another camera has the view (a
/// cutscene, a grab, a climb), so the head alone tilts the view up and down. A mounted gun's or a
/// driven machine's keeps its pitch, which the stick aims.
fn levels_camera() -> bool {
    crate::providers::camera_taken::controlled() && !crate::providers::camera_taken::mounted()
}

/// Where the eyes sit against the camera they are built on for the head's `pose`, `sighted` into
/// aiming down sights ([`sights::head_position`]): on that camera while [`levels_camera`]. A
/// cutscene's camera sits where the player's head was drawn, its own head hidden; the real head's
/// steps and leans would carry the eyes out of the body, so only its turn is taken.
fn eyes_position(shared: &Shared, pose: &HeadPose, sighted: f32) -> [f32; 3] {
    if levels_camera() { [0.0; 3] } else { sights::head_position(shared, pose, sighted) }
}

/// The camera the eyes are built on for the game camera `own` (levelled while
/// [`levels_camera`]).
fn eyes_camera(own: View) -> View {
    if levels_camera() { levelled(own) } else { own }
}

/// The view `view` levelled: its place and the way it faces kept, its pitch and roll dropped (id
/// axes: forward, left, up rows, up +z). Its facing is its forward axis on the ground, with its up
/// axis leaning back (looking up) or forward (looking down) taking over as it steepens: steady
/// through straight up and down, where the forward axis alone flips round with any tip past them
/// (a rope climb's camera, each step).
fn levelled(mut view: View) -> View {
    let a = view.origin_axis;
    let (forward_up, x, y) = (a[5], a[3], a[4]);
    let (x, y) = (x - forward_up * a[9], y - forward_up * a[10]);
    let n = (x * x + y * y).sqrt();
    if n < 1e-3 {
        return view;
    }
    let (fx, fy) = (x / n, y / n);
    view.origin_axis[3..].copy_from_slice(&[fx, fy, 0.0, -fy, fx, 0.0, 0.0, 0.0, 1.0]);
    view
}

/// The view `view` the engine splits into the eyes (each `separation` units along its left axis),
/// moved onto eye `eye` (0 left): where the engine draws that eye from.
fn at_eye(mut view: View, eye: usize, separation: f32) -> View {
    let side = if eye == 0 { separation } else { -separation };
    for k in 0..3 {
        view.origin_axis[k] += side * view.origin_axis[6 + k];
    }
    view
}

/// How an eye was rendered, for the warp: its `camera` (canonical, metres) and the engine's
/// projection for it (`gl`: the frustum maker's row-major matrix, GL depth -1..1, game units,
/// `near` its near distance in game units) made canonical: depth 0..1 (with GL's default depth
/// range the window depth is what that projection writes) and metres (`units` game units a metre).
pub fn warp_rendered(camera: &Mat34, gl: &[f32; 16], near: f32, units: f32) -> Option<Rendered> {
    let mut projection = *gl;
    for column in 0..4 {
        projection[8 + column] = 0.5 * (gl[8 + column] + gl[12 + column]);
    }
    projection[11] /= units;
    Rendered::new(*camera, projection, near / units)
}

/// At the frame's `SwapBuffers`: the frame warping shot of the frame being presented (its ticket
/// taken in step with the loop's), or none for a frame without one (not built with an eye, or
/// without how it was rendered).
pub fn warp_shot() -> Option<GlShot> {
    // The image a `SwapBuffers` shows is the frame built one view build before the newest ticket:
    // the ticket taken at the previous `SwapBuffers`.
    let newest = WARP_TICKETS.locked().show().ticket;
    let ticket = std::mem::replace(&mut *SHOWING.locked(), newest)?;
    let Ticket::Eye(eye) = ticket.view else { return None };
    let Some((rendered, jitter)) = ticket.extra else {
        if UNRENDERED.fetch_add(1, Relaxed) < 5 {
            log!("frame warping: a frame of eye {eye} without its camera or projection; not warped");
        }
        return None;
    };
    let setup = SETUP.get()?;
    let separation = setup.shared.separation_for(&ticket.pose);
    let other = <Wolf as EyeGeometry>::frustum(&ticket.pose, 1 - eye);
    Some(GlShot { shot: Shot::labelled(Pending { eye, pose: ticket.pose }, rendered, separation, other), jitter })
}

/// FSR's jitter phases for `scale` (eye size over render size per side): AMD's count, 8 times the
/// pixel ratio.
pub fn fsr_phases(scale: f32) -> u32 {
    ((8.0 * scale * scale).ceil() as u32).max(1)
}

/// FSR's jitter at `phase` (render pixels, x right, y down): the Halton (2, 3) point, centred.
pub fn fsr_jitter(phase: u32) -> [f32; 2] {
    let halton = |mut index: u32, base: u32| {
        let (mut fraction, mut result) = (1.0f32, 0.0f32);
        while index > 0 {
            fraction /= base as f32;
            result += fraction * (index % base) as f32;
            index /= base;
        }
        result
    };
    [halton(phase + 1, 2) - 0.5, halton(phase + 1, 3) - 0.5]
}

/// With frame warping, the eye of the frame the render thread draws now (the oldest ticket not
/// shown); none without frame warping.
pub fn drawing_eye() -> Option<usize> {
    if !WARPING.load(Acquire) {
        return None;
    }
    match WARP_TICKETS.locked().drawing()?.view {
        Ticket::Eye(eye) => Some(eye),
        _ => None,
    }
}

/// The near plane the eyes were last drawn with (game units, f32 bits; 0 before the first).
static NEAR_UNITS: AtomicU32 = AtomicU32::new(0);

/// The near plane the eyes were last drawn with, in metres (the depth sent with each pair is
/// standard from it, its far plane infinite).
pub fn near_metres() -> Option<f32> {
    let near = f32::from_bits(NEAR_UNITS.load(Relaxed));
    let units = SETUP.get()?.config.units_per_metre;
    (near > 0.0 && near.is_finite() && units > 0.0).then(|| near / units)
}

/// The scene's depth texture now (`engine::SceneDepth`), behind the frame being presented.
pub fn scene_depth() -> Option<GlDepth> {
    static SEEN: AtomicBool = AtomicBool::new(false);
    let global = SETUP.get()?.scene_depth?;
    let destination = mem::read::<usize>(global).filter(|p| *p != 0)?;
    let image = mem::read::<usize>(destination + engine::DESTINATION_DEPTH).filter(|p| *p != 0)?;
    let texture = mem::read::<u32>(image + engine::IMAGE_TEXTURE).filter(|t| *t != 0)?;
    let width = mem::read::<i32>(image + engine::IMAGE_WIDTH)?;
    let height = mem::read::<i32>(image + engine::IMAGE_HEIGHT)?;
    let samples = mem::read::<i32>(image + engine::IMAGE_SAMPLES)?;
    let depth = GlDepth { texture, width: u32::try_from(width).ok()?, height: u32::try_from(height).ok()?, multisampled: samples > 1 };
    if !SEEN.swap(true, Relaxed) {
        log!("frame warping: scene depth is GL texture {texture}, {width}x{height}, {samples} samples");
    }
    Some(depth)
}

pub unsafe extern "system" fn projection(fov_x: f32, fov_y: f32, near: f32, far: f32, shift_x: f32, shift_y: f32, out: *mut f32) {
    let flight = InFlight::enter();
    if BUILDING_THREAD.load(Acquire) == thread_id()
        && let Some(setup) = SETUP.get()
        && setup.shared.headset_fov
    {
        let call = EYE_CALLS.fetch_add(1, Relaxed) as usize;
        let warping = WARPING.load(Acquire);
        // With frame warping the one view is the frame's eye.
        let eye = if warping && call == 0 { FRAME_EYE.load(Relaxed) } else { call };
        let pose = *FRAME_POSE.locked();
        if eye < 2
            && (!warping || call == 0)
            && let Some(frustum) = pose.and_then(|p| Frustum::from_fov(p.fov[eye]))
            && near.is_finite()
            && near > 0.0
        {
            // With FSR the eye's projection is shifted by its next jitter (render pixels).
            // With DLSS both eyes are shifted by the build's jitter (render pixels, each eye half the
            // frame).
            let dlss = DLSS.get().filter(|_| !warping);
            let jitter = match (warping, SETUP_SCALE.get().copied().flatten(), gl_frames::frame_size()) {
                (true, Some(scale), Some((width, height))) => {
                    let render = gl_frames::region_size(width, height, warp_region(scale));
                    let phases = fsr_phases(scale);
                    let phase = JITTER_PHASE[eye].fetch_add(1, Relaxed) % phases;
                    Some((fsr_jitter(phase), render))
                }
                (false, _, Some((width, height))) if dlss.is_some() => Some((monaka_dlss11::jitter(DLSS_PHASE.load(Relaxed)), (width / 2, height))),
                _ => None,
            };
            let drawn = jitter.map_or(frustum, |(jitter, render)| jittered_at(frustum, jitter, render));
            let [left, right, bottom, top] = drawn.extents(near);
            PROJECTED.fetch_add(1, Relaxed);
            NEAR_UNITS.store(near.to_bits(), Relaxed);
            // SAFETY: the engine's own frustum maker, with the output the game passed.
            unsafe { (setup.make_frustum)(left, right, bottom, top, near, far, out) };
            if warping || dlss.is_some() {
                // SAFETY: the 4x4 matrix the frustum maker just wrote.
                let matrix = mem::read::<[f32; 16]>(out as usize);
                let projection = matrix.map(|m| (m, near, frustum, jitter.map_or([0.0; 2], |(j, _)| j)));
                match warping {
                    true => *FRAME_PROJECTION.locked() = projection,
                    false => DLSS_PROJECTIONS.locked()[eye] = projection,
                }
            }
            return;
        }
    }
    // SAFETY: forwards the game's own call.
    unsafe { PROJECTION_ORIGINAL.get(&flight)(fov_x, fov_y, near, far, shift_x, shift_y, out) }
}

#[cfg(test)]
pub(crate) mod tests;
