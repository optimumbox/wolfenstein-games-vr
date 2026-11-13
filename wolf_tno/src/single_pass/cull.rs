//! The left eye's culling widened to both eyes (RDRVR's `ViUnionCull`): single pass draws the
//! left eye's scene into both eyes and skips the right eye's draws, so what only the right eye
//! sees (its outer edge, past the left eye's frustum) must be in the left eye's scene.
//!
//! The engine culls a bounds against a view's model-view-projection with its bounds cull
//! ([`CullBoundsFn`], id Tech's `CullBoundsToMVPbits`: true when culled). The left eye's view is
//! known from the view setup ([`ViewSetupFn`]): of the frame's views, the one the eye matrix `e`
//! (left clip to right clip, measured by [`super::draws`]) carries onto the next. A bounds the
//! left eye's view culls is tried again in the right eye's (`e · mvp`, the same object seen from
//! the right eye); seen there, it is not culled. A matrix is the left eye's view when the left
//! eye's view-projection's inverse leaves a model matrix of it (its last row `0 0 0 1`); a light's
//! or a shadow's are not.

use super::eye_matrix::{self, Rows};
use monaka_core::sync::Locked;
use monaka_hook::module::Module;
use monaka_hook::{Hooks, InFlight, Original, Site, mem};
use monaka_producer::log;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::*};
use std::sync::{Mutex, RwLock};

/// `(mvp, bounds (min xyz, max xyz), out planes outside, zero to one) -> culled`.
pub type CullBoundsFn = unsafe extern "system" fn(*const f32, *const f32, *mut u8, u8) -> u8;
/// `(view, ...)`: a view's matrices made from its render view, its view-projection at
/// [`VIEW_MVP`].
pub type ViewSetupFn = unsafe extern "system" fn(usize, usize, usize, usize, usize) -> usize;
pub const VIEW_MVP: usize = 0x34c0;

static CULL: Original<CullBoundsFn> = Original::new();
static VIEW_SETUP: Original<ViewSetupFn> = Original::new();
static SITES: Mutex<Option<(Site<CullBoundsFn>, Site<ViewSetupFn>)>> = Mutex::new(None);
/// Widening on: single pass with an eye matrix.
static ON: AtomicBool = AtomicBool::new(false);

/// The views the culls are judged by.
struct Views {
    /// The eye matrix.
    eye: Option<Rows>,
    /// The last view set up, to pair with the next.
    last: Option<Rows>,
    /// The left eye's view-projection's inverse.
    left_inverse: Option<Rows>,
}

static VIEWS: RwLock<Views> = RwLock::new(Views { eye: None, last: None, left_inverse: None });

/// Counts since the last report: the culls of the left eye's view tried in the right eye's, those
/// seen there (kept), and the views paired.
static TRIED: AtomicU64 = AtomicU64::new(0);
static KEPT: AtomicU64 = AtomicU64::new(0);
static PAIRED: AtomicU64 = AtomicU64::new(0);
static VIEWS_SET_UP: AtomicU64 = AtomicU64::new(0);

/// How close `e ·` the left eye's view-projection must be to the right eye's to pair them.
const PAIRS: f64 = 2e-3;
/// How close the last row of what the left eye's inverse leaves must be to `0 0 0 1`.
const MODEL: f32 = 1e-3;

/// The build's sites ([`crate::providers::engine::Build`]), hooked with the draws.
pub fn sites(cull: Option<Site<CullBoundsFn>>, view_setup: Option<Site<ViewSetupFn>>) {
    *SITES.locked() = cull.zip(view_setup);
}

/// Hooks the cull and the view setup, if mapped.
pub fn install(hooks: &mut Hooks, exe: &Module) {
    let Some((cull, view_setup)) = *SITES.locked() else {
        log!("single pass: the engine's bounds cull is not mapped for this build: the left eye's pass culls for the left eye only (the right eye's edge may miss things)");
        return;
    };
    let result = hooks.site(&CULL, exe, &cull, cull_bounds as CullBoundsFn).and_then(|()| hooks.site(&VIEW_SETUP, exe, &view_setup, view_setup_detour as ViewSetupFn));
    match result {
        Ok(()) => log!("single pass: the left eye's culling widened to both eyes ({cull:?}, {view_setup:?})"),
        Err(e) => log!("single pass: the engine's bounds cull not hooked ({e}): the left eye's pass culls for the left eye only"),
    }
}

/// The eye matrix of the frame (None: single pass off), from the draws.
pub fn eye_matrix(eye: Option<Rows>) {
    ON.store(eye.is_some(), Relaxed);
    if let Ok(mut views) = VIEWS.write() {
        views.eye = eye;
        if eye.is_none() {
            views.left_inverse = None;
        }
    }
}

/// The counts since the last report, for the draws' log.
pub fn report(frames: f64) -> String {
    let take = |a: &AtomicU64| a.swap(0, Relaxed) as f64 / frames;
    format!(
        "the left eye's culling over both eyes: views set up {:.1} a frame, paired {:.1}; the left eye's culled bounds tried in the right eye's view {:.0} a frame, seen there and kept {:.1}",
        take(&VIEWS_SET_UP),
        take(&PAIRED),
        take(&TRIED),
        take(&KEPT)
    )
}

unsafe extern "system" fn view_setup_detour(view: usize, a: usize, b: usize, c: usize, d: usize) -> usize {
    let flight = InFlight::enter();
    // SAFETY: forwards the game's own call.
    let made = unsafe { VIEW_SETUP.get(&flight)(view, a, b, c, d) };
    if ON.load(Relaxed)
        && let Some(mvp) = mem::read::<Rows>(view + VIEW_MVP)
    {
        VIEWS_SET_UP.fetch_add(1, Relaxed);
        if let Ok(mut views) = VIEWS.write() {
            if let (Some(eye), Some(last)) = (views.eye, views.last)
                && eye_matrix::apart(&eye_matrix::carry(&eye, &last), &mvp) <= PAIRS
            {
                views.left_inverse = eye_matrix::inverse(&last);
                PAIRED.fetch_add(1, Relaxed);
            }
            views.last = Some(mvp);
        }
    }
    made
}

unsafe extern "system" fn cull_bounds(mvp: *const f32, bounds: *const f32, outside: *mut u8, zero_to_one: u8) -> u8 {
    let flight = InFlight::enter();
    let original = CULL.get(&flight);
    // SAFETY: forwards the game's own call.
    let culled = unsafe { original(mvp, bounds, outside, zero_to_one) };
    if culled == 0 || !ON.load(Relaxed) || mvp.is_null() {
        return culled;
    }
    let Ok(views) = VIEWS.try_read() else { return culled };
    let (Some(eye), Some(inverse)) = (views.eye, views.left_inverse) else { return culled };
    drop(views);
    // SAFETY: the engine's matrix it culls by, 16 floats.
    let rows: Rows = unsafe { std::ptr::read_unaligned(mvp.cast::<Rows>()) };
    if !of_left_view(&inverse, &rows) {
        return culled;
    }
    TRIED.fetch_add(1, Relaxed);
    let right = eye_matrix::carry(&eye, &rows);
    let mut right_outside = 0u8;
    // SAFETY: the game's cull with the right eye's matrix and the game's bounds.
    let right_culled = unsafe { original(right.as_ptr().cast(), bounds, &mut right_outside, zero_to_one) };
    if right_culled != 0 {
        return culled;
    }
    KEPT.fetch_add(1, Relaxed);
    if !outside.is_null() {
        // SAFETY: the game's output byte.
        unsafe { *outside = right_outside };
    }
    0
}

/// Whether `mvp` is a model's in the left eye's view: `inverse · mvp`'s last row is `0 0 0 1`.
fn of_left_view(inverse: &Rows, mvp: &Rows) -> bool {
    (0..4).all(|j| {
        let terms = (0..4).map(|k| inverse[3][k] * mvp[k][j]);
        let (sum, size) = terms.fold((0.0f32, 0.0f32), |(s, m), t| (s + t, m + t.abs()));
        let target = if j == 3 { 1.0 } else { 0.0 };
        (sum - target).abs() <= MODEL * size.max(1.0)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_model_in_the_left_view_is_known_and_a_light_is_not() {
        // A perspective view-projection, and a model moved and turned in it.
        let projection: Rows = [[1.2, 0.0, 0.1, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, -1.0, -0.2], [0.0, 0.0, -1.0, 0.0]];
        let view: Rows = [[0.0, -1.0, 0.0, 3.0], [0.0, 0.0, 1.0, -1.0], [-1.0, 0.0, 0.0, 2.0], [0.0, 0.0, 0.0, 1.0]];
        let vp = eye_matrix::carry(&projection, &view);
        let model: Rows = [[0.8, 0.6, 0.0, 10.0], [-0.6, 0.8, 0.0, -4.0], [0.0, 0.0, 1.0, 0.5], [0.0, 0.0, 0.0, 1.0]];
        let mvp = eye_matrix::carry(&vp, &model);
        let inverse = eye_matrix::inverse(&vp).unwrap();
        assert!(of_left_view(&inverse, &mvp));
        assert!(of_left_view(&inverse, &vp));
        // Another perspective (a light's) on the same model.
        let light: Rows = [[0.5, 0.0, 0.0, 0.0], [0.0, 0.5, 0.0, 0.0], [0.0, 0.0, -1.0, -0.1], [0.3, 0.2, -1.0, 4.0]];
        assert!(!of_left_view(&inverse, &eye_matrix::carry(&light, &model)));
    }
}
