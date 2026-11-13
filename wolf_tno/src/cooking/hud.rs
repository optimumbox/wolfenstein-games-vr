//! Cooking, waiting for a HUD layer ready-made for OpenGL (placing a 2D layer the game draws per
//! eye by its viewport, as `monaka_stereo::hud_layer::d3d11` places D3D11 draws), reading the HUD layer
//! provider (here the 2D layer function and which eye a record is for, [`record_eye`]) and the
//! frame labels (the pose of the frame being drawn, [`head::rendering`]).
//!
//! The HUD and menus in the headset. The engine draws its 2D layer per eye through one function
//! (`engine::Build::gui_draw`): in play into each eye's own render target after the 3D,
//! in a paused menu straight into that eye's half of the window. It draws with whatever viewport
//! is set. Around it the viewport is replaced by the layer's place in the eye
//! ([`monaka_core::hud::Placement`]: at the eye's straight-ahead point, its own shape in angle,
//! toward the nose so it floats at a set distance), then put back. The engine's own per-eye GUI
//! shift (`stereoRender_guiOffset`) is set to 0.
//!
//! A frame built without views (a paused menu, a loading screen) goes out without a head pose,
//! and the viewer shows such a frame on its flat screen in the room, each eye's image on the one
//! screen. Its layer is then the same in both eyes ([`flat_rect`]): placed for the lenses, the
//! eyes' different straight-ahead points would cross on the screen far nearer than it. Such a
//! frame's views draw no world ([`draws_world`]). Whether a ticket is waiting does not tell: the
//! render thread shows the last frame as it starts the next, so in play the frame being drawn has
//! already given up its ticket.

use crate::cooking::head;
use crate::providers::engine::{self, GuiDrawFn};
use crate::providers::eye_site;
use crate::{Facts, Wolf};
use monaka_core::camera::Frustum;
use monaka_core::feature::Feature;
use monaka_core::hud::{Placement, Rect};
use monaka_contract::manifest::{HUD_DISTANCE, HUD_SIZE};
use monaka_core::options::{Hud, Shared};
use monaka_hook::{InFlight, Original, mem};
use monaka_producer::game::{Component, Lack, Need, On, Start};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering::*};
use windows::Win32::Graphics::OpenGL::{
    GL_COLOR_BUFFER_BIT, GL_COLOR_CLEAR_VALUE, GL_COLOR_WRITEMASK, GL_SCISSOR_BOX, GL_SCISSOR_TEST, GL_VIEWPORT, glClear, glClearColor, glColorMask, glDisable, glEnable,
    glGetBooleanv, glGetFloatv, glGetIntegerv, glIsEnabled, glScissor, glViewport,
};

/// The 2D layer at a depth in each eye: drawn through its own viewport, placed per eye.
pub const HUD: Component<Wolf> = Component {
    name: "hud",
    features: &[Feature::HudLayer],
    options: &[HUD_SIZE.default_number(0.35), HUD_DISTANCE],
    needs: &[Need::On(&eye_site::EYES)],
    on: |run| On::when(run.shared.head_tracking && run.shared.hud.shown, "the HUD is off"),
    install: install_hud,
    summary: Some(summary),
    ..Component::NONE
};

fn install_hud(start: &mut Start<'_, Wolf>) -> Result<(), Lack> {
    let Facts { exe, build } = *start.build;
    install(*start.shared);
    start.hooks.site(&GUI_DRAW_ORIGINAL, &exe, &build.gui_draw, gui_draw)?;
    eng_idt::set_float(start.ledger, &exe, &build.stereo_gui_offset, 0.0, "the engine shifts the HUD per eye as well");
    Ok(())
}

pub static GUI_DRAW_ORIGINAL: Original<GuiDrawFn> = Original::new();

static PLACED: AtomicU64 = AtomicU64::new(0);
static DRAWS: AtomicU64 = AtomicU64::new(0);
static FLAT: AtomicU64 = AtomicU64::new(0);

/// The run's settings: the HUD's size and distance, and the eye distance the eyes are drawn with.
static SHARED: OnceLock<Shared> = OnceLock::new();

pub fn install(shared: Shared) {
    let _ = SHARED.set(shared);
}

pub fn summary() -> String {
    format!(
        "{} of {} 2D layer draws placed, {} of them flat (no world drawn); the laser sight drawn {} times",
        PLACED.load(Relaxed),
        DRAWS.load(Relaxed),
        FLAT.load(Relaxed),
        BEAMS_DRAWN.load(Relaxed)
    )
}

/// The viewport for one eye's 2D layer: `eye` 0 left, 1 right; `rect` the eye's own viewport.
pub fn placed_rect(rect: [i32; 4], eye: usize, frustum: &Frustum, ipd: f32, hud: &Hud) -> Option<[i32; 4]> {
    let [x, y, w, h] = rect.map(|v| v as f32);
    if !(w > 0.0 && h > 0.0 && hud.distance > 0.0) {
        return None;
    }
    // The layer is laid out for the whole side-by-side frame: two eye halves wide.
    let placed = Placement::new(*frustum, eye, ipd, hud.distance).rect(Rect::new(x, 0.0, w, h), 2.0 * w / h, hud.scale);
    // GL viewports go up from the bottom.
    let bottom = y + h - (placed.y + placed.height);
    let placed = [placed.x.round(), bottom.round(), placed.width.round(), placed.height.round()];
    placed.iter().all(|v| v.is_finite() && v.abs() < 32768.0).then(|| placed.map(|v| v as i32))
}

/// The pixels (x right, y up) from the 2D layer's centre to `aim` (tangents in the eye: x right, y
/// up) in an eye viewport `rect` drawn with `frustum`, the layer placed as [`placed_rect`] places it.
pub fn aim_shift(rect: [i32; 4], eye: usize, frustum: &Frustum, ipd: f32, hud: &Hud, aim: [f32; 2]) -> Option<[i32; 2]> {
    let [_, _, w, h] = rect.map(|v| v as f32);
    let (span_x, span_y) = (frustum.right - frustum.left, frustum.up - frustum.down);
    if !(w > 0.0 && h > 0.0 && span_x > 0.0 && span_y > 0.0 && hud.distance > 0.0) {
        return None;
    }
    let centre = Placement::new(*frustum, eye, ipd, hud.distance).centre;
    let shift = [(aim[0] - centre[0]) / span_x * w, (aim[1] - centre[1]) / span_y * h];
    shift.iter().all(|v| v.is_finite() && v.abs() < 32768.0).then(|| shift.map(|v| v.round() as i32))
}

/// The laser sight's colour (linear RGB).
const LASER_COLOUR: [f32; 3] = [1.0, 0.06, 0.03];
/// The beam's width in pixels at the muzzle and at its end, the spot's where it hits, and the
/// pixels between the beam's squares.
const BEAM_WIDE: f32 = 4.0;
const BEAM_THIN: f32 = 2.0;
const SPOT: f32 = 8.0;
const BEAM_STEP: f32 = 1.5;
/// The most squares a beam is drawn with.
const BEAM_SQUARES: usize = 2400;

static BEAMS_DRAWN: AtomicU64 = AtomicU64::new(0);

/// The squares (GL scissor boxes: x, y up from the bottom, width, height) that draw a laser sight's
/// beam with ends `ends` (tangents: the muzzle, then where it stops) in an eye viewport `rect`
/// drawn with `frustum`: tapering from the muzzle, a spot at its end where it hits something.
pub fn beam_boxes(rect: [i32; 4], frustum: &Frustum, ends: [[f32; 2]; 2], hit: bool) -> Option<Vec<[i32; 4]>> {
    let [x, y, w, h] = rect.map(|v| v as f32);
    let (span_x, span_y) = (frustum.right - frustum.left, frustum.up - frustum.down);
    if !(w > 0.0 && h > 0.0 && span_x > 0.0 && span_y > 0.0) {
        return None;
    }
    let pixel = |t: [f32; 2]| [x + (t[0] - frustum.left) / span_x * w, y + (t[1] - frustum.down) / span_y * h];
    let (a, b) = (pixel(ends[0]), pixel(ends[1]));
    let length = (b[0] - a[0]).hypot(b[1] - a[1]);
    if !length.is_finite() {
        return None;
    }
    let steps = ((length / BEAM_STEP).ceil() as usize).clamp(1, BEAM_SQUARES);
    let square = |c: [f32; 2], size: f32| {
        let (x0, y0) = ((c[0] - size / 2.0).round(), (c[1] - size / 2.0).round());
        let (x1, y1) = ((x0 + size.round()).min(x + w), (y0 + size.round()).min(y + h));
        let (x0, y0) = (x0.max(x), y0.max(y));
        (x1 > x0 && y1 > y0).then(|| [x0 as i32, y0 as i32, (x1 - x0) as i32, (y1 - y0) as i32])
    };
    let mut boxes: Vec<[i32; 4]> = (0..=steps)
        .filter_map(|i| {
            let s = i as f32 / steps as f32;
            square([a[0] + s * (b[0] - a[0]), a[1] + s * (b[1] - a[1])], BEAM_WIDE + s * (BEAM_THIN - BEAM_WIDE))
        })
        .collect();
    boxes.dedup();
    if hit && let Some(spot) = square(b, SPOT) {
        boxes.push(spot);
    }
    (!boxes.is_empty()).then_some(boxes)
}

/// The beam's squares cleared to the laser's colour in the eye's render target, the GL state it
/// touches put back.
fn draw_beam(boxes: &[[i32; 4]]) {
    if BEAMS_DRAWN.fetch_add(1, Relaxed) == 0 {
        monaka_producer::log!("hud: a laser sight drawn into the eyes ({} squares)", boxes.len());
    }
    let mut scissor = [0i32; 4];
    let mut clear = [0f32; 4];
    let mut mask = [0u8; 4];
    // SAFETY: state queries and plain GL calls on the render thread, the game's context current;
    // the state is put back as it was.
    unsafe {
        let scissoring = glIsEnabled(GL_SCISSOR_TEST) != 0;
        glGetIntegerv(GL_SCISSOR_BOX, scissor.as_mut_ptr());
        glGetFloatv(GL_COLOR_CLEAR_VALUE, clear.as_mut_ptr());
        glGetBooleanv(GL_COLOR_WRITEMASK, mask.as_mut_ptr());
        glEnable(GL_SCISSOR_TEST);
        glColorMask(1, 1, 1, 1);
        glClearColor(LASER_COLOUR[0], LASER_COLOUR[1], LASER_COLOUR[2], 1.0);
        for &[x, y, w, h] in boxes {
            glScissor(x, y, w, h);
            glClear(GL_COLOR_BUFFER_BIT);
        }
        glClearColor(clear[0], clear[1], clear[2], clear[3]);
        glColorMask(mask[0], mask[1], mask[2], mask[3]);
        glScissor(scissor[0], scissor[1], scissor[2], scissor[3]);
        if !scissoring {
            glDisable(GL_SCISSOR_TEST);
        }
    }
}

/// The viewport for a 2D layer going to the flat screen, the same in each eye: the eye's whole
/// width and the layer's own shape (two eye halves wide, so half the eye's height), centred.
pub fn flat_rect(rect: [i32; 4]) -> Option<[i32; 4]> {
    let [x, y, w, h] = rect;
    if w <= 0 || h <= 0 {
        return None;
    }
    let height = h / 2;
    Some([x, y + (h - height) / 2, w, height])
}

/// The eye a 2D layer record belongs to, from where its rectangle lies in the window.
fn record_eye(record: usize) -> Option<usize> {
    let left = mem::read::<i32>(record + engine::GUI_RECORD_RECT)?;
    Some(usize::from(left > 0))
}

/// Whether a 2D layer record's view draws the 3D world (the layer goes over it, in play).
fn draws_world(record: usize) -> bool {
    mem::read::<usize>(record + engine::GUI_RECORD_WORLD).is_some_and(|world| world != 0)
}
pub unsafe extern "system" fn gui_draw(backend: usize, guis: usize, count: i32, a: usize, record: usize, shift: f32) {
    let flight = InFlight::enter();
    DRAWS.fetch_add(1, Relaxed);
    let mut base = [0i32; 4];
    let mut placed = None;
    let mut crosshair = None;
    let mut beam = None;
    // With frame warping the one view is the eye of the frame being drawn, wherever it sits.
    let eye = record_eye(record).filter(|_| count > 0).map(|eye| crate::providers::eye_site::drawing_eye().unwrap_or(eye));
    if crate::on(crate::HUD_AT)
        && let Some(shared) = SHARED.get()
        && let Some(eye) = eye
    {
        // SAFETY: a state query on the render thread's current context.
        unsafe { glGetIntegerv(GL_VIEWPORT, base.as_mut_ptr()) };
        placed = match draws_world(record) {
            true => head::rendering().or_else(|| head::HEAD.current()).and_then(|pose| {
                let frustum = Frustum::from_fov(pose.fov[eye])?;
                let ipd = shared.separation_for(&pose);
                // The marksman rifle's laser sight, under the HUD.
                beam = eye_site::beam_for(&pose).and_then(|b| beam_boxes(base, &frustum, b.ends[eye], b.hit));
                // A driven machine's crosshair goes where it aims in this eye.
                crosshair = eye_site::aim_for(&pose).and_then(|aim| aim_shift(base, eye, &frustum, ipd, &shared.hud, aim[eye]));
                placed_rect(base, eye, &frustum, ipd, &shared.hud)
            }),
            // A paused menu builds no views: the frame goes to the flat screen.
            false => flat_rect(base).inspect(|_| {
                FLAT.fetch_add(1, Relaxed);
            }),
        };
    }
    if let Some(boxes) = beam {
        draw_beam(&boxes);
    }
    if let Some([x, y, w, h]) = placed {
        // SAFETY: GL call on the render thread with the game's context current.
        unsafe { glViewport(x, y, w, h) };
        PLACED.fetch_add(1, Relaxed);
    }
    if let Some(eye) = eye {
        monaka_stereo::hud_panels::gl::layer_begins(eye);
    }
    crate::providers::hud_pieces::set_crosshair_shift(crosshair.filter(|_| placed.is_some()));
    // SAFETY: forwards the game's own call.
    unsafe { GUI_DRAW_ORIGINAL.get(&flight)(backend, guis, count, a, record, shift) };
    crate::providers::hud_pieces::set_crosshair_shift(None);
    if let Some(eye) = eye {
        monaka_stereo::hud_panels::gl::layer_ends(eye);
        crate::research::panels::layer_ended(eye);
    }
    if placed.is_some() {
        // SAFETY: as above; puts the game's viewport back.
        unsafe { glViewport(base[0], base[1], base[2], base[3]) };
    }
}

#[cfg(test)]
mod tests;
