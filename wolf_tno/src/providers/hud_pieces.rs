//! The HUD pieces provider for Monaka's dynamic HUD (`monaka_stereo::hud_panels`, its OpenGL
//! panels): which of the HUD's draws are which piece's, and where each piece lies in the HUD's
//! layout. The pieces are the guns' ammo, left then right, on the inside of each hand, and the
//! item checklists (the items a task gathers, top left) on the back of the left hand ([`PIECES`]).
//!
//! The HUD is a Flash movie drawn into a GUI model on the game thread
//! ([`engine::HudPieces`]); each gun's ammo is a sprite of its own (`ammo_group_right_new`,
//! `ammo_group_left_new`), which the HUD's ammo groups keep, and so is each item checklist
//! (`combiner1`..`combiner3`), which the checklist group keeps ([`hud_render`]). When the movie draws one of them
//! ([`render_sprite`]), the model is made to start a new surface before and after it, so its draws
//! are surfaces of their own, and those surfaces' GL state is marked with its piece in bits the
//! engine never uses ([`MARKS`]). The render thread draws every surface with its state
//! ([`draw_elements`]), so the mark arrives with each of the piece's draws ([`piece_of`]): the draw
//! goes to its piece's panel, the mark taken off before the engine sees the state. The vertices it
//! wrote give its box in the layout (all of a piece's sprites drawn in a frame: one box).
//!
//! In a driven machine (the mech) the crosshair (`hudmain/hud/crosshair`, in the layout's centre)
//! shows where the machine aims, not the centre of the HUD in front of the eyes: its surfaces are
//! marked the same way ([`CROSSHAIR_MARK`]) and each eye draws them moved by the pixels from the
//! layer's centre to the machine's aim in that eye ([`set_crosshair_shift`]).
//!
//! Out of a machine the crosshair marks the HUD's centre, where the guns no longer aim (they aim
//! with the hands): with `crosshair=hide` ([`CROSSHAIR_COMPONENT`]) its sprite is not drawn at all.

use crate::providers::engine::{self, DrawElementsFn, HudPieces, HudRenderFn, RenderSpriteFn};
use crate::{Facts as Build, Wolf};
use monaka_contract::protocol::{LEFT_HAND, RIGHT_HAND};
use monaka_hook::module::Module;
use monaka_hook::{InFlight, Original, mem};
use monaka_contract::manifest::{Opt, Shown, choice, opt};
use monaka_core::options::Options;
use monaka_producer::game::{Component, Lack, Need, OrMissing, Start};
use monaka_producer::log;
use monaka_stereo::hud_panels::{self, Piece, Place, Side, places};
use std::sync::{Mutex, OnceLock};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering::*};
use monaka_core::sync::Locked;
use windows::Win32::Graphics::OpenGL::{GL_VIEWPORT, glGetIntegerv, glViewport};

pub static RENDER_SPRITE_ORIGINAL: Original<RenderSpriteFn> = Original::new();
pub static HUD_RENDER_ORIGINAL: Original<HudRenderFn> = Original::new();
pub static DRAW_ORIGINAL: Original<DrawElementsFn> = Original::new();

/// The dynamic HUD (Monaka's): each gun's ammo on the inside of its hand and the item checklists
/// on the back of the left hand, out of the HUD in front of the eyes.
pub const HUD_PANELS: Component<Wolf> = Component { needs: &[Need::On(&crate::cooking::hud::HUD)], install: install_panels, summary: Some(summary), ..hud_panels::hud_panels() };

/// The HUD's centre crosshair out of a driven machine (`crosshair`): hidden (the default) or shown.
pub const CROSSHAIR_OPTION: Opt = opt(
    "crosshair",
    "Crosshair",
    "Hide: the crosshair in the middle of the HUD is not drawn (the guns aim with your hands). The mech's still shows where it aims.",
    choice(&[("hide", "Hide"), ("show", "Show")], "hide"),
)
.on(Shown::Game)
.only_when("world_hud", "1");

/// The crosshair as `crosshair` asks, read as the run starts; its sprite is drawn by the HUD
/// panels' hooks ([`HUD_PANELS`]).
pub const CROSSHAIR_COMPONENT: Component<Wolf> = Component {
    name: "crosshair",
    options: &[CROSSHAIR_OPTION],
    needs: &[Need::On(&HUD_PANELS)],
    settings: |options, _| {
        hides_crosshair(options);
    },
    install: install_crosshair,
    ..Component::NONE
};

/// Whether `options` hide the HUD's centre crosshair.
pub fn hides_crosshair(options: &Options) -> bool {
    options.choice("crosshair", &[("hide", true), ("show", false)], true)
}

/// The crosshair hidden out of a driven machine, and its draws left out.
static HIDE_CROSSHAIR: AtomicBool = AtomicBool::new(false);
static CROSSHAIR_HIDDEN: AtomicU64 = AtomicU64::new(0);

fn install_crosshair(start: &mut Start<'_, Wolf>) -> Result<(), Lack> {
    HIDE_CROSSHAIR.store(hides_crosshair(start.options), Relaxed);
    Ok(())
}

/// Whether the crosshair is hidden now: out of a driven machine, with `crosshair=hide`.
fn hiding_crosshair() -> bool {
    HIDE_CROSSHAIR.load(Relaxed) && !crate::providers::camera_taken::driving()
}

/// The item checklists' piece, after the guns' ammo (by side).
pub const CHECKLISTS: usize = 2;

/// The item checklists: on the back of the left hand, a hand's width.
const CHECKLISTS_PLACE: Place = Place { hand: LEFT_HAND, side: Side::BackOfHand, width: 0.08, beside: false };

/// The pieces: a second gun's ammo on the left palm, the gun's on the right (by side), then the
/// item checklists.
pub const PIECES: [Piece; 3] = [
    Piece { name: "weapon_left", size: (768, 384), place: places::WEAPON_LEFT },
    Piece { name: "weapon", size: (768, 384), place: places::WEAPON },
    Piece { name: "checklists", size: (768, 384), place: CHECKLISTS_PLACE },
];

impl hud_panels::HudPieces for Wolf {
    const PIECES: &'static [Piece] = &PIECES;
}

fn install_panels(start: &mut Start<'_, Wolf>) -> Result<(), Lack> {
    let Build { exe, build } = *start.build;
    let pieces = build.hud_pieces.as_ref().or_missing(&format!("where {} draws its HUD's ammo is not mapped", build.game))?;
    hud_panels::install::<Wolf>(start)?;
    let _ = FACTS.set(Facts { pieces, exe });
    start.hooks.site(&HUD_RENDER_ORIGINAL, &exe, &pieces.hud_render, hud_render)?;
    start.hooks.site(&RENDER_SPRITE_ORIGINAL, &exe, &pieces.render_sprite, render_sprite)?;
    start.hooks.site(&DRAW_ORIGINAL, &exe, &pieces.draw_elements, draw_elements)?;
    Ok(())
}

/// The GL state bits marking a surface as a piece's, by piece (left ammo, right ammo, checklists).
pub const MARKS: [u64; 3] = [1 << 62, 1 << 63, 1 << 61];

/// The GL state bit marking a surface as the crosshair's (the engine tests bits up to 50).
pub const CROSSHAIR_MARK: u64 = 1 << 60;

/// The piece a surface of GL state `state` draws, if marked.
pub fn piece_of(state: u64) -> Option<usize> {
    (0..MARKS.len()).find(|&piece| state & MARKS[piece] != 0)
}

/// `state` without the marks.
pub fn unmarked(state: u64) -> u64 {
    state & !MARKS.iter().fold(CROSSHAIR_MARK, |all, mark| all | mark)
}

/// The crosshair's sprite instance while a machine is driven (0: not found yet), the time in the
/// machine it was found in (`camera_taken::drives`), and the HUD frames since it was last drawn.
static CROSSHAIR: AtomicUsize = AtomicUsize::new(0);
static CROSSHAIR_DRIVE: AtomicU64 = AtomicU64::new(0);
static CROSSHAIR_UNSEEN: AtomicU64 = AtomicU64::new(0);
/// HUD frames without the crosshair after which it is looked for again (the HUD rebuilt).
const CROSSHAIR_UNSEEN_FRAMES: u64 = 30;
/// The crosshair drawn apart, and its draws moved in an eye.
static CROSSHAIRS: AtomicU64 = AtomicU64::new(0);
static CROSSHAIR_MOVED: AtomicU64 = AtomicU64::new(0);
/// The pixels the crosshair moves by in the eye whose 2D layer is being drawn (render thread).
static CROSSHAIR_SHIFT: Mutex<Option<[i32; 2]>> = Mutex::new(None);

/// While the 2D layer of an eye is drawn: the pixels (x right, y up) from the layer's centre to
/// where the driven machine aims in that eye; None: the crosshair stays where the HUD has it.
pub fn set_crosshair_shift(shift: Option<[i32; 2]>) {
    *CROSSHAIR_SHIFT.locked() = shift;
}

/// The name of sprite instance `sprite`.
fn sprite_name(sprite: usize) -> Option<String> {
    mem::read::<usize>(sprite + engine::SPRITE_NAME).filter(|&p| p != 0).and_then(|p| mem::read_c_string(p, 32))
}

/// Whether `sprite` is the HUD's crosshair (`hud/crosshair`): found by name once each time in a
/// machine (or while it is hidden), then by instance.
fn is_crosshair(sprite: usize) -> bool {
    let found = match CROSSHAIR.load(Relaxed) {
        0 => {
            let found = sprite_name(sprite).is_some_and(|n| n == "crosshair")
                && mem::read::<usize>(sprite + engine::SPRITE_PARENT).and_then(sprite_name).is_some_and(|n| n == "hud");
            if found {
                CROSSHAIR.store(sprite, Relaxed);
                CROSSHAIR_DRIVE.store(crate::providers::camera_taken::drives(), Relaxed);
                static LOGGED: AtomicU64 = AtomicU64::new(0);
                if LOGGED.fetch_add(1, Relaxed) < 20 {
                    let does = if hiding_crosshair() { "hidden" } else { "it shows where the machine aims" };
                    log!("HUD pieces: the crosshair sprite is {sprite:#x}: {does}");
                }
            }
            found
        }
        crosshair => crosshair == sprite,
    };
    if found {
        CROSSHAIR_UNSEEN.store(0, Relaxed);
    }
    found
}

/// Each HUD frame: the crosshair is looked for again out of a machine (unless it is hidden), in a
/// new time in one (a checkpoint's reload rebuilds the HUD), or once the one found has not been
/// drawn for a while.
fn forget_stale_crosshair() {
    let unseen = CROSSHAIR_UNSEEN.fetch_add(1, Relaxed) + 1;
    let stale = !(crate::providers::camera_taken::driving() || HIDE_CROSSHAIR.load(Relaxed))
        || CROSSHAIR_DRIVE.load(Relaxed) != crate::providers::camera_taken::drives()
        || unseen > CROSSHAIR_UNSEEN_FRAMES;
    if stale && CROSSHAIR.swap(0, Relaxed) != 0 {
        CROSSHAIR_UNSEEN.store(0, Relaxed);
    }
}

/// The running build's facts and executable.
struct Facts {
    pieces: &'static HudPieces,
    exe: Module,
}

static FACTS: OnceLock<Facts> = OnceLock::new();
/// Each gun's ammo sprite instance (0: not found yet), by side.
static AMMO: [AtomicUsize; 2] = [AtomicUsize::new(0), AtomicUsize::new(0)];
/// The item checklists' sprite instances (0: not found yet).
static CHECKLIST_SPRITES: [AtomicUsize; 3] = [AtomicUsize::new(0), AtomicUsize::new(0), AtomicUsize::new(0)];
/// Pieces' sprites drawn on surfaces of their own, by piece.
static SPLIT: [AtomicU64; 3] = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];

pub fn summary() -> String {
    format!(
        "HUD pieces: ammo drawn apart {} times (left), {} (right), checklists {}; {} draws into the panels; crosshair drawn apart {} times in a driven machine, {} draws moved onto its aim, {} times left out (hidden)",
        SPLIT[LEFT_HAND].load(Relaxed),
        SPLIT[RIGHT_HAND].load(Relaxed),
        SPLIT[CHECKLISTS].load(Relaxed),
        hud_panels::gl::draws(),
        CROSSHAIRS.load(Relaxed),
        CROSSHAIR_MOVED.load(Relaxed),
        CROSSHAIR_HIDDEN.load(Relaxed)
    )
}

/// The GL draw of a GUI surface (render thread): a piece's draw goes to its panel (Monaka's,
/// in the eye's 2D layer); every other draw as the game makes it.
unsafe extern "system" fn draw_elements(material: usize, geometry: usize, state: u64, flag: usize) {
    let flight = InFlight::enter();
    // SAFETY: forwards the game's own call, with the state as the engine knows it.
    let original = || unsafe { DRAW_ORIGINAL.get(&flight)(material, geometry, unmarked(state), flag) };
    if let Some(piece) = piece_of(state)
        && hud_panels::gl::draw(piece, &original)
    {
        return;
    }
    if state & CROSSHAIR_MARK != 0
        && let Some([dx, dy]) = *CROSSHAIR_SHIFT.locked()
    {
        let mut at = [0i32; 4];
        // SAFETY: GL calls on the render thread with the game's context current: the eye's 2D
        // layer viewport moved for this draw, then put back.
        unsafe {
            glGetIntegerv(GL_VIEWPORT, at.as_mut_ptr());
            glViewport(at[0] + dx, at[1] + dy, at[2], at[3]);
        }
        original();
        // SAFETY: as above.
        unsafe { glViewport(at[0], at[1], at[2], at[3]) };
        CROSSHAIR_MOVED.fetch_add(1, Relaxed);
        return;
    }
    original();
}

/// The HUD's frame: each gun's ammo sprite read from its ammo group (none while the group has
/// not found what it draws) and the item checklists' sprites from their group, then the HUD as the
/// game draws it.
unsafe extern "system" fn hud_render(hud: usize, gui: usize, player: usize, time: usize) {
    let flight = InFlight::enter();
    forget_stale_crosshair();
    if let Some(facts) = FACTS.get() {
        for side in [LEFT_HAND, RIGHT_HAND] {
            let group = hud + facts.pieces.ammo_groups[side];
            let ready = mem::read::<u8>(group + engine::AMMO_GROUP_READY[side]).is_some_and(|r| r != 0);
            let sprite = if ready { mem::read::<usize>(group + engine::AMMO_GROUP_SPRITE[side]).unwrap_or(0) } else { 0 };
            if AMMO[side].swap(sprite, Relaxed) != sprite {
                note_sprite(PIECES[side].name, sprite);
            }
        }
        let lists = &facts.pieces.checklists;
        for (held, at) in CHECKLIST_SPRITES.iter().zip(lists.sprites) {
            let sprite = mem::read::<usize>(hud + lists.group + at).unwrap_or(0);
            if held.swap(sprite, Relaxed) != sprite {
                note_sprite(PIECES[CHECKLISTS].name, sprite);
            }
        }
    }
    // SAFETY: forwards the game's own call.
    unsafe { HUD_RENDER_ORIGINAL.get(&flight)(hud, gui, player, time) };
}

fn note_sprite(piece: &str, sprite: usize) {
    static LOGGED: AtomicU64 = AtomicU64::new(0);
    if LOGGED.fetch_add(1, Relaxed) < 20 {
        log!("HUD pieces: a {piece} sprite is {sprite:#x}");
    }
}

/// The piece whose sprite `sprite` is, if any.
fn piece_of_sprite(sprite: usize) -> Option<usize> {
    if sprite == 0 {
        return None;
    }
    AMMO.iter().position(|a| a.load(Relaxed) == sprite).or_else(|| CHECKLIST_SPRITES.iter().any(|c| c.load(Relaxed) == sprite).then_some(CHECKLISTS))
}

/// A GUI model's current surface, the surface count, and surface `i`.
fn current(gui: usize, facts: &Facts) -> Option<usize> {
    mem::read::<usize>(gui + facts.pieces.current_surface).filter(|&s| s != 0)
}

fn count(gui: usize, facts: &Facts) -> Option<i32> {
    mem::read::<i32>(gui + facts.pieces.current_surface + 0xc)
}

fn surface(gui: usize, facts: &Facts, i: i32) -> usize {
    gui + facts.pieces.current_surface + 0x10 + i as usize * engine::GUI_SURFACE_SIZE
}

/// Starts the GUI model's next surface if the current one holds anything.
fn advance(gui: usize, facts: &Facts) {
    let Some(surface) = current(gui, facts) else { return };
    if mem::read::<i32>(surface + engine::GUI_SURFACE_VERTICES).is_some_and(|n| n > 0) {
        // SAFETY: the game's own `idRenderModelGui::AdvanceSurf` on a live GUI model, on the game
        // thread the model is being drawn into.
        unsafe { std::mem::transmute::<usize, unsafe extern "system" fn(usize)>(facts.exe.at(facts.pieces.advance_surface))(gui) };
    }
}

fn vertex_count(facts: &Facts) -> Option<i32> {
    mem::read::<i32>(facts.exe.at(facts.pieces.gui_vertex_count))
}

/// The box of the vertices `from..to` the model wrote (layout units), if any are readable.
fn written_box(facts: &Facts, from: i32, to: i32) -> Option<[f32; 4]> {
    let base = mem::read::<usize>(facts.exe.at(facts.pieces.gui_vertices)).filter(|&b| b != 0)?;
    let mut bounds: Option<[f32; 4]> = None;
    for i in from.max(0)..to.min(from.max(0) + 4096) {
        let Some([x, y]) = mem::read::<[f32; 2]>(base + i as usize * engine::GUI_VERTEX_SIZE).filter(|p| p.iter().all(|v| v.is_finite())) else { continue };
        bounds = Some(match bounds {
            None => [x, y, x, y],
            Some(b) => [b[0].min(x), b[1].min(y), b[2].max(x), b[3].max(y)],
        });
    }
    bounds
}

/// The layout size of GUI model `gui`.
fn layout(gui: usize, facts: &Facts) -> Option<(f32, f32)> {
    let size = |rva: usize| {
        // SAFETY: the game's own getter on a live GUI model, on the game thread.
        unsafe { std::mem::transmute::<usize, unsafe extern "system" fn(usize) -> f32>(facts.exe.at(rva))(gui) }
    };
    let (w, h) = (size(facts.pieces.virtual_width), size(facts.pieces.virtual_height));
    (w.is_finite() && h.is_finite() && w > 1.0 && h > 1.0).then_some((w, h))
}

/// Logs the first places `piece` is drawn at.
fn note_box(piece: usize, bounds: [f32; 4], layout: (f32, f32)) {
    static LOGGED: AtomicU64 = AtomicU64::new(0);
    static LAST: [AtomicU64; 3] = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];
    let key = bounds.iter().fold(0u64, |k, v| k.rotate_left(16) ^ u64::from(v.to_bits()));
    if LAST[piece].swap(key, Relaxed) != key && LOGGED.fetch_add(1, Relaxed) < 20 {
        log!("HUD pieces: {} drawn at {bounds:.0?} of a {:.0}x{:.0} layout", PIECES[piece].name, layout.0, layout.1);
    }
}

/// `idSWF::RenderSprite`: a piece's sprite (and in a driven machine the crosshair) is drawn on
/// surfaces of their own, marked with its piece (the crosshair's mark); a hidden crosshair not at
/// all; everything else as the game draws it.
unsafe extern "system" fn render_sprite(swf: usize, gui: usize, sprite: usize, state: usize, a5: usize, a6: usize) {
    let flight = InFlight::enter();
    // SAFETY: forwards the game's own call.
    let original = || unsafe { RENDER_SPRITE_ORIGINAL.get(&flight)(swf, gui, sprite, state, a5, a6) };
    let piece = hud_panels::active().then(|| piece_of_sprite(sprite)).flatten();
    let hiding = hiding_crosshair();
    let crosshair = piece.is_none() && sprite != 0 && (hiding || crate::providers::camera_taken::driving()) && is_crosshair(sprite);
    if crosshair && hiding {
        CROSSHAIR_HIDDEN.fetch_add(1, Relaxed);
        return;
    }
    let mark = piece.map(|piece| MARKS[piece]).or(crosshair.then_some(CROSSHAIR_MARK));
    let (Some(mark), Some(facts)) = (mark, FACTS.get()) else {
        // Which sprite is drawn where, once each (`research::hud_sprites`).
        if let Some(facts) = FACTS.get()
            && crate::research::hud_sprites::wanted(sprite)
        {
            let from = vertex_count(facts);
            original();
            if let (Some(from), Some(to)) = (from, vertex_count(facts))
                && to > from
                && let (Some(bounds), Some(layout)) = (written_box(facts, from, to), layout(gui, facts))
            {
                crate::research::hud_sprites::drawn(sprite, bounds, layout);
            }
            return;
        }
        return original();
    };
    // Room for the two surfaces this adds, or it is drawn as the game draws it.
    if count(gui, facts).is_none_or(|n| !(1..=engine::GUI_SURFACES - 4).contains(&n)) {
        return original();
    }
    advance(gui, facts);
    let (Some(first), Some(from)) = (count(gui, facts), vertex_count(facts)) else { return original() };
    original();
    let (Some(last), Some(to)) = (count(gui, facts), vertex_count(facts)) else { return };
    advance(gui, facts);
    if to <= from {
        return;
    }
    for i in first - 1..last {
        let at = surface(gui, facts, i) + engine::GUI_SURFACE_STATE;
        if let Some(state) = mem::read::<u64>(at) {
            mem::write(at, state | mark);
        }
    }
    let Some(piece) = piece else {
        CROSSHAIRS.fetch_add(1, Relaxed);
        return;
    };
    SPLIT[piece].fetch_add(1, Relaxed);
    if let (Some(bounds), Some(layout)) = (written_box(facts, from, to), layout(gui, facts)) {
        note_box(piece, bounds, layout);
        hud_panels::show(piece, bounds, layout);
    }
}

#[cfg(test)]
mod tests;
