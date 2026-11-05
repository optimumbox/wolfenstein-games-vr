//! Provider, to be: settings (changing a game setting and putting it back), once Monaka has the
//! trait; the ledger (`monaka_producer::ledger`) already puts each change back.
//!
//! What the game needs set while VR runs ([`SETTINGS`]), and the game's rate ([`GameRate`]) and
//! how long each frame waited for its tick ([`spinning`]), for the frame rate
//! ([`crate::cooking::frame_rate`]).
//!
//! The debug menu's "Freeze the world" (`Category::FreezeWorld`) sets the engine's `g_stopTime`
//! while it is on ([`freeze_frame`]): the world stands still, the player's view goes on. The ledger
//! puts it back when VR stops.

use crate::providers::engine::{self, GameRateFn};
use crate::{Facts, Wolf};
use eng_idt::Cvar;
use monaka_contract::debug_draw::Category;
use monaka_hook::module::Module;
use monaka_producer::log;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering::*};
use monaka_contract::manifest::{DISABLE_VIEW_BOBBING, Kind, Opt, Shown, opt};
use monaka_hook::mem;
use monaka_producer::game::{Component, Lack, Start};
use monaka_producer::ledger::Setting;

/// The view's shake (explosions, hits, the ground trembling) off (`view_allowCameraShake` 0), as
/// [`DISABLE_VIEW_BOBBING`] takes the bob.
pub const DISABLE_SCREEN_SHAKE: Opt = opt("disable_screen_shake", "Disable screen shake", "", Kind::Switch { default: false }).on(Shown::Game);

/// What the game needs set while VR runs: it keeps playing without focus (the viewer, SteamVR or a
/// PC check have it), its view bob goes (Monaka's own setting), and its screen shake when the
/// player says ([`DISABLE_SCREEN_SHAKE`]).
pub const SETTINGS: Component<Wolf> = Component {
    name: "settings",
    options: &[DISABLE_VIEW_BOBBING, DISABLE_SCREEN_SHAKE],
    settings: |options, manifest| {
        no_view_bob(options, manifest);
        options.switch(DISABLE_SCREEN_SHAKE.id, false);
    },
    install: install_settings,
    ..Component::NONE
};

/// The game's view bob (and weapon sway) off while VR runs (`disable_view_bobbing`, the
/// manifest's default).
pub(crate) fn no_view_bob(options: &monaka_core::options::Options, manifest: &monaka_contract::manifest::Manifest) -> bool {
    options.switch("disable_view_bobbing", manifest.switch_default("disable_view_bobbing").unwrap_or(true))
}

fn install_settings(start: &mut Start<'_, Wolf>) -> Result<(), Lack> {
    let Facts { exe, build } = *start.build;
    eng_idt::set_int(start.ledger, &exe, &build.pause_on_alt_tab, 0, "the game pauses without focus");
    if no_view_bob(start.options, start.manifest) {
        eng_idt::set_int(start.ledger, &exe, &build.no_bob, 1, "the view bobs with the character's steps");
    }
    if start.options.switch(DISABLE_SCREEN_SHAKE.id, false) {
        no_screen_shake(start.ledger, &exe, build);
    }
    no_screen_blur(start.ledger, &exe, build);
    log_streaming(&exe);
    gpu_decoding(start.ledger, &exe);
    hold_unique(start.ledger, &exe);
    crate::research::cvars::apply(start.ledger, &exe);
    crate::research::cvars::prepare_cycle(start.ledger, &exe);
    // Kept at 0 by the ledger (what it holds at the stop goes back), so the world never stays
    // frozen after VR.
    if build.stop_time.present(&exe) && build.stop_time.set_int(&exe, 0) && start.ledger.keep(eng_idt::IntCvar { exe, cvar: &build.stop_time }).is_some() {
        let _ = FREEZE.set((exe, &build.stop_time));
        monaka_producer::debug::offer(Category::FreezeWorld);
    }
    Ok(())
}

/// The game's texture streaming settings, as the run starts (read only): with the streaming
/// lines of [`crate::single_pass::uploads`], what a hitch was made of and what to turn.
const STREAMING: &[&str] = &[
    "vt_maxPPF",
    "vt_pageImageSizeVmtr",
    "vt_pageImageSizeUnique",
    "vt_pageImageSizeUniqueDiffuseOnly",
    "vt_uncompressedPhysicalImages",
    "vt_useCudaTranscode",
    "vt_useTranscodeJobs",
    "vt_splitTranscodeJobs",
    "vt_usePageFileCache",
    "vt_usePageTablePBOs",
    "vt_useUnsynchronizedPageTablePBOs",
    "vt_feedbackLodBias",
    "vt_feedbackHighWater",
    "vt_feedbackLowWater",
    "vt_lodBias",
    "vt_maxAniso",
    "vt_pageFadeInSteps",
    "r_usePBOTextureFeedback",
    "jobs_numThreads",
];

fn log_streaming(exe: &Module) {
    let found: Vec<(&'static str, &'static Cvar)> = STREAMING.iter().filter_map(|&name| Some((name, Cvar::find(exe, name, eng_idt::Layout::ID_TECH_5)?))).collect();
    let values: Vec<String> = found.iter().filter_map(|(name, cvar)| Some(format!("{name} {} ({})", cvar.int(exe)?, cvar.float(exe)?))).collect();
    log!("texture streaming settings: {}", values.join(", "));
    let seen = found.iter().map(|(_, c)| c.int(exe)).collect();
    let _ = WATCHED.set(Watched { exe: *exe, found, seen: std::sync::Mutex::new(seen) });
}

/// The streaming settings as last seen, for [`watch_streaming`].
struct Watched {
    exe: Module,
    found: Vec<(&'static str, &'static Cvar)>,
    seen: std::sync::Mutex<Vec<Option<i32>>>,
}

static WATCHED: OnceLock<Watched> = OnceLock::new();
/// `vt_useCudaTranscode`, held on ([`gpu_decoding`]).
static GPU_DECODING: OnceLock<(Module, &'static Cvar)> = OnceLock::new();
static WATCH_FRAMES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The texture pages decoded on the GPU (`vt_useCudaTranscode` 1, the game's own CUDA decoder)
/// rather than on the CPU's job threads: the work that grows with the streaming settings raised
/// moves off the CPU. The game's options menu writes it back to 0 whenever its video settings are
/// applied, so it is held on ([`watch_streaming`]); the ledger puts the game's own back at the stop.
fn gpu_decoding(ledger: &mut monaka_producer::ledger::Ledger, exe: &Module) {
    let Some(cvar) = Cvar::find(exe, "vt_useCudaTranscode", eng_idt::Layout::ID_TECH_5) else { return };
    eng_idt::set_int(ledger, exe, cvar, 1, "texture pages are decoded on the CPU");
    let _ = GPU_DECODING.set((*exe, cvar));
    log!("texture streaming: pages decoded on the GPU (vt_useCudaTranscode 1), held on while VR runs");
}

/// The largest `vt_pageImageSizeUnique` (the unique texture cache image) let through, or none.
/// The game's texture cache on high (Vmtr 4096, Unique 8192) throttled the headset a minute into
/// every run; Vmtr held at 2048 with Unique 8192 did the same, while Vmtr 4096 with Unique 4096 held
/// 90. That was the viewer's copy waiting behind the game's GPU work with Windows' hardware GPU
/// scheduling off; with it on, high and ultra held 90, so nothing is held (`Some(size)` holds again).
const UNIQUE_MOST: Option<i32> = None;
/// `vt_pageImageSizeUnique`, held at [`UNIQUE_MOST`] at most.
static UNIQUE: OnceLock<(Module, &'static Cvar, i32)> = OnceLock::new();

fn hold_unique(ledger: &mut monaka_producer::ledger::Ledger, exe: &Module) {
    let Some(most) = UNIQUE_MOST else { return };
    let Some(cvar) = Cvar::find(exe, "vt_pageImageSizeUnique", eng_idt::Layout::ID_TECH_5) else { return };
    let _ = UNIQUE.set((*exe, cvar, most));
    let game = cvar.int(exe);
    if game.is_some_and(|v| v > most) {
        eng_idt::set_int(ledger, exe, cvar, most, "the unique texture cache image is the game's size");
        log!("texture streaming: vt_pageImageSizeUnique held at {most} (the game's {}), vt_pageImageSizeVmtr left as the game has it", game.unwrap_or(0));
    }
}

/// On the render thread, every 45th frame: a streaming setting changed (the options menu) is logged,
/// and GPU decoding turned off is turned on again before the engine rebuilds its texture cache.
pub fn watch_streaming() {
    if WATCH_FRAMES.fetch_add(1, Relaxed) % 45 != 0 {
        return;
    }
    if let Some((exe, cvar)) = GPU_DECODING.get()
        && cvar.int(exe) == Some(0)
    {
        cvar.set_int(exe, 1);
        log!("texture streaming: the game turned GPU decoding off (vt_useCudaTranscode 0); on again");
    }
    if let Some((exe, cvar, most)) = UNIQUE.get()
        && cvar.int(exe).is_some_and(|v| v > *most)
    {
        cvar.set_int(exe, *most);
        log!("texture streaming: the game set vt_pageImageSizeUnique above {most}; held at {most} again");
    }
    let Some(w) = WATCHED.get() else { return };
    let Ok(mut seen) = w.seen.lock() else { return };
    let mut changed = Vec::new();
    for ((name, cvar), last) in w.found.iter().zip(seen.iter_mut()) {
        let now = cvar.int(&w.exe);
        if now != *last {
            changed.push(format!("{name} {} -> {}", last.map_or("?".to_owned(), |v| v.to_string()), now.map_or("?".to_owned(), |v| v.to_string())));
            *last = now;
        }
    }
    if !changed.is_empty() {
        log!("texture streaming settings changed: {}", changed.join(", "));
    }
}

/// What shakes the view, each set to 0 while the screen shake is off: the camera shakes
/// (explosions, hits) and the weapons' kick as they fire (switched off, and its amplitudes);
/// whether each is a float.
const SHAKES: &[(&str, bool)] = &[
    ("view_allowCameraShake", false),
    ("g_weaponkick", false),
    ("g_kickAmplitude", true),
    ("view_kickAmplitude_min", true),
    ("view_kickAmplitude_max", true),
];

/// The screen shake off ([`SHAKES`]): cvars no build's table lists, found in the running game
/// ([`eng_idt::Cvar::find`]; the lookup checked once against `pm_noBob`, which the table does
/// list). One the game does not have is left out.
fn no_screen_shake(ledger: &mut monaka_producer::ledger::Ledger, exe: &Module, build: &engine::Build) {
    if let Some(bob) = Cvar::find(exe, build.no_bob.name, eng_idt::Layout::ID_TECH_5)
        && bob.name_pointer != build.no_bob.name_pointer
    {
        log!("cvar lookup: {} found at {:#x}, the table has {:#x}; the screen shake stays", bob.name, bob.name_pointer, build.no_bob.name_pointer);
        return;
    }
    for &(name, float) in SHAKES {
        let Some(cvar) = Cvar::find(exe, name, eng_idt::Layout::ID_TECH_5) else { continue };
        if float {
            eng_idt::set_float(ledger, exe, cvar, 0.0, "the view shakes");
        } else {
            eng_idt::set_int(ledger, exe, cvar, 0, "the view shakes");
        }
    }
}

/// The blurs laid over the whole screen when the player is hit (the damage blur and its
/// amplitudes), each 0 while VR runs: a flat image's smear, not the same in the two eyes (each eye
/// drawn on its own, or one warped from the other), and a blurred frame upscaled and warped turns
/// to noise. The game has no camera motion blur to turn off.
const BLURS: &[(&str, bool)] = &[("view_damageBlur", false), ("view_blur_min", true), ("view_blur_max", true)];

/// The screen blurs off ([`BLURS`]), found in the running game as the shakes are; one the game
/// does not have is left out.
fn no_screen_blur(ledger: &mut monaka_producer::ledger::Ledger, exe: &Module, build: &engine::Build) {
    if let Some(bob) = Cvar::find(exe, build.no_bob.name, eng_idt::Layout::ID_TECH_5)
        && bob.name_pointer != build.no_bob.name_pointer
    {
        return;
    }
    for &(name, float) in BLURS {
        let Some(cvar) = Cvar::find(exe, name, eng_idt::Layout::ID_TECH_5) else { continue };
        if float {
            eng_idt::set_float(ledger, exe, cvar, 0.0, "the screen blurs when the player is hit");
        } else {
            eng_idt::set_int(ledger, exe, cvar, 0, "the screen blurs when the player is hit");
        }
    }
}

/// The executable and `g_stopTime`, once the ledger holds it.
static FREEZE: OnceLock<(Module, &'static Cvar)> = OnceLock::new();
/// Whether the world is frozen now.
static FROZEN: AtomicBool = AtomicBool::new(false);

/// Each view build (the game thread): the world frozen while the debug menu's "Freeze the world"
/// is on, or once `probe_freeze_at` says ([`crate::research::freeze`]).
pub fn freeze_frame() {
    let Some((exe, stop_time)) = FREEZE.get() else { return };
    let wanted = monaka_producer::debug::on(Category::FreezeWorld) || crate::research::freeze::due();
    if FROZEN.load(Relaxed) == wanted {
        return;
    }
    if stop_time.set_int(exe, i32::from(wanted)) {
        FROZEN.store(wanted, Relaxed);
        log!("the world {} (g_stopTime {})", if wanted { "is frozen: the view goes on" } else { "moves again" }, i32::from(wanted));
    }
}

/// The game's rate (ticks, and so frames, a second), set through the engine's own setter
/// (`engine::Build::set_game_rate`); a new rate holds from the game's next frame.
#[derive(Clone, Copy)]
pub struct GameRate {
    set: usize,
    manager: usize,
}

impl GameRate {
    pub fn of(facts: &Facts) -> GameRate {
        GameRate { set: facts.exe.at(facts.build.set_game_rate), manager: facts.exe.at(facts.build.time_manager) }
    }
}

impl Setting for GameRate {
    type Value = i32;
    fn name(&self) -> String {
        "the game's rate".to_owned()
    }
    fn read(&self) -> Option<i32> {
        mem::read::<i32>(self.manager).filter(|hz| (1..=1000).contains(hz))
    }
    fn write(&self, hz: &i32) -> bool {
        // SAFETY: the engine's setter at this build's address (checked by the build's fingerprint),
        // which takes the rate alone.
        unsafe { std::mem::transmute::<usize, GameRateFn>(self.set)(*hz) };
        self.read() == Some(*hz)
    }
}

/// How long the last frame waited for its tick, in microseconds, read at `address`
/// (`engine::Build::time_spinning`, absolute).
pub fn spinning(address: usize) -> Option<i32> {
    mem::read::<i32>(address).filter(|us| *us >= 0)
}
