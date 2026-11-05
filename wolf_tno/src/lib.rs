//! Wolfenstein: The New Order and The Old Blood (id Tech 5, OpenGL) in stereo: the engine's own
//! side-by-side layout renders both eyes every frame, and each half of the frame goes to the viewer
//! as one eye through the D3D11 keyed-mutex channel. Injected by `monaka_loader`; options come from
//! `wolf-options.txt` (`config`) and the channel from `live-channel.txt` beside the DLL copy. One DLL
//! serves every build in [`engine::ROWS`]; the running executable picks its facts.
//!
//! The adapter ([`Wolf::ADAPTER`]), in the order its components start:
//!
//! - [`SETTINGS`](providers::settings::SETTINGS): what the game needs set while VR runs, first so
//!   it goes back last (the look hands back its share in the next command, which a paused game
//!   would not build);
//! - [`FRAME_PACING_COMPONENT`](cooking::frame_pacing::FRAME_PACING_COMPONENT): each frame started
//!   on the headset's refresh;
//! - [`FRAME_RATE_COMPONENT`](cooking::frame_rate::FRAME_RATE_COMPONENT): the game's rate
//!   following what the PC holds, up to the player's ceiling;
//! - [`PRESENT`]: the same-frame loop (Monaka's), each frame's halves to the viewer at
//!   `SwapBuffers` ([`providers::present_entry`]; a full-screen video to both eyes, flat), the eye
//!   site installed with it ([`providers::eye_site`]);
//! - [`EYES`](providers::eye_site::EYES): stereo by the engine's side-by-side layout, the player's
//!   render view turned by the head at the view builder, each eye given the headset's frustum;
//! - [`SINGLE_PASS`]: both eyes drawn by the left eye's view, the right eye's view skipping what
//!   is drawn (`stereo=single`; [`single_pass`]);
//! - [`LOOK`]: the player looks and shoots where the head points (Monaka's look, through the
//!   command's angles, [`providers::look`]);
//! - [`GUNS`]: each gun on its controller ([`providers::held_item`]); [`GUN`]: each shot from the
//!   gun that fired it, along its barrel (Monaka's, [`providers::fire`]); [`HAND_RIG`]: the arms
//!   with them (Monaka's hand rig); [`FINGERS`]: a free hand's fingers; [`MELEE`]: a swing of the
//!   knife or the pipe hits where it goes (Monaka's physical melee, [`providers::melee`]);
//! - [`GAMEPAD`]: the controllers as the game's pad (Monaka's);
//!   [`PAD_CONNECTED`](providers::pad_connected::PAD_CONNECTED): the game's mark that a pad is
//!   there;
//! - [`HUD`](cooking::hud::HUD): the 2D layer at a depth in each eye;
//!   [`HUD_PANELS`](providers::hud_pieces::HUD_PANELS): each gun's ammo on the inside of its hand
//!   (Monaka's dynamic HUD);
//! - [`AUTO_PICKUP`](cooking::auto_pickup::AUTO_PICKUP): ammo taken by walking to it
//!   ([`providers::pickups`]);
//! - [`MECH_LEVERS`](extras::mech_levers::MECH_LEVERS): the mech aimed up and down by pushing and
//!   pulling both controllers;
//! - [`HOTWIRE`](extras::hotwire::HOTWIRE): the hotwiring minigame's hands moved by the
//!   controllers;
//! - [`PROBES`]: the research probes.
//!
//! The code is sorted by who should own it: [`providers`] (what only these games know: their
//! facts, their hooks, reading and writing the game), [`extras`] (what only these games have: the
//! cutting laser aimed by its gun), [`cooking`] (their own versions of features a Monaka ready-made
//! should own, each waiting for a provider: the HUD layer and the sights), `research` (the
//! probes) and [`single_pass`] (both eyes drawn by the left eye's pass, on trial). The components
//! are listed here, except those still cooking.

mod eyes;
mod cooking;
mod extras;
mod manifest;
#[cfg(test)]
mod launch_tests;
mod providers;
mod research;
pub mod single_pass;

use eyes::EyeSettings;
use cooking::hud;
use monaka_body::finger_rig::ChainSource;
use monaka_body::hand_rig::{self, Climbing, HandRig, Holds, Method, Rig};
use monaka_body::held::Kind;
use monaka_core::feature::Feature;
use monaka_contract::manifest::{AIM_MOTION, AIM_MOUSE, Manifest, TURN, aim, aims};
use monaka_core::pad::StickTurn;
use monaka_core::options::Options;
use monaka_core::situation::Situation;
use monaka_hook::module::Module;
use monaka_producer::game::{Adapter, Component, Game, Lack, Need, On, Start, place};
use monaka_producer::{Rejection, Session, log};
use providers::engine::{self, Build};
use providers::{eye_site, held_item, hud_pieces, pad_connected, settings, skeleton};
use std::sync::atomic::Ordering;

monaka_producer::producer!(game: crate::Wolf);

/// Both Wolfensteins, as Monaka runs them.
pub struct Wolf;

/// This build's facts: the executable and its table.
pub struct Facts {
    pub exe: Module,
    pub build: &'static Build,
}

impl Game for Wolf {
    const MANIFESTS: &'static [Manifest] = manifest::MANIFESTS;
    const ADAPTER: Adapter<Wolf> = Adapter {
        components: &[
            &settings::SETTINGS,
            &cooking::frame_rate::FRAME_RATE_COMPONENT,
            &cooking::frame_pacing::FRAME_PACING_COMPONENT,
            &providers::window_size::EYE_SIZE,
            &PRESENT,
            &eye_site::EYES,
            &SINGLE_PASS,
            &LOOK,
            &cooking::body_forward::MOVE_HEADING_COMPONENT,
            &GUNS,
            &cooking::room_scale::ROOM_SCALE,
            &GUN,
            &HAND_RIG,
            &FINGERS,
            &MELEE,
            &GAMEPAD,
            &pad_connected::PAD_CONNECTED,
            &hud::HUD,
            &hud_pieces::HUD_PANELS,
            &hud_pieces::CROSSHAIR_COMPONENT,
            &cooking::auto_pickup::AUTO_PICKUP,
            &extras::mech_levers::MECH_LEVERS,
            &extras::hotwire::HOTWIRE,
            &PROBES,
        ],
    };
    type Build = Facts;
    type Frame = ();

    fn prepare(_: &'static Session, _: &'static Manifest, options: &Options) -> Result<Facts, Rejection> {
        let host = monaka_hook::module::host_name();
        let (build, exe) = engine::BUILDS.pick(|b| b.exe.eq_ignore_ascii_case(&host), &host)?;
        // Joined as it starts, nothing is touched before the engine has a screen layout (its
        // renderer up, its settings read): the launcher asks again.
        if engine::current_layout(&exe, build, &[build.layout_single, build.layout_left_right]).is_none() {
            return Err(Rejection::not_ready("the engine is not up yet (or is in another layout)"));
        }
        log!("{}", build.game);
        research::PROBES.configure(options);
        research::view::PLAYER_FIELDS.store(research::options().player && build.exe == "WolfOldBlood_x64.exe", Ordering::Relaxed);
        Ok(Facts { exe, build })
    }
}

/// How the run's eyes are made and placed ([`EyeSettings`]).
fn eyes(start: &Start<'_, Wolf>) -> EyeSettings {
    EyeSettings::from_options(start.options, start.manifest)
}

/// The components' places, for the detours to ask whether each is on.
pub(crate) const EYES_AT: usize = place::<Wolf>(&eye_site::EYES);
pub(crate) const SINGLE_PASS_AT: usize = place::<Wolf>(&SINGLE_PASS);
pub(crate) const GUNS_AT: usize = place::<Wolf>(&GUNS);
pub(crate) const GUN_AT: usize = place::<Wolf>(&GUN);
pub(crate) const HUD_AT: usize = place::<Wolf>(&hud::HUD);
pub(crate) const GAMEPAD_AT: usize = place::<Wolf>(&GAMEPAD);
pub(crate) const MELEE_AT: usize = place::<Wolf>(&MELEE);

/// Whether the component at `place` is on (off from its stop's first step).
pub(crate) fn on(place: usize) -> bool {
    monaka_producer::game::on::<Wolf>(place)
}

// ---- Stereo, look ----

/// The same-frame loop (`monaka_stereo::same_frame`, its OpenGL path): the head, each frame read
/// back at `SwapBuffers` and split into the eyes ([`providers::present_entry`]), the eye site
/// installed ([`providers::eye_site`]). Always on: with stereo off the whole frame goes to both
/// eyes. It reports a full-screen video, which goes to the flat screen.
pub const PRESENT: Component<Wolf> = Component { reports: &[Situation::Video], ..monaka_stereo::same_frame::same_frame() };

/// Single pass (the stereo method `stereo=single`, [`eyes::SINGLE_PASS_STEREO`]): the engine's scene programs converted to draw both
/// eyes at once, the left eye's view drawing the right eye's too and the right eye's view skipping
/// those draws ([`single_pass`]). Its work is at each `SwapBuffers`; at the stop the render thread
/// turns the converted programs back to one eye before the hooks come out.
pub const SINGLE_PASS: Component<Wolf> = Component {
    name: "single_pass",
    needs: &[Need::On(&eye_site::EYES)],
    settings: |options, _| {
        eyes::single_pass_on(options);
    },
    on: |run| {
        let on = eyes::single_pass_on(run.options);
        // Both eyes framed alike up and down: the hardware keeps only the left eye's height.
        eye_site::share_vertical(on);
        On::when(on, "the stereo method is not Single-pass (Same-frame)")
    },
    give_back: single_pass::draws::give_back,
    let_go: Some(single_pass::draws::let_go),
    summary: Some(single_pass::draws::summary),
    ..Component::NONE
};

/// The player looks and shoots where the head points (`monaka_body::look`, through the input
/// command's angles: [`providers::look`]): the head's turn shared into them after the game added
/// mouse and stick, handed back at the stop. Stands down while another camera has the view (a
/// cutscene, a takedown), which it reports.
pub const LOOK: Component<Wolf> = Component {
    options: &[aim(&[AIM_MOTION, ("head", "Look-based"), AIM_MOUSE], aims::HAND_RIG), cooking::body_forward::BODY_TURN, TURN],
    settings: |options, manifest| {
        options.switch("pause_hold", true);
        options.switch(cooking::body_forward::BODY_TURN.id, true);
        StickTurn::from_options(options, manifest, StickTurn::smooth(0.0), StickTurn::SNAP);
    },
    summary: Some(|| format!("{}\n{}\n{}", monaka_body::look::command::summary(), providers::camera_taken::summary(), providers::focus::summary())),
    ..monaka_body::look::look()
};

// ---- Guns, hand rig: the arms, the fingers ----

/// The guns in the controllers' hands (`aim=hand_rig`): each gun the game places is put on its
/// controller (`monaka_body::gun_hold::on_controller`; model, muzzle, laser follow), one gun or two
/// (dual wielding), what each hand holds reported ([`providers::held_item`]). The view stays on
/// the head.
pub const GUNS: Component<Wolf> = Component {
    name: "guns",
    features: &[Feature::HandRig],
    options: &[held_item::HIDE_FREE_HAND],
    settings: |options, _| {
        options.switch(held_item::HIDE_FREE_HAND.id, false);
    },
    needs: &[Need::On(&eye_site::EYES)],
    needs_channel: true,
    on: |run| On::when(run.shared.head_tracking && run.shared.aim.hand, "aim is not motion controls"),
    install: held_item::install_guns,
    release: monaka_body::controllers::close,
    summary: Some(|| format!("{}; {}", held_item::summary(), providers::fire::summary())),
    ..Component::NONE
};

/// Shooting from the barrel (`monaka_body::gun`, at the fire queue: [`providers::fire`]): each
/// shot from the gun of the hand that fired it, along its barrel, the game's spread on top.
pub const GUN: Component<Wolf> = monaka_body::gun::gun();

/// The hand rig (Monaka's, [`HandRig`]): the arms with the guns, the hands' skeleton moved onto
/// them each frame (`crate::providers::skeleton`).
pub const HAND_RIG: Component<Wolf> = Component {
    needs: &[Need::On(&GUNS)],
    settings: |options, _| {
        options.switch("arms", true);
    },
    on: |run| On::when(run.shared.aim.rig && run.options.switch("arms", true), "the arms stay as animated"),
    install: install_arms,
    summary: Some(skeleton::summary),
    ..hand_rig::hand_rig()
};

/// The hands' skeleton hangs from the hands, named nothing: each attach joint onto its gun, the
/// shoulders kept where the game's rifle hold has them, a free left hand's fingers found by the
/// skeleton's shape.
impl HandRig for Wolf {
    const RIG: Rig = Rig {
        method: Method::HandRooted { shoulders: skeleton::SHOULDERS, poles: skeleton::POLES, left_grip: skeleton::LEFT_GRIP, right_melee_grip: skeleton::melee_rest },
        holds: Holds::TwoHanded,
        climbing: Climbing::None,
        fingers: Some(ChainSource::Shape),
    };

    fn held(side: usize) -> Kind {
        held_item::held(side)
    }
}

/// The posing moment: captured at the hands' blend kick (the guns give the targets), posed in the
/// blend job at its bounds.
fn install_arms(start: &mut Start<'_, Wolf>) -> Result<(), Lack> {
    hand_rig::install::<Wolf>(start)?;
    let Facts { exe, build } = *start.build;
    let Some(facts) = build.guns.as_ref() else { return Err(Lack::Missing("no guns mapped".to_owned())) };
    start.hooks.site(&skeleton::KICK_ORIGINAL, &exe, &facts.arms_kick, skeleton::arms_kick)?;
    start.hooks.site(&skeleton::BOUNDS_ORIGINAL, &exe, &facts.pose_bounds, skeleton::pose_bounds)?;
    if let Some(follow) = &facts.follow_head {
        providers::follow_head::install(start.hooks, &exe, follow, &facts.layout)?;
    }
    Ok(())
}

/// A free hand's fingers on the tracked fingers (the left, while the right holds a one-handed gun).
pub const FINGERS: Component<Wolf> = Component { needs: &[Need::On(&HAND_RIG)], ..monaka_body::feature::fingers() };

/// Physical melee (`monaka_body::melee_hit`, on [`providers::melee`]): with the knife or the pipe
/// in the right hand, a swing of its controller sweeps the game's melee trace along the weapon,
/// with no canned attack; the trigger does nothing. The melee button stays the game's.
pub const MELEE: Component<Wolf> = Component { needs: &[Need::On(&GUNS)], quiet: providers::melee::quiet, ..monaka_body::melee_hit::physical_melee() };

// ---- Pad ----

/// The controllers as the game's pad, with the guns: with dual wielding its triggers fire the left
/// and right gun.
pub const GAMEPAD: Component<Wolf> = Component { needs: &[Need::On(&GUNS)], ..monaka_pad::feature::gamepad() };

// ---- Research ----

/// The research probes: their hooks and summaries. Each probe checks its own option.
pub const PROBES: Component<Wolf> = Component {
    install: install_probes,
    summary: Some(|| [research::melee::summary(), research::cutting::summary()].into_iter().flatten().collect::<Vec<_>>().join("\n")),
    ..monaka_producer::game::probes()
};

fn install_probes(start: &mut Start<'_, Wolf>) -> Result<(), Lack> {
    let Facts { exe, build } = *start.build;
    research::gl::install(start.hooks, &exe)?;
    research::melee::install(start.hooks, &exe, build)?;
    research::cutting::install(start.hooks, &exe, build)?;
    Ok(())
}

#[cfg(test)]
mod tests;
