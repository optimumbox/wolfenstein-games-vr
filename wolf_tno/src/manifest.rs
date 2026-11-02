//! The two Wolfensteins' game info, for the launcher and the loader (`monaka_contract::manifest`); the
//! options are those of the components ([`crate::Wolf`], `monaka_producer::game::options`).

use crate::Wolf;
use monaka_contract::manifest::*;
use monaka_producer::game::{options, options_len};

/// Each game's options: those of the components that serve it ([`crate::Wolf`]; the mech's levers
/// and hotwiring by hand are The New Order's only).
const OPTIONS: [Opt; options_len::<Wolf>("wolf_tno")] = options::<Wolf, { options_len::<Wolf>("wolf_tno") }>("wolf_tno");
const TOB_OPTIONS: [Opt; options_len::<Wolf>("wolf_tob")] = options::<Wolf, { options_len::<Wolf>("wolf_tob") }>("wolf_tob");

const BEFORE: &str = "Start the game from Steam first. Set its display mode to windowed: only then can it render at the headset's resolution.God rays, lens flares, dim shadows and screen-space reflections are switched off while VR runs (the game only gets them right in one eye).";

/// One producer for both games: the running executable picks its engine facts (`engine::ROWS`).
pub const MANIFESTS: &[Manifest] = &[
    Manifest {
        id: "wolf_tno",
        name: "Wolfenstein: The New Order",
        status: Status::Alpha,
        executables: &["WolfNewOrder_x64.exe"],
        steam_app: 201810,
        producer: "monaka_wolf_tno",
        log: "monaka_wolf.log",
        options_file: "wolf-options.txt",
        run_prefix: "wolf",
        before: BEFORE,
        options: &OPTIONS,
        // No aim ring on the guns: they aim with the hands, and the sights show where (as the HUD's
        // crosshair, hidden by default).
        left_hand_aims: false,
        weapon_ring: false,
        ..Manifest::GAME
    },
    Manifest {
        id: "wolf_tob",
        name: "Wolfenstein: The Old Blood",
        status: Status::Experimental,
        executables: &["WolfOldBlood_x64.exe"],
        steam_app: 350080,
        producer: "monaka_wolf_tno",
        log: "monaka_wolf.log",
        options_file: "wolf-options.txt",
        run_prefix: "wolf",
        before: BEFORE,
        options: &TOB_OPTIONS,
        left_hand_aims: false,
        weapon_ring: false,
        ..Manifest::GAME
    },
];
