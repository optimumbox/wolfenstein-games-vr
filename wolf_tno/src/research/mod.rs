//! Research: the probes that find out how id Tech 5 (Wolfenstein) works, kept apart from the
//! adapter.
//!
//! The adapter calls in here at the moments a probe watches (`research::view::seen(..)` at the
//! view builder, `research::gl::frame(..)` at each SwapBuffers), one line each, and never says
//! which probes listen: each checks its own option ([`options`]) and returns at once when it is
//! off. The options (`probe_*`) come from the same options file as the adapter's and are never
//! shown by the launcher.
//!
//! A probe that has found its answer is a candidate for deletion: its finding belongs in the
//! research notes, its facts in `providers/engine.rs`.

pub mod cutting;
pub mod cvars;
pub mod freeze;
pub mod gl;
pub mod guns;
pub mod hud_sprites;
pub mod melee;
pub mod panels;
pub mod profile;
pub mod shaders;
pub mod view;

use monaka_producer::research::Research;

/// What this run probes.
#[derive(Clone, Debug, Default)]
pub struct Options {
    /// `probe_view`: log the player's render view for the first frames (field checks).
    pub view: bool,
    /// `probe_gl`: log the viewport and scissor calls of two frames (finding the HUD).
    pub gl: bool,
    /// `probe_guns`: synthetic controllers in front of the head hold the guns (PC checks of
    /// `aim=controller` without a headset).
    pub guns: bool,
    /// `probe_melee`: log the player's melee trace ([`melee`]).
    pub melee: bool,
    /// `probe_panels`: write the hands' ammo panels as images ([`panels`]).
    pub panels: bool,
    /// `probe_profile`: sample where the render thread's time goes ([`profile`]).
    pub profile: bool,
    /// `probe_fixed_rate`: the game's rate goes to the `frame_rate` ceiling and stays there,
    /// whatever the PC holds (measuring the frames at that rate).
    pub fixed_rate: bool,
    /// `probe_freeze_at=<seconds>`: the world frozen that long after the first view build, to the
    /// end (0: never; [`freeze`]).
    pub freeze_at: f32,
    /// `probe_cvars=<name>@<rva>=<value>,...`: integer cvars set for the run ([`cvars`]).
    pub cvars: String,
    /// `probe_cycle=<name>=<value>[+...],...`: candidates tried in turn in play ([`cvars::prepare_cycle`]).
    pub cycle: String,
    /// `probe_extra_yaw=<degrees>`: the eyes' views turned this much more (left positive) as they
    /// are written: whether the world is drawn from them or from the game camera.
    pub extra_yaw: f32,
    /// `probe_items_on_view`: held items placed from the player's view alone, not on the arms model
    /// as drawn (A/B of `skeleton::onto_arms_as_drawn`).
    pub items_on_view: bool,
    /// `probe_cutting`: log what the laser's cut does to what it hits ([`cutting`]).
    pub cutting: bool,
    /// `probe_hud_sprites`: log where each of the HUD movie's sprites first draws ([`hud_sprites`]).
    pub hud_sprites: bool,
    /// `probe_shaders`: write every shader the game gave the driver beside the run's log
    /// ([`shaders`]).
    pub shaders: bool,
    /// `probe_player`: log The Old Blood's player fields that can hand the view to something else
    /// ([`view::player`]).
    pub player: bool,
}

impl Options {
    pub fn from_options(options: &monaka_core::options::Options) -> Self {
        Self {
            view: options.switch("probe_view", false),
            gl: options.switch("probe_gl", false),
            guns: options.switch("probe_guns", false),
            melee: options.switch("probe_melee", false),
            panels: options.switch("probe_panels", false),
            profile: options.switch("probe_profile", false),
            fixed_rate: options.switch("probe_fixed_rate", false),
            freeze_at: options.number("probe_freeze_at", 0.0, 3600.0, 0.0),
            cvars: options.text("probe_cvars").unwrap_or_default().to_owned(),
            cycle: options.text("probe_cycle").unwrap_or_default().to_owned(),
            extra_yaw: options.number("probe_extra_yaw", -90.0, 90.0, 0.0),
            items_on_view: options.switch("probe_items_on_view", false),
            cutting: options.switch("probe_cutting", false),
            hud_sprites: options.switch("probe_hud_sprites", false),
            player: options.switch("probe_player", false),
            shaders: options.switch("probe_shaders", false),
        }
    }
}

/// This run's probes, read at the start.
pub static PROBES: Research<Options> = Research::new(Options::from_options);

/// This run's probes (none before they are read).
pub fn options() -> &'static Options {
    PROBES.get()
}

