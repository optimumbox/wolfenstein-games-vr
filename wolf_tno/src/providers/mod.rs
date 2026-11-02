//! What only the Wolfensteins know: their engine facts, their hooks, and reading and writing the
//! game's state. The rule for this folder: a provider supplies something once for Monaka's
//! ready-mades to read, and does no feature's work itself (that is cooking, in
//! [`crate::cooking`]). Each file says which provider it is, or will be once Monaka has the trait.

pub(crate) mod camera_taken;
pub(crate) mod engine;
pub(crate) mod eye_site;
pub(crate) mod fire;
pub(crate) mod focus;
pub(crate) mod follow_head;
pub(crate) mod head_conformance;
pub(crate) mod held_item;
pub(crate) mod hud_pieces;
pub(crate) mod look;
pub(crate) mod melee;
pub(crate) mod pad_connected;
pub(crate) mod pickups;
pub(crate) mod present_entry;
pub(crate) mod settings;
pub(crate) mod skeleton;
pub(crate) mod window_size;
