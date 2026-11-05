//! The Wolfensteins' own versions of features Monaka should own. The rule for this folder: each
//! file does a feature's work itself, which belongs in a Monaka ready-made, and is here only
//! because a provider (or the ready-made) is still missing; each file's first line says which one
//! it waits for. Once that exists, the file goes, and the adapter lists the ready-made instead.

pub(crate) mod auto_pickup;
pub(crate) mod body_forward;
pub(crate) mod frame_pacing;
pub(crate) mod frame_rate;
pub(crate) mod head;
pub(crate) mod hud;
pub(crate) mod journal;
pub(crate) mod knife_throw;
pub(crate) mod room_scale;
pub(crate) mod sights;
pub(crate) mod turret_aim;
