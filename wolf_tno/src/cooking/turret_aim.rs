//! Waits for a Monaka ready-made for machines aimed by a hand: a mounted gun, and TNO's mech, aimed
//! by the right controller.
//!
//! A mounted gun has the view while the player uses it and turns by the change in the player's
//! command angles (`engine::Build::mounted_gun_vtable`), so the stick aims it and the view turns
//! with it; the mech the same way (`engine::Driven`, `engine::MechAim`), unless its levers aim it
//! (`mech_levers`). With the controllers placing the guns, the right controller aims them instead:
//!
//! - The eyes are built on the gun's view as it was when the player got on, levelled and turned
//!   by its yaw only ([`anchored`]): the world stays put while the gun turns in it, as a gun on a
//!   stand turns in front of whoever holds it (the gun's own camera turning with it would turn the
//!   world under the player, and the controller with it).
//! - Each command, the gun's aim (its view now) is turned toward where the right controller points
//!   in that world ([`steer`]): a share of the difference goes into the command angles, which the
//!   gun follows by a factor of its own.
//! - The stick's turn turns the world the eyes keep with it ([`steer`]): the mech walks where the
//!   stick turns it, and the controller still aims within that.

use crate::extras::mech_levers;
use crate::providers::{camera_taken, eye_site, held_item};
use monaka_contract::protocol::RIGHT_HAND;
use monaka_core::sync::Locked;
use monaka_producer::log;
use monaka_stereo::same_frame::PAIRS;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering::*};

/// The share of the gun's aim error put into each command: below one, so the gun settles on the
/// controller over a few frames without overshooting a frame behind.
const GAIN: f32 = 0.35;
/// The most the command angles move in one command (degrees).
const MOST_STEP: f32 = 15.0;
/// An aim error below this (degrees) is left alone: the controller's tremor.
const DEAD_ZONE: f32 = 0.15;
/// How often the aim's tallies are logged (commands steered).
const LOG_EVERY: u64 = 900;
/// A turn larger than this in one command (degrees) is no stick's (a teleport, a reload).
const MOST_STICK: f32 = 30.0;

/// The gun's view rows (forward, left, up) the eyes keep while on a mounted gun, levelled; None
/// off one.
static ANCHOR: Mutex<Option<[f32; 9]>> = Mutex::new(None);
/// The yaw the last steered command went out with (degrees): what the game's own turn (the stick)
/// is measured from at the next.
static LAST_YAW: Mutex<Option<f32>> = Mutex::new(None);
static STEERED: AtomicU64 = AtomicU64::new(0);
static TALLY: Mutex<Tally> = Mutex::new(Tally::new());

/// The aim's tallies since the last log: the error, and the gun's turn against the commands' (its
/// factor).
struct Tally {
    error_sum: f32,
    error_most: f32,
    count: u32,
    commanded: f32,
    turned: f32,
    last: Option<(f32, f32, u64)>,
}

impl Tally {
    const fn new() -> Self {
        Self { error_sum: 0.0, error_most: 0.0, count: 0, commanded: 0.0, turned: 0.0, last: None }
    }
}

/// Whether the right controller aims a mounted gun or the mech now: on one, with the controllers
/// placing the guns (the mech's levers not aiming it).
fn aiming() -> bool {
    let mech = camera_taken::driving() && mech_levers::in_mech() && !mech_levers::levers_on();
    (camera_taken::on_mounted_gun() || mech) && crate::on(crate::GUNS_AT) && held_item::setup().is_some()
}

/// Levelled rows (forward, left, up) turned by `yaw` degrees (positive left) from `rows`'.
pub fn turned_rows(rows: &[f32; 9], yaw: f32) -> [f32; 9] {
    let now = rows[1].atan2(rows[0]) + yaw.to_radians();
    let (s, c) = now.sin_cos();
    [c, s, 0.0, -s, c, 0.0, 0.0, 0.0, 1.0]
}

/// Levelled rows (forward, left, up) turned by the yaw of `rows`' forward.
pub fn level_rows(rows: &[f32; 9]) -> [f32; 9] {
    let yaw = rows[1].atan2(rows[0]);
    let (s, c) = yaw.sin_cos();
    [c, s, 0.0, -s, c, 0.0, 0.0, 0.0, 1.0]
}

/// At a view build, player `player`'s view `origin_axis` (origin, then rows) as the eyes are built
/// on: on a mounted gun aimed by the controller, the gun's view kept as it was when the player got
/// on (levelled), its origin the gun's now.
pub fn anchored(player: usize, origin_axis: [f32; 12]) -> [f32; 12] {
    if player != 0 {
        return origin_axis;
    }
    let mut anchor = ANCHOR.locked();
    if !aiming() {
        if anchor.take().is_some() {
            *LAST_YAW.locked() = None;
            log!("turret: off the mounted gun or the mech: the view follows the game's camera again");
        }
        return origin_axis;
    }
    let rows = *anchor.get_or_insert_with(|| {
        log!("turret: on a mounted gun or the mech: the right controller aims it, the view stays where the player got on and turns with the stick");
        level_rows(&std::array::from_fn(|k| origin_axis[3 + k]))
    });
    let mut out = origin_axis;
    out[3..].copy_from_slice(&rows);
    out
}

/// A forward axis's yaw (degrees, positive left) and pitch (degrees, positive down), as the
/// command's angles are.
pub fn yaw_pitch(forward: [f32; 3]) -> (f32, f32) {
    let yaw = forward[1].atan2(forward[0]).to_degrees();
    let length = (forward[0] * forward[0] + forward[1] * forward[1] + forward[2] * forward[2]).sqrt().max(f32::EPSILON);
    let pitch = -(forward[2] / length).clamp(-1.0, 1.0).asin().to_degrees();
    (yaw, pitch)
}

/// `a - b` in degrees, wrapped into -180..180.
pub fn wrapped(a: f32, b: f32) -> f32 {
    (a - b + 540.0).rem_euclid(360.0) - 180.0
}

/// The turn (pitch, yaw: degrees, as the command's) that moves a gun aimed at `gun` toward `wanted`
/// (yaw, pitch each).
pub fn step(gun: (f32, f32), wanted: (f32, f32)) -> (f32, f32) {
    let toward = |error: f32| if error.abs() < DEAD_ZONE { 0.0 } else { (error * GAIN).clamp(-MOST_STEP, MOST_STEP) };
    (toward(wanted.1 - gun.1), toward(wrapped(wanted.0, gun.0)))
}

/// Where the right controller points now in the world the eyes show (yaw, pitch), and the gun's
/// aim (its view's), while the right controller aims a mounted gun.
fn aims() -> Option<((f32, f32), (f32, f32))> {
    let rows = (*ANCHOR.locked())?;
    let setup = held_item::setup()?;
    let gun_view = eye_site::render_view()?;
    let mut view = gun_view;
    view[3..].copy_from_slice(&rows);
    let head = PAIRS.head().current()?;
    let space = eye_site::tracking_space(&view, &head)?;
    let palm = monaka_body::controllers::palm(RIGHT_HAND)?;
    let (_, axis) = held_item::gun_in_world(&space, &palm, setup.units_per_metre())?;
    let gun = yaw_pitch([gun_view[3], gun_view[4], gun_view[5]]);
    Some((gun, yaw_pitch([axis[0], axis[1], axis[2]])))
}

/// At each command, the command's angles (pitch, yaw: degrees) turned so the mounted gun follows
/// the right controller; as they were otherwise.
pub fn steer(pitch: f32, yaw: f32) -> Option<(f32, f32)> {
    if !aiming() {
        return None;
    }
    // The game's own turn since the last command (the stick's): the world the eyes keep turns too.
    let stick = LAST_YAW.locked().map_or(0.0, |last| wrapped(yaw, last));
    if stick != 0.0
        && stick.abs() <= MOST_STICK
        && let Some(rows) = ANCHOR.locked().as_mut()
    {
        *rows = turned_rows(rows, stick);
    }
    let Some((gun, wanted)) = aims() else {
        *LAST_YAW.locked() = Some(yaw);
        return None;
    };
    let (d_pitch, d_yaw) = step(gun, wanted);
    tally(gun, wanted, d_yaw);
    *LAST_YAW.locked() = Some(yaw + d_yaw);
    Some((pitch + d_pitch, yaw + d_yaw))
}

/// Counts the error and how far the gun turned for the turn asked, and logs them now and then.
fn tally(gun: (f32, f32), wanted: (f32, f32), d_yaw: f32) {
    let Ok(mut t) = TALLY.try_lock() else { return };
    let error = wrapped(wanted.0, gun.0).hypot(wanted.1 - gun.1);
    t.error_sum += error;
    t.error_most = t.error_most.max(error);
    t.count += 1;
    let build = eye_site::builds();
    if let Some((last_yaw, asked, at)) = t.last
        && build != at
    {
        t.commanded += asked.abs();
        t.turned += wrapped(gun.0, last_yaw).abs();
    }
    t.last = Some((gun.0, d_yaw, build));
    let n = STEERED.fetch_add(1, Relaxed) + 1;
    if n % LOG_EVERY == 1 {
        let factor = if t.commanded > 1.0 { t.turned / t.commanded } else { f32::NAN };
        log!(
            "turret: aimed by the right controller {n} commands; off the controller {:.1} degrees on average, at most {:.1}; the gun turned {:.2} of the turn asked",
            t.error_sum / t.count.max(1) as f32,
            t.error_most,
            factor
        );
        *t = Tally { last: t.last, ..Tally::new() };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn yaw_is_positive_left_and_pitch_positive_down() {
        let (yaw, pitch) = yaw_pitch([0.0, 1.0, 0.0]);
        assert!((yaw - 90.0).abs() < 1e-3 && pitch.abs() < 1e-3);
        let (_, pitch) = yaw_pitch([1.0, 0.0, -1.0]);
        assert!((pitch - 45.0).abs() < 1e-3, "{pitch}");
    }

    #[test]
    fn the_gun_turns_the_short_way_round() {
        assert!((wrapped(-170.0, 170.0) - 20.0).abs() < 1e-3);
        let (_, d_yaw) = step((170.0, 0.0), (-170.0, 0.0));
        assert!(d_yaw > 0.0, "{d_yaw}");
    }

    #[test]
    fn a_step_is_a_share_of_the_error_and_settles() {
        let mut gun = (0.0f32, 0.0f32);
        let wanted = (30.0, -10.0);
        for _ in 0..60 {
            let (d_pitch, d_yaw) = step(gun, wanted);
            gun = (gun.0 + d_yaw, gun.1 + d_pitch);
        }
        assert!(wrapped(wanted.0, gun.0).abs() < 0.5 && (wanted.1 - gun.1).abs() < 0.5, "{gun:?}");
        assert_eq!(step(wanted, wanted), (0.0, 0.0));
    }

    #[test]
    fn the_stick_turns_the_kept_view_left() {
        let rows = turned_rows(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], 90.0);
        assert!(rows[0].abs() < 1e-6 && (rows[1] - 1.0).abs() < 1e-6, "{rows:?}");
    }

    #[test]
    fn levelled_rows_keep_the_yaw_only() {
        let s = std::f32::consts::FRAC_1_SQRT_2;
        let rows = level_rows(&[s, 0.0, -s, 0.0, 1.0, 0.0, s, 0.0, s]);
        assert_eq!(rows, [1.0, 0.0, 0.0, -0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
    }
}
