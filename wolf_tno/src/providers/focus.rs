//! Provider part, to be: what the player looks at is where the head looks (the look's, once
//! Monaka has the trait).
//!
//! Each think the player's focus traces along its view object for what it looks at
//! (`engine::Build::focus_update`): the usable whose prompt shows, a choice a scene asks for by
//! looking at it (two cards on a table), the crosshair's target. In a cutscene that view is the
//! cutscene's camera, turned by the mouse's free look on a flat screen; with a headset the head
//! turns the eyes' view from the game camera instead (`eye_site`), which the trace never saw. The
//! trace is given the view moved and turned by the head as it was at the last view build
//! ([`follow_head::head_from_camera`]): the focus follows the eyes, in a cutscene and in play.
//!
//! A weapon that finds its target by the focus (the laser that cuts chains:
//! `engine::AIMED_BY_FOCUS`) aims with the gun, not the head: while the right hand holds one, the
//! trace runs along its barrel as the hand placed it, in play and while another camera has the view
//! (a scene that hands the player the laser to cut with: the gun is still in the hand, its beam
//! drawn from the muzzle).
//!
//! Traced along the barrel, where the trace hit is kept ([`barrel_hit`]): where the marksman
//! rifle's laser sight ends (`engine::DOTTED`, `eye_site::laser_beam`).

use crate::providers::engine::{self, FocusUpdateFn};
use crate::providers::follow_head::{self, turned_rows};
use monaka_hook::module::Module;
use monaka_hook::{Hooks, InFlight, Original, mem};
use monaka_producer::game::Lack;
use monaka_producer::log;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::*};

pub static FOCUS_ORIGINAL: Original<FocusUpdateFn> = Original::new();
static FOLLOWED: AtomicU64 = AtomicU64::new(0);
/// Updates traced along the gun's barrel, and of them while another camera had the view.
static ALONG_BARREL: AtomicU64 = AtomicU64::new(0);
static ALONG_BARREL_TAKEN: AtomicU64 = AtomicU64::new(0);
static LOGGED: AtomicBool = AtomicBool::new(false);

/// Hooks the player's focus update, where this build has it mapped.
pub fn install(hooks: &mut Hooks, exe: &Module, build: &engine::Build) -> Result<(), Lack> {
    let Some(site) = &build.focus_update else {
        log!("focus: not mapped for this build: the player's focus stays on the game camera");
        return Ok(());
    };
    hooks.site(&FOCUS_ORIGINAL, exe, site, focus_update)?;
    Ok(())
}

pub fn summary() -> String {
    format!(
        "focus: {} updates traced where the head looks, {} along the gun's barrel ({} of them while another camera had the view; {} hit found on it, {} not)",
        FOLLOWED.load(Relaxed),
        ALONG_BARREL.load(Relaxed),
        ALONG_BARREL_TAKEN.load(Relaxed),
        HITS.load(Relaxed),
        MISSES.load(Relaxed)
    )
}

/// The view `origin`, `rows` moved by `shift` and turned by `turn` (rows · turn).
pub fn followed(origin: [f32; 3], rows: &[f32; 9], shift: [f32; 3], turn: &[f32; 9]) -> ([f32; 3], [f32; 9]) {
    (std::array::from_fn(|k| origin[k] + shift[k]), turned_rows(rows, turn))
}

/// A placement older than this (ms) is not the gun's now.
const BARREL_FRESH_MS: u64 = 200;
static BARREL_LOGGED: AtomicBool = AtomicBool::new(false);

static BARREL_TAKEN_LOGGED: AtomicBool = AtomicBool::new(false);
static KNIFE_LOGGED: AtomicBool = AtomicBool::new(false);
static BARREL_SCENE_LOGGED: AtomicBool = AtomicBool::new(false);

/// The last trace along the barrel: the barrel (origin, axis rows), how far along it the trace hit
/// (units; None: nothing found on it), when (tick).
type BarrelHit = (([f32; 3], [f32; 9]), Option<f32>, u64);
static BARREL_HIT: Mutex<Option<BarrelHit>> = Mutex::new(None);
/// Barrel traces whose hit was found on the barrel, and not.
static HITS: AtomicU64 = AtomicU64::new(0);
static MISSES: AtomicU64 = AtomicU64::new(0);

/// The right gun's barrel (origin, axis rows) as last traced along, and how far along it the trace
/// hit (None: nothing found on it), if traced within [`BARREL_FRESH_MS`].
pub fn barrel_hit() -> Option<(([f32; 3], [f32; 9]), Option<f32>)> {
    let (barrel, hit, at) = (*BARREL_HIT.lock().ok()?)?;
    (monaka_channel::tick().saturating_sub(at) < BARREL_FRESH_MS).then_some((barrel, hit))
}

/// How far along the ray from `origin` along unit `forward` the trace that began at `start` and
/// reaches as far as `end` hit, from its candidate points (the result's hit point, where it
/// stopped): the first that lies on the ray (within 2% of its distance and 2 units of it) short of
/// its reach, with the trace begun at the ray's origin. A trace that stopped only at its reach hit
/// nothing.
pub fn hit_along(origin: [f32; 3], forward: [f32; 3], start: [f32; 3], end: [f32; 3], candidates: &[[f32; 3]]) -> Option<f32> {
    let sub = |a: [f32; 3], b: [f32; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    let dot = |a: [f32; 3], b: [f32; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let off = sub(start, origin);
    if !(dot(off, off).sqrt() < 2.0) {
        return None;
    }
    let reach = dot(sub(end, start), sub(end, start)).sqrt();
    candidates.iter().find_map(|&point| {
        let to = sub(point, origin);
        let along = dot(to, forward);
        let aside = (dot(to, to) - along * along).max(0.0).sqrt();
        (along > 1.0 && along.is_finite() && aside <= 2.0 + 0.02 * along && !(along >= 0.99 * reach)).then_some(along)
    })
}

/// Barrel traces logged with what each candidate read.
static SAMPLES: AtomicU64 = AtomicU64::new(0);

/// After a trace along the barrel `(origin, axis)` by `tracker`: where it hit, kept.
fn keep_hit(tracker: usize, (origin, axis): ([f32; 3], [f32; 9])) {
    let trace = tracker + engine::FOCUS_TRACE;
    let result = trace + engine::FOCUS_TRACE_RESULT;
    let read = |at: usize| mem::read::<[f32; 3]>(at);
    let hit = read(trace + engine::FOCUS_TRACE_START).zip(read(trace + engine::FOCUS_TRACE_END)).and_then(|(start, end)| {
        let candidates: Vec<[f32; 3]> = [result + engine::TRACE_POINT, result + engine::TRACE_END_POS].into_iter().filter_map(read).collect();
        let forward = [axis[0], axis[1], axis[2]];
        if SAMPLES.fetch_add(1, Relaxed) % 900 == 0 && SAMPLES.load(Relaxed) < 900 * 6 {
            let along = |p: [f32; 3]| (0..3).map(|k| (p[k] - origin[k]) * forward[k]).sum::<f32>();
            log!(
                "focus: a barrel trace: start {:.1} units off the barrel, reach {:.0}, hit point {:.0}, stop {:.0} along it",
                (0..3).map(|k| (start[k] - origin[k]).powi(2)).sum::<f32>().sqrt(),
                along(end),
                candidates.first().map_or(f32::NAN, |&p| along(p)),
                candidates.get(1).map_or(f32::NAN, |&p| along(p))
            );
        }
        hit_along(origin, forward, start, end, &candidates)
    });
    let n = if hit.is_some() { HITS.fetch_add(1, Relaxed) } else { MISSES.fetch_add(1, Relaxed) };
    if n == 0 {
        match hit {
            Some(d) => log!("focus: the barrel's trace hit {d:.0} units along it: a laser sight ends there"),
            None => log!("focus: a barrel's trace with no hit found on the barrel: a laser sight reaches {} m", crate::providers::eye_site::LASER_REACH),
        }
    }
    if let Ok(mut kept) = BARREL_HIT.lock() {
        *kept = Some(((origin, axis), hit, monaka_channel::tick()));
    }
}

/// The right hand's gun (origin, axis rows) while it finds its target by the focus.
fn barrel() -> Option<([f32; 3], [f32; 9])> {
    let knife = crate::cooking::knife_throw::readied();
    // A gun with its laser sight switched on aims by its barrel too.
    let lasered = crate::providers::held_item::right_laser_switched() && crate::cooking::sights::laser_switched_on();
    if !crate::on(crate::GUNS_AT) || !(knife || lasered || crate::providers::held_item::right_aimed_by_focus()) {
        return None;
    }
    let placed = crate::providers::held_item::placed()[monaka_contract::protocol::RIGHT_HAND]?;
    (monaka_channel::tick().saturating_sub(placed.at) < BARREL_FRESH_MS).then_some(())?;
    if knife {
        // A readied knife throw: traced along the blade, for the red dot where it goes.
        if !KNIFE_LOGGED.swap(true, Relaxed) {
            log!("focus: a knife throw readied: the trace runs along the blade");
        }
        return Some((placed.origin, crate::cooking::knife_throw::knife_axis(&placed.axis)));
    }
    ALONG_BARREL.fetch_add(1, Relaxed);
    if !BARREL_LOGGED.swap(true, Relaxed) {
        log!("focus: the gun in hand finds its target by the focus: it traces along the gun's barrel");
    }
    if crate::providers::camera_taken::controlled() {
        // A scene: only the laser cutter (handed to the player to cut with) aims by its barrel;
        // any other gun leaves the focus where the head looks (a scene may wait on what the
        // player looks at).
        if !crate::providers::held_item::right_named(engine::LASER_CUTTER) {
            if !BARREL_SCENE_LOGGED.swap(true, Relaxed) {
                log!("focus: another camera has the view with a gun in hand: the trace follows the head, not the gun");
            }
            return None;
        }
        ALONG_BARREL_TAKEN.fetch_add(1, Relaxed);
        if !BARREL_TAKEN_LOGGED.swap(true, Relaxed) {
            log!("focus: another camera has the view with the gun in hand: the trace still runs along its barrel");
        }
    }
    // From the muzzle where it is known, as the shots leave: the game's beam from the muzzle to
    // the hit then runs along the barrel.
    // Along its sights where they point off its axis.
    Some((crate::providers::held_item::aimed_origin(&placed), crate::providers::held_item::aimed_axis(&placed)))
}

static FROM_BODY_LOGGED: AtomicBool = AtomicBool::new(false);

/// The head's shift for the focus: none while the player's own camera has the view. What the
/// focus finds there the game checks again from the player's own eye as it acts on it (a
/// takedown's prompt shown from the head leaning in, refused from the body behind it: a plain
/// swing instead), so it traces from that eye, turned where the head looks, as on a flat screen.
/// A cutscene's camera has no body to check from: the head's shift stays.
fn play_shift(shift: [f32; 3]) -> [f32; 3] {
    if crate::providers::camera_taken::controlled() {
        return shift;
    }
    if !FROM_BODY_LOGGED.swap(true, Relaxed) {
        log!("focus: in play the focus traces from the player's own eye, turned where the head looks (the head {:.1} units off it now)", shift.iter().map(|v| v * v).sum::<f32>().sqrt());
    }
    [0.0; 3]
}

unsafe extern "system" fn focus_update(tracker: usize, player: usize, origin: *const f32, axis: *const f32, extra: usize) {
    let flight = InFlight::enter();
    let view = (!origin.is_null() && !axis.is_null())
        .then(|| Some((mem::read::<[f32; 3]>(origin as usize)?, mem::read::<[f32; 9]>(axis as usize)?)))
        .flatten();
    let barrel = barrel();
    let moved = match barrel {
        Some(barrel) => Some(barrel),
        None => crate::on(crate::EYES_AT).then(follow_head::head_from_camera).flatten().zip(view).map(|((shift, turn), (o, rows))| followed(o, &rows, play_shift(shift), &turn)),
    };
    match moved {
        Some((o, rows)) if o.iter().chain(&rows).all(|v| v.is_finite()) => {
            if !LOGGED.swap(true, Relaxed) {
                log!("focus: the player's focus traces where the head looks");
            }
            FOLLOWED.fetch_add(1, Relaxed);
            let far = far_use(tracker, player, o, &rows);
            // SAFETY: forwards the game's own call with the view moved, in this frame's locals.
            unsafe { FOCUS_ORIGINAL.get(&flight)(tracker, player, o.as_ptr(), rows.as_ptr(), extra) };
            far_done(tracker, far);
            if let Some(barrel) = barrel
                && tracker != 0
            {
                keep_hit(tracker, barrel);
            }
        }
        _ => {
            let far = view.and_then(|(o, rows)| far_use(tracker, player, o, &rows));
            // SAFETY: forwards the game's own call.
            unsafe { FOCUS_ORIGINAL.get(&flight)(tracker, player, origin, axis, extra) };
            far_done(tracker, far);
        }
    }
}

static FAR_USES: AtomicU64 = AtomicU64::new(0);

/// Before the focus update traced from `origin` along `rows`: an interaction usable now within reach
/// from far (`pickups::far_usable`) set as the focus's override, where none is set; the override
/// it replaced, to put back ([`far_done`]).
fn far_use(tracker: usize, player: usize, origin: [f32; 3], rows: &[f32; 9]) -> Option<u32> {
    if tracker == 0 || player == 0 {
        return None;
    }
    let at = tracker + engine::FOCUS_ENTITY_OVERRIDE;
    let old = mem::read::<u32>(at)?;
    if old != 0 && crate::providers::pickups::handle_live(old) {
        return None;
    }
    let units = crate::providers::held_item::setup().map_or(crate::eyes::UNITS_PER_METRE, |s| s.units_per_metre());
    let handle = crate::providers::pickups::far_usable(player, origin, [rows[0], rows[1], rows[2]], units)?;
    let _ = mem::write(at, handle);
    if FAR_USES.fetch_add(1, Relaxed) == 0 {
        log!("far use: an interaction within {} m where the head looks takes the focus: the use button reaches it", crate::providers::pickups::FAR_USE_METRES);
    }
    Some(old)
}

/// After the focus update: the override [`far_use`] replaced put back.
fn far_done(tracker: usize, old: Option<u32>) {
    if let Some(old) = old {
        let _ = mem::write(tracker + engine::FOCUS_ENTITY_OVERRIDE, old);
    }
}

#[cfg(test)]
mod tests;
