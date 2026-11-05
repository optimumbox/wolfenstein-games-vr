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

use crate::providers::engine::{self, FocusUpdateFn};
use crate::providers::follow_head::{self, turned_rows};
use monaka_hook::module::Module;
use monaka_hook::{Hooks, InFlight, Original, mem};
use monaka_producer::game::Lack;
use monaka_producer::log;
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
        "focus: {} updates traced where the head looks, {} along the gun's barrel ({} of them while another camera had the view)",
        FOLLOWED.load(Relaxed),
        ALONG_BARREL.load(Relaxed),
        ALONG_BARREL_TAKEN.load(Relaxed)
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

/// The right hand's gun (origin, axis rows) while it finds its target by the focus.
fn barrel() -> Option<([f32; 3], [f32; 9])> {
    if !crate::on(crate::GUNS_AT) || !crate::providers::held_item::right_aimed_by_focus() {
        return None;
    }
    let placed = crate::providers::held_item::placed()[monaka_contract::protocol::RIGHT_HAND]?;
    (monaka_channel::tick().saturating_sub(placed.at) < BARREL_FRESH_MS).then_some(())?;
    ALONG_BARREL.fetch_add(1, Relaxed);
    if !BARREL_LOGGED.swap(true, Relaxed) {
        log!("focus: the gun in hand finds its target by the focus: it traces along the gun's barrel");
    }
    if crate::providers::camera_taken::controlled() {
        ALONG_BARREL_TAKEN.fetch_add(1, Relaxed);
        if !BARREL_TAKEN_LOGGED.swap(true, Relaxed) {
            log!("focus: another camera has the view with the gun in hand: the trace still runs along its barrel");
        }
    }
    Some((placed.origin, placed.axis))
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
    let moved = match barrel() {
        Some(barrel) => Some(barrel),
        None => crate::on(crate::EYES_AT).then(follow_head::head_from_camera).flatten().zip(view).map(|((shift, turn), (o, rows))| followed(o, &rows, play_shift(shift), &turn)),
    };
    match moved {
        Some((o, rows)) if o.iter().chain(&rows).all(|v| v.is_finite()) => {
            if !LOGGED.swap(true, Relaxed) {
                log!("focus: the player's focus traces where the head looks");
            }
            FOLLOWED.fetch_add(1, Relaxed);
            // SAFETY: forwards the game's own call with the view moved, in this frame's locals.
            unsafe { FOCUS_ORIGINAL.get(&flight)(tracker, player, o.as_ptr(), rows.as_ptr(), extra) }
        }
        // SAFETY: forwards the game's own call.
        _ => unsafe { FOCUS_ORIGINAL.get(&flight)(tracker, player, origin, axis, extra) },
    }
}

#[cfg(test)]
mod tests;
