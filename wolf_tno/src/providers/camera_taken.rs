//! Provider, to be: situations (a detector for `Situation::CameraTaken`), once Monaka has the
//! trait; today it reports the situation itself and answers the look's stand-down
//! ([`controlled`]).
//!
//! Cutscenes: while something else controls the player's view (a cutscene's camera, a takedown's:
//! `engine::Build::view_camera`), the game camera does not follow the player's view angles. Head
//! aim's turn in those angles would then be taken back out of a camera that never had it, leaving
//! the view on the cutscene's camera wherever the head looks. Meanwhile the look stands down
//! ([`look`](crate::providers::look)) and the head turns the cutscene's camera like any other
//! ([`eye_site`]).
//!
//! A mounted gun has the view the same way while the player uses it, and the look stands down for
//! it too: the gun turns by the change in the command angles (times a factor of its own, up to
//! its limits), so the head's share would pull it about, and keeping the player level would keep
//! pushing it toward the head. The share stays in the angles as it was ([`mounted`]): taken out,
//! the gun would turn by it as the player got on. The right stick aims it, its pitch lent back to
//! the game meanwhile (`monaka_pad::lend_right_stick_y`), and the head looks around the gun's
//! view.
//!
//! A machine the player drives (TNO's mech: `engine::Driven`) takes the view the same way and
//! turns by the change in the command angles too, so it counts as mounted ([`mounted`]): the
//! stick aims it and the head looks around its view.
//!
//! The player's view update asks the engine's own getter every frame; its answer is kept here. The
//! player it is asked for also says whether the weapon wheel is open, or a minigame of the game's
//! runs (hotwiring: `engine::Minigame`): both are steered by the right stick, so its pitch is lent
//! back too. While the wheel is open the game leaves the view angles alone, so the look holds the
//! share as it was ([`holds`]): the camera keeps the share it had, and the head looks around it.
//!
//! A grab's spring camera takes no control: it is the player's control camera
//! (`engine::ControlCamera`), which rewrites the player's own view each frame after the game built
//! it from the angles. While one is set the view counts as taken too. The render view
//! then leaves the player's first-person view, and the hands and guns follow the camera
//! (`skeleton::hands_view`, [`camera_away`]). Both are asked on the game thread only (the one that
//! builds the views; the getter is called on others too), where the game itself asks them.

use crate::providers::engine::{self, CastFn, ControlCameraFn, ControlInhibitedFn, Driven, Entities, HandsLayout, Minigame, ViewCameraFn};
use crate::providers::eye_site;
use monaka_hook::thread_id;
use monaka_hook::module::Module;
use monaka_hook::{InFlight, Original, mem};
use monaka_producer::log;
use monaka_producer::situation::{self, Situation};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering::*};

pub static VIEW_CAMERA_ORIGINAL: Original<ViewCameraFn> = Original::new();

/// The getter's last answer (a camera controls the view), the view build it came at, and the
/// view builds made while one did.
static CONTROLLED: AtomicBool = AtomicBool::new(false);
static SEEN_AT: AtomicU64 = AtomicU64::new(0);
static LINES: AtomicU32 = AtomicU32::new(0);
static TAKEOVERS: AtomicU64 = AtomicU64::new(0);
/// The mounted gun's vtable (absolute) and the player's weapon selection flags (an offset), from
/// the install.
static MOUNTED_GUN_VTABLE: AtomicUsize = AtomicUsize::new(0);
static SELECTION_FLAGS: AtomicUsize = AtomicUsize::new(0);
/// Whether a mounted gun had the view at the getter's last answer, and how often one took it.
static MOUNTED: AtomicBool = AtomicBool::new(false);
static MOUNTS: AtomicU64 = AtomicU64::new(0);
/// The driven machine's facts (the cast absolute), from the install; whether one had the view at
/// the game thread's last answer, and how often one took it.
static DRIVEN: OnceLock<Driven> = OnceLock::new();
static DRIVING: AtomicBool = AtomicBool::new(false);
static DRIVES: AtomicU64 = AtomicU64::new(0);
/// Whether the weapon wheel was open at the last answer, and how often it opened.
static WHEEL: AtomicBool = AtomicBool::new(false);
static WHEELS: AtomicU64 = AtomicU64::new(0);
/// The hands' layout, from the install (none: the build's hands are not mapped).
static LAYOUT: OnceLock<&'static HandsLayout> = OnceLock::new();
/// The executable's base, the game's entities and the minigame's place, from the install; whether
/// a minigame ran at the last answer, and how often one did.
static MINIGAME_FACTS: OnceLock<(usize, Entities, Minigame)> = OnceLock::new();
static MINIGAME: AtomicBool = AtomicBool::new(false);
static MINIGAMES: AtomicU64 = AtomicU64::new(0);
/// The `idPresentablePlayer` the getter was last asked for on the game thread.
static GAME_PLAYER: AtomicUsize = AtomicUsize::new(0);
/// The threads the getter was asked on, as logged (the first few).
static THREADS_SEEN: std::sync::Mutex<Vec<u32>> = std::sync::Mutex::new(Vec::new());
/// Whether the camera is off the player's first-person view.
static AWAY: AtomicBool = AtomicBool::new(false);
/// The `idPresentablePlayer` -> `idPlayer` offset and the engine's control camera getter
/// (absolute), from the install; whether a camera was set at the last answer, and how often one was.
static CONTROL_CAMERA: OnceLock<(usize, usize)> = OnceLock::new();
static SPRING: AtomicBool = AtomicBool::new(false);
static SPRINGS: AtomicU64 = AtomicU64::new(0);
/// The engine's control-held check (absolute, `engine::Build::control_inhibited`), from the
/// install; whether it held the player's control at the last answer, and how often it began to.
static CONTROL_INHIBITED: AtomicUsize = AtomicUsize::new(0);
static INHIBITED: AtomicBool = AtomicBool::new(false);
static INHIBITS: AtomicU64 = AtomicU64::new(0);

/// The camera counts as off the first-person view past this far (units) or this angle (degrees)
/// between their forward axes, and back on it within the second pair.
const OFF_VIEW: (f32, f32) = (8.0, 10.0);
const ON_VIEW: (f32, f32) = (2.0, 3.0);

/// An answer older than this many view builds no longer counts (the getter stopped being asked).
const FRESH_BUILDS: u64 = 10;
/// Each answer also goes into the run's situations (`Situation::CameraTaken`), holding this long
/// (ms): [`FRESH_BUILDS`] builds at a low frame rate.
const FRESH_MS: u64 = 250;

/// Whether something other than the player's own angles controls the view now: another camera, or
/// a control camera rewriting it.
pub fn controlled() -> bool {
    (CONTROLLED.load(Acquire) || SPRING.load(Acquire)) && fresh()
}

/// Whether a control camera rewrites the view now (a grab: a latch cut, a takedown): the game
/// builds it from the player's angles, so the head's share must be out of them for the head to
/// look around it once.
pub fn grabbed() -> bool {
    SPRING.load(Acquire) && fresh()
}

/// Whether `handle` names a live entity, checked as the game checks it ([`Entities`]).
fn live_entity(base: usize, entities: &Entities, handle: u32) -> bool {
    let mask = (1u32 << entities.bits) - 1;
    let number = (handle & mask) as usize;
    if number == mask as usize {
        return false;
    }
    let Some(game) = mem::read::<usize>(base + entities.game_local).filter(|&g| g != 0) else { return false };
    let spawn = mem::read::<i32>(game + entities.spawn_ids + number * 4);
    let entity = mem::read::<usize>(game + entities.entities + number * 8);
    spawn == Some((handle as i32) >> entities.bits) && entity.is_some_and(|e| e != 0)
}

/// Whether `player` (`idPresentablePlayer`) is in a minigame of the game's (hotwiring).
fn minigame_running(player: usize) -> bool {
    let Some((base, entities, minigame)) = MINIGAME_FACTS.get() else { return false };
    mem::read::<u32>(player + minigame.hud + minigame.handle).is_some_and(|handle| live_entity(*base, entities, handle))
}

/// Whether `player` (`idPresentablePlayer`) has a control camera set: the engine's own getter.
fn control_camera_set(player: usize) -> bool {
    let Some(&(offset, get)) = CONTROL_CAMERA.get() else { return false };
    let Some(game_player) = mem::read::<usize>(player + offset).filter(|&p| p != 0) else { return false };
    // SAFETY: `engine::ControlCamera::get` of this build (its fingerprint checked), which reads the
    // player's camera handle and the entity tables only, on the player the game is drawing.
    let get: ControlCameraFn = unsafe { std::mem::transmute(get) };
    // SAFETY: see above.
    unsafe { get(game_player) != 0 }
}

/// Whether the camera the eyes are built from is off the player's first-person view now (a grab):
/// the hands and guns are placed from it.
pub fn camera_away() -> bool {
    AWAY.load(Acquire) && fresh()
}

/// How far apart two views are (origin, then axis rows): their origins (units) and forward axes
/// (degrees).
pub fn views_apart(a: &[f32; 12], b: &[f32; 12]) -> (f32, f32) {
    let forward = |v: &[f32; 12]| [v[3], v[4], v[5]];
    let (fa, fb) = (forward(a), forward(b));
    let length = |v: [f32; 3]| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    let lengths = length(fa) * length(fb);
    let cos = if lengths > f32::EPSILON { (fa[0] * fb[0] + fa[1] * fb[1] + fa[2] * fb[2]) / lengths } else { 1.0 };
    (length([a[0] - b[0], a[1] - b[1], a[2] - b[2]]), cos.clamp(-1.0, 1.0).acos().to_degrees())
}

/// Whether the camera is off `player`'s first-person view, as last decided (`was`), with the views
/// `units` and `degrees` apart.
pub fn away_now(was: bool, units: f32, degrees: f32) -> bool {
    match was {
        true => units > ON_VIEW.0 || degrees > ON_VIEW.1,
        false => units > OFF_VIEW.0 || degrees > OFF_VIEW.1,
    }
}

/// Each answer: the render view (the last view build's) against `player`'s first-person view.
fn update_away(player: usize) {
    let (Some(layout), Some(render)) = (LAYOUT.get(), eye_site::render_view()) else { return };
    let Some(first) = mem::read::<[f32; 12]>(player + layout.player_view) else { return };
    let (units, degrees) = views_apart(&first, &render);
    let was = AWAY.load(Acquire);
    let away = away_now(was, units, degrees);
    if away != was {
        AWAY.store(away, Release);
        if LINES.fetch_add(1, Relaxed) < 40 {
            let build = eye_site::builds();
            match away {
                true => log!("view build {build}: the camera is {units:.1} units and {degrees:.0} degrees off the first-person view: the hands and guns follow the camera"),
                false => log!("view build {build}: the camera is back on the first-person view"),
            }
        }
    }
}

/// Whether that camera is a machine the player drives.
pub fn driving() -> bool {
    DRIVING.load(Acquire) && fresh()
}

/// How many times a machine the player drives took the view (one more each time the player gets
/// in, a checkpoint's reload too).
pub fn drives() -> u64 {
    DRIVES.load(Relaxed)
}

/// The `idPresentablePlayer` whose view the game thread last updated (0 before the first), and
/// its `idPlayer` (`engine::ControlCamera::player`), if it reads.
pub fn game_player() -> Option<(usize, usize)> {
    let presentable = GAME_PLAYER.load(Acquire);
    let &(offset, _) = CONTROL_CAMERA.get().filter(|_| presentable != 0)?;
    let player = mem::read::<usize>(presentable + offset).filter(|&p| p != 0)?;
    Some((presentable, player))
}

/// Whether that camera is a mounted gun or a machine the player drives.
pub fn mounted() -> bool {
    (MOUNTED.load(Acquire) || DRIVING.load(Acquire)) && fresh()
}

/// Whether that camera is a mounted gun (not a machine the player drives).
pub fn on_mounted_gun() -> bool {
    MOUNTED.load(Acquire) && fresh()
}

/// Whether the look keeps the head's share in the angles as it was: on a mounted gun ([`mounted`]),
/// while the weapon wheel is open with the game's pad on, or while a menu holds the player's
/// control with no other camera on the view (the journal). The player's view update then leaves
/// the view angles alone, so the camera keeps the share it had and the head looks around it.
pub fn holds() -> bool {
    let menu = (WHEEL.load(Acquire) && crate::on(crate::GAMEPAD_AT)) || (INHIBITED.load(Acquire) && !CONTROLLED.load(Acquire));
    mounted() || (menu && fresh())
}

/// Whether the engine holds `player`'s control now (asked on the game thread, as the engine asks).
fn control_held(player: usize) -> bool {
    let check = CONTROL_INHIBITED.load(Acquire);
    if check == 0 {
        return false;
    }
    // SAFETY: `engine::Build::control_inhibited` of this build (its fingerprint checked), asked of
    // the presentable player the game is updating, on the game thread, as its view update asks it.
    let check: ControlInhibitedFn = unsafe { std::mem::transmute(check) };
    // SAFETY: see above.
    unsafe { check(player) != 0 }
}

/// Whether the getter's last answer still counts.
fn fresh() -> bool {
    eye_site::builds().saturating_sub(SEEN_AT.load(Relaxed)) <= FRESH_BUILDS
}

pub fn install(exe: &Module, build: &'static engine::Build) {
    MOUNTED_GUN_VTABLE.store(exe.at(build.mounted_gun_vtable), Release);
    if let Some(driven) = build.driven {
        let _ = DRIVEN.set(Driven { cast: exe.at(driven.cast), ..driven });
    }
    SELECTION_FLAGS.store(build.weapon_selection_flags, Release);
    CONTROL_INHIBITED.store(exe.at(build.control_inhibited), Release);
    let _ = CONTROL_CAMERA.set((build.control_camera.player, exe.at(build.control_camera.get)));
    let _ = MINIGAME_FACTS.set((exe.base(), build.entities, build.minigame));
    if let Some(guns) = build.guns.as_ref() {
        let _ = LAYOUT.set(&guns.layout);
    }
}

pub fn summary() -> String {
    format!(
        "{} times a camera took the view, {} of them a mounted gun and {} a machine the player drives (aimed by the stick), {} times a control camera rewrote it (a grab), weapon wheel opened {} times, {} minigames, {} times a menu held the player's control",
        TAKEOVERS.load(Relaxed),
        MOUNTS.load(Relaxed),
        DRIVES.load(Relaxed),
        SPRINGS.load(Relaxed),
        WHEELS.load(Relaxed),
        MINIGAMES.load(Relaxed),
        INHIBITS.load(Relaxed)
    )
}

/// Whether `camera`, the getter's answer, is a mounted gun.
fn mounted_gun(camera: usize) -> bool {
    camera != 0 && mem::read::<usize>(camera) == Some(MOUNTED_GUN_VTABLE.load(Acquire))
}

/// Whether `camera`, the getter's answer, is a machine the player drives: its entity cast as the
/// game casts it.
fn driven_machine(camera: usize) -> bool {
    let Some(driven) = DRIVEN.get() else { return false };
    let Some(entity) = mem::read::<usize>(camera + driven.entity).filter(|&e| e != 0) else { return false };
    // The cast calls the entity's first virtual: only on an entity whose table reads.
    if mem::read::<usize>(entity).and_then(mem::read::<usize>).is_none() {
        return false;
    }
    // SAFETY: `engine::Driven::cast` of this build (its fingerprint checked), which asks the
    // entity's type only, on the entity of the presentable the getter just checked, on the game
    // thread.
    let cast: CastFn = unsafe { std::mem::transmute(driven.cast) };
    // SAFETY: see above.
    unsafe { cast(entity) != 0 }
}

/// Whether `player`'s weapon wheel is open.
fn wheel_open(player: usize) -> bool {
    mem::read::<u32>(player + SELECTION_FLAGS.load(Acquire)).is_some_and(|flags| flags & engine::WEAPON_WHEEL_OPEN != 0)
}

/// Logs a change of `flag` to `now` (within the getter's line budget).
fn changed(flag: &AtomicBool, now: bool, count: &AtomicU64, began: &str, ended: &str) {
    if flag.swap(now, AcqRel) == now {
        return;
    }
    if now {
        count.fetch_add(1, Relaxed);
    }
    if LINES.fetch_add(1, Relaxed) < 40 {
        log!("view build {}: {}", eye_site::builds(), if now { began } else { ended });
    }
}

/// Logs each of the first few threads the getter is asked on, and whether it is the game thread.
fn note_thread(thread: u32) {
    let Ok(mut seen) = THREADS_SEEN.try_lock() else { return };
    if seen.len() < 4 && !seen.contains(&thread) {
        seen.push(thread);
        let whose = match eye_site::game_thread() {
            0 => "before the first view build",
            game if game == thread => "the game thread",
            _ => "not the game thread",
        };
        log!("view camera getter on thread {thread} ({whose})");
    }
}

pub unsafe extern "system" fn view_camera(player: usize) -> usize {
    let flight = InFlight::enter();
    // SAFETY: forwards the game's own call.
    let camera = unsafe { VIEW_CAMERA_ORIGINAL.get(&flight)(player) };
    let mounted = mounted_gun(camera);
    changed(&MOUNTED, mounted, &MOUNTS, "a mounted gun has the view: the stick aims it, the head looks around", "off the mounted gun");
    let wheel = wheel_open(player);
    changed(&WHEEL, wheel, &WHEELS, "the weapon wheel opened: the stick's pitch goes to it", "the weapon wheel closed");
    let minigame = minigame_running(player);
    changed(&MINIGAME, minigame, &MINIGAMES, "a minigame (hotwiring) has the sticks: the right stick's pitch goes to it (the controllers)", "the minigame is over");
    // The control camera, the driven machine and the camera's place: on the game thread only,
    // where the game asks them.
    let thread = thread_id();
    note_thread(thread);
    if thread == eye_site::game_thread() {
        GAME_PLAYER.store(player, Release);
        let driven = driven_machine(camera);
        changed(&DRIVING, driven, &DRIVES, "a machine the player drives has the view (the mech): the stick aims it, the head looks around", "out of the machine");
        update_away(player);
        let spring = control_camera_set(player);
        changed(&SPRING, spring, &SPRINGS, "a control camera rewrites the view (a grab): the head looks around it", "the control camera lets the view go");
        let held = control_held(player);
        changed(&INHIBITED, held, &INHIBITS, "a menu holds the player's control (the journal): the head looks around", "the player has control again");
    }
    if (wheel || mounted || DRIVING.load(Acquire) || minigame) && crate::on(crate::GAMEPAD_AT) {
        monaka_pad::lend_right_stick_y();
    }
    let now = camera != 0;
    SEEN_AT.store(eye_site::builds(), Relaxed);
    situation::report(Situation::CameraTaken, now || SPRING.load(Acquire), FRESH_MS);
    crate::research::view::player(eye_site::builds(), player);
    changed(&CONTROLLED, now, &TAKEOVERS, "a camera controls the view (a cutscene, a mounted gun): the head looks around it", "the player's own view again");
    camera
}

#[cfg(test)]
mod tests;
