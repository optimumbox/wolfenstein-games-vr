//! Cooking, waiting for a Monaka ready-made for the run's heading (part of the look): the left
//! stick on the head's heading, the body's, or halfway between ([`MOVE_HEADING`]), the body's
//! turning with the player (`body_turn`).
//!
//! With head aim the game camera carries the head's yaw (`monaka_body::look`'s share in the
//! command angles), and the game moves the player relative to its camera: the run goes where the
//! head looks. Fully on the body's heading (the right stick's turn) a glance aside felt like
//! moving sideways; fully on the head's, the head steers the run, which is what physical turning
//! needs (as KF2VR's head-relative movement, its default). Before the game reads the pad, the
//! left stick is turned back by the chosen share of the head's turn ([`head_share_out`]). The
//! view, the aim and the shots stay the head's.
//!
//! Off the head's heading, with `body_turn` ([`BODY_TURN`]) the player turning in the room turns
//! the body: a heading of
//! the mod's own follows the head's ([`Body`]), never more than [`DEAD_ZONE`] behind and settling
//! onto it over [`SETTLE_SECONDS`], so a glance aside barely steers but a real turn does, and the
//! left stick is turned onto that heading. The right stick still turns the body as before.
//!
//! With `turn=snap` (Monaka's [`monaka_contract::manifest::TURN`]) the right stick turns the body
//! [`StickTurn::SNAP`]'s degrees at a push instead of smoothly ([`take_snapped`]: into the command's
//! angles, `providers::look`).

use crate::Wolf;
use crate::providers::camera_taken;
use monaka_body::look;
use monaka_contract::gamepad::Gamepad;
use monaka_contract::manifest::{Kind, Opt, Shown, choice, opt};
use monaka_core::options::Options;
use monaka_core::pad::{StickTurn, TurnLatch};
use monaka_core::sync::Locked;
use monaka_producer::game::{Component, Lack, Start};
use std::f32::consts::{PI, TAU};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering::*};
use std::time::Instant;

/// Where the left stick moves the player (`move_heading`): where the body faces (the default, only
/// the right stick steers), halfway, or where the head looks (physical turns steer; experimental,
/// offered in developer mode only).
pub const MOVE_HEADING: Opt = opt(
    "move_heading",
    "Movement direction",
    "Body: only the right stick turns. Halfway: between where you look and where the body faces.",
    choice(&[("head", "Head, experimental"), ("halfway", "Halfway"), ("body", "Body")], "body"),
)
.on(Shown::Controls)
.turns_off("head", BODY_TURN.id, "Movement follows your head already; room-scale movement is its own setting.");

/// The run's heading as `move_heading` asks, read as the run starts.
pub const MOVE_HEADING_COMPONENT: Component<Wolf> = Component {
    name: "move heading",
    options: &[MOVE_HEADING],
    settings: |options, _| {
        share_out(options);
    },
    install,
    ..Component::NONE
};

/// The shares of the head's turn taken out of the run's heading, by `move_heading` id.
const SHARES: [(&str, f32); 3] = [("head", 0.0), ("halfway", 0.5), ("body", 1.0)];

/// The run's share taken out, as `f32` bits (0.0 until read: the head's heading).
static SHARE_OUT: AtomicU32 = AtomicU32::new(0);

fn install(start: &mut Start<'_, Wolf>) -> Result<(), Lack> {
    SHARE_OUT.store(share_out(start.options).to_bits(), Relaxed);
    Ok(())
}

/// How much of the head's turn `options` take out of the run's heading (0: where the head looks,
/// 1: where the body faces).
pub fn share_out(options: &Options) -> f32 {
    options.choice("move_heading", &SHARES, 1.0)
}

/// The run's share taken out ([`MOVE_HEADING_COMPONENT`]).
pub fn head_share_out() -> f32 {
    f32::from_bits(SHARE_OUT.load(Relaxed))
}

/// The body turning with the player (the left stick walking where they face, not halfway to where
/// the right stick last turned them).
pub const BODY_TURN: Opt =
    opt("body_turn", "Room-scale turning", "Turning in your room turns your character.", Kind::Switch { default: true }).on(Shown::Controls);

/// How far the head may turn from the body before it takes the body along (radians).
pub const DEAD_ZONE: f32 = 45.0 * PI / 180.0;
/// How long the body takes to settle onto where the head faces (seconds, to about a third).
pub const SETTLE_SECONDS: f32 = 1.0;
/// A head turn this far between two pad reads is a recentre, not a turn (radians): the body jumps
/// with it.
const RECENTRE: f32 = 60.0 * PI / 180.0;
/// The longest step the settling takes at once (seconds): a pause is not a long settle.
const LONGEST_STEP: f32 = 0.1;

static TURNS: AtomicBool = AtomicBool::new(false);

/// The right stick's snap turn (`turn=snap`), and its latch; none: the game's own smooth turn.
static SNAP: Mutex<Option<(StickTurn, TurnLatch)>> = Mutex::new(None);
/// Snapped degrees (right positive) not yet in a command ([`take_snapped`]).
static SNAPPED: Mutex<f32> = Mutex::new(0.0);

/// At the look's install: whether the body turns with the player this run, and how the right
/// stick turns.
pub fn configure(options: &monaka_core::options::Options, manifest: &monaka_contract::manifest::Manifest) {
    TURNS.store(options.switch(BODY_TURN.id, true), Relaxed);
    let turn = StickTurn::from_options(options, manifest, StickTurn::smooth(0.0), StickTurn::SNAP);
    *SNAP.locked() = matches!(turn, StickTurn::Snap { .. }).then(|| (turn, TurnLatch::new()));
}

/// The snap turn on the pad: while the player's own angles turn the view (not a mounted gun, the
/// mech, the weapon wheel or another camera), the right stick's sideways push turns once per push,
/// into the next command ([`take_snapped`]), and the game sees no sideways push.
fn snap(pad: &mut Gamepad) {
    let mut snap = SNAP.locked();
    let Some((turn, latch)) = snap.as_mut() else { return };
    // The look holds the head's share while the player's own angles turn the view (the command
    // delivery leaves `look::live` unset).
    if look::baked::<Wolf>(None).is_none() || camera_taken::holds() || camera_taken::controlled() {
        return;
    }
    let degrees = turn.turn(f32::from(pad.thumb_rx) / 32767.0, 0.0, latch);
    if degrees != 0.0 {
        *SNAPPED.locked() += degrees;
    }
    pad.thumb_rx = 0;
}

/// The snapped degrees (right positive) for the command being built, taken.
pub fn take_snapped() -> f32 {
    std::mem::take(&mut *SNAPPED.locked())
}

/// The pad the game reads, last (`monaka_pad::rewrite`): the knife thrown by the trigger
/// (`cooking::knife_throw`), the right trigger let go while swings attack (`providers::melee`),
/// the right stick's snap turn, the left stick onto the body's heading, the left trigger let go
/// while it would only zoom (`cooking::sights`), and Y held opening the journal
/// (`cooking::journal`).
pub fn pad(pad: &mut Gamepad) {
    if crate::providers::eye_site::in_menu() {
        menu_stick(pad);
        return;
    }
    crate::cooking::sights::no_zoom(pad);
    crate::cooking::journal::pad(pad);
    // The knife's throw reads the trigger before the swings let go of it.
    crate::cooking::knife_throw::pad(pad);
    crate::providers::melee::let_go_of_trigger(pad);
    snap(pad);
    let share = look::baked::<Wolf>(None);
    let share_out = head_share_out();
    let out = match (share_out == 0.0, TURNS.load(Relaxed), share) {
        // On the head's heading: as pushed (the head steers, physical turns with it).
        (true, ..) => None,
        (false, true, Some(share)) => Some(BODY.locked().follow(share.yaw, Instant::now())),
        (false, true, None) => {
            // Not looking by the head (a cutscene, another camera): the body starts again from
            // where the head faces.
            BODY.locked().reset();
            None
        }
        (false, false, share) => share.map(|share| share.yaw * share_out),
    };
    if let Some(out) = out {
        (pad.thumb_lx, pad.thumb_ly) = turned_stick(pad.thumb_lx, pad.thumb_ly, out);
    }
    // Last, on the game camera's own heading: room-scale following's push.
    crate::cooking::room_scale::pad(pad);
}

/// A stick past this share of its range is a menu step, held until it is back under
/// [`MENU_REST`] (a light push steps as on a pad; the stick's noise at the edge does not).
const MENU_STEP: f32 = 0.35;
const MENU_REST: f32 = 0.2;

/// In a menu: the left stick as a steady step along its main axis, full or centred. The VR stick's
/// own noise and the pad's dither change the state every few reads, and the menu takes a changing
/// stick as a push that has not settled, stepping seconds late.
fn menu_stick(pad: &mut Gamepad) {
    static LOGGED: AtomicU32 = AtomicU32::new(0);
    static LAST: Mutex<(i16, i16)> = Mutex::new((0, 0));
    let mut last = LAST.locked();
    let step = menu_step(pad.thumb_lx, pad.thumb_ly, *last != (0, 0));
    (pad.thumb_lx, pad.thumb_ly) = step;
    if *last != step {
        if LOGGED.fetch_add(1, Relaxed) < 200 {
            monaka_producer::log!("menu stick: {:?} -> {:?} handed to the game", *last, step);
        }
        *last = step;
    }
}

/// The step a left stick at `x`, `y` is in a menu: its main axis at full, past [`MENU_STEP`]
/// (past [`MENU_REST`] while a step is `held`).
pub fn menu_step(x: i16, y: i16, held: bool) -> (i16, i16) {
    let (fx, fy) = (f32::from(x) / 32767.0, f32::from(y) / 32767.0);
    if fx.hypot(fy) < if held { MENU_REST } else { MENU_STEP } {
        return (0, 0);
    }
    let full = |v: f32| if v < 0.0 { -32767 } else { 32767 };
    if fx.abs() > fy.abs() { (full(fx), 0) } else { (0, full(fy)) }
}

/// The body's heading of its own, in tracking space (radians, positive left, as the head's share).
#[derive(Clone, Copy, Debug, Default)]
pub struct Body {
    /// Its heading and the head's at the last read, and when.
    yaw: Option<(f32, f32, Instant)>,
}

static BODY: Mutex<Body> = Mutex::new(Body { yaw: None });

impl Body {
    /// The head faces `head` at `now`: the body follows, and what the head is turned from it (what
    /// the left stick is turned back by).
    pub fn follow(&mut self, head: f32, now: Instant) -> f32 {
        let body = match self.yaw {
            None => head,
            Some((body, last_head, at)) => {
                let turned = wrapped(head - last_head);
                // A recentre: the body goes round with the head.
                let body = if turned.abs() > RECENTRE { body + turned } else { body };
                let step = now.duration_since(at).as_secs_f32().min(LONGEST_STEP);
                followed(body, head, step)
            }
        };
        self.yaw = Some((body, head, now));
        wrapped(head - body)
    }

    pub fn reset(&mut self) {
        self.yaw = None;
    }
}

/// The body at `body` after the head facing `head` for `seconds`: settled toward it, and never
/// more than [`DEAD_ZONE`] away.
pub fn followed(body: f32, head: f32, seconds: f32) -> f32 {
    let off = wrapped(head - body).clamp(-DEAD_ZONE, DEAD_ZONE) * (-seconds / SETTLE_SECONDS).exp();
    wrapped(head - off)
}

/// `angle` in (-pi, pi].
fn wrapped(angle: f32) -> f32 {
    let a = (angle + PI).rem_euclid(TAU) - PI;
    if a <= -PI { a + TAU } else { a }
}

/// The stick `(x, y)` (x right, y forward) given on a heading `yaw` radians left of the one wanted,
/// turned onto it.
pub fn turned_stick(x: i16, y: i16, yaw: f32) -> (i16, i16) {
    let (s, c) = (-yaw).sin_cos();
    let (x, y) = (f32::from(x), f32::from(y));
    let axis = |v: f32| v.round().clamp(-32768.0, 32767.0) as i16;
    (axis(x * c - y * s), axis(x * s + y * c))
}

#[cfg(test)]
mod tests;
