//! Provider: the held item (`monaka_body::held::HeldItems`, with the hand rig's `held`): what each
//! hand holds (a gun, one or two: [`held`]; the left hand a gun of its own while dual wielding,
//! [`dual`]), which hand a held item's tag is ([`side_of_tag`]), whether the right hand's weapon is
//! one-handed ([`right_one_handed`]), and where each hand's gun is, which the game places itself
//! (`HeldItems::placed`): its barrel runs along its forward axis from its origin.
//!
//! Each frame the hands' item placer (`engine::Guns::place_item`) puts each held item's model on
//! its hand tag of the hands model, then points the muzzle and laser from it. For the right and
//! left hand tags the tag's world transform (`engine::Guns::tag_transform`, its first call inside
//! the placer) is replaced by where the gun goes on its controller ([`gun_target`]:
//! `monaka_body::gun_hold::on_controller` on the eye site's tracking space,
//! [`eye_site::tracking_space`]), so the gun model and everything the game derives from it follow
//! the controller.
//!
//! `probe_guns` puts synthetic controllers in front of the head ([`crate::research::guns`]).
//!
//! The log follows the right gun against what the eyes see ([`watch`]): when the first-person view
//! the gun is placed from leaves the render view the eyes are built from, when the placed gun leaves
//! the sight of that view, when the game stops placing it or holds another weapon.

use crate::providers::engine::{self, HandsLayout, PlaceItemFn, TagTransformFn};
use crate::providers::{eye_site, skeleton};
use crate::{Facts, Wolf};
use monaka_body::held::{HeldItems, InWorld, Kind, Line, NoModel, WorldLine};
use monaka_core::camera::{Mat34, all_finite, apply3, compose, from_rotation, mul3, position, rotation_part, transpose3};
use monaka_core::math::{Vec3, cross, dot, length, midpoint, scale, sub, unit};
use monaka_contract::manifest;
use monaka_contract::protocol::{HandPose, HeadPose, LEFT_HAND, RIGHT_HAND};
use monaka_core::sync::Locked;
use monaka_hook::{InFlight, Original, mem};
use monaka_producer::game::{Lack, Start};
use monaka_producer::log;
use monaka_stereo::same_frame::PAIRS;
use std::cell::Cell;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering::*};
use std::sync::{Mutex, OnceLock};

/// What each hand holds as of the latest kick: the right hand its gun or melee weapon; the left one
/// a gun of its own (two guns) or the right one's (both hands on it), or nothing (beside a
/// one-handed weapon).
pub static HELD: Mutex<[Kind; 2]> = Mutex::new([Kind::Gun; 2]);

/// What hand `side` holds as of the latest kick (the hand rig's `held`).
pub fn held(side: usize) -> Kind {
    HELD.lock().map_or(Kind::Gun, |held| held[side.min(1)])
}

/// The guns' barrels: along each gun's forward axis from its origin (no muzzle length known).
impl HeldItems for Wolf {
    type Model = NoModel;

    fn line(_side: usize) -> Option<Line> {
        Some(BARREL)
    }

    fn model(_side: usize) -> Option<NoModel> {
        None
    }

    fn placed(side: usize) -> Option<InWorld> {
        let gun = PLACED.locked()[side.min(1)]?;
        let [f0, f1, f2, l0, l1, l2, u0, u1, u2] = gun.axis;
        let [o0, o1, o2] = gun.origin;
        let holder: Mat34 = [f0, l0, u0, o0, f1, l1, u1, o1, f2, l2, u2, o2];
        let line = WorldLine { from: gun.origin, along: [f0, f1, f2], length: None, edge: None };
        Some(InWorld { holder, line: Some(line), posed: gun.at })
    }
}

/// A gun's barrel in its own frame (the holder's columns are its forward, left and up axes).
const BARREL: Line = Line { from: [0.0; 3], along: [1.0, 0.0, 0.0], length: None, edge: None };

/// The side whose controller holds what sits on hand tag `tag`.
pub fn side_of_tag(tag: i32) -> Option<usize> {
    match tag {
        0 => Some(RIGHT_HAND),
        1 => Some(LEFT_HAND),
        _ => None,
    }
}

static ONE_HANDED_LOGGED: [AtomicBool; 2] = [AtomicBool::new(false), AtomicBool::new(false)];

/// The right hand's weapon (`idWeapon`), if it holds one.
pub fn right_weapon(layout: &HandsLayout, hands: usize) -> Option<usize> {
    mem::read::<usize>(hands + layout.right_item + engine::ITEM_WEAPON).filter(|&w| w != 0)
}

/// The right hand's weapon decl (`idDeclWeapon`), if it holds a weapon.
pub(crate) fn right_decl(layout: &HandsLayout, hands: usize) -> Option<usize> {
    right_weapon(layout, hands).and_then(|w| mem::read::<usize>(w + engine::INVENTORY_ITEM_DECL)).filter(|&d| d != 0)
}

/// A decl's name (`weapon/...`).
pub fn decl_name(decl: usize) -> Option<String> {
    mem::read::<usize>(decl + engine::DECL_NAME).filter(|&p| p != 0).and_then(|p| mem::read_c_string(p, 128))
}

/// The right hand's item by name: its decl and name, kept while the decl stays.
static RIGHT_NAME: Mutex<(usize, Option<String>)> = Mutex::new((0, None));
static NAMES_LOGGED: AtomicU64 = AtomicU64::new(0);
/// The most changes of the right hand's item the log names.
const NAME_LINES: u64 = 60;

/// `f` of the right hand's item's decl name (`weapon/...`; None: no weapon). Logged each time it
/// changes.
fn with_right_name<R>(layout: &HandsLayout, hands: usize, f: impl FnOnce(Option<&str>) -> R) -> R {
    let decl = right_decl(layout, hands).unwrap_or(0);
    let mut kept = RIGHT_NAME.locked();
    if kept.0 != decl {
        let name = (decl != 0).then(|| decl_name(decl)).flatten();
        if NAMES_LOGGED.fetch_add(1, Relaxed) < NAME_LINES {
            let carried = name.as_deref().is_some_and(|n| engine::CARRIED_TWO_HANDED.contains(&n));
            log!("held: the right hand holds {} (decl {decl:#x}){}", name.as_deref().unwrap_or("no weapon"), if carried { ": carried in both hands, between the controllers" } else { "" });
        }
        *kept = (decl, name);
    }
    f(kept.1.as_deref())
}

/// Whether the right hand's item, as last placed, throws knives (`engine::THROWS_KNIVES`).
pub fn right_throws_knives() -> bool {
    RIGHT_NAME.locked().1.as_deref().is_some_and(|n| engine::THROWS_KNIVES.contains(&n))
}

/// Whether the right hand's item, as last placed, finds its target by the player's focus
/// (`engine::AIMED_BY_FOCUS`).
pub fn right_aimed_by_focus() -> bool {
    RIGHT_NAME.locked().1.as_deref().is_some_and(|n| engine::AIMED_BY_FOCUS.contains(&n))
}

/// Whether the right hand's item is carried in both hands from its middle
/// (`engine::CARRIED_TWO_HANDED`).
pub fn right_carried(layout: &HandsLayout, hands: usize) -> bool {
    with_right_name(layout, hands, |name| name.is_some_and(|n| engine::CARRIED_TWO_HANDED.contains(&n)))
}

/// Whether the right hand holds a one-handed weapon (`idDeclWeapon::oneHandedWeapon`).
pub fn right_one_handed(layout: &HandsLayout, hands: usize) -> bool {
    let one_handed = right_decl(layout, hands).and_then(|d| mem::read::<u8>(d + layout.one_handed)).is_some_and(|b| b == 1);
    if !ONE_HANDED_LOGGED[one_handed as usize].swap(true, Relaxed) {
        log!("arms: the right hand holds a {}-handed weapon: the left hand {}", if one_handed { "one" } else { "two" }, if one_handed { "follows its controller" } else { "stays on the weapon" });
    }
    one_handed
}

/// What the right hand holds: a melee weapon (the knife; The Old Blood's pipes: one-handed, in a
/// melee slot) or a gun.
pub fn right_kind(layout: &HandsLayout, hands: usize) -> Kind {
    let melee = right_decl(layout, hands).is_some_and(|d| {
        mem::read::<u8>(d + layout.one_handed) == Some(1) && mem::read::<i32>(d + layout.weapon_slot).is_some_and(|slot| layout.melee_slots.contains(&slot))
    });
    if melee { Kind::Melee } else { Kind::Gun }
}

// ---- Where the guns go ----

pub static PLACE_ORIGINAL: Original<PlaceItemFn> = Original::new();
pub static TAG_ORIGINAL: Original<TagTransformFn> = Original::new();

/// Whether the arms follow the guns in the controllers' hands (`arms`; off: as animated).
pub(crate) fn arms(options: &monaka_core::options::Options) -> bool {
    options.switch("arms", true)
}

/// The guns on the controllers: the item placer and the tag transform hooked, the controllers
/// opened (the guns' component, `crate::GUNS`).
pub fn install_guns(start: &mut Start<'_, Wolf>) -> Result<(), Lack> {
    let Facts { exe, build } = *start.build;
    let Some(facts) = build.guns.as_ref() else {
        return Err(Lack::Missing(format!("where {} places and fires its guns is not mapped; the head aims", build.game)));
    };
    monaka_body::controllers::open(start.channel()?);
    // With the guns in the controllers' hands, the arms follow them (`arms=0`: as animated).
    let rig = start.shared.aim.rig && arms(start.options);
    let hide_free_hand = rig && start.options.switch(HIDE_FREE_HAND.id, false);
    install(&facts.layout, crate::eyes(start).units_per_metre, crate::research::options().guns, rig, rig && start.shared.fingers.tracking, hide_free_hand);
    start.hooks.site(&PLACE_ORIGINAL, &exe, &facts.place_item, place_item)?;
    start.hooks.site(&TAG_ORIGINAL, &exe, &facts.tag_transform, tag_transform)?;
    crate::extras::laser::install(start.hooks, &exe, facts)?;
    Ok(())
}

pub struct Setup {
    pub layout: &'static HandsLayout,
    units_per_metre: f32,
    /// Synthetic controllers (`probe_guns`).
    synthetic: bool,
    /// The arms follow the guns ([`skeleton`]).
    pub arms: bool,
    /// A free hand's fingers follow tracking.
    pub fingers: bool,
    /// The left arm hidden while its hand holds nothing ([`HIDE_FREE_HAND`]).
    pub hide_free_hand: bool,
}

/// The left arm out of sight while its hand holds nothing (the right holding a one-handed weapon:
/// the knife, a pipe, a grenade); a rifle's front hand and a second gun stay.
pub const HIDE_FREE_HAND: manifest::Opt =
    manifest::opt("hide_free_hand", "Hide free hand", "Hides your left hand when it holds nothing.", manifest::Kind::Switch { default: false })
        .on(manifest::Shown::Skeleton)
        .only_when("aim", "hand_rig");

static SETUP: OnceLock<Setup> = OnceLock::new();

/// The guns' setup, once installed.
pub fn setup() -> Option<&'static Setup> {
    SETUP.get()
}

impl Setup {
    /// World units per metre.
    pub fn units_per_metre(&self) -> f32 {
        self.units_per_metre
    }
}

/// A hand's gun as last placed: its weapon (to know its shots), where it is in the world (origin
/// and axis rows forward, left, up, world units) and when (`monaka_channel::tick`).
#[derive(Clone, Copy)]
pub struct Placed {
    pub weapon: usize,
    pub origin: [f32; 3],
    pub axis: [f32; 9],
    pub at: u64,
}

/// By side (`LEFT_HAND`, `RIGHT_HAND`).
static PLACED: Mutex<[Option<Placed>; 2]> = Mutex::new([None; 2]);

thread_local! {
    /// The placement the next tag transform on this thread is replaced by (set for one placer call).
    static PENDING: Cell<Option<([f32; 3], [f32; 9])>> = const { Cell::new(None) };
    /// The side of the hand tag the next tag transform on this thread places (set for one placer call).
    static TAG_SIDE: Cell<Option<usize>> = const { Cell::new(None) };
    /// The hands of the latest placer call on this thread.
    static HANDS: Cell<usize> = const { Cell::new(0) };
}

static PLACEMENTS: AtomicU64 = AtomicU64::new(0);

static MOVED: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];
static UNTRACKED: AtomicU64 = AtomicU64::new(0);

fn install(layout: &'static HandsLayout, units_per_metre: f32, synthetic: bool, arms: bool, fingers: bool, hide_free_hand: bool) {
    let _ = SETUP.set(Setup { layout, units_per_metre, synthetic, arms, fingers, hide_free_hand });
    if synthetic {
        log!("guns: synthetic controllers (probe_guns)");
    }
}

pub fn summary() -> String {
    format!(
        "guns: {} item placements, moved right {} left {}, {} without a tracked controller; the right hand last held {}; {}",
        PLACEMENTS.load(Relaxed),
        MOVED[RIGHT_HAND].load(Relaxed),
        MOVED[LEFT_HAND].load(Relaxed),
        UNTRACKED.load(Relaxed),
        RIGHT_NAME.locked().1.clone().unwrap_or_else(|| "no weapon".to_owned()),
        crate::extras::laser::summary()
    )
}

/// Each hand's gun as last placed, by side.
pub fn placed() -> [Option<Placed>; 2] {
    *PLACED.locked()
}

/// Where the item of `side` was last placed (world units), while the controllers place them.
pub fn placed_origin(side: usize) -> Option<[f32; 3]> {
    PLACED.locked()[side].map(|p| p.origin)
}

/// The controller of `side`, as its palm in tracking space.
fn palm(side: usize, head: &HeadPose, synthetic: bool) -> Option<HandPose> {
    if synthetic { Some(crate::research::guns::synthetic_palm(side, head)) } else { monaka_body::controllers::palm(side) }
}

/// Where a gun held by the controller at `palm` goes (origin in world units, axis rows forward,
/// left, up), with the headset's tracking space at `space` in the world (canonical).
pub fn gun_in_world(space: &Mat34, palm: &HandPose, units_per_metre: f32) -> Option<([f32; 3], [f32; 9])> {
    eng_idt::view::origin_axis(&eng_idt::camera_convention(units_per_metre), &monaka_body::gun_hold::on_controller(space, palm))
}

/// Where the headset's tracking space sits in the world (canonical) now, for the hands `hands`,
/// with the head pose it was taken from.
pub fn hands_space(setup: &Setup, hands: usize) -> Option<(Mat34, HeadPose)> {
    let head = PAIRS.head().current()?;
    let view = skeleton::hands_view(setup.layout, hands)?;
    Some((eye_site::tracking_space(&view, &head)?, head))
}

/// Where the gun of `side` goes in the tracking space `space` ([`hands_space`]; None: its
/// controller is not tracked).
pub fn gun_at(setup: &Setup, (space, head): &(Mat34, HeadPose), side: usize) -> Option<([f32; 3], [f32; 9])> {
    gun_in_world(space, &palm(side, head, setup.synthetic)?, setup.units_per_metre)
}

/// Where the item of `side` goes in the tracking space `tracked` ([`hands_space`]) for the hands
/// `hands`: a gun on its controller ([`gun_at`]); the right hand's melee weapon, where its resting
/// grip is measured (`HandsLayout::melee_rest`), along the controller's grip ([`melee_at`]); an item
/// carried in both hands from its middle, between both controllers ([`carried_at`]).
pub fn item_at(setup: &Setup, tracked: &(Mat34, HeadPose), hands: usize, side: usize) -> Option<([f32; 3], [f32; 9])> {
    if side == RIGHT_HAND
        && right_carried(setup.layout, hands)
        && let Some(wrists) = skeleton::carried_wrists()
        && let (Some(right), Some(left)) = (gun_at(setup, tracked, RIGHT_HAND), gun_at(setup, tracked, LEFT_HAND))
    {
        return carried_at(&right, &left, &wrists);
    }
    if side == RIGHT_HAND && setup.layout.melee_rest.is_some() && right_kind(setup.layout, hands) == Kind::Melee {
        let (space, head) = tracked;
        let grip = if setup.synthetic { palm(side, head, true) } else { monaka_body::controllers::grip(side) };
        return melee_at(&gun_in_world(space, &grip?, setup.units_per_metre)?);
    }
    gun_at(setup, tracked, side)
}

/// Where a melee weapon goes, held by the controller grip whose frame is `(origin, axis)` (world
/// origin and axis rows forward, left, up, as [`gun_in_world`] places it): the weapon's handle
/// along the controller's handle, as the hand holds both. The game models the knife's blade along
/// its own up axis, so that axis goes on the grip's forward (out of the fist) and the weapon's
/// forward on the grip's down; the hand then holds it at the game's resting grip.
pub fn melee_at((origin, axis): &([f32; 3], [f32; 9])) -> Option<([f32; 3], [f32; 9])> {
    let [f0, f1, f2, l0, l1, l2, u0, u1, u2] = *axis;
    Some((*origin, [-u0, -u1, -u2, l0, l1, l2, f0, f1, f2]))
}

/// A placement (origin, axis rows forward, left, up) as a joint's matrix (rotation columns the axes).
fn as_joint((origin, rows): &([f32; 3], [f32; 9])) -> Mat34 {
    from_rotation(&transpose3(rows), *origin)
}

fn from_joint(m: &Mat34) -> ([f32; 3], [f32; 9]) {
    ([m[3], m[7], m[11]], transpose3(&rotation_part(m)))
}

/// The turn taking the direction `from` onto `to` the shortest way (None: either is too short).
pub fn turn_between(from: Vec3, to: Vec3) -> Option<[f32; 9]> {
    let (a, b) = (unit(from)?, unit(to)?);
    let (axis, cos) = (cross(a, b), dot(a, b).clamp(-1.0, 1.0));
    let sin = length(axis);
    if sin < 1e-6 {
        if cos > 0.0 {
            return Some([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        }
        // Opposite: half a turn about any axis square to them.
        let side = if a[0].abs() < 0.9 { [1.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0] };
        let k = unit(cross(a, side))?;
        return Some(axis_turn(k, -1.0, 0.0));
    }
    Some(axis_turn(scale(axis, 1.0 / sin), cos, sin))
}

/// The turn about the unit axis `k` by the angle of cosine `c` and sine `s` (Rodrigues).
fn axis_turn(k: Vec3, c: f32, s: f32) -> [f32; 9] {
    let t = 1.0 - c;
    let [x, y, z] = k;
    [t * x * x + c, t * x * y - s * z, t * x * z + s * y, t * x * y + s * z, t * y * y + c, t * y * z - s * x, t * x * z - s * y, t * y * z + s * x, t * z * z + c]
}

/// Where an item carried in both hands from its middle goes (its tag: origin and axis rows), with
/// the right and left controllers holding where a gun of each would go (`right`, `left`: placements
/// as [`gun_at`] gives them) and the game's wrists on it (`wrists`: right, then left, each in the
/// tag's frame). Each controller's hand is where its wrist would be on a gun of its own
/// (`skeleton::RIGHT_GRIP`, `skeleton::LEFT_GRIP`); the item turns with the right hand, then so the
/// line between its wrists runs along the line between those hands, and sits with its wrists'
/// middle on theirs.
pub fn carried_at(right: &([f32; 3], [f32; 9]), left: &([f32; 3], [f32; 9]), wrists: &(Mat34, Mat34)) -> Option<([f32; 3], [f32; 9])> {
    let hand_r = compose(&as_joint(right), &skeleton::RIGHT_GRIP);
    let hand_l = compose(&as_joint(left), &skeleton::LEFT_GRIP);
    let (grip_r, grip_l) = wrists;
    let turned = mul3(&rotation_part(&hand_r), &transpose3(&rotation_part(grip_r)));
    let (on_r, on_l) = (position(grip_r), position(grip_l));
    let (at_r, at_l) = (position(&hand_r), position(&hand_l));
    let swing = turn_between(apply3(&turned, sub(on_l, on_r)), sub(at_l, at_r)).unwrap_or([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
    let rotation = mul3(&swing, &turned);
    let origin = sub(midpoint(at_r, at_l), apply3(&rotation, midpoint(on_r, on_l)));
    let placed = from_joint(&from_rotation(&rotation, origin));
    all_finite(&placed.0).then_some(())?;
    all_finite(&placed.1).then_some(placed)
}

/// The axis row (of a target's three) the right hand's melee weapon's blade runs along: its up
/// axis when it is held along the controller's grip ([`melee_at`]), else its forward.
pub fn blade_row(setup: &Setup) -> usize {
    if setup.layout.melee_rest.is_some() { 2 } else { 0 }
}

/// Where the item of `side` goes now, for the hands `hands` (None: its controller is not tracked).
pub fn gun_target(setup: &Setup, hands: usize, side: usize) -> Option<([f32; 3], [f32; 9])> {
    item_at(setup, &hands_space(setup, hands)?, hands, side)
}

/// The frame the left hand last placed an item of its own (MAX: never).
static LEFT_ITEM_FRAME: AtomicU64 = AtomicU64::new(u64::MAX);

/// Whether the left hand holds a gun of its own (dual wielding): it placed an item of its own
/// within the last two view builds.
/// The left hand's item at its last placement (an `idWeapon`, 0: none).
static LEFT_ITEM: AtomicUsize = AtomicUsize::new(0);
/// Placements that never asked for their hand tag (the controller's placement not applied).
static UNPLACED: AtomicU64 = AtomicU64::new(0);

/// What the left hand last held of its own, by name.
pub fn left_item_name() -> Option<String> {
    let weapon = LEFT_ITEM.load(Relaxed);
    (weapon != 0).then(|| mem::read::<usize>(weapon + engine::INVENTORY_ITEM_DECL).filter(|&d| d != 0).and_then(decl_name)).flatten()
}

pub fn dual() -> bool {
    let last_left = LEFT_ITEM_FRAME.load(Relaxed);
    last_left != u64::MAX && eye_site::builds().saturating_sub(last_left) <= 2
}

static ALONG_GUN: AtomicU64 = AtomicU64::new(0);

/// `place` with the player's view axis (its rows after the origin) set to `axis`, then given back
/// unless the placement made it anew meanwhile.
fn along_gun(layout: &HandsLayout, hands: usize, axis: &[f32; 9], place: impl FnOnce()) {
    let rows = mem::read::<usize>(hands + engine::HANDS_PLAYER).filter(|&p| p != 0).map(|p| p + layout.player_view + 12);
    let Some((rows, view)) = rows.and_then(|r| Some((r, mem::read::<[f32; 9]>(r)?))) else {
        place();
        return;
    };
    mem::write(rows, *axis);
    place();
    if mem::read::<[f32; 9]>(rows) == Some(*axis) {
        mem::write(rows, view);
    }
    if ALONG_GUN.fetch_add(1, Relaxed) == 0 {
        log!("guns: a gun in hand is placed with the view along its barrel: its laser goes straight out of it");
    }
}

pub unsafe extern "system" fn place_item(item: usize, hands: usize) {
    let flight = InFlight::enter();
    PLACEMENTS.fetch_add(1, Relaxed);
    let mut moved = None;
    let tag_side = mem::read::<i32>(item + engine::ITEM_TAG).and_then(side_of_tag);
    TAG_SIDE.set(tag_side);
    HANDS.set(hands);
    let frame = eye_site::builds();
    // The left hand holds an item of its own only when it is a weapon: an animation's empty item on
    // the left hand tag (a wall shuffle's) leaves the left hand free on its controller.
    let weapon = mem::read::<usize>(item + engine::ITEM_WEAPON).unwrap_or(0);
    if tag_side == Some(LEFT_HAND) && weapon != 0 {
        LEFT_ITEM_FRAME.store(frame, Relaxed);
        LEFT_ITEM.store(weapon, Relaxed);
    }
    // While the controllers move the hotwiring minigame's hands, an item stays on its hand.
    if crate::on(crate::GUNS_AT)
        && !crate::extras::hotwire::active()
        && let Some(setup) = SETUP.get()
        && let Some(side) = tag_side
    {
        // With the arms posed on the controllers, on the arms model as drawn (where the hands are).
        match gun_target(setup, hands, side).map(|placed| if setup.arms && !crate::research::options().items_on_view { skeleton::onto_arms_as_drawn(setup.layout, hands, placed) } else { placed }) {
            Some(placed) => {
                PENDING.set(Some(placed));
                moved = Some((side, placed));
            }
            None => {
                UNTRACKED.fetch_add(1, Relaxed);
            }
        }
    }
    // SAFETY: forwards the game's own call.
    let place = || unsafe { PLACE_ORIGINAL.get(&flight)(item, hands) };
    // A gun placed on its controller draws its laser (a sight's, the cutting laser) from its muzzle
    // along the player's view: along the gun instead, the view's axis is the gun's for the placement.
    let gun = moved.filter(|_| weapon != 0).zip(SETUP.get());
    match (tag_side, gun) {
        (Some(_), Some(((_, (_, axis)), setup))) => along_gun(setup.layout, hands, &axis, place),
        (Some(_), None) => place(),
        // On no hand's tag (goggles worn): placed on the player's view, which follows the head.
        (None, _) => {
            let _ = crate::providers::follow_head::moved_view(hands, place);
        }
    }
    // A placement that never asked for its hand tag: the item stays where the game's animation put
    // it (counted, the first few named).
    if PENDING.take().is_some() {
        let n = UNPLACED.fetch_add(1, Relaxed) + 1;
        if log::seldom(n, 10) {
            let name = mem::read::<usize>(weapon + engine::INVENTORY_ITEM_DECL).filter(|&d| d != 0).and_then(decl_name);
            log!(
                "guns: {} (item {item:#x}, kind {}, on the {} hand tag) placed without its hand tag {n} times: it stays where the game's animation puts it",
                name.as_deref().unwrap_or("an item"),
                mem::read::<i32>(item).unwrap_or(-1),
                if tag_side == Some(LEFT_HAND) { "left" } else { "right" }
            );
        }
    }
    TAG_SIDE.set(None);
    if tag_side == Some(RIGHT_HAND)
        && let Some(setup) = SETUP.get()
    {
        let _ = right_carried(setup.layout, hands);
    }
    if let Some((side, (origin, axis))) = moved {
        let count = MOVED[side].fetch_add(1, Relaxed);
        if count < 2 {
            log!("guns: {} gun (weapon {weapon:#x}) placed at {origin:?}, forward {:?}", if side == RIGHT_HAND { "right" } else { "left" }, &axis[..3]);
        }
        PLACED.locked()[side] = Some(Placed { weapon, origin, axis, at: monaka_channel::tick() });
        if side == RIGHT_HAND && right_throws_knives() {
            crate::cooking::knife_throw::note_knife(origin);
        }
        if side == RIGHT_HAND
            && let Some(setup) = SETUP.get()
        {
            watch(setup, hands, weapon, origin);
        }
    }
}

// ---- The right gun against what the eyes see ----

/// The first-person view counts as off the render view past this far (units) or this angle
/// (degrees) between their forward axes.
const SPLIT_UNITS: f32 = 16.0;
const SPLIT_DEGREES: f32 = 15.0;
/// The gun counts as out of sight past this far from the render view's origin (metres) or this
/// angle off its forward axis (degrees).
const SIGHT_METRES: f32 = 1.5;
const SIGHT_DEGREES: f32 = 70.0;
/// The game counts as no longer placing the gun after this many view builds without a placement.
const UNPLACED_BUILDS: u64 = 30;
/// Each kind of line is logged at most this often.
const WATCH_LINES: u64 = 20;

/// The watch's state: the views apart, the gun out of sight, the last weapon, the build of the
/// last right placement and whether placements stopped; the lines logged.
struct Watch {
    split: bool,
    out_of_sight: bool,
    weapon: usize,
    placed_at: u64,
    unplaced: bool,
    lines: u64,
}

static WATCH: Mutex<Watch> = Mutex::new(Watch { split: false, out_of_sight: false, weapon: 0, placed_at: 0, unplaced: false, lines: 0 });

fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn degrees_between(a: [f32; 3], b: [f32; 3]) -> f32 {
    let lengths = dot3(a, a).sqrt() * dot3(b, b).sqrt();
    if lengths <= f32::EPSILON { 0.0 } else { (dot3(a, b) / lengths).clamp(-1.0, 1.0).acos().to_degrees() }
}

/// The views' origin and forward axis (origin, then axis rows forward, left, up).
fn origin_forward(view: &[f32; 12]) -> ([f32; 3], [f32; 3]) {
    ([view[0], view[1], view[2]], [view[3], view[4], view[5]])
}

/// At each right gun placement (weapon `weapon`, placed at `origin`): logs when the first-person
/// view leaves the render view or comes back, when the gun leaves the render view's sight or comes
/// back, and a change of weapon.
fn watch(setup: &Setup, hands: usize, weapon: usize, origin: [f32; 3]) {
    let mut w = WATCH.locked();
    w.placed_at = eye_site::builds();
    if w.unplaced {
        w.unplaced = false;
        if w.lines < WATCH_LINES {
            w.lines += 1;
            log!("guns: the game places the right gun again (view build {})", w.placed_at);
        }
    }
    if weapon != w.weapon {
        if w.weapon != 0 && w.lines < WATCH_LINES {
            w.lines += 1;
            log!("guns: the right hand holds another weapon: {:#x} -> {weapon:#x} (decl {:#x})", w.weapon, right_decl(setup.layout, hands).unwrap_or(0));
        }
        w.weapon = weapon;
    }
    let Some(render) = eye_site::render_view() else { return };
    let (render_origin, render_forward) = origin_forward(&render);
    if let Some(player) = skeleton::player_view(setup.layout, hands) {
        let (player_origin, player_forward) = origin_forward(&player);
        let apart = dot3(sub3(player_origin, render_origin), sub3(player_origin, render_origin)).sqrt();
        let turned = degrees_between(player_forward, render_forward);
        let split = apart > SPLIT_UNITS || turned > SPLIT_DEGREES;
        if split != w.split {
            w.split = split;
            if w.lines < WATCH_LINES {
                w.lines += 1;
                match split {
                    true => log!(
                        "guns: the first-person view (where the gun is placed from) left the render view (what the eyes see): {apart:.1} units apart, {turned:.0} degrees; first-person {player_origin:?} forward {player_forward:?}, render {render_origin:?} forward {render_forward:?}"
                    ),
                    false => log!("guns: the first-person view is back on the render view"),
                }
            }
        }
    }
    let to_gun = sub3(origin, render_origin);
    let metres = dot3(to_gun, to_gun).sqrt() / setup.units_per_metre;
    let off = degrees_between(to_gun, render_forward);
    let out = metres > SIGHT_METRES || off > SIGHT_DEGREES;
    if out != w.out_of_sight {
        w.out_of_sight = out;
        if w.lines < WATCH_LINES {
            w.lines += 1;
            match out {
                true => log!("guns: the right gun is out of sight: {metres:.2} m from the render view, {off:.0} degrees off its forward axis (gun at {origin:?})"),
                false => log!("guns: the right gun is in sight again ({metres:.2} m, {off:.0} degrees)"),
            }
        }
    }
}

/// At each view build (`build`): logs when the game stops placing the right gun.
pub fn watch_build(build: u64) {
    let mut w = WATCH.locked();
    if w.placed_at == 0 || w.unplaced || build.saturating_sub(w.placed_at) < UNPLACED_BUILDS {
        return;
    }
    w.unplaced = true;
    if w.lines < WATCH_LINES {
        w.lines += 1;
        log!("guns: the game stopped placing the right gun (last at view build {}, now {build})", w.placed_at);
    }
}

pub unsafe extern "system" fn tag_transform(entity: usize, tag: usize, origin: *mut f32, axis: *mut f32) {
    let flight = InFlight::enter();
    // SAFETY: forwards the game's own call.
    unsafe { TAG_ORIGINAL.get(&flight)(entity, tag, origin, axis) };
    if let Some(side) = TAG_SIDE.take()
        && !origin.is_null()
        && !axis.is_null()
    {
        // SAFETY: the game's own outputs, three floats and nine.
        let (o, a) = unsafe { (*(origin as *const [f32; 3]), *(axis as *const [f32; 9])) };
        if crate::on(crate::GUNS_AT)
            && let Some(setup) = SETUP.get().filter(|s| s.arms)
        {
            skeleton::observe_tag(setup.layout, HANDS.get(), side, o, a);
        }
    }
    if let Some((o, a)) = PENDING.take()
        && !origin.is_null()
        && !axis.is_null()
    {
        // SAFETY: the game's own outputs: three floats and nine (the placer's locals).
        unsafe {
            std::ptr::copy_nonoverlapping(o.as_ptr(), origin, 3);
            std::ptr::copy_nonoverlapping(a.as_ptr(), axis, 9);
        }
    }
}

#[cfg(test)]
mod tests;
