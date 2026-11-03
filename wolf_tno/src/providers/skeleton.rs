//! Provider: the skeleton (`monaka_body::Skeleton` over the blend job's joint array: bones, the
//! first pose seen as the rest, reading and writing the pose) and its posing moment (captured at
//! the hands' blend kick, [`arms_kick`], posed in the job at its bounds, [`pose_bounds`]), with the
//! rig's facts ([`SHOULDERS`], [`POLES`], [`LEFT_GRIP`]) for Monaka's hand-rooted solver. The
//! solving is Monaka's (`monaka_body::hand_rig::Method::HandRooted`); the targets are where the
//! guns go ([`held_item::gun_target`]).
//!
//! The first-person arms on the controllers (with the guns, `aim=hand_rig`), on Monaka's hand rig
//! ([`crate::Wolf`]'s `HandRig`: a skeleton hung from the hands, `monaka_body::hand_rig::Method::HandRooted`).
//! The hands model's skeleton hangs from the hands: each hand's attach joint (where its gun's tag
//! sits) carries the wrist, the fingers and a chain back up the arm (forearm twist joints, the
//! elbow, the shoulder as the last joint). Its joints are model-space matrices the game has
//! finished animating when the hands place their items ([`held_item`]); changed then, the guns'
//! tags and the drawn arms follow.
//!
//! Each frame, at the hands' blend kick on the game thread ([`kick`]), each gun's target (the
//! controller's palm in the world) is taken into the model's space and the moment captured; the
//! blend job (a worker thread) poses the arms in its joint array once its joints are final
//! ([`job_pose`], from [`pose_bounds`]): each attach joint and everything under it onto its gun's
//! target, each arm bent at the elbow so its shoulder stays where the game had it. The left hand
//! follows the left controller while it holds its own gun (dual wielding) or the right hand holds
//! a one-handed weapon (the knife); otherwise it rides the right hand's gun as the game holds it (a
//! rifle's front hand). The joints are drawn with the next frame.
//!
//! The joints' matrices are 3x4 (rotation and translation, column vectors), in the model's space
//! and the game's units: the rig works there. Which way their rotation maps to the game's axis
//! rows is the engine's (`engine::HandsLayout::joint_basis`); once a run the log checks it
//! against the game's own tag placement, from the joints the tag was placed from.

use crate::providers::engine::{self, ArmsKickFn, Basis, HandsLayout, PoseBoundsFn};
use crate::providers::held_item::{self, HELD};
use monaka_body::hand_rig::{self, Hands, Inputs, Moment};
use monaka_body::held::Kind;
use monaka_body::skeleton::{attach_joint, find_arm};
use monaka_body::{Skeleton, Snapshot};
use monaka_core::camera::{Mat34, apply3, compose, from_rotation, mul3, rigid_inverse, rotation_part, transpose3};
use monaka_core::math::{Vec3, add, length, sub};
use monaka_contract::protocol::{LEFT_HAND, RIGHT_HAND};
use monaka_core::sync::Locked;
use monaka_hook::{InFlight, Original, mem};
use monaka_producer::log;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering::*};
use std::sync::{Arc, Mutex};

pub static BOUNDS_ORIGINAL: Original<PoseBoundsFn> = Original::new();
/// Whether the left hand held an item of its own at the last kick, and the changes logged.
static DUAL_LOGGED: AtomicBool = AtomicBool::new(false);
static DUAL_LINES: AtomicU64 = AtomicU64::new(0);
pub static KICK_ORIGINAL: Original<ArmsKickFn> = Original::new();

/// The joint's axes as rows (forward, left, up), in model space.
fn basis_rows(j: &Mat34, basis: Basis) -> [f32; 9] {
    match basis {
        Basis::Columns => transpose3(&rotation_part(j)),
        Basis::Rows => rotation_part(j),
    }
}

/// The rotation of the joint whose axes are `rows` (forward, left, up) in model space.
fn rotation_from_rows(rows: &[f32; 9], basis: Basis) -> [f32; 9] {
    match basis {
        Basis::Columns => transpose3(rows),
        Basis::Rows => *rows,
    }
}

/// The model's place in the world: origin, axis rows (forward, left, up), scale.
#[derive(Clone, Copy, Debug)]
pub struct Model {
    origin: Vec3,
    axis: [f32; 9],
    scale: Vec3,
}

impl Model {
    /// A model-space point in the world: `origin + A^T (scale * p)`.
    fn to_world(self, p: Vec3) -> Vec3 {
        let s = [p[0] * self.scale[0], p[1] * self.scale[1], p[2] * self.scale[2]];
        add(self.origin, apply3(&transpose3(&self.axis), s))
    }

    fn to_model(self, w: Vec3) -> Vec3 {
        let p = apply3(&self.axis, sub(w, self.origin));
        [p[0] / self.scale[0], p[1] / self.scale[1], p[2] / self.scale[2]]
    }

    /// World axis rows as model axis rows.
    fn rows_to_model(&self, rows: &[f32; 9]) -> [f32; 9] {
        mul3(rows, &transpose3(&self.axis))
    }

    fn rows_to_world(&self, rows: &[f32; 9]) -> [f32; 9] {
        mul3(rows, &self.axis)
    }

    /// Model space to the canonical world (OpenXR axes, metres) in an engine of
    /// `units_per_metre`: each model axis is its axis row times its scale, in game units.
    fn to_canonical(self, units_per_metre: f32) -> Option<Mat34> {
        let convention = eng_idt::camera_convention(units_per_metre);
        let (a, s) = (self.axis, self.scale);
        let scaled = [a[0] * s[0], a[1] * s[0], a[2] * s[0], a[3] * s[1], a[4] * s[1], a[5] * s[1], a[6] * s[2], a[7] * s[2], a[8] * s[2]];
        let m = convention.world.from_origin_axis(self.origin, scaled)?;
        Some(m.map(|v| v / convention.units_per_metre))
    }
}

// ---- The rig's facts.

/// Where each shoulder stays, in model space (x forward, y left, z up, from the eyes; left first):
/// the right one where the game's rifle hold has it, the left one mirrored.
pub const SHOULDERS: [Vec3; 2] = [[-10.9, 10.0, -10.8], [-10.9, -10.0, -10.8]];
/// Each elbow bends down and out (left first).
pub const POLES: [Vec3; 2] = [[0.0, 0.6, -1.0], [0.0, -0.6, -1.0]];
/// The left wrist in the left attach joint's frame as the game holds a gun in the left hand (two
/// guns): a free left hand takes this grip on its controller.
pub const LEFT_GRIP: Mat34 = [0.101, 0.085, 0.991, -3.687, -0.344, -0.932, 0.115, 1.577, 0.934, -0.353, -0.064, 1.149];
/// The right hand's gun grip, the left one's mirror image (the arms mirror left to right: y): the
/// grip a right hand keeps on a melee weapon placed where a gun would be, in a build whose resting
/// grip is not measured (`HandsLayout::melee_rest`).
pub const RIGHT_GRIP: Mat34 = mirrored(LEFT_GRIP);

/// The running build's grip of the right hand on its melee weapon (the hand rig's
/// `right_melee_grip`): its measured resting grip, or [`RIGHT_GRIP`].
pub fn melee_rest() -> Option<Mat34> {
    held_item::setup().map(|setup| setup.layout.melee_rest.unwrap_or(RIGHT_GRIP))
}

/// A joint-to-joint transform between left-right mirrored frames, mirrored across y.
const fn mirrored(m: Mat34) -> Mat34 {
    [m[0], -m[1], m[2], m[3], -m[4], m[5], -m[6], -m[7], m[8], -m[9], m[10], m[11]]
}

// ---- The live skeleton.

/// The skeleton's parents, kept while its header stays (the joint matrices move: the animator
/// keeps more than one set).
struct Found {
    header: usize,
    count: usize,
    parents: Arc<Vec<i32>>,
}

static FOUND: Mutex<Option<Found>> = Mutex::new(None);
/// Whether the log's check of the joint basis was made this run.
static BASIS_CHECKED: AtomicBool = AtomicBool::new(false);
static REFUSED: AtomicU64 = AtomicU64::new(0);

pub fn summary() -> String {
    format!(
        "arms: {} blend kicks, {} frames prepared ({} never reached their job, {} reached it after the next kick), {} poses edited, {} refused\n{}",
        KICKS.load(Relaxed),
        PREPARED.load(Relaxed),
        UNUSED.load(Relaxed),
        LATE.load(Relaxed),
        EDITED.load(Relaxed),
        REFUSED.load(Relaxed),
        crate::providers::follow_head::summary(),
    )
}

/// The player's first-person view (origin, then axis rows) for the hands `hands`.
pub fn player_view(layout: &HandsLayout, hands: usize) -> Option<[f32; 12]> {
    mem::read::<usize>(hands + engine::HANDS_PLAYER).filter(|&p| p != 0).and_then(|p| mem::read::<[f32; 12]>(p + layout.player_view))
}

/// Where the hands and their guns are placed from (origin, then axis rows) for the hands `hands`:
/// the player's first-person view, read fresh here, where the game draws the arms model and the
/// camera sits; while a grab has the camera off that view (`camera_taken::camera_away`), the
/// camera's render view (the last view build's, as the game made it).
pub fn hands_view(layout: &HandsLayout, hands: usize) -> Option<[f32; 12]> {
    let view = player_view(layout, hands)?;
    let render = crate::providers::camera_taken::camera_away().then(crate::providers::eye_site::render_view).flatten();
    Some(render.unwrap_or(view))
}

/// Items placed off the arms model as drawn, and the most one moved (units).
static ONTO_ARMS: AtomicU64 = AtomicU64::new(0);
static ONTO_ARMS_MOST: AtomicU32 = AtomicU32::new(0);

/// Where an item placed at `placed` (world origin and axis rows, from the player's view as the rig
/// takes it) goes with the arms model where the game actually drew it this frame: the hands are
/// posed in that model, so an item held stays in its hand when the model is drawn off the view (a
/// ledge shuffle moves the arms, and the camera riding them, with its animation). Unchanged while
/// the model is where the view says.
pub fn onto_arms_as_drawn(layout: &HandsLayout, hands: usize, placed: (Vec3, [f32; 9])) -> (Vec3, [f32; 9]) {
    let (Some(live), Some(view)) = (live(layout, hands), hands_view(layout, hands)) else { return placed };
    let Some(actual) = model(live.entity) else { return placed };
    let view = crate::providers::follow_head::as_placed(view);
    let assumed = Model { origin: [view[0], view[1], view[2]], axis: std::array::from_fn(|i| view[3 + i]), scale: actual.scale };
    let origin = actual.to_world(assumed.to_model(placed.0));
    let axis = actual.rows_to_world(&assumed.rows_to_model(&placed.1));
    if !origin.iter().chain(&axis).all(|v| v.is_finite()) {
        return placed;
    }
    let moved = length(sub(origin, placed.0));
    if moved > 0.5 {
        let n = ONTO_ARMS.fetch_add(1, Relaxed) + 1;
        let most = f32::from_bits(ONTO_ARMS_MOST.load(Relaxed));
        if moved > most {
            ONTO_ARMS_MOST.store(moved.to_bits(), Relaxed);
        }
        if log::seldom(n, 5) {
            log!("arms: an item placed {moved:.1} units off the view's place, where the arms model is drawn ({n} so far)");
        }
    }
    (origin, axis)
}

/// The arms model's animated entity, its joint animator, its joint matrices, its skeleton header
/// and joint count.
struct Live {
    entity: usize,
    animator: usize,
    joints: usize,
    header: usize,
    count: usize,
}

fn live(layout: &HandsLayout, hands: usize) -> Option<Live> {
    let presentable = mem::read::<usize>(hands + engine::HANDS_ARMS_MODEL).filter(|&p| p != 0)?;
    let entity = mem::read::<usize>(presentable + layout.model_animated).filter(|&p| p != 0)?;
    let animator = mem::read::<usize>(entity + layout.joint_animator).filter(|&p| p != 0)?;
    let joints = mem::read::<usize>(animator + layout.joints).filter(|&p| p != 0)?;
    let decl = mem::read::<usize>(animator).filter(|&p| p != 0)?;
    let model = mem::read::<usize>(decl + 0x60).filter(|&p| p != 0)?;
    let skel = mem::read::<usize>(model + 0x128).filter(|&p| p != 0)?;
    let header = mem::read::<usize>(skel + layout.skeleton_header).filter(|&p| p != 0)?;
    let count = mem::read::<u16>(header + 2)? as usize;
    Some(Live { entity, animator, joints, header, count })
}

fn model(entity: usize) -> Option<Model> {
    let origin = mem::read::<Vec3>(entity + engine::ANIMATED_ORIGIN)?;
    let axis = mem::read::<[f32; 9]>(entity + engine::ANIMATED_AXIS)?;
    let scale = mem::read::<Vec3>(entity + engine::ANIMATED_SCALE).filter(|s| s.iter().all(|v| v.is_finite() && v.abs() > 1e-3)).unwrap_or([1.0; 3]);
    Some(Model { origin, axis, scale })
}

fn read_joints(at: usize, count: usize) -> Option<Vec<Mat34>> {
    (0..count).map(|j| mem::read::<Mat34>(at + j * engine::JOINT_SIZE)).collect()
}

/// The skeleton's parents, found again when its joints move.
fn found(header: usize, count: usize) -> Option<Arc<Vec<i32>>> {
    let mut slot = FOUND.locked();
    if let Some(f) = slot.as_ref().filter(|f| f.header == header && f.count == count) {
        return Some(f.parents.clone());
    }
    let offset = mem::read::<u16>(header + 0xe)? as usize;
    let parents: Vec<i32> = (0..count).map(|j| mem::read::<i16>(header + offset + 2 * j).map(i32::from)).collect::<Option<_>>()?;
    if !parents.iter().enumerate().all(|(j, &p)| p >= -1 && p < j as i32) {
        log!("arms: the parent table does not read as parents first; the arms stay as the game animates them");
        return None;
    }
    let parents = Arc::new(parents);
    *slot = Some(Found { header, count, parents: parents.clone() });
    Some(parents)
}

/// The right attach joint, found from the game's right hand tag ([`attach_joint`]), once and
/// before anything is moved. The left one is its sibling under the hands root (found by the
/// skeleton, not by its tag: in a moved pose, or the game's own two-handed holds, both attach
/// joints can sit on the same spot).
static RIGHT_ROOT: Mutex<Option<usize>> = Mutex::new(None);

/// From the tag transform, the game's own placement of a hand tag (before any replacement): finds
/// the right hand's attach joint before anything is moved (nothing is until it is found), and
/// once a run checks the joint basis for the log: the tag's axes against the attach joint's, read
/// here from the joints the tag was just placed from.
pub fn observe_tag(layout: &HandsLayout, hands: usize, side: usize, origin: Vec3, axis: [f32; 9]) {
    if side != RIGHT_HAND {
        return;
    }
    let Some(live) = live(layout, hands) else { return };
    let (Some(m), Some(joints)) = (model(live.entity), read_joints(live.joints, live.count)) else { return };
    let mut root = RIGHT_ROOT.locked();
    if root.is_none()
        && let Some(parents) = found(live.header, live.count)
    {
        let distances: Vec<f32> = joints.iter().map(|j| length(sub(m.to_world([j[3], j[7], j[11]]), origin))).collect();
        if let Some(attach) = attach_joint(&parents, &distances) {
            log!("arms: the right hand's tag sits on joint {attach} ({:.2} units)", distances[attach]);
            *root = Some(attach);
        }
    }
    let attach = *root;
    drop(root);
    if let Some(attach) = attach
        && let Some(parents) = found(live.header, live.count)
    {
        note_tag_off_attach(layout, hands, &m, &joints, &parents, attach, origin, axis);
    }
    if let Some(attach) = attach
        && let Some(joint) = joints.get(attach)
        && !BASIS_CHECKED.swap(true, AcqRel)
    {
        let error = |b: Basis| {
            let world = m.rows_to_world(&basis_rows(joint, b));
            world.iter().zip(&axis).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max)
        };
        let (columns, rows) = (error(Basis::Columns), error(Basis::Rows));
        let expected = layout.joint_basis;
        let off = if expected == Basis::Columns { columns } else { rows };
        match off < 0.02 {
            true => log!("arms: joint basis check against the game's tag: columns off by {columns:.4}, rows by {rows:.4}: {expected:?}, as the engine's"),
            false => log!("arms: joint basis check against the game's tag: columns off by {columns:.4}, rows by {rows:.4}: NOT the engine's {expected:?}; the arms may be turned wrong"),
        }
    }
}

/// Both wrists in the right attach joint's frame (right, then left), as the game posed the arms in
/// the latest blend job before the rig moved them ([`note_game_wrists`]): on an item carried in both
/// hands from its middle, where the game holds its edges. Read from the game's own pose, never the
/// drawn one: the rig bends the drawn arms, and a placement made from them would feed on itself.
pub fn carried_wrists() -> Option<(Mat34, Mat34)> {
    *GAME_WRISTS.locked()
}

/// The game's own wrists in the right attach joint's frame, from the latest blend job.
static GAME_WRISTS: Mutex<Option<(Mat34, Mat34)>> = Mutex::new(None);
/// The right attach joint and both wrists (right, then left), by skeleton header.
static WRIST_JOINTS: Mutex<Option<(usize, [usize; 3])>> = Mutex::new(None);

/// The right attach joint and both wrists of the skeleton `header` with `parents`.
fn wrist_joints(header: usize, parents: &[i32]) -> Option<[usize; 3]> {
    let mut kept = WRIST_JOINTS.locked();
    if let Some((h, joints)) = *kept
        && h == header
    {
        return Some(joints);
    }
    let right = (*RIGHT_ROOT.locked())?;
    let hands_root = *parents.get(right)?;
    let left = (0..parents.len()).find(|&j| j != right && parents[j] == hands_root)?;
    let joints = [right, find_arm(parents, right)?.wrist, find_arm(parents, left)?.wrist];
    *kept = Some((header, joints));
    Some(joints)
}

/// The left arm's joints, the shoulder first (everything under the left attach joint: the hand,
/// the fingers, the forearm, the elbow), by skeleton header.
static LEFT_ARM: Mutex<Option<(usize, Arc<Vec<usize>>)>> = Mutex::new(None);
/// The scale a hidden arm's joints get: every vertex they carry drawn at the shoulder.
const HIDDEN_SCALE: f32 = 1e-3;

/// The left arm's joints of the skeleton `header` with `parents`, the shoulder first.
fn left_arm(header: usize, parents: &[i32]) -> Option<Arc<Vec<usize>>> {
    let mut kept = LEFT_ARM.locked();
    if let Some((h, joints)) = kept.as_ref()
        && *h == header
    {
        return Some(joints.clone());
    }
    let right = (*RIGHT_ROOT.locked())?;
    let hands_root = *parents.get(right)?;
    let left = (0..parents.len()).find(|&j| j != right && parents[j] == hands_root)?;
    let arm = find_arm(parents, left)?;
    let joints: Arc<Vec<usize>> = Arc::new(std::iter::once(arm.shoulder).chain(arm.subtree.into_iter().filter(|&j| j != arm.shoulder)).collect());
    *kept = Some((header, joints.clone()));
    Some(joints)
}

/// The arm `joints` (the shoulder first) shrunk to a point at the shoulder: out of sight, its
/// bounds no wider.
fn hide(skeleton: &JobPose<'_>, joints: &[usize]) {
    let Some(shoulder) = joints.first().and_then(|&s| skeleton.world(s as i32)) else { return };
    let s = HIDDEN_SCALE;
    let point: Mat34 = [s, 0.0, 0.0, shoulder[3], 0.0, s, 0.0, shoulder[7], 0.0, 0.0, s, shoulder[11]];
    for &joint in joints {
        skeleton.set_world(joint as i32, &point);
    }
}

/// In the blend job, before the rig moves anything: the game's own wrists in `pose` (the hands'
/// joint array of the skeleton `header` with `parents`) kept for [`carried_wrists`].
fn note_game_wrists(header: usize, pose: usize, parents: &[i32]) {
    let Some([right, wrist_r, wrist_l]) = wrist_joints(header, parents) else { return };
    let read = |j: usize| mem::read::<Mat34>(pose + j * engine::JOINT_SIZE);
    let (Some(tag), Some(r), Some(l)) = (read(right), read(wrist_r), read(wrist_l)) else { return };
    let tag = rigid_inverse(&tag);
    *GAME_WRISTS.locked() = Some((compose(&tag, &r), compose(&tag, &l)));
}

/// The right hand's weapons (decls) whose tag was found off the attach joint, each logged once.
static TAG_OFF_LOGGED: Mutex<Vec<usize>> = Mutex::new(Vec::new());
/// The tag counts as off the attach joint past this far (units) or this turn (degrees).
const TAG_OFF: (f32, f32) = (0.5, 5.0);

/// Logs, once per weapon, when the game's right hand tag (`origin`, `axis`: world) is not on the
/// right attach joint in `joints` (the model at `m`): the item then hangs from another joint than
/// the one the rig puts on the controller. Names the nearest joint and its parent.
#[allow(clippy::too_many_arguments)]
fn note_tag_off_attach(layout: &HandsLayout, hands: usize, m: &Model, joints: &[Mat34], parents: &[i32], attach: usize, origin: Vec3, axis: [f32; 9]) {
    let Some(decl) = held_item::right_decl(layout, hands) else { return };
    let joint = &joints[attach];
    let units = length(sub(m.to_world([joint[3], joint[7], joint[11]]), origin));
    let rows = m.rows_to_world(&basis_rows(joint, layout.joint_basis));
    let cos = ((0..3).map(|i| rows[i] * axis[i]).sum::<f32>()).clamp(-1.0, 1.0);
    let degrees = cos.acos().to_degrees();
    if units <= TAG_OFF.0 && degrees <= TAG_OFF.1 {
        return;
    }
    let mut logged = TAG_OFF_LOGGED.locked();
    if logged.contains(&decl) || logged.len() >= 32 {
        return;
    }
    logged.push(decl);
    drop(logged);
    let distances: Vec<f32> = joints.iter().map(|j| length(sub(m.to_world([j[3], j[7], j[11]]), origin))).collect();
    let nearest = (0..distances.len()).min_by(|&a, &b| distances[a].total_cmp(&distances[b]));
    let name = held_item::decl_name(decl).unwrap_or_else(|| "?".to_owned());
    match nearest {
        Some(n) => {
            let parent = parents.get(n).copied().unwrap_or(-1);
            log!("arms: {name}'s tag is {units:.1} units and {degrees:.0} degrees off the right attach joint {attach}; nearest joint {n} at {:.2} units (parent {parent})", distances[n]);
        }
        None => log!("arms: {name}'s tag is {units:.1} units and {degrees:.0} degrees off the right attach joint {attach}"),
    }
}

// ---- From the kick to the job.

/// What the game thread hands the blend job for one frame: the joint array the job builds (the
/// animator's back pose), the skeleton, and the moment captured for it.
struct Frame {
    pose: usize,
    header: usize,
    parents: Arc<Vec<i32>>,
    inputs: Inputs,
    /// The left arm hidden: its hand holds nothing and the player hides it.
    hide_left: bool,
}

/// The frames prepared for blend jobs that have not reached their bounds yet, oldest first. A job
/// can still be on its way when the next frame's kick comes (the game waits for it only inside the
/// kick), so a frame stays until its own job takes it; the two back poses alternate, so two are
/// enough.
static FRAMES: Mutex<Vec<Frame>> = Mutex::new(Vec::new());
/// The poses of [`FRAMES`], for every blend job to check without the lock (0: none).
static PENDING: [AtomicUsize; 2] = [AtomicUsize::new(0), AtomicUsize::new(0)];
static KICKS: AtomicU64 = AtomicU64::new(0);
static PREPARED: AtomicU64 = AtomicU64::new(0);
/// Frames dropped before their job took them.
static UNUSED: AtomicU64 = AtomicU64::new(0);
/// Frames taken after the next frame's kick.
static LATE: AtomicU64 = AtomicU64::new(0);
static STALE_LOGS: AtomicU64 = AtomicU64::new(0);

/// The frame for `pose`, replacing any earlier one for it (None: the job leaves it as it is).
fn set_frame(pose: usize, frame: Option<Frame>) {
    let mut frames = FRAMES.locked();
    let before = frames.len();
    frames.retain(|f| f.pose != pose);
    let mut unused = before - frames.len();
    if let Some(frame) = frame {
        frames.push(frame);
        PREPARED.fetch_add(1, Relaxed);
    }
    while frames.len() > PENDING.len() {
        frames.remove(0);
        unused += 1;
    }
    UNUSED.fetch_add(unused as u64, Relaxed);
    publish(&frames);
}

fn publish(frames: &[Frame]) {
    for (i, slot) in PENDING.iter().enumerate() {
        slot.store(frames.get(i).map_or(0, |f| f.pose), Release);
    }
}

/// Before the hands' blend kick, on the game thread: where the arms go in the blend being started
/// (`targets` by side: the world origin and axis rows each hand's gun gets; the left one only while
/// it follows its controller; `tracking`: the headset's tracking space in the world (canonical) the
/// targets were placed from, in an engine of `units_per_metre`; `dual` with a gun in each hand;
/// `fingers`: a free hand's fingers follow tracking; `hide_free`: a free left hand is hidden). The
/// job puts them there ([`job_pose`]); they are drawn with the next frame.
#[allow(clippy::too_many_arguments)]
pub fn kick(layout: &HandsLayout, hands: usize, targets: [Option<(Vec3, [f32; 9])>; 2], tracking: Option<Mat34>, units_per_metre: f32, dual: bool, fingers: bool, hide_free: bool) {
    KICKS.fetch_add(1, Relaxed);
    // The left hand beside a one-handed weapon holds nothing; otherwise it holds a gun, its own
    // or the right one's. The right hand holds its gun or melee weapon.
    let free = targets[LEFT_HAND].is_some() && !dual;
    *HELD.locked() = [if free { Kind::Nothing } else { Kind::Gun }, held_item::right_kind(layout, hands)];
    let Some(live) = live(layout, hands) else { return };
    let Some(pose) = mem::read::<usize>(live.animator + layout.back_joints).filter(|&p| p != 0) else { return };
    let frame = prepare(layout, hands, &live, pose, targets, tracking, units_per_metre, fingers).map(|frame| Frame { hide_left: free && hide_free, ..frame });
    set_frame(pose, frame);
}

// The kick's facts passed on as they are.
#[allow(clippy::too_many_arguments)]
fn prepare(layout: &HandsLayout, hands: usize, live: &Live, pose: usize, targets: [Option<(Vec3, [f32; 9])>; 2], tracking: Option<Mat34>, units_per_metre: f32, fingers: bool) -> Option<Frame> {
    let m = model(live.entity)?;
    let parents = found(live.header, live.count)?;
    let right_root = (*RIGHT_ROOT.locked())?;
    let basis = layout.joint_basis;
    // No right controller: nothing to move, but a debug view still shows the arms as animated.
    if targets[RIGHT_HAND].is_none() && !hand_rig::views_on() {
        return None;
    }
    // The targets in the model's space taken on the player's view now, where the arms model is
    // drawn (against the model's placement as read here, the arms trailed the guns while walking).
    let view = crate::providers::follow_head::as_placed(hands_view(layout, hands)?);
    let drawn = Model { origin: [view[0], view[1], view[2]], axis: [view[3], view[4], view[5], view[6], view[7], view[8], view[9], view[10], view[11]], scale: m.scale };
    let behind = length(sub(drawn.origin, m.origin));
    if behind > 0.5 && STALE_LOGS.fetch_add(1, Relaxed) < 5 {
        log!("arms: the arms model's placement is {behind:.2} units from the player's view at the kick");
    }
    let target = |(origin, axis): (Vec3, [f32; 9])| from_rotation(&rotation_from_rows(&drawn.rows_to_model(&axis), basis), drawn.to_model(origin));
    // The bones' space in tracking space, as the targets were placed: model to world, then world
    // to tracking.
    let space = compose(&rigid_inverse(&tracking?), &drawn.to_canonical(units_per_metre)?);
    let inputs = hand_rig::capture::<crate::Wolf>(Moment { hands: Hands::Attached { right: right_root, targets: targets.map(|t| t.map(target)), space }, fingers });
    Some(Frame { pose, header: live.header, parents, inputs, hide_left: false })
}

static EDITED: AtomicU64 = AtomicU64::new(0);

/// The frame prepared for `pose`, taken out (every blend job asks; only the hands' pose matches).
fn take_frame(pose: usize) -> Option<Frame> {
    if !PENDING.iter().any(|p| p.load(Acquire) == pose) {
        return None;
    }
    // The game thread holds the lock only to swap a frame in, never while waiting on a job.
    let mut frames = FRAMES.locked();
    let at = frames.iter().position(|f| f.pose == pose)?;
    if at + 1 < frames.len() {
        LATE.fetch_add(1, Relaxed);
    }
    let frame = frames.remove(at);
    publish(&frames);
    Some(frame)
}

/// A blend job's joint array as the hand rig's skeleton (model space, the game's units): written
/// in place, before the bounds and the skinning read it.
struct JobPose<'a> {
    header: usize,
    pose: usize,
    parents: &'a [i32],
}

impl Skeleton for JobPose<'_> {
    fn id(&self) -> usize {
        self.header
    }

    fn count(&self) -> i32 {
        self.parents.len() as i32
    }

    fn name(&self, _: i32) -> String {
        String::new()
    }

    fn parent(&self, element: i32) -> i32 {
        usize::try_from(element).ok().and_then(|e| self.parents.get(e)).copied().unwrap_or(-1)
    }

    fn world(&self, element: i32) -> Option<Mat34> {
        let e = usize::try_from(element).ok().filter(|&e| e < self.parents.len())?;
        mem::read::<Mat34>(self.pose + e * engine::JOINT_SIZE)
    }

    fn set_world(&self, element: i32, matrix: &Mat34) {
        if let Some(e) = usize::try_from(element).ok().filter(|&e| e < self.parents.len()) {
            mem::write(self.pose + e * engine::JOINT_SIZE, *matrix);
        }
    }

    fn is_bone(&self, _: i32) -> bool {
        true
    }
}

/// In the blend job, once its joints are final (model space, overrides applied) and before the
/// bounds and the skinning: the arms onto their targets, if `pose` is the hands' joint array of a
/// frame [`kick`] prepared.
pub fn job_pose(pose: usize) {
    let Some(frame) = take_frame(pose) else { return };
    note_game_wrists(frame.header, pose, &frame.parents);
    let skeleton = JobPose { header: frame.header, pose, parents: &frame.parents };
    let Some(animated) = Snapshot::take(&skeleton) else { return };
    match hand_rig::pose::<crate::Wolf>(&skeleton, &animated, &frame.inputs) {
        Ok(_) => {
            EDITED.fetch_add(1, Relaxed);
            if frame.hide_left
                && let Some(arm) = left_arm(frame.header, &frame.parents)
            {
                hide(&skeleton, &arm);
            }
        }
        Err(why) => {
            if REFUSED.fetch_add(1, Relaxed) < 5 {
                log!("arms: not placed: {why}");
            }
        }
    }
}

/// The hands' blend kick (game thread): the moment captured with each gun's target. The left hand
/// follows its own controller while it holds a gun of its own (dual wielding) or the right hand a
/// one-handed weapon (the knife, the pipes, a grenade); otherwise both hands hold the right one's.
pub unsafe extern "system" fn arms_kick(hands: usize) {
    let flight = InFlight::enter();
    // While the controllers move the hotwiring minigame's hands, the arms stay as it animates them.
    if crate::on(crate::GUNS_AT)
        && !crate::extras::hotwire::active()
        && let Some(setup) = held_item::setup()
        && setup.arms
    {
        let dual = held_item::dual();
        if DUAL_LOGGED.swap(dual, Relaxed) != dual && DUAL_LINES.fetch_add(1, Relaxed) < 20 {
            match dual {
                true => log!("arms: the left hand holds an item of its own ({}; right hand: {}): it holds it on its controller", held_item::left_item_name().as_deref().unwrap_or("no weapon"), held_item::right_decl(setup.layout, hands).and_then(held_item::decl_name).as_deref().unwrap_or("no weapon")),
                false => log!("arms: the left hand holds nothing of its own again"),
            }
        }
        let free = !dual && held_item::right_one_handed(setup.layout, hands);
        // One tracking space for both targets and the skeleton's debug views.
        let tracking = held_item::hands_space(setup, hands);
        let target = |side| tracking.as_ref().and_then(|t| held_item::item_at(setup, t, hands, side));
        let left = if dual || free { target(LEFT_HAND) } else { None };
        let right = target(RIGHT_HAND);
        kick(setup.layout, hands, [left, right], tracking.map(|(space, _)| space), setup.units_per_metre(), dual, setup.fingers, setup.hide_free_hand);
    }
    // SAFETY: forwards the game's own call.
    unsafe { KICK_ORIGINAL.get(&flight)(hands) }
}

/// Every blend job's pose bounds (worker threads): the hands' pose gets its arms first.
pub unsafe extern "system" fn pose_bounds(skeleton: usize, pose: usize, skip: i16, outputs: usize) {
    let flight = InFlight::enter();
    if crate::on(crate::GUNS_AT) {
        job_pose(pose);
    }
    // SAFETY: forwards the game's own call.
    unsafe { BOUNDS_ORIGINAL.get(&flight)(skeleton, pose, skip, outputs) }
}

#[cfg(test)]
mod tests;
