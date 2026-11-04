use super::*;
use crate::manifest::MANIFESTS;
use monaka_core::options::Options;

/// Looking down id +X from (100, 200, 87).
pub(crate) const EAST: [f32; 12] = [100.0, 200.0, 87.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

/// A head turned `yaw` degrees (positive left) with headset-like fields of view.
pub(crate) fn synthetic(yaw: f32) -> HeadPose {
    HeadPose::synthetic(yaw, 0.0, HeadPose::SYNTHETIC_FOV, 0.064)
}

/// [`turned_view`] at 50 units a metre without a dominant eye, the head's position as the pose has it.
fn turn(origin_axis: &[f32; 12], fov: [f32; 2], pose: &HeadPose, headset_fov: bool, baked: Option<Share>) -> Option<([f32; 12], [f32; 2])> {
    let options = Options::parse("dominant_eye=none");
    let config = EyeSettings::from_options(&options, &MANIFESTS[0]);
    let shared = Shared::from_options(&options, &MANIFESTS[0]);
    let convention = eng_idt::camera_convention(config.units_per_metre);
    let own = View { origin_axis: *origin_axis, fov };
    turned_view(&own, pose, pose.position, &config, &shared, &convention, headset_fov, baked).map(|view| (view.origin_axis, view.fov))
}

#[test]
fn a_head_turned_left_turns_the_view_left_and_widens_its_fov() {
    let pose = synthetic(90.0);
    let (turned, fov) = turn(&EAST, [80.0, 50.0], &pose, true, None).unwrap();
    let close = |a: &[f32], b: &[f32]| a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-4);
    assert!(close(&turned[..3], &EAST[..3]), "no head movement: same origin {:?}", &turned[..3]);
    assert!(close(&turned[3..6], &[0.0, 1.0, 0.0]), "forward is id left: {:?}", &turned[3..6]);
    assert!(close(&turned[6..9], &[-1.0, 0.0, 0.0]), "left is id back: {:?}", &turned[6..9]);
    assert!(close(&turned[9..12], &[0.0, 0.0, 1.0]));
    // Synthetic fov: outer edge 0.87 rad each side, 0.80 up and down.
    let degrees = |tangent: f32| 2.0 * tangent.atan().to_degrees();
    assert!((fov[0] - degrees(0.87f32.tan())).abs() < 1e-3 && (fov[1] - degrees(0.80f32.tan())).abs() < 1e-3, "{fov:?}");
    let (_, kept) = turn(&EAST, [80.0, 50.0], &pose, false, None).unwrap();
    assert_eq!(kept, [80.0, 50.0]);
}

#[test]
fn head_position_moves_the_view_in_game_units() {
    let mut pose = synthetic(0.0);
    pose.position = [0.1, 0.2, -0.3];
    let (turned, _) = turn(&EAST, [80.0, 50.0], &pose, true, None).unwrap();
    // 0.1 m right is 5 units id -Y, 0.2 m up 10 units +Z, 0.3 m forward 15 units +X.
    assert!(turned[..3].iter().zip([115.0, 195.0, 97.0]).all(|(a, b)| (a - b).abs() < 1e-3), "{:?}", &turned[..3]);
}

#[test]
fn baked_head_aim_is_swapped_not_added() {
    // The game camera turned 10 left by the mouse plus 30 left and 20 up by the head's aim.
    let (y, p) = (40f32.to_radians(), 20f32.to_radians());
    let forward = [p.cos() * y.cos(), p.cos() * y.sin(), p.sin()];
    let left = [-y.sin(), y.cos(), 0.0];
    let up = [-p.sin() * y.cos(), -p.sin() * y.sin(), p.cos()];
    let mut aimed = [0.0f32; 12];
    aimed[..3].copy_from_slice(&EAST[..3]);
    aimed[3..6].copy_from_slice(&forward);
    aimed[6..9].copy_from_slice(&left);
    aimed[9..12].copy_from_slice(&up);
    let baked = Share { yaw: 30f32.to_radians(), pitch: 20f32.to_radians() };
    let close = |a: &[f32], b: &[f32]| a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-4);
    // The head still where the command had it: the view is the game camera, not 70 left.
    let (sy, cy) = (baked.yaw * 0.5).sin_cos();
    let (sp, cp) = (baked.pitch * 0.5).sin_cos();
    let mut pose = synthetic(0.0);
    pose.orientation = [cy * sp, sy * cp, -sy * sp, cy * cp];
    let (turned, _) = turn(&aimed, [80.0, 50.0], &pose, true, Some(baked)).unwrap();
    assert!(close(&turned[3..6], &forward), "{:?} vs {forward:?}", &turned[3..6]);
    // The head 5 degrees further left since: the view is 5 further left, still 20 up.
    let y2 = 35f32.to_radians();
    let (sy, cy) = (y2 * 0.5).sin_cos();
    pose.orientation = [cy * sp, sy * cp, -sy * sp, cy * cp];
    let (turned, _) = turn(&aimed, [80.0, 50.0], &pose, true, Some(baked)).unwrap();
    let y3 = 45f32.to_radians();
    let expected = [p.cos() * y3.cos(), p.cos() * y3.sin(), p.sin()];
    assert!(close(&turned[3..6], &expected), "{:?} vs {expected:?}", &turned[3..6]);
}

#[test]
fn garbage_is_refused() {
    let pose = synthetic(0.0);
    let mut skewed = EAST;
    skewed[4] = 0.5;
    assert!(turn(&skewed, [80.0, 50.0], &pose, true, None).is_none());
    assert!(turn(&EAST, [0.0, 50.0], &pose, false, None).is_none());
    assert!(turn(&[f32::NAN; 12], [80.0, 50.0], &pose, true, None).is_none());
}

#[test]
fn frame_warping_draws_the_dominant_eye_and_the_other_every_refresh() {
    let right: Vec<usize> = (0..REFRESH * 2).map(|n| warp_eye(n, Some(DominantEye::Right))).collect();
    let others = right.iter().filter(|&&eye| eye == 0).count();
    assert_eq!(others, 2, "the left eye twice in two refreshes: {right:?}");
    assert_eq!(right[REFRESH as usize - 1], 0);
    assert_eq!(warp_eye(3, Some(DominantEye::Left)), 0);
    assert_eq!([warp_eye(4, Some(DominantEye::None)), warp_eye(5, None)], [0, 1], "without a dominant eye they take turns");
}

#[test]
fn a_warped_frame_is_drawn_from_the_eye_the_engine_would_have_offset() {
    // The engine's left eye is origin + separation * left, its right eye origin - separation * left.
    let view = View { origin_axis: EAST, fov: [80.0, 50.0] };
    let left = at_eye(view, 0, 1.6);
    let right = at_eye(view, 1, 1.6);
    assert_eq!(&left.origin_axis[..3], &[100.0, 201.6, 87.0]);
    assert_eq!(&right.origin_axis[..3], &[100.0, 198.4, 87.0]);
    assert_eq!(&left.origin_axis[3..], &EAST[3..], "only moved");
}

/// The engine's frustum maker: row-major, GL depth;
/// `far` at or below `near` is infinite.
fn engine_projection(left: f32, right: f32, bottom: f32, top: f32, near: f32, far: f32) -> [f32; 16] {
    let mut m = [0.0f32; 16];
    m[0] = 2.0 * near / (right - left);
    m[2] = (right + left) / (right - left);
    m[5] = 2.0 * near / (top - bottom);
    m[6] = (top + bottom) / (top - bottom);
    if near < far {
        m[10] = -(near + far) / (far - near);
        m[11] = -(2.0 * far * near) / (far - near);
    } else {
        m[10] = -1.0;
        m[11] = -2.0 * near;
    }
    m[14] = -1.0;
    m
}

#[test]
fn the_warps_depth_reads_what_the_engine_wrote() {
    let units = 50.0;
    let camera = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    for far in [0.0, 4000.0] {
        let gl = engine_projection(-3.0, 2.5, -2.8, 3.0, 2.0, far);
        let rendered = warp_rendered(&camera, &gl, 2.0, units).unwrap();
        for distance in [2.5f32, 40.0, 900.0, 3500.0] {
            // The window depth the engine's matrix writes for a point `distance` units ahead.
            let (z, w) = (gl[10] * -distance + gl[11], distance);
            let window = 0.5 * (z / w) + 0.5;
            // What the warp reads it as: depth = -P[10] + P[11] / distance (metres).
            let p = rendered.projection;
            let read = -p[10] + p[11] / (distance / units);
            assert!((window - read).abs() < 1e-5, "far {far}, {distance} units: {window} written, {read} read");
            assert!((rendered.mapping.nearness(window) - 2.0 / distance).abs() < 1e-4, "nearness at {distance}");
        }
        assert_eq!((rendered.projection[0], rendered.projection[2]), (gl[0], gl[2]), "x and y terms kept");
    }
}

#[test]
fn the_one_eye_layout_ends_on_a_named_record() {
    let named = |name: u64, sign: f32| {
        let mut record = [0u8; engine::LAYOUT_RECORD];
        record[..8].copy_from_slice(&name.to_le_bytes());
        record[engine::VIEW_EYE_SIGN..engine::VIEW_EYE_SIGN + 4].copy_from_slice(&sign.to_le_bytes());
        record
    };
    let (header, left, unused) = (named(0x1000, 0.0), named(0x2000, -1.0), named(0x2001, 0.0));
    let records = one_eye_records(header, left, unused);
    assert_eq!(records[3], header, "the walk stops at the header's name, never a null one");
    assert_eq!(records[1], left, "the left view, its eye sign set per frame");
    assert_eq!(records[2], unused);
}

#[test]
fn fsr_jitter_runs_amds_halton_points_inside_the_pixel() {
    assert_eq!(fsr_phases(1.5), 18, "8 times the pixel ratio 2.25, rounded up");
    assert_eq!(fsr_phases(2.0), 32);
    assert_eq!(fsr_jitter(0), [0.0, 1.0 / 3.0 - 0.5], "Halton (2, 3) from its first point");
    assert_eq!(fsr_jitter(1), [-0.25, 2.0 / 3.0 - 0.5]);
    for phase in 0..64 {
        let [x, y] = fsr_jitter(phase);
        assert!(x.abs() <= 0.5 && y.abs() <= 0.5, "phase {phase}: {x}, {y}");
    }
}

#[test]
fn a_smaller_eye_keeps_to_the_windows_bottom_left() {
    assert_eq!(WARP_REGION, [0.0, 0.0, 0.5, 1.0], "the left half");
    // DLSS Quality at 2560: a 1706 render, the window 3412 wide; the eye brought back to 2560.
    let display = warp_display(Some(((1706, 1706), (2560, 2560))), 1.5);
    assert_eq!(gl_frames::region_size(3412, 1706, [0.0, 0.0, display[0], display[1]]), (2560, 2560));
    assert_eq!(warp_display(None, 2.0), [1.0, 2.0], "no eye size: the scale");
}

#[test]
fn a_levelled_camera_keeps_its_place_and_heading_without_pitch_or_roll() {
    // Facing +Y (left in id axes), pitched 30 degrees down, and rolled.
    let (s, c) = (0.5f32, 0.8660254f32);
    let forward = [0.0, c, -s];
    let left = [-1.0, 0.0, 0.0];
    let up = [0.0, s, c];
    let mut origin_axis = [0.0f32; 12];
    origin_axis[..3].copy_from_slice(&[4.0, 5.0, 60.0]);
    origin_axis[3..6].copy_from_slice(&forward);
    origin_axis[6..9].copy_from_slice(&left);
    origin_axis[9..].copy_from_slice(&up);
    let level = levelled(View { origin_axis, fov: [90.0, 90.0] });
    assert_eq!(&level.origin_axis[..3], &[4.0, 5.0, 60.0], "the place kept");
    let near = |a: &[f32], b: &[f32]| a.iter().zip(b).all(|(a, b)| (a - b).abs() < 1e-5);
    assert!(near(&level.origin_axis[3..], &[0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0]), "{:?}", level.origin_axis);
}

#[test]
fn a_camera_tipping_over_straight_up_keeps_its_heading() {
    // Facing +X and pitched up to 80, 89, 91 and 100 degrees (past vertical: tipped back over the
    // head): it still faces +X.
    for degrees in [80.0f32, 89.0, 91.0, 100.0] {
        let (s, c) = degrees.to_radians().sin_cos();
        let mut origin_axis = [0.0f32; 12];
        origin_axis[3..].copy_from_slice(&[c, 0.0, s, 0.0, 1.0, 0.0, -s, 0.0, c]);
        let level = levelled(View { origin_axis, fov: [90.0, 90.0] });
        let forward = &level.origin_axis[3..6];
        assert!((forward[0] - 1.0).abs() < 1e-5 && forward[1].abs() < 1e-5, "{degrees} degrees: {forward:?}");
    }
}

#[test]
fn a_camera_looking_straight_down_takes_its_heading_from_its_up() {
    // Looking down, its up axis leaning toward +X: it faces +X.
    let mut origin_axis = [0.0f32; 12];
    origin_axis[3..].copy_from_slice(&[0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0]);
    let level = levelled(View { origin_axis, fov: [90.0, 90.0] });
    assert_eq!(&level.origin_axis[3..6], &[1.0, 0.0, 0.0]);
}

#[test]
fn a_machine_aiming_where_the_head_looks_sits_toward_each_eyes_nose() {
    // Eyes 3 units either side, the aim 1000 ahead: the left eye sees it right of straight ahead.
    let [left, right] = aim_tangents(&EAST, &EAST, 3.0, 1000.0).unwrap();
    assert!((left[0] - 0.003).abs() < 1e-5 && left[1].abs() < 1e-6, "{left:?}");
    assert!((right[0] + 0.003).abs() < 1e-5 && right[1].abs() < 1e-6, "{right:?}");
}

#[test]
fn a_machine_aiming_up_and_left_of_the_head_shows_up_and_left_in_both_eyes() {
    // The machine looks 10 degrees up and 20 left of where the head looks (+Y is id's left).
    let (yaw, pitch) = (20f32.to_radians(), 10f32.to_radians());
    let mut own = EAST;
    own[3..6].copy_from_slice(&[pitch.cos() * yaw.cos(), pitch.cos() * yaw.sin(), pitch.sin()]);
    for tangent in aim_tangents(&own, &EAST, 0.0, 1000.0).unwrap() {
        assert!((tangent[0] + yaw.tan()).abs() < 1e-4, "{tangent:?}");
        assert!((tangent[1] - pitch.tan() / yaw.cos()).abs() < 1e-4, "{tangent:?}");
    }
}

#[test]
fn a_machine_aiming_behind_the_eyes_has_no_aim_in_them() {
    let mut own = EAST;
    own[3] = -1.0;
    assert_eq!(aim_tangents(&own, &EAST, 3.0, 1000.0), None);
}
