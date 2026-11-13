use super::*;

#[test]
fn a_beam_runs_from_the_muzzle_to_a_spot_where_it_hits() {
    let frustum = Frustum::centred([1.0, 1.0]).unwrap();
    // From the middle of a 200x200 eye to its right edge's middle, ahead.
    let boxes = beam_boxes([0, 0, 200, 200], &frustum, [[0.0, 0.0], [0.5, 0.0]], true).unwrap();
    let first = boxes[0];
    assert!((first[0] - 98).abs() <= 1 && (first[1] - 98).abs() <= 1, "{first:?}");
    let spot = *boxes.last().unwrap();
    assert_eq!((spot[2], spot[3]), (8, 8));
    assert!((spot[0] + 4 - 150).abs() <= 1, "{spot:?}");
    // Nothing hit: no spot. Off the eye: clipped to it.
    let missed = beam_boxes([0, 0, 200, 200], &frustum, [[0.0, 0.0], [3.0, 0.0]], false).unwrap();
    assert!(missed.iter().all(|b| b[0] >= 0 && b[0] + b[2] <= 200));
}

const HUD: Hud = Hud { shown: true, world: false, scale: 0.5, distance: 2.0, markers: false };

#[test]
fn a_centred_eye_gets_a_centred_box_of_the_layers_shape() {
    let frustum = Frustum { left: -1.0, right: 1.0, up: 1.0, down: -1.0 };
    let [x, y, w, h] = placed_rect([0, 0, 1920, 2160], 0, &frustum, 0.0, &HUD).unwrap();
    assert_eq!((y, h), (540, 1080), "half the height, centred");
    // 16:9 in angle: 1 tangent unit high is 1.778 wide, of 2 across 1920 pixels.
    assert_eq!(w, (1920.0f32 * 1.7778 / 2.0).round() as i32);
    assert!((x - (1920 - w) / 2).abs() <= 1, "{x} {w}");
}

#[test]
fn off_centre_eyes_put_it_straight_ahead_and_toward_the_nose() {
    // A left eye reaching further out (left) than in: straight ahead is right of its centre.
    let frustum = Frustum { left: -1.2, right: 0.8, up: 1.0, down: -1.0 };
    let [x, _, w, _] = placed_rect([0, 0, 1920, 2160], 0, &frustum, 0.0, &HUD).unwrap();
    let centre = x as f32 + w as f32 * 0.5;
    assert!((centre - 1.2 / 2.0 * 1920.0).abs() <= 1.0, "straight ahead at {centre}");
    // With an IPD the left eye's box moves right (toward the nose), the right eye's left.
    let [x2, ..] = placed_rect([0, 0, 1920, 2160], 0, &frustum, 0.064, &HUD).unwrap();
    assert!(x2 > x);
    let mirrored = Frustum { left: -0.8, right: 1.2, up: 1.0, down: -1.0 };
    let [r0, ..] = placed_rect([1920, 0, 1920, 2160], 1, &mirrored, 0.0, &HUD).unwrap();
    let [r1, ..] = placed_rect([1920, 0, 1920, 2160], 1, &mirrored, 0.064, &HUD).unwrap();
    assert!(r1 < r0, "{r1} {r0}");
}

#[test]
fn a_frame_without_a_pose_gets_the_same_box_in_each_eye() {
    // Both eye halves of a 3840x2160 window: the layer's own 16:9 shape across each whole eye.
    let left = flat_rect([0, 0, 1920, 2160]).unwrap();
    let right = flat_rect([1920, 0, 1920, 2160]).unwrap();
    assert_eq!(left, [0, 540, 1920, 1080]);
    assert_eq!(right, [1920 + left[0], left[1], left[2], left[3]], "the same place in each eye");
    assert_eq!(flat_rect([0, 0, 0, 2160]), None);
}

#[test]
fn the_crosshair_moves_from_the_layers_centre_to_the_aim() {
    let frustum = Frustum { left: -1.0, right: 1.0, up: 1.0, down: -1.0 };
    // Aim straight ahead of an eye without an IPD: the layer's centre, no move.
    assert_eq!(aim_shift([0, 0, 1920, 2160], 0, &frustum, 0.0, &HUD, [0.0, 0.0]), Some([0, 0]));
    // A quarter of the field right and up: a quarter of the image's width and height.
    assert_eq!(aim_shift([0, 0, 1920, 2160], 0, &frustum, 0.0, &HUD, [0.5, 0.5]), Some([480, 540]));
    // With an IPD the layer sits toward the nose, so an aim straight ahead lies outward of it.
    let [dx, _] = aim_shift([0, 0, 1920, 2160], 0, &frustum, 0.064, &HUD, [0.0, 0.0]).unwrap();
    assert!(dx < 0, "{dx}");
}
