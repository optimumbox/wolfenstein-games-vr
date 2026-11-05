use super::*;

#[test]
fn hand_tags_are_sides() {
    assert_eq!(side_of_tag(0), Some(RIGHT_HAND));
    assert_eq!(side_of_tag(1), Some(LEFT_HAND));
    assert_eq!(side_of_tag(2), None);
}

#[test]
fn a_placed_gun_is_held_along_its_forward_axis() {
    // A gun 2 units up from the origin, turned to face id +Y (left): forward +Y, left -X, up +Z.
    let gun = Placed { weapon: 1, origin: [10.0, 20.0, 2.0], axis: [0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0], at: 5, build: 0 };
    *PLACED.locked() = [None, Some(gun)];
    let placed = <Wolf as HeldItems>::placed(RIGHT_HAND).unwrap();
    let line = placed.line.unwrap();
    assert_eq!((line.from, line.along, placed.posed), (gun.origin, [0.0, 1.0, 0.0], 5));
    // The barrel in the gun's own frame lands on the same line.
    assert_eq!(BARREL.in_world(&placed.holder).map(|world| (world.from, world.along)), Some((line.from, line.along)));
    assert!(<Wolf as HeldItems>::placed(LEFT_HAND).is_none());
    *PLACED.locked() = [None; 2];
}

#[test]
fn a_melee_weapon_lies_along_the_controller_grip_blade_out_of_the_fist() {
    // A grip 2 units up, facing id +Y (left): forward +Y, left -X, up +Z.
    let grip = ([10.0, -4.0, 60.0], [0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
    let (origin, axis) = melee_at(&grip).unwrap();
    assert_eq!(origin, grip.0, "the handle at the grip");
    assert_eq!(&axis[6..9], &grip.1[0..3], "the blade (the knife's up) out of the fist, along the grip's forward");
    assert_eq!(&axis[0..3], &[0.0, 0.0, -1.0], "the knife's forward along the grip's down");
    assert_eq!(&axis[3..6], &grip.1[3..6], "its left the grip's left");
}

/// A turn about z by `degrees`, then about x by `tilt` (rotation columns the axes).
fn turned(degrees: f32, tilt: f32) -> [f32; 9] {
    let (s, c) = degrees.to_radians().sin_cos();
    let (st, ct) = tilt.to_radians().sin_cos();
    mul3(&[c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0], &[1.0, 0.0, 0.0, 0.0, ct, -st, 0.0, st, ct])
}

/// The tray's wrists as the game holds it: each 17 units to a side of its middle, a little back.
fn tray_wrists() -> (Mat34, Mat34) {
    (from_rotation(&turned(-20.0, 10.0), [-7.8, -17.3, 2.5]), from_rotation(&turned(20.0, -10.0), [-7.8, 17.3, 2.5]))
}

/// A placement: origin, then axis rows.
type Placement = ([f32; 3], [f32; 9]);

/// The controller placements whose hands (a gun's grip on each) are `hand_r` and `hand_l`.
fn controllers(hand_r: &Mat34, hand_l: &Mat34) -> (Placement, Placement) {
    let palm = |hand: &Mat34, grip: &Mat34| from_joint(&compose(hand, &monaka_core::camera::rigid_inverse(grip)));
    (palm(hand_r, &skeleton::RIGHT_GRIP), palm(hand_l, &skeleton::LEFT_GRIP))
}

fn near(a: Vec3, b: Vec3, tolerance: f32) -> bool {
    length(sub(a, b)) < tolerance
}

#[test]
fn a_carried_tray_puts_the_games_wrists_on_the_hands_holding_it() {
    let wrists = tray_wrists();
    let tray = from_rotation(&turned(35.0, 5.0), [100.0, -40.0, 60.0]);
    let (hand_r, hand_l) = (compose(&tray, &wrists.0), compose(&tray, &wrists.1));
    let (right, left) = controllers(&hand_r, &hand_l);
    let placed = as_joint(&carried_at(&right, &left, &wrists).unwrap());
    let miss = length(sub(position(&compose(&placed, &wrists.0)), position(&hand_r)));
    assert!(miss < 0.05, "right wrist on the right hand: {miss}");
    assert!(near(position(&compose(&placed, &wrists.1)), position(&hand_l), 0.05), "left wrist on the left hand");
    assert!(near(position(&placed), position(&tray), 0.05), "the tray where it was held");
}

#[test]
fn hands_wider_than_the_tray_hold_it_centred_along_their_line() {
    let wrists = tray_wrists();
    let tray = from_rotation(&turned(-60.0, 0.0), [0.0, 0.0, 50.0]);
    let (hand_r, mut hand_l) = (compose(&tray, &wrists.0), compose(&tray, &wrists.1));
    // The left hand 10 units further out and 6 higher than the tray's edge.
    let out = sub(position(&hand_l), position(&hand_r));
    let further = monaka_core::math::add(position(&hand_l), monaka_core::math::add(scale(unit(out).unwrap(), 10.0), [0.0, 0.0, 6.0]));
    (hand_l[3], hand_l[7], hand_l[11]) = (further[0], further[1], further[2]);
    let (right, left) = controllers(&hand_r, &hand_l);
    let placed = as_joint(&carried_at(&right, &left, &wrists).unwrap());
    let (on_r, on_l) = (position(&compose(&placed, &wrists.0)), position(&compose(&placed, &wrists.1)));
    assert!(near(midpoint(on_r, on_l), midpoint(position(&hand_r), position(&hand_l)), 1e-3), "centred between the hands");
    let (held, hands) = (unit(sub(on_l, on_r)).unwrap(), unit(sub(position(&hand_l), position(&hand_r))).unwrap());
    assert!(dot(held, hands) > 0.9999, "the wrists' line along the hands' line");
}

#[test]
fn a_turn_between_two_directions_takes_one_onto_the_other() {
    for (from, to) in [([1.0, 0.0, 0.0], [0.0, 3.0, 0.0]), ([0.3, -1.0, 0.2], [-2.0, 0.5, 1.0]), ([0.0, 0.0, 1.0], [0.0, 0.0, -2.0]), ([1.0, 1.0, 0.0], [2.0, 2.0, 0.0])] {
        let turn = turn_between(from, to).unwrap();
        assert!(near(unit(apply3(&turn, from)).unwrap(), unit(to).unwrap(), 1e-4), "{from:?} -> {to:?}");
    }
    assert!(turn_between([0.0; 3], [1.0, 0.0, 0.0]).is_none());
}

/// An item placed during the view build before the eyes' (or during theirs) is where those eyes see
/// it; one placed two builds before is stale (the game stopped placing it).
#[test]
fn an_item_is_fresh_for_the_build_after_its_placement_only() {
    assert!(fresh_for(5, 5) && fresh_for(5, 6));
    assert!(!fresh_for(5, 7));
    let gun = Placed { weapon: 1, origin: [1.0, 2.0, 3.0], axis: [4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0], at: 0, build: 0 };
    assert_eq!(origin_axis(&gun), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
}
