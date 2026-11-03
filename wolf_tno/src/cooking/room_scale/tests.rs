use super::*;
use monaka_core::camera::from_back_up;

fn camera_facing(yaw_left_degrees: f32) -> Mat34 {
    // Facing -z at 0 (canonical: back is +z), turned left by `yaw_left_degrees`, pitched down a bit.
    let yaw = yaw_left_degrees.to_radians();
    from_back_up([yaw.sin(), 0.3, yaw.cos()], [0.0, 1.0, 0.0], [4.0, 1.7, -2.0]).unwrap()
}

#[test]
fn a_step_ahead_pushes_the_stick_forward() {
    let push = push_for([0.0, -FULL_SPEED], &camera_facing(0.0));
    assert!(push[0].abs() < 1e-4 && (push[1] - 1.0).abs() < 1e-4, "{push:?}");
}

#[test]
fn a_step_to_the_side_pushes_along_the_cameras_heading() {
    // Facing world -x (a quarter turn left): a step toward world -x is forward, toward world -z right.
    let camera = camera_facing(90.0);
    let ahead = push_for([-2.0, 0.0], &camera);
    assert!(ahead[0].abs() < 1e-4 && ahead[1] > 0.0, "{ahead:?}");
    let right = push_for([0.0, -2.0], &camera);
    assert!(right[0] > 0.0 && right[1].abs() < 1e-4, "{right:?}");
}

#[test]
fn a_slow_follow_still_clears_the_dead_zone() {
    let push = push_for([0.0, -0.05], &camera_facing(0.0));
    let length = push[0].hypot(push[1]);
    assert!(length > STICK_DEAD_ZONE && length < STICK_DEAD_ZONE + 0.02, "{length}");
    assert_eq!(push_for([0.0, 0.0], &camera_facing(0.0)), [0.0; 2], "standing still: no push");
}
