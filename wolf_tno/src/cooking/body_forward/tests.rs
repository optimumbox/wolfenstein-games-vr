use super::*;
use std::time::Duration;

#[test]
fn a_stick_given_on_a_heading_turns_onto_the_one_wanted() {
    // Given a quarter turn left of the heading wanted: forward becomes the given heading's right.
    let (x, y) = turned_stick(0, 32767, std::f32::consts::FRAC_PI_2);
    assert!((x - 32767).abs() <= 1 && y.abs() <= 1, "{x} {y}");
    assert_eq!(turned_stick(1200, -3000, 0.0), (1200, -3000), "on it already: as pushed");
}

fn degrees(radians: f32) -> f32 {
    radians.to_degrees()
}

#[test]
fn a_glance_aside_barely_turns_the_body() {
    let start = Instant::now();
    let mut body = Body::default();
    assert_eq!(body.follow(0.0, start), 0.0, "the body starts where the head faces");
    // Thirty degrees left for a fifth of a second, read every 10 ms.
    let mut off = 0.0;
    for step in 1..=20 {
        off = body.follow(30f32.to_radians(), start + Duration::from_millis(10 * step));
    }
    assert!(degrees(off) > 22.5, "most of the glance is the head's alone: {:.1} degrees off the body", degrees(off));
}

#[test]
fn a_real_turn_takes_the_body_round_and_it_settles() {
    let start = Instant::now();
    let mut body = Body::default();
    body.follow(0.0, start);
    // A half turn to the left over a second, read every 10 ms.
    let mut off = 0.0;
    for step in 1..=100 {
        off = body.follow(PI * step as f32 / 100.0, start + Duration::from_millis(10 * step));
    }
    assert!(degrees(off) <= 45.0 + 1e-3, "never more than the dead zone behind: {:.1}", degrees(off));
    // Then facing that way for four seconds.
    for step in 101..=500 {
        off = body.follow(PI, start + Duration::from_millis(10 * step));
    }
    assert!(degrees(off).abs() < 3.0, "settled onto the head: {:.1} degrees off", degrees(off));
}

#[test]
fn the_body_follows_across_the_half_turn() {
    // The head just past a half turn one way and the body just short of it: they are close.
    let near = followed(PI - 0.05, -PI + 0.05, 0.0);
    assert!(wrapped(near - (PI - 0.05)).abs() < 1e-4, "within the dead zone, it stays: {near}");
}

#[test]
fn a_recentre_takes_the_body_with_it() {
    let start = Instant::now();
    let mut body = Body::default();
    body.follow(0.2, start);
    let off = body.follow(0.2 + 1.5, start + Duration::from_millis(10));
    assert!(degrees(off).abs() < 0.5, "the head is where it was against the body: {:.1}", degrees(off));
}

#[test]
fn halfway_runs_between_the_head_and_the_body() {
    // The head a quarter turn left: forward goes an eighth of a turn right of where it looks.
    let out = share_out(&Options::parse("move_heading=halfway"));
    let (x, y) = turned_stick(0, 32767, std::f32::consts::FRAC_PI_2 * out);
    let angle = f32::from(x).atan2(f32::from(y)).to_degrees();
    assert!((angle - 45.0).abs() < 0.1, "{angle}");
}

#[test]
fn the_body_steers_by_default() {
    assert_eq!(share_out(&Options::parse("")), 1.0, "unset: where the body faces");
    assert_eq!(share_out(&Options::parse("move_heading=head")), 0.0);
    assert_eq!(share_out(&Options::parse("move_heading=body")), 1.0);
    assert_eq!(share_out(&Options::parse("move_heading=sideways")), 1.0, "unknown: the default (the body)");
}

#[test]
fn a_menu_takes_the_stick_as_a_steady_step() {
    assert_eq!(menu_step(0, -30000, false), (0, -32767), "down, full");
    assert_eq!(menu_step(0, -29999, true), (0, -32767), "the dither's flip is the same step");
    assert_eq!(menu_step(20000, 9000, false), (32767, 0), "its main axis alone");
    assert_eq!(menu_step(0, -13000, false), (0, -32767), "a light push steps");
    assert_eq!(menu_step(5000, -5000, false), (0, 0), "under a third: at rest");
    assert_eq!(menu_step(0, -8000, true), (0, -32767), "held until back under a fifth");
    assert_eq!(menu_step(0, -6000, true), (0, 0), "then let go");
}
