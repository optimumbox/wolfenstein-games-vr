use super::*;

#[test]
fn a_stick_given_on_a_heading_turns_onto_the_one_wanted() {
    // Given a quarter turn left of the heading wanted: forward becomes the given heading's right.
    let (x, y) = turned_stick(0, 32767, std::f32::consts::FRAC_PI_2);
    assert!((x - 32767).abs() <= 1 && y.abs() <= 1, "{x} {y}");
    assert_eq!(turned_stick(1200, -3000, 0.0), (1200, -3000), "on it already: as pushed");
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
fn the_head_steers_by_default() {
    assert_eq!(share_out(&Options::parse("")), 0.0, "unset: where the head looks");
    assert_eq!(share_out(&Options::parse("move_heading=head")), 0.0);
    assert_eq!(share_out(&Options::parse("move_heading=body")), 1.0);
    assert_eq!(share_out(&Options::parse("move_heading=sideways")), 0.0, "unknown: the default");
}
