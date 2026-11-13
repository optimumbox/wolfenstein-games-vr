use super::*;

#[test]
fn a_hit_counts_only_on_the_barrel_from_its_origin() {
    let (origin, forward, end) = ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [4096.0, 0.0, 0.0]);
    assert_eq!(hit_along(origin, forward, origin, end, &[[300.0, 1.0, 0.0]]), Some(300.0));
    // Off the barrel: the next candidate.
    assert_eq!(hit_along(origin, forward, origin, end, &[[300.0, 80.0, 0.0], [500.0, 0.0, 0.0]]), Some(500.0));
    // Behind it, or a trace begun elsewhere (the head's): none.
    assert_eq!(hit_along(origin, forward, origin, end, &[[-50.0, 0.0, 0.0]]), None);
    assert_eq!(hit_along(origin, forward, [0.0, 30.0, 0.0], end, &[[300.0, 0.0, 0.0]]), None);
    // Stopped only at its reach: nothing hit.
    assert_eq!(hit_along(origin, forward, origin, end, &[[4096.0, 0.0, 0.0]]), None);
}

#[test]
fn the_focus_view_turns_and_moves_with_the_head() {
    // The camera looking along +x; the head 4 units left and turned a quarter left: the trace starts
    // at the head and looks along +y.
    let camera = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let head = [0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
    let turn = follow_head::turn_between(&camera, &head);
    let (origin, rows) = followed([10.0, 20.0, 30.0], &camera, [0.0, 4.0, 0.0], &turn);
    assert_eq!(origin, [10.0, 24.0, 30.0]);
    assert!(rows.iter().zip(&head).all(|(a, b)| (a - b).abs() < 1e-6), "{rows:?}");
}

#[test]
fn far_use_reaches_fifty_metres_where_the_head_looks() {
    use crate::providers::pickups::far_reached;
    let units = 50.0;
    assert!(far_reached([0.0; 3], [1.0, 0.0, 0.0], [49.0 * units, 0.0, 0.0], units).is_some());
    assert!(far_reached([0.0; 3], [1.0, 0.0, 0.0], [51.0 * units, 0.0, 0.0], units).is_none());
    assert!(far_reached([0.0; 3], [1.0, 0.0, 0.0], [10.0 * units, 5.0 * units, 0.0], units).is_some());
    assert!(far_reached([0.0; 3], [1.0, 0.0, 0.0], [0.0, 10.0 * units, 0.0], units).is_none());
}
