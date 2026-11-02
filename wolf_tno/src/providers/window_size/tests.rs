use super::*;

#[test]
fn the_window_holds_both_eyes_side_by_side() {
    assert_eq!(window_for((2688, 2688), true), (5376, 2688));
    assert_eq!(window_for((1440, 1600), false), (1440, 1600), "one eye fills it");
}
