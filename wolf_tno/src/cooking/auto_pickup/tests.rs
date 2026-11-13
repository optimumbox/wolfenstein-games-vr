use super::*;

const PLAYER: [f32; 6] = [-16.0, -16.0, 0.0, 16.0, 16.0, 74.0];

/// A 4-unit box at `(x, y, z)`.
fn at(x: f32, y: f32, z: f32) -> [f32; 6] {
    [x - 2.0, y - 2.0, z - 2.0, x + 2.0, y + 2.0, z + 2.0]
}

#[test]
fn an_item_within_reach_of_the_players_bounds_is_in_reach() {
    assert!(in_reach(&PLAYER, &at(0.0, 0.0, 1.0), REACH), "at the feet");
    assert!(in_reach(&PLAYER, &at(16.0 + REACH, 0.0, 40.0), REACH), "on a table, an arm's length ahead");
    assert!(in_reach(&PLAYER, &at(-30.0, 30.0, 74.0 + REACH - 4.0), REACH), "on a shelf above, to one side");
}

#[test]
fn an_item_past_reach_on_any_axis_is_not() {
    assert!(!in_reach(&PLAYER, &at(16.0 + REACH + 3.0, 0.0, 40.0), REACH), "too far ahead");
    assert!(!in_reach(&PLAYER, &at(0.0, -16.0 - REACH - 3.0, 40.0), REACH), "too far behind");
    assert!(!in_reach(&PLAYER, &at(0.0, 0.0, -REACH - 3.0), REACH), "a floor below");
}

#[test]
fn a_refused_item_waits_its_thinks_then_is_tried_again() {
    let mut refused = Refused::new();
    refused.note(0x1000, 100);
    assert!(refused.waits(0x1000, 100));
    assert!(refused.waits(0x1000, 100 + RETRY_THINKS - 1));
    assert!(!refused.waits(0x1000, 100 + RETRY_THINKS), "tried again");
    assert!(!refused.waits(0x2000, 101), "another item");
}

#[test]
fn the_oldest_refusal_gives_way_when_more_are_noted() {
    let mut refused = Refused::new();
    for i in 0..=REMEMBERED {
        refused.note(0x1000 + i, 5);
    }
    assert!(!refused.waits(0x1000, 5), "written over");
    assert!(refused.waits(0x1000 + REMEMBERED, 5));
}

#[test]
fn health_is_taken_only_while_all_of_it_fits() {
    assert!(heals_without_overfill(60.0, 100.0, 25.0), "60 + 25 fits");
    assert!(heals_without_overfill(75.0, 100.0, 25.0), "exactly to full");
    assert!(!heals_without_overfill(80.0, 100.0, 25.0), "would overfill");
    assert!(!heals_without_overfill(100.0, 100.0, 10.0), "full");
    assert!(!heals_without_overfill(150.0, 100.0, 10.0), "overcharged");
    assert!(!heals_without_overfill(50.0, 100.0, 0.0), "nothing in it");
}

#[test]
fn health_is_taken_up_to_120() {
    assert_eq!(HEALTH_CAP, 120.0);
    assert!(heals_without_overfill(110.0, HEALTH_CAP, 10.0), "110 + 10 fits");
    assert!(heals_without_overfill(95.0, HEALTH_CAP, 25.0), "exactly to 120");
    assert!(!heals_without_overfill(100.0, HEALTH_CAP, 25.0), "would pass 120");
    assert!(!heals_without_overfill(120.0, HEALTH_CAP, 10.0), "at 120");
}

#[test]
fn armor_is_taken_while_under_its_most() {
    assert!(armor_wanted(20.0, 100.0, 0.0));
    assert!(!armor_wanted(100.0, 100.0, 0.0));
    assert!(armor_wanted(20.0, 0.0, 50.0), "the item's most where the player's reads 0");
    assert!(!armor_wanted(0.0, 0.0, 0.0), "no most known");
}
