use super::*;

/// 120 Hz: a refresh every 8,333,333 ns.
const P: u64 = 8_333_333;

fn grid(woke: u64) -> DisplayTiming {
    DisplayTiming { period_ns: P, woke_ns: woke }
}

#[test]
fn rates_that_divide_the_refresh_take_whole_refreshes() {
    assert_eq!(refreshes_per_frame(P, 120), Some(1));
    assert_eq!(refreshes_per_frame(P, 60), Some(2));
    assert_eq!(refreshes_per_frame(P, 96), None, "1.25 refreshes");
    assert_eq!(refreshes_per_frame(P, 160), None, "faster than the headset");
    assert_eq!(refreshes_per_frame(11_111_111, 60), None, "90 Hz: one and a half");
    assert_eq!(refreshes_per_frame(0, 60), None);
}

#[test]
fn frames_go_every_two_refreshes_on_the_grid() {
    let timing = grid(1_000_000_000);
    let (first, late) = next_slot(&timing, 0, None, 2, 1_000_000_000 + 3_000_000);
    assert_eq!((first, late), (1_000_000_000 + P, false), "the next grid point");
    let (second, late) = next_slot(&timing, 0, Some(first), 2, first + 10_000_000);
    assert_eq!((second, late), (first + 2 * P, false));
}

#[test]
fn a_grid_that_slid_a_little_is_followed() {
    // The viewer's latest wake 0.3 ms later than the grid the last slot was on.
    let last = 1_000_000_000;
    let timing = grid(last + P + 300_000);
    let (slot, _) = next_slot(&timing, 0, Some(last), 2, last + P);
    assert_eq!(slot, last + 2 * P + 300_000);
}

#[test]
fn a_late_wake_does_not_skip_a_refresh() {
    // The viewer's latest wake came 4.5 ms late (over half a refresh): the slot stays on the beat,
    // drawn toward that grid by an eighth of a refresh at most.
    let last = 1_000_000_000;
    for late in [4_500_000, P - 4_500_000] {
        let timing = grid(last + P + late);
        let (slot, skipped) = next_slot(&timing, 0, Some(last), 2, last + P);
        assert!(!skipped);
        assert!(slot.abs_diff(last + 2 * P) <= P / 8, "{late}: {}", slot as i64 - (last + 2 * P) as i64);
    }
}

#[test]
fn a_frame_that_ran_over_starts_on_the_next_point() {
    let last = 1_000_000_000;
    let timing = grid(last);
    // Two refreshes wanted, but it is already past that by most of a refresh.
    let now = last + 2 * P + 6_000_000;
    let (slot, late) = next_slot(&timing, 0, Some(last), 2, now);
    assert!(late);
    assert_eq!(slot, last + 3 * P);
}

#[test]
fn pairs_are_moved_toward_halfway_between_wakes() {
    let timing = grid(1_000_000_000);
    // Published just after a wake: 3.9 ms early of halfway.
    let published = 1_000_000_000 + 250_000;
    assert!(Phase::off(&timing, published) < -3_000_000);
    let phase = Phase::after(&timing, 0, published);
    // Frames start later: the offset grows by a share of what was off.
    let moved = phase as i64;
    assert!(moved > 100_000 && moved < 400_000, "{moved}");
    // Halfway: nothing moves.
    assert_eq!(Phase::after(&timing, 500, 1_000_000_000 + P / 2), 500);
}

#[test]
fn a_reprojecting_steamvrs_half_rate_keeps_the_headsets_grid() {
    let at = |period_ns| DisplayTiming { period_ns, woke_ns: 5 };
    assert_eq!(steady(at(11_111_111)).period_ns, 11_111_111);
    assert_eq!(steady(at(22_222_222)), at(11_111_111), "45 Hz reported: still 90's grid");
    assert_eq!(steady(at(11_111_111)).period_ns, 11_111_111);
}
