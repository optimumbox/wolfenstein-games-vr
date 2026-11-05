use super::*;

/// The step whose tick runs the game's clock at 0.99 of real time rather than 0.96.
const HEADSET_90: i32 = 90;

#[test]
fn every_step_keeps_the_games_own_speed() {
    for step in STEPS.into_iter().filter(|&s| s != HEADSET_90) {
        assert_eq!(1000 / step * step, 960, "{step} a second: a whole-millisecond tick at 0.96 of real time");
    }
    assert_eq!(1000 / HEADSET_90 * HEADSET_90, 990, "90 a second: an 11 ms tick, 0.99 of real time");
}

#[test]
fn a_90_hz_headset_gets_90() {
    assert_eq!(fitting(105.0, 96), 90, "94 to spare: 90 rather than 80");
    assert_eq!(fitting(300.0, 90), 90);
}

#[test]
fn the_rate_fits_what_the_pc_holds_with_room_to_spare() {
    assert_eq!(fitting(300.0, 240), 240);
    assert_eq!(fitting(300.0, 120), 120, "never above the ceiling");
    assert_eq!(fitting(140.0, 240), 120, "140 with a tenth to spare is 126: 120");
    assert_eq!(fitting(130.0, 240), 96, "117 to spare: 96");
    assert_eq!(fitting(40.0, 240), OWN, "never below the game's own");
}

#[test]
fn the_rate_rises_only_after_room_to_spare_holds() {
    let start = Instant::now();
    let mut steps = Steps::new(96);
    for window in 1..RISE_WINDOWS {
        assert_eq!(steps.next(140.0, 140.0, 240, start), 96, "window {window}: not yet");
    }
    assert_eq!(steps.next(140.0, 140.0, 240, start), 120);
    steps.next(400.0, 120.0, 240, start);
    assert_eq!(steps.next(125.0, 120.0, 240, start), 120, "a measure without room starts the count again");
    assert_eq!(steps.rises, 0);
}

#[test]
fn behind_the_rate_it_drops_to_what_the_pc_held_and_stays_under_it() {
    let start = Instant::now();
    let after = |seconds: u64| start + Duration::from_secs(seconds);
    // At 120 the game thread's measure promises far more than the PC holds.
    let mut steps = Steps::new(120);
    for _ in 0..RISE_WINDOWS {
        steps.next(450.0, 120.0, 240, after(0));
    }
    assert_eq!(steps.current, 240);
    // At 240 the frames fall behind: 172 a second is what the PC holds (twice: once is a hitch).
    assert_eq!(steps.next(172.0, 172.0, 240, after(1)), 240, "one slow measure is a hitch");
    assert_eq!(steps.next(172.0, 172.0, 240, after(1)), 160, "down to the step under 172 with 3% to spare");
    for window in 0..RISE_WINDOWS * 5 {
        assert_eq!(steps.next(450.0, 160.0, 240, after(2)), 160, "window {window}: held under the ceiling");
    }
    // After its wait it may try again; behind again, it waits twice as long.
    for _ in 0..RISE_WINDOWS {
        steps.next(450.0, 160.0, 240, after(32));
    }
    assert_eq!(steps.current, 240, "tried again after 30 s");
    steps.next(170.0, 170.0, 240, after(33));
    assert_eq!(steps.next(170.0, 170.0, 240, after(33)), 160);
    for _ in 0..RISE_WINDOWS {
        steps.next(450.0, 160.0, 240, after(70));
    }
    assert_eq!(steps.current, 160, "the second wait is 60 s");
}

#[test]
fn never_above_the_ceiling_or_below_the_games_own() {
    let start = Instant::now();
    let mut steps = Steps::new(60);
    for _ in 0..RISE_WINDOWS {
        steps.next(900.0, 60.0, 120, start);
    }
    assert_eq!(steps.current, 120);
    steps.next(30.0, 30.0, 120, start);
    assert_eq!(steps.next(30.0, 30.0, 120, start), OWN);
}

#[test]
fn a_rate_promised_but_not_achieved_drops() {
    // At 160 the game thread still waits now and then (it promises 400), but only 154 frames a
    // second come: the game runs slow, so the rate drops under what was achieved.
    let start = Instant::now();
    let mut steps = Steps::new(160);
    steps.next(400.0, 154.0, 240, start);
    assert_eq!(steps.next(400.0, 154.0, 240, start), 120);
    for _ in 0..RISE_WINDOWS * 3 {
        steps.next(400.0, 120.0, 240, start + Duration::from_secs(5));
    }
    assert_eq!(steps.current, 120, "held under the 154 achieved");
}

#[test]
fn a_hitch_at_the_games_own_rate_blocks_nothing() {
    let start = Instant::now();
    let mut steps = Steps::new(OWN);
    steps.next(400.0, 35.0, 240, start);
    steps.next(400.0, 35.0, 240, start);
    for _ in 0..RISE_WINDOWS {
        let achieved = f64::from(steps.current);
        steps.next(400.0, achieved, 240, start);
    }
    assert_eq!(steps.current, 240, "risen as promised: nothing was learnt at 60");
}

#[test]
fn paced_to_the_headset_a_hitch_keeps_the_rate() {
    let start = Instant::now();
    let mut steps = Steps::new(90);
    // Two refreshes missed in each half second (87 a second): a hitch, not the PC's limit.
    for _ in 0..BEHIND_PACED_WINDOWS * 3 {
        assert_eq!(steps.step(900.0, 86.0, 90, start, true), 90);
    }
    // Most refreshes missed for two seconds: the PC cannot hold 90.
    for _ in 1..BEHIND_PACED_WINDOWS {
        assert_eq!(steps.step(900.0, 62.0, 90, start, true), 90);
    }
    assert_eq!(steps.step(900.0, 62.0, 90, start, true), OWN);
}

#[test]
fn on_the_headsets_refresh_frames_already_achieved_raise_the_rate() {
    let start = Instant::now();
    let mut steps = Steps::new(OWN);
    // The present holds the render thread: the game thread promises 92, the frames make 89.5.
    for window in 1..RISE_WINDOWS {
        assert_eq!(steps.step(92.0, 89.5, 90, start, true), OWN, "window {window}: not yet");
    }
    assert_eq!(steps.step(92.0, 89.5, 90, start, true), 90);
    let mut off_grid = Steps::new(OWN);
    for _ in 0..RISE_WINDOWS * 2 {
        assert!(off_grid.step(92.0, 89.5, 90, start, false) < 90, "no headset refresh: the promise alone");
    }
    let mut short = Steps::new(OWN);
    for _ in 0..RISE_WINDOWS * 2 {
        assert!(short.step(80.0, 80.0, 90, start, true) < 90, "80 achieved does not keep 90");
    }
}
