use super::*;

#[test]
fn marks_tell_the_piece_and_come_off() {
    let state = 0x4000000000065u64;
    assert_eq!(piece_of(state), None);
    assert_eq!(piece_of(state | MARKS[RIGHT_HAND]), Some(RIGHT_HAND));
    assert_eq!(piece_of(state | MARKS[LEFT_HAND]), Some(LEFT_HAND));
    assert_eq!(piece_of(state | MARKS[CHECKLISTS]), Some(CHECKLISTS));
    assert_eq!(unmarked(state | MARKS[LEFT_HAND] | MARKS[RIGHT_HAND] | MARKS[CHECKLISTS]), state);
}

#[test]
fn the_marks_are_bits_the_engine_never_tests() {
    // The engine's GL state setter tests up to bit 50 (front face).
    assert!(MARKS.iter().all(|&mark| mark.trailing_zeros() > 50));
    assert_eq!(MARKS.len(), PIECES.len());
}

#[test]
fn a_screen_fill_is_a_rectangle_over_nearly_all_the_layout() {
    let layout = (1280.0, 720.0);
    assert!(is_fill(4, [0.0, 0.0, 1280.0, 720.0], layout));
    assert!(is_fill(6, [-10.0, 20.0, 1290.0, 710.0], layout));
    // A strip, a corner piece, or text drawn over the screen are not.
    assert!(!is_fill(4, [0.0, 600.0, 1280.0, 720.0], layout));
    assert!(!is_fill(4, [0.0, 0.0, 300.0, 200.0], layout));
    assert!(!is_fill(40, [0.0, 0.0, 1280.0, 720.0], layout));
}
