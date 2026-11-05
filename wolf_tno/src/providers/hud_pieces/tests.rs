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
    let quad = |x0: f32, y0: f32, x1: f32, y1: f32| vec![[x0, y0], [x1, y0], [x1, y1], [x0, y1]];
    assert_eq!(fill_quads(&quad(0.0, 0.0, 1280.0, 720.0), layout), vec![0]);
    // Among other shapes, after a few vertices of text.
    let mut drawn = vec![[10.0, 10.0], [20.0, 10.0], [20.0, 30.0]];
    drawn.extend(quad(-10.0, 20.0, 1290.0, 710.0));
    drawn.extend(quad(100.0, 100.0, 200.0, 140.0));
    assert_eq!(fill_quads(&drawn, layout), vec![3]);
    // A strip, a corner piece, or a shape spread over the screen are not.
    assert!(fill_quads(&quad(0.0, 600.0, 1280.0, 720.0), layout).is_empty());
    assert!(fill_quads(&quad(0.0, 0.0, 300.0, 200.0), layout).is_empty());
    assert!(fill_quads(&[[0.0, 0.0], [640.0, 0.0], [1280.0, 720.0], [0.0, 720.0]], layout).is_empty());
}
