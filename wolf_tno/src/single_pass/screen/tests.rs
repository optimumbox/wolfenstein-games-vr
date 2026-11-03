use super::*;

const MATRIX: Rows = [[1.0, 0.0, 0.0, 0.1], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]];
/// The right eye's window from the left eye's.
const K: [Row; 4] = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [-0.07, 0.0, 0.003, 1.0]];

fn position(i: usize) -> Rows {
    let t = i as f32;
    [[1.0 + t, 0.2, 0.0, t], [0.0, 1.0, 0.3 * t, 0.0], [0.0, 0.1, 1.0, 2.0 + t], [0.0, 0.0, 1.0, 3.0 + t]]
}

fn right_position(i: usize) -> Rows {
    let l = position(i);
    std::array::from_fn(|r| std::array::from_fn(|c| (0..4).map(|k| MATRIX[r][k] * l[k][c]).sum()))
}

/// A light's draw: row 0 the screen's bias and scale (both eyes'), rows 1-4 the window carried
/// into the light (each eye's own), row 5 its colour.
fn light(i: usize, right: bool) -> Drawn {
    let t = i as f32 * 0.37;
    let window: [Row; 4] = [[1.0 + t, 0.5, -t, 0.2], [0.1, 2.0 - t, 0.4, t], [t, 0.3, 1.5, -0.6], [0.0, 0.2 * t, 0.7, 1.0]];
    let fragment = std::iter::once([0.5, 0.5, 1.0 / 1792.0, 1.0 / 1792.0])
        .chain(window.iter().map(|row| if right { carry(row, &K) } else { *row }))
        .chain(std::iter::once([1.0, 0.8, 0.6, 1.0]))
        .collect();
    Drawn { program: 7, name: "blendlight", position: if right { right_position(i) } else { position(i) }, vertex: vec![[0.0; 4]], fragment }
}

#[test]
fn a_lights_window_rows_are_carried_to_the_right_eye() {
    let mut learner = Learner::default();
    for frame in 0..12 {
        for i in 0..6 {
            learner.left.push(light(frame * 6 + i, false));
        }
        // The right eye's pass makes the same draws, in another order.
        for i in (0..6).rev() {
            learner.right.push(light(frame * 6 + i, true));
        }
        learner.end_frame(Some(&MATRIX));
    }
    let (carried, k) = learner.shared(7).expect("judged shared");
    assert_eq!(carried, &[false, true, true, true, true, false]);
    let l = light(999, false);
    let r = light(999, true);
    for (i, row) in l.fragment.iter().enumerate() {
        let worked_out = if carried[i] { carry(row, k) } else { *row };
        assert!(near(&worked_out, &r.fragment[i]), "row {i}: {worked_out:?} against {:?}", r.fragment[i]);
    }
    assert_eq!(learner.counts(), (1, 0));
    assert!(!learner.wants(7, false));
    assert!(learner.wants(7, true));
}

#[test]
fn a_program_whose_rows_move_otherwise_is_drawn_by_each_eye() {
    let mut learner = Learner::default();
    for frame in 0..40 {
        for i in 0..6 {
            let n = frame * 6 + i;
            learner.left.push(light(n, false));
            let mut right = light(n, true);
            // An eye position: moved by the eye's own place, not by the window matrix.
            right.fragment[5] = [n as f32 * 0.3 + 0.032, 1.0, (n as f32).sin(), 1.0];
            learner.left.last_mut().expect("pushed").fragment[5] = [n as f32 * 0.3, 1.0, (n as f32).sin(), 1.0];
            learner.right.push(right);
        }
        learner.end_frame(Some(&MATRIX));
    }
    assert!(learner.shared(7).is_none());
    assert_eq!(learner.counts(), (0, 1));
    assert!(!learner.wants(7, true));
}

#[test]
fn vertex_rows_differing_between_the_eyes_keep_a_program_to_each_eye() {
    let mut learner = Learner::default();
    for i in 0..30 {
        learner.left.push(light(i, false));
        let mut right = light(i, true);
        right.vertex[0] = [0.032, 0.0, 0.0, 1.0];
        learner.right.push(right);
    }
    learner.end_frame(Some(&MATRIX));
    assert!(learner.shared(7).is_none());
    assert_eq!(learner.counts(), (0, 1));
}

#[test]
fn draws_whose_positions_do_not_pair_are_not_compared() {
    let mut learner = Learner::default();
    for i in 0..30 {
        learner.left.push(light(i, false));
        learner.right.push(light(i + 100, true));
    }
    learner.end_frame(Some(&MATRIX));
    assert!(learner.shared(7).is_none());
    assert_eq!(learner.counts(), (0, 0));
}
