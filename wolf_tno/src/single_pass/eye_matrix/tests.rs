use super::*;

/// An off-centre perspective projection (GL convention) by rows.
fn frustum(left: f64, right: f64, bottom: f64, top: f64, near: f64, far: f64) -> M {
    [
        [2.0 * near / (right - left), 0.0, (right + left) / (right - left), 0.0],
        [0.0, 2.0 * near / (top - bottom), (top + bottom) / (top - bottom), 0.0],
        [0.0, 0.0, -(far + near) / (far - near), -2.0 * far * near / (far - near)],
        [0.0, 0.0, -1.0, 0.0],
    ]
}

fn translate(x: f64, y: f64, z: f64) -> M {
    [[1.0, 0.0, 0.0, x], [0.0, 1.0, 0.0, y], [0.0, 0.0, 1.0, z], [0.0, 0.0, 0.0, 1.0]]
}

fn turn(yaw: f64) -> M {
    let (s, c) = yaw.sin_cos();
    [[c, 0.0, s, 0.0], [0.0, 1.0, 0.0, 0.0], [-s, 0.0, c, 0.0], [0.0, 0.0, 0.0, 1.0]]
}

fn narrow(m: &M) -> Rows {
    m.map(|row| row.map(|v| v as f32))
}

/// Both eyes' rows for a draw of `model` seen from a head at `head` (world to head).
fn pair(head: &M, model: &M) -> (Rows, Rows) {
    let ipd = 0.064;
    // Mirrored off-centre frustums, the nose side narrower.
    let left_eye = multiply(&frustum(-0.11, 0.09, -0.1, 0.1, 0.05, 500.0), &translate(ipd / 2.0, 0.0, 0.0));
    let right_eye = multiply(&frustum(-0.09, 0.11, -0.1, 0.1, 0.05, 500.0), &translate(-ipd / 2.0, 0.0, 0.0));
    let view_model = multiply(head, model);
    (narrow(&multiply(&left_eye, &view_model)), narrow(&multiply(&right_eye, &view_model)))
}

#[test]
fn the_matrix_carries_every_draw_from_the_left_eye_to_the_right() {
    let head = multiply(&turn(0.7), &translate(-3.0, -1.6, 2.0));
    let models = [translate(0.0, 0.0, 0.0), translate(5.0, 0.0, -2.0), multiply(&translate(1.0, 2.0, 3.0), &turn(1.3)), translate(-4.0, 1.0, 7.0)];
    let pairs: Vec<_> = models.iter().map(|m| pair(&head, m)).collect();
    let measured = measure(&pairs).unwrap();
    assert_eq!((measured.agreeing, measured.looked_at), (4, 4));
    // A point of a draw not among the pairs lands where the right eye draws it.
    let (left, right) = pair(&head, &translate(2.0, -1.0, -6.0));
    let point = [0.3f64, 0.2, -0.4, 1.0];
    let apply = |m: &M, v: &[f64; 4]| -> [f64; 4] { std::array::from_fn(|i| (0..4).map(|k| m[i][k] * v[k]).sum()) };
    let left_clip = apply(&widen(&left), &point);
    let carried = apply(&widen(&measured.matrix), &left_clip);
    let expected = apply(&widen(&right), &point);
    for i in 0..4 {
        assert!((carried[i] - expected[i]).abs() < 1e-4, "{carried:?} vs {expected:?}");
    }
}

#[test]
fn a_pair_from_another_draw_is_left_out_and_too_few_give_none() {
    let head = translate(0.0, -1.6, 0.0);
    let mut pairs: Vec<_> = [translate(0.0, 0.0, -5.0), translate(1.0, 0.0, -5.0), translate(2.0, 0.0, -5.0)].iter().map(|m| pair(&head, m)).collect();
    // The right eye's rows of a different draw.
    pairs.push((pair(&head, &translate(0.0, 0.0, -9.0)).0, pair(&head, &turn(2.0)).1));
    let measured = measure(&pairs).unwrap();
    assert_eq!((measured.agreeing, measured.looked_at), (3, 4));
    assert_eq!(measure(&pairs[2..]), None);
}

#[test]
fn an_orthographic_or_singular_draw_gives_nothing() {
    let zero = [[0.0f32; 4]; 4];
    assert_eq!(measure(&[(zero, zero); 4]), None);
}

#[test]
fn draws_one_eye_culled_are_paired_by_the_last_matrix_not_their_order() {
    // Each pass culls for its own eye: the right eye's pass makes a draw the left's does not, and
    // the left's one the right's does not, so pairing by order puts different draws together.
    let head = multiply(&turn(0.4), &translate(-1.0, -1.6, 3.0));
    let models = [translate(0.0, 0.0, -5.0), translate(1.0, 0.5, -6.0), multiply(&translate(-2.0, 0.0, -4.0), &turn(0.9)), translate(3.0, -1.0, -8.0), translate(0.5, 2.0, -7.0)];
    let both: Vec<_> = models.iter().map(|m| pair(&head, m)).collect();
    let right_only = pair(&head, &translate(6.0, 0.0, -3.0)).1;
    let left_only = pair(&head, &translate(-6.0, 0.0, -3.0)).0;
    let left: Vec<(u32, Rows)> = std::iter::once((7, left_only)).chain(both.iter().map(|p| (7, p.0))).collect();
    let right: Vec<(u32, Rows)> = std::iter::once((7, right_only)).chain(both.iter().map(|p| (7, p.1))).rev().collect();
    let by_order: Vec<(Rows, Rows)> = left.iter().zip(&right).map(|(l, r)| (l.1, r.1)).collect();
    assert_eq!(measure(&by_order), None, "paired by order, the draws disagree");
    let last = measure(&both).unwrap().matrix;
    let measured = measure_with(&last, &left, &right).unwrap();
    assert_eq!(measured.agreeing, models.len());
    // Carried by it, each left draw is its right eye's draw, and the right eye's own draw is none
    // of them.
    for (l, r) in &both {
        assert!(apart(&carry(&measured.matrix, l), r) <= AGREES * 10.0);
    }
    assert!(both.iter().all(|(l, _)| apart(&carry(&measured.matrix, l), &right_only) > AGREES * 10.0));
}
