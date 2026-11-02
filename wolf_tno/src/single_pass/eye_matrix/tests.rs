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
