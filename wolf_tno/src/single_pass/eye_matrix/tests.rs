use super::*;

#[test]
fn the_right_unprojection_takes_the_right_eyes_pixels_to_the_world() {
    // Asymmetric eyes 0.064 apart, the world turned and moved off the origin.
    let view = [[0.8, 0.0, -0.6, 3.0], [0.0, 1.0, 0.0, -2.0], [0.6, 0.0, 0.8, 5.0], [0.0, 0.0, 0.0, 1.0]];
    let shift = |x: f64| [[1.0, 0.0, 0.0, -x], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]];
    let left = multiply(&frustum(-0.14, 0.09, -0.13, 0.095, 0.1, 1000.0), &multiply(&shift(-0.032), &view));
    let right = multiply(&frustum(-0.09, 0.14, -0.13, 0.095, 0.1, 1000.0), &multiply(&shift(0.032), &view));
    let e = multiply(&right, &invert(&left).unwrap());
    // The engine's unprojection for the left eye: window to world.
    let left_unprojection = invert(&multiply(&TO_WINDOW, &left)).unwrap();
    let given = widen(&right_unprojection(&narrow(&e), &narrow(&left_unprojection)).unwrap());
    for world in [[0.3, -0.2, -4.0, 1.0], [-2.0, 1.0, 7.0, 1.0], [10.0, 3.0, -20.0, 1.0]] {
        // The point's window position and depth in the right eye.
        let clip: Vec<f64> = (0..4).map(|i| (0..4).map(|k| right[i][k] * world[k]).sum()).collect();
        let window = [clip[0] / clip[3] * 0.5 + 0.5, clip[1] / clip[3] * 0.5 + 0.5, clip[2] / clip[3] * 0.5 + 0.5, 1.0];
        let back: Vec<f64> = (0..4).map(|i| (0..4).map(|k| given[i][k] * window[k]).sum()).collect();
        for k in 0..3 {
            let found = back[k] / back[3];
            assert!((found - world[k]).abs() < 1e-2 * (1.0 + world[k].abs()), "{world:?}: {back:?}");
        }
    }
}

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

/// Rows rounded as a single-precision upload would, with one part in `noise` added to each.
fn noisy(rows: &Rows, seed: u64, noise: f32) -> Rows {
    let mut state = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    rows.map(|row| {
        row.map(|v| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let unit = (state % 2001) as f32 / 1000.0 - 1.0;
            v * (1.0 + unit / noise)
        })
    })
}

#[test]
fn the_average_over_many_frames_is_nearer_than_a_frames_own_measure() {
    let head = multiply(&turn(0.4), &translate(-30.0, -1.6, 20.0));
    let models: Vec<M> = (0..6).map(|i| multiply(&translate(i as f64 * 7.0 - 20.0, (i % 3) as f64, -40.0 + i as f64 * 3.0), &turn(i as f64))).collect();
    let truth = {
        let (left, right) = pair(&head, &models[0]);
        multiply(&widen(&right), &invert(&widen(&left)).unwrap())
    };
    let truth = narrow(&truth);
    let mut fit = Fit::default();
    let mut worst_single = 0.0f64;
    for frame in 0..200u64 {
        let pairs: Vec<(Rows, Rows)> =
            models.iter().map(|m| pair(&head, m)).enumerate().map(|(i, (l, r))| (noisy(&l, frame * 31 + i as u64, 3e5), noisy(&r, frame * 37 + i as u64 + 7, 3e5))).collect();
        let fresh = measure(&pairs);
        if let Some(fresh) = &fresh {
            worst_single = worst_single.max(apart(&fresh.matrix, &truth));
        }
        fit.add(&pairs, fresh.as_ref().map(|m| &m.matrix));
    }
    let averaged = apart(&fit.matrix().unwrap(), &truth);
    assert!(worst_single > 0.0);
    assert!(averaged * 4.0 < worst_single, "averaged {averaged:.2e}, a frame's own at worst {worst_single:.2e}");
}

#[test]
fn the_average_starts_afresh_when_the_eyes_move_apart() {
    let head = translate(0.0, -1.6, 0.0);
    let models = [translate(0.0, 0.0, -5.0), translate(1.0, 0.0, -6.0), translate(-2.0, 1.0, -8.0), multiply(&translate(2.0, 0.0, -9.0), &turn(0.5))];
    let pairs: Vec<_> = models.iter().map(|m| pair(&head, m)).collect();
    let mut fit = Fit::default();
    for _ in 0..10 {
        fit.add(&pairs, measure(&pairs).as_ref().map(|m| &m.matrix));
    }
    let before = fit.matrix().unwrap();
    // The same draws with the eyes twice as far apart: one frame of it is held, a few start afresh.
    let wider: Vec<_> = pairs.iter().map(|(l, r)| (*l, narrow(&multiply(&translate(-0.5, 0.0, 0.0), &widen(r))))).collect();
    let fresh = measure(&wider).unwrap();
    assert_eq!(fit.add(&wider, Some(&fresh.matrix)), FitStep::Held);
    assert_eq!(fit.matrix().unwrap(), before);
    let mut last = FitStep::Nothing;
    for _ in 1..FIT_SETTLES {
        last = fit.add(&wider, Some(&fresh.matrix));
    }
    assert_eq!(last, FitStep::Reset);
    assert!(apart(&fit.matrix().unwrap(), &fresh.matrix) < 1e-4);
}

#[test]
fn the_average_holds_through_frames_measuring_different_wrong_matrices() {
    let head = translate(0.0, -1.6, 0.0);
    let models = [translate(0.0, 0.0, -5.0), translate(1.0, 0.0, -6.0), translate(-2.0, 1.0, -8.0), multiply(&translate(2.0, 0.0, -9.0), &turn(0.5))];
    let pairs: Vec<_> = models.iter().map(|m| pair(&head, m)).collect();
    let mut fit = Fit::default();
    for _ in 0..10 {
        fit.add(&pairs, measure(&pairs).as_ref().map(|m| &m.matrix));
    }
    let before = fit.matrix().unwrap();
    // Streaming: each frame's pairs agree on another wrong matrix; none starts the average afresh.
    for frame in 0..(FIT_SETTLES * 4) {
        let off = 0.3 + 0.2 * f64::from(frame % 2);
        let wrong: Vec<_> = pairs.iter().map(|(l, r)| (*l, narrow(&multiply(&translate(-off, 0.0, 0.0), &widen(r))))).collect();
        let fresh = measure(&wrong).unwrap();
        assert_eq!(fit.add(&wrong, Some(&fresh.matrix)), FitStep::Held);
    }
    assert_eq!(fit.matrix().unwrap(), before);
}
