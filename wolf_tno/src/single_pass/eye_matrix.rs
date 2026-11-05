//! The eye matrix: the right eye's clip position from the left eye's (`right = E · left`).
//!
//! Every scene draw's position is its model-view-projection rows times the vertex (`_va_` rows,
//! [`super::convert::position_rows`]). The right eye's matrix is its view-projection times the
//! same model, so `E = MVP_right · MVP_left⁻¹` is the same for every draw of a frame: the right
//! eye's view-projection over the left eye's. It changes only with the eye distance and the eyes'
//! frustums (the head moves both eyes together), and with the jitter both eyes share it changes by
//! less than a hundredth of a pixel, so the matrix measured in one frame serves the next.
//!
//! It is measured from the frame's draws themselves: the left eye's pass records each scene draw's
//! rows, the right eye's pass the same draws' rows in the same order ([`measure`]). Pairs that
//! disagree (draws that came in another order) are left out; too few agreeing ones and there is no
//! matrix, and the next frame draws each eye on its own.

/// A 4x4 matrix by rows (`m[row][column]`), as the position rows are uploaded.
pub type Rows = [[f32; 4]; 4];

type M = [[f64; 4]; 4];

fn widen(rows: &Rows) -> M {
    rows.map(|row| row.map(f64::from))
}

pub(super) fn multiply(a: &M, b: &M) -> M {
    let mut out = [[0.0; 4]; 4];
    for (i, row) in out.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            *cell = (0..4).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    out
}

/// The inverse, none when the matrix is (nearly) singular for its scale.
pub(super) fn invert(m: &M) -> Option<M> {
    let mut a = *m;
    let mut inverse = [[0.0; 4]; 4];
    for (i, row) in inverse.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    let scale = m.iter().flatten().fold(0.0f64, |s, v| s.max(v.abs()));
    if scale == 0.0 || !scale.is_finite() {
        return None;
    }
    for column in 0..4 {
        let pivot = (column..4).max_by(|&x, &y| a[x][column].abs().total_cmp(&a[y][column].abs()))?;
        if a[pivot][column].abs() < scale * 1e-7 {
            return None;
        }
        a.swap(column, pivot);
        inverse.swap(column, pivot);
        let divisor = a[column][column];
        for k in 0..4 {
            a[column][k] /= divisor;
            inverse[column][k] /= divisor;
        }
        for row in 0..4 {
            if row != column {
                let factor = a[row][column];
                for k in 0..4 {
                    a[row][k] -= factor * a[column][k];
                    inverse[row][k] -= factor * inverse[column][k];
                }
            }
        }
    }
    Some(inverse)
}

/// Window coordinates (x, y and depth 0..1, GL's default depth range) from clip space: `T`.
const TO_WINDOW: M = [[0.5, 0.0, 0.0, 0.5], [0.0, 0.5, 0.0, 0.5], [0.0, 0.0, 0.5, 0.5], [0.0, 0.0, 0.0, 1.0]];
const FROM_WINDOW: M = [[2.0, 0.0, 0.0, -1.0], [0.0, 2.0, 0.0, -1.0], [0.0, 0.0, 2.0, -1.0], [0.0, 0.0, 0.0, 1.0]];

/// The right eye's rows of a fragment shader's unprojection: `left` takes a left-eye pixel's
/// window position and depth (`vec4( window, depth, 1.0 )`) to where it is in the world; the
/// right eye's take a right-eye pixel there: `left · T · e⁻¹ · T⁻¹` (`e` the eye matrix, left
/// clip to right clip). The engine gives the decals the left eye's in both passes, and the eyes'
/// frusta differ (asymmetric): the same pixel of the right eye is another direction of the left,
/// its point far off the decal's box, which then clips every pixel.
pub fn right_unprojection(e: &Rows, left: &Rows) -> Option<Rows> {
    let inverse = invert(&widen(e))?;
    let m = multiply(&multiply(&multiply(&widen(left), &TO_WINDOW), &inverse), &FROM_WINDOW);
    let out = m.map(|row| row.map(|v| v as f32));
    out.iter().flatten().all(|v| v.is_finite()).then_some(out)
}

/// How far `e · left` is from `right`, relative to `right`'s size.
pub fn misfit(e: &Rows, left: &Rows, right: &Rows) -> f64 {
    disagreement(&widen(e), left, right)
}

/// How far `e · left` is from `right`, relative to `right`'s size.
fn disagreement(e: &M, left: &Rows, right: &Rows) -> f64 {
    let predicted = multiply(e, &widen(left));
    let right = widen(right);
    let scale = right.iter().flatten().fold(0.0f64, |s, v| s.max(v.abs())).max(1e-12);
    let worst = predicted.iter().flatten().zip(right.iter().flatten()).fold(0.0f64, |s, (p, r)| s.max((p - r).abs()));
    worst / scale
}

/// `e · left`: the rows a left-eye draw is carried to in the right eye.
pub fn carry(e: &Rows, left: &Rows) -> Rows {
    multiply(&widen(e), &widen(left)).map(|row| row.map(|v| v as f32))
}

/// How far rows `a` are from `b`, relative to `b`'s size.
pub fn apart(a: &Rows, b: &Rows) -> f64 {
    let scale = b.iter().flatten().fold(0.0f32, |s, v| s.max(v.abs())).max(1e-12);
    let worst = a.iter().flatten().zip(b.iter().flatten()).fold(0.0f32, |s, (p, r)| s.max((p - r).abs()));
    f64::from(worst / scale)
}

/// The largest disagreement a pair may have and still agree.
pub const AGREES: f64 = 2e-4;
/// Agreeing pairs needed for a matrix.
pub const LEAST_AGREEING: usize = 3;

/// What a frame's pairs gave.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Measured {
    pub matrix: Rows,
    /// Pairs that agreed with it, and pairs looked at.
    pub agreeing: usize,
    pub looked_at: usize,
}

/// The eye matrix from the frame's pairs (`(left rows, right rows)` of the same draws): built from
/// the first pair that gives one most others agree with; none with fewer than
/// [`LEAST_AGREEING`] agreeing.
pub fn measure(pairs: &[(Rows, Rows)]) -> Option<Measured> {
    let mut best: Option<(M, usize)> = None;
    for (left, right) in pairs.iter().take(8) {
        let Some(inverse) = invert(&widen(left)) else { continue };
        let e = multiply(&widen(right), &inverse);
        let agreeing = pairs.iter().filter(|(l, r)| disagreement(&e, l, r) <= AGREES).count();
        if best.as_ref().is_none_or(|(_, n)| agreeing > *n) {
            best = Some((e, agreeing));
        }
        if agreeing * 2 > pairs.len() {
            break;
        }
    }
    let (e, agreeing) = best?;
    (agreeing >= LEAST_AGREEING).then(|| Measured { matrix: e.map(|row| row.map(|v| v as f32)), agreeing, looked_at: pairs.len() })
}

/// [`measure`] from the passes' recorded draws (`(program, rows)` each) paired by a known matrix
/// `previous` rather than by their order: the passes cull for their own eye, so a draw only one
/// of them makes shifts every pair after it and the frame's pairs disagree, though the matrix
/// (the eyes' distance and frustums) is the same. Each right draw pairs with a left draw of its
/// program that `previous` carries onto it; the matrix is measured afresh from those pairs.
pub fn measure_with(previous: &Rows, left: &[(u32, Rows)], right: &[(u32, Rows)]) -> Option<Measured> {
    measure(&pairs_with(previous, left, right))
}

/// The passes' recorded draws paired by a known matrix `previous` ([`measure_with`]).
pub fn pairs_with(previous: &Rows, left: &[(u32, Rows)], right: &[(u32, Rows)]) -> Vec<(Rows, Rows)> {
    right
        .iter()
        .filter_map(|(program, r)| left.iter().find(|(p, l)| p == program && misfit(previous, l, r) <= AGREES * 10.0).map(|(_, l)| (*l, *r)))
        .collect()
}

/// The eye matrix averaged over many frames: each agreeing pair's own matrix
/// (`right · left⁻¹`) added in. One pair's matrix is only as good as its rows' single precision,
/// and the pair a frame measures from changes from frame to frame: measured afresh each frame the
/// right eye's whole world moves by up to a pixel between frames (thin things and lights shimmer,
/// the right eye's own decals flicker on the carried surfaces). The matrix does not change (the
/// eyes' distance and frustums), so the average of thousands of pairs stands still and is nearer the
/// truth. Older pairs fade once [`FIT_KEPT`] are held, and a frame's own measure far from the average
/// [`FIT_SETTLES`] frames in a row, its pairs agreeing with none of it and each frame's measure the
/// same new matrix, starts it afresh (the eye distance changed). While textures and meshes stream in,
/// the passes draw different things and a frame's pairs can agree on a wrong matrix: those differ
/// from frame to frame, and starting afresh from one shifted the right eye's near things for a few
/// frames (a flicker in the right eye only).
#[derive(Clone, Copy, Debug, Default)]
pub struct Fit {
    sum: M,
    weight: f64,
    misses: u32,
    /// The frames' own measure while they miss the average: the same each frame, or no reset.
    candidate: Option<M>,
}

/// Pairs' weight the average holds before older ones fade.
pub const FIT_KEPT: f64 = 2000.0;
/// Frames in a row whose own measure is far from the average before it starts afresh.
pub const FIT_SETTLES: u32 = 6;
/// How far (relative) a frame's own measure may be from the average and still be the same matrix.
pub const FIT_SAME: f64 = 1e-3;
/// How far (relative) one pair's own matrix may be from the average and still be added (a pair of
/// nearly singular rows gives a matrix far off though its rows agree).
const FIT_PAIR: f64 = 1e-2;

/// What a frame's pairs did to the average.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FitStep {
    /// This many pairs added.
    Added(usize),
    /// None added; the frame's own measure far from the average (kept for now).
    Held,
    /// Started afresh from the frame's own measure.
    Reset,
    /// Nothing to go on.
    Nothing,
}

impl Fit {
    /// The averaged matrix, none before any pair.
    pub fn matrix(&self) -> Option<Rows> {
        (self.weight > 0.0).then(|| self.mean().map(|row| row.map(|v| v as f32)))
    }

    fn mean(&self) -> M {
        self.sum.map(|row| row.map(|v| v / self.weight))
    }

    /// Adds the pairs that agree with the average (or, before one, with `fresh`, the frame's own
    /// measure).
    pub fn add(&mut self, pairs: &[(Rows, Rows)], fresh: Option<&Rows>) -> FitStep {
        if self.weight <= 0.0 {
            return match fresh {
                Some(fresh) => {
                    self.seed(pairs, fresh);
                    FitStep::Reset
                }
                None => FitStep::Nothing,
            };
        }
        let estimate = self.mean();
        let added = self.add_agreeing(pairs, &estimate);
        if added > 0 {
            self.misses = 0;
            self.candidate = None;
            return FitStep::Added(added);
        }
        let Some(fresh) = fresh else { return FitStep::Nothing };
        if apart(fresh, &estimate.map(|row| row.map(|v| v as f32))) <= FIT_SAME {
            return FitStep::Nothing;
        }
        let same = self.candidate.is_some_and(|c| apart(fresh, &c.map(|row| row.map(|v| v as f32))) <= FIT_SAME);
        self.misses = if same { self.misses + 1 } else { 1 };
        if !same {
            self.candidate = Some(widen(fresh));
        }
        if self.misses < FIT_SETTLES {
            return FitStep::Held;
        }
        self.seed(pairs, fresh);
        FitStep::Reset
    }

    fn seed(&mut self, pairs: &[(Rows, Rows)], fresh: &Rows) {
        *self = Self::default();
        if self.add_agreeing(pairs, &widen(fresh)) == 0 {
            self.sum = widen(fresh);
            self.weight = 1.0;
        }
    }

    fn add_agreeing(&mut self, pairs: &[(Rows, Rows)], estimate: &M) -> usize {
        let narrow = estimate.map(|row| row.map(|v| v as f32));
        let mut added = 0;
        for (left, right) in pairs {
            if disagreement(estimate, left, right) > AGREES {
                continue;
            }
            let Some(inverse) = invert(&widen(left)) else { continue };
            let own = multiply(&widen(right), &inverse);
            if apart(&own.map(|row| row.map(|v| v as f32)), &narrow) > FIT_PAIR {
                continue;
            }
            for (sum, own) in self.sum.iter_mut().flatten().zip(own.iter().flatten()) {
                *sum += own;
            }
            self.weight += 1.0;
            added += 1;
        }
        if self.weight > FIT_KEPT {
            let keep = FIT_KEPT / self.weight;
            self.sum = self.sum.map(|row| row.map(|v| v * keep));
            self.weight = FIT_KEPT;
        }
        added
    }
}

#[cfg(test)]
mod tests;
