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

#[cfg(test)]
mod tests;
