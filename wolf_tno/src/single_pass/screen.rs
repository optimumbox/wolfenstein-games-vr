//! The other eye's rows for the programs reading the scene's maps ([`super::programs::Screen`]).
//!
//! Such a program (a light, a decal, fog, a soft particle) places what it reads by its pixel and
//! the depth under it, through rows of its own: a light's are the window's position carried into
//! the light's projection, and the left eye's carry the left eye's window. Drawn once for both
//! eyes, its copy in layer 1 reads the right eye's maps ([`super::convert::layered_fragment`]) and
//! the rows given for it ([`super::convert::RIGHT_ROWS`]), worked out from the left eye's here.
//!
//! The right eye's window is the left eye's through a fixed matrix (the eyes' projections and their
//! distance fix it, as they fix the eye matrix), so a row carrying the window is the left eye's row
//! times that matrix, `K`; the other rows are the same in both eyes. Which rows are which, and `K`,
//! are learnt from the frames' own draws: each eye's pass makes the same draws with its own rows,
//! paired by their positions through the eye matrix. A program is drawn for both eyes once its
//! right eye's rows have come out as worked out over [`JUDGED_AFTER`] pairs; one whose don't (a
//! row moved some other way, or vertex rows differing between the eyes, which would move what the
//! vertex shader passes on) is drawn by each eye. Shared programs are checked again on later
//! frames.

use super::eye_matrix::{self, Rows};
use monaka_producer::log;
use std::collections::HashMap;

pub type Row = [f32; 4];

/// A draw of a program reading the scene's maps: its position rows, its other vertex rows and its
/// fragment rows, as uploaded.
pub struct Drawn {
    pub program: u32,
    pub name: &'static str,
    pub position: Rows,
    pub vertex: Vec<Row>,
    pub fragment: Vec<Row>,
}

/// Pairs compared before a program is judged.
pub const JUDGED_AFTER: u64 = 24;
/// The share of pairs a shared program's rows may miss by (one-off pairings gone wrong).
const MISSES_ALLOWED: f64 = 0.02;
/// Pairs a program is compared over before it is given up on unjudged (its rows never fitted).
const GIVEN_UP_AFTER: u64 = 2000;

/// What a program's right eye's rows are.
#[derive(Clone, Debug, PartialEq)]
pub enum Verdict {
    /// Worked out from the left eye's: each row carried by `K` or the same.
    Shared { carried: Vec<bool>, k: [Row; 4] },
    /// Drawn by each eye, for the reason given.
    Own(String),
}

#[derive(Default)]
struct Learning {
    /// Normal equations of `right = left · K` over the rows that differ between the eyes.
    a: [[f64; 4]; 4],
    b: [[f64; 4]; 4],
    fitted: u64,
    k: Option<[[f64; 4]; 4]>,
    /// Pairs compared; per row, pairs where it differed and where `left · K` missed it (with a fit).
    pairs: u64,
    differed: Vec<u64>,
    missed: Vec<u64>,
    with_fit: u64,
    vertex_differed: u64,
    /// Pairs compared since the verdict, and those it missed.
    checked: u64,
    check_missed: u64,
    verdict: Option<Verdict>,
}

#[derive(Default)]
pub struct Learner {
    pub left: Vec<Drawn>,
    pub right: Vec<Drawn>,
    programs: HashMap<u32, Learning>,
}

impl Learner {
    /// Whether the program's draws are recorded this frame: while it is being learnt, and on the
    /// checking frames once shared.
    pub fn wants(&self, program: u32, checking: bool) -> bool {
        match self.programs.get(&program).and_then(|l| l.verdict.as_ref()) {
            None => true,
            Some(Verdict::Shared { .. }) => checking,
            Some(Verdict::Own(_)) => false,
        }
    }

    /// The program's right eye's rows when they are worked out from the left eye's.
    pub fn shared(&self, program: u32) -> Option<(&[bool], &[Row; 4])> {
        match self.programs.get(&program)?.verdict.as_ref()? {
            Verdict::Shared { carried, k } => Some((carried, k)),
            Verdict::Own(_) => None,
        }
    }

    /// The programs drawn for both eyes and those drawn by each.
    pub fn counts(&self) -> (usize, usize) {
        let verdicts = self.programs.values().filter_map(|l| l.verdict.as_ref());
        verdicts.fold((0, 0), |(shared, own), v| if matches!(v, Verdict::Shared { .. }) { (shared + 1, own) } else { (shared, own + 1) })
    }

    /// The frame's recorded draws compared, their eyes paired through `matrix` (the frame's eye
    /// matrix); the lists cleared.
    pub fn end_frame(&mut self, matrix: Option<&Rows>) {
        let (left, right) = (std::mem::take(&mut self.left), std::mem::take(&mut self.right));
        let Some(matrix) = matrix else { return };
        let mut used = vec![false; left.len()];
        for r in &right {
            let found = (0..left.len()).find(|&i| !used[i] && left[i].program == r.program && eye_matrix::misfit(matrix, &left[i].position, &r.position) <= eye_matrix::AGREES * 10.0);
            let Some(i) = found else { continue };
            used[i] = true;
            let l = &left[i];
            if l.fragment.len() != r.fragment.len() || l.vertex.len() != r.vertex.len() {
                continue;
            }
            self.programs.entry(r.program).or_default().compare(l, r);
        }
    }
}

/// Whether `a` and `b` are the same row (to the rows' own precision).
fn same(a: &Row, b: &Row) -> bool {
    let scale = a.iter().chain(b).fold(0.0f32, |s, v| s.max(v.abs()));
    a.iter().zip(b).all(|(x, y)| (x - y).abs() <= 1e-4 * scale + 1e-6)
}

/// `row · k`.
pub fn carry(row: &Row, k: &[Row; 4]) -> Row {
    std::array::from_fn(|j| (0..4).map(|i| row[i] * k[i][j]).sum())
}

fn carry_wide(row: &Row, k: &[[f64; 4]; 4]) -> Row {
    std::array::from_fn(|j| (0..4).map(|i| f64::from(row[i]) * k[i][j]).sum::<f64>() as f32)
}

/// Whether `carried` is `right` to within what a fit over single-precision rows keeps.
fn near(carried: &Row, right: &Row) -> bool {
    let scale = carried.iter().chain(right).fold(0.0f32, |s, v| s.max(v.abs()));
    carried.iter().zip(right).all(|(x, y)| (x - y).abs() <= 2e-3 * scale + 1e-5)
}

impl Learning {
    fn compare(&mut self, l: &Drawn, r: &Drawn) {
        let rows = l.fragment.len();
        self.differed.resize(rows, 0);
        self.missed.resize(rows, 0);
        if let Some(Verdict::Shared { carried, k }) = &self.verdict {
            let wrong = l.vertex.iter().zip(&r.vertex).any(|(a, b)| !same(a, b))
                || l.fragment.iter().zip(&r.fragment).zip(carried).any(|((a, b), &carried)| if carried { !near(&carry(a, k), b) } else { !same(a, b) });
            self.checked += 1;
            self.check_missed += u64::from(wrong);
            if self.checked >= JUDGED_AFTER && self.check_missed as f64 > self.checked as f64 * MISSES_ALLOWED {
                let why = format!("its right eye's rows stopped coming out as worked out ({} of {} pairs)", self.check_missed, self.checked);
                log!("single pass: {} ({}) reading the scene's maps: each eye draws it from here: {why}", l.name, l.program);
                self.verdict = Some(Verdict::Own(why));
            }
            return;
        }
        if self.verdict.is_some() {
            return;
        }
        self.pairs += 1;
        self.vertex_differed += u64::from(l.vertex.iter().zip(&r.vertex).any(|(a, b)| !same(a, b)));
        for (i, (a, b)) in l.fragment.iter().zip(&r.fragment).enumerate() {
            if same(a, b) {
                continue;
            }
            self.differed[i] += 1;
            if let Some(k) = &self.k {
                self.missed[i] += u64::from(!near(&carry_wide(a, k), b));
            }
            for x in 0..4 {
                for y in 0..4 {
                    self.a[x][y] += f64::from(a[x]) * f64::from(a[y]);
                    self.b[x][y] += f64::from(a[x]) * f64::from(b[y]);
                }
            }
            self.fitted += 1;
        }
        self.with_fit += u64::from(self.k.is_some());
        if self.k.is_none() && self.fitted >= 8 {
            self.k = fit(&self.a, &self.b);
        }
        self.judge(l);
    }

    fn judge(&mut self, l: &Drawn) {
        let differing = self.differed.iter().any(|&n| n > 0);
        let verdict = if self.vertex_differed > 0 {
            Some(Verdict::Own(format!("its vertex rows differ between the eyes ({} of {} pairs)", self.vertex_differed, self.pairs)))
        } else if !differing && self.pairs >= JUDGED_AFTER {
            Some(Verdict::Shared { carried: vec![false; self.differed.len()], k: [[0.0; 4]; 4] })
        } else if let Some(k) = self.k.filter(|_| self.with_fit >= JUDGED_AFTER) {
            let rows = self.differed.len();
            match (0..rows).find(|&i| self.missed[i] as f64 > self.with_fit as f64 * MISSES_ALLOWED) {
                Some(i) => Some(Verdict::Own(format!("its row {i} is not carried by the eyes' window matrix ({} of {} pairs)", self.missed[i], self.with_fit))),
                None => Some(Verdict::Shared { carried: self.differed.iter().map(|&n| n > 0).collect(), k: k.map(|row| row.map(|v| v as f32)) }),
            }
        } else if self.pairs >= GIVEN_UP_AFTER {
            Some(Verdict::Own("its rows were never fitted".to_owned()))
        } else {
            None
        };
        if let Some(verdict) = verdict {
            match &verdict {
                Verdict::Shared { carried, .. } => {
                    let n = carried.iter().filter(|&&c| c).count();
                    log!("single pass: {} ({}) reading the scene's maps: both eyes in one draw, {n} of its {} rows carried to the right eye's window ({} pairs)", l.name, l.program, carried.len(), self.pairs);
                }
                Verdict::Own(why) => log!("single pass: {} ({}) reading the scene's maps: each eye draws it: {why}", l.name, l.program),
            }
            self.verdict = Some(verdict);
        }
    }
}

/// `K` from the normal equations (`a · K = b`), a little ridge keeping a fit over rows that span
/// less than all four directions.
fn fit(a: &[[f64; 4]; 4], b: &[[f64; 4]; 4]) -> Option<[[f64; 4]; 4]> {
    let trace: f64 = (0..4).map(|i| a[i][i]).sum();
    if trace <= 0.0 || !trace.is_finite() {
        return None;
    }
    let mut ridged = *a;
    for (i, row) in ridged.iter_mut().enumerate() {
        row[i] += trace * 1e-9;
    }
    Some(eye_matrix::multiply(&eye_matrix::invert(&ridged)?, b))
}

#[cfg(test)]
mod tests;
