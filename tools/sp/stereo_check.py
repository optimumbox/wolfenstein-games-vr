"""Find what does not line up between the eyes in a capture, with no hint of where to look.

    python stereo_check.py <capture folder> [--pair N] [--scale 2]

A capture folder (`captures/capture-N` of a run) holds each pair as the headset showed it
(`pair-N-left.jpg`, `pair-N-right.jpg`) and the pack's `draws.txt`, whose header gives the eye
matrix E: the right eye's clip position from the left eye's, `right = E . left`. With the game's
NDC (x right, y up, z -1 near .. 1 far), a thing the left eye sees at (x, y) and depth z the right
eye sees at x' = E00 x + E01 y + E02 z + E03, y' = E10 x + E11 y + E12 z + E13 (over w' = E3 . p).
So each left-eye patch has exactly one line in the right eye it may be on (its depth unknown): the
checker searches that line (z from near to past the far plane) and a few pixels above and below.

Flagged (in both directions: what the left eye shows that the right does not, and back):
- missing: a textured patch with no match anywhere on its line (black or other geometry in one eye,
  something drawn in one eye only, doubled things, a light or decal in one eye);
- height: it matches clearly better a few pixels above or below its line than on it (lights or
  things at different heights in each eye);
- the eyes' heights: the whole right eye's picture higher or lower than the frustum it is shown
  with (`pairs.txt`'s fields of view) puts it, against the left's (single pass's hardware keeps only
  x of the right eye's position, so eyes framed differently up and down lose that difference);
- depth: it matches best beyond the far plane (further apart than infinity: drawn in the wrong
  place sideways).

Writes `check.txt` (each region: kind, where in the eye in percent from the left and top, size,
the offset found) and `check.jpg` (both eyes side by side, regions outlined and numbered: red
missing, yellow height, magenta depth) into the capture folder. numpy and Pillow only.
"""
import argparse
import os
import re
import sys

import numpy as np
from PIL import Image, ImageDraw

PATCH = 5            # half size of the matched window, pixels at the working scale
TEXTURED = 6.0       # a window's grey-level spread needed to be judged (0..255)
MATCHED = 0.55       # normalised correlation counted a match
BETTER = 0.15        # how much better off its line a match must be to be a height fault
HEIGHTS = (-6, -3, 3, 6)
BLOCK = 16           # regions are gathered on blocks this size
BLOCK_SHARE = 0.35   # share of a block's judged pixels flagged that flags the block
MIN_BLOCKS = 3
NEAR_Z = 0.8         # NDC depth down to which a column's line must be inside the other image


def eye_matrix(folder):
    path = os.path.join(folder, 'draws.txt')
    if not os.path.exists(path):
        return None
    with open(path, encoding='utf-8', errors='replace') as f:
        for line in f:
            m = re.search(r'eye matrix Some\((\[\[.*?\]\])\)', line)
            if m:
                return np.array(eval(m.group(1), {}, {}), dtype=np.float64)  # a list of lists of numbers
    return None


def grey(path, scale):
    im = Image.open(path).convert('L')
    if scale != 1:
        im = im.resize((im.width // scale, im.height // scale), Image.BILINEAR)
    return np.asarray(im, dtype=np.float32)


def box(a, r):
    """Sum of each (2r+1)^2 window, same size (edges clamped)."""
    p = np.pad(a, r + 1, mode='edge').astype(np.float64)
    c = p.cumsum(0).cumsum(1)
    n = 2 * r + 1
    s = c[n:, n:] - c[:-n, n:] - c[n:, :-n] + c[:-n, :-n]
    return s[:a.shape[0], :a.shape[1]]


def shifted(img, dx, dy, fill=0.0):
    """img sampled at (x + dx, y + dy), integer shifts, outside filled with `fill`."""
    h, w = img.shape
    out = np.full_like(img, fill)
    xs0, xs1 = max(0, -dx), min(w, w - dx)
    ys0, ys1 = max(0, -dy), min(h, h - dy)
    if xs0 < xs1 and ys0 < ys1:
        out[ys0:ys1, xs0:xs1] = img[ys0 + dy:ys1 + dy, xs0 + dx:xs1 + dx]
    return out


class Matcher:
    """Normalised correlation of a's windows with b's shifted ones: a's and b's window sums kept,
    so each shift costs one window sum (of a times shifted b)."""

    def __init__(self, a, b, r):
        self.a, self.b, self.r = a.astype(np.float64), b.astype(np.float64), r
        n = (2 * r + 1) ** 2
        self.n = n
        self.sa = box(self.a, r)
        self.va = box(self.a * self.a, r) - self.sa * self.sa / n
        self.sb = box(self.b, r)
        self.vb = box(self.b * self.b, r) - self.sb * self.sb / n
        self.spread = np.sqrt(np.maximum(self.va / n, 0))

    def at(self, dx, dy):
        h, w = self.a.shape
        r = self.r
        sab = box(self.a * shifted(self.b, dx, dy), r)
        sb = shifted(self.sb, dx, dy)
        vb = shifted(self.vb, dx, dy)
        out = (sab - self.sa * sb / self.n) / np.sqrt(np.maximum(self.va * vb, 1e-6))
        # Windows reaching outside b are not judged.
        ys, xs = np.mgrid[0:1, 0:w]
        inside = np.ones((h, w), bool)
        x = np.arange(w) + dx
        inside[:, (x < r) | (x >= w - r)] = False
        y = np.arange(h) + dy
        inside[(y < r) | (y >= h - r), :] = False
        return np.where(inside, out, -2.0)


def align_rows(right, E, z_mid):
    """The right image resampled so each left row's line lies on the same row (E's y terms)."""
    h, w = right.shape
    v = np.arange(h, dtype=np.float64)
    y = 1.0 - (v + 0.5) / h * 2.0
    yr = E[1, 1] * y + E[1, 2] * z_mid + E[1, 3]
    wr = E[3, 1] * y + E[3, 2] * z_mid + E[3, 3]
    vr = (1.0 - yr / wr) * h / 2.0 - 0.5
    lo = np.clip(np.floor(vr).astype(int), 0, h - 1)
    hi = np.clip(lo + 1, 0, h - 1)
    t = (vr - np.floor(vr))[:, None].astype(np.float32)
    return right[lo] * (1 - t) + right[hi] * t


def shift_range(E, w, forward):
    """Pixel shifts (right x minus left x at the image middle) from the near plane to past the far one."""
    near = E[0, 2] * -1.0 + E[0, 3]
    far = E[0, 2] * 1.0 + E[0, 3]
    a, b = sorted((near * w / 2.0, far * w / 2.0))
    lo, hi = int(np.floor(a)) - 12, int(np.ceil(b)) + 12
    far_px = far * w / 2.0
    if not forward:
        lo, hi, far_px = -hi, -lo, -far_px
    return lo, hi, far_px


def search(a, b, E, r):
    """For each pixel of a: the best match on its line in b and its shift, the best a few rows off
    it (within 6 pixels of that shift), and the far plane's shift."""
    h, w = a.shape
    lo, hi, far_px = shift_range(E, w, True)
    m = Matcher(a, b, r)
    best = np.full(a.shape, -2.0)
    best_dx = np.zeros(a.shape, np.int32)
    for dx in range(lo, hi + 1):
        s = m.at(dx, 0)
        better = s > best
        best = np.where(better, s, best)
        best_dx = np.where(better, dx, best_dx)
    off = np.full(a.shape, -2.0)
    off_dy = np.zeros(a.shape, np.int32)
    for dy in HEIGHTS:
        for dx in range(lo, hi + 1):
            near = np.abs(best_dx - dx) <= 6
            if not near.any():
                continue
            s = np.where(near, m.at(dx, dy), -2.0)
            better = s > off
            off = np.where(better, s, off)
            off_dy = np.where(better, dy, off_dy)
    # The x term (E00 - 1) x, small, so the shift is measured from each column's own far plane.
    x = (np.arange(w) + 0.5) / w * 2.0 - 1.0
    far_cols = far_px + (E[0, 0] - 1.0) * x * w / 2.0
    # Columns judged: the line inside the other eye's image from the far plane to NEAR_Z (nearer
    # things at the edge are outside the other eye's view, not missing from it).
    near_cols = far_cols + (E[0, 2] * NEAR_Z + E[0, 3] - (E[0, 2] + E[0, 3])) * w / 2.0
    xs = np.arange(w)
    ends = np.stack([xs + far_cols, xs + near_cols])
    room = (ends.min(0) >= r) & (ends.max(0) < w - r)
    return best, best_dx, off, off_dy, far_cols[None, :], m.spread, room[None, :]


def regions(mask, judged):
    """Blocks flagged in mask (share of judged pixels), gathered into 4-connected regions."""
    h, w = mask.shape
    bh, bw = h // BLOCK, w // BLOCK
    m = mask[:bh * BLOCK, :bw * BLOCK].reshape(bh, BLOCK, bw, BLOCK).sum((1, 3))
    j = judged[:bh * BLOCK, :bw * BLOCK].reshape(bh, BLOCK, bw, BLOCK).sum((1, 3))
    flagged = (j >= BLOCK * BLOCK * 0.2) & (m >= np.maximum(j, 1) * BLOCK_SHARE)
    seen = np.zeros_like(flagged)
    out = []
    for i in range(bh):
        for k in range(bw):
            if not flagged[i, k] or seen[i, k]:
                continue
            stack, cells = [(i, k)], []
            seen[i, k] = True
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for ny, nx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                    if 0 <= ny < bh and 0 <= nx < bw and flagged[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            if len(cells) >= MIN_BLOCKS:
                ys = [c[0] for c in cells]
                xs = [c[1] for c in cells]
                out.append((min(xs) * BLOCK, min(ys) * BLOCK, (max(xs) + 1) * BLOCK, (max(ys) + 1) * BLOCK, len(cells)))
    return out


def judge(a, b, E, r):
    best, best_dx, off, off_dy, far, spread, room = search(a, b, E, r)
    # Judged: textured, and its line inside the other eye's image somewhere (the outer edge of each
    # eye is seen by that eye alone).
    judged = (spread >= TEXTURED) & (best > -1.5) & room
    missing = judged & (best < MATCHED) & (off < MATCHED)
    height = judged & ~missing & (off > best + BETTER) & (off >= MATCHED + 0.1)
    # Beyond the far plane: past it on the side away from the near plane.
    beyond = (best_dx - far) * np.sign(E[0, 2])
    depth = judged & ~missing & ~height & (best >= MATCHED + 0.1) & (beyond > 3)
    return judged, {'missing': (missing, None), 'height': (height, off_dy), 'depth': (depth, beyond)}


def where(x0, y0, x1, y1, w, h):
    cx, cy = (x0 + x1) / 2 / w, (y0 + y1) / 2 / h
    col = 'left' if cx < 0.33 else 'right' if cx > 0.67 else 'middle'
    row = 'top' if cy < 0.33 else 'bottom' if cy > 0.67 else 'middle'
    return f'{row}-{col}' if (row, col) != ('middle', 'middle') else 'centre'


def told_fov(folder, pair):
    """Each eye's field of view (left, right, up, down; radians) in the pose record of `pair`."""
    path = os.path.join(folder, 'pairs.txt')
    if not os.path.exists(path):
        return None
    with open(path, encoding='utf-8', errors='replace') as f:
        for line in f:
            if line.startswith(f'pair {pair} '):
                fovs = re.findall(r'fov \[([^\]]+)\]', line)
                if len(fovs) == 2:
                    return [np.array([float(v) for v in f.split(',')]) for f in fovs]
    return None


def eye_heights(folder, pair, left_path, right_path):
    """How far the right eye's whole picture sits from the height its own frustum gives it.

    Each eye is shown with the frustum in its pose record. A thing in direction tangent t (up) is
    at row (tan(up) - t) / (tan(up) - tan(down)) of each eye's image, whatever its distance (the eyes
    sit side by side), so between the eyes it moves by a known number of rows at each height. The
    picture is matched block by block (any sideways shift, a band of heights) and the rows it
    actually moved are compared with that: a whole eye drawn higher or lower than its frustum
    (single pass drawing the right eye at the left eye's height) shows as one offset everywhere."""
    fov = told_fov(folder, pair)
    if fov is None:
        return None, []
    s = 4
    L = grey(left_path, s)
    R = grey(right_path, s)
    h, w = L.shape
    up = [np.tan(f[2]) for f in fov]
    down = [np.tan(f[3]) for f in fov]
    lines = []
    if abs(up[0] - up[1]) > 1e-4 or abs(down[0] - down[1]) > 1e-4:
        expected_mid = ((up[1] - 0.0) / (up[1] - down[1]) - (up[0] - 0.0) / (up[0] - down[0])) * h * s
        lines.append(f'the eyes are framed differently up and down (right eye needs things {expected_mid:+.0f} rows from the left at the centre):'
                     " single pass keeps only the left eye's height for the right eye's copy of a draw")
    B = 12
    residuals = []
    for by in range(B * 2, h - B * 2, B * 2):
        for bx in range(B * 2, w - B * 2, B * 2):
            a = L[by - B:by + B, bx - B:bx + B]
            if a.std() < TEXTURED:
                continue
            t = up[0] - (by + 0.5) * s / (h * s) * (up[0] - down[0])
            row_r = (up[1] - t) / (up[1] - down[1]) * h
            want = row_r - by
            best = (-2.0, 0, 0)
            an = (a - a.mean()) / (a.std() + 1e-6)
            for dy in range(-12, 13):
                y = by + dy
                if y - B < 0 or y + B > h:
                    continue
                band = R[y - B:y + B]
                for dx in range(-w // 3, w // 12):
                    x = bx + dx
                    if x - B < 0 or x + B > w:
                        continue
                    b = band[:, x - B:x + B]
                    sd = b.std()
                    if sd < 1e-3:
                        continue
                    c = float((an * (b - b.mean())).mean() / sd)
                    if c > best[0]:
                        best = (c, dx, dy)
            if best[0] >= 0.8:
                residuals.append((best[2] - want) * s)
    if len(residuals) >= 8:
        off = float(np.median(residuals))
        spread = float(np.percentile(residuals, 75) - np.percentile(residuals, 25))
        verdict = "height: the right eye's picture is off its frustum" if abs(off) > 3 + spread / 2 else "the eyes' heights match their frustums"
        lines.append(f'{verdict}: {off:+.0f} rows (lower +) from where its frustum puts it, over {len(residuals)} matched blocks (middle half within {spread:.0f})')
    else:
        lines.append(f'eye heights: too few blocks matched ({len(residuals)}) to judge')
    return residuals, lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('folder')
    ap.add_argument('--pair', type=int, default=1)
    ap.add_argument('--scale', type=int, default=2)
    args = ap.parse_args()
    E = eye_matrix(args.folder)
    if E is None:
        sys.exit('no eye matrix in draws.txt (single pass off?): nothing to check against')
    left_path = os.path.join(args.folder, f'pair-{args.pair}-left.jpg')
    right_path = os.path.join(args.folder, f'pair-{args.pair}-right.jpg')
    L = grey(left_path, args.scale)
    R = align_rows(grey(right_path, args.scale), E, 0.99)
    h, w = L.shape
    # Back from the right eye: rows aligned the other way (the same small scale, undone).
    Rinv_E = np.linalg.inv(E)
    L_from_R = align_rows(grey(left_path, args.scale), Rinv_E, 0.99)
    R_plain = grey(right_path, args.scale)
    found = []
    judged_l, kinds_l = judge(L, R, E, PATCH)
    judged_r, kinds_r = judge(R_plain, L_from_R, Rinv_E, PATCH)
    for eye, judged, kinds in (('left', judged_l, kinds_l), ('right', judged_r, kinds_r)):
        for kind, (mask, amount) in kinds.items():
            for (x0, y0, x1, y1, n) in regions(mask, judged):
                detail = ''
                if amount is not None:
                    vals = amount[y0:y1, x0:x1][mask[y0:y1, x0:x1]]
                    if vals.size:
                        med = float(np.median(vals)) * args.scale
                        detail = f', {"rows" if kind == "height" else "pixels beyond the far plane"} {med:+.0f}'
                found.append((eye, kind, x0, y0, x1, y1, n, detail))
    _, height_lines = eye_heights(args.folder, args.pair, left_path, right_path)
    lines = [f'{args.folder} pair {args.pair}: eye matrix {E.tolist()}', *height_lines,
             'Regions (eye they were found from: "left" = shown by the left eye, not lined up in the right; "right" = the other way). Positions are % of the eye image from its left and top.']
    colours = {'missing': (255, 40, 40), 'height': (255, 220, 0), 'depth': (255, 0, 255)}
    li = Image.open(left_path).convert('RGB')
    ri = Image.open(right_path).convert('RGB')
    draws = {'left': ImageDraw.Draw(li), 'right': ImageDraw.Draw(ri)}
    for i, (eye, kind, x0, y0, x1, y1, n, detail) in enumerate(sorted(found, key=lambda f: -f[6]), 1):
        lines.append(f'{i}. {kind} in the {"right" if eye == "left" else "left"} eye of what the {eye} eye shows, {where(x0, y0, x1, y1, w, h)}: '
                     f'x {100 * x0 / w:.0f}-{100 * x1 / w:.0f}%, y {100 * y0 / h:.0f}-{100 * y1 / h:.0f}% ({n} blocks){detail}')
        s = args.scale
        d = draws[eye]
        d.rectangle([x0 * s, y0 * s, x1 * s, y1 * s], outline=colours[kind], width=6)
        d.text((x0 * s + 8, y0 * s + 8), str(i), fill=colours[kind])
    if not found:
        lines.append('nothing out of line found')
    both = Image.new('RGB', (li.width + ri.width, li.height))
    both.paste(li, (0, 0))
    both.paste(ri, (li.width, 0))
    both.thumbnail((2560, 1280))
    both.save(os.path.join(args.folder, 'check.jpg'), quality=85)
    with open(os.path.join(args.folder, 'check.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
