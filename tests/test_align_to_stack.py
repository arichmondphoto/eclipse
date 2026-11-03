"""The alignment found on 2x2 superpixels must line up the full-size frames that Stack warps.

Align measures on binned frames; pipeline._to_full turns each result into a full-resolution
matrix, and stack.warp_region resamples the full-size frame with it. A half-pixel slip in that
hand-over would pass every test that writes the matrices by hand.
"""

import math

import numpy as np
import pytest

from tse_core.align import AlignFrame, AlignParams, align_sequence
from tse_core.pipeline import _to_full
from tse_core.stack import warp_region
from tse_core.synthetic import SyntheticScene


def _bin2(img):
    h, w = img.shape[0] // 2 * 2, img.shape[1] // 2 * 2
    return img[:h, :w].reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))


def test_superpixel_alignment_lines_up_the_full_size_frames():
    """Frames shifted by sub-pixel amounts and turned by field rotation (0.4 deg per frame); the
    rotation is what makes the superpixel/full-size pixel-centre convention matter."""
    sc = SyntheticScene(size=896, sun_radius=110)
    c = 447.5
    shifts = [(3.37, -2.81), (-1.62, 4.18), (0.0, 0.0), (2.71, 1.29), (-4.46, -0.73)]
    exps = [1e-3 * 2 ** (k / 1.25) for k in range(len(shifts))]
    full, binned = [], []
    for i, (e, (sx, sy)) in enumerate(zip(exps, shifts)):
        img, _ = sc.frame(e, c + sx, c + sy, -24 + 8 * i, 0, angle_deg=0.4 * (i - 2), noise=False, seed=40 + i)
        full.append(img.astype(np.float32))
        binned.append(AlignFrame(i, _bin2(img).astype(np.float32), e, time_s=8.0 * i))
    res = align_sequence(binned, AlignParams(), expected_moon_radius=116 / 2)
    ref = next(r.index for r in res if math.isinf(r.psr))
    size = sc.size
    for r in res:
        M = np.asarray(_to_full(r.transform, 0, 0, 2))
        sx, sy = c + shifts[r.index][0], c + shifts[r.index][1]
        rx, ry = c + shifts[ref][0], c + shifts[ref][1]
        x, y = M @ np.array([sx, sy, 1.0])
        # The hand-over is exact: the full-size error is twice the superpixel error (a superpixel
        # centre is at full-size (2k + 0.5)).
        bx, by = r.transform.apply_point((sx - 0.5) / 2, (sy - 0.5) / 2)
        assert x - rx == pytest.approx(2 * (bx - (rx - 0.5) / 2), abs=1e-6)
        assert y - ry == pytest.approx(2 * (by - (ry - 0.5) / 2), abs=1e-6)
        # And the alignment itself is sub-pixel at full size (here up to 0.16 px without noise,
        # 0.09 px with it: the measurement, not the hand-over).
        assert math.hypot(x - rx, y - ry) < 0.25, (r.index, x - rx, y - ry)
        # And the warped full-size frame matches the reference frame's corona (scaled by exposure).
        warped = warp_region(full[r.index], M.tolist(), (0, 0, size, size), (0, 0), order=3)
        yy, xx = np.mgrid[0:size, 0:size]
        rr = np.hypot(xx - rx, yy - ry)
        ring = (rr > 1.25 * 110) & (rr < 2.5 * 110) & np.isfinite(warped)
        a = warped[ring] / exps[r.index]
        b = full[ref][ring] / exps[ref]
        ok = (a < 0.9 / exps[r.index]) & (b < 0.9 / exps[ref])  # below clipping in both
        err = np.abs(a[ok] - b[ok]) / np.median(b[ok])
        assert np.median(err) < 0.01, (r.index, float(np.median(err)))
