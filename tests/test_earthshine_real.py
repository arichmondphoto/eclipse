"""Regression test on the real 2024 earthshine stack (Moon-aligned, linear), binned 2 x 2.

``data/earthshine2024_bin2.npz`` is ``<name>_earthshine.fit`` from 'totality earthshine'
on the Lake Wappapello frames (Nikon D850, 400 mm), binned 2 x 2 (Moon radius 218 px),
float16 scaled by 1e4. It keeps what synthetic data did not show: glare just inside
the limb 10-20x the earthshine, changing within a few degrees round the limb (inner
corona, chromosphere, a bright prominence), and far more noise in the last pixels,
where the long exposures are saturated.

The 2017 set is kept back as a blind test and must never be added here.
"""

from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage

from tse_core import earthshine as ES

DATA = Path(__file__).parent / "data" / "earthshine2024_bin2.npz"
pytestmark = pytest.mark.skipif(not DATA.exists(), reason="real-data fixture not present")


@pytest.fixture(scope="module")
def disc():
    z = np.load(DATA)
    lum = z["lum"].astype(np.float32) * float(z["scale"])
    cx, cy, r = float(z["MOONX"]), float(z["MOONY"]), float(z["MOONR"])
    yy, xx = np.mgrid[0 : lum.shape[0], 0 : lum.shape[1]]
    rr = np.hypot(xx - cx, yy - cy)
    good = (rr < r - 1.5) & (lum != 0)
    # Stand-in for the stack weights: inverse variance from the local scatter.
    hp = lum - ndimage.median_filter(lum, 3)
    var = ndimage.uniform_filter(hp**2, 9) + 1e-14
    weight = np.where(good, 1.0 / var, 0.0).astype(np.float32)
    return lum, weight, good, cx, cy, r, rr, np.arctan2(yy - cy, xx - cx)


def _spread(img, good, rr, phi, r, a, b):
    """Spread of 5-degree sector medians in the ring a..b px inside the limb."""
    band = good & (rr > r - b) & (rr < r - a)
    secs = [np.median(img[band & (phi >= t) & (phi < t + np.radians(5))]) for t in np.radians(np.arange(-180, 180, 5))]
    return float(np.std(secs))


def test_limb_glare_is_removed_round_the_limb(disc):
    lum, weight, good, cx, cy, r, rr, phi = disc
    detail, shown, snr, keep, notes = ES.process_disc(lum, weight, good, cx, cy, r, ES.EarthshineParams(smooth=0.5))
    plain = np.where(good, lum - ES.glare_surface(lum, good, cx, cy, r, 2), 0.0)
    # The lunar surface: large-scale contrast in the middle of the disc.
    big = ndimage.gaussian_filter(detail, 5)
    surface = np.percentile(big[rr < r * 0.6], 95) - np.percentile(big[rr < r * 0.6], 5)
    assert snr > 10
    for a, b in ((3, 6), (6, 12), (12, 24)):
        before = _spread(plain, good, rr, phi, r, a, b)
        after = _spread(detail, good, rr, phi, r, a, b)
        assert after < before / 4, (a, b, before, after)
    # 6-24 px (12-48 at full size) inside the limb: what is left round the limb is below the surface contrast.
    assert _spread(detail, good, rr, phi, r, 6, 24) < 0.5 * surface
    # The middle is not touched by the limb terms.
    mid = rr < r * 0.6
    assert abs(np.median(detail[mid])) < 0.2 * surface


def test_shown_layer_is_calm_at_the_limb(disc):
    """No bright or dark arcs: near the limb the shown picture stays within the range of the body."""
    lum, weight, good, cx, cy, r, rr, phi = disc
    detail, shown, snr, keep, notes = ES.process_disc(lum, weight, good, cx, cy, r, ES.EarthshineParams(smooth=0.5))
    body = good & (rr < r * 0.9)
    lo, hi = np.percentile(shown[body], [0.5, 99.5])
    rim = good & (rr > r - keep - 15) & (rr < r - keep)
    outside = (shown[rim] < lo) | (shown[rim] > hi)
    assert outside.mean() < 0.05
