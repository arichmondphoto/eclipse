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
    # Stand-in for the stack weights: inverse variance from the scatter, per ring (the real
    # weights follow the exposures used, which change with distance from the limb).
    hp = lum - ndimage.median_filter(lum, 3)
    ri = rr.astype(int)
    var_ring = ndimage.median(hp**2, ri, index=np.arange(ri.max() + 1))
    var = np.asarray(var_ring)[ri] / 0.45 + 1e-14
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
    # 6-24 px (12-48 at full size) inside the limb: what varies round the limb (left-over glare,
    # and now real maria such as Crisium and Marginis, which are kept) stays near the surface contrast.
    assert _spread(detail, good, rr, phi, r, 6, 24) < 1.25 * surface
    # The middle is not touched by the limb terms.
    mid = rr < r * 0.6
    assert abs(np.median(detail[mid])) < 0.2 * surface


def test_spikes_are_not_smoothed(disc):
    """Diffraction spikes from the bright prominence (bottom of the disc) keep their sharpness:
    no extra smoothing for the glare taken away there, only the base amount."""
    lum, weight, good, cx, cy, r, rr, phi = disc
    p = ES.EarthshineParams(smooth=0.5)
    detail, shown, snr, keep, notes = ES.process_disc(lum, weight, good, cx, cy, r, p)
    base = ES.adaptive_smooth(detail, good, np.ones_like(detail), p.smooth)
    band = good & (rr > r - 8) & (rr < r - 3)
    spot = np.unravel_index(np.argmax(np.where(band, lum, 0)), lum.shape)
    dist = np.hypot(np.arange(lum.shape[1])[None, :] - spot[1], np.arange(lum.shape[0])[:, None] - spot[0])
    zone = good & (dist > 20) & (dist < 75) & (rr < r - 20)

    def fine(x):
        return np.std((x - ndimage.gaussian_filter(x, 4))[zone])

    assert fine(shown) > 0.9 * fine(base)


def test_shown_layer_is_calm_at_the_limb(disc):
    """No bright or dark arcs: near the limb the shown picture stays within the range of the body."""
    lum, weight, good, cx, cy, r, rr, phi = disc
    detail, shown, snr, keep, notes = ES.process_disc(lum, weight, good, cx, cy, r, ES.EarthshineParams(smooth=0.5))
    body = good & (rr < r * 0.9)
    lo, hi = np.percentile(shown[body], [0.5, 99.5])
    rim = good & (rr > r - keep - 15) & (rr < r - keep)
    secs = np.array([np.median(shown[rim & (phi >= t) & (phi < t + np.radians(5))])
                     for t in np.radians(np.arange(-180, 180, 5))])
    lo, hi = np.percentile(shown[body], [5, 95])
    assert ((secs < lo) | (secs > hi)).mean() < 0.1  # a few: maria kept by the limb (Crisium)


def test_edge_fill_has_no_teeth(disc):
    """The filled edge follows the brightness round the limb smoothly: no dark or bright
    teeth from single noisy pixels copied outwards."""
    lum, weight, good, cx, cy, r, rr, phi = disc
    detail, shown, snr, keep, notes = ES.process_disc(lum, weight, good, cx, cy, r, ES.EarthshineParams(smooth=0.5))
    body = good & (rr < r * 0.9)
    lo, hi = np.percentile(shown[body], [0.5, 99.5])
    img = np.clip((shown - lo) / (hi - lo), 0, 1)
    out = ES.extend_to_limb(img, good & (rr < r - keep), cx, cy, r - keep, r + 1.0)
    band = (rr > r - keep + 1) & (rr < r)
    # Neighbours round the limb differ by far less than the grain of the body.
    shifted = ndimage.shift(out, (0, 1), order=0)
    step = np.abs(out - shifted)[band & (np.abs(np.cos(phi)) < 0.3)]
    grain = np.abs(img - ndimage.shift(img, (0, 1), order=0))[body]
    assert np.percentile(step, 95) < 0.3 * np.percentile(grain, 95)


def test_edge_fill_does_not_depend_on_the_weights(disc):
    """The real stack weights can understate the noise by the limb (this run showed a dashed
    edge where the preview, with other weights, did not): the fill depth must not rely on them.
    With Filled edge at 0 (the default since Nov 2026) the noise alone decides it: 4 px on this
    binned stack with either weights; with the old 2.5% minimum both were 2.5% of the radius."""
    lum, weight, good, cx, cy, r, rr, phi = disc
    flat = np.where(good, 1.0, 0.0).astype(np.float32)
    for fill in (0.0, 2.5):
        p = ES.EarthshineParams(smooth=0.5, edge_fill=fill)
        _, _, _, keep_flat, _ = ES.process_disc(lum, flat, good, cx, cy, r, p)
        _, _, _, keep, _ = ES.process_disc(lum, weight, good, cx, cy, r, p)
        assert keep_flat >= max(p.margin(r), fill / 100 * r) - 1e-6
        assert abs(keep_flat - keep) <= 3


def test_detail_is_kept_near_the_limb(disc):
    """Maria near the limb (Crisium at the right) must survive: the glare correction round the
    limb widens quickly with depth, so 15-60 px in (30-120 at full size) it keeps the large-scale
    lunar contrast. Measured away from the bright prominence: its fan of glare used to count as
    "detail" here (1.2 x the middle's contrast with it, 0.66 without)."""
    lum, weight, good, cx, cy, r, rr, phi = disc
    detail, *_ = ES.process_disc(lum, weight, good, cx, cy, r, ES.EarthshineParams(smooth=0.5, source_glare=True))
    big = ndimage.gaussian_filter(detail, 4)
    band = good & (rr > r - 60) & (rr < r - 15) & (np.abs(np.degrees(phi) - 74) > 25)
    body = good & (rr < r * 0.6)
    assert big[band].std() > 0.6 * big[body].std()
    # The prominence fit changes nothing away from the prominence.
    plain, *_ = ES.process_disc(lum, weight, good, cx, cy, r, ES.EarthshineParams(smooth=0.5, source_glare=False))
    assert np.allclose(detail[band], plain[band], atol=1e-9)


def test_prominence_fan_is_taken_away(disc):
    """The bright prominence at the bottom right throws a fan onto the Moon; once removed, nothing
    there stands far above the range of the lunar surface, and there is no fan of streaks."""
    lum, weight, good, cx, cy, r, rr, phi = disc
    on = ES.process_disc(lum, weight, good, cx, cy, r, ES.EarthshineParams(smooth=0.5, source_glare=True))
    off = ES.process_disc(lum, weight, good, cx, cy, r, ES.EarthshineParams(smooth=0.5, source_glare=False))
    assert any("prominence" in n for n in on[4])
    body = good & (rr < 0.8 * r)
    near = good & (rr > r - 0.12 * r) & (rr < r - 0.025 * r) & (np.abs(np.degrees(phi) - 74) < 12)

    def excess(det):
        sm = ndimage.gaussian_filter(det, 2)
        p1, p99 = np.percentile(sm[body], [1, 99])
        return float(np.percentile(np.abs(sm[near] - np.median(sm[body])), 99) / (p99 - p1))

    assert excess(on[0]) < 0.5 * excess(off[0]), (excess(on[0]), excess(off[0]))


def test_filled_edge_minimum_is_a_setting_in_percent_of_the_radius(disc):
    """The filled band by the limb is at least ``edge_fill`` % of the Moon's radius, so it scales with
    any lens and camera (2.5%: 10.9 px on the full-size 2024 Moon). At 0, the default since Nov 2026
    (Andy), the noise rule alone decides: on the full-size 2024 stack that was the last 6 px (1.4%)."""
    lum, weight, good, cx, cy, r, rr, phi = disc
    keep = {}
    for pct in (0.0, 2.5, 5.0):
        *_, keep[pct], notes = ES.process_disc(lum, weight, good, cx, cy, r, ES.EarthshineParams(edge_fill=pct))
        assert any(f"{keep[pct]:.1f} px inside the limb" in n for n in notes), notes
    assert ES.EarthshineParams().edge_fill == 0.0
    assert keep[2.5] == pytest.approx(0.025 * r, abs=0.01)
    assert keep[5.0] == pytest.approx(0.05 * r, abs=0.01)
    assert keep[0.0] < 0.025 * r
