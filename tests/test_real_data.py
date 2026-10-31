"""Regression tests on real data: the 2024-04-08 stack from Lake Wappapello (Nikon D850, 400 mm).

``data/eclipse2024_bin8.npz`` is the linear HDR stack made by ``totality stack``, binned 8 x 8
(1036 x 690 px, Sun radius 51.7 px), stored as float16 scaled by 1e4. It keeps what made
the first enhancement runs go wrong and synthetic data did not show:

* the outermost RAW columns are not image (one reads about six times too bright);
* the twilight sky is uneven and slightly different in each colour channel, and the red
  sky level is below zero;
* the corona reaches the frame edges, so circles round the Sun leave the frame;
* dust shadows and stars.

The 2017 set is kept back as a blind test and must never be added here.
"""

from pathlib import Path

import numpy as np
import pytest

from tse_core import enhance as E

DATA = Path(__file__).parent / "data" / "eclipse2024_bin8.npz"
pytestmark = pytest.mark.skipif(not DATA.exists(), reason="real-data fixture not present")


@pytest.fixture(scope="module")
def real():
    z = np.load(DATA)
    hdr = np.ascontiguousarray(np.moveaxis(z["hdr"].astype(np.float32) * float(z["scale"]), 0, -1))
    geo = {k: float(z[k]) for k in ("SUNX", "SUNY", "SUNR", "MOONX", "MOONY", "MOONR")}
    weight = (np.abs(hdr).sum(axis=2) > 0).astype(np.float32)
    res, layers = E.enhance(hdr.copy(), weight, geo, E.EnhanceParams())
    g = E.geometry_from_header(geo)
    r, phi = E.polar(hdr.shape[:2], g)
    return res, layers, g, r, phi


def test_no_ring_where_circles_leave_the_frame(real):
    """No step in brightness at the radius where circles first leave the frame.

    Measured within 30 degrees of horizontal, where the image carries on past that radius.
    Before the fixes the NRGF stepped by 0.074 and the FNRGF by 0.022 here.
    """
    res, layers, g, r, phi = real
    h = r.shape[0]
    edge = min(g.sun_y, h - 1 - g.sun_y)
    side = np.cos(phi) ** 2 > np.cos(np.radians(30)) ** 2
    inside = side & (r > edge - 40) & (r < edge - 4)
    outside = side & (r > edge + 4) & (r < edge + 40)
    limits = {"nrgf": 0.03, "fnrgf": 0.012, "fnrgf_n": 0.012}
    for name, limit in limits.items():
        y = layers[name].mean(axis=2)
        assert abs(y[outside].mean() - y[inside].mean()) < limit, name


def test_no_thin_rings(real):
    """Averaged round each circle, the filter layers must vary smoothly with radius:
    a dust spot or a bad column at one radius shows up as a spike."""
    from scipy import ndimage

    res, layers, g, r, phi = real
    ri = r.astype(int)
    keep = np.zeros(r.shape, bool)
    keep[12:-12, 12:-12] = True
    for name in ("fnrgf", "fnrgf_n", "blend"):
        y = layers[name].mean(axis=2)
        for a0 in range(0, 360, 90):
            sec = keep & (np.degrees(phi) >= a0) & (np.degrees(phi) < a0 + 90) & (r > 2.2 * g.sun_r)
            n = np.bincount(ri[sec])
            s = np.bincount(ri[sec], y[sec])
            ok = n > 80
            p = (s / np.maximum(n, 1))[ok]
            spikes = np.abs(p - ndimage.median_filter(p, 9))
            assert spikes.max() < 0.05, (name, a0, float(spikes.max()))


def test_sky_is_dark_in_blend_and_grey_in_colour(real):
    res, layers, g, r, phi = real
    b = layers["blend"]
    far = (r > 6 * g.sun_r) & (b.mean(axis=2) > 0)
    near = (r > 1.2 * g.sun_r) & (r < 1.6 * g.sun_r)
    assert np.median(b[far].mean(axis=1)) < 0.35  # dark sky
    assert np.median(b[near].mean(axis=1)) > 0.7  # bright inner corona
    for name in ("nrgf", "fnrgf_n"):
        y = layers[name]
        out = (r > 6 * g.sun_r) & (y.mean(axis=2) > 0.05)
        cast = (y[..., 2] - y[..., 0]) / np.maximum(y.mean(axis=2), 1e-6)
        assert abs(np.median(cast[out])) < 0.03, name  # no cyan or red cast in the faint corona


def test_white_balance_is_plausible_for_a_d850(real):
    res, *_ = real
    assert res.white[0] == pytest.approx(2.0, abs=0.15)
    assert res.white[2] == pytest.approx(1.34, abs=0.1)
