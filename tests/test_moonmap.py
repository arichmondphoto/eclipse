"""Albedo map guide for the earthshine: orientation, benefit near the limb, and no leaking.

Simulated: a made-up Moon map (dark maria on brighter highlands), rendered at a known
orientation, plus glare that is steep at the limb and changes quickly round it, plus noise.
"""

import math

import numpy as np
import pytest
from scipy import ndimage

from tse_core import earthshine as ES
from tse_core import moonmap as MM

TRUTH = MM.Orientation(37.0, False, 4.0, -3.0, 1.0)
R, N = 150.0, 340


def make_map(seed=0, h=256):
    rng = np.random.default_rng(seed)
    w = 2 * h
    m = np.full((h, w), 0.6, np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    for _ in range(25):
        cy, cx = rng.uniform(0.2 * h, 0.8 * h), rng.uniform(0, w)
        s = rng.uniform(0.03, 0.09) * w
        d = np.minimum(np.abs(xx - cx), w - np.abs(xx - cx))
        m -= 0.35 * np.exp(-((d**2 + (yy - cy) ** 2) / (2 * s * s)))
    m += 0.05 * ndimage.gaussian_filter(rng.normal(0, 1, (h, w)), 2, mode="wrap")
    return np.clip(m, 0.05, 1).astype(np.float32)


@pytest.fixture(scope="module")
def sim():
    amap = make_map()
    cx = cy = N / 2 - 0.3
    surf = 1e-3 * MM.render(amap, (N, N), cx, cy, R, TRUTH)
    yy, xx = np.mgrid[0:N, 0:N]
    rr = np.hypot(xx - cx, yy - cy)
    phi = np.arctan2(yy - cy, xx - cx)
    d = np.clip(R - rr, 0, None)
    rng = np.random.default_rng(1)
    ang = ndimage.gaussian_filter1d(rng.normal(0, 1, 720), 6, mode="wrap")
    ang = np.clip(1 + 0.6 * ang / ang.std(), 0.2, None)
    a = np.interp((phi + math.pi) / (2 * math.pi) * 720, np.arange(720), ang, period=720)
    glare = 1e-3 * (12 * a * np.exp(-d / (0.02 * R)) + 3 * a * np.exp(-d / (0.1 * R)) + 0.3 * (1 + 0.3 * (xx - cx) / R))
    good = rr < R - 3
    img = np.where(good, surf + glare + rng.normal(0, 2e-5, (N, N)), 0).astype(np.float32)
    p = ES.EarthshineParams(colour_limb=False, albedo_strength=1.0)
    det0, _ = ES.remove_glare(img, good, cx, cy, R, p)
    return amap, img, good, cx, cy, rr, surf, p, det0


def _near_limb_match(det, surf, good, rr, a=15, b=40):
    band = good & (R - rr > a) & (R - rr < b)
    return float(np.corrcoef(ndimage.gaussian_filter(det, 2)[band], ndimage.gaussian_filter(surf, 2)[band])[0, 1])


def test_render_centre_and_orientation():
    """The disc centre shows the map at the libration point; rotation turns the picture."""
    h, w = 90, 180
    amap = np.zeros((h, w), np.float32)
    amap[:, w // 2 - 2 : w // 2 + 2] = 1.0  # a bright line along the 0-degree meridian
    img = MM.render(amap, (101, 101), 50, 50, 45, (0.0, False, 0.0, 0.0))
    assert img[50, 50] > 0.5 and img[10, 50] > 0.5 and img[50, 10] < 0.1  # vertical through the centre
    turned = MM.render(amap, (101, 101), 50, 50, 45, (90.0, False, 0.0, 0.0))
    assert turned[50, 10] > 0.5 and turned[10, 50] < 0.1  # now horizontal
    shifted = MM.render(amap, (101, 101), 50, 50, 45, (0.0, False, 20.0, 0.0))
    assert shifted[50, 50] < 0.1  # the meridian has moved off the centre
    assert img[0, 0] == 0  # black outside the disc


def test_orientation_is_found(sim):
    amap, img, good, cx, cy, rr, surf, p, det0 = sim
    o = MM.fit_orientation(amap, det0, good, cx, cy, R)
    assert not o.mirror
    assert abs(((o.rotation_deg - TRUTH.rotation_deg + 180) % 360) - 180) < 5
    assert abs(o.lib_lon - TRUTH.lib_lon) < 3 and abs(o.lib_lat - TRUTH.lib_lat) < 3
    assert o.correlation > 0.6


def test_map_recovers_maria_by_the_limb(sim):
    amap, img, good, cx, cy, rr, surf, p, det0 = sim
    base = _near_limb_match(det0, surf, good, rr)
    prior, o, note = ES.albedo_prior(amap, det0, good, cx, cy, R, p, TRUTH)
    assert prior is not None, note
    det, _ = ES.remove_glare_guided(img, good, cx, cy, R, p, prior)
    assert _near_limb_match(det, surf, good, rr) > base + 0.3
    # Half strength gives part of the gain.
    half = ES.EarthshineParams(colour_limb=False, albedo_strength=0.5)
    prior_h, *_ = ES.albedo_prior(amap, det0, good, cx, cy, R, half, TRUTH)
    det_h, _ = ES.remove_glare_guided(img, good, cx, cy, R, half, prior_h)
    assert base < _near_limb_match(det_h, surf, good, rr) < _near_limb_match(det, surf, good, rr)


def test_false_feature_in_the_map_hardly_leaks(sim):
    """A mare the map has and the Moon does not: at most ~1/4 of it comes through 25 px inside
    the limb, nothing deeper in."""
    amap, img, good, cx, cy, rr, surf, p, det0 = sim
    h, w = amap.shape
    lonmap = np.tile(np.linspace(-180, 180, w, endpoint=False) + 180 / w, (h, 1)).astype(np.float32)
    latmap = np.tile((90 - (np.arange(h) + 0.5) / h * 180)[:, None], (1, w)).astype(np.float32)
    lonimg = MM.render(lonmap + 500, img.shape, cx, cy, R, TRUTH) - 500
    latimg = MM.render(latmap + 500, img.shape, cx, cy, R, TRUTH) - 500
    true_t, _, k = MM.guide(amap, det0, good, cx, cy, R, TRUTH)
    d_true, _ = ES.remove_glare_guided(img, good, cx, cy, R, p, true_t)
    yy, xx = np.mgrid[0:N, 0:N]
    for depth, ang, limit in ((25, 0.0, 0.25), (45, 2.2, 0.05), (90, -1.0, 0.05)):
        px, py = cx + (R - depth) * math.cos(ang), cy + (R - depth) * math.sin(ang)
        lon0, lat0 = lonimg[int(py), int(px)], latimg[int(py), int(px)]
        my, mx = np.mgrid[0:h, 0:w]
        dl = np.radians(((mx + 0.5) / w * 360 - 180 - lon0 + 180) % 360 - 180) * np.cos(np.radians(lat0))
        db = np.radians(90 - (my + 0.5) / h * 180 - lat0)
        fake = (amap - 0.3 * np.exp(-(dl**2 + db**2) / (2 * np.radians(4) ** 2))).astype(np.float32)
        t = MM._masked_blur(MM.render(fake, img.shape, cx, cy, R, TRUTH), good, max(1.0, 0.02 * R))
        fake_t = np.where(good, k * t, 0).astype(np.float32)
        d_fake, _ = ES.remove_glare_guided(img, good, cx, cy, R, p, fake_t)
        diff = fake_t - true_t
        c = np.unravel_index(np.argmin(np.where(good, diff, 0)), diff.shape)
        dist = np.hypot(yy - c[0], xx - c[1])
        core, ring = good & (dist < 4), good & (dist > 15) & (dist < 25)
        in_map = diff[core].mean() - diff[ring].mean()
        in_result = (d_fake - d_true)[core].mean() - (d_fake - d_true)[ring].mean()
        assert abs(in_result / in_map) < limit, (depth, in_result / in_map)


def test_wrong_map_is_not_used(sim):
    amap, img, good, cx, cy, rr, surf, p, det0 = sim
    other = make_map(seed=13)  # a different made-up Moon: best chance match about 0.4
    prior, o, note = ES.albedo_prior(other, det0, good, cx, cy, R, p)
    assert prior is None and "not used" in note


def test_load_map_checks_shape(tmp_path):
    import tifffile

    tifffile.imwrite(tmp_path / "ok.tif", (np.random.default_rng(0).random((64, 128, 3)) * 65535).astype(np.uint16),
                     photometric="rgb")
    m = MM.load_map(tmp_path / "ok.tif")
    assert m.shape == (64, 128) and m.max() <= 1.5
    tifffile.imwrite(tmp_path / "bad.tif", np.zeros((64, 64), np.uint16))
    with pytest.raises(ValueError, match="twice as wide"):
        MM.load_map(tmp_path / "bad.tif")


def test_process_disc_uses_the_map(sim):
    amap, img, good, cx, cy, rr, surf, p, det0 = sim
    weight = np.where(good, 1.0, 0.0).astype(np.float32)
    q = ES.EarthshineParams(colour_limb=False, albedo_strength=1.0)
    det, shown, snr, keep, notes = ES.process_disc(img, weight, good, cx, cy, R, q, albedo=amap)
    assert any(n.startswith("albedo map: rotation") for n in notes), notes
    assert _near_limb_match(det, surf, good, rr) > _near_limb_match(det0, surf, good, rr) + 0.3
    off = ES.EarthshineParams(colour_limb=False, albedo_strength=0.0)
    det_off, *_ = ES.process_disc(img, weight, good, cx, cy, R, off, albedo=amap)
    assert np.allclose(det_off, det0, atol=1e-7)
