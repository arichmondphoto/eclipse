"""Earthshine: a Moon-aligned stack recovers the lunar surface that a corona-aligned one blurs."""

import math
from pathlib import Path


import numpy as np
import pytest
from scipy import ndimage

from tse_core import earthshine as ES
from tse_core.earthshine import EarthshineParams, glare_surface, moon_aligned, stack_earthshine
from tse_core.stack import FrameInfo, Match
from tse_core.synthetic import SyntheticScene
from test_stack import mosaic_from

SIZE, SUN, RS = 400, (200.0, 200.0), 40.0
MOON_R = RS * 1.055


# The settings these method tests were written for. The test Moon is 42 px in radius, a tenth of a
# 400 mm Moon, so its maria are large against the disc: the default glare surface (order 6, chosen
# by eye on the 2024 Moon, R = 436 px) is flexible enough here to take them away with the glare.
SMALL_MOON = dict(glare_order=2, smooth=1.0, edge_smooth=3.0, brightness=0.6)


def _surface(seed=3):
    """Maria and highlands: smooth blobs, fixed to the Moon (coordinates relative to its centre)."""
    rng = np.random.default_rng(seed)
    n = 128
    tex = ndimage.gaussian_filter(rng.normal(0, 1, (n, n)), 3.0)
    tex /= tex.std()
    return tex  # sampled at (dy, dx) + 64


def _scene(moon_offsets, exposures, frame_shifts, earth=4e-4, seed=0, limb_glare=3e-3, spot=0.0):
    scene = SyntheticScene(size=SIZE, sun_radius=RS, seed=5)
    tex = _surface()
    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(float)
    mos, infos, truth = {}, [], {}
    for i, ((mdx, mdy), t, (sx, sy)) in enumerate(zip(moon_offsets, exposures, frame_shifts)):
        sun = (SUN[0] + sx, SUN[1] + sy)
        corona = scene.corona(*sun) * 1e-3
        mx, my = sun[0] + mdx, sun[1] + mdy
        r = np.hypot(xx - mx, yy - my)
        disc = np.clip(MOON_R - r + 0.5, 0, 1)
        surf = ndimage.map_coordinates(tex, [yy - my + 64, xx - mx + 64], order=1, mode="nearest")
        glare = 3e-3 * (1 + 0.4 * (xx - sun[0]) / RS + 0.2 * (yy - sun[1]) / RS)  # fixed to the Sun, smooth
        # Scattered light from the inner corona just beyond the limb: steep, stronger to one side.
        phi = np.arctan2(yy - my, xx - mx)
        limb = limb_glare * np.exp(-np.clip(MOON_R - r, 0, None) / 2.5) * (1 + 0.5 * np.cos(phi - 0.7)
                                                                     + spot * np.exp(-(((phi - 2.0 + np.pi) % (2 * np.pi) - np.pi) / 0.3) ** 2))
        moonlight = earth * (1 + 0.3 * surf) + glare + limb
        img = (corona * (1 - disc) + moonlight * disc) * t
        rng = np.random.default_rng(seed + i)
        img = rng.poisson(np.clip(img, 0, None) * 3e4) / 3e4 + rng.normal(0, 2e-4, img.shape)
        mos[i] = mosaic_from(np.minimum(img, 1.0).astype(np.float32))
        M = [[1.0, 0.0, -sx], [0.0, 1.0, -sy]]
        infos.append(FrameInfo(i, f"f{i}", t, M, mx - sx, my - sy, MOON_R))
        truth[i] = (mx - sx, my - sy)
    return mos, infos, tex


def _corr_with_truth(detail, crop, moon_xy, tex, margin=6):
    X0, Y0, w, h = crop
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    cx, cy = moon_xy[0] - X0, moon_xy[1] - Y0
    inside = np.hypot(xx - cx, yy - cy) < MOON_R - margin
    true = ndimage.map_coordinates(tex, [yy - cy + 64, xx - cx + 64], order=1, mode="nearest")
    a = ndimage.gaussian_filter(detail, 1.0)[inside]
    return float(np.corrcoef(a, true[inside])[0, 1])


@pytest.fixture(scope="module")
def run():
    moons = [(-14, -5), (-5, -2), (4, 1), (13, 5)]  # the Moon crossing the Sun during totality
    exposures = [4.0, 2.0, 4.0, 1.0]
    shifts = [(1.2, -0.7), (0.0, 0.0), (-0.9, 1.1), (0.4, 0.3)]
    mos, infos, tex = _scene(moons, exposures, shifts)
    matches = {f.index: Match(f.index, f.nominal_ratio, 0.0, f.nominal_ratio, 0.0, 0) for f in infos}
    ref = infos[1]
    return mos, infos, tex, matches, ref


def test_moon_aligned_stack_recovers_surface(run, tmp_path):
    import tifffile

    mos, infos, tex, matches, ref = run
    res, lum, detail = stack_earthshine(infos, lambda f: mos[f.index], {}, matches, (ref.moon_x, ref.moon_y), MOON_R,
                                        (0, 0, SIZE, SIZE), EarthshineParams(**SMALL_MOON), out_dir=tmp_path, name="t")
    c = _corr_with_truth(detail, res.crop, (ref.moon_x, ref.moon_y), tex)
    # (The test Moon is small, 42 px: the band by the limb where glare is fitted round the limb
    # is a fifth of it. On a real 400 mm Moon it is a few percent.)
    assert c > 0.78, c
    assert res.snr > 3
    assert max(abs(d[0]) for d in res.shifts.values()) > 8  # the Moon really was moved
    layer = tifffile.imread(res.outputs["earthshine"])
    assert layer.shape == (SIZE, SIZE, 3)
    # Black outside the disc, lunar detail inside, at the Moon frame's Moon.
    assert layer[5, 5].max() == 0
    assert layer[int(ref.moon_y), int(ref.moon_x)].max() > 0
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    p = EarthshineParams()
    outside = np.hypot(xx - ref.moon_x, yy - ref.moon_y) > MOON_R + p.limb_reach  # the real limb, then nothing
    assert layer[outside].max() == 0


def test_corona_aligned_stack_blurs_surface(run):
    """Control: without the Moon shifts the same data gives a much worse surface."""
    mos, infos, tex, matches, ref = run
    still = [FrameInfo(f.index, f.file_name, f.nominal_ratio, f.matrix, ref.moon_x, ref.moon_y, f.moon_r) for f in infos]
    res, lum, detail = stack_earthshine(still, lambda f: mos[f.index], {}, matches, (ref.moon_x, ref.moon_y), MOON_R,
                                        (0, 0, SIZE, SIZE), EarthshineParams())
    c_still = _corr_with_truth(detail, res.crop, (ref.moon_x, ref.moon_y), tex)
    assert c_still < 0.6, c_still


def test_short_exposures_skipped(run):
    mos, infos, tex, matches, ref = run
    res, *_ = stack_earthshine(infos, lambda f: mos[f.index], {}, matches, (ref.moon_x, ref.moon_y), MOON_R,
                               (0, 0, SIZE, SIZE), EarthshineParams(stops=1.5))
    assert sorted(res.frames) == [0, 1, 2]  # the 1 s frame is more than 1.5 stops below 4 s


def test_glare_surface_removes_smooth_gradient():
    h = w = 100
    yy, xx = np.mgrid[0:h, 0:w]
    inside = np.hypot(xx - 50, yy - 50) < 45
    glare = 1 + 0.01 * xx + 0.0002 * (yy - 50) ** 2
    fit = glare_surface(glare.astype(np.float32), inside, 50, 50, 45, 4)
    assert np.abs(fit - glare)[inside].max() < 1e-3


def test_moon_aligned_matrix():
    m = moon_aligned([[1, 0, 2.0], [0, 1, -1.0]], (100.0, 50.0), (103.0, 49.0))
    assert m[0][2] == pytest.approx(5.0) and m[1][2] == pytest.approx(-2.0)


def test_workflow_and_cli(tmp_path, monkeypatch, run, capsys):
    """totality earthshine: frames from the project, alignment and brightness from earlier steps."""
    import json
    from dataclasses import asdict

    from conftest import make_frames
    from tse_core import cli, rawio
    from tse_core.inventory import Inventory
    from tse_core.project import EventProject

    mos, infos, tex, matches, ref = run
    frames = make_frames(["4", "2", "4", "1"])
    by_path = {}
    for f, info in zip(frames, infos):
        f.index = info.index
        f.path = str(tmp_path / f.file_name)
        (tmp_path / f.file_name).write_bytes(b"")
        by_path[f.path] = mos[info.index]
    monkeypatch.setattr(rawio, "read_mosaic", lambda p: by_path[p])
    align = {"reference_index": 1, "image_size": [SIZE, SIZE], "sun": None, "frames": [
        {"index": i.index, "file_name": i.file_name, "matrix": i.matrix, "moon_ref": [i.moon_x, i.moon_y],
         "moon_frame": {"radius": MOON_R}} for i in infos]}
    (tmp_path / "e_align.json").write_text(json.dumps(align))
    (tmp_path / "e_stack.json").write_text(json.dumps({"crop": [0, 0, SIZE, SIZE], "moon_frame": 1, "reference": 1,
                                                       "matches": [asdict(m) for m in matches.values()]}))
    p = EventProject(name="e")
    p.set_inventory(Inventory(frames))
    p.use_lensfun = False
    p.parameters["align"] = {"file": "e_align.json"}
    p.parameters["stack"] = {"info": "e_stack.json"}
    p.save(tmp_path / "e.json")
    assert cli.main(["earthshine", str(tmp_path / "e.json"), "--lensfun-db", str(tmp_path / "none")]) == 0
    out = capsys.readouterr().out
    assert "Lunar surface detail" in out
    saved = EventProject.load(tmp_path / "e.json").parameters["earthshine"]
    assert Path(saved["earthshine"]).exists() and sorted(saved["frames"]) == [0, 1, 2, 3]
    # Andy, Nov 2026: data files in their own folders inside the project folder, pictures beside it;
    # the align and stack files at the top (an existing project's layout) are still found.
    for k in ("linear", "rgb", "weight", "moon_surface"):
        assert Path(saved[k]).parent == tmp_path / "fits", (k, saved[k])
    assert Path(saved["info"]).parent == tmp_path / "json"
    assert Path(saved["earthshine"]).parent == tmp_path


def test_composite_fills_disc_and_keeps_corona():
    from tse_core.earthshine import composite

    base = np.full((10, 10, 3), 0.7, np.float32)
    base[3:7, 3:7] = 0.0  # black lunar disc in the corona layer
    earth = np.zeros_like(base)
    earth[3:7, 3:7] = 0.3
    for mode in ("screen", "lighten", "add"):
        out = composite(base, earth, 1.0, mode)
        assert np.allclose(out[0, 0], 0.7) and np.allclose(out[4, 4], 0.3), mode
    assert np.allclose(composite(base, earth, 2.0)[4, 4], 0.6)
    with pytest.raises(ValueError):
        composite(base, earth, 1.0, "multiply")




def test_limb_glare_removed_and_layer_meets_the_corona(run, tmp_path):
    """The steep glare just inside the limb is taken out (no bright rim), and the layer
    reaches the corona layers' black disc (moon_r + 1.5 px) with no dark ring between."""
    import tifffile

    from tse_core.earthshine import composite

    mos, infos, tex, matches, ref = run
    # Evenly lit, so a dark ring would stand out. The scene's Moon is tiny (42 px radius) but its
    # limb is blurred like a big one's, so the margin is set as for one (the scaled one is 2 px).
    p = EarthshineParams(margin_px=3.0, **SMALL_MOON)
    res, lum, detail = stack_earthshine(infos, lambda f: mos[f.index], {}, matches, (ref.moon_x, ref.moon_y), MOON_R,
                                        (0, 0, SIZE, SIZE), p, out_dir=tmp_path, name="t")
    X0, Y0, w, h = res.crop
    yy, xx = np.mgrid[0:h, 0:w]
    rr = np.hypot(xx - (ref.moon_x - X0), yy - (ref.moon_y - Y0))
    sm = ndimage.gaussian_filter(detail, 2.0)
    edge = (rr > MOON_R - 7) & (rr < MOON_R - p.margin(MOON_R) - 1)
    middle = rr < MOON_R * 0.6
    assert abs(np.median(sm[edge]) - np.median(sm[middle])) < 3e-5
    _, _, plain = stack_earthshine(infos, lambda f: mos[f.index], {}, matches, (ref.moon_x, ref.moon_y), MOON_R,
                                   (0, 0, SIZE, SIZE), EarthshineParams(limb_glare=False, margin_px=3.0, **SMALL_MOON))
    sp = ndimage.gaussian_filter(plain, 2.0)
    assert np.median(sp[edge]) - np.median(sp[middle]) > 6e-5  # the test does contain a rim to remove
    # The layer is filled right out to the corona's edge and soft only beyond it.
    layer = tifffile.imread(res.outputs["earthshine"]).mean(axis=2) / 65535
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    rf = np.hypot(xx - ref.moon_x, yy - ref.moon_y)
    disc = rf < MOON_R * 0.9
    ring = (rf > MOON_R - 4) & (rf < MOON_R + 1.0)
    assert np.median(layer[ring]) > 0.5 * np.median(layer[disc])
    assert layer[rf > MOON_R + p.limb_reach].max() == 0
    # Into a corona layer that is black out to moon_r + 1.5: no dark ring at the limb.
    corona = np.where(rf > MOON_R + 1.5, 0.8, 0.0)[..., None].repeat(3, axis=2)
    out = composite(corona, np.repeat(layer[..., None], 3, axis=2), 1.0, "screen").mean(axis=2)
    for k in np.arange(MOON_R - 6, MOON_R + 3, 0.5):
        band = (rf >= k) & (rf < k + 0.5)
        assert np.median(out[band]) > 0.4 * np.median(out[disc]), k


def test_extend_to_limb_carries_values_outward():
    from tse_core.earthshine import extend_to_limb

    n = 60
    yy, xx = np.mgrid[0:n, 0:n]
    rr = np.hypot(xx - 30, yy - 30)
    img = (0.5 + 0.2 * np.cos(np.arctan2(yy - 30, xx - 30))).astype(np.float32)
    good = rr < 20
    out = extend_to_limb(np.where(good, img, 0), good, 30, 30, 20, 24)
    band = (rr > 20.5) & (rr < 24)
    assert np.abs(out[band] - img[band]).max() < 0.03
    assert out[rr > 25.5].max() == 0


def test_limb_glare_that_changes_quickly_round_the_limb():
    """Glare from a prominence or a bright patch of chromosphere: a few degrees wide.
    Harmonics alone leave it as a bright patch on the rim; the per-angle terms take it out."""
    moons = [(-14, -5), (-5, -2), (4, 1), (13, 5)]
    mos, infos, tex = _scene(moons, [4.0, 2.0, 4.0, 1.0], [(1.2, -0.7), (0.0, 0.0), (-0.9, 1.1), (0.4, 0.3)], spot=4.0)
    matches = {f.index: Match(f.index, f.nominal_ratio, 0.0, f.nominal_ratio, 0.0, 0) for f in infos}
    ref = infos[1]
    res, lum, detail = stack_earthshine(infos, lambda f: mos[f.index], {}, matches, (ref.moon_x, ref.moon_y), MOON_R,
                                        (0, 0, SIZE, SIZE), EarthshineParams(**SMALL_MOON))
    X0, Y0, w, h = res.crop
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = ref.moon_x - X0, ref.moon_y - Y0
    rr = np.hypot(xx - cx, yy - cy)
    phi = np.arctan2(yy - cy, xx - cx)
    rim = (rr > MOON_R - 7) & (rr < MOON_R - 3.5)
    at_spot = rim & (np.abs(((phi - 2.0 + np.pi) % (2 * np.pi)) - np.pi) < 0.1)
    away = rim & (np.abs(((phi - 2.0 + np.pi) % (2 * np.pi)) - np.pi) > 0.6)
    sm = ndimage.gaussian_filter(detail, 1.0)
    assert abs(np.median(sm[at_spot]) - np.median(sm[away])) < 1.5e-4  # earthshine itself is 4e-4
    assert _corr_with_truth(detail, res.crop, (ref.moon_x, ref.moon_y), tex) > 0.78


def test_the_layer_is_evenly_lit(run, tmp_path):
    """No vignette is drawn on the Moon: the edge of the lunar surface is as bright as the middle
    (the old Darker towards the limb setting made it 40% of the middle by default)."""
    import tifffile

    mos, infos, tex, matches, ref = run
    res, *_ = stack_earthshine(infos, lambda f: mos[f.index], {}, matches, (ref.moon_x, ref.moon_y), MOON_R,
                               (0, 0, SIZE, SIZE), EarthshineParams(keep_glare=0.0, limb_reach=0.0),
                               out_dir=tmp_path, name="d")
    layer = tifffile.imread(res.outputs["earthshine"]).mean(axis=2)
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    rf = np.hypot(xx - ref.moon_x, yy - ref.moon_y) / MOON_R
    # The maria and highlands differ from place to place, so take them out first: fit the layer as
    # the true surface plus a term that grows towards the limb, and that term must be about zero.
    true = ndimage.map_coordinates(tex, [yy - ref.moon_y + 64, xx - ref.moon_x + 64], order=1, mode="nearest")
    m = rf < 0.9
    A = np.column_stack([np.ones(m.sum()), true[m], rf[m] ** 2])
    (a, _b, c), *_ = np.linalg.lstsq(A, layer[m], rcond=None)
    # With the old default vignette c was -0.75 a. What is left (about -0.2 a) is the glare removal
    # on this small 42 px test Moon, whose steep limb glare takes some of the surface with it.
    assert abs(c) < 0.35 * a, (a, c)

def test_channel_choice_avoids_red_glare():
    """Glare that is mostly red (chromosphere) by the limb: the automatic choice leaves red out."""
    from tse_core.earthshine import EarthshineParams, choose_channel

    n, r = 260, 110.0
    yy, xx = np.mgrid[0:n, 0:n]
    rr = np.hypot(xx - 130, yy - 130)
    phi = np.arctan2(yy - 130, xx - 130)
    good = rr < r - 3
    surface = ndimage.gaussian_filter(np.random.default_rng(1).normal(0, 1, (n, n)), 6)
    surface = 1e-3 * (1 + 0.1 * surface / surface.std())
    d = np.clip(r - rr, 0, None)
    pattern = 1 + 0.8 * np.cos(9 * phi) + 0.5 * np.sin(23 * phi)
    red_glare = 2e-3 * np.exp(-d / 4) * pattern + 1e-3 * np.exp(-d / 12) * pattern
    rng = np.random.default_rng(2)
    rgb = np.stack([surface + red_glare + 0.2 * red_glare.mean(), surface, surface]) + rng.normal(0, 2e-6, (3, n, n))
    rgb = np.where(good, rgb, 0).astype(np.float32)
    name, scores = choose_channel(rgb, good, 130, 130, r, EarthshineParams())
    assert name in ("green", "green+blue"), scores
    assert scores["all"][0] > 2 * scores[name][0]


def test_limb_falloff_keeps_a_mare_by_the_limb():
    """A fading-inward glare is taken out; a dark mare band near the limb (a dip, which no
    fall-off can make) is left."""
    from tse_core.earthshine import limb_falloff

    n, r = 400, 180.0
    yy, xx = np.mgrid[0:n, 0:n]
    rr = np.hypot(xx - 200, yy - 200)
    phi = np.arctan2(yy - 200, xx - 200)
    good = rr < r - 3
    d = r - rr
    glare = 0.5 * np.exp(-d / 3) * (1 + 0.8 * np.cos(3 * phi)) + 0.1 * np.exp(-d / 10)
    mare = -0.05 * np.exp(-(((d - 20) / 5) ** 2)) * (np.cos(phi) > 0.5)  # dark band 20 px in, on one side
    img = np.where(good, glare + mare, 0).astype(np.float32)
    fit = limb_falloff(img, good, 200, 200, r)
    left = img - fit
    band = good & (d > 16) & (d < 24)
    on, off = band & (np.cos(phi) > 0.7), band & (np.cos(phi) < 0.3)
    assert np.median(left[on]) - np.median(left[off]) < -0.03  # the mare is still there
    rim = good & (d < 10) & (np.cos(phi) < 0.3)
    assert np.abs(left[rim]).mean() < 0.15 * glare[rim].mean()  # most glare taken away


def _limb_scene(bumps=True, n=300, r=110.0, seed=4):
    """A Moon-aligned linear stack with a bumpy real limb, steep glare inside it and corona outside."""
    yy, xx = np.mgrid[0:n, 0:n]
    cx = cy = n / 2 - 0.4
    rr = np.hypot(xx - cx, yy - cy)
    phi = np.arctan2(yy - cy, xx - cx)
    true_r = r + (1.5 * np.cos(17 * phi) + 1.0 * np.sin(29 * phi) if bumps else 0.0)  # mountains and valleys
    d = true_r - rr
    surface = 1e-3 * (1 + 0.15 * ndimage.gaussian_filter(np.random.default_rng(seed).normal(0, 1, (n, n)), 5) * 6)
    glare = 3e-3 * np.exp(-np.clip(d, 0, None) / 4) + 1e-3 * np.exp(-np.clip(d, 0, None) / 25)
    corona = 0.05 * np.exp(-np.clip(-d, 0, None) / 30)
    lin = np.where(d > 0, surface + glare, corona)
    lin = ndimage.gaussian_filter(lin, 0.8)  # lens and seeing
    return lin.astype(np.float32), cx, cy, r, rr, phi, true_r


def test_layer_follows_the_real_limb_and_glows():
    from tse_core.earthshine import EarthshineParams, compose_layer, remove_glare

    lin, cx, cy, r, rr, phi, true_r = _limb_scene()
    p = EarthshineParams(keep_glare=0.5, limb_soften=0.0, **SMALL_MOON)
    good = rr < r - p.margin(r) - 2
    weight = (rr < r + p.limb_reach + 2).astype(np.float32)
    detail, glare = remove_glare(lin, good, cx, cy, r, EarthshineParams(limb_shape=False, **SMALL_MOON))
    layer = compose_layer(detail, good, weight, cx, cy, r, p, keep_from=6.0, linear=lin, glare=glare)
    # The edge (the steepest rise from the dark Moon to the corona) follows the bumps.
    edges = []
    for a in np.linspace(-np.pi, np.pi, 180, endpoint=False):
        rs = np.arange(r - 8, r + 4, 0.1)
        v = ndimage.map_coordinates(layer, [cy + rs * np.sin(a), cx + rs * np.cos(a)], order=1)
        edges.append(rs[np.argmax(np.gradient(ndimage.gaussian_filter1d(v, 3)))])
    want = r + 1.5 * np.cos(17 * np.linspace(-np.pi, np.pi, 180, endpoint=False)) \
        + 1.0 * np.sin(29 * np.linspace(-np.pi, np.pi, 180, endpoint=False))
    assert np.corrcoef(edges, want)[0, 1] > 0.75  # a drawn circle: about -0.5 (no relation)
    # A glow: brighter near the limb than in the middle, fading inward; and the middle is the surface.
    mid = np.median(layer[rr < 0.5 * r])
    near = np.median(layer[(rr > r - 10) & (rr < r - 4)])
    assert near > mid * 1.15
    off = compose_layer(detail, good, weight, cx, cy, r, EarthshineParams(keep_glare=0.0, limb_soften=0.0, **SMALL_MOON), keep_from=6.0, linear=lin, glare=glare)
    assert np.median(off[(rr > r - 40) & (rr < r - 25)]) < np.median(layer[(rr > r - 40) & (rr < r - 25)])
    # Nothing beyond the reach of the stack.
    assert layer[rr > r + p.limb_reach].max() == 0


def test_filled_edge_gets_grain():
    from tse_core.earthshine import EarthshineParams, compose_layer

    n, r = 200, 80.0
    yy, xx = np.mgrid[0:n, 0:n]
    rr = np.hypot(xx - 100, yy - 100)
    good = rr < r - 3
    shown = np.where(good, np.random.default_rng(0).normal(0, 1, (n, n)), 0).astype(np.float32)
    shown = ndimage.gaussian_filter(shown, 0.8)
    band = (rr > r - 8) & (rr < r - 1)
    for grain, expect in ((True, True), (False, False)):
        p = EarthshineParams(edge_grain=grain, limb_reach=0.0)
        out = compose_layer(shown, good, good.astype(np.float32), 100, 100, r, p, keep_from=10.0)
        hp = out - ndimage.gaussian_filter(out, 2)
        inner = good & (rr < r - 15) & (rr > r - 30)
        ratio = hp[band].std() / hp[inner].std()
        assert (ratio > 0.5) == expect, ratio


def test_composite_matches_the_limb_to_the_corona():
    """Round the limb the layer's light is brought to the corona's level just outside, and over
    the corona layers' disc edge the brighter one is kept: no bright ring, no dark ring."""
    from tse_core.earthshine import composite

    n, r = 240, 90.0
    yy, xx = np.mgrid[0:n, 0:n]
    rr = np.hypot(xx - 120, yy - 120)
    corona = np.where(rr > r + 1.5, 0.75 - 0.05 * np.exp(-(rr - r - 1.5) / 2), 0.0)  # a little dimmer at its edge
    earth = np.where(rr < r, 0.3, 0.0) + np.where(np.abs(rr - r - 1) < 3, 1.0, 0.0) * np.clip(r + 6 - rr, 0, 1)
    earth = np.clip(earth, 0, 1)
    base = np.repeat(corona[..., None], 3, axis=2).astype(np.float32)
    e = np.repeat(earth[..., None], 3, axis=2).astype(np.float32)
    plain = composite(base, e, 1.0, "screen").mean(axis=2)
    matched = composite(base, e, 1.0, "screen", moon=(120, 120, r), limb_px=1.5, reach=6.0).mean(axis=2)
    ring = (rr > r) & (rr < r + 4)
    outside = (rr > r + 8) & (rr < r + 12)
    assert np.median(plain[ring]) > np.median(plain[outside]) + 0.15  # without matching: a bright ring
    assert np.median(matched[ring]) < np.median(matched[outside]) + 0.05
    prof = [np.median(matched[(rr >= r + d) & (rr < r + d + 1)]) for d in range(0, 10)]
    assert min(prof) > 0.6  # and no dark gap where the layers meet


def test_glare_far_above_the_surface_is_shown_as_recorded():
    """Glare far brighter than any lunar surface (by a big prominence, with its diffraction spikes)
    is shown as the data has it, not replaced by a blur of its surroundings. Andy, Nov 2026, on the
    2024 Moon: the blur made a faceted grey blob with a dark wedge where the glare and spikes are;
    he wants "that area represented as close to what the data holds as possible", nothing inferred.
    Here a patch of stripes 1.10 / 1.25 x the brightest surface (as the fan's spikes): it must stay
    bright, and the stripes must stay stripes."""
    from tse_core.earthshine import EarthshineParams, compose_layer, moon_surface

    n, r = 200, 80.0
    yy, xx = np.mgrid[0:n, 0:n]
    rr = np.hypot(xx - 100, yy - 100)
    good = rr < r - 3
    shown = np.random.default_rng(0).uniform(0, 1, (n, n)).astype(np.float32)  # the lunar surface, 0..1
    patch = (np.abs(yy - 44) < 4) & (np.abs(xx - 100) < 4)  # 64 px: too few to move the stretch
    stripe = patch & (xx % 2 == 0)
    shown = np.where(stripe, 1.25, np.where(patch, 1.10, shown))
    shown = np.where(good, shown, 0).astype(np.float32)
    p = EarthshineParams(limb_reach=0.0, edge_grain=False, brightness=1.0)
    layer = compose_layer(shown, good, good.astype(np.float32), 100, 100, r, p, keep_from=5.0)
    surf, _ = moon_surface(shown, good, np.zeros_like(shown), 100, 100, r, p, 5.0)
    for name, out in (("layer", layer), ("surface", surf[..., 1])):
        assert out[patch].mean() > 0.9, (name, out[patch].mean())  # bright, as recorded (a blur gave ~0.5)
        assert out[stripe].mean() > out[patch & ~stripe].mean() + 0.01, name  # the stripes are still there


def test_layer_keeps_the_colour_at_the_limb():
    """A red prominence at the limb: on the Moon side the glow is red too, not grey; the
    surface far inside takes the given tint."""
    from tse_core.earthshine import EarthshineParams, compose_layer, remove_glare

    lin, cx, cy, r, rr, phi, true_r = _limb_scene(bumps=False)
    prom = np.exp(-((phi - 0.5) ** 2) / (2 * 0.06**2)) * np.exp(-np.clip(np.abs(rr - r), 0, None) / 3)
    rgb = np.stack([lin + 0.05 * prom, lin, lin]).astype(np.float32)  # red light at angle 0.5 rad
    lum = rgb.mean(axis=0)
    good = rr < r - 5
    weight = (rr < r + 8).astype(np.float32)
    detail, glare = remove_glare(lum, good, cx, cy, r, EarthshineParams(limb_shape=False, **SMALL_MOON))
    # A clear glow (the defaults before Nov 2026), so its colour can be measured.
    p = EarthshineParams(keep_glare=0.5, limb_soften=1.5, **SMALL_MOON)
    tint = np.array([0.9, 1.0, 1.1], np.float32)
    layer = compose_layer(detail, good, weight, cx, cy, r, p, keep_from=6.0, linear=lum, glare=glare,
                          rgb=rgb, white=(1.0, 1.0, 1.0), tint=tint)
    assert layer.ndim == 3
    inside_prom = (rr > r - 4) & (rr < r) & (np.abs(phi - 0.5) < 0.05)
    inside_other = (rr > r - 4) & (rr < r) & (np.abs(phi + 2.0) < 0.05)
    red = lambda m: np.median(layer[m][:, 0] / np.maximum(layer[m][:, 1], 1e-6))
    assert red(inside_prom) > 1.3 and red(inside_other) < 1.1
    middle = rr < 0.4 * r
    rb = np.median(layer[middle][:, 2] / layer[middle][:, 0])
    assert rb == pytest.approx(1.1 / 0.9, rel=0.1)
    grey = compose_layer(detail, good, weight, cx, cy, r, p, keep_from=6.0, linear=lum, glare=glare)
    assert grey.ndim == 2


def _moon_project(tmp_path, holes=True, es_blur=0.0, cover=3.0, scale=1):
    """A small HDR stack and earthshine outputs, as the Stack and Earthshine steps write them.
    ``es_blur``: the earthshine stack's limb blurred further (Gaussian sigma, px); ``cover``: over
    how many px outside the limb the stack's weights rise (each frame's Moon covers some);
    ``scale``: the whole scene that many times larger (the Moon's radius 80 px x ``scale``)."""
    import json

    from tse_core.fits import write_fits
    from tse_core.project import EventProject

    n, r, c = 320 * scale, 80.0 * scale, 160.0 * scale
    yy, xx = np.mgrid[0:n, 0:n]
    rr = np.hypot(xx - c, yy - c)
    phi = np.arctan2(yy - c, xx - c)
    d = r - rr
    tex = ndimage.gaussian_filter(np.random.default_rng(3).normal(0, 1, (n, n)), 4)
    light = np.where(d > 0, 1e-3 * (1 + 0.3 * tex / tex.std()) + 4e-3 * np.exp(-d / 4),
                     0.06 * np.exp(-np.clip(-d, 0, None) / (25 * scale)) * (1 + 0.3 * np.cos(4 * phi)))
    light = ndimage.gaussian_filter(light, 0.7)
    prom = np.exp(-(phi**2) / (2 * 0.05**2)) * np.exp(-np.abs(d) / 3)  # a red prominence at angle 0
    rgb = np.stack([light * (1 + 3 * prom), light, light * 0.95]).astype(np.float32)
    hdr = rgb.copy()
    rng = np.random.default_rng(4)
    hdr[:, d > 0] *= 1 + 0.5 * rng.normal(0, 1, (d > 0).sum())  # one noisy frame inside the disc
    hdr[:, d <= 0] += 2e-4 * rng.normal(0, 1, (3, (d <= 0).sum()))  # many frames on the corona
    if es_blur > 0:  # long Moon-aligned exposures: the limb softer than in the stack
        rgb = np.stack([ndimage.gaussian_filter(ch, es_blur) for ch in rgb]).astype(np.float32)
    rgb = rgb + 3e-5 * rng.normal(0, 1, rgb.shape).astype(np.float32)  # the earthshine stack's noise
    # Real weights: many frames on the corona, one on the Moon; the earthshine stack's weights
    # are in units of their own.
    # Just outside the limb fewer frames go in (each frame's Moon covers some): over 3 px here.
    weight = np.where(d > 0, 0.02, np.clip(0.1 + 0.9 * (-d) / cover, 0.1, 1.0)).astype(np.float32)
    if holes:
        hole = (np.abs(d - 0.5) < 1) & (np.sin(9 * phi) > 0.6)  # saturated in every frame, by the limb
        hdr[:, hole] = 0
        weight[hole] = 0
    head = {"MOONX": c, "MOONY": c, "MOONR": r, "SUNX": c + 3, "SUNY": c - 2, "SUNR": r / 1.05}
    write_fits(tmp_path / "e_hdr.fit", hdr, head)
    write_fits(tmp_path / "e_weight.fit", weight)
    # Earthshine crop at (40, 30): the Moon-aligned stack (clean), its weights and the surface.
    X0, Y0, m = 40 * scale, 30 * scale, 240 * scale
    crop = (slice(Y0, Y0 + m), slice(X0, X0 + m))
    es_w = np.where(rr < r + 8, 5e4, 0.0)[crop].astype(np.float32)
    write_fits(tmp_path / "e_earthshine_rgb.fit", rgb[(slice(None),) + crop],
               {"MOONX": c - X0, "MOONY": c - Y0, "MOONR": r, "CROPX0": X0, "CROPY0": Y0})
    write_fits(tmp_path / "e_earthshine_weight.fit", es_w)
    surf = np.where(rr < r, 0.25 * (1 + 0.5 * tex / tex.std()), 0)[crop]
    vis = np.clip(r + 0.5 - rr, 0, 1)[crop]
    write_fits(tmp_path / "e_moon_surface.fit", np.stack([surf, surf, surf, vis]).astype(np.float32))
    (tmp_path / "e_earthshine.json").write_text(json.dumps({"crop": [X0, Y0, m, m], "canvas": [0, 0, n, n]}))
    p = EventProject(name="e")
    p.parameters["stack"] = {"hdr": "e_hdr.fit", "weight": "e_weight.fit"}
    p.parameters["earthshine"] = {"info": "e_earthshine.json", "rgb": "e_earthshine_rgb.fit",
                                  "weight": "e_earthshine_weight.fit", "moon_surface": "e_moon_surface.fit"}
    p.save(tmp_path / "e.json")
    return tmp_path / "e.json", rr, phi, r


def test_moon_and_corona_stretched_together_have_no_seam(tmp_path):
    """The final picture with the Moon put into the stack: across the limb the brightness changes
    smoothly (no line), there are no black specks where the stack had holes, and a red prominence
    has the same colour on both sides of the limb."""
    import tifffile

    from tse_core import workflow as W
    from tse_core.enhance import EnhanceParams

    proj, rr, phi, r = _moon_project(tmp_path)
    params = EnhanceParams(layers=(), blend={"natural": 1.0})
    out = W.final_with_moon(proj, "blend", 1.0, enhance_params=params, output_file=str(tmp_path / "f.tif"))
    img = tifffile.imread(out).astype(np.float32) / 65535
    lum = img.mean(axis=2)
    # Reference: the same light stretched as one image (no join at all), same surface on top.
    from tse_core.earthshine import merge_moon  # noqa: F401  (the path under test)
    from tse_core.enhance import MoonInput, enhance
    from tse_core.fits import read_fits

    clean, head = read_fits(tmp_path / "e_earthshine_rgb.fit")
    full = np.zeros((3,) + rr.shape, np.float32)
    full[:, 30:270, 40:280] = clean
    hdr_ref, h0 = read_fits(tmp_path / "e_hdr.fit")
    outside = rr >= r + 8
    full[:, outside] = hdr_ref[:, outside]
    md, _ = read_fits(tmp_path / "e_moon_surface.fit")
    mask = rr < r + params.moon_margin_px + 1.0
    mi = MoonInput(mask=mask, x0=40, y0=30, surface=np.moveaxis(md[:3], 0, -1), visibility=md[3])
    _, col = enhance(np.ascontiguousarray(np.moveaxis(full, 0, -1)), np.ones(rr.shape, np.float32), h0, params,
                     moon=mi, keep={"blend"})
    ref = col["blend"]
    band = (rr > r - 4) & (rr < r + 5)
    diff = np.abs(img - ref).mean(axis=2)
    assert np.percentile(diff[band], 99) < 0.03, np.percentile(diff[band], 99)  # the join does not show
    # No line just outside the limb (where the two stacks hand over): each 1 px ring matches.
    for a0 in (0, 1, 2):
        m = (rr > r + a0) & (rr < r + a0 + 1)
        assert abs(float(img[m].mean() - ref[m].mean())) < 0.01, (a0, img[m].mean(), ref[m].mean())
    # No dark specks where the stack had holes by the limb.
    ring = (rr > r - 1) & (rr < r + 2.5)
    assert (ref.mean(axis=2) - lum)[ring].max() < 0.08
    # The prominence: red on the Moon side as on the corona side.
    def redness(im, m):
        return np.median(im[m][:, 0] / np.maximum(im[m][:, 1], 1e-3))
    at = np.abs(phi) < 0.03
    for m in (at & (rr > r - 2) & (rr < r), at & (rr > r + 2) & (rr < r + 4)):  # Moon side, corona side
        assert redness(img, m) > 1.05
        assert abs(redness(img, m) - redness(ref, m)) < 0.1 * redness(ref, m)  # as if drawn as one image
    # The lunar surface is in the middle.
    assert np.median(lum[rr < 0.5 * r]) > 0.1
    prev = W.final_with_moon(proj, "blend", 1.0, enhance_params=params, preview_side=160)
    assert prev.shape[0] == 160


def test_surface_steps_back_where_the_moon_is_bright(tmp_path):
    """A bright prominence spilling over the limb keeps its colour: the lunar surface added on
    top does not wash it out to white."""
    from tse_core import workflow as W
    from tse_core.enhance import EnhanceParams
    from tse_core.fits import read_fits, write_fits

    proj, rr, phi, r = _moon_project(tmp_path, holes=False)
    md, _ = read_fits(tmp_path / "e_moon_surface.fit")
    md[:3] = np.where(md[3] > 0, 0.6, 0.0)  # a bright, even surface right up to the limb
    write_fits(tmp_path / "e_moon_surface.fit", md)
    img = W.final_with_moon(proj, "blend", 1.0, enhance_params=EnhanceParams(layers=(), blend={"natural": 1.0}),
                            preview_side=10_000)
    at = (np.abs(phi) < 0.03) & (rr > r - 2) & (rr < r)
    red = np.median(img[at][:, 0] / np.maximum(img[at][:, 1], 1e-3))
    assert red > 1.05  # still red, not washed to white


def test_saturated_prominence_core_is_not_white(tmp_path):
    """By a bright prominence the earthshine frames are saturated (little or no weight) and the
    stack's own one-frame Moon is clipped white: neither is used there; the gap is filled from
    the reliable light around it, so the prominence stays pink across the limb."""
    from tse_core import workflow as W
    from tse_core.enhance import EnhanceParams
    from tse_core.fits import read_fits, write_fits

    proj, rr, phi, r = _moon_project(tmp_path, holes=False)
    core = (np.abs(phi) < 0.04) & (rr > r - 4) & (rr < r + 1.5)  # as on the 2024 data: white to +1 px
    hdr, head = read_fits(tmp_path / "e_hdr.fit")
    hdr[:, core] = 5.0  # clipped: equal in every channel, very bright
    write_fits(tmp_path / "e_hdr.fit", hdr, head)
    es, eh = read_fits(tmp_path / "e_earthshine_rgb.fit")
    ew, _ = read_fits(tmp_path / "e_earthshine_weight.fit")
    core_c = core[30:270, 40:280]
    es[:, core_c] = 4.0  # the few frames left are near full well: grey
    ew = np.where(core_c, 0.01 * ew.max(), ew).astype(np.float32)
    write_fits(tmp_path / "e_earthshine_rgb.fit", es, eh)
    write_fits(tmp_path / "e_earthshine_weight.fit", ew)
    img = W.final_with_moon(proj, "blend", 1.0, enhance_params=EnhanceParams(layers=(), blend={"natural": 1.0}),
                            preview_side=10_000, join="rebuilt")
    m = core & (rr < r)
    red = np.median(img[m][:, 0] / np.maximum(img[m][:, 1], 1e-3))
    assert red > 1.1, red


def test_prominence_fan_is_removed_and_maria_kept():
    """A bright prominence on the limb throws a fan of glare with diffraction spikes onto the
    Moon. It is found, fitted from the prominence and taken away; a mare inside the fan stays."""
    n, r = 400, 170.0
    cx = cy = n / 2 - 0.3
    yy, xx = np.mgrid[0:n, 0:n]
    rr = np.hypot(xx - cx, yy - cy)
    good = rr < r - 3
    ang = math.radians(60.0)
    sx, sy = cx + r * math.cos(ang), cy + r * math.sin(ang)
    rho = np.maximum(np.hypot(xx - sx, yy - sy), 2.0)
    th = np.arctan2(yy - sy, xx - sx)
    spikes = 1 + 0.8 * np.cos(6 * th) ** 8  # six diffraction spikes
    fan = 4e-3 * spikes * (rho / 4) ** -1.2 * np.exp(-rho / 30)  # fades within ~0.3 radius
    mare_c = (cx + 135 * math.cos(ang), cy + 135 * math.sin(ang))  # 35 px in, inside the fan
    mare = -6e-5 * np.exp(-((xx - mare_c[0]) ** 2 + (yy - mare_c[1]) ** 2) / (2 * 12.0**2))
    rng = np.random.default_rng(5)
    detail = np.where(good, fan + mare + 2e-5 * rng.normal(0, 1, (n, n)), 0).astype(np.float32)
    g, angles = ES.source_glare(detail, good, cx, cy, r, 3.0)
    assert len(angles) == 1 and abs(angles[0] - 60.0) < 2.5, angles
    left = detail - g
    zone = good & (rho > 12) & (rho < 40)
    before = np.std(ndimage.gaussian_filter(np.where(good, fan, 0), 2)[zone])
    after = np.std(ndimage.gaussian_filter(left - mare, 2)[zone])
    assert after < 0.15 * before, (after, before)
    # The mare survives (its depth at the centre, against its surroundings 30-40 px away).
    dm = np.hypot(xx - mare_c[0], yy - mare_c[1])
    depth = left[dm < 4].mean() - left[(dm > 30) & (dm < 40) & good].mean()
    assert depth < -0.8 * 6e-5 * (1 - math.exp(-35**2 / 288)), depth  # at least 80% of it
    # No source: nothing found, nothing taken away.
    quiet = np.where(good, mare + 2e-5 * rng.normal(0, 1, (n, n)), 0).astype(np.float32)
    g0, a0 = ES.source_glare(quiet, good, cx, cy, r, 3.0)
    assert a0 == [] and not g0.any()


def test_light_by_a_bright_prominence_is_smooth_round_the_limb(tmp_path):
    """By a bright prominence ever fewer earthshine frames are usable towards it, and the last
    ones are near full well (colour and level off). The merged light there must not be ragged:
    it is interpolated along each radius, not patched from neighbours. Round the limb it varies
    from pixel to pixel little more than the true light does (the older patching: 4-8x)."""
    from tse_core.earthshine import merge_moon
    from tse_core.fits import read_fits

    proj, rr, phi, r = _moon_project(tmp_path, holes=False)
    hdr, head = read_fits(tmp_path / "e_hdr.fit")
    wt, _ = read_fits(tmp_path / "e_weight.fit")
    ew, _ = read_fits(tmp_path / "e_earthshine_weight.fit")
    es, eh = read_fits(tmp_path / "e_earthshine_rgb.fit")
    truth = es.copy()
    crop = (slice(30, 270), slice(40, 280))
    rc, pc = rr[crop], phi[crop]
    zone = (np.abs(pc) < 0.12) & (rc > r - 8) & (rc < r + 1)
    rng = np.random.default_rng(7)
    dist = np.hypot(rc * np.cos(pc) - r, rc * np.sin(pc))[zone]
    f = np.clip((0.02 + 0.6 * np.clip(dist / 10, 0, 1)) * rng.uniform(0.8, 1.2, zone.sum()), 0, 1)
    ew = ew.copy()
    ew[zone] *= f
    es = es.copy()
    bad = zone.copy()
    bad[zone] = f < 0.3
    es[:, bad] = es[:, bad].mean(axis=0) * rng.uniform(0.3, 1.5, bad.sum())
    H = np.moveaxis(hdr, 0, -1).astype(np.float32).copy()
    merge_moon(H, wt.astype(np.float32).copy(), head, es.astype(np.float32), ew.astype(np.float32), (40, 30),
               (float(eh["MOONX"]), float(eh["MOONY"])), join="rebuilt")
    ref = np.zeros_like(H)
    ref[crop] = np.moveaxis(truth, 0, -1)

    def rough(img, d):
        a = np.linspace(-0.1, 0.1, 60)
        ys, xs = 159.5 + (r + d) * np.sin(a), 159.5 + (r + d) * np.cos(a)
        v = np.stack([ndimage.map_coordinates(img[..., c], [ys, xs], order=1) for c in range(3)]).mean(0)
        return float(np.std(np.diff(np.log(np.maximum(v, 1e-9)))))

    for d in (-1.5, -3.0):
        assert rough(H, d) < 0.05, (d, rough(H, d), rough(ref, d))


def test_no_step_where_the_stacks_hand_over_outside_the_limb(tmp_path):
    """Outside the limb the earthshine stack's corona is not the HDR stack's (aligned on the Moon,
    so smeared; long frames saturated). Using it out to 3 px and then switching to the HDR stack
    made a step, seen as a line round the limb. Now the corona is continued in from 3 px out."""
    from tse_core.earthshine import merge_moon
    from tse_core.fits import read_fits

    proj, rr, phi, r = _moon_project(tmp_path, holes=False)
    hdr, head = read_fits(tmp_path / "e_hdr.fit")
    wt, _ = read_fits(tmp_path / "e_weight.fit")
    ew, _ = read_fits(tmp_path / "e_earthshine_weight.fit")
    es, eh = read_fits(tmp_path / "e_earthshine_rgb.fit")
    truth = np.moveaxis(hdr, 0, -1).copy()
    crop = (slice(30, 270), slice(40, 280))
    es = es.copy()
    es[:, rr[crop] > r] *= 0.75  # the earthshine stack's corona: darker than the HDR's
    H = np.moveaxis(hdr, 0, -1).astype(np.float32).copy()
    merge_moon(H, wt.astype(np.float32).copy(), head, es.astype(np.float32), ew.astype(np.float32), (40, 30),
               (float(eh["MOONX"]), float(eh["MOONY"])), join="rebuilt")
    away = np.abs(phi) > 0.3

    def ratio(a):
        m = (rr > r + a) & (rr < r + a + 0.5) & away
        return float(np.median(H[m].mean(axis=1) / truth[m].mean(axis=1)))

    steps = [abs(ratio(a + 0.5) - ratio(a)) for a in np.arange(1.0, 4.0, 0.5)]
    assert max(steps) < 0.1, steps
    print(steps)  # a gradual hand-over; the old switch at 3 px: one 25% step
    assert ratio(2.5) > 0.9


def test_corona_darkened_by_the_moons_edge_is_not_used(tmp_path):
    """Andy, Nov 2026: a faint line round the limb in the final image. On the 2024 stack the corona
    just outside the limb is darkened by the Moon's blurred edge in frames whose Moon sat just past
    the stacking margin, mostly on one side: at 6-8 o'clock 30 % below the corona's own trend 3 px
    out, 5 % at 10 px; elsewhere a few % at 3 px. The weights do not show it, so the HDR stack was
    used from 3 px all round, and the corona continued inwards to the limb met a 1 px darker ring
    there. Now the darkening is measured per direction from the corona's own trend and taken off,
    so the corona keeps its structure, and a prominence there (darkened the same way) its shape."""
    from tse_core.earthshine import merge_moon
    from tse_core.fits import read_fits

    proj, rr, phi, r = _moon_project(tmp_path, holes=False)
    hdr, head = read_fits(tmp_path / "e_hdr.fit")
    wt, _ = read_fits(tmp_path / "e_weight.fit")
    ew, _ = read_fits(tmp_path / "e_earthshine_weight.fit")
    es, eh = read_fits(tmp_path / "e_earthshine_rgb.fit")
    truth = np.moveaxis(hdr, 0, -1).copy()
    d = rr - r
    side = (phi > -0.5) & (phi < 2.4)                     # one side, with the prominence at 0
    dark = np.where(side & (d > 0), 0.3 * np.exp(-np.clip(d - 3.0, 0, None) / 4.0), 0.0)
    H = (truth * (1 - dark)[..., None]).astype(np.float32)   # as the 2024 stack at 6-8 o'clock
    merge_moon(H, wt.astype(np.float32).copy(), head, es.astype(np.float32), ew.astype(np.float32), (40, 30),
               (float(eh["MOONX"]), float(eh["MOONY"])), join="rebuilt")

    def ratio(a, where):
        m = (rr > r + a) & (rr < r + a + 0.5) & where
        return float(np.median(H[m].mean(axis=1) / truth[m].mean(axis=1)))

    core = (phi > 1.2) & (phi < 2.1)
    got = [round(ratio(a, core), 3) for a in np.arange(1.0, 12.0, 0.5)]
    assert min(got) > 0.92, got                          # the old code: 0.70 at 3 px
    steps = np.abs(np.diff(got))
    assert steps.max() < 0.04, got                       # no line
    other = (phi < -0.6) & (phi > -2.6)                  # the side not darkened: left as it was
    assert all(abs(ratio(a, other) - 1) < 0.03 for a in np.arange(4.0, 12.0, 1.0))
    prom = np.abs(phi) < 0.03                            # the prominence: as it is, not smoothed over
    assert all(abs(ratio(a, prom) - 1) < 0.08 for a in np.arange(3.0, 10.0, 1.0)), \
        [round(ratio(a, prom), 3) for a in np.arange(3.0, 10.0, 1.0)]


def test_hdr_reliable_zone_is_measured_from_the_weights():
    """Where the HDR stack's corona is used from follows its weights (the Moon covering frames),
    not a fixed number of pixels: 3 px at one scale, 8 px at a longer focal length, while the
    weight keeps rising outwards (fewer long frames saturated as the corona fades)."""
    n, r = 600, 250.0
    yy, xx = np.mgrid[0:n, 0:n]
    rr = np.hypot(xx - n / 2, yy - n / 2)
    rng = np.random.default_rng(2)
    d = rr - r
    for width in (3.0, 8.0):
        cover = np.clip((d + 0.5) / width, 0.02, 1) ** 2
        w = cover * np.exp(np.clip(d, 0, None) / 25.0)  # x10 over ~60 px, as on the 2024 stack
        w = np.where(d > 0, w * rng.uniform(0.9, 1.1, w.shape), 0.0).astype(np.float32)
        edge = ES.hdr_reliable_from(w, rr, r)
        assert width - 1.5 <= edge <= width + 1.0, (width, edge)
    assert ES.hdr_reliable_from(np.ones((n, n), np.float32), rr, r) == 2.0


def test_hdr_reliable_zone_on_the_2024_weights():
    """The ring medians of the 2024 D850 HDR weights, 0-59 px outside the limb: the Moon's
    coverage ends at about 3 px; the steady rise beyond (saturation) once gave 39 px."""
    prof = [2.12, 9.24, 19.5, 27.1, 28.3, 29.1, 31.7, 30.4, 32.2, 32.3, 33.2, 33.8, 35, 37.7, 39.7, 41.8,
            44.2, 46.2, 49.1, 50.8, 52.2, 53.6, 55, 57.5, 60.3, 63.5, 66.7, 69.9, 72.8, 76.2, 79.5, 81.9,
            83.6, 85.6, 91.2, 99.5, 105, 113, 120, 127, 134, 142, 150, 154, 158, 160, 166, 176, 186, 196,
            206, 218, 228, 240, 252, 262, 274, 278, 284, 289]
    r = 435.9
    n = 1100
    yy, xx = np.mgrid[0:n, 0:n]
    rr = np.hypot(xx - n / 2, yy - n / 2)
    d = rr - r
    w = np.interp(d, np.arange(len(prof)) + 0.5, prof, left=0.0, right=prof[-1])
    w = np.where(d > 0, w, 0.0).astype(np.float32)
    assert 2.5 <= ES.hdr_reliable_from(w, rr, r) <= 4.0


def test_inner_margin_scales_with_the_moon():
    p = EarthshineParams()
    assert p.margin(435.9) == 3.0  # 2024 D850 at 400 mm, as before
    assert p.margin(1090.0) == 7.5  # 1000 mm
    assert p.margin(150.0) == 2.0
    assert EarthshineParams(margin_px=4.0).margin(435.9) == 4.0


def test_edge_preserving_keeps_mare_edges_and_removes_noise():
    """The surface smoothing averages only across similar brightness: noise inside a 'mare' goes
    down about as much as a Gaussian of the same size would take it, but the mare's edge stays
    sharp (a Gaussian of that size spreads it over several pixels)."""
    n = 200
    yy, xx = np.mgrid[0:n, 0:n]
    truth = np.where(xx < 100, 0.3, 0.7).astype(np.float32)
    rng = np.random.default_rng(0)
    img = (truth + 0.03 * rng.normal(0, 1, truth.shape)).astype(np.float32)
    mask = np.ones_like(img, bool)
    out = ES.edge_preserving(img, mask, 4.0)
    gauss = ndimage.gaussian_filter(img, 4.0)
    flat = (xx > 20) & (xx < 80) & (yy > 20) & (yy < 180)
    assert out[flat].std() < 0.35 * img[flat].std()

    def edge_width(a):  # 10-90 % rise across the edge, mean over rows
        row = a[20:180].mean(axis=0)
        lo, hi = row[60:80].mean(), row[120:140].mean()
        t = (row - lo) / (hi - lo)
        return int(((t > 0.1) & (t < 0.9)).sum())

    assert edge_width(out) <= 3 < edge_width(gauss)


def test_the_filled_edge_band_is_named_in_the_log(run):
    """The last pixels inside the limb are filled from further in; how many, and why, is data-dependent,
    so the log says it (on the 2024 Moon it was the 2.5% minimum, 10.9 px, and the user could not tell)."""
    mos, infos, tex, matches, ref = run
    res, *_ = stack_earthshine(infos, lambda f: mos[f.index], {}, matches, (ref.moon_x, ref.moon_y), MOON_R,
                               (0, 0, SIZE, SIZE), EarthshineParams())
    line = [n for n in res.notes if "filled" in n and "inside the limb" in n]
    assert line, res.notes
    assert "% of the radius" in line[0] and ("minimum" in line[0] or "noise" in line[0]), line[0]


def test_defaults_as_andy_uses_them():
    """Nov 2026, from Andy's work on the 2024 Moon: a faint glow (0.10) and no extra softening are
    what blend best when pasting, and the prominence glare is left as the data has it (removing it
    cannot bring back what is under it). Later in Nov: no smoothing of any kind by default (the
    surface as the stack has it), and full brightness. Glare smoothness 3.5 (7 Nov 2026, after
    comparing 3, 3.5 and 4 on his stack: most of order 4's glare removal with more of order 3's
    maria contrast). The command line has the same defaults."""
    import sys
    from pathlib import Path

    p = EarthshineParams()
    assert (p.keep_glare, p.limb_soften, p.source_glare) == (0.1, 0.0, False)
    assert (p.smooth, p.edge_smooth, p.surface_denoise, p.brightness, p.glare_order) == (0.0, 0.0, 0.0, 1.0, 3.5)
    assert p.limb_shape is False
    sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))
    from cli_reference import parser

    sp = parser()._subparsers._group_actions[0].choices["earthshine"]
    a = sp.parse_args(["proj.json"])
    assert (a.keep_glare, a.limb_soften, a.prominence_glare) == (0.1, 0.0, False)
    assert (a.smooth, a.edge_smooth, a.surface_denoise, a.brightness, a.glare_order) == (0.0, 0.0, 0.0, 1.0, 3.5)
    assert a.limb_shape is False and sp.parse_args(["proj.json", "--limb-shape"]).limb_shape is True



def _saturating_limb_scene(n_frames=8, size=700, rs=150.0, seed=3, glare_step=0.3, limb_amp=2.75, noise=True,
                           ticks=None):
    """Frames whose long exposures saturate a few px inside the limb (deeper the longer the
    exposure, varying round the limb) and whose limb glare differs a little from frame to frame,
    as on the 2024 frames (10-30% as third contact neared). Moon fixed; returns the stack inputs."""
    scene = SyntheticScene(size=size, sun_radius=rs, seed=5)
    tex = _surface()
    yy, xx = np.mgrid[0:size, 0:size].astype(float)
    c = size / 2.0
    r_moon = rs * 1.055
    rng = np.random.default_rng(seed)
    corona = scene.corona(c, c) * 1e-3
    rr = np.hypot(xx - c, yy - c)
    phi = np.arctan2(yy - c, xx - c)
    disc = np.clip(r_moon - rr + 0.5, 0, 1)
    surf = ndimage.map_coordinates(tex, [(yy - c) / 3 + 64, (xx - c) / 3 + 64], order=1, mode="nearest")
    mos, infos = {}, []
    exposures = [0.25, 0.5, 1.0, 2.0, 4.0, 3.0, 1.5, 0.75][:n_frames]
    for i, t in enumerate(exposures):
        f = 1.0 + glare_step * (i % 2)                # this frame's glare, a little brighter or not
        d = np.clip(r_moon - rr, 0, None)
        limb = limb_amp * f * np.exp(-d / 2.5) * (1 + 0.4 * np.cos(3 * phi + i))
        moonlight = 4e-4 * (1 + 0.3 * surf) + 3e-3 + limb
        img = (corona * (1 - disc) + moonlight * disc) * t
        if ticks is not None and i == ticks[0]:   # this frame alone: a comb of ticks 8-12 px inside the limb
            band = (d > 8) & (d < 12) & (rr < r_moon)
            img = img + np.where(band, ticks[1] * t * np.sign(np.sin(phi * r_moon / 1.5)), 0.0)
        if noise:
            img = rng.poisson(np.clip(img, 0, None) * 3e4) / 3e4 + rng.normal(0, 2e-4, img.shape)
        mos[i] = mosaic_from(np.minimum(img, 1.0).astype(np.float32))
        infos.append(FrameInfo(i, f"f{i}", t, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], c, c, r_moon))
    matches = {f.index: Match(f.index, f.nominal_ratio, 0.0, f.nominal_ratio, 0.0, 0) for f in infos}
    return mos, infos, matches, (c, c), r_moon, size


def _row_jumps(detail, crop, centre, r, lo=4.0, hi=15.0):
    """Lines parallel to the limb: in 10-degree sectors, how much each half-pixel ring differs from the next."""
    X0, Y0, w, h = crop
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = centre[0] - X0, centre[1] - Y0
    dep = r - np.hypot(xx - cx, yy - cy)
    sec = ((np.arctan2(yy - cy, xx - cx) + np.pi) / (2 * np.pi) * 36).astype(int) % 36
    ds = np.arange(lo, hi, 0.5)
    prof = np.array([[np.median(detail[(sec == s) & (np.abs(dep - d) < 0.25)]) for d in ds] for s in range(36)])
    return float(np.median(np.std(np.diff(prof, axis=1), axis=1)))


def test_frames_fade_out_near_their_saturation_so_no_lines_follow_the_limb():
    """Each long exposure stops counting where it saturates, a few px inside the limb, at a depth of
    its own. Cut off at once, every frame whose glare differs a little from the rest leaves a step
    there, and the steps show as thin lines parallel to the limb (Andy's 'beads', Nov 2026; on 22
    of his 2024 frames the row-to-row jumps 6-15 px inside the limb fell from 2.6 to 1.1x the maria).
    Faded out over 1% of the Moon's radius instead, the steps go.

    Here the drop is smaller (about 0.68x): part of what this measure sees on the simulated limb is
    glare the removal leaves behind whatever the masks (with every frame's glare the same and no
    noise it still falls only from 65 to 49 x 1e-4), and fading cannot touch that."""
    mos, infos, matches, centre, r, size = _saturating_limb_scene()
    out = {}
    for fade in (0.0, 1.0):
        res, lum, detail = stack_earthshine(infos, lambda f: mos[f.index], {}, matches, centre, r,
                                            (0, 0, size, size), EarthshineParams(mask_fade=fade, glare_order=2))
        out[fade] = _row_jumps(detail, res.crop, centre, r)
        assert any("fades out" in n for n in res.notes) == (fade > 0)
    assert EarthshineParams().mask_fade == 1.0
    assert out[1.0] < 0.8 * out[0.0], out


def test_frames_off_the_moons_track_count_less():
    """Camera shake puts a frame's Moon off the smooth track Align fits, and its limb is smeared. On
    22 of Andy's 2024 frames the steadiest 10 alone gave 0.99x the maria in row-to-row jumps by the
    limb, the other 12 5.6x; weighting by distance from the track (on top of faded masks) took the
    whole stack from 0.99 to 0.69. A frame counts 1 / (1 + (off / s)^2), s twice the typical scatter."""
    from tse_core.earthshine import track_factors
    from tse_core.workflow import _frame_infos
    from tse_core.inventory import Inventory
    from tse_core.project import EventProject
    from conftest import make_frames

    infos = [FrameInfo(i, f"f{i}", 1.0, [[1, 0, 0], [0, 1, 0]], 0, 0, 436.0, off)
             for i, off in enumerate([0.2, 0.3, 0.4, 0.4, 0.5, 1.3, 2.2])]
    f = track_factors(infos, EarthshineParams(), 436.0)
    s = 2 * 1.4826 * 0.4
    assert f[0] == pytest.approx(1 / (1 + (0.2 / s) ** 2)) and f[0] > 0.9
    assert f[5] == pytest.approx(1 / (1 + (1.3 / s) ** 2)) and f[5] < 0.5      # like DSC_9126
    assert track_factors(infos, EarthshineParams(track_weight=False), 436.0) == {}
    assert track_factors(infos[:2], EarthshineParams(), 436.0) == {}            # too few to say
    # Align's residuals reach the earthshine stack, frame by frame
    frames = make_frames(["1/4", "1/2", "1"], aperture=8.0)
    project = EventProject(name="e")
    project.set_inventory(Inventory(frames))
    alignment = {"reference_index": frames[0].index,
                 "frames": [{"index": fr.index, "file_name": fr.file_name, "matrix": [[1, 0, 0], [0, 1, 0]],
                             "moon_ref": [10.0, 10.0], "moon_frame": {"radius": 5.0}} for fr in frames],
                 "sun": {"rejected": [], "residuals_px": [0.3, 1.4, 0.2]}}
    _, _, got, _ = _frame_infos(project, alignment, include_rejected=False)
    assert [g.track_off for g in got] == [0.3, 1.4, 0.2]


def test_files_keep_their_name_when_shaken_frames_are_named_in_the_log(run, tmp_path):
    """Nov 2026: the log line naming frames off the Moon's track reused the variable holding the
    output name, so the files were named after a list of frames ('File name too long' in Siril)."""
    from dataclasses import replace

    mos, infos, tex, matches, ref = run
    offs = [0.2, 0.3, 2.5, 0.25]
    shaken = [replace(f, track_off=o) for f, o in zip(infos, offs)]
    res, *_ = stack_earthshine(shaken, lambda f: mos[f.index], {}, matches, (ref.moon_x, ref.moon_y), MOON_R,
                               (0, 0, SIZE, SIZE), EarthshineParams(), out_dir=tmp_path, name="ecl")
    assert any("counted less" in n and "f2" in n for n in res.notes), res.notes
    assert all(Path(v).name.startswith("ecl_") for v in res.outputs.values() if isinstance(v, str)), res.outputs


def test_robust_factors_turn_down_a_frame_that_disagrees_only_where_it_does():
    from tse_core.earthshine import robust_factors
    rng = np.random.default_rng(0)
    F, H, Wd = 8, 40, 60
    yy, xx = np.mgrid[0:H, 0:Wd]
    where = (yy.ravel(), xx.ravel())
    w = np.full((F, H * Wd), 4.0, np.float32)               # noise 0.5 in every frame
    v = (1.0 + rng.normal(0, 0.5, (F, H * Wd))).astype(np.float32)
    patch = ((yy > 10) & (yy < 20) & (xx > 20) & (xx < 40)).ravel()
    v[3, patch] += 4.0                                       # one frame off by 8 sigma, over a patch
    f = robust_factors(v, w, np.full(H * Wd, 5.0), 3.0, where, (H, Wd), smooth=1.0)
    assert f[3, patch].mean() < 0.15
    assert f[3, ~patch].mean() > 0.85 and f[[0, 1, 2, 4, 5, 6, 7]].mean() > 0.85


def test_a_frame_with_ticks_by_the_limb_counts_less_there():
    """On Andy's 2024 stack two frames (DSC_9132/9133) left a zipper of ticks 9-11 px inside the limb
    that no fade or track weighting removed. Near the limb a frame's pixel now counts less the further
    it lies from the other frames, in its own noise; there the ticks fell from 2.9 to 2.15x the
    disc's fine structure (1.93 with the two frames left out): what the two frames added beyond the
    noise fell by 55%. Here one of 8 simulated frames carries ticks, and it is the longest exposure,
    with over half the weight there: soft weighting (it counts a fifth at 6 sigma, not nothing) takes
    about 37% of the ticks away. The log names that frame."""
    out, notes = {}, {}
    for k in (0.0, 3.0):
        diff = []
        for tk in (None, (4, 2e-3)):
            mos, infos, matches, centre, r, size = _saturating_limb_scene(ticks=tk)
            res, lum, detail = stack_earthshine(infos, lambda f: mos[f.index], {}, matches, centre, r,
                                                (0, 0, size, size),
                                                # the test Moon is 158 px: 10% reaches the ticks, as 5% (22 px) does on a 400 mm Moon
                                                EarthshineParams(robust_k=k, robust_zone=10.0, glare_order=2))
            diff.append(detail)
            notes[k] = res.notes
        X0, Y0, w, h = res.crop
        yy, xx = np.mgrid[0:h, 0:w]
        dep = r - np.hypot(xx - (centre[0] - X0), yy - (centre[1] - Y0))
        band = (dep > 8) & (dep < 12)
        out[k] = float(np.std((diff[1] - diff[0])[band]))
    assert EarthshineParams().robust_k == 3.0
    assert out[3.0] < 0.75 * out[0.0], out
    assert any("count less" in n and "f4" in n for n in notes[3.0]), notes[3.0]
    assert not any("frame pixels far from" in n for n in notes[0.0])






def test_moon_layer_over_the_blend_gives_the_blend_final_image(tmp_path):
    """Andy, Nov 2026: to blend the Moon in Photoshop he needed the Moon as Blend draws it, with the
    Lunar surface strength in it (opacity in Photoshop changed the limb too). Write Moon layer:
    the Moon alone, transparent elsewhere, over exactly the pixels where the corona layers are
    black, and over the band outside the limb where the final image blends the Moon's stack into the
    corona (where it differs visibly from the written blend); over that blend at 100 % in Normal
    mode it is the Blend final image."""
    import tifffile

    from tse_core import workflow as W
    from tse_core.enhance import EnhanceParams, enhance
    from tse_core.imageout import read_image, write_image
    from tse_core.project import EventProject

    proj, rr, phi, r = _moon_project(tmp_path)
    params = EnhanceParams(layers=(), blend={"natural": 1.0})
    data = W.load_hdr(proj)
    _, col = enhance(data.hdr.copy(), data.weight, data.header, params, out_dir=None, keep={"blend"})
    blend = read_image(write_image(tmp_path / "e_blend.tif", col["blend"]))
    p = EventProject.load(proj)
    p.parameters["enhance"] = {"blend": "e_blend.tif"}   # the blend as last written
    p.save(proj)
    want = tifffile.imread(W.final_with_moon(proj, "blend", 0.4, enhance_params=params,
                                             output_file=str(tmp_path / "a.tif"))).astype(np.float32) / 65535
    msgs = []
    out = W.moon_layer(proj, "blend", 0.4, enhance_params=params, output_file=str(tmp_path / "m.tif"),
                       say=msgs.append)
    lay = read_image(out, alpha=True)
    rgb, a = lay[..., :3], lay[..., 3:4]
    normal = blend * (1 - a) + rgb * a
    assert np.abs(normal - want).max() < 0.5 / 255 + 2 / 65535     # no visible step anywhere
    assert np.abs(normal - want)[a[..., 0] == 1].max() < 3 / 65535  # exact where the layer is opaque
    assert a[rr < r - 2].min() == 1 and a[rr > r + 1.6].sum() > 0    # the disc, and a band past its edge
    assert a[rr > r + 30].max() == 0                                                     # only the Moon
    # A soft edge grown outward over 1.15 % of the Moon's radius (5 px for a 400 mm lens on a
    # 45-megapixel full-frame camera; Andy, Nov 2026: 2.5 px still left a faint line where the two
    # edges meet in Photoshop): the edges then meet without a line.
    from scipy import ndimage
    core = a[..., 0] == 1
    d = ndimage.distance_transform_edt(~core)
    w = max(2.0, 0.0115 * r)                                        # at least 2 px
    one = (d > 0.99) & (d < 1.01)                                   # the first pixels outside
    assert np.allclose(a[..., 0][one], 1 - 1 / w, atol=2 / 65535)   # partly opaque
    assert a[..., 0][d >= w].max() == 0
    assert EventProject.load(proj).parameters["moon_layer"]["strength"] == 0.4
    assert any("Moon layer" in m for m in msgs), msgs


def test_half_steps_of_glare_smoothness_average_the_two_neighbouring_fits():
    """Andy, Nov 2026: Glare smoothness in half steps, 0-6. A polynomial has whole orders only, so
    3.5 is the average of the order-3 and order-4 glare surfaces fitted to the data; whole numbers
    are exactly as before, and 0 (one even level) is allowed."""
    from tse_core.earthshine import glare_fit, glare_surface

    n, r, c = 240, 90.0, 120.0
    yy, xx = np.mgrid[0:n, 0:n]
    rr = np.hypot(xx - c, yy - c)
    u, v = (xx - c) / r, (yy - c) / r
    rng = np.random.default_rng(5)
    img = (1.0 + 0.3 * u + 0.2 * u * v + 0.15 * u ** 3 - 0.1 * v ** 4 + 0.01 * rng.normal(size=(n, n))).astype(np.float32)
    inside = rr < r - 3
    g3 = glare_surface(img, inside, c, c, r, 3)
    g4 = glare_surface(img, inside, c, c, r, 4)
    assert np.allclose(glare_fit(img, inside, c, c, r, 3.5), 0.5 * (g3 + g4), atol=1e-6)
    assert np.array_equal(glare_fit(img, inside, c, c, r, 4.0), g4)
    g0 = glare_fit(img, inside, c, c, r, 0)
    assert np.ptp(g0[inside]) < 1e-6 and abs(float(g0[inside].mean()) - float(np.median(img[inside]))) < 0.1


def test_filter_layers_do_not_see_the_moons_edge(tmp_path):
    """Andy, Nov 2026: a dark 1 px line round the limb with MGN at 45 % on Weighted Average. The
    filter layers have data from the black disc out (1.5 px outside the limb), but the stack there
    still holds the Moon's blurred edge, where the earthshine stack hands over to the corona: on his
    2024 stack MGN's input rose from 0.77 at 1.5 px to the corona's 0.88 at 2.75 px, MGN read that
    as a dark feature (0.35 at 1.5 px, 0.60 at 2 px, 0.75 from 2.5 px) and the final image dipped
    to 0.730 at 1.5 px between 0.757 and 0.771. The filter layers now see the corona continued to
    the limb there; the natural layer keeps the limb."""
    from tse_core import workflow as W
    from tse_core.enhance import EnhanceParams

    # The earthshine stack's limb as on Andy's: 10-90 % of the corona over 4.7 px (sigma 1.8 px).
    # The stack used from 4 px out, as his (the scene twice the usual size: at most 3 % of the radius).
    proj, rr, phi, r = _moon_project(tmp_path, holes=False, es_blur=1.8, cover=6.0, scale=2)
    p = EnhanceParams(layers=(), blend={"natural": 1.0, "adaptive": 0.45}, blend_modes={"adaptive": "average"})
    img = W.final_with_moon(proj, "blend", 1.0, enhance_params=p, preview_side=10000).mean(axis=2)
    cap = {}
    import tse_core.enhance as En
    orig = En.soft_light_blend

    def keep(layers, *a, **k):
        cap.update(layers)
        return orig(layers, *a, **k)
    En.soft_light_blend = keep
    try:
        W.final_with_moon(proj, "blend", 1.0, enhance_params=p, preview_side=10000)
    finally:
        En.soft_light_blend = orig
    got = cap["adaptive"].mean(axis=2) if cap["adaptive"].ndim == 3 else cap["adaptive"]
    far = (rr >= r + 6) & (rr < r + 12)
    level = float(np.median(got[far]))
    for a in (1.5, 2.0, 2.5, 3.0):                # MGN's first rings: no dark edge
        ring = (rr >= r + a) & (rr < r + a + 0.5)
        assert float(np.median(got[ring])) > level - 0.05, (a, round(float(np.median(got[ring])), 3), round(level, 3))
    def at(a):
        return float(np.median(img[(rr >= r + a) & (rr < r + a + 0.5)]))
    for a in np.arange(1.0, 4.0, 0.5):            # the final image: no ring darker than both its neighbours
        assert at(a) > min(at(a - 0.5), at(a + 0.5)) - 0.01, (a, round(at(a - 0.5), 3), round(at(a), 3), round(at(a + 0.5), 3))


@pytest.mark.parametrize("layer, weight", [("adaptive", 0.43), ("fnrgf", 0.4)])
def test_linear_layers_leave_no_bright_ring_round_the_moon(tmp_path, layer, weight):
    """Andy, Nov 2026: with the Moon put in, a blend on Linear (MGN at 43 % with the Gamma stretch)
    had a white ring just outside the limb. The filter layers have no data on the Moon, so they fade
    out over the 4 px outside the limb; on Linear their weight faded, and with it the division by
    K1 + K2, so the corona there lost the Linear form's darkening (on the 2024 crop 0.876 grey at
    1.5-3 px against 0.806 at 3-4.5 px). Now the layers keep full strength wherever they have data
    (out to the black disc), and on the Moon go to their neutral value at full weight, as Soft Light
    does, over 1 px: just outside the limb the blend is the blend without the Moon (on the 2024
    crop, MGN 43 % Linear with Gamma: 0.743 / 0.766 / 0.802 at 1.5-3 / 3-4.5 / 4.5-6 px, against
    0.798 / 0.762 / 0.795 without the Moon)."""
    from tse_core import workflow as W
    from tse_core.enhance import EnhanceParams

    from tse_core.enhance import enhance

    proj, rr, phi, r = _moon_project(tmp_path, holes=False)
    p = EnhanceParams(layers=(), blend={"natural": 1.0, layer: weight}, blend_modes={layer: "linear"})
    img = W.final_with_moon(proj, "blend", 1.0, enhance_params=p, preview_side=10000).mean(axis=2)
    data = W.load_hdr(proj)                       # the same blend without the Moon (as Enhance writes it)
    _, col = enhance(data.hdr.copy(), data.weight, data.header, p, out_dir=None, keep={"blend"})
    plain = col["blend"].mean(axis=2)
    for a in (1.5, 2.5, 3.5, 4.5):                # the 4 px where the layer has no data beyond
        ring = (rr >= r + a) & (rr < r + a + 1)
        got, want = float(np.median(img[ring])), float(np.median(plain[ring]))
        assert abs(got - want) < 0.03, (a, round(got, 3), round(want, 3))


def test_final_image_box_is_the_written_image(tmp_path):
    """Andy, Nov 2026: the Prominences tab's preview did not look like the written image (it was the
    linear stack with a quick stretch of its own). It now draws part of the final image, full size,
    exactly as Write final image does."""
    from tse_core import imageout as IO
    from tse_core import workflow as W

    proj, rr, phi, r = _moon_project(tmp_path, holes=False)
    out = W.final_with_moon(proj, "blend", 1.0, output_file=str(tmp_path / "f.tif"))
    full = IO.read_image(out)
    part = W.final_with_moon(proj, "blend", 1.0, box=(60, 260, 40, 200))
    assert part.shape == (200, 160, 3)
    assert np.abs(part - full[60:260, 40:200]).max() < 2 / 65535


def test_limb_colour_takes_the_red_glare_off_and_can_make_the_moon_neutral():
    """As on the 2024 frame-9 earthshine stack: the Moon's middle R/G 0.47, B/G 0.79; the
    chromosphere's glare reddens the last pixels inside the limb to R/G 0.64, B/G 0.69 at 2-4 px
    and fades by about 40 px. The corona's white balance: R x 2.03, B x 1.34."""
    from tse_core.earthshine import limb_colour

    n, cx, cy, r = 400, 199.5, 199.5, 150.0
    yy, xx = np.mgrid[0:n, 0:n]
    rr = np.hypot(xx - cx, yy - cy)
    good = rr < r
    d = r - rr
    rng = np.random.default_rng(3)
    G = (0.002 * (1 + 0.05 * rng.standard_normal((n, n))) + 0.03 * np.exp(-np.clip(d, 0, None) / 4.0)).astype(np.float32)
    red = np.exp(-np.clip(d, 0, None) / 12.0)  # glare's share of the colour excess
    rgb = np.stack([G * (0.47 + 0.17 * red), G, G * (0.79 - 0.10 * red)]).astype(np.float32)
    rgb = np.where(good, rgb, 0.0)
    band = good & (d >= 2) & (d < 4)
    mid = good & (rr < 0.5 * r)

    same, note = limb_colour(rgb, good, cx, cy, r, "recorded")
    assert same is rgb and note == ""

    out, note = limb_colour(rgb, good, cx, cy, r, "no red")
    assert abs(np.median(out[0][band] / out[1][band]) - 0.47) < 0.01   # was 0.63
    assert abs(np.median(out[2][band] / out[1][band]) - 0.79) < 0.01
    assert np.array_equal(out[1], rgb[1])                               # brightness untouched
    assert abs(np.median(out[0][mid] / out[1][mid]) - 0.47) < 0.005     # the middle as it was
    assert "R/G" in note

    white = (2.03, 1.0, 1.34)
    neu, note = limb_colour(rgb, good, cx, cy, r, "neutral", white)
    for c in (0, 2):  # the Moon (middle and edge) in the corona's white: R/G 1/2.03, B/G 1/1.34
        for zone in (mid, band):
            assert abs(np.median(neu[c][zone] / neu[1][zone]) * white[c] - 1.0) < 0.02
    assert "white" in note
    with pytest.raises(ValueError):
        limb_colour(rgb, good, cx, cy, r, "pink")


def test_corona_first_keeps_the_blend_and_uses_the_earthshine_only_where_the_blend_is_black(tmp_path):
    """Andy, Nov 2026: the Blend without the Moon looked right up to its black disc; with the Moon
    put in, the corona in the 4-6 px outside the limb was rebuilt (a red ring, a bigger pink
    prominence, a small prominence lost). The rule: the HDR stack wherever the Blend has it, the
    earthshine stack elsewhere in the disc, and black where neither has data."""
    from tse_core.earthshine import merge_moon
    from tse_core.fits import read_fits

    proj, rr, phi, r = _moon_project(tmp_path, holes=False)
    hdr, head = read_fits(tmp_path / "e_hdr.fit")
    wt, _ = read_fits(tmp_path / "e_weight.fit")
    ew, _ = read_fits(tmp_path / "e_earthshine_weight.fit")
    es, eh = read_fits(tmp_path / "e_earthshine_rgb.fit")
    H = np.moveaxis(hdr, 0, -1).astype(np.float32).copy()
    W = wt.astype(np.float32).copy()
    ew = ew.astype(np.float32).copy()
    crop = (slice(30, 270), slice(40, 280))
    # A few pixels by the limb that neither stack has: the HDR's covered, the earthshine's saturated.
    gap = (np.abs(phi) < 0.03) & (rr > r - 3) & (rr < r + 1.4)
    ew[gap[crop]] = 0.0
    before = H.copy()
    moon, _ = merge_moon(H, W, head, es.astype(np.float32), ew, (40, 30), (float(eh["MOONX"]), float(eh["MOONY"])))
    blend_has = (rr > r + 1.5) & (wt > 0)
    assert np.array_equal(H[blend_has], before[blend_has])            # the corona exactly as in the Blend
    inside = np.zeros(W.shape, bool); inside[crop] = (rr[crop] < r - 2) & (ew > 0) & ~gap[crop]
    es_c = np.moveaxis(es, 0, -1)
    assert np.allclose(H[crop][inside[crop]], es_c[inside[crop]], rtol=1e-4, atol=1e-7)  # the earthshine
    g = np.zeros(W.shape, bool); g[crop] = gap[crop] & ~blend_has[crop]
    assert g.any() and (W[g] == 0).all() and (H[g] == 0).all()        # neither: black
    assert moon.mask[inside].all() and not moon.mask[blend_has].any()




def test_corona_first_uses_earthshine_pixels_from_few_frames(tmp_path):
    """On Andy's frame-9 stack (Nov 2026) 1119 px by the limb were left black under the rule, most of
    them just inside the limb by the 6 o'clock prominence: the earthshine stack had data there, but
    from only 0.5-9.5 % of its usual frames (the longer ones saturated by the prominence's glare),
    and the rule used it only from 10 %. Those pixels hold the glare recorded by the short frames
    (red, 150x the earthshine); in his final image without the limit they continue the prominence's
    glare and 280 px are left, which no frame recorded. Now any earthshine data is used."""
    from tse_core.earthshine import merge_moon
    from tse_core.fits import read_fits

    proj, rr, phi, r = _moon_project(tmp_path, holes=False)
    hdr, head = read_fits(tmp_path / "e_hdr.fit")
    wt, _ = read_fits(tmp_path / "e_weight.fit")
    es, eh = read_fits(tmp_path / "e_earthshine_rgb.fit")
    ew, _ = read_fits(tmp_path / "e_earthshine_weight.fit")
    crop = (slice(30, 270), slice(40, 280))
    few = (np.abs(phi) < 0.05) & (rr > r - 6) & (rr < r - 1)
    fc = few[crop]
    es = es.astype(np.float32).copy()
    es[0][fc], es[1][fc], es[2][fc] = 0.30, 0.10, 0.12           # red glare, as by the prominence
    ew = ew.astype(np.float32).copy()
    ew[fc] = 0.05 * float(np.median(ew[ew > 0]))                 # 5 % of the usual frames
    H = np.moveaxis(hdr, 0, -1).astype(np.float32).copy()
    W = wt.astype(np.float32).copy()
    msgs = []
    merge_moon(H, W, head, es, ew, (40, 30), (float(eh["MOONX"]), float(eh["MOONY"])), say=msgs.append)
    assert (W[few] > 0).all(), msgs                                # used, not left black
    assert np.allclose(H[few], [0.30, 0.10, 0.12], atol=1e-4)      # the earthshine stack's own values
