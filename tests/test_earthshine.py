"""Earthshine: a Moon-aligned stack recovers the lunar surface that a corona-aligned one blurs."""

from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage

from tse_core.earthshine import EarthshineParams, glare_surface, moon_aligned, stack_earthshine
from tse_core.stack import FrameInfo, Match
from tse_core.synthetic import SyntheticScene
from test_stack import mosaic_from

SIZE, SUN, RS = 400, (200.0, 200.0), 40.0
MOON_R = RS * 1.055


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
                                        (0, 0, SIZE, SIZE), EarthshineParams(), out_dir=tmp_path, name="t")
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
    outside = np.hypot(xx - ref.moon_x, yy - ref.moon_y) > MOON_R + p.limb_px + 2  # the real limb, then nothing
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


def test_final_image_workflow(tmp_path):
    import tifffile

    from tse_core import workflow as W
    from tse_core.project import EventProject

    base = (np.full((40, 60, 3), 0.5) * 65535).astype(np.uint16)
    base[10:30, 20:40] = 0
    earth = np.zeros_like(base)
    earth[12:28, 22:38] = 20000
    tifffile.imwrite(tmp_path / "e_blend.tif", base, photometric="rgb")
    tifffile.imwrite(tmp_path / "e_earthshine.tif", earth, photometric="rgb")
    p = EventProject(name="e")
    p.parameters["enhance"] = {"blend": "e_blend.tif"}
    p.parameters["earthshine"] = {"earthshine": "e_earthshine.tif"}
    p.save(tmp_path / "e.json")
    prev = W.final_image(tmp_path / "e.json", preview_side=30)
    assert prev.shape == (20, 30, 3)
    out = W.final_image(tmp_path / "e.json", strength=1.5, mode="lighten")
    img = tifffile.imread(out)
    assert img[20, 30, 0] == pytest.approx(min(65535, 20000 * 1.5), abs=2)
    assert img[0, 0, 0] == pytest.approx(32768, abs=2)
    assert EventProject.load(tmp_path / "e.json").parameters["final"]["mode"] == "lighten"
    with pytest.raises(W.WorkflowError, match="'natural' layer"):
        W.final_image(tmp_path / "e.json", base_layer="natural")
    # Any image of the right size can be the corona; a cropped one is refused with a clear message.
    tifffile.imwrite(tmp_path / "other.tif", (np.full((40, 60, 3), 0.25) * 65535).astype(np.uint16), photometric="rgb")
    out2 = W.final_image(tmp_path / "e.json", base_file=str(tmp_path / "other.tif"), output_file=str(tmp_path / "f2.tif"))
    assert tifffile.imread(out2)[0, 0, 0] == pytest.approx(16384, abs=2)
    tifffile.imwrite(tmp_path / "small.tif", np.zeros((20, 30, 3), np.uint16), photometric="rgb")
    with pytest.raises(W.WorkflowError, match="size of the HDR stack"):
        W.final_image(tmp_path / "e.json", base_file=str(tmp_path / "small.tif"))


def test_limb_glare_removed_and_layer_meets_the_corona(run, tmp_path):
    """The steep glare just inside the limb is taken out (no bright rim), and the layer
    reaches the corona layers' black disc (moon_r + 1.5 px) with no dark ring between."""
    import tifffile

    from tse_core.earthshine import composite

    mos, infos, tex, matches, ref = run
    p = EarthshineParams(limb_darkening=0.0)  # evenly lit, so a dark ring would stand out
    res, lum, detail = stack_earthshine(infos, lambda f: mos[f.index], {}, matches, (ref.moon_x, ref.moon_y), MOON_R,
                                        (0, 0, SIZE, SIZE), p, out_dir=tmp_path, name="t")
    X0, Y0, w, h = res.crop
    yy, xx = np.mgrid[0:h, 0:w]
    rr = np.hypot(xx - (ref.moon_x - X0), yy - (ref.moon_y - Y0))
    sm = ndimage.gaussian_filter(detail, 2.0)
    edge = (rr > MOON_R - 7) & (rr < MOON_R - p.margin_px - 1)
    middle = rr < MOON_R * 0.6
    assert abs(np.median(sm[edge]) - np.median(sm[middle])) < 3e-5
    _, _, plain = stack_earthshine(infos, lambda f: mos[f.index], {}, matches, (ref.moon_x, ref.moon_y), MOON_R,
                                   (0, 0, SIZE, SIZE), EarthshineParams(limb_glare=False))
    sp = ndimage.gaussian_filter(plain, 2.0)
    assert np.median(sp[edge]) - np.median(sp[middle]) > 6e-5  # the test does contain a rim to remove
    # The layer is filled right out to the corona's edge and soft only beyond it.
    layer = tifffile.imread(res.outputs["earthshine"]).mean(axis=2) / 65535
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    rf = np.hypot(xx - ref.moon_x, yy - ref.moon_y)
    disc = rf < MOON_R * 0.9
    ring = (rf > MOON_R - 4) & (rf < MOON_R + 1.0)
    assert np.median(layer[ring]) > 0.5 * np.median(layer[disc])
    assert layer[rf > MOON_R + p.limb_px + 2].max() == 0
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
                                        (0, 0, SIZE, SIZE), EarthshineParams())
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


def test_limb_darkening_in_the_layer(run, tmp_path):
    import tifffile

    mos, infos, tex, matches, ref = run
    res, *_ = stack_earthshine(infos, lambda f: mos[f.index], {}, matches, (ref.moon_x, ref.moon_y), MOON_R,
                               (0, 0, SIZE, SIZE), EarthshineParams(limb_darkening=0.6, keep_glare=0.0, limb_reach=0.0),
                               out_dir=tmp_path, name="d")
    layer = tifffile.imread(res.outputs["earthshine"]).mean(axis=2)
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    rf = np.hypot(xx - ref.moon_x, yy - ref.moon_y) / MOON_R
    middle = np.median(layer[rf < 0.3])
    edge = np.median(layer[(rf > 0.95) & (rf < 1.0)])
    assert 0.3 < edge / middle < 0.6  # about 40-50% of the middle near the limb


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
    p = EarthshineParams(keep_glare=0.3, limb_soften=0.0, limb_darkening=0.0)
    good = rr < r - p.margin_px - 2
    weight = (rr < r + p.limb_reach + 2).astype(np.float32)
    detail, glare = remove_glare(lin, good, cx, cy, r, EarthshineParams(limb_shape=False))
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
    off = compose_layer(detail, good, weight, cx, cy, r, EarthshineParams(keep_glare=0.0, limb_soften=0.0,
                        limb_darkening=0.0), keep_from=6.0, linear=lin, glare=glare)
    assert np.median(off[(rr > r - 40) & (rr < r - 25)]) < np.median(layer[(rr > r - 40) & (rr < r - 25)])
    # Nothing beyond the reach of the stack.
    assert layer[rr > r + p.limb_px + 2].max() == 0


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
        p = EarthshineParams(edge_grain=grain, limb_darkening=0.0, limb_reach=0.0)
        out = compose_layer(shown, good, good.astype(np.float32), 100, 100, r, p, keep_from=10.0)
        hp = out - ndimage.gaussian_filter(out, 2)
        inner = good & (rr < r - 15) & (rr > r - 30)
        ratio = hp[band].std() / hp[inner].std()
        assert (ratio > 0.5) == expect, ratio
