"""Phase 3: demosaic, full-resolution regions, brightness matching and the HDR stack."""

import math

import numpy as np
import pytest

from tse_core.demosaic import channel_masks, cosmetic_correct, demosaic_mhc
from tse_core.fits import read_fits
from tse_core.rawio import RawMosaic, region
from tse_core.stack import FrameInfo, StackParams, crop_box, robust_line, source_box, stack_frames, warp_region
from tse_core.synthetic import SyntheticScene

BLACK, WHITE = 600.0, 16383.0


def mosaic_from(img: np.ndarray, pattern: str = "RGGB", colour=(1.0, 1.0, 1.0)) -> RawMosaic:
    """A Bayer mosaic of a grey (or tinted) linear image, in raw units."""
    masks = channel_masks(pattern, img.shape)
    raw = np.empty(img.shape, np.float32)
    for k, c in enumerate("RGB"):
        raw[masks[c]] = img[masks[c]] * colour[k]
    raw = np.minimum(BLACK + raw * (WHITE - BLACK), WHITE)
    return RawMosaic(raw.astype(np.float32), pattern, {"R": BLACK, "G": BLACK, "B": BLACK}, WHITE, (2.0, 1.0, 1.5))


@pytest.mark.parametrize("pattern", ["RGGB", "BGGR", "GRBG", "GBRG"])
def test_demosaic_smooth(pattern):
    yy, xx = np.mgrid[0:64, 0:64].astype(np.float32)
    truth = 0.2 + 0.3 * np.sin(xx / 9.0) * np.cos(yy / 11.0)
    cfa = np.empty_like(truth)
    m = channel_masks(pattern, truth.shape)
    for c in "RGB":
        cfa[m[c]] = truth[m[c]]
    rgb = demosaic_mhc(cfa, pattern)
    err = np.abs(rgb[4:-4, 4:-4] - truth[4:-4, 4:-4, None]).max()
    assert err < 0.005


def test_cosmetic_correct_hot_pixel_only():
    rng = np.random.default_rng(3)
    cfa = (0.1 + rng.normal(0, 0.002, (64, 64))).astype(np.float32)
    cfa[20, 30] = 0.9  # hot
    cfa[40:43, 10:13] = 0.6  # a real bright feature, several photosites wide
    out, n = cosmetic_correct(cfa, "RGGB")
    assert abs(out[20, 30] - 0.1) < 0.01
    assert n >= 1
    assert out[41, 11] == pytest.approx(0.6)


def test_region_calibrates_and_snaps():
    img = np.full((40, 60), 0.25, np.float32)
    mos = mosaic_from(img, colour=(0.5, 1.0, 0.8))
    fr = region(mos, 5, 7, 20, 10)
    assert fr.x0 % 2 == 0 and fr.y0 % 2 == 0
    assert fr.rgb[2:-2, 2:-2, 0].mean() == pytest.approx(0.125, rel=1e-4)
    assert fr.rgb[2:-2, 2:-2, 1].mean() == pytest.approx(0.25, rel=1e-4)
    assert fr.rgb[2:-2, 2:-2, 2].mean() == pytest.approx(0.2, rel=1e-4)
    assert not fr.saturated.any()


def test_robust_line_ignores_outliers():
    rng = np.random.default_rng(0)
    x = rng.uniform(0.01, 0.4, 2000)
    y = 2.1 * x + 0.003
    y[:100] *= 3  # e.g. a hot spot or a moving feature
    a, b, n = robust_line(x, y)
    assert a == pytest.approx(2.1, rel=1e-3)
    assert b == pytest.approx(0.003, abs=5e-4)
    assert n < 2000


def test_warp_and_source_box_identity():
    img = np.arange(100 * 100, dtype=np.float32).reshape(100, 100)
    M = [[1, 0, 0], [0, 1, 0]]
    crop = (20, 30, 40)
    x0, y0, w, h = source_box(M, crop)
    out = warp_region(img[y0 : y0 + h, x0 : x0 + w], M, crop, (x0, y0), order=1)
    assert np.allclose(out, img[30:70, 20:60])


def _synthetic_stack(noise: bool, errors: dict):
    scene = SyntheticScene(size=480, sun_radius=40.0, seed=4)
    sun = (240.0, 240.0)
    nominal = [1 / 256, 1 / 64, 1 / 16, 1 / 4, 1.0]
    ref = 2
    shifts = [(-2.3, 1.1), (0.7, -1.6), (0.0, 0.0), (1.4, 2.2), (-0.9, -0.4)]
    moons = [(-6, -2), (3, 1), (0, 0), (-3, -1), (6, 2)]  # the Moon drifting (exposures not in time order)
    mosaics, infos = {}, []
    for i, (t, (dx, dy), (mdx, mdy)) in enumerate(zip(nominal, shifts, moons)):
        actual = t * (1 + errors.get(i, 0.0)) * 0.25  # brightness scale: the shortest frame just reaches full scale at the limb
        img, moon = scene.frame(actual, sun[0] + dx, sun[1] + dy, mdx, mdy, noise=noise, seed=i, gain=5e4)
        mosaics[i] = mosaic_from(img)
        # frame -> reference: undo the shift
        M = [[1.0, 0.0, -dx], [0.0, 1.0, -dy]]
        infos.append(FrameInfo(i, f"f{i}", t / nominal[ref], M, moon["moon_x"] - dx, moon["moon_y"] - dy, moon["moon_r"]))
    sun_info = {"x": sun[0], "y": sun[1], "radius_px": 40.0, "arcsec_per_px": 24.0, "rotation_deg": 0.0}
    params = StackParams(crop_radii=4.0)
    result, hdr, weight = stack_frames(infos, lambda f: mosaics[f.index], {}, (480, 480), ref, sun_info, params)
    truth = scene.frame(nominal[ref] * (1 + errors.get(ref, 0.0)) * 0.25, sun[0], sun[1], 0, 0, noise=False,
                        full_scale=1e9)[0]
    # Without the Moon: the corona everywhere.
    truth_free = scene.corona(*sun) * nominal[ref] * (1 + errors.get(ref, 0.0)) * 0.25
    from scipy import ndimage

    truth_free = ndimage.gaussian_filter(truth_free, 0.7)
    X0, Y0, s = result.crop
    return result, hdr, weight, truth[Y0:Y0 + s, X0:X0 + s], truth_free[Y0:Y0 + s, X0:X0 + s], infos


def test_stack_recovers_shutter_errors_and_radiance():
    errors = {0: 0.08, 1: -0.05, 3: 0.12, 4: -0.10}
    result, hdr, weight, truth, truth_free, infos = _synthetic_stack(noise=False, errors=errors)
    ref_actual = 1.0
    for m in result.matches:
        expected = (1 + errors.get(m.index, 0.0)) / ref_actual * infos[m.index].nominal_ratio
        assert m.gain == pytest.approx(expected, rel=0.01), m
        assert abs(m.offset) < 2e-3
    X0, Y0, s = result.crop
    yy, xx = np.mgrid[0:s, 0:s]
    r = np.hypot(xx + X0 - 240, yy + Y0 - 240) / 40.0
    moon = infos[2]
    rm = np.hypot(xx + X0 - moon.moon_x, yy + Y0 - moon.moon_y)
    # Compare everywhere
    # outside the kept Moon, from the inner corona out to where the longest exposure still has signal.
    sel = (rm > moon.moon_r + 3) & (r < 3.5)
    lum = hdr.mean(axis=2)
    rel = np.abs(lum[sel] - truth_free[sel]) / truth_free[sel]
    assert np.median(rel) < 0.01
    assert np.percentile(rel, 99) < 0.06
    # Inside the kept Moon: that frame's own (earthshine) data.
    inside = rm < moon.moon_r - 6  # clear of the blurred limb and the demosaic kernel
    assert np.allclose(lum[inside], truth[inside], rtol=0.05, atol=2e-4)
    # With exposures 2 stops apart a thin ring next to the limb can be saturated or covered everywhere.
    assert (weight[sel] > 0).mean() > 0.97


def test_stack_with_noise_writes_outputs(tmp_path):
    result, hdr, weight, truth, truth_free, infos = _synthetic_stack(noise=True, errors={3: 0.1})
    from tse_core.stack import write_outputs

    by = {f.index: f for f in infos}
    out = write_outputs(hdr, weight, result.crop, {m.index: m for m in result.matches}, by, 2, 2, result.sun,
                        tmp_path, "test")
    data, header = read_fits(out["hdr"])
    assert data.shape == (3, result.crop[2], result.crop[2])
    assert header["SUNX"] == pytest.approx(240.0 - result.crop[0])
    assert header["MOONFRM"] == 2
    m3 = [m for m in result.matches if m.index == 3][0]
    assert m3.gain == pytest.approx(4 * 1.1, rel=0.02)
    # Faint outer corona: the stack is less noisy than the reference frame alone would be.
    X0, Y0, s = result.crop
    yy, xx = np.mgrid[0:s, 0:s]
    r = np.hypot(xx + X0 - 240, yy + Y0 - 240) / 40.0
    sel = (r > 2.5) & (r < 3.5)
    rel = (hdr.mean(axis=2)[sel] - truth_free[sel]) / truth_free[sel]
    assert abs(np.median(rel)) < 0.02
    import os

    assert os.path.exists(out["preview"])


def test_crop_box_inside_image():
    assert crop_box((100, 100), 50, (1000, 800), 5) == (0, 0, 500)
    x0, y0, s = crop_box((500, 400), 30, (1000, 800), 5)
    assert (x0, y0, s) == (350, 250, 300)


def test_cli_stack(tmp_path, monkeypatch, capsys):
    """tse stack: frames from the project, transforms from the alignment file, rejected frames skipped."""
    import json

    from conftest import make_frames
    from tse_core import cli, rawio
    from tse_core.inventory import Inventory
    from tse_core.project import EventProject

    scene = SyntheticScene(size=320, sun_radius=30.0, seed=2)
    exposures = ["1/250", "1/60", "1/15", "1/4"]
    frames = make_frames(exposures, aperture=8.0)
    for f in frames:
        f.path = str(tmp_path / f.file_name)
    shifts = [(1.0, -0.5), (0.0, 0.0), (-1.2, 0.8), (0.4, 0.3)]
    mosaics = {}
    align_frames = []
    for f, (dx, dy) in zip(frames, shifts):
        img, moon = scene.frame(f.exposure_s * 2.0, 160 + dx, 160 + dy, 0, 0, noise=False, full_scale=1.0)
        mosaics[f.path] = mosaic_from(img)
        align_frames.append({"index": f.index, "file_name": f.file_name, "matrix": [[1, 0, -dx], [0, 1, -dy]],
                             "moon_ref": [moon["moon_x"] - dx, moon["moon_y"] - dy],
                             "moon_frame": {"x": moon["moon_x"], "y": moon["moon_y"], "radius": moon["moon_r"]}})
    monkeypatch.setattr(rawio, "read_mosaic", lambda p: mosaics[p])
    sun = {"x": 160.0, "y": 160.0, "radius_px": 30.0, "arcsec_per_px": 30.0, "rotation_deg": 2.0,
           "rejected": [3]}
    align_path = tmp_path / "ecl_align.json"
    align_path.write_text(json.dumps({"reference_index": frames[1].index, "image_size": [320, 320],
                                      "frames": align_frames, "sun": sun}))
    project = EventProject(name="ecl")
    project.set_inventory(Inventory(frames))
    project.use_lensfun = False
    path = tmp_path / "ecl.json"
    project.save(path)
    assert cli.main(["stack", str(path), "--radii", "4", "--lensfun-db", str(tmp_path / "none")]) == 0
    out = capsys.readouterr().out
    assert "off the Moon's track; skipped" in out
    saved = EventProject.load(path).parameters["stack"]
    data, header = read_fits(saved["hdr"])
    assert data.shape == (3, 240, 240)
    assert header["NORTHANG"] == pytest.approx(2.0)
    info = json.loads(open(saved["info"]).read())
    gains = {m["index"]: m["gain"] for m in info["matches"]}
    assert set(gains) == {frames[0].index, frames[1].index, frames[2].index}
    assert gains[frames[2].index] == pytest.approx(4.0, rel=0.02)
