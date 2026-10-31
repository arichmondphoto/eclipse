"""Prominences: found by colour at the limb, never on the Moon, chromosphere optional."""

import numpy as np
import pytest
from scipy import ndimage

from tse_core.prominences import ProminenceParams, find_prominences, place, sun_region
from tse_core.synthetic import SyntheticScene

H, W = 500, 700
SX, SY, R = 350.0, 250.0, 60.0
MX, MY, MR = 353.0, 250.0, 63.0  # the Moon a little east: a thin crescent of chromosphere shows in the west


def _hdr(seed=1):
    sc = SyntheticScene(size=W, sun_radius=R, seed=4)
    corona = sc.corona(SX, SY, shape=(H, W)).astype(np.float32) * 1e-3
    rgb = np.stack([corona, corona, corona], axis=-1)  # white corona
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    r = np.hypot(xx - SX, yy - SY)
    phi = np.degrees(np.arctan2(yy - SY, xx - SX)) % 360
    red = np.zeros((H, W), np.float32)
    proms = [(30, 14, 4), (150, 30, 6), (250, 8, 3)]  # (angle, height px, half-width deg)
    for ang, hgt, half in proms:
        d = np.minimum(np.abs(phi - ang), 360 - np.abs(phi - ang))
        red += ((r > R) & (r < R + hgt) & (d < half)).astype(np.float32) * 0.6
    chromo = ((r > R) & (r < R + 2.5)).astype(np.float32) * 0.4  # thin rim all round (mostly behind the Moon)
    red = ndimage.gaussian_filter(red + chromo, 0.8)
    rgb[..., 0] += red * 1.0  # H-alpha
    rgb[..., 2] += red * 0.35  # H-beta: pink
    rng = np.random.default_rng(seed)
    rgb += rng.normal(0, 2e-4, rgb.shape).astype(np.float32) * np.sqrt(1 + rgb / 1e-3)
    # The Moon frame's disc: black, except a red blob that must never show (it would sit on the earthshine).
    disc = np.hypot(xx - MX, yy - MY) < MR
    rgb[disc] = 0.0
    blob = np.hypot(xx - (MX + 30), yy - MY) < 6
    rgb[blob, 0] = 0.5
    header = {"SUNX": SX, "SUNY": SY, "SUNR": R, "MOONX": MX, "MOONY": MY, "MOONR": MR, "PIXSCALE": 2.4}  # like a 400 mm lens on a D850
    return rgb, np.ones((H, W), np.float32), header, r, phi


def _at(alpha, region, x, y):
    x0, y0 = region.origin
    return alpha[int(y - y0), int(x - x0)]


def test_finds_prominences_not_corona_not_moon():
    rgb, wt, hdr, r, phi = _hdr()
    region = sun_region(rgb, wt, hdr)
    res, layer, alpha = find_prominences(region, ProminenceParams())
    for ang, hgt in ((30, 14), (150, 30)):
        x = SX + (R + hgt / 2) * np.cos(np.radians(ang))
        y = SY + (R + hgt / 2) * np.sin(np.radians(ang))
        assert _at(alpha, region, x, y) > 0.8, ang
    # White corona between prominences: nothing.
    x0, y0 = region.origin
    yy, xx = np.mgrid[0 : alpha.shape[0], 0 : alpha.shape[1]]
    rr = np.hypot(xx + x0 - SX, yy + y0 - SY)
    pp = np.degrees(np.arctan2(yy + y0 - SY, xx + x0 - SX)) % 360
    corona = (rr > R + 6) & (rr < R + 25) & (np.abs(((pp - 90 + 180) % 360) - 180) < 20)
    assert (alpha[corona] > 0.2).mean() < 0.01
    # Never on the Moon (the red blob there stays out).
    on_moon = np.hypot(xx + x0 - MX, yy + y0 - MY) < MR
    assert alpha[on_moon].max() == 0
    # Pink: red above blue above green.
    k = alpha > 0.9
    m = layer[k].mean(axis=0)
    assert m[0] > m[2] > m[1]
    assert res.count >= 3


def test_chromosphere_optional():
    rgb, wt, hdr, r, phi = _hdr()
    region = sun_region(rgb, wt, hdr)
    # The uncovered crescent of the rim, in the west (angle 180), away from any prominence.
    x, y = SX - (R + 1.3), SY
    _, _, with_c = find_prominences(region, ProminenceParams(chromosphere=True))
    res, _, without_c = find_prominences(region, ProminenceParams(chromosphere=False))
    assert _at(with_c, region, x, y) > 0.5
    assert _at(without_c, region, x, y) < 0.1
    # The tall prominence keeps its base.
    bx = SX + (R + 2) * np.cos(np.radians(150))
    by = SY + (R + 2) * np.sin(np.radians(150))
    assert _at(without_c, region, bx, by) > 0.5


def test_place_on_canvas():
    rgb, wt, hdr, r, phi = _hdr()
    region = sun_region(rgb, wt, hdr)
    _, layer, _ = find_prominences(region)
    full = place(region, layer)
    assert full.shape == (H, W, 3)
    assert full[0, 0].max() == 0


def test_composite_top_normal_keeps_colour_over_bright_corona():
    from tse_core.prominences import composite_top

    base = np.full((4, 4, 3), 0.9, np.float32)  # bright inner corona
    layer = np.zeros_like(base)
    layer[1:3, 1:3] = [0.9, 0.2, 0.5]  # pink prominence
    alpha = np.zeros((4, 4), np.float32)
    alpha[1:3, 1:3] = 1.0
    out = composite_top(base, layer, alpha)
    assert np.allclose(out[1, 1], [0.9, 0.2, 0.5]) and np.allclose(out[0, 0], 0.9)
    scr = composite_top(base, layer, alpha, mode="screen")
    assert scr[1, 1, 1] > 0.9  # screen washes it out: why normal is the default
    half = composite_top(base, layer, alpha, strength=0.5)
    assert np.allclose(half[1, 1], [0.9, 0.55, 0.7])


def test_workflow_layer_and_final(tmp_path):
    from tse_core import imageout as IO
    from tse_core import workflow as WF
    from tse_core.fits import write_fits
    from tse_core.project import EventProject

    rgb, wt, hdr, r, phi = _hdr()
    write_fits(tmp_path / "e_hdr.fit", np.moveaxis(rgb, -1, 0), hdr)
    write_fits(tmp_path / "e_weight.fit", wt)
    IO.write_image(tmp_path / "e_final.tif", np.full((H, W, 3), 0.3, np.float32))
    p = EventProject(name="e")
    p.parameters["stack"] = {"hdr": "e_hdr.fit", "weight": "e_weight.fit"}
    p.parameters["final"] = {"file": "e_final.tif"}
    p.save(tmp_path / "e.json")
    res, layer, alpha = WF.prominence_layer(tmp_path / "e.json", preview=True)
    assert res.count >= 3 and layer.shape[:2] == alpha.shape
    res = WF.prominence_layer(tmp_path / "e.json", output_file=str(tmp_path / "out" / "proms.png"))
    lay = IO.read_image(res.outputs["prominences"], alpha=True)
    assert lay.shape == (H, W, 4) and lay[0, 0, 3] == 0  # transparent away from prominences
    prev = WF.final_with_prominences(tmp_path / "e.json", preview_side=350)
    assert prev.shape == (250, 350, 3)
    out = WF.final_with_prominences(tmp_path / "e.json", output_file=str(tmp_path / "f.jpg"))
    img = IO.read_image(out)
    x = int(SX + (R + 7) * np.cos(np.radians(30)))
    y = int(SY + (R + 7) * np.sin(np.radians(30)))
    assert img[y, x, 0] > img[y, x, 1] + 0.1  # a pink prominence over the grey base
    assert abs(img[5, 5, 0] - 0.3) < 0.01
    with pytest.raises(WF.WorkflowError, match="'natural' layer"):
        WF.final_with_prominences(tmp_path / "e.json", base="natural")


def test_cli(tmp_path, capsys):
    from tse_core import cli
    from tse_core import imageout as IO
    from tse_core.fits import write_fits
    from tse_core.project import EventProject

    rgb, wt, hdr, r, phi = _hdr()
    write_fits(tmp_path / "e_hdr.fit", np.moveaxis(rgb, -1, 0), hdr)
    write_fits(tmp_path / "e_weight.fit", wt)
    IO.write_image(tmp_path / "e_blend.tif", np.full((H, W, 3), 0.3, np.float32))
    p = EventProject(name="e")
    p.parameters["stack"] = {"hdr": "e_hdr.fit", "weight": "e_weight.fit"}
    p.parameters["enhance"] = {"blend": "e_blend.tif"}
    p.save(tmp_path / "e.json")
    assert cli.main(["prominences", str(tmp_path / "e.json"), "--no-chromosphere", "--quiet"]) == 0
    assert "Prominences found" in capsys.readouterr().out
    assert cli.main(["final-prominences", str(tmp_path / "e.json"), "--base", "blend", "-o",
                     str(tmp_path / "x.png")]) == 0
    assert IO.read_image(tmp_path / "x.png").shape == (H, W, 3)
    assert cli.main(["final-prominences", str(tmp_path / "e.json")]) == 1  # no final image yet: a message
