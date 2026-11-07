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
    # The opacity is the prominence's share of the light (no outline is drawn): here its red and
    # blue against a corona about as bright, so about 2/3.
    for ang, hgt in ((30, 14), (150, 30)):
        x = SX + (R + hgt / 2) * np.cos(np.radians(ang))
        y = SY + (R + hgt / 2) * np.sin(np.radians(ang))
        assert _at(alpha, region, x, y) > 0.5, ang
        assert _at(layer[..., 0], region, x, y) > 0.3, ang
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
    k = alpha > 0.5
    m = layer[k].mean(axis=0)
    assert m[0] > m[2] > m[1]
    assert res.count >= 3


def test_chromosphere_optional():
    rgb, wt, hdr, r, phi = _hdr()
    region = sun_region(rgb, wt, hdr)
    # The uncovered crescent of the rim, in the west (angle 180), away from any prominence.
    x, y = SX - (R + 1.3), SY
    # (The layer itself: by the limb the corona is brightest, so the rim's share of the light is small.)
    _, with_c, _ = find_prominences(region, ProminenceParams(chromosphere=True))
    res, without_c, _ = find_prominences(region, ProminenceParams(chromosphere=False))
    with_c, without_c = with_c[..., 0], without_c[..., 0]
    assert _at(with_c, region, x, y) > 0.2
    assert _at(without_c, region, x, y) < 0.02
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


def _stacked_project(tmp_path, monkeypatch, exposures=("1/1000", "1/500", "1/250", "1/60", "1/15")):
    """A project with RAW frames (synthetic mosaics), aligned and stacked as the Stack step does."""
    import json

    from conftest import make_frames
    from tse_core import rawio
    from tse_core import workflow as W
    from tse_core.inventory import Inventory
    from tse_core.project import EventProject
    from tse_core.synthetic import SyntheticScene

    from tse_core.demosaic import channel_masks
    from tse_core.rawio import RawMosaic

    scene = SyntheticScene(size=400, sun_radius=60.0, seed=2)
    frames = make_frames(list(exposures), aperture=8.0)
    yy, xx = np.mgrid[0:400, 0:400]

    def prominence(dx, dy):  # pink (H-alpha and H-beta) at 3 o'clock, 10 px tall, as bright as the limb
        r = np.hypot(xx - 200 - dx, yy - 200 - dy)
        ang = np.arctan2(yy - 200 - dy, xx - 200 - dx)
        blob = ((r > 60) & (r < 73) & (np.abs(ang) < 0.08)).astype(np.float32) * 2400.0
        return ndimage.gaussian_filter(blob, 0.7)

    def mosaic(grey, red):
        masks = channel_masks("RGGB", grey.shape)
        raw = np.empty(grey.shape, np.float32)
        for c, k in (("R", 1.0), ("G", 0.0), ("B", 0.35)):
            raw[masks[c]] = (grey + k * red)[masks[c]]
        black, white = 600.0, 15520.0
        raw = np.minimum(black + np.minimum(raw, 1.0) * (white - black), white)
        return RawMosaic(raw.astype(np.float32), "RGGB", {"R": black, "G": black, "B": black}, white, (2.0, 1.0, 1.5))
    shifts = [(1.0, -0.5), (0.0, 0.0), (-1.2, 0.8), (0.4, 0.3), (0.8, -0.9)]
    mosaics, align_frames = {}, []
    for f, (dx, dy) in zip(frames, shifts):
        f.path = str(tmp_path / f.file_name)
        (tmp_path / f.file_name).write_bytes(b"")
        # the corona peaks at 0.5 of full scale in the 1/250 s frame: the longer ones saturate by the limb
        img, moon = scene.frame(f.exposure_s * 0.05, 200 + dx, 200 + dy, 0, 0, noise=False, full_scale=1.0)
        disc = np.clip(np.hypot(xx - moon["moon_x"], yy - moon["moon_y"]) - moon["moon_r"] + 0.5, 0, 1)
        mosaics[f.path] = mosaic(img, prominence(dx, dy) * disc * f.exposure_s * 0.05)
        align_frames.append({"index": f.index, "file_name": f.file_name, "matrix": [[1, 0, -dx], [0, 1, -dy]],
                             "moon_ref": [moon["moon_x"] - dx, moon["moon_y"] - dy],
                             "moon_frame": {"x": moon["moon_x"], "y": moon["moon_y"], "radius": moon["moon_r"]}})
    monkeypatch.setattr(rawio, "read_mosaic", lambda p: mosaics[p])
    sun = {"x": 200.0, "y": 200.0, "radius_px": 60.0, "arcsec_per_px": 16.0, "rotation_deg": 0.0}
    # The reference a middle exposure, not one of the short frames (as on the 2024 set: 1/10 s)
    (tmp_path / "e_align.json").write_text(json.dumps({"reference_index": frames[3].index, "image_size": [400, 400],
                                                       "frames": align_frames, "sun": sun}))
    p = EventProject(name="e")
    p.set_inventory(Inventory(frames))
    p.use_lensfun = False
    path = tmp_path / "e.json"
    p.save(path)
    W.stack(path, W.StackOptions(align=str(tmp_path / "e_align.json"), lensfun_db=str(tmp_path / "none")))
    return path, frames


def test_prominence_stack_uses_the_shortest_frames_in_the_stacks_units(tmp_path, monkeypatch):
    """The prominences are sharpest in the shortest frames (2024: edges 2.2-2.6 px at 1/1600-1/1000 s
    against 3.25 px in the HDR stack). Stack prominences takes the frames up to 2 stops longer than
    the shortest, with the HDR stack's own gains, so it is in the same units; and measures the blur
    on each frame's own Moon edge (the synthetic optics: sigma 0.7 px, and the demosaicing)."""
    from tse_core import workflow as W
    from tse_core.fits import read_fits
    from tse_core.project import EventProject

    path, frames = _stacked_project(tmp_path, monkeypatch)
    msgs = []
    entry = W.prominence_stack(path, say=msgs.append)
    assert entry["frames"] == [frames[0].index, frames[1].index, frames[2].index]
    assert 0.5 < entry["blur_px"] < 1.4, entry["blur_px"]
    assert any("blur" in m and "px" in m for m in msgs)
    st = EventProject.load(path).parameters
    hdr, hh = read_fits(st["stack"]["hdr"])
    pr, ph = read_fits(st["prominence_stack"]["hdr"])
    x0, y0 = int(ph["CROPX0"]), int(ph["CROPY0"])
    assert pr.shape[1] < hdr.shape[1] and x0 > 0          # only round the Sun
    yy, xx = np.mgrid[0:pr.shape[1], 0:pr.shape[2]]
    r = np.hypot(xx + x0 - 200, yy + y0 - 200)
    ring = (r > 60 * 1.15) & (r < 60 * 1.4)
    ratio = pr[1][ring] / hdr[1, y0:y0 + pr.shape[1], x0:x0 + pr.shape[2]][ring]
    assert abs(float(np.median(ratio)) - 1) < 0.02


def test_prominence_layer_from_the_prominence_stack(tmp_path, monkeypatch):
    """Write prominence layer after Stack prominences: the layer comes from the short frames, the
    size of the HDR stack, with the prominence's own light only (the corona taken off), and
    nothing on the HDR stack's Moon (the one in the final image)."""
    from tse_core import imageout as IO
    from tse_core import workflow as W
    from tse_core.fits import read_fits_header
    from tse_core.project import EventProject

    path, frames = _stacked_project(tmp_path, monkeypatch)
    W.prominence_stack(path)
    msgs = []
    res = W.prominence_layer(path, output_file=str(tmp_path / "p.tif"), say=msgs.append)
    assert any("prominence stack" in m for m in msgs)
    lay = IO.read_image(res.outputs["prominences"], alpha=True)
    assert lay.shape == (400, 400, 4)
    head = read_fits_header(EventProject.load(path).parameters["stack"]["hdr"])
    yy, xx = np.mgrid[0:400, 0:400]
    on_moon = np.hypot(xx - head["MOONX"], yy - head["MOONY"]) < head["MOONR"]
    assert lay[on_moon].max() == 0
    r = np.hypot(xx - 200, yy - 200)
    prom = (r > 63.5) & (r < 70) & (np.abs(np.arctan2(yy - 200, xx - 200)) < 0.04)
    assert np.median(lay[prom][:, 0]) > 0.3                            # pink: red over green
    assert np.median(lay[prom][:, 0]) > np.median(lay[prom][:, 1]) + 0.2
    corona = (r > 66) & (r < 80) & (np.abs(np.arctan2(yy - 200, xx - 200)) > 0.5)
    assert np.percentile(lay[corona][:, :3].max(axis=1), 99) < 0.05   # the white corona taken off



def test_cli_prominence_stack_and_final_settings(tmp_path, monkeypatch, capsys):
    """totality prominence-stack: stacks the short frames and saves the settings 'totality final' uses."""
    from tse_core import cli
    from tse_core.project import EventProject

    path, frames = _stacked_project(tmp_path, monkeypatch)
    assert cli.main(["prominence-stack", str(path), "--brightness", "0.3", "--quiet"]) == 0
    out = capsys.readouterr().out
    assert "Prominence stack: 3 frame(s)" in out
    pr = EventProject.load(path).parameters
    assert pr["prominences"]["params"]["brightness"] == 0.3 and pr["prominence_stack"]["use"] is True
    assert cli.main(["prominence-stack", str(path), "--settings-only", "--off"]) == 0
    assert EventProject.load(path).parameters["prominence_stack"]["use"] is False



def test_moon_cover_says_when_the_chromosphere_is_hidden():
    """Andy, Nov 2026: Include the chromosphere and Rim height changed nothing. In his final image
    the Moon (436 px) covers the Sun's edge (413 px) all round, so the rim never shows; the tab now
    says so. Near a contact the Sun's edge shows on one side."""
    from tse_core.enhance import Geometry
    from tse_core.prominences import moon_cover_px

    mid = Geometry(3966.1, 2793.8, 413.25, 3972.4, 2798.3, 435.9)  # the 2024 stack's header
    assert 14.0 < moon_cover_px(mid) < 16.0                       # 22.7 px larger, 7.7 px off centre
    near_contact = Geometry(100.0, 100.0, 50.0, 140.0, 100.0, 52.0)
    assert moon_cover_px(near_contact) < 0


def _soft_and_sharp(seed=5, extra=1.0, gap=0.0):
    """The 2024 situation, scaled: a prominence as bright as on the 2024 stack (brighter than the
    corona beside it), sharp in the short frames (with their noise), softer by ``extra`` px in the
    HDR stack; the short frames' Moon covering ``gap`` px more outside the final image's limb."""
    rgb, wt, head, r, phi = _hdr()
    corona = rgb[..., 1].copy()
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    strands = (np.sin(np.arctan2(yy - SY, xx - SX) * 400) > 0.3)          # fine structure across it
    shape = ((r > R) & (r < R + 15) & (np.abs(((phi - 90 + 180) % 360) - 180) < 6) & strands).astype(np.float32) * 2.5
    colour = np.float32([1.0, 0.5, 0.35])                                 # with some green (helium)
    sharp = ndimage.gaussian_filter(shape, 0.8)
    base = np.stack([corona, corona, corona], axis=-1)
    hdr = base + ndimage.gaussian_filter(sharp, extra)[..., None] * colour
    prom = base + sharp[..., None] * colour
    prom += np.random.default_rng(seed).normal(0, 4e-3, prom.shape).astype(np.float32)
    d_moon = np.hypot(xx - MX, yy - MY) - MR
    hdr[d_moon < 0] = 0.0
    pw = np.ones((H, W), np.float32)
    pw[d_moon < gap] = 0.0
    prom[d_moon < gap] = 0.0
    return hdr, prom, pw, sharp, base, colour, r, phi, d_moon, xx, yy


def test_sharpening_undoes_the_hdr_stacks_extra_blur():
    """Andy, Nov 2026: the prominences are already in the HDR stack (little noise, right down to the
    limb), only soft; use the short frames to sharpen them, not to replace them. The blur that turns
    the short frames' prominence into the HDR stack's is measured (2024: 0.6-1.3 px) and undone in
    the HDR stack only where the prominences are. Sharpening 0.5 brings it to the short frames'
    sharpness; the corona elsewhere and the Moon are left as they were."""
    from tse_core.enhance import Geometry
    from tse_core.prominences import ProminenceParams, sharpen_prominences

    hdr, prom, pw, sharp, base, colour, r, phi, d_moon, xx, yy = _soft_and_sharp(extra=1.0, gap=4.0)
    before = hdr.copy()
    info = sharpen_prominences(hdr, np.ones((H, W), np.float32), prom, pw, Geometry(SX, SY, R, MX, MY, MR),
                               ProminenceParams(sharpen=0.5), blur_px=0.8)
    assert info["blurs"] and abs(info["blurs"][0] - 1.0) < 0.25, info
    prom_zone = (r > R + 5) & (r < R + 13) & (np.abs(((phi - 90 + 180) % 360) - 180) < 4)
    want = base[..., 0] + sharp * colour[0]
    err_before = np.abs(before[..., 0] - want)[prom_zone].mean()
    err_after = np.abs(hdr[..., 0] - want)[prom_zone].mean()
    assert err_after < 0.5 * err_before, (err_before, err_after)     # strands brought back
    away = (np.abs(((phi - 90 + 180) % 360) - 180) > 30) & (d_moon > 0)
    assert np.abs(hdr[away] - before[away]).max() < 1e-6              # the corona elsewhere untouched
    assert np.array_equal(hdr[d_moon < 0], before[d_moon < 0])         # nothing on the Moon
    # Where the short frames have no data (their Moon), the HDR stack's prominence is still there.
    band = (d_moon > 0.5) & (d_moon < 3.5) & (np.abs(xx - SX) < 2) & (yy > SY)
    kept = (hdr[..., 0] - base[..., 0])[band].sum() / (before[..., 0] - base[..., 0])[band].sum()
    assert kept > 0.8, kept


def test_prominence_brightness_takes_their_glow_too():
    """At Prominence brightness 0.3 a pale outline was left round the prominences: their faint red
    glow is below the colour test, so it stayed at full brightness. Their glow's colour excess is
    dimmed too (the corona's own light there is left)."""
    from tse_core.enhance import Geometry
    from tse_core.prominences import ProminenceParams, sharpen_prominences

    hdr, prom, pw, sharp, base, colour, r, phi, d_moon, xx, yy = _soft_and_sharp(extra=1.0)
    hdr += (0.02 * ndimage.gaussian_filter(sharp, 4.0))[..., None] * colour   # a faint glow round it
    before = hdr.copy()
    sharpen_prominences(hdr, np.ones((H, W), np.float32), prom, pw, Geometry(SX, SY, R, MX, MY, MR),
                        ProminenceParams(sharpen=0.0, brightness=0.3), blur_px=0.8)
    ring = (r > R + 17) & (r < R + 21) & (np.abs(((phi - 90 + 180) % 360) - 180) < 4)
    red_before = (before[..., 0] - before[..., 1])[ring]
    red_after = (hdr[..., 0] - hdr[..., 1])[ring]
    assert red_before.max() > 0.01
    assert (red_after < 0.5 * red_before + 2e-3).all()
    assert np.allclose(hdr[..., 1][ring], before[..., 1][ring], atol=1e-6)    # the corona's own light kept
