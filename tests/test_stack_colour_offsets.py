"""Brightness matching fits each frame's offset per colour.

The matching between brackets finds one gain per frame (the exposure ratio, the same for every
colour) and an offset for the light that does not scale with exposure: twilight that changes during
totality, scattered light, small black-level differences. That light has a colour of its own (the
eclipse sky is blue). Until Nov 2026 the offset was fitted on the mean of the three colours and
taken off red, green and blue alike, so blue additive light left red over-corrected. On Andy's 2024
stack the red channel was below zero over 87% of the sky between 6 and 9 solar radii and 99.8%
beyond (green and blue positive); with the sky removed this was hidden, with "Keep the sky" the
sky turned cyan and the corners fell to black.
"""

import json

import numpy as np

from conftest import make_frames
from tse_core import metadata, rawio
from tse_core.fits import read_fits
from tse_core.inventory import Inventory
from tse_core.project import EventProject
from tse_core.rawio import RawMosaic
from tse_core.synthetic import SyntheticScene

BLACK, WHITE = 400.0, 15520.0
SIZE = 320
BLUE = np.array([0.25, 1.0, 1.6])  # additive light per colour (R, G, B): the eclipse sky's colour


def _cfa(rgb):
    """Per-pixel values of an RGGB mosaic from one value per colour."""
    m = np.empty((SIZE, SIZE), np.float32)
    m[0::2, 0::2], m[0::2, 1::2], m[1::2, 0::2], m[1::2, 1::2] = rgb[0], rgb[1], rgb[1], rgb[2]
    return m


def _stack(tmp_path, monkeypatch, extra):
    """Stack three frames; ``extra[k]`` is frame k's additive light (fraction of full scale) per colour."""
    from tse_core.workflow import StackOptions, stack

    scene = SyntheticScene(size=SIZE, sun_radius=30.0, seed=2)
    lights = make_frames(["1/250", "1/60", "1/15"], aperture=8.0, iso=200)
    mosaics, align = {}, []
    for k, f in enumerate(lights):
        f.path = str(tmp_path / f.file_name)
        (tmp_path / f.file_name).write_bytes(b"")
        img, moon = scene.frame(f.exposure_s * 2.0, 160, 160, 0, 0, noise=False, full_scale=1.0)
        raw = BLACK + (img + _cfa(extra[k])) * (WHITE - BLACK)
        mosaics[f.path] = RawMosaic(np.minimum(raw, WHITE).astype(np.float32), "RGGB",
                                    {"R": BLACK, "G": BLACK, "B": BLACK}, WHITE, (2.0, 1.0, 1.5))
        align.append({"index": f.index, "file_name": f.file_name, "matrix": [[1, 0, 0], [0, 1, 0]],
                      "moon_ref": [moon["moon_x"], moon["moon_y"]],
                      "moon_frame": {"x": moon["moon_x"], "y": moon["moon_y"], "radius": moon["moon_r"]}})
    project = EventProject(name="ecl")
    project.set_inventory(Inventory(lights))
    project.use_lensfun = False
    monkeypatch.setattr(rawio, "read_mosaic", lambda p: mosaics[p])
    monkeypatch.setattr(metadata, "read_frames", lambda paths, **kw: [m for m in lights if m.path in map(str, paths)])
    (tmp_path / "ecl_align.json").write_text(json.dumps(
        {"reference_index": lights[1].index, "image_size": [SIZE, SIZE], "frames": align,
         "sun": {"x": 160.0, "y": 160.0, "radius_px": 30.0, "arcsec_per_px": 30.0, "rotation_deg": 0.0,
                 "rejected": []}}))
    path = tmp_path / "ecl.json"
    project.save(path)
    msgs = []
    stack(path, StackOptions(lensfun_db=str(tmp_path / "none"), hot_pixels=False), say=msgs.append)
    data, _ = read_fits(EventProject.load(path).parameters["stack"]["hdr"])
    return data, "\n".join(msgs)


def test_blue_additive_light_is_taken_off_each_colour(tmp_path, monkeypatch):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    truth, _ = _stack(tmp_path / "a", monkeypatch, [np.zeros(3)] * 3)
    # The short and long frames carry blue light the reference (1/60 s) does not, as bright as the
    # faint corona further out (on the 2024 stack the sky matched the corona by 4-6 solar radii).
    got, log = _stack(tmp_path / "b", monkeypatch, [0.012 * BLUE, np.zeros(3), 0.024 * BLUE])
    yy, xx = np.mgrid[0:truth.shape[1], 0:truth.shape[2]]
    r = np.hypot(xx - truth.shape[2] / 2, yy - truth.shape[1] / 2)
    far = (r > 110) & (r < 150)  # the faint outer corona, 3.7-5 solar radii
    # Since Nov 2026 the stack's zero comes from the longest exposures (see test_stack_zero.py), so the
    # longest frame's own extra light (0.024 x BLUE of its full scale, gain 4 against the reference)
    # stays in the sky: expect the stack to differ from the truth by exactly that, per colour.
    zero = 0.024 * BLUE / 4.0
    for c, name in enumerate("RGB"):
        level = float(np.median(truth[c][far]))
        err = float(np.median(got[c][far] - truth[c][far])) - zero[c]
        # Each colour as without the extra light, to 5% of the faint corona there. The old shared
        # offset (fitted on the mean colour) left red low and blue high by about a third of it.
        assert abs(err) < 0.05 * level, (name, err, level)
