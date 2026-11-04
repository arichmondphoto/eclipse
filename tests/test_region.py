"""The full-resolution detail box: rendering a region must give exactly what the full-size write
gives there, for every layer and the blend.

Everything measured over the whole image (sky, white balance, noise, dust spots, circle
statistics, the natural layer's black and white points and profiles, the gated noise model) is
measured once by ``prepare`` and used by both; the local filters run on the box plus a margin
as wide as their reach; the filter layers and the blend have fixed output scales.
"""

from pathlib import Path

import numpy as np
import pytest

from tse_core import enhance as E

DATA = Path(__file__).parent / "data" / "eclipse2024_bin8.npz"

PARAMS = E.EnhanceParams(
    layers=E.LAYERS, flatten=0.8, even=0.5, midtone=0.6, shoulder=0.3, noise_gate=3.0, saturation=1.2,
    blend={"natural": 1.0, "nrgf": 0.2, "fnrgf": 0.3, "fnrgf_n": 0.6, "adaptive": 0.5, "achf": 0.4},
    blend_modes={"adaptive": "overlay"},
    achf_weights=(1.0, 1.0, 1.0, 1.0, 0.5, 0.5),  # the large (circular) scales on as well
)


@pytest.fixture(scope="module")
def real():
    z = np.load(DATA)
    hdr = np.ascontiguousarray(np.moveaxis(z["hdr"].astype(np.float32) * float(z["scale"]), 0, -1))
    geo = {k: float(z[k]) for k in ("SUNX", "SUNY", "SUNR", "MOONX", "MOONY", "MOONR")}
    weight = (np.abs(hdr).sum(axis=2) > 0).astype(np.float32)
    return hdr, weight, geo


def _boxes(geo, shape):
    h, w = shape
    sx, sy, sr = geo["SUNX"], geo["SUNY"], geo["SUNR"]
    return [
        (int(sy - 0.6 * sr), int(sy + 0.9 * sr) + 3, int(sx + 0.4 * sr) + 1, int(sx + 2.1 * sr)),  # through the limb
        (0, 101, w - 137, w),  # the frame's corner
        (int(sy - 3.3 * sr) + 2, int(sy - 2.1 * sr), int(sx - 1.7 * sr), int(sx - 0.2 * sr) - 5),  # outer corona
    ]


def test_box_matches_the_full_render_on_the_2024_stack(real):
    hdr, weight, geo = real
    frame = E.prepare(hdr.copy(), weight, geo, PARAMS)
    full = E.render(frame, PARAMS)
    assert set(full) == set(E.LAYERS) | {"blend"}
    for box in _boxes(geo, hdr.shape[:2]):
        y0, y1, x0, x1 = box
        part = E.render(frame, PARAMS, box=box)
        for k, img in full.items():
            d = np.abs(part[k] - img[y0:y1, x0:x1])
            assert part[k].shape == (y1 - y0, x1 - x0, 3)
            # A thousandth of the output range: a quarter of one 8-bit step (Gaussian tails beyond
            # the margin are all that can differ).
            assert d.max() < 1e-3, (k, box, float(d.max()))


def test_enhance_is_prepare_then_render(real):
    """The full-size write goes through the same two steps, so it is the same picture."""
    hdr, weight, geo = real
    p = E.EnhanceParams(layers=("natural", "adaptive"), blend={"natural": 1.0, "adaptive": 0.5})
    _, layers = E.enhance(hdr.copy(), weight, geo, p)
    frame = E.prepare(hdr.copy(), weight, geo, p)
    again = E.render(frame, p)
    for k in layers:
        assert np.abs(layers[k] - again[k]).max() < 1e-6, k


def test_measured_once_reused_after_changes(real):
    """Changing a tone setting re-measures only what depends on it; the frame is not prepared again."""
    hdr, weight, geo = real
    p = E.EnhanceParams(layers=("natural",))
    frame = E.prepare(hdr.copy(), weight, geo, p)
    E.render(frame, p, box=(100, 200, 300, 420))
    keys = set(frame.cache)
    from dataclasses import replace

    E.render(frame, replace(p, flatten=0.5), box=(100, 200, 300, 420))
    assert keys < set(frame.cache)  # one more natural-statistics entry, the rest reused
