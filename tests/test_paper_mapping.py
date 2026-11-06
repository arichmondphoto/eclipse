"""The filter layers as the papers define them, with no display curve of their own (Andy, 4 Nov
2026: "as close to the way they were implemented in the papers as possible").

* MGN (Morgan & Druckmueller 2014, eq. 5): the local part is the mean over the scales of
  arctan(k C_i), combined linearly with the global image (our blend). Its range is (-pi/2, pi/2),
  so the layer is 0.5 + mean / pi: no constant of ours at all.
* NRGF (Morgan, Habbal & Woo 2006) and FNRGF / FNRGF-N (Druckmuellerova, Morgan & Habbal 2011,
  eqs. 7 and 9): Y is combined linearly, Z = K1 X + K2 Y, with the same K2 for every image. The
  layer is Y on a straight line about mid-grey, clipped where it leaves 0..1; the blend strength
  plays K2.

Until now each went through an extra arctan S-curve (1.6x the contrast near mid-grey, compressed
towards black and white) that the papers do not have.
"""

from pathlib import Path

import numpy as np
from scipy import ndimage

from tse_core import enhance as E

DATA = Path(__file__).parent / "data" / "eclipse2024_bin8.npz"


def test_mgn_layer_is_the_papers_mean_of_arctan_on_a_straight_line():
    """B is the original image, linear (Morgan & Druckmueller 2014: "the original image normalised
    by exposure time", negative pixels replaced first). Until 6 Nov 2026 the layer used log(B)."""
    rng = np.random.default_rng(4)
    x = np.exp(ndimage.gaussian_filter(rng.normal(size=(160, 200)), 2.0)).astype(np.float32)
    x[5, 5] = -0.3   # a spurious negative pixel: replaced by 0
    valid = np.ones(x.shape, bool)
    scales, k = [1.5, 3.0, 6.0], 0.7
    b = np.maximum(x, 0).astype(np.float32)
    acc = np.zeros_like(b)
    for w in scales:  # the paper's C_i: (B - mean) / local deviation, at each scale
        d = b - ndimage.gaussian_filter(b, w, mode="reflect")
        sd = np.sqrt(np.maximum(ndimage.gaussian_filter(d * d, w, mode="reflect"), 1e-12))
        acc += np.arctan(k * d / sd)
    expect = 0.5 + acc / len(scales) / np.pi
    got = E.mgn(x, valid, scales, k, 1e-6)
    assert np.abs(got - expect).max() < 1e-5
    # The noise-gated version with the gate off is the same filter, so the same line.
    gated = E.mgn_gated(x, valid, scales, k, 1e-6, np.zeros_like(x), 50.0, gate=0.0)
    assert np.abs(gated - expect).max() < 1e-5


def test_nrgf_and_fnrgf_layers_are_y_on_a_straight_line(real=None):
    z = np.load(DATA)
    hdr = np.ascontiguousarray(np.moveaxis(z["hdr"].astype(np.float32) * float(z["scale"]), 0, -1))
    geo = {k: float(z[k]) for k in ("SUNX", "SUNY", "SUNR", "MOONX", "MOONY", "MOONR")}
    weight = (np.abs(hdr).sum(axis=2) > 0).astype(np.float32)
    p = E.EnhanceParams(layers=("nrgf", "fnrgf", "fnrgf_n"), saturation=0.0, blend={})  # grey: no colour
    frame = E.prepare(hdr, weight, geo, p)
    out = E.render(frame, p)
    r, phi = E.polar(frame.shape, frame.g)
    stats = E._ring_cache(frame, p)
    att = E._attenuation(None, p.order)
    ys = {"nrgf": (E.nrgf(frame.lum, r, frame.valid, stats), E.NRGF_SPREAD),
          "fnrgf": (E.fnrgf(frame.lum, r, phi, frame.valid, stats, att, att), E.FNRGF_SPREAD),
          "fnrgf_n": (E.fnrgf(frame.lum, r, phi, frame.valid, stats, att, att, p.noise_m), E.FNRGF_N_SPREAD)}
    v = frame.valid
    for k, (y, s) in ys.items():
        expect = np.clip(0.5 + y / (2 * s), 0.0, 1.0)
        assert np.abs(out[k][..., 0][v] - expect[v]).max() < 1e-5, k


def test_mgn_default_scales_and_weights_are_the_papers():
    """Morgan & Druckmueller (2014): "w = 1.25, 2.5, 5, 10, 20, 40" px; the weights g_i "may be
    adjusted", or "set equal for all scales so that a straightforward mean" results."""
    p = E.EnhanceParams()
    assert tuple(p.mgn_scales) == (1.25, 2.5, 5.0, 10.0, 20.0, 40.0)
    assert tuple(p.mgn_weights) == (1.0,) * 6


def test_mgn_scale_weights_as_in_the_paper():
    """(1 - h) / n * sum g_i C'_i: each scale's arctan times its weight, over the number of scales;
    a weight of 0 leaves the scale out (and lowers the layer, as in the paper)."""
    rng = np.random.default_rng(5)
    x = np.exp(ndimage.gaussian_filter(rng.normal(size=(120, 150)), 1.5)).astype(np.float32)
    valid = np.ones(x.shape, bool)
    scales, k, g = [1.5, 3.0, 6.0], 0.7, [0.25, 1.0, 0.0]
    b = x.astype(np.float32)   # B: the linear image, as in the paper
    acc = np.zeros_like(b)
    for w, gi in zip(scales, g):
        d = b - ndimage.gaussian_filter(b, w, mode="reflect")
        sd = np.sqrt(np.maximum(ndimage.gaussian_filter(d * d, w, mode="reflect"), 1e-12))
        acc += gi * np.arctan(k * d / sd)
    expect = 0.5 + acc / len(scales) / np.pi
    assert np.abs(E.mgn(x, valid, scales, k, 1e-6, weights=g) - expect).max() < 1e-5
    gated = E.mgn_gated(x, valid, scales, k, 1e-6, np.zeros_like(x), 50.0, gate=0.0, weights=g)
    assert np.abs(gated - expect).max() < 1e-5


def test_mgn_scales_follow_the_preview_reduction():
    """The sizes are full-size pixels: the reduced preview divides them by its reduction."""
    from tse_core import workflow as W

    p = W._scaled_params(E.EnhanceParams(), 1 / 6)
    assert np.allclose(p.mgn_scales, [v / 6 for v in (1.25, 2.5, 5, 10, 20, 40)])



def test_natural_gamma_stretch_is_the_mgn_papers_global_image():
    """Morgan & Druckmueller (2014) combine MGN with C'_g = ((B - a0) / (a1 - a0))^(1/gamma), a0 and
    a1 the image's minimum and maximum, gamma 3.2 in the paper. Natural layer, Stretch: Gamma."""
    z = np.load(DATA)
    hdr = np.ascontiguousarray(np.moveaxis(z["hdr"].astype(np.float32) * float(z["scale"]), 0, -1))
    geo = {k: float(z[k]) for k in ("SUNX", "SUNY", "SUNR", "MOONX", "MOONY", "MOONR")}
    weight = (np.abs(hdr).sum(axis=2) > 0).astype(np.float32)
    p = E.EnhanceParams(layers=("natural",), saturation=0.0, blend={}, natural_stretch="gamma", gamma=3.2)
    assert E.EnhanceParams().natural_stretch == "log" and E.EnhanceParams().gamma == 3.2
    frame = E.prepare(hdr, weight, geo, p)
    out = E.render(frame, p)["natural"][..., 0]
    b = np.maximum(frame.lum, 0)
    sv = frame.stats_valid
    a0, a1 = float(b[sv].min()), float(b[sv].max())
    expect = np.clip((b - a0) / (a1 - a0), 0, 1) ** (1 / 3.2)
    v = frame.valid
    assert np.abs(out[v] - expect[v]).max() < 1e-4
    # A different gamma gives a different layer, and the log stretch is unchanged by these settings.
    assert np.abs(E.render(frame, E.EnhanceParams(layers=("natural",), saturation=0.0, blend={},
                                                  natural_stretch="gamma", gamma=2.0))["natural"][..., 0][v]
                  - expect[v]).max() > 0.01
