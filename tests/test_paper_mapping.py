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
    rng = np.random.default_rng(4)
    x = np.exp(ndimage.gaussian_filter(rng.normal(size=(160, 200)), 2.0)).astype(np.float32)
    valid = np.ones(x.shape, bool)
    scales, k = [1.5, 3.0, 6.0], 0.7
    b = np.log(x + 1e-6).astype(np.float32)
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
