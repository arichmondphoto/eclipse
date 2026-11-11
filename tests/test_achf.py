"""Adaptive (ACHF): Druckmüller's adaptive circular high-pass filter, as defined in
H. Druckmüllerová's thesis (2013), section 5.2, equations 5.1-5.2:

    C_{r,phi}(rho, theta) = exp(-((rho - r)^2 + (r (theta - phi))^2) / (2 sigma^2))

a Gaussian of width sigma across the radius and sigma along the Sun-centred circle, used in an
incomplete convolution in which the Moon and pixels outside the image have weight 0; the filtered
image is the pixel minus that weighted local mean.
"""

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage

from tse_core import enhance as E
from tse_core.enhance import achf, achf_mean

N = 400
CX, CY = 200.0, 200.0
R_MOON = 60.0


def _grid():
    yy, xx = np.mgrid[0:N, 0:N].astype(np.float64)
    return np.hypot(xx - CX, yy - CY), np.arctan2(yy - CY, xx - CX)


def test_the_kernel_follows_the_circle():
    """A corona that depends on the distance from the Sun only: along each circle nothing changes,
    so (eq. 5.1) the local mean is the radial profile blurred by a 1-D Gaussian of width sigma.
    A straight (Cartesian) kernel bends away from the circle and gets it wrong near the Sun."""
    r, _ = _grid()
    prof = lambda x: np.where(x < 110, 1.0, 0.0) + 0.002 * x  # noqa: E731  a sharp ring edge at r = 110
    f = prof(r)
    w = (r > R_MOON).astype(np.float64)
    sigma = 12.0
    mean = achf_mean(f, w, CX, CY, sigma)
    # Expected: the 1-D Gaussian blur of the profile along the radius, at the same radius.
    xs = np.arange(-6 * sigma, 6 * sigma + 0.01, 0.05)
    kern = np.exp(-0.5 * (xs / sigma) ** 2)
    sel = (r > 100) & (r < 120) & (np.abs(_grid()[1]) < 2.5)  # by the edge, away from the frame
    expect = np.array([np.sum(prof(rv + xs) * kern) / kern.sum() for rv in r[sel]])
    err = np.abs(mean[sel] - expect)
    assert err.max() < 0.01, err.max()
    cart = ndimage.gaussian_filter(f * w, sigma) / ndimage.gaussian_filter(w, sigma)
    assert np.abs(cart[sel] - expect).max() > 3 * err.max()  # a straight kernel is measurably off


def test_the_moon_does_not_count():
    """Weight 0 inside the Moon (eq. 5.2): what is there cannot change the result outside."""
    r, _ = _grid()
    rng = np.random.default_rng(1)
    f = ndimage.gaussian_filter(rng.normal(size=(N, N)), 3) + 1.0
    w = (r > R_MOON).astype(np.float64)
    a = achf_mean(f, w, CX, CY, 20.0)
    f2 = f.copy()
    f2[r <= R_MOON] = 1e6  # a blazing Moon
    b = achf_mean(f2, w, CX, CY, 20.0)
    out = r > R_MOON
    assert np.abs(a[out] - b[out]).max() < 1e-9


def test_small_scales_far_from_the_sun_are_an_ordinary_gaussian():
    """Where sigma is small against the radius the circle is nearly straight: the mean is a
    weighted Gaussian blur (here computed through the circular path anyway)."""
    r, _ = _grid()
    rng = np.random.default_rng(2)
    f = ndimage.gaussian_filter(rng.normal(size=(N, N)), 4)
    w = np.ones((N, N))
    sigma = 3.0
    m = achf_mean(f, w, CX, CY, sigma, circular=True)
    cart = ndimage.gaussian_filter(f, sigma)
    sel = (r > 120) & (r < 180)
    assert np.abs(m[sel] - cart[sel]).max() < 0.03 * f[sel].std()


def test_achf_layer_brings_out_detail_and_nothing_round_a_smooth_corona():
    """The layer is the (log) pixel minus its local means over the scales. A smooth corona varies
    only with height (a high-pass of its curved fall-off leaves an even offset round each circle,
    not zero), so round each circle it is flat; a loop stands out from the rest of its circle."""
    r, phi = _grid()
    corona = 1e3 * np.maximum(r / R_MOON, 1.0) ** -4
    valid = r > R_MOON + 2
    scales = [2.0, 4.0, 8.0, 16.0]
    smooth = achf(corona, valid, scales, 1e-6, CX, CY)
    ring = valid & (np.abs(r - 120) < 0.5)
    assert smooth[ring].std() < 0.01
    loop = corona * (1 + 0.3 * np.exp(-0.5 * ((r - 120) / 2.0) ** 2) * (np.abs(phi - 1.0) < 0.3))
    out = achf(loop, valid, scales, 1e-6, CX, CY)
    on = valid & (np.abs(r - 120) < 1) & (np.abs(phi - 1.0) < 0.2)
    off = valid & (np.abs(r - 120) < 1) & (np.abs(phi + 2.0) < 0.2)
    assert out[on].mean() - out[off].mean() > 0.1
    # Away from the loop the circle stays flat (each image gets its own stretch, so values are
    # compared within the image, not with the smooth one).
    rest = valid & (np.abs(r - 120) < 0.5) & (np.abs(np.angle(np.exp(1j * (phi - 1.0)))) > 0.8)
    assert out[rest].std() < 0.01


def test_scale_weights():
    """Per-scale strengths (like the eight frequency sliders of Druckmüller's Corona program): a
    scale at 0 adds nothing, so with only one scale on, the layer is that scale's filter alone;
    all equal is the plain sum; all 0 is flat mid-grey."""
    r, phi = _grid()
    rng = np.random.default_rng(3)
    img = 1e3 * np.maximum(r / R_MOON, 1.0) ** -4 * (1 + 0.1 * ndimage.gaussian_filter(rng.normal(size=(N, N)), 2))
    valid = r > R_MOON + 2
    scales = [2.0, 4.0, 8.0, 16.0]
    only = achf(img, valid, [4.0], 1e-6, CX, CY)
    picked = achf(img, valid, scales, 1e-6, CX, CY, weights=[0, 1, 0, 0])
    assert np.abs(only - picked)[valid].max() < 1e-5
    assert np.abs(achf(img, valid, scales, 1e-6, CX, CY, weights=[2, 2, 2, 2])
                  - achf(img, valid, scales, 1e-6, CX, CY))[valid].max() < 1e-5
    flat = achf(img, valid, scales, 1e-6, CX, CY, weights=[0, 0, 0, 0])
    assert np.abs(flat[valid] - 0.5).max() < 1e-6


def test_default_scales_are_the_users_pixels_on_the_2024_sun_and_scale_with_the_sun(tmp_path):
    """Defaults: 0.5, 1, 3 and 10 px on the 2024 frames (Sun radius 413.29 px, the stack's SUNR), then 10% and 20% of the radius
    switched off; kept as fractions of the Sun's radius, so a Sun twice the size gets twice the px."""
    from tse_core.enhance import ACHF_SCALES, ACHF_WEIGHTS, EnhanceParams
    from tse_core.fits import read_fits, read_fits_header, write_fits

    p = EnhanceParams()
    assert [round(f * 413.29, 3) for f in p.achf_scales[:4]] == [0.5, 1.0, 3.0, 10.0]
    assert tuple(p.achf_weights) == ACHF_WEIGHTS == (1, 1, 1, 1, 0, 0) and p.achf_scales[4:] == (0.1, 0.2)
    assert [round(f * 2 * 413.29, 3) for f in ACHF_SCALES[:4]] == [1.0, 2.0, 6.0, 20.0]
    # The window reads the stack's Sun from its header without loading the image.
    path = tmp_path / "s.fit"
    write_fits(path, np.zeros((3, 40, 50), np.float32), {"SUNR": 436.25, "OBJECT": "Sun", "MIRROR": True})
    assert read_fits_header(path) == {k: v for k, v in read_fits(path)[1].items()}


def _prominence_scene():
    r, phi = _grid()
    corona = 1e3 * np.maximum(r / R_MOON, 1.0) ** -4
    px, py = CX + (R_MOON + 7) * math.cos(1.0), CY + (R_MOON + 7) * math.sin(1.0)
    yy, xx = np.mgrid[0:N, 0:N]
    blob = np.hypot(xx - px, yy - py) < 6
    img = np.where(blob, corona * 10.0, corona)  # a prominence 10x brighter than the corona by it
    return r, phi, img, blob, np.hypot(xx - px, yy - py)


def test_a_bright_prominence_leaves_no_dark_fringe():
    """Requirement (d) of the thesis (section 5.2): a kernel weight is 0 where two pixels lie in
    significantly different parts of the image, by a maximum acceptable difference in value. Without
    it, a prominence 10x brighter than the corona drags the local mean of the corona beside it up,
    and a dark fringe forms round it (seen on the 2024 crop); with it, the corona there comes out
    as the corona elsewhere on the same circle."""
    r, phi, img, blob, d = _prominence_scene()
    valid = r > R_MOON + 1.5
    scales = [1.0, 3.0, 10.0]
    near = valid & ~blob & (d > 7) & (d < 16)  # the corona round the prominence
    same_r = lambda out, sel: np.array([np.median(out[valid & (np.abs(r - rv) < 0.5) & (np.abs(phi + 1.5) < 0.6)])  # noqa: E731
                                        for rv in r[sel]])
    off = achf(img, valid, scales, 1e-6, CX, CY, part_limit=0)
    on = achf(img, valid, scales, 1e-6, CX, CY, part_limit=2.0)
    fringe_off = float(np.median(same_r(off, near) - off[near]))
    fringe_on = float(np.median(same_r(on, near) - on[near]))
    assert fringe_off > 0.03, fringe_off  # the fringe is real without the rule
    assert abs(fringe_on) < 0.25 * fringe_off, (fringe_on, fringe_off)


def test_part_rule_leaves_ordinary_structure_alone():
    """Structure well under the limit (a loop of 30% contrast) is one part: the rule changes nothing."""
    r, phi = _grid()
    corona = 1e3 * np.maximum(r / R_MOON, 1.0) ** -4
    loop = corona * (1 + 0.3 * np.exp(-0.5 * ((r - 120) / 2.0) ** 2) * (np.abs(phi - 1.0) < 0.3))
    valid = r > R_MOON + 2
    a = achf(loop, valid, [2.0, 4.0, 8.0], 1e-6, CX, CY, part_limit=0)
    b = achf(loop, valid, [2.0, 4.0, 8.0], 1e-6, CX, CY, part_limit=2.0)
    assert np.abs(a - b)[valid].max() < 1e-5


# --- The thesis's own combination, h = K1 f + K2 g (Druckmuellerova 2013, section 5.2) -------------
# f is the composed image, "monotone (ideally linear)" in the light (section 4.1.4), and g is f minus
# its incomplete convolution with the circular kernel (eq. 5.2), on that same linear f. "The
# processed image is then a linear combination of the original and the filtered image": nothing
# is taken in log and no curve is put on g. Only afterwards is h drawn (here, with the natural
# layer's own stretch); her chapter 7 lists a stretch of f before combining as future work.

DATA = Path(__file__).parent / "data" / "eclipse2024_bin8.npz"


def _real_frame(p):
    z = np.load(DATA)
    hdr = np.ascontiguousarray(np.moveaxis(z["hdr"].astype(np.float32) * float(z["scale"]), 0, -1))
    geo = {k: float(z[k]) for k in ("SUNX", "SUNY", "SUNR", "MOONX", "MOONY", "MOONR")}
    weight = (np.abs(hdr).sum(axis=2) > 0).astype(np.float32)
    return E.prepare(hdr, weight, geo, p)


def _thesis_g(frame, fracs, weights, part_limit=0.0):
    g = frame.g
    f = np.maximum(frame.lum, 0).astype(np.float64)
    w = frame.valid.astype(np.float64)
    acc = np.zeros(f.shape)
    for s, k in zip(fracs, weights):
        if k > 0:
            acc += k * (f - achf_mean(f, w, g.sun_x, g.sun_y, s * g.sun_r, r_ref=g.moon_r))
    return np.where(frame.valid, acc / sum(weights), 0.0)


@pytest.mark.parametrize("stretch", ["log", "gamma"])
def test_achf_on_linear_is_the_theses_h_k1_f_plus_k2_g(stretch):
    fracs, ws = (0.005, 0.02), (1.0, 1.0)
    p = E.EnhanceParams(layers=("natural", "achf"), saturation=0.0, natural_stretch=stretch,
                        achf_scales=fracs + (0.1, 0.2, 0.3, 0.4), achf_weights=ws + (0, 0, 0, 0),
                        achf_part_limit=0.0, achf_contrast=1.5,
                        blend={"natural": 1.0, "achf": 4.0}, blend_modes={"achf": "linear"})
    frame = _real_frame(p)
    out = E.render(frame, p)["blend"][..., 0]
    gl = _thesis_g(frame, fracs, ws)
    h = frame.lum + 4.0 * 1.5 * gl                         # K1 = 1, K2 = strength x Contrast
    st = E._natural_cache(frame, p)
    r, _ = E.polar(frame.shape, frame.g)
    expect = E.natural_apply(h.astype(np.float32), r, st, p.flatten, show=frame.show)
    v = frame.valid
    assert np.abs(out[v] - expect[v]).max() < 2e-3
    # Not the old form (the ACHF layer's 0..1 picture averaged with the natural layer):
    nat = E.render(frame, replace(p, blend={}))["natural"][..., 0]
    assert np.abs(out[v] - nat[v]).max() > 0.02


def test_achf_weighted_average_is_the_earlier_linear_form():
    """Weighted average keeps how Linear (paper) treated ACHF until 6 Nov 2026: the layer's own
    0..1 picture (log detail through its soft curve), (K1 B + K2 L) / (K1 + K2)."""
    p = E.EnhanceParams(layers=("natural", "achf"), saturation=0.0, achf_scales=(0.005, 0.02, 0.1, 0.2, 0.3, 0.4),
                        achf_weights=(1, 1, 0, 0, 0, 0), blend={"natural": 1.0, "achf": 0.6},
                        blend_modes={"achf": "average"})
    frame = _real_frame(p)
    lay = E.render(frame, p)
    v = frame.valid
    expect = (lay["natural"][..., 0] + 0.6 * lay["achf"][..., 0]) / 1.6
    assert np.abs(lay["blend"][..., 0][v] - expect[v]).max() < 1e-5


def test_achf_layer_on_linear_shows_the_detail_as_a_share_of_the_local_mean():
    """Seen on its own, the thesis-form layer is g / (f - g), the pixel's difference from its local
    mean as a share of that mean, on ACHF's usual display scale (close to the log form's)."""
    fracs, ws = (0.005, 0.02), (1.0, 1.0)
    p = E.EnhanceParams(layers=("natural", "achf"), saturation=0.0, achf_scales=fracs + (0.1, 0.2, 0.3, 0.4),
                        achf_weights=ws + (0, 0, 0, 0), achf_part_limit=0.0, blend={"natural": 1.0, "achf": 1.0},
                        blend_modes={"achf": "linear"})
    frame = _real_frame(p)
    lay = E.render(frame, p)["achf"][..., 0]
    gl = _thesis_g(frame, fracs, ws)
    mean = np.maximum(np.maximum(frame.lum, 0) - gl, frame.nlev)
    expect = E.to_fixed((gl / mean).astype(np.float32), E.ACHF_SPREAD)
    v = frame.valid
    assert np.abs(lay[v] - expect[v]).max() < 2e-3
