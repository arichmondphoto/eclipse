"""Adaptive (ACHF): Druckmüller's adaptive circular high-pass filter, as defined in
H. Druckmüllerová's thesis (2013), section 5.2, equations 5.1-5.2:

    C_{r,phi}(rho, theta) = exp(-((rho - r)^2 + (r (theta - phi))^2) / (2 sigma^2))

a Gaussian of width sigma across the radius and sigma along the Sun-centred circle, used in an
incomplete convolution in which the Moon and pixels outside the image have weight 0; the filtered
image is the pixel minus that weighted local mean.
"""

import math

import numpy as np
import pytest
from scipy import ndimage

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
