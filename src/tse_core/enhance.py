"""Phase 4: reveal coronal structure in the linear HDR stack.

Every filter works on the luminance of the HDR image, in heliocentric polar
coordinates around the Sun's centre (from the stack header). Colour is put
back afterwards from the smoothed colour ratios of the stack, so all layers
share the same colour and line up pixel for pixel.

Layers (each written as a 16-bit TIFF, ready to stack in Photoshop):

``natural``
    Logarithmic stretch with most of the radial fall-off removed; keeps the
    look of the eye-view corona.
``nrgf``
    Normalizing-radial-graded filter (Morgan, Habbal & Woo 2006): subtract
    the mean and divide by the standard deviation along each circle.
``fnrgf``
    Fourier NRGF (Druckmüllerová, Morgan & Habbal 2011): the mean and the
    standard deviation along each circle are attenuated Fourier series in
    position angle, so contrast is evened out around the Sun as well as
    outwards.
``fnrgf_n``
    FNRGF-N: the same, with the noise contribution subtracted from the local
    standard deviation (their eq. 9). The noise is estimated automatically
    from the stack's weight map instead of from a hand-picked region.
``adaptive``
    Multi-scale Gaussian normalization (Morgan & Druckmüller 2014) of the
    logarithm of the image: local contrast normalized at several scales
    independent of orientation, the published successor to the adaptive
    circular high-pass filter of Druckmüller et al. (2006).

``blend`` is a suggested mix of these, with adjustable weights.
"""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
from scipy import ndimage

Say = Callable[[str], None]

LAYERS = ("natural", "nrgf", "fnrgf", "fnrgf_n", "adaptive", "achf")
# Written unless asked otherwise: Adaptive (ACHF) takes minutes on a full frame, so only on request.
DEFAULT_LAYERS = ("natural", "nrgf", "fnrgf", "fnrgf_n", "adaptive")
# Adaptive (ACHF) scales, finest first, as fractions of the Sun's radius (so they suit any lens):
# Andy's choice (4 Nov 2026) of 0.5, 1, 3 and 10 px on his 2024 frames (Sun radius 413.29 px, from
# the stack's SUNR; not the Moon's 435.92), within the thesis's 0.5-64 px, plus 10% and 20% of the
# radius switched off (strength 0).
MGN_SCALES = (1.25, 2.5, 5.0, 10.0, 20.0, 40.0)
MGN_WEIGHTS = (1.0,) * 6
SUN_R_2024 = 413.29
ACHF_SCALES = tuple(v / SUN_R_2024 for v in (0.5, 1.0, 3.0, 10.0)) + (0.1, 0.2)
ACHF_WEIGHTS = (1.0, 1.0, 1.0, 1.0, 0.0, 0.0)
# The natural layer is the base; the others are soft-light layers at these opacities. All filter
# layers start at 0 (Andy, Nov 2026: every Enhance option off by default), so the blend starts as the
# natural layer and each layer is added by choice.
DEFAULT_BLEND = {"natural": 1.0}


@dataclass
class EnhanceParams:
    layers: tuple[str, ...] = DEFAULT_LAYERS
    segments: int = 50  # angular segments per circle (paper: 50)
    order: int = 10  # Fourier order (paper: 10)
    mean_attenuation: Optional[Sequence[float]] = None  # A_0..A_order; default 1, 0.9, ..., 0
    std_attenuation: Optional[Sequence[float]] = None  # S_0..S_order; default as above
    noise_m: float = 0.7  # FNRGF-N: fraction of the noise removed from the local deviation, in [0, 1)
    ring_smooth: float = 1.5  # smoothing of the circle statistics along the radius, px
    flatten: float = 0.0  # natural layer: fraction of the radial fall-off (in log) removed; 0 = off
    # Highlights: 0 = a hard white point (the brightest 0.1% clip); above 0, the top part of the
    # range (this fraction) rolls off smoothly towards white and nothing clips, colours keep their hue.
    shoulder: float = 0.0
    # Evenness: this fraction of the large-scale brightness differences round the Sun (streamers
    # against coronal holes, over ~25% of the Sun's radius) is taken out too, like the radial
    # fall-off. 0 = off.
    even: float = 0.0
    # Midtones: above 0, a gamma puts the inner corona's median brightness (1.05-2.5 solar radii)
    # at this level (e.g. 0.6 for a light grey corona). 0 = off.
    midtone: float = 0.0
    # Adaptive (MGN): the paper's six kernel widths (px of the full-size stack) and their weights g_i
    # (Morgan & Druckmueller 2014: "w = 1.25, 2.5, 5, 10, 20, 40"; g_i "may be adjusted").
    mgn_scales: Optional[Sequence[float]] = MGN_SCALES
    mgn_weights: Sequence[float] = MGN_WEIGHTS
    # Adaptive (ACHF): the six scales (kernel widths sigma) as fractions of the Sun's radius, finest
    # first, and the strength of each.
    achf_scales: Sequence[float] = ACHF_SCALES
    achf_weights: Sequence[float] = ACHF_WEIGHTS
    # Requirement (d): a pixel this many times brighter or darker than the corona round it (a
    # prominence, the bright rim at the limb) is a part of its own; 0 = off.
    achf_part_limit: float = 2.0
    # Adaptive (ACHF): contrast of the layer (the thesis's K2), set by eye; see ACHF_SPREAD.
    achf_contrast: float = 1.0
    mgn_k: float = 0.7  # adaptive layer strength: k of arctan(k C), 0.7 as in Morgan & Druckmueller (2014)
    # Adaptive layer, noise-aware (0 = off):
    # noise_gate: each scale's detail counts only as far as it stands above the noise at that scale
    # (measured from the stack); the number is how many times the noise counts as "clearly above"
    # (3 is a good start). Keeps the finest scales from turning grain into texture.
    noise_gate: float = 0.0
    saturation: float = 1.0  # colour saturation of every layer (0 = monochrome)
    white: str = "corona"  # "corona" (inner corona is white), "camera" (as-shot), or "none"
    neutral_background: bool = True  # model and remove the sky (level, colour and gradient) before filtering
    moon_margin_px: float = 1.5
    blend: dict = field(default_factory=lambda: dict(DEFAULT_BLEND))
    blend_modes: dict = field(default_factory=dict)  # layer -> "soft light" (default), "overlay" or "linear"


@dataclass
class Geometry:
    sun_x: float
    sun_y: float
    sun_r: float
    moon_x: float
    moon_y: float
    moon_r: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class MoonInput:
    """The Moon put into the stack (see earthshine.merge_moon): which pixels carry it, and the
    lunar surface to add on top after the stretch (a crop at ``x0``, ``y0`` of the canvas)."""

    mask: np.ndarray  # (h, w) bool over the whole canvas: Moon pixels to render
    x0: int = 0
    y0: int = 0
    surface: Optional[np.ndarray] = None  # (hc, wc, 3) display 0..1: the seas and highlands
    visibility: Optional[np.ndarray] = None  # (hc, wc) 0..1: how much of the surface to show


def geometry_from_header(header: dict) -> Geometry:
    mx, my, mr = float(header["MOONX"]), float(header["MOONY"]), float(header["MOONR"])
    if "SUNX" in header:
        return Geometry(float(header["SUNX"]), float(header["SUNY"]), float(header["SUNR"]), mx, my, mr)
    return Geometry(mx, my, mr / 1.04, mx, my, mr)


def polar(shape: tuple[int, int], g: Geometry, origin: tuple[int, int] = (0, 0)) -> tuple[np.ndarray, np.ndarray]:
    """Distance from the Sun's centre (px) and position angle (radians, 0..2pi) of every pixel of
    the region of ``shape`` starting at ``origin`` (row, column) of the whole image."""
    h, w = shape
    yy = (np.arange(origin[0], origin[0] + h, dtype=np.float64) - g.sun_y).astype(np.float32)[:, None]
    xx = (np.arange(origin[1], origin[1] + w, dtype=np.float64) - g.sun_x).astype(np.float32)[None, :]
    r = np.sqrt(xx * xx + yy * yy)
    phi = np.arctan2(yy, xx)
    phi = np.where(phi < 0, phi + np.float32(2 * math.pi), phi).astype(np.float32)
    return r, phi


def blur(img: np.ndarray, sigma: float, origin: tuple[int, int] = (0, 0)) -> np.ndarray:
    """Gaussian blur; large kernels run on a reduced copy (fast on 45-megapixel frames).

    ``origin`` is where ``img`` starts in the whole image: the reduced copy's blocks are laid on the
    whole image's grid, so a region (a detail box with its margin) gets exactly the whole image's
    result away from its edges."""
    if sigma <= 6:
        return ndimage.gaussian_filter(img, sigma, mode="reflect")
    f = int(sigma // 3)
    h, w = img.shape
    oy, ox = int(origin[0]) % f, int(origin[1]) % f
    ph, pw = (-(h + oy)) % f, (-(w + ox)) % f
    a = np.pad(img, ((oy, ph), (ox, pw)), mode="reflect")
    small = a.reshape(a.shape[0] // f, f, a.shape[1] // f, f).mean(axis=(1, 3))
    s2 = math.sqrt(max(sigma**2 - (f**2 - 1) / 12.0, 0.25)) / f
    small = ndimage.gaussian_filter(small, s2, mode="reflect")
    big = ndimage.zoom(small, f, order=1, mode="nearest", grid_mode=True)
    return big[oy:oy + h, ox:ox + w].astype(img.dtype, copy=False)


def _blur_reach(sigma: float) -> int:
    """How far :func:`blur` reaches, in px (Gaussians are cut at 4 sigma; the reduced copy adds
    up to two of its blocks)."""
    return int(math.ceil(4 * sigma)) + (2 * int(sigma // 3) if sigma > 6 else 0) + 2


def _attenuation(values: Optional[Sequence[float]], order: int) -> np.ndarray:
    if values is None:
        return np.clip(1.0 - np.arange(order + 1) / max(order, 1), 0.0, 1.0)
    a = np.zeros(order + 1)
    v = np.asarray(values, float)[: order + 1]
    a[: len(v)] = v
    return a


# How the filter layers are shown, as the papers combine them (linearly, with the user's weights)
# and with fixed scales, never measured on each image (so a detail box gives exactly the full-size
# result), in each method's own units, the same for any camera, lens or eclipse:
#   NRGF, FNRGF, FNRGF-N (Morgan, Habbal & Woo 2006; Druckmuellerova, Morgan & Habbal 2011): Y, the
#     pixel less the local mean over the local deviation, on a straight line about mid-grey,
#     0.5 + Y / (2 s), clipped to 0..1: Y = +s is white, -s black. The papers combine Z = K1 X + K2 Y
#     with the same K2 for every image; the blend strength plays K2. s in standard deviations
#     (FNRGF-N: deviations less Noise compensation x the noise).
#   MGN (Morgan & Druckmueller 2014): the mean over the scales of arctan(k C), whose range is
#     (-pi/2, pi/2), as 0.5 + mean / pi: no constant of ours.
#   ACHF (thesis eq. 5.2 has h = K1 f + K2 g, K2 the user's): Contrast x the strength-weighted mean
#     of ln(pixel / local mean) through a soft arctan curve, 0.5 + arctan(1.6 y / s) / pi, so
#     y = s shows at 82% grey; s = 0.0106, about 1.1% brighter than the local mean at Contrast 1.
NRGF_SPREAD = 3.5
FNRGF_SPREAD = 6.8
FNRGF_N_SPREAD = 20.0
ACHF_SPREAD = 0.0106


def to_fixed(y: np.ndarray, spread: float) -> np.ndarray:
    """A zero-centred filter output to 0..1 with a soft arctan curve and a fixed scale (ACHF)."""
    return (0.5 + np.arctan(1.6 * y / spread) / math.pi).astype(np.float32)


def to_linear(y: np.ndarray, spread: float) -> np.ndarray:
    """A zero-centred filter output on a straight line about mid-grey: +spread white, -spread black."""
    return np.clip(0.5 + y / np.float32(2.0 * spread), 0.0, 1.0).astype(np.float32)


def mgn_unit(acc: np.ndarray) -> np.ndarray:
    """MGN's mean of arctan(k C) over the scales, range (-pi/2, pi/2), to 0..1 on a straight line."""
    return np.clip(0.5 + acc / np.float32(math.pi), 0.0, 1.0).astype(np.float32)


def roll_off(y: np.ndarray, shoulder: float) -> np.ndarray:
    """Above 1 - ``shoulder`` the values approach 1 smoothly (slope 1 at the knee, never reaching
    it), so the brightest parts keep their gradations instead of clipping. 0: unchanged."""
    if shoulder <= 0:
        return y
    knee = 1.0 - shoulder
    return np.where(y > knee, knee + shoulder * (1.0 - np.exp(-(y - knee) / shoulder)), y).astype(np.float32)


def stretch(x: np.ndarray, valid: np.ndarray, lo: float = 0.5, hi: float = 99.8, shoulder: float = 0.0) -> np.ndarray:
    a, b = np.percentile(x[valid], [lo, hi]) if valid.any() else (0.0, 1.0)
    y = (x - a) / ((b - a) or 1.0)
    return np.clip(roll_off(y, shoulder), 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Noise
# ---------------------------------------------------------------------------


def noise_constant(lum: np.ndarray, weight: np.ndarray, valid: np.ndarray) -> Optional[float]:
    """``c`` in noise = c / sqrt(weight), from the pixel-to-pixel scatter of the faintest half of
    ``valid`` (where structure contributes least). None when there are too few pixels."""
    wpos = np.where(valid & (weight > 0), weight, np.nan)
    resid = lum - ndimage.gaussian_filter(lum, 1.0)
    scaled = np.abs(resid) * np.sqrt(wpos)
    ok = np.isfinite(scaled)
    if ok.sum() < 100:
        return None
    faint = ok & (lum < np.nanpercentile(np.where(ok, lum, np.nan), 50))
    return 1.4826 * float(np.median(scaled[faint if faint.sum() > 100 else ok])) / 0.96


def noise_map(lum: np.ndarray, weight: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Per-pixel noise of the stacked luminance.

    The stack weights are inverse variances up to one common factor, so the
    noise is ``c / sqrt(weight)``; ``c`` is measured from the pixel-to-pixel
    scatter in the faint outer corona, where structure is smooth.
    """
    c = noise_constant(lum, weight, valid)
    if c is None:
        return np.zeros_like(lum)
    with np.errstate(divide="ignore", invalid="ignore"):
        n = c / np.sqrt(np.where(weight > 0, weight, np.nan))
    return np.nan_to_num(n, nan=0.0, posinf=0.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Circle statistics and the FNRGF
# ---------------------------------------------------------------------------


@dataclass
class RingStats:
    """Fourier coefficients (per ring) of the mean, standard deviation and noise along circles."""

    mean: np.ndarray  # (rings, 2*order+1): [c0, a1, b1, a2, b2, ...]
    std: np.ndarray
    noise: np.ndarray
    order: int
    ring_mean: np.ndarray  # (rings,) whole-circle mean (NRGF)
    ring_std: np.ndarray  # (rings,) whole-circle standard deviation (NRGF)
    seg_std: np.ndarray  # (rings,) average within-segment deviation (a floor for the FNRGF)


def spot_mask(lum: np.ndarray, valid: np.ndarray, r: Optional[np.ndarray] = None, r_min: float = 0.0,
              size: int = 9, k: float = 5.0) -> np.ndarray:
    """Small compact blemishes (dust shadows, stars, hot pixels) to leave out of the statistics.

    A dark dust spot at one distance from the Sun pulls down the circle
    statistics at exactly that distance, and the filters turn it into a
    faint arc around the Sun. Spots are found on a reduced copy (about 1400
    px high) as departures from a local median, far beyond the local scatter.
    They are only excluded from the statistics; the image keeps them.
    """
    h, w = lum.shape
    f = max(1, int(round(min(h, w) / 1400)))
    hh, ww = h // f, w // f
    x = np.where(valid, lum, np.nan)[: hh * f, : ww * f].reshape(hh, f, ww, f)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        red = np.nanmean(x, axis=(1, 3))
    ok = np.isfinite(red) & (red > 0)
    if ok.sum() < 1000:
        return np.zeros((h, w), bool)
    lg = np.log(np.where(ok, red, np.nanmedian(red[ok])))
    lg = np.where(ok, lg, np.median(lg[ok]))
    d = lg - ndimage.median_filter(lg, size=size, mode="nearest")
    scatter = 1.25 * ndimage.uniform_filter(np.abs(d), size=4 * size + 1, mode="nearest") + 1e-6
    spots = ok & (np.abs(d) > k * scatter)
    if r is not None and r_min > 0:
        # Near the Sun the corona itself is full of compact, sharp structure; leave it alone.
        spots &= r[: hh * f : f, : ww * f : f][:hh, :ww] > r_min
    spots = ndimage.binary_dilation(spots, iterations=2)
    full = np.zeros((h, w), bool)
    full[: hh * f, : ww * f] = np.repeat(np.repeat(spots, f, axis=0), f, axis=1)
    return full


def ring_stats(x: np.ndarray, r: np.ndarray, phi: np.ndarray, valid: np.ndarray, segments: int, order: int,
               smooth: float, noise: Optional[np.ndarray] = None) -> RingStats:
    """Mean and deviation per 1-px ring and angular segment, fitted with Fourier series.

    The paper integrates over complete circles; on a rectangular frame many
    circles leave the image, so segments without data are first filled from
    further in (see ``_fill_segments``).
    """
    nr = int(np.ceil(r.max())) + 2
    ring = r.astype(np.int32)
    seg = np.minimum((phi * (segments / (2 * math.pi))).astype(np.int32), segments - 1)
    idx = (ring * segments + seg)[valid]
    xv = x[valid].astype(np.float64)
    size = nr * segments
    cnt = np.bincount(idx, minlength=size).astype(np.float64)
    s1 = np.bincount(idx, xv, minlength=size)
    s2 = np.bincount(idx, xv * xv, minlength=size)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = s1 / cnt
        var = np.maximum(s2 / cnt - mean * mean, 0.0)
    if noise is not None:
        nv = noise[valid].astype(np.float64)
        sn = np.bincount(idx, nv * nv, minlength=size)
        with np.errstate(invalid="ignore", divide="ignore"):
            nvar = sn / cnt
    else:
        nvar = np.zeros(size)
    # Whole-circle statistics for the NRGF.
    rc = cnt.reshape(nr, segments).sum(axis=1)
    rs1 = s1.reshape(nr, segments).sum(axis=1)
    rs2 = s2.reshape(nr, segments).sum(axis=1)
    # Variance within each 1-px circle first, then smoothed: pooling neighbouring circles
    # before taking the variance would add the steep radial gradient to it.
    with np.errstate(invalid="ignore", divide="ignore"):
        m1 = np.nan_to_num(rs1 / rc)
        v1 = np.nan_to_num(np.maximum(rs2 / rc - m1 * m1, 0.0))
    sg = max(smooth, 0.5)
    rcs = ndimage.gaussian_filter1d(rc, sg, mode="nearest") + 1e-12
    ring_mean = ndimage.gaussian_filter1d(m1 * rc, sg, mode="nearest") / rcs
    ring_std = np.sqrt(ndimage.gaussian_filter1d(v1 * rc, sg, mode="nearest") / rcs)
    cnt = cnt.reshape(nr, segments)
    # A segment "has data" when at least a third of its arc is inside the frame and off the
    # Moon; slivers at the frame edge are too few pixels to trust.
    arc = 2 * math.pi * (np.arange(nr) + 0.5) / segments
    have = cnt >= np.maximum(0.33 * arc[:, None], 1.0)
    mean = np.nan_to_num(mean.reshape(nr, segments))
    std = np.sqrt(np.nan_to_num(var.reshape(nr, segments)))
    nstd = np.sqrt(np.nan_to_num(nvar.reshape(nr, segments)))
    # Smooth along the radius: 1-px rings far out hold few pixels per segment.
    if smooth > 0:
        wsum = ndimage.gaussian_filter1d(cnt, smooth, axis=0, mode="nearest") + 1e-12
        mean = ndimage.gaussian_filter1d(mean * cnt, smooth, axis=0, mode="nearest") / wsum
        std = ndimage.gaussian_filter1d(std * cnt, smooth, axis=0, mode="nearest") / wsum
        nstd = ndimage.gaussian_filter1d(nstd * cnt, smooth, axis=0, mode="nearest") / wsum
        cnt = wsum
    seg_std = (std * cnt).sum(axis=1) / (cnt.sum(axis=1) + 1e-12)
    ring_mean, ring_std, seg_std = (_carry(v, rc > 0) for v in (ring_mean, ring_std, seg_std))
    # Circles that leave the frame have segments without data. Fill each empty segment from
    # the last ring where it had data, scaled by the smooth radial trend: coronal structure
    # is close to radial, so this continues streamers outwards instead of inventing an edge.
    mean = _fill_segments(mean, have, additive=True)
    std = _fill_segments(std, have, additive=False)
    nstd = _fill_segments(nstd, have, additive=False)
    # Whole-circle statistics again, now from the completed circles, so the NRGF and the natural
    # layer do not jump where one side of a circle leaves the frame and the sky there differs.
    data = rc > 0
    ring_mean = np.where(data, mean.mean(axis=1), ring_mean)
    ring_std = np.where(data, np.sqrt((std**2 + (mean - mean.mean(axis=1, keepdims=True)) ** 2).mean(axis=1)),
                        ring_std)
    seg_std = np.where(data, std.mean(axis=1), seg_std)
    # Fourier coefficients of the (now complete) circles: the paper's eqs. 2-4.
    centres = 2 * math.pi * (np.arange(segments) + 0.5) / segments
    cols = [np.full(segments, 1.0 / segments)]
    for k in range(1, order + 1):
        cols += [2.0 / segments * np.cos(k * centres), 2.0 / segments * np.sin(k * centres)]
    P = np.stack(cols, axis=1)  # (segments, 2*order+1)
    out = [y @ P for y in (mean, std, nstd)]
    return RingStats(out[0], out[1], out[2], order, ring_mean, ring_std, seg_std)


def _trend(prof: np.ndarray, complete: np.ndarray, sigma: float = 15.0) -> np.ndarray:
    """Smooth radial trend of a profile, for scaling values carried outwards.

    Taken only from circles that lie wholly inside the frame (partial circles
    are biased by whichever side is left), and continued beyond them as a
    power law fitted to the outer quarter of those circles.
    """
    n = len(prof)
    idx = np.nonzero(complete & (prof > 0))[0]
    if len(idx) < 10:
        return np.ones(n)
    v = np.log(np.maximum(prof, 1e-30))
    filled = np.interp(np.arange(n), idx, v[idx])
    sm = ndimage.gaussian_filter1d(filled, sigma, mode="nearest")
    last = idx[-1]
    fit = idx[idx >= idx[0] + 0.75 * (last - idx[0])]
    if len(fit) >= 5:
        slope, icpt = np.polyfit(np.log(fit + 0.5), sm[fit], 1)
        slope = min(slope, 0.0)  # never brighten outwards
        rr = np.arange(last + 1, n)
        sm[last + 1:] = sm[last] + slope * (np.log(rr + 0.5) - np.log(last + 0.5))
    return np.exp(sm)


def _fill_segments(y: np.ndarray, have: np.ndarray, additive: bool) -> np.ndarray:
    """Complete circles that leave the frame.

    A segment with no data at ring ``i`` takes its own value from the last
    ring ``L`` where it had data, changed by how much the segments that
    still have data at ``i`` changed between ``L`` and ``i``: the same
    difference (``additive``, for means, which approach zero once the sky
    is removed) or the same ratio (for deviations, which stay positive).
    Seen from the Sun the frame is convex, so a segment with data at ``i``
    also has data at every ring inside ``i``.
    """
    nr, ns = y.shape
    out = y.copy()
    idx = np.arange(nr)[:, None]
    last = np.maximum.accumulate(np.where(have, idx, -1), axis=0)
    # Rings inside the first data (behind the Moon) take the nearest ring outside.
    nxt = np.minimum.accumulate(np.where(have, idx, nr)[::-1], axis=0)[::-1]
    for i in range(nr):
        miss = ~have[i]
        if not miss.any():
            continue
        avail = have[i]
        for L in np.unique(last[i][miss]):
            segs = miss & (last[i] == L)
            if L < 0:
                n = nxt[i][segs]
                ok = n < nr
                cols = np.nonzero(segs)[0]
                out[i, cols[ok]] = y[n[ok], cols[ok]]
                continue
            if avail.any():
                now, then = y[i, avail].mean(), y[L, avail].mean()
                if additive:
                    out[i, segs] = y[L, segs] + (now - then)
                else:
                    out[i, segs] = y[L, segs] * (now / then if then > 0 else 1.0)
            else:
                out[i, segs] = out[i - 1, segs] if i > 0 else y[L, segs]
    return out


def _carry(v: np.ndarray, have: np.ndarray) -> np.ndarray:
    """Fill rings without data from the nearest ring that has some."""
    if not have.any():
        return v
    idx = np.nonzero(have)[0]
    out = v.copy()
    out[: idx[0]] = v[idx[0]]
    out[idx[-1] + 1:] = v[idx[-1]]
    return out


def _profile_at(prof: np.ndarray, r: np.ndarray) -> np.ndarray:
    return np.interp(r, np.arange(len(prof)) + 0.5, prof).astype(np.float32)


def nrgf(x: np.ndarray, r: np.ndarray, valid: np.ndarray, stats: RingStats) -> np.ndarray:
    """Morgan et al. (2006): Y = (X - E_r) / sqrt(D_r) along each circle."""
    y = (x - _profile_at(stats.ring_mean, r)) / np.maximum(_profile_at(stats.ring_std, r), 1e-12)
    y[~valid] = 0.0
    return y.astype(np.float32)


def evaluate(coeffs: np.ndarray, att: np.ndarray, r: np.ndarray, phi: np.ndarray, rows: slice) -> np.ndarray:
    """Attenuated Fourier series at each pixel of ``rows``, interpolated between rings."""
    rr = r[rows] - 0.5
    i0 = np.clip(np.floor(rr).astype(np.int32), 0, coeffs.shape[0] - 2)
    t = np.clip(rr - i0, 0.0, 1.0).astype(np.float32)
    c = coeffs.astype(np.float32)
    p = phi[rows]
    out = att[0] * ((1 - t) * c[i0, 0] + t * c[i0 + 1, 0])
    order = (coeffs.shape[1] - 1) // 2
    for k in range(1, min(order, len(att) - 1) + 1):
        if att[k] == 0:
            continue
        a = (1 - t) * c[i0, 2 * k - 1] + t * c[i0 + 1, 2 * k - 1]
        b = (1 - t) * c[i0, 2 * k] + t * c[i0 + 1, 2 * k]
        out += att[k] * (a * np.cos(k * p) + b * np.sin(k * p))
    return out


def fnrgf(x: np.ndarray, r: np.ndarray, phi: np.ndarray, valid: np.ndarray, stats: RingStats,
          mean_att: np.ndarray, std_att: np.ndarray, noise_m: float = 0.0, chunk: int = 512) -> np.ndarray:
    """Y = (X - F_E) / (F_sigma - m * F_N), Druckmüllerová et al. (2011) eqs. 7 and 9."""
    h = x.shape[0]
    y = np.zeros_like(x, dtype=np.float32)
    for r0 in range(0, h, chunk):
        rows = slice(r0, min(h, r0 + chunk))
        fe = evaluate(stats.mean, mean_att, r, phi, rows)
        fs = evaluate(stats.std, std_att, r, phi, rows)
        # Never let the local deviation fall far below the circle's average (gaps, edges).
        floor = 0.25 * _profile_at(stats.seg_std, r[rows])
        if noise_m > 0:
            fn = evaluate(stats.noise, std_att[:1], r, phi, rows)
            fs = fs - noise_m * fn
        fs = np.maximum(fs, np.maximum(floor, 1e-12))
        y[rows] = (x[rows] - fe) / fs
    y[~valid] = 0.0
    return y


# ---------------------------------------------------------------------------
# Other layers
# ---------------------------------------------------------------------------


def radial_profile(x: np.ndarray, r: np.ndarray, valid: np.ndarray, smooth: float = 2.0) -> np.ndarray:
    ring = r.astype(np.int32)
    nr = int(ring.max()) + 2
    cnt = np.bincount(ring[valid], minlength=nr).astype(np.float64)
    s = np.bincount(ring[valid], x[valid].astype(np.float64), minlength=nr)
    w = ndimage.gaussian_filter1d(cnt, smooth, mode="nearest") + 1e-12
    prof = ndimage.gaussian_filter1d(s, smooth, mode="nearest") / w
    # Carry the last value outwards where there is no data.
    good = cnt > 0
    if good.any():
        last = np.nonzero(good)[0].max()
        prof[last + 1:] = prof[last]
    return prof


@dataclass
class NaturalStats:
    """What the natural layer measures over the whole image (see :func:`natural_stats`)."""

    floor: float
    prof: np.ndarray  # mean log brightness per 1-px circle (completed circles)
    black: float
    white: float
    ring: Optional[np.ndarray] = None  # Evenness: the circles' mean of the large-scale map
    loc: Optional[tuple] = None  # Evenness: the large-scale map as (blurred lg, blurred base, blurred mask)
    mid: Optional[float] = None  # Midtone: the inner corona's median before the curve


def _ring_at(prof: np.ndarray, r: np.ndarray) -> np.ndarray:
    ri = np.clip(r - 0.5, 0, len(prof) - 1.001)
    i0 = ri.astype(np.int32)
    t = ri - i0
    return ((1 - t) * prof[i0] + t * prof[i0 + 1]).astype(np.float32)


def natural_stats(x: np.ndarray, r: np.ndarray, phi: np.ndarray, valid: np.ndarray, noise_level: float,
                  flatten: float, stats_valid: Optional[np.ndarray] = None, shoulder: float = 0.0,
                  even: float = 0.0, sun_r: float = 0.0, midtone: float = 0.0,
                  base_cache: Optional[dict] = None) -> NaturalStats:
    """The natural layer's whole-image measurements: the circles' mean log brightness, the black
    point (the sky), the white point (99.9th percentile), and for Evenness and Midtone their maps
    and medians. ``base_cache`` keeps what does not depend on the settings between calls."""
    cache = base_cache if base_cache is not None else {}
    floor = max(noise_level, 1e-9)
    lg = np.log(np.maximum(x, 0) + floor).astype(np.float32)
    sv = valid if stats_valid is None else stats_valid
    if "prof" not in cache:
        # Circle averages from completed circles (see ring_stats), so no arcs at the frame edges.
        lo = float(lg[valid].min()) if valid.any() else 0.0
        prof = ring_stats(lg - lo, r, phi, sv, 36, 0, 2.0).ring_mean + lo
        cache["prof"] = np.maximum(prof, math.log(4 * floor))  # no lifting of what is mostly noise
    prof = cache["prof"]
    base = _ring_at(prof, r)
    out = lg - np.float32(flatten) * base
    # Black point: the sky (pixels at the noise level), so it comes out dark (taken before any
    # evening out, which must not move the sky).
    sky = valid & (x < floor)
    black = float(np.median(out[sky])) if sky.sum() > 1000 else float(np.percentile(out[valid], 0.5))
    ring = loc = None
    if even > 0 and sun_r > 0:
        if "loc" not in cache:
            m = sv.astype(np.float32)
            sig = max(4.0, 0.25 * sun_r)
            cache["loc"] = (blur(np.where(sv, lg, 0.0).astype(np.float32), sig),
                            blur(np.where(sv, base, 0.0).astype(np.float32), sig), blur(m, sig))
        loc = cache["loc"]
        lmap = _loc_map(loc, flatten)
        lo2 = float(lmap[sv].min()) if sv.any() else 0.0
        ring = ring_stats(lmap - lo2, r, phi, sv, 36, 0, 2.0).ring_mean + lo2
        out = _even(out, lg, lmap, _ring_at(ring, r), r, floor, even, sun_r)
    white = float(np.percentile(out[valid], 99.9))
    mid = None
    if midtone > 0 and sun_r > 0:
        y = np.clip(roll_off((out - black) / ((white - black) or 1.0), shoulder), 0.0, 1.0)
        sel = valid & (r > 1.05 * sun_r) & (r < 2.5 * sun_r)
        mid = float(np.median(y[sel])) if sel.any() else None
    return NaturalStats(floor, prof, black, white, ring, loc, mid)


def _loc_map(loc: tuple, flatten: float) -> np.ndarray:
    """The large-scale mean of (lg - flatten * base), from its three blurred parts (blurs are
    linear, so this is exactly the blur of the flattened image, for any flatten)."""
    b_lg, b_base, b_m = loc
    return ((b_lg - np.float32(flatten) * b_base) / np.maximum(b_m, 1e-6)).astype(np.float32)


def _even(out, lg, lmap, ring, r, floor, even, sun_r):
    # Streamers against coronal holes: the large-scale differences round each circle, only where
    # the corona stands well above the noise (the sky stays where it is), and fading in from 1.15
    # to 1.4 solar radii: right by the limb the corona's contrast is so steep that taking out its
    # local mean pushed the dark gaps between bright loops to black.
    above = np.clip((lg - math.log(4 * floor)) / 2.0, 0.0, 1.0)
    rin = np.clip((r / sun_r - 1.15) / 0.25, 0.0, 1.0)
    return out - np.float32(even) * (above * rin * (lmap - ring)).astype(np.float32)


def natural_apply(x: np.ndarray, r: np.ndarray, st: NaturalStats, flatten: float, shoulder: float = 0.0,
                  even: float = 0.0, sun_r: float = 0.0, midtone: float = 0.0,
                  loc: Optional[tuple] = None, show: Optional[np.ndarray] = None) -> np.ndarray:
    """The natural layer at each pixel from the whole image's measurements: a region (``x``, ``r``
    and the ``loc`` maps cut to it) gives exactly the whole image's result there."""
    lg = np.log(np.maximum(x, 0) + st.floor).astype(np.float32)
    out = lg - np.float32(flatten) * _ring_at(st.prof, r)
    if even > 0 and sun_r > 0 and st.ring is not None:
        out = _even(out, lg, _loc_map(loc if loc is not None else st.loc, flatten), _ring_at(st.ring, r), r,
                    st.floor, even, sun_r)
    out = np.clip(roll_off((out - st.black) / ((st.white - st.black) or 1.0), shoulder), 0.0, 1.0).astype(np.float32)
    if midtone > 0 and st.mid is not None:
        out = _mtf(out, st.mid, midtone)
    if show is not None:
        out[~show] = 0.0
    return out


def natural(x: np.ndarray, r: np.ndarray, phi: np.ndarray, valid: np.ndarray, noise_level: float,
            flatten: float, stats_valid: Optional[np.ndarray] = None, show: Optional[np.ndarray] = None,
            shoulder: float = 0.0, even: float = 0.0, sun_r: float = 0.0, midtone: float = 0.0) -> np.ndarray:
    """Log stretch with part of the radial fall-off removed (the whole image at once).

    ``x`` has the sky removed. The flattening stops where the corona sinks
    into the noise, so the sky beyond stays dark instead of being lifted.
    """
    st = natural_stats(x, r, phi, valid, noise_level, flatten, stats_valid, shoulder, even, sun_r, midtone)
    return natural_apply(x, r, st, flatten, shoulder, even, sun_r, midtone, show=valid if show is None else show)


def _mtf(out: np.ndarray, a: float, midtone: float) -> np.ndarray:
    """The midtones transfer curve that takes ``a`` to ``midtone``."""
    t = min(max(midtone, 0.05), 0.95)
    if not 0.0 < a < 1.0:
        return out
    m = a * (1 - t) / (a - 2 * t * a + t)
    x = np.clip(out, 0.0, 1.0)
    return ((m - 1) * x / np.minimum((2 * m - 1) * x - m, -1e-9)).astype(np.float32)


def soft_light(base: np.ndarray, top: np.ndarray) -> np.ndarray:
    """Photoshop-style soft light (Pegtop form): 50 % grey on top leaves the base unchanged."""
    return (1 - 2 * top) * base * base + 2 * top * base


def overlay(base: np.ndarray, top: np.ndarray) -> np.ndarray:
    """Photoshop's overlay: 50 % grey on top leaves the base unchanged; stronger than soft light,
    and it boosts contrast most in the mid-tones of the base."""
    return np.where(base < 0.5, 2 * base * top, 1 - 2 * (1 - base) * (1 - top))


BLEND_MODES = {"soft light": soft_light, "overlay": overlay}
# Linear (paper) is not a layer mode applied in turn: see soft_light_blend.
MODE_NAMES = ("soft light", "overlay", "linear")


def blend_mode_name(text) -> str:
    """'Soft Light', 'softlight', 'Linear (paper)', ... -> 'soft light', 'overlay' or 'linear'."""
    m = str(text).strip().lower().replace("_", " ").replace("softlight", "soft light")
    if m.startswith("linear"):
        return "linear"
    return m


def soft_light_blend(layers: dict, weights: dict, valid: np.ndarray, modes: Optional[dict] = None,
                     show: Optional[np.ndarray] = None, fade: Optional[np.ndarray] = None,
                     shoulder: float = 0.0) -> np.ndarray:
    """The natural layer as the base, each filter layer applied at its weight (like layer opacity
    in Photoshop) in its mode, soft light unless ``modes`` says overlay. The filter layers are
    centred on mid-grey, which both modes treat as 'no change', so they add structure without
    lifting the dark sky. Layers are applied in the order natural, NRGF, FNRGF, FNRGF-N, adaptive.

    Linear (paper) layers are then combined with that base as the papers combine the original and
    the filtered image: a weighted average on the display range (FNRGF: Z = K1 X + K2 Y with only
    the ratio mattering, both images scaled to 0..1 in the thesis's software; MGN: h C'g + (1 - h)
    C'), the base with the natural layer's weight (1, K1) and each layer with its own (K2, which may
    exceed 1). Nothing clips and nothing is measured, so a detail box matches the full size."""
    modes = modes or {}
    base = layers["natural"].copy() if "natural" in weights else np.full_like(next(iter(layers.values())), 0.5)
    lin = []
    for k in LAYERS:
        if k == "natural" or k not in weights:
            continue
        name = blend_mode_name(modes.get(k, "soft light"))
        if name not in MODE_NAMES:
            raise ValueError(f"unknown blend mode {modes[k]!r} for {k}; use soft light, overlay or linear")
        if name == "linear":
            lin.append((k, max(float(weights[k]), 0.0)))
            continue
        w = min(max(weights[k], 0.0), 1.0)
        fn = BLEND_MODES[name]
        if fade is not None:  # filter layers fade out towards the Moon, where they have no data
            wf = w * fade
            base = (1 - wf) * base + wf * fn(base, layers[k])
        else:
            base = (1 - w) * base + w * fn(base, layers[k])
    if lin:
        k1 = max(float(weights.get("natural", 1.0)), 0.0) if "natural" in weights else 1.0
        num = k1 * base
        den = np.full_like(base, k1)
        for k, w in lin:
            wk = w * fade if fade is not None else np.float32(w)
            num = num + wk * layers[k]
            den = den + wk
        base = (num / np.maximum(den, 1e-12)).astype(np.float32)
    # No stretch measured over the whole image (a detail box must give the full-size result): the
    # natural base already runs from black to white, and every mode keeps 0..1. The natural
    # layer's midtone curve is already in the base, so it is not measured again here.
    out = np.clip(roll_off(base, shoulder), 0.0, 1.0).astype(np.float32)
    out[~(valid if show is None else show)] = 0.0
    return out


def _masked_fill(b: np.ndarray, valid: np.ndarray, r: float) -> np.ndarray:
    """``b`` with the invalid pixels (the Moon, frame edges) filled smoothly from around them:
    normalized blurs of widening size until every pixel is reached, then smoothed, so the filled
    part carries no edge or level step into the filters."""
    m = valid.astype(np.float32)
    f = np.where(valid, b, 0.0).astype(np.float32)
    have = valid.copy()
    sig = max(2.0, 0.02 * r)
    while not have.all() and sig < 4 * max(b.shape):
        den = blur(m, sig)
        num = blur(np.where(valid, b, 0.0).astype(np.float32), sig)
        ok = ~have & (den > 0.02)
        f = np.where(ok, num / np.maximum(den, 1e-6), f)
        have |= ok
        sig *= 2.0
    s2 = max(4.0, 0.04 * r)
    for _ in range(3):
        f = np.where(valid, b, blur(f, s2)).astype(np.float32)
    return f


def noise_kernel(valid: np.ndarray, sample: np.ndarray, patch: int = 64) -> Optional[np.ndarray]:
    """The convolution kernel giving unit noise the spatial spectrum measured in ``sample`` over
    ``valid`` (demosaicing and the alignment's resampling correlate neighbouring pixels, so white
    noise understates the noise left after any smoothing); None if too little to measure. The
    spectrum is the average over patches of the residual after a 6 px blur, so smooth structure in
    the sample does not count as noise."""
    r = (sample - ndimage.gaussian_filter(sample, 6.0)).astype(np.float32)
    h, w = r.shape
    win = np.outer(np.hanning(patch), np.hanning(patch))
    psd = np.zeros((patch, patch))
    count = 0
    for need in (0.8, 0.0):  # patches mostly in ``valid``; if too few, any fully usable patch
        for y0 in range(0, h - patch + 1, patch):
            for x0 in range(0, w - patch + 1, patch):
                if (np.isfinite(r[y0:y0 + patch, x0:x0 + patch]).all()
                        and valid[y0:y0 + patch, x0:x0 + patch].mean() >= max(need, 1e-9)):
                    q = r[y0:y0 + patch, x0:x0 + patch]
                    psd += np.abs(np.fft.fft2((q - q.mean()) * win)) ** 2
                    count += 1
        if count >= 4:
            break
    if count < 4:
        return None
    # Undo the 6 px high-pass where it let enough through; below that frequency the noise is
    # taken as flat (demosaicing and resampling correlate only neighbouring pixels).
    fy = np.fft.fftfreq(patch)[:, None]
    fx = np.fft.fftfreq(patch)[None, :]
    f = np.hypot(fy, fx)
    t = 1.0 - np.exp(-2.0 * math.pi**2 * 36.0 * f**2)
    psd = psd / count
    ok = t > 0.5
    corr = np.where(ok, psd / np.maximum(t, 1e-6) ** 2, 0.0)
    f_cut = float(f[ok].min())
    ring = ok & (f < f_cut * 1.5)
    corr = np.where(ok, corr, float(corr[ring].mean()))
    kern = np.real(np.fft.fftshift(np.fft.ifft2(np.sqrt(corr))))
    kern /= np.sqrt((kern**2).sum())
    return kern.astype(np.float32)


NOISE_TILE = 256


def tiled_noise(shape: tuple[int, int], origin: tuple[int, int], seed: int,
                kern: Optional[np.ndarray]) -> np.ndarray:
    """Noise for the region of the whole image starting at ``origin``: white noise drawn per fixed
    256 px tile of the whole image (each tile from its own seed), then given the measured spectrum.
    A region gets exactly the noise the whole image has there."""
    h, w = shape
    oy, ox = int(origin[0]), int(origin[1])
    pad = (kern.shape[0] // 2 + 1) if kern is not None else 0
    y0, y1, x0, x1 = oy - pad, oy + h + pad, ox - pad, ox + w + pad
    T = NOISE_TILE
    ty0, ty1 = y0 // T, (y1 - 1) // T
    tx0, tx1 = x0 // T, (x1 - 1) // T
    big = np.empty(((ty1 - ty0 + 1) * T, (tx1 - tx0 + 1) * T), np.float32)
    for ty in range(ty0, ty1 + 1):
        for tx in range(tx0, tx1 + 1):
            rng = np.random.default_rng([int(seed) + 7, ty + 1_000_000, tx + 1_000_000])
            big[(ty - ty0) * T:(ty - ty0 + 1) * T, (tx - tx0) * T:(tx - tx0 + 1) * T] = rng.normal(0.0, 1.0, (T, T))
    n = big[y0 - ty0 * T:y1 - ty0 * T, x0 - tx0 * T:x1 - tx0 * T]
    if kern is not None:
        from scipy.signal import fftconvolve

        n = fftconvolve(n, kern, mode="same").astype(np.float32)
    return n[pad:pad + h, pad:pad + w]


def noise_norm(kern: Optional[np.ndarray], seed: int = 0) -> float:
    """The factor that makes :func:`tiled_noise` read 1 on the 1-px scatter statistic used by
    :func:`noise_map` (measured on a fixed 1024 px sample, so the same everywhere)."""
    n = tiled_noise((1024, 1024), (0, 0), seed, kern)
    hp = n - ndimage.gaussian_filter(n, 1.0)
    return 1.4826 * float(np.median(np.abs(hp))) / 0.96


def correlated_noise(shape, valid: np.ndarray, sample: np.ndarray, seed: int = 0, patch: int = 64) -> np.ndarray:
    """Unit noise with the spatial spectrum measured in ``sample`` over ``valid``, scaled so that
    the 1-px scatter statistic used by :func:`noise_map` reads 1."""
    kern = noise_kernel(valid, sample, patch)
    return (tiled_noise(shape, (0, 0), seed, kern) / noise_norm(kern, seed)).astype(np.float32)


def gated_prep(x: np.ndarray, valid: np.ndarray, floor: float, sun_r: float, seed: int = 0) -> dict:
    """What the noise-gated adaptive layer measures over the whole image: the log image with the
    Moon and frame edges filled (``bf``), and the noise model (kernel and scale)."""
    b = np.log(np.maximum(x, 0) + max(floor, 1e-9)).astype(np.float32)
    bf = _masked_fill(b, valid, sun_r)
    smooth_b = blur(bf, max(4.0, 0.05 * sun_r))  # the faint part, as a region (not single pixels)
    faint = valid & (smooth_b < np.percentile(smooth_b[valid], 30))
    kern = noise_kernel(faint, bf)
    return {"bf": bf, "kern": kern, "norm": noise_norm(kern, seed), "seed": seed}


def mgn_gated(x: np.ndarray, valid: np.ndarray, scales: Sequence[float], k: float, floor: float,
              noise: np.ndarray, sun_r: float, gate: float = 3.0, seed: int = 0, prep: Optional[dict] = None,
              origin: tuple[int, int] = (0, 0), weights: Optional[Sequence[float]] = None) -> np.ndarray:
    """:func:`mgn` with the noise kept down: each scale's detail weighted by how far it stands above
    the noise at that scale (a Wiener gain; the noise at each scale comes from a noise image with
    the stack's measured noise and correlation). Pure noise gives almost no contrast.

    ``prep`` (from :func:`gated_prep` on the whole image) and ``origin`` let it run on a region."""
    if prep is None:
        prep = gated_prep(x, valid, floor, sun_r, seed)
    bf = prep["bf"]
    sig = np.where(valid, noise / np.maximum(np.maximum(x, 0) + max(floor, 1e-9), 1e-12), 0.0)
    vm = valid.astype(np.float32)
    sig = ndimage.gaussian_filter(sig.astype(np.float32), 4.0) / np.maximum(ndimage.gaussian_filter(vm, 4.0), 1e-3)
    nz = tiled_noise(bf.shape, origin, prep["seed"], prep["kern"]) / prep["norm"] * sig
    del sig, vm  # full frames are 45 megapixels: free what is done with
    nz = np.where(valid, nz, 0.0).astype(np.float32)
    acc = np.zeros_like(bf)
    g2 = gate * gate
    gs = [1.0] * len(scales) if weights is None else [max(float(v), 0.0) for v in weights]
    if len(gs) != len(scales):
        raise ValueError(f"Adaptive (MGN) needs one strength per scale ({len(scales)}), got {len(gs)}")
    for w, gi in zip(scales, gs):
        if gi <= 0:
            continue
        d = bf - blur(bf, w, origin)
        sd2 = blur(d * d, w, origin)
        dn = nz - blur(nz, w, origin)
        # Both judged over a wider area than the scale itself: over a few pixels the estimates
        # scatter so much that noise passes the gate by chance.
        wide = max(3.0 * w, 8.0)
        n2 = blur(dn * dn, wide, origin)
        s2 = blur(d * d, wide, origin)
        gain = np.maximum(s2 - n2, 0.0) / np.maximum(s2 + (g2 - 1.0) * n2, 1e-12) if gate > 0 else 1.0
        acc += np.float32(gi) * np.arctan(k * gain * d / np.sqrt(np.maximum(sd2, 1e-12)))
        del d, sd2, dn, n2, s2, gain
    acc /= len(scales)
    # The paper's straight mapping, as the plain filter (never stretched to the image's spread, which
    # would bring noise back up wherever there is little real structure): the gate only lowers each
    # scale's share.
    out = mgn_unit(acc)
    out[~valid] = 0.0
    return out


def _gauss_rows(a: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur down the columns (axis 0) by FFT, edges extended by their own values (as
    ``mode="nearest"``); the same result as ``gaussian_filter1d`` and far faster for wide kernels."""
    from scipy.fft import irfft, next_fast_len, rfft

    pad = int(math.ceil(4 * sigma)) + 1
    ap = np.pad(a, ((pad, pad), (0, 0)), mode="edge")
    n = next_fast_len(ap.shape[0])
    k = np.arange(n // 2 + 1, dtype=np.float64)[:, None]
    tf = np.exp(-0.5 * (2 * math.pi * sigma * k / n) ** 2)
    out = irfft(rfft(ap, n=n, axis=0) * tf, n=n, axis=0)
    # The FFT wraps round: the padded edges (4 sigma) keep the far end out of the kept rows.
    return out[pad:pad + a.shape[0]]


def achf_mean(f: np.ndarray, w: np.ndarray, cx: float, cy: float, sigma: float,
              circular: Optional[bool] = None, band: int = 192, sigma_radial: Optional[float] = None,
              r_ref: Optional[float] = None) -> np.ndarray:
    """The local mean under Druckmüller's adaptive circular kernel (Druckmüllerová 2013, thesis
    section 5.2, eq. 5.1-5.2): a Gaussian of width ``sigma`` across the radius and ``sigma`` along
    the circle round the Sun's centre (cx, cy), in an incomplete convolution where pixels with
    ``w`` = 0 (the Moon, outside the image) do not count.

    Computed in Sun-centred polar coordinates, one ring of radii at a time: rows 1 px apart in
    radius, the circle sampled at least 1 px apart. Blurring each column across the radius and then
    each row along its circle (width sigma / r in angle, by FFT, as the circle is periodic) gives
    exactly the kernel of eq. 5.1. Where the circle is nearly straight over the kernel (its sagitta
    2 sigma^2 / r stays under 0.1 px down to the smallest radius used) an ordinary weighted Gaussian
    is the same thing and much faster; ``circular`` forces either way. ``sigma_radial`` (default
    ``sigma``) gives the kernel another width across the radius (always computed on the circles).
    """
    from scipy.fft import irfft, next_fast_len, rfft

    f = np.asarray(f, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    h, wd = f.shape
    yy = np.arange(h, dtype=np.float64)[:, None] - cy
    xx = np.arange(wd, dtype=np.float64)[None, :] - cx
    r = np.sqrt(xx * xx + yy * yy)
    used = w > 0
    # The smallest radius that matters: given (the Moon's radius, so a detail box decides as the
    # whole image does), or the smallest radius with data.
    r_min = float(r_ref) if r_ref is not None else (float(r[used].min()) if used.any() else 0.0)
    sr = sigma if sigma_radial is None else float(sigma_radial)
    if sr != sigma:
        circular = True
    if circular is None:
        circular = 2.0 * sigma * sigma / max(r_min, 1.0) >= 0.1
    fw = f * w
    if not circular:
        num = ndimage.gaussian_filter(fw, sigma, mode="constant")
        den = ndimage.gaussian_filter(w, sigma, mode="constant")
        return np.where(den > 1e-9, num / np.maximum(den, 1e-300), f)
    phi = np.mod(np.arctan2(yy, xx), 2 * math.pi)
    out = f.copy()
    # Cubic-spline coefficients once (map_coordinates would recompute them for every ring).
    cf = ndimage.spline_filter(fw, order=3, mode="constant")
    cw = ndimage.spline_filter(w, order=3, mode="constant")
    margin = int(math.ceil(4 * sr)) + 2
    r_max = float(r.max())
    for r0 in range(0, int(math.ceil(r_max)) + 1, band):
        r1 = r0 + band
        sel = (r >= r0) & (r < r1)
        if not sel.any():
            continue
        rows = np.arange(max(0, r0 - margin), r1 + margin + 1, dtype=np.float64)
        n_phi = next_fast_len(max(64, int(math.ceil(2 * math.pi * rows[-1]))))
        ang = np.arange(n_phi) * (2 * math.pi / n_phi)
        py = cy + rows[:, None] * np.sin(ang)[None, :]
        px = cx + rows[:, None] * np.cos(ang)[None, :]
        # Cubic in and out: linear interpolation both ways blurred fine structure measurably (3.6%
        # of a texture's spread at sigma 3 px, against 1.8%; the rest is eq. 5.1's even sampling
        # in angle, which weights the outer side of the kernel a little less than a flat Gaussian).
        pf = ndimage.map_coordinates(cf, [py, px], order=3, mode="constant", cval=0.0, prefilter=False)
        pw = ndimage.map_coordinates(cw, [py, px], order=3, mode="constant", cval=0.0, prefilter=False)
        del py, px
        # Across the radius (constant sigma, rows are 1 px apart) ...
        if sr > 0:
            pf = _gauss_rows(pf, sr)
            pw = _gauss_rows(pw, sr)
        # ... then along each circle, sigma measured along the circle of that row (eq. 5.1).
        k = np.arange(n_phi // 2 + 1, dtype=np.float64)[None, :]
        s_ang = sigma / np.maximum(rows, 0.5)[:, None]
        tf = np.exp(-0.5 * (s_ang * k) ** 2)
        pf = irfft(rfft(pf, axis=1) * tf, n=n_phi, axis=1)
        pw = irfft(rfft(pw, axis=1) * tf, n=n_phi, axis=1)
        mean = np.where(pw > 1e-9, pf / np.maximum(pw, 1e-300), 0.0)
        del pf, tf
        ri = r[sel] - rows[0]
        pj = phi[sel] * (n_phi / (2 * math.pi))
        m = ndimage.map_coordinates(mean, [ri, pj], order=3, mode="grid-wrap")
        ok = ndimage.map_coordinates(pw, [ri, pj], order=1, mode="grid-wrap") > 1e-9
        out[sel] = np.where(ok, m, f[sel])
        del mean, pw
    return out


def achf_parts(b: np.ndarray, valid: np.ndarray, cx: float, cy: float, limit: float,
               ref_sigma: float, r_ref: Optional[float] = None) -> list[np.ndarray]:
    """Requirement (d) of the thesis (section 5.2): pixels in significantly different parts of the
    image do not count in each other's kernels, judged by a maximum acceptable difference in value.
    Here: a pixel more than ``limit`` times brighter (or darker) than the corona at its height (the
    mean along its circle, ``ref_sigma`` each way, taken without such pixels, twice so a wide
    prominence does not hide itself) is in a part of its own. Returns the masks of the parts:
    the corona, the brighter part, the darker part (empty ones left out)."""
    if limit <= 1.0:
        return [valid]
    t = math.log(limit)
    normal = valid
    for _ in range(2):
        # Along the pixel's own circle only (1 px across the radius): a mean reaching across the
        # radius is pulled by the corona's steep fall-off, and by the Moon and the frame edges
        # (where it is one-sided) far enough to make a whole ring by the limb a "part".
        ref = achf_mean(b, normal.astype(np.float64), cx, cy, ref_sigma, sigma_radial=1.0, r_ref=r_ref)
        bright = valid & (b - ref > t)
        dark = valid & (b - ref < -t)
        normal = valid & ~bright & ~dark
    return [m for m in (normal, bright, dark) if m.any()]


def achf(x: np.ndarray, valid: np.ndarray, scales: Sequence[float], floor: float,
         cx: float, cy: float, weights: Optional[Sequence[float]] = None, part_limit: float = 2.0,
         ref_sigma: Optional[float] = None, r_ref: Optional[float] = None, contrast: float = 1.0,
         raw: bool = False) -> np.ndarray:
    """Adaptive (ACHF): Druckmüller's adaptive circular high-pass filter (Druckmüller, Rušin &
    Minarovjech 2006; defined in Druckmüllerová 2013, section 5.2). For each scale sigma the
    filtered image is the pixel minus its local mean under the circular kernel (:func:`achf_mean`),
    the Moon and unusable pixels left out; the scales are summed with equal weights, which gives
    an amplification rising with spatial frequency; ``weights`` (one per scale, default all 1)
    change each scale's share, like the frequency sliders of Druckmüller's Corona program.

    Not published, so chosen here: it works on log brightness (as Adaptive (MGN) does, so the same
    contrast counts the same at every height), uses the same scales as Adaptive (MGN), and is
    centred on mid-grey for the blend, whose natural-layer base plays the part of the original
    image's share (K1 in the thesis). Druckmüller's later "nonlinearity in the use of filtered
    images" (Corona 4.1) is not published and not reproduced.
    """
    b = np.log(np.maximum(x, 0) + max(floor, 1e-9))
    w = valid.astype(np.float64)
    ws = [1.0] * len(scales) if weights is None else [max(float(v), 0.0) for v in weights]
    if len(ws) != len(scales):
        raise ValueError(f"Adaptive (ACHF) needs one strength per scale ({len(scales)}), got {len(ws)}")
    used = [s for s, ks in zip(scales, ws) if ks > 0]
    parts = [valid]
    if part_limit > 1.0 and used:
        rs = ref_sigma if ref_sigma else max(4.0 * max(used), 8.0)
        parts = achf_parts(b, valid, cx, cy, part_limit, rs, r_ref)
    acc = np.zeros(b.shape, np.float64)
    for s, ks in zip(scales, ws):
        if ks > 0:  # a scale at 0 is not computed at all (and saves its time)
            # Each part's pixels are averaged only with pixels of the same part (eq. 5.2's weight 0
            # for the others), so a prominence neither darkens the corona beside it nor the reverse.
            mean = np.zeros(b.shape, np.float64)
            for m in parts:
                mean[m] = achf_mean(b, m.astype(np.float64), cx, cy, float(s), r_ref=r_ref)[m]
            acc += ks * np.where(valid, b - mean, 0.0)
    if sum(ws) <= 0:
        if raw:
            return np.zeros(b.shape, np.float32)
        out = np.where(valid, 0.5, 0.0).astype(np.float32)
        return out
    acc /= sum(ws)
    if raw:  # the filtered image before Contrast and the display scale (see achf_display)
        return np.where(valid, acc, 0.0).astype(np.float32)
    # A fixed output scale (the thesis's K2 is the Contrast setting): measured over the whole image
    # it would make a detail box differ from the full-size result.
    out = to_fixed((acc * contrast).astype(np.float32), ACHF_SPREAD)
    out[~valid] = 0.0
    return out


def achf_display(acc: np.ndarray, valid: np.ndarray, contrast: float) -> np.ndarray:
    """ACHF's filtered image (``achf(..., raw=True)``) as the layer: Contrast, then the fixed scale."""
    out = to_fixed((acc * np.float32(contrast)).astype(np.float32), ACHF_SPREAD)
    out[~valid] = 0.0
    return out


def _achf_reach(scales: Sequence[float], weights: Sequence[float], part_limit: float, ref_sigma: float) -> int:
    used = [s for s, k in zip(scales, weights) if k > 0]
    if not used:
        return 0
    reach = int(math.ceil(4 * max(used))) + 8
    if part_limit > 1.0:  # the parts' reference reaches 4 ref_sigma along the circle, twice
        reach += 2 * int(math.ceil(4 * ref_sigma)) + 8
    return reach


def mgn(x: np.ndarray, valid: np.ndarray, scales: Sequence[float], k: float, floor: float,
        origin: tuple[int, int] = (0, 0), weights: Optional[Sequence[float]] = None) -> np.ndarray:
    """Multi-scale Gaussian normalization (Morgan & Druckmüller 2014) of log(X); local part only,
    on the paper's own scale: 0.5 + (1/n sum g_i arctan(k C_i)) / pi (see :func:`mgn_unit`), the
    weights g_i (default all 1, the plain mean) as in the paper's eq. 5; a scale at 0 is skipped."""
    b = np.log(np.maximum(x, 0) + max(floor, 1e-9)).astype(np.float32)
    # Fill invalid pixels with their surroundings so the Moon and frame edges do not ring.
    m = valid.astype(np.float32)
    fill = blur(b * m, 8.0, origin) / np.maximum(blur(m, 8.0, origin), 1e-6)
    b = np.where(valid, b, fill).astype(np.float32)
    acc = np.zeros_like(b)
    gs = [1.0] * len(scales) if weights is None else [max(float(v), 0.0) for v in weights]
    if len(gs) != len(scales):
        raise ValueError(f"Adaptive (MGN) needs one strength per scale ({len(scales)}), got {len(gs)}")
    for w, gi in zip(scales, gs):
        if gi <= 0:
            continue
        d = b - blur(b, w, origin)
        sd = np.sqrt(np.maximum(blur(d * d, w, origin), 1e-12))
        acc += np.float32(gi) * np.arctan(k * d / sd)
    acc /= len(scales)
    out = mgn_unit(acc)
    out[~valid] = 0.0
    return out


def _mgn_reach(scales: Sequence[float], gated: bool) -> int:
    """How far the adaptive layer reaches from a pixel (for a detail box's margin)."""
    w = max(scales)
    reach = _blur_reach(8.0) + _blur_reach(w) + _blur_reach(w)  # fill, detail, its spread
    if gated:  # the noise image (kernel 64 px, 4 px smoothing) and the 3x wider judging blurs
        reach = max(reach, _blur_reach(8.0) + _blur_reach(w) + _blur_reach(max(3 * w, 8.0)) + 32 + 16)
    return reach


# ---------------------------------------------------------------------------
# Colour
# ---------------------------------------------------------------------------


def sky_background(rgb: np.ndarray, r: np.ndarray, valid: np.ndarray, g: Geometry) -> np.ndarray:
    """Per-channel level of the darkest sky, far from the Sun.

    Twilight sky, airglow and small black-level residuals add a constant to
    each channel. It is invisible in the bright inner corona but a log
    stretch or a radial filter brings the faint outer corona up to the same
    brightness, and then a few counts' difference between channels shows as
    an overall colour cast.
    """
    far = valid & (r > 3.0 * g.sun_r)
    if far.sum() < 1000:
        far = valid & (r > np.percentile(r[valid], 80)) if valid.any() else valid
    if far.sum() < 100:
        return np.zeros(3, np.float32)
    lum = rgb[..., :3].mean(axis=2)[far]
    dark = lum <= np.percentile(lum, 5)
    return np.array([np.median(rgb[..., c][far][dark]) for c in range(3)], np.float32)


def _sky_cells(img: np.ndarray, r: np.ndarray, valid: np.ndarray, g: Geometry, cell: int):
    """Darkest-pixel level (20th percentile) in grid cells clear of the inner corona."""
    h, w = img.shape
    xs, ys, rs, vals = [], [], [], []
    far = valid & (r > 2.5 * g.sun_r)
    for y0 in range(0, h - cell + 1, cell):
        for x0 in range(0, w - cell + 1, cell):
            m = far[y0:y0 + cell, x0:x0 + cell]
            if m.mean() < 0.5:
                continue
            v = img[y0:y0 + cell, x0:x0 + cell][m]
            xs.append(x0 + cell / 2)
            ys.append(y0 + cell / 2)
            rs.append(float(r[y0 + cell // 2, x0 + cell // 2]))
            vals.append(float(np.percentile(v, 20)))
    return np.array(xs), np.array(ys), np.array(rs), np.array(vals)


def sky_model(rgb: np.ndarray, r: np.ndarray, valid: np.ndarray, g: Geometry) -> tuple[list, str]:
    """Smooth sky (twilight glow, gradient, black-level residual) per channel, as polynomial coefficients.

    Fitted to the darkest pixels of grid cells away from the inner corona,
    as a quadratic surface plus radial power-law terms. The power laws
    soak up the faint outer corona, so only the quadratic part (the sky) is
    removed. With too few cells (a tight crop) the sky is a constant.
    """
    h, w = r.shape
    cell = max(16, min(h, w) // 24)
    cx, cy, sc = w / 2.0, h / 2.0, max(h, w) / 2.0
    models = []
    kind = "gradient"
    for c in range(rgb.shape[2]):
        xs, ys, rs, vals = _sky_cells(rgb[..., c], r, valid, g, cell)
        if len(vals) < 30:
            far = valid & (r > 3.0 * g.sun_r)
            if far.sum() < 100:
                far = valid
            v = rgb[..., c][far]
            level = float(np.median(v[v <= np.percentile(v, 5)])) if v.size else 0.0
            models.append([level, 0, 0, 0, 0, 0])
            kind = "constant"
            continue
        u, v = (xs - cx) / sc, (ys - cy) / sc
        rr = np.maximum(rs / g.sun_r, 1.0)
        A = np.stack([np.ones_like(u), u, v, u * u, u * v, v * v, rr**-2, rr**-3, rr**-5], axis=1)
        wts = np.ones(len(vals))
        coef = np.zeros(A.shape[1])
        for _ in range(6):
            sw = np.sqrt(wts)[:, None]
            coef, *_ = np.linalg.lstsq(A * sw, vals * sw[:, 0], rcond=None)
            res = vals - A @ coef
            s_ = 1.4826 * np.median(np.abs(res)) + 1e-12
            t = res / (4.685 * s_)
            wts = np.where(np.abs(t) < 1, (1 - t * t) ** 2, 0.0)
        models.append(coef[:6].tolist())
    return models, kind


def sky_surface(coef: Sequence[float], shape: tuple[int, int], rows: slice = slice(None)) -> np.ndarray:
    h, w = shape
    cx, cy, sc = w / 2.0, h / 2.0, max(h, w) / 2.0
    u = ((np.arange(w, dtype=np.float32) - cx) / sc)[None, :]
    v = ((np.arange(h, dtype=np.float32)[rows] - cy) / sc)[:, None]
    a = [np.float32(x) for x in coef]
    return a[0] + a[1] * u + a[2] * v + a[3] * u * u + a[4] * u * v + a[5] * v * v


def white_balance(rgb: np.ndarray, r: np.ndarray, valid: np.ndarray, g: Geometry, mode: str,
                  camera_wb: Optional[Sequence[float]], background: Optional[np.ndarray] = None) -> np.ndarray:
    """Channel multipliers. 'corona': the inner corona (1.2-2.5 R) is white, as K-corona light is
    scattered sunlight; 'camera': the as-shot multipliers stored with the stack."""
    if mode == "camera" and camera_wb:
        m = np.asarray(camera_wb, float)
    elif mode == "corona":
        ring = valid & (r > 1.2 * g.sun_r) & (r < 2.5 * g.sun_r)
        if ring.sum() < 100:
            ring = valid
        bg = np.zeros(3) if background is None else background
        med = np.array([np.median(rgb[..., c][ring]) - bg[c] for c in range(3)])
        m = med[1] / np.maximum(med, 1e-12)
    else:
        m = np.ones(3)
    return (m / m[1]).astype(np.float32)


def chroma(rgb: np.ndarray, lum: np.ndarray, valid: np.ndarray, noise: np.ndarray, saturation: float,
           colour_floor: float = 0.0) -> np.ndarray:
    """Smoothed colour ratios rgb/lum, faded to grey where the signal is buried in noise or is
    not clearly brighter than ``colour_floor`` (the uncertainty of the removed sky)."""
    out = np.ones_like(rgb)
    if saturation <= 0:
        return out
    sl = blur(np.where(valid, lum, 0).astype(np.float32), 3.0)
    for c in range(3):
        sc = blur(np.where(valid, rgb[..., c], 0).astype(np.float32), 3.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[..., c] = np.clip(np.nan_to_num(sc / sl, nan=1.0), 0.3, 3.0)
    snr = np.where(noise > 0, lum / np.maximum(noise, 1e-12) / 3.0, 10.0)
    fade = np.clip(snr / 20.0, 0.0, 1.0) * saturation
    if colour_floor > 0:
        fade *= np.clip((lum / colour_floor - 1.0) / 4.0, 0.0, 1.0)
    out = 1.0 + (out - 1.0) * fade[..., None]
    out /= np.maximum(out.mean(axis=2, keepdims=True), 1e-6)  # keep luminance
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


@dataclass
class EnhanceResult:
    layers: dict  # name -> output path (or array when not written)
    white: list
    noise_scale: float
    background: list
    scales: list  # Adaptive (MGN) kernel widths, px of the image rendered
    notes: list = field(default_factory=list)
    reduction: int = 1  # the preview's reduction (1 = full size)

    def describe(self) -> str:
        full = ", ".join(f"{s * self.reduction:g}" for s in self.scales)
        mgn = f"Adaptive (MGN) scales: {full} px of the full-size stack"
        if self.reduction > 1:
            mgn += (f" ({', '.join(f'{s:.2g}' for s in self.scales)} px in this preview, reduced "
                    f"{self.reduction}x)")
        lines = [f"White balance multipliers (R, G, B): {', '.join(f'{v:.3f}' for v in self.white)}",
                 f"Sky removed, typical level after white balance (R, G, B): "
                 f"{', '.join(f'{v:.3g}' for v in self.background)}",
                 mgn]
        lines += [f"{k}: {v}" for k, v in self.layers.items()]
        lines += [f"Note: {n}" for n in self.notes]
        return "\n".join(lines)


@dataclass
class Frame:
    """A stack made ready for enhancing (see :func:`prepare`): the sky removed and white balanced,
    and everything the layers measure over the whole image, so any region can be rendered alone
    (:func:`render`) and comes out exactly as in the full-size picture."""

    g: Geometry
    shape: tuple
    rgb: np.ndarray  # (h, w, 3) linear, sky removed, white balanced
    lum: np.ndarray
    weight: np.ndarray
    valid: np.ndarray  # pixels with corona data (off the Moon, inside the frame)
    show: np.ndarray  # pixels drawn by the natural layer and the blend (with the Moon if put in)
    stats_valid: np.ndarray  # valid without dust spots and stars (for the circle statistics)
    noise: np.ndarray
    nlev: float
    wb: np.ndarray
    bg: np.ndarray
    sky_kind: str
    colour_floor: float
    moon: Optional[MoonInput] = None
    moon_margin_px: float = 1.5
    cache: dict = field(default_factory=dict)  # whole-image measurements, by what they depend on


def prepare(hdr: np.ndarray, weight: np.ndarray, header: dict, params: Optional[EnhanceParams] = None,
            moon: Optional[MoonInput] = None, progress: Optional[Say] = None) -> Frame:
    """The whole-image part of enhancing: masks, sky, white balance, noise, dust spots. Depends only
    on the settings White and Neutral background (and the Moon); ``hdr`` is changed in place."""
    say = progress or (lambda _m: None)
    p = params or EnhanceParams()
    g = geometry_from_header(header)
    h, w = hdr.shape[:2]
    r, _ = polar((h, w), g)
    finite = np.isfinite(hdr).all(axis=2)
    hdr = np.asarray(hdr, dtype=np.float32)
    np.nan_to_num(hdr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)  # in place: saves 0.5 GB
    yy = np.arange(h, dtype=np.float32)[:, None] - np.float32(g.moon_y)
    xx = np.arange(w, dtype=np.float32)[None, :] - np.float32(g.moon_x)
    outside_moon = np.sqrt(xx * xx + yy * yy) > g.moon_r + p.moon_margin_px
    del xx, yy
    valid = finite & (weight > 0) & outside_moon
    # The outermost rows and columns of a RAW frame are often not image (Nikon includes a few
    # masked or edge columns) and the stack edges hold fewer frames; leave a thin border out.
    edge = max(4, int(round(0.004 * min(h, w))))
    valid[:edge] = False
    valid[-edge:] = False
    valid[:, :edge] = False
    valid[:, -edge:] = False
    # With the Moon put into the stack: the statistics still come from the corona only (valid),
    # but the natural layer and the blend are drawn over the Moon too, with the same curve.
    show = valid if moon is None else (valid | moon.mask)

    camera_wb = None
    if all(k in header for k in ("WBR", "WBG", "WBB")):
        camera_wb = [header["WBR"], header["WBG"], header["WBB"]]
    sky_kind = "kept"
    bg = np.zeros(3, np.float32)
    if p.neutral_background:
        models, sky_kind = sky_model(hdr, r, valid, g)
        for c in range(3):
            for r0 in range(0, h, 1024):
                rows = slice(r0, min(h, r0 + 1024))
                hdr[rows, :, c] -= sky_surface(models[c], (h, w), rows)
            bg[c] = float(np.median(sky_surface(models[c], (h, w))[valid])) if valid.any() else 0.0
    wb = white_balance(hdr, r, valid, g, p.white, camera_wb)
    hdr *= wb[None, None, :]
    bg = bg * wb
    lum = hdr.mean(axis=2)
    say(f"white balance {wb.round(3).tolist()}; sky removed ({sky_kind}) {bg.tolist()}")
    stats_valid = valid & ~spot_mask(lum, valid, r, 2.0 * g.sun_r)
    say(f"{int((valid & ~stats_valid).sum())} pixels in dust spots or stars left out of the circle statistics")
    del r
    noise = noise_map(lum, weight, valid)
    nlev = float(np.median(noise[valid & (noise > 0)])) if (valid & (noise > 0)).any() else 0.0
    # Per-channel sky offsets are only known to about their own size; where the corona is no
    # brighter than that, its colour cannot be trusted and is shown grey.
    return Frame(g, (h, w), hdr, lum, weight, valid, show, stats_valid, noise, nlev, wb, bg, sky_kind,
                 2.0 * float(np.abs(bg).max()), moon, p.moon_margin_px)


def _mgn_scales(p: EnhanceParams, g: Geometry) -> list:
    """The MGN kernel widths in px of the image rendered (the paper's by default)."""
    return list(p.mgn_scales) if p.mgn_scales else list(MGN_SCALES)


def _mgn_used(p: EnhanceParams, g: Geometry) -> list:
    """The widths of the scales with a strength above 0 (for the margin a box needs)."""
    return [w for w, gi in zip(_mgn_scales(p, g), p.mgn_weights) if gi > 0] or [1.0]


def needed_layers(p: EnhanceParams) -> set:
    return set(p.layers) | {k for k, v in p.blend.items() if v}


def render_margin(p: EnhanceParams, g: Geometry, need: Optional[set] = None) -> int:
    """How far outside a box the layers look (px): a box rendered with this margin is exact."""
    need = needed_layers(p) if need is None else need
    reach = _blur_reach(3.0)  # the colour ratios
    if "adaptive" in need:
        reach = max(reach, _mgn_reach(_mgn_used(p, g), p.noise_gate > 0))
    if "achf" in need:
        reach = max(reach, _achf_reach([g.sun_r * f for f in p.achf_scales], list(p.achf_weights),
                                       p.achf_part_limit, 0.1 * g.sun_r))
    return reach


def _ring_cache(frame: Frame, p: EnhanceParams) -> RingStats:
    key = ("ring", p.segments, p.order, p.ring_smooth)
    if key not in frame.cache:
        r, phi = polar(frame.shape, frame.g)
        frame.cache[key] = ring_stats(frame.lum, r, phi, frame.stats_valid, p.segments, p.order, p.ring_smooth,
                                      frame.noise)
    return frame.cache[key]


def _natural_cache(frame: Frame, p: EnhanceParams) -> NaturalStats:
    key = ("nat", float(p.flatten), float(p.even), float(p.shoulder), float(p.midtone))
    if key not in frame.cache:
        r, phi = polar(frame.shape, frame.g)
        frame.cache[key] = natural_stats(frame.lum, r, phi, frame.valid, frame.nlev, p.flatten, frame.stats_valid,
                                         p.shoulder, p.even, frame.g.sun_r, p.midtone,
                                         base_cache=frame.cache.setdefault("nat_base", {}))
    return frame.cache[key]


def _gated_cache(frame: Frame) -> dict:
    if "gated" not in frame.cache:
        frame.cache["gated"] = gated_prep(frame.lum, frame.valid, frame.nlev, frame.g.sun_r)
    return frame.cache["gated"]


# The settings each layer depends on (besides the prepared frame); see render(cache=True).
LAYER_SETTINGS = {
    "natural": ("flatten", "even", "shoulder", "midtone"),
    "nrgf": ("segments", "order", "ring_smooth"),
    "fnrgf": ("segments", "order", "ring_smooth", "mean_attenuation", "std_attenuation"),
    "fnrgf_n": ("segments", "order", "ring_smooth", "mean_attenuation", "std_attenuation", "noise_m"),
    "adaptive": ("mgn_scales", "mgn_weights", "mgn_k", "noise_gate"),
    "achf": ("achf_scales", "achf_weights", "achf_part_limit"),  # Contrast is applied after (achf_display)
}


def layer_key(name: str, p: EnhanceParams) -> tuple:
    def h(v):
        return tuple(h(x) for x in v) if isinstance(v, (list, tuple)) else v

    return tuple(h(getattr(p, k)) for k in LAYER_SETTINGS[name])


def render(frame: Frame, params: Optional[EnhanceParams] = None, box: Optional[tuple] = None,
           keep: Optional[set] = None, progress: Optional[Say] = None,
           emit: Optional[Callable[[str, np.ndarray], None]] = None, cache: bool = False) -> dict:
    """The coloured layers ({name: (h, w, 3) float32 0..1}) of ``box`` = (y0, y1, x0, x1) of the
    whole image (default: all of it), exactly as in the full-size picture: whole-image measurements
    come from ``frame`` (and are kept in ``frame.cache``), the local filters run on the box plus
    their reach. With ``emit``, each coloured layer is handed to it as soon as it is made and not
    kept (a full 45 MP colour layer is half a gigabyte). With ``cache`` (previews), each layer's
    result for this area is kept in ``frame.cache`` with its own settings (:func:`layer_key`) and
    used again while they stay the same, so a change redraws only the layers it concerns."""
    say = progress or (lambda _m: None)
    p = params or EnhanceParams()
    g = frame.g
    H, W = frame.shape
    y0, y1, x0, x1 = (0, H, 0, W) if box is None else (int(v) for v in box)
    y0, x0 = max(0, y0), max(0, x0)
    y1, x1 = min(H, y1), min(W, x1)
    if y1 <= y0 or x1 <= x0:
        raise ValueError(f"empty box {box}")
    need = needed_layers(p)
    whole = box is None
    inner_box = (y0, y1, x0, x1)
    store = frame.cache.setdefault("layers", {}) if cache else None

    def region(m):
        """The box plus ``m`` px (inside the image): slices, origin, and the box within it."""
        Y0, Y1, X0, X1 = max(0, y0 - m), min(H, y1 + m), max(0, x0 - m), min(W, x1 + m)
        return ((slice(Y0, Y1), slice(X0, X1)), (Y0, X0),
                (slice(y0 - Y0, y1 - Y0), slice(x0 - X0, x1 - X0)), (Y1 - Y0, X1 - X0))

    box_reg = (slice(y0, y1), slice(x0, x1))
    valid, show = frame.valid[box_reg], frame.show[box_reg]
    fade = None
    if frame.moon is not None:
        yy = (np.arange(y0, y1, dtype=np.float64) - g.moon_y).astype(np.float32)[:, None]
        xx = (np.arange(x0, x1, dtype=np.float64) - g.moon_x).astype(np.float32)[None, :]
        t = np.clip((np.sqrt(xx * xx + yy * yy) - g.moon_r - frame.moon_margin_px) / 4.0, 0.0, 1.0)
        fade = (t * t * (3 - 2 * t)).astype(np.float32)
        del xx, yy, t

    def compute(name_l):
        """One grey layer for the box, on the box plus that layer's own reach (so it is exact
        whatever the other layers need)."""
        if name_l == "adaptive":
            m = _mgn_reach(_mgn_used(p, g), p.noise_gate > 0)
        elif name_l == "achf":
            m = _achf_reach([g.sun_r * f for f in p.achf_scales], list(p.achf_weights), p.achf_part_limit,
                            0.1 * g.sun_r)
        else:
            m = 0  # natural and the NRGF family are per pixel, from whole-image statistics
        reg, origin, inner, shape = region(0 if whole else m)
        lum, rvalid = frame.lum[reg], frame.valid[reg]
        r, phi = polar(shape, g, origin)
        if name_l == "natural":
            st = _natural_cache(frame, p)
            loc = None if st.loc is None else tuple(a_[reg] for a_ in st.loc)
            y = natural_apply(lum, r, st, p.flatten, p.shoulder, p.even, g.sun_r, p.midtone, loc=loc,
                              show=frame.show[reg])
        elif name_l in ("nrgf", "fnrgf", "fnrgf_n"):
            stats = _ring_cache(frame, p)
            mean_att = _attenuation(p.mean_attenuation, p.order)
            std_att = _attenuation(p.std_attenuation, p.order)
            if name_l == "nrgf":
                y = to_linear(nrgf(lum, r, rvalid, stats), NRGF_SPREAD)
            elif name_l == "fnrgf":
                y = to_linear(fnrgf(lum, r, phi, rvalid, stats, mean_att, std_att), FNRGF_SPREAD)
            else:
                y = to_linear(fnrgf(lum, r, phi, rvalid, stats, mean_att, std_att, p.noise_m), FNRGF_N_SPREAD)
        elif name_l == "achf":
            # Raw (before Contrast), so a Contrast change needs no filtering: see below.
            y = achf(lum, rvalid, [g.sun_r * f for f in p.achf_scales], frame.nlev, g.sun_x - origin[1],
                     g.sun_y - origin[0], list(p.achf_weights), p.achf_part_limit, ref_sigma=0.1 * g.sun_r,
                     r_ref=g.moon_r, raw=True)
        else:
            scales = _mgn_scales(p, g)
            if p.noise_gate > 0:
                say(f"adaptive: noise gate {p.noise_gate:g}")
                prep = _gated_cache(frame)
                y = mgn_gated(lum, rvalid, scales, p.mgn_k, frame.nlev, frame.noise[reg], g.sun_r, p.noise_gate,
                              prep=dict(prep, bf=prep["bf"][reg]), origin=origin, weights=list(p.mgn_weights))
            else:
                y = mgn(lum, rvalid, scales, p.mgn_k, frame.nlev, origin=origin, weights=list(p.mgn_weights))
        y = np.ascontiguousarray(y[inner])
        if name_l != "achf":
            y[~(show if name_l == "natural" else valid)] = 0.0
        return y

    layers: dict[str, np.ndarray] = {}
    for name_l in LAYERS:
        if name_l not in need:
            continue
        key = layer_key(name_l, p)
        hit = store.get((name_l, inner_box)) if store is not None else None
        if hit is not None and hit[0] == key:
            say(f"layer {name_l} (unchanged)" if name_l != "achf" else "layer achf (unchanged; Contrast applied)")
            layers[name_l] = hit[1]
            continue
        say(f"layer {name_l}")
        layers[name_l] = compute(name_l)
        if store is not None:
            store[(name_l, inner_box)] = (key, layers[name_l])
    if "achf" in layers:
        layers["achf"] = achf_display(layers["achf"], valid, p.achf_contrast)
    if store is not None:  # only the latest area is kept
        for k in [k for k in store if k[1] != inner_box]:
            del store[k]
    blend_w = {k: float(v) for k, v in p.blend.items() if v and k in layers}
    if blend_w:
        layers["blend"] = soft_light_blend(layers, blend_w, valid, p.blend_modes, show=show, fade=fade,
                                           shoulder=p.shoulder)
    ckey = ("chroma", inner_box)
    hit = store.get(ckey) if store is not None else None
    if hit is not None and hit[0] == float(p.saturation):
        col = hit[1]
    else:
        reg, _, inner, _ = region(0 if whole else _blur_reach(3.0))
        col = chroma(frame.rgb[reg], frame.lum[reg], frame.show[reg], frame.noise[reg], p.saturation,
                     colour_floor=frame.colour_floor)[inner]
        if store is not None:
            store[ckey] = (float(p.saturation), col)
    moon = frame.moon
    out = {}
    for k in layers:
        if keep is not None and k not in keep:
            continue
        img = layers[k][..., None] * col
        if p.shoulder > 0:  # keep the hue where a colour would go past white (a bright prominence)
            img = img / np.maximum(img.max(axis=2, keepdims=True), 1.0)
        img = np.clip(img, 0.0, 1.0)
        if moon is not None and moon.surface is not None and k in ("natural", "blend"):
            # The seas and highlands, added (screen) where the stretched light of the Moon itself
            # leaves room for them: nothing is cut or pasted, so the limb is the stack's own.
            hc, wc = moon.surface.shape[:2]
            ys = slice(max(moon.y0, y0), min(moon.y0 + hc, y1))
            xs = slice(max(moon.x0, x0), min(moon.x0 + wc, x1))
            if ys.stop > ys.start and xs.stop > xs.start:
                sy = slice(ys.start - moon.y0, ys.stop - moon.y0)
                sx = slice(xs.start - moon.x0, xs.stop - moon.x0)
                iy = slice(ys.start - y0, ys.stop - y0)
                ix = slice(xs.start - x0, xs.stop - x0)
                add = moon.surface[sy, sx] * (moon.visibility[sy, sx, None] if moon.visibility is not None else 1.0)
                # Where the Moon's own stretched light is already bright (the limb, a prominence
                # spilling over it), the surface steps back: it could not be seen there, and adding
                # it would wash the prominence out to white.
                own = img[iy, ix].mean(axis=2)
                t = np.clip((0.7 - own) / 0.4, 0.0, 1.0)
                add = add * (t * t * (3 - 2 * t))[..., None]
                img[iy, ix] = 1.0 - (1.0 - img[iy, ix]) * (1.0 - np.clip(add, 0.0, 1.0))
        if emit is not None:
            emit(k, img.astype(np.float32, copy=False))
        else:
            out[k] = img.astype(np.float32)
        del img
    return out


def enhance(hdr: np.ndarray, weight: np.ndarray, header: dict, params: Optional[EnhanceParams] = None,
            out_dir=None, name: str = "corona", progress: Optional[Say] = None,
            fmt: str = "tif", moon: Optional[MoonInput] = None, keep: Optional[set] = None) -> tuple[EnhanceResult, dict]:
    """``hdr`` is (h, w, 3) linear. Writes one image per layer to ``out_dir`` (``fmt``: tif, png or
    jpg; see ``imageout``); without ``out_dir`` returns the layers instead as {name: (h, w, 3)
    float32 in 0..1}. The same as :func:`prepare` then :func:`render` of the whole image."""
    say = progress or (lambda _m: None)
    p = params or EnhanceParams()
    frame = prepare(hdr, weight, header, p, moon=moon, progress=say)
    notes = []
    unknown = set(p.blend) - set(LAYERS)
    if unknown:
        notes.append(f"unknown blend layer(s) ignored: {', '.join(sorted(unknown))}")
    scales = _mgn_scales(p, frame.g)
    paths: dict = {}
    emit = None
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        def emit(k, img):
            paths[k] = str(write_image(with_format(out / f"{name}_{k}.tif", fmt), img))
            say(f"wrote {paths[k]}")

    colour = render(frame, p, keep=keep, progress=say, emit=emit)
    wb, bg, nlev = frame.wb, frame.bg, frame.nlev
    del frame
    if out_dir is not None:
        info = {"params": {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(p).items()},
                "white": wb.tolist(), "background": bg.tolist(), "noise_level": nlev, "scales": scales,
                "geometry": asdict(geometry_from_header(header))}
        (out / f"{name}_enhance.json").write_text(json.dumps(info, indent=2, default=float), encoding="utf-8")
        paths["info"] = str(out / f"{name}_enhance.json")
    return EnhanceResult(paths, wb.tolist(), nlev, bg.tolist(), scales, notes), colour


from .imageout import with_format, write_image  # noqa: E402
