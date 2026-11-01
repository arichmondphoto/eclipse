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

LAYERS = ("natural", "nrgf", "fnrgf", "fnrgf_n", "adaptive")
# The natural layer is the base; the others are soft-light layers at these opacities.
DEFAULT_BLEND = {"natural": 1.0, "fnrgf_n": 0.7, "adaptive": 0.5}


@dataclass
class EnhanceParams:
    layers: tuple[str, ...] = LAYERS
    segments: int = 50  # angular segments per circle (paper: 50)
    order: int = 10  # Fourier order (paper: 10)
    mean_attenuation: Optional[Sequence[float]] = None  # A_0..A_order; default 1, 0.9, ..., 0
    std_attenuation: Optional[Sequence[float]] = None  # S_0..S_order; default as above
    noise_m: float = 0.7  # FNRGF-N: fraction of the noise removed from the local deviation, in [0, 1)
    ring_smooth: float = 1.5  # smoothing of the circle statistics along the radius, px
    flatten: float = 0.8  # natural layer: fraction of the radial fall-off (in log) removed
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
    mgn_scales: Optional[Sequence[float]] = None  # px; default from the Sun's radius
    mgn_k: float = 0.7
    mgn_skip_fine: int = 0  # leave out this many of the finest adaptive scales (they are mostly noise)
    # Adaptive layer, noise-aware (both 0 = off, as before):
    # strands: smoothing along the coronal strands before the filter, in units of 2% of the Sun's
    # radius (1 = about 8 px on a 2024 D850 frame at 400 mm); grain is smoothed out along each
    # strand while the strand stays sharp across.
    strands: float = 0.0
    # noise_gate: each scale's detail counts only as far as it stands above the noise at that scale
    # (measured from the stack); the number is how many times the noise counts as "clearly above"
    # (3 is a good start). Keeps the finest scales from turning grain into texture.
    noise_gate: float = 0.0
    saturation: float = 1.0  # colour saturation of every layer (0 = monochrome)
    white: str = "corona"  # "corona" (inner corona is white), "camera" (as-shot), or "none"
    neutral_background: bool = True  # model and remove the sky (level, colour and gradient) before filtering
    moon_margin_px: float = 1.5
    blend: dict = field(default_factory=lambda: dict(DEFAULT_BLEND))
    blend_modes: dict = field(default_factory=dict)  # layer -> "soft light" (default) or "overlay"


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


def polar(shape: tuple[int, int], g: Geometry) -> tuple[np.ndarray, np.ndarray]:
    """Distance from the Sun's centre (px) and position angle (radians, 0..2pi) of every pixel."""
    h, w = shape
    yy = np.arange(h, dtype=np.float32)[:, None] - np.float32(g.sun_y)
    xx = np.arange(w, dtype=np.float32)[None, :] - np.float32(g.sun_x)
    r = np.sqrt(xx * xx + yy * yy)
    phi = np.arctan2(yy, xx)
    phi = np.where(phi < 0, phi + np.float32(2 * math.pi), phi).astype(np.float32)
    return r, phi


def blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur; large kernels run on a reduced copy (fast on 45-megapixel frames)."""
    if sigma <= 6:
        return ndimage.gaussian_filter(img, sigma, mode="reflect")
    f = int(sigma // 3)
    h, w = img.shape
    ph, pw = (-h) % f, (-w) % f
    a = np.pad(img, ((0, ph), (0, pw)), mode="reflect")
    small = a.reshape(a.shape[0] // f, f, a.shape[1] // f, f).mean(axis=(1, 3))
    s2 = math.sqrt(max(sigma**2 - (f**2 - 1) / 12.0, 0.25)) / f
    small = ndimage.gaussian_filter(small, s2, mode="reflect")
    big = ndimage.zoom(small, f, order=1, mode="nearest", grid_mode=True)
    return big[:h, :w].astype(img.dtype, copy=False)


def _attenuation(values: Optional[Sequence[float]], order: int) -> np.ndarray:
    if values is None:
        return np.clip(1.0 - np.arange(order + 1) / max(order, 1), 0.0, 1.0)
    a = np.zeros(order + 1)
    v = np.asarray(values, float)[: order + 1]
    a[: len(v)] = v
    return a


def to_unit(y: np.ndarray, valid: np.ndarray, spread: float = 99.5) -> np.ndarray:
    """Map a zero-centred filter output to 0..1 with a soft arctan curve."""
    s = float(np.percentile(np.abs(y[valid]), spread)) if valid.any() else 1.0
    s = s or 1.0
    return (0.5 + np.arctan(1.6 * y / s) / math.pi).astype(np.float32)


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


def natural(x: np.ndarray, r: np.ndarray, phi: np.ndarray, valid: np.ndarray, noise_level: float,
            flatten: float, stats_valid: Optional[np.ndarray] = None, show: Optional[np.ndarray] = None,
            shoulder: float = 0.0, even: float = 0.0, sun_r: float = 0.0, midtone: float = 0.0) -> np.ndarray:
    """Log stretch with part of the radial fall-off removed.

    ``x`` has the sky removed. The flattening stops where the corona sinks
    into the noise, so the sky beyond stays dark instead of being lifted.
    """
    floor = max(noise_level, 1e-9)
    lg = np.log(np.maximum(x, 0) + floor).astype(np.float32)
    # Circle averages from completed circles (see ring_stats), so no arcs at the frame edges.
    lo = float(lg[valid].min()) if valid.any() else 0.0
    sv = valid if stats_valid is None else stats_valid
    prof = ring_stats(lg - lo, r, phi, sv, 36, 0, 2.0).ring_mean + lo
    prof = np.maximum(prof, math.log(4 * floor))  # no lifting of what is mostly noise
    ri = np.clip(r - 0.5, 0, len(prof) - 1.001)
    i0 = ri.astype(np.int32)
    t = ri - i0
    base = (1 - t) * prof[i0] + t * prof[i0 + 1]
    out = lg - np.float32(flatten) * base.astype(np.float32)
    # Black point: the sky (pixels at the noise level), so it comes out dark (taken before any
    # evening out, which must not move the sky).
    sky = valid & (x < floor)
    black = float(np.median(out[sky])) if sky.sum() > 1000 else float(np.percentile(out[valid], 0.5))
    if even > 0 and sun_r > 0:
        # Streamers against coronal holes: the large-scale differences round each circle.
        m = sv.astype(np.float32)
        sig = max(4.0, 0.25 * sun_r)
        loc = blur(np.where(sv, out, 0.0).astype(np.float32), sig) / np.maximum(blur(m, sig), 1e-6)
        lo2 = float(loc[sv].min()) if sv.any() else 0.0
        rp = ring_stats(loc - lo2, r, phi, sv, 36, 0, 2.0).ring_mean + lo2
        ring = ((1 - t) * rp[i0] + t * rp[i0 + 1]).astype(np.float32)
        # Only where the corona stands well above the noise (the sky stays where it is), and
        # fading in from 1.15 to 1.4 solar radii: right by the limb the corona's contrast is so
        # steep that taking out its local mean pushed the dark gaps between bright loops to black.
        above = np.clip((lg - math.log(4 * floor)) / 2.0, 0.0, 1.0)
        rin = np.clip((r / sun_r - 1.15) / 0.25, 0.0, 1.0)
        out = out - np.float32(even) * (above * rin * (loc - ring)).astype(np.float32)
    white = float(np.percentile(out[valid], 99.9))
    out = np.clip(roll_off((out - black) / ((white - black) or 1.0), shoulder), 0.0, 1.0).astype(np.float32)
    if midtone > 0 and sun_r > 0:
        out = set_midtone(out, valid & (r > 1.05 * sun_r) & (r < 2.5 * sun_r), midtone)
    out[~(valid if show is None else show)] = 0.0
    return out


def set_midtone(out: np.ndarray, sel: np.ndarray, midtone: float) -> np.ndarray:
    """A midtones transfer curve (as in PixInsight's) that puts the median of ``out`` over ``sel``
    at ``midtone``; black and white stay, and unlike a gamma it has a finite slope at black, so the
    dark sky is lifted only a little."""
    if not sel.any():
        return out
    a = float(np.median(out[sel]))
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


def soft_light_blend(layers: dict, weights: dict, valid: np.ndarray, modes: Optional[dict] = None,
                     show: Optional[np.ndarray] = None, fade: Optional[np.ndarray] = None,
                     shoulder: float = 0.0, midtone: float = 0.0, tone_sel: Optional[np.ndarray] = None) -> np.ndarray:
    """The natural layer as the base, each filter layer applied at its weight (like layer opacity
    in Photoshop) in its mode, soft light unless ``modes`` says overlay. The filter layers are
    centred on mid-grey, which both modes treat as 'no change', so they add structure without
    lifting the dark sky. Layers are applied in the order natural, NRGF, FNRGF, FNRGF-N, adaptive."""
    modes = modes or {}
    base = layers["natural"].copy() if "natural" in weights else np.full_like(next(iter(layers.values())), 0.5)
    for k in LAYERS:
        if k == "natural" or k not in weights:
            continue
        w = min(max(weights[k], 0.0), 1.0)
        fn = BLEND_MODES.get(str(modes.get(k, "soft light")).lower().replace("_", " ").replace("softlight", "soft light"))
        if fn is None:
            raise ValueError(f"unknown blend mode {modes[k]!r} for {k}; use soft light or overlay")
        if fade is not None:  # filter layers fade out towards the Moon, where they have no data
            wf = w * fade
            base = (1 - wf) * base + wf * fn(base, layers[k])
        else:
            base = (1 - w) * base + w * fn(base, layers[k])
    out = stretch(base, valid, 0.05, 99.95, shoulder)
    if midtone > 0 and tone_sel is not None:
        out = set_midtone(out, tone_sel, midtone)
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


def strand_orientation(b: np.ndarray, r: float) -> tuple[np.ndarray, np.ndarray]:
    """Direction along the coronal strands (radians) and how clear it is (coherence 0..1).

    From the structure tensor of the mid-scale detail (``b`` minus a blur of 10% of the Sun's
    radius): taken from the image itself, the corona's overall fall-off would set the direction
    (across the strands), and the finest scales are mostly noise.
    """
    d = b - blur(b, max(4.0, 0.1 * r))
    sg, st = max(1.0, 0.0073 * r), max(2.0, 0.024 * r)
    gx = ndimage.gaussian_filter(d, sg, order=(0, 1))
    gy = ndimage.gaussian_filter(d, sg, order=(1, 0))
    jxx, jyy, jxy = blur(gx * gx, st), blur(gy * gy, st), blur(gx * gy, st)
    theta = 0.5 * np.arctan2(2 * jxy, jxx - jyy) + math.pi / 2
    coh = np.sqrt((jxx - jyy) ** 2 + 4 * jxy**2) / np.maximum(jxx + jyy, 1e-30)
    return theta.astype(np.float32), coh.astype(np.float32)


def smooth_along(b: np.ndarray, theta: np.ndarray, coh: np.ndarray, length: float,
                 iso: float = 2.0, chunk: int = 512) -> np.ndarray:
    """Gaussian smoothing along the strands (sigma ``length`` px) where the direction is clear,
    a light isotropic one (``iso``) where it is not. Done in row blocks to bound memory."""
    h, w = b.shape
    n = int(math.ceil(2.5 * length))
    ts = np.arange(-n, n + 1, dtype=np.float32)
    wts = np.exp(-0.5 * (ts / length) ** 2)
    wts /= wts.sum()
    base = ndimage.gaussian_filter(b, 0.6)
    out = np.empty_like(b, dtype=np.float32)
    iso_img = ndimage.gaussian_filter(b, iso)
    # Where two families of strands meet the direction flips over a few pixels; a mix that switched
    # as fast between along and round made speckles there, so the mix changes over a strand length.
    kmap = ndimage.gaussian_filter(np.clip((coh - 0.2) / 0.3, 0.0, 1.0).astype(np.float32), max(1.0, length))
    xx = np.arange(w, dtype=np.float32)[None, :]
    for r0 in range(0, h, chunk):
        rows = slice(r0, min(h, r0 + chunk))
        yy = np.arange(rows.start, rows.stop, dtype=np.float32)[:, None]
        c, s_ = np.cos(theta[rows]), np.sin(theta[rows])
        acc = np.zeros((rows.stop - rows.start, w), np.float32)
        for t, wt in zip(ts, wts):
            acc += wt * ndimage.map_coordinates(base, [yy + t * s_, xx + t * c], order=1, mode="nearest")
        k = kmap[rows]
        out[rows] = k * acc + (1 - k) * iso_img[rows]
    return out


def correlated_noise(shape, valid: np.ndarray, sample: np.ndarray, seed: int = 0, patch: int = 64) -> np.ndarray:
    """Unit noise with the spatial spectrum measured in ``sample`` over ``valid`` (demosaicing and
    the alignment's resampling correlate neighbouring pixels, so white noise understates the noise
    left after any smoothing), scaled so that the 1-px scatter statistic used by
    :func:`noise_map` reads 1. The spectrum is the average over patches of the residual after a
    6 px blur, so smooth structure in the sample does not count as noise."""
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
    n = np.random.default_rng(seed).normal(0.0, 1.0, shape).astype(np.float32)
    if count >= 4:
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
        from scipy.signal import fftconvolve
        n = fftconvolve(n, kern.astype(np.float32), mode="same").astype(np.float32)
    hp = n - ndimage.gaussian_filter(n, 1.0)
    return (n / (1.4826 * float(np.median(np.abs(hp))) / 0.96)).astype(np.float32)


def mgn_gated(x: np.ndarray, valid: np.ndarray, scales: Sequence[float], k: float, floor: float,
              noise: np.ndarray, sun_r: float, strands: float = 0.0, gate: float = 3.0,
              seed: int = 0) -> np.ndarray:
    """:func:`mgn` with the noise kept down: optional smoothing along the strands first, and each
    scale's detail weighted by how far it stands above the noise at that scale (a Wiener gain;
    the noise at each scale comes from a noise image with the stack's measured noise and
    correlation, put through the same smoothing). Pure noise gives almost no contrast."""
    b = np.log(np.maximum(x, 0) + max(floor, 1e-9)).astype(np.float32)
    bf = _masked_fill(b, valid, sun_r)
    sig = np.where(valid, noise / np.maximum(np.maximum(x, 0) + max(floor, 1e-9), 1e-12), 0.0)
    vm = valid.astype(np.float32)
    sig = ndimage.gaussian_filter(sig.astype(np.float32), 4.0) / np.maximum(ndimage.gaussian_filter(vm, 4.0), 1e-3)
    smooth_b = blur(bf, max(4.0, 0.05 * sun_r))  # the faint part, as a region (not single pixels)
    faint = valid & (smooth_b < np.percentile(smooth_b[valid], 30))
    nz = correlated_noise(b.shape, faint, bf, seed) * sig
    del b, sig, vm, smooth_b, faint  # full frames are 45 megapixels: free what is done with
    if strands > 0:
        theta, coh = strand_orientation(bf, sun_r)
        # How clear a direction noise alone would seem: only clearer directions than that are
        # followed, or the smoothing would draw strands out of the noise itself.
        theta_n, coh_n = strand_orientation(nz, sun_r)
        base_n = blur(coh_n, max(8.0, 0.1 * sun_r))

        def clear(c):
            return np.clip((c - base_n) / np.maximum(1.0 - base_n, 1e-3), 0.0, 1.0).astype(np.float32)

        length = max(1.0, strands * 0.02 * sun_r)
        k_data = clear(coh)
        bf = smooth_along(bf, theta, k_data, length)
        del theta, coh
        # The noise image is smoothed along its *own* ridges, as the data's noise is wherever the
        # direction comes from the noise: noise smoothed along itself keeps more of its strength,
        # and judging against that keeps strands from being drawn out of noise. (Where the
        # direction comes from real strands this overstates the noise a little: safe side.)
        # Mixed between along and round in the same places as the data (k_data): where the data
        # is only smoothed round, its noise stays grainier, and the noise image must too.
        nz = smooth_along(nz, theta_n, k_data, length)
        del theta_n, coh_n, base_n, k_data
    nz = np.where(valid, nz, 0.0).astype(np.float32)
    acc = np.zeros_like(bf)
    g2 = gate * gate
    for w in scales:
        d = bf - blur(bf, w)
        sd2 = blur(d * d, w)
        dn = nz - blur(nz, w)
        # Both judged over a wider area than the scale itself: over a few pixels the estimates
        # scatter so much that noise passes the gate by chance.
        wide = max(3.0 * w, 8.0)
        n2 = blur(dn * dn, wide)
        s2 = blur(d * d, wide)
        gain = np.maximum(s2 - n2, 0.0) / np.maximum(s2 + (g2 - 1.0) * n2, 1e-12) if gate > 0 else 1.0
        acc += np.arctan(k * gain * d / np.sqrt(np.maximum(sd2, 1e-12)))
        del d, sd2, dn, n2, s2, gain
    acc /= len(scales)
    # A fixed scale, not the image's own spread: stretching to the spread would bring noise back
    # up to full contrast wherever there is little real structure.
    out = (0.5 + np.arctan(1.6 * acc / 0.6) / math.pi).astype(np.float32)
    out[~valid] = 0.0
    return out



def mgn(x: np.ndarray, valid: np.ndarray, scales: Sequence[float], k: float, floor: float) -> np.ndarray:
    """Multi-scale Gaussian normalization (Morgan & Druckmüller 2014) of log(X); local part only."""
    b = np.log(np.maximum(x, 0) + max(floor, 1e-9)).astype(np.float32)
    # Fill invalid pixels with their surroundings so the Moon and frame edges do not ring.
    m = valid.astype(np.float32)
    fill = blur(b * m, 8.0) / np.maximum(blur(m, 8.0), 1e-6)
    b = np.where(valid, b, fill).astype(np.float32)
    acc = np.zeros_like(b)
    for w in scales:
        d = b - blur(b, w)
        sd = np.sqrt(np.maximum(blur(d * d, w), 1e-12))
        acc += np.arctan(k * d / sd)
    acc /= len(scales)
    out = to_unit(acc, valid, 99.9)
    out[~valid] = 0.0
    return out


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
    scales: list
    notes: list = field(default_factory=list)

    def describe(self) -> str:
        lines = [f"White balance multipliers (R, G, B): {', '.join(f'{v:.3f}' for v in self.white)}",
                 f"Sky removed, typical level after white balance (R, G, B): "
                 f"{', '.join(f'{v:.3g}' for v in self.background)}",
                 f"Adaptive filter scales (px): {', '.join(f'{s:.1f}' for s in self.scales)}"]
        lines += [f"{k}: {v}" for k, v in self.layers.items()]
        lines += [f"Note: {n}" for n in self.notes]
        return "\n".join(lines)


def enhance(hdr: np.ndarray, weight: np.ndarray, header: dict, params: Optional[EnhanceParams] = None,
            out_dir=None, name: str = "corona", progress: Optional[Say] = None,
            fmt: str = "tif", moon: Optional[MoonInput] = None, keep: Optional[set] = None) -> tuple[EnhanceResult, dict]:
    """``hdr`` is (h, w, 3) linear. Writes one image per layer to ``out_dir`` (``fmt``: tif, png or
    jpg; see ``imageout``); without
    ``out_dir`` returns the layers instead as {name: (h, w, 3) float32 in 0..1}."""
    say = progress or (lambda _m: None)
    p = params or EnhanceParams()
    g = geometry_from_header(header)
    h, w = hdr.shape[:2]
    r, phi = polar((h, w), g)
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
    fade = None
    if moon is not None:
        dmoon = np.sqrt((np.arange(h, dtype=np.float32)[:, None] - np.float32(g.moon_y)) ** 2
                        + (np.arange(w, dtype=np.float32)[None, :] - np.float32(g.moon_x)) ** 2) - g.moon_r
        t = np.clip((dmoon - p.moon_margin_px) / 4.0, 0.0, 1.0)
        fade = (t * t * (3 - 2 * t)).astype(np.float32)
        del dmoon, t

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
    rgb = hdr
    lum = rgb.mean(axis=2)
    say(f"white balance {wb.round(3).tolist()}; sky removed ({sky_kind}) {bg.tolist()}")

    stats_valid = valid & ~spot_mask(lum, valid, r, 2.0 * g.sun_r)
    say(f"{int((valid & ~stats_valid).sum())} pixels in dust spots or stars left out of the circle statistics")
    noise = noise_map(lum, weight, valid)
    nlev = float(np.median(noise[valid & (noise > 0)])) if (valid & (noise > 0)).any() else 0.0
    # Per-channel sky offsets are only known to about their own size; where the corona is no
    # brighter than that, its colour cannot be trusted and is shown grey.
    col = chroma(rgb, lum, show, noise, p.saturation, colour_floor=2.0 * float(np.abs(bg).max()))
    del rgb

    order = p.order
    mean_att = _attenuation(p.mean_attenuation, order)
    std_att = _attenuation(p.std_attenuation, order)
    layers: dict[str, np.ndarray] = {}
    stats = None
    if {"fnrgf", "fnrgf_n", "nrgf"} & set(p.layers) or "fnrgf_n" in p.blend or "fnrgf" in p.blend:
        say("circle statistics")
        stats = ring_stats(lum, r, phi, stats_valid, p.segments, order, p.ring_smooth, noise)

    need = set(p.layers) | {k for k, v in p.blend.items() if v}
    scales = list(p.mgn_scales) if p.mgn_scales else [g.sun_r * f for f in (0.006, 0.012, 0.025, 0.05, 0.1, 0.2)]
    scales = sorted(scales)[min(max(int(p.mgn_skip_fine), 0), len(scales) - 1):]
    for name_l in LAYERS:
        if name_l not in need:
            continue
        say(f"layer {name_l}")
        if name_l == "natural":
            y = natural(lum, r, phi, valid, nlev, p.flatten, stats_valid, show=show, shoulder=p.shoulder,
                        even=p.even, sun_r=g.sun_r, midtone=p.midtone)
        elif name_l == "nrgf":
            y = to_unit(nrgf(lum, r, valid, stats), valid)
        elif name_l == "fnrgf":
            y = to_unit(fnrgf(lum, r, phi, valid, stats, mean_att, std_att), valid)
        elif name_l == "fnrgf_n":
            y = to_unit(fnrgf(lum, r, phi, valid, stats, mean_att, std_att, p.noise_m), valid)
        else:
            if p.strands > 0 or p.noise_gate > 0:
                say(f"adaptive: noise gate {p.noise_gate:g}, smoothing along the strands {p.strands:g}")
                y = mgn_gated(lum, valid, scales, p.mgn_k, nlev, noise, g.sun_r, p.strands, p.noise_gate)
            else:
                y = mgn(lum, valid, scales, p.mgn_k, nlev)
        y[~(show if name_l == "natural" else valid)] = 0.0
        layers[name_l] = y

    notes = []
    blend_w = {k: float(v) for k, v in p.blend.items() if v and k in layers}
    if blend_w:
        layers["blend"] = soft_light_blend(layers, blend_w, valid, p.blend_modes, show=show, fade=fade,
                                             shoulder=p.shoulder, midtone=p.midtone,
                                             tone_sel=valid & (r > 1.05 * g.sun_r) & (r < 2.5 * g.sun_r))
    unknown = set(p.blend) - set(LAYERS)
    if unknown:
        notes.append(f"unknown blend layer(s) ignored: {', '.join(sorted(unknown))}")

    def coloured(k: str) -> np.ndarray:
        img = layers[k][..., None] * col
        if p.shoulder > 0:  # keep the hue where a colour would go past white (a bright prominence)
            img = img / np.maximum(img.max(axis=2, keepdims=True), 1.0)
        img = np.clip(img, 0.0, 1.0)
        if moon is not None and moon.surface is not None and k in ("natural", "blend"):
            # The seas and highlands, added (screen) where the stretched light of the Moon itself
            # leaves room for them: nothing is cut or pasted, so the limb is the stack's own.
            hc, wc = moon.surface.shape[:2]
            ys = slice(max(moon.y0, 0), min(moon.y0 + hc, h))
            xs = slice(max(moon.x0, 0), min(moon.x0 + wc, w))
            sy = slice(ys.start - moon.y0, ys.stop - moon.y0)
            sx = slice(xs.start - moon.x0, xs.stop - moon.x0)
            add = moon.surface[sy, sx] * (moon.visibility[sy, sx, None] if moon.visibility is not None else 1.0)
            # Where the Moon's own stretched light is already bright (the limb, a prominence spilling
            # over it), the surface steps back: it could not be seen there, and adding it would wash
            # the prominence out to white.
            own = img[ys, xs].mean(axis=2)
            t = np.clip((0.7 - own) / 0.4, 0.0, 1.0)
            add = add * (t * t * (3 - 2 * t))[..., None]
            img[ys, xs] = 1.0 - (1.0 - img[ys, xs]) * (1.0 - np.clip(add, 0.0, 1.0))
        return img.astype(np.float32)

    paths: dict = {}
    colour: dict = {}
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        for k in list(layers):
            if keep is not None and k not in keep:
                continue
            # One colour image at a time: a full 45 MP RGB layer is half a gigabyte.
            img = coloured(k)
            paths[k] = str(write_image(with_format(out / f"{name}_{k}.tif", fmt), img))
            del img
            say(f"wrote {paths[k]}")
        info = {"params": {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(p).items()},
                "white": wb.tolist(), "background": bg.tolist(), "noise_level": nlev, "scales": scales, "geometry": asdict(g)}
        (out / f"{name}_enhance.json").write_text(json.dumps(info, indent=2, default=float), encoding="utf-8")
        paths["info"] = str(out / f"{name}_enhance.json")
    else:
        colour = {k: coloured(k) for k in layers if keep is None or k in keep}
    return EnhanceResult(paths, wb.tolist(), nlev, bg.tolist(), scales, notes), colour


from .imageout import with_format, write_image  # noqa: E402
