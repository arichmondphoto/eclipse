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
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
from scipy import ndimage

Say = Callable[[str], None]

LAYERS = ("natural", "nrgf", "fnrgf", "fnrgf_n", "adaptive")
DEFAULT_BLEND = {"natural": 0.4, "fnrgf_n": 0.3, "adaptive": 0.3}


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
    mgn_scales: Optional[Sequence[float]] = None  # px; default from the Sun's radius
    mgn_k: float = 0.7
    saturation: float = 1.0  # colour saturation of every layer (0 = monochrome)
    white: str = "corona"  # "corona" (inner corona is white), "camera" (as-shot), or "none"
    moon_margin_px: float = 1.5
    blend: dict = field(default_factory=lambda: dict(DEFAULT_BLEND))


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


def stretch(x: np.ndarray, valid: np.ndarray, lo: float = 0.5, hi: float = 99.8) -> np.ndarray:
    a, b = np.percentile(x[valid], [lo, hi]) if valid.any() else (0.0, 1.0)
    return np.clip((x - a) / ((b - a) or 1.0), 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Noise
# ---------------------------------------------------------------------------


def noise_map(lum: np.ndarray, weight: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Per-pixel noise of the stacked luminance.

    The stack weights are inverse variances up to one common factor, so the
    noise is ``c / sqrt(weight)``; ``c`` is measured from the pixel-to-pixel
    scatter in the faint outer corona, where structure is smooth.
    """
    wpos = np.where(valid & (weight > 0), weight, np.nan)
    resid = lum - ndimage.gaussian_filter(lum, 1.0)
    scaled = np.abs(resid) * np.sqrt(wpos)
    ok = np.isfinite(scaled)
    if ok.sum() < 100:
        return np.zeros_like(lum)
    # Use the faintest half, where coronal structure contributes least to the scatter.
    faint = ok & (lum < np.nanpercentile(np.where(ok, lum, np.nan), 50))
    c = 1.4826 * float(np.median(scaled[faint if faint.sum() > 100 else ok])) / 0.96
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
    have = cnt > 1e-3 * np.maximum(cnt.max(axis=1, keepdims=True), 1e-12)
    trend_m = _trend(ring_mean, rc > 0)
    trend_s = _trend(ring_std, rc > 0)
    mean = _fill_segments(mean, have, trend_m)
    std = _fill_segments(std, have, trend_s)
    nstd = _fill_segments(nstd, have, trend_s)
    # Fourier coefficients of the (now complete) circles: the paper's eqs. 2-4.
    centres = 2 * math.pi * (np.arange(segments) + 0.5) / segments
    cols = [np.full(segments, 1.0 / segments)]
    for k in range(1, order + 1):
        cols += [2.0 / segments * np.cos(k * centres), 2.0 / segments * np.sin(k * centres)]
    P = np.stack(cols, axis=1)  # (segments, 2*order+1)
    out = [y @ P for y in (mean, std, nstd)]
    return RingStats(out[0], out[1], out[2], order, ring_mean, ring_std, seg_std)


def _trend(prof: np.ndarray, have: np.ndarray, sigma: float = 15.0) -> np.ndarray:
    """Smooth, positive radial trend of a ring profile (for scaling carried-over values)."""
    v = np.log(np.maximum(_carry(prof, have), 1e-30))
    return np.exp(ndimage.gaussian_filter1d(v, sigma, mode="nearest"))


def _fill_segments(y: np.ndarray, have: np.ndarray, trend: np.ndarray) -> np.ndarray:
    """Fill (ring, segment) cells without data from the nearest ring with data, scaled by ``trend``."""
    nr = y.shape[0]
    idx = np.arange(nr)[:, None]
    last = np.maximum.accumulate(np.where(have, idx, -1), axis=0)  # nearest inner ring with data
    nxt = np.minimum.accumulate(np.where(have, idx, nr)[::-1], axis=0)[::-1]  # nearest outer ring
    src = np.where(last >= 0, last, np.where(nxt < nr, nxt, idx))
    seg = np.arange(y.shape[1])[None, :]
    filled = y[src, seg] * (trend[:, None] / np.maximum(trend[src], 1e-30))
    # A circle with no data at all keeps its ring average.
    return np.where(have, y, filled)


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


def natural(x: np.ndarray, r: np.ndarray, valid: np.ndarray, noise_level: float, flatten: float) -> np.ndarray:
    """Log stretch with part of the radial fall-off removed."""
    floor = max(noise_level, 1e-9)
    lg = np.log(np.maximum(x, 0) + floor).astype(np.float32)
    prof = radial_profile(lg, r, valid)
    ri = np.clip(r - 0.5, 0, len(prof) - 1.001)
    i0 = ri.astype(np.int32)
    t = ri - i0
    base = (1 - t) * prof[i0] + t * prof[i0 + 1]
    out = lg - np.float32(flatten) * base.astype(np.float32)
    out = stretch(out, valid, 0.5, 99.9)
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


def white_balance(rgb: np.ndarray, r: np.ndarray, valid: np.ndarray, g: Geometry, mode: str,
                  camera_wb: Optional[Sequence[float]]) -> np.ndarray:
    """Channel multipliers. 'corona': the inner corona (1.2-2.5 R) is white, as K-corona light is
    scattered sunlight; 'camera': the as-shot multipliers stored with the stack."""
    if mode == "camera" and camera_wb:
        m = np.asarray(camera_wb, float)
    elif mode == "corona":
        ring = valid & (r > 1.2 * g.sun_r) & (r < 2.5 * g.sun_r)
        if ring.sum() < 100:
            ring = valid
        med = np.array([np.median(rgb[..., c][ring]) for c in range(3)])
        m = med[1] / np.maximum(med, 1e-12)
    else:
        m = np.ones(3)
    return (m / m[1]).astype(np.float32)


def chroma(rgb: np.ndarray, lum: np.ndarray, valid: np.ndarray, noise: np.ndarray, saturation: float) -> np.ndarray:
    """Smoothed colour ratios rgb/lum, faded to grey where the signal is buried in noise."""
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
    scales: list
    notes: list = field(default_factory=list)

    def describe(self) -> str:
        lines = [f"White balance multipliers (R, G, B): {', '.join(f'{v:.3f}' for v in self.white)}",
                 f"Adaptive filter scales (px): {', '.join(f'{s:.1f}' for s in self.scales)}"]
        lines += [f"{k}: {v}" for k, v in self.layers.items()]
        lines += [f"Note: {n}" for n in self.notes]
        return "\n".join(lines)


def enhance(hdr: np.ndarray, weight: np.ndarray, header: dict, params: Optional[EnhanceParams] = None,
            out_dir=None, name: str = "corona", progress: Optional[Say] = None) -> tuple[EnhanceResult, dict]:
    """``hdr`` is (h, w, 3) linear. Writes one 16-bit TIFF per layer to ``out_dir``; without
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

    camera_wb = None
    if all(k in header for k in ("WBR", "WBG", "WBB")):
        camera_wb = [header["WBR"], header["WBG"], header["WBB"]]
    wb = white_balance(hdr, r, valid, g, p.white, camera_wb)
    hdr *= wb[None, None, :]
    rgb = hdr
    lum = rgb.mean(axis=2)
    say(f"white balance {wb.round(3).tolist()}")

    noise = noise_map(lum, weight, valid)
    nlev = float(np.median(noise[valid & (noise > 0)])) if (valid & (noise > 0)).any() else 0.0
    col = chroma(rgb, lum, valid, noise, p.saturation)
    del rgb

    order = p.order
    mean_att = _attenuation(p.mean_attenuation, order)
    std_att = _attenuation(p.std_attenuation, order)
    layers: dict[str, np.ndarray] = {}
    stats = None
    if {"fnrgf", "fnrgf_n", "nrgf"} & set(p.layers) or "fnrgf_n" in p.blend or "fnrgf" in p.blend:
        say("circle statistics")
        stats = ring_stats(lum, r, phi, valid, p.segments, order, p.ring_smooth, noise)

    need = set(p.layers) | {k for k, v in p.blend.items() if v}
    scales = list(p.mgn_scales) if p.mgn_scales else [g.sun_r * f for f in (0.006, 0.012, 0.025, 0.05, 0.1, 0.2)]
    for name_l in LAYERS:
        if name_l not in need:
            continue
        say(f"layer {name_l}")
        if name_l == "natural":
            y = natural(lum, r, valid, nlev, p.flatten)
        elif name_l == "nrgf":
            y = to_unit(nrgf(lum, r, valid, stats), valid)
        elif name_l == "fnrgf":
            y = to_unit(fnrgf(lum, r, phi, valid, stats, mean_att, std_att), valid)
        elif name_l == "fnrgf_n":
            y = to_unit(fnrgf(lum, r, phi, valid, stats, mean_att, std_att, p.noise_m), valid)
        else:
            y = mgn(lum, valid, scales, p.mgn_k, nlev)
        y[~valid] = 0.0
        layers[name_l] = y

    notes = []
    blend_w = {k: float(v) for k, v in p.blend.items() if v and k in layers}
    if blend_w:
        tot = sum(blend_w.values())
        b = sum(layers[k] * (v / tot) for k, v in blend_w.items())
        b = stretch(b, valid, 0.2, 99.9)
        b[~valid] = 0.0
        layers["blend"] = b
    unknown = set(p.blend) - set(LAYERS)
    if unknown:
        notes.append(f"unknown blend layer(s) ignored: {', '.join(sorted(unknown))}")

    paths: dict = {}
    colour: dict = {}
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        for k in list(layers):
            # One colour image at a time: a full 45 MP RGB layer is half a gigabyte.
            img = np.clip(layers[k][..., None] * col, 0.0, 1.0)
            paths[k] = str(_write_tiff(out / f"{name}_{k}.tif", img))
            del img
            say(f"wrote {paths[k]}")
        info = {"params": {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(p).items()},
                "white": wb.tolist(), "noise_level": nlev, "scales": scales, "geometry": asdict(g)}
        (out / f"{name}_enhance.json").write_text(json.dumps(info, indent=2, default=float), encoding="utf-8")
        paths["info"] = str(out / f"{name}_enhance.json")
    else:
        colour = {k: np.clip(v[..., None] * col, 0.0, 1.0).astype(np.float32) for k, v in layers.items()}
    return EnhanceResult(paths, wb.tolist(), nlev, scales, notes), colour


def _write_tiff(path: Path, img: np.ndarray) -> Path:
    data = (np.clip(img, 0, 1) * 65535 + 0.5).astype(np.uint16)
    try:
        import tifffile

        tifffile.imwrite(path, data, photometric="rgb")
        return path
    except ImportError:
        from PIL import Image

        path = path.with_suffix(".png")
        Image.fromarray((data >> 8).astype(np.uint8)).save(path)
        return path
