"""Prominences: the red and pink hydrogen glow at the solar limb, as a layer of its own.

Prominences shine in emission lines, mostly hydrogen-alpha at 656 nm (deep
red) with weaker blue hydrogen lines, which together look pink or magenta.
The corona is scattered sunlight, which is white once the corona sets the
white balance. So prominences are found by colour: pixels much redder than
the corona at the same height, by more than the noise.

The HDR stack already holds them unclipped: at the limb it is built from the
short exposures, since the long ones are saturated there. Because the stack
merges every frame, prominences that the moving Moon uncovered at any time
during totality are all there. The layer is cut off at the limb of the Moon
in the HDR stack's Moon frame, so nothing lands on the lunar disc (or on the
earthshine).

The chromosphere, the thin pink rim along the solar limb seen near second
and third contact, is the same emission. It can be left out: thin red
regions that never rise more than ``chromosphere_arcsec`` above the solar
limb are removed, while taller prominences keep their bases.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import ndimage

from .enhance import Geometry, geometry_from_header, white_balance

SUN_RADIUS_ARCSEC = 959.6  # mean; only used when the stack has no pixel scale


@dataclass
class ProminenceParams:
    height: float = 0.3  # how far above the solar limb to look, in solar radii
    sensitivity: float = 4.0  # red excess needed, in noise standard deviations
    min_redness: float = 1.1  # how much redder than the corona (2024: 99 % of the corona within 1.08)
    chromosphere: bool = True  # keep the thin rim along the solar limb
    chromosphere_arcsec: float = 12.0  # the rim's height above the solar limb
    brightness: float = 1.0
    saturation: float = 1.2
    feather_px: float = 1.5  # (not used: the layer has no outline)
    white: str = "corona"
    sharpen: float = 0.5  # deconvolution with the blur measured on the Moon's edge: 0 off, 1 strongest
    structure: float = 0.0  # fine structure brought out with MGN: 0 off, 1 strong


@dataclass
class ProminenceResult:
    count: int  # separate prominences (and chromosphere pieces) kept
    removed_chromosphere: int  # chromosphere pixels left out
    area_px: int
    white: list
    notes: list = field(default_factory=list)
    outputs: dict = field(default_factory=dict)

    def describe(self) -> str:
        lines = [f"Prominences found: {self.count} ({self.area_px} px)",
                 f"White balance (R, G, B): {', '.join(f'{v:.3f}' for v in self.white)}"]
        if self.removed_chromosphere:
            lines.append(f"Chromosphere left out: {self.removed_chromosphere} px")
        lines += [f"{k}: {v}" for k, v in self.outputs.items()]
        lines += [f"Note: {n}" for n in self.notes]
        return "\n".join(lines)


@dataclass
class SunRegion:
    """The part of the HDR stack around the Sun (prominences live within a few tenths of a radius)."""

    rgb: np.ndarray  # (h, w, 3) linear
    weight: np.ndarray
    geometry: Geometry  # in region coordinates
    origin: tuple  # (x0, y0) in the HDR stack
    canvas: tuple  # (height, width) of the HDR stack
    pixscale: float  # arcsec per pixel
    blur_px: Optional[float] = None  # blur measured on the Moon's edge (the prominence stack), px
    source: str = "HDR stack"


def sun_region(hdr: np.ndarray, weight: np.ndarray, header: dict, radii: float = 2.6) -> SunRegion:
    g = geometry_from_header(header)
    h, w = hdr.shape[:2]
    half = int(math.ceil(radii * g.sun_r))
    x0, y0 = max(0, int(g.sun_x) - half), max(0, int(g.sun_y) - half)
    x1, y1 = min(w, int(g.sun_x) + half + 1), min(h, int(g.sun_y) + half + 1)
    sub = np.ascontiguousarray(hdr[y0:y1, x0:x1], dtype=np.float32)
    wt = np.ascontiguousarray(weight[y0:y1, x0:x1], dtype=np.float32)
    gg = Geometry(g.sun_x - x0, g.sun_y - y0, g.sun_r, g.moon_x - x0, g.moon_y - y0, g.moon_r)
    scale = float(header.get("PIXSCALE") or SUN_RADIUS_ARCSEC / g.sun_r)
    return SunRegion(sub, wt, gg, (x0, y0), (h, w), scale)


def _ring_profile(values: np.ndarray, ring: np.ndarray, sel: np.ndarray, stat, n: int, smooth: float = 2.0):
    out = np.full(n, np.nan)
    order = np.argsort(ring[sel], kind="stable")
    rs, vs = ring[sel][order], values[sel][order]
    bounds = np.searchsorted(rs, np.arange(n + 1))
    for i in range(n):
        a, b = bounds[i], bounds[i + 1]
        if b - a >= 8:
            out[i] = stat(vs[a:b])
    good = np.isfinite(out)
    if not good.any():
        return np.zeros(n)
    idx = np.arange(n)
    out = np.interp(idx, idx[good], out[good])
    return ndimage.gaussian_filter1d(out, smooth, mode="nearest")


def richardson_lucy(img: np.ndarray, sigma: float, iterations: int) -> np.ndarray:
    """Richardson-Lucy deconvolution of a non-negative image with a Gaussian blur ``sigma`` (px)."""
    x = np.maximum(img, 1e-12).astype(np.float64)
    est = x.copy()
    for _ in range(max(int(iterations), 0)):
        conv = ndimage.gaussian_filter(est, sigma)
        est *= ndimage.gaussian_filter(x / np.maximum(conv, 1e-12), sigma)
    return est.astype(np.float32)


def find_prominences(region: SunRegion, params: Optional[ProminenceParams] = None):
    """The prominences' own light, sharpened and stretched as a layer.

    Returns (result, layer (h, w, 3) display image in 0..1 over the region, opacity (h, w)).

    The corona is white light; prominences glow in hydrogen's red (H-alpha) and blue lines. The
    corona's colour is measured away from them (its red and blue against its green), and the
    corona behind them, which shows in the green channel alone, is taken off in every channel:
    what is left is the prominences' own light, with no outline drawn round it. With a blur
    measured on the Moon's edge (``region.blur_px``), it is deconvolved (``params.sharpen``),
    then stretched on its own (they are far brighter than the corona near them), with fine
    structure brought out by MGN (``params.structure``). The opacity is the share of the light at
    each pixel that is the prominence's (in the linear data), so faint threads let the corona
    show through. Nothing lands on the Moon: cut at its limb, 1 px soft.
    """
    p = params or ProminenceParams()
    g = region.geometry
    h, w = region.rgb.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.hypot(xx - g.sun_x, yy - g.sun_y)
    d_moon = np.hypot(xx - g.moon_x, yy - g.moon_y)
    finite = np.isfinite(region.rgb).all(axis=2) & (region.weight > 0)
    outside = d_moon > g.moon_r
    valid = finite & outside

    wb = white_balance(np.nan_to_num(region.rgb), r, valid, g, p.white, None)
    rgb = np.nan_to_num(region.rgb) * wb[None, None, :]
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    height_px = p.height * g.sun_r
    search = valid & (r < g.sun_r + height_px)
    ring_px = 2
    ring = np.clip(((r - g.sun_r) / ring_px).astype(np.int32), 0, None)
    n = int(height_px / ring_px) + 3
    rc = np.clip(ring, 0, n - 1)

    # The corona's colour, height by height, from pixels that are not red themselves.
    redness = R / np.maximum(G, 1e-12)
    k0 = float(np.median(redness[search])) if search.any() else 1.0
    plain = search & ~ndimage.binary_dilation(redness > k0 * p.min_redness, iterations=2)
    plain &= ~ndimage.binary_dilation(~outside, iterations=2)
    kr = _ring_profile(redness, ring, plain, np.median, n)[rc]
    kb = _ring_profile(B / np.maximum(G, 1e-12), ring, plain, np.median, n)[rc]
    emis = np.stack([R - kr * G, np.zeros_like(G), B - kb * G], axis=-1).astype(np.float32)
    corona = np.stack([kr * G, G, kb * G], axis=-1)
    lum = emis[..., 0] + emis[..., 2]
    hp = lum - ndimage.median_filter(lum, size=3)
    noise = _ring_profile(np.abs(hp), ring, plain, np.median, n) * 1.4826 / 0.85
    sigma_n = np.maximum(noise[rc], 1e-12)

    # Sharpened with the blur measured on the Moon's edge.
    notes = []
    blur = getattr(region, "blur_px", None)
    if p.sharpen > 0 and blur:
        its = int(round(p.sharpen * 30))
        bias = 3.0 * float(np.median(noise))  # keeps the background positive for the deconvolution
        for c in (0, 2):
            emis[..., c] = richardson_lucy(emis[..., c] + bias, blur, its) - bias
        notes.append(f"sharpened: blur {blur:.2f} px, {its} iterations")
    elif p.sharpen > 0:
        notes.append("not sharpened: no blur measured (stack the prominences first)")
    lum = emis[..., 0] + emis[..., 2]

    # Pieces, for the count and the chromosphere.
    core = (ndimage.gaussian_filter(lum, 0.7) > p.sensitivity * sigma_n) & search
    labels, count = ndimage.label(core, structure=np.ones((3, 3)))
    keep = np.zeros(count + 1, bool)
    removed_chromo = 0
    if count:
        idx = np.arange(1, count + 1)
        area = ndimage.sum(core, labels, idx)
        keep[1:] = area >= max(4.0, (g.sun_r / 100.0) ** 2)
    cut = np.ones((h, w), np.float32)
    if count and not p.chromosphere:
        chromo_px = p.chromosphere_arcsec / region.pixscale
        low = (r - g.sun_r) < chromo_px
        tall = keep[labels] & ~low
        bases = ndimage.binary_dilation(tall, iterations=int(math.ceil(chromo_px)) + 2)
        rim = ndimage.binary_dilation(keep[labels] & low & ~bases, iterations=2) & low & ~bases
        removed_chromo = int(rim.sum())
        cut[rim] = 0.0
    cut *= np.clip(d_moon - g.moon_r, 0, 1)  # never on the Moon: cut at its limb, 1 px soft
    cut[~search] = 0.0
    emis *= cut[..., None]
    lum = emis[..., 0] + emis[..., 2]

    # Its own stretch: black at twice the noise, asinh so faint threads and bright cores both show.
    kept = keep[labels] & (cut > 0.5)
    if kept.sum() < 4:
        notes.append("no prominences found; try a lower sensitivity or a greater height")
        top = float(np.percentile(np.clip(emis, 0, None).max(axis=2)[search], 99.5)) if search.any() else 1.0
    else:
        top = float(np.percentile(np.clip(emis, 0, None).max(axis=2)[kept], 99.5))
    floor = 2.0 * sigma_n[..., None]
    s = 8.0
    y = np.arcsinh(s * np.clip(emis - floor, 0, None) / max(top, 1e-12)) / math.asinh(s)
    if p.structure > 0:
        from .enhance import mgn

        scales = [max(1.0, f * g.sun_r) for f in (0.003, 0.006, 0.012, 0.024)]
        m = mgn(np.clip(lum, 0, None), cut > 0, scales, 0.7, 0.0)
        y = y * np.clip(1.0 + p.structure * 2.0 * (m - 0.5), 0.0, None)[..., None]
    # The display colour about its luminance; green from the hydrogen lines' mix left as measured.
    yl = y.mean(axis=2, keepdims=True)
    y = yl + p.saturation * (y - yl)
    layer = np.clip(y * p.brightness, 0, 1).astype(np.float32)
    share = np.clip(lum, 0, None) / np.maximum(np.clip(lum, 0, None) + 2 * corona[..., 1], 1e-12)
    alpha = np.clip(share * cut, 0, 1).astype(np.float32)
    alpha[layer.max(axis=2) <= 0] = 0.0
    result = ProminenceResult(int(keep[1:].sum()) if count else 0, removed_chromo, int(kept.sum()), wb.tolist(), notes)
    return result, layer, alpha


def place(region: SunRegion, layer: np.ndarray) -> np.ndarray:
    """The region's layer on a black canvas the size of the HDR stack."""
    h, w = region.canvas
    out = np.zeros((h, w, layer.shape[2]), np.float32)
    x0, y0 = region.origin
    hh, ww = layer.shape[:2]
    out[y0 : y0 + hh, x0 : x0 + ww] = layer
    return out


MODES = ("normal", "screen", "lighten")


def composite_top(base: np.ndarray, layer: np.ndarray, alpha: np.ndarray, strength: float = 1.0,
                  mode: str = "normal") -> np.ndarray:
    """The prominence layer on top of ``base`` (display images, 0..1).

    normal (default): the prominences cover what is under them, with ``alpha`` (their soft outline)
    times ``strength`` as opacity, so their colour shows even over the bright inner corona.
    screen / lighten: as Photoshop's modes, with ``strength`` scaling the layer.
    """
    a = np.clip(alpha * strength, 0.0, 1.0)[..., None]
    if mode == "normal":
        out = base * (1 - a) + layer * a
    elif mode == "screen":
        out = 1.0 - (1.0 - base) * (1.0 - np.clip(layer * strength, 0, 1))
    elif mode == "lighten":
        out = np.maximum(base, np.clip(layer * strength, 0, 1))
    else:
        raise ValueError(f"unknown mode {mode!r}; choose from {', '.join(MODES)}")
    return np.clip(out, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# The prominence stack: sharpness measured on the Moon's edge
# ---------------------------------------------------------------------------


def limb_blur(rgb: np.ndarray, moon_x: float, moon_y: float, moon_r: float) -> Optional[tuple[float, float]]:
    """How blurred one frame is, from the sharp step at its own Moon's edge.

    The brightness across the limb (median of each quarter-pixel ring, all round, so prominences
    do not count): the Moon's level just inside (4-6 px in), the corona's just outside, from its
    trend further out continued to the edge. The blur is the distance between the points a
    quarter and three quarters of the way up the step, which is 1.349 sigma for a Gaussian: the
    core of the blur, not the slow tail of glare inside the limb (on the 2024 frames still 7 % of
    the step 4 px in). Returns (sigma, where the step is half way, relative to the Moon's radius)
    in px, or None if the frame does not show the edge (saturated, or no corona).
    """
    h, w = rgb.shape[:2]
    lum = np.nanmean(rgb, axis=2)
    yy, xx = np.mgrid[0:h, 0:w]
    d = np.hypot(xx - moon_x, yy - moon_y) - moon_r
    far = max(15.0, 0.035 * moon_r)
    sel = (d > -8) & (d < far) & np.isfinite(lum)
    if sel.sum() < 2000:
        return None
    q = 0.25
    k = np.floor((d[sel] + 8) / q).astype(int)
    n = int((far + 8) / q) + 1
    prof = np.full(n, np.nan)
    have = np.unique(k)
    prof[have] = ndimage.median(lum[sel], k, index=have)
    dd = np.arange(n) * q - 8 + q / 2
    ok = np.isfinite(prof)
    inner = (dd > -6) & (dd < -4) & ok
    out = (dd > 5) & ok & (prof > 0)
    if not inner.any() or out.sum() < 8:
        return None
    moon = float(np.median(prof[inner]))
    c = np.polyfit(dd[out], np.log(prof[out]), 1)
    top = float(np.exp(np.polyval(c, 0.0)))
    if top <= 2 * max(moon, 0.0):
        return None  # no step to measure
    frac = (prof - moon) / (top - moon)
    step = (dd > -4) & (dd < 4) & ok
    x, f = dd[step], np.maximum.accumulate(np.nan_to_num(frac[step]))  # rising through the edge

    def cross(level):
        j = int(np.argmax(f >= level))
        if j == 0 or f[j] < level:
            return None
        return float(x[j - 1] + (level - f[j - 1]) / max(f[j] - f[j - 1], 1e-12) * (x[j] - x[j - 1]))

    a, m, b = cross(0.25), cross(0.5), cross(0.75)
    if a is None or b is None or m is None:
        return None
    sigma = (b - a) / 1.349
    if not 0.1 < sigma < 5.0:
        return None
    return float(sigma), float(m)


# ---------------------------------------------------------------------------
# Swapping the prominences into the HDR stack (linear, before the stretch)
# ---------------------------------------------------------------------------


def _emission(rgb: np.ndarray, r: np.ndarray, sun_r: float, search: np.ndarray, plain_from: np.ndarray,
              min_redness: float, ring_px: float = 2.0):
    """A stack's prominence light, the corona taken off: (emission (h, w, 3) with green 0, noise of
    its luminance (h, w), corona red/green and blue/green (h, w)). Linear, camera colour."""
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    n = int((r[search].max() - sun_r) / ring_px) + 3 if search.any() else 3
    ring = np.clip(((r - sun_r) / ring_px).astype(np.int32), 0, n - 1)
    redness = R / np.maximum(G, 1e-30)
    k0 = float(np.median(redness[plain_from])) if plain_from.any() else 1.0
    plain = plain_from & ~ndimage.binary_dilation(redness > k0 * min_redness, iterations=2)
    kr = _ring_profile(redness, ring, plain, np.median, n)[ring]
    kb = _ring_profile(B / np.maximum(G, 1e-30), ring, plain, np.median, n)[ring]
    emis = np.stack([R - kr * G, np.zeros_like(G), B - kb * G], axis=-1).astype(np.float32)
    lum = emis[..., 0] + emis[..., 2]
    hp = lum - ndimage.median_filter(lum, size=3)
    noise = _ring_profile(np.abs(hp), ring, plain, np.median, n)[ring] * 1.4826 / 0.85
    return emis, np.maximum(noise, 1e-30).astype(np.float32), kr, kb


def _significance(emis: np.ndarray, noise: np.ndarray, corona: np.ndarray, tolerance: float,
                  valid: Optional[np.ndarray] = None) -> np.ndarray:
    """How surely a pixel's light is a prominence's: 0 up to a threshold, rising to 1 over as much
    again (smoothed a little, so single noisy pixels do not count). The threshold is 3x the noise,
    or ``tolerance`` times the corona's own red and blue there if that is more: the corona's colour
    varies a little from place to place, which a clean stack shows far above its noise (on the
    2024 HDR stack 4-21 % of the corona would otherwise count)."""
    lum = emis[..., 0] + emis[..., 2]
    if valid is None:
        lum = ndimage.gaussian_filter(lum, 0.7)
    else:  # smoothed over the valid pixels only (not pulled down by the Moon's zeros at its edge)
        m = valid.astype(np.float32)
        lum = ndimage.gaussian_filter(lum * m, 0.7) / np.maximum(ndimage.gaussian_filter(m, 0.7), 1e-6)
    t = np.maximum(3.0 * noise, tolerance * corona)
    return np.clip((lum - t) / np.maximum(t, 1e-30), 0.0, 1.0).astype(np.float32)


def corona_behind(rgb: np.ndarray, zone: np.ndarray, usable: np.ndarray, g: Geometry, step: float = 0.5) -> np.ndarray:
    """The corona behind the pixels of ``zone`` (where prominences are), from the corona beside
    them: at each distance from the Sun's centre, carried across the gap in angle from both sides
    (the corona changes steeply with distance near the limb, slowly with angle). Only ``usable``
    pixels (off the Moon, with data, outside the zone) are carried. Returns (h, w, 3): ``rgb``
    with the zone's pixels replaced."""
    out = np.array(rgb, np.float32, copy=True)
    ys, xs = np.nonzero(zone)
    if ys.size == 0:
        return out
    rz = np.hypot(xs - g.sun_x, ys - g.sun_y)
    r0, r1 = max(1.0, float(rz.min()) - 2.0), float(rz.max()) + 2.0
    radii = np.arange(r0, r1 + step, step)
    na = max(64, int(math.ceil(2 * math.pi * r1 / step)))
    ang = np.arange(na) * 2 * math.pi / na
    X = g.sun_x + radii[:, None] * np.cos(ang)[None, :]
    Y = g.sun_y + radii[:, None] * np.sin(ang)[None, :]
    ok = ndimage.map_coordinates(usable.astype(np.float32), [Y, X], order=0, cval=0.0) > 0.5
    P = np.stack([ndimage.map_coordinates(np.nan_to_num(rgb[..., c]), [Y, X], order=1, cval=0.0)
                  for c in range(3)], axis=-1)
    for i in range(len(radii)):
        good = ok[i]
        if good.sum() < 2 or good.all():
            continue
        bad = ~good
        for c in range(3):
            P[i, bad, c] = np.interp(ang[bad], ang[good], P[i, good, c], period=2 * math.pi)
    Pw = np.concatenate([P, P[:, :1]], axis=1)  # wrap in angle
    jr = (rz - r0) / step
    ja = np.mod(np.arctan2(ys - g.sun_y, xs - g.sun_x), 2 * math.pi) / (2 * math.pi) * na
    for c in range(3):
        out[ys, xs, c] = ndimage.map_coordinates(Pw[..., c], [jr, ja], order=1, mode="nearest")
    return out


def swap_prominences(hdr: np.ndarray, weight: np.ndarray, prom: np.ndarray, prom_weight: np.ndarray,
                     g: Geometry, params: Optional[ProminenceParams] = None, blur_px: Optional[float] = None,
                     pixscale: float = 2.4, say=None) -> dict:
    """Put the prominence stack's prominences into the HDR stack, in place (linear, the same units).

    In the HDR stack the prominences come mostly from the longest frames not saturated there,
    swollen and soft with a red glow; the short frames show them sharp. In both stacks their own
    light is the light left when the corona (white: its red and blue in the corona's own ratio to
    green) is taken off, so swapping it leaves the corona as it was: the HDR stack's prominence
    light is taken out and the prominence stack's, sharpened with the blur measured on the Moon's
    edge, put in, each where it stands above its own noise (0 within 3x the noise, all of it by
    5x), nowhere else. Where the HDR stack has no data (saturated in every frame) the prominence
    stack's pixels are used. Never on the Moon (the HDR stack's, which the final image draws):
    nothing changes inside its limb. ``hdr``, ``prom``: (h, w, 3) camera RGB over the same pixels.
    Returns what was done, for the log.
    """
    p = params or ProminenceParams()
    say = say or (lambda _m: None)
    h, w = hdr.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.hypot(xx - g.sun_x, yy - g.sun_y)
    d_moon = np.hypot(xx - g.moon_x, yy - g.moon_y)
    outside = d_moon > g.moon_r
    both = outside & (weight > 0) & (prom_weight > 0) & np.isfinite(hdr).all(axis=2) & np.isfinite(prom).all(axis=2)
    # The whole prominence stack's square (no height limit: a limit would cut a tall prominence's
    # glow), including where its frames' Moons covered the limb.
    pys, pxs = np.nonzero(prom_weight > 0)
    square = np.zeros((h, w), bool)
    if pys.size:
        square[pys.min():pys.max() + 1, pxs.min():pxs.max() + 1] = True
    search = outside & square
    plain = both & search & (d_moon > g.moon_r + 2)
    eh, nh, krh, kbh = _emission(np.nan_to_num(hdr), r, g.sun_r, search, plain, p.min_redness)
    ep, npn, krp, kbp = _emission(np.nan_to_num(prom), r, g.sun_r, search, plain, p.min_redness)
    tol = max(p.min_redness - 1.0, 0.0)  # how much redder than the corona (Redness 1.2: 20 %)
    ch = (krh + kbh) * np.nan_to_num(hdr[..., 1])
    cp = (krp + kbp) * np.nan_to_num(prom[..., 1])
    # One weight for both: where either stack shows a prominence, its light is the short frames'.
    # (Found by colour: redder than the corona there, see _significance.)
    vh = outside & (weight > 0) & np.isfinite(hdr).all(axis=2)
    vp = outside & (prom_weight > 0) & np.isfinite(prom).all(axis=2)
    sw = np.maximum(_significance(eh, nh, ch, tol, vh) * vh, _significance(ep, npn, cp, tol, vp) * vp)
    sw[~search] = 0.0
    # Prominences stand on the limb: pieces that come no nearer than a tenth of the Sun's radius
    # (specks far out in the corona or the sky) are left alone.
    lab, n = ndimage.label(sw > 0, structure=np.ones((3, 3)))
    if n:
        near = np.zeros(n + 1, bool)
        near[1:] = ndimage.minimum(r - g.sun_r, lab, np.arange(1, n + 1)) < 0.1 * g.sun_r
        sw *= near[lab]
    sp = sw
    # The HDR stack's own soft glow round them (2024: 1 % of the peak 3-8 superpixels out) is often
    # fainter than the corona's colour tolerance: its light is taken out too wherever it is above
    # its noise, within 5 % of the Sun's radius of a prominence (only taken out: the short frames'
    # light goes in where it counts as prominence, not their noise).
    reach = max(2, int(round(0.05 * g.sun_r)))
    grown = ndimage.binary_dilation(sw > 0, iterations=reach)
    lum_h = ndimage.gaussian_filter(eh[..., 0] + eh[..., 2], 0.7)
    glow = grown & search & (lum_h > 3.0 * nh)
    wh = np.maximum(sw, glow.astype(np.float32))
    # There, their own light in every colour (hydrogen's red and blue, and helium's yellow, partly
    # in green) against the corona behind them, carried in from beside them: so their brightness
    # can be set without changing their colour.
    zone = ndimage.binary_dilation(wh > 0, iterations=2) & search
    use_h = outside & (weight > 0) & np.isfinite(hdr).all(axis=2) & ~zone
    eh_colour = eh  # the HDR stack's red and blue beyond the corona's colour (green 0)
    # The corona behind them from the HDR stack (clean) for both stacks: they are in the same units
    # with the same corona, so at brightness 1 a swapped pixel is simply the short frames' pixel.
    # (From each stack separately, the short frames' estimate by their own Moon's edge came out
    # too dark, and the swap overshot: stepped lines, on Andy's 2024 stack +0.16 at 1.1-1.3.)
    bg = corona_behind(hdr, zone, use_h, g)
    eh = np.where(zone[..., None], np.nan_to_num(hdr) - bg, 0.0).astype(np.float32)
    ep = np.where(zone[..., None], np.nan_to_num(prom) - bg, 0.0).astype(np.float32)
    # The short frames' pixels next to where they have no data (their own Moon's edge, blurred
    # into them: on the 2024 stack 4-6 % darker for 3 px) are not used: within 3x the measured
    # blur, the HDR stack's stay.
    have_p = (prom_weight > 0) & np.isfinite(prom).all(axis=2)
    clear = ndimage.distance_transform_edt(have_p) > 3.0 * float(blur_px or 1.0)
    sw_all = sw.copy()  # where either stack shows a prominence
    sw, wh = sw * clear, wh * clear  # (where the short frames have no data, the HDR stack's stays)
    if p.sharpen > 0 and blur_px:
        its = int(round(p.sharpen * 30))
        bias = 3.0 * float(np.median(npn[search]))
        for c in range(3):
            ep[..., c] = np.where(zone, richardson_lucy(ep[..., c] + bias, blur_px, its) - bias, 0.0)
    if p.structure > 0:
        from .enhance import mgn

        scales = [max(1.0, f * g.sun_r) for f in (0.003, 0.006, 0.012, 0.024)]
        m = mgn(np.clip(ep.sum(axis=2), 0, None), zone, scales, 0.7, 0.0)
        ep *= np.clip(1.0 + p.structure * 2.0 * (m - 0.5), 0.0, None)[..., None]
    ep *= np.float32(p.brightness)
    if not p.chromosphere:  # the thin rim along the solar limb, except the bases of taller prominences
        chromo_px = p.chromosphere_arcsec / pixscale
        low = (r - g.sun_r) < chromo_px
        lab, n = ndimage.label(sp > 0.5, structure=np.ones((3, 3)))
        tall = np.zeros(n + 1, bool)
        if n:
            tall[1:] = ndimage.maximum(r - g.sun_r, lab, np.arange(1, n + 1)) >= chromo_px
        bases = ndimage.binary_dilation(tall[lab] & ~low, iterations=int(math.ceil(chromo_px)) + 2)
        rim = low & ~bases
        ep[rim] = 0.0  # the rim's light is taken out, none put back
    # Where they are: the HDR stack's prominence light out, the short frames' in. In the glow round
    # them only its colour excess out (red and blue beyond the corona's colour), so the corona keeps
    # its own texture there.
    # Where the short frames have no data the HDR stack's prominence light stays, at the same
    # brightness setting (else a seam where the short frames' data begins).
    keep = (sw_all - sw)[..., None] * np.float32(p.brightness - 1.0) * eh
    change = (sw[..., None] * (ep - eh) - (wh - sw)[..., None] * eh_colour + keep) * outside[..., None]
    hdr += change.astype(np.float32)
    holes = outside & ~(weight > 0) & (prom_weight > 0) & np.isfinite(prom).all(axis=2)
    hdr[holes] = prom[holes]
    weight[holes] = prom_weight[holes]
    kept = int(((sp > 0.5) & outside).sum())
    lab, n = ndimage.label((sp > 0.5) & outside, structure=np.ones((3, 3)))
    area = ndimage.sum(np.ones_like(sp), lab, np.arange(1, n + 1)) if n else np.zeros(0)
    count = int((area >= max(4.0, (g.sun_r / 100.0) ** 2)).sum())
    info = {"count": count, "pixels": kept,
            "filled_px": int(holes.sum()), "sharpened": bool(p.sharpen > 0 and blur_px)}
    say(f"prominences: {count} from the short frames ({kept} px)"
        + (f", sharpened with the blur measured on the Moon's edge ({blur_px:.2f} px)" if info["sharpened"] else "")
        + (f"; {info['filled_px']} px the HDR stack had no data for filled from them" if info["filled_px"] else ""))
    return info


def moon_cover_px(g: Geometry, n: int = 720) -> float:
    """How far the Moon's edge reaches above the Sun's edge where it reaches least, all round (px);
    negative where the Sun's edge shows (near the contacts). In mid-totality the Moon is larger than
    the Sun and covers the chromosphere everywhere (2024 stack: Moon 436 px, Sun 413 px)."""
    dx, dy = g.moon_x - g.sun_x, g.moon_y - g.sun_y
    a = np.arange(n) * 2 * math.pi / n
    ud = np.cos(a) * dx + np.sin(a) * dy
    disc = g.moon_r ** 2 - (dx * dx + dy * dy) + ud * ud
    if (disc < 0).any():
        return -float("inf")
    return float(np.min(ud + np.sqrt(disc) - g.sun_r))
