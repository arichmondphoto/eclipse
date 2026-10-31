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
    min_redness: float = 1.2  # red/green ratio needed (the white corona is about 1)
    chromosphere: bool = True  # keep the thin rim along the solar limb
    chromosphere_arcsec: float = 12.0  # the rim's height above the solar limb
    brightness: float = 1.0
    saturation: float = 1.2
    feather_px: float = 1.5
    white: str = "corona"


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


def find_prominences(region: SunRegion, params: Optional[ProminenceParams] = None):
    """Returns (result, layer (h, w, 3) display image in 0..1 over the region, soft mask)."""
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
    R, G = rgb[..., 0], rgb[..., 1]
    height_px = p.height * g.sun_r
    search = valid & (r < g.sun_r + height_px)
    ring_px = 2
    ring = ((r - g.sun_r) / ring_px).astype(np.int32)
    ring = np.clip(ring, 0, None)
    n = int(height_px / ring_px) + 3
    # The corona's own colour can drift slightly with height: take it out, circle by circle.
    # (Only from pixels that are not red themselves: low down, the uncovered limb may be all chromosphere.)
    excess = R - G
    redness = R / np.maximum(G, 1e-12)
    plain = search & ~ndimage.binary_dilation(redness >= p.min_redness, iterations=2)
    plain &= ~ndimage.binary_dilation(~outside, iterations=2)  # and not right at the lunar limb
    base = _ring_profile(excess, ring, plain, np.median, n)
    e = excess - base[np.clip(ring, 0, n - 1)]
    # Pixel noise from a 3x3 median residual, which edges (limb, rim) do not inflate.
    hp = e - ndimage.median_filter(e, size=3)
    noise = _ring_profile(np.abs(hp), ring, plain, np.median, n) * 1.4826 / 0.85
    sigma = np.maximum(noise[np.clip(ring, 0, n - 1)], 1e-12)
    soft = np.clip((e / sigma - p.sensitivity) / 2.0, 0, 1) * np.clip((redness - p.min_redness) / 0.1, 0, 1)
    soft = ndimage.gaussian_filter(soft, 0.7)
    soft[~search] = 0.0

    # Separate pieces; drop specks, and the chromosphere when asked.
    core = soft > 0.5
    labels, count = ndimage.label(core, structure=np.ones((3, 3)))
    keep = np.zeros(count + 1, bool)
    removed_chromo = 0
    if count:
        idx = np.arange(1, count + 1)
        area = ndimage.sum(core, labels, idx)
        top = ndimage.maximum(r - g.sun_r, labels, idx)
        min_area = max(4.0, (g.sun_r / 100.0) ** 2)
        keep[1:] = area >= min_area
    grown = ndimage.binary_dilation(keep[labels], iterations=3)
    if count and not p.chromosphere:
        # The rim: everything lower than chromosphere_arcsec, except the bases of taller prominences
        # (low pixels within reach of a part that rises above the rim).
        chromo_px = p.chromosphere_arcsec / region.pixscale
        low = (r - g.sun_r) < chromo_px
        tall = keep[labels] & ~low
        bases = ndimage.binary_dilation(tall, iterations=int(math.ceil(chromo_px)) + 2)
        removed_chromo = int((grown & low & ~bases).sum())
        grown &= ~low | bases
    alpha = soft * grown
    if p.feather_px > 0:
        alpha = ndimage.gaussian_filter(alpha, p.feather_px / 2)
    alpha *= np.clip(d_moon - g.moon_r, 0, 1)  # never on the Moon: cut at its limb, 1 px soft
    alpha = np.clip(alpha, 0, 1).astype(np.float32)

    # Display: an asinh stretch keeps faint threads and bright cores; colour about the luminance.
    kept = alpha > 0.5
    notes = []
    if kept.sum() < 4:
        notes.append("no prominences found; try a lower sensitivity or redness, or a greater height")
        top_v = float(np.percentile(rgb.max(axis=2)[search], 99.5)) if search.any() else 1.0
    else:
        top_v = float(np.percentile(rgb.max(axis=2)[kept], 99.5))
    s = 8.0
    y = np.arcsinh(s * np.clip(rgb, 0, None) / max(top_v, 1e-12)) / math.asinh(s)
    lum = y.mean(axis=2, keepdims=True)
    y = lum + p.saturation * (y - lum)
    layer = np.clip(y * p.brightness, 0, 1) * alpha[..., None]
    result = ProminenceResult(int(keep[1:].sum()) if count else 0, removed_chromo, int(kept.sum()), wb.tolist(), notes)
    return result, layer.astype(np.float32), alpha


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
