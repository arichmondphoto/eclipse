"""Earthshine: the night side of the Moon, lit by sunlight reflected from the Earth.

During totality the Moon moves across the Sun by about its own radius per
hour (about 40 px in three minutes at 400 mm), so a stack aligned on the
corona blurs the lunar surface. Here every frame keeps its corona alignment
(rotation and all) plus a shift that puts its Moon exactly where the Moon is
in the HDR stack's Moon frame. The result lines up with the black disc of
the corona layers and drops straight into it.

Inside the disc the light is earthshine plus glare: corona light scattered
in the lens and the atmosphere, a smooth brightness gradient, strongest
towards the brightest inner corona. The glare is removed with a smooth
surface fitted to the disc, leaving the lunar surface: dark maria on
brighter highlands. Close to the limb the glare climbs steeply (the inner
corona is only a few pixels away), far faster than any gentle surface can
follow, so the fit also has terms that fall off with distance inside the
limb, varying round it as the inner corona does.

Only the longer exposures carry earthshine; frames more than ``stops``
below the longest are skipped, as they would add read noise only.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
from scipy import ndimage

from .fits import write_fits
from .rawio import RawMosaic, region
from .stack import FrameInfo, Match, source_box, warp_region

Say = Callable[[str], None]


@dataclass
class EarthshineParams:
    stops: float = 6.0  # use frames within this many stops of the longest exposure
    margin_px: float = 3.0  # stay this far inside the lunar limb (limb glare, chromosphere), full-res px
    glare_order: int = 2  # polynomial order of the glare (a gentle gradient); higher also eats large maria
    smooth: float = 1.0  # Gaussian smoothing of the shown layer, px (0 = none)
    brightness: float = 0.6  # brightest lunar surface in the layer, 0..1 (the corona stays brighter)
    limb_darkening: float = 0.6  # how much darker the limb is than the middle (0 = even, 1 = black at the limb)
    vignette_shape: float = 1.5  # how the darkening falls from the centre (1 = straight line, higher = later)
    edge_smooth: float = 3.0  # most smoothing where the noise is high near the limb, px
    limb_glare: bool = True  # also model the steep glare just inside the limb
    limb_shape: bool = True  # fit the limb glare as light fading inward, round the limb (keeps maria by the limb)
    source_glare: bool = True  # glare round a bright prominence (a fan with diffraction spikes), fitted from the prominence
    channel: str = "auto"  # colour channels to use: auto, all, green, green+blue, blue
    albedo_map: str = ""  # lunar albedo map (equirectangular, 0 deg longitude centred) to guide the glare removal
    albedo_strength: float = 0.5  # how much the map guides it, 0..1 (only used with a map)
    albedo_min_match: float = 0.6  # the map is not used if it matches the middle of the disc less well than this
    limb_px: float = 1.5  # the corona layers' black disc reaches this far past the limb (enhance's margin)
    feather_px: float = 1.0  # soft edge of the disc in the layer (only without a measured limb)
    keep_glare: float = 0.5  # how much of the removed glare is kept as a glow inside the limb, 0..1
    limb_reach: float = 6.0  # px past the limb the stack and layer reach: the real limb, from the data
    limb_soften: float = 1.5  # extra softening of the limb, px (Gaussian), on top of the real one
    edge_grain: bool = True  # put grain like the rest of the surface into the filled last pixels
    colour: bool = True  # natural colour: the glow and limb (prominences red) and the earthshine's own tint
    saturation: float = 1.0  # colour saturation, as for the corona layers (0 = grey)
    saturation_level: float = 0.95
    saturation_taper: float = 0.15
    noise_floor: float = 0.002
    hot_pixels: bool = True


@dataclass
class EarthshineResult:
    frames: list  # frame indices used
    shifts: dict  # frame index -> (dx, dy) applied to put its Moon on the Moon frame's
    crop: tuple  # x0, y0, w, h of the Moon crop in reference coordinates
    snr: float  # contrast of the lunar surface over the noise (a guide: < 3 means little to see)
    outputs: dict
    notes: list = field(default_factory=list)

    def describe(self) -> str:
        lines = [f"Frames used: {len(self.frames)} ({', '.join('#' + str(i) for i in self.frames)})",
                 f"Moon shifts applied: up to {max((math.hypot(*d) for d in self.shifts.values()), default=0):.1f} px",
                 f"Lunar surface detail / noise: {self.snr:.1f}"]
        lines += [f"{k}: {v}" for k, v in self.outputs.items()]
        lines += [f"Note: {n}" for n in self.notes]
        return "\n".join(lines)


def moon_aligned(matrix, moon_xy: tuple[float, float], target_xy: tuple[float, float]) -> list:
    """The corona alignment followed by the shift that moves this frame's Moon onto ``target``."""
    m = np.asarray(matrix, float).copy()
    m[0, 2] += target_xy[0] - moon_xy[0]
    m[1, 2] += target_xy[1] - moon_xy[1]
    return m.tolist()


# Fall-off lengths of the limb glare, in lunar radii, each with how fast it may vary round the
# limb (cos/sin of up to that multiple of the angle). The short ones hug the limb and may vary
# more; the long ones reach far into the disc, so they are kept simple to leave the maria alone.
LIMB_TERMS = ((0.004, 2), (0.01, 2), (0.025, 2), (0.06, 1))


def glare_surface(img: np.ndarray, inside: np.ndarray, cx: float, cy: float, r: float, order: int,
                  iters: int = 6, limb: bool = False, max_points: int = 300_000) -> np.ndarray:
    """Robust fit of the glare over the disc; returned over the whole crop.

    A 2-D polynomial of ``order`` (the gentle gradient across the disc), plus,
    with ``limb``, terms exp(-d/s) times low harmonics of the position angle,
    where d is the distance inside the limb: the steep rise of scattered light
    towards the bright inner corona.
    """
    h, w = img.shape
    yy, xx = np.mgrid[0:h, 0:w]
    u = (xx - cx) / r
    v = (yy - cy) / r
    terms = [(i, j) for i in range(order + 1) for j in range(order + 1 - i)]

    def basis(sel):
        cols = [(u[sel] ** i) * (v[sel] ** j) for i, j in terms]
        if limb:
            us, vs = u[sel], v[sel]
            rho = np.hypot(us, vs)
            d = np.clip(1.0 - rho, 0.0, None)
            phi = np.arctan2(vs, us)
            for s, nh in LIMB_TERMS:
                e = np.exp(-d / s)
                cols.append(e)
                for k in range(1, nh + 1):
                    cols += [e * np.cos(k * phi), e * np.sin(k * phi)]
        return np.stack(cols, axis=1)

    idx = np.flatnonzero(inside)
    if idx.size > max_points:
        idx = idx[:: int(math.ceil(idx.size / max_points))]
    sel = np.unravel_index(idx, inside.shape)
    A = basis(sel)
    y = img[sel].astype(np.float64)
    wts = np.ones_like(y)
    coef = np.zeros(A.shape[1])
    for _ in range(iters):
        sw = np.sqrt(wts)
        coef, *_ = np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)
        res = y - A @ coef
        s = 1.4826 * np.median(np.abs(res)) + 1e-30
        t = res / (4.685 * s)
        wts = np.where(np.abs(t) < 1, (1 - t * t) ** 2, 0.0)
    out = np.empty(h * w)
    for a in range(0, h, 256):  # in bands, to keep the design matrix small
        band = np.zeros((h, w), bool)
        band[a : a + 256] = True
        out[a * w : min(h, a + 256) * w] = basis(band) @ coef
    return out.reshape(h, w).astype(np.float32)


# Fall-off lengths in lunar radii, with a floor in pixels (lens blur does not shrink with a
# smaller Moon); the fit reaches 3x the longest inside the limb.
def limb_background(lum: np.ndarray, glare: np.ndarray, good: np.ndarray, cx: float, cy: float, r: float,
                    depth: float = 0.275, width0: float = 0.014, growth: float = 1.0,
                    quad: float = 15.0) -> np.ndarray:
    """What is left of the limb glare after :func:`glare_surface`, followed closely round the limb.

    Returns a factor (1 away from the limb) for the glare surface. The light just
    inside the limb comes from whatever is just outside it in each frame: inner
    corona, chromosphere and prominences, which change within a few degrees and
    are uncovered differently as the Moon moves. Here the ratio of the image to
    the glare surface (which varies far more slowly with depth than the glare
    itself) is resampled onto circles, and each circle is median-filtered round
    the limb. The window is a few degrees at the limb and widens quickly with depth
    (``width0`` x radius + ``growth`` x depth + ``quad`` x depth^2 / radius, in px:
    about 1 degree at the limb, 13 at 40 px in, 50 at 90 px in on a 436 px Moon),
    because glare from
    farther away is spread over a wider angle; lunar features smaller than the
    window pass through, larger ones close to the limb are flattened. The
    correction fades out by ``depth`` x radius inside the limb.
    """
    h, w = lum.shape
    reach = max(depth * r, 8.0)
    n = max(64, int(round(2 * math.pi * r)))
    step = 0.5
    d_in = np.arange(0.0, reach + step, step)
    t = np.arange(n) / n * 2 * math.pi - math.pi
    rad = r - d_in[:, None]
    ys = cy + rad * np.sin(t)[None, :]
    xs = cx + rad * np.cos(t)[None, :]
    q = np.where(good, lum / np.where(np.abs(glare) > 1e-12, glare, 1e-12), 0.0)
    g = good.astype(np.float32)
    cover = ndimage.map_coordinates(g, [ys, xs], order=1)
    polar = ndimage.map_coordinates(q, [ys, xs], order=1) / np.maximum(cover, 1e-6)
    bg = np.ones_like(polar)
    idx = np.arange(n)
    width0_px = width0 * r
    for i, dv in enumerate(d_in):
        ok = cover[i] > 0.5
        if ok.sum() < 10:
            continue
        row = np.interp(idx, idx[ok], polar[i][ok], period=n)
        win = int(min(width0_px + growth * dv + quad * dv * dv / r, n / 2)) | 1
        win = max(win, 5)
        row = ndimage.median_filter(row, win, mode="wrap")
        bg[i] = ndimage.uniform_filter1d(row, max(3, win // 2) | 1, mode="wrap")
    yy, xx = np.mgrid[0:h, 0:w]
    dist = r - np.hypot(xx - cx, yy - cy)
    sel = good & (dist < reach)
    out = np.ones((h, w), np.float32)
    if not sel.any():
        return out
    fi = np.clip(dist[sel] / step, 0, len(d_in) - 1)
    fj = (np.arctan2(yy[sel] - cy, xx[sel] - cx) + math.pi) / (2 * math.pi) * n
    v = ndimage.map_coordinates(bg, [fi, fj], order=1, mode="grid-wrap")
    fade = np.clip((reach - dist[sel]) / (0.4 * reach), 0.0, 1.0)
    fade = 0.5 - 0.5 * np.cos(math.pi * fade)
    out[sel] = 1.0 + (v - 1.0) * fade
    return out


def _decreasing(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted least-squares fit of a non-increasing sequence (pool adjacent violators)."""
    vals, wts, lens = [], [], []
    for yi, wi in zip(y.tolist(), np.maximum(w, 1e-30).tolist()):
        vals.append(yi), wts.append(wi), lens.append(1)
        while len(vals) > 1 and vals[-2] < vals[-1]:
            v2, w2, l2 = vals.pop(), wts.pop(), lens.pop()
            vals[-1] = (vals[-1] * wts[-1] + v2 * w2) / (wts[-1] + w2)
            wts[-1] += w2
            lens[-1] += l2
    return np.repeat(vals, lens).astype(np.float64)


def _source_glare(detail: np.ndarray, use: np.ndarray, sx: float, sy: float, inward: float, reach: float,
                  dth: float = 0.5, dl: float = 0.04, iters: int = 30) -> np.ndarray:
    """Glare round one bright source on the limb, as seen from the source itself.

    In polar coordinates about the source (distance rho, direction theta) its glare, diffraction
    spikes included, is a fan: an angular pattern a(theta) times a fall-off f(rho) shared by
    every direction. The lunar surface does not follow that pattern, so a robust rank-1 fit of
    the residual in (log rho, theta) takes the fan and leaves the maria. Returns the fan.
    """
    h, w = detail.shape
    yy, xx = np.mgrid[0:h, 0:w]
    rho = np.hypot(xx - sx, yy - sy)
    th = np.degrees(np.arctan2(yy - sy, xx - sx))
    dev = (th - inward + 180.0) % 360.0 - 180.0  # direction relative to straight into the disc
    near = (rho > 2.0) & (rho < reach) & (np.abs(dev) < 90.0)
    li = (np.log(np.maximum(rho, 2.0) / 2.0) / dl).astype(int)
    ti = ((dev + 90.0) / dth).astype(int)
    nl, nt = int(math.log(reach / 2.0) / dl) + 1, int(180.0 / dth) + 1
    li, ti = np.clip(li, 0, nl - 1), np.clip(ti, 0, nt - 1)
    m = use & near
    lab = li * nt + ti
    cnt = np.bincount(lab[m], minlength=nl * nt).reshape(nl, nt).astype(np.float64)
    tot = np.bincount(lab[m], weights=detail[m].astype(np.float64), minlength=nl * nt).reshape(nl, nt)
    g = tot / np.maximum(cnt, 1.0)
    # Each bin counts the same (bins far out hold many more pixels, and more of the surface).
    base = (cnt > 0).astype(np.float64)
    wgt = base.copy()
    f = np.sqrt(np.maximum((wgt * g * g).sum(axis=1), 1e-30))
    a = np.zeros(nt)
    for _ in range(iters):
        a = (wgt * f[:, None] * g).sum(axis=0) / np.maximum((wgt * f[:, None] ** 2).sum(axis=0), 1e-30)
        a = ndimage.uniform_filter1d(a, 3, mode="nearest")
        fw = (wgt * a[None] ** 2).sum(axis=1)
        f = (wgt * a[None] * g).sum(axis=1) / np.maximum(fw, 1e-30)
        # Glare only fades with distance from its source: a mare crossing the fan cannot put a
        # dip into the fall-off (which every direction shares).
        if f[: max(3, nl // 10)].mean() < 0:
            f, a = -f, -a
        f = _decreasing(np.maximum(f, 0.0), fw)
        res = g - np.outer(f, a)
        sc = 1.4826 * float(np.median(np.abs(res[cnt > 0]))) + 1e-30
        wgt = base / (1.0 + (res / (3.0 * sc)) ** 2)  # maria and noise count less
    # Read back between bin centres (the fall-off is steep near the source).
    lf = np.log(np.maximum(rho, 2.0) / 2.0) / dl - 0.5
    tf = (dev + 90.0) / dth - 0.5
    model = np.interp(lf, np.arange(nl), f) * np.interp(tf, np.arange(nt), a)
    fade = np.clip((reach - rho) / (0.3 * reach), 0.0, 1.0) * np.clip((90.0 - np.abs(dev)) / 3.0, 0.0, 1.0)
    return np.where(near, model * fade, 0.0).astype(np.float32)


def source_glare(detail: np.ndarray, good: np.ndarray, cx: float, cy: float, r: float, margin: float,
                 reach: float = 0.35, max_sources: int = 4, strength: float = 3.0, passes: int = 1):
    """Glare from bright prominences, which the limb terms (smooth round the limb) cannot follow.

    A source is where the residual just inside the limb stands well above the range of the
    lunar surface (``strength`` x its 1-99 % spread, both smoothed by 3 px). Each is fitted by
    :func:`_source_glare`, brightest first. Returns (glare, list of source angles in degrees).
    """
    h, w = detail.shape
    yy, xx = np.mgrid[0:h, 0:w]
    rr = np.hypot(xx - cx, yy - cy)
    phi = np.degrees(np.arctan2(yy - cy, xx - cx))
    use = good & (rr < r - max(margin, 0.025 * r))
    body = good & (rr < 0.8 * r)
    total = np.zeros((h, w), np.float32)
    found = []
    if body.sum() < 100 or use.sum() < 100:
        return total, found
    den = np.maximum(ndimage.gaussian_filter(use.astype(np.float32), 3.0), 1e-6)
    band = use & (rr > r - max(margin, 0.025 * r) - 0.07 * r)
    nb = 720
    ib = (((phi + 180.0) / 360.0) * nb).astype(int) % nb
    res = detail.astype(np.float32)
    for _ in range(max_sources):
        sm = ndimage.gaussian_filter(np.where(use, res, 0.0), 3.0) / den
        p1, p99 = np.percentile(sm[body], [1, 99])
        prof = np.asarray(ndimage.maximum(np.where(band, sm, -np.inf), np.where(band, ib, -1), index=np.arange(nb)))
        for a0 in found:  # one fit per source
            prof[np.abs(((np.arange(nb) + 0.5) / nb * 360.0 - 180.0 - a0 + 180.0) % 360.0 - 180.0) < 6.0] = -np.inf
        k = int(np.argmax(prof))
        if not np.isfinite(prof[k]) or prof[k] < p99 + strength * (p99 - p1):
            break
        peak = (k + 0.5) / nb * 360.0 - 180.0
        # The brightest side of the fan is not where the source is: try nearby places on the
        # limb (and just outside it) and keep the one whose fan leaves the least behind.
        best = None
        for ang in peak + np.arange(-2.0, 2.01, 0.5):
            for out in (0.0, 0.01 * r):
                sx, sy = cx + (r + out) * math.cos(math.radians(ang)), cy + (r + out) * math.sin(math.radians(ang))
                left, g = res, np.zeros_like(total)
                for _ in range(passes):  # each pass a further rank-1 term (spikes fade unlike the halo)
                    gi = _source_glare(left, use, sx, sy, ang + 180.0, reach * r)
                    left, g = left - gi, g + gi
                zone = use & (np.hypot(xx - sx, yy - sy) < 0.12 * r)
                cost = float(np.mean(np.abs(left[zone]))) if zone.any() else np.inf
                if best is None or cost < best[0]:
                    best = (cost, float(ang), g)
        _, ang, g = best
        res = res - g
        total += g
        found.append(ang)
    return total, found


def limb_profile(rho: np.ndarray, darkening: float, shape: float = 1.5) -> np.ndarray:
    """Brightness across the disc, like a vignette: 1 in the middle, ``1 - darkening`` at the limb.

    It falls gradually all the way from the centre: 1 - darkening x rho^shape (rho = distance
    from the centre in lunar radii). ``shape`` 1.5 starts gently in the middle and darkens
    steadily outwards (half way out it is already 21% down at darkening 0.6); higher values
    keep the middle brighter for longer, values near 1 make a straight-line fall.

    The glare removal takes away the disc's overall brightness along with the glare,
    leaving it evenly lit; this gives back a brighter middle and keeps the noisier last
    pixels by the limb dim instead of smoothing them away.
    """
    r = np.clip(np.asarray(rho, np.float64), 0.0, 1.0)
    d = float(np.clip(darkening, 0.0, 1.0))
    return (1.0 - d * r ** max(float(shape), 0.5)).astype(np.float32)


def adaptive_smooth(img: np.ndarray, good: np.ndarray, factor: np.ndarray, base: float, cap: float = 12.0) -> np.ndarray:
    """Gaussian smoothing of ``base`` px, widened to ``base`` x ``factor`` (up to ``cap``) pixel by pixel.

    Only good pixels are averaged, so the edge is not darkened.
    """
    base = max(base, 0.3)
    target = np.clip(base * factor, base, cap)
    levels = [base] + [s for s in (2.0, 3.0, 4.5, 7.0, 10.0, cap) if s > base * 1.2]
    g = good.astype(np.float32)
    src = np.where(good, img, 0.0).astype(np.float32)
    stack = []
    for s in levels:
        stack.append(ndimage.gaussian_filter(src, s) / np.maximum(ndimage.gaussian_filter(g, s), 1e-6))
    stack = np.stack(stack)
    pos = np.interp(np.log(target), np.log(levels), np.arange(len(levels)))
    i0 = np.floor(pos).astype(int)
    i1 = np.minimum(i0 + 1, len(levels) - 1)
    f = (pos - i0).astype(np.float32)
    out = np.take_along_axis(stack, i0[None], 0)[0] * (1 - f) + np.take_along_axis(stack, i1[None], 0)[0] * f
    return np.where(good, out, 0.0).astype(np.float32)


def extend_to_limb(img: np.ndarray, good: np.ndarray, cx: float, cy: float, r_in: float, r_out: float,
                   depth: float = 8.0, along: float = 16.0, blend: float = 6.0) -> np.ndarray:
    """Fill from ``r_in`` out to ``r_out`` with the brightness just inside, followed round the limb.

    The stack stays a few pixels inside the limb (where the frames' limbs and the
    chromosphere would leak in), and the last pixels it has are mostly noise; the
    corona layers only start just outside the limb. Without a fill the gap shows as
    a dark ring; copying single noisy pixels outwards shows as dark teeth. So the
    fill is the median of a ``depth`` px ring just inside ``r_in``, in bins round
    the limb about ``along`` px long (smoothed between bins), and it is blended in
    over ``blend`` px so there is no seam.
    """
    h, w = img.shape
    yy, xx = np.mgrid[0:h, 0:w]
    rr = np.hypot(xx - cx, yy - cy)
    phi = np.arctan2(yy - cy, xx - cx)
    out = np.where(good, img, 0.0).astype(np.float32)
    band = (rr >= r_in - blend) & (rr <= r_out + 1.0)
    if not band.any():
        return out
    nb = max(16, int(round(2 * math.pi * r_in / max(along, 1.0))))
    ring = good & (rr < r_in) & (rr >= r_in - depth)
    if ring.sum() < nb:
        ring = good & (rr < r_in) & (rr >= r_in - 3 * depth)
    if not ring.any():
        return out
    bins = ((phi + math.pi) / (2 * math.pi) * nb).astype(int) % nb
    prof = np.full(nb, np.nan)
    vals = img[ring]
    bi = bins[ring]
    order = np.argsort(bi, kind="stable")
    bi, vals = bi[order], vals[order]
    starts = np.searchsorted(bi, np.arange(nb))
    ends = np.searchsorted(bi, np.arange(nb), side="right")
    for k in range(nb):
        if ends[k] > starts[k]:
            prof[k] = np.median(vals[starts[k]:ends[k]])
    ok = np.isfinite(prof)
    idx = np.arange(nb)
    prof = np.interp(idx, idx[ok], prof[ok], period=nb)
    prof = ndimage.gaussian_filter1d(prof, 1.0, mode="wrap")
    pos = (phi[band] + math.pi) / (2 * math.pi) * nb - 0.5
    fill = ndimage.map_coordinates(prof, [pos], order=1, mode="grid-wrap")
    t = np.clip((rr[band] - (r_in - blend)) / max(blend, 1e-6), 0.0, 1.0)
    t = 0.5 - 0.5 * np.cos(math.pi * t)
    inner = np.where(good[band], out[band], fill)
    out[band] = (inner * (1 - t) + fill * t).astype(np.float32)
    return out
    # Sample a little inside the edge, from a copy smoothed with only the good pixels counted.
    g = good.astype(np.float32)
    num = ndimage.gaussian_filter(out, 1.0)
    den = ndimage.gaussian_filter(g, 1.0)
    filled = np.where(den > 0.05, num / np.maximum(den, 1e-6), 0.0)
    ys, xs = np.nonzero(band)
    f = (r_in - 1.5) / np.maximum(rr[ys, xs], 1e-6)
    sx = cx + (xs - cx) * f
    sy = cy + (ys - cy) * f
    out[ys, xs] = ndimage.map_coordinates(filled, [sy, sx], order=1, mode="nearest")
    return out


LIMB_FALLOFF = ((0.0035, 0.7), (0.009, 1.5), (0.023, 2.5), (0.057, 4.0))  # lunar radii, px floor


def limb_falloff(res: np.ndarray, good: np.ndarray, cx: float, cy: float, r: float, d0: float = 3.0,
                 bin_px: float = 20.0, smooth_bins: float = 2.0) -> np.ndarray:
    """Glare that fades inward from the limb, fitted round the limb in short arcs.

    Glare from just outside the limb can only get fainter going inward; a mare near the
    limb is a dip and highlands a bump, which no such fall-off can make. So in each arc
    (``bin_px`` along the limb) the profile with depth is fitted with a sum of decaying
    exponentials whose strengths all have the same sign (non-negative least squares,
    both signs tried), and only that is taken away. Strengths are smoothed between arcs.
    """
    from scipy import optimize

    h, w = res.shape
    reach = min(5.0 * max(max(f * r, floor) for f, floor in LIMB_FALLOFF), 0.3 * r)
    scales = [sc for sc in (max(f * r, floor) for f, floor in LIMB_FALLOFF) if sc <= reach / 3] or [reach / 3]
    n = max(64, int(round(2 * math.pi * r)))
    step = 0.5
    ds = np.arange(d0, reach + step, step)
    t = np.arange(n) / n * 2 * math.pi - math.pi
    rad = r - ds[:, None]
    ys = cy + rad * np.sin(t)[None, :]
    xs = cx + rad * np.cos(t)[None, :]
    cover = ndimage.map_coordinates(good.astype(np.float32), [ys, xs], order=1)
    polar = ndimage.map_coordinates(np.where(good, res, 0.0), [ys, xs], order=1) / np.maximum(cover, 1e-6)
    polar[cover < 0.5] = np.nan
    nb = max(24, int(round(n / bin_px)))
    edges = np.linspace(0, n, nb + 1).astype(int)
    E = np.stack([np.exp(-(ds - d0) / sc) for sc in scales], axis=1)
    amps = np.zeros((nb, len(scales)))
    for b in range(nb):
        blk = polar[:, edges[b] : edges[b + 1]]
        rows = np.isfinite(blk).any(axis=1)
        if rows.sum() < 10:
            continue
        prof = np.nanmedian(blk[rows], axis=1)
        sc = float(np.max(np.abs(prof))) + 1e-30
        best = None
        for sign in (1.0, -1.0):
            try:
                x, rn = optimize.nnls(E[rows], sign * prof / sc, maxiter=2000)
            except RuntimeError:
                continue
            if best is None or rn < best[1]:
                best = (sign * x * sc, rn)
        if best is not None:
            amps[b] = best[0]
    if smooth_bins > 0:
        amps = ndimage.gaussian_filter1d(amps, smooth_bins, axis=0, mode="wrap")
    yy, xx = np.mgrid[0:h, 0:w]
    dist = r - np.hypot(xx - cx, yy - cy)
    sel = good & (dist < reach)
    out = np.zeros((h, w), np.float32)
    if not sel.any():
        return out
    pos = (np.arctan2(yy[sel] - cy, xx[sel] - cx) + math.pi) / (2 * math.pi) * nb - 0.5
    dd = np.clip(dist[sel] - d0, 0.0, None)
    for k, sc in enumerate(scales):
        a = ndimage.map_coordinates(amps[:, k], [pos], order=1, mode="grid-wrap")
        out[sel] += (a * np.exp(-dd / sc)).astype(np.float32)
    return out


def remove_glare(img: np.ndarray, good: np.ndarray, cx: float, cy: float, r: float, p: "EarthshineParams",
                 prior: Optional[np.ndarray] = None):
    """The lunar surface left after taking away the glare: returns (detail, glare).

    ``prior``: what the lunar surface is expected to add (from an albedo map). It is taken
    away only while the glare is *estimated*, so the estimate does not mistake maria for
    glare; the detail returned is always ``img`` minus the glare, i.e. your data.
    """
    est = img if prior is None else np.where(good, img - prior, 0.0).astype(np.float32)
    glare = glare_surface(est, good, cx, cy, r, p.glare_order, limb=p.limb_glare)
    if p.limb_glare:
        if p.limb_shape:
            glare = glare + limb_falloff(np.where(good, est - glare, 0.0).astype(np.float32), good, cx, cy, r,
                                         d0=p.margin_px)
        glare = glare * limb_background(est, glare, good, cx, cy, r)
    return np.where(good, img - glare, 0.0).astype(np.float32), glare


def remove_glare_guided(img: np.ndarray, good: np.ndarray, cx: float, cy: float, r: float, p: "EarthshineParams",
                        prior: np.ndarray, iters: int = 2, sigma: float = 0.02, taper: tuple = (0.05, 0.12)):
    """:func:`remove_glare` with an albedo-map prior, corrected towards the data first.

    Wherever the data can tell (what is left after the glare estimate is not glare-shaped),
    the prior is pulled towards the data, a blurred step at a time; so a feature the map has
    but your Moon does not is mostly rejected, and the map only fills in what the data cannot
    separate from glare. The correction fades in from ``taper`` (fractions of the radius
    inside the limb): in the outer few % the residual is mostly glare, and feeding it back
    made a string of beads along the limb on real data. On simulations a false mare at 17%
    of the radius inside the limb comes through at about 17% of its strength in the map
    (times ``albedo_strength``), and not at all at 30% inside.
    """
    pr = prior.astype(np.float32).copy()  # correction kept off the outer few % (see the taper below)
    sig = max(1.0, sigma * r)
    h, w = img.shape
    yy, xx = np.mgrid[0:h, 0:w]
    depth = (r - np.hypot(xx - cx, yy - cy)) / r
    wt = np.ones((h, w), np.float32) if taper[1] <= taper[0] else \
        np.clip((depth - taper[0]) / (taper[1] - taper[0]), 0.0, 1.0).astype(np.float32)
    for _ in range(iters):
        _, glare = remove_glare(img, good, cx, cy, r, p, pr)
        resid = np.where(good, img - pr - glare, 0.0).astype(np.float32)
        num = ndimage.gaussian_filter(resid, sig)
        den = ndimage.gaussian_filter(good.astype(np.float32), sig)
        pr = np.where(good, pr + wt * num / np.maximum(den, 1e-6), 0.0).astype(np.float32)
    return remove_glare(img, good, cx, cy, r, p, pr)


def albedo_prior(amap: np.ndarray, detail: np.ndarray, good: np.ndarray, cx: float, cy: float, r: float,
                 p: "EarthshineParams", orientation=None):
    """The albedo map turned to the image and scaled to ``detail`` (see moonmap.guide).

    Returns (prior or None, orientation, note): None when the map does not match well enough.
    """
    from .moonmap import guide

    tmpl, o, k = guide(amap, detail, good, cx, cy, r, orientation)
    if o.correlation < p.albedo_min_match or k <= 0:
        return None, o, (f"{o.describe()}: too poor a match (below {p.albedo_min_match:.2f}), map not used")
    return (p.albedo_strength * tmpl).astype(np.float32), o, o.describe()


CHANNELS = {"all": (1.0, 1.0, 1.0), "green": (0.0, 1.0, 0.0), "green+blue": (0.0, 1.0, 1.0),
            "blue": (0.0, 0.0, 1.0)}


def channel_image(rgb: np.ndarray, name: str) -> np.ndarray:
    w = CHANNELS[name]
    return (sum(wi * rgb[i] for i, wi in enumerate(w) if wi) / sum(w)).astype(np.float32)


def choose_channel(rgb: np.ndarray, good: np.ndarray, cx: float, cy: float, r: float, p: "EarthshineParams"):
    """The colour channels to make the earthshine from, judged on this stack.

    The glare that changes quickly round the limb is largely red (hydrogen light from the
    chromosphere and prominences), and on a colour camera the green channel has twice the
    pixels, so green (or green + blue) usually beats the average of all three. Each candidate
    is scored by how strong the glare pattern 3-6% of the radius inside the limb (12-24 px at 400 mm)
    is against the lunar contrast in the middle; within 5% of the best, the least noisy wins.
    Returns (name, scores) with scores[name] = (glare, noise), both relative to the lunar contrast.
    """
    h, w = good.shape
    yy, xx = np.mgrid[0:h, 0:w]
    rr = np.hypot(xx - cx, yy - cy)
    phi = np.arctan2(yy - cy, xx - cx)
    d_in = r - rr
    band = good & (d_in > max(0.03 * r, p.margin_px + 1)) & (d_in < max(0.06 * r, p.margin_px + 7))
    body = good & (rr < r * 0.5)
    quick = EarthshineParams(glare_order=p.glare_order, limb_shape=False, margin_px=p.margin_px)
    scores = {}
    for name in ("all", "green", "green+blue"):
        img = channel_image(rgb, name)
        plain = np.where(good, img - glare_surface(img, good, cx, cy, r, p.glare_order, limb=True), 0.0)
        det, _ = remove_glare(img, good, cx, cy, r, quick)
        big = ndimage.gaussian_filter(det, max(2.0, 0.014 * r))
        surface = float(np.percentile(big[body], 95) - np.percentile(big[body], 5)) + 1e-30
        secs = [np.median(plain[band & (phi >= t) & (phi < t + np.radians(5))])
                for t in np.radians(np.arange(-180, 180, 5)) if (band & (phi >= t) & (phi < t + np.radians(5))).any()]
        noise = float(np.std((det - ndimage.gaussian_filter(det, 1.0))[body]))
        glare = float(np.std(secs)) / surface if len(secs) > 3 else float("inf")
        scores[name] = (glare, noise / surface)
    best = min(v[0] for v in scores.values())
    if not np.isfinite(best):
        return "all", scores
    name = min((k for k, v in scores.items() if v[0] <= 1.05 * best), key=lambda k: scores[k][1])
    return name, scores


def process_disc(lum: np.ndarray, weight: np.ndarray, good: np.ndarray, cx: float, cy: float, moon_r: float,
                 p: EarthshineParams, albedo: Optional[np.ndarray] = None, extras: Optional[dict] = None):
    """From the Moon-aligned stack to the picture: glare removed, smoothed by how certain it is.

    Returns (detail, shown, snr, keep_from, notes): ``keep_from`` is the depth inside the
    limb (px) from which the picture is kept; nearer the limb it is carried out from there.
    """
    notes = []
    h, w = lum.shape
    yy, xx = np.mgrid[0:h, 0:w]
    rr = np.hypot(xx - cx, yy - cy)
    detail, glare = remove_glare(lum, good, cx, cy, moon_r, p)
    orientation = None
    if albedo is not None and p.albedo_strength > 0:
        prior, orientation, note = albedo_prior(albedo, detail, good, cx, cy, moon_r, p)
        notes.append(note)
        if prior is not None:
            detail, glare = remove_glare_guided(lum, good, cx, cy, moon_r, p, prior)
        else:
            orientation = None
    if p.limb_glare and p.source_glare:
        extra, angles = source_glare(detail, good, cx, cy, moon_r, p.margin_px)
        if angles:
            detail = np.where(good, detail - extra, 0.0).astype(np.float32)
            glare = glare + extra
            notes.append("glare round a bright prominence taken away at " +
                         ", ".join(f"{(a + 360) % 360:.0f}" for a in angles) + " deg")
    # Noise from the stack weights (inverse variances up to one factor), measured on fine scales.
    gm = good.astype(np.float32)
    local = ndimage.gaussian_filter(detail, 1.0) / np.maximum(ndimage.gaussian_filter(gm, 1.0), 1e-6)
    resid = np.where(good, detail - local, 0.0)
    scale = 1.4826 * np.median(np.abs(resid[good] * np.sqrt(weight[good]))) / 0.96
    pix_noise = np.where(good, scale / np.sqrt(np.maximum(weight, 1e-30)), 0.0)
    noise = float(np.median(pix_noise[good]))
    body = good & (rr < moon_r * 0.5)
    noise_body = float(np.median(pix_noise[body])) if body.sum() > 100 else noise
    # How noisy each ring is, measured on the picture itself (the scatter of neighbouring pixels),
    # as well as from the stack weights: the weights alone can understate it near the limb.
    ri = np.clip((moon_r - rr).astype(int), 0, None)
    scatter = np.abs(resid)
    nring = int(ri[good].max()) + 1 if good.any() else 1
    ring_med = np.asarray(ndimage.median(scatter, np.where(good, ri, -1), index=np.arange(nring)), float)
    ring_med = np.nan_to_num(ring_med, nan=0.0)
    body_scatter = float(np.median(scatter[body])) if body.sum() > 100 else float(np.median(scatter[good]))
    ring_ratio = ring_med / max(body_scatter, 1e-30)
    weight_ratio = pix_noise / max(noise_body, 1e-30)
    factor = np.maximum(weight_ratio, ring_ratio[np.clip(ri, 0, nring - 1)])
    # A little more smoothing where the noise is higher (near the limb the long exposures are
    # saturated), up to ``edge_smooth`` px; diffraction spikes and other real structure are left alone.
    shown = (adaptive_smooth(detail, good, factor, p.smooth, cap=max(p.edge_smooth, p.smooth))
             if p.smooth > 0 else detail)
    # Nearer the limb than where the noise is still ~10x that of the middle, fill the edge from
    # just inside instead: only noise, and glare shaped by the bumpy real limb, is left in those
    # last pixels. At least the last 2.5% of the radius (11 px at 400 mm on a full-frame camera).
    keep_from = max(p.margin_px, 0.025 * moon_r)
    for d in range(int(math.ceil(p.margin_px)), min(nring, int(moon_r * 0.2))):
        if ring_ratio[d] > 0 and ring_ratio[d] < 10.0 and np.median(weight_ratio[good & (ri == d)]) < 10.0:
            keep_from = max(float(d), keep_from)
            break
    big = ndimage.gaussian_filter(detail, max(3.0, moon_r / 40))
    snr = float((np.percentile(big[good], 95) - np.percentile(big[good], 5)) / (noise / max(3.0, moon_r / 40) + 1e-30))
    if snr < 3:
        notes.append("very little lunar detail above the noise; the earthshine layer will look mostly like noise")

    if extras is not None:
        extras["glare"] = glare
    return detail, shown, snr, keep_from, notes


def stack_earthshine(
    infos: Sequence[FrameInfo],
    load: Callable[[FrameInfo], RawMosaic],
    calibs: dict,
    matches: dict,
    moon_xy: tuple[float, float],
    moon_r: float,
    canvas: tuple[int, int, int, int],
    params: Optional[EarthshineParams] = None,
    out_dir=None,
    name: str = "earthshine",
    image_crop: float = 1.0,
    progress: Optional[Say] = None,
    fmt: str = "tif",
    white: Optional[Sequence[float]] = None,
) -> tuple[EarthshineResult, np.ndarray, np.ndarray]:
    """Stack the frames on the Moon. ``matches``: frame index -> Match (gain, offset) from the HDR stack.

    ``moon_xy``/``moon_r``: the Moon in the HDR stack's Moon frame (reference coordinates).
    ``canvas``: the HDR stack's output region (x0, y0, w, h), so the layer lines up with it.
    Returns (result, linear luminance over the Moon crop, glare-removed detail).
    """
    say = progress or (lambda _m: None)
    p = params or EarthshineParams()
    infos = [f for f in infos if f.index in matches]
    if not infos:
        raise ValueError("no frames with a measured brightness; run the stack first")
    gmax = max(matches[f.index].gain for f in infos)
    use = [f for f in infos if matches[f.index].gain >= gmax * 2.0 ** (-p.stops)]
    use.sort(key=lambda f: -matches[f.index].gain)
    cx0, cy0 = moon_xy
    half = int(math.ceil(moon_r * 1.15)) + 4
    X0, Y0 = int(round(cx0)) - half, int(round(cy0)) - half
    crop = (X0, Y0, 2 * half, 2 * half)
    cx, cy = cx0 - X0, cy0 - Y0
    yy, xx = np.mgrid[0 : 2 * half, 0 : 2 * half]
    rr = np.hypot(xx - cx, yy - cy)
    inside = rr < moon_r - p.margin_px  # used for fitting the glare and the surface
    stacked = rr < moon_r + max(p.limb_reach, 0.0) + 2  # stacked: also the real limb just outside

    total = np.zeros((2 * half, 2 * half), np.float64)
    total_c = np.zeros((3, 2 * half, 2 * half), np.float64)
    weight = np.zeros_like(total)
    shifts = {}
    for f in use:
        m = matches[f.index]
        mat = moon_aligned(f.matrix, (f.moon_x, f.moon_y), moon_xy)
        shifts[f.index] = (cx0 - f.moon_x, cy0 - f.moon_y)
        mos = load(f)
        bx, by, bw, bh = source_box(mat, crop)
        fr = region(mos, bx, by, bw, bh, calibs.get(f.index), image_crop, hot_pixels=p.hot_pixels)
        del mos
        vc = [warp_region(fr.rgb[..., c], mat, crop, (fr.x0, fr.y0), order=3) for c in range(3)]
        v = (vc[0] + vc[1] + vc[2]) / 3.0
        vmax = warp_region(fr.rgb.max(axis=2), mat, crop, (fr.x0, fr.y0), order=1)
        sat = warp_region(fr.saturated.astype(np.float32), mat, crop, (fr.x0, fr.y0), order=1, cval=1.0)
        taper = np.clip((p.saturation_level - np.nan_to_num(vmax, nan=1.0)) / p.saturation_taper, 0.0, 1.0)
        w = (m.gain**2) / (np.clip(np.nan_to_num(v, nan=0.0), 0, None) + p.noise_floor) * taper * (sat < 0.01)
        w = np.where(np.isfinite(v) & stacked, w, 0.0)
        total += np.nan_to_num((v - m.offset) / m.gain) * w
        for c in range(3):
            total_c[c] += np.nan_to_num((vc[c] - m.offset) / m.gain) * w
        weight += w
        say(f"earthshine: {f.file_name} (gain {m.gain:.3g}, Moon shifted {math.hypot(*shifts[f.index]):.1f} px)")
    lum = np.where(weight > 0, total / np.maximum(weight, 1e-30), 0.0).astype(np.float32)
    rgb = np.where(weight > 0, total_c / np.maximum(weight, 1e-30), 0.0).astype(np.float32)
    good = inside & (weight > 0)
    notes = []
    if good.sum() < 100:
        raise ValueError("no usable pixels inside the lunar disc")

    albedo = None
    if p.albedo_map and p.albedo_strength > 0:
        from .moonmap import load_map

        try:
            albedo = load_map(p.albedo_map)
        except (OSError, ValueError) as exc:
            notes.append(f"albedo map not used: {exc}")
    chan = p.channel
    if chan == "auto":
        chan, scores = choose_channel(rgb, good, cx, cy, moon_r, p)
        say("earthshine channels: " + ", ".join(f"{k} glare {v[0]:.1f} noise {v[1]:.2f}" for k, v in scores.items())
            + f" -> {chan}")
    if chan not in CHANNELS:
        raise ValueError(f"unknown channel {p.channel!r}; choose from auto, {', '.join(CHANNELS)}")
    notes.append(f"made from the {chan} channel{'s' if chan in ('all', 'green+blue') else ''}")
    img = lum if chan == "all" else channel_image(rgb, chan)
    extras: dict = {}
    detail, shown, snr, keep_from, more = process_disc(img, weight, good, cx, cy, moon_r, p, albedo=albedo,
                                                       extras=extras)
    notes += more

    outputs = {}
    if out_dir is not None:
        outputs = _write(lum, detail, shown, good, weight, crop, canvas, (cx, cy), moon_r, p, out_dir, name, fmt,
                         keep_from, rgb, img=img, glare=extras.get("glare"), white=white,
                         tint=(surface_tint(rgb, detail, chan, good, cx, cy, moon_r, p, white, notes)
                               if p.colour and p.saturation > 0 else None))
    return EarthshineResult([f.index for f in use], shifts, crop, snr, outputs, notes), lum, detail


def _ring_floor(values: np.ndarray, mask: np.ndarray, phi: np.ndarray, nb: int = 72) -> np.ndarray:
    """Median of ``values`` in ``mask`` per angle bin round the Moon, smoothed; as a function of angle."""
    bins = ((phi + math.pi) / (2 * math.pi) * nb).astype(int) % nb
    prof = np.full(nb, np.nan)
    for k in range(nb):
        m = mask & (bins == k)
        if m.sum() > 3:
            prof[k] = np.median(values[m])
    ok = np.isfinite(prof)
    if not ok.any():
        return np.zeros_like(values)
    idx = np.arange(nb)
    prof = ndimage.gaussian_filter1d(np.interp(idx, idx[ok], prof[ok], period=nb), 1.5, mode="wrap")
    return ndimage.map_coordinates(prof, [(phi + math.pi) / (2 * math.pi) * nb - 0.5], order=1, mode="grid-wrap")


def soft_shoulder(x: np.ndarray, knee: float = 0.85) -> np.ndarray:
    """0..1 like a clip, but above ``knee`` it rolls off smoothly towards 1 instead of a hard cut."""
    x = np.clip(np.asarray(x, np.float32), 0.0, None)
    k = 1.0 - knee
    return np.where(x > knee, knee + k * (1.0 - np.exp(-(x - knee) / k)), x).astype(np.float32)


def surface_tint(rgb: np.ndarray, detail: np.ndarray, chan: str, good: np.ndarray, cx: float, cy: float,
                 r: float, p: "EarthshineParams", white: Optional[Sequence[float]], notes: Optional[list] = None):
    """The earthshine's own colour (R, G, B, mean 1), after the corona's white balance.

    Measured from how strongly the lunar pattern (maria against highlands) shows in each
    channel once its glare is taken away: the same albedo pattern lit by earthlight, so the
    ratios are the earthlight's colour. Returns None when a channel does not show it well.
    """
    wb = np.asarray(white if white is not None else (1.0, 1.0, 1.0), float)
    h, w = good.shape
    yy, xx = np.mgrid[0:h, 0:w]
    body = good & (np.hypot(xx - cx, yy - cy) < 0.75 * r)
    quick = EarthshineParams(glare_order=p.glare_order, limb_shape=False, margin_px=p.margin_px)
    sig = max(3.0, 0.015 * r)
    ref = ndimage.gaussian_filter(detail, sig)[body]
    ref = ref - ref.mean()
    slopes = []
    for c in range(3):
        d, _ = remove_glare(rgb[c], good, cx, cy, r, quick)
        v = ndimage.gaussian_filter(d, sig)[body]
        v = v - v.mean()
        corr = float(np.sum(v * ref) / (np.sqrt(np.sum(v * v) * np.sum(ref * ref)) + 1e-30))
        if corr < 0.5:
            if notes is not None:
                notes.append("earthshine colour not measured (a channel does not show the lunar pattern); grey")
            return None
        slopes.append(float(np.sum(v * ref) / (np.sum(ref * ref) + 1e-30)))
    t = np.asarray(slopes) * wb
    if np.any(t <= 0):
        return None
    t = t / t.mean()
    if notes is not None:
        notes.append("earthshine colour (R, G, B): " + ", ".join(f"{x:.2f}" for x in t))
    return t.astype(np.float32)


def _limb_colour(rgb: np.ndarray, weight: np.ndarray, white: Optional[Sequence[float]], saturation: float,
                 strength: np.ndarray) -> np.ndarray:
    """Colour (R, G, B ratios, mean 1) of the light in the stack, smoothed; grey where faint."""
    wb = np.asarray(white if white is not None else (1.0, 1.0, 1.0), np.float32)
    have = (weight > 0).astype(np.float32)
    den = ndimage.gaussian_filter(have, 2.0)
    ch = [ndimage.gaussian_filter(np.where(weight > 0, rgb[c] * wb[c], 0.0), 2.0) / np.maximum(den, 1e-6)
          for c in range(3)]
    m = (ch[0] + ch[1] + ch[2]) / 3.0
    col = np.stack([np.clip(np.nan_to_num(c / np.where(m > 0, m, np.inf), nan=1.0), 0.3, 3.0) for c in ch], axis=-1)
    col = np.where((m > 0)[..., None], col, 1.0)
    fade = np.clip(strength / 0.15, 0.0, 1.0) * saturation  # colour only where the glow carries light
    col = 1.0 + (col - 1.0) * fade[..., None]
    return (col / np.maximum(col.mean(axis=2, keepdims=True), 1e-6)).astype(np.float32)


def compose_layer(shown: np.ndarray, good: np.ndarray, weight: np.ndarray, cx: float, cy: float, moon_r: float,
                  p: "EarthshineParams", keep_from: Optional[float], linear: Optional[np.ndarray] = None,
                  glare: Optional[np.ndarray] = None, seed: int = 1, rgb: Optional[np.ndarray] = None,
                  white: Optional[Sequence[float]] = None, tint: Optional[np.ndarray] = None) -> np.ndarray:
    """The earthshine layer as a picture (0..1) over the Moon crop.

    * The lunar surface (``shown``) is stretched on the body of the disc, filled out over the
      noisy last pixels from just inside (with grain like the rest, ``edge_grain``), and given
      the vignette (``limb_darkening``).
    * A glow: ``keep_glare`` of the glare that was taken away, fading inward from the limb (what
      the eye sees at totality), compressed so it cannot clip.
    * The real limb: in the last few pixels and ``limb_reach`` px beyond, the stack itself (the
      Moon-aligned frames, linear) is shown, compressed the same way and at full strength, so
      the edge has the Moon's own shape and softness and runs into the corona, instead of a
      drawn circle. ``limb_soften`` px of extra Gaussian softening on top.
    Without ``linear``/``glare`` (or with ``limb_reach`` 0) the old cut at the circle is used.

    With ``rgb`` (the stack per channel) the result is in colour: the surface in the
    earthshine's own ``tint``, and the glow and limb in their measured colour (after the corona's
    ``white`` balance, ``saturation`` as for the corona), so a prominence at the limb stays red.
    """
    h, w = shown.shape
    yy, xx = np.mgrid[0:h, 0:w]
    rr = np.hypot(xx - cx, yy - cy)
    phi = np.arctan2(yy - cy, xx - cx)
    body = good & (rr < moon_r * 0.9)
    lo, hi = np.percentile(shown[body if body.sum() > 100 else good], [0.5, 99.5])
    span = float(hi - lo) or 1.0
    xs = (shown - lo) / span
    surf = soft_shoulder(xs)
    # Light far above the brightest surface is glare left over (by a bright prominence): it goes
    # into the glow below, on its gentle logarithmic scale, not into the surface as a flat patch.
    excess = np.where(good, np.clip(xs - 1.0, 0.0, None) * span, 0.0)
    # Under such glare the surface is not known: fill it smoothly from around (no flat patch).
    over = ndimage.binary_dilation(good & (xs > 1.15), iterations=2)
    if over.any():
        ok = good & ~over
        sig = 6.0
        fill = ndimage.gaussian_filter(np.where(ok, surf, 0.0), sig) / np.maximum(
            ndimage.gaussian_filter(ok.astype(np.float32), sig), 1e-6)
        wgt = ndimage.gaussian_filter(over.astype(np.float32), 1.5)
        surf = np.where(good, surf * (1 - wgt) + np.clip(fill, 0, 1) * wgt, surf).astype(np.float32)
    keep_from = p.margin_px if keep_from is None else keep_from
    r_in = moon_r - keep_from
    shown_good = good & (rr < r_in)
    real_limb = linear is not None and glare is not None and p.limb_reach > 0
    edge = moon_r + (p.limb_reach if real_limb else p.limb_px)
    surf = extend_to_limb(surf, shown_good, cx, cy, r_in, min(edge, moon_r + p.limb_px))
    if p.edge_grain and keep_from > p.margin_px:
        # Grain like the surface just inside the fill, so the filled band is not smoother than the rest.
        ring = shown_good & (rr > r_in - 12)
        hp = surf - ndimage.gaussian_filter(surf, 2.0)
        grain = float(np.std(hp[ring])) if ring.sum() > 50 else 0.0
        band = (rr >= r_in) & (rr <= moon_r + p.limb_px)
        noise = ndimage.gaussian_filter(np.random.default_rng(seed).normal(0, 1, (h, w)), max(p.smooth, 0.5))
        noise *= grain / (float(np.std(noise)) + 1e-12)
        t = np.clip((rr - (r_in - 3)) / 3, 0, 1)
        surf = np.where(band | (t > 0) & (rr < r_in), np.clip(surf + noise * t, 0, 1), surf)
    rho = np.minimum(rr / moon_r, 1.0)
    surf = surf * limb_profile(rho, p.limb_darkening, p.vignette_shape) * p.brightness
    if not real_limb:
        fe = max(p.feather_px, 0.5)
        alpha = np.clip((edge + fe / 2 - rr) / fe, 0.0, 1.0)
        return (surf * alpha).astype(np.float32)

    # Glow and real limb, both in units of the surface's stretch.
    floor = _ring_floor(glare, good & (rr > 0.66 * moon_r) & (rr < 0.74 * moon_r), phi)
    g_in = np.where(good, np.clip(glare - floor, 0.0, None), 0.0) + ndimage.gaussian_filter(excess, 2.0)
    have = weight > 0
    num = ndimage.gaussian_filter(np.where(have, linear, 0.0), 1.0)
    den = ndimage.gaussian_filter(have.astype(np.float32), 1.0)
    lin = np.where(den > 0.2, num / np.maximum(den, 1e-6), np.nan)
    g_out = np.clip(lin - floor, 0.0, None)
    # From the model glare to the data itself across the fill band (data noise is calmed by the
    # compression: it only matters where the light is already many times the surface).
    t = np.clip((rr - (r_in - 4)) / 6.0, 0.0, 1.0)
    t = t * t * (3 - 2 * t)
    g = np.where(np.isfinite(g_out), g_in * (1 - t) + np.nan_to_num(g_out) * t, g_in)
    # Where the stack has nothing just outside (every frame saturated), keep the light just inside.
    lost = ~np.isfinite(lin) & (rr < edge) & (rr > r_in)
    if lost.any():
        g = np.where(lost, extend_to_limb(np.where(rr < r_in, g, 0.0), rr < r_in, cx, cy, r_in, edge), g)
    # Logarithmic, scaled so the corona just outside the limb comes to 1: the glare inside stays
    # well below it, so the step at the limb (where the data jumps) shows, with the Moon's own shape.
    g0 = span / max(p.brightness, 1e-6)
    ring_out = np.isfinite(lin) & (rr > moon_r + 1.5) & (rr < edge - 1)
    top = float(np.median(g_out[ring_out])) if ring_out.sum() > 20 else float(np.percentile(g[good], 99.9))
    glow = np.log1p(np.clip(g, 0, None) / g0) / np.log1p(max(top, g0) / g0)
    glow = np.clip(glow, 0.0, 1.0)
    # Strength: keep_glare inside, rising to full over the last few px so the real limb meets the corona.
    k = p.keep_glare + (1.0 - p.keep_glare) * np.clip((rr - (moon_r - 3)) / 4.0, 0.0, 1.0)
    gl = k * glow
    if rgb is not None and p.saturation > 0:
        col = _limb_colour(rgb, weight, white, p.saturation, gl)
        ts = np.ones(3, np.float32) if tint is None else 1.0 + (np.asarray(tint, np.float32) - 1.0) * p.saturation
        out = 1.0 - (1.0 - surf[..., None] * ts) * (1.0 - gl[..., None] * col)
        out = np.where((rr < edge)[..., None], out, 0.0)
    else:
        out = 1.0 - (1.0 - surf) * (1.0 - gl)
        out = np.where(rr < edge, out, 0.0)
    if p.limb_soften > 0:
        sig = (p.limb_soften, p.limb_soften, 0) if out.ndim == 3 else p.limb_soften
        soft = ndimage.gaussian_filter(out, sig)
        tb = np.clip(1.0 - np.abs(rr - moon_r) / (p.limb_reach + 3 * p.limb_soften), 0.0, 1.0)
        if out.ndim == 3:
            tb = tb[..., None]
        out = out * (1 - tb) + soft * tb
    # The stack ends limb_reach px out: fade over the last 2 px. Where this overlaps the corona
    # layers (past their black disc), the final composite keeps the brighter of the two.
    fade = np.clip((edge - rr) / 2.0, 0.0, 1.0)
    if out.ndim == 3:
        fade = fade[..., None]
    return np.clip(out * fade, 0.0, 1.0).astype(np.float32)


def _write(lum, detail, shown, good, weight, crop, canvas, centre, moon_r, p, out_dir, name, fmt="tif",
           keep_from: Optional[float] = None, rgb: Optional[np.ndarray] = None,
           img: Optional[np.ndarray] = None, glare: Optional[np.ndarray] = None,
           white: Optional[Sequence[float]] = None, tint: Optional[np.ndarray] = None) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    X0, Y0, w, h = crop
    cx, cy = centre
    header = {"OBJECT": "Earthshine", "CROPX0": X0, "CROPY0": Y0, "MOONX": cx, "MOONY": cy, "MOONR": moon_r,
              "COMMENT": "Moon-aligned linear stack in units of the reference exposure"}
    lin_path = out / f"{name}_earthshine.fit"
    write_fits(lin_path, lum, header)
    write_fits(out / f"{name}_earthshine_weight.fit", weight.astype(np.float32), header)
    if rgb is not None:
        write_fits(out / f"{name}_earthshine_rgb.fit", rgb, header)
    img = compose_layer(shown, good, weight, cx, cy, moon_r, p, keep_from, linear=img, glare=glare,
                        rgb=rgb if p.colour else None, white=white, tint=tint)
    moon_path = None
    if glare is not None:
        surface, vis = moon_surface(shown, good, glare, cx, cy, moon_r, p,
                                    p.margin_px if keep_from is None else keep_from, tint)
        moon_path = out / f"{name}_moon_surface.fit"
        write_fits(moon_path, np.concatenate([np.moveaxis(surface, -1, 0), vis[None]], axis=0).astype(np.float32),
                   {**header, "COMMENT": "planes: surface R, G, B (display 0..1), visibility"})
    crop_tif = write_image(with_format(out / f"{name}_earthshine_crop.tif", fmt), img)
    # The layer on the HDR stack's canvas: black outside the disc, so it drops into the corona layers.
    cx0, cy0, cw, ch = canvas
    layer = np.zeros((ch, cw) + img.shape[2:], np.float32)
    ox, oy = X0 - cx0, Y0 - cy0
    sy0, sx0 = max(0, -oy), max(0, -ox)
    ty0, tx0 = max(0, oy), max(0, ox)
    hh = min(h - sy0, ch - ty0)
    ww = min(w - sx0, cw - tx0)
    if hh > 0 and ww > 0:
        layer[ty0 : ty0 + hh, tx0 : tx0 + ww] = img[sy0 : sy0 + hh, sx0 : sx0 + ww]
    layer_tif = write_image(with_format(out / f"{name}_earthshine.tif", fmt), layer)
    info = out / f"{name}_earthshine.json"
    moon = [cx + X0 - canvas[0], cy + Y0 - canvas[1], moon_r]  # in the layer's (canvas) pixels
    info.write_text(json.dumps({"crop": crop, "canvas": canvas, "moon": moon, "params": asdict(p)}, indent=2),
                    encoding="utf-8")
    outputs = {"earthshine": str(layer_tif), "crop": str(crop_tif), "linear": str(lin_path), "info": str(info)}
    if rgb is not None:
        outputs["rgb"] = str(out / f"{name}_earthshine_rgb.fit")
        outputs["weight"] = str(out / f"{name}_earthshine_weight.fit")
    if moon_path is not None:
        outputs["moon_surface"] = str(moon_path)
    return outputs


from .imageout import with_format, write_image  # noqa: E402


# ---------------------------------------------------------------------------
# Compositing into the corona image
# ---------------------------------------------------------------------------

MODES = ("screen", "lighten", "add")


def composite(base: np.ndarray, earth: np.ndarray, strength: float = 1.0, mode: str = "screen",
              moon: Optional[tuple] = None, limb_px: float = 1.5, reach: float = 6.0) -> np.ndarray:
    """Put the earthshine layer into the corona image, as Photoshop's blend modes would.

    Both are display images in 0..1 of the same size. The corona layers are
    black inside the lunar disc and the earthshine layer is black outside it,
    so any of the modes leaves the corona untouched and fills the disc;
    they differ only where both have light (the soft edge of the limb).
    ``strength`` scales the earthshine first (like layer opacity, but it can
    go above 1 to brighten a faint Moon).

    With ``moon`` (x, y, radius in these pixels) the limb is matched: round the limb the
    earthshine layer's light at the edge is scaled to the corona's brightness just outside
    (it was made from the linear data, the corona from the enhanced layers, so they differ),
    and across the corona layers' disc edge the brighter of the two is kept. So the Moon's
    real outline runs into the corona with no bright or dark ring.
    """
    e = np.clip(earth * strength, 0.0, 1.0)
    if moon is not None:
        e = match_limb(base, e, moon, limb_px, reach=reach)
    if mode == "screen":
        out = 1.0 - (1.0 - base) * (1.0 - e)
    elif mode == "lighten":
        out = np.maximum(base, e)
    elif mode == "add":
        out = base + e
    else:
        raise ValueError(f"unknown mode {mode!r}; choose from {', '.join(MODES)}")
    if moon is not None:
        cx, cy, r = moon
        h, w = base.shape[:2]
        y0, y1 = max(0, int(cy - r - 12)), min(h, int(cy + r + 13))
        x0, x1 = max(0, int(cx - r - 12)), min(w, int(cx + r + 13))
        if y1 > y0 and x1 > x0:
            yy, xx = np.mgrid[y0:y1, x0:x1]
            rr = np.hypot(xx - cx, yy - cy)
            band = (rr > r + limb_px - 1.0) & (rr < r + max(reach, limb_px + 2.0) + 1.0)
            sub = out[y0:y1, x0:x1]
            sub[band] = np.maximum(base[y0:y1, x0:x1][band], e[y0:y1, x0:x1][band])
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def match_limb(base: np.ndarray, e: np.ndarray, moon: tuple, limb_px: float = 1.5, nb: int = 180,
               reach: float = 6.0) -> np.ndarray:
    """Scale the earthshine layer's light at the limb to the corona just outside, angle by angle.

    The brightest of the layer's limb light (out to ``reach``) is brought to the corona layers'
    level a little further out (past their own dimmer edge), so the limb rises smoothly into the
    corona with no bright ring."""
    cx, cy, r = moon
    h, w = base.shape[:2]
    m = int(limb_px + reach + 14)
    y0, y1 = max(0, int(cy - r - m)), min(h, int(cy + r + m + 1))
    x0, x1 = max(0, int(cx - r - m)), min(w, int(cx + r + m + 1))
    if y1 <= y0 or x1 <= x0:
        return e
    yy, xx = np.mgrid[y0:y1, x0:x1]
    rr = np.hypot(xx - cx, yy - cy)
    phi = np.arctan2(yy - cy, xx - cx)
    bl = base[y0:y1, x0:x1]
    el = e[y0:y1, x0:x1]
    bl1 = bl.mean(axis=2) if bl.ndim == 3 else bl
    el1 = el.mean(axis=2) if el.ndim == 3 else el
    bins = ((phi + math.pi) / (2 * math.pi) * nb).astype(int) % nb
    outside = (rr > r + limb_px + 4.0) & (rr < r + limb_px + 9.0)
    edge = (rr > r - 2.0) & (rr < r + reach)
    f = np.full(nb, np.nan)
    for k in range(nb):
        mo, me = outside & (bins == k), edge & (bins == k)
        if mo.sum() > 2 and me.sum() > 2:
            pk = float(np.percentile(el1[me], 95))
            if pk > 1e-3:
                f[k] = float(np.median(bl1[mo])) / pk
    ok = np.isfinite(f)
    if not ok.any():
        return e
    idx = np.arange(nb)
    f = np.clip(np.interp(idx, idx[ok], f[ok], period=nb), 0.2, 1.5)
    f = ndimage.gaussian_filter1d(f, 2.0, mode="wrap")
    fmap = ndimage.map_coordinates(f, [(phi + math.pi) / (2 * math.pi) * nb - 0.5], order=1, mode="grid-wrap")
    t = np.clip((rr - (r - 8.0)) / 8.0, 0.0, 1.0)
    t = t * t * (3 - 2 * t)
    scale = 1.0 + (fmap - 1.0) * t
    out = e.copy()
    out[y0:y1, x0:x1] = np.clip(el * (scale[..., None] if el.ndim == 3 else scale), 0.0, 1.0)
    return out


# ---------------------------------------------------------------------------
# The Moon in the stack: one stretch for Moon and corona
# ---------------------------------------------------------------------------


def moon_surface(shown: np.ndarray, good: np.ndarray, glare: np.ndarray, cx: float, cy: float, moon_r: float,
                 p: "EarthshineParams", keep_from: float, tint: Optional[np.ndarray] = None):
    """The seas and highlands as a picture to add on top of the stretched Moon, and where to show them.

    Returns (surface (h, w, 3) in 0..1, visibility (h, w) in 0..1). The surface is the glare-free
    detail, stretched on the body of the disc, with the vignette, ``brightness`` and the
    earthshine's ``tint``. The visibility is 1 where the earthshine itself dominates and falls
    smoothly to 0 towards the limb as the glare grows (1 / (1 + (glare / surface)^2)): near the
    limb the stack's own light, stretched with the corona, is all there is, so nothing needs
    cutting, filling or softening there.
    """
    h, w = shown.shape
    yy, xx = np.mgrid[0:h, 0:w]
    rr = np.hypot(xx - cx, yy - cy)
    phi = np.arctan2(yy - cy, xx - cx)
    body = good & (rr < moon_r * 0.9)
    lo, hi = np.percentile(shown[body if body.sum() > 100 else good], [0.5, 99.5])
    span = float(hi - lo) or 1.0
    xs = (shown - lo) / span
    surf = soft_shoulder(xs)
    over = ndimage.binary_dilation(good & (xs > 1.15), iterations=2)  # glare by a bright prominence
    if over.any():
        ok = good & ~over
        fill = ndimage.gaussian_filter(np.where(ok, surf, 0.0), 6.0) / np.maximum(
            ndimage.gaussian_filter(ok.astype(np.float32), 6.0), 1e-6)
        wgt = ndimage.gaussian_filter(over.astype(np.float32), 1.5)
        surf = np.where(good, surf * (1 - wgt) + np.clip(fill, 0, 1) * wgt, surf)
    # Over the noisy last pixels, carry the surface out to the limb (with matching grain).
    r_in = moon_r - keep_from
    known = good & (rr < r_in)
    surf = extend_to_limb(np.where(known, surf, 0.0), known, cx, cy, r_in, moon_r + 1.0)
    if p.edge_grain:
        ring = known & (rr > r_in - 12)
        hp = surf - ndimage.gaussian_filter(surf, 2.0)
        grain = float(np.std(hp[ring])) if ring.sum() > 50 else 0.0
        noise = ndimage.gaussian_filter(np.random.default_rng(1).normal(0, 1, (h, w)), max(p.smooth, 0.5))
        noise *= grain / (float(np.std(noise)) + 1e-12)
        t = np.clip((rr - (r_in - 3)) / 3, 0, 1)
        surf = np.where(rr < moon_r + 1.0, np.clip(surf + noise * t, 0, 1), surf)
    rho = np.minimum(rr / moon_r, 1.0)
    surf = surf * limb_profile(rho, p.limb_darkening, p.vignette_shape) * p.brightness
    # Shown over the whole disc, ending with a 1 px soft edge at the limb; added with screen, so
    # near the limb, where the stretched light of the Moon itself is brighter, that light wins.
    vis = np.clip(moon_r + 0.5 - rr, 0.0, 1.0)
    ts = np.ones(3, np.float32) if tint is None else 1.0 + (np.asarray(tint, np.float32) - 1.0) * p.saturation
    surface = np.clip(surf[..., None] * ts, 0.0, 1.0).astype(np.float32)
    return surface, vis.astype(np.float32)


def register_moon(hdr_lum: np.ndarray, es_lum: np.ndarray, es_good: np.ndarray, cx: float, cy: float, r: float,
                  search: float = 3.0) -> tuple[float, float]:
    """Shift (dx, dy) that puts the earthshine stack onto the HDR stack, from the limb itself.

    Both hold the same light across the limb (the steep jump from Moon to corona), so the
    shift is the one that best matches log brightness in a ring round the limb.
    """
    h, w = es_lum.shape
    yy, xx = np.mgrid[0:h, 0:w]
    rr = np.hypot(xx - cx, yy - cy)
    ring = (rr > r - 6) & (rr < r + 4) & es_good & (hdr_lum > 0)
    if ring.sum() < 200:
        return 0.0, 0.0
    a = np.log(np.maximum(hdr_lum, 1e-9))
    b = np.log(np.maximum(np.where(es_good, es_lum, 1e-9), 1e-9))
    ys, xs = np.nonzero(ring)
    best = (np.inf, 0.0, 0.0)
    for step, centre in ((0.5, (0.0, 0.0)), (0.125, None)):
        c0 = centre if centre is not None else (best[1], best[2])
        rng = np.arange(-search, search + 1e-9, step) if centre is not None else np.arange(-0.5, 0.5 + 1e-9, step)
        for dy in rng + c0[1]:
            for dx in rng + c0[0]:
                v = ndimage.map_coordinates(b, [ys - dy, xs - dx], order=1, mode="nearest")
                d = a[ys, xs] - v
                cost = float(np.median(np.abs(d - np.median(d))))
                if cost < best[0]:
                    best = (cost, float(dx), float(dy))
    return best[1], best[2]


def merge_moon(hdr: np.ndarray, weight: np.ndarray, header: dict, es_rgb: np.ndarray, es_weight: np.ndarray,
               origin: tuple[int, int], es_centre: tuple[float, float], margin_px: float = 1.5,
               say: Optional[Say] = None):
    """Put the Moon-aligned stack (linear, same units) into the HDR stack inside the lunar disc.

    ``origin``: where the earthshine crop starts in the HDR's pixels; ``es_centre``: the Moon's
    centre in the crop. The two are first registered on the limb. Inside the limb the earthshine
    stack replaces the stack's one-frame Moon, crossing over in the 2 px just outside the limb,
    where both hold the same light. Returns the MoonInput mask (the pixels to draw) and the shift applied.
    """
    from .enhance import MoonInput

    say = say or (lambda _m: None)
    mx, my, mr = float(header["MOONX"]), float(header["MOONY"]), float(header["MOONR"])
    ox, oy = origin
    hc, wc = es_rgb.shape[1:]
    h, w = hdr.shape[:2]
    ys = slice(max(oy, 0), min(oy + hc, h))
    xs = slice(max(ox, 0), min(ox + wc, w))
    sy = slice(ys.start - oy, ys.stop - oy)
    sx = slice(xs.start - ox, xs.stop - ox)
    cx, cy = mx - ox, my - oy  # the HDR's Moon in crop pixels
    es_lum = es_rgb.mean(axis=0)
    es_good = es_weight > 0
    hdr_lum = np.zeros((hc, wc), np.float32)
    hdr_lum[sy, sx] = hdr[ys, xs].mean(axis=2)
    dx, dy = register_moon(hdr_lum, es_lum, es_good, cx, cy, mr)
    dx0, dy0 = cx - es_centre[0], cy - es_centre[1]
    say(f"Moon: earthshine stack moved by {dx:+.2f}, {dy:+.2f} px onto the HDR stack (centres differed by "
        f"{dx0:+.2f}, {dy0:+.2f})")
    if abs(dx) + abs(dy) > 1e-3:
        es_rgb = np.stack([ndimage.shift(c, (dy, dx), order=1, mode="nearest") for c in es_rgb])
        es_good = ndimage.shift(es_good.astype(np.float32), (dy, dx), order=1, mode="constant") > 0.99
    yy, xx = np.mgrid[0:hc, 0:wc]
    rr = np.hypot(xx - cx, yy - cy)
    # Trust the earthshine stack only where enough of its frames went in: by a bright prominence
    # the long frames are saturated, and the few that are left are near full well (colours off).
    ew = np.where(es_good, np.asarray(es_weight, np.float32), 0.0)
    if abs(dx) + abs(dy) > 1e-3:
        ew = ndimage.shift(ew, (dy, dx), order=1, mode="constant")
    ri = np.clip(rr.astype(int), 0, None)
    med = ndimage.median(ew, labels=np.where(ew > 0, ri + 1, 0), index=np.arange(1, ri.max() + 2))
    med = np.nan_to_num(np.asarray(med, np.float32))
    es_ok = es_good & (ew >= 0.1 * med[ri])
    # Fully the earthshine stack out to the limb (the stack's own Moon is a single noisy frame),
    # crossing over to the HDR stack's corona in the next 2 px, where both hold the same light.
    t = np.clip((mr + 2.0 - rr) / 2.0, 0.0, 1.0)
    wm = np.where(es_ok, t * t * (3 - 2 * t), 0.0).astype(np.float32)
    sub = hdr[ys, xs]
    wsub = weight[ys, xs]
    # The two stacks' weights are in different units. Both are inverse variances up to a factor
    # (noise = c / sqrt(weight)); each c is measured from its own pixel scatter, so the Moon's
    # pixels get weights that tell the stretch their real noise. (With the stack's one-frame
    # weights, or none, the colour of everything inside the limb was faded to grey as "noise".)
    from .enhance import noise_constant

    rs = rr[sy, sx]
    c_hdr = noise_constant(sub.mean(axis=2), wsub, (wsub > 0) & (rs >= mr + 3.0) & (rs < mr + 0.5 * mr))
    c_es = noise_constant(es_rgb.mean(axis=0), ew, es_ok & (rr < mr - 5.0))
    if c_hdr and c_es:
        w_es = ew * (c_hdr / c_es) ** 2
        say(f"Moon: earthshine noise matched to the stack's (weights x {(c_hdr / c_es) ** 2:.3g})")
    else:  # too few pixels to measure: as reliable as the corona just outside the limb
        near = (wsub > 0) & (rs >= mr + 3.0) & (rs < mr + 10.0)
        ref = float(np.median(wsub[near])) if near.any() else 1.0
        w_es = np.where(ew > 0, ref, 0.0)
    # The HDR stack's own pixels inside the Moon (out to 3 px past the limb) come from one
    # frame, saturated by a bright prominence: not used. Its corona further out is.
    had = (weight[ys, xs] > 0) & np.isfinite(sub).all(axis=2) & (rr[sy, sx] >= mr + 3.0)
    wm = wm.copy()
    wm[sy, sx] = np.where(~had & es_ok[sy, sx], 1.0, wm[sy, sx])
    sub[:] = np.where(had[..., None], sub, 0.0) * (1 - wm[sy, sx, None]) \
        + np.moveaxis(es_rgb, 0, -1)[sy, sx] * wm[sy, sx, None]
    # Pixels that neither stack has reliably (every frame saturated there, or covered):
    # filled from their neighbours, widening until they are all filled.
    # Weight of the mix (inverse variance of (1-wm) * hdr + wm * es).
    a = wm[sy, sx]
    wh = np.where(had, wsub, 0.0)
    we = np.where(es_ok, w_es, 0.0)[sy, sx]
    with np.errstate(divide="ignore", invalid="ignore"):
        var = np.where(a < 1, (1 - a) ** 2 / wh, 0.0) + np.where(a > 0, a**2 / we, 0.0)
        wmix = np.nan_to_num(1.0 / var, nan=0.0, posinf=0.0)
    have = had | (a > 0.5)
    wnew = np.where(have, wmix, 0.0).astype(np.float32)
    band = rs < mr + 3.0
    hole = band & ~have
    n_hole = int(hole.sum())
    if n_hole:
        m = have.astype(np.float32)
        left = hole.copy()
        for sig in (1.5, 3.0, 6.0, 12.0, 24.0):
            den = ndimage.gaussian_filter(m, sig)
            ok = left & (den > 0.05)
            if ok.any():
                for c in range(3):
                    num = ndimage.gaussian_filter(np.where(have, sub[..., c], 0.0), sig)
                    sub[..., c] = np.where(ok, num / np.maximum(den, 1e-6), sub[..., c])
                # As certain as the light it came from (an average of it).
                wn = ndimage.gaussian_filter(np.where(have, wnew, 0.0), sig)
                wnew = np.where(ok, wn / np.maximum(den, 1e-6), wnew)
                left &= ~ok
            if not left.any():
                break
        say(f"Moon: {n_hole} pixels by the limb that neither stack has reliably filled from around")
    wsub[:] = np.where((a > 0) | hole, np.maximum(wnew, 1e-12), wsub)
    mask = np.zeros((h, w), bool)
    inside = rr < mr + margin_px + 1.0  # overlaps the corona's own region: no gap between them
    mask[ys, xs] = inside[sy, sx]
    return MoonInput(mask=mask, x0=ox, y0=oy), (dx, dy)
