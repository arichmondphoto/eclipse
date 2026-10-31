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
    limb_glare: bool = True  # also model the steep glare just inside the limb
    limb_px: float = 1.5  # the layer reaches this far past the limb, to meet the corona layers' black disc
    feather_px: float = 1.0  # soft edge of the disc in the layer
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
                    depth: float = 0.275, width0: float = 0.014, growth: float = 1.0) -> np.ndarray:
    """What is left of the limb glare after :func:`glare_surface`, followed closely round the limb.

    Returns a factor (1 away from the limb) for the glare surface. The light just
    inside the limb comes from whatever is just outside it in each frame: inner
    corona, chromosphere and prominences, which change within a few degrees and
    are uncovered differently as the Moon moves. Here the ratio of the image to
    the glare surface (which varies far more slowly with depth than the glare
    itself) is resampled onto circles, and each circle is median-filtered round
    the limb. The window is a few degrees at the limb and widens with depth
    (``width0`` x radius + ``growth`` x depth, in px), because glare from
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
        win = int(width0_px + growth * dv) | 1
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


def extend_to_limb(img: np.ndarray, good: np.ndarray, cx: float, cy: float, r_in: float, r_out: float) -> np.ndarray:
    """Carry the values just inside ``r_in`` straight out to ``r_out`` (along each radius).

    The stack stays a few pixels inside the limb, where the frames' limbs and
    the chromosphere would leak in; the corona layers only start just outside
    it. Without this the gap between the two shows as a dark ring.
    """
    h, w = img.shape
    yy, xx = np.mgrid[0:h, 0:w]
    rr = np.hypot(xx - cx, yy - cy)
    band = (rr >= r_in) & (rr <= r_out + 1.0)
    out = np.where(good, img, 0.0).astype(np.float32)
    if not band.any():
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


def process_disc(lum: np.ndarray, weight: np.ndarray, good: np.ndarray, cx: float, cy: float, moon_r: float,
                 p: EarthshineParams):
    """From the Moon-aligned stack to the picture: glare removed, smoothed by how certain it is.

    Returns (detail, shown, snr, keep_from, notes): ``keep_from`` is the depth inside the
    limb (px) from which the picture is kept; nearer the limb it is carried out from there.
    """
    notes = []
    h, w = lum.shape
    yy, xx = np.mgrid[0:h, 0:w]
    rr = np.hypot(xx - cx, yy - cy)
    glare = glare_surface(lum, good, cx, cy, moon_r, p.glare_order, limb=p.limb_glare)
    detail = np.where(good, lum - glare, 0.0).astype(np.float32)
    if p.limb_glare:
        glare = glare * limb_background(lum, glare, good, cx, cy, moon_r)
        detail = np.where(good, lum - glare, 0.0).astype(np.float32)
    # Noise from the stack weights (inverse variances up to one factor), measured on fine scales.
    resid = detail - ndimage.gaussian_filter(detail, 1.0)
    scale = 1.4826 * np.median(np.abs(resid[good] * np.sqrt(weight[good]))) / 0.96
    pix_noise = np.where(good, scale / np.sqrt(np.maximum(weight, 1e-30)), 0.0)
    noise = float(np.median(pix_noise[good]))
    body = good & (rr < moon_r * 0.5)
    noise_body = float(np.median(pix_noise[body])) if body.sum() > 100 else noise
    # Smooth more where the picture is less certain: where the noise is higher (near the limb the
    # long exposures are saturated) and where much glare was taken away (by a bright prominence).
    floor = float(np.median(glare[body])) if body.sum() > 100 else float(np.median(glare[good]))
    excess = ndimage.gaussian_filter(np.clip((glare - floor) / max(abs(floor), 1e-12), 0, None), 4.0)
    factor = np.maximum(pix_noise / max(noise_body, 1e-30), (1.0 + excess) ** 1.5)
    shown = adaptive_smooth(detail, good, factor, p.smooth) if p.smooth > 0 else detail
    # Nearer the limb than where the noise is still ~10x that of the middle (more than the widest
    # smoothing can calm), carry the picture out to the limb instead: only noise is left there.
    rings = np.arange(int(p.margin_px), int(moon_r * 0.2))
    ri = np.clip((moon_r - rr).astype(int), 0, rings[-1] + 1) if len(rings) else None
    keep_from = p.margin_px
    if ri is not None:
        ratio = pix_noise / max(noise_body, 1e-30)
        for d in rings:
            m = good & (ri == d)
            if m.sum() > 20 and np.median(ratio[m]) < 10.0:
                keep_from = max(float(d), p.margin_px)
                break
    # Over the last few percent of the radius what is left of the glare is about as large as the
    # lunar detail, so the contrast there is eased down (to 30%) towards the limb.
    if p.limb_glare:
        mid = float(np.median(shown[good & (rr < moon_r * 0.9)]))
        c = np.clip((moon_r - rr - keep_from) / (0.06 * moon_r), 0.0, 1.0)
        c = 0.3 + 0.7 * (0.5 - 0.5 * np.cos(math.pi * c))
        shown = np.where(good, mid + (shown - mid) * c, 0.0).astype(np.float32)
    big = ndimage.gaussian_filter(detail, max(3.0, moon_r / 40))
    snr = float((np.percentile(big[good], 95) - np.percentile(big[good], 5)) / (noise / max(3.0, moon_r / 40) + 1e-30))
    if snr < 3:
        notes.append("very little lunar detail above the noise; the earthshine layer will look mostly like noise")

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
    inside = rr < moon_r - p.margin_px

    total = np.zeros((2 * half, 2 * half), np.float64)
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
        v = warp_region(fr.rgb.mean(axis=2), mat, crop, (fr.x0, fr.y0), order=3)
        vmax = warp_region(fr.rgb.max(axis=2), mat, crop, (fr.x0, fr.y0), order=1)
        sat = warp_region(fr.saturated.astype(np.float32), mat, crop, (fr.x0, fr.y0), order=1, cval=1.0)
        taper = np.clip((p.saturation_level - np.nan_to_num(vmax, nan=1.0)) / p.saturation_taper, 0.0, 1.0)
        w = (m.gain**2) / (np.clip(np.nan_to_num(v, nan=0.0), 0, None) + p.noise_floor) * taper * (sat < 0.01)
        w = np.where(np.isfinite(v) & inside, w, 0.0)
        total += np.nan_to_num((v - m.offset) / m.gain) * w
        weight += w
        say(f"earthshine: {f.file_name} (gain {m.gain:.3g}, Moon shifted {math.hypot(*shifts[f.index]):.1f} px)")
    lum = np.where(weight > 0, total / np.maximum(weight, 1e-30), 0.0).astype(np.float32)
    good = inside & (weight > 0)
    notes = []
    if good.sum() < 100:
        raise ValueError("no usable pixels inside the lunar disc")

    detail, shown, snr, keep_from, more = process_disc(lum, weight, good, cx, cy, moon_r, p)
    notes += more

    outputs = {}
    if out_dir is not None:
        outputs = _write(lum, detail, shown, good, weight, crop, canvas, (cx, cy), moon_r, p, out_dir, name, fmt,
                         keep_from)
    return EarthshineResult([f.index for f in use], shifts, crop, snr, outputs, notes), lum, detail


def _write(lum, detail, shown, good, weight, crop, canvas, centre, moon_r, p, out_dir, name, fmt="tif",
           keep_from: Optional[float] = None) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    X0, Y0, w, h = crop
    cx, cy = centre
    header = {"OBJECT": "Earthshine", "CROPX0": X0, "CROPY0": Y0, "MOONX": cx, "MOONY": cy, "MOONR": moon_r,
              "COMMENT": "Moon-aligned linear stack in units of the reference exposure"}
    lin_path = out / f"{name}_earthshine.fit"
    write_fits(lin_path, lum, header)
    write_fits(out / f"{name}_earthshine_weight.fit", weight.astype(np.float32), header)
    # Stretch on the body of the disc: the last few pixels by the limb are noisier
    # (photon noise of the glare there) and would otherwise set the range.
    yy, xx = np.mgrid[0:h, 0:w]
    body = good & (np.hypot(xx - cx, yy - cy) < moon_r * 0.9)
    lo, hi = np.percentile(shown[body if body.sum() > 100 else good], [0.5, 99.5])
    img = np.clip((shown - lo) / ((hi - lo) or 1.0), 0.0, 1.0) * p.brightness
    # Fill out to where the corona layers start (just outside the limb), then a short soft edge.
    edge = moon_r + p.limb_px
    r_in = moon_r - (p.margin_px if keep_from is None else keep_from)
    shown_good = good & (np.hypot(xx - cx, yy - cy) < r_in)
    img = extend_to_limb(img, shown_good, cx, cy, r_in, edge)
    yy, xx = np.mgrid[0:h, 0:w]
    rr = np.hypot(xx - cx, yy - cy)
    fe = max(p.feather_px, 0.5)
    alpha = np.clip((edge + fe / 2 - rr) / fe, 0.0, 1.0)
    img = (img * alpha).astype(np.float32)
    crop_tif = write_image(with_format(out / f"{name}_earthshine_crop.tif", fmt), img)
    # The layer on the HDR stack's canvas: black outside the disc, so it drops into the corona layers.
    cx0, cy0, cw, ch = canvas
    layer = np.zeros((ch, cw), np.float32)
    ox, oy = X0 - cx0, Y0 - cy0
    sy0, sx0 = max(0, -oy), max(0, -ox)
    ty0, tx0 = max(0, oy), max(0, ox)
    hh = min(h - sy0, ch - ty0)
    ww = min(w - sx0, cw - tx0)
    if hh > 0 and ww > 0:
        layer[ty0 : ty0 + hh, tx0 : tx0 + ww] = img[sy0 : sy0 + hh, sx0 : sx0 + ww]
    layer_tif = write_image(with_format(out / f"{name}_earthshine.tif", fmt), layer)
    info = out / f"{name}_earthshine.json"
    info.write_text(json.dumps({"crop": crop, "canvas": canvas, "params": asdict(p)}, indent=2), encoding="utf-8")
    return {"earthshine": str(layer_tif), "crop": str(crop_tif), "linear": str(lin_path), "info": str(info)}


from .imageout import with_format, write_image  # noqa: E402


# ---------------------------------------------------------------------------
# Compositing into the corona image
# ---------------------------------------------------------------------------

MODES = ("screen", "lighten", "add")


def composite(base: np.ndarray, earth: np.ndarray, strength: float = 1.0, mode: str = "screen") -> np.ndarray:
    """Put the earthshine layer into the corona image, as Photoshop's blend modes would.

    Both are display images in 0..1 of the same size. The corona layers are
    black inside the lunar disc and the earthshine layer is black outside it,
    so any of the modes leaves the corona untouched and fills the disc;
    they differ only where both have light (the soft edge of the limb).
    ``strength`` scales the earthshine first (like layer opacity, but it can
    go above 1 to brighten a faint Moon).
    """
    e = np.clip(earth * strength, 0.0, 1.0)
    if mode == "screen":
        out = 1.0 - (1.0 - base) * (1.0 - e)
    elif mode == "lighten":
        out = np.maximum(base, e)
    elif mode == "add":
        out = base + e
    else:
        raise ValueError(f"unknown mode {mode!r}; choose from {', '.join(MODES)}")
    return np.clip(out, 0.0, 1.0).astype(np.float32)
