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
brighter highlands.

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
    feather_px: float = 2.0  # soft edge of the disc in the layer
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


def glare_surface(img: np.ndarray, inside: np.ndarray, cx: float, cy: float, r: float, order: int,
                  iters: int = 6) -> np.ndarray:
    """Robust 2-D polynomial fitted over the disc (the smooth glare); returned over the whole crop."""
    h, w = img.shape
    yy, xx = np.mgrid[0:h, 0:w]
    u = (xx - cx) / r
    v = (yy - cy) / r
    terms = [(i, j) for i in range(order + 1) for j in range(order + 1 - i)]
    A = np.stack([(u[inside] ** i) * (v[inside] ** j) for i, j in terms], axis=1)
    y = img[inside].astype(np.float64)
    wts = np.ones_like(y)
    coef = np.zeros(len(terms))
    for _ in range(iters):
        sw = np.sqrt(wts)
        coef, *_ = np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)
        res = y - A @ coef
        s = 1.4826 * np.median(np.abs(res)) + 1e-30
        t = res / (4.685 * s)
        wts = np.where(np.abs(t) < 1, (1 - t * t) ** 2, 0.0)
    out = np.zeros((h, w))
    for c, (i, j) in zip(coef, terms):
        out += c * (u**i) * (v**j)
    return out.astype(np.float32)


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

    glare = glare_surface(lum, good, cx, cy, moon_r, p.glare_order)
    detail = np.where(good, lum - glare, 0.0).astype(np.float32)
    # Noise from the stack weights (inverse variances up to one factor), measured on fine scales.
    resid = detail - ndimage.gaussian_filter(detail, 1.0)
    scale = 1.4826 * np.median(np.abs(resid[good] * np.sqrt(weight[good]))) / 0.96
    noise = float(np.median(scale / np.sqrt(weight[good])))
    shown = ndimage.gaussian_filter(detail, p.smooth) if p.smooth > 0 else detail
    big = ndimage.gaussian_filter(detail, max(3.0, moon_r / 40))
    snr = float((np.percentile(big[good], 95) - np.percentile(big[good], 5)) / (noise / max(3.0, moon_r / 40) + 1e-30))
    if snr < 3:
        notes.append("very little lunar detail above the noise; the earthshine layer will look mostly like noise")

    outputs = {}
    if out_dir is not None:
        outputs = _write(lum, detail, shown, good, weight, crop, canvas, (cx, cy), moon_r, p, out_dir, name, fmt)
    return EarthshineResult([f.index for f in use], shifts, crop, snr, outputs, notes), lum, detail


def _write(lum, detail, shown, good, weight, crop, canvas, centre, moon_r, p, out_dir, name, fmt="tif") -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    X0, Y0, w, h = crop
    cx, cy = centre
    header = {"OBJECT": "Earthshine", "CROPX0": X0, "CROPY0": Y0, "MOONX": cx, "MOONY": cy, "MOONR": moon_r,
              "COMMENT": "Moon-aligned linear stack in units of the reference exposure"}
    lin_path = out / f"{name}_earthshine.fit"
    write_fits(lin_path, lum, header)
    lo, hi = np.percentile(shown[good], [0.5, 99.5])
    img = np.clip((shown - lo) / ((hi - lo) or 1.0), 0.0, 1.0) * p.brightness
    yy, xx = np.mgrid[0:h, 0:w]
    rr = np.hypot(xx - cx, yy - cy)
    edge = moon_r - p.margin_px
    alpha = np.clip((edge - rr) / max(p.feather_px, 0.5), 0.0, 1.0)
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
