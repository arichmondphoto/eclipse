"""Phase 3: merge the aligned frames into one linear HDR image of the corona.

Following Druckmüller, Rušin & Minarovjech (2006) and the descriptions by
Vuorinen and astrofalls:

1. **Brightness matching.** Nominal exposure times are not exact, so each
   frame's real brightness relation to the reference frame is measured:
   ``value_i = g_i * L + c_i`` where L is the radiance in units of the
   reference frame. ``g_i`` and ``c_i`` come from robust linear fits between
   neighbouring exposures (on a reduced copy of each aligned frame, using
   pixels well exposed in both and away from the Moon), chained to the
   reference.
2. **Weighted average.** Every frame is calibrated, demosaiced at full
   resolution and resampled onto the reference grid. Each pixel's radiance
   estimate ``(value - c_i) / g_i`` is weighted by its inverse variance,
   ``g_i^2 / (value + read-noise floor)``, tapered to zero towards
   saturation.
3. **Moon awareness.** The Moon moves across the corona during totality, so
   in every frame the pixels covered by (or right next to) that frame's Moon
   get zero weight. Inside the Moon as it was in the chosen Moon frame the
   result is that frame's own data, so the lunar limb is sharp and in one
   place.

Everything is linear; the output radiance is in units of the reference
frame's exposure, in camera RGB (colour calibration is a later step).
"""

from __future__ import annotations

import json
import math
import os
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
from scipy import ndimage

from .fits import write_fits
from .rawio import RawMosaic, region

Say = Callable[[str], None]


@dataclass
class StackParams:
    crop_radii: Optional[float] = None  # None = full frame; else a square this many solar radii from the centre
    moon_margin_px: float = 3.0  # drop pixels this far outside each frame's lunar limb, full-res px
    saturation_taper: float = 0.15  # weights fall to zero over this range below the saturation level
    saturation_level: float = 0.95  # of full scale
    noise_floor: float = 0.002  # read-noise term in the weights, in full-scale units
    match_scale: int = 4  # brightness matching on images reduced by this factor
    match_range: tuple[float, float] = (0.003, 0.85)  # usable values for matching
    match_smooth: float = 2.0  # blur (reduced px) before matching, so seeing/shake differences don't bias gains
    match_strong: float = 0.02  # gains from pixels above this level in both frames
    hot_pixels: bool = True
    moon_frame: Optional[int] = None  # frame index whose Moon is kept; None = the alignment reference


@dataclass
class FrameInfo:
    index: int
    file_name: str
    nominal_ratio: float  # exposure relative to the reference, from EXIF
    matrix: list[list[float]]  # 2x3, frame full-res -> reference full-res
    moon_x: float  # Moon centre in reference coordinates
    moon_y: float
    moon_r: float


@dataclass
class Match:
    index: int
    gain: float  # g_i
    offset: float  # c_i
    nominal: float
    stops_vs_nominal: float
    pixels: int


@dataclass
class StackResult:
    crop: tuple[int, int, int, int]  # x0, y0, width, height in reference coordinates
    matches: list[Match]
    moon_frame: int
    sun: Optional[dict]
    outputs: dict
    notes: list[str] = field(default_factory=list)

    def describe(self) -> str:
        lines = [f"Output: {self.crop[2]} x {self.crop[3]} px from reference ({self.crop[0]}, {self.crop[1]})",
                 f"Moon from frame #{self.moon_frame}",
                 f"{'#':>3}  {'gain':>10}  {'offset':>9}  {'vs nominal':>10}  {'pixels':>7}"]
        for m in self.matches:
            lines.append(f"{m.index:>3}  {m.gain:10.5f}  {m.offset:+9.5f}  {m.stops_vs_nominal:+9.2f}EV  {m.pixels:7d}")
        for k, v in self.outputs.items():
            lines.append(f"{k}: {v}")
        lines += [f"Note: {n}" for n in self.notes]
        return "\n".join(lines)


# ---------------------------------------------------------------------------


def crop_box(sun_xy: tuple[float, float], sun_r: float, size_wh: tuple[int, int],
             radii: Optional[float] = None) -> tuple[int, int, int, int]:
    """Output region (x0, y0, w, h): the whole reference frame, or a square around the Sun inside it."""
    w, h = size_wh
    if radii is None:
        return 0, 0, int(w), int(h)
    s = int(min(2 * radii * sun_r, w, h))
    s -= s % 2
    x0 = int(round(sun_xy[0] - s / 2))
    y0 = int(round(sun_xy[1] - s / 2))
    return max(0, min(x0, w - s)), max(0, min(y0, h - s)), s, s


def _inverse(matrix) -> tuple[np.ndarray, np.ndarray]:
    M = np.vstack([np.asarray(matrix, float), [0, 0, 1]])
    Mi = np.linalg.inv(M)
    return Mi[:2, :2], Mi[:2, 2]


def source_box(matrix, crop: tuple[int, int, int, int], margin: int = 8) -> tuple[int, int, int, int]:
    """Region of the frame (x0, y0, w, h) that lands in the crop."""
    A, t = _inverse(matrix)
    x0, y0, cw, ch = crop
    corners = np.array([[x0, y0], [x0 + cw, y0], [x0, y0 + ch], [x0 + cw, y0 + ch]], float)
    p = corners @ A.T + t
    lo = np.floor(p.min(axis=0)) - margin
    hi = np.ceil(p.max(axis=0)) + margin
    return int(lo[0]), int(lo[1]), int(hi[0] - lo[0]), int(hi[1] - lo[1])


def warp_region(img: np.ndarray, matrix, crop: tuple[int, int, int, int], origin: tuple[int, int],
                order: int = 3, scale: int = 1, cval: float = np.nan) -> np.ndarray:
    """Resample a frame region onto the crop grid of the reference (optionally reduced by ``scale``)."""
    A, t = _inverse(matrix)
    X0, Y0, cw, ch = crop
    x0, y0 = origin
    shape = (ch // scale, cw // scale)
    # Output pixel (r, c) is reference point (X0 + (c + 0.5) * scale - 0.5, Y0 + ...).
    off_ref = np.array([X0 + (scale - 1) / 2.0, Y0 + (scale - 1) / 2.0])
    p0 = A @ off_ref + t  # frame point of output (0, 0)
    mat = np.array([[A[1, 1], A[1, 0]], [A[0, 1], A[0, 0]]]) * scale
    offset = np.array([p0[1] - y0, p0[0] - x0])
    if img.ndim == 2:
        return ndimage.affine_transform(img, mat, offset=offset, output_shape=shape, order=order,
                                        mode="constant", cval=cval, prefilter=order > 1)
    out = np.empty(shape + (img.shape[2],), np.float32)
    for ch in range(img.shape[2]):
        out[..., ch] = ndimage.affine_transform(img[..., ch], mat, offset=offset, output_shape=shape,
                                                order=order, mode="constant", cval=cval, prefilter=order > 1)
    return out


def moon_mask(crop: tuple[int, int, int, int], fi: FrameInfo, margin: float, scale: int = 1) -> np.ndarray:
    """True where this frame's Moon (plus margin) covers the crop."""
    X0, Y0, cw, ch = crop
    off = (scale - 1) / 2.0
    xx = X0 + np.arange(cw // scale)[None, :] * scale + off
    yy = Y0 + np.arange(ch // scale)[:, None] * scale + off
    return np.hypot(xx - fi.moon_x, yy - fi.moon_y) < fi.moon_r + margin


def moon_box(crop: tuple[int, int, int, int], fi: FrameInfo, margin: float) -> tuple[slice, slice]:
    """Array slices (rows, cols) of the crop that contain this frame's Moon plus margin."""
    X0, Y0, cw, ch = crop
    r = fi.moon_r + margin + 1
    c0 = max(0, int(math.floor(fi.moon_x - r)) - X0)
    c1 = min(cw, int(math.ceil(fi.moon_x + r)) - X0 + 1)
    r0 = max(0, int(math.floor(fi.moon_y - r)) - Y0)
    r1 = min(ch, int(math.ceil(fi.moon_y + r)) - Y0 + 1)
    return slice(r0, max(r0, r1)), slice(c0, max(c0, c1))


# ---------------------------------------------------------------------------
# Brightness matching
# ---------------------------------------------------------------------------


def robust_line(x: np.ndarray, y: np.ndarray, iters: int = 8) -> tuple[float, float, int]:
    """Fit y = a x + b, down-weighting outliers (Tukey biweight). Returns (a, b, points used)."""
    w = np.ones_like(x)
    a, b = 1.0, 0.0
    for _ in range(iters):
        sw = w.sum()
        if sw <= 0:
            break
        mx, my = (w * x).sum() / sw, (w * y).sum() / sw
        vx = (w * (x - mx) ** 2).sum()
        if vx <= 0:
            break
        a = (w * (x - mx) * (y - my)).sum() / vx
        b = my - a * mx
        r = y - (a * x + b)
        # Residuals relative to the signal: noise grows with the value.
        rel = r / (np.abs(y) + 1e-3)
        s = 1.4826 * np.median(np.abs(rel)) + 1e-9
        u = rel / (4.685 * s)
        w = np.where(np.abs(u) < 1, (1 - u**2) ** 2, 0.0)
    return float(a), float(b), int((w > 0).sum())


def pair_gain(vi: np.ndarray, vj: np.ndarray, strong: float = 0.02,
              value_range: tuple[float, float] = (0.003, 0.85)) -> tuple[float, float, int]:
    """Brightness relation vj = a * vi + b between two aligned frames.

    A least-squares slope is biased low whenever one frame is noisier or
    blurrier than the other (errors in the x values), and across a chain of
    30 brackets that adds up to several percent. So the slope comes from the
    median ratio of well-exposed pixels, which is symmetric in the two
    frames; the line fit only supplies the small offset. Pixels are chosen
    by their mean brightness across both frames, not by either frame alone,
    because cutting on one noisy frame biases the ratio.

    ``vi`` and ``vj`` are the pixels usable in both (unsaturated, off the Moon).
    """
    lo, hi = value_range
    sel = (vi > lo) & (vi < hi) & (vj > lo) & (vj < hi)
    if sel.sum() < 50:
        return float("nan"), float("nan"), int(sel.sum())
    a0, b, n = robust_line(vi[sel], vj[sel])
    if not (math.isfinite(a0) and a0 > 0):
        return a0, b, n
    u = 0.5 * (a0 * vi + vj)  # in frame j's units
    top = 0.9 * hi
    good = (u > strong) & (u / a0 > strong) & (u < top) & (u / a0 < top)
    if good.sum() < 50:
        return a0, b, n
    xi, xj, uu = vi[good], vj[good], u[good]
    a = float(np.exp(np.median(np.log(np.clip(xj - b, 1e-9, None) / xi))))
    w = np.ones_like(xi)
    for _ in range(4):
        # Ratio of (weighted) sums: linear in both frames, so noise does not bias it, and
        # sharpness differences only move light between pixels. Outliers (hot pixels, a
        # prominence that changed, residual Moon edge) are down-weighted symmetrically.
        r = (xj - b - a * xi) / uu
        sc = 1.4826 * np.median(np.abs(r)) + 1e-9
        t = r / (4.685 * sc)
        w = np.where(np.abs(t) < 1, (1 - t**2) ** 2, 0.0)
        a = float((w * (xj - b)).sum() / (w * xi).sum())
        b = float(np.median(vj[sel] - a * vi[sel]))
    return a, b, int((w > 0).sum())


def match_exposures(reduced: dict[int, tuple[np.ndarray, np.ndarray]], infos: dict[int, FrameInfo], ref: int,
                    params: StackParams, say: Say) -> dict[int, Match]:
    """Chain robust fits between neighbouring exposures to get g_i and c_i relative to ``ref``."""
    order = sorted(reduced, key=lambda i: infos[i].nominal_ratio)
    pos = {i: k for k, i in enumerate(order)}
    lo_v, hi_v = params.match_range
    gains = {ref: 1.0}
    offsets = {ref: 0.0}
    pixels = {ref: 0}

    def pair(i: int, j: int) -> tuple[float, float, int]:
        vi, oki = reduced[i]
        vj, okj = reduced[j]
        both = oki & okj
        if params.match_smooth > 0:
            # Blur both frames identically (same mask) to even out sharpness differences, and
            # keep only pixels whose blur kernel lies wholly inside the common mask.
            sm = params.match_smooth
            vi = ndimage.gaussian_filter(np.where(both, vi, 0.0), sm)
            vj = ndimage.gaussian_filter(np.where(both, vj, 0.0), sm)
            both = ~ndimage.binary_dilation(~both, iterations=int(math.ceil(3 * sm)))
        return pair_gain(vi[both].astype(np.float64), vj[both].astype(np.float64), params.match_strong,
                         params.match_range)

    k = pos[ref]
    for direction in (1, -1):
        prev = ref
        idx = k + direction
        while 0 <= idx < len(order):
            cur = order[idx]
            # Fit the new frame against the already-matched one: v_cur = a * v_prev + b.
            a, b, n = pair(prev, cur)
            if not (math.isfinite(a) and a > 0):
                nominal = infos[cur].nominal_ratio / infos[prev].nominal_ratio
                say(f"  frame {cur}: too little overlap with frame {prev}; using the nominal exposure ratio")
                a, b, n = nominal, 0.0, 0
            gains[cur] = a * gains[prev]
            offsets[cur] = a * offsets[prev] + b
            pixels[cur] = n
            prev = cur
            idx += direction
    out = {}
    for i in order:
        nom = infos[i].nominal_ratio
        out[i] = Match(i, gains[i], offsets[i], nom, math.log2(gains[i] / nom) if nom > 0 else float("nan"), pixels[i])
    return out


# ---------------------------------------------------------------------------
# Stacking
# ---------------------------------------------------------------------------


def stack_frames(
    infos: Sequence[FrameInfo],
    load: Callable[[FrameInfo], RawMosaic],
    calibs: dict,
    image_size: tuple[int, int],
    ref_index: int,
    sun: Optional[dict],
    params: Optional[StackParams] = None,
    out_dir: Optional[str | os.PathLike] = None,
    name: str = "stack",
    image_crop: float = 1.0,
    progress: Optional[Say] = None,
) -> tuple[StackResult, np.ndarray, np.ndarray]:
    """Run brightness matching and the weighted, Moon-aware average.

    Returns (result, hdr, weight): hdr is (h, w, 3) float32 radiance in units
    of the reference exposure; weight is (h, w) float32. The output covers the
    whole reference frame unless ``params.crop_radii`` asks for a square.
    A full D850 frame needs roughly 3 GB of memory at the peak.
    """
    say = progress or (lambda _m: None)
    params = params or StackParams()
    infos = list(infos)
    by_index = {f.index: f for f in infos}
    moon_frame = params.moon_frame if params.moon_frame is not None else ref_index
    if moon_frame not in by_index:
        raise ValueError(f"Moon frame #{moon_frame} is not among the frames being stacked")

    if sun:
        centre, sun_r = (sun["x"], sun["y"]), sun["radius_px"]
    else:
        ref = by_index[ref_index]
        centre, sun_r = (ref.moon_x, ref.moon_y), ref.moon_r / 1.04
    crop = crop_box(centre, sun_r, image_size, params.crop_radii)
    X0, Y0, cw, ch = crop
    say(f"output region {cw} x {ch} px at ({X0}, {Y0})")

    # Pass 1: reduced, aligned copies for brightness matching.
    sc = params.match_scale
    reduced: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    cache: dict[int, RawMosaic] = {}
    wb = None
    for f in infos:
        mos = load(f)
        wb = wb or mos.wb
        bx, by, bw, bh = source_box(f.matrix, crop)
        fr = region(mos, bx, by, bw, bh, calibs.get(f.index), image_crop, hot_pixels=False)
        lum = fr.rgb.mean(axis=2)
        v = warp_region(lum, f.matrix, crop, (fr.x0, fr.y0), order=1, scale=1)
        sat = warp_region(fr.saturated.astype(np.float32), f.matrix, crop, (fr.x0, fr.y0), order=1, scale=1, cval=1.0)
        ny, nx = ch // sc, cw // sc
        nan = ~np.isfinite(v[: ny * sc, : nx * sc])
        v = np.nan_to_num(v[: ny * sc, : nx * sc], nan=0.0).reshape(ny, sc, nx, sc).mean(axis=(1, 3))
        bad = ((sat[: ny * sc, : nx * sc] > 0.01) | nan).reshape(ny, sc, nx, sc).any(axis=(1, 3))
        ok = ~bad & ~moon_mask(crop, f, params.moon_margin_px + 2 * sc, sc)
        reduced[f.index] = (v.astype(np.float32), ok)
        if len(cache) < 2:
            cache[f.index] = mos  # keep a couple to save one read; the rest are reloaded
        say(f"matching: read {f.file_name}")
    matches = match_exposures(reduced, by_index, ref_index, params, say)
    reduced.clear()
    notes = []
    for m in matches.values():
        if m.index != ref_index and abs(m.stops_vs_nominal) > 0.3:
            notes.append(f"frame #{m.index} is {m.stops_vs_nominal:+.2f} stops from its nominal exposure")

    # Pass 2: full resolution.
    total = np.zeros((ch, cw, 3), np.float32)
    weight = np.zeros((ch, cw), np.float32)
    moon_rgb = None  # the Moon frame's data, kept only around its Moon
    moon_slices = moon_box(crop, by_index[moon_frame], params.moon_margin_px)
    # Where no frame has usable data (all saturated or behind the Moon), fall back to the
    # shortest uncovered exposure: process frames from longest to shortest so it ends up on top.
    fallback = np.zeros((ch, cw, 3), np.float32)
    have_fallback = np.zeros((ch, cw), bool)
    infos = sorted(infos, key=lambda f: -matches[f.index].gain)
    for f in infos:
        mos = cache.pop(f.index, None) or load(f)
        bx, by, bw, bh = source_box(f.matrix, crop)
        fr = region(mos, bx, by, bw, bh, calibs.get(f.index), image_crop, hot_pixels=params.hot_pixels)
        del mos
        v = warp_region(fr.rgb, f.matrix, crop, (fr.x0, fr.y0), order=3)
        sat = warp_region(fr.saturated.astype(np.float32), f.matrix, crop, (fr.x0, fr.y0), order=1, cval=1.0)
        hot = fr.hot_pixels
        del fr
        m = matches[f.index]
        with warnings.catch_warnings():  # NaN = outside this frame after alignment; weight 0 below
            warnings.simplefilter("ignore", RuntimeWarning)
            vmax = np.nan_to_num(np.nanmax(v, axis=2), nan=params.saturation_level)
            mean = np.nanmean(v, axis=2)
        taper = np.clip((params.saturation_level - vmax) / params.saturation_taper, 0.0, 1.0)
        w = (m.gain**2) / (np.clip(mean, 0, None) + params.noise_floor)
        del mean
        w = np.nan_to_num(w * taper * (sat < 0.01), nan=0.0).astype(np.float32)
        del vmax, taper, sat
        radiance = v  # converted in place to save memory on full frames
        radiance -= m.offset
        radiance /= m.gain
        covered = moon_mask(crop, f, params.moon_margin_px)
        if f.index == moon_frame:
            moon_rgb = np.nan_to_num(radiance[moon_slices], nan=0.0).astype(np.float32)
        w[covered] = 0.0
        valid = ~covered & np.isfinite(radiance).all(axis=2)
        np.nan_to_num(radiance, copy=False, nan=0.0)
        fallback[valid] = radiance[valid]
        have_fallback |= valid
        del covered, valid
        radiance *= w[..., None]
        total += radiance
        weight += w
        del radiance, v, w
        say(f"stacked {f.file_name}: gain {m.gain:.4f}, {hot} hot pixels fixed")

    hdr = total
    hdr /= np.maximum(weight, 1e-12)[..., None]
    del total
    # The Moon as it was in the Moon frame, and any pixel no frame covered.
    mf = by_index[moon_frame]
    inside = moon_mask(crop, mf, 0.0)
    empty = weight <= 0
    gap = empty & ~inside
    hdr[gap & have_fallback] = fallback[gap & have_fallback]
    hdr[gap & ~have_fallback] = 0.0
    del fallback
    if moon_rgb is not None:
        sub = hdr[moon_slices]
        sel = inside[moon_slices]
        sub[sel] = moon_rgb[sel]
    nodata = int((gap & ~have_fallback).sum())
    clipped = int((gap & have_fallback).sum())
    if clipped:
        notes.append(f"{clipped} pixel(s) outside the Moon were saturated or covered in every frame; "
                     "filled from the shortest exposure that shows them (may be clipped)")
    if nodata:
        notes.append(f"{nodata} pixel(s) near the frame edges are outside every aligned frame; set to zero")

    outputs = {}
    if out_dir is not None:
        outputs = write_outputs(hdr, weight, crop, matches, by_index, ref_index, moon_frame, sun, out_dir, name,
                                wb=wb)
    result = StackResult(crop, [matches[i] for i in sorted(matches)], moon_frame, sun, outputs, notes)
    return result, hdr, weight


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def stretch_preview(hdr: np.ndarray, wb: Optional[Sequence[float]] = None, strength: float = 300.0) -> np.ndarray:
    """Asinh-stretched 16-bit RGB for a quick look (not for further processing)."""
    img = hdr.astype(np.float32).copy()
    if wb is not None:
        img *= np.asarray(wb, np.float32)[None, None, :]
    bg = np.percentile(img, 5, axis=(0, 1))
    img -= bg[None, None, :]
    top = float(np.percentile(img, 99.9)) or 1.0
    x = np.clip(img / top, 0, None)
    y = np.arcsinh(strength * x) / np.arcsinh(strength)
    return (np.clip(y, 0, 1) * 65535).astype(np.uint16)


def write_outputs(hdr, weight, crop, matches, infos, ref_index, moon_frame, sun, out_dir, name, wb=None) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    X0, Y0 = crop[0], crop[1]
    header = {"OBJECT": "Solar corona", "BUNIT": "relative", "CROPX0": X0, "CROPY0": Y0,
              "REFFRAME": ref_index, "MOONFRM": moon_frame,
              "COMMENT": "Linear HDR stack in units of the reference exposure, camera RGB"}
    mf = infos[moon_frame]
    header.update({"MOONX": mf.moon_x - X0, "MOONY": mf.moon_y - Y0, "MOONR": mf.moon_r})
    if sun:
        header.update({"SUNX": sun["x"] - X0, "SUNY": sun["y"] - Y0, "SUNR": sun["radius_px"],
                       "PIXSCALE": sun["arcsec_per_px"], "NORTHANG": sun["rotation_deg"]})
    hdr_path = out / f"{name}_hdr.fit"
    write_fits(hdr_path, np.moveaxis(hdr, -1, 0), header)
    w_path = out / f"{name}_weight.fit"
    write_fits(w_path, weight, {"COMMENT": "Sum of stacking weights per pixel"})
    preview = stretch_preview(hdr, wb)
    prev_path = out / f"{name}_preview.tif"
    try:
        import tifffile

        tifffile.imwrite(prev_path, preview, photometric="rgb")
    except ImportError:
        from PIL import Image

        prev_path = out / f"{name}_preview.png"
        Image.fromarray((preview >> 8).astype(np.uint8)).save(prev_path)
    meta_path = out / f"{name}_stack.json"
    meta_path.write_text(json.dumps({"crop": crop, "matches": [asdict(m) for m in matches.values()],
                                     "reference": ref_index, "moon_frame": moon_frame, "sun": sun,
                                     "header": header}, indent=2), encoding="utf-8")
    return {"hdr": str(hdr_path), "weight": str(w_path), "preview": str(prev_path), "info": str(meta_path)}
