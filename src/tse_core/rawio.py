"""Load frames as linear, calibrated images.

RAW files are read with rawpy (LibRaw) and reduced to 2x2 "superpixels"
(one R, two G and one B photosite each), which needs no demosaicing and
introduces no interpolation, at half the resolution. That is what the
alignment works on; transforms are scaled back to full resolution after.

Calibration applied here, matching the calibration plan:
  * black level subtracted per CFA channel (never clipped at zero, so
    the noise statistics stay honest);
  * scaled so 1.0 is the sensor's white level;
  * vignetting divided out per colour channel (own-flats or lensfun
    profile), when the plan gives one;
  * saturation flagged where any of the four photosites reaches the
    white level.

FITS files (for example frames Siril has already calibrated) are read as
they are; CFA FITS with a BAYERPAT keyword are reduced to superpixels too.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .calibration import FrameCalibration
from .demosaic import cosmetic_correct, demosaic_mhc
from .lensfun import vignetting_map

SATURATION_FRACTION = 0.98


@dataclass
class LoadedFrame:
    rgb: np.ndarray  # (h, w, 3) float32, 1.0 = full scale, vignetting corrected
    saturated: np.ndarray  # (h, w) bool
    binning: int  # sensor pixels per image pixel along each axis

    @property
    def luminance(self) -> np.ndarray:
        return (self.rgb[..., 0] + 2.0 * self.rgb[..., 1] + self.rgb[..., 2]) / 4.0


def load_frame(path: str, calib: Optional[FrameCalibration] = None, image_crop: float = 1.0) -> LoadedFrame:
    if path.lower().endswith((".fit", ".fits", ".fts")):
        return _load_fits(path, calib, image_crop)
    return _load_raw(path, calib, image_crop)


def _load_raw(path: str, calib: Optional[FrameCalibration], image_crop: float) -> LoadedFrame:
    try:
        import rawpy
    except ImportError as exc:
        raise RuntimeError("reading RAW files needs rawpy: pip install rawpy") from exc
    with rawpy.imread(path) as raw:
        bayer = raw.raw_image_visible.astype(np.float32)
        colors = raw.raw_colors_visible
        desc = raw.color_desc.decode("ascii", "replace")
        black = np.asarray(raw.black_level_per_channel, dtype=np.float32)
        white = float(raw.white_level)
    bayer_pattern(colors, desc, path)  # refuse a sensor without a 2x2 Bayer pattern
    return superpixel(bayer, colors, desc, black, white, calib, image_crop)


def superpixel(bayer: np.ndarray, colors: np.ndarray, desc: str, black: np.ndarray, white: float,
               calib: Optional[FrameCalibration] = None, image_crop: float = 1.0) -> LoadedFrame:
    """Reduce a Bayer mosaic to calibrated 2x2 superpixels."""
    h, w = (bayer.shape[0] // 2) * 2, (bayer.shape[1] // 2) * 2
    bayer, colors = bayer[:h, :w], colors[:h, :w]
    out = np.zeros((h // 2, w // 2, 3), np.float32)
    count = np.zeros(3, np.float32)
    saturated = np.zeros((h // 2, w // 2), bool)
    for dy in (0, 1):
        for dx in (0, 1):
            idx = int(colors[dy, dx])
            name = desc[idx] if idx < len(desc) else "G"
            ch = "RGB".index(name) if name in "RGB" else 1
            plane = bayer[dy::2, dx::2]
            saturated |= plane >= SATURATION_FRACTION * white
            scale = 1.0 / max(white - float(black[idx]), 1.0)
            out[..., ch] += (plane - black[idx]) * scale
            count[ch] += 1
    out /= np.maximum(count, 1)[None, None, :]
    _devignette(out, calib, image_crop)
    return LoadedFrame(out, saturated, 2)


def _devignette(rgb: np.ndarray, calib: Optional[FrameCalibration], image_crop: float) -> None:
    if calib is None or calib.lens_terms is None or calib.vignetting not in ("lensfun", "profile"):
        return
    h, w = rgb.shape[:2]
    for ch, name in enumerate("RGB"):
        terms = (calib.channel_terms or {}).get(name, calib.lens_terms)
        rgb[..., ch] /= vignetting_map(w, h, tuple(terms), image_crop, calib.profile_crop)


def _load_fits(path: str, calib: Optional[FrameCalibration], image_crop: float) -> LoadedFrame:
    from .fits import read_fits

    data, header = read_fits(path)
    if data.ndim == 3:
        rgb = np.moveaxis(data[:3], 0, -1).astype(np.float32)
        if rgb.shape[2] == 1:
            rgb = np.repeat(rgb, 3, axis=2)
        sat = (rgb >= SATURATION_FRACTION * float(np.nanmax(rgb))).any(axis=2)
        _devignette(rgb, calib, image_crop)
        return LoadedFrame(rgb, sat, 1)
    pattern = str(header.get("BAYERPAT", "")).strip().upper()
    if len(pattern) == 4 and set(pattern) <= set("RGB"):
        colors = np.zeros(data.shape, np.uint8)
        idx = {"R": 0, "G": 1, "B": 2}
        for i, (dy, dx) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
            colors[dy::2, dx::2] = idx[pattern[i]]
        top = float(np.nanmax(data))
        return superpixel(data, colors, "RGB", np.zeros(3, np.float32), top, calib, image_crop)
    rgb = np.repeat(data[..., None], 3, axis=2).astype(np.float32)
    sat = data >= SATURATION_FRACTION * float(np.nanmax(data))
    _devignette(rgb, calib, image_crop)
    return LoadedFrame(rgb, sat, 1)


# ---------------------------------------------------------------------------
# Full resolution
# ---------------------------------------------------------------------------


@dataclass
class RawMosaic:
    """A RAW file's Bayer data, read once; calibrate any part of it with ``region``."""

    bayer: np.ndarray  # (H, W) raw values
    pattern: str  # 2x2 CFA pattern at (0, 0), e.g. "RGGB"
    black: dict  # channel letter -> black level (G uses the mean of both greens)
    white: float
    wb: tuple[float, float, float]  # camera as-shot multipliers (R, G, B), for previews only

    @property
    def shape(self) -> tuple[int, int]:
        return self.bayer.shape


@dataclass
class FullFrame:
    rgb: np.ndarray  # (h, w, 3) float32 linear, 1.0 = full scale, vignetting corrected
    saturated: np.ndarray  # (h, w) bool: any channel near full scale (dilated over the demosaic kernel)
    x0: int  # region origin in sensor pixels
    y0: int
    hot_pixels: int
    wb: tuple[float, float, float]


def _part(master: Optional[np.ndarray], mosaic: RawMosaic, what: str, x0: int, y0: int, w: int, h: int):
    """The part of a master calibration frame under a region of the sensor."""
    if master is None:
        return None
    if master.shape != mosaic.shape:
        raise ValueError(f"the master {what} is {master.shape[1]} x {master.shape[0]} px but the light frames are "
                         f"{mosaic.shape[1]} x {mosaic.shape[0]}; use calibration frames from the same camera")
    return master[y0 : y0 + h, x0 : x0 + w]


def bayer_pattern(colors: np.ndarray, desc: str, path: str = "") -> str:
    """The 2x2 Bayer pattern at (0, 0) ("RGGB", "GBRG", ...) from LibRaw's colour index per pixel.
    The whole toolkit (superpixels, demosaicing, saturation per photosite) needs a 2x2 Bayer
    sensor; any other layout, such as Fujifilm's X-Trans (6x6), is refused here rather than misread."""
    c = np.asarray(colors)
    block = c[: min(12, c.shape[0]), : min(12, c.shape[1])]
    two = block[:2, :2]
    tiled = np.tile(two, ((block.shape[0] + 1) // 2, (block.shape[1] + 1) // 2))[: block.shape[0], : block.shape[1]]
    names = [desc[int(i)] if int(i) < len(desc) else "?" for i in two.flatten()]
    names = [n if n in "RGB" else ("G" if n == "G" or n == "?" else n) for n in names]
    if not np.array_equal(block, tiled) or sorted(set(names)) != ["B", "G", "R"] or names.count("G") != 2:
        where = f" ({path})" if path else ""
        raise ValueError(f"this camera's sensor does not use a 2x2 Bayer colour pattern{where}, which the toolkit "
                         "needs (Fujifilm X-Trans and Sigma Foveon sensors are not supported)")
    return "".join(names)


def read_mosaic(path: str) -> RawMosaic:
    try:
        import rawpy
    except ImportError as exc:
        raise RuntimeError("reading RAW files needs rawpy: pip install rawpy") from exc
    with rawpy.imread(path) as raw:
        bayer = raw.raw_image_visible.astype(np.float32)
        colors = raw.raw_colors_visible
        desc = raw.color_desc.decode("ascii", "replace")
        black_c = np.asarray(raw.black_level_per_channel, dtype=np.float32)
        white = float(raw.white_level)
        try:
            wb_raw = [float(v) for v in raw.camera_whitebalance]
        except Exception:
            wb_raw = [1.0, 1.0, 1.0, 1.0]
    pattern = bayer_pattern(colors, desc, path)
    black: dict = {}
    for k, (dy, dx) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
        black.setdefault(pattern[k], []).append(float(black_c[int(colors[dy, dx])]))
    black = {k: sum(v) / len(v) for k, v in black.items()}
    g = wb_raw[1] or 1.0
    wb = (wb_raw[0] / g if wb_raw[0] else 1.0, 1.0, wb_raw[2] / g if wb_raw[2] else 1.0)
    return RawMosaic(bayer, pattern, black, white, wb)


def region(mosaic: RawMosaic, x0: int, y0: int, w: int, h: int, calib: Optional[FrameCalibration] = None,
           image_crop: float = 1.0, hot_pixels: bool = True) -> FullFrame:
    """Calibrate and demosaic one rectangle of the sensor (origin snapped to the CFA grid)."""
    H, W = mosaic.shape
    x0 = max(0, min(int(x0) - int(x0) % 2, W - 2))
    y0 = max(0, min(int(y0) - int(y0) % 2, H - 2))
    w = min(int(w) + 1, W - x0)
    h = min(int(h) + 1, H - y0)
    w -= w % 2
    h -= h % 2
    raw = mosaic.bayer[y0 : y0 + h, x0 : x0 + w]
    pattern = mosaic.pattern  # origin is even, so the pattern is unchanged
    from .demosaic import channel_masks

    masks = channel_masks(pattern, (h, w))
    sat_cfa = raw >= SATURATION_FRACTION * mosaic.white
    cfa = np.empty((h, w), np.float32)
    offset = _part(calib.offset_map, mosaic, "dark or bias", x0, y0, w, h) if calib is not None else None
    for c in "RGB":
        # 1.0 stays the white level above the black level, whatever offset is subtracted.
        scale = 1.0 / max(mosaic.white - mosaic.black[c], 1.0)
        sub = offset[masks[c]] if offset is not None else mosaic.black[c]
        cfa[masks[c]] = (raw[masks[c]] - sub) * scale
    flat = _part(calib.flat_map, mosaic, "flat", x0, y0, w, h) if calib is not None else None
    if flat is not None:
        cfa /= flat
    elif calib is not None and calib.lens_terms is not None and calib.vignetting in ("lensfun", "profile"):
        for c in "RGB":
            terms = (calib.channel_terms or {}).get(c, calib.lens_terms)
            vmap = vignetting_map(W, H, tuple(terms), image_crop, calib.profile_crop, box=(x0, y0, w, h))
            cfa[masks[c]] /= vmap[masks[c]]
    n_hot = 0
    if hot_pixels:
        cfa, n_hot = cosmetic_correct(cfa, pattern)
    rgb = demosaic_mhc(cfa, pattern)
    from scipy import ndimage

    saturated = ndimage.binary_dilation(sat_cfa, iterations=2)
    return FullFrame(rgb, saturated, x0, y0, n_hot, mosaic.wb)
