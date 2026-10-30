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
