"""Smooth vignetting profiles fitted from the user's own flat frames.

A raw master flat only fits the camera, image size and dust it was shot
with. A *profile* keeps only what belongs to the lens: a radially symmetric
falloff in the same "pa" form lensfun uses,

    observed / true = 1 + k1 r^2 + k2 r^4 + k3 r^6,

with ``r = 1`` at the corner of the camera that took the flats, measured on
the sensor in millimetres. The same profile therefore corrects any camera
with that lens (a D800 and a D850, for example), and dust in the flats
never reaches the eclipse frames.

Fitting works on block averages of every CFA channel. Dust shadows,
hot pixels and other local defects are rejected iteratively, and a linear
brightness tilt is fitted alongside the profile and discarded, so a light
panel that is slightly brighter on one side does not bias the result.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from .frames import FrameMeta
from .lensfun import FULL_FRAME_HALF_DIAGONAL_MM, corner_falloff, relative_illumination

PROFILE_DIR = Path(os.environ.get("TSE_PROFILE_DIR", Path.home() / ".solar-eclipse-toolkit" / "profiles"))
CHANNELS = ("R", "G", "B")


@dataclass
class VignettingProfile:
    lens_model: str
    focal_length_mm: float
    aperture_f: float
    terms: tuple[float, float, float]  # all channels together
    channel_terms: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    crop_factor: float = 1.0  # of the camera that took the flats
    camera_model: str = ""
    flat_count: int = 0
    rms_residual: float = 0.0  # after the fit, as a fraction of the centre level
    light_tilt_pct: float = 0.0  # brightness difference across the frame removed as light-source tilt
    rejected_pct: float = 0.0  # blocks rejected as dust or defects
    name: str = ""
    created: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    # ----------------------------------------------------------------- use
    def corner(self, image_crop: float = 1.0) -> tuple[float, float]:
        return corner_falloff(self.terms, image_crop, self.crop_factor)

    def label(self) -> str:
        return self.name or f"{self.lens_model} at {self.focal_length_mm:g} mm f/{self.aperture_f:g} (own flats)"

    def describe(self) -> str:
        b, stops = self.corner()
        lines = [
            f"Profile: {self.label()}",
            f"  from {self.flat_count} flat(s), {self.camera_model or 'unknown camera'} (crop {self.crop_factor:g})",
            f"  k1={self.terms[0]:+.5f}  k2={self.terms[1]:+.5f}  k3={self.terms[2]:+.5f}",
            f"  corners at {b:.1%} of centre ({stops:+.2f} stops)",
        ]
        for ch, t in self.channel_terms.items():
            cb, cs = corner_falloff(t, 1.0, 1.0)
            lines.append(f"  {ch}: corners {cb:.1%} ({cs:+.2f} stops)")
        lines.append(
            f"  fit residual {self.rms_residual:.2%}, light tilt removed {self.light_tilt_pct:.1f}%, "
            f"rejected {self.rejected_pct:.1f}% of blocks"
        )
        if self.light_tilt_pct > 3:
            lines.append("  note: the flat light was noticeably uneven; check the light source")
        if self.rms_residual > 0.01:
            lines.append("  note: the residual is high; the flats may be uneven or not radially symmetric")
        return "\n".join(lines)

    # --------------------------------------------------------- persistence
    def save(self, path: Optional[str | os.PathLike] = None) -> Path:
        if path is None:
            PROFILE_DIR.mkdir(parents=True, exist_ok=True)
            slug = re.sub(r"[^A-Za-z0-9.]+", "_", f"{self.lens_model}_{self.focal_length_mm:g}mm_f{self.aperture_f:g}").strip("_")
            path = PROFILE_DIR / f"{slug}.json"
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return Path(path)

    @classmethod
    def load(cls, path: str | os.PathLike) -> "VignettingProfile":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        data["terms"] = tuple(data["terms"])
        data["channel_terms"] = {k: tuple(v) for k, v in data.get("channel_terms", {}).items()}
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in known})


def load_profile_library(directory: Optional[str | os.PathLike] = None) -> list[VignettingProfile]:
    folder = Path(directory) if directory else PROFILE_DIR
    profiles = []
    if folder.is_dir():
        for p in sorted(folder.glob("*.json")):
            try:
                profiles.append(VignettingProfile.load(p))
            except (ValueError, KeyError, TypeError):
                continue
    return profiles


def match_profile(frame: FrameMeta, profiles: Iterable[VignettingProfile]) -> Optional[VignettingProfile]:
    """Closest profile for the same lens, within 1/3 stop and 5 % focal length."""
    best, best_d = None, float("inf")
    lens = _norm(frame.lens_model)
    for p in profiles:
        if lens and _norm(p.lens_model) and _norm(p.lens_model) != lens:
            continue
        if not (frame.focal_length_mm and frame.aperture_f):
            continue
        df = abs(frame.focal_length_mm - p.focal_length_mm) / p.focal_length_mm
        da = abs(2 * math.log2(frame.aperture_f / p.aperture_f))  # in stops
        if df > 0.05 or da > 0.34:
            continue
        d = df + da
        if d < best_d:
            best, best_d = p, d
    return best


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


@dataclass
class _Samples:
    r2: list[np.ndarray] = field(default_factory=list)
    x: list[np.ndarray] = field(default_factory=list)
    y: list[np.ndarray] = field(default_factory=list)
    v: list[np.ndarray] = field(default_factory=list)


def fit_profile_from_planes(
    frames: list[dict[str, tuple[np.ndarray, float]]],
    full_width: int,
    full_height: int,
    crop_factor: float = 1.0,
    block: int = 32,
    **meta,
) -> VignettingProfile:
    """Fit a profile from flat images already split into colour planes.

    ``frames`` holds one dict per flat: channel name -> (plane, scale),
    where ``scale`` is sensor pixels per plane pixel (2 for a Bayer
    sub-plane, 1 for a full-resolution plane). Planes must be linear and
    black-subtracted.
    """
    per_channel: dict[str, _Samples] = {}
    half_diag_px = math.hypot(full_width, full_height) / 2.0
    # r = 1 at the corner of this camera's frame, as lensfun measures it.
    r_scale = 1.0 / half_diag_px
    cx, cy = full_width / 2.0, full_height / 2.0
    for planes in frames:
        for channel, (plane, scale) in planes.items():
            means, xs, ys = _block_means(np.asarray(plane, dtype=np.float64), block, scale)
            centre = _centre_level(means, xs - cx, ys - cy, half_diag_px)
            if not centre > 0:
                continue
            s = per_channel.setdefault(_channel_name(channel), _Samples())
            s.r2.append(((xs - cx) ** 2 + (ys - cy) ** 2) * r_scale**2)
            s.x.append((xs - cx) / cx)
            s.y.append((ys - cy) / cy)
            s.v.append(means / centre)
    if not per_channel:
        raise ValueError("no usable flat data")

    channel_terms, all_samples = {}, _Samples()
    for channel, s in per_channel.items():
        r2, x, y, v = (np.concatenate(a) for a in (s.r2, s.x, s.y, s.v))
        terms, tilt, rms, rejected = _robust_fit(r2, x, y, v)
        channel_terms[channel] = terms
        for a, b in ((all_samples.r2, r2), (all_samples.x, x), (all_samples.y, y), (all_samples.v, v)):
            a.append(b)
    r2, x, y, v = (np.concatenate(a) for a in (all_samples.r2, all_samples.x, all_samples.y, all_samples.v))
    terms, tilt, rms, rejected = _robust_fit(r2, x, y, v)
    return VignettingProfile(
        terms=terms,
        channel_terms=channel_terms,
        crop_factor=crop_factor,
        flat_count=len(frames),
        rms_residual=rms,
        light_tilt_pct=tilt,
        rejected_pct=rejected,
        **meta,
    )


def _block_means(plane: np.ndarray, block: int, scale: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = plane.shape
    bh, bw = h // block, w // block
    if bh < 4 or bw < 4:
        block = max(1, min(h, w) // 8)
        bh, bw = h // block, w // block
    cropped = plane[: bh * block, : bw * block]
    means = cropped.reshape(bh, block, bw, block).mean(axis=(1, 3))
    ys = (np.arange(bh) + 0.5) * block * scale
    xs = (np.arange(bw) + 0.5) * block * scale
    gx, gy = np.meshgrid(xs, ys)
    return means.ravel(), gx.ravel(), gy.ravel()


def _centre_level(values: np.ndarray, dx: np.ndarray, dy: np.ndarray, half_diag: float) -> float:
    near = np.hypot(dx, dy) < 0.1 * half_diag
    pick = values[near] if near.sum() >= 4 else values
    return float(np.median(pick))


def _robust_fit(r2, x, y, v, iterations: int = 6, clip: float = 3.0):
    """Fit v = c (1 + k1 r2 + k2 r2^2 + k3 r2^3) + a x + b y, rejecting outliers."""
    keep = np.isfinite(v) & (v > 0)
    design = np.column_stack([np.ones_like(r2), r2, r2**2, r2**3, x, y])
    coef = None
    for _ in range(iterations):
        coef, *_ = np.linalg.lstsq(design[keep], v[keep], rcond=None)
        resid = v - design @ coef
        sigma = 1.4826 * np.median(np.abs(resid[keep] - np.median(resid[keep])))
        if not sigma > 0:
            break
        new_keep = np.isfinite(v) & (np.abs(resid) < clip * sigma)
        if np.array_equal(new_keep, keep):
            break
        keep = new_keep
    c = coef[0]
    terms = (float(coef[1] / c), float(coef[2] / c), float(coef[3] / c))
    tilt_pct = float(200 * math.hypot(coef[4], coef[5]) / c)  # edge-to-edge brightness change
    resid = v - design @ coef
    rms = float(np.sqrt(np.mean(resid[keep] ** 2)) / c)
    rejected = float(100 * (1 - keep.mean()))
    return terms, tilt_pct, rms, rejected


def _channel_name(channel: str) -> str:
    """Bayer sub-plane names (R, G, G2, B) to colour channels (R, G, B)."""
    return channel.upper()[:1] if channel[:1].upper() in "RGB" else channel


# ---------------------------------------------------------------------------
# Reading flats
# ---------------------------------------------------------------------------


def planes_from_raw(path: str) -> tuple[dict[str, tuple[np.ndarray, float]], int, int]:
    """Black-subtracted Bayer sub-planes of a RAW flat (needs rawpy)."""
    import rawpy

    with rawpy.imread(path) as raw:
        img = raw.raw_image_visible.astype(np.float32)
        colors = raw.raw_colors_visible
        desc = raw.color_desc.decode("ascii", "replace")
        black = np.asarray(raw.black_level_per_channel, dtype=np.float32)
        h, w = img.shape
        planes: dict[str, tuple[np.ndarray, float]] = {}
        for dy in (0, 1):
            for dx in (0, 1):
                idx = int(colors[dy, dx])
                name = desc[idx] if idx < len(desc) else str(idx)
                key = name if name not in planes else name + "2"
                sub = img[dy::2, dx::2] - black[idx]
                planes[key] = (sub, 2.0)
    return planes, w, h


def planes_from_fits(path: str) -> tuple[dict[str, tuple[np.ndarray, float]], int, int]:
    """Colour planes of a FITS flat (CFA with BAYERPAT, RGB cube, or mono)."""
    from .fits import read_fits

    data, header = read_fits(path)
    if data.ndim == 3:
        names = CHANNELS if data.shape[0] == 3 else [f"C{i}" for i in range(data.shape[0])]
        return {n: (data[i], 1.0) for i, n in enumerate(names)}, data.shape[2], data.shape[1]
    pattern = str(header.get("BAYERPAT", "")).strip().upper()
    h, w = data.shape
    if len(pattern) == 4 and set(pattern) <= set("RGB"):
        planes: dict[str, tuple[np.ndarray, float]] = {}
        for i, (dy, dx) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
            name = pattern[i]
            key = name if name not in planes else name + "2"
            planes[key] = (data[dy::2, dx::2], 2.0)
        return planes, w, h
    return {"L": (data, 1.0)}, w, h


def fit_profile_from_files(
    paths: list[str], frames_meta: Optional[list[FrameMeta]] = None, crop_factor: Optional[float] = None, block: int = 32
) -> VignettingProfile:
    """Read flats (RAW via rawpy, or FITS) and fit one profile."""
    frames, width, height = [], None, None
    for p in paths:
        planes, w, h = planes_from_fits(p) if p.lower().endswith((".fit", ".fits", ".fts")) else planes_from_raw(p)
        if width is None:
            width, height = w, h
        elif (w, h) != (width, height):
            raise ValueError(f"{Path(p).name} has a different size from the other flats")
        frames.append(planes)
    sample = frames_meta[0] if frames_meta else None
    return fit_profile_from_planes(
        frames, width, height,
        crop_factor=crop_factor or (sample.crop_factor if sample and sample.crop_factor else 1.0),
        block=block,
        lens_model=sample.lens_model if sample else "",
        focal_length_mm=float(sample.focal_length_mm) if sample and sample.focal_length_mm else 0.0,
        aperture_f=float(sample.aperture_f) if sample and sample.aperture_f else 0.0,
        camera_model=sample.camera_model if sample else "",
    )


def profile_map(profile: VignettingProfile, width: int, height: int, image_crop: float = 1.0,
                cfa_pattern: str = "", dtype=np.float32) -> np.ndarray:
    """Relative illumination for an image, per CFA channel when the pattern is given."""
    from .lensfun import vignetting_map

    if len(cfa_pattern) != 4 or not profile.channel_terms:
        return vignetting_map(width, height, profile.terms, image_crop, profile.crop_factor, dtype)
    out = np.empty((height, width), dtype=dtype)
    for i, (dy, dx) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
        ch = _channel_name(cfa_pattern[i])
        terms = profile.channel_terms.get(ch, profile.terms)
        full = vignetting_map(width, height, terms, image_crop, profile.crop_factor, dtype)
        out[dy::2, dx::2] = full[dy::2, dx::2]
    return out


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


__all__ = [
    "VignettingProfile", "fit_profile_from_planes", "fit_profile_from_files", "load_profile_library",
    "match_profile", "profile_map", "planes_from_raw", "planes_from_fits", "relative_illumination",
    "FULL_FRAME_HALF_DIAGONAL_MM",
]
