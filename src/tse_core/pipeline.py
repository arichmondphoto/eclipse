"""Phase 2 pipeline: load the light frames, align them on the corona, find
the Moon in each, and locate the Sun's centre.

Everything is computed on 2x2 superpixels inside a square window around
the Moon, then reported in full-resolution sensor coordinates.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

from .align import AlignFrame, AlignParams, FrameAlignment, Transform, align_sequence, auto_window
from .calibration import CalibrationPlan, FrameCalibration
from .eclipse import LocalEclipse, Site, elements_for
from .frames import FrameMeta
from .geometry import MoonFit, SunCentre, expected_moon_radius_px, find_moon, solve_sun_centre
from .rawio import LoadedFrame, load_frame

FULL_FRAME_DIAGONAL_MM = math.hypot(36.0, 24.0)

Loader = Callable[[FrameMeta, Optional[FrameCalibration]], LoadedFrame]


@dataclass
class FrameResult:
    index: int
    file_name: str
    exposure_text: str
    time_ut: str
    matrix: list[list[float]]  # 2x3, full-resolution frame pixel -> reference pixel
    angle_deg: float
    shift_px: tuple[float, float]  # full-resolution translation part (for a quick look)
    psr: float
    overlap: float
    moon_frame: dict  # Moon fit in the frame's own full-resolution coordinates
    moon_ref: tuple[float, float]  # Moon centre in reference coordinates
    notes: list[str] = field(default_factory=list)


@dataclass
class AlignmentResult:
    reference_index: int
    image_size: tuple[int, int]  # full-resolution (width, height)
    window: dict  # analysis window on the superpixel grid
    frames: list[FrameResult]
    moon_radius_px: float
    sun: Optional[dict]
    params: dict
    notes: list[str] = field(default_factory=list)

    def save(self, path: str | os.PathLike) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    def describe(self) -> str:
        lines = [f"Reference frame: #{self.reference_index}; Moon radius {self.moon_radius_px:.2f} px",
                 f"{'#':>3}  {'exposure':>8}  {'dx':>8} {'dy':>8} {'rot deg':>8}  {'psr':>6}  notes"]
        for f in self.frames:
            lines.append(f"{f.index:>3}  {f.exposure_text:>8}  {f.shift_px[0]:+8.2f} {f.shift_px[1]:+8.2f} "
                         f"{f.angle_deg:+8.4f}  {f.psr:6.0f}  {'; '.join(f.notes)}")
        if self.sun:
            lines.append("")
            lines.append(self.sun["text"])
        lines += [f"Note: {n}" for n in self.notes]
        return "\n".join(lines)


def default_loader(plan: Optional[CalibrationPlan], crop_factor: float) -> Loader:
    by_index = {fc.frame_index: fc for fc in plan.frames} if plan else {}

    def load(frame: FrameMeta, _calib: Optional[FrameCalibration]) -> LoadedFrame:
        return load_frame(frame.path, by_index.get(frame.index), crop_factor)

    return load


def run_alignment(
    frames: Sequence[FrameMeta],
    loader: Loader,
    params: Optional[AlignParams] = None,
    window: Optional[int] = None,
    site: Optional[Site] = None,
    clock_error_s: Optional[float] = None,
    crop_factor: float = 1.0,
    utc_offset_h: Optional[float] = None,
    mirrored: bool = False,
    progress: Optional[Callable[[str], None]] = None,
) -> AlignmentResult:
    """Align the light frames; ``utc_offset_h`` is the camera's time zone when EXIF has none."""
    say = progress or (lambda _m: None)

    def _aware(f: FrameMeta) -> datetime:
        return _aware_time(f, utc_offset_h)

    params = params or AlignParams()
    frames = [f for f in frames if f.role == "light"]
    if len(frames) < 2:
        raise ValueError("need at least two light frames")

    # 1. Load and find the Moon in each frame (superpixel grid).
    loaded: list[LoadedFrame] = []
    moons: list[MoonFit] = []
    expected = None
    for k, f in enumerate(frames):
        lf = loader(f, None)
        lum = lf.luminance
        if expected is None:
            expected = _expected_radius(f, lum.shape, lf.binning, crop_factor)
        moons.append(find_moon(lum, expected))
        loaded.append(lf)
        say(f"loaded {f.file_name}: Moon at ({moons[-1].x:.1f}, {moons[-1].y:.1f}) r={moons[-1].radius:.1f}")
    b = loaded[0].binning
    h, w = loaded[0].luminance.shape

    # 2. Common window around the Moon.
    cx = float(np.median([m.x for m in moons]))
    cy = float(np.median([m.y for m in moons]))
    sun_r = float(np.median([m.radius for m in moons])) / 1.04
    size = auto_window(sun_r, w, h) if window is None else min(window, h, w)
    size -= size % 2
    x0 = int(round(cx)) - size // 2
    y0 = int(round(cy)) - size // 2
    x0 = min(max(x0, 0), w - size)
    y0 = min(max(y0, 0), h - size)

    try:
        t0 = min(_aware(f) for f in frames) if all(f.timestamp for f in frames) else None
    except ValueError as exc:
        say(str(exc))
        t0 = None
    align_frames = []
    for k, (f, lf, m) in enumerate(zip(frames, loaded, moons)):
        lum = lf.luminance[y0 : y0 + size, x0 : x0 + size].astype(np.float32)
        sat = lf.saturated[y0 : y0 + size, x0 : x0 + size]
        ev = f.exposure_value()
        exposure = 2.0 ** ev if ev is not None else (f.exposure_s or 1.0)
        t = (_aware(f) - t0).total_seconds() if t0 is not None else None
        moon_w = MoonFit(m.x - x0, m.y - y0, m.radius, m.rms, m.points, m.rejected)
        align_frames.append(AlignFrame(k, lum, exposure, sat, moon_w, time_s=t))
    loaded.clear()

    # 3. Align.
    results = align_sequence(align_frames, params, progress=say)
    ref_k = next(r.index for r in results if math.isinf(r.psr))

    # 4. Full-resolution transforms and Moon positions in reference coordinates.
    out_frames = []
    moon_ref_bin = []
    for f, r, m in zip(frames, results, moons):
        M = _to_full(r.transform, x0, y0, b)
        mx_w, my_w = r.transform.apply_point(m.x - x0, m.y - y0)
        moon_ref_bin.append((mx_w, my_w))
        mfx, mfy = _bin_to_full(m.x, m.y, b)
        mrx, mry = _bin_to_full(mx_w + x0, my_w + y0, b)
        out_frames.append(FrameResult(
            index=f.index, file_name=f.file_name, exposure_text=f.exposure_text,
            time_ut=_aware(f).astimezone(timezone.utc).isoformat() if t0 is not None else "",
            matrix=M.tolist(), angle_deg=r.transform.angle_deg,
            shift_px=_shift_at(M, mfx, mfy),
            psr=r.psr, overlap=r.overlap,
            moon_frame={"x": mfx, "y": mfy, "radius": m.radius * b, "rms": m.rms * b},
            moon_ref=(mrx, mry), notes=r.notes))
    moon_r_full = float(np.median([m.radius for m in moons])) * b

    # 5. Sun centre.
    sun = None
    notes: list[str] = []
    if site is not None and t0 is not None:
        try:
            el = elements_for(t0.astimezone(timezone.utc).date())
            ecl = LocalEclipse(el, site)
            times = [_aware(f) for f in frames]
            sc = solve_sun_centre(times, [fr.moon_ref for fr in out_frames], moon_r_full, ecl,
                                  clock_error_s=clock_error_s, mirrored=mirrored)
            sun = asdict(sc)
            sun["text"] = sc.describe()
        except (KeyError, ValueError) as exc:
            notes.append(f"Sun centre not computed: {exc}")
    elif site is None:
        notes.append("no observing site given; Sun centre not computed (use --lat/--lon)")
    else:
        notes.append("frames have no capture times; Sun centre not computed")

    return AlignmentResult(
        reference_index=frames[ref_k].index,
        image_size=(w * b, h * b),
        window={"x0": x0, "y0": y0, "size": size, "binning": b},
        frames=out_frames,
        moon_radius_px=moon_r_full,
        sun=sun,
        params=asdict(params),
        notes=notes,
    )


def _expected_radius(f: FrameMeta, shape: tuple[int, int], binning: int, crop_factor: float) -> Optional[float]:
    if not f.focal_length_mm:
        return None
    h, w = shape
    diag_px = math.hypot(w * binning, h * binning)
    pitch_um = (FULL_FRAME_DIAGONAL_MM / (f.crop_factor or crop_factor or 1.0)) / diag_px * 1000.0
    return expected_moon_radius_px(f.focal_length_mm, pitch_um, binning)


def _bin_to_full(x: float, y: float, b: int) -> tuple[float, float]:
    return x * b + (b - 1) / 2.0, y * b + (b - 1) / 2.0


def _to_full(tf: Transform, x0: int, y0: int, b: int) -> np.ndarray:
    """2x3 matrix mapping full-resolution frame pixels to full-resolution reference pixels."""
    M = np.vstack([tf.matrix(), [0, 0, 1]])
    # full -> binned window: p_win = (p_full - (b-1)/2) / b - (x0, y0)
    to_win = np.array([[1 / b, 0, -(b - 1) / 2 / b - x0], [0, 1 / b, -(b - 1) / 2 / b - y0], [0, 0, 1]])
    to_full = np.linalg.inv(to_win)
    return (to_full @ M @ to_win)[:2]


def _shift_at(M: np.ndarray, x: float, y: float) -> tuple[float, float]:
    """Displacement the transform applies at (x, y): the 'shift' of that frame there."""
    return (float(M[0, 0] * x + M[0, 1] * y + M[0, 2] - x), float(M[1, 0] * x + M[1, 1] * y + M[1, 2] - y))


def _aware_time(f: FrameMeta, utc_offset_h: Optional[float] = None) -> datetime:
    dt = f.datetime()
    if dt is None:
        raise ValueError(f"{f.file_name} has no capture time")
    if dt.tzinfo is None:
        if utc_offset_h is None:
            raise ValueError(f"{f.file_name}: capture time has no time zone; give one with --camera-utc-offset")
        dt = dt.replace(tzinfo=timezone(timedelta(hours=utc_offset_h)))
    return dt
