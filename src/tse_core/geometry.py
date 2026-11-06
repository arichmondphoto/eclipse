"""Find the Moon in each frame and the Sun's centre behind it.

``find_moon`` locates the lunar limb in one frame and fits a circle to it
with sub-pixel accuracy. It works on any exposure: in short ones the limb
is the edge of the bright inner corona, in long ones the edge of the
saturated region. Prominences and lunar valleys are rejected as outliers.

``solve_sun_centre`` follows the appendix of Druckmüllerová, Morgan &
Habbal (2011): once the frames are aligned on the corona, the Moon moves
along a straight line across the Sun, and an eclipse calculator says where
the Sun's centre lies relative to that track. The model fitted here is

    moon_px(t) = H + R_sun_px * Rot(theta) * P * offset(t + clock_error)

where ``offset`` is the Moon-minus-Sun vector in Sun radii (east, north)
from :mod:`tse_core.eclipse`, ``P`` maps sky east/north to image axes, and
``theta`` is the camera's rotation on the sky. Track direction gives
theta, the track's side gives the across-track position of H, and the
camera clock error sets the along-track position. When the clock error is
unknown it is bounded by requiring every frame to lie inside totality,
and the remaining uncertainty is reported.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional, Sequence

import numpy as np
from scipy import ndimage

from .eclipse import LocalEclipse

MOON_RADIUS_ARCSEC = 1000.0  # rough, for the first guess only


# ---------------------------------------------------------------------------
# Moon in one frame
# ---------------------------------------------------------------------------


@dataclass
class MoonFit:
    x: float
    y: float
    radius: float
    rms: float  # residual of limb points about the circle, px
    points: int  # limb points kept
    rejected: int

    def as_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "radius": self.radius, "rms": self.rms,
                "points": self.points, "rejected": self.rejected}


def find_moon(
    image: np.ndarray,
    expected_radius: Optional[float] = None,
    angles: int = 720,
    saturation: Optional[float] = None,
) -> MoonFit:
    """Fit the lunar limb in a 2-D image (any exposure, linear data)."""
    img = np.asarray(image, dtype=np.float32)
    h, w = img.shape
    # 1. Coarse: work on a copy about 800 px across.
    f = max(1, int(round(max(h, w) / 800)))
    small = _block_mean(img, f)
    small = ndimage.gaussian_filter(small, 1.5)
    cx, cy = _bright_ring_centre(small)
    r_exp = expected_radius / f if expected_radius else None
    fit = _limb_fit(small, cx, cy, r_exp, angles=360, half_window=None, smooth=1.0)
    if r_exp is not None and not _plausible(fit):
        # The expected size may be wrong (focal length or pixel size): search freely too.
        free = _limb_fit(small, cx, cy, None, angles=360, half_window=None, smooth=1.0)
        if _quality(free) < _quality(fit):
            fit = free
    # 2. Fine: full resolution, a narrow band around the coarse limb.
    cx, cy, r = fit.x * f + (f - 1) / 2.0, fit.y * f + (f - 1) / 2.0, fit.radius * f
    for half in (max(12.0, 3.0 * f), 8.0):
        fit = _limb_fit(img, cx, cy, r, angles=angles, half_window=half, smooth=1.0, saturation=saturation)
        cx, cy, r = fit.x, fit.y, fit.radius
    return fit


def _quality(fit: MoonFit) -> float:
    """Lower is better: residual, penalised by the share of rejected limb points."""
    total = fit.points + fit.rejected
    return fit.rms * (1.0 + 4.0 * fit.rejected / max(total, 1))


def _plausible(fit: MoonFit) -> bool:
    total = fit.points + fit.rejected
    return fit.rms < 1.0 and fit.points > 0.7 * total


def _block_mean(img: np.ndarray, f: int) -> np.ndarray:
    if f == 1:
        return img
    h, w = (img.shape[0] // f) * f, (img.shape[1] // f) * f
    return img[:h, :w].reshape(h // f, f, w // f, f).mean(axis=(1, 3))


def _bright_ring_centre(img: np.ndarray) -> tuple[float, float]:
    """Centroid of the brightest pixels: the inner corona ring surrounds the Moon."""
    thr = np.percentile(img, 99.5)
    yy, xx = np.nonzero(img >= thr)
    return float(xx.mean()), float(yy.mean())


def _limb_fit(img, cx, cy, r_guess, angles, half_window, smooth, saturation=None) -> MoonFit:
    h, w = img.shape
    theta = np.linspace(0, 2 * np.pi, angles, endpoint=False)
    if half_window is None:
        r_max = 0.45 * min(h, w)
        radii = np.arange(2.0, r_max, 0.5)
    else:
        radii = np.arange(max(2.0, r_guess - half_window), r_guess + half_window, 0.25)
    xs = cx + np.outer(np.cos(theta), radii)
    ys = cy + np.outer(np.sin(theta), radii)
    prof = ndimage.map_coordinates(img, [ys, xs], order=1, mode="nearest")
    if smooth:
        prof = ndimage.gaussian_filter1d(prof, smooth / (radii[1] - radii[0]), axis=1)
    if saturation is not None:
        prof = np.minimum(prof, saturation)
    grad = np.gradient(prof, radii, axis=1)
    if half_window is None and r_guess is not None:
        # Only accept steps near the expected radius.
        keep = (radii > 0.5 * r_guess) & (radii < 2.0 * r_guess)
        grad[:, ~keep] = -1e30
    elif half_window is None:
        # The Moon is dark inside: the limb is the first big rise from a dark interior.
        inside = prof[:, : max(3, len(radii) // 20)].mean(axis=1, keepdims=True)
        grad[np.maximum.accumulate(prof, axis=1) > inside + 0.5 * (prof.max(axis=1, keepdims=True) - inside)] = -1e30
    i = np.argmax(grad, axis=1)
    i = np.clip(i, 1, len(radii) - 2)
    rows = np.arange(len(theta))
    g0, g1, g2 = grad[rows, i - 1], grad[rows, i], grad[rows, i + 1]
    denom = g0 - 2 * g1 + g2
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        frac = np.where(np.abs(denom) > 1e-12, 0.5 * (g0 - g2) / denom, 0.0)
    frac = np.nan_to_num(frac)
    r_edge = radii[i] + np.clip(frac, -1, 1) * (radii[1] - radii[0])
    strength = g1
    px = cx + r_edge * np.cos(theta)
    py = cy + r_edge * np.sin(theta)
    good = np.isfinite(strength) & (strength > 0) & (strength < 1e29)
    # Weak edges (behind prominences or in noise) are less reliable.
    if good.sum() > 10:
        good &= strength > 0.2 * np.median(strength[good])
    return _robust_circle(px[good], py[good], len(theta))


def _robust_circle(px: np.ndarray, py: np.ndarray, total: int) -> MoonFit:
    keep = np.ones(len(px), bool)
    cx = cy = r = 0.0
    for _ in range(10):
        cx, cy, r = _circle_lsq(px[keep], py[keep])
        res = np.hypot(px - cx, py - cy) - r
        mad = 1.4826 * np.median(np.abs(res[keep] - np.median(res[keep])))
        new = np.abs(res) < max(3.0 * mad, 0.3)
        # Mountains and prominences stick out; valleys are small. Reject both.
        if np.array_equal(new, keep):
            break
        keep = new
    res = np.hypot(px[keep] - cx, py[keep] - cy) - r
    return MoonFit(float(cx), float(cy), float(r), float(np.sqrt(np.mean(res**2))), int(keep.sum()), int(total - keep.sum()))


def _circle_lsq(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Algebraic fit, then Gauss-Newton on geometric distances."""
    A = np.column_stack([x, y, np.ones_like(x)])
    bvec = x * x + y * y
    c, *_ = np.linalg.lstsq(A, bvec, rcond=None)
    cx, cy = c[0] / 2, c[1] / 2
    r = math.sqrt(max(c[2] + cx * cx + cy * cy, 1e-9))
    for _ in range(10):
        dx, dy = x - cx, y - cy
        d = np.hypot(dx, dy)
        d[d == 0] = 1e-9
        J = np.column_stack([-dx / d, -dy / d, -np.ones_like(d)])
        res = d - r
        step, *_ = np.linalg.lstsq(J, -res, rcond=None)
        cx, cy, r = cx + step[0], cy + step[1], r + step[2]
        if np.max(np.abs(step)) < 1e-6:
            break
    return float(cx), float(cy), float(r)


def expected_moon_radius_px(focal_length_mm: float, pixel_pitch_um: float, binning: int = 1) -> float:
    arcsec_per_px = 206264.806 * pixel_pitch_um * 1e-3 * binning / focal_length_mm
    return MOON_RADIUS_ARCSEC / arcsec_per_px


# ---------------------------------------------------------------------------
# Sun centre from the Moon's track
# ---------------------------------------------------------------------------


@dataclass
class SunCentre:
    x: float
    y: float
    radius_px: float  # solar radius in image pixels
    arcsec_per_px: float
    rotation_deg: float  # angle of celestial north from image "up", towards image right... see describe()
    clock_error_s: float  # true UT minus camera UT
    clock_error_range_s: tuple[float, float]
    along_track_uncertainty_px: float
    track_rms_px: float
    speed_ratio: float  # measured / predicted Moon speed; ~1 when the scale is right
    mirrored: bool
    notes: list[str] = field(default_factory=list)
    px_per_clock_second: float = 0.0  # how far the centre moves along the track per second of clock error
    rejected: list[int] = field(default_factory=list)  # input positions left out of the track fit
    residuals_px: list[float] = field(default_factory=list)  # each frame's distance from the fitted track
    # The answer had the image the other way round (mirrored vs not). The Moon's track cannot
    # tell the two apart, so both are kept; they differ by twice the closest approach of centres.
    other_parity_xy: Optional[tuple[float, float]] = None
    clock_error_given: bool = False
    parity_difference_px: float = 0.0

    def describe(self) -> str:
        lo, hi = self.clock_error_range_s
        lines = [
            f"Sun centre: x={self.x:.2f}, y={self.y:.2f} px (radius {self.radius_px:.1f} px, {self.arcsec_per_px:.3f}\"/px)",
            f"Celestial north is {self.rotation_deg:+.2f} deg from image up (positive = towards image right)",
            f"Camera clock error used: {self.clock_error_s:+.1f} s (true UT minus camera UT)",
            f"Image orientation: {'mirrored' if self.mirrored else 'not mirrored'}; east is "
            f"{'clockwise' if self.mirrored else 'counter-clockwise'} from north on the image",
        ]
        if self.clock_error_given:
            lines.append(f"  each second of clock error moves the centre {self.px_per_clock_second:.2f} px along the Moon's track")
        elif hi > lo:
            lines.append(
                f"  allowed by 'all frames in totality': {lo:+.0f} to {hi:+.0f} s; "
                f"along-track uncertainty +/-{self.along_track_uncertainty_px:.1f} px"
            )
        if self.other_parity_xy is not None and self.parity_difference_px > max(2.0, 0.005 * self.radius_px):
            ox, oy = self.other_parity_xy
            lines.append(
                f"If the optics {'do not ' if self.mirrored else ''}mirror the image, the Sun centre would be at "
                f"x={ox:.2f}, y={oy:.2f} ({self.parity_difference_px:.1f} px away)"
            )
        lines.append(f"Moon track fit residual {self.track_rms_px:.2f} px; measured/predicted Moon speed {self.speed_ratio:.3f}")
        lines += [f"Note: {n}" for n in self.notes]
        return "\n".join(lines)


def solve_sun_centre(
    times_ut: Sequence[datetime],
    moon_xy: Sequence[tuple[float, float]],
    moon_radius_px: float,
    eclipse: LocalEclipse,
    clock_error_s: Optional[float] = None,
    mirrored: bool = False,
    limb_rms: Optional[Sequence[float]] = None,
) -> SunCentre:
    """Sun centre in aligned-image coordinates from the Moon centres over time.

    ``times_ut`` are the camera's time stamps converted to UT with the
    camera's own time-zone tag; ``clock_error_s`` (true minus camera) is
    applied on top. Leave it None to bound it by totality.

    ``limb_rms`` (the lunar-limb fit residual of each frame) weights the
    frames: in long exposures glare from the saturated corona softens the
    limb and biases the fitted centre. Frames whose Moon lies far off the
    fitted track (camera shake, a doubled image) are rejected and reported.
    """
    m = np.asarray(moon_xy, dtype=float)
    if len(m) < 2:
        raise ValueError("need the Moon's position in at least two frames")
    t_cam = [_utc(t) for t in times_ut]
    circ = eclipse.circumstances()
    notes: list[str] = []
    if circ.kind != "total":
        raise ValueError(f"the eclipse is {circ.kind} at this site, not total")

    # Clock error window: every frame inside totality.
    first, last = min(t_cam), max(t_cam)
    lo = (circ.c2 - first).total_seconds()
    hi = (circ.c3 - last).total_seconds()
    if lo > hi:
        notes.append("the frames span longer than totality at this site; check the location and frames")
        lo = hi = 0.5 * (lo + hi)
    if clock_error_s is None:
        tau = 0.5 * (lo + hi)
    else:
        tau = float(clock_error_s)
        if not lo - 2 <= tau <= hi + 2:
            notes.append(f"with this clock error some frames fall outside totality ({lo:+.0f} to {hi:+.0f} s allowed)")

    P = np.array([[1.0, 0.0], [0.0, -1.0]]) if mirrored else np.array([[-1.0, 0.0], [0.0, -1.0]])

    base_w = np.ones(len(m))
    if limb_rms is not None:
        base_w = 1.0 / np.maximum(np.asarray(limb_rms, dtype=float), 0.3) ** 2
    keep = np.ones(len(m), bool)

    def fit(tau_s: float):
        states = [eclipse.state_at(t + timedelta(seconds=tau_s)) for t in t_cam]
        ratio = float(np.mean([s.ratio for s in states]))
        sun_r = moon_radius_px / ratio
        wv = np.array([P @ np.array(s.offset()) for s in states]) * sun_r
        w = base_w * keep
        w = w / w.sum()
        mc, wc = (w[:, None] * m).sum(axis=0), (w[:, None] * wv).sum(axis=0)
        dm, dw = m - mc, wv - wc
        theta = math.atan2(np.sum(w * (dw[:, 0] * dm[:, 1] - dw[:, 1] * dm[:, 0])),
                           np.sum(w * (dw[:, 0] * dm[:, 0] + dw[:, 1] * dm[:, 1])))
        R = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
        Hc = mc - R @ wc
        pred = Hc + (R @ wv.T).T
        resid = np.linalg.norm(pred - m, axis=1)
        rms = float(np.sqrt(np.sum(w * resid**2)))
        # Measured / predicted speed: least-squares scale between the two tracks.
        rdw = (R @ dw.T).T
        denom = float(np.sum(w * np.sum(rdw * rdw, axis=1)))
        speed = float(np.sum(w * np.sum(dm * rdw, axis=1)) / denom) if denom > 0 else float("nan")
        return Hc, theta, sun_r, rms, speed, states, resid

    # Reject frames whose Moon is far off the track (shake, doubled image, bad limb fit): over 3x the
    # typical scatter, and at least 0.35% of the Moon's radius off (1.5 px with a 400 mm lens on a
    # 45 MP full-frame camera), so the limit is the same whatever the lens and camera.
    limit = 0.0035 * float(moon_radius_px)
    for _ in range(5):
        resid = fit(tau)[6]
        scale = 1.4826 * float(np.median(resid[keep])) if keep.any() else 0.0
        new = resid < max(3.0 * scale, limit)
        if new.sum() < 2 or np.array_equal(new, keep):
            break
        keep = new
    rejected = [int(i) for i in np.nonzero(~keep)[0]]
    if rejected:
        notes.append(f"{len(rejected)} frame(s) left out of the Moon track: their Moon lies off the track "
                     "(camera shake, a doubled image, or a poor limb fit)")

    H, theta, sun_r, rms, speed, states, resid = fit(tau)
    H_lo = fit(lo)[0]
    H_hi = fit(hi)[0]
    along = 0.5 * float(np.linalg.norm(H_hi - H_lo)) if clock_error_s is None else 0.0
    if len(m) >= 3 and abs(speed - 1) > 0.05:
        notes.append("the Moon moved at a different speed than predicted; check the site, date and alignment")
    if np.linalg.norm(m - m.mean(axis=0), axis=1).max() < 2:
        notes.append("the Moon barely moved between frames; the track direction is poorly determined")
    sun_arcsec = circ.sun_radius_arcsec
    # Image "up" is -y. North vector on the image is R @ P @ (0, 1).
    north = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]]) @ P @ np.array([0.0, 1.0])
    rot = math.degrees(math.atan2(north[0], -north[1]))
    # Same fit with the other image parity.
    P_alt = np.array([[-1.0, 0.0], [0.0, -1.0]]) if mirrored else np.array([[1.0, 0.0], [0.0, -1.0]])
    P_keep = P
    P = P_alt
    H_alt = fit(tau)[0]
    P = P_keep
    diff = float(np.linalg.norm(H_alt - H))
    # Below about 0.5 % of the solar radius (or 2 px) the difference does not matter in practice.
    if diff > max(2.0, 0.005 * sun_r) and not mirrored:
        notes.append(
            f"the Sun's centre depends on whether your optics mirror the image ({diff:.1f} px apart here). "
            "Camera lenses do not; telescopes with a star diagonal do: use --mirrored for those"
        )
    return SunCentre(
        x=float(H[0]), y=float(H[1]), radius_px=float(sun_r), arcsec_per_px=sun_arcsec / sun_r,
        rotation_deg=rot, clock_error_s=tau, clock_error_range_s=(lo, hi),
        along_track_uncertainty_px=along, track_rms_px=rms, speed_ratio=speed, mirrored=mirrored, notes=notes,
        other_parity_xy=(float(H_alt[0]), float(H_alt[1])), parity_difference_px=diff,
        rejected=rejected, residuals_px=[float(r) for r in resid],
        clock_error_given=clock_error_s is not None,
        px_per_clock_second=float(np.linalg.norm(fit(tau + 1.0)[0] - H)),
    )


def _utc(t: datetime) -> datetime:
    if t.tzinfo is None:
        raise ValueError("frame times must carry a time zone (camera EXIF offset or one you give)")
    return t.astimezone(timezone.utc)
