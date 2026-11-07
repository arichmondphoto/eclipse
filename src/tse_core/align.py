"""Align eclipse frames on the corona: the modified phase correlation of
Druckmüller (2009), ApJ 706, 1605.

For two images a (reference) and b, both are first replaced by

    a' = H_rho(a) * m_a,      b' = H_rho(b) * m_b

where ``H_rho`` subtracts a blur along circles centred on the Moon (a
tangential unsharp mask that removes the steep radial brightness gradient
and every radial frequency that saturation, glare or film non-linearity
corrupts) and ``m`` is a soft mask removing the Moon, image edges,
saturated areas, stars and defective pixels. The shift is the peak of

    n(x, y) = F^-1[ A' conj(B')... ]   with damping p, q and a Gaussian low-pass sigma,

located to sub-pixel accuracy by the centroid of the peak within radius
epsilon. Rotation and scale come first, from the same correlation applied
to the Fourier amplitudes in log-polar coordinates.

Notes on the paper's formula (4): it is printed with typos ("r + cos").
It is implemented here as the intended tangential unsharp mask: subtract
a Gaussian blur along the circle through each pixel, with standard
deviation ``rho`` pixels of arc length, truncated at +/-2 rho.

Many frames are aligned against a reference that grows as frames are
added (an HDR composite of the frames aligned so far, each scaled by its
exposure), starting from the middle exposure and working outwards, so
every frame meets a reference with plenty of overlap in brightness. A
second pass re-aligns every frame against the composite of all the
others.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np
from scipy import fft as sfft
from scipy import ndimage

from .geometry import MoonFit, find_moon


# ---------------------------------------------------------------------------
# Parameters and results
# ---------------------------------------------------------------------------


@dataclass
class AlignParams:
    rho: Optional[float] = None  # tangential blur, px of arc length; None = from the Sun's size (see scaled_params)
    p_frac: float = 0.0005  # damping, fraction of the largest Fourier amplitude (paper: 0.01-0.1 %)
    sigma_frac: Optional[float] = None  # low-pass, fraction of the window width; None = choose automatically
    # Low-pass widths to try, as fractions of the window. Above ~0.16 the peak is narrower than a
    # pixel and the sub-pixel estimate locks onto whole pixels.
    sigma_candidates: tuple[float, ...] = (0.02, 0.04, 0.08, 0.16)
    epsilon: float = 4.0  # sub-pixel centroid radius (paper: 3-5)
    moon_margin: float = 0.06  # mask the Moon out to (1 + margin) * radius
    edge_margin: float = 0.04  # keep inside (1 - margin) * incircle
    mask_blur: Optional[float] = None  # px, softness of the mask; None = rho / 2
    saturation: float = 0.97  # of full scale, in normalised data
    star_sigma: float = 8.0  # local outliers this many sigma above their surroundings are masked
    min_snr: float = 8.0  # mask where the local brightness is below this many noise sigmas
    solve_rotation: bool = True  # small rotations (tracking mounts), refined from local shifts
    coarse_rotation: bool = False  # also search any angle and scale (log-polar); for alt-az or handheld
    rotation_model: str = "drift"  # "drift": smooth rotation over time; "per-frame"; "none"
    rotation_max_stops: float = 4.0  # measure rotation only on frames within this many stops of the anchor
    rotation_rounds: int = 6  # at most this many measure-model-realign rounds
    max_rotation_deg: float = 10.0  # larger refinements are treated as misfits
    window: Optional[int] = None  # analysis window (square, px); None = the largest fast size that fits
    passes: int = 2


@dataclass
class Transform:
    """Maps frame pixel coordinates into reference coordinates:

    ref = scale * Rot(angle) * (frame - centre) + centre + (dx, dy)
    """

    dx: float = 0.0
    dy: float = 0.0
    angle_deg: float = 0.0
    scale: float = 1.0
    centre_x: float = 0.0
    centre_y: float = 0.0

    def matrix(self) -> np.ndarray:
        a = math.radians(self.angle_deg)
        c, s = self.scale * math.cos(a), self.scale * math.sin(a)
        A = np.array([[c, -s], [s, c]])
        ctr = np.array([self.centre_x, self.centre_y])
        t = ctr + np.array([self.dx, self.dy]) - A @ ctr
        return np.array([[A[0, 0], A[0, 1], t[0]], [A[1, 0], A[1, 1], t[1]]])

    def apply_point(self, x: float, y: float) -> tuple[float, float]:
        M = self.matrix()
        return float(M[0, 0] * x + M[0, 1] * y + M[0, 2]), float(M[1, 0] * x + M[1, 1] * y + M[1, 2])

    @classmethod
    def from_matrix(cls, M: np.ndarray, centre_x: float, centre_y: float) -> "Transform":
        """Similarity transform from a 2x3 matrix, expressed about the given centre."""
        A = np.asarray(M)[:, :2]
        t = np.asarray(M)[:, 2]
        scale = math.sqrt(abs(np.linalg.det(A)))
        angle = math.degrees(math.atan2(A[1, 0], A[0, 0]))
        c = np.array([centre_x, centre_y])
        d = A @ c + t - c
        return cls(float(d[0]), float(d[1]), angle, scale, centre_x, centre_y)

    def then(self, other: "Transform") -> "Transform":
        """This transform followed by ``other``."""
        A = np.vstack([self.matrix(), [0, 0, 1]])
        B = np.vstack([other.matrix(), [0, 0, 1]])
        return Transform.from_matrix((B @ A)[:2], self.centre_x, self.centre_y)

    def inverse(self) -> "Transform":
        A = np.linalg.inv(np.vstack([self.matrix(), [0, 0, 1]]))
        return Transform.from_matrix(A[:2], self.centre_x, self.centre_y)

    def scaled(self, factor: float) -> "Transform":
        """The same transform for an image scaled by ``factor`` (e.g. 2 for binned -> full)."""
        return Transform(self.dx * factor, self.dy * factor, self.angle_deg, self.scale,
                         (self.centre_x + 0.5) * factor - 0.5, (self.centre_y + 0.5) * factor - 0.5)


@dataclass
class FrameAlignment:
    index: int
    transform: Transform
    peak: float  # correlation peak height
    psr: float  # peak-to-sidelobe ratio: > ~8 is a confident match
    sigma_frac: float
    overlap: float  # fraction of the window both masks keep
    moon: Optional[dict] = None
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        d = asdict(self)
        return d


# ---------------------------------------------------------------------------
# Preparation: tangential high-pass and mask
# ---------------------------------------------------------------------------


def sun_radius_from_moon(moon_radius_px: float) -> float:
    """Approximate solar radius in pixels from the lunar one (the Moon is 1.00-1.08 x the Sun in a total eclipse)."""
    return moon_radius_px / 1.04


def scaled_params(params: AlignParams, sun_radius_px: float) -> AlignParams:
    """Fill pixel-based settings from the Sun's size in the image.

    Druckmuller used rho = 8-16 px at 0.89"/px (a solar radius of ~1080 px), i.e. about 1 % of
    the radius; small images need relatively more blur to stay above the noise. The rule below
    gives 8 px at the D850 + 400 mm superpixel scale (~210 px radius) and stays within 4-16 px.
    """
    from dataclasses import replace

    rho = params.rho if params.rho is not None else float(min(16.0, max(4.0, 0.038 * sun_radius_px)))
    blur = params.mask_blur if params.mask_blur is not None else max(2.0, rho / 2.0)
    return replace(params, rho=rho, mask_blur=blur)


def auto_window(sun_radius_px: float, width: int, height: int) -> int:
    """Analysis window: the corona out to about 4 solar radii, 512-4096 px, inside the image."""
    want = sfft.next_fast_len(int(8 * sun_radius_px))
    size = int(min(max(want, 512), 4096, width, height))
    return size - size % 2


def tangential_highpass(img: np.ndarray, cx: float, cy: float, rho: float, r_max: Optional[float] = None,
                        return_blur: bool = False):
    """H_rho: the image minus its Gaussian blur along circles about (cx, cy)."""
    img = np.asarray(img, dtype=np.float32)
    h, w = img.shape
    if r_max is None:
        r_max = math.hypot(max(cx, w - cx), max(cy, h - cy))
    radii = np.arange(0.0, r_max + 2.0, 1.0, dtype=np.float32)
    nphi = sfft.next_fast_len(max(64, int(math.ceil(2 * math.pi * r_max))), real=True)
    phi = (np.arange(nphi, dtype=np.float32) * (2 * math.pi / nphi))
    cos_p, sin_p = np.cos(phi), np.sin(phi)
    polar = np.empty((len(radii), nphi), dtype=np.float32)
    chunk = max(1, int(4e6 // nphi))
    for start in range(0, len(radii), chunk):
        r = radii[start : start + chunk, None]
        polar[start : start + chunk] = ndimage.map_coordinates(
            img, [cy + r * sin_p, cx + r * cos_p], order=1, mode="nearest")
    # Blur each row by sigma = rho / r radians, done in the Fourier domain (rows are periodic).
    k = np.arange(nphi // 2 + 1, dtype=np.float32) / nphi
    sig_samples = rho / np.maximum(radii, 0.5) * (nphi / (2 * math.pi))
    spec = sfft.rfft(polar, axis=1, workers=-1)
    spec *= np.exp(-2.0 * (math.pi**2) * (sig_samples[:, None] ** 2) * (k[None, :] ** 2)).astype(np.float32)
    blurred = sfft.irfft(spec, n=nphi, axis=1, workers=-1).astype(np.float32)
    # Back to the image grid.
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    rr = np.hypot(xx - cx, yy - cy)
    pp = np.mod(np.arctan2(yy - cy, xx - cx), 2 * math.pi) * (nphi / (2 * math.pi))
    wrapped = np.concatenate([blurred, blurred[:, :1]], axis=1)
    back = ndimage.map_coordinates(wrapped, [rr, pp], order=1, mode="nearest")
    if return_blur:
        return img - back, back
    return img - back


def noise_sigma(img: np.ndarray) -> float:
    """Per-pixel noise in the faintest part of the image (robust)."""
    resid = img - ndimage.median_filter(img, size=3)
    faint = img <= np.percentile(img, 30)
    vals = resid[faint] if faint.sum() > 100 else resid.ravel()
    return float(1.4826 * np.median(np.abs(vals - np.median(vals))) * 1.2) + 1e-12


def defect_mask(img: np.ndarray, n_sigma: float = 8.0) -> np.ndarray:
    """True on stars, hot pixels and other small bright outliers."""
    med = ndimage.median_filter(img, size=5)
    resid = img - med
    # Local noise from the mean absolute residual (cheaper than a second median).
    noise = ndimage.uniform_filter(np.abs(resid), size=15) * 1.25 + 1e-6
    bad = resid > n_sigma * noise
    return ndimage.binary_dilation(bad, iterations=2)


def prepare(
    img: np.ndarray,
    moon: MoonFit,
    params: AlignParams,
    saturated: Optional[np.ndarray] = None,
    valid: Optional[np.ndarray] = None,
    defects: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (a', mask) for one window-sized image.

    The tangential high-pass is divided by the local (tangentially blurred)
    brightness, giving relative tangential contrast. That is the same for
    any exposure where the data are linear, and it stops the bright inner
    corona next to the Moon from dominating the correlation, which would
    otherwise lock onto the Moon's motion instead of the corona's.
    Pixels with too little signal for a useful contrast are masked.
    """
    h, w = img.shape
    if params.rho is None or params.mask_blur is None:
        params = scaled_params(params, sun_radius_from_moon(moon.radius))
    hp, back = tangential_highpass(img, moon.x, moon.y, params.rho, return_blur=True)
    noise = noise_sigma(img) if valid is None else noise_sigma(np.where(valid, img, np.median(img[valid]) if valid.any() else 0))
    floor = params.min_snr * noise
    hp = hp / np.maximum(back, floor)
    yy, xx = np.mgrid[0:h, 0:w]
    r_moon = np.hypot(xx - moon.x, yy - moon.y)
    r_edge = np.hypot(xx - (w - 1) / 2.0, yy - (h - 1) / 2.0)
    keep = (r_moon > moon.radius * (1 + params.moon_margin)) & (r_edge < 0.5 * min(h, w) * (1 - params.edge_margin))
    if saturated is not None:
        keep &= ~ndimage.binary_dilation(saturated, iterations=int(math.ceil(2 * params.rho)))
    if valid is not None:
        keep &= valid
    keep &= back > floor
    if defects is not None:
        keep &= ~defects
    mask = ndimage.gaussian_filter(keep.astype(np.float32), params.mask_blur)
    mask *= keep  # soft inside, exactly zero on removed pixels
    return hp * mask, mask


# ---------------------------------------------------------------------------
# Phase correlation
# ---------------------------------------------------------------------------


@dataclass
class Peak:
    x: float
    y: float
    height: float
    psr: float


def phase_correlation_surface(a: np.ndarray, b: np.ndarray, p_frac: float, sigma_frac: float) -> np.ndarray:
    """n_{p,q,sigma}(x, y) of the paper, with the zero shift at index (0, 0)."""
    A = sfft.fft2(a, workers=-1)
    B = sfft.fft2(b, workers=-1)
    absA, absB = np.abs(A), np.abs(B)
    p = p_frac * float(absA.max())
    q = p_frac * float(absB.max())
    N = (np.conj(A) * B) / ((absA + p) * (absB + q))
    h, w = a.shape
    fy = sfft.fftfreq(h) * h
    fx = sfft.fftfreq(w) * w
    sigma = sigma_frac * w
    G = np.exp(-0.5 * (fy[:, None] ** 2 + fx[None, :] ** 2) / sigma**2)
    return np.real(sfft.ifft2(N * G, workers=-1))


def find_peak(n: np.ndarray, epsilon: float, window: Optional[tuple[float, float, float]] = None) -> Peak:
    """Sub-pixel peak of a correlation surface. ``window`` = (x, y, radius): only look for the
    maximum within ``radius`` px of the shift (x, y) (the PSR is still judged on the whole surface)."""
    h, w = n.shape
    if not np.isfinite(n).all():
        n = np.nan_to_num(n, nan=0.0, posinf=0.0, neginf=0.0)
    if window is not None:
        ex, ey, rad = window
        ky_ = (np.arange(h) + h // 2) % h - h // 2  # signed shift of each row / column
        kx_ = (np.arange(w) + w // 2) % w - w // 2
        inside = (kx_[None, :] - ex) ** 2 + (ky_[:, None] - ey) ** 2 <= rad**2
        iy, ix = np.unravel_index(int(np.argmax(np.where(inside, n, -np.inf))), n.shape)
    else:
        iy, ix = np.unravel_index(int(np.argmax(n)), n.shape)
    e = int(math.ceil(epsilon))
    ky, kx = np.mgrid[-e : e + 1, -e : e + 1]
    sel = kx**2 + ky**2 < epsilon**2
    vals = n[(iy + ky) % h, (ix + kx) % w]
    # Centroid of the peak itself: subtract half the peak height first. A plain centroid over
    # radius epsilon is dominated by the surrounding noise floor when the peak is sharp, which
    # biases the result towards the whole-pixel maximum.
    vals = np.where(sel, np.clip(vals - 0.5 * n[iy, ix], 0.0, None), 0.0)
    m00 = vals.sum()
    dx = float((kx * vals).sum() / m00) if m00 > 0 else 0.0
    dy = float((ky * vals).sum() / m00) if m00 > 0 else 0.0
    x = ix + dx
    y = iy + dy
    # Signed shift (wrap to -N/2..N/2).
    if x > w / 2:
        x -= w
    if y > h / 2:
        y -= h
    # Peak-to-sidelobe ratio outside a small box around the peak.
    box = max(5, 3 * e)
    side = n.copy()
    yy = (iy + np.arange(-box, box + 1)) % h
    xx = (ix + np.arange(-box, box + 1)) % w
    side[np.ix_(yy, xx)] = np.nan
    mu, sd = float(np.nanmean(side)), float(np.nanstd(side)) + 1e-12
    return Peak(x, y, float(n[iy, ix]), (float(n[iy, ix]) - mu) / sd)


def measure_shift(a: np.ndarray, b: np.ndarray, params: AlignParams,
                  window: Optional[tuple[float, float, float]] = None) -> tuple[Peak, float]:
    """Shift of b relative to a (features at p in a are at p + shift in b); ``window`` limits
    where the peak may be (see :func:`find_peak`)."""
    if params.sigma_frac is not None:
        return (find_peak(phase_correlation_surface(a, b, params.p_frac, params.sigma_frac), params.epsilon, window),
                params.sigma_frac)
    # Automatic sigma. Small sigma gives a broad, safe but imprecise peak; large sigma a sharp,
    # precise one that can be fooled by noise. Take the largest sigma whose peak is confident
    # and agrees with the consensus position of all candidates.
    peaks = [(s, find_peak(phase_correlation_surface(a, b, params.p_frac, s), params.epsilon, window))
             for s in sorted(params.sigma_candidates)]
    xs = np.array([p.x for _, p in peaks])
    ys = np.array([p.y for _, p in peaks])
    cx, cy = float(np.median(xs)), float(np.median(ys))
    for s, pk in reversed(peaks):
        if pk.psr >= 8 and math.hypot(pk.x - cx, pk.y - cy) < 1.5:
            return pk, s
    s, pk = max(peaks, key=lambda sp: sp[1].psr)
    return pk, s


def measure_rotation_scale(a: np.ndarray, b: np.ndarray, params: AlignParams) -> tuple[float, float, float]:
    """(angle_deg, scale, psr) of b relative to a, from log-polar Fourier amplitudes."""
    h, w = a.shape
    win = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
    Aa = np.abs(sfft.fftshift(sfft.fft2(a * win, workers=-1)))
    Ab = np.abs(sfft.fftshift(sfft.fft2(b * win, workers=-1)))
    n = min(h, w)
    n_theta, n_r = 1024, 512
    r_min, r_max = 4.0, 0.45 * n
    log_base = math.log(r_max / r_min) / n_r
    theta = np.arange(n_theta) * (math.pi / n_theta)
    rad = r_min * np.exp(np.arange(n_r) * log_base)
    cy, cx = h / 2.0, w / 2.0
    ys = cy + np.outer(np.sin(theta), rad)
    xs = cx + np.outer(np.cos(theta), rad)
    # Emphasise mid frequencies; very low ones carry the radial gradient.
    lp_a = ndimage.map_coordinates(np.log1p(Aa), [ys, xs], order=1)
    lp_b = ndimage.map_coordinates(np.log1p(Ab), [ys, xs], order=1)
    lp_a -= lp_a.mean(axis=0, keepdims=True)
    lp_b -= lp_b.mean(axis=0, keepdims=True)
    surf = phase_correlation_surface(lp_a, lp_b, params.p_frac, 0.25)
    pk = find_peak(surf, params.epsilon)
    angle = pk.y * 180.0 / n_theta  # rows are theta
    scale = math.exp(pk.x * log_base)
    # Spectra rotate with the image; the image rotation is minus the spectrum shift here because
    # b's spectrum at angle t equals a's at t - alpha.
    return float(angle), float(scale), pk.psr


def warp(img: np.ndarray, tf: Transform, shape: Optional[tuple[int, int]] = None, order: int = 3,
         cval: float = 0.0) -> np.ndarray:
    """Resample a frame into reference coordinates."""
    M = tf.matrix()
    A = M[:, :2]
    t = M[:, 2]
    Ainv = np.linalg.inv(A)
    # scipy works in (row, col) = (y, x).
    mat = np.array([[Ainv[1, 1], Ainv[1, 0]], [Ainv[0, 1], Ainv[0, 0]]])
    off_xy = -Ainv @ t
    offset = np.array([off_xy[1], off_xy[0]])
    return ndimage.affine_transform(img, mat, offset=offset, output_shape=shape or img.shape, order=order,
                                    mode="constant", cval=cval, prefilter=order > 1)


def refine_rotation(ref: np.ndarray, img: np.ndarray, ref_mask: np.ndarray, centre: tuple[float, float],
                    params: AlignParams) -> tuple[float, float, float, int, float]:
    """Residual small rotation of ``img`` relative to ``ref`` about ``centre``.

    The shift is measured in sub-windows on a ring around the corona's
    centre; a rotation by delta moves a sub-window at offset d from the
    centre by delta * (-d_y, d_x). Fitting that to the measured shifts gives
    delta (and any common translation). Returns (delta_deg, tx, ty, windows used,
    rms misfit of the window shifts in px).
    """
    h, w = ref.shape
    size = sfft.next_fast_len(int(min(h, w) * 0.4))
    half = size // 2
    cx, cy = centre
    ring = 0.5 * min(h, w) * 0.42
    taper = np.outer(np.hanning(size), np.hanning(size)).astype(np.float32)
    rows, rhs = [], []
    for k in range(8):
        a = 2 * math.pi * k / 8
        px, py = cx + ring * math.cos(a), cy + ring * math.sin(a)
        x0, y0 = int(round(px)) - half, int(round(py)) - half
        if x0 < 0 or y0 < 0 or x0 + size > w or y0 + size > h:
            continue
        if float(ref_mask[y0 : y0 + size, x0 : x0 + size].mean()) < 0.15:
            continue
        ra = ref[y0 : y0 + size, x0 : x0 + size] * taper
        ia = img[y0 : y0 + size, x0 : x0 + size] * taper
        pk, _ = measure_shift(ra, ia, params)
        if pk.psr < 8:
            continue
        # Lever arm: where the window's information actually is, not its geometric centre.
        # Where both images carry the same structure: the locally smoothed product is positive
        # there, while uncorrelated noise averages out (|a*b| would favour noisy outer regions).
        energy = np.clip(ndimage.gaussian_filter((ra * ia).astype(np.float64), 3.0), 0, None)
        tot = float(energy.sum())
        if tot <= 0:
            continue
        gy, gx = np.mgrid[0:size, 0:size]
        ex = x0 + float((gx * energy).sum() / tot)
        ey = y0 + float((gy * energy).sum() / tot)
        dx, dy = ex - cx, ey - cy
        rows += [[1.0, 0.0, -dy], [0.0, 1.0, dx]]
        rhs += [pk.x, pk.y]
    if len(rows) < 6:
        return 0.0, 0.0, 0.0, len(rows) // 2, float("inf")
    A, y = np.array(rows), np.array(rhs)
    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    rms = float(np.sqrt(np.mean((A @ sol - y) ** 2)))
    # Features of img sit at ref + shift: to map img onto ref, rotate by -delta.
    return -math.degrees(sol[2]), float(sol[0]), float(sol[1]), len(rows) // 2, rms


def _rotate_about(tf: Transform, delta_deg: float, cx: float, cy: float) -> Transform:
    """Follow ``tf`` by a rotation of ``delta_deg`` about (cx, cy) in reference space."""
    d = math.radians(delta_deg)
    R = np.array([[math.cos(d), -math.sin(d)], [math.sin(d), math.cos(d)]])
    c = np.array([tf.centre_x, tf.centre_y])
    C = np.array([cx, cy])
    new_d = R @ (c + np.array([tf.dx, tf.dy]) - C) + C - c
    return Transform(float(new_d[0]), float(new_d[1]), tf.angle_deg + delta_deg, tf.scale, tf.centre_x, tf.centre_y)


def align_pair(ref: np.ndarray, ref_mask: np.ndarray, img: np.ndarray, img_mask: np.ndarray,
               params: AlignParams, centre: tuple[float, float],
               corona_centre: Optional[tuple[float, float]] = None,
               angle_deg: Optional[float] = None, refine: bool = False,
               expected: Optional[tuple[float, float, float]] = None) -> tuple[Transform, Peak, float]:
    """Transform taking ``img`` (prepared) onto ``ref`` (prepared).

    The rotation is either given (``angle_deg``, about ``corona_centre``),
    searched coarsely over any angle (``params.coarse_rotation``), and/or
    refined from local shifts (``refine``). The translation is always
    measured by phase correlation, last.
    """
    tf = Transform(0.0, 0.0, 0.0, 1.0, centre[0], centre[1])
    if params.coarse_rotation:
        ang, sc, rpsr = measure_rotation_scale(ref, img, params)
        # Amplitude spectra are symmetric, so the angle is known modulo 180 deg; small is expected.
        if ang > 90:
            ang -= 180
        if 0.9 < sc < 1.1 and rpsr > 8:
            tf = Transform(0.0, 0.0, -ang, 1.0 / sc, centre[0], centre[1])
    if angle_deg and corona_centre is not None:
        tf = _rotate_about(tf, angle_deg, *corona_centre)

    def shift_step(t: Transform) -> tuple[Transform, Peak, float]:
        moved = warp(img, t, order=1) if (t.angle_deg or t.scale != 1.0 or t.dx or t.dy) else img
        # ``expected`` = (dx, dy, tolerance) of the whole transform: the peak must lie near it.
        win = None if expected is None else (t.dx - expected[0], t.dy - expected[1], expected[2])
        pk, sig = measure_shift(ref, moved, params, win)
        # Features of ``moved`` sit at ref position + shift; move them back.
        return Transform(t.dx - pk.x, t.dy - pk.y, t.angle_deg, t.scale, t.centre_x, t.centre_y), pk, sig

    tf, pk, sig = shift_step(tf)
    if refine and corona_centre is not None:
        def residual(t: Transform) -> tuple[float, int]:
            d, _tx, _ty, used, _rms = refine_rotation(ref, warp(img, t, order=1), ref_mask, corona_centre, params)
            return d, used

        delta, used = residual(tf)
        for _ in range(4):
            if used < 3 or abs(delta) > params.max_rotation_deg or abs(delta) < 1e-3:
                break
            cand, cpk, csig = shift_step(_rotate_about(tf, delta, *corona_centre))
            cdelta, cused = residual(cand)
            # Accept while the residual rotation keeps shrinking (the iteration converges).
            if cused < 3 or abs(cdelta) >= abs(delta):
                break
            tf, pk, sig, delta, used = cand, cpk, csig, cdelta, cused
    return tf, pk, sig


# ---------------------------------------------------------------------------
# Sequences
# ---------------------------------------------------------------------------


@dataclass
class AlignFrame:
    """One frame as the aligner needs it: a 2-D linear image in a common window."""

    index: int
    image: np.ndarray  # linear, black-subtracted, scaled so 1.0 = full scale
    exposure: float  # relative exposure (time x ISO / f^2); used only to build the reference
    saturated: Optional[np.ndarray] = None
    moon: Optional[MoonFit] = None
    defects: Optional[np.ndarray] = None
    time_s: Optional[float] = None  # capture time, seconds from any origin (for the rotation drift model)


def align_sequence(
    frames: Sequence[AlignFrame],
    params: Optional[AlignParams] = None,
    expected_moon_radius: Optional[float] = None,
    progress: Optional[Callable[[str], None]] = None,
) -> list[FrameAlignment]:
    """Align every frame onto the middle exposure, growing an HDR reference as it goes."""
    params = params or AlignParams()
    say = progress or (lambda _m: None)
    frames = list(frames)
    if not frames:
        return []
    h, w = frames[0].image.shape
    centre = ((w - 1) / 2.0, (h - 1) / 2.0)
    nonfinite: dict[int, np.ndarray] = {}
    for f in frames:
        if f.image.shape != (h, w):
            raise ValueError("all frames must have the same size")
        bad = ~np.isfinite(f.image)
        if bad.any():
            # One NaN pixel turns every Fourier transform of the frame into NaN: no peak, shift 0.
            # Such pixels are left out like hot pixels (they say something upstream went wrong).
            if bad.mean() > 0.5:
                raise ValueError(f"frame {f.index}: {bad.mean():.0%} of its pixels are not numbers (NaN or "
                                 "infinite). Check the lens profile / vignetting correction and calibration "
                                 "frames on the Frames tab.")
            say(f"frame {f.index}: {int(bad.sum())} pixels are not numbers (NaN or infinite); left out")
            f.image = np.where(bad, 0.0, f.image).astype(np.float32)
            nonfinite[id(f)] = bad
        if f.saturated is None:
            f.saturated = f.image >= params.saturation
        if f.moon is None:
            f.moon = find_moon(f.image, expected_moon_radius, saturation=None)
        if f.defects is None:
            f.defects = defect_mask(f.image, params.star_sigma)
        if id(f) in nonfinite:
            f.defects = f.defects | ndimage.binary_dilation(nonfinite[id(f)], iterations=2)
    params = scaled_params(params, sun_radius_from_moon(float(np.median([f.moon.radius for f in frames]))))
    say(f"settings: rho {params.rho:.1f} px, mask blur {params.mask_blur:.1f} px, window {w} px")

    mid, sequence = visit_order([f.exposure for f in frames])
    order = sorted(range(len(frames)), key=lambda i: math.log(frames[i].exposure))
    log_e = {i: math.log(frames[i].exposure) for i in order}

    results: dict[int, FrameAlignment] = {
        mid: FrameAlignment(frames[mid].index, Transform(centre_x=centre[0], centre_y=centre[1]), 1.0, float("inf"), 0.0, 1.0,
                            frames[mid].moon.as_dict())
    }
    acc = _Accumulator((h, w), params)
    acc.add(frames[mid], results[mid].transform)

    corona_centre = (frames[mid].moon.x, frames[mid].moon.y)
    _cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def prepared(f: AlignFrame) -> tuple[np.ndarray, np.ndarray]:
        """A frame's prepared image never changes between passes; only the reference does."""
        key = id(f)
        if key not in _cache:
            _cache[key] = prepare(f.image, f.moon, params, saturated=f.saturated, defects=f.defects)
        return _cache[key]
    model = params.rotation_model if params.solve_rotation else "none"
    rotation: dict[int, float] = {i: 0.0 for i in order}
    rot_notes: list[str] = []

    # Where the corona can be: the Moons' offset from the anchor, give or take how far the Moon moves
    # against the Sun during totality (well under a quarter of its radius in any sequence).
    anchor_moon = frames[mid].moon
    tol = max(10.0, 0.25 * anchor_moon.radius)
    in_acc: set[int] = {mid}

    def align_one(i: int, pass_label: str, refine: bool) -> None:
        f = frames[i]
        if i in in_acc:
            acc.remove(f, results[i].transform)
            in_acc.discard(i)
        ref_img, ref_valid, ref_moon = acc.reference(frames[mid].moon)
        ref_p, ref_m = prepare(ref_img, ref_moon, params, valid=ref_valid)
        img_p, img_m = prepared(f)
        expected = (anchor_moon.x - f.moon.x, anchor_moon.y - f.moon.y, tol)
        tf, pk, sig = align_pair(ref_p, ref_m, img_p, img_m, params, centre, corona_centre=corona_centre,
                                 angle_deg=rotation[i], refine=refine, expected=expected)
        overlap = float(((ref_m > 0.5) & (warp(img_m, tf, order=1) > 0.5)).mean())
        notes = []
        near = math.hypot(tf.dx - expected[0], tf.dy - expected[1]) <= 1.5 * tol
        ok = np.isfinite(pk.psr) and pk.psr >= 5 and overlap >= 0.02 and near
        if not np.isfinite(pk.psr) or overlap < 0.02:
            notes.append("little usable overlap with the reference")
        elif pk.psr < 8:
            notes.append("weak correlation peak; check this frame")
        if not ok:
            # A failed frame must not spoil the reference for all the others (one frame placed
            # hundreds of px away once grew the reference's Moon mask over the whole window,
            # and every later frame failed). Use the Moon's offset instead, and leave it out.
            tf = Transform(expected[0], expected[1], tf.angle_deg, 1.0, tf.centre_x, tf.centre_y)
            notes.append("alignment failed: placed by its Moon only and left out of the reference; "
                         "exclude it if it is a partial-phase or poor frame")
        results[i] = FrameAlignment(f.index, tf, pk.height, pk.psr, sig, overlap, f.moon.as_dict(), notes)
        if ok:
            acc.add(f, tf)
            in_acc.add(i)
        say(f"{pass_label}: frame {f.index} shift ({tf.dx:+.2f}, {tf.dy:+.2f}) rot {tf.angle_deg:+.3f} deg, psr {pk.psr:.1f}")

    # Pass 1: translation only, growing the reference outwards from the anchor.
    for i in sequence[1:]:
        align_one(i, "pass 1", refine=False)

    anchor = frames[mid]
    anchor_p, anchor_m = prepared(anchor)

    def measure_rotations() -> dict[int, tuple[float, int, float]]:
        """Each frame's residual rotation against the anchor frame itself.

        A composite reference smears the fine tangential structure that the
        rotation measurement relies on, so frames are compared with the single
        anchor exposure. Frames too far from it in exposure share little
        signal; their measurements get low weight and the drift model fills in.
        """
        meas = {}
        for i in sequence[1:]:
            f = frames[i]
            if round(abs(log_e[i] - log_e[mid]) / math.log(2), 6) > params.rotation_max_stops:
                continue
            img_p, _ = prepared(f)
            d, _tx, _ty, used, rms = refine_rotation(anchor_p, warp(img_p, results[i].transform, order=1), anchor_m,
                                                     corona_centre, params)
            if used >= 3 and math.isfinite(rms) and rms < 2.0:
                meas[i] = (d, used, rms)
            say(f"  rotation measured: frame {f.index} {d:+.4f} deg from {used} windows, misfit {rms:.2f} px")
        return meas

    # Rotation rounds: measure, model, apply, re-align; repeat until the model stops changing.
    if model != "none":
        for rnd in range(params.rotation_rounds):
            meas = measure_rotations()
            if model == "per-frame":
                update = {i: m[0] for i, m in meas.items()}
            else:
                notes_round: list[str] = []
                update = _drift_model(frames, mid, meas, notes_round)
                rot_notes = notes_round  # the last round's verdict; see _rotation_summary below
            for i, dlt in update.items():
                rotation[i] += dlt
            say("rotation: " + ", ".join(f"{frames[i].index}:{rotation[i]:+.3f}" for i in sequence))
            for i in sequence[1:]:
                align_one(i, f"rotation round {rnd + 1}", refine=False)
            if not update or max(abs(v) for v in update.values()) < 0.002:
                break
    # Final passes: every frame against the composite of all the others.
    for pass_no in range(1, max(2, params.passes)):
        for i in sequence[1:]:
            align_one(i, f"pass {pass_no + 1}", refine=False)
    # Re-anchor. Every frame was last measured against the composite of all the others, so the
    # non-anchor frames can agree with each other yet drift together away from the anchor (it is
    # only one voice in the composite). Measure where the anchor sits against the composite of the
    # others and remove that common offset, so the anchor frame really is the reference.
    if len(sequence) > 1:
        acc.remove(frames[mid], results[mid].transform)
        ref_img, ref_valid, ref_moon = acc.reference(frames[mid].moon)
        ref_p, ref_m = prepare(ref_img, ref_moon, params, valid=ref_valid)
        tf_a, pk_a, _ = align_pair(ref_p, ref_m, anchor_p, anchor_m, params, centre, corona_centre=corona_centre)
        if pk_a.psr >= 8:
            # tf_a maps anchor -> composite space; bring every other frame into anchor space.
            back = tf_a.inverse()
            for i in sequence[1:]:
                results[i].transform = results[i].transform.then(back)
            say(f"re-anchored: composite offset from the anchor ({tf_a.dx:+.3f}, {tf_a.dy:+.3f}) px, "
                f"{tf_a.angle_deg:+.4f} deg")
        acc.add(frames[mid], results[mid].transform)
    if model == "drift":
        rot_notes = _rotation_summary(frames, mid, rotation, rot_notes)
        if rot_notes:
            say(rot_notes[0])
    for i in sequence[1:]:
        results[i].notes += rot_notes
    return [results[i] for i in range(len(frames))]


def _rotation_summary(frames, mid: int, rotation: dict[int, float], last: list[str]) -> list[str]:
    """The note for the rotation the frames actually carry.

    Each round measures what is left after the previous rounds' correction, so the last round
    usually finds nothing significant; its note alone ("not applied") would hide a rotation
    applied earlier (seen on the 2024 set).
    """
    if not any(abs(v) > 1e-9 for v in rotation.values()):
        return last
    times = {i: frames[i].time_s for i in rotation}
    pairs = [(times[i] - times[mid], v) for i, v in rotation.items()
             if times[i] is not None and times[mid] is not None and i != mid]
    if not pairs:
        return last
    t = np.array([p[0] for p in pairs], dtype=float)
    v = np.array([p[1] for p in pairs], dtype=float)
    if float(np.sum(t * t)) <= 0:
        return last
    slope = float(np.sum(t * v) / np.sum(t * t))
    span = float(np.ptp([x for x in times.values() if x is not None]))
    note = f"field rotation {slope * 60:+.4f} deg/min applied ({slope * span:+.4f} deg over the sequence)"
    rest = [n for n in last if "not significant" in n]
    if rest:
        m = rest[0]
        inner = m[m.find("(") + 1:m.find(")")] if "(" in m else ""
        note += f"; what was left after correcting it is not significant ({inner})"
    return [note]


def _drift_model(frames, mid: int, meas: dict, notes: list[str]) -> dict[int, float]:
    """Fit rotation = a + b * time through the reliable measurements (anchor fixed at 0).

    Field rotation on a tracking mount changes smoothly with time; single
    frames (very short or very long exposures) can give poor measurements,
    so a robust straight-line fit is used for every frame.
    """
    times = {i: frames[i].time_s for i in range(len(frames))}
    if not meas:
        notes.append("rotation could not be measured; assumed zero")
        return {}
    have_time = all(times[i] is not None for i in list(meas) + [mid])
    idx = list(meas)
    d = np.array([meas[i][0] for i in idx])
    wts = np.array([meas[i][1] / (meas[i][2] + 0.05) for i in idx])
    if have_time and len(idx) >= 3:
        t = np.array([times[i] - times[mid] for i in idx], dtype=float)
        keep = np.ones(len(idx), bool)
        slope = 0.0
        for _ in range(5):
            # Anchor has zero rotation by definition: fit d = slope * t (through the origin).
            ww = wts * keep
            slope = float(np.sum(ww * t * d) / max(np.sum(ww * t * t), 1e-12))
            res = d - slope * t
            mad = 1.4826 * np.median(np.abs(res[keep])) + 1e-4
            new = np.abs(res) < 3 * mad + 0.02
            if np.array_equal(new, keep):
                break
            keep = new
        # Uncertainty of the slope from the scatter about the fit.
        ww = wts * keep
        res = (d - slope * t)[keep]
        dof = max(1, int(keep.sum()) - 1)
        scatter = math.sqrt(float(np.sum(ww[keep] * res**2) / max(np.sum(ww[keep]), 1e-12)) * keep.sum() / dof)
        slope_err = scatter / math.sqrt(max(float(np.sum((t[keep]) ** 2)), 1e-12))
        span = float(np.ptp([times[i] for i in range(len(frames)) if times[i] is not None]))
        if abs(slope) < 3 * slope_err:
            notes.append(
                f"field rotation not significant ({slope * span:+.4f} +/- {slope_err * span:.4f} deg over the sequence); not applied"
            )
            return {}
        notes.append(
            f"field rotation {slope * 60:+.4f} deg/min ({slope * span:+.4f} +/- {slope_err * span:.4f} deg over the sequence)"
        )
        return {i: slope * (times[i] - times[mid]) for i in range(len(frames)) if times[i] is not None and i != mid}
    # No times: a constant rotation relative to the anchor would be unphysical; use the median offset.
    med = float(np.median(d))
    notes.append(f"no capture times; rotation set to the median measured value {med:+.4f} deg")
    return {i: med for i in range(len(frames)) if i != mid}


def visit_order(exposures: list[float]) -> tuple[int, list[int]]:
    """The middle exposure, and the order frames are aligned in: outward from it by distance in
    stops, the shorter first where two are as far (rounded to a millionth of a stop, so the order
    does not hang on rounding noise and is the same for any common scale of the exposures, such as
    an f-number that is not recorded)."""
    log_e = [math.log(e) for e in exposures]
    order = sorted(range(len(exposures)), key=lambda i: log_e[i])
    mid = order[len(order) // 2]
    return mid, sorted(order, key=lambda i: (round(abs(log_e[i] - log_e[mid]) / math.log(2), 6),
                                             round((log_e[i] - log_e[mid]) / math.log(2), 6), i))


class _Accumulator:
    """Running exposure-normalised weighted sum of aligned frames."""

    def __init__(self, shape: tuple[int, int], params: AlignParams):
        self.sum = np.zeros(shape, np.float64)
        self.weight = np.zeros(shape, np.float64)
        self.params = params
        self.moons: list[tuple[float, float, float]] = []

    def _parts(self, f: AlignFrame, tf: Transform):
        h, w = f.image.shape
        yy, xx = np.mgrid[0:h, 0:w]
        wgt = (~f.saturated).astype(np.float32)
        if f.defects is not None:
            wgt *= ~f.defects
        wgt *= np.hypot(xx - f.moon.x, yy - f.moon.y) > f.moon.radius * (1 + self.params.moon_margin)
        # Favour well-exposed pixels: weight rises with signal up to 10 % of full scale.
        wgt *= np.clip(f.image / 0.1, 0.0, 1.0)
        img_w = warp(f.image / f.exposure, tf, order=3)
        wgt_w = warp(wgt, tf, order=1)
        return img_w * wgt_w, wgt_w

    def add(self, f: AlignFrame, tf: Transform) -> None:
        s, wgt = self._parts(f, tf)
        self.sum += s
        self.weight += wgt
        mx, my = tf.apply_point(f.moon.x, f.moon.y)
        self.moons.append((mx, my, f.moon.radius))

    def remove(self, f: AlignFrame, tf: Transform) -> None:
        s, wgt = self._parts(f, tf)
        self.sum -= s
        self.weight -= wgt
        np.maximum(self.weight, 0, out=self.weight)

    def reference(self, anchor_moon: MoonFit):
        valid = self.weight > 1e-3
        img = np.where(valid, self.sum / np.maximum(self.weight, 1e-6), 0.0).astype(np.float32)
        xs = [m[0] for m in self.moons] or [anchor_moon.x]
        ys = [m[1] for m in self.moons] or [anchor_moon.y]
        rs = [m[2] for m in self.moons] or [anchor_moon.radius]
        moon = MoonFit(float(np.mean(xs)), float(np.mean(ys)), float(np.mean(rs)), 0.0, 0, 0)
        # The masked Moon must cover every frame's Moon: grow it by the spread of positions.
        spread = float(np.max(np.hypot(np.array(xs) - moon.x, np.array(ys) - moon.y))) if len(xs) > 1 else 0.0
        moon.radius += spread
        return img, valid, moon
