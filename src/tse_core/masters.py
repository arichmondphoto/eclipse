"""Master bias, dark and flat frames for Stack and the earthshine stack.

The calibration plan (``calibration.plan_calibration``) decides which bias, dark and flat
group each light frame uses; this module reads those frames, averages them into masters and
attaches them to each frame's :class:`~tse_core.calibration.FrameCalibration`, where
``rawio.region`` applies them to the sensor data:

* **Offset** (subtracted in raw ADU): the matching master dark (it contains the bias), else the
  master bias for the ISO, else the black level from the RAW file.
* **Flat**, one of three ways (``FLAT_MODES``):

  - ``master``: divide by the master flat, pixel by pixel. Corrects vignetting, dust shadows
    and pixel-to-pixel response, provided the dust has not moved since the flats were taken.
  - ``profile``: fit a smooth vignetting profile to the flats and divide by that. Corrects
    vignetting only; dust in the flats is ignored.
  - ``compare``: as ``master``, and also report (and save as a FITS image) how far the master
    flat differs from the smooth profile, so dust or a bad flat can be seen.

Each master is the mean of its frames (bias, darks and flats here are fixed patterns, so a mean
gives the lowest noise). Every flat has its own offset removed and is scaled to 1.0 in the
middle of the frame, per CFA position, before averaging, so flats of slightly different
brightness combine correctly and the colour balance of the lights is unchanged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .calibration import CalibrationFrames, CalibrationPlan, describe_dark_key
from .frames import FrameMeta
from .lensfun import MIN_ILLUMINATION

FLAT_MODES = ("master", "profile", "compare")
FLAT_MODE_HELP = {
    "master": "divide by the master flat (vignetting, dust and pixel response)",
    "profile": "divide by a smooth profile fitted to the flats (vignetting only; dust ignored)",
    "compare": "master flat, and report how it differs from the smooth profile",
}

Say = Callable[[str], None]


@dataclass
class FlatReport:
    """How the master flat of one group compares with its smooth profile."""

    rms_pct: float  # rms of (master / profile - 1), in %
    dark_pct: float  # share of pixels more than 2% darker than the profile (dust shadows)
    deepest_pct: float  # how much darker the deepest spot is, in %
    path: Optional[str] = None  # FITS image of master / profile


def _cfa_positions():
    return ((0, 0), (0, 1), (1, 0), (1, 1))


def _centre_norm(plane: np.ndarray) -> float:
    """Level in the middle of a CFA sub-plane (median within 10% of the half-diagonal)."""
    h, w = plane.shape
    yy, xx = np.ogrid[0:h, 0:w]
    near = np.hypot(xx - (w - 1) / 2, yy - (h - 1) / 2) < 0.1 * math.hypot(w, h) / 2
    vals = plane[near] if near.sum() >= 4 else plane
    return float(np.median(vals))


def _mean(paths: list[str], load, offset_of=None, normalise=False, info: Optional[dict] = None) -> tuple[np.ndarray, int]:
    """Mean of the frames' raw data (minus each one's offset; scaled to 1 in the middle). ``info``
    receives the colour-filter pattern."""
    total, shape, n = None, None, 0
    for p in paths:
        mos = load(p)
        if info is not None and "pattern" not in info:
            info["pattern"] = mos.pattern
        data = mos.bayer.astype(np.float64)
        if shape is None:
            shape = data.shape
        elif data.shape != shape:
            raise ValueError(f"{Path(p).name} is {data.shape[1]} x {data.shape[0]} px, the others "
                             f"{shape[1]} x {shape[0]}; calibration frames must all be the same size")
        if offset_of is not None:
            data = data - offset_of(mos)
        if normalise:
            for dy, dx in _cfa_positions():
                c = _centre_norm(data[dy::2, dx::2])
                if not c > 0:
                    raise ValueError(f"{Path(p).name} is black in the middle; it cannot be used as a flat")
                data[dy::2, dx::2] /= c
        total = data if total is None else total + data
        n += 1
    return (total / max(n, 1)).astype(np.float32), n


MASTER_FORMAT = 1  # raise when the way a master is made changes, so saved ones are built again


class MasterCache:
    """Masters saved in a folder (FITS, so they open in Siril) and read back while their frames stay
    the same: each carries a fingerprint of its files (name, size and time modified, so a file
    added, removed or replaced changes it) and of how it was made."""

    def __init__(self, folder: str | Path):
        self.folder = Path(folder)

    @staticmethod
    def fingerprint(paths: list[str], how: str = "") -> Optional[str]:
        import hashlib
        import json
        import os

        try:
            files = sorted((str(Path(p).resolve()), os.stat(p).st_size, os.stat(p).st_mtime_ns) for p in paths)
        except OSError:
            return None
        text = json.dumps([MASTER_FORMAT, how, files])
        return hashlib.sha1(text.encode()).hexdigest()

    def _stem(self, kind: str, key: str) -> str:
        import re

        return f"master_{kind}_{re.sub(r'[^A-Za-z0-9._-]', 'x', key)}_"

    def get(self, kind: str, key: str, fp: Optional[str]):
        if fp is None:
            return None
        p = self.folder / f"{self._stem(kind, key)}{fp[:12]}.fit"
        if not p.exists():
            return None
        from .fits import read_fits

        try:
            data, header = read_fits(p)
        except (OSError, ValueError):
            return None
        if header.get("TSEHASH") != fp:
            return None
        return (np.ascontiguousarray(data, dtype=np.float32), int(header.get("NFRAMES", 0)), p,
                str(header.get("BAYERPAT", "")).strip())

    def put(self, kind: str, key: str, fp: Optional[str], data: np.ndarray, n: int, pattern: str = "") -> Optional[Path]:
        if fp is None:
            return None
        from .fits import write_fits

        try:
            self.folder.mkdir(parents=True, exist_ok=True)
            stem = self._stem(kind, key)
            for old in self.folder.glob(f"{stem}*.fit"):  # the same group made from other files
                old.unlink()
            p = self.folder / f"{stem}{fp[:12]}.fit"
            tmp = p.with_suffix(".tmp")
            write_fits(tmp, data, {"IMAGETYP": f"Master {kind}", "NFRAMES": int(n), "TSEHASH": fp,
                                   "BAYERPAT": pattern or "",
                                   "COMMENT": "Made by Totality; built again when its frames change"})
            tmp.replace(p)
            return p
        except OSError:
            return None


def _mean_cached(cache: Optional[MasterCache], kind: str, key: str, paths: list[str], load, how: str = "",
                 info: Optional[dict] = None, **kw) -> tuple[np.ndarray, int, str]:
    """``_mean`` of the frames, read back from ``cache`` while they are unchanged. Returns (master,
    number of frames, what happened, for the log); ``info`` receives the colour-filter pattern."""
    info = {} if info is None else info
    fp = MasterCache.fingerprint(paths, how) if cache is not None else None
    hit = cache.get(kind, key, fp) if cache is not None else None
    if hit is not None and hit[3]:
        data, n, p, info["pattern"] = hit
        return data, n or len(paths), f"reused from {p.parent.name}/{p.name}"
    data, n = _mean(paths, load, info=info, **kw)
    p = cache.put(kind, key, fp, data, n, info.get("pattern", "")) if cache is not None else None
    return data, n, (f"saved as {p.parent.name}/{p.name} for next time" if p is not None else "")


def _black_array(mos) -> np.ndarray:
    """The RAW file's black level per CFA position, as an array the size of the frame."""
    out = np.empty(mos.bayer.shape, np.float32)
    for i, (dy, dx) in enumerate(_cfa_positions()):
        out[dy::2, dx::2] = mos.black.get(mos.pattern[i], 0.0)
    return out


def _flat_label(key: str) -> str:
    # "f8.0_fl400" -> "f/8, 400 mm"; "f?_fl?" -> "f-number and focal length not recorded"
    try:
        f, fl = key[1:].split("_fl")
        if f == "?" and fl == "?":
            return "f-number and focal length not recorded"
        return f"{'f-number not recorded' if f == '?' else f'f/{float(f):g}'}, " \
               f"{'focal length not recorded' if fl == '?' else fl + ' mm'}"
    except ValueError:
        return key


def _profile_of(flat: np.ndarray, pattern: str, image_crop: float, meta: Optional[FrameMeta] = None,
                with_map: bool = True):
    """A smooth profile fitted to a master flat, and its map on the sensor."""
    from .vignetting_profile import fit_profile_from_planes, profile_map

    h, w = flat.shape
    planes = {}
    for i, (dy, dx) in enumerate(_cfa_positions()):
        name = pattern[i] if pattern[i] not in planes else pattern[i] + "2"
        planes[name] = (flat[dy::2, dx::2], 2.0)
    prof = fit_profile_from_planes([planes], w, h, crop_factor=image_crop,
                                   lens_model=meta.lens_model if meta else "",
                                   focal_length_mm=float(meta.focal_length_mm or 0) if meta else 0.0,
                                   aperture_f=float(meta.aperture_f or 0) if meta else 0.0)
    return prof, (profile_map(prof, w, h, image_crop, pattern) if with_map else None)


def _light_slope(flat: np.ndarray, prof, image_crop: float, step: int = 16) -> tuple[float, float]:
    """The straight slope of the light across a master flat, beyond the lens's smooth profile:
    (bx, by) = relative change from the middle to the right and the bottom edge."""
    from .lensfun import vignetting_map

    h, w = flat.shape
    hh, ww = h // step * step, w // step * step
    small = flat[:hh, :ww].reshape(hh // step, step, ww // step, step).mean(axis=(1, 3)).astype(np.float64)
    sh, sw = small.shape
    pm = vignetting_map(sw, sh, tuple(prof.terms), image_crop, prof.crop_factor).astype(np.float64)
    ratio = (small / np.maximum(pm, MIN_ILLUMINATION)).ravel()
    yy, xx = np.mgrid[0:sh, 0:sw]
    xn = ((xx - (sw - 1) / 2) / max((sw - 1) / 2, 1)).ravel()
    yn = ((yy - (sh - 1) / 2) / max((sh - 1) / 2, 1)).ravel()
    design = np.column_stack([np.ones_like(xn), xn, yn])
    keep = np.isfinite(ratio)
    coef = np.array([1.0, 0.0, 0.0])
    for _ in range(6):  # dust and hot spots are left out of the fit
        coef, *_ = np.linalg.lstsq(design[keep], ratio[keep], rcond=None)
        res = ratio - design @ coef
        sigma = 1.4826 * float(np.median(np.abs(res[keep]))) + 1e-6
        new = np.isfinite(ratio) & (np.abs(res) < 3 * sigma)
        if np.array_equal(new, keep):
            break
        keep = new
    return float(coef[1] / coef[0]), float(coef[2] / coef[0])


def _slope_words(bx: float, by: float) -> str:
    parts = []
    if abs(bx) > 0.002:
        parts.append(f"brighter on the {'right' if bx > 0 else 'left'} ({abs(bx) * 100:.1f}% at the edge)")
    if abs(by) > 0.002:
        parts.append(f"brighter at the {'bottom' if by > 0 else 'top'} ({abs(by) * 100:.1f}% at the edge)")
    return " and ".join(parts)


def attach_masters(
    plan: CalibrationPlan,
    calib: CalibrationFrames,
    load: Callable,
    lights: list[FrameMeta],
    flat_mode: str = "master",
    image_crop: float = 1.0,
    say: Optional[Say] = None,
    out_dir: Optional[str | Path] = None,
    name: str = "stack",
    cache_dir: Optional[str | Path] = None,
) -> dict:
    """Build the masters the plan calls for and attach them to its frames.

    Returns what was done, for the project file: ``{"flat_mode", "flat_vs_profile"?}``.
    """
    say = say or (lambda _m: None)
    cache = MasterCache(cache_dir) if cache_dir is not None else None

    def note(what: str) -> str:
        return f"; {what}" if what else ""

    if flat_mode not in FLAT_MODES:
        raise ValueError(f"flat mode must be one of {', '.join(FLAT_MODES)}")
    from .calibration import _dark_key, _flat_key, _iso_key

    by_index = {fc.frame_index: fc for fc in plan.frames}
    meta_by_index = {f.index: f for f in lights}
    used_iso = {_iso_key(meta_by_index[fc.frame_index]) for fc in plan.frames
                if fc.bias == "frames" and not fc.dark_group and fc.frame_index in meta_by_index}
    out: dict = {"flat_mode": flat_mode}

    # Bias: needed for lights without darks, and for the flats.
    bias: dict[str, np.ndarray] = {}
    flat_isos = {_iso_key(f) for f in calib.flats}
    for key, paths in plan.bias_groups.items():
        if key in used_iso or key in flat_isos:
            bias[key], n, what = _mean_cached(cache, "bias", key, paths, load)
            say(f"master bias {key.replace('iso', 'ISO ')}: mean of {n} bias frame(s){note(what)}")

    # Darks.
    darks: dict[str, np.ndarray] = {}
    for key in sorted({fc.dark_group for fc in plan.frames if fc.dark_group}):
        darks[key], n, what = _mean_cached(cache, "dark", key, plan.dark_groups[key], load)
        say(f"master dark {describe_dark_key(key)}: mean of {n} dark frame(s){note(what)}")

    # Flats: each one's own offset (its ISO's master bias, else its black level) removed.
    flat_meta = {f.path: f for f in calib.flats}

    flats: dict[str, np.ndarray] = {}
    profiles: dict[str, tuple] = {}
    for key in sorted({fc.flat_group for fc in plan.frames if fc.flat_group}):
        paths = plan.flat_groups[key]
        isos = {_iso_key(flat_meta[p]) for p in paths if p in flat_meta}
        biased = [k for k in isos if k in bias]
        if biased and len(isos) == 1:
            b = bias[biased[0]]
            offset = lambda mos, b=b: b  # noqa: E731
            how = "master bias"
            made = "bias " + str(MasterCache.fingerprint(plan.bias_groups[biased[0]]))
        else:
            offset = _black_array
            how = "black level"
            made = "black level"
        info: dict = {}
        flat, n, what = _mean_cached(cache, "flat", key, paths, load, how=made, info=info, offset_of=offset,
                                     normalise=True)
        pattern = info["pattern"]
        label = _flat_label(key)
        prof, _ = _profile_of(flat, pattern, image_crop, flat_meta.get(paths[0]), with_map=False)
        profiles[key] = prof
        if flat_mode == "profile":
            corner = prof.corner(image_crop)[0]
            say(f"flats {label}: smooth profile fitted to {n} flat(s) (corners at {corner:.0%} of the middle); "
                "dust in the flats is ignored")
        else:
            # A brightness slope straight across the frame comes from the light source (a panel or the
            # sky brighter on one side), not the lens: take it out, keeping vignetting and dust.
            bx, by = _light_slope(flat, prof, image_crop)
            if max(abs(bx), abs(by)) > 0.002:
                h, w = flat.shape
                xn = ((np.arange(w, dtype=np.float32) - (w - 1) / 2) / ((w - 1) / 2))[None, :]
                yn = ((np.arange(h, dtype=np.float32) - (h - 1) / 2) / ((h - 1) / 2))[:, None]
                flat /= 1 + bx * xn + by * yn
                say(f"flats {label}: the light on the flats was uneven: " + _slope_words(bx, by)
                    + "; taken out of the master flat (it is the light, not the lens)")
            np.maximum(flat, MIN_ILLUMINATION, out=flat)
            flats[key] = flat
            say(f"flats {label}: master flat, mean of {n} flat(s) ({how} removed); corrects vignetting and dust"
                + note(what))
        if flat_mode == "compare":
            from scipy import ndimage

            from .vignetting_profile import profile_map

            h, w = flat.shape
            ratio = flat / np.maximum(profile_map(prof, w, h, image_crop, pattern), MIN_ILLUMINATION)
            rms = float(np.sqrt(np.mean((ratio - 1.0) ** 2))) * 100
            # Dust: small dark spots, judged against the ratio's own large-scale level (so a broad
            # difference from the profile is not counted as dust).
            size = max(31, min(h, w) // 10)
            local = ndimage.uniform_filter(ratio, size)
            sm = ndimage.uniform_filter(ratio, 5) / np.maximum(local, 1e-3)
            dark = float(np.mean(sm < 0.98)) * 100
            deepest = float(max(0.0, 1.0 - sm.min())) * 100
            path = None
            if out_dir is not None:
                from .fits import write_fits

                path = str(Path(out_dir) / f"{name}_flat_vs_profile_{key}.fit")
                write_fits(path, ratio, {"IMAGETYP": "Flat / smooth profile",
                                         "COMMENT": "1.0 = as the smooth profile; dark spots are dust"})
                out["flat_vs_profile"] = path
            say(f"flats {label}: the master flat differs from the smooth profile by {rms:.2f}% rms; "
                f"{dark:.2f}% of pixels are in dark spots more than 2% deep (dust); the deepest is {deepest:.0f}% "
                "darker" + (f"; saved as {Path(path).name}" if path else ""))
            del ratio, local, sm
            if deepest > 30 or rms > 3:
                say("  warning: that is a large difference; check the flats (dust moved, uneven light, "
                    "or a different lens setting) before trusting the master flat")

    # Attach.
    for fc in plan.frames:
        f = meta_by_index.get(fc.frame_index)
        if fc.dark_group and fc.dark_group in darks:
            fc.offset_map = darks[fc.dark_group]
            fc.offset_label = f"master dark {describe_dark_key(fc.dark_group)}"
        elif fc.bias == "frames" and f is not None and _iso_key(f) in bias:
            fc.offset_map = bias[_iso_key(f)]
            fc.offset_label = "master bias"
        if fc.flat_group:
            if fc.flat_group in flats:
                fc.flat_map = flats[fc.flat_group]
                fc.lens_label = f"master flat ({_flat_label(fc.flat_group)})"
            else:
                prof = profiles[fc.flat_group]
                fc.vignetting = "profile"
                fc.lens_label = f"smooth profile from flats ({_flat_label(fc.flat_group)})"
                fc.lens_terms = prof.terms
                fc.channel_terms = dict(prof.channel_terms) or None
                fc.profile_crop = image_crop
    return out


def describe_frames(plan: CalibrationPlan) -> list[str]:
    """Log lines: which correction the frames got."""
    from collections import Counter

    vign = Counter()
    offs = Counter()
    for fc in plan.frames:
        if fc.vignetting == "flat" and fc.flat_map is not None:
            vign[fc.lens_label] += 1
        elif fc.vignetting == "profile":
            label = fc.lens_label
            vign[label if label.startswith("smooth profile") else f"saved own-flats profile {label}"] += 1
        elif fc.vignetting == "lensfun":
            vign[f"lensfun profile {fc.lens_label}"] += 1
        else:
            vign["NO vignetting correction"] += 1
        offs[fc.offset_label or "black level from the RAW file"] += 1
    lines = ["vignetting: " + "; ".join(f"{n} frame(s) by {k}" for k, n in vign.items()),
             "dark/bias: " + "; ".join(f"{n} frame(s) by {k}" for k, n in offs.items())]
    return lines


def flat_mismatch(lights: list[FrameMeta], calib: CalibrationFrames) -> Optional[str]:
    """Why flats were given but no light frame can use them, or None."""
    from .calibration import _flat_key

    if not calib.flats:
        return None
    fk = {_flat_key(f) for f in calib.flats}
    lk = {_flat_key(f) for f in lights}
    if fk & lk:
        return None
    return (f"your flats ({', '.join(sorted(_flat_label(k) for k in fk if k))}) do not match the light frames "
            f"({', '.join(sorted(_flat_label(k) for k in lk if k))}): flats must be taken at the same aperture "
            "and focal length; they were not used")


def correction_preview(fc, size: tuple[int, int], image_crop: float = 1.0, step: int = 8):
    """The vignetting or flat correction a frame gets (what it is divided by; 1.0 = none), reduced
    ``step`` times, and a line for the log. Returns (image or None, text)."""
    from .lensfun import vignetting_map

    w, h = size
    if fc is not None and fc.flat_map is not None:
        f = fc.flat_map
        hh, ww = f.shape[0] // step * step, f.shape[1] // step * step
        small = f[:hh, :ww].reshape(hh // step, step, ww // step, step).mean(axis=(1, 3))
        what = fc.lens_label or "master flat"
    elif fc is not None and fc.vignetting in ("profile", "lensfun") and fc.lens_terms is not None:
        small = vignetting_map(max(w // step, 8), max(h // step, 8), tuple(fc.lens_terms), image_crop, fc.profile_crop)
        what = fc.lens_label if fc.vignetting == "profile" else f"lensfun profile {fc.lens_label}"
    else:
        return None, "vignetting correction: NONE applied (no flats, own-flats profile or lensfun profile matched)"
    sh, sw = small.shape
    k = max(2, min(sh, sw) // 20)
    centre = float(np.median(small[sh // 2 - k:sh // 2 + k, sw // 2 - k:sw // 2 + k]))
    corners = float(np.median([np.median(small[:k, :k]), np.median(small[:k, -k:]),
                               np.median(small[-k:, :k]), np.median(small[-k:, -k:])]))
    rel = corners / centre if centre > 0 else float("nan")
    text = (f"vignetting correction ({what}): the corners are {rel:.0%} as bright as the middle "
            f"({math.log2(rel):+.2f} stops) and are brightened by that")
    return (small / centre).astype(np.float32), text
