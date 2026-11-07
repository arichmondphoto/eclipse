"""Decide how each light frame is calibrated.

Rules, from the development plan:

* **Vignetting**, first match wins and only one is ever applied:
  a profile the user chose explicitly; the user's own master flat if one
  matches the frame's aperture and focal length; a saved profile fitted
  from the user's flats (``totality flat-profile``) for the same lens and
  settings; a lensfun profile; otherwise none.
* **Darks**: matched to each light by exposure time and ISO. Without a
  match, hot pixels are removed by cosmetic correction instead.
* **Bias**: matched by ISO. Without bias frames, the black level read from
  each RAW file is used as a synthetic bias.

The result is a :class:`CalibrationPlan`, which the Siril script carries out.
Nothing here touches Siril.
"""

from __future__ import annotations

import math
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .fits import write_fits
from .frames import FrameMeta, format_exposure
from .lensfun import Lens, LensfunDatabase, corner_falloff, vignetting_map
from .vignetting_profile import VignettingProfile, match_profile, profile_map


@dataclass
class CalibrationFrames:
    """User-supplied calibration frames, already read into FrameMeta."""

    bias: list[FrameMeta] = field(default_factory=list)
    darks: list[FrameMeta] = field(default_factory=list)
    flats: list[FrameMeta] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not (self.bias or self.darks or self.flats)


@dataclass
class FrameCalibration:
    frame_index: int
    vignetting: str  # "profile", "flat", "lensfun", "none"
    flat_group: Optional[str] = None
    lens_label: str = ""  # lensfun lens or own-flats profile used
    lens_terms: Optional[tuple[float, float, float]] = None
    channel_terms: Optional[dict[str, tuple[float, float, float]]] = None
    profile_crop: float = 1.0  # crop factor of the camera the profile was measured on
    corner_brightness: Optional[float] = None
    corner_stops: Optional[float] = None
    dark_group: Optional[str] = None
    bias: str = "synthetic"  # "frames" or "synthetic"
    synthetic_bias_adu: Optional[float] = None
    cosmetic: str = "find hot pixels"  # or "from dark"
    notes: list[str] = field(default_factory=list)
    # Set by masters.attach_masters for Stack and the earthshine stack (raw ADU, frame-sized):
    offset_map: Optional[np.ndarray] = field(default=None, repr=False, compare=False)  # dark or bias
    offset_label: str = ""
    flat_map: Optional[np.ndarray] = field(default=None, repr=False, compare=False)  # 1.0 in the middle


@dataclass
class CalibrationPlan:
    frames: list[FrameCalibration]
    lens: Optional[Lens]
    lens_candidates: list[str]
    dark_groups: dict[str, list[str]]
    flat_groups: dict[str, list[str]]
    bias_groups: dict[str, list[str]]
    warnings: list[str] = field(default_factory=list)

    def describe(self) -> str:
        lines = []
        counts = defaultdict(int)
        for fc in self.frames:
            counts[fc.vignetting] += 1
        own = sorted({fc.lens_label for fc in self.frames if fc.vignetting == "profile"})
        for label in own:
            lines.append(f"Own-flats profile: {label}")
        if own and not self.lens and not self.lens_candidates:
            pass
        elif self.lens:
            lines.append(f"Lens profile: {self.lens.label()} ({self.lens.source_file})")
        elif self.lens_candidates:
            lines.append("Lens profile: not chosen; candidates: " + "; ".join(self.lens_candidates))
        else:
            lines.append("Lens profile: none found")
        lines.append(
            "Vignetting: "
            + ", ".join(f"{n} frame(s) by {k}" for k, n in sorted(counts.items()))
        )
        falloffs = sorted({(round(fc.corner_brightness, 3), round(fc.corner_stops, 2)) for fc in self.frames if fc.corner_brightness})
        for brightness, stops in falloffs:
            lines.append(f"  corners at {brightness:.0%} of centre ({stops:+.2f} stops)")
        lines.append(f"Darks: {len(self.dark_groups)} matched group(s)" if self.dark_groups else "Darks: none; hot pixels found from the lights")
        lines.append(f"Bias: {len(self.bias_groups)} group(s)" if self.bias_groups else "Bias: synthetic, from each file's black level")
        for w in self.warnings:
            lines.append(f"Warning: {w}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------


def plan_calibration(
    lights: list[FrameMeta],
    db: Optional[LensfunDatabase] = None,
    calib: Optional[CalibrationFrames] = None,
    lens_name: Optional[str] = None,
    use_lensfun: bool = True,
    use_focus_distance: bool = False,
    profile: Optional[VignettingProfile] = None,
    profiles: Optional[list[VignettingProfile]] = None,
) -> CalibrationPlan:
    """Choose flat, dark and bias handling for every light frame.

    ``lens_name`` forces a lensfun lens (as shown by ``Lens.label()`` or any
    of its database names). Otherwise the best match for the frames' EXIF
    lens is used, provided the match is unambiguous.

    The eclipse is always focused at infinity, so the EXIF focus distance is
    ignored unless ``use_focus_distance`` is set. Cameras report it coarsely
    and a wrong value would change the vignetting profile.

    ``profile`` forces one own-flats profile for every frame; ``profiles``
    is the saved library, searched per frame for the same lens and settings.
    """
    calib = calib or CalibrationFrames()
    warnings: list[str] = []

    dark_groups = _group(calib.darks, lambda f: _dark_key(f))
    flat_groups = _group(calib.flats, lambda f: _flat_key(f))
    bias_groups = _group(calib.bias, lambda f: _iso_key(f))

    # lensfun is only consulted for frames that no profile or flat covers.
    def covered(f: FrameMeta) -> bool:
        return profile is not None or _flat_key(f) in flat_groups or bool(profiles and match_profile(f, profiles))

    needs_lensfun = any(not covered(f) for f in lights)
    lens, candidates = None, []
    if not needs_lensfun:
        pass
    elif use_lensfun and db is not None and not db.empty:
        lens, candidates, lens_warning = choose_lens(lights, db, lens_name)
        if lens_warning:
            warnings.append(lens_warning)
    elif use_lensfun:
        warnings.append("lensfun database not found; run 'totality lensfun-update' to download it")

    camera_crop = _camera_crop(lights, db)
    results: list[FrameCalibration] = []
    for f in lights:
        fc = FrameCalibration(frame_index=f.index, vignetting="none")
        fkey = _flat_key(f)
        own = profile if profile is not None else None
        if own is None and fkey not in flat_groups and profiles:
            own = match_profile(f, profiles)
        if own is not None:
            crop = f.crop_factor or camera_crop or 1.0
            fc.vignetting = "profile"
            fc.lens_label = own.label()
            fc.lens_terms = own.terms
            fc.channel_terms = dict(own.channel_terms) or None
            fc.profile_crop = own.crop_factor
            fc.corner_brightness, fc.corner_stops = own.corner(crop)
        elif fkey in flat_groups:
            fc.vignetting, fc.flat_group = "flat", fkey
        elif lens is not None and f.focal_length_mm and f.aperture_f:
            distance = 1000.0
            if use_focus_distance and f.focus_distance_m and f.focus_distance_m > 0:
                distance = f.focus_distance_m
            terms = lens.vignetting_terms(f.focal_length_mm, f.aperture_f, distance)
            if terms:
                crop = f.crop_factor or camera_crop or 1.0
                brightness, stops = corner_falloff(terms, crop, lens.crop_factor)
                fc.vignetting = "lensfun"
                fc.lens_label = lens.label()
                fc.lens_terms = terms
                fc.profile_crop = lens.crop_factor
                fc.corner_brightness, fc.corner_stops = brightness, stops
            else:
                fc.notes.append(
                    f"no lensfun vignetting data near {f.focal_length_mm:g} mm f/{f.aperture_f:g}"
                )
        dkey = _dark_key(f)
        if dkey in dark_groups:
            fc.dark_group, fc.cosmetic = dkey, "from dark"
        ikey = _iso_key(f)
        if ikey in bias_groups:
            fc.bias = "frames"
        else:
            fc.synthetic_bias_adu = f.black_level
            if f.black_level is None:
                fc.notes.append("black level unknown; synthetic bias cannot be set")
        results.append(fc)

    unmatched_darks = [f.index for f, fc in zip(lights, results) if calib.darks and not fc.dark_group]
    if unmatched_darks:
        warnings.append(f"{len(unmatched_darks)} light frame(s) have no dark with the same exposure and ISO")
    unknown_black = [fc.frame_index for fc in results if fc.bias == "synthetic" and fc.synthetic_bias_adu is None]
    if unknown_black:
        warnings.append(
            f"black level unknown for {len(unknown_black)} frame(s); install rawpy or exiftool, or supply bias frames"
        )
    no_vign = [fc.frame_index for fc in results if fc.vignetting == "none"]
    if no_vign:
        warnings.append(
            f"{len(no_vign)} light frame(s) will not be corrected for vignetting; shoot flats with the same lens, "
            "focal length and aperture, then run 'totality flat-profile <flats folder>'"
        )

    return CalibrationPlan(
        frames=results,
        lens=lens,
        lens_candidates=[m for m in candidates],
        dark_groups={k: [f.path for f in v] for k, v in dark_groups.items()},
        flat_groups={k: [f.path for f in v] for k, v in flat_groups.items()},
        bias_groups={k: [f.path for f in v] for k, v in bias_groups.items()},
        warnings=warnings,
    )


def choose_lens(
    frames: list[FrameMeta], db: LensfunDatabase, lens_name: Optional[str] = None
) -> tuple[Optional[Lens], list[str], str]:
    """Pick the lensfun lens for these frames. Returns (lens, candidates, warning)."""
    if lens_name:
        lens = db.find_lens_by_name(lens_name)
        if lens is None:
            return None, [], f"lens '{lens_name}' is not in the lensfun database"
        return lens, [lens.label()], ""
    names = {(f.lens_model, tuple(f.lens_spec) if f.lens_spec else None) for f in frames if f.lens_model or f.lens_spec}
    if not names:
        return None, [], "the files do not say which lens was used; choose a lens profile"
    if len(names) > 1:
        return None, [], "frames were taken with more than one lens; choose a lens profile"
    name, spec = names.pop()
    sample = frames[0]
    maker = sample.lens_make or sample.camera_make
    matches = db.match_lens(name, list(spec) if spec else None, sample.focal_length_mm, maker)
    labels = [m.lens.label() for m in matches]
    if not matches:
        return None, [], f"no lensfun profile matches '{name}'"
    best = matches[0]
    runner_up = matches[1].score if len(matches) > 1 else -math.inf
    if best.score - runner_up < 1.0:
        return None, labels, f"lens '{name}' matches several profiles equally well; choose one"
    if not best.lens.has_vignetting:
        # Only suggest another profile from the same maker: a Sigma or Tokina
        # profile is never right for a lens the camera reports as a Nikkor.
        maker_word = (maker or "").lower().split(" ")[0]
        others = [
            m.lens.label() for m in matches[1:]
            if m.lens.has_vignetting and maker_word and m.lens.maker.lower().startswith(maker_word)
        ]
        hint = f"; if your lens is the {others[0]}, choose it with --lens \"{others[0]}\"" if others else ""
        return None, labels, (
            f"'{best.lens.label()}' has no vignetting data in lensfun{hint}; "
            "or shoot flats and run 'totality flat-profile'"
        )
    return best.lens, labels, ""


def write_synthetic_flat(
    path: str | os.PathLike,
    width: int,
    height: int,
    terms: tuple[float, float, float],
    image_crop: float = 1.0,
    calib_crop: float = 1.0,
    channel_terms: Optional[dict[str, tuple[float, float, float]]] = None,
    cfa_pattern: str = "",
) -> Path:
    """Write a vignetting correction (lensfun or own flats) as a FITS master flat.

    The flat is the relative illumination itself (1.0 at the centre), so
    Siril's flat division applies exactly this correction. With per-channel
    terms and a Bayer pattern such as "RGGB", each CFA pixel gets its own
    channel's falloff.
    """
    if channel_terms and len(cfa_pattern) == 4:
        prof = VignettingProfile("", 0.0, 0.0, terms, channel_terms, calib_crop)
        flat = profile_map(prof, width, height, image_crop, cfa_pattern)
    else:
        flat = vignetting_map(width, height, terms, image_crop, calib_crop)
    write_fits(
        path,
        flat,
        {"IMAGETYP": "Master Flat", "COMMENT": "Synthetic flat from a smooth vignetting profile",
         "LF_K1": terms[0], "LF_K2": terms[1], "LF_K3": terms[2]},
    )
    return Path(path)


# ---------------------------------------------------------------------------


def _group(frames: list[FrameMeta], key) -> dict[str, list[FrameMeta]]:
    groups: dict[str, list[FrameMeta]] = defaultdict(list)
    for f in frames:
        k = key(f)
        if k:
            groups[k].append(f)
    return dict(groups)


def _dark_key(f: FrameMeta) -> Optional[str]:
    if not f.exposure_s or not f.iso:
        return None
    # Group by exposure rounded to 1/20 stop so EXIF rounding does not split groups.
    return f"t{round(math.log2(f.exposure_s) * 20)}_iso{int(f.iso)}"


def _flat_key(f: FrameMeta) -> Optional[str]:
    """Flats match lights taken with the same f-number and focal length. What a file does not record
    (a camera on a telescope) is '?': flats with no lens data match lights with none, as through the
    same optics, but never lights that record other settings."""
    a = f"{f.aperture_f:.1f}" if f.aperture_f else "?"
    fl = f"{round(f.focal_length_mm)}" if f.focal_length_mm else "?"
    return f"f{a}_fl{fl}"


def _iso_key(f: FrameMeta) -> Optional[str]:
    return f"iso{int(f.iso)}" if f.iso else None


def _camera_crop(frames: list[FrameMeta], db: Optional[LensfunDatabase]) -> Optional[float]:
    if db is None or not frames:
        return None
    cam = db.find_camera(frames[0].camera_make, frames[0].camera_model)
    return cam.crop_factor if cam else None


def describe_dark_key(key: str) -> str:
    t, iso = key[1:].split("_iso")
    return f"{format_exposure(2 ** (int(t) / 20))} s, ISO {iso}"
