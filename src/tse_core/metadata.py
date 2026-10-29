"""Read capture settings and sensor levels from DSLR RAW files.

Three readers are tried in order and each fills only what the previous one
left empty:

1. **exiftool** (if installed): reads every format, including Canon CR3, and
   decodes maker notes, so it gives the full lens name and the black and
   white levels. Set ``TSE_EXIFTOOL`` to its path if it is not on ``PATH``.
2. **exifread** (pure Python, always available): reads NEF, NRW, CR2, ARW and
   DNG, plus CR3 through a small box parser in this module.
3. **rawpy** (LibRaw, optional): the authoritative source for black level,
   white level, image size and CFA pattern, the same library Siril uses.

Nothing here assumes a particular camera, lens or exposure scheme.
"""

from __future__ import annotations

import io
import json
import os
import shlex
import shutil
import struct
import subprocess
from pathlib import Path
from typing import Any, Iterable, Optional

from .frames import FrameMeta, format_exposure, parse_exif_datetime, to_float

# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def read_frames(
    paths: Iterable[str | os.PathLike],
    use_exiftool: bool = True,
    use_rawpy: bool = True,
) -> list[FrameMeta]:
    """Read metadata for every file, using all available readers."""
    paths = [str(p) for p in paths]
    frames = [FrameMeta(path=p, file_name=Path(p).name) for p in paths]

    exiftool = ExifToolReader.find() if use_exiftool else None
    if exiftool is not None:
        try:
            for frame, data in zip(frames, exiftool.read_many(paths)):
                _merge(frame, data, "exiftool")
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            for frame in frames:
                frame.warnings.append(f"exiftool failed: {exc}")

    for frame in frames:
        if _needs_more(frame):
            try:
                _merge(frame, read_with_exifread(frame.path), "exifread")
            except Exception as exc:  # exifread raises many types on odd files
                frame.warnings.append(f"exifread failed: {exc}")

    if use_rawpy:
        for frame in frames:
            levels = read_raw_levels(frame.path)
            if levels:
                # LibRaw is authoritative for levels and geometry.
                _merge(frame, levels, "rawpy", overwrite=True)

    for frame in frames:
        _finalise(frame)
    return frames


ESSENTIAL = ("exposure_s", "aperture_f", "iso", "focal_length_mm", "timestamp")


def _needs_more(frame: FrameMeta) -> bool:
    return any(not getattr(frame, name) for name in ESSENTIAL) or not frame.lens_model


def _merge(frame: FrameMeta, data: dict[str, Any], source: str, overwrite: bool = False) -> None:
    used = False
    for key, value in data.items():
        if value in (None, "", []):
            continue
        if key in frame.overrides:
            continue
        current = getattr(frame, key, None)
        if overwrite or current in (None, "", []):
            setattr(frame, key, value)
            used = True
    if used and source not in frame.sources:
        frame.sources.append(source)


def _finalise(frame: FrameMeta) -> None:
    if frame.exposure_s and not frame.exposure_text:
        frame.exposure_text = format_exposure(frame.exposure_s)
    if frame.black_level is None and frame.black_level_per_channel:
        vals = frame.black_level_per_channel
        frame.black_level = float(sum(vals) / len(vals))
    missing = [name for name in ESSENTIAL if not getattr(frame, name)]
    if missing:
        frame.warnings.append("missing " + ", ".join(missing))
    if frame.black_level is None or frame.white_level is None:
        frame.warnings.append(
            "black/white level unknown (install rawpy or exiftool for exact values)"
        )


# ---------------------------------------------------------------------------
# exiftool
# ---------------------------------------------------------------------------

_EXIFTOOL_TAGS = [
    "-Make", "-Model", "-SerialNumber", "-LensMake", "-LensModel", "-LensID", "-Lens",
    "-LensInfo#", "-ExposureTime#", "-FNumber#", "-ISO#",
    "-FocalLength#", "-FocusDistance#", "-DateTimeOriginal", "-SubSecTimeOriginal",
    "-OffsetTimeOriginal", "-ShutterCount#", "-ImageCount#", "-BlackLevel#",
    "-WhiteLevel#", "-ScaleFactor35efl#",
]


class ExifToolReader:
    def __init__(self, command: list[str]):
        self.command = command

    @classmethod
    def find(cls) -> Optional["ExifToolReader"]:
        env = os.environ.get("TSE_EXIFTOOL")
        if env:
            return cls(shlex.split(env))
        exe = shutil.which("exiftool") or shutil.which("exiftool.exe")
        return cls([exe]) if exe else None

    def read_many(self, paths: list[str], chunk: int = 200) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for start in range(0, len(paths), chunk):
            part = paths[start : start + chunk]
            cmd = self.command + ["-json", "-q", "-q"] + _EXIFTOOL_TAGS + part
            out = subprocess.run(cmd, capture_output=True, check=False, timeout=600)
            records = json.loads(out.stdout.decode("utf-8", "replace") or "[]")
            by_file = {os.path.normcase(os.path.abspath(r.get("SourceFile", ""))): r for r in records}
            for p in part:
                rec = by_file.get(os.path.normcase(os.path.abspath(p)), {})
                results.append(_from_exiftool(rec))
        return results


def _from_exiftool(rec: dict[str, Any]) -> dict[str, Any]:
    if not rec:
        return {}
    lens_info = rec.get("LensInfo")
    lens_spec = None
    if isinstance(lens_info, str):
        nums = [to_float(x) for x in lens_info.split()]
        if len(nums) == 4 and all(n is not None for n in nums):
            lens_spec = nums
    lens_name = (
        _clean(rec.get("LensID"))
        or _clean(rec.get("LensModel"))
        or _clean(rec.get("Lens"))
    )
    if lens_name.lower().startswith("unknown"):
        lens_name = _clean(rec.get("LensModel")) or lens_name
    # When the decoded lens ID and the EXIF lens model disagree on focal range,
    # trust the EXIF model: it is what the camera wrote for this shot.
    model = _clean(rec.get("LensModel"))
    if model and lens_name != model and _focal_text(model) and _focal_text(model) != _focal_text(lens_name):
        lens_name = model
    black = rec.get("BlackLevel")
    black_list = None
    if isinstance(black, str):
        black_list = [v for v in (to_float(x) for x in black.split()) if v is not None]
        black = None
    shutter_count = rec.get("ShutterCount") or rec.get("ImageCount")
    dt = parse_exif_datetime(
        str(rec.get("DateTimeOriginal", "")),
        str(rec.get("SubSecTimeOriginal", "") or ""),
        str(rec.get("OffsetTimeOriginal", "") or ""),
    )
    white = rec.get("WhiteLevel")
    if isinstance(white, str):
        white = to_float(white.split()[0])
    scale = to_float(rec.get("ScaleFactor35efl"))
    return {
        "camera_make": _clean(rec.get("Make")),
        "camera_model": _clean(rec.get("Model")),
        "serial_number": _clean(rec.get("SerialNumber")),
        "lens_make": _clean(rec.get("LensMake")),
        "lens_model": lens_name,
        "lens_spec": lens_spec,
        "exposure_s": to_float(rec.get("ExposureTime")),
        "aperture_f": to_float(rec.get("FNumber")),
        "iso": to_float(rec.get("ISO")),
        "focal_length_mm": to_float(rec.get("FocalLength")),
        "focus_distance_m": to_float(rec.get("FocusDistance")),
        "timestamp": dt.isoformat() if dt else "",
        "shutter_count": int(shutter_count) if isinstance(shutter_count, (int, float)) else None,
        "black_level": to_float(black) if black is not None else None,
        "black_level_per_channel": black_list,
        "white_level": to_float(white),
        "crop_factor": scale if scale else None,
    }


# ---------------------------------------------------------------------------
# exifread (pure Python)
# ---------------------------------------------------------------------------

# Nikon LensType bit flags, as decoded by exiftool.
_NIKON_LENS_TYPE = [(0x02, "D"), (0x04, "G"), (0x08, "VR"), (0x40, "E"), (0x80, "AF-P")]


def read_with_exifread(path: str) -> dict[str, Any]:
    import exifread  # imported here so the package imports without it

    if path.lower().endswith(".cr3"):
        tags: dict[str, Any] = {}
        for block in extract_cr3_tiff_blocks(path):
            tags.update(exifread.process_file(io.BytesIO(block), details=False))
    else:
        with open(path, "rb") as fh:
            tags = exifread.process_file(fh, details=True)

    def get(*names: str) -> Any:
        for name in names:
            for prefix in ("EXIF ", "Image ", "MakerNote ", ""):
                tag = tags.get(prefix + name)
                if tag is not None:
                    return tag
        return None

    def values(tag: Any) -> list[Any]:
        if tag is None:
            return []
        vals = getattr(tag, "values", tag)
        return list(vals) if isinstance(vals, (list, tuple)) else [vals]

    exposure_tag = get("ExposureTime")
    dt = parse_exif_datetime(
        _clean(get("DateTimeOriginal")),
        _clean(get("SubSecTimeOriginal")),
        _clean(get("OffsetTimeOriginal")),
    )

    lens_spec = None
    spec_vals = values(get("LensSpecification")) or values(get("LensMinMaxFocalMaxAperture"))
    if len(spec_vals) == 4:
        spec = [to_float(v) for v in spec_vals]
        if all(v is not None for v in spec) and spec[0] > 0:
            lens_spec = spec

    lens_name = _clean(get("LensModel"))
    if not lens_name and lens_spec:
        lens_name = describe_lens_spec(lens_spec)
        lens_type = values(get("LensType"))
        if lens_type and isinstance(lens_type[0], int):
            flags = [name for bit, name in _NIKON_LENS_TYPE if lens_type[0] & bit]
            if "G" in flags and "D" in flags:
                flags.remove("D")  # G lenses also carry the distance (D) bit
            if flags:
                lens_name = f"{lens_name} {' '.join(flags)}"

    shutter = values(get("ShutterCount", "TotalShutterReleases"))
    iso = to_float(values(get("ISOSpeedRatings", "PhotographicSensitivity"))[:1] or None)
    return {
        "camera_make": _clean(get("Make")),
        "camera_model": _clean(get("Model")),
        "serial_number": _clean(get("BodySerialNumber", "SerialNumber")),
        "lens_make": _clean(get("LensMake")),
        "lens_model": lens_name,
        "lens_spec": lens_spec,
        "exposure_s": to_float(values(exposure_tag)[:1] or None),
        "aperture_f": to_float(values(get("FNumber"))[:1] or None),
        "iso": iso,
        "focal_length_mm": to_float(values(get("FocalLength"))[:1] or None),
        "timestamp": dt.isoformat() if dt else "",
        "shutter_count": int(shutter[0]) if shutter and isinstance(shutter[0], int) else None,
    }


def describe_lens_spec(spec: list[float]) -> str:
    """[80, 400, 4.5, 5.6] -> '80-400mm f/4.5-5.6'. Zero apertures mean unknown."""
    fmin, fmax, amin, amax = spec

    def num(x: float) -> str:
        return f"{x:.1f}".rstrip("0").rstrip(".")

    focal = f"{num(fmin)}mm" if abs(fmin - fmax) < 0.01 else f"{num(fmin)}-{num(fmax)}mm"
    if not amin:
        return focal
    aperture = f"f/{num(amin)}" if not amax or abs(amin - amax) < 0.01 else f"f/{num(amin)}-{num(amax)}"
    return f"{focal} {aperture}"


# --------------------------------------------------------------- Canon CR3

_CANON_UUID = bytes.fromhex("85c0b687820f11e08111f4ce462b6a48")


def extract_cr3_tiff_blocks(path: str) -> list[bytes]:
    """Return the TIFF-structured EXIF blocks (CMT1, CMT2, CMT3) of a CR3 file.

    CR3 is an ISO base media file. Its EXIF lives in ``moov/uuid(Canon)``
    as small TIFF files named CMT1 (main IFD), CMT2 (Exif IFD) and CMT3
    (Canon maker notes). Only the first few MB of the file are read.
    """
    with open(path, "rb") as fh:
        data = fh.read(8 * 1024 * 1024)
    blocks: dict[str, bytes] = {}

    def walk(buf: bytes, start: int, end: int, depth: int) -> None:
        pos = start
        while pos + 8 <= end and depth < 6:
            size, kind = struct.unpack(">I4s", buf[pos : pos + 8])
            header = 8
            if size == 1:
                size = struct.unpack(">Q", buf[pos + 8 : pos + 16])[0]
                header = 16
            elif size == 0:
                size = end - pos
            if size < header or pos + size > end:
                break
            name = kind.decode("latin-1")
            body = pos + header
            if name == "moov":
                walk(buf, body, pos + size, depth + 1)
            elif name == "uuid" and buf[body : body + 16] == _CANON_UUID:
                walk(buf, body + 16, pos + size, depth + 1)
            elif name in ("CMT1", "CMT2", "CMT3"):
                blocks[name] = buf[body : pos + size]
            pos += size

    walk(data, 0, len(data), 0)
    return [blocks[k] for k in ("CMT1", "CMT2", "CMT3") if k in blocks]


# ---------------------------------------------------------------------------
# rawpy (LibRaw)
# ---------------------------------------------------------------------------


def rawpy_available() -> bool:
    try:
        import rawpy  # noqa: F401
    except ImportError:
        return False
    return True


def read_raw_levels(path: str) -> dict[str, Any]:
    """Black level, white level, size and CFA pattern straight from LibRaw."""
    try:
        import rawpy
    except ImportError:
        return {}
    try:
        with rawpy.imread(path) as raw:
            black = [float(v) for v in raw.black_level_per_channel]
            colors = raw.color_desc.decode("ascii", "replace")
            pattern = "".join(colors[i] for i in raw.raw_pattern.flatten()) if raw.raw_pattern is not None else ""
            return {
                "black_level_per_channel": black,
                "black_level": sum(black) / len(black) if black else None,
                "white_level": float(raw.white_level),
                "width": int(raw.sizes.width),
                "height": int(raw.sizes.height),
                "cfa_pattern": pattern,
            }
    except Exception:  # LibRaw errors on unsupported or corrupt files
        return {}


def _focal_text(name: str) -> str:
    import re

    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:-\s*(\d+(?:\.\d+)?))?\s*mm", name or "", re.I)
    return m.group(0).replace(" ", "") if m else ""


def _clean(value: Any) -> str:
    if value is None:
        return ""
    text = str(getattr(value, "printable", value)).replace("\x00", "").strip()
    return "" if text in ("None", "[]") else text
