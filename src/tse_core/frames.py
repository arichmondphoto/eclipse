"""Per-frame metadata record shared by every stage of the pipeline."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timedelta, timezone
from fractions import Fraction
from typing import Any, Optional

RAW_EXTENSIONS = {
    ".nef": "Nikon",
    ".nrw": "Nikon",
    ".cr2": "Canon",
    ".cr3": "Canon",
    ".arw": "Sony",
    ".srf": "Sony",
    ".sr2": "Sony",
    ".dng": "DNG",
}


@dataclass
class FrameMeta:
    """Everything the pipeline needs to know about one RAW frame.

    Values come from the file's EXIF and RAW metadata. Any field can be
    overridden by the user; overridden field names are listed in
    ``overrides`` so they survive a re-scan of the folder.
    """

    path: str
    file_name: str = ""
    index: int = -1  # position after sorting by capture time

    camera_make: str = ""
    camera_model: str = ""
    serial_number: str = ""
    lens_make: str = ""
    lens_model: str = ""
    # Lens specification: min focal, max focal, max aperture at min, at max.
    lens_spec: Optional[list[float]] = None

    exposure_s: Optional[float] = None
    exposure_text: str = ""  # nominal value as written in EXIF, e.g. "1/1600"
    aperture_f: Optional[float] = None
    iso: Optional[float] = None
    focal_length_mm: Optional[float] = None
    focus_distance_m: Optional[float] = None

    # Capture time as ISO 8601 text (local camera time unless an offset is known).
    timestamp: str = ""
    shutter_count: Optional[int] = None

    # Sensor data levels in raw ADU.
    black_level: Optional[float] = None
    black_level_per_channel: Optional[list[float]] = None
    white_level: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    cfa_pattern: str = ""
    crop_factor: Optional[float] = None

    # Bookkeeping.
    role: str = "light"  # light, partial, excluded
    bracket_id: Optional[int] = None
    sequence_id: Optional[int] = None
    sources: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    overrides: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------ time
    def datetime(self) -> Optional[datetime]:
        return parse_iso(self.timestamp) if self.timestamp else None

    # -------------------------------------------------------------- exposure
    def exposure_value(self) -> Optional[float]:
        """Relative light gathered, in stops, normalised to 1 s, ISO 100, f/1.

        This is only a starting estimate. The HDR stage replaces it with the
        brightness scale measured from the image data.
        """
        if not (self.exposure_s and self.aperture_f and self.iso):
            return None
        return math.log2(self.exposure_s * (self.iso / 100.0) / (self.aperture_f**2))

    # ---------------------------------------------------------- persistence
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FrameMeta":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in names})

    def apply_override(self, name: str, value: Any) -> None:
        if name not in {f.name for f in fields(self)}:
            raise KeyError(f"Unknown frame field: {name}")
        setattr(self, name, value)
        if name not in self.overrides:
            self.overrides.append(name)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def parse_exif_datetime(
    value: str, subsec: str | None = None, offset: str | None = None
) -> Optional[datetime]:
    """Parse EXIF "YYYY:MM:DD HH:MM:SS" plus optional sub-seconds and offset."""
    if not value:
        return None
    value = str(value).strip().replace("\x00", "")
    # exiftool's SubSecDateTimeOriginal already carries ".ss" and "+hh:mm".
    if "." in value[17:] or "+" in value[17:] or value[19:20] == "-":
        dt = _parse_composite(value)
        if dt is not None:
            return dt
    try:
        dt = datetime.strptime(value[:19], "%Y:%m:%d %H:%M:%S")
    except ValueError:
        try:
            dt = datetime.fromisoformat(value)
        except ValueError:
            return None
    if subsec:
        digits = "".join(ch for ch in str(subsec) if ch.isdigit())
        if digits:
            dt += timedelta(microseconds=int(round(int(digits) / 10 ** len(digits) * 1e6)))
    if offset:
        tz = _parse_offset(str(offset))
        if tz is not None:
            dt = dt.replace(tzinfo=tz)
    return dt


def _parse_composite(value: str) -> Optional[datetime]:
    base, rest = value[:19], value[19:]
    subsec, offset = "", ""
    if rest.startswith("."):
        rest = rest[1:]
        i = 0
        while i < len(rest) and rest[i].isdigit():
            i += 1
        subsec, offset = rest[:i], rest[i:]
    else:
        offset = rest
    try:
        datetime.strptime(base, "%Y:%m:%d %H:%M:%S")
    except ValueError:
        return None
    return parse_exif_datetime(base, subsec or None, offset or None)


def _parse_offset(text: str) -> Optional[timezone]:
    text = text.strip()
    if not text or text[0] not in "+-":
        return None
    sign = 1 if text[0] == "+" else -1
    parts = text[1:].split(":")
    try:
        hours = int(parts[0])
        minutes = int(parts[1]) if len(parts) > 1 else 0
    except ValueError:
        return None
    return timezone(sign * timedelta(hours=hours, minutes=minutes))


def parse_iso(text: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(text)
    except (TypeError, ValueError):
        return None


def to_float(value: Any) -> Optional[float]:
    """Convert EXIF values ("1/1600", Fraction, Ratio, "8.0 mm", [x]) to float."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return to_float(value[0]) if value else None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, Fraction):
        return float(value)
    num = getattr(value, "num", None)
    den = getattr(value, "den", None)
    if num is not None and den:
        return float(num) / float(den)
    text = str(value).strip().split(" ")[0].replace("\x00", "")
    if not text:
        return None
    try:
        if "/" in text:
            n, d = text.split("/", 1)
            return float(n) / float(d) if float(d) else None
        return float(text)
    except ValueError:
        return None


def format_exposure(seconds: Optional[float]) -> str:
    """Human-readable shutter speed: 1/1600, 1/2.5, 0.6, 4."""
    if not seconds:
        return ""
    if seconds >= 0.55:
        text = f"{seconds:.1f}".rstrip("0").rstrip(".")
        return text
    inv = 1.0 / seconds
    text = f"{inv:.1f}".rstrip("0").rstrip(".") if inv < 10 else f"{round(inv)}"
    return f"1/{text}"
