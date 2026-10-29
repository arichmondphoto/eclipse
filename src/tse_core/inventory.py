"""Build and save the frame inventory for one eclipse event."""

from __future__ import annotations

import csv
import json
import os
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

from .frames import RAW_EXTENSIONS, FrameMeta
from .metadata import read_frames

CSV_COLUMNS = [
    "index", "file_name", "timestamp", "exposure_text", "exposure_s", "aperture_f",
    "iso", "focal_length_mm", "camera_model", "lens_model", "black_level",
    "white_level", "role", "sequence_id", "bracket_id", "warnings",
]


def find_raw_files(folder: str | os.PathLike, recursive: bool = False) -> list[str]:
    """All RAW files in a folder, by extension, case-insensitively."""
    root = Path(folder)
    pattern = "**/*" if recursive else "*"
    files = [
        str(p) for p in root.glob(pattern)
        if p.is_file() and p.suffix.lower() in RAW_EXTENSIONS and not p.name.startswith("._")
    ]
    return sorted(files)


@dataclass
class Inventory:
    frames: list[FrameMeta] = field(default_factory=list)

    # ------------------------------------------------------------- building
    @classmethod
    def from_paths(cls, paths: Iterable[str], **reader_options) -> "Inventory":
        inv = cls(read_frames(paths, **reader_options))
        inv.sort()
        return inv

    @classmethod
    def from_folder(cls, folder: str, recursive: bool = False, **reader_options) -> "Inventory":
        return cls.from_paths(find_raw_files(folder, recursive), **reader_options)

    def sort(self) -> None:
        """Order by capture time, then shutter count, then file name."""

        def key(f: FrameMeta):
            dt = f.datetime()
            stamp = dt.replace(tzinfo=None).timestamp() if dt else float("inf")
            return (stamp, f.shutter_count if f.shutter_count is not None else 0, f.file_name)

        self.frames.sort(key=key)
        for i, frame in enumerate(self.frames):
            frame.index = i

    # -------------------------------------------------------------- queries
    def lights(self) -> list[FrameMeta]:
        return [f for f in self.frames if f.role == "light"]

    def cameras(self) -> Counter:
        return Counter(f.camera_model or "unknown" for f in self.frames)

    def lenses(self) -> Counter:
        return Counter(f.lens_model or "unknown" for f in self.frames)

    def settings(self, name: str) -> Counter:
        return Counter(getattr(f, name) for f in self.frames)

    def time_span(self) -> tuple[Optional[datetime], Optional[datetime]]:
        times = [f.datetime() for f in self.frames if f.datetime()]
        return (min(times), max(times)) if times else (None, None)

    def mark_outside(
        self, second_contact: datetime, third_contact: datetime, margin_s: float = 0.0, clock_offset_s: float = 0.0
    ) -> int:
        """Mark frames taken outside totality as 'partial'. Returns how many changed.

        ``clock_offset_s`` is added to each frame's camera time before the
        comparison, so contact times can be given in true time when the
        camera clock was wrong. With no offset, give them in camera time.
        """
        changed = 0
        for f in self.frames:
            dt = f.datetime()
            if dt is None or f.role == "excluded":
                continue
            a = _naive(second_contact)
            b = _naive(third_contact)
            t = _naive(dt) + _seconds(clock_offset_s)
            inside = (a - _seconds(margin_s)) <= t <= (b + _seconds(margin_s))
            new_role = "light" if inside else "partial"
            if new_role != f.role:
                f.role = new_role
                changed += 1
        return changed

    # ----------------------------------------------------------- persistence
    def to_json(self, path: str | os.PathLike) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump([f.to_dict() for f in self.frames], fh, indent=2)

    @classmethod
    def from_json(cls, path: str | os.PathLike) -> "Inventory":
        with open(path, encoding="utf-8") as fh:
            return cls([FrameMeta.from_dict(d) for d in json.load(fh)])

    def to_csv(self, path: str | os.PathLike) -> None:
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(CSV_COLUMNS)
            for f in self.frames:
                row = []
                for col in CSV_COLUMNS:
                    value = getattr(f, col)
                    row.append("; ".join(value) if isinstance(value, list) else ("" if value is None else value))
                writer.writerow(row)


def _naive(dt: datetime) -> datetime:
    return dt.replace(tzinfo=None) if dt.tzinfo else dt


def _seconds(s: float):
    from datetime import timedelta

    return timedelta(seconds=s)
