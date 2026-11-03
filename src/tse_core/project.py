"""Per-event project file: everything needed to reproduce one eclipse run."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from . import __version__
from .frames import FrameMeta
from .inventory import Inventory

PROJECT_FORMAT = 1


@dataclass
class EventProject:
    name: str = ""
    raw_folder: str = ""
    created: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    modified: str = ""
    toolkit_version: str = __version__
    format: int = PROJECT_FORMAT

    # Which eclipse (its date, as in NASA's catalogue); empty = the one on the frames' date.
    eclipse_date: str = ""
    # In the camera's clock (ISO 8601), as the RAW files record time: computed from the eclipse, the
    # site, the camera time zone and the clock error (event.compute_contacts), or typed in.
    second_contact: str = ""
    third_contact: str = ""
    # Older projects: time added to the camera clock to compare with contacts typed in another clock.
    # Converted on loading (the contacts are moved into the camera clock) and then always 0.
    camera_clock_offset_s: float = 0.0

    # Observing site (for the eclipse circumstances and the Sun's centre).
    latitude: Optional[float] = None
    longitude: Optional[float] = None  # east positive
    elevation_m: float = 0.0
    camera_utc_offset_h: Optional[float] = None  # only needed when the files carry no time zone
    mirrored: bool = False  # optics give a mirror image (e.g. a telescope with a star diagonal)
    clock_error_s: Optional[float] = None  # true UT minus camera UT, when known

    lens_profile: str = ""  # forced lensfun lens label, empty = automatic
    use_lensfun: bool = True
    bias_files: list[str] = field(default_factory=list)
    dark_files: list[str] = field(default_factory=list)
    flat_files: list[str] = field(default_factory=list)

    # Parameters for later stages, keyed by stage name ("align", "stack", ...).
    parameters: dict[str, dict[str, Any]] = field(default_factory=dict)
    frames: list[dict[str, Any]] = field(default_factory=list)

    # ----------------------------------------------------------- inventory
    @property
    def inventory(self) -> Inventory:
        return Inventory([FrameMeta.from_dict(d) for d in self.frames])

    def store_inventory(self, inventory: Inventory) -> None:
        """Store the inventory exactly as it is now."""
        self.frames = [f.to_dict() for f in inventory.frames]

    def set_inventory(self, inventory: Inventory) -> None:
        """Store a new inventory, keeping the user's overrides and roles."""
        old = {d["path"]: d for d in self.frames}
        merged = []
        for frame in inventory.frames:
            previous = old.get(frame.path)
            if previous:
                for name in previous.get("overrides", []):
                    frame.apply_override(name, previous.get(name))
                if previous.get("role") == "excluded":
                    frame.role = "excluded"
            merged.append(frame.to_dict())
        self.frames = merged

    # --------------------------------------------------------- persistence
    def save(self, path: str | os.PathLike) -> None:
        self.modified = datetime.now().isoformat(timespec="seconds")
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | os.PathLike) -> "EventProject":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("format", 1) > PROJECT_FORMAT:
            raise ValueError("This project was saved by a newer version of the toolkit")
        known = set(cls.__dataclass_fields__)
        project = cls(**{k: v for k, v in data.items() if k in known})
        project._contacts_to_camera_clock()
        return project

    def _contacts_to_camera_clock(self) -> None:
        """Older projects kept typed contacts plus a clock offset (frames were compared after adding
        it); the same comparison is: contact minus offset, in the camera's own clock."""
        off = float(self.camera_clock_offset_s or 0.0)
        if not off:
            return
        from datetime import timedelta

        for name in ("second_contact", "third_contact"):
            text = getattr(self, name)
            try:
                t = datetime.fromisoformat(text) if text else None
            except ValueError:
                t = None
            if t is not None:
                setattr(self, name, (t - timedelta(seconds=off)).isoformat())
        self.camera_clock_offset_s = 0.0

    def contact_times(self) -> tuple[Optional[datetime], Optional[datetime]]:
        def parse(text: str) -> Optional[datetime]:
            try:
                return datetime.fromisoformat(text) if text else None
            except ValueError:
                return None

        return parse(self.second_contact), parse(self.third_contact)
