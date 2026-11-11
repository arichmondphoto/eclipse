"""The event: which eclipse, where, and how the camera's clock relates to true time.

Second and third contact are computed from NASA's Besselian elements for the observing site and
shown in the camera's own clock (what the RAW files record), so frames can be compared with them
directly:

    camera clock = UT + camera time zone - clock error        (clock error = true UT - camera UT)

The camera time zone is taken from the files when they record one, otherwise from the setting.
An unknown clock error counts as 0 here. Because a wrong clock would mark frames taken in
totality as partial, :func:`check_fit` looks for brackets that straddle a contact (a bracket is
shot in one go, so it is either in totality or not) and says what clock error would fit them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

from .frames import FrameMeta

Say = Callable[[str], None]


def ensure_elements(day: str, say: Optional[Say] = None):
    """Besselian elements for this date. NASA's catalogue ships with the toolkit; it is fetched
    only if that file is missing (a damaged installation) and no downloaded copy exists."""
    from . import eclipse as E

    try:
        return E.elements_for(day)
    except KeyError:
        if E.BUNDLED_CANON.is_file() or E.CANON_CACHE.is_file():
            raise
    (say or (lambda _m: None))("NASA's eclipse catalogue is missing from this installation; fetching it once.")
    E.update_canon()
    return E.elements_for(day)


def known_eclipses(kinds: tuple[str, ...] = ("T", "H")) -> list[tuple[str, str]]:
    """(date, label) of the eclipses with elements on this computer: total and hybrid by default."""
    from .eclipse import all_elements

    names = {"T": "total", "H": "hybrid", "A": "annular", "P": "partial"}
    out = []
    for day, el in sorted(all_elements().items()):
        kind = (el.type or "")[:1].upper()
        if kinds and kind not in kinds:
            continue
        out.append((day, f"{day} ({names.get(kind, el.type or '?')})"))
    return out


def frames_date(frames: list[FrameMeta]) -> Optional[str]:
    """The date the frames were taken (camera clock), or None."""
    for f in frames:
        dt = f.datetime()
        if dt is not None:
            return dt.date().isoformat()
    return None


def camera_offset_h(frames: list[FrameMeta], setting: Optional[float]) -> tuple[Optional[float], str]:
    """The camera's time zone in hours and where it came from ("files" or "setting")."""
    for f in frames:
        dt = f.datetime()
        if dt is not None and dt.tzinfo is not None:
            return dt.utcoffset().total_seconds() / 3600.0, "files"
    if setting is not None:
        return float(setting), "setting"
    return None, ""


@dataclass
class Contacts:
    eclipse_date: str
    kind: str  # the eclipse here: total, partial, annular...
    c2_ut: datetime
    c3_ut: datetime
    c2_camera: datetime  # naive, in the camera's clock
    c3_camera: datetime
    duration_s: float
    offset_h: float
    offset_source: str
    clock_error_s: float
    clock_error_known: bool
    circumstances: object = None
    notes: list[str] = field(default_factory=list)

    def describe(self) -> str:
        zone = f"UTC{self.offset_h:+g} h ({'from the files' if self.offset_source == 'files' else 'your setting'})"
        clock = (f"clock error {self.clock_error_s:+.1f} s" if self.clock_error_known
                 else "clock error unknown (taken as 0)")
        m, s = divmod(self.duration_s, 60)
        lines = [f"Eclipse {self.eclipse_date}: totality here lasts {int(m)}m{s:04.1f}s",
                 f"  second contact {self.c2_ut:%H:%M:%S} UT = {self.c2_camera:%H:%M:%S} on the camera",
                 f"  third contact  {self.c3_ut:%H:%M:%S} UT = {self.c3_camera:%H:%M:%S} on the camera",
                 f"  camera clock: {zone}, {clock}"]
        return "\n".join(lines + [f"  note: {n}" for n in self.notes])


def compute_contacts(frames: list[FrameMeta], latitude: float, longitude: float, elevation_m: float = 0.0,
                     utc_offset_h: Optional[float] = None, clock_error_s: Optional[float] = None,
                     eclipse_date: str = "", say: Optional[Say] = None) -> Contacts:
    """Second and third contact at the site, in UT and in the camera's clock."""
    from .eclipse import LocalEclipse, Site

    day = eclipse_date or frames_date(frames)
    if not day:
        raise ValueError("Choose the eclipse: the frames have no capture date.")
    try:
        el = ensure_elements(day, say)
    except KeyError as exc:
        raise ValueError(str(exc.args[0])) from exc
    except OSError as exc:
        raise ValueError(f"Could not download NASA's eclipse catalogue ({exc}). Check the internet connection, "
                         "or type the contact times in.") from exc
    circ = LocalEclipse(el, Site(latitude, longitude, elevation_m or 0.0)).circumstances()
    if circ.c2 is None or circ.c3 is None:
        raise ValueError(f"At {latitude:.4f}, {longitude:.4f} the eclipse of {el.date} is {circ.kind}, not total: "
                         "there is no second or third contact. Check the latitude and longitude (west and south "
                         "are negative) and the eclipse.")
    offset, source = camera_offset_h(frames, utc_offset_h)
    if offset is None:
        raise ValueError("The RAW files do not record a time zone: enter the camera's time zone (the zone its "
                         "clock was set to, e.g. -6 for CST, -5 for CDT).")
    ce = float(clock_error_s) if clock_error_s is not None else 0.0

    def cam(t: datetime) -> datetime:
        return (t.astimezone(timezone.utc) + timedelta(hours=offset) - timedelta(seconds=ce)).replace(tzinfo=None)

    out = Contacts(el.date, circ.kind, circ.c2, circ.c3, cam(circ.c2), cam(circ.c3),
                   float(circ.duration_s or (circ.c3 - circ.c2).total_seconds()), offset, source, ce,
                   clock_error_s is not None, circ)
    if frames_date(frames) and abs((cam(circ.c2).date() - datetime.fromisoformat(frames_date(frames)).date()).days) > 1:
        out.notes.append(f"the frames were taken on {frames_date(frames)}, not on the day of this eclipse")
    return out


@dataclass
class Fit:
    inside: int
    outside: int
    warnings: list[str]
    suggested_clock_error: Optional[tuple[float, float]] = None  # a range that fits every bracket, seconds

    @property
    def ok(self) -> bool:
        return not self.warnings


def check_fit(frames: list[FrameMeta], c2: datetime, c3: datetime, clock_error_s: float = 0.0) -> Fit:
    """Do the frames sit sensibly against the contacts (both in the camera clock)?

    A bracket is shot in one go: if some of its frames fall inside totality and some outside, the
    camera clock (or time zone, or site) is probably wrong, and marking would throw away frames
    taken in totality. Says which clock error would put every such bracket inside.
    """
    from .brackets import detect_brackets

    use = [f for f in frames if f.role != "excluded" and f.datetime() is not None]
    times = {f.index: f.datetime().replace(tzinfo=None) for f in use}
    inside = {i for i, t in times.items() if c2 <= t <= c3}
    warnings: list[str] = []
    if use and not inside:
        first, last = min(times.values()), max(times.values())
        warnings.append(f"none of your frames falls inside the computed totality ({c2:%H:%M:%S} to {c3:%H:%M:%S} on "
                        f"the camera; your frames run {first:%H:%M:%S} to {last:%H:%M:%S}). Check the eclipse, the "
                        "site, the camera time zone and the clock error.")
    # The brackets as numbered in the frame table, when they have been found; otherwise find them.
    groups: dict[int, list[int]] = {}
    if use and all(f.bracket_id is not None for f in use):
        for f in use:
            groups.setdefault(f.bracket_id, []).append(f.index)
    else:
        groups = {b.bracket_id: list(b.frame_indices) for b in detect_brackets(use, assign=False).brackets}
    lo_all, hi_all = -1e9, 1e9
    straddled = False
    for bid, members in sorted(groups.items()):
        idx = [i for i in members if i in times]
        if not idx:
            continue
        n_in = sum(1 for i in idx if i in inside)
        if n_in == 0 or n_in == len(idx):
            continue
        straddled = True
        t0, t1 = min(times[i] for i in idx), max(times[i] for i in idx)
        before = sum(1 for i in idx if times[i] < c2)
        after = sum(1 for i in idx if times[i] > c3)
        # Shift s added to the camera times so the whole bracket fits: c2 <= t0 + s, t1 + s <= c3.
        s_lo, s_hi = (c2 - t0).total_seconds(), (c3 - t1).total_seconds()
        where = (f"{before} of its frames before second contact" if before else "") + \
                (" and " if before and after else "") + (f"{after} after third contact" if after else "")
        warnings.append(f"bracket {bid} ({len(idx)} frames, {t0:%H:%M:%S} to {t1:%H:%M:%S} on the camera) "
                        f"is cut by the contact times: {where}.")
        lo_all, hi_all = max(lo_all, s_lo), min(hi_all, s_hi)
    suggestion = None
    if straddled and lo_all <= hi_all:
        suggestion = (clock_error_s + lo_all, clock_error_s + hi_all)
        warnings.append(f"if those brackets were all shot in totality, the camera clock error (true minus camera) is "
                        f"between {suggestion[0]:+.0f} and {suggestion[1]:+.0f} s; enter it and compute again. If "
                        "the brackets really began before totality, mark their frames by hand instead.")
    elif straddled:
        warnings.append("no single clock error puts all of those brackets inside totality; check the site, the "
                        "eclipse and the camera time zone.")
    return Fit(len(inside), len(times) - len(inside), warnings, suggestion)
