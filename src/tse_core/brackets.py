"""Find exposure brackets in a set of frames, whatever the bracketing scheme.

Frames are first split into *sequences* wherever there is a long pause
between shots (for example partial-phase frames taken minutes before
totality). Each sequence is then split into *brackets*: a bracket ends as
soon as an exposure repeats. This handles one long bracket, many repeated
short brackets, ascending or descending order, and any step size.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Optional

from .frames import FrameMeta, format_exposure

# Nominal shutter speeds marked on cameras, in seconds.
THIRD_STOP_SPEEDS = [
    30, 25, 20, 15, 13, 10, 8, 6, 5, 4, 3, 2.5, 2, 1.6, 1.3, 1, 0.8, 0.6, 1 / 2, 1 / 2.5,
    1 / 3, 1 / 4, 1 / 5, 1 / 6, 1 / 8, 1 / 10, 1 / 13, 1 / 15, 1 / 20, 1 / 25, 1 / 30,
    1 / 40, 1 / 50, 1 / 60, 1 / 80, 1 / 100, 1 / 125, 1 / 160, 1 / 200, 1 / 250, 1 / 320,
    1 / 400, 1 / 500, 1 / 640, 1 / 800, 1 / 1000, 1 / 1250, 1 / 1600, 1 / 2000, 1 / 2500,
    1 / 3200, 1 / 4000, 1 / 5000, 1 / 6400, 1 / 8000,
]
HALF_STOP_SPEEDS = [
    30, 20, 15, 10, 8, 6, 4, 3, 2, 1.5, 1, 0.7, 1 / 2, 1 / 3, 1 / 4, 1 / 6, 1 / 8, 1 / 10,
    1 / 15, 1 / 20, 1 / 30, 1 / 45, 1 / 60, 1 / 90, 1 / 125, 1 / 180, 1 / 250, 1 / 350,
    1 / 500, 1 / 750, 1 / 1000, 1 / 1500, 1 / 2000, 1 / 3000, 1 / 4000, 1 / 6000, 1 / 8000,
]
STANDARD_STEPS = (1 / 3, 1 / 2, 2 / 3, 1.0)


@dataclass
class Gap:
    after_exposure: str
    before_exposure: str
    stops: float
    missing_count: int
    missing_speeds: list[str] = field(default_factory=list)


@dataclass
class Bracket:
    bracket_id: int
    sequence_id: int
    frame_indices: list[int]
    exposures_s: list[float]
    shortest: str
    longest: str
    range_stops: float
    step_stops: Optional[float]
    gaps: list[Gap]
    duration_s: Optional[float]
    iso_values: list[float]
    aperture_values: list[float]
    order: str  # ascending, descending, mixed

    @property
    def count(self) -> int:
        return len(self.frame_indices)

    def describe(self) -> str:
        step = f"{_step_name(self.step_stops)} steps" if self.step_stops else "irregular steps"
        text = (
            f"Bracket {self.bracket_id}: {self.count} frames, {self.shortest} to {self.longest} s, "
            f"{self.range_stops:.1f} stops, {step}, {self.order}"
        )
        if self.duration_s is not None:
            text += f", {self.duration_s:.0f} s long"
        if len(set(self.iso_values)) > 1:
            text += f", ISO varies {sorted(set(self.iso_values))}"
        if len(set(self.aperture_values)) > 1:
            text += f", aperture varies {sorted(set(self.aperture_values))}"
        return text


@dataclass
class BracketReport:
    brackets: list[Bracket]
    sequence_count: int
    unplaced: list[int]  # frames without usable exposure data

    def describe(self) -> str:
        lines = [f"{len(self.brackets)} bracket(s) in {self.sequence_count} sequence(s)"]
        for b in self.brackets:
            lines.append("  " + b.describe())
            for g in b.gaps:
                missing = ", ".join(g.missing_speeds) if g.missing_speeds else f"{g.missing_count} step(s)"
                lines.append(
                    f"    gap of {g.stops:.2f} stops between {g.after_exposure} and "
                    f"{g.before_exposure} s (missing {missing})"
                )
        if self.unplaced:
            lines.append(f"  {len(self.unplaced)} frame(s) without exposure data were not placed")
        return "\n".join(lines)


# ---------------------------------------------------------------------------


def detect_brackets(
    frames: list[FrameMeta],
    min_pause_s: float = 60.0,
    pause_factor: float = 6.0,
    assign: bool = True,
) -> BracketReport:
    """Group light frames into sequences and brackets.

    A new sequence starts when the idle time between the end of one exposure
    and the start of the next exceeds ``max(min_pause_s, pause_factor *
    median idle time)``.
    """
    lights = [f for f in sorted(frames, key=lambda f: f.index) if f.role == "light"]
    usable = [f for f in lights if f.exposure_value() is not None]
    unplaced = [f.index for f in lights if f.exposure_value() is None]

    sequences = _split_sequences(usable, min_pause_s, pause_factor)
    brackets: list[Bracket] = []
    for seq_id, seq in enumerate(sequences):
        current: list[FrameMeta] = []
        seen: set[int] = set()
        for frame in seq:
            key = round(frame.exposure_value() * 20)  # 0.05-stop bins
            if key in seen:
                brackets.append(_analyse(len(brackets), seq_id, current))
                current, seen = [], set()
            current.append(frame)
            seen.add(key)
        if current:
            brackets.append(_analyse(len(brackets), seq_id, current))

    if assign:
        by_index = {f.index: f for f in frames}
        for b in brackets:
            for i in b.frame_indices:
                by_index[i].bracket_id = b.bracket_id
                by_index[i].sequence_id = b.sequence_id
    return BracketReport(brackets, len(sequences), unplaced)


def _split_sequences(frames: list[FrameMeta], min_pause_s: float, factor: float) -> list[list[FrameMeta]]:
    if not frames:
        return []
    times = [f.datetime() for f in frames]
    if any(t is None for t in times):
        return [frames]
    times = [t.replace(tzinfo=None) for t in times]
    idle = []
    for i in range(1, len(frames)):
        elapsed = (times[i] - times[i - 1]).total_seconds()
        idle.append(elapsed - (frames[i - 1].exposure_s or 0.0))
    threshold = max(min_pause_s, factor * statistics.median(idle)) if idle else min_pause_s
    sequences, current = [], [frames[0]]
    for i in range(1, len(frames)):
        if idle[i - 1] > threshold:
            sequences.append(current)
            current = []
        current.append(frames[i])
    sequences.append(current)
    return sequences


def _analyse(bracket_id: int, sequence_id: int, frames: list[FrameMeta]) -> Bracket:
    evs = [f.exposure_value() for f in frames]
    order_desc = _order(evs)
    ordered = sorted(frames, key=lambda f: f.exposure_value())
    sorted_ev = [f.exposure_value() for f in ordered]
    diffs = [b - a for a, b in zip(sorted_ev, sorted_ev[1:])]

    step = None
    if diffs:
        median = statistics.median(diffs)
        step = min(STANDARD_STEPS, key=lambda s: abs(s - median))
        if abs(step - median) > 0.1:
            step = median

    only_shutter = len({f.iso for f in frames}) == 1 and len({f.aperture_f for f in frames}) == 1
    gaps: list[Gap] = []
    if step:
        for lo, hi, d in zip(ordered, ordered[1:], diffs):
            if d > step * 1.4:
                missing_speeds = _missing_speeds(lo.exposure_s, hi.exposure_s, step) if only_shutter else []
                count = len(missing_speeds) or max(1, round(d / step) - 1)
                gaps.append(Gap(_name(lo), _name(hi), d, count, missing_speeds))

    times = [f.datetime() for f in frames]
    duration = None
    if all(times):
        times = [t.replace(tzinfo=None) for t in times]
        last = max(range(len(frames)), key=lambda i: times[i])
        duration = (times[last] - min(times)).total_seconds() + (frames[last].exposure_s or 0.0)

    return Bracket(
        bracket_id=bracket_id,
        sequence_id=sequence_id,
        frame_indices=[f.index for f in frames],
        exposures_s=[f.exposure_s for f in frames],
        shortest=_name(ordered[0]),
        longest=_name(ordered[-1]),
        range_stops=sorted_ev[-1] - sorted_ev[0],
        step_stops=step,
        gaps=gaps,
        duration_s=duration,
        iso_values=[f.iso for f in frames],
        aperture_values=[f.aperture_f for f in frames],
        order=order_desc,
    )


def _order(evs: list[float]) -> str:
    if len(evs) < 2:
        return "single"
    ups = sum(b > a for a, b in zip(evs, evs[1:]))
    downs = sum(b < a for a, b in zip(evs, evs[1:]))
    if downs == 0:
        return "ascending"
    if ups == 0:
        return "descending"
    return "mixed order"


def _missing_speeds(short_s: float, long_s: float, step: float) -> list[str]:
    series = HALF_STOP_SPEEDS if abs(step - 0.5) < 0.05 else THIRD_STOP_SPEEDS
    lo, hi = math.log2(short_s), math.log2(long_s)
    found = [s for s in series if lo + 0.12 < math.log2(s) < hi - 0.12]
    return [format_exposure(s) for s in sorted(found)]


def _name(frame: FrameMeta) -> str:
    return frame.exposure_text or format_exposure(frame.exposure_s)


def _step_name(step: float) -> str:
    for value, name in ((1 / 3, "1/3-stop"), (1 / 2, "1/2-stop"), (2 / 3, "2/3-stop"), (1.0, "1-stop")):
        if abs(step - value) < 1e-6:
            return name
    return f"{step:.2f}-stop"
