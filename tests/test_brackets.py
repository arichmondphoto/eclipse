from datetime import datetime, timedelta

import pytest

from conftest import ECLIPSE_2024, make_frames
from tse_core.brackets import detect_brackets


def test_2024_single_bracket(frames_2024):
    report = detect_brackets(frames_2024)
    assert report.sequence_count == 1
    assert len(report.brackets) == 1
    b = report.brackets[0]
    assert b.count == 36
    assert b.step_stops == pytest.approx(1 / 3)
    assert b.range_stops == pytest.approx(12.6, abs=0.1)
    assert b.order == "ascending"
    assert b.shortest == "1/1600" and b.longest == "4.0"
    assert [g.missing_speeds for g in b.gaps] == [["1/3", "1/2.5"], ["2"]]
    assert all(f.bracket_id == 0 for f in frames_2024)


def test_repeated_short_brackets_are_split():
    seq = ["1/1000", "1/250", "1/60", "1/15", "1/4"] * 3
    report = detect_brackets(make_frames(seq))
    assert len(report.brackets) == 3
    assert all(b.count == 5 and b.step_stops == pytest.approx(2.0, abs=0.01) for b in report.brackets)


def test_descending_and_repeat_back_up():
    seq = ["4", "1", "1/4", "1/15", "1/60", "1/60", "1/15", "1/4", "1", "4"]
    report = detect_brackets(make_frames(seq))
    assert [b.count for b in report.brackets] == [5, 5]
    assert [b.order for b in report.brackets] == ["descending", "ascending"]


def test_partial_phase_frames_form_their_own_sequence():
    partial = make_frames(["1/1000"] * 3, start=datetime(2024, 4, 8, 13, 30, 0), pause=20.0)
    totality = make_frames(ECLIPSE_2024, start=datetime(2024, 4, 8, 13, 47, 30), first_index=3)
    report = detect_brackets(partial + totality)
    assert report.sequence_count == 2
    assert report.brackets[-1].count == 36
    assert report.brackets[-1].sequence_id == 1


def test_half_stop_series_and_iso_bracketing():
    seq = ["1/500", "1/350", "1/250", "1/125", "1/90"]
    report = detect_brackets(make_frames(seq))
    b = report.brackets[0]
    assert b.step_stops == pytest.approx(0.5)
    assert [g.missing_speeds for g in b.gaps] == [["1/180"]]

    frames = make_frames(["1/100"] * 4)
    for f, iso in zip(frames, (100, 200, 400, 800)):
        f.iso = iso
    report = detect_brackets(frames)
    assert len(report.brackets) == 1 and report.brackets[0].step_stops == pytest.approx(1.0)
    assert "ISO varies" in report.brackets[0].describe()


def test_frames_without_exposure_are_reported():
    frames = make_frames(["1/100", "1/50"])
    frames[1].exposure_s = None
    report = detect_brackets(frames)
    assert report.unplaced == [1]


def test_excluded_frames_are_ignored(frames_2024):
    frames_2024[5].role = "excluded"
    report = detect_brackets(frames_2024)
    assert report.brackets[0].count == 35
