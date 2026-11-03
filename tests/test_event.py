"""The event in one place: contacts computed from the eclipse and the site, in the camera's clock."""

from datetime import datetime, timedelta

import pytest

from conftest import make_frames
from tse_core.event import camera_offset_h, check_fit, compute_contacts
from tse_core.project import EventProject

# Poplar Bluff, 2024: published contacts 18:56:21 and 19:00:29 UT (as in test_phase2).
POPLAR = (36.757, -90.393, 150.0)


def _secs(a: datetime, b: datetime) -> float:
    return abs((a - b).total_seconds())


def test_contacts_in_the_camera_clock():
    """A camera on CST (UTC-6) running 117 s slow shows second contact at 12:56:21 - 1:57 = 12:54:24."""
    frames = make_frames(["1/250", "1/60"], start=datetime(2024, 4, 8, 12, 55, 0))
    con = compute_contacts(frames, *POPLAR, utc_offset_h=-6, clock_error_s=117)
    assert _secs(con.c2_ut.replace(tzinfo=None), datetime(2024, 4, 8, 18, 56, 21)) < 3
    assert _secs(con.c3_ut.replace(tzinfo=None), datetime(2024, 4, 8, 19, 0, 29)) < 3
    assert _secs(con.c2_camera, datetime(2024, 4, 8, 12, 54, 24)) < 3
    assert _secs(con.c3_camera, datetime(2024, 4, 8, 12, 58, 32)) < 3
    assert con.c2_camera.tzinfo is None and "UTC-6" in con.describe()


def test_time_zone_from_the_files_wins():
    frames = make_frames(["1/250"], start=datetime(2024, 4, 8, 13, 55, 0))
    frames[0].timestamp = "2024-04-08T13:55:00-05:00"
    assert camera_offset_h(frames, -6) == (-5.0, "files")
    con = compute_contacts(frames, *POPLAR, utc_offset_h=-6)
    assert _secs(con.c2_camera, datetime(2024, 4, 8, 13, 56, 21)) < 3


def test_needs_a_time_zone_and_a_total_eclipse():
    frames = make_frames(["1/250"], start=datetime(2024, 4, 8, 13, 55, 0))
    with pytest.raises(ValueError, match="time zone"):
        compute_contacts(frames, *POPLAR)
    with pytest.raises(ValueError, match="not total"):  # Chicago: partial only
        compute_contacts(frames, 41.88, -87.63, 180, utc_offset_h=-5)
    with pytest.raises(ValueError, match="not total"):  # east longitude by mistake: no eclipse there
        compute_contacts(frames, 36.757, 90.393, 150, utc_offset_h=-5)


def test_fit_accepts_frames_inside_and_partial_phase_outside():
    c2, c3 = datetime(2024, 4, 8, 13, 56, 21), datetime(2024, 4, 8, 14, 0, 29)
    frames = (make_frames(["1/250", "1/60", "1/15"], start=datetime(2024, 4, 8, 13, 40)) +
              make_frames(["1/250", "1/60", "1/15"], start=datetime(2024, 4, 8, 13, 57), first_index=3))
    fit = check_fit(frames, c2, c3)
    assert fit.ok and (fit.inside, fit.outside) == (3, 3)


def test_fit_warns_when_a_bracket_is_cut_and_says_which_clock_error_fits():
    """Five frames shot in one go from 13:56:15: four before computed second contact (13:56:21). If
    they were all in totality the camera was slow by at least 6 s (13:56:21 - 13:56:15)."""
    c2, c3 = datetime(2024, 4, 8, 13, 56, 21), datetime(2024, 4, 8, 14, 0, 29)
    frames = make_frames(["1/250", "1/60", "1/15", "1/4", "1", "2", "4"], start=datetime(2024, 4, 8, 13, 56, 15))
    fit = check_fit(frames, c2, c3, clock_error_s=10)
    assert not fit.ok and "cut by the contact times" in fit.warnings[0]
    lo, hi = fit.suggested_clock_error
    assert lo == pytest.approx(10 + 6, abs=0.01)
    t_last = max(f.datetime() for f in frames)
    assert hi == pytest.approx(10 + (c3 - t_last).total_seconds(), abs=0.01)


def test_fit_warns_when_no_frame_is_inside():
    c2, c3 = datetime(2024, 4, 8, 13, 56, 21), datetime(2024, 4, 8, 14, 0, 29)
    frames = make_frames(["1/250", "1/60"], start=datetime(2024, 4, 8, 12, 56, 30))  # an hour off: zone wrong
    fit = check_fit(frames, c2, c3)
    assert not fit.ok and "none of your frames" in fit.warnings[0]


def test_old_projects_move_typed_contacts_into_the_camera_clock(tmp_path):
    """Before, frames were compared after adding the clock offset; now the contacts are kept in the
    camera clock: the same frames are inside."""
    old = EventProject(name="e", second_contact="2024-04-08T13:56:21", third_contact="2024-04-08T14:00:29",
                       camera_clock_offset_s=3600 + 4 * 60)
    old.save(tmp_path / "e.json")
    p = EventProject.load(tmp_path / "e.json")
    assert p.camera_clock_offset_s == 0.0
    assert p.second_contact == "2024-04-08T12:52:21" and p.third_contact == "2024-04-08T12:56:29"
    frame = make_frames(["1/250"], start=datetime(2024, 4, 8, 12, 54, 0))[0]
    t = frame.datetime()
    assert (datetime(2024, 4, 8, 13, 56, 21) <= t + timedelta(seconds=3840) <= datetime(2024, 4, 8, 14, 0, 29))
    c2, c3 = p.contact_times()
    assert c2 <= t <= c3
