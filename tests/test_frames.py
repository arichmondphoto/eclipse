from datetime import timedelta
from fractions import Fraction

import pytest

from conftest import ECLIPSE_2024
from tse_core.frames import FrameMeta, format_exposure, parse_exif_datetime, to_float


def test_parse_subseconds_and_offset():
    dt = parse_exif_datetime("2024:04:08 13:47:31", "25", "-05:00")
    assert dt.microsecond == 250000
    assert dt.utcoffset() == timedelta(hours=-5)


def test_parse_exiftool_composite():
    dt = parse_exif_datetime("2024:04:08 13:47:31.07-05:00")
    assert dt.microsecond == 70000 and dt.utcoffset() == timedelta(hours=-5)


@pytest.mark.parametrize("value,expected", [
    ("1/1600", 1 / 1600), (Fraction(5, 8), 0.625), ("18.0 mm", 18.0), ([Fraction(7, 2)], 3.5), (None, None), ("", None),
])
def test_to_float(value, expected):
    assert to_float(value) == pytest.approx(expected) if expected is not None else to_float(value) is None


def test_format_exposure_round_trips_camera_names():
    for text in ECLIPSE_2024:
        expected = text[:-2] if text.endswith(".0") else text
        assert format_exposure(to_float(text)) == expected


def test_exposure_value_combines_time_iso_aperture():
    a = FrameMeta(path="a", exposure_s=1.0, iso=100, aperture_f=8.0)
    b = FrameMeta(path="b", exposure_s=0.5, iso=200, aperture_f=8.0)
    c = FrameMeta(path="c", exposure_s=1.0, iso=100, aperture_f=5.6)
    assert a.exposure_value() == pytest.approx(b.exposure_value())
    assert c.exposure_value() - a.exposure_value() == pytest.approx(1.03, abs=0.01)


def test_overrides_are_recorded():
    f = FrameMeta(path="a", iso=100)
    f.apply_override("iso", 200)
    assert f.iso == 200 and f.overrides == ["iso"]
    with pytest.raises(KeyError):
        f.apply_override("nonsense", 1)


@pytest.mark.parametrize("text,seconds", [
    ("+3848", 3848), ("-64:08", -3848), ("1:04:08", 3848), ("1h4m8s", 3848), ("-4m", -240), ("45s", 45), (90, 90),
])
def test_parse_duration(text, seconds):
    from tse_core.frames import parse_duration

    assert parse_duration(text) == seconds


def test_clock_offset_moves_frames_into_totality():
    from datetime import datetime

    from conftest import make_frames
    from tse_core.inventory import Inventory

    # Camera clock an hour behind (standard time) and 4 minutes slow.
    inv = Inventory(make_frames(ECLIPSE_2024, start=datetime(2024, 4, 8, 12, 43, 40)))
    c2, c3 = datetime(2024, 4, 8, 13, 47, 33), datetime(2024, 4, 8, 13, 51, 50)
    inv.mark_outside(c2, c3)
    assert all(f.role == "partial" for f in inv.frames)
    inv.mark_outside(c2, c3, clock_offset_s=3600 + 4 * 60)
    assert all(f.role == "light" for f in inv.frames)
