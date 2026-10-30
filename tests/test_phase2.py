"""Phase 2: eclipse circumstances, Moon fit, alignment, Sun centre."""

import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from tse_core import cli
from tse_core.align import AlignFrame, AlignParams, Transform, align_pair, align_sequence, prepare, warp
from tse_core.eclipse import LocalEclipse, Site, elements_for, load_elements
from tse_core.frames import FrameMeta
from tse_core.geometry import find_moon, solve_sun_centre
from tse_core.pipeline import run_alignment
from tse_core.rawio import LoadedFrame, superpixel
from tse_core.synthetic import SyntheticScene

WAPPAPELLO = Site(36.942202, -90.334287, 110)
CDT = timezone(timedelta(hours=-5))


# --------------------------------------------------------------- eclipse

# EclipseWise Table 5 (2024 Apr 08), local daylight time -> UT. City coordinates are approximate,
# so contacts are checked to 3 s.
CITIES = [
    ("Poplar Bluff", 36.757, -90.393, "18:56:21", "19:00:29"),
    ("Dallas", 32.7767, -96.797, "18:40:44", "18:44:32"),
    ("Cleveland", 41.4993, -81.6944, "19:13:46", "19:17:35"),
    ("Indianapolis", 39.7684, -86.1581, "19:06:05", "19:09:55"),
    ("Waco", 31.5493, -97.1467, "18:38:02", "18:42:14"),
]


@pytest.mark.parametrize("name,lat,lon,c2,c3", CITIES)
def test_contacts_match_published_tables(name, lat, lon, c2, c3):
    circ = LocalEclipse(elements_for("2024-04-08"), Site(lat, lon, 150), delta_t=69.1).circumstances()
    for got, want in ((circ.c2, c2), (circ.c3, c3)):
        ref = datetime.fromisoformat(f"2024-04-08T{want}+00:00")
        assert abs((got - ref).total_seconds()) <= 3.0, name


def test_wappapello_circumstances():
    circ = LocalEclipse(elements_for("2024-04-08"), WAPPAPELLO).circumstances()
    assert circ.kind == "total"
    assert 245 < circ.duration_s < 256
    assert circ.umbral_depth_pct > 90
    assert circ.ratio == pytest.approx(1.055, abs=0.002)


def test_elements_lookup():
    assert elements_for("2017-08-21").type == "T"
    assert elements_for(datetime(2024, 4, 9)).date == "2024-04-08"  # next day (UT) still matches
    with pytest.raises(KeyError):
        elements_for("2020-01-01")
    assert "2026-08-12" in load_elements()


def test_eclipse_command(capsys):
    assert cli.main(["eclipse", "--date", "2024-04-08", "--lat", "36.942202", "--lon", "-90.334287",
                     "--utc-offset", "-5"]) == 0
    out = capsys.readouterr().out
    assert "C2 totality begins" in out and "(13:56:" in out


# ------------------------------------------------------------ Moon and Sun


def test_find_moon_subpixel():
    sc = SyntheticScene(size=384, sun_radius=50)
    img, info = sc.frame(0.02, 191.3, 190.7, 2.4, -1.1, seed=4)
    m = find_moon(img, 53)
    assert math.hypot(m.x - info["moon_x"], m.y - info["moon_y"]) < 0.3
    assert m.radius == pytest.approx(info["moon_r"], abs=2.0)


def _track(ecl, times, H, theta_deg, sun_r, clock_error):
    th = math.radians(theta_deg)
    R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    P = np.array([[-1.0, 0.0], [0.0, -1.0]])
    pts = []
    for t in times:
        e, n = ecl.state_at(t + timedelta(seconds=clock_error)).offset()
        pts.append(tuple(np.array(H) + sun_r * (R @ P @ np.array([e, n]))))
    return pts


def test_sun_centre_with_known_clock():
    ecl = LocalEclipse(elements_for("2024-04-08"), WAPPAPELLO)
    start = datetime(2024, 4, 8, 18, 55, 5, tzinfo=timezone.utc)  # camera UT; camera ~2 min slow
    times = [start + timedelta(seconds=4 * k) for k in range(37)]
    H, sun_r = (4000.0, 2800.0), 413.0
    moons = _track(ecl, times, H, 12.0, sun_r, 120.0)
    ratio = ecl.state_at(times[18]).ratio
    sc = solve_sun_centre(times, moons, sun_r * ratio, ecl, clock_error_s=120.0)
    assert math.hypot(sc.x - H[0], sc.y - H[1]) < 0.05
    assert sc.track_rms_px < 0.01
    assert sc.speed_ratio == pytest.approx(1.0, abs=0.01)


def test_sun_centre_with_unknown_clock_is_bounded():
    ecl = LocalEclipse(elements_for("2024-04-08"), WAPPAPELLO)
    start = datetime(2024, 4, 8, 18, 55, 5, tzinfo=timezone.utc)
    times = [start + timedelta(seconds=4 * k) for k in range(37)]
    H, sun_r = (4000.0, 2800.0), 413.0
    moons = _track(ecl, times, H, -30.0, sun_r, 120.0)
    sc = solve_sun_centre(times, moons, sun_r * ecl.state_at(times[18]).ratio, ecl)
    lo, hi = sc.clock_error_range_s
    assert lo <= 120.0 <= hi  # the true clock error lies in the window that keeps every frame in totality
    assert math.hypot(sc.x - H[0], sc.y - H[1]) <= sc.along_track_uncertainty_px + 0.1


# --------------------------------------------------------------- alignment


def test_pair_alignment_shift_and_rotation():
    sc = SyntheticScene(size=512, sun_radius=60)
    c = 255.5
    p = AlignParams()
    a, _ = sc.frame(0.016, c, c, 0, 0, seed=1)
    ma = find_moon(a, 63)
    ap, am = prepare(a, ma, p, saturated=a >= 0.97)
    b, _ = sc.frame(0.016, c + 4.1, c + 2.2, 4, 1, angle_deg=0.3, seed=3)
    mb = find_moon(b, 63)
    bp, bm = prepare(b, mb, p, saturated=b >= 0.97)
    tf, pk, _ = align_pair(ap, am, bp, bm, p, (c, c), corona_centre=(ma.x, ma.y), refine=True)
    x, y = tf.apply_point(c + 4.1, c + 2.2)
    assert math.hypot(x - c, y - c) < 0.1
    assert tf.angle_deg == pytest.approx(-0.3, abs=0.06)
    assert pk.psr > 20


def test_sequence_alignment_over_eight_stops():
    sc = SyntheticScene(size=448, sun_radius=55)
    c = 223.5
    rng = np.random.default_rng(5)
    exps = [1e-3 * 2 ** (k / 1.25) for k in range(11)]
    frames, truth = [], []
    for i, e in enumerate(exps):
        sx, sy = rng.uniform(-5, 5, 2)
        md = -12 + 24 * i / 10  # the Moon crosses the Sun during the sequence
        img, _ = sc.frame(e, c + sx, c + sy, md, 0.3 * md, seed=20 + i)
        frames.append(AlignFrame(i, img, e, time_s=8.0 * i))
        truth.append((sx, sy))
    res = align_sequence(frames, AlignParams(), expected_moon_radius=58)
    anchor = next(r for r in res if math.isinf(r.psr)).index
    for r, (sx, sy) in zip(res, truth):
        x, y = r.transform.apply_point(c + sx, c + sy)
        assert math.hypot(x - (c + truth[anchor][0]), y - (c + truth[anchor][1])) < 0.35, r.index
        assert abs(r.transform.angle_deg) < 0.02  # no rotation invented


def test_warp_moves_content_by_the_transform():
    img = np.random.default_rng(0).random((64, 80)).astype(np.float32)
    tf = Transform(3.0, -1.0, 0.0, 1.0, 39.5, 31.5)
    moved = warp(img, tf, order=1)
    # Content at (x, y) in the frame lands at (x + 3, y - 1) in the reference.
    assert moved[20, 30] == pytest.approx(img[21, 27])
    assert tf.apply_point(10.0, 12.0) == pytest.approx((13.0, 11.0))
    small_rot = Transform(0.0, 0.0, 90.0, 1.0, 0.0, 0.0)
    assert small_rot.apply_point(1.0, 0.0) == pytest.approx((0.0, 1.0))


# ------------------------------------------------------------------- RAW


def test_superpixel_calibration():
    h, w = 8, 10
    bayer = np.full((h, w), 600.0, np.float32)  # 200 above black 400
    bayer[0::2, 0::2] = 16383  # red photosites saturated
    colors = np.zeros((h, w), np.uint8)
    colors[0::2, 1::2] = 1
    colors[1::2, 0::2] = 3
    colors[1::2, 1::2] = 2
    lf = superpixel(bayer, colors, "RGBG", np.array([400, 400, 400, 400], np.float32), 16383.0)
    assert lf.rgb.shape == (4, 5, 3) and lf.binning == 2
    assert lf.rgb[0, 0, 1] == pytest.approx(200 / (16383 - 400))
    assert lf.saturated.all()


# --------------------------------------------------------------- pipeline


def test_pipeline_end_to_end_with_eclipse_geometry(tmp_path):
    """A synthetic 2024 bracket whose Moon moves as it did at Lake Wappapello."""
    ecl = LocalEclipse(elements_for("2024-04-08"), WAPPAPELLO)
    sc = SyntheticScene(size=448, sun_radius=55)
    rng = np.random.default_rng(9)
    clock_error = 115.0
    cam0 = datetime(2024, 4, 8, 12, 55, 5, tzinfo=timezone(timedelta(hours=-6)))
    frames, scenes = [], {}
    sun_r_bin = 55.0
    for i in range(9):
        t = cam0 + timedelta(seconds=15 * i)
        e = 1e-3 * 2 ** (i / 1.3)
        east, north = ecl.state_at(t + timedelta(seconds=clock_error)).offset()
        # Image: north up, east left (theta = 0); offsets in superpixels.
        mdx, mdy = -east * sun_r_bin, -north * sun_r_bin
        sx, sy = rng.uniform(-4, 4, 2)
        img, _ = sc.frame(e, 223.5 + sx, 223.5 + sy, mdx, mdy, seed=40 + i)
        fm = FrameMeta(path=f"f{i}", file_name=f"f{i}.nef", index=i, exposure_s=e, exposure_text=f"{e:.4f}",
                       aperture_f=1.0, iso=100, focal_length_mm=400.0, timestamp=t.isoformat())
        frames.append(fm)
        scenes[i] = (img, (sx, sy))

    def loader(fm, _calib):
        img, _ = scenes[fm.index]
        return LoadedFrame(np.repeat(img[..., None], 3, axis=2), img >= 0.97, 2)

    res = run_alignment(frames, loader, AlignParams(), window=448, site=WAPPAPELLO,
                        clock_error_s=clock_error)
    ref = res.reference_index
    sx, sy = scenes[ref][1]
    true_sun_full = ((223.5 + sx) * 2 + 0.5, (223.5 + sy) * 2 + 0.5)
    assert res.sun is not None
    assert math.hypot(res.sun["x"] - true_sun_full[0], res.sun["y"] - true_sun_full[1]) < 2.0
    assert abs(res.sun["rotation_deg"]) < 2.0  # north is up in this synthetic set
    res.save(tmp_path / "a.json")
    assert (tmp_path / "a.json").stat().st_size > 100
    assert "Sun centre" in res.describe()
