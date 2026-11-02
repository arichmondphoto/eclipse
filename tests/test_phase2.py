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


# ------------------------------------------------------------------ canon

CANON_SAMPLE = __import__("pathlib").Path(__file__).parent / "data" / "nasa_canon_sample.csv"


def test_parse_nasa_canon_sample():
    from tse_core.eclipse import parse_canon_csv

    rows = parse_canon_csv(CANON_SAMPLE.read_text())
    assert set(rows) == {"2024-04-08", "2017-08-21", "2025-03-29"}  # the -1999 row is outside 1600-2400
    e = rows["2024-04-08"]
    assert e["type"] == "T" and e["t0_tdt_hours"] == 18.0 and len(e["mu"]) == 3
    assert e["x"][0] == pytest.approx(-0.318244) and e["tan_f2"] == pytest.approx(0.004645)


def test_canon_and_bulletin_agree_for_2024(tmp_path, monkeypatch):
    from tse_core import eclipse as ecl_mod

    monkeypatch.setattr(ecl_mod, "CANON_CACHE", tmp_path / "canon.json")
    path, n = ecl_mod.update_canon(CANON_SAMPLE)
    assert n == 3 and path.is_file()
    canon = ecl_mod.load_elements(path)["2024-04-08"]
    bulletin = ecl_mod.load_elements()["2024-04-08"]
    a = LocalEclipse(canon, WAPPAPELLO, delta_t=69.2).circumstances()
    b = LocalEclipse(bulletin, WAPPAPELLO, delta_t=69.2).circumstances()
    assert abs((a.c2 - b.c2).total_seconds()) < 2 and abs((a.c3 - b.c3).total_seconds()) < 2
    # Bundled bulletin values win over the canon; a canon-only eclipse is now found.
    assert ecl_mod.elements_for("2024-04-08").source.startswith("NASA GSFC, Besselian")
    assert ecl_mod.elements_for("2025-03-29").type == "P"


def test_missing_eclipse_suggests_update(tmp_path, monkeypatch):
    from tse_core import eclipse as ecl_mod

    monkeypatch.setattr(ecl_mod, "CANON_CACHE", tmp_path / "none.json")
    with pytest.raises(KeyError, match="elements-update"):
        ecl_mod.elements_for("2045-08-12")


# ------------------------------------------------------- scale-aware defaults


def test_scaled_defaults():
    from tse_core.align import auto_window, scaled_params

    p = scaled_params(AlignParams(), 210.0)  # D850 + 400 mm, superpixels
    assert 7.5 < p.rho < 8.5 and p.mask_blur == pytest.approx(p.rho / 2)
    assert scaled_params(AlignParams(), 20.0).rho == 4.0  # short lens: floor
    assert scaled_params(AlignParams(), 1000.0).rho == 16.0  # long telescope: Druckmuller's upper value
    assert scaled_params(AlignParams(rho=11.0), 210.0).rho == 11.0  # user setting wins
    assert auto_window(210.0, 4144, 2760) == 1680
    assert auto_window(20.0, 4144, 2760) == 512
    assert auto_window(1000.0, 4144, 2760) == 2760


@pytest.mark.parametrize("radius,size", [(20, 192), (90, 768)])
def test_alignment_at_other_scales(radius, size):
    sc = SyntheticScene(size=size, sun_radius=radius)
    c = (size - 1) / 2
    rng = np.random.default_rng(3)
    frames, truth = [], []
    for i in range(5):
        e = 2e-3 * 2 ** i
        sx, sy = rng.uniform(-4, 4, 2)
        md = (-0.2 + 0.1 * i) * radius
        img, _ = sc.frame(e, c + sx, c + sy, md, 0.3 * md, seed=50 + i)
        frames.append(AlignFrame(i, img, e, time_s=10.0 * i))
        truth.append((sx, sy))
    res = align_sequence(frames, AlignParams(), expected_moon_radius=radius * 1.055)
    a = next(r for r in res if math.isinf(r.psr)).index
    for r, (sx, sy) in zip(res, truth):
        x, y = r.transform.apply_point(c + sx, c + sy)
        assert math.hypot(x - c - truth[a][0], y - c - truth[a][1]) < 0.2


# ------------------------------------------------------------- mirroring


def _track_parity(ecl, times, H, theta_deg, sun_r, clock_error, mirrored):
    th = math.radians(theta_deg)
    R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    P = np.array([[1.0, 0.0], [0.0, -1.0]]) if mirrored else np.array([[-1.0, 0.0], [0.0, -1.0]])
    return [tuple(np.array(H) + sun_r * (R @ P @ np.array(ecl.state_at(t + timedelta(seconds=clock_error)).offset())))
            for t in times]


def test_mirrored_optics_near_the_path_edge():
    # Dallas was about half-way from the central line to the edge: the centres never coincided.
    ecl = LocalEclipse(elements_for("2024-04-08"), Site(32.7767, -96.797, 150))
    c2 = ecl.circumstances().c2
    times = [c2 + timedelta(seconds=20 + 10 * k) for k in range(18)]
    H, sun_r = (2000.0, 1500.0), 800.0
    moons = _track_parity(ecl, times, H, 35.0, sun_r, 0.0, mirrored=True)
    r = sun_r * ecl.state_at(times[9]).ratio
    right = solve_sun_centre(times, moons, r, ecl, clock_error_s=0.0, mirrored=True)
    wrong = solve_sun_centre(times, moons, r, ecl, clock_error_s=0.0, mirrored=False)
    assert math.hypot(right.x - H[0], right.y - H[1]) < 0.05
    assert wrong.parity_difference_px > 5  # the choice matters here...
    assert math.hypot(wrong.other_parity_xy[0] - H[0], wrong.other_parity_xy[1] - H[1]) < 0.05
    assert any("--mirrored" in n for n in wrong.notes)  # ...and the user is told


def test_parity_irrelevant_on_the_central_line():
    ecl = LocalEclipse(elements_for("2024-04-08"), WAPPAPELLO)
    c2 = ecl.circumstances().c2
    times = [c2 + timedelta(seconds=20 + 10 * k) for k in range(18)]
    moons = _track_parity(ecl, times, (2000.0, 1500.0), 0.0, 413.0, 0.0, mirrored=False)
    sc = solve_sun_centre(times, moons, 413.0 * 1.0547, ecl, clock_error_s=0.0)
    assert sc.parity_difference_px < 1.5 and not any("--mirrored" in n for n in sc.notes)


def test_off_track_frames_are_rejected_and_reported():
    ecl = LocalEclipse(elements_for("2024-04-08"), WAPPAPELLO)
    start = datetime(2024, 4, 8, 18, 55, 5, tzinfo=timezone.utc)
    times = [start + timedelta(seconds=4 * k) for k in range(37)]
    H, sun_r = (3965.0, 2791.0), 413.5
    moons = [list(p) for p in _track(ecl, times, H, 2.7, sun_r, 117.0)]
    rng = np.random.default_rng(1)
    for p in moons:
        p[0] += rng.normal(0, 0.4)
        p[1] += rng.normal(0, 0.4)
    moons[21][1] -= 7.8  # like DSC_9108: a doubled image at 1/5 s
    moons[34][0] -= 3.1  # like DSC_9131: smeared 2.5 s frame
    moons[34][1] += 4.9
    limb = [0.4] * 37
    limb[34] = 2.9
    sc = solve_sun_centre(times, [tuple(p) for p in moons], sun_r * ecl.state_at(times[18]).ratio, ecl,
                          clock_error_s=117.0, limb_rms=limb)
    assert sc.rejected == [21, 34]
    assert math.hypot(sc.x - H[0], sc.y - H[1]) < 0.5
    assert sc.track_rms_px < 0.7 and sc.speed_ratio == pytest.approx(1.0, abs=0.05)
    assert 0.1 < sc.px_per_clock_second < 0.3


def test_nan_or_infinite_pixels_do_not_break_alignment():
    """One NaN (or infinite) pixel made every Fourier transform of a frame NaN: no peak, every
    shift 0, PSR nan, 'little usable overlap' (seen on real frames after a vignetting correction
    divided by zero). Such pixels are now left out like hot pixels; the shifts match the clean run."""
    sc = SyntheticScene(size=448, sun_radius=55)
    c = 223.5
    rng = np.random.default_rng(5)
    exps = [1e-3 * 2 ** (k / 1.25) for k in range(5)]
    clean, spoiled, truth = [], [], []
    for i, e in enumerate(exps):
        sx, sy = rng.uniform(-5, 5, 2)
        img, _ = sc.frame(e, c + sx, c + sy, -12 + 6 * i, 0, seed=20 + i)
        img = img.astype(np.float32)
        clean.append(AlignFrame(i, img.copy(), e, time_s=8.0 * i))
        bad = img.copy()
        bad[3, 3] = np.nan
        bad[400, 10] = np.inf
        spoiled.append(AlignFrame(i, bad, e, time_s=8.0 * i))
        truth.append((sx, sy))
    a = align_sequence(clean, AlignParams(), expected_moon_radius=58)
    b = align_sequence(spoiled, AlignParams(), expected_moon_radius=58)
    for ra, rb in zip(a, b):
        assert np.isfinite(rb.transform.dx) and (math.isinf(rb.psr) or rb.psr > 20)
        assert abs(ra.transform.dx - rb.transform.dx) < 0.05 and abs(ra.transform.dy - rb.transform.dy) < 0.05


def test_vignetting_map_never_reaches_zero():
    """A profile used beyond its calibration's corner can fall to zero or below; dividing by it
    made NaN pixels. The map stops at 5%."""
    from tse_core.lensfun import MIN_ILLUMINATION, vignetting_map

    m = vignetting_map(200, 150, (-2.0, 0.5, 0.0), image_crop=1.0, calib_crop=1.5)
    assert np.isfinite(m).all() and m.min() >= MIN_ILLUMINATION


def test_one_failed_frame_does_not_spoil_the_others(monkeypatch):
    """Once, one frame (a 1/8000 s exposure) was placed ~850 px away with PSR 9; that grew the
    reference's Moon mask over the whole window and every later frame failed (shift 0, PSR nan;
    seen on the 2024 D850 set). A frame placed far from where its Moon says it must be is now
    treated as failed: placed by its Moon and kept out of the reference; the others still align."""
    from tse_core import align as A

    sc = SyntheticScene(size=448, sun_radius=55)
    c = 223.5
    rng = np.random.default_rng(5)
    exps = [1e-3 * 2 ** (k / 1.25) for k in range(7)]
    frames, truth = [], []
    for i, e in enumerate(exps):
        sx, sy = rng.uniform(-5, 5, 2)
        img, _ = sc.frame(e, c + sx, c + sy, -12 + 4 * i, 0, seed=20 + i)
        frames.append(AlignFrame(i, img.astype(np.float32), e, time_s=8.0 * i))
        truth.append((sx, sy))
    real_pair = A.align_pair
    victim = frames[1].image

    def flaky(ref, ref_mask, img, img_mask, params, centre, **kw):
        tf, pk, sig = real_pair(ref, ref_mask, img, img_mask, params, centre, **kw)
        if img.shape == victim.shape and np.allclose(img, flaky.victim_prepared):
            return A.Transform(170.0, -190.0, 0.0, 1.0, *centre), A.Peak(170.0, -190.0, 0.1, 9.2), sig
        return tf, pk, sig

    p0 = AlignParams()
    m = A.find_moon(victim, 58)
    from tse_core.align import defect_mask, scaled_params, sun_radius_from_moon
    pp = scaled_params(p0, sun_radius_from_moon(m.radius))
    flaky.victim_prepared = A.prepare(victim, m, pp, saturated=victim >= pp.saturation,
                                      defects=defect_mask(victim, pp.star_sigma))[0]
    monkeypatch.setattr(A, "align_pair", flaky)
    res = align_sequence(frames, p0, expected_moon_radius=58)
    anchor = next(r for r in res if math.isinf(r.psr)).index
    for r, (sx, sy) in zip(res, truth):
        if r.index == 1:
            assert any("alignment failed" in n for n in r.notes)
            continue
        assert math.isinf(r.psr) or (np.isfinite(r.psr) and r.psr > 8), r.index
        x, y = r.transform.apply_point(c + sx, c + sy)
        assert math.hypot(x - (c + truth[anchor][0]), y - (c + truth[anchor][1])) < 0.35, r.index


def test_rotation_note_matches_the_rotation_applied():
    """On the 2024 set every frame's note said "field rotation not significant ...; not applied"
    while the frames carried rotations of -0.028 to +0.033 deg: round 1 found the drift and applied
    it, round 2 measured what was left (nothing) and its note replaced round 1's. The note must
    describe the rotation the frames actually carry."""
    sc = SyntheticScene(size=448, sun_radius=55)
    c = 223.5
    rng = np.random.default_rng(5)
    rate = 0.3  # deg per frame, 8 s apart
    frames = []
    for i, e in enumerate([1e-3 * 2 ** (k / 1.25) for k in range(7)]):
        sx, sy = rng.uniform(-5, 5, 2)
        img, _ = sc.frame(e, c + sx, c + sy, -12 + 4 * i, 0, angle_deg=rate * (i - 3), seed=20 + i)
        frames.append(AlignFrame(i, img.astype(np.float32), e, time_s=8.0 * i))
    res = align_sequence(frames, AlignParams(), expected_moon_radius=58)
    assert max(abs(r.transform.angle_deg) for r in res) > 0.8  # the drift was found and applied
    for r in res:
        if math.isinf(r.psr):
            continue
        text = " ".join(r.notes)
        assert "not applied" not in text, text
        import re
        total = re.search(r"applied \(([-+][0-9.]+) deg over the sequence", text)
        assert total and abs(abs(float(total.group(1))) - 6 * rate) < 0.05, text
