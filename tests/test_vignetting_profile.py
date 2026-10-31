import numpy as np
import pytest

from conftest import ECLIPSE_2024, make_frames
from tse_core import cli
from tse_core.calibration import plan_calibration, write_synthetic_flat
from tse_core.fits import read_fits, write_fits
from tse_core.lensfun import corner_falloff, vignetting_map
from tse_core.vignetting_profile import (
    VignettingProfile, fit_profile_from_files, fit_profile_from_planes, load_profile_library,
    match_profile, profile_map,
)

D_LENS = "AF VR Zoom-Nikkor 80-400mm f/4.5-5.6D ED"
TRUE = {"R": (-0.30, 0.04, -0.02), "G": (-0.28, 0.03, -0.015), "B": (-0.33, 0.05, -0.03)}
PATTERN = {(0, 0): "R", (0, 1): "G", (1, 0): "G", (1, 1): "B"}


def synthetic_cfa_flat(rng, w=900, h=600, tilt=0.02, dust=8, level=8000.0):
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.zeros((h, w))
    light = 1 + tilt * (xx - w / 2) / (w / 2)
    for (dy, dx), c in PATTERN.items():
        img[dy::2, dx::2] = (level * vignetting_map(w, h, TRUE[c]) * light)[dy::2, dx::2]
    for _ in range(dust):
        cx, cy, r = rng.integers(60, w - 60), rng.integers(60, h - 60), rng.integers(8, 25)
        img[(xx - cx) ** 2 + (yy - cy) ** 2 < r * r] *= 0.8
    return rng.poisson(img).astype(np.float32)


def split(img):
    planes = {}
    for (dy, dx), c in PATTERN.items():
        planes[c if c not in planes else c + "2"] = (img[dy::2, dx::2], 2.0)
    return planes


def test_fit_recovers_each_channel_despite_dust_and_tilt():
    rng = np.random.default_rng(3)
    flats = [split(synthetic_cfa_flat(rng)) for _ in range(3)]
    prof = fit_profile_from_planes(flats, 900, 600, 1.0, block=12, lens_model=D_LENS,
                                   focal_length_mm=400, aperture_f=8)
    for c, t in TRUE.items():
        assert corner_falloff(prof.channel_terms[c])[0] == pytest.approx(corner_falloff(t)[0], abs=0.004)
    assert 2.5 < prof.light_tilt_pct < 5.0  # the 4 % edge-to-edge tilt was found and removed
    assert prof.rejected_pct > 0  # dust blocks rejected
    assert prof.rms_residual < 0.01


def test_profile_carries_to_another_camera_with_the_same_sensor_size():
    prof = VignettingProfile(D_LENS, 400, 8, TRUE["G"], {}, crop_factor=1.0)
    d850 = profile_map(prof, 828, 552)   # D850-like aspect, scaled down
    d800 = profile_map(prof, 736, 491)   # D800-like aspect, scaled down
    assert d850[0, 0] == pytest.approx(d800[0, 0], abs=0.003)
    assert d850[276, 414] == pytest.approx(1.0, abs=1e-3)


def test_cfa_flat_uses_each_channel():
    prof = VignettingProfile(D_LENS, 400, 8, TRUE["G"], dict(TRUE), 1.0)
    m = profile_map(prof, 400, 300, cfa_pattern="RGGB")
    full_r = vignetting_map(400, 300, TRUE["R"])
    full_b = vignetting_map(400, 300, TRUE["B"])
    assert m[0, 0] == pytest.approx(full_r[0, 0]) and m[-1, -1] == pytest.approx(full_b[-1, -1])


def test_save_load_and_library(tmp_path):
    prof = VignettingProfile(D_LENS, 400, 8, TRUE["G"], dict(TRUE), 1.0, flat_count=20)
    prof.save(tmp_path / "a.json")
    (tmp_path / "junk.json").write_text("{}")
    lib = load_profile_library(tmp_path)
    assert len(lib) == 1 and lib[0].channel_terms["B"] == TRUE["B"]


def test_matching_rules():
    prof = VignettingProfile(D_LENS, 400, 8, TRUE["G"], {}, 1.0)
    f = make_frames(["1/100"])[0]
    f.lens_model = D_LENS
    assert match_profile(f, [prof]) is prof
    f.aperture_f = 9.0  # 1/3 stop away: accepted
    assert match_profile(f, [prof]) is prof
    f.aperture_f = 11.0  # one stop away: refused
    assert match_profile(f, [prof]) is None
    f.aperture_f, f.lens_model = 8.0, "AF-S Nikkor 80-400mm f/4.5-5.6G ED VR"
    assert match_profile(f, [prof]) is None


def test_plan_uses_library_profile_for_the_d_lens(lens_db):
    frames = make_frames(ECLIPSE_2024)
    for f in frames:
        f.lens_model = D_LENS
    prof = VignettingProfile(D_LENS, 400, 8, TRUE["G"], dict(TRUE), 1.0)
    plan = plan_calibration(frames, lens_db, profiles=[prof])
    assert {fc.vignetting for fc in plan.frames} == {"profile"}
    assert plan.frames[0].corner_brightness == pytest.approx(corner_falloff(TRUE["G"])[0])
    assert "Own-flats profile" in plan.describe()
    # Without the profile the frames go uncorrected and the warning says what to do.
    plan = plan_calibration(frames, lens_db)
    assert any("totality flat-profile" in w for w in plan.warnings)


def test_fits_master_flat_and_cli(tmp_path, capsys):
    rng = np.random.default_rng(5)
    stack = np.mean([synthetic_cfa_flat(rng, dust=0) for _ in range(2)], axis=0)
    write_fits(tmp_path / "master_flat.fit", stack, {"BAYERPAT": "RGGB"})
    prof = fit_profile_from_files([str(tmp_path / "master_flat.fit")], block=12)
    assert corner_falloff(prof.channel_terms["B"])[0] == pytest.approx(corner_falloff(TRUE["B"])[0], abs=0.004)

    out = tmp_path / "profile.json"
    code = cli.main(["flat-profile", str(tmp_path / "master_flat.fit"), "--block", "12", "--lens", D_LENS,
                     "--focal", "400", "--aperture", "8", "-o", str(out)])
    assert code == 0 and "Profile saved" in capsys.readouterr().out
    loaded = VignettingProfile.load(out)
    assert loaded.lens_model == D_LENS and loaded.aperture_f == 8


def test_synthetic_cfa_flat_file(tmp_path):
    path = write_synthetic_flat(tmp_path / "f.fit", 200, 100, TRUE["G"], channel_terms=dict(TRUE), cfa_pattern="RGGB")
    data, _ = read_fits(path)
    assert data[0, 0] == pytest.approx(vignetting_map(200, 100, TRUE["R"])[0, 0])


@pytest.mark.parametrize("a,b,same", [
    ("80-400mm f/4.5-5.6 D VR", "AF VR Zoom-Nikkor 80-400mm f/4.5-5.6D ED", True),   # exifread vs exiftool
    ("80-400mm f/4.5-5.6 D VR", "80-400mm f/4.5-5.6 D VR", True),
    ("80-400mm f/4.5-5.6 D VR", "AF-S Nikkor 80-400mm f/4.5-5.6G ED VR", False),    # G lens
    ("80-400mm f/4.5-5.6 D VR", "80-400mm f/4.5-5.6", True),                        # designation unknown
    ("80-400mm f/4.5-5.6 D VR", "70-300mm f/4.5-5.6 G VR", False),
    ("80-400mm f/4.5-5.6 D VR", "80-400mm f/4-5.6", False),
    ("Some Lens", "Other Lens", False),
])
def test_same_lens(a, b, same):
    from tse_core.vignetting_profile import same_lens

    assert same_lens(a, b) is same


def test_user_profile_matches_frames_read_by_either_reader():
    # The profile fitted from the 2024 flats, as saved on the user's machine.
    prof = VignettingProfile("80-400mm f/4.5-5.6 D VR", 400, 8, (-0.07130, -0.07298, 0.04993), {}, 1.0)
    assert prof.corner()[0] == pytest.approx(0.906, abs=0.001)
    for name in ("80-400mm f/4.5-5.6 D VR", "AF VR Zoom-Nikkor 80-400mm f/4.5-5.6D ED"):
        f = make_frames(["1/100"])[0]
        f.lens_model = name
        assert match_profile(f, [prof]) is prof


def test_no_lensfun_warning_when_profile_covers_every_frame(lens_db):
    frames = make_frames(ECLIPSE_2024)
    for f in frames:
        f.lens_model, f.lens_spec = "80-400mm f/4.5-5.6 D VR", [80, 400, 4.5, 5.6]
    prof = VignettingProfile("80-400mm f/4.5-5.6 D VR", 400, 8, (-0.0713, -0.07298, 0.04993), {}, 1.0)
    plan = plan_calibration(frames, lens_db, profiles=[prof])
    assert {fc.vignetting for fc in plan.frames} == {"profile"}
    assert plan.warnings == [] and plan.lens is None
    assert "Lens profile: not chosen" not in plan.describe()
    # Without the profile, the D lens warning must not point at another maker's lens.
    plan = plan_calibration(frames, lens_db)
    text = " ".join(plan.warnings)
    assert "no vignetting data" in text and "Sigma" not in text and "totality flat-profile" in text
