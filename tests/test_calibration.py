import pytest

from conftest import ECLIPSE_2024, make_frames
from tse_core.calibration import CalibrationFrames, choose_lens, plan_calibration, write_synthetic_flat
from tse_core.fits import read_fits


def test_lensfun_used_when_no_flats(frames_2024, lens_db):
    plan = plan_calibration(frames_2024, lens_db)
    assert plan.lens.model == "Nikon AF-S Nikkor 80-400mm f/4.5-5.6G ED VR"
    assert {fc.vignetting for fc in plan.frames} == {"lensfun"}
    assert plan.frames[0].corner_brightness == pytest.approx(0.8248, abs=1e-3)
    assert all(fc.bias == "synthetic" and fc.synthetic_bias_adu == 400.0 for fc in plan.frames)
    assert all(fc.cosmetic == "find hot pixels" for fc in plan.frames)
    assert "82% of centre" in plan.describe()


def test_user_flats_take_priority_over_lensfun(frames_2024, lens_db):
    flats = make_frames(["1/100"] * 5, first_index=100)
    plan = plan_calibration(frames_2024, lens_db, CalibrationFrames(flats=flats))
    assert {fc.vignetting for fc in plan.frames} == {"flat"}
    assert all(fc.lens_terms is None for fc in plan.frames)


def test_flats_at_other_aperture_do_not_match(frames_2024, lens_db):
    flats = make_frames(["1/100"] * 5, aperture=5.6, first_index=100)
    plan = plan_calibration(frames_2024, lens_db, CalibrationFrames(flats=flats))
    assert {fc.vignetting for fc in plan.frames} == {"lensfun"}


def test_darks_matched_by_exposure_and_iso(frames_2024, lens_db):
    darks = make_frames(["1/1600", "4.0", "4.0"], first_index=200)
    darks += make_frames(["1/2"], iso=400, first_index=300)
    plan = plan_calibration(frames_2024, lens_db, CalibrationFrames(darks=darks))
    matched = {frames_2024[fc.frame_index].exposure_text for fc in plan.frames if fc.dark_group}
    assert matched == {"1/1600", "4.0"}
    assert any("no dark with the same exposure" in w for w in plan.warnings)


def test_lensfun_can_be_turned_off(frames_2024, lens_db):
    plan = plan_calibration(frames_2024, lens_db, use_lensfun=False)
    assert {fc.vignetting for fc in plan.frames} == {"none"}


def test_forced_lens_profile(frames_2024, lens_db):
    lens, _, warning = choose_lens(frames_2024, lens_db, "Nikon AF-S Nikkor 80-400mm f/4.5-5.6G ED VR")
    assert lens is not None and warning == ""
    lens, _, warning = choose_lens(frames_2024, lens_db, "Made Up 12mm f/1.0")
    assert lens is None and "not in the lensfun database" in warning


def test_mixed_lenses_need_a_choice(lens_db):
    frames = make_frames(ECLIPSE_2024[:3])
    frames[1].lens_model = "Something Else 100-500mm"
    lens, _, warning = choose_lens(frames, lens_db)
    assert lens is None and "more than one lens" in warning


def test_synthetic_flat_file(tmp_path, lens_db, frames_2024):
    plan = plan_calibration(frames_2024, lens_db)
    path = write_synthetic_flat(tmp_path / "flat.fit", 120, 80, plan.frames[0].lens_terms)
    data, header = read_fits(path)
    assert data.shape == (80, 120)
    assert data[40, 60] == pytest.approx(1.0, abs=1e-3)  # centre; f/8 brightens slightly mid-field
    assert data.min() == pytest.approx(0.8248, abs=0.01)
    assert header["IMAGETYP"] == "Master Flat"
