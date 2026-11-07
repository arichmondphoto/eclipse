"""Frames whose files do not record the lens (a camera on a telescope, a manual lens with no chip):
no f-number and no focal length, or 0. Any camera, any lens: nothing may depend on them.

On a telescope the f-number never changes, so the light gathered follows exposure time and ISO;
flats through the same optics match the lights by definition; and the focal length only gives a
first guess at the Moon's size, which the Moon search does without.
"""

from tse_core.brackets import detect_brackets
from tse_core.calibration import CalibrationFrames, plan_calibration
from tse_core.inventory import Inventory
from tse_core.project import EventProject

from conftest import ECLIPSE_2024, make_frames


def _bare(exposures, **kw):
    """Frames as a camera on a telescope writes them: no lens, no f-number, no focal length."""
    frames = make_frames(exposures, aperture=None, focal=None, **kw)
    for f in frames:
        f.lens_model, f.lens_spec = "", None
    return frames


def test_brackets_are_found_without_an_f_number(frames_2024):
    """The 2024 bracket without its f-number: the same bracket, nothing left unplaced."""
    with_lens = detect_brackets(frames_2024)
    bare = detect_brackets(_bare(ECLIPSE_2024))
    assert bare.unplaced == []
    assert len(bare.brackets) == len(with_lens.brackets)
    assert [b.frame_indices for b in bare.brackets] == [b.frame_indices for b in with_lens.brackets]
    assert abs(bare.brackets[0].range_stops - with_lens.brackets[0].range_stops) < 1e-9
    assert "f-number" in bare.describe()                  # says what it assumed


def test_stacking_gets_exposure_ratios_without_an_f_number():
    """Stack orders frames by their nominal brightness and falls back on it where two frames share
    too few pixels: without an f-number every frame was taken as the reference's exposure (1:1)."""
    from tse_core.workflow import _frame_infos

    frames = _bare(["1/1000", "1/250", "1/60", "1/15"])
    project = EventProject(name="e")
    project.set_inventory(Inventory(frames))
    alignment = {"reference_index": frames[0].index,
                 "frames": [{"index": f.index, "file_name": f.file_name, "matrix": [[1, 0, 0], [0, 1, 0]],
                             "moon_ref": [10.0, 10.0], "moon_frame": {"radius": 5.0}} for f in frames]}
    _, _, infos, _ = _frame_infos(project, alignment, include_rejected=False)
    ratios = [i.nominal_ratio for i in infos]
    assert ratios[0] == 1.0
    assert 3.5 < ratios[1] < 4.5 and 14 < ratios[2] < 18 and 55 < ratios[3] < 75


def test_flats_without_lens_data_match_lights_without_it(lens_db):
    """Flats through the same telescope, no lens data on either: used. (They were ignored, so the
    frames got no vignetting correction at all.)"""
    lights = _bare(ECLIPSE_2024)
    flats = _bare(["1/100"] * 5, first_index=100)
    plan = plan_calibration(lights, lens_db, CalibrationFrames(flats=flats))
    assert {fc.vignetting for fc in plan.frames} == {"flat"}


def test_flats_with_lens_data_still_need_the_same_settings(lens_db, frames_2024):
    """Recorded settings still decide: flats at another f-number do not match, and flats with no
    lens data do not match lights that have it (they may be through other optics)."""
    flats = make_frames(["1/100"] * 5, aperture=5.6, first_index=100)
    assert {fc.vignetting for fc in plan_calibration(frames_2024, lens_db, CalibrationFrames(flats=flats)).frames} \
        != {"flat"}
    bare_flats = _bare(["1/100"] * 5, first_index=100)
    assert {fc.vignetting for fc in plan_calibration(frames_2024, lens_db,
                                                    CalibrationFrames(flats=bare_flats)).frames} != {"flat"}


def test_typed_focal_length_and_f_number_fill_only_what_the_files_lack():
    """The Frames tab's Focal length and f-number: used for frames whose files do not say, never
    over a recorded value; changing or clearing them updates those frames."""
    project = EventProject(name="e")
    frames = _bare(["1/1000", "1/250"]) + make_frames(["1/60"], aperture=8.0, focal=400.0, first_index=2)
    project.set_inventory(Inventory(frames))
    project.focal_length_mm, project.aperture_f = 1000.0, 10.0
    got = project.inventory.frames
    assert [f.focal_length_mm for f in got] == [1000.0, 1000.0, 400.0]
    assert [f.aperture_f for f in got] == [10.0, 10.0, 8.0]
    project.focal_length_mm, project.aperture_f = None, None
    got = project.inventory.frames
    assert [f.focal_length_mm for f in got] == [None, None, 400.0]
    # Saved and loaded with the project
    import json
    d = json.loads(json.dumps(project.__dict__, default=str))
    assert "focal_length_mm" in d and "aperture_f" in d


def test_align_visits_frames_in_the_same_order_whatever_the_exposure_scale():
    """Align visits frames outward from the middle exposure. Frames the same number of stops above and
    below it were ordered by rounding noise in the 16th digit, which changes when every exposure is
    scaled (f/1 assumed without an f-number instead of f/8): on 27 of Andy's 2024 frames two pairs
    swapped, the reference grew differently, and the field rotation moved by 0.008 degrees (1.07 px at
    a frame corner). Ties now go to the shorter exposure, the same at any scale."""
    from tse_core.align import visit_order
    from tse_core.frames import to_float

    times = [to_float(t) for t in ECLIPSE_2024]
    a = visit_order(times)
    for k in (64.0, 1 / 64.0, 3.7, 1e-6):
        assert visit_order([t * k for t in times]) == a
    # one stop either side of the middle: the shorter first
    mid, seq = visit_order([1 / 1000, 1 / 500, 1 / 250])
    assert mid == 1 and seq == [1, 0, 2]
