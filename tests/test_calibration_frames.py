"""Stack uses the calibration frames chosen on the Frames tab: flats, darks and bias.

Until Nov 2026 the Frames tab planned with them ("Vignetting: N frame(s) by flat") but Stack
rebuilt the plan without them, so they were silently ignored (seen on the 2024 set). These
tests run the real Stack step on simulated RAW frames whose sensor adds vignetting, a dust
shadow, a dark-current glow and a bias pattern, and compare with the same scene without them.
"""

import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest

from conftest import make_frames
from tse_core import metadata, rawio
from tse_core.fits import read_fits
from tse_core.inventory import Inventory
from tse_core.project import EventProject
from tse_core.rawio import RawMosaic
from tse_core.synthetic import SyntheticScene

BLACK, WHITE = 400.0, 15520.0
SIZE = 320
DUST = (215, 120, 5.0)  # x, y, sigma (px): in the corona, clear of the Moon


def _field(shift=0.0):
    """What the optics and sensor do to a flat light: vignetting (its centre ``shift`` px right of
    the frame's) times a dust shadow."""
    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)
    r2 = ((xx - SIZE / 2 - shift) ** 2 + (yy - SIZE / 2) ** 2) / (SIZE / 2) ** 2 / 2
    vign = 1 - 0.45 * r2
    dust = 1 - 0.2 * np.exp(-((xx - DUST[0]) ** 2 + (yy - DUST[1]) ** 2) / (2 * DUST[2] ** 2))
    return (vign * dust).astype(np.float32)


def _glow():
    """Dark current: a warm corner (amp glow), 60 ADU at most, smooth so no hot-pixel fix hides it."""
    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)
    return (60.0 * np.exp(-(xx + yy) / 120.0)).astype(np.float32)


def _bias_pattern():
    """Fixed column pattern of the read-out, +/-12 ADU."""
    xx = np.arange(SIZE, dtype=np.float32)
    return np.broadcast_to(12.0 * np.sin(xx / 7.0), (SIZE, SIZE)).astype(np.float32)


def _mosaic(raw):
    return RawMosaic(np.minimum(raw, WHITE).astype(np.float32), "RGGB", {"R": BLACK, "G": BLACK, "B": BLACK},
                     WHITE, (2.0, 1.0, 1.5))


def _project(tmp_path, monkeypatch, sensor=True, flats=True, darks=True, bias=True, panel_tilt=0.0, shift=0.0):
    """A project of three light frames (and calibration frames); returns its path."""
    scene = SyntheticScene(size=SIZE, sun_radius=30.0, seed=2)
    lights = make_frames(["1/250", "1/60", "1/15"], aperture=8.0, iso=200)
    rng = np.random.default_rng(1)
    field, glow, pattern = (_field(shift), _glow(), _bias_pattern()) if sensor else (1.0, 0.0, 0.0)
    mosaics, metas, align_frames = {}, {}, []
    for f in lights:
        f.path = str(tmp_path / f.file_name)
        (tmp_path / f.file_name).write_bytes(b"")
        # Clipped at the sensor (in _mosaic), after the lens and dust, as in a camera; clipping the
        # scene before the vignetting left "saturated" pixels below the saturation mark.
        img, moon = scene.frame(f.exposure_s * 2.0, 160, 160, 0, 0, noise=False, full_scale=1e9)
        # Darks exist only for the 1/15 s frame; the others need the bias.
        dark = glow if f.exposure_text == "1/15" else 0.0
        mosaics[f.path] = _mosaic(BLACK + pattern + dark + img * field * (WHITE - BLACK))
        align_frames.append({"index": f.index, "file_name": f.file_name, "matrix": [[1, 0, 0], [0, 1, 0]],
                             "moon_ref": [moon["moon_x"], moon["moon_y"]],
                             "moon_frame": {"x": moon["moon_x"], "y": moon["moon_y"], "radius": moon["moon_r"]}})
    project = EventProject(name="ecl")
    project.set_inventory(Inventory(lights))
    project.use_lensfun = False

    def calib(kind, n, exposure, level, iso=200):
        metas_k = make_frames([exposure] * n, aperture=8.0, iso=iso, first_index=100 + 10 * len(metas))
        paths = []
        for k, m in enumerate(metas_k):
            m.path = str(tmp_path / f"{kind}_{k}.NEF")
            m.file_name = f"{kind}_{k}.NEF"
            (tmp_path / m.file_name).write_bytes(b"")
            noise = rng.normal(0, 3.0, (SIZE, SIZE))
            mosaics[m.path] = _mosaic(level(m) + noise)
            metas[m.path] = m
            paths.append(m.path)
        return paths

    if flats:
        # panel_tilt: the light source is that much brighter on the right than in the middle
        xx = np.arange(SIZE, dtype=np.float32)
        panel = 1 + panel_tilt * (xx - SIZE / 2) / (SIZE / 2)
        project.flat_files = calib("flat", 6, "1/100",
                                   lambda m: BLACK + pattern + 0.5 * field * panel[None, :] * (WHITE - BLACK))
    if darks:
        project.dark_files = calib("dark", 4, "1/15", lambda m: BLACK + pattern + glow)
    if bias:
        project.bias_files = calib("bias", 4, "1/8000", lambda m: BLACK + pattern)
    monkeypatch.setattr(rawio, "read_mosaic", lambda p: mosaics[p])
    monkeypatch.setattr(metadata, "read_frames", lambda paths, **kw: [metas[str(p)] for p in paths])
    (tmp_path / "ecl_align.json").write_text(json.dumps(
        {"reference_index": lights[1].index, "image_size": [SIZE, SIZE], "frames": align_frames,
         "sun": {"x": 160.0, "y": 160.0, "radius_px": 30.0, "arcsec_per_px": 30.0, "rotation_deg": 0.0,
                 "rejected": []}}))
    path = tmp_path / "ecl.json"
    project.save(path)
    return path


def _stack(path, tmp_path, **opts):
    from tse_core.workflow import StackOptions, stack

    msgs = []
    opts.setdefault("darks", "use")  # these tests stack with all the calibration frames unless they say otherwise
    stack(path, StackOptions(lensfun_db=str(tmp_path / "none"), hot_pixels=False, **opts), say=msgs.append)
    data, _ = read_fits(EventProject.load(path).parameters["stack"]["hdr"])
    return data.mean(axis=0), "\n".join(msgs)


@pytest.fixture
def truth(tmp_path_factory, monkeypatch):
    """The same scene stacked from a perfect sensor."""
    d = tmp_path_factory.mktemp("truth")
    return _stack(_project(d, monkeypatch, sensor=False, flats=False, darks=False, bias=False), d)[0]


def _zones():
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    r = np.hypot(xx - 160, yy - 160)
    corona = (r > 40) & (r < 120)
    dust = np.hypot(xx - DUST[0], yy - DUST[1]) < 2
    return corona & ~(np.hypot(xx - DUST[0], yy - DUST[1]) < 4 * DUST[2]), dust


def _err(a, b, where):
    return float(np.median(np.abs(a[where] / b[where] - 1)))


def test_master_flat_darks_and_bias_are_used_by_stack(tmp_path, monkeypatch, truth):
    corona, dust = _zones()
    got, log = _stack(_project(tmp_path, monkeypatch), tmp_path, vignetting="master")
    assert _err(got, truth, corona) < 0.01, log  # vignetting and dark glow gone (were up to ~30%)
    assert abs(float(np.median(got[dust] / truth[dust])) - 1) < 0.02, log  # dust shadow divided out
    assert "master flat" in log and "6 flat" in log
    assert "dark" in log and "bias" in log


def test_smooth_profile_from_the_flats_leaves_dust(tmp_path, monkeypatch, truth):
    corona, dust = _zones()
    path = _project(tmp_path, monkeypatch)
    got, log = _stack(path, tmp_path, vignetting="profile")
    # The log and a small file say what was divided out: the simulated corners are at 1 - 0.45 = 55%.
    assert "corners are 5" in log or "corners are 6" in log, log
    small, _ = read_fits(EventProject.load(path).parameters["stack"]["vignetting"])
    assert small.shape == (SIZE // 8, SIZE // 8) and abs(float(small[0, 0]) - 0.55) < 0.06
    assert _err(got, truth, corona) < 0.015, log  # vignetting corrected by the fitted profile
    assert float(np.median(got[dust] / truth[dust])) < 0.88, log  # the dust shadow stays (it is 20% deep)
    assert "smooth surface" in log  # since Nov 2026 a smooth surface (it can sit off-centre)


def test_compare_uses_the_master_and_reports_the_dust(tmp_path, monkeypatch, truth):
    corona, dust = _zones()
    path = _project(tmp_path, monkeypatch)
    got, log = _stack(path, tmp_path, vignetting="compare")
    assert _err(got, truth, corona) < 0.01, log
    assert abs(float(np.median(got[dust] / truth[dust])) - 1) < 0.02, log
    assert "differs from the smooth profile" in log
    ratio_path = EventProject.load(path).parameters["stack"]["flat_vs_profile"]
    ratio, _ = read_fits(ratio_path)
    ratio = ratio if ratio.ndim == 2 else ratio[0]
    assert ratio[DUST[1], DUST[0]] < 0.85  # the dust shows as a dark spot in the comparison
    assert abs(float(np.median(ratio)) - 1) < 0.01


def test_flats_that_do_not_match_are_reported(tmp_path, monkeypatch):
    path = _project(tmp_path, monkeypatch)
    project = EventProject.load(path)
    for m in metadata.read_frames(project.flat_files):
        m.aperture_f = 5.6  # flats shot at another aperture
    _, log = _stack(path, tmp_path, vignetting="master")
    assert "f/5.6" in log and "f/8" in log  # says why they were not used
    assert "NONE applied" in log  # and that nothing took their place (lensfun is off here)


def test_an_uneven_light_source_is_not_copied_into_the_stack(tmp_path, monkeypatch, truth):
    """Flats from a light panel 3% brighter on one side (simulated at quarter D850 size through the
    window, Nov 2026): the master flat put that slope into the stack, left edge +2.4%, right -2.8%,
    and compare called 12.7% of the pixels dust. A slope straight across the frame comes from the
    light, not the lens: it is taken out of the master flat; vignetting and dust stay."""
    corona, dust = _zones()
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    band = (np.abs(yy - 160) < 40) & ((xx < 30) | (xx > SIZE - 30))
    left, right = band & (xx < 30), band & (xx > SIZE - 30)
    path = _project(tmp_path, monkeypatch, panel_tilt=0.03)
    # Since Nov 2026 the slope is taken out only when the user says the light was uneven: from the
    # flats alone it cannot be told from a lens whose darkening is off-centre (see the next test).
    got, log = _stack(path, tmp_path, vignetting="compare", flats_uneven_light=True)
    ratio = got / truth
    assert abs(float(np.median(ratio[left])) - float(np.median(ratio[right]))) < 0.006, log
    assert _err(got, truth, corona) < 0.01, log
    assert abs(float(np.median(got[dust] / truth[dust])) - 1) < 0.02, log  # the dust is still corrected
    assert "brighter on the right" in log, log  # the slope is reported, not called dust
    share = float(log.split("% of pixels")[0].split(";")[-1].strip())
    assert share < 1.0, log


def _counting(monkeypatch):
    """Count the calibration frames read (bias, darks, flats) through rawio.read_mosaic."""
    reads = []
    orig = rawio.read_mosaic
    monkeypatch.setattr(rawio, "read_mosaic", lambda p: (reads.append(Path(p).name), orig(p))[1])
    return reads


def _calib_reads(reads, kind=""):
    return [r for r in reads if r.startswith(("bias_", "dark_", "flat_")) and r.startswith(kind)]


def test_masters_are_built_once_and_reused_until_the_folder_changes(tmp_path, monkeypatch):
    """The user, Nov 2026: 300 darks were read again on every Stack and earthshine run (about 14 GB of
    47 MB files). Each master is saved (FITS, in masters/ beside the project) with a fingerprint of
    its files, and read back while the files stay the same."""
    path = _project(tmp_path, monkeypatch)
    reads = _counting(monkeypatch)
    first, log1 = _stack(path, tmp_path, vignetting="master")
    assert _calib_reads(reads, "dark_") and _calib_reads(reads, "flat_") and _calib_reads(reads, "bias_")
    saved = sorted(p.name for p in (tmp_path / "fits" / "masters").glob("*.fit"))
    assert any("dark" in n for n in saved) and any("flat" in n for n in saved) and any("bias" in n for n in saved)
    assert "saved" in log1

    reads.clear()
    second, log2 = _stack(path, tmp_path, vignetting="master")
    assert _calib_reads(reads) == [], reads                      # nothing read again
    assert np.array_equal(first, second)                        # the very same stack
    assert "reused" in log2

    # A dark frame replaced: the darks are built again, the flats and bias are not.
    (tmp_path / "dark_0.NEF").write_bytes(b"changed")
    reads.clear()
    _stack(path, tmp_path, vignetting="master")
    assert _calib_reads(reads, "dark_") and not _calib_reads(reads, "flat_") and not _calib_reads(reads, "bias_")
    assert len([p for p in (tmp_path / "fits" / "masters").glob("*dark*.fit")]) == 1   # the old one removed

    # A bias frame replaced: the bias, and the flats it is taken off, are built again.
    (tmp_path / "bias_1.NEF").write_bytes(b"changed")
    reads.clear()
    _stack(path, tmp_path, vignetting="master")
    assert _calib_reads(reads, "bias_") and _calib_reads(reads, "flat_") and not _calib_reads(reads, "dark_")


def test_data_files_go_in_fits_and_json_folders(tmp_path, monkeypatch):
    """The user, Nov 2026: every .fit file the steps write goes in fits/ inside the project folder and every
    .json file (but the project file) in json/; pictures stay beside the project. The alignment file
    here sits at the top, as in an existing project, and is still found."""
    path = _project(tmp_path, monkeypatch)
    _stack(path, tmp_path, vignetting="compare")
    st = EventProject.load(path).parameters["stack"]
    assert Path(st["hdr"]).parent == tmp_path / "fits" and Path(st["weight"]).parent == tmp_path / "fits"
    assert Path(st["vignetting"]).parent == tmp_path / "fits"
    assert Path(st["flat_vs_profile"]).parent == tmp_path / "fits"
    assert Path(st["info"]).parent == tmp_path / "json"
    assert Path(st["preview"]).parent == tmp_path
    assert sorted(p.name for p in tmp_path.glob("*.fit")) == []
    assert sorted(p.name for p in tmp_path.glob("*.json")) == ["ecl.json", "ecl_align.json"]


def _rings():
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    r = np.hypot(xx - 160, yy - 160)
    return (r >= 120) & (r < 160), r >= 160  # the outer corona and the corners: the sky


@pytest.mark.parametrize("mode", ["master", "profile"])
def test_an_off_centre_lens_is_corrected_out_to_the_corners(tmp_path, monkeypatch, truth, mode):
    """A lens whose darkening is centred 25 px (8 % of the half-width) right of the frame's centre,
    as real lenses often are. Before Nov 2026 the master flat took the asymmetry for an uneven light
    and removed it, and the smooth profile (round, centred) could not follow it: the corners came
    out 5 % too dark on average, 15 % in places (the user: "the stack does not look right if I keep the
    sky"). Now the sky is right to 1 % (median) and 3 % (5-95 %); the slope is reported, not taken
    out."""
    outer, corners = _rings()
    got, log = _stack(_project(tmp_path, monkeypatch, shift=25.0), tmp_path, vignetting=mode)
    for zone in (outer, corners):
        q = got[zone] / truth[zone]
        assert abs(float(np.median(q)) - 1) < 0.01, log
        assert float(np.percentile(q, 5)) > 0.97 and float(np.percentile(q, 95)) < 1.03, log
    assert "brighter on the right" in log and "kept" in log, log


def test_stack_can_leave_the_flats_out(tmp_path, monkeypatch):
    """The user, Nov 2026: there was no way to stack without the flats chosen on the Frames tab."""
    path = _project(tmp_path, monkeypatch)
    got, log = _stack(path, tmp_path, vignetting="none")
    assert "master flat" not in log and "smooth profile" not in log, log
    assert "not used: 6 flat(s)" in log, log
    assert "master dark" in log, log  # darks and bias are still used


def test_calibration_defaults_and_the_hot_pixel_rule(tmp_path, monkeypatch):
    """The user, Nov 2026: calibration is chosen on the Stack tab. Vignetting Automatic = the master flat
    when flats are chosen; darks are not used unless asked (then the separate hot-pixel fix is off);
    bias frames are used when there are any."""
    from tse_core.workflow import StackOptions, stack

    path = _project(tmp_path, monkeypatch)
    msgs = []
    stack(path, StackOptions(lensfun_db=str(tmp_path / "none")), say=msgs.append)
    log = "\n".join(msgs)
    assert "vignetting by master flat (automatic)" in log and "darks not used" in log, log
    assert "bias frames used" in log and "master dark" not in log, log
    st = EventProject.load(path).parameters["stack"]["calibration"]
    assert st["vignetting"] == "auto" and st["darks"] == "none" and st["bias"] == "use"
    msgs.clear()
    stack(path, StackOptions(lensfun_db=str(tmp_path / "none"), darks="use", hot_pixels=True), say=msgs.append)
    log = "\n".join(msgs)
    assert "darks used" in log and "separate hot-pixel fix is off" in log, log
    msgs.clear()
    stack(path, StackOptions(lensfun_db=str(tmp_path / "none"), vignetting="lensfun", bias="none"), say=msgs.append)
    log = "\n".join(msgs)
    assert "vignetting by lensfun profile" in log and "not used: 6 flat(s)" in log, log
    assert "black level of each file" in log and "master bias" not in log, log


def test_none_ignores_saved_profiles_and_lensfun_uses_them(tmp_path, monkeypatch):
    """The user, Nov 2026: with 'None' Stack still used a profile fitted from his flats earlier (the
    saved library is matched by lens and settings). Now only the lensfun choice uses it."""
    from tse_core import workflow
    from tse_core.vignetting_profile import VignettingProfile

    path = _project(tmp_path, monkeypatch, flats=False, darks=False, bias=False)
    project = EventProject.load(path)
    lights = project.inventory.lights()
    f = lights[0]
    saved = VignettingProfile(lens_model=f.lens_model or "", focal_length_mm=float(f.focal_length_mm or 0),
                              aperture_f=float(f.aperture_f or 0), terms=(-0.3, 0.0, 0.0), camera_model=f.camera_model or "")
    monkeypatch.setattr(workflow, "load_profile_library", lambda d=None: [saved])
    plan, _ = workflow._plan(project, lights, str(tmp_path / "none"), None, vignetting="lensfun")
    assert any(fc.vignetting == "profile" for fc in plan.frames)
    plan, _ = workflow._plan(project, lights, str(tmp_path / "none"), None, vignetting="none")
    assert all(fc.vignetting == "none" for fc in plan.frames)


def test_align_loads_the_frames_without_a_lens_correction(tmp_path, monkeypatch):
    """The user, Nov 2026: Align used the lens profile (and the last Stack's lens choice). It does not need
    one: the darkening is the same in every frame and smooth, and phase correlation compares the
    corona's fine structure. On 12 of his 2024 frames a typical correction (corners at 59 %) moved the
    alignments by 0.4-1.2 px and put the Moons further off their straight track: rms 1.34 px with it,
    0.76 px without."""
    from tse_core import workflow

    path = _project(tmp_path, monkeypatch)
    project = EventProject.load(path)
    project.parameters["stack"] = {"calibration": {"vignetting": "lensfun", "lens": "", "uneven_light": False,
                                                   "darks": "use", "bias": "use"}}
    project.save(path)
    seen = {}

    class Stop(Exception):
        pass

    def loader(plan, crop):
        seen["plan"] = plan
        return lambda *a: None

    def run(*a, **k):
        raise Stop

    from tse_core import pipeline

    monkeypatch.setattr(pipeline, "default_loader", loader)
    monkeypatch.setattr(pipeline, "run_alignment", run)
    with pytest.raises(Stop):
        workflow.align(path, workflow.AlignOptions(lensfun_db=str(tmp_path / "none")))
    assert seen["plan"] is None
