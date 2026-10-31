"""The shared steps behind the command line and the GUI."""

from pathlib import Path

import numpy as np
import pytest

from tse_core import workflow as W
from tse_core.enhance import EnhanceParams

DATA = Path(__file__).parent / "data" / "eclipse2024_bin8.npz"


def test_reporter_cancels_on_next_message():
    seen = []
    flag = {"stop": False}
    say = W.reporter(seen.append, lambda: flag["stop"])
    say("one")
    flag["stop"] = True
    with pytest.raises(W.Cancelled):
        say("two")
    assert seen == ["one"]


def test_parse_blend_and_layers():
    assert W.parse_blend("natural=1, fnrgf_n=0.7") == {"natural": 1.0, "fnrgf_n": 0.7}
    assert W.parse_blend("none") == {}
    with pytest.raises(W.WorkflowError):
        W.parse_blend("natural")
    with pytest.raises(W.WorkflowError):
        W.parse_blend("bogus=1")
    assert W.parse_layers("natural,adaptive") == ("natural", "adaptive")
    with pytest.raises(W.WorkflowError):
        W.parse_layers("natural,nope")


def test_reduce_for_preview_scales_geometry_and_keeps_empty_bins_out():
    hdr = np.ones((100, 160, 3), np.float32)
    hdr[:, :10] = 5.0
    weight = np.ones((100, 160), np.float32)
    weight[:, :3] = 0  # no data at the left edge
    header = {"SUNX": 79.5, "SUNY": 49.5, "SUNR": 20.0, "MOONX": 80.0, "MOONY": 50.0, "MOONR": 21.0}
    data = W.HdrData(hdr, weight, header, "x")
    small = W.reduce_for_preview(data, max_side=40)
    assert small.binning == 4
    assert small.hdr.shape == (25, 40, 3)
    assert small.header["SUNX"] == pytest.approx((79.5 + 0.5) / 4 - 0.5)
    assert small.header["SUNR"] == pytest.approx(5.0)
    assert small.weight[0, 0] == 0  # partly empty bin
    assert small.weight[0, 5] == pytest.approx(16.0)
    assert small.hdr[0, 1] == pytest.approx(5.0)
    assert W.reduce_for_preview(data, max_side=500) is data


@pytest.mark.skipif(not DATA.exists(), reason="real-data fixture not present")
def test_preview_matches_full_result_on_real_data():
    """A 2x-reduced preview looks like the full-size result shrunk (same tones, no new artifacts)."""
    z = np.load(DATA)
    hdr = np.ascontiguousarray(np.moveaxis(z["hdr"].astype(np.float32) * float(z["scale"]), 0, -1))
    header = {k: float(z[k]) for k in ("SUNX", "SUNY", "SUNR", "MOONX", "MOONY", "MOONR")}
    weight = (np.abs(hdr).sum(axis=2) > 0).astype(np.float32)
    data = W.HdrData(hdr, weight, header, "fixture")
    params = EnhanceParams(layers=("natural", "fnrgf_n"))
    _, full = W.enhance_preview(data, params)
    small = W.reduce_for_preview(data, max_side=520)
    assert small.binning == 2
    _, prev = W.enhance_preview(small, params)
    for k in ("natural", "blend"):
        a = full[k].mean(axis=2)
        h, w = prev[k].shape[:2]
        a = a[: h * 2, : w * 2].reshape(h, 2, w, 2).mean(axis=(1, 3))
        b = prev[k].mean(axis=2)
        inner = (slice(10, -10), slice(10, -10))
        assert np.median(np.abs(a[inner] - b[inner])) < 0.03, k


def test_steps_report_missing_inputs(tmp_path):
    from tse_core.project import EventProject

    p = tmp_path / "e.json"
    EventProject(name="e").save(p)
    with pytest.raises(W.WorkflowError, match="two light frames"):
        W.align(p, W.AlignOptions())
    with pytest.raises(W.WorkflowError, match="Align step"):
        W.stack(p, W.StackOptions())
    with pytest.raises(W.WorkflowError, match="Stack step"):
        W.enhance_project(p, EnhanceParams())


def test_paths_stored_relative_to_another_folder(tmp_path, monkeypatch):
    """A project made by 'tse' run inside its folder stores bare file names; the GUI (Siril) runs
    from elsewhere and must still find the alignment and stack files."""
    import json

    from tse_core.fits import write_fits
    from tse_core.project import EventProject

    folder = tmp_path / "event"
    folder.mkdir()
    (folder / "e_align.json").write_text(json.dumps({"reference_index": 0, "frames": []}))
    write_fits(folder / "e_hdr.fit", np.zeros((3, 8, 8), np.float32), {"MOONX": 4.0, "MOONY": 4.0, "MOONR": 2.0})
    write_fits(folder / "e_weight.fit", np.ones((8, 8), np.float32))
    p = EventProject(name="e")
    p.parameters["align"] = {"file": "e_align.json"}
    p.parameters["stack"] = {"hdr": "e_hdr.fit", "weight": "e_weight.fit"}
    p.save(folder / "e.json")
    monkeypatch.chdir(tmp_path)  # somewhere else
    alignment, apath = W.load_alignment(folder / "e.json")
    assert apath == (folder / "e_align.json").resolve()
    data = W.load_hdr(folder / "e.json")
    assert Path(data.path) == (folder / "e_hdr.fit").resolve()
    assert not data.notes  # the weight map was found too
    assert W.resolve(folder / "e.json", "missing.fit") == ""
    # Moved folder: absolute paths that no longer exist fall back to the name next to the project.
    assert W.resolve(folder / "e.json", "/old/place/e_hdr.fit") == str((folder / "e_hdr.fit").resolve())


def test_project_folder_layout(tmp_path):
    from tse_core.inventory import ProjectFolder

    root = tmp_path / "Eclipse 2024"
    for sub in ("Lights", "darks", "Flat", "offsets", "other"):
        (root / sub).mkdir(parents=True)
    pf = ProjectFolder.scan(root)
    assert pf.is_project_folder
    assert pf.lights.name == "Lights" and pf.darks.name == "darks" and pf.flats.name == "Flat"
    assert pf.bias.name == "offsets"
    assert pf.project_file == root.resolve() / "Eclipse 2024.json"
    plain = ProjectFolder.scan(root / "Lights")
    assert not plain.is_project_folder and plain.lights == (root / "Lights").resolve()


def test_cli_inventory_uses_project_folder(tmp_path, monkeypatch, capsys):
    """A project folder: frames from lights/, calibration from its subfolders, project file in the folder."""
    from conftest import make_frames
    from tse_core import cli
    from tse_core import inventory as inv_mod
    from tse_core.project import EventProject

    root = tmp_path / "ev"
    (root / "lights").mkdir(parents=True)
    (root / "darks").mkdir()
    frames = make_frames(["1/250", "1/60"])
    for f in frames:
        (root / "lights" / f.file_name).write_bytes(b"")
    (root / "darks" / "D1.NEF").write_bytes(b"")

    def fake_from_paths(paths, use_exiftool=True):
        assert all("lights" in p for p in paths)
        return inv_mod.Inventory(make_frames(["1/250", "1/60"]))

    monkeypatch.setattr(cli.Inventory, "from_paths", staticmethod(fake_from_paths))
    monkeypatch.setattr(cli, "read_frames", lambda files: [])
    assert cli.main(["inventory", str(root), "--lensfun-db", str(tmp_path / "none")]) == 0
    out = capsys.readouterr().out
    assert "Project folder" in out
    project = EventProject.load(root / "ev.json")
    assert project.dark_files and project.dark_files[0].endswith("D1.NEF")
    assert Path(project.raw_folder) == (root / "lights").resolve()
