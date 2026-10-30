import json

from conftest import ECLIPSE_2024, DATA, make_frames
from tse_core import cli, inventory


def test_inventory_command_end_to_end(tmp_path, monkeypatch, capsys):
    raw_dir = tmp_path / "raws"
    raw_dir.mkdir()
    for i in range(len(ECLIPSE_2024)):
        (raw_dir / f"DSC_{i:04d}.NEF").write_bytes(b"")
    frames = make_frames(ECLIPSE_2024)
    monkeypatch.setattr(inventory, "read_frames", lambda paths, **kw: frames)

    project = tmp_path / "event.json"
    code = cli.main(["inventory", str(raw_dir), "--lensfun-db", str(DATA), "--project", str(project),
                     "--csv", str(tmp_path / "frames.csv")])
    out = capsys.readouterr().out
    assert code == 0
    assert "37 frames, 1/1600 to 4.0 s, 12.6 stops, 1/3-stop steps" in out
    assert "missing 1/2.5)" in out and "missing 2)" in out
    assert "corners at 82% of centre" in out
    saved = json.loads(project.read_text())
    assert len(saved["frames"]) == 37 and saved["frames"][0]["bracket_id"] == 0
    assert (tmp_path / "frames.csv").read_text().count("\n") == 38


def test_lens_command(capsys):
    assert cli.main(["lens", "AF-S Nikkor 80-400mm f/4.5-5.6G ED VR", "--focal", "400", "--aperture", "8",
                     "--lensfun-db", str(DATA)]) == 0
    out = capsys.readouterr().out.splitlines()[0]
    assert "Nikon AF-S Nikkor 80-400mm f/4.5-5.6G ED VR" in out and "82%" in out


def test_frames_exclude_and_include(tmp_path, capsys):
    from tse_core.inventory import Inventory
    from tse_core.project import EventProject

    project = EventProject(name="t")
    project.set_inventory(Inventory(make_frames(ECLIPSE_2024)))
    path = tmp_path / "p.json"
    project.save(path)
    assert cli.main(["frames", str(path), "--exclude", "DSC_0021.NEF", "dsc_0034.nef"]) == 0
    loaded = EventProject.load(path)
    roles = {f.file_name: f.role for f in loaded.inventory.frames}
    assert roles["DSC_0021.NEF"] == roles["DSC_0034.NEF"] == "excluded"
    assert len(loaded.inventory.lights()) == 35
    # A re-scan of the folder keeps the exclusions.
    loaded.set_inventory(Inventory(make_frames(ECLIPSE_2024)))
    assert len(loaded.inventory.lights()) == 35
    assert cli.main(["frames", str(path), "--include", "DSC_0021.NEF"]) == 0
    assert len(EventProject.load(path).inventory.lights()) == 36
    assert cli.main(["frames", str(path), "--exclude", "nope.nef"]) == 1
    assert "not in this project" in capsys.readouterr().out
