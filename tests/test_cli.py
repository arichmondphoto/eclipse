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
    assert "36 frames, 1/1600 to 4.0 s, 12.6 stops, 1/3-stop steps" in out
    assert "missing 1/3, 1/2.5" in out and "missing 2)" in out
    assert "corners at 82% of centre" in out
    saved = json.loads(project.read_text())
    assert len(saved["frames"]) == 36 and saved["frames"][0]["bracket_id"] == 0
    assert (tmp_path / "frames.csv").read_text().count("\n") == 37


def test_lens_command(capsys):
    assert cli.main(["lens", "AF-S Nikkor 80-400mm f/4.5-5.6G ED VR", "--focal", "400", "--aperture", "8",
                     "--lensfun-db", str(DATA)]) == 0
    out = capsys.readouterr().out.splitlines()[0]
    assert "Nikon AF-S Nikkor 80-400mm f/4.5-5.6G ED VR" in out and "82%" in out
