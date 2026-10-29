import struct

import numpy as np
import pytest

from conftest import ECLIPSE_2024, make_frames, sample_images_dir
from tse_core.fits import read_fits, write_fits
from tse_core.inventory import Inventory
from tse_core.metadata import describe_lens_spec, extract_cr3_tiff_blocks, read_frames
from tse_core.project import EventProject


def test_describe_lens_spec():
    assert describe_lens_spec([80, 400, 4.5, 5.6]) == "80-400mm f/4.5-5.6"
    assert describe_lens_spec([50, 50, 1.8, 1.8]) == "50mm f/1.8"
    assert describe_lens_spec([15, 45, 0, 0]) == "15-45mm"


def _box(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", 8 + len(payload)) + kind + payload


def test_cr3_box_parser(tmp_path):
    canon_uuid = bytes.fromhex("85c0b687820f11e08111f4ce462b6a48")
    cmt1, cmt2 = b"II*\x00" + b"\x01" * 12, b"II*\x00" + b"\x02" * 12
    uuid_box = _box(b"uuid", canon_uuid + _box(b"CNCV", b"x") + _box(b"CMT1", cmt1) + _box(b"CMT2", cmt2))
    data = _box(b"ftyp", b"crx \x00\x00\x00\x01") + _box(b"moov", uuid_box) + _box(b"mdat", b"\x00" * 64)
    path = tmp_path / "x.CR3"
    path.write_bytes(data)
    assert extract_cr3_tiff_blocks(str(path)) == [cmt1, cmt2]


def test_fits_round_trip(tmp_path):
    img = np.random.default_rng(1).random((3, 20, 30)).astype(np.float32)
    write_fits(tmp_path / "a.fit", img, {"EXPTIME": 0.5, "OBJECT": "Corona"})
    back, header = read_fits(tmp_path / "a.fit")
    assert np.array_equal(back, img)
    assert header["EXPTIME"] == 0.5 and header["OBJECT"] == "Corona"


def test_project_round_trip_keeps_overrides(tmp_path):
    inv = Inventory(make_frames(ECLIPSE_2024[:4]))
    project = EventProject(name="2024 Hot Springs")
    project.set_inventory(inv)
    frames = project.inventory.frames
    frames[0].apply_override("iso", 64.0)
    frames[1].role = "excluded"
    project.store_inventory(project.inventory.__class__(frames))
    project.save(tmp_path / "p.json")

    loaded = EventProject.load(tmp_path / "p.json")
    rescanned = Inventory(make_frames(ECLIPSE_2024[:4]))  # a fresh read of the same files
    loaded.set_inventory(rescanned)
    again = loaded.inventory.frames
    assert again[0].iso == 64.0 and again[0].overrides == ["iso"]
    assert again[1].role == "excluded"
    assert again[2].iso == 100


def test_contact_times_mark_partial_frames():
    inv = Inventory(make_frames(ECLIPSE_2024))
    from datetime import datetime

    changed = inv.mark_outside(datetime(2024, 4, 8, 13, 47, 40), datetime(2024, 4, 8, 13, 51, 0))
    partial = [f.index for f in inv.frames if f.role == "partial"]
    assert changed == len(partial) and partial and partial[0] == 0


@pytest.mark.skipif(sample_images_dir() is None, reason="set TSE_SAMPLE_RAWS to a folder of sample RAW files")
def test_real_samples_read_without_exiftool():
    folder = sample_images_dir()
    files = [str(p) for p in folder.iterdir() if p.suffix.lower() in (".nef", ".cr2", ".cr3")]
    frames = read_frames(files, use_exiftool=False, use_rawpy=False)
    for f in frames:
        assert f.exposure_s and f.iso and f.aperture_f and f.focal_length_mm and f.timestamp, f.file_name
