import math

from conftest import DATA

import numpy as np
import pytest

from tse_core.lensfun import (
    Lens, VignettingCalibration, corner_falloff, parse_aperture_range, parse_focal_range,
    relative_illumination, vignetting_map,
)


def test_database_subset_loads(lens_db):
    assert len(lens_db.lenses) == 3
    cam = lens_db.find_camera("NIKON CORPORATION", "NIKON D850")
    assert cam and cam.crop_factor == 1.0


@pytest.mark.parametrize("name,spec", [
    ("AF-S Nikkor 80-400mm f/4.5-5.6G ED VR", None),   # exiftool LensID
    ("80-400mm f/4.5-5.6 G VR", [80, 400, 4.5, 5.6]),   # exifread, built from Nikon maker notes
])
def test_matches_the_g_lens(lens_db, name, spec):
    best = lens_db.match_lens(name, spec, 400, "NIKON CORPORATION")[0]
    assert best.lens.model == "Nikon AF-S Nikkor 80-400mm f/4.5-5.6G ED VR"


def test_d_lens_is_not_confused_with_g(lens_db):
    best = lens_db.match_lens("80-400mm f/4.5-5.6 D VR", [80, 400, 4.5, 5.6], 400)[0]
    assert "5.6D" in best.lens.model


def test_exact_calibration_at_400mm_f8(lens_db):
    lens = lens_db.find_lens_by_name("Nikon AF-S Nikkor 80-400mm f/4.5-5.6G ED VR")
    terms = lens.vignetting_terms(400, 8.0)
    assert terms == pytest.approx((0.0723851, -0.5104349, 0.2628477))
    brightness, stops = corner_falloff(terms)
    assert brightness == pytest.approx(0.8248, abs=1e-4)
    assert stops == pytest.approx(-0.278, abs=1e-3)


def test_interpolation_lies_between_neighbours(lens_db):
    lens = lens_db.find_lens_by_name("Nikon AF-S Nikkor 80-400mm f/4.5-5.6G ED VR")
    b56 = corner_falloff(lens.vignetting_terms(400, 5.6))[0]
    b8 = corner_falloff(lens.vignetting_terms(400, 8.0))[0]
    b7 = corner_falloff(lens.vignetting_terms(400, 7.1))[0]
    assert min(b56, b8) - 0.02 <= b7 <= max(b56, b8) + 0.02


def test_far_from_any_calibration_returns_none():
    lens = Lens("X", "X 50mm f/1.8", ["X 50mm f/1.8"], [], 1.0,
                [VignettingCalibration(50, 1.8, 1000, -0.5, 0, 0)], "pa")
    assert lens.vignetting_terms(50, 1.8) == (-0.5, 0, 0)
    assert lens.vignetting_terms(50, 0.95) is None  # 4/0.95 - 4/1.8 > 1


def test_map_matches_corner_falloff():
    terms = (0.0723851, -0.5104349, 0.2628477)
    m = vignetting_map(301, 201, terms)
    assert m[100, 150] == pytest.approx(1.0, abs=1e-4)
    # The pixel-centre corner sits just inside r = 1.
    r_corner = math.hypot(150, 100) / (math.hypot(301, 201) / 2)
    assert m[0, 0] == pytest.approx(float(relative_illumination(r_corner, terms)), rel=1e-5)
    assert m[0, 0] == pytest.approx(m[-1, -1]) == pytest.approx(m[0, -1])


def test_crop_sensor_sees_less_falloff():
    terms = (-0.3678867, 0.0054641, -0.0222676)
    full = corner_falloff(terms, image_crop=1.0)[0]
    aps_c = corner_falloff(terms, image_crop=1.5)[0]
    assert aps_c > full
    m = vignetting_map(300, 200, terms, image_crop=1.5)
    assert float(m.min()) == pytest.approx(aps_c, abs=0.01)


def test_name_parsing():
    assert parse_focal_range("AF-S Nikkor 80-400mm f/4.5-5.6G ED VR") == (80, 400)
    assert parse_aperture_range("AF-S Nikkor 80-400mm f/4.5-5.6G ED VR") == (4.5, 5.6)
    assert parse_focal_range("EF 50mm f/1.8 STM") == (50, 50)
    assert parse_aperture_range("Sony FE 24-70mm F2.8 GM") == (2.8, 2.8)


def test_download_database_saves_only_xml(tmp_path, monkeypatch):
    import io
    import json as _json

    from tse_core import lensfun

    xml = (DATA / "lensfun_subset.xml").read_bytes()
    listing = [
        {"name": "slr-nikon.xml", "download_url": "https://example/slr-nikon.xml"},
        {"name": "lensfun-database.dtd", "download_url": "https://example/x.dtd"},
    ]

    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=0, **kw):
        url = getattr(req, "full_url", req)
        return Resp(_json.dumps(listing).encode() if url == lensfun.GITHUB_DB_API else xml)

    monkeypatch.setattr(lensfun.urllib.request, "urlopen", fake_urlopen)
    path = lensfun.download_database(tmp_path / "db")
    assert [p.name for p in path.iterdir()] == ["slr-nikon.xml"]
    assert len(lensfun.LensfunDatabase.load([path]).lenses) == 3


def test_import_database_from_zip_and_folder(tmp_path):
    import zipfile

    from tse_core import lensfun

    xml = (DATA / "lensfun_subset.xml").read_bytes()
    zpath = tmp_path / "lensfun-master.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("lensfun-master/data/db/slr-nikon.xml", xml)
        zf.writestr("lensfun-master/data/db/lensfun-database.dtd", "x")
        zf.writestr("lensfun-master/docs/other.xml", "<notlensfun/>")
    out = lensfun.import_database(zpath, tmp_path / "a")
    assert [p.name for p in out.iterdir()] == ["slr-nikon.xml"]
    out2 = lensfun.import_database(out, tmp_path / "b")
    assert len(lensfun.LensfunDatabase.load([out2]).lenses) == 3


def test_lensfun_update_explains_certificate_errors(monkeypatch, capsys):
    import ssl
    import urllib.error

    from tse_core import cli

    def boom(dest=None):
        raise urllib.error.URLError(ssl.SSLCertVerificationError("certificate verify failed: unable to get local issuer certificate"))

    monkeypatch.setattr(cli, "download_database", boom)
    assert cli.main(["lensfun-update"]) == 1
    out = capsys.readouterr().out
    assert "certifi" in out and "--from" in out
