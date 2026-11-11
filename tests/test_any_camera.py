"""RAW files from any camera with a Bayer sensor (Nov 2026).

Only Nikon, Canon, Sony and DNG files were picked up from a folder, although LibRaw reads nearly
every RAW format. Sensors without a 2x2 Bayer pattern (Fujifilm X-Trans, Sigma Foveon) are refused
with a clear message instead of being misread.
"""

from pathlib import Path

import numpy as np
import pytest

from tse_core import frames as F
from tse_core import rawio
from tse_core.frames import FrameMeta
from tse_core.inventory import find_raw_files
from tse_core.metadata import _finalise


@pytest.mark.parametrize("ext", [".NEF", ".cr2", ".CR3", ".arw", ".dng", ".RAF", ".orf", ".rw2", ".pef", ".rwl",
                                 ".srw", ".3fr", ".iiq", ".erf", ".mrw", ".crw", ".gpr"])
def test_raw_files_from_any_camera_are_found(tmp_path, ext):
    (tmp_path / f"frame{ext}").write_bytes(b"x")
    (tmp_path / "notes.txt").write_text("x")
    assert [Path(p).name for p in find_raw_files(tmp_path)] == [f"frame{ext}"]


def test_foveon_is_not_listed():
    assert ".x3f" not in F.RAW_EXTENSIONS  # no colour filter pattern at all


def _tile(pattern_idx, shape=(12, 12)):
    p = np.asarray(pattern_idx)
    return np.tile(p, (shape[0] // p.shape[0], shape[1] // p.shape[1]))


@pytest.mark.parametrize("idx, desc, want", [([[0, 1], [3, 2]], "RGBG", "RGGB"), ([[1, 2], [0, 1]], "RGBG", "GBRG"),
                                             ([[2, 1], [1, 0]], "RGB", "BGGR")])
def test_bayer_patterns_are_read(idx, desc, want):
    assert rawio.bayer_pattern(_tile(idx), desc) == want


def test_xtrans_is_refused_clearly():
    xtrans = [[1, 1, 0, 1, 1, 2], [1, 1, 2, 1, 1, 0], [2, 0, 1, 0, 2, 1],
              [1, 1, 2, 1, 1, 0], [1, 1, 0, 1, 1, 2], [0, 2, 1, 2, 0, 1]]
    with pytest.raises(ValueError, match="2x2 Bayer"):
        rawio.bayer_pattern(_tile(xtrans), "RGBG")


def test_inventory_warns_about_a_sensor_that_is_not_bayer():
    f = FrameMeta(path="x.raf", file_name="x.raf")
    f.cfa_pattern = "GGRGGB" * 6  # X-Trans: 36 entries
    _finalise(f)
    assert any("Bayer" in w for w in f.warnings)
    g = FrameMeta(path="y.nef", file_name="y.nef")
    g.cfa_pattern = "RGGB"
    _finalise(g)
    assert not any("Bayer" in w for w in g.warnings)
