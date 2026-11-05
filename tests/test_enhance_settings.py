"""Enhance settings saved to a file and loaded again (the Enhance tab's Save settings... and Load
settings..., and `totality enhance --settings`), also from the _enhance.json written with layers."""

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from tse_core import enhance as E
from tse_core import workflow as W

DATA = Path(__file__).parent / "data" / "eclipse2024_bin8.npz"

CUSTOM = E.EnhanceParams(
    layers=("natural", "fnrgf_n", "achf"), segments=40, order=8, noise_m=0.6, flatten=0.55, shoulder=0.2,
    even=0.3, midtone=0.45, mgn_k=0.82, noise_gate=2.5, saturation=1.4, white="camera", neutral_background=False,
    achf_scales=(0.002, 0.004, 0.01, 0.03, 0.1, 0.2), achf_weights=(1.0, 0.5, 1.0, 1.0, 0.25, 0.0),
    achf_part_limit=3.0, achf_contrast=1.35, blend={"natural": 1.0, "fnrgf_n": 0.6, "achf": 2.5},
    blend_modes={"fnrgf_n": "soft light", "achf": "linear"})


def test_saved_settings_load_back_the_same(tmp_path):
    path = W.save_enhance_settings(CUSTOM, tmp_path / "my look")  # .json added
    assert path.name == "my look.json"
    doc = json.loads(path.read_text())
    assert doc["kind"] == W.SETTINGS_KIND and "moon_margin_px" not in doc["settings"]
    got, notes = W.load_enhance_settings(path)
    assert got == CUSTOM and notes == []


def test_the_info_file_written_with_layers_loads_too(tmp_path):
    z = np.load(DATA)
    hdr = np.ascontiguousarray(np.moveaxis(z["hdr"].astype(np.float32) * float(z["scale"]), 0, -1))
    geo = {k: float(z[k]) for k in ("SUNX", "SUNY", "SUNR", "MOONX", "MOONY", "MOONR")}
    p = replace(CUSTOM, layers=("natural",), blend={"natural": 1.0})
    result, _ = E.enhance(hdr, (np.abs(hdr).sum(axis=2) > 0).astype(np.float32), geo, p, out_dir=tmp_path,
                          name="ecl", fmt="png")
    got, notes = W.load_enhance_settings(result.layers["info"])
    assert got == p and notes == []


def test_older_files_keep_what_they_lack_and_say_what_is_gone(tmp_path):
    """An _enhance.json from before the ACHF settings and with removed ones (Oct 2026)."""
    old = {"params": {"flatten": 0.7, "strands": 0.4, "mgn_skip_fine": 1, "blend": {"natural": 1, "adaptive": 0.3}}}
    f = tmp_path / "old_enhance.json"
    f.write_text(json.dumps(old))
    base = replace(E.EnhanceParams(), saturation=1.7)
    got, notes = W.load_enhance_settings(f, base=base)
    assert got.flatten == 0.7 and got.blend == {"natural": 1.0, "adaptive": 0.3}
    assert got.saturation == 1.7 and got.achf_contrast == 1.0  # not in the file: as they were
    assert "Smooth along scales" in notes[0] and "Skip finest scales" in notes[0]
    assert "achf_contrast" in notes[1]


@pytest.mark.parametrize("bad", [{"layers": ["natural", "strands"]}, {"white": "daylight"},
                                 {"achf_weights": [1, 1]}, {"blend_modes": {"nrgf": "multiply"}},
                                 {"flatten": "lots"}])
def test_invalid_settings_are_refused(tmp_path, bad):
    f = tmp_path / "s.json"
    f.write_text(json.dumps({"kind": W.SETTINGS_KIND, "version": 1, "settings": bad}))
    with pytest.raises(W.WorkflowError):
        W.load_enhance_settings(f)


def test_not_a_settings_file(tmp_path):
    f = tmp_path / "project.json"
    f.write_text(json.dumps({"name": "ecl", "frames": []}))
    with pytest.raises(W.WorkflowError, match="not an Enhance settings file"):
        W.load_enhance_settings(f)


def _enhance_parser():
    sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))
    from cli_reference import parser

    sp = parser()._subparsers._group_actions[0].choices["enhance"]
    return sp


def test_command_line_settings_file_with_options_typed_over_it(tmp_path):
    from tse_core import cli

    f = W.save_enhance_settings(CUSTOM, tmp_path / "look.json")
    sp = _enhance_parser()
    argv = ["proj.json", "--settings", str(f), "--saturation", "0.8", "--blend=natural=1,adaptive=0.4:overlay",
            "--flat", "0.25"]  # --flat: abbreviated, as argparse allows
    args = sp.parse_args(argv)
    params, notes = cli.enhance_params(args, cli.given_options(sp, argv))
    assert params.saturation == 0.8 and params.flatten == 0.25
    assert params.blend == {"natural": 1.0, "adaptive": 0.4} and params.blend_modes == {"adaptive": "overlay"}
    assert params.achf_contrast == 1.35 and params.white == "camera" and params.layers == CUSTOM.layers  # from the file
    assert "saturation" in notes[-1]
    # Without --settings: the options alone, as before.
    plain, _ = cli.enhance_params(sp.parse_args(["proj.json"]), set())
    assert plain == replace(E.EnhanceParams(), layers=E.DEFAULT_LAYERS)


def test_every_enhance_option_is_a_setting_or_says_it_is_not():
    """A new option must be added to ENHANCE_OPTION_SETTINGS (or to the list of non-settings), or
    --settings would let the file silently win over it."""
    from tse_core import cli

    dests = {a.dest for a in _enhance_parser()._actions}
    assert dests == set(cli.ENHANCE_OPTION_SETTINGS) | cli.ENHANCE_NOT_SETTINGS, \
        dests ^ (set(cli.ENHANCE_OPTION_SETTINGS) | cli.ENHANCE_NOT_SETTINGS)
