"""Previews redraw only the layers whose settings changed (Andy, Nov 2026: changing an ACHF setting
while viewing the blend should redraw ACHF and the blend, not every filter).

``render(..., cache=True)`` keeps each layer's last result for the area drawn, keyed by that
layer's own settings; the blend is mixed again from them. Results must equal a fresh render."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from tse_core import enhance as E

DATA = Path(__file__).parent / "data" / "eclipse2024_bin8.npz"
P = E.EnhanceParams(layers=E.LAYERS, flatten=0.6, noise_gate=3.0,
                    blend={"natural": 1.0, "fnrgf_n": 0.5, "adaptive": 0.4, "achf": 1.0})


@pytest.fixture(scope="module")
def frame():
    z = np.load(DATA)
    hdr = np.ascontiguousarray(np.moveaxis(z["hdr"].astype(np.float32) * float(z["scale"]), 0, -1))
    geo = {k: float(z[k]) for k in ("SUNX", "SUNY", "SUNR", "MOONX", "MOONY", "MOONR")}
    return lambda: E.prepare(hdr.copy(), (np.abs(hdr).sum(axis=2) > 0).astype(np.float32), geo, P)


def _counting(monkeypatch):
    calls = []
    for name in ("natural_apply", "nrgf", "fnrgf", "mgn", "mgn_gated", "achf"):
        orig = getattr(E, name)

        def wrap(*a, _o=orig, _n=name, **k):
            calls.append(_n)
            return _o(*a, **k)

        monkeypatch.setattr(E, name, wrap)
    return calls


@pytest.mark.parametrize("box", [None, (60, 260, 300, 520)])
def test_an_achf_change_redraws_only_achf(frame, monkeypatch, box):
    f = frame()
    E.render(f, P, box=box, cache=True)
    calls = _counting(monkeypatch)
    p2 = replace(P, achf_weights=(1, 0.5, 1, 1, 0, 0))
    got = E.render(f, p2, box=box, cache=True)
    assert calls == ["achf"], calls
    # Contrast only rescales the filtered image: nothing is filtered again.
    p3 = replace(p2, achf_contrast=1.7)
    got3 = E.render(f, p3, box=box, cache=True)
    assert calls == ["achf"], calls
    monkeypatch.undo()
    fresh = E.render(frame(), p2, box=box)
    for k in fresh:
        assert np.abs(got[k] - fresh[k]).max() < 1e-6, k
    fresh3 = E.render(frame(), p3, box=box)
    for k in fresh3:
        assert np.abs(got3[k] - fresh3[k]).max() < 1e-6, k
    p2 = p3
    # A blend-only change redraws no layer at all; a natural-layer change only the natural layer.
    calls = _counting(monkeypatch)
    E.render(f, replace(p2, blend={"natural": 1.0, "achf": 0.3}), box=box, cache=True)
    assert calls == []
    E.render(f, replace(p2, flatten=0.3), box=box, cache=True)
    assert calls == ["natural_apply"]


def test_cached_layers_stay_exact_when_another_layers_margin_changes(frame):
    """Turning the large ACHF scales on widens ACHF's margin; MGN, cached from before, must still be
    exactly what a fresh render gives for the box."""
    f = frame()
    box = (60, 260, 300, 520)
    E.render(f, P, box=box, cache=True)
    p2 = replace(P, achf_weights=(1, 1, 1, 1, 1, 1))
    got = E.render(f, p2, box=box, cache=True)
    fresh = E.render(frame(), p2, box=box)
    assert np.abs(got["adaptive"] - fresh["adaptive"]).max() < 1e-6
    assert np.abs(got["blend"] - fresh["blend"]).max() < 1e-6


def test_without_cache_nothing_is_kept(frame):
    f = frame()
    E.render(f, P, box=(60, 160, 300, 420))
    assert "layers" not in f.cache


def test_preview_log_gives_the_mgn_sizes_as_set():
    """The reduced preview divides the MGN sizes by its reduction; its log line must still give the
    sizes as set on the tab (full-size px), with the preview's own pixels named as such (it said
    "Adaptive filter scales (px): 0.2, ..." for 1.25 px, Nov 2026)."""
    from tse_core import workflow as W

    z = np.load(DATA)
    hdr = np.ascontiguousarray(np.moveaxis(z["hdr"].astype(np.float32) * float(z["scale"]), 0, -1))
    geo = {k: float(z[k]) for k in ("SUNX", "SUNY", "SUNR", "MOONX", "MOONY", "MOONR")}
    data = W.HdrData(hdr, (np.abs(hdr).sum(axis=2) > 0).astype(np.float32), geo, "x", binning=6)
    p = E.EnhanceParams(layers=("adaptive",), blend={})
    _, result, _ = W.enhance_whole_preview(data, p)
    text = result.describe()
    assert "1.25, 2.5, 5, 10, 20, 40 px of the full-size stack" in text, text
    assert "in this preview, reduced 6x" in text, text
