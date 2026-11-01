"""Zoom and pan for the preview views (the arithmetic; the GUI only draws it)."""

import pytest

from tse_core.viewzoom import MAX_ZOOM, ViewZoom


def make(iw=1400, ih=930, vw=700, vh=600):
    z = ViewZoom()
    z.set_view(vw, vh)
    z.set_image(iw, ih)
    return z


def test_fit_shows_the_whole_image_centred():
    z = make()
    assert z.label() == "Fit" and z.scale() == pytest.approx(0.5)
    target, source = z.rects()
    assert source == pytest.approx((0, 0, 1400, 930))
    assert target[2] == pytest.approx(700) and target[1] == pytest.approx((600 - 465) / 2)


def test_zoom_keeps_the_point_under_the_mouse():
    z = make()
    before = z.to_image(100, 200)
    z.zoom_at(2.5, 100, 200)
    assert z.to_image(100, 200) == pytest.approx(before)
    assert z.label() == "125%"


def test_zoom_out_past_fit_returns_to_fit_and_limits():
    z = make()
    z.zoom_at(4)
    z.zoom_at(0.1)
    assert z.zoom is None
    for _ in range(20):
        z.zoom_at(2)
    assert z.zoom == MAX_ZOOM


def test_pan_stays_on_the_image():
    z = make()
    z.set_zoom(2.0)
    z.pan(10_000, 10_000)  # dragged far to the right and down: the top-left corner shows
    target, source = z.rects()
    assert source[0] == pytest.approx(0) and source[1] == pytest.approx(0)
    assert target[:2] == pytest.approx((0, 0))
    z.pan(-1e6, -1e6)
    _, source = z.rects()
    assert source[0] + source[2] == pytest.approx(1400) and source[1] + source[3] == pytest.approx(930)


def test_refreshed_preview_keeps_the_zoom_new_size_resets():
    z = make()
    z.set_zoom(3.0, 350, 300)
    cx = z.cx
    z.set_image(1400, 930)  # the preview was run again with other settings
    assert z.zoom == 3.0 and z.cx == cx
    z.set_image(800, 800)
    assert z.zoom is None


def test_double_click_toggles_fit_and_100():
    z = make()
    z.toggle(350, 300)
    assert z.zoom == pytest.approx(1.0)
    z.toggle(10, 10)
    assert z.zoom is None
    small = make(200, 150)  # smaller than the view: twice the fit instead
    small.toggle(350, 300)
    assert small.zoom == pytest.approx(2 * small.fit_scale())
