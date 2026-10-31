"""Phase 4: NRGF, FNRGF, FNRGF-N, adaptive filter, natural layer, colour, CLI."""

import math

import numpy as np
import pytest
from scipy import ndimage

from tse_core import enhance as E
from tse_core.synthetic import SyntheticScene


def _scene(size=(400, 600), sun=(300.0, 200.0), r_sun=50.0, noise=0.0, seed=1):
    h, w = size
    sc = SyntheticScene(size=max(size), sun_radius=r_sun, seed=seed)
    img = sc.corona(sun[0], sun[1], shape=size).astype(np.float32)
    rng = np.random.default_rng(seed)
    if noise:
        img = img + rng.normal(0, noise, img.shape).astype(np.float32)
    g = E.Geometry(sun[0], sun[1], r_sun, sun[0] + 2, sun[1] - 1, r_sun * 1.05)
    return img, g


def _valid(shape, g):
    r, phi = E.polar(shape, g)
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    return r, phi, np.hypot(xx - g.moon_x, yy - g.moon_y) > g.moon_r + 1.5


def test_nrgf_normalizes_each_circle():
    img, g = _scene(size=(500, 500), sun=(250.0, 250.0))
    r, phi, valid = _valid(img.shape, g)
    st = E.ring_stats(img, r, phi, valid, 50, 10, 1.0)
    y = E.nrgf(img, r, valid, st)
    for R in (70, 110, 160, 220):
        ring = valid & (np.abs(r - R) < 1.0)
        assert abs(y[ring].mean()) < 0.1
        assert y[ring].std() == pytest.approx(1.0, abs=0.15)


def test_fnrgf_evens_contrast_around_the_sun():
    """A low-contrast hemisphere is boosted by the FNRGF but not by the NRGF."""
    img, g = _scene(size=(500, 500), sun=(250.0, 250.0))
    r, phi, valid = _valid(img.shape, g)
    yy, xx = np.mgrid[0:500, 0:500]
    left = xx < 250
    # Lower the contrast of the left half: pull it towards its circle average.
    ring = r.astype(int)
    prof = np.bincount(ring.ravel(), img.ravel()) / np.maximum(np.bincount(ring.ravel()), 1)
    img = np.where(left, prof[ring] + 0.3 * (img - prof[ring]), img).astype(np.float32)
    st = E.ring_stats(img, r, phi, valid, 50, 10, 1.0)
    att = E._attenuation(None, 10)
    yf = E.fnrgf(img, r, phi, valid, st, att, att)
    yn = E.nrgf(img, r, valid, st)
    band = valid & (r > 80) & (r < 200)

    def ratio(y):
        return y[band & left].std() / y[band & ~left].std()

    assert ratio(yn) < 0.5
    assert ratio(yf) > 0.7


def test_fnrgf_no_arc_where_circles_leave_the_frame():
    img, g = _scene(size=(400, 700), sun=(350.0, 200.0))
    r, phi, valid = _valid(img.shape, g)
    st = E.ring_stats(img, r, phi, valid, 50, 10, 1.5)
    att = E._attenuation(None, 10)
    y = E.fnrgf(img, r, phi, valid, st, att, att)
    # Circles stay whole up to r = 200 (half the height). Contrast just outside should match inside.
    inside = valid & (r > 170) & (r < 195)
    outside = valid & (r > 205) & (r < 230)
    assert np.abs(y[outside]).mean() == pytest.approx(np.abs(y[inside]).mean(), rel=0.25)
    assert abs(y[outside].mean() - y[inside].mean()) < 0.3
    assert np.isfinite(y).all() and np.abs(y).max() < 50


def test_noise_map_recovers_noise():
    rng = np.random.default_rng(4)
    h, w = 300, 300
    weight = np.tile(np.linspace(1.0, 16.0, w, dtype=np.float32), (h, 1))  # noise 1 -> 0.25
    sigma = 0.01 / np.sqrt(weight)
    clean = np.full((h, w), 0.5, np.float32)
    lum = clean + rng.normal(0, 1, (h, w)).astype(np.float32) * sigma
    valid = np.ones((h, w), bool)
    n = E.noise_map(lum, weight, valid)
    assert np.median(n[:, :20] / sigma[:, :20]) == pytest.approx(1.0, rel=0.15)
    assert np.median(n[:, -20:] / sigma[:, -20:]) == pytest.approx(1.0, rel=0.15)


def test_fnrgf_n_boosts_faint_noisy_corona():
    img, g = _scene(size=(500, 500), sun=(250.0, 250.0), noise=0.02)
    r, phi, valid = _valid(img.shape, g)
    noise = np.full(img.shape, 0.02, np.float32)
    st = E.ring_stats(img, r, phi, valid, 50, 10, 1.5, noise)
    att = E._attenuation(None, 10)
    y0 = E.fnrgf(img, r, phi, valid, st, att, att, 0.0)
    y1 = E.fnrgf(img, r, phi, valid, st, att, att, 0.7)
    far = valid & (r > 200) & (r < 240)
    near = valid & (r > 70) & (r < 100)
    assert y1[far].std() > 1.2 * y0[far].std()  # faint, noise-dominated: more contrast
    assert y1[near].std() == pytest.approx(y0[near].std(), rel=0.1)  # bright: unchanged


def test_blur_matches_gaussian_for_large_sigma():
    rng = np.random.default_rng(0)
    img = ndimage.gaussian_filter(rng.normal(0, 1, (256, 300)), 3).astype(np.float32)
    a = E.blur(img, 20.0)
    b = ndimage.gaussian_filter(img, 20.0, mode="reflect")
    core = (slice(40, -40), slice(40, -40))
    assert np.abs(a[core] - b[core]).max() < 0.05 * np.abs(b[core]).max() + 1e-3


def test_enhance_layers_and_colour():
    img, g = _scene(size=(300, 400), sun=(200.0, 150.0), r_sun=30.0, noise=0.002)
    hdr = np.stack([img * 1.3, img, img * 0.7], axis=-1)
    weight = np.ones(img.shape, np.float32)
    header = {"SUNX": g.sun_x, "SUNY": g.sun_y, "SUNR": g.sun_r, "MOONX": g.moon_x, "MOONY": g.moon_y,
              "MOONR": g.moon_r}
    res, layers = E.enhance(hdr.copy(), weight, header, E.EnhanceParams())
    assert set(layers) == set(E.LAYERS) | {"blend"}
    # Corona white balance: the tint is removed.
    assert res.white == pytest.approx([1 / 1.3, 1.0, 1 / 0.7], rel=0.02)
    for name, img_out in layers.items():
        assert img_out.shape == (300, 400, 3)
        assert np.isfinite(img_out).all() and img_out.min() >= 0 and img_out.max() <= 1
        # Black Moon.
        assert img_out[int(g.moon_y), int(g.moon_x)].max() == 0
        # Grey after white balance.
        sel = img_out.mean(axis=2) > 0.2
        assert np.abs(img_out[sel][:, 0] - img_out[sel][:, 2]).mean() < 0.02, name


def test_cli_enhance(tmp_path, capsys):
    import tifffile

    from tse_core import cli
    from tse_core.fits import write_fits
    from tse_core.project import EventProject

    img, g = _scene(size=(240, 320), sun=(160.0, 120.0), r_sun=25.0, noise=0.002)
    hdr = np.stack([img, img, img], axis=0)
    header = {"SUNX": g.sun_x, "SUNY": g.sun_y, "SUNR": g.sun_r, "MOONX": g.moon_x, "MOONY": g.moon_y,
              "MOONR": g.moon_r, "WBR": 2.0, "WBG": 1.0, "WBB": 1.5}
    write_fits(tmp_path / "e_hdr.fit", hdr, header)
    write_fits(tmp_path / "e_weight.fit", np.ones((240, 320), np.float32))
    project = EventProject(name="e")
    project.parameters["stack"] = {"hdr": str(tmp_path / "e_hdr.fit"), "weight": str(tmp_path / "e_weight.fit")}
    path = tmp_path / "e.json"
    project.save(path)
    assert cli.main(["enhance", str(path), "--layers", "natural,fnrgf_n,adaptive", "--quiet"]) == 0
    saved = EventProject.load(path).parameters["enhance"]
    for k in ("natural", "fnrgf_n", "adaptive", "blend"):
        t = tifffile.imread(saved[k])
        assert t.shape == (240, 320, 3) and t.dtype == np.uint16
    assert "nrgf" not in saved
    assert cli.main(["enhance", str(path), "--blend", "bogus=1"]) == 1
    assert cli.main(["enhance", str(path), "--layers", "nope"]) == 1
    assert cli.main(["enhance", str(path), "--white", "camera", "--blend", "none", "--layers", "natural",
                     "--quiet"]) == 0
    assert "blend" not in EventProject.load(path).parameters["enhance"]


def test_sky_background_cast_is_removed():
    """A small green sky offset would tint the whole stretched image green."""
    img, g = _scene(size=(300, 400), sun=(200.0, 150.0), r_sun=30.0, noise=0.0005)
    sky = np.array([0.002, 0.006, 0.003], np.float32)  # green-heavy sky / black-level residual
    hdr = np.stack([img, img, img], axis=-1) + sky
    weight = np.ones(img.shape, np.float32)
    header = {"SUNX": g.sun_x, "SUNY": g.sun_y, "SUNR": g.sun_r, "MOONX": g.moon_x, "MOONY": g.moon_y,
              "MOONR": g.moon_r}
    res, layers = E.enhance(hdr.copy(), weight, header, E.EnhanceParams(layers=("natural", "fnrgf_n")))
    assert res.white == pytest.approx([1.0, 1.0, 1.0], abs=0.02)  # the corona itself is white
    r, _, _ = _valid(img.shape, g)
    outer = r > 3.5 * g.sun_r  # faint corona, where the sky matters

    def green(out):
        sel = outer & (out.mean(axis=2) > 0.05)
        return np.median(((out[..., 1] - 0.5 * (out[..., 0] + out[..., 2])) / out.mean(axis=2))[sel])

    for name in ("natural", "fnrgf_n"):
        assert abs(green(layers[name])) < 0.005, name
    _, cast = E.enhance(hdr.copy(), weight, header, E.EnhanceParams(layers=("natural",), blend={},
                                                                         neutral_background=False))
    assert green(cast["natural"]) > 0.01  # the cast it fixes


def test_sky_model_removes_gradient_but_keeps_corona():
    H, W = 460, 690
    g = E.Geometry(330.0, 232.0, 35.0, 331.0, 232.0, 36.8)
    sc = SyntheticScene(size=W, sun_radius=35.0, seed=7)
    corona = sc.corona(g.sun_x, g.sun_y, shape=(H, W)).astype(np.float32) * 0.002
    xx = np.arange(W)[None, :]
    yy = np.arange(H)[:, None]
    sky = (0.004 + 0.002 * xx / W + 0.001 * yy / H) * (1 - 0.5 * ((xx - W / 2) ** 2 + (yy - H / 2) ** 2)
                                                      / (W * W / 4 + H * H / 4))
    img = (corona + sky).astype(np.float32)
    r, phi, valid = _valid((H, W), g)
    models, kind = E.sky_model(img[..., None], r, valid, g)
    assert kind == "gradient"
    fitted = E.sky_surface(models[0], (H, W))
    far = valid & (r > 4 * g.sun_r)
    err = np.abs(fitted - sky)[far]
    assert np.median(err) < 0.1 * (sky.max() - sky.min())
    # The corona at 2-3 solar radii is not taken for sky.
    mid = valid & (r > 2 * g.sun_r) & (r < 3 * g.sun_r)
    assert np.median((img - fitted)[mid] / corona[mid]) == pytest.approx(1.0, abs=0.15)


def test_overlay_and_blend_modes():
    base = np.array([0.2, 0.5, 0.8], np.float32)
    grey = np.full(3, 0.5, np.float32)
    assert np.allclose(E.overlay(base, grey), base)  # mid-grey: no change
    assert np.allclose(E.soft_light(base, grey), base)
    top = np.full(3, 0.8, np.float32)
    # Overlay is the stronger of the two.
    assert np.all(np.abs(E.overlay(base, top) - base) >= np.abs(E.soft_light(base, top) - base) - 1e-6)
    valid = np.ones((4, 4), bool)
    layers = {"natural": np.full((4, 4), 0.4, np.float32), "adaptive": np.full((4, 4), 0.9, np.float32),
              "fnrgf": np.full((4, 4), 0.9, np.float32)}
    layers["natural"][0, 0] = 0.0
    layers["natural"][3, 3] = 1.0  # so the final stretch keeps the scale
    a = E.soft_light_blend(layers, {"natural": 1, "adaptive": 0.5}, valid)
    b = E.soft_light_blend(layers, {"natural": 1, "adaptive": 0.5}, valid, {"adaptive": "overlay"})
    assert b[1, 1] > a[1, 1]
    with pytest.raises(ValueError):
        E.soft_light_blend(layers, {"natural": 1, "fnrgf": 0.1}, valid, {"fnrgf": "multiply"})


def test_parse_blend_with_modes():
    from tse_core import workflow as W

    text = "natural=1,adaptive=0.1:soft light,fnrgf=0.05:overlay"
    assert W.parse_blend(text) == {"natural": 1.0, "adaptive": 0.1, "fnrgf": 0.05}
    assert W.parse_blend_modes(text) == {"adaptive": "soft light", "fnrgf": "overlay"}
    with pytest.raises(W.WorkflowError):
        W.parse_blend_modes("fnrgf=0.1:screen")
