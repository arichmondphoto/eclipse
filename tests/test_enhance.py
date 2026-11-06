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
    assert E.EnhanceParams().layers == E.DEFAULT_LAYERS and "achf" not in E.DEFAULT_LAYERS  # slow: on request
    res, layers = E.enhance(hdr.copy(), weight, header, E.EnhanceParams(layers=E.LAYERS))
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


def test_highlight_shoulder_rolls_off_without_clipping():
    """With a shoulder the brightest values keep their order and gradations below 1 instead of
    clipping; below the knee nothing changes; 0 leaves the hard white point as before."""
    import numpy as np

    from tse_core.enhance import roll_off, stretch

    y = np.linspace(0, 3, 301).astype(np.float32)
    out = roll_off(y, 0.3)
    assert np.allclose(out[y <= 0.7], y[y <= 0.7])
    assert np.all(np.diff(out) > 0) and out.max() < 1.0
    assert np.array_equal(roll_off(y, 0.0), y)
    valid = np.ones_like(y, bool)
    assert (stretch(y, valid, 0, 90) >= 1.0).sum() > 20  # hard white point: the top clips
    assert (stretch(y, valid, 0, 90, shoulder=0.3) >= 0.999).sum() <= 1


def _noisy_corona(strands_amp, seed=1):
    """A corona falling off from the limb, 2% noise correlated like a demosaiced, resampled stack,
    optionally with radial strands; weights are the true inverse variances."""
    import numpy as np
    from scipy import ndimage

    n, R = 700, 100.0
    yy, xx = np.mgrid[0:n, 0:n]
    r = np.hypot(xx - n / 2, yy - n / 2)
    phi = np.arctan2(yy - n / 2, xx - n / 2)
    img = 0.1 * np.exp(-(r - R) / 60) + 1e-3
    img = img * (1 + strands_amp * np.cos(40 * phi) * np.exp(-(r - R) / 200))
    z = ndimage.gaussian_filter(np.random.default_rng(seed).normal(0, 1, img.shape), 0.45)
    img = (img * (1 + 0.02 * z / z.std())).astype(np.float32)
    return img, (1 / (0.02 * img) ** 2).astype(np.float32), r > R * 1.06, r, R


def test_gated_adaptive_does_not_make_texture_out_of_noise():
    """The noise-gated adaptive filter brings out real strands
    but leaves pure noise nearly flat; the plain filter turns that noise into full-contrast
    texture. Measured as fine texture (minus a 10 px blur) away from the limb."""
    import numpy as np
    from scipy import ndimage

    from tse_core.enhance import mgn, mgn_gated, noise_map

    def texture(strands_amp, gated):
        img, w, valid, r, R = _noisy_corona(strands_amp)
        scales = [R * f for f in (0.006, 0.012, 0.025, 0.05, 0.1, 0.2)]
        if gated:
            out = mgn_gated(img, valid, scales, 0.7, 1e-4, noise_map(img, w, valid), R, gate=3.0)
        else:
            out = mgn(img, valid, scales, 0.7, 1e-4)
        tex = out - ndimage.gaussian_filter(out, 10)
        return float(tex[valid & (r > 1.3 * R) & (r < 3.5 * R)].std())

    noise_gated, noise_plain = texture(0.0, True), texture(0.0, False)
    strands_gated = texture(0.15, True)
    assert noise_gated < 0.4 * noise_plain, (noise_gated, noise_plain)
    assert noise_gated < 0.4 * strands_gated, (noise_gated, strands_gated)


def test_linear_blend_is_the_papers_weighted_average():
    """Linear (paper): the images combined as the papers do, a weighted average on the display range
    (FNRGF paper: Z = K1 X + K2 Y with K1, K2 non-negative and only their ratio mattering; the
    thesis's FNRGF software scales both images to 0..1 first; MGN: h C'g + (1 - h) C'; the 2011
    paper's Fig. 9: 0.4 FNRGF + 0.6 ACHF). The natural layer has weight 1 (K1), each Linear layer
    its strength (K2), which may exceed 1. The order does not matter, and the result is not
    re-stretched, so a detail box still matches the full-size result. Since 6 Nov 2026 MGN's detail
    goes in as its paper adds it, centred on 0 and in radians ((layer - 0.5) pi), the result
    clipped to 0..1 (see test_paper_mapping)."""
    valid = np.ones((3, 4), bool)
    nat = np.array([[0.0, 0.3, 0.5, 1.0]] * 3, np.float32)
    a = np.array([[0.2, 0.6, 0.4, 0.95]] * 3, np.float32)
    f = np.array([[0.7, 0.5, 0.45, 0.5]] * 3, np.float32)
    layers = {"natural": nat, "adaptive": a, "fnrgf": f}
    w = {"natural": 1, "adaptive": 0.6, "fnrgf": 2.0}  # K2 above 1 (the thesis: "common values are 10 or lower")
    out = E.soft_light_blend(layers, w, valid, {"adaptive": "linear", "fnrgf": "linear"})
    d = (a - 0.5) * np.pi
    expect = np.clip((nat + 0.6 * d + 2.0 * f) / 3.6, 0, 1)
    assert np.allclose(out, expect, atol=1e-6)
    # The paper's MGN form: h C'g + (1 - h) C' with h = 0.7 is the weight (1 - h) / h against the natural layer.
    mgn = E.soft_light_blend({"natural": nat, "adaptive": a}, {"natural": 1, "adaptive": 0.3 / 0.7}, valid,
                             {"adaptive": "linear"})
    assert np.allclose(mgn, np.clip(0.7 * nat + 0.3 * d, 0, 1), atol=1e-6)
    # Soft Light layers are applied to the base first, then averaged with the Linear ones.
    mixed = E.soft_light_blend(layers, {"natural": 1, "adaptive": 0.6, "fnrgf": 0.5}, valid, {"fnrgf": "linear"})
    sl = (1 - 0.6) * nat + 0.6 * E.soft_light(nat, a)
    assert np.allclose(mixed, (sl + 0.5 * f) / 1.5, atol=1e-6)
    # Near the Moon the filter layers fade out (their data ends there): each weight times the fade.
    fade = np.full_like(nat, 0.5)
    half = E.soft_light_blend(layers, w, valid, {"adaptive": "linear", "fnrgf": "linear"}, fade=fade)
    assert np.allclose(half, np.clip((nat + 0.3 * d + 1.0 * f) / 2.3, 0, 1), atol=1e-6)


def test_parse_linear_blend_mode():
    from tse_core import workflow as W

    assert W.parse_blend_modes("fnrgf_n=0.6:linear,adaptive=0.4:Linear (paper)") == {"fnrgf_n": "linear",
                                                                                     "adaptive": "linear"}
