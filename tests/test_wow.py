"""WOW, wavelet-optimized whitening (Auchère, Soubrié, Pelouze & Buchlin 2023, A&A 670, A66).

Synthetic tests check the paper's properties; the limb tests use the 2024 stack binned 8x in
``tests/data``. Nothing here depends on one camera or eclipse: the result must not change with the
stack's units.
"""

from pathlib import Path

import numpy as np
import pytest

from tse_core import enhance as E

DATA = Path(__file__).parent / "data" / "eclipse2024_bin8.npz"


@pytest.mark.parametrize("bilateral", [False, True])
def test_planes_and_residual_add_up_to_the_image(bilateral):
    """Eqs. 4-5: {w_1..w_S, c_S} adds up exactly to the image, plain and edge-aware."""
    x = np.random.default_rng(3).random((120, 170)).astype(np.float32) * 7
    planes = E.atrous(x, E.wow_scales(170), bilateral)
    assert len(planes) == 7 + 1  # ln 170 / ln 2 = 7.4 scales and the residual
    assert np.abs(sum(planes) - x).max() < 1e-5


def test_white_noise_response_is_the_b3_starlet():
    """sigma'_s measured on white noise: for the B3-spline à trous transform the known values are
    0.890, 0.201, 0.086 (Starck & Murtagh, Astronomical Image and Data Analysis, table 2.1)."""
    got = E.wow_noise_response(3, False)
    assert got == pytest.approx([0.890, 0.201, 0.086], abs=0.004)


def test_whitened_white_noise_has_a_flat_spectrum():
    """The point of whitening: every scale of the result carries about the same power."""
    n = np.random.default_rng(5).standard_normal((512, 512)).astype(np.float32)
    f = E.wow(n, np.ones(n.shape, bool), denoise=(), residual=False)
    stds = [float(p.std()) for p in E.atrous(f, 8)[1:7]]
    assert max(stds) / min(stds) < 1.3, stds


def test_result_does_not_depend_on_the_stack_units():
    """Any camera, any exposure: the stack's units are arbitrary, so scaling the stack and its noise
    together must not change WOW (this is why the coarse residual c_S of eq. 9 is left out: in the
    stack's own units it would make the picture depend on them)."""
    rng = np.random.default_rng(7)
    yy, xx = np.mgrid[0:200, 0:260]
    x = (1.0 / (1 + np.hypot(xx - 130, yy - 100) / 20.0) ** 3 + 0.01 * rng.random((200, 260))).astype(np.float32)
    noise = np.full(x.shape, 0.003, np.float32)
    valid = np.ones(x.shape, bool)
    a = E.wow_display(E.wow(x, valid, noise, residual=False), valid)
    b = E.wow_display(E.wow(x * 1000.0, valid, noise * 1000.0, residual=False), valid)
    assert np.abs(a - b).max() < 1e-4


def test_noise_thresholds_smooth_the_finest_scale():
    """Eq. 12: a higher n_s fades more of the noise at that scale."""
    rng = np.random.default_rng(9)
    x = (1.0 + 0.05 * rng.standard_normal((256, 256))).astype(np.float32)
    noise = np.full(x.shape, 0.05, np.float32)
    v = np.ones(x.shape, bool)
    fine = [float(E.atrous(E.wow(x, v, noise, denoise=(n,), residual=False), 1)[0].std()) for n in (0, 1, 5)]
    assert fine[0] > fine[1] > fine[2], fine


def _real_frame():
    z = np.load(DATA)
    hdr = np.ascontiguousarray(np.moveaxis(z["hdr"].astype(np.float32) * float(z["scale"]), 0, -1))
    geo = {k: float(z[k]) for k in ("SUNX", "SUNY", "SUNR", "MOONX", "MOONY", "MOONR")}
    weight = (np.abs(hdr).sum(axis=2) > 0).astype(np.float32)
    return E.prepare(hdr, weight, geo, E.EnhanceParams(layers=("natural", "wow")))


def _limb_profile(y, fr, dists=(1, 2, 3, 6, 12)):
    g = fr.g
    yy, xx = np.mgrid[0:y.shape[0], 0:y.shape[1]]
    d = np.hypot(xx - g.moon_x, yy - g.moon_y) - g.moon_r
    return [float(np.median(y[fr.valid & (d >= a) & (d < a + 1)])) for a in dists]


@pytest.mark.parametrize("bilateral", [False, True])
def test_no_white_band_at_the_limb_on_the_2024_stack(bilateral):
    """The Moon's disc has no data. Filled with the limb's own brightness (a plateau), every scale
    read the limb as the top of a ridge: on the 2024 stack (binned 8x) WOW was white (1.0) for the
    first 2-3 px outside the limb, against 0.82 (plain) and 0.87 (edge-aware) at 6 px. With the corona
    continued through the limb (odd reflection), the first pixels stay within 0.15 of the level 6 px
    out and below white."""
    fr = _real_frame()
    lum, valid, noise = fr.flum, fr.valid, fr.noise
    old = E.wow_display(E.wow(lum, valid, noise, bilateral=bilateral, residual=False), valid)
    po = _limb_profile(old, fr)
    assert po[0] > 0.99 and po[1] > 0.99  # the band, as it was
    x, v = E.wow_disc_fill(lum, valid, fr.g, fr.moon_margin_px)
    new = E.wow_display(E.wow(x, v, noise, bilateral=bilateral, residual=False), valid)
    pn = _limb_profile(new, fr)
    assert max(pn[:3]) < 0.97 and max(pn[:3]) - pn[3] < 0.15, pn
    assert np.array_equal(x[valid], lum[valid])  # pixels with data untouched


def test_wow_layer_detail_box_is_the_whole_image_render():
    fr = _real_frame()
    p = E.EnhanceParams(layers=("natural", "wow"), blend={"natural": 1.0, "wow": 0.43},
                        blend_modes={"wow": "linear"})
    whole = E.render(fr, p)
    part = E.render(fr, p, box=(150, 400, 300, 700))
    for k in ("wow", "blend"):
        assert np.abs(part[k] - whole[k][150:400, 300:700]).max() == 0


def test_wow_settings_in_a_settings_file_and_on_the_command_line(tmp_path):
    import sys

    from tse_core import workflow as W
    from tse_core.cli import enhance_params

    sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))
    from cli_reference import parser

    p = E.EnhanceParams(wow_edge_aware=True, wow_denoise=(4.0, 2.0, 0.0))
    f = W.save_enhance_settings(p, tmp_path / "look.json")
    back, _ = W.load_enhance_settings(f)
    assert back.wow_edge_aware is True and tuple(back.wow_denoise) == (4.0, 2.0, 0.0)
    sp = parser()._subparsers._group_actions[0].choices["enhance"]
    args = sp.parse_args(["x.json", "--wow-edge-aware", "--wow-denoise", "3,1"])
    got, _ = enhance_params(args)
    assert got.wow_edge_aware and got.wow_denoise == (3.0, 1.0)
    assert "wow" in E.LAYERS and "wow" not in E.DEFAULT_LAYERS  # on request, like ACHF


def test_noise_spread_over_neighbours_is_measured_at_each_scale():
    """A debayered, aligned stack's noise is spread over neighbouring pixels: with white noise's
    sigma'_s the thresholds acted at about half strength (on the 2024 stack the noise at scales 2
    and 3 was 1.8-2.2x white noise's). sigma'_s is measured on the image's faint half instead; on
    white noise it gives the white-noise values, on noise blurred over ~1 px it gives more at
    scales 2 and 3."""
    from scipy import ndimage

    rng = np.random.default_rng(11)
    white = rng.standard_normal((512, 512)).astype(np.float32)
    corr = ndimage.gaussian_filter(white, 1.0).astype(np.float32)
    v = np.ones(white.shape, bool)
    ref = E.wow_noise_response(3, False)
    for img, more in ((white, False), (corr, True)):
        x = img + 10.0
        pix = np.full(x.shape, float(img.std()), np.float32)  # the pixel noise
        got = E.wow_measured_response(E.atrous(x, 3), v, x, pix, ref)
        if more:
            # Gaussian blur of 1 px: 1.3x at scale 2 and 2.5x at scale 3 (on the 2024 stack 1.8x, 2.2x)
            assert got[1] > 1.2 * ref[1] and got[2] > 2.0 * ref[2], (got, ref)
        else:
            assert got == pytest.approx(ref, rel=0.06), (got, ref)
