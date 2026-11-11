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
    planes = E.atrous(x, E.wow_scales(x.shape), bilateral)
    assert len(planes) == 5 + 1  # log2 120 - log2 5 = 4.6 scales (as the authors' code), and the residual
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
    S = E.wow_scales(n.shape)  # 7: the scales that are whitened; the last one borders the residual
    stds = [float(p.std()) for p in E.atrous(f, S + 1)[1:S - 1]]
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
    read the limb as the top of a ridge: on the test data (binned 8x) WOW was white (1.0) for the first
    pixels outside the limb. With the corona continued through the limb (odd reflection) it stays well
    below white, and what rises towards the limb is a small fraction of the plateau's band. The layer
    is made as in the toolkit (the coarse residual added, as the authors' code does), so the corona
    also keeps its gentle overall fall outwards (plain: 0.86 at the limb, 0.69 at 6 px, 0.50 at 80 px)."""
    fr = _real_frame()
    lum, valid, noise = fr.flum, fr.valid, fr.noise
    old = E.wow_display(E.wow(lum, valid, noise, bilateral=bilateral), valid)
    po = _limb_profile(old, fr)
    x, v = E.wow_disc_fill(lum, valid, fr.g, fr.moon_margin_px)
    new = E.wow_display(E.wow(x, v, noise, bilateral=bilateral), valid)
    pn = _limb_profile(new, fr)
    # The plateau's band is white (clipped) over the first 2 px, so its size cannot be measured
    # beyond that; the reflection keeps those pixels clearly below white (0.86 plain, 0.72 edge-aware).
    assert min(po[:2]) > 0.97  # the band, as it was
    assert max(pn[:3]) < 0.9, (po, pn)
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


# --- Matching the authors' own code (watroo, Auchère et al.), Nov 2026 -------------------------------
# Compared with frederic-auchere/wavelets (watroo/utils.py, wow): three differences were found and
# put right; the references below are written from the formulas, not from their code.


def _reference_wow(x, n_scales, denoise=(), noise=None, white=None):
    """The authors' order of operations: each plane's local power P_s = h_s * w_s^2 from the plane
    as transformed, then the plane denoised (erf), then divided by sqrt(P_s); the residual divided by
    its own standard deviation."""
    from scipy.special import erf

    planes = E.atrous(x, n_scales)
    out = np.zeros_like(x)
    for s, w in enumerate(planes[:-1]):
        power = E.atrous_smooth(w * w, s)
        if s < len(denoise) and denoise[s] > 0:
            w = w * erf(np.abs(w) / (denoise[s] * white[s] * noise))
        out += w / np.sqrt(power)
    c = planes[-1]
    return out + c / c.std()


def test_denoised_planes_are_divided_by_their_power_before_denoising():
    """Ours measured each plane's local power after the noise threshold, so in areas holding only
    noise the faded noise was divided by a correspondingly smaller power and came back: the noise
    thresholds acted far more weakly than set. The authors measure the power first."""
    rng = np.random.default_rng(21)
    yy, xx = np.mgrid[0:256, 0:320]
    x = (1.0 / (1 + np.hypot(xx - 160, yy - 128) / 15.0) ** 2 + 0.01 * rng.standard_normal((256, 320)))
    x = x.astype(np.float32)
    v = np.ones(x.shape, bool)
    noise = np.full(x.shape, 0.01, np.float32)
    S = E.wow_scales(x.shape)
    got = E.wow(x, v, noise, denoise=(5.0, 2.0, 1.0))
    ref = _reference_wow(x, S, (5.0, 2.0, 1.0), 0.01, E.wow_noise_response(3, False))
    assert np.abs(got - ref).max() < 1e-3 * np.abs(ref).max()
    # and the threshold now takes the noise out: the finest plane of the result, in the noise-only corner
    corner = (slice(0, 40), slice(0, 40))
    fine = lambda f: float(E.atrous(f, 1)[0][corner].std())
    assert fine(got) < 0.5 * fine(E.wow(x, v, noise, denoise=()))


@pytest.mark.parametrize("shape, n", [((5504, 8256), 10), ((3000, 3000), 9), ((933, 1400), 8), ((120, 170), 5)])
def test_number_of_scales_as_the_authors(shape, n):
    """The authors stop at log2 of the shorter side less log2 of the kernel's 5 samples, so the
    coarsest smoothing never spans the frame; ours took ln N / ln 2 of the longer side, 3 scales more
    on a 45-megapixel frame, where the kernel's not quite round shape showed as squarish gradients."""
    assert E.wow_scales(shape) == n


def test_coarse_residual_is_added_divided_by_its_spread():
    """The authors add the coarse residual divided by its own standard deviation, which keeps WOW
    independent of the stack's units; ours left it out."""
    rng = np.random.default_rng(4)
    yy, xx = np.mgrid[0:200, 0:260]
    x = (1.0 / (1 + np.hypot(xx - 130, yy - 100) / 20.0) ** 3 + 0.002 * rng.random((200, 260))).astype(np.float32)
    v = np.ones(x.shape, bool)
    S = E.wow_scales(x.shape)
    got = E.wow(x, v)
    c = E.atrous(x, S)[-1]
    assert np.abs(got - (E.wow(x, v, residual=False) + c / c.std())).max() < 1e-4
    assert np.abs(E.wow(x * 1000.0, v) - got).max() < 1e-3
