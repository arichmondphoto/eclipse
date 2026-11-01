import numpy as np, math
from scipy import ndimage

def orientation(b, sigma_g=3.0, sigma_t=10.0, detail=40.0):
    """Direction along the structure (radians) and coherence 0..1, from the structure tensor of
    the detail (``b`` minus a blur): the corona's overall fall-off would otherwise dominate."""
    b = b - ndimage.gaussian_filter(b, detail)
    gx = ndimage.gaussian_filter(b, sigma_g, order=(0, 1))
    gy = ndimage.gaussian_filter(b, sigma_g, order=(1, 0))
    jxx = ndimage.gaussian_filter(gx * gx, sigma_t)
    jyy = ndimage.gaussian_filter(gy * gy, sigma_t)
    jxy = ndimage.gaussian_filter(gx * gy, sigma_t)
    theta = 0.5 * np.arctan2(2 * jxy, jxx - jyy) + math.pi / 2  # perpendicular to the gradient
    tr = jxx + jyy
    coh = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2) / np.maximum(tr, 1e-30)
    return theta.astype(np.float32), coh.astype(np.float32)

def along(b, theta, coh, length=4.0, across=0.6, iso=2.0):
    """Gaussian smoothing along the structure (sigma ``length`` px) where it is coherent,
    a light isotropic smoothing (``iso``) where it is not. Line integral by sampling."""
    h, w = b.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    c, s = np.cos(theta), np.sin(theta)
    n = int(math.ceil(2.5 * length))
    ts = np.arange(-n, n + 1, dtype=np.float32)
    wts = np.exp(-0.5 * (ts / length) ** 2); wts /= wts.sum()
    acc = np.zeros_like(b)
    base = ndimage.gaussian_filter(b, across) if across > 0 else b
    for t, wt in zip(ts, wts):
        acc += wt * ndimage.map_coordinates(base, [yy + t * s, xx + t * c], order=1, mode='nearest')
    isot = ndimage.gaussian_filter(b, iso)
    k = np.clip((coh - 0.2) / 0.3, 0, 1)
    return (k * acc + (1 - k) * isot).astype(np.float32)


# ---------------------------------------------------------------------------
# Prototype driver (plan step 2), as run on 1 Nov 2026 on hdr_crop_2500.fit.
# Not part of the toolkit yet: see docs/PLAN_bray_falls_look.md, step 2.
# ---------------------------------------------------------------------------

def blur(img, sigma):
    return ndimage.gaussian_filter(img, sigma)


def strands(lum, valid, sun_r, length=8.0, c=3.0, k=0.8, seed=0):
    """Local-contrast layer with noise kept down: log image, Moon filled from its surroundings,
    smoothed along the strands (direction from the mid-scale structure, so the corona's overall
    fall-off and the Moon's edge do not set it), then multi-scale normalization where each scale's
    detail is weighted by how far it stands above the noise (a Wiener gain from a noise image put
    through the same smoothing). Returns values in about [-1.2, 1.2]."""
    b = np.log(np.maximum(lum, 1e-5)).astype(np.float32)
    hp = lum - ndimage.gaussian_filter(lum, 1.0)
    loc = 1.4826 * ndimage.median_filter(np.abs(hp), size=9) / 0.96
    sig_log = ndimage.gaussian_filter(loc, 8) / np.maximum(ndimage.gaussian_filter(np.maximum(lum, 1e-5), 8), 1e-6)
    mm = valid.astype(np.float32)
    bf = np.where(valid, b, blur(b * mm, 8.0) / np.maximum(blur(mm, 8.0), 1e-6)).astype(np.float32)
    for _ in range(3):
        bf = np.where(valid, b, blur(bf, 16.0)).astype(np.float32)
    th, coh = orientation(bf)  # defaults used: sigma_g=3, sigma_t=10, detail=40
    bd = along(bf, th, coh, length)  # across=0.6, iso=2.0, coherence ramp 0.2-0.5
    noise = (np.random.default_rng(seed).normal(0, 1, b.shape) * sig_log).astype(np.float32)
    nd = np.where(valid, along(noise, th, coh, length), 0).astype(np.float32)
    acc = np.zeros_like(bd)
    scales = [sun_r * f for f in (0.006, 0.012, 0.025, 0.05, 0.1, 0.2)]
    for w in scales:
        d = bd - blur(bd, w)
        sd2 = blur(d * d, w)
        dn = nd - blur(nd, w)
        n2 = blur(dn * dn, w)
        g = np.maximum(sd2 - n2, 0) / np.maximum(sd2 + (c * c - 1) * n2, 1e-12)
        acc += np.arctan(k * g * d / np.sqrt(np.maximum(sd2, 1e-12)))
    return acc / len(scales)

# The look shown to Andy ("p8"): global = log image minus 0.9 x its ring median, stretched with
# roll_off(.., 0.3); local = strands() stretched 0.2-99.8 %; result = overlay(global, local) at 100%.
