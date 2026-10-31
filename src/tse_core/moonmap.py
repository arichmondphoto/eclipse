"""A lunar albedo map as a guide for the earthshine glare removal.

Near the limb the glare is many times the earthshine and changes on the
same scales as the maria, so from the data alone a mare by the limb and a
dip in the glare look alike. A map of the Moon's surface brightness (an
albedo map, such as NASA's CGI Moon Kit colour map, which is centred on 0
degrees longitude in an equirectangular 2:1 layout) says what pattern the
surface should make there.

How it is used, so that the map's own features cannot leak into the picture:

* The map is turned to match *your* image: rotation, mirror and libration are
  searched for the best match with the lunar pattern in the middle of your
  disc (no clock, site or ephemeris needed). If the match is poor it is not used.
* It is blurred (no sharp features) and scaled to your data, and only ever
  *subtracted before the glare is estimated*; the glare estimate is then
  subtracted from your data. Nothing from the map is added to the picture.
* Before that, the map is pulled towards your data wherever the data can tell
  (see earthshine.remove_glare_guided), so a feature the map has and your
  Moon does not is mostly rejected. What remains is the map's say where the
  data cannot separate surface from glare: the outer ~30 px, scaled by
  ``albedo_strength``.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import ndimage


@dataclass
class Orientation:
    rotation_deg: float  # turn of the map's north from image up, counter-clockwise
    mirror: bool
    lib_lon: float  # longitude at the disc centre (libration), degrees
    lib_lat: float  # latitude at the disc centre, degrees
    correlation: float  # match with the data in the middle of the disc (0..1)

    def describe(self) -> str:
        return (f"albedo map: rotation {self.rotation_deg:.1f} deg{', mirrored' if self.mirror else ''}, "
                f"centre at lon {self.lib_lon:+.1f}, lat {self.lib_lat:+.1f} deg, "
                f"match with your Moon {self.correlation:.2f}")

    def as_dict(self) -> dict:
        return asdict(self)


def load_map(path, max_width: int = 2048) -> np.ndarray:
    """Equirectangular map (2:1, 0 deg longitude in the middle, north at the top) as grey 0..1."""
    from .imageout import read_image

    img = read_image(str(path))
    img = np.asarray(img, np.float32)
    if img.ndim == 3:
        img = img[..., :3].mean(axis=2)
    h, w = img.shape
    if abs(w / h - 2.0) > 0.05:
        raise ValueError(f"{Path(path).name}: an albedo map must be twice as wide as it is high (got {w} x {h})")
    if w > max_width:
        f = w // max_width
        img = img[: h - h % f, : w - w % f].reshape(h // f, f, w // f, f).mean(axis=(1, 3))
    hi = float(np.percentile(img, 99.9)) or 1.0
    return np.clip(img / hi, 0.0, None).astype(np.float32)


def render(amap: np.ndarray, shape, cx: float, cy: float, r: float, o: Orientation | tuple) -> np.ndarray:
    """The map as the Moon would look from the observer: orthographic, on an image grid.

    Outside the disc the result is 0. ``o`` is an Orientation or (rotation, mirror, lon, lat).
    """
    if isinstance(o, Orientation):
        rot, mirror, lon0, lat0 = o.rotation_deg, o.mirror, o.lib_lon, o.lib_lat
    else:
        rot, mirror, lon0, lat0 = o
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    u = (xx - cx) / r
    v = -(yy - cy) / r  # up
    rho2 = u * u + v * v
    inside = rho2 < 1.0
    a = math.radians(rot)
    up = u * math.cos(a) + v * math.sin(a)
    vp = -u * math.sin(a) + v * math.cos(a)
    if mirror:
        up = -up
    X, Y = up[inside], vp[inside]
    Z = np.sqrt(np.clip(1.0 - X * X - Y * Y, 0.0, 1.0))
    b, l = math.radians(lat0), math.radians(lon0)
    # Tilt by the latitude at the centre, then turn by the longitude.
    Y1 = Y * math.cos(b) + Z * math.sin(b)
    Z1 = -Y * math.sin(b) + Z * math.cos(b)
    X2 = X * math.cos(l) + Z1 * math.sin(l)
    Z2 = -X * math.sin(l) + Z1 * math.cos(l)
    lat = np.arcsin(np.clip(Y1, -1.0, 1.0))
    lon = np.arctan2(X2, Z2)
    mh, mw = amap.shape
    col = (lon / (2 * math.pi) + 0.5) * mw - 0.5
    row = (0.5 - lat / math.pi) * mh - 0.5
    vals = ndimage.map_coordinates(amap, [row, col], order=1, mode="grid-wrap")
    out = np.zeros((h, w), np.float32)
    out[inside] = vals
    return out


def _poly_residual(img: np.ndarray, mask: np.ndarray, cx: float, cy: float, r: float, order: int = 2) -> np.ndarray:
    """``img`` minus a least-squares 2-D polynomial over ``mask`` (the same broad removal the data gets)."""
    h, w = img.shape
    yy, xx = np.mgrid[0:h, 0:w]
    u, v = (xx - cx) / r, (yy - cy) / r
    terms = [(i, j) for i in range(order + 1) for j in range(order + 1 - i)]
    A = np.stack([(u[mask] ** i) * (v[mask] ** j) for i, j in terms], 1)
    coef, *_ = np.linalg.lstsq(A, img[mask].astype(np.float64), rcond=None)
    fit = sum(c * (u**i) * (v**j) for c, (i, j) in zip(coef, terms))
    return np.where(mask, img - fit, 0.0).astype(np.float32)


def _masked_blur(img: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur averaging only pixels inside ``mask`` (no darkening at the limb)."""
    m = mask.astype(np.float32)
    num = ndimage.gaussian_filter(np.where(mask, img, 0.0).astype(np.float32), sigma)
    den = ndimage.gaussian_filter(m, sigma)
    return np.where(mask, num / np.maximum(den, 1e-6), 0.0).astype(np.float32)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    return float(np.sum(a * b) / (math.sqrt(float(np.sum(a * a) * np.sum(b * b))) + 1e-30))


def fit_orientation(amap: np.ndarray, detail: np.ndarray, good: np.ndarray, cx: float, cy: float, r: float,
                    size: int = 160, max_libration: float = 8.0) -> Orientation:
    """Turn the map to match the lunar pattern in the middle of the disc (inside 0.8 radius).

    Searched over rotation (all of it), mirror, then libration and fine rotation, on a
    reduced copy; the pattern is compared after the same broad polynomial removal and blur.
    """
    f = max(1, int(round(2 * r / size)))
    h, w = detail.shape
    hh, ww = h // f, w // f
    d = detail[: hh * f, : ww * f].reshape(hh, f, ww, f).mean(axis=(1, 3))
    g = good[: hh * f, : ww * f].reshape(hh, f, ww, f).all(axis=(1, 3))
    scx, scy, sr = (cx + 0.5) / f - 0.5, (cy + 0.5) / f - 0.5, r / f
    yy, xx = np.mgrid[0:hh, 0:ww]
    body = g & (np.hypot(xx - scx, yy - scy) < 0.8 * sr)
    sigma = max(1.0, 0.02 * sr)
    dd = _poly_residual(ndimage.gaussian_filter(d, sigma), body, scx, scy, sr)[body]

    def score(o):
        t = render(amap, (hh, ww), scx, scy, sr, o)
        t = _poly_residual(ndimage.gaussian_filter(t, sigma), body, scx, scy, sr)[body]
        return _corr(dd, t)

    best = (-2.0, (0.0, False, 0.0, 0.0))
    for mirror in (False, True):
        for rot in np.arange(0.0, 360.0, 3.0):
            c = score((rot, mirror, 0.0, 0.0))
            if c > best[0]:
                best = (c, (rot, mirror, 0.0, 0.0))
    c0, (rot, mirror, _, _) = best
    step = 2.0
    lon, lat = 0.0, 0.0
    for _ in range(3):  # coordinate refinement: libration grid, then rotation, finer each time
        for dl in np.arange(-max_libration, max_libration + 1e-9, step):
            for db in np.arange(-max_libration, max_libration + 1e-9, step):
                c = score((rot, mirror, dl, db))
                if c > best[0]:
                    best = (c, (rot, mirror, dl, db))
        c0, (rot, mirror, lon, lat) = best
        for dr in np.arange(-3 * step, 3 * step + 1e-9, step / 2):
            c = score((rot + dr, mirror, lon, lat))
            if c > best[0]:
                best = (c, (rot + dr, mirror, lon, lat))
        c0, (rot, mirror, lon, lat) = best
        max_libration = min(max_libration, 3 * step)
        step /= 2
    return Orientation(float(rot % 360.0), bool(mirror), float(lon), float(lat), float(best[0]))


def guide(amap: np.ndarray, detail: np.ndarray, good: np.ndarray, cx: float, cy: float, r: float,
          orientation: Optional[Orientation] = None, blur: float = 0.02):
    """The map turned, blurred and scaled to the data: what the lunar surface should add.

    Returns (template, orientation, scale). The template is blurred by ``blur`` x radius
    (so it carries no sharp features) and scaled by a fit of its pattern to the data's
    inside 0.8 radius (both with the same broad polynomial removed).
    """
    o = orientation or fit_orientation(amap, detail, good, cx, cy, r)
    t = render(amap, detail.shape, cx, cy, r, o)
    sig = max(1.0, blur * r)
    t = _masked_blur(t, good, sig)
    h, w = detail.shape
    yy, xx = np.mgrid[0:h, 0:w]
    rr = np.hypot(xx - cx, yy - cy)
    # The scale comes from the patterns (both with the same broad polynomial removed); the
    # template itself keeps its full level, as the surface adds that to the image too.
    tp = _poly_residual(t, good, cx, cy, r)
    body = good & (rr < 0.8 * r)
    ds = _poly_residual(_masked_blur(detail, good, sig), good, cx, cy, r)
    a = tp[body] - tp[body].mean()
    b = ds[body] - ds[body].mean()
    k = float(np.sum(a * b) / (np.sum(a * a) + 1e-30))
    t = np.where(good, t, 0.0)
    return (t * k).astype(np.float32), o, k
