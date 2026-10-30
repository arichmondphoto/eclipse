"""Linear demosaicing of calibrated Bayer data, in floating point.

Malvar, He & Cutler (2004), "High-quality linear interpolation for
demosaicing of Bayer-patterned color images": bilinear interpolation
corrected by the local Laplacian of the channel that was sampled. It is
linear (so it keeps the data's radiometry, which HDR stacking needs), has
no thresholds, and works on negative values, so faint noisy sky is not
clipped and biased upwards as it would be by an 8- or 16-bit converter.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

# Kernels from the paper, divided by 8.
_G_AT_RB = np.array([
    [0, 0, -1, 0, 0],
    [0, 0, 2, 0, 0],
    [-1, 2, 4, 2, -1],
    [0, 0, 2, 0, 0],
    [0, 0, -1, 0, 0]], np.float32) / 8
# Red (or blue) at a green site whose row holds red (or blue).
_RB_AT_G_SAME_ROW = np.array([
    [0, 0, 0.5, 0, 0],
    [0, -1, 0, -1, 0],
    [-1, 4, 5, 4, -1],
    [0, -1, 0, -1, 0],
    [0, 0, 0.5, 0, 0]], np.float32) / 8
_RB_AT_G_SAME_COL = _RB_AT_G_SAME_ROW.T.copy()
# Red at blue sites, blue at red sites.
_RB_AT_BR = np.array([
    [0, 0, -1.5, 0, 0],
    [0, 2, 0, 2, 0],
    [-1.5, 0, 6, 0, -1.5],
    [0, 2, 0, 2, 0],
    [0, 0, -1.5, 0, 0]], np.float32) / 8


def channel_masks(pattern: str, shape: tuple[int, int]) -> dict[str, np.ndarray]:
    """Boolean masks for R, G and B photosites; ``pattern`` like "RGGB" (row-major 2x2)."""
    pattern = pattern.upper()
    if len(pattern) != 4 or sorted(pattern) != ["B", "G", "G", "R"]:
        raise ValueError(f"unsupported CFA pattern {pattern!r}")
    h, w = shape
    masks = {c: np.zeros((h, w), bool) for c in "RGB"}
    for i, (dy, dx) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
        masks[pattern[i]][dy::2, dx::2] = True
    return masks


def demosaic_mhc(cfa: np.ndarray, pattern: str) -> np.ndarray:
    """(h, w) mosaic -> (h, w, 3) RGB, float32."""
    cfa = np.asarray(cfa, dtype=np.float32)
    h, w = cfa.shape
    m = channel_masks(pattern, (h, w))
    rows_with_r = np.zeros((h, w), bool)
    rows_with_b = np.zeros((h, w), bool)
    r_rows = np.any(m["R"], axis=1)
    b_rows = np.any(m["B"], axis=1)
    rows_with_r[r_rows, :] = True
    rows_with_b[b_rows, :] = True

    def conv(k):
        return ndimage.correlate(cfa, k, mode="mirror")

    g_rb = conv(_G_AT_RB)
    same_row = conv(_RB_AT_G_SAME_ROW)
    same_col = conv(_RB_AT_G_SAME_COL)
    cross = conv(_RB_AT_BR)

    out = np.empty((h, w, 3), np.float32)
    # Green.
    out[..., 1] = np.where(m["G"], cfa, g_rb)
    # Red: at R sites the sample; at G sites in red rows the row kernel, in blue rows the column
    # kernel; at B sites the cross kernel.
    red = np.where(m["R"], cfa, cross)
    red = np.where(m["G"] & rows_with_r, same_row, red)
    red = np.where(m["G"] & rows_with_b, same_col, red)
    out[..., 0] = red
    blue = np.where(m["B"], cfa, cross)
    blue = np.where(m["G"] & rows_with_b, same_row, blue)
    blue = np.where(m["G"] & rows_with_r, same_col, blue)
    out[..., 2] = blue
    return out


def cosmetic_correct(cfa: np.ndarray, pattern: str, n_sigma: float = 6.0) -> tuple[np.ndarray, int]:
    """Replace hot (and dead) photosites by the median of their same-colour neighbours.

    Without dark frames, hot pixels are found from the light frame itself: a
    photosite far above (or below) its eight same-colour neighbours, measured
    against the local noise, is replaced. Returns (corrected, count).
    """
    out = np.array(cfa, dtype=np.float32, copy=True)
    h, w = out.shape
    resid = np.zeros((h, w), np.float32)
    med_full = np.zeros((h, w), np.float32)
    bad = np.zeros((h, w), bool)
    for dy in (0, 1):
        for dx in (0, 1):
            plane = out[dy::2, dx::2]
            med = ndimage.median_filter(plane, size=3, mode="mirror")
            r = plane - med
            noise = ndimage.uniform_filter(np.abs(r), size=9) * 1.25 + 1e-6
            resid[dy::2, dx::2] = r
            med_full[dy::2, dx::2] = med
            bad[dy::2, dx::2] = np.abs(r) > n_sigma * noise
    # A real feature (a star, a prominence, a sharp limb) is wider than one photosite, so the
    # adjacent photosites of other colours deviate the same way. A defect stands alone.
    ring = np.ones((3, 3), bool)
    ring[1, 1] = False
    up = ndimage.maximum_filter(resid, footprint=ring, mode="mirror")
    down = ndimage.minimum_filter(resid, footprint=ring, mode="mirror")
    alone = np.where(resid > 0, up < 0.25 * resid, down > 0.25 * resid)
    bad &= alone
    out[bad] = med_full[bad]
    return out, int(bad.sum())
