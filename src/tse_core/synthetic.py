"""Synthetic total-eclipse frames with known geometry, for testing.

The corona is a steep radial falloff modulated by streamers and fine
plumes, fixed to the Sun. The Moon is a dark disc that moves across it.
Frames can be shifted, rotated and scaled by known amounts, exposed for
any time (with clipping at full scale) and given photon and read noise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass
class SyntheticScene:
    size: int = 512
    sun_radius: float = 60.0
    moon_ratio: float = 1.055
    seed: int = 1

    def corona(self, sun_x: float, sun_y: float, angle_deg: float = 0.0, scale: float = 1.0,
               shape: tuple[int, int] | None = None) -> np.ndarray:
        """Coronal brightness (arbitrary linear units) for a Sun at (sun_x, sun_y)."""
        h, w = shape or (self.size, self.size)
        rng = np.random.default_rng(self.seed)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
        dx, dy = (xx - sun_x) / scale, (yy - sun_y) / scale
        a = math.radians(angle_deg)
        dx, dy = dx * math.cos(a) + dy * math.sin(a), -dx * math.sin(a) + dy * math.cos(a)
        r = np.hypot(dx, dy) / self.sun_radius
        phi = np.arctan2(dy, dx)
        r = np.maximum(r, 0.98)
        base = 1e3 * r**-8 + 30 * r**-3
        ang = np.zeros_like(phi)
        for k in range(3, 40):
            ang += rng.normal() * (1.0 / k) * np.cos(k * phi + rng.uniform(0, 2 * math.pi))
        streamers = np.exp(0.6 * ang)
        fine = np.zeros_like(phi)
        for k in range(40, 160, 7):
            fine += rng.normal() * 0.03 * np.cos(k * phi + rng.uniform(0, 2 * math.pi) + 0.4 * r)
        return base * streamers * (1 + fine)

    def frame(self, exposure: float, sun_x: float, sun_y: float, moon_dx: float, moon_dy: float,
              angle_deg: float = 0.0, scale: float = 1.0, noise: bool = True, full_scale: float = 1.0,
              read_noise: float = 2e-4, gain: float = 2e4, seed: int = 0) -> tuple[np.ndarray, dict]:
        """One exposure. Brightness = corona * exposure, clipped at ``full_scale``."""
        img = self.corona(sun_x, sun_y, angle_deg, scale) * exposure
        mx, my = sun_x + moon_dx * scale, sun_y + moon_dy * scale
        moon_r = self.sun_radius * self.moon_ratio * scale
        yy, xx = np.mgrid[0:self.size, 0:self.size]
        rm = np.hypot(xx - mx, yy - my)
        disc = np.clip(rm - moon_r + 0.5, 0.0, 1.0)  # anti-aliased limb
        img = img * disc + 0.002 * exposure * (1 - disc)  # faint earthshine
        img = ndimage.gaussian_filter(img, 0.7)  # optics
        if noise:
            rng = np.random.default_rng(seed)
            img = rng.poisson(np.clip(img, 0, None) * gain) / gain + rng.normal(0, read_noise, img.shape)
        img = np.minimum(img, full_scale).astype(np.float32)
        return img, {"moon_x": mx, "moon_y": my, "moon_r": moon_r}
