"""Zoom and pan for an image shown in a view (no Qt here: the GUI draws what this works out).

Coordinates: image pixels (x right, y down) and view pixels. ``zoom`` is view pixels per image
pixel; ``None`` means fit the whole image in the view. ``cx, cy`` is the image point at the
middle of the view.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

MAX_ZOOM = 16.0


@dataclass
class ViewZoom:
    img_w: int = 0
    img_h: int = 0
    view_w: int = 1
    view_h: int = 1
    zoom: Optional[float] = None
    cx: float = 0.0
    cy: float = 0.0

    # -- state -------------------------------------------------------------------------------

    def set_image(self, w: int, h: int) -> None:
        """A new image: keeps the zoom and place if it is the same size (a refreshed preview)."""
        if (w, h) != (self.img_w, self.img_h):
            self.img_w, self.img_h = int(w), int(h)
            self.fit()

    def set_view(self, w: int, h: int) -> None:
        self.view_w, self.view_h = max(int(w), 1), max(int(h), 1)
        self._clamp()

    def fit(self) -> None:
        self.zoom = None
        self.cx, self.cy = self.img_w / 2.0, self.img_h / 2.0

    def fit_scale(self) -> float:
        if self.img_w <= 0 or self.img_h <= 0:
            return 1.0
        return min(self.view_w / self.img_w, self.view_h / self.img_h)

    def scale(self) -> float:
        return self.fit_scale() if self.zoom is None else self.zoom

    def label(self) -> str:
        return "Fit" if self.zoom is None else f"{self.zoom * 100:.0f}%"

    # -- conversions -------------------------------------------------------------------------

    def to_image(self, vx: float, vy: float) -> tuple[float, float]:
        s = self.scale()
        return self.cx + (vx - self.view_w / 2.0) / s, self.cy + (vy - self.view_h / 2.0) / s

    def to_view(self, ix: float, iy: float) -> tuple[float, float]:
        """Image pixel coordinates to view coordinates (the reverse of :meth:`to_image`)."""
        s = self.scale()
        return self.view_w / 2.0 + (ix - self.cx) * s, self.view_h / 2.0 + (iy - self.cy) * s

    def box_from_drag(self, vx0: float, vy0: float, vx1: float, vy1: float) -> Optional[tuple[int, int, int, int]]:
        """A box dragged in the view as (x0, y0, x1, y1) whole image pixels inside the image (x1, y1
        exclusive); None for a drag under 4 view px either way (a click) or one that misses the image."""
        if abs(vx1 - vx0) < 4 or abs(vy1 - vy0) < 4:
            return None
        ax, ay = self.to_image(vx0, vy0)
        bx, by = self.to_image(vx1, vy1)
        x0, x1 = sorted((ax, bx))
        y0, y1 = sorted((ay, by))
        x0, y0 = max(0, int(x0)), max(0, int(y0))
        x1, y1 = min(self.img_w, int(x1 + 0.999)), min(self.img_h, int(y1 + 0.999))
        if x1 - x0 < 1 or y1 - y0 < 1:
            return None
        return x0, y0, x1, y1

    def rects(self):
        """(target, source): where to draw (view px) and which part of the image (image px),
        each as (x, y, w, h); None when there is no image."""
        if self.img_w <= 0 or self.img_h <= 0:
            return None
        s = self.scale()
        x0, y0 = self.cx - self.view_w / 2.0 / s, self.cy - self.view_h / 2.0 / s
        sx0, sy0 = max(x0, 0.0), max(y0, 0.0)
        sx1 = min(x0 + self.view_w / s, float(self.img_w))
        sy1 = min(y0 + self.view_h / s, float(self.img_h))
        if sx1 <= sx0 or sy1 <= sy0:
            return None
        target = ((sx0 - x0) * s, (sy0 - y0) * s, (sx1 - sx0) * s, (sy1 - sy0) * s)
        return target, (sx0, sy0, sx1 - sx0, sy1 - sy0)

    # -- actions -----------------------------------------------------------------------------

    def zoom_at(self, factor: float, vx: Optional[float] = None, vy: Optional[float] = None) -> None:
        """Zoom by ``factor`` keeping the image point under (vx, vy) (default: the middle) still."""
        if self.img_w <= 0:
            return
        vx = self.view_w / 2.0 if vx is None else vx
        vy = self.view_h / 2.0 if vy is None else vy
        px, py = self.to_image(vx, vy)
        fit = self.fit_scale()
        s1 = min(max(self.scale() * factor, fit), MAX_ZOOM)
        if s1 <= fit * 1.0001:
            self.fit()
            return
        self.zoom = s1
        self.cx = px - (vx - self.view_w / 2.0) / s1
        self.cy = py - (vy - self.view_h / 2.0) / s1
        self._clamp()

    def set_zoom(self, zoom: float, vx: Optional[float] = None, vy: Optional[float] = None) -> None:
        self.zoom_at(zoom / self.scale(), vx, vy)

    def toggle(self, vx: float, vy: float) -> None:
        """Double-click: from fit to 100% (or twice the fit, if the image is smaller than the
        view) at the clicked point; otherwise back to fit."""
        if self.zoom is None:
            self.set_zoom(max(1.0, 2.0 * self.fit_scale()), vx, vy)
        else:
            self.fit()

    def pan(self, dx: float, dy: float) -> None:
        """Drag by (dx, dy) view pixels: the image follows the mouse."""
        if self.zoom is None:
            return
        s = self.scale()
        self.cx -= dx / s
        self.cy -= dy / s
        self._clamp()

    def _clamp(self) -> None:
        if self.zoom is None:
            self.cx, self.cy = self.img_w / 2.0, self.img_h / 2.0
            return
        s = self.scale()
        for attr, size, view in (("cx", self.img_w, self.view_w), ("cy", self.img_h, self.view_h)):
            half = view / 2.0 / s
            if size <= 2 * half:
                setattr(self, attr, size / 2.0)
            else:
                setattr(self, attr, min(max(getattr(self, attr), half), size - half))
