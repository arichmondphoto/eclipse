"""Read the lensfun database and build vignetting corrections from it.

This is a small, dependency-free reader for lensfun's XML files (database
version 1 or 2). It reproduces the parts of the lensfun library this
toolkit needs:

* camera lookup (for the crop factor),
* lens matching from EXIF lens names and lens specifications,
* vignetting interpolation (inverse-distance weighting with power 3.5 over
  focal length, 4/aperture and 0.1/distance, exactly as lensfun does), and
* the Pablo d'Angelo ("pa") vignetting model, in which the observed image
  is the true image multiplied by ``1 + k1 r^2 + k2 r^4 + k3 r^6`` and
  ``r = 1`` is the corner of the calibration camera's frame.

The database itself is not bundled. It is found in the usual install
locations, in ``TSE_LENSFUN_DB``, or downloaded once with
:func:`download_database`.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import ssl
import urllib.request
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

FULL_FRAME_HALF_DIAGONAL_MM = math.hypot(36.0, 24.0) / 2.0
CACHE_DIR = Path(os.environ.get("TSE_CACHE_DIR", Path.home() / ".cache" / "solar-eclipse-toolkit"))
GITHUB_DB_API = "https://api.github.com/repos/lensfun/lensfun/contents/data/db"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class VignettingCalibration:
    focal: float
    aperture: float
    distance: float
    k1: float
    k2: float
    k3: float


@dataclass
class Camera:
    maker: str
    model: str
    variant: str
    mount: str
    crop_factor: float


@dataclass
class Lens:
    maker: str
    model: str
    names: list[str]
    mounts: list[str]
    crop_factor: float
    vignetting: list[VignettingCalibration] = field(default_factory=list)
    vignetting_model: str = ""
    source_file: str = ""

    @property
    def focal_range(self) -> Optional[tuple[float, float]]:
        return parse_focal_range(self.model)

    @property
    def aperture_range(self) -> Optional[tuple[float, float]]:
        return parse_aperture_range(self.model)

    @property
    def has_vignetting(self) -> bool:
        return bool(self.vignetting) and self.vignetting_model == "pa"

    def label(self) -> str:
        return f"{self.maker} {self.model}".strip() if self.maker and not self.model.lower().startswith(self.maker.lower()) else self.model

    # ------------------------------------------------------------ vignetting
    def vignetting_terms(
        self, focal: float, aperture: float, distance: float = 1000.0
    ) -> Optional[tuple[float, float, float]]:
        """Interpolated (k1, k2, k3) for these settings, or None if unknown.

        Same method as lensfun's ``lfLens::InterpolateVignetting``. Returns
        None when the nearest calibration is too far away to trust.
        """
        if not self.has_vignetting:
            return None
        fr = self.focal_range or (min(c.focal for c in self.vignetting), max(c.focal for c in self.vignetting))
        fmin, fmax = fr
        span = (fmax - fmin) or 1.0
        distance = distance if distance and distance > 0 else 1000.0

        def dist(c: VignettingCalibration) -> float:
            df = (focal - fmin) / span - (c.focal - fmin) / span
            da = 4.0 / aperture - 4.0 / c.aperture
            dd = 0.1 / distance - 0.1 / c.distance
            return math.sqrt(df * df + da * da + dd * dd)

        weights, terms, nearest = 0.0, np.zeros(3), float("inf")
        for c in self.vignetting:
            d = dist(c)
            if d < 1e-4:
                return (c.k1, c.k2, c.k3)
            nearest = min(nearest, d)
            w = 1.0 / d**3.5
            weights += w
            terms += w * np.array([c.k1, c.k2, c.k3])
        if nearest > 1.0 or weights == 0:
            return None
        k = terms / weights
        return (float(k[0]), float(k[1]), float(k[2]))


@dataclass
class LensMatch:
    lens: Lens
    score: float
    reasons: list[str]


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------


class LensfunDatabase:
    def __init__(self, cameras: list[Camera], lenses: list[Lens], files: list[str]):
        self.cameras = cameras
        self.lenses = lenses
        self.files = files

    # ------------------------------------------------------------- loading
    @classmethod
    def load(cls, directories: Optional[Iterable[str | os.PathLike]] = None) -> "LensfunDatabase":
        dirs = [Path(d) for d in directories] if directories else default_database_dirs()
        cameras: list[Camera] = []
        lenses: list[Lens] = []
        files: list[str] = []
        seen_names: set[str] = set()
        for d in dirs:
            if not d.is_dir():
                continue
            for path in sorted(d.glob("*.xml")):
                if path.name in seen_names:  # earlier (newer) directories win
                    continue
                try:
                    c, l = parse_database_file(path)
                except ET.ParseError:
                    continue
                seen_names.add(path.name)
                files.append(str(path))
                cameras.extend(c)
                lenses.extend(l)
        return cls(cameras, lenses, files)

    @property
    def empty(self) -> bool:
        return not self.lenses

    # ------------------------------------------------------------- cameras
    def find_camera(self, make: str, model: str) -> Optional[Camera]:
        make_n, model_n = _norm(make), _norm(model)
        best, best_score = None, 0
        for cam in self.cameras:
            cm = _norm(cam.model)
            score = 0
            if cm == model_n or _norm(f"{cam.maker} {cam.model}") == model_n:
                score = 3
            elif model_n and (model_n.endswith(cm) or cm.endswith(model_n)):
                score = 2
            if score and make_n and _norm(cam.maker).split(" ")[0] in make_n:
                score += 1
            if score > best_score:
                best, best_score = cam, score
        return best

    # --------------------------------------------------------------- lenses
    def match_lens(
        self,
        name: str = "",
        spec: Optional[list[float]] = None,
        focal: Optional[float] = None,
        maker: str = "",
        limit: int = 5,
    ) -> list[LensMatch]:
        """Rank database lenses against what the EXIF says about the lens."""
        name_range = parse_focal_range(name)
        name_ap = parse_aperture_range(name)
        spec_range = (spec[0], spec[1]) if spec and spec[0] else None
        spec_ap = (spec[2], spec[3]) if spec and spec[2] else None
        want_range = name_range or spec_range
        want_ap = name_ap or spec_ap
        want_tokens = _tokens(name)
        maker_n = _norm(maker)

        matches: list[LensMatch] = []
        for lens in self.lenses:
            score, reasons = 0.0, []
            lr = lens.focal_range
            if want_range and lr:
                if _close(lr[0], want_range[0]) and _close(lr[1], want_range[1]):
                    score += 4
                    reasons.append("focal range matches")
                else:
                    continue
            elif focal and lr and not (lr[0] - 0.5 <= focal <= lr[1] + 0.5):
                continue
            la = lens.aperture_range
            if want_ap and la:
                if _close(la[0], want_ap[0], 0.04, 0.1) and (not want_ap[1] or _close(la[1], want_ap[1], 0.04, 0.1)):
                    score += 2
                    reasons.append("aperture matches")
                else:
                    score -= 2
            if want_tokens:
                lens_tokens = set().union(*(_tokens(n) for n in lens.names))
                overlap = len(want_tokens & lens_tokens) / max(1, len(want_tokens | lens_tokens))
                score += 4 * overlap
                if overlap:
                    reasons.append(f"name overlap {overlap:.0%}")
                # A G/D/E designation that disagrees is a different lens.
                for flag in ("g", "d", "e"):
                    if (flag in want_tokens) != (flag in lens_tokens) and (flag in want_tokens or flag in lens_tokens):
                        if {"g", "d", "e"} & want_tokens:
                            score -= 1.5
            if maker_n and _norm(lens.maker) and _norm(lens.maker).split(" ")[0] in maker_n:
                score += 1
            if lens.has_vignetting:
                score += 0.5
                reasons.append("has vignetting data")
            if score > 0:
                matches.append(LensMatch(lens, score, reasons))
        matches.sort(key=lambda m: m.score, reverse=True)
        return matches[:limit]

    def find_lens_by_name(self, name: str) -> Optional[Lens]:
        target = _norm(name)
        for lens in self.lenses:
            if any(_norm(n) == target for n in lens.names) or _norm(lens.label()) == target:
                return lens
        return None


def parse_database_file(path: str | os.PathLike) -> tuple[list[Camera], list[Lens]]:
    root = ET.parse(path).getroot()
    cameras, lenses = [], []
    for el in root.findall("camera"):
        cameras.append(
            Camera(
                maker=_text(el, "maker"),
                model=_text(el, "model"),
                variant=_text(el, "variant"),
                mount=_text(el, "mount"),
                crop_factor=float(_text(el, "cropfactor") or 1.0),
            )
        )
    for el in root.findall("lens"):
        names = [m.text.strip() for m in el.findall("model") if m.text]
        lens = Lens(
            maker=_text(el, "maker"),
            model=_text(el, "model"),
            names=names,
            mounts=[m.text.strip() for m in el.findall("mount") if m.text],
            crop_factor=float(_text(el, "cropfactor") or 1.0),
            source_file=Path(path).name,
        )
        for calib in el.findall("calibration"):
            for v in calib.findall("vignetting"):
                model = v.get("model", "")
                if not lens.vignetting_model:
                    lens.vignetting_model = model
                if model != lens.vignetting_model:
                    continue
                lens.vignetting.append(
                    VignettingCalibration(
                        focal=float(v.get("focal", 0)),
                        aperture=float(v.get("aperture", 0)),
                        distance=float(v.get("distance", 1000)),
                        k1=float(v.get("k1", 0)),
                        k2=float(v.get("k2", 0)),
                        k3=float(v.get("k3", 0)),
                    )
                )
        lenses.append(lens)
    return cameras, lenses


def default_database_dirs() -> list[Path]:
    dirs: list[Path] = []
    env = os.environ.get("TSE_LENSFUN_DB")
    if env:
        dirs += [Path(p) for p in env.split(os.pathsep) if p]
    dirs.append(CACHE_DIR / "lensfun")
    home = Path.home()
    for base in (
        home / ".local/share/lensfun/updates",
        home / ".local/share/lensfun",
        Path(os.environ.get("APPDATA", home)) / "lensfun",
        Path("/usr/local/share/lensfun"),
        Path("/usr/share/lensfun"),
        Path("/opt/homebrew/share/lensfun"),
        Path("/opt/local/share/lensfun"),
    ):
        dirs += [base / "version_2", base / "version_1", base]
    return dirs


def download_database(dest: Optional[str | os.PathLike] = None, timeout: float = 30.0) -> Path:
    """Download the current lensfun XML files from GitHub into the cache.

    Uses certifi's certificate list when it is installed, so it also works
    with Python installs that cannot see the system certificates (common
    with the python.org installer on macOS).
    """
    target = Path(dest) if dest else CACHE_DIR / "lensfun"
    target.mkdir(parents=True, exist_ok=True)
    ctx = _ssl_context()
    req = urllib.request.Request(GITHUB_DB_API, headers={"Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
        listing = json.load(resp)
    for item in listing:
        name = item.get("name", "")
        url = item.get("download_url")
        if not name.endswith(".xml") or not url:
            continue
        with urllib.request.urlopen(url, timeout=timeout, context=ctx) as resp:
            data = resp.read()
        ET.fromstring(data)  # refuse to save anything that is not valid XML
        (target / name).write_bytes(data)
    return target


def import_database(source: str | os.PathLike, dest: Optional[str | os.PathLike] = None) -> Path:
    """Install lensfun XML files from a folder or a downloaded .zip.

    Works offline: download the lensfun repository as a zip in a browser
    (github.com/lensfun/lensfun, Code, Download ZIP) and pass its path, or
    point at any folder holding the database's .xml files.
    """
    src = Path(source)
    target = Path(dest) if dest else CACHE_DIR / "lensfun"
    target.mkdir(parents=True, exist_ok=True)
    count = 0
    if src.is_file() and zipfile.is_zipfile(src):
        with zipfile.ZipFile(src) as zf:
            names = [n for n in zf.namelist() if n.endswith(".xml") and "/data/db/" in "/" + n]
            if not names:
                names = [n for n in zf.namelist() if n.endswith(".xml")]
            for n in names:
                data = zf.read(n)
                if _is_lensfun_xml(data):
                    (target / Path(n).name).write_bytes(data)
                    count += 1
    elif src.is_dir():
        db = src / "data" / "db"
        folder = db if db.is_dir() else src
        for p in folder.glob("*.xml"):
            data = p.read_bytes()
            if _is_lensfun_xml(data):
                shutil.copyfile(p, target / p.name)
                count += 1
    else:
        raise FileNotFoundError(f"{src} is not a folder or a .zip file")
    if count == 0:
        raise ValueError(f"no lensfun database files found in {src}")
    return target


def _is_lensfun_xml(data: bytes) -> bool:
    try:
        return ET.fromstring(data).tag == "lensdatabase"
    except ET.ParseError:
        return False


def _ssl_context() -> ssl.SSLContext:
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


# ---------------------------------------------------------------------------
# Vignetting model
# ---------------------------------------------------------------------------


def relative_illumination(r: np.ndarray | float, terms: tuple[float, float, float]) -> np.ndarray | float:
    """Observed / true brightness at normalised radius r (pa model)."""
    k1, k2, k3 = terms
    r2 = np.asarray(r, dtype=np.float64) ** 2
    return 1.0 + k1 * r2 + k2 * r2**2 + k3 * r2**3


def vignetting_map(
    width: int,
    height: int,
    terms: tuple[float, float, float],
    image_crop: float = 1.0,
    calib_crop: float = 1.0,
    dtype=np.float32,
    box: Optional[tuple[int, int, int, int]] = None,
) -> np.ndarray:
    """Relative illumination for every pixel of a width x height image.

    Divide a linear image by this map to correct vignetting. The radius is
    measured on the sensor in millimetres, so a calibration made on a
    different sensor size is rescaled correctly. ``box`` = (x0, y0, w, h)
    returns only that part of the map.
    """
    if box is None:
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
    else:
        bx, by, bw, bh = box
        yy, xx = np.mgrid[by : by + bh, bx : bx + bw].astype(np.float64)
    cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
    pixel_half_diag = math.hypot(width, height) / 2.0
    image_half_diag_mm = FULL_FRAME_HALF_DIAGONAL_MM / image_crop
    calib_half_diag_mm = FULL_FRAME_HALF_DIAGONAL_MM / calib_crop
    mm_per_pixel = image_half_diag_mm / pixel_half_diag
    r = np.hypot(xx - cx, yy - cy) * mm_per_pixel / calib_half_diag_mm
    return relative_illumination(r, terms).astype(dtype)


def corner_falloff(terms: tuple[float, float, float], image_crop: float = 1.0, calib_crop: float = 1.0) -> tuple[float, float]:
    """(relative brightness, stops) at the image corner."""
    r = calib_crop / image_crop
    value = float(relative_illumination(r, terms))
    return value, (math.log2(value) if value > 0 else float("-inf"))


# ---------------------------------------------------------------------------
# Name helpers
# ---------------------------------------------------------------------------

_FOCAL_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:-\s*(\d+(?:\.\d+)?))?\s*mm", re.I)
_APERTURE_RE = re.compile(r"(?:f\s*/?\s*|1:)(\d+(?:\.\d+)?)(?:\s*-\s*(\d+(?:\.\d+)?))?", re.I)


def parse_focal_range(text: str) -> Optional[tuple[float, float]]:
    m = _FOCAL_RE.search(text or "")
    if not m:
        return None
    a = float(m.group(1))
    b = float(m.group(2)) if m.group(2) else a
    return (a, b)


def parse_aperture_range(text: str) -> Optional[tuple[float, float]]:
    m = _APERTURE_RE.search(text or "")
    if not m:
        return None
    a = float(m.group(1))
    b = float(m.group(2)) if m.group(2) else a
    return (a, b)


def _tokens(text: str) -> set[str]:
    text = (text or "").lower()
    text = _FOCAL_RE.sub(" ", text)
    text = _APERTURE_RE.sub(" ", text)
    # "5.6G" -> "g": split designations glued to apertures.
    text = re.sub(r"(\d)([a-z])", r"\1 \2", text)
    raw = re.split(r"[^a-z0-9\-]+", text)
    out = set()
    for tok in raw:
        tok = tok.strip("-")
        if not tok or tok.replace(".", "").isdigit():
            continue
        out.add(tok)
        out.update(t for t in tok.split("-") if t)
    out -= {"lens", "zoom", "mm", "f"}
    return out


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower().replace("corporation", "").replace(",", " ")).strip()


def _close(a: float, b: float, rel: float = 0.02, abs_tol: float = 0.6) -> bool:
    return abs(a - b) <= max(abs_tol, rel * max(abs(a), abs(b)))


def _text(el: ET.Element, tag: str) -> str:
    """First untranslated child text (lensfun puts translations in lang=…)."""
    for child in el.findall(tag):
        if child.get("lang") is None and child.text:
            return child.text.strip()
    child = el.find(tag)
    return child.text.strip() if child is not None and child.text else ""
