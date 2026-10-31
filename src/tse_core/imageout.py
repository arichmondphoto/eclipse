"""Writing and reading the picture outputs (TIFF, PNG, JPG).

Every picture is written at the highest quality the format allows, tagged
300 dpi (a printing label only: the pixels are not changed) and tagged sRGB
so other programs show the colours consistently:

* TIFF: 16-bit, uncompressed.
* PNG: 16-bit, lossless (PNG has no lossy mode; the compression level only
  trades file size against writing time).
* JPG: 8-bit, which is all the JPEG format holds; quality 100 with no colour
  subsampling, dithered from 16 bits so smooth gradients (the outer corona)
  do not band.

FITS files stay as they are: they hold the linear data for further processing.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path
from typing import Optional

import numpy as np

DPI = 300
FORMATS = {".tif": "tiff", ".tiff": "tiff", ".png": "png", ".jpg": "jpeg", ".jpeg": "jpeg"}
FILTER = "TIFF, 16-bit (*.tif);;PNG, 16-bit (*.png);;JPG, 8-bit (*.jpg)"  # for save dialogs


def format_of(path) -> str:
    ext = Path(path).suffix.lower()
    if ext not in FORMATS:
        raise ValueError(f"unsupported image type {ext!r}; use .tif, .png or .jpg")
    return FORMATS[ext]


def with_format(path, fmt: Optional[str]) -> Path:
    """``path`` with the extension for ``fmt`` ('tif', 'png' or 'jpg'), or unchanged when fmt is None."""
    p = Path(path)
    if not fmt:
        return p
    ext = {"tif": ".tif", "tiff": ".tif", "png": ".png", "jpg": ".jpg", "jpeg": ".jpg"}[fmt.lower()]
    return p.with_suffix(ext)


def _srgb_icc() -> Optional[bytes]:
    try:
        from PIL import ImageCms

        return ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    except Exception:
        return None


def to_uint16(img: np.ndarray) -> np.ndarray:
    """RGB, or RGBA when the image has a fourth (alpha) channel."""
    a = np.asarray(img, dtype=np.float32)
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, axis=2)
    return (np.clip(a[..., :4], 0.0, 1.0) * 65535.0 + 0.5).astype(np.uint16)


def to_uint8_dithered(img: np.ndarray, seed: int = 0) -> np.ndarray:
    """8-bit with triangular dither of one step: no banding in smooth gradients, no visible grain."""
    a = np.asarray(img, dtype=np.float32)
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, axis=2)
    rng = np.random.default_rng(seed)
    noise = rng.random(a.shape[:2] + (3,), dtype=np.float32) - rng.random(a.shape[:2] + (3,), dtype=np.float32)
    return np.clip(np.clip(a[..., :3], 0.0, 1.0) * 255.0 + noise + 0.5, 0, 255).astype(np.uint8)


def write_image(path, img: np.ndarray, dpi: int = DPI) -> Path:
    """Write a display image (float, 0..1, (h, w), (h, w, 3), or (h, w, 4) with transparency) in the
    format its extension names. JPG has no transparency: an RGBA image is flattened onto black."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fmt = format_of(path)
    icc = _srgb_icc()
    if fmt == "tiff":
        import tifffile

        extratags = [(34675, "B", len(icc), icc, True)] if icc else []
        data = to_uint16(img)
        extra = {"extrasamples": ["unassalpha"]} if data.shape[2] == 4 else {}
        tifffile.imwrite(path, data, photometric="rgb", compression=None,
                         resolution=(dpi, dpi), resolutionunit="INCH", extratags=extratags, **extra)
    elif fmt == "png":
        _write_png16(path, to_uint16(img), dpi)
    else:
        from PIL import Image

        a = np.asarray(img, dtype=np.float32)
        if a.ndim == 3 and a.shape[2] == 4:
            a = a[..., :3] * a[..., 3:4]  # flatten onto black
        im = Image.fromarray(to_uint8_dithered(a), "RGB")
        kw = {"quality": 100, "subsampling": 0, "optimize": False, "dpi": (dpi, dpi)}
        if icc:
            kw["icc_profile"] = icc
        im.save(path, "JPEG", **kw)
    return path


def _chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def _write_png16(path: Path, data: np.ndarray, dpi: int) -> None:
    """16-bit RGB PNG (no row filters, so it can also be read back quickly), with pHYs and sRGB chunks."""
    h, w, c = data.shape
    raw = np.empty((h, 1 + w * 2 * c), np.uint8)
    raw[:, 0] = 0  # filter type None
    raw[:, 1:] = data.astype(">u2").view(np.uint8).reshape(h, w * 2 * c)
    ppm = int(round(dpi / 0.0254))
    png = b"\x89PNG\r\n\x1a\n"
    png += _chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 16, 6 if c == 4 else 2, 0, 0, 0))
    png += _chunk(b"sRGB", b"\x00")  # perceptual rendering intent
    png += _chunk(b"pHYs", struct.pack(">IIB", ppm, ppm, 1))
    png += _chunk(b"IDAT", zlib.compress(raw.tobytes(), 6))
    png += _chunk(b"IEND", b"")
    path.write_bytes(png)


def _read_png16_fast(path) -> Optional[np.ndarray]:
    """Read a 16-bit RGB PNG written without row filters (as ours are); None for anything else."""
    data = Path(path).read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    pos, idat, w, h = 8, [], 0, 0
    while pos < len(data):
        n = struct.unpack(">I", data[pos : pos + 4])[0]
        kind = data[pos + 4 : pos + 8]
        body = data[pos + 8 : pos + 8 + n]
        if kind == b"IHDR":
            w, h, depth, ctype, _c, _f, inter = struct.unpack(">IIBBBBB", body)
            if depth != 16 or ctype not in (2, 6) or inter != 0:
                return None
            chans = 4 if ctype == 6 else 3
        elif kind == b"IDAT":
            idat.append(body)
        pos += 12 + n
    raw = np.frombuffer(zlib.decompress(b"".join(idat)), np.uint8).reshape(h, 1 + w * 2 * chans)
    if raw[:, 0].any():
        return None  # filtered rows: leave it to Pillow
    return raw[:, 1:].copy().view(">u2").reshape(h, w, chans).astype(np.uint16)


def read_image(path, step: int = 1, alpha: bool = False) -> np.ndarray:
    """A TIFF, PNG or JPG as float RGB in 0..1 (every ``step``-th pixel). 16-bit files keep 16 bits,
    except PNGs from other programs, which are read at 8 bits. With ``alpha``, RGBA (opaque where
    the file has no transparency); otherwise transparency is ignored."""
    path = str(path)
    fmt = format_of(path)
    data = None
    if fmt == "tiff":
        import tifffile

        data = tifffile.imread(path)
    elif fmt == "png":
        data = _read_png16_fast(path)
    if data is None:
        from PIL import Image

        data = np.asarray(Image.open(path).convert("RGBA" if alpha else "RGB"))
    data = data[::step, ::step]
    if data.ndim == 2:
        data = np.repeat(data[..., None], 3, axis=2)
    scale = 65535.0 if data.dtype == np.uint16 else 255.0
    out = data.astype(np.float32) / scale
    if not alpha:
        return out[..., :3]
    if out.shape[2] == 3:
        out = np.concatenate([out, np.ones(out.shape[:2] + (1,), np.float32)], axis=2)
    return out[..., :4]


def image_size(path) -> tuple[int, int]:
    """(width, height) without reading the pixels where possible."""
    fmt = format_of(path)
    if fmt == "tiff":
        import tifffile

        with tifffile.TiffFile(path) as t:
            h, w = t.pages[0].shape[:2]
        return w, h
    from PIL import Image

    with Image.open(path) as im:
        return im.size
