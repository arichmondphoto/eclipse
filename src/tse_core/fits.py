"""Minimal FITS reader and writer for single-image 32-bit float files.

Enough to write synthetic flats and intermediate products that Siril can
open, without requiring astropy. Images are stored as (height, width) or
(channels, height, width) arrays, top row first, which is how Siril reads
FITS when "ROWORDER=TOP-DOWN" is set.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import numpy as np

_BLOCK = 2880


def _card(key: str, value: Any = None, comment: str = "") -> bytes:
    if value is None:
        text = f"{key:<8}"
    else:
        if isinstance(value, bool):
            val = "T" if value else "F"
            text = f"{key:<8}= {val:>20}"
        elif isinstance(value, (int, np.integer)):
            text = f"{key:<8}= {int(value):>20}"
        elif isinstance(value, (float, np.floating)):
            text = f"{key:<8}= {float(value):>20.10G}"
        else:
            sval = str(value).replace("'", "''")[:68]
            text = f"{key:<8}= '{sval:<8}'"
        if comment:
            text += f" / {comment}"
    return text[:80].ljust(80).encode("ascii", "replace")


def write_fits(path: str | os.PathLike, data: np.ndarray, header: Optional[dict[str, Any]] = None) -> None:
    arr = np.asarray(data, dtype=">f4")
    if arr.ndim not in (2, 3):
        raise ValueError("FITS image must be 2-D or 3-D")
    cards = [_card("SIMPLE", True), _card("BITPIX", -32), _card("NAXIS", arr.ndim)]
    for i, n in enumerate(reversed(arr.shape), start=1):
        cards.append(_card(f"NAXIS{i}", int(n)))
    cards.append(_card("ROWORDER", "TOP-DOWN"))
    for key, value in (header or {}).items():
        cards.append(_card(key.upper()[:8], value))
    cards.append(_card("END"))
    head = b"".join(cards)
    head += b" " * (-len(head) % _BLOCK)
    body = arr.tobytes()
    body += b"\0" * (-len(body) % _BLOCK)
    with open(path, "wb") as fh:
        fh.write(head)
        fh.write(body)


def read_fits_header(path: str | os.PathLike) -> dict[str, Any]:
    """The header only (reads the first blocks, not the image)."""
    header: dict[str, Any] = {}
    with open(path, "rb") as fh:
        while True:
            block = fh.read(_BLOCK)
            if not block:
                return header
            for i in range(0, len(block), 80):
                card = block[i:i + 80].decode("ascii", "replace")
                key = card[:8].strip()
                if key == "END":
                    return header
                if card[8:10] == "= ":
                    value = card[10:].split(" /")[0].strip()
                    if value.startswith("'"):
                        header[key] = value.strip("'").rstrip()
                        continue
                    for conv in (int, float):
                        try:
                            header[key] = conv(value)
                            break
                        except ValueError:
                            pass
                    else:
                        header[key] = value == "T" if value in ("T", "F") else value


def read_fits(path: str | os.PathLike) -> tuple[np.ndarray, dict[str, Any]]:
    with open(path, "rb") as fh:
        raw = fh.read()
    header: dict[str, Any] = {}
    pos = 0
    while True:
        card = raw[pos : pos + 80].decode("ascii", "replace")
        pos += 80
        key = card[:8].strip()
        if key == "END":
            break
        if card[8:10] == "= ":
            value = card[10:].split(" /")[0].strip()
            if value.startswith("'"):
                value = value.strip("'").rstrip()
            elif value in ("T", "F"):
                value = value == "T"
            else:
                try:
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        pass
            header[key] = value
    pos += -pos % _BLOCK
    shape = tuple(int(header[f"NAXIS{i}"]) for i in range(int(header["NAXIS"]), 0, -1))
    dtype = {8: ">u1", 16: ">i2", 32: ">i4", -32: ">f4", -64: ">f8"}[int(header["BITPIX"])]
    count = int(np.prod(shape))
    data = np.frombuffer(raw, dtype=dtype, count=count, offset=pos).reshape(shape).astype(np.float32)
    data = data * float(header.get("BSCALE", 1.0)) + float(header.get("BZERO", 0.0))
    if header.get("ROWORDER", "BOTTOM-UP") != "TOP-DOWN":
        data = data[..., ::-1, :]
    return data, header
