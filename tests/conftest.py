import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tse_core.frames import FrameMeta, to_float  # noqa: E402
from tse_core.lensfun import LensfunDatabase  # noqa: E402

DATA = Path(__file__).resolve().parent / "data"

# The 2024 bracket as shot: Nikon D850, ISO 100, f/8, 400 mm, 37 frames.
ECLIPSE_2024 = (
    "1/1600 1/1250 1/1000 1/800 1/640 1/500 1/400 1/320 1/250 1/200 1/160 1/125 1/100 "
    "1/80 1/60 1/50 1/40 1/30 1/25 1/20 1/15 1/13 1/10 1/8 1/6 1/5 1/4 1/3 1/2 0.6 0.8 1.0 "
    "1.3 1.6 2.5 3.0 4.0"
).split()


def make_frames(exposures, start=datetime(2024, 4, 8, 13, 47, 30), pause=1.2, iso=100, aperture=8.0,
                focal=400.0, first_index=0, **extra):
    frames, t = [], start
    for i, text in enumerate(exposures):
        e = to_float(text)
        frames.append(FrameMeta(
            path=f"/data/DSC_{first_index + i:04d}.NEF", file_name=f"DSC_{first_index + i:04d}.NEF",
            index=first_index + i, exposure_s=e, exposure_text=text, aperture_f=aperture, iso=iso,
            focal_length_mm=focal, timestamp=t.isoformat(), camera_make="NIKON CORPORATION",
            camera_model="NIKON D850", lens_model="AF-S Nikkor 80-400mm f/4.5-5.6G ED VR",
            lens_spec=[80.0, 400.0, 4.5, 5.6], black_level=400.0, white_level=15520.0,
            width=8288, height=5520, **extra))
        t += timedelta(seconds=e + pause)
    return frames


@pytest.fixture
def frames_2024():
    return make_frames(ECLIPSE_2024)


@pytest.fixture
def lens_db():
    return LensfunDatabase.load([DATA])


def sample_images_dir():
    """Optional real RAW samples (for example exiftool's t/images folder)."""
    path = os.environ.get("TSE_SAMPLE_RAWS")
    return Path(path) if path and Path(path).is_dir() else None
