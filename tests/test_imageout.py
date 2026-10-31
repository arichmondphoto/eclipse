"""Picture outputs: 16-bit TIFF and PNG, 8-bit dithered JPG, all 300 dpi and sRGB-tagged."""

import numpy as np
import pytest

from tse_core import imageout as IO


def _img(h=40, w=64):
    yy, xx = np.mgrid[0:h, 0:w]
    return np.stack([xx / w, yy / h, 0.5 + 0 * xx], axis=-1).astype(np.float32)


@pytest.mark.parametrize("ext", [".tif", ".png"])
def test_16bit_roundtrip_is_exact(tmp_path, ext):
    img = _img()
    p = IO.write_image(tmp_path / f"a{ext}", img)
    back = IO.read_image(p)
    assert back.shape == img.shape
    assert np.abs(back - img).max() <= 0.5 / 65535 + 1e-7
    assert IO.image_size(p) == (64, 40)
    assert IO.read_image(p, step=2).shape == (20, 32, 3)


def test_dpi_and_profile(tmp_path):
    import tifffile
    from PIL import Image

    img = _img()
    t = IO.write_image(tmp_path / "a.tif", img)
    with tifffile.TiffFile(t) as tf:
        tags = tf.pages[0].tags
        assert tags["XResolution"].value == (300, 1) and tags["ResolutionUnit"].value == 2
        assert tags["Compression"].value == 1  # none
        assert 34675 in tags  # ICC profile
    with Image.open(IO.write_image(tmp_path / "a.png", img)) as im:
        assert round(im.info["dpi"][0]) == 300
    j = IO.write_image(tmp_path / "a.jpg", img)
    with Image.open(j) as im:
        assert im.info["dpi"] == (300, 300)
        assert im.info.get("icc_profile")
        assert im.mode == "RGB"


def test_jpg_quality_and_dither(tmp_path):
    # A very smooth gradient: dithering keeps the average right, no banding steps.
    ramp = np.tile(np.linspace(0.2, 0.21, 512, dtype=np.float32), (64, 1))
    img = np.repeat(ramp[..., None], 3, axis=2)
    back = IO.read_image(IO.write_image(tmp_path / "r.jpg", img))
    cols = back.mean(axis=(0, 2))
    assert np.abs(cols - ramp[0]).max() < 0.004  # within about one 8-bit step, column by column
    assert np.all(np.diff(np.convolve(cols, np.ones(32) / 32, "valid")) > -0.0005)  # no staircase


def test_with_format_and_errors(tmp_path):
    assert IO.with_format("x/a.tif", "png").name == "a.png"
    assert IO.with_format("x/a.tif", None).name == "a.tif"
    with pytest.raises(ValueError):
        IO.write_image(tmp_path / "a.bmp", _img())


@pytest.mark.parametrize("ext", [".tif", ".png"])
def test_rgba_roundtrip(tmp_path, ext):
    img = _img()
    rgba = np.concatenate([img, np.linspace(0, 1, 64, dtype=np.float32)[None, :, None].repeat(40, 0)], axis=2)
    p = IO.write_image(tmp_path / f"a{ext}", rgba)
    back = IO.read_image(p, alpha=True)
    assert back.shape == (40, 64, 4)
    assert np.abs(back - rgba).max() <= 0.5 / 65535 + 1e-7
    assert IO.read_image(p).shape == (40, 64, 3)
    j = IO.read_image(IO.write_image(tmp_path / "a.jpg", rgba))
    assert j[:, 0].max() < 0.02  # alpha 0: black
