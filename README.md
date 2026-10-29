# Solar Eclipse Toolkit

Processing for DSLR photographs of total solar eclipses: alignment on the
corona, linear HDR stacking and coronal enhancement, built to run as a
script inside [Siril](https://siril.org) 1.4 or later.

The methods come from Druckmüller (2009, phase correlation alignment),
Druckmüller, Rušin & Minarovjech (2006, Moon-aware HDR composition and
adaptive filtering) and Druckmüllerová, Morgan & Habbal (2011, the Fourier
normalizing-radial-graded filter). See the development plan for details.

Nothing is tied to one eclipse, camera or bracket: every setting is read
from each frame's metadata.

## Status

| Phase | Scope | State |
| --- | --- | --- |
| 1 | RAW inventory, bracket detection, lensfun lookup, calibration planning, project files | **done** |
| 2 | Geometry (lunar limb, Moon track, Sun centre) and phase correlation alignment | next |
| 3 | Moon-aware linear HDR stacking | planned |
| 4 | NRGF, FNRGF, FNRGF-N, adaptive kernel filter | planned |
| 5 | Full GUI and packaging | planned |

## Supported files

Nikon NEF and NRW, Canon CR2 and CR3, Sony ARW, and DNG. Sony users should
shoot uncompressed or lossless RAW.

Metadata readers, used in this order and each filling gaps left by the one
before:

1. **exiftool**, if installed (optional). Gives full lens names for every
   brand. Set `TSE_EXIFTOOL` if it is not on your `PATH`.
2. **exifread** (required, pure Python). Reads everything the pipeline
   needs from NEF, CR2, CR3, ARW and DNG.
3. **rawpy** (strongly recommended). Exact black level, white level, image
   size and CFA pattern from LibRaw, the same library Siril uses.

## Install

From the repository folder:

```bash
pip install -e ".[all]"      # exifread, numpy, rawpy, PyQt6
tse doctor                   # shows which readers are available
tse lensfun-update           # downloads the lensfun lens database once
```

If the download fails with a certificate error, run `pip install --upgrade certifi`
and try again (on macOS you can also run *Install Certificates.command* in the
Python folder in Applications). Or download
<https://github.com/lensfun/lensfun/archive/refs/heads/master.zip> in a browser
and install it offline:

```bash
tse lensfun-update --from ~/Downloads/lensfun-master.zip
```

### In Siril

1. In Siril, open **Preferences → Scripts** and add this repository's
   `siril` folder to the script paths.
2. Run **Scripts → SolarEclipseToolkit**. On first run Siril installs
   PyQt6, exifread, numpy and rawpy into its own Python environment.

The script finds the core package in `../src`, so keep the folders together.

## Command line

```bash
tse inventory "D:/Eclipse 2024/RAW" --project eclipse2024.json --csv frames.csv
tse inventory RAW --c2 2024-04-08T13:47:30 --c3 2024-04-08T13:51:45   # mark frames outside totality
tse inventory RAW --flats FLATS --darks DARKS --bias BIAS              # optional calibration frames
tse inventory RAW --lens "Nikon AF-S Nikkor 80-400mm f/4.5-5.6G ED VR" # force a lens profile
tse lens "AF-S Nikkor 80-400mm f/4.5-5.6G ED VR" --focal 400 --aperture 8
```

`inventory` prints the frame table, the detected brackets (step size, range,
missing shutter speeds), and the calibration plan: which vignetting source
each frame uses, how darks and bias are matched, and the corner falloff.

Contact times are in the camera's clock, which may differ from true time.

## Calibration rules

* **Vignetting:** your own master flat if one matches the frame's aperture
  and focal length, otherwise the lensfun profile, otherwise none. Only one
  is ever applied. The lensfun profile can be written as a synthetic FITS
  master flat for Siril. The eclipse is assumed focused at infinity.
* **Darks:** matched by exposure time and ISO; without a match, hot pixels
  are found from the light frames.
* **Bias:** matched by ISO; without bias frames, each file's black level is
  used as a synthetic bias.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

Set `TSE_SAMPLE_RAWS` to a folder of sample RAW files to also test the
metadata readers on real files. Development uses only the 2024 data; the
2017 set is held back as a blind test of the finished tool.

## Licence

MIT, see `LICENSE`.

The lensfun database is not bundled; it is downloaded from the lensfun
project (CC BY-SA 3.0). `tests/data/lensfun_subset.xml` is a short excerpt
used only by the tests.
