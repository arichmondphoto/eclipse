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
| 2 | Geometry (lunar limb, Moon track, Sun centre) and phase correlation alignment | **done** (command line) |
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
tse flat-profile FLATS                       # fit and save a vignetting profile from your own flats
tse inventory RAW --profile my_profile.json  # use one profile for every frame
tse frame DSC_4100.NEF                       # show everything read from one file
tse eclipse --date 2024-04-08 --lat 36.9422 --lon -90.3343 --utc-offset -5   # contacts and duration
tse align eclipse2024.json --lat 36.9422 --lon -90.3343 --elev 110           # align and find the Sun
```

## Alignment (Phase 2)

`tse align` loads every light frame as 2x2 superpixels (no demosaicing),
applies the calibration plan, and then:

1. **Finds the Moon** in each frame: the limb is located along 720 radial
   profiles and fitted with a robust circle (prominences and valleys rejected).
2. **Aligns on the corona** with Druckmüller's modified phase correlation:
   a tangential high-pass filter about the Moon, a soft mask (Moon, edges,
   saturation, stars, hot pixels), damped phase correlation with a Gaussian
   low-pass, and a sub-pixel peak centroid. Two changes from the paper make
   it work across a 12-stop bracket:
   * the correlation uses *relative* tangential contrast (high-pass divided
     by local brightness), which is the same in any exposure and stops the
     bright inner corona from locking the result onto the Moon's motion;
   * frames are aligned against a reference that grows from the middle
     exposure outwards, then every frame against all the others, and the
     whole set is re-anchored to the middle exposure at the end.
3. **Field rotation** is measured against the middle exposure and modelled
   as a smooth drift over time; it is applied only when statistically
   significant. Use `--rotation per-frame` or `--any-angle` for mounts that
   do not track.
4. **Finds the Sun's centre** (Druckmüllerová et al. 2011, appendix): the
   Moon's track across the aligned frames is fitted to the eclipse
   geometry computed for your site from NASA Besselian elements. This also
   gives the image scale and where celestial north points. If the camera
   clock was wrong, the clock error is bounded by requiring every frame to
   fall inside totality and the remaining along-track uncertainty is
   reported; pass `--clock-error` if you know it.

Results go to `<project>_align.json`: a full-resolution 2x3 transform per
frame, the Moon in each frame, and the Sun's centre and radius.

Measured accuracy: about 0.1 px (full resolution) on copies of a real D850
frame up to four stops apart, and 0.1 px on synthetic brackets spanning 8
stops with a moving Moon. A 37-frame bracket takes several minutes.

Bundled Besselian elements: 2017 Aug 21, 2024 Apr 08, 2026 Aug 12 and
2027 Aug 02 (reduced precision). Add others to
`src/tse_core/data/besselian.json` or pass `--elements`.

`inventory` prints the frame table, the detected brackets (step size, range,
missing shutter speeds), and the calibration plan: which vignetting source
each frame uses, how darks and bias are matched, and the corner falloff.

Contact times are compared with the camera's clock. If the clock was wrong,
give the correction with `--clock-offset` (added to camera time, for example
`+1:04:08`) and enter the contact times in true time. Bracket detection and
Moon tracking only use time differences, so a wrong clock never affects them.

## Calibration rules

* **Vignetting**, first match wins and only one is ever applied:
  1. a profile you name with `--profile`;
  2. your own flats (`--flats`), used directly when they match the frame's
     aperture and focal length;
  3. a saved own-flats profile for the same lens, within 5% focal length
     and 1/3 stop;
  4. the lensfun profile;
  5. none, with a warning.

  Profiles of either kind can be written as a synthetic FITS master flat
  for Siril. The eclipse is assumed focused at infinity.
* **Darks:** matched by exposure time and ISO; without a match, hot pixels
  are found from the light frames.
* **Bias:** matched by ISO; without bias frames, each file's black level is
  used as a synthetic bias.

## Own-flats profiles

When lensfun has no vignetting data for your lens, shoot flats and fit a
profile once:

1. Same lens, focal length, aperture and focus (infinity) as the eclipse
   frames; any ISO; any camera with the same sensor size.
2. An even light source filling the frame: a light panel, a white screen,
   or a twilight sky through a white T-shirt. Expose so the histogram peak
   is near the middle. 20 to 30 frames.
3. `tse flat-profile FLATS` fits a smooth, radially symmetric profile per
   colour channel and saves it to your profile library
   (`~/.solar-eclipse-toolkit/profiles`, or `TSE_PROFILE_DIR`).

The profile keeps only the lens's falloff: dust in the flats is rejected,
and a light source that is brighter on one side is fitted as a tilt and
removed. Because the profile is measured on the sensor in millimetres, it
corrects any camera with the same lens, so one set of flats serves both a
D800 and a D850. `tse inventory` then picks it up automatically. A FITS
master flat stacked in Siril can be used instead of RAW flats.

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
