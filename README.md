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
| 3 | Moon-aware linear HDR stacking | **done** (command line) |
| 4 | NRGF, FNRGF, FNRGF-N, adaptive filter, natural layer, blend | **done** (command line) |
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

Downloads check certificates against your system's store (the same
certificates your browser uses, via the `truststore` package), then
certifi's list. That way they also work where an antivirus program, a
company network or a VPN inspects HTTPS traffic, which otherwise shows up as
"self-signed certificate in certificate chain". Certificate checking is
never switched off. If a download still fails, download the file in a
browser and use `--from`:

```bash
# https://github.com/lensfun/lensfun/archive/refs/heads/master.zip
tse lensfun-update --from ~/Downloads/lensfun-master.zip
# https://eclipse.gsfc.nasa.gov/eclipse_besselian_from_mysqldump2.csv
tse elements-update --from ~/Downloads/eclipse_besselian_from_mysqldump2.csv
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
tse frames eclipse2024.json --exclude DSC_9108.nef   # leave a frame out (--include to undo)
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

**Any eclipse, any site.** Elements from NASA's eclipse bulletins are
bundled for 2017, 2024, 2026 and 2027. For every other eclipse from 1600 to
2400, the tool downloads NASA's Five Millennium Canon once (about 6 MB) the
first time it needs it, or on request:

```bash
tse elements-update                    # download NASA's catalogue now
tse elements-update --from canon.csv   # or install a copy downloaded in a browser
```

**Any focal length.** The tangential filter length, mask softness and
analysis window follow the Sun's size in the image (tested from a 20 px to a
180 px solar radius); `--rho` and `--window` override them.

**Mirrored optics.** Camera lenses give a true image; telescopes with a star
diagonal give a mirror image. The Moon's track cannot tell the two apart,
and the choice moves the computed Sun centre by twice the closest approach
of the Sun's and Moon's centres: negligible near the central line, tens of
pixels near the edge of the path. Use `--mirrored` for mirrored optics; the
tool reports both answers and warns when the difference matters.

`inventory` prints the frame table, the detected brackets (step size, range,
missing shutter speeds), and the calibration plan: which vignetting source
each frame uses, how darks and bias are matched, and the corner falloff.

Contact times are compared with the camera's clock. If the clock was wrong,
give the correction with `--clock-offset` (added to camera time, for example
`+1:04:08`) and enter the contact times in true time. Bracket detection and
Moon tracking only use time differences, so a wrong clock never affects them.

## HDR stacking (Phase 3)

```bash
tse stack eclipse2024.json            # uses the alignment saved by 'tse align'
tse stack eclipse2024.json --square   # quicker: a square 10 solar radii wide
```

Every frame is read again at full resolution: black level subtracted per
channel (negative values kept), vignetting divided out per colour, isolated
hot pixels replaced, demosaiced with the linear Malvar-He-Cutler method in
32-bit floating point, and resampled onto the reference frame with the
alignment transform (bicubic).

**Brightness matching.** Shutter speeds are rarely exact. Each frame's
brightness relative to the reference, `value = gain * radiance + offset`,
is measured by robust straight-line fits between neighbouring exposures on
pixels well exposed in both frames and clear of the Moon. Frames more than
0.3 EV from their nominal exposure are reported.

**Merging.** Each pixel is the inverse-variance weighted mean of the frames'
radiance estimates. Weights fall to zero approaching saturation and inside
each frame's own Moon (plus 3 px), so the moving Moon never darkens the
inner corona. Inside the Moon as it was in the Moon frame (the alignment
reference unless `--moon-frame` says otherwise) the result is that frame's
data, which keeps the lunar limb sharp. Pixels that are saturated or covered
in every frame (rare with 1/3-stop brackets) take the shortest exposure that
shows them and are counted in the report.

Frames the Sun-centre fit flagged as off the Moon's track are skipped
unless `--include-rejected` is given.

**Outputs** go next to the project unless `-o` says otherwise. By default
they are the full frame, the same size as the RAW files and with the Sun
where it is in the reference frame, so they line up with your originals in
Photoshop and keep faint streamers out to the frame edges. Pixels near the
edges that no aligned frame covers are set to zero (reported).

For a quicker run, `--square` outputs only a square 10 solar radii wide
around the Sun (about a third of the pixels, so roughly three times faster
and smaller); `--radii N` sets the half-width in solar radii.

A full 45-megapixel frame needs about 3 GB of free memory at the peak.

| File | Contents |
| --- | --- |
| `<name>_hdr.fit` | linear 32-bit RGB in units of the reference exposure, camera colour; header has `SUNX/SUNY/SUNR`, `MOONX/MOONY/MOONR`, `PIXSCALE`, `NORTHANG` and the crop origin (0, 0 for the full frame) |
| `<name>_weight.fit` | sum of weights per pixel (where the data came from) |
| `<name>_preview.tif` | 16-bit asinh stretch with the camera white balance, for a quick look only |
| `<name>_stack.json` | gains, offsets and settings |

## Enhancement (Phase 4)

```bash
tse enhance eclipse2024.json          # uses the HDR stack saved by 'tse stack'
```

Writes one 16-bit RGB TIFF per layer, all the same size and lined up pixel
for pixel, ready to stack as layers in Photoshop, plus a suggested blend:

| Layer | What it shows |
| --- | --- |
| `natural` | log stretch with most of the radial fall-off removed (`--flatten`, default 0.8); the eye-view look |
| `nrgf` | normalizing-radial-graded filter (Morgan et al. 2006): each circle around the Sun normalized to the same mean and spread |
| `fnrgf` | Fourier NRGF (Druckmüllerová et al. 2011): mean and spread follow position angle too, so faint and bright regions at the same height get similar contrast |
| `fnrgf_n` | FNRGF-N: the same with noise compensation (`--noise-m`, default 0.7); the noise is estimated automatically from the stack's weight map |
| `adaptive` | multi-scale Gaussian normalization (Morgan & Druckmüller 2014) of the log image: fine structure at all scales and orientations, the published successor to the adaptive circular high-pass filter |
| `blend` | suggested mix, `--blend natural=0.4,fnrgf_n=0.3,adaptive=0.3` by default (`--blend none` to skip) |

The Sun's centre and radius come from the stack header, so the circles are
centred on the true Sun, not the Moon. Circles that leave a rectangular
frame are completed from further in, so the frame edge does not show as an
arc.

Colour: by default the inner corona (1.2-2.5 solar radii) is made neutral,
since K-corona light is scattered sunlight; `--white camera` uses the
camera's as-shot balance, `--saturation 0` gives monochrome. Colour is faded
to grey where the signal is buried in noise. The Moon is black (an
earthshine layer is planned).

FNRGF settings: `--order` (10), `--segments` (50), `--mean-att` and
`--std-att` (attenuation factors A0..A10 and S0..S10, default 1, 0.9, ..., 0).

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
