# Totality - A Solar Eclipse Toolkit

Processing for DSLR photographs of total solar eclipses, from the RAW
files to finished pictures: alignment on the corona, linear HDR stacking,
coronal enhancement, an earthshine layer for the Moon and a prominence
layer. It runs as a script with its own window inside
[Siril](https://siril.org) 1.4 or later, or from the command line as
`totality`.

The methods come from Druckmüller (2009, phase correlation alignment),
Druckmüller, Rušin & Minarovjech (2006, Moon-aware HDR composition),
Druckmüllerová, Morgan & Habbal (2011, the Fourier normalizing-radial-graded
filter and the Sun's centre), Morgan, Habbal & Woo (2006, NRGF), Morgan &
Druckmüller (2014, multi-scale Gaussian normalization) and H. Druckmüllerová's
PhD thesis (2013, the adaptive circular high-pass filter).

Works on any total solar eclipse. Eclipse geometry comes from NASA's Besselian
elements for any eclipse from 1600 to 2400.

## Status

| Step | Scope | State |
| --- | --- | --- |
| 1 Frames | RAW inventory, bracket detection, lensfun or own-flats vignetting, calibration planning, project folders | done |
| 2 Align | Lunar limb, phase correlation alignment, field rotation, the Sun's centre | done |
| 3 Stack | Moon-aware linear HDR stacking with measured brightness matching | done |
| 4 Enhance | Natural, NRGF, FNRGF, FNRGF-N, Adaptive (MGN) and Adaptive (ACHF) layers; Soft Light / Overlay / Linear / Weighted average blend (Linear as each method's own publication combines it); full-resolution detail box; settings files | done |
| 5 Earthshine | Moon-aligned stack of the long exposures; glare removed; the Moon put into the HDR stack and stretched with the corona | done |
| 6 Prominences | The short exposures stacked round the Sun, to measure how soft the HDR stack's prominences are; those sharpened before the stretch | done (checked on the 2024 frames) |

Each step works in the Siril window and on the command line. Tested on the
2024 eclipse (Nikon D850, 400 mm); the 2017 set is kept back as a blind test.
Still to come: the blind test, and help for other cameras and operating
systems before sharing the tool more widely. Ideas kept for later, if needed
or requested (such as files from astronomy cameras like ZWO's), are in
[docs/FUTURE.md](docs/FUTURE.md).

## Workflow

1. **Frames:** make a project from the event folder (see *Project folders*),
   and enter the event once: the eclipse, your site, the camera's time zone
   and clock error. **Compute contacts** works out second and third contact
   and marks the frames taken outside totality.
2. **Align:** the frames are aligned and the Sun's centre found (from the
   event entered on the Frames tab).
3. **Stack:** the frames are merged into one linear 32-bit HDR image.
4. **Enhance:** tune the layers in a live preview, then write them full size.
5. **Earthshine:** stack the Moon, then write the final image: the Moon put
   into the HDR stack and stretched together with the corona.
6. **Prominences:** stack the short exposures, then write the final image
   with the prominences sharpened.

Then finish in Photoshop (or not: steps 5 and 6 write a finished picture).

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
pip install -e ".[all]"       # numpy, scipy, exifread, tifffile, Pillow, rawpy, PyQt6, ...
totality doctor               # shows which readers are available
totality lensfun-update       # downloads the lensfun lens database once
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
totality lensfun-update --from ~/Downloads/lensfun-master.zip
# https://eclipse.gsfc.nasa.gov/eclipse_besselian_from_mysqldump2.csv
totality elements-update --from ~/Downloads/eclipse_besselian_from_mysqldump2.csv
```

### In Siril

1. In Siril, open **Preferences → Scripts** and add this repository's
   `siril` folder to the script paths.
2. Run **Scripts → Totality** (the script is `siril/Totality.py`; the window
   is titled "Totality - A Solar Eclipse Toolkit", with the version and the
   commit). On first run Siril installs
   PyQt6, numpy, scipy, exifread, rawpy, tifffile, Pillow, certifi and
   truststore into its own Python environment.

The script finds the core package in `../src`, so keep the folders together.

To update, pull the repository (`git pull`) and run the script again. The
window title shows the version and the commit it is running (for example
`Totality - A Solar Eclipse Toolkit 0.1.0 (8e54fa8)`), so you can check an
update took effect.

The window has one tab per step, and each step saves into the project file:

| Tab | What you do | What you see |
| --- | --- | --- |
| 1 Frames | Read the RAW folder, check the frames and brackets. **Event:** the eclipse, the site (latitude, longitude, elevation), the camera time zone and clock error if known; **Compute contacts** fills in second and third contact. **Lens:** the focal length and f-number for frames that do not record them (a telescope; optional); the darks, flats and bias folders | Frame table (right-click to exclude a frame), the contacts and the check of your brackets against them, bracket report and the calibration frames found |
| 2 Align | Whether the optics mirror the image; the rotation model (drift, per-frame or none) and **Search any rotation and scale** (the event is shown, from the Frames tab) | Per-frame shift, rotation and peak strength; the Sun centre report |
| 3 Stack | Full frame or a square around the Sun; which frame's Moon to keep; **Calibration:** vignetting (automatic, master flat, profile from the flats, master flat + compare, lensfun, none) and the lensfun lens, whether the light on the flats was uneven, darks (off by default) with the hot-pixel fix when they are off, and bias | The stacked preview and each frame's measured brightness |
| 4 Enhance | **Save settings…** / **Load settings…**; **Layers to write** (all but Adaptive (ACHF) ticked: it is slow); **Blend:** each layer's strength (all 0 % by default) and mode, Soft Light, Overlay or Linear, and for Adaptive (MGN) and Adaptive (ACHF) also Weighted average; **Natural layer and colour:** Stretch (Log; or Gamma, the base of Morgan & Druckmüller 2014) with its Gamma (3.2) and black and white points (0 % and 100 %), Flatten, Highlight roll-off, Evenness, Midtone (all 0 by default), Saturation (1), White balance (corona), **Keep the sky** (off); **Adaptive layer (MGN):** Strength (0.7, as Morgan & Druckmüller 2014), Noise gate (off) and the size and strength of each of its six scales (1.25, 2.5, 5, 10, 20, 40 px at strength 1, as Morgan & Druckmüller 2014); **Adaptive layer (ACHF):** the size and strength of each of its six scales (0.121%, 0.242%, 0.726% and 2.42% of the Sun's radius at strength 1; 10% and 20% at 0), **Separate parts at** (2x) and **Contrast** (1); **FNRGF:** Order (10), Segments (50), Noise compensation (0.7) | A live preview on a reduced copy (about 1400 px) that updates as you change settings; Shift-drag on it for a **detail box** shown at full resolution on its own tab, exactly as written; then **Write full-size layers**. **Save settings…** and **Load settings…** keep every setting of the tab in a file (see [Settings files](#settings-files)) |
| 5 Earthshine | 1 Stack the Moon: frames within N stops, colour channels, frames off the Moon's track. 2 Lunar surface: glare smoothness, limb glare, glare round a bright prominence, maria by the limb, smoothing (overall and by the limb), edge-preserving smoothing, filled edge width, edge grain, brightness, natural colour; **Photoshop paste only (old way)**: kept glare and limb softening. 3 Final image: Blend or Natural (stretched with the Moon, using your Enhance settings) or paste into a file, lunar surface strength, blend mode when pasting, use the prominence stack | The Moon close up, then a preview of the final picture; **Write final image**, **Write Moon layer (for Photoshop)** |
| 6 Prominences | 1 Prominence stack: longest exposure (stops above the shortest), **Stack prominences**. 2 In the final image: sharpening, structure, prominence brightness, redness needed, chromosphere on or off, rim height (greyed out when the Moon hides the chromosphere) | The final image round the Sun at full size, exactly as it will be written; **Write final image** (as on the Earthshine tab) |

Every setting has hover help explaining what it does and when to change it.
Every picture view zooms: scroll (or **+** / **-**) to zoom about the pointer,
drag to move round, double-click for 100% at that point (again for fit),
**0** fit, **1** 100%; the zoom shows in the corner. A preview run again with
other settings keeps the zoom and place, so settings can be compared on the
same detail. Previews are reduced copies (about 1400 px), so 100% there is a
preview pixel, not a full-size one. On the Enhance tab, hold **Shift** and
drag on the whole picture to choose a **detail box**: the **Detail box (full
resolution)** tab then shows that part at full resolution, the very pixels
**Write full-size layers** will write (see [The detail box](#the-detail-box)).
Long steps run in the background: progress shows in the log under each tab
and **Cancel** stops at the next step. **Open … in Siril** loads a result
into Siril. **Project → New from folder…**, **Open…** and **Save** manage
the project file. The same steps run from the command line with
`totality`, using the same project file.

### Project folders

Keep each event in its own folder:

```
Eclipse 2024/
    lights/      the RAW frames from totality
    darks/       optional
    flats/       optional
    bias/        optional
```

**Project → New from folder…** in the GUI, or `totality inventory "Eclipse 2024"`
on the command line, reads the light frames from `lights/` and the
calibration frames from the other subfolders and saves the project file as
`Eclipse 2024/Eclipse 2024.json`. Opening that folder again reopens the
project. The steps add two folders of data files:

```
Eclipse 2024/
    Eclipse 2024.json   the project file (open this)
    fits/        every .fit file: the HDR stack and its weights, the
                 earthshine stacks, the vignetting correction, and
                 masters/ (the master bias, darks and flats)
    json/        every other .json file: the alignment, the stack's and
                 earthshine's details, the Enhance settings used
```

Files from an older project at the top of the folder are still found.
Wherever a picture is written (enhanced layers, earthshine and prominence
layers, final images), the GUI asks where to save, under what name and in
which format, starting in the project folder.

## Command line

The command is `totality` (run `pip install -e .` again after updating).

A whole event, from a project folder:

```bash
totality inventory "Eclipse 2024" --lat 36.9422 --lon -90.3343 --elev 110   # reads lights/, writes "Eclipse 2024/Eclipse 2024.json"
P="Eclipse 2024/Eclipse 2024.json"
totality align "$P"                                      # the site from the project
totality stack "$P"
totality enhance "$P" --blend natural=1,fnrgf_n=0.7,adaptive=0.1:overlay
totality earthshine "$P"
totality prominence-stack "$P" --brightness 0.3
totality final "$P" -o "Eclipse 2024/finished.tif"
```

`totality <command> --help` lists every option (all of them are also in the
*Command-line reference* below). More examples:

```bash
totality inventory "D:/Eclipse 2024/RAW" --project eclipse2024.json --csv frames.csv
totality inventory RAW --lat 36.9422 --lon -90.3343 --elev 110 --camera-utc-offset -6   # contacts computed, frames outside totality marked
totality inventory RAW --c2 2024-04-08T12:56:37 --c3 2024-04-08T13:00:48   # or type the contacts (camera clock)
totality inventory RAW --flats FLATS --darks DARKS --bias BIAS              # optional calibration frames
totality inventory RAW --lens "Nikon AF-S Nikkor 80-400mm f/4.5-5.6G ED VR" # force a lens profile
totality lens "AF-S Nikkor 80-400mm f/4.5-5.6G ED VR" --focal 400 --aperture 8
totality flat-profile FLATS                       # fit and save a vignetting profile from your own flats
totality inventory RAW --profile my_profile.json  # use one profile for every frame
totality frame DSC_4100.NEF                       # show everything read from one file
totality frames eclipse2024.json --exclude DSC_9108.nef   # leave a frame out (--include to undo)
totality eclipse --date 2024-04-08 --lat 36.9422 --lon -90.3343 --utc-offset -5   # contacts and duration
totality align eclipse2024.json                    # align and find the Sun (site from the project)
```

## Alignment (Phase 2)

`totality align` loads every light frame as 2x2 superpixels (no demosaicing),
applies the calibration plan, and then:

1. **Finds the Moon** in each frame: the limb is located along 720 radial
   profiles and fitted with a robust circle (prominences and valleys rejected).
2. **Aligns on the corona** with Druckmüller's modified phase correlation:
   a tangential high-pass filter about the Moon, a soft mask (Moon, edges,
   saturation, stars, hot pixels), damped phase correlation with a Gaussian
   low-pass, and a sub-pixel peak centroid. Two changes from that paper (Druckmüller 2009) make
   it work across a 12-stop bracket:
   * the correlation uses *relative* tangential contrast (high-pass divided
     by local brightness), which is the same in any exposure and stops the
     bright inner corona from locking the result onto the Moon's motion;
   * frames are aligned against a reference that grows from the middle
     exposure outwards, then every frame against all the others, and the
     whole set is re-anchored to the middle exposure at the end.

   A frame counts as aligned when its correlation peak is confident (peak to
   sidelobe ratio, PSR, of 8 or more) and it shares at least 2% of the Sun's
   disc with the reference; a weaker peak (PSR 5 to 8) needs at least the
   disc's whole area. The shortest exposures show only a thin ring of inner
   corona (the 2024 1/6400 s frame: 5% of the disc, PSR 8-12) and still
   align. A frame that fails, or whose result lies far from where its Moon
   says the corona must be, is placed by its Moon only, left out of the
   reference and noted ("little usable overlap with the reference", "weak
   correlation peak").
3. **Field rotation** is measured for every frame against the middle
   exposure. **drift** (the default, **Rotation** on the Align tab) fits one
   rotation rate through those measurements (rotation = rate x time since
   the middle exposure; robust, each frame weighted by how reliable its
   measurement was) and applies it only when the rate is at least three times
   its uncertainty; every frame gets the fitted value, so the very short and
   very long exposures, whose own measurements are the shakiest, borrow
   accuracy from the rest. Measuring and correcting repeat until nothing
   significant is left; each frame's note gives the total rotation applied
   (deg/min and over the sequence) and what was left afterwards. Without the
   frames' capture times a single median rotation is used. The sky's field
   rotation changes smoothly on any mount (imperfect polar alignment, an
   alt-az mount, a fixed tripod), so drift is right whenever the camera was
   not touched during totality. **per-frame** keeps each frame's own
   measurement, unsmoothed and untested (a frame whose measurement failed
   gets none): only for a rotation that jumps, when the camera was bumped,
   re-aimed or turned between frames. **none** applies shifts only. For a
   large rotation (several degrees: an alt-az mount over a long totality, a
   turned camera), also tick **Search any rotation and scale**
   (`--any-angle`), which finds it coarsely before the fine measurement.
4. **Finds the Sun's centre** (Druckmüllerová et al. 2011, appendix): the
   Moon's track across the aligned frames is fitted to the eclipse
   geometry computed for your site from NASA Besselian elements. This also
   gives the image scale and where celestial north points. If the camera
   clock was wrong, the clock error is bounded by requiring every frame to
   fall inside totality and the remaining along-track uncertainty is
   reported; pass `--clock-error` if you know it. Without it, the clock
   error used is the middle of that allowed range: it comes from the frames'
   times and the eclipse's times at your site, not from the Moon's track.
   Frames whose Moon lies off the fitted track (camera shake, a doubled
   image) by more than 3x the typical scatter, and by at least 0.35% of the
   Moon's radius (1.5 px at 400 mm on a 45 MP full-frame camera, so the same
   on any lens), are flagged as off the track; each frame's distance from the
   track is kept for the earthshine stack.

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
totality elements-update                    # download NASA's catalogue now
totality elements-update --from canon.csv   # or install a copy downloaded in a browser
```

**Any focal length.** The tangential filter length, mask softness and
analysis window follow the Sun's size in the image (tested from a 20 px to a
180 px solar radius); `--rho` and `--window` override them.

**Mirrored optics.** Camera lenses give a true image; telescopes with a star
diagonal give a mirror image. The Moon's track cannot tell the two apart,
and the choice moves the computed Sun centre by twice the closest approach
of the Sun's and Moon's centres: negligible near the central line, tens of
pixels near the edge of the path. Tick **Optics mirror the image** on the
Align tab (`--mirrored`) for mirrored optics; the tool reports both answers
and warns when the difference matters.

`inventory` prints the frame table, the detected brackets (step size, range,
missing shutter speeds), and the calibration plan: which vignetting source
each frame uses, how darks and bias are matched, and the corner falloff.

**Contact times** are computed for your site from NASA's Besselian elements
(bundled for recent eclipses; NASA's catalogue of every eclipse from 1600 to
2400 is downloaded once when needed) and kept in the camera's own clock, the
one the RAW files record:

    camera clock = UT + camera time zone - clock error

The camera time zone comes from the files when they record one, otherwise
from the setting; an unknown clock error counts as 0. A wrong clock or time
zone would mark frames taken in totality as partial, so before marking, the
toolkit checks your exposure brackets against the contacts. A bracket is shot
in one go, so if one is cut in two by a contact (or no frame falls inside
totality), nothing is marked without asking, and it says which clock error
would put every bracket inside. You can also type the contacts, in the
camera's clock. Bracket detection and Moon tracking only use time differences, so a wrong
clock never affects them.

## HDR stacking (Phase 3)

Each frame leaves out the corona just round its own Moon (the limb is
blurred there and the Moon moves between frames). That margin is scaled to
the Moon's size in the image: 0.7% of its radius, at least 2 px, so 3 px for
the 2024 D850 at 400 mm and about 7.5 px at 1000 mm. The log names it.

```bash
totality stack eclipse2024.json            # uses the alignment saved by 'totality align'
totality stack eclipse2024.json --square   # quicker: a square 10 solar radii wide
```

Every frame is read again at full resolution and calibrated with the
frames chosen on the Frames tab (see [Calibration](#calibration)):
the matching master dark, else the master bias, else the RAW file's black
level subtracted (negative values kept); the flat or vignetting profile
divided out; isolated hot pixels replaced, demosaiced with the linear Malvar-He-Cutler method in
32-bit floating point, and resampled onto the reference frame with the
alignment transform (bicubic).

**Brightness matching.** Shutter speeds are rarely exact. Each frame's
brightness relative to the reference, `value = gain * radiance + offset`,
is measured against its neighbour in exposure on pixels well exposed in
both and clear of the Moon: both frames are blurred identically and the
gain is a weighted ratio of sums, so differences in noise or sharpness
(seeing) between frames do not bias it, and errors do not build up along a
long bracket. It is compared with the shutter's true time behind its
rounded label (1/30 s is really 1/32 s); frames more than 0.3 EV off are
reported. If two neighbours share too few usable pixels, the shutter speeds
are used instead and the log says so. The gain is one number per frame (an
exposure ratio is the same for every colour), measured on the mean of the
colours; the offset, the light that does not scale with exposure (twilight
that changes during totality, scattered light, small black-level
differences), is then fitted **per colour** with that gain, because that
light has a colour of its own: the eclipse sky is blue (one offset for all
three colours would over-correct red under a blue sky). The stack log's table lists the offsets per colour.

**Merging.** Each pixel is the inverse-variance weighted mean of the frames'
radiance estimates. Weights fall to zero approaching saturation and inside
each frame's own Moon (plus the margin above), so the moving Moon never
darkens the inner corona. Inside the Moon as it was in the Moon frame (the alignment
reference unless `--moon-frame` says otherwise) the result is that frame's
data, which keeps the lunar limb sharp. Pixels that are saturated or covered
in every frame (rare with 1/3-stop brackets) take the shortest exposure that
shows them and are counted in the report. Rows and columns at the frame edge
that are not image (a D850's last column reads several times too bright) are
detected and given zero weight.

Frames the Sun-centre fit flagged as off the Moon's track are skipped
unless `--include-rejected` is given.

**Outputs:** the `.fit` files go in a `fits` folder and the `.json` files in a
`json` folder inside the project folder; the preview picture goes next to the
project (or in the `-o` folder). By default
they are the full frame, the same size as the RAW files and with the Sun
where it is in the reference frame, so they line up with your originals in
Photoshop and keep faint streamers out to the frame edges. Pixels near the
edges that no aligned frame covers are set to zero (reported).

For a quicker run, `--square` outputs only a square 10 solar radii wide
around the Sun (about a third of the pixels, so roughly three times faster
and smaller); `--radii N` sets the half-width in solar radii.

A full 45-megapixel frame needs about 3-5 GB of free memory at the peak.

| File | Contents |
| --- | --- |
| `<name>_hdr.fit` | linear 32-bit RGB in units of the reference exposure, camera colour; header has `SUNX/SUNY/SUNR`, `MOONX/MOONY/MOONR`, `PIXSCALE`, `NORTHANG` and the crop origin (0, 0 for the full frame) |
| `<name>_weight.fit` | sum of weights per pixel (where the data came from) |
| `<name>_preview.tif` | 16-bit asinh stretch with the camera white balance, for a quick look only |
| `<name>_stack.json` | each frame's gain and offsets (per colour), and the settings |
| `<name>_vignetting.fit` | the vignetting correction divided out of the reference frame, 1/8 size (1.0 = the middle) |
| `<name>_flat_vs_profile_<settings>.fit` | with **Master flat + compare** only: the master flat over the smooth profile (dark spots are dust) |

## Enhancement (Phase 4)

```bash
totality enhance eclipse2024.json          # uses the HDR stack saved by 'totality stack'
```

Writes one picture per layer (16-bit TIFF by default; `--format png|jpg`,
`-o` folder, `--name` prefix), all the same size and lined up pixel for
pixel, ready to stack as layers in Photoshop, plus a suggested blend:

| Layer | What it shows |
| --- | --- |
| `natural` | log stretch (default); `--flatten` (off by default, e.g. 0.8) removes that much of the radial fall-off. Or `--natural-stretch gamma`: the base Morgan & Druckmüller (2014) combine MGN with, ((B - min) / (max - min))^(1/γ) of the linear stack, `--gamma` 3.2 as in Morgan & Druckmüller (2014) (below) |
| `nrgf` | normalizing-radial-graded filter (Morgan et al. 2006): each circle around the Sun normalized to the same mean and spread |
| `fnrgf` | Fourier NRGF (Druckmüllerová et al. 2011): mean and spread follow position angle too, so faint and bright regions at the same height get similar contrast |
| `fnrgf_n` | FNRGF-N: the same with noise compensation (`--noise-m`, default 0.7); the noise is estimated automatically from the stack's weight map |
| `adaptive` | multi-scale Gaussian normalization (Morgan & Druckmüller 2014) of the linear image, as in Morgan & Druckmüller 2014 ("the original image normalised by exposure time"): fine structure at all scales and orientations, the published successor to the adaptive circular high-pass filter |
| `achf` | **Adaptive (ACHF)**: Druckmüller's adaptive circular high-pass filter (Druckmüller, Rušin & Minarovjech 2006; defined in Druckmüllerová 2013): the log image minus its local mean under a kernel that follows the circles round the Sun, at its own six scales (size and strength each), Moon left out. Written only on request (`--layers ...,achf`): a few minutes on a full frame |
| `blend` | suggested mix: the natural layer as the base with the filter layers applied in Soft Light, Overlay or Linear at the given strengths; by default every filter layer is at 0 (`--blend natural=1`), so the blend is the natural layer until you add layers, e.g. `--blend natural=1,fnrgf_n=0.7,adaptive=0.5`; add `:overlay` to a layer for Overlay, e.g. `--blend natural=1,adaptive=0.1,fnrgf=0.05:overlay`, or `:linear` for the combination each method's own publication uses (below) (`--blend none` to skip); the same thing you would do with layers in Photoshop |

**Names used**

| Short name | Full name | Source |
| --- | --- | --- |
| HDR | High Dynamic Range: all exposures merged into one picture | |
| NRGF | Normalizing-Radial-Graded Filter | Morgan, Habbal & Woo (2006) |
| FNRGF | Fourier Normalizing-Radial-Graded Filter | Druckmüllerová, Morgan & Habbal (2011) |
| FNRGF-N | FNRGF with noise compensation | Druckmüllerová, Morgan & Habbal (2011) |
| MGN | Multi-scale Gaussian Normalization (the **Adaptive (MGN)** layer) | Morgan & Druckmüller (2014) |
| ACHF | Adaptive Circular High-pass Filter (the **Adaptive (ACHF)** layer) | Druckmüller, Rušin & Minarovjech (2006); Druckmüllerová's thesis (2013) |

### Getting a good picture

Set the natural layer up first: it is the base every other layer is
blended onto. Then add filter layers in the **Blend** box, one at a time,
and judge fine detail in the [detail box](#the-detail-box). Good starting
points:

| Look | Settings |
| --- | --- |
| Natural, with structure (a good first try) | Stretch **Log**; **FNRGF-N** 60 % and **Adaptive (MGN)** 60 %, both **Soft Light** |
| Morgan & Druckmüller (2014) | Stretch **Gamma**, Gamma 3.2, Gamma white point 99.9 %; **Adaptive (MGN)** 43 % on **Linear** |
| The same structure on a brighter base | Stretch **Log**; **Adaptive (MGN)** 43 % on **Linear** |
| A gentler MGN | **Adaptive (MGN)** 43 % on **Weighted average** |
| ACHF as Druckmüllerová's thesis (2013) combines it | Stretch **Log**; **Adaptive (ACHF)** 2000-5000 % on **Linear**, **Contrast** 1 (about twice the strength with Gamma) |
| Druckmüllerová, Morgan & Habbal (2011), Figure 9 | **FNRGF** 40 % on **Linear** and **Adaptive (ACHF)** 60 % on **Weighted average** |

Use Adaptive (MGN) *or* Adaptive (ACHF), not both: they bring out the same
fine detail in different ways. MGN evens contrast out everywhere (strong,
graphic structure, flatter overall brightness); ACHF keeps bright structure
brighter than faint structure (closer to the natural look, more noise in
the faint corona).

The layers come out already stretched, so run Enhance on the linear HDR
stack (never on a stretched copy) and use only gentle curves or levels on
the results.

**In Photoshop:** `natural` at the bottom in Normal mode, `fnrgf_n` above it
in Soft Light (40-70 %), `adaptive` above that in Soft Light (50-80 %).
Overlay is a stronger alternative at a lower opacity. The filter layers are
centred on mid-grey, which both modes treat as "no change"; desaturate them
so the colour comes from the natural layer.

### Blend modes

| Mode | What it does | Typical strength |
| --- | --- | --- |
| **Soft Light** (default) | Gentle: adds structure, keeps the natural tones, leaves the dark sky dark. Safe for every layer. | 50-80 % |
| **Overlay** | Like Soft Light but much stronger, mostly in the middle tones. | 5-20 % |
| **Linear** | Combines the layer with the natural layer as the method's own publication does (see below). Strength may go above 100 %. | see below |
| **Weighted average** | Adaptive (MGN) and Adaptive (ACHF) only: a gentler alternative to Linear; the sky comes up a little towards grey. | 40-60 % |

What **Linear** does depends on the layer:

* **NRGF, FNRGF, FNRGF-N** (Druckmüllerová, Morgan & Habbal 2011): an
  average of the natural layer and the filter layer, weighted by the
  strength (the natural layer counts as 100 %). As in their published
  pictures, the dark sky comes up towards grey and the natural tones
  flatten. 30-100 %.
* **Adaptive (MGN)** (Morgan & Druckmüller 2014): the detail is added at
  full strength and the sky stays dark. 43 % is their mix. With Stretch
  **Gamma** at 3.2 this is their own picture; on an eclipse it comes out very
  dark unless **Gamma white point** is lowered to about 99.9 % (at 100 %,
  about a third of the corona between 1.2 and 3 solar radii went black on the
  2024 stack; at 99.9 %, 7 %). With Stretch **Log** the same 43 % puts the
  structure on a brighter base.
* **Adaptive (ACHF)** (Druckmüllerová's thesis, 2013): the detail is added
  to the stack *before* it is stretched, so the corona keeps its brightness,
  the sky stays dark, and only the detail grows. It needs far higher
  strengths than any other mode: 2000-5000 % with Stretch **Log**, about twice
  that with **Gamma** (up to 10000 %). Leave **Contrast** at 1 and use the
  strength. At high strengths the darkest fine detail right by the Moon's
  edge shows as black specks: that is noise in the stack, shown as it is.
  Lower the strength or the finest scale if it bothers you.

<details>
<summary>How the modes are computed</summary>

The natural layer (after any Soft Light or Overlay layers) is X with
weight K1 = 1; each Linear or Weighted average layer is Y with its strength
as K2.

* NRGF family on Linear, and every Weighted average layer: Z = (K1 X + K2 Y)
  / (K1 + K2), both images on 0..1, as in Druckmüllerová's FNRGF software
  (thesis appendix: it "first normalizes both the original and the filtered
  image into the interval <0, 1>"; Druckmüllerová, Morgan & Habbal 2011:
  "it is their ratio rather than their values that affects the result").
  Dividing by K1 + K2 matters: a plain sum turns much of the picture flat
  black, because the natural layer puts the sky at black and the filter
  layers carry its noise as detail of both signs. In Photoshop, for one
  layer: Normal mode at opacity K2 / (1 + K2).
* MGN on Linear: I = h C'g + ((1 - h) / n) sum g_i arctan(k C_i) (Morgan &
  Druckmüller 2014, eq. 5), with h = K1 / (K1 + K2), so 43 % is h = 0.7. The
  detail goes in centred on 0 and in radians.
* ACHF on Linear: h = K1 f + K2 g (thesis, section 5.2), where f is the
  linear stack and g is f minus its local mean under the circular kernel,
  also on the linear stack; K2 is the strength times **Contrast**. The natural
  layer's own stretch, measured on f, then draws h. On the 2024 crop,
  fine-detail contrast at 1.1-1.5 solar radii matched Weighted average at
  60 % with K2 of about 40-50 on Log and about 100 on Gamma.
* Weighted average for MGN keeps about a third of the Linear form's detail;
  for ACHF it uses the layer as shown on its own (detail measured on log
  brightness, through a soft curve).

Nothing is measured on the picture, so a detail box matches the full-size
result exactly. The blend is clipped to 0..1 and not re-stretched.

</details>

### Natural layer and colour

| Setting | What it does | Default | Typical |
| --- | --- | --- | --- |
| **Stretch** (`--natural-stretch`) | **Log**: a natural look with the faint outer corona visible. **Gamma**: the base Morgan & Druckmüller (2014) put under their MGN detail; much darker on an eclipse. | Log | |
| **Gamma** (`--gamma`) | Gamma only. Higher lifts the faint outer corona and holds back the bright inner corona. | 3.2 | 2.5-4 (Morgan & Druckmüller 2014) |
| **Gamma black point** (`--gamma-black`) | Gamma only. How dark the darkest parts go, as a percentile of the stack. | 0 % | 0.5-2 % if the sky looks grey |
| **Gamma white point** (`--gamma-white`) | Gamma only. What comes out white. At 100 % the single brightest pixel sets white and an eclipse is very dark. | 100 % | 99.9 % for eclipses |
| **Flatten** (`--flatten`) | Log only. How much of the corona's fading away from the Sun is taken out. The sky stays dark either way. | 0 | 0.5-0.7; 0.8-1 for the outer corona as bright as the inner |
| **Evenness** (`--even`) | Log only. Evens out bright streamers against dark coronal holes; the fine strands stay. | 0 | 0.6-0.8 |
| **Midtone** (`--midtone`) | Lightens or darkens the middle tones so the inner corona sits at this brightness. Most useful with Gamma. | 0 (off) | 0.4-0.6 |
| **Highlight roll-off** (`--shoulder`) | How the brightest parts (inner corona by the Moon, bright prominences) reach white. At 0 the brightest 0.1 % clip to flat white; above 0 they keep their gradations and prominences keep their colour. | 0 | 0.3 |
| **Saturation** (`--saturation`) | Colour strength in every layer; 0 is black and white. | 1 | |
| **White balance** (`--white`) | **corona**: the inner corona (1.2-2.5 solar radii) is made white, which it really is (scattered sunlight). **camera**: the camera's as-shot balance. **none**: raw camera colour (green). | corona | |
| **Keep the sky** (`--keep-sky`) | Off: the sky (twilight glow and its gradient) is measured and removed before filtering, so the filters do not turn it into arcs. On: the sky is kept, for example its blue. | off | |

Colour fades to grey on its own where the corona is too faint for its colour
to be trusted. The Moon is black in these layers; the Earthshine step fills
it.

<details>
<summary>What happens before the layers are made</summary>

* **The sky** is modelled for each colour as a smooth surface, fitted to the
  darkest parts of the frame away from the inner corona, and removed. Terms
  that fall off from the Sun take up the faint outer corona, so it is not
  mistaken for sky.
* **The Sun's centre and radius** come from the stack, so the circles the
  filters work on are centred on the Sun, not the Moon. Circles that run off
  a rectangular frame are completed from further in, a thin border at the
  frame edge is left out, and dust shadows and stars beyond 2 solar radii are
  kept out of the circle statistics, so none of these show as arcs or rings.
* **Log stretch:** black is set at the sky and white at the brightest 0.1 %.
  Flatten stops where the corona sinks into the noise, so the sky beyond is
  not lifted.
* **Gamma stretch:** ((B - a0) / (a1 - a0))^(1/Gamma) of the linear stack B,
  with a0 and a1 the black and white points (Morgan & Druckmüller 2014: they
  "set the minimum and maximum input values"). Only Midtone applies to it.
  On the 2024 stack, with black and white at the stack's extremes, the median
  grey at 1.5 and 4 solar radii was 0.28 and 0.07 (Log: 0.75 and 0.29);
  Midtone 0.5 brought it to 0.65 and 0.27.

</details>

### Adaptive layer (MGN)

| Setting | What it does | Default | Typical |
| --- | --- | --- | --- |
| **Strength** (`--adaptive-k`) | How hard the detail is brought out. Keep 0.7 with the Linear blend. | 0.7 (Morgan & Druckmüller 2014) | 1-1.5 harder, 0.3-0.5 gentler |
| **Noise gate** (`--noise-gate`) | Stops the finest scales from turning noise into grain. 0 is the filter as published. | 0 (off) | 3 for a grainy layer |
| **Scale 1-6: size** (`--mgn-scales`) | The size of the structure each scale brings out, in pixels of the full-size stack. | 1.25, 2.5, 5, 10, 20, 40 px (Morgan & Druckmüller 2014) | raise in proportion for a longer lens or smaller pixels |
| **Scale 1-6: strength** (`--mgn-weights`) | How much each scale counts, 0-1. Lowering one also weakens the whole layer a little; 0 switches it off. | 1 | lower the first one or two if grainy |

The default sizes suit a 400 mm lens on a 45-megapixel full-frame camera,
where the Sun is about 410 px in radius (the Align log's "Sun centre" line
gives yours). On a typical stack the 1.25 px scale is mostly noise: lower
its strength or use the noise gate.

<details>
<summary>How the noise gate works</summary>

At its finest scales a stack is mostly noise (on the 2024 stack, structure
stood only 2x and 3.6x above the noise at 2.5 and 5 px, against 9-250x at
10-83 px). The gate weights each scale's detail by how far it stands above
the noise at that scale. The noise at each scale comes from a noise image
with the stack's own noise level (from its weights) and grain (measured in
the faint corona), put through the same filter. Pure noise comes out with
under 0.4x the texture of the plain filter. The gated layer is shown on the
same scale as the plain one, so where there is little real structure it
stays flat.

</details>

### Adaptive layer (ACHF)

| Setting | What it does | Default | Typical |
| --- | --- | --- | --- |
| **Scale 1-6: size** (`--achf-scales`) | The size of the structure each scale brings out. Shown in pixels of the open stack, kept as a share of the Sun's radius, so saved settings suit any lens. | about 0.5, 1, 3 and 10 px on a 400 mm / 45-megapixel setup; the last two scales off | |
| **Scale 1-6: strength** (`--achf-weights`) | How much each scale counts in the mix of sizes; only the balance between them matters. 0 switches a scale off and saves time. | 1, 1, 1, 1, 0, 0 | lower the finest if grainy; lower the large ones if the Moon's edge glows |
| **Separate parts at** (`--achf-part-limit`) | Keeps prominences from leaving a dark halo: anything this many times brighter or darker than the corona at its height is filtered on its own. 0 turns it off (three times faster). | 2 | 2 |
| **Contrast** (`--achf-contrast`) | How strongly the layer shows its detail on Soft Light, Overlay and Weighted average. On Linear it multiplies the strength: leave it at 1 there. | 1 | 1.5-3 if flat |

ACHF is slow: on a 2530 px crop it took 28 s at the default settings (4 s
for Adaptive (MGN)), and a full 45-megapixel frame has 7x the pixels. So it
is only made when ticked under **Layers to write**, blended or shown
(`--layers ...,achf` on the command line).

<details>
<summary>How ACHF is computed</summary>

As defined in Druckmüllerová's thesis (*Application of Adaptive Filters in
Processing of Solar Corona Images*, Brno University of Technology, 2013),
section 5.2: for each scale, each pixel's local mean is taken under a
Gaussian kernel that bends along the circle round the Sun (eq. 5.1), with
the Moon and pixels outside the frame left out (eq. 5.2), and the filtered
image is the pixel minus that mean. The scales are added with their
strengths, which boosts finer detail more. Tests check the circular kernel,
that the Moon has no influence, and the small-scale limit.

**Separate parts at** is the thesis's requirement (d): pixels in very
different parts of the picture do not count in each other's means. Here a
pixel more than that many times brighter or darker than the mean along its
own circle is a part of its own. Without it a bright prominence darkened the
corona beside it on the 2024 crop; with it 247 pixels (the prominences)
became a part and the coronal structure was unchanged.

Chosen here, as the thesis does not say: on Soft Light, Overlay and
Weighted average the detail is measured on log brightness (the same
contrast counts the same at every height) and shown through a soft curve:
at Contrast 1, a pixel 1.1 % brighter than its surroundings shows at 82 %
grey. On Linear it follows the thesis exactly (see Blend modes). The
"nonlinearity in the use of filtered images" of Druckmüller's Corona 4.1
software is not published and not reproduced.

</details>

### FNRGF and FNRGF-N

| Setting | What it does | Default | Typical |
| --- | --- | --- | --- |
| **Order** (`--order`) | How closely FNRGF follows brightness changes round the Sun. Lower keeps streamers brighter than coronal holes; higher lifts faint regions to match bright ones, and too high looks flat and blotchy. | 10 (Druckmüllerová, Morgan & Habbal 2011) | 3-6 or 14-20 |
| **Segments** (`--segments`) | How many pieces each circle is measured in. | 50 (Druckmüllerová, Morgan & Habbal 2011) | 80-100 with Order above 15 |
| **Noise compensation** (`--noise-m`) | FNRGF-N only. How much of the noise is allowed for, 0-0.99. Higher shows more faint streamers and more grain. | 0.7 | compare 0.5 and 0.9; 0.3-0.5 for noisy stacks |

The command line also takes the attenuation factors of the paper
(`--mean-att`, `--std-att`; default 1, 0.9, ..., 0).

### Settings files

**Save settings…** on the Enhance tab writes every setting on the tab to a
file you name. **Load settings…** sets the tab from such a file, on any
project, or from the `<name>_enhance.json` that **Write full-size layers**
saves in the project's `json` folder,
so the exact settings of an earlier picture can be had back. On the command
line, `totality enhance <project> --settings <file>` starts from the file,
and options typed on the same line take its place for what they set (for
example `--settings look.json --saturation 0.8`).

The ACHF sizes are saved as a share of the Sun's radius, so they keep their
size against the Sun on a stack taken with another lens; the MGN sizes are
saved in pixels. The detail box is not saved. Settings a file does not have
stay as they are, and the log lists them.

### The detail box

The whole-picture preview is a reduced copy (about 1400 px wide), so fine
settings (the Adaptive scales, the noise gate, Noise compensation) cannot be
judged on it. Hold **Shift** and drag a box on the whole picture: the
**Detail box (full resolution)** tab shows that part of the full-size stack
(at most 2000 px a side), exactly as **Write full-size layers** will make
it, and redraws it when a setting changes. The first box takes a while (the
full-size stack is read and measured once); after that only what a change
concerns is redrawn, so a blend strength, mode or Contrast change shows in
under a second. The box is kept in the project.

<details>
<summary>Why the box matches the full-size picture exactly</summary>

Writing the layers is two steps, and the box runs only the second:

1. **Prepare**, once per stack and again only when **White balance** or
   **Keep the sky** changes: the sky model, white balance, noise, dust spots
   and masks; then, measured on the whole picture and kept, the circle
   statistics of the NRGF family, the natural layer's black and white
   points, and the noise model of the noise gate.
2. **Render** the box plus a margin as wide as the filters reach, then cut
   the margin off.

The filter layers are therefore shown on **fixed scales**, set in each
method's own units, instead of being stretched to each picture's range:

| Layer | Shown as |
| --- | --- |
| NRGF | white at +3.5 standard deviations from the circle's mean, black at -3.5 |
| FNRGF | white at +6.8 local deviations |
| FNRGF-N | white at +20 local deviations (less the noise) |
| Adaptive (MGN) | its own range, -pi/2 to pi/2, as Morgan & Druckmüller (2014) |
| Adaptive (ACHF) | 82 % grey at 1.1 % brighter than the surroundings, at Contrast 1 |

So a corona with little structure looks flatter than one with much, as it
is, and each look is set by eye with the blend strengths and Contrast, as in
Druckmüller's own software. Tests compare boxes through the limb, at a frame
corner and in the outer corona of the 2024 stack with the full-size result.

</details>

## Earthshine (Phase 5)

```bash
totality earthshine eclipse2024.json       # after 'totality stack'
```

The Moon's night side, lit by sunlight reflected from the Earth. The Moon
moves across the Sun during totality (about 40 px in three minutes at
400 mm), so the corona-aligned HDR stack blurs it. This step makes a
separate stack aligned on the Moon, removes the glare of the corona from the
lunar disc, and keeps the seas and highlands. On the Earthshine tab it has
three parts.

### 1 Stack the Moon

| Setting | What it does | Default |
| --- | --- | --- |
| **Use frames within** (`--stops`) | Only the longer exposures carry earthshine: frames within this many stops of the longest are used. | 6 stops |
| **Colour channels** (`--channel`) | Which channels the earthshine is made from. **auto** tries green, green + blue and all three, and uses the one with the least glare round the limb (on the 2024 stack: green). | auto |
| **Include frames flagged as off the Moon's track** (`--include-rejected`) | Also use frames Align flagged as shaken. | off |

How much each frame counts is decided from the data, so the limb comes out
clean:

* each frame by its own noise, not at all where it is saturated, fading in
  gently from there (so no frame starts counting all at once, which left
  thin lines along the limb);
* frames whose Moon lies off its track (camera shake, from Align) count
  less; the log names them;
* close to the limb, a pixel that disagrees with the other frames there
  counts less (this removed a row of ticks inside the limb that two frames
  near third contact left on the 2024 stack; the log names such frames).

### 2 Lunar surface (seas and highlands)

| Setting | What it does | Default | Typical |
| --- | --- | --- | --- |
| **Glare smoothness** (`--glare-order`) | How flexible the glare surface taken off the disc is, 0-6 in half steps. Higher removes uneven glare better (bright arcs by the edge) but flattens the large maria more; lower keeps them bolder. A half step averages the two neighbouring whole orders' fits. | 3.5 | 3-4; 2 removes only a gentle gradient |
| **Remove the bright glare along the limb** (`--no-limb-glare` turns it off) | The glare climbs steeply in the last pixels inside the limb; this follows it round the limb and takes it away. | on | |
| **Remove the glare fanning out from a bright prominence** (`--prominence-glare`) | Fits the fan of glare and spikes a bright prominence throws onto the Moon, and takes it away. Off, the fan is shown as recorded; on, what was under it cannot come back (on the 2024 Moon, a mare became noise). | off | |
| **Keep maria by the limb** (`--limb-shape`) | Takes away only light that fades inward from the limb, so a mare right by the limb is kept. | off | |
| **Smoothing** (`--smooth`) | Plain smoothing of the surface, px. | 0 | about 1 |
| **Smoothing near the limb** (`--edge-smooth`) | Smoothing only where the noise is high, by the limb. | 0 | 3 |
| **Edge-preserving smoothing** (`--surface-denoise`) | Smooths the noise inside the maria and keeps their edges crisp. | 0 | 3-5 |
| **Filled edge, at least** (`--edge-fill`) | The last pixels inside the limb are only noise and are filled from just inside; this sets a minimum width, as a share of the Moon's radius. 0 lets the noise decide. | 0 % | 2.5 % hides leftover glare at the edge |
| **Grain in the filled edge** (`--no-edge-grain` turns it off) | Gives the filled pixels grain like the rest of the surface. | on | |
| **Brightness** (`--brightness`) | The brightest lunar surface in the layer, 0-1. | 1 | 0.35 for a darker Moon |
| **Natural colour** (`--no-colour` turns it off) | The earthshine's own tint, and prominences red where they spill over the limb. | on | |
| **Colour by the limb** (`--limb-colour`) | The long exposures catch the lens's glare of the red chromosphere just inside the limb (2024: R/G 0.64 at 2-4 px inside, 0.47 in the Moon's middle). **As recorded** keeps that red band; **No red** gives each ring near the limb (to a quarter of the radius in) the colour of the Moon's middle, brightness unchanged; **No red, neutral Moon** also gives the whole Moon the corona's white (earthshine itself is a little blue-green: 4 % less red, 5 % more blue than the corona in 2024). Next to **Colour channels** on the Earthshine tab; needs **Stack earthshine**. | As recorded | No red |

A darker, flat, neutral Moon: **Brightness** 0.35, **Natural colour** off,
**Edge-preserving smoothing** 4. Nothing is added from outside your
photographs: the lunar surface is your data with the glare taken away. On
the 2024 stack the surface is mostly noise at every scale below about 35 px:
the maria are really there, craters and rays are not.

The pixel settings (the smoothing widths) suit a 400 mm lens on a
45-megapixel full-frame camera, where the Sun is about 410 px in radius;
with a longer lens or smaller pixels raise them in proportion. Every other
width by the limb scales with the Moon's size on its own.

The report gives the lunar detail over the noise; below about 3 the layer is
mostly noise (short exposures, low ISO or a bright sky).

<details>
<summary>How the glare is removed</summary>

* The glare from the inner corona across the disc, a gentle gradient, is
  taken off with a smooth surface (**Glare smoothness**), fitted from a
  margin just inside the limb (0.7 % of the Moon's radius, at least 2 px).
  The surface is a 2-D polynomial of that order, robustly fitted; a half step
  is the average of the two neighbouring orders' surfaces. On the 2024 stack
  the large-scale maria contrast was 0.42 of the shown range at 3, 0.32 at
  3.5 and 0.29 at 4, while 3 left a bright arc of glare by the upper-left
  limb that 3.5 mostly and 4 fully removed.
* Close to the limb the glare is 10-20x the earthshine and changes within a
  few degrees round the limb. Terms that fall off with distance inside the
  limb take out its average shape; the rest is followed round the limb with
  a window that is about 1° wide at the limb and widens further in (13° at
  40 px, 50° at 90 px on a 436 px Moon), so maria near the limb such as
  Crisium survive. Only within about 20 px of the limb is most detail
  flattened.
* **Keep maria by the limb** fits, in short arcs round the limb, light that
  fades inward and takes only that away: glare can only fade inward, while a
  mare is a dip and highlands a bump.
* **Remove the glare fanning out from a bright prominence** fits the fan as
  a pattern of directions times a fall-off with distance, centred on the
  prominence (up to four, brightest first).
* Glare far brighter than any lunar surface is shown as recorded, rolled off
  towards white: nothing is filled in or smoothed over.
* **Edge-preserving smoothing** is a joint bilateral filter, judged on a
  lightly smoothed copy, so it only averages across similar brightness.

</details>

### 3 Final image: Moon and corona together

`totality final eclipse2024.json`, or **Write final image** on the
Earthshine tab (with a preview), puts the Moon-aligned earthshine stack into
the HDR stack inside the lunar disc, lined up on the limb, and stretches
Moon and corona together with the settings of your last Enhance run. The
limb is the stacks' own: one curve, one colour treatment, no seam. A
prominence is the same red on both sides of the limb. You do not need to run
Enhance again.

| Setting | What it does | Default |
| --- | --- | --- |
| **Corona image** (`--base`) | The blend or the natural layer, stretched with the Moon; or paste into a file (below). | blend |
| **Lunar surface strength** (`--strength`) | How bright the seas and highlands are, 1 = as made in part 2. | 1 |
| **Moon meets corona** (`--join`) | How Moon and corona meet, pixel by pixel, before the stretch: **Corona wherever the stack has it** or **Band by the limb rebuilt** (both below). | Corona wherever the stack has it |

**Corona wherever the stack has it** (`corona-first`) follows one rule for
each pixel:

1. Where the Blend without the Moon has the corona (outside its black disc,
   1.5 px past the limb, and the HDR stack has data there), the HDR stack is
   used, exactly as in that Blend.
2. Otherwise, where the earthshine stack has data (at least 10 % of its usual
   frames: by a bright prominence the last few are near full well and their
   colours are off), the earthshine stack is used.
3. Otherwise the pixel has no data and stays black.

Nothing is rebuilt or blended. The log gives the counts ("the earthshine
stack on N px (M of them outside the limb, where the HDR stack has no
data); K px by the limb with neither, left black"). On the 2024 stack with
the Moon from frame 9, 1119 px had neither, mostly at the base of the 6 o'clock
prominence, at 3 o'clock and at 10 o'clock.

**Band by the limb rebuilt** (`rebuilt`), the older way:
just outside the limb neither stack is good on its own: the earthshine
stack's corona is smeared (it is aligned on the Moon) and the HDR stack
holds few frames there (the Moon covered it part of the time). So the
earthshine stack's limb hands over to the HDR corona from where the HDR
stack is reliable, measured from its weights; the log says where ("the HDR
stack's corona is used from N px outside the limb"; 3 px on the 2024
stack). Just outside that, the HDR stack's corona is still darkened by the
Moon's blurred edge in frames whose Moon was just past the stacking margin,
mostly on one side of the Moon (on the 2024 stack up to a third at 3 px out,
at 6-8 o'clock). Measured direction by direction against the corona's own
trend further out, the darkening is taken off (linear; a prominence there
keeps its shape), and the log says by how much ("the corona just outside the
limb was darkened by the Moon's edge: brightened by up to N %"). The corona
continued inwards to the limb follows that same trend. By a bright prominence, where the long earthshine frames saturate,
the light inside the limb is taken from the last well-exposed earthshine
pixel to the limb. The filter layers have no data on the Moon: they keep full strength wherever
they have data (out to the corona layers' black disc), and on the Moon count
as neutral, no detail, at full weight, over a 1 px soft edge. Between their
black disc and where the HDR stack's corona is used, the natural layer draws
the Moon's own blurred edge, but the filter layers see the corona continued to
the limb: MGN in particular read the edge as a dark feature, a 1 px dark line
round the limb. So just outside the limb the blend is the blend without the
Moon, and on Linear the Moon gets the same overall treatment as the corona.

**Another way: pasting.** **Paste into a file** on the tab (`totality final
--paste` / `--base-file`) pastes the Moon into a picture you choose, for a
corona finished elsewhere. The Moon pasted is the Moon as Blend draws it
(with your Enhance settings and **Lunar surface strength**), cut where the
corona layers' black disc ends, combined with **Blend mode** (screen by
default); the rest of the picture is left as it is. So the pasted Moon is
the same as in the Blend final image; only the first few pixels outside the
edge differ, where the Blend final image blends the two stacks and a pasted
picture keeps its own corona. Before any **Write full-size layers** run there
are no settings to draw the Moon with, so the earthshine layer itself is
pasted, and the log says so.

**A Moon layer for Photoshop.** **Write Moon layer (for Photoshop)** on the
tab (`totality final --moon-layer`) writes only the Moon, exactly as Blend (or
Natural, from **Corona image**) draws it, with **Lunar surface strength**
built in, transparent everywhere else (TIFF or PNG, 16-bit). Put it above the
blend (or natural) layer you last wrote with **Write full-size layers**, in
Normal mode at 100 %: the result is the final image, limb included. Besides
the black disc it covers the band just outside the limb where the final image
blends the Moon's stack into the corona (wherever the final image differs
from that layer by more than half an 8-bit level), and its edge fades out over
1.15 % of the Moon's radius (5 px for a 400 mm lens on a 45-megapixel
full-frame camera, at least 2 px), so the two edges meet without a line.
To change the lunar
surface, change **Lunar surface strength** and write the layer again: the
layer's opacity would change the limb too. Put corona adjustments above both
layers. On the 2024 crop (surface strength 0.35) the layer over the blend
matched the Blend final image exactly where it is opaque (the disc and about
4-8 px outside the limb) and within half an 8-bit level everywhere else.

After **Stack earthshine**, the tab shows the Moon as the final image draws it
(a preview with your Enhance settings), so what you set there is what Blend,
Natural and pasting give.

The earthshine layer file (`<name>_earthshine.tif`, for Photoshop) is drawn
on its own: the seas and highlands stretched by themselves, with **Keep
glare (glow)** (`--keep-glare`, default 0.1) of the removed glare kept as a
glow inside the limb and **Soften the limb** (`--limb-soften`, default 0).
These two settings, under **Photoshop paste only (old way)**, are
used only for that file.

| File | Contents |
| --- | --- |
| `<name>_final.tif` | the finished picture (or `-o`) |
| `<name>_earthshine.tif` | layer the size of the HDR stack: the lunar surface inside the disc, black outside, for Screen or Lighten in Photoshop |
| `<name>_earthshine_crop.tif` | the same, cropped to the Moon |
| `<name>_earthshine.fit` | the Moon-aligned linear stack before glare removal |
| `<name>_earthshine_rgb.fit` | the same, per colour channel (used by the final image) |
| `<name>_earthshine_weight.fit` | how much went into each pixel of that stack |
| `<name>_moon_surface.fit` | the seas and highlands, glare removed, as added to the final image |
| `<name>_earthshine.json` | where the crop sits, the Moon's centre and radius, and the settings used |
| `<name>_moon.tif` | **Write Moon layer**: the Moon as Blend draws it, transparent elsewhere, for Photoshop (Normal mode over the blend) |

The final image needs `_earthshine_rgb.fit` and `_moon_surface.fit`; if they
are missing it asks you to stack the earthshine again.

## Prominences (Phase 6)

```bash
totality prominence-stack eclipse2024.json --brightness 0.3   # stack the short frames, save the settings
totality final eclipse2024.json                                # the final image, with them
```

Prominences glow in red hydrogen light (with weaker blue lines, and some
yellow from helium), so they look pink once the corona is white. In the HDR
stack they come mostly from the longest exposures not saturated there,
where they are swollen and soft with a red glow round them; in the shortest
exposures they are sharp. On the 2024 frames the edge of the prominence at
5 o'clock spread over 2.5 double pixels in the HDR stack and 1.9-2.0 in
single frames of 1/4000-1/1000 s.

**Stack prominences** stacks the frames from the shortest aligned exposure
up to **Longest exposure** stops longer (`--longest-stops`, default 2: on the
2024 set the 7 frames from 1/1600 to 1/400 s), in a square round the Sun
(`--radii`, 1.6 solar radii). It uses the HDR stack's own brightness
matching, so the result is in the same units, and leaves each frame's Moon
out as Stack does. It also measures how blurred each frame is, on the sharp
step at its own Moon's edge (the distance between the points a quarter and
three quarters of the way up the step; the slow glare inside the limb does
not count): 1.30-1.46 px on the 2024 frames, 1.40 px the median. It writes
`fits/<name>_prominence_hdr.fit` and `_prominence_weight.fit`.

Then **Write final image** (on this tab or the Earthshine tab, or
`totality final`) sharpens the HDR stack's own prominences, in the linear
stack before the stretch; **Use the prominence stack** on the Earthshine tab
turns this on or off. The prominences are already in the HDR stack, with
little noise and right down to the limb, only soft; the short frames show
how soft. For each prominence, the blur that turns the short frames'
version into the HDR stack's is measured (on the 2024 stack 0.6, 0.9 and
1.3 px), comparing the two over the prominence and just round it (where
the short frames have data, clear of their own Moon's blurred edge: 3
times the blur measured there). The HDR stack's prominence light is then
deconvolved by that much (Richardson-Lucy, 20 rounds), only where the
prominences are. A prominence's own light is the stack minus the corona
behind it, carried in from beside it at the same distance from the Sun
(near the limb the corona changes steeply with distance and slowly with
angle), so the corona is left as it is. Prominences are found by colour:
light at least **Redness needed** times redder than the corona there and
above 3 times the noise, on pieces that reach the limb. Their faint red
glow round them takes the same **Prominence brightness** (only its colour
excess, so the corona keeps its texture there). Where the HDR stack has no
data (saturated in every frame) the short frames fill it. Nothing changes
inside the Moon (the one in the final image), so it still hides their
bases. The log names the blur found for each prominence ("prominences: 3
sharpened; softer than in the short frames by 0.6, 0.9, 1.3 px").

**Preview** on the Prominences tab draws the final image round the Sun at
full size, exactly as **Write final image** will (your Enhance settings, and
the base and lunar surface strength on the Earthshine tab), so the settings
can be judged on what will be written.

| Setting | What it does | Default |
| --- | --- | --- |
| **Use the prominence stack** (Earthshine tab; `--off` turns it off) | Off, the final image keeps the HDR stack's own prominences. | on |
| **Sharpening** (`--sharpen`) | 0.5: the HDR stack's prominences as sharp as in the short frames (their extra blur undone). 1: also the blur measured on the Moon's edge (the lens and the air), sharper still, with more noise in faint ones. In between, in proportion. | 0.5 |
| **Structure** (`--structure`) | Brings out strands and loops inside them (the multi-scale method of Adaptive (MGN), on their own light). 0.3 with Sharpening 0.5 works well on the 2024 frames. | 0 |
| **Prominence brightness** (`--brightness`) | Their light against the corona's, before the stretch. They are far brighter than the corona, so at 1 their insides turn flat pink; about 0.5 shows their structure. Much lower turns them pale (their own green light, from helium, then shows). | 1 |
| **Redness needed** (`--min-redness`) | How much redder than the corona light must be to count. On the 2024 stack 99 % of the corona was within 1.08, and a faint prominence low by the limb reached 1.1-1.2. | 1.1 |
| **Include the chromosphere** (`--no-chromosphere` leaves it out) | The thin pink rim along the limb. Left out, the bases of taller prominences stay. | on |
| **Rim height** (`--chromosphere-arcsec`) | When the chromosphere is left out: red light lower than this counts as the rim. | 12 arcsec |

The chromosphere is a few arcseconds high. In mid-totality the Moon is
usually larger than the Sun and covers its edge all round (on the 2024
stack by at least 35″, 15 px), so the rim never shows in the final image;
the tab then says so and greys out these two settings. They matter for a
final image whose Moon is from a frame near second or third contact.

Within the first few pixels outside the limb (4 px on the 2024 stack), the
final image takes the light from the earthshine stack and the corona
continued in from further out (see *Earthshine*), so the part of a
prominence lower than that is not sharpened.

The command line can still write the prominences as a separate layer for
Photoshop, transparent outside them (`totality prominences`, from the
prominence stack if there is one), and put it on top of a picture
(`totality final-prominences`).

## Picture formats

Every picture (layers, earthshine, prominences, final images) can be
written as:

| Format | Bits | Compression | |
| --- | --- | --- | --- |
| TIFF (default) | 16 | none | highest quality, largest |
| PNG | 16 | lossless | same quality, smaller |
| JPG | 8 | quality 100, no colour subsampling, dithered | smallest; no 16 bits or transparency |

All are tagged 300 dpi (a printing label; the pixels are not changed) and
sRGB. In the window the Save dialog's file type chooses; on the command line
`--format tif|png|jpg` or the output file's extension.

## Calibration

The Frames tab only says where the darks, flats and bias frames are (and
reports how many it found, and whether the flats match the lights). How
they are used is chosen in the **Calibration** box on the Stack tab, and
the earthshine and prominence stacks repeat that choice:

| Setting | Choices | Default |
| --- | --- | --- |
| **Vignetting** (`--vignetting`) | **Automatic**; **Master flat** (flats from the same session); **Profile from the flats** (flats from another time); **Master flat + compare**; **Lensfun profile**; **None** | Automatic: the master flat if flats are chosen, otherwise the lensfun profile, otherwise none |
| **Lensfun lens** (`--lens`) | The lens for the lensfun profile: matched from the files, or one you choose | Matched from the files |
| **The light on the flats was uneven** (`--flats-uneven-light`) | Takes a straight slope across the flats out (below) | Off |
| **Darks** (`--darks`) | **Use the darks** or **Don't use** | Don't use |
| **Fix isolated hot pixels** (`--no-hot-pixels`) | Single hot pixels cleaned from the frames themselves; only when Darks is Don't use (the box is unticked and greyed out while darks are used) | On |
| **Bias** (`--bias`) | **Use the bias frames** or **Don't use** (the black level written in each file) | Use |

### Vignetting

Only one correction is applied to a frame:

* **Master flat**: the flats are averaged and every frame is divided by
  that, pixel by pixel. This corrects vignetting *and* dust shadows,
  provided nothing was changed or cleaned between the eclipse and the flats.
* **Profile from the flats**: a smooth surface is fitted to the flats (a
  polynomial in x and y, each colour on its own, dust left out of the fit)
  and used instead. It follows a lens whose darkening is off-centre.
  Vignetting only, so dust that moved since cannot add dark rings.
* **Master flat + compare**: the master flat, and the log reports how far it
  differs from that smooth surface, with a warning when the difference is
  large. The difference is saved as `<name>_flat_vs_profile_<settings>.fit`
  (dark spots are dust).
* **Lensfun profile**: the lens's lensfun profile (or a profile saved from
  your own flats with `totality flat-profile`, which comes first); the flats
  are not used.
* **None**: no vignetting correction at all.

A straight slope across the flats is measured and named in the log
("brighter on the right (3.2 % at the edge)"). It is kept: from flats alone,
a light source brighter on one side and a lens whose darkening is a little
off-centre give the same slope, and taking out the lens's leaves the sky
too dark on one side and too bright on the other. If you know the light was
uneven, tick **The light on the flats was uneven** and the slope is taken
out.

Flats that match no light frame are named in the log and not used.

**A camera on a telescope, or a manual lens.** Many such frames do not
record the f-number or focal length. Nothing needs them:

* without an f-number, every frame is taken to be at the same f-number, so
  exposure time and ISO set how bright each one is (the Frames tab report
  says so);
* flats with no lens data match light frames with no lens data, as taken
  through the same optics (but never frames that record other settings);
* without a focal length, Align searches for the Moon at every size.

If you know them, you can type the **Focal length** and **f-number** on the
Frames tab (`totality inventory --focal-length --f-number`). They are used
only for frames and flats whose files leave them out, never over a recorded
value. lensfun has no telescope profiles, so a telescope's vignetting comes
from your own flats.

### Darks and bias

Put the camera's RAW files in the folders as they are, any number of them;
the toolkit averages them itself (pre-stacked masters are not read from
these folders).

* **Darks** (when **Darks** is Use the darks) are matched by exposure time
  and ISO and averaged into a master dark, which is subtracted (it includes
  the bias and removes hot pixels, thermal signal and amp glow). Frames
  without a match use the bias.
* **Bias** frames (when **Bias** is Use the bias frames) are matched by ISO
  and averaged into a master bias. Without them, each file's own black level
  is used.

A dark only corrects light frames with its own exposure time and ISO, so
for a bracket shoot darks at each shutter speed (10-20 each is plenty).

**Masters are built once.** The first Stack (or earthshine stack) saves each
master bias, dark and flat as a FITS file in `fits/masters` inside the
project folder (they open in Siril), and later runs read them back instead of the
frames. A master is built again when a file in its group is added, removed
or replaced, and a master flat also when the bias taken off it changes; the
old one is then deleted. The log says which masters were reused. Each is
the size of one frame in 32-bit (about 180 MB for a 45-megapixel camera).

### What the log tells you

Stack logs which correction every frame got ("vignetting: 48 frame(s) by
master flat (f/8, 400 mm)"), how much darker the corners were ("the corners
are 62 % as bright as the middle (-0.69 stops)"), and saves the correction
for the reference frame as `<name>_vignetting.fit` at 1/8 size (open it in
Siril to see what was divided out; 1.0 = the middle). When nothing matched
it says "NONE applied". The folders chosen on the Frames tab are saved with
the project as soon as a step runs. The earthshine stack is calibrated
exactly as the HDR stack was, so the two stay in the same units. Align uses no
vignetting correction at all: the darkening is the same in every frame and
smooth, and phase correlation compares the corona's fine structure. On the
2024 frames a typical correction moved the alignments by up to 1 px and put
the Moons further off their straight track (rms 1.34 px with it, 0.76 px
without).

## Own-flats profiles

When lensfun has no vignetting data for your lens, shoot flats and fit a
profile once:

1. Same lens, focal length, aperture and focus (infinity) as the eclipse
   frames; any ISO; any camera with the same sensor size.
2. An even light filling the frame: a light panel, a white screen, or a
   twilight sky through a white T-shirt. Expose so the histogram peak is
   near the middle. 20 to 30 frames.
3. Run `totality flat-profile FLATS`. It fits a smooth profile for each
   colour and saves it to your profile library
   (`~/.solar-eclipse-toolkit/profiles`, or `TSE_PROFILE_DIR`). If the flats
   do not record the lens, give it with `--lens`, `--focal` and `--aperture`;
   a profile is matched to frames by focal length and f-number, so for frames
   that do not record them, type them on the Frames tab too.

The profile keeps only the lens's fall-off: dust in the flats is ignored,
and a light that is brighter on one side is taken out. Because the profile
is measured on the sensor in millimetres, one set of flats serves any camera
with the same lens (a D800 and a D850, say). `totality inventory` then picks
it up on its own. A FITS master flat stacked in Siril can be used instead of
RAW flats.

## Command-line reference

Every option of every command, as `totality <command> --help` gives it.
Generated from the program by `python tools/cli_reference.py`; a test checks it is current.

<!-- cli-reference:start -->
### `totality inventory`

read every RAW file in a folder and report brackets and calibration.

| Option | What it does |
| --- | --- |
| `folder` | the folder of light frames, or a project folder holding lights/ (and optionally darks/, flats/, bias/); with a project folder the project file and every result go in it |
| `--recursive` | also read sub-folders |
| `--csv` | write the frame table to this CSV file |
| `--project` | create or update this project file (.json) |
| `--lens` | force this lensfun lens profile |
| `--focal-length` | focal length in mm, used only for frames (and flats) whose files do not record it, e.g. a camera on a telescope; optional, nothing needs it (saved in the project; 0 clears) |
| `--f-number` | f-number (focal ratio), used only for frames (and flats) whose files do not record it; optional: without it every frame is taken to be at the same f-number (saved in the project; 0 clears) |
| `--no-exiftool` | do not use exiftool even if installed |
| `--eclipse` | the eclipse, by its date, YYYY-MM-DD (default: the frames' date) |
| `--lat` | site latitude in degrees, north positive (saved in the project) |
| `--lon` | site longitude in degrees, east positive (west is negative) |
| `--elev` | site elevation in metres |
| `--camera-utc-offset` | the camera clock's time zone in hours, e.g. -6, when the files do not record one |
| `--clock-error` | seconds the camera clock was behind true time (true minus camera), when known |
| `--c2` | second contact typed in, YYYY-MM-DDThh:mm:ss, in the camera's clock (or in true time with --clock-offset); default: computed from the eclipse and site |
| `--c3` | third contact typed in (with --c2) |
| `--clock-offset` | with --c2/--c3 typed in another clock: add this to the camera clock to get that clock, e.g. +1:04:08, -45s, 1h |
| `--bias` | folder of bias frames |
| `--darks` | folder of dark frames |
| `--flats` | folder of flat frames |
| `--lensfun-db` | folder holding lensfun XML files |
| `--profile` | use this own-flats vignetting profile (.json) for every frame |
| `--profile-dir` | folder of saved own-flats profiles (default: your profile library) |

### `totality flat-profile`

fit a smooth vignetting profile from your own flat frames.

| Option | What it does |
| --- | --- |
| `source` | folder of RAW flats, or a FITS master flat |
| `--name` | a name for the profile |
| `--output`, `-o` | save to this .json file instead of the profile library |
| `--block` | block size in pixels for the fit (default 32) |
| `--crop` | crop factor of the camera that took the flats (default: from lensfun, else 1.0) |
| `--lens` | lens name to store, if the files do not say |
| `--focal` | focal length to store, if the files do not say |
| `--aperture` | aperture to store, if the files do not say |
| `--lensfun-db` | folder holding lensfun XML files (default: the installed database) |

### `totality lens`

find a lensfun profile and show its vignetting.

| Option | What it does |
| --- | --- |
| `name` | lens name or part of it (empty: list the lenses lensfun knows) |
| `--focal` | focal length, mm |
| `--aperture` | f-number |
| `--crop` | camera crop factor (1.0 = full frame) (default 1.0) |
| `--lensfun-db` | folder holding lensfun XML files (default: the installed database) |

### `totality lensfun-update`

download the lensfun database from GitHub.

| Option | What it does |
| --- | --- |
| `--dest` | folder to put the database in (default: the toolkit's own data folder) |
| `--from` | install from a downloaded lensfun .zip or a folder instead |

### `totality frame`

show everything read from one RAW file (for troubleshooting).

| Option | What it does |
| --- | --- |
| `file` | one RAW file |
| `--lensfun-db` | folder holding lensfun XML files (default: the installed database) |

### `totality eclipse`

eclipse circumstances (contacts, duration) for a site.

| Option | What it does |
| --- | --- |
| `--date` | eclipse date, YYYY-MM-DD |
| `--lat` | latitude, degrees north |
| `--lon` | longitude, degrees east (west is negative) |
| `--elev` | elevation, metres |
| `--utc-offset` | also show local times at this UTC offset, hours |
| `--delta-t` | TT-UT in seconds (default: the measured or predicted value) |
| `--elements` | JSON file with Besselian elements for other eclipses |

### `totality frames`

list a project's frames, or exclude / include some.

| Option | What it does |
| --- | --- |
| `project` | the project file (.json) |
| `--exclude` | leave these frames out (file names) |
| `--include` | use these frames again |

### `totality elements-update`

download NASA's Besselian elements for every eclipse (1600-2400).

| Option | What it does |
| --- | --- |
| `--from` | use a downloaded copy of NASA's canon CSV instead |

### `totality align`

align the light frames of a project on the corona and find the Sun's centre.

| Option | What it does |
| --- | --- |
| `project` | project file made by 'totality inventory --project' |
| `--lat` | observing site latitude (saved in the project) |
| `--lon` | observing site longitude, east positive (saved in the project) |
| `--elev` | observing site elevation, metres |
| `--clock-error` | seconds to add to the camera's UT to get true UT, if known (saved in the project); otherwise bounded from totality |
| `--camera-utc-offset` | camera time zone, hours, when the files carry none |
| `--mirrored`, `--no-mirrored` | the optics mirror the image (telescope with a star diagonal); saved in the project |
| `--window` | analysis window in superpixels (default: about 8 solar radii) |
| `--rotation` | field rotation: smooth drift over time (default), per frame, or none |
| `--any-angle` | search any rotation and scale (alt-az mounts) |
| `--rho` | tangential filter length in superpixels (default: from the Sun's size) |
| `--output`, `-o` | alignment results file (default: json/<project>_align.json in the project folder) |
| `--profile-dir` | folder of saved own-flats profiles |
| `--lensfun-db` | folder holding lensfun XML files (default: the installed database) |
| `--quiet` | print only the result, not each step |

### `totality stack`

merge the aligned frames into one linear HDR image.

| Option | What it does |
| --- | --- |
| `project` | project file (run 'totality align' on it first) |
| `--align` | alignment results file (default: the one saved by 'totality align') |
| `--output`, `-o` | folder for the pictures (default: next to the project); the .fit and .json files always go in the fits and json folders inside the project folder |
| `--name` | output file name prefix (default: the project name) |
| `--square` | quicker run: output only a square 10 solar radii wide around the Sun (default: the full frame, the same size as the RAW files) |
| `--radii` | like --square, but this many solar radii from the centre to the edge |
| `--moon-frame` | frame number whose Moon (and earthshine) is kept (default: the alignment reference) |
| `--include-rejected` | also stack frames the Sun-centre fit flagged as off the Moon's track |
| `--no-hot-pixels` | skip the hot-pixel correction |
| `--vignetting` | vignetting correction: auto = master flat if flats are chosen, else the lensfun profile (default); master = master flat (flats from the same session: vignetting and dust); profile = a smooth surface fitted to the flats (flats from another time: vignetting only); compare = master flat and report how it differs from that surface; lensfun = the lens's lensfun profile; none = no correction |
| `--lens` | lensfun lens for --vignetting lensfun/auto (default: matched from the files) |
| `--flats-uneven-light` | the light on the flats was brighter on one side: take a straight slope across them out (off by default: an off-centre lens gives the same slope) |
| `--darks` | use the project's dark frames (hot pixels, thermal signal, amp glow; the separate hot-pixel fix is then off), or not (default) |
| `--bias` | use the project's bias frames for the offset (default), or the black level in each file |
| `--profile-dir` | folder of saved own-flats profiles |
| `--lensfun-db` | folder holding lensfun XML files (default: the installed database) |
| `--quiet` | print only the result, not each step |

### `totality enhance`

reveal coronal structure: NRGF, FNRGF, adaptive filter, natural, blend.

| Option | What it does |
| --- | --- |
| `project` | project file (run 'totality stack' on it first) |
| `--settings` | Enhance settings file to start from (saved with 'Save settings...' on the Enhance tab, or the <name>_enhance.json in the project's json folder); options given here as well override the file |
| `--hdr` | HDR FITS file (default: the one saved by 'totality stack') |
| `--output`, `-o` | folder for the pictures (default: next to the project); the .fit and .json files always go in the fits and json folders inside the project folder |
| `--name` | output file name prefix (default: the project name) |
| `--layers` | comma-separated layers to write, from natural, nrgf, fnrgf, fnrgf_n, adaptive, achf (default: all but achf, which takes minutes on a full frame) |
| `--blend` | suggested blend as layer=weight pairs, each optionally ':overlay' or ':linear' (combined as each method's own publication does: NRGF, FNRGF and FNRGF-N as Druckmuellerova, Morgan & Habbal 2011, typically 0.3-1; adaptive as Morgan & Druckmueller 2014, 0.43 for their mix; achf as Druckmuellerova's thesis 2013, added to the linear stack before the natural stretch, typically 20-50 with the log stretch, about twice that with gamma) or, for adaptive and achf only, ':average' (a gentler look, typically 0.4-0.6); default soft light, e.g. natural=1,fnrgf_n=0.7,adaptive=0.1:overlay; or 'none' (default: natural=1.0) |
| `--order` | FNRGF Fourier order (default 10) |
| `--segments` | FNRGF angular segments (default 50) |
| `--mean-att` | FNRGF attenuation A0,A1,... for the mean (default 1,0.9,...,0) |
| `--std-att` | FNRGF attenuation S0,S1,... for the deviation (default 1,0.9,...,0) |
| `--noise-m` | FNRGF-N noise compensation, 0..0.99 (default 0.7) |
| `--adaptive-k` | adaptive layer: strength of the local contrast at each scale, k in arctan(k C) (default 0.7, as in Morgan & Druckmueller 2014) |
| `--achf-scales` | Adaptive (ACHF): the size (kernel sigma) of each of its six scales, finest first, in % of the Sun's radius (default 0.121,0.242,0.7259,2.42,10,20; the last two start at strength 0) |
| `--mgn-scales` | Adaptive (MGN): the width (kernel sigma) of each of its six scales, finest first, in px of the full-size stack (default 1.25,2.5,5,10,20,40, as Morgan & Druckmueller 2014; a frame of reference: a 400 mm lens on a 45-megapixel full-frame camera; raise them with a longer lens or smaller pixels) |
| `--mgn-weights` | Adaptive (MGN): the weight g_i of each scale, 0-1, finest first (default 1,1,1,1,1,1: the plain mean); the layer is 1/6 of the weighted sum, as in Morgan & Druckmueller 2014, so lowering one weakens it; 0 leaves the scale out |
| `--achf-weights` | Adaptive (ACHF): strength of each of its six scales, finest first; 0 leaves a scale out (default 1,1,1,1,0,0) |
| `--achf-part-limit` | Adaptive (ACHF): a pixel this many times brighter or darker than the corona at its height (a prominence) is a separate part, left out of the corona's kernels and filtered with its own pixels only, so no dark fringe forms round it; 0 = off (default 2) |
| `--achf-contrast` | Adaptive (ACHF): how strongly the layer shows its detail with soft light, overlay or average: at 1, a pixel about 1.1% brighter than its surroundings shows at 82% grey; raise it (1.5-3) if the layer looks flat. With achf on ':linear' it multiplies that blend weight instead: leave it at 1 there (default 1) |
| `--noise-gate` | adaptive layer: each scale's detail counts only as far as it stands above the stack's noise at that scale (e.g. 3); 0 = off |
| `--even` | take out this fraction of the large brightness differences round the Sun (streamers against coronal holes); 0 = off, e.g. 0.7 |
| `--midtone` | put the inner corona's median brightness at this level with a midtones transfer curve; 0 = off, e.g. 0.6 |
| `--shoulder` | highlight roll-off: 0 = hard white point (the brightest 0.1% clip); e.g. 0.3 = the top 30% of the range rolls off smoothly to white, nothing clips and colours keep their hue |
| `--natural-stretch` | natural layer: log brightness (default) or gamma, the base Morgan & Druckmueller 2014 put under their MGN detail (darker on an eclipse: use --gamma-white 99.9) |
| `--gamma` | gamma of --natural-stretch gamma (3.2 as in Morgan & Druckmueller 2014; 2.5-4 suggested) (default 3.2) |
| `--gamma-black` | --natural-stretch gamma: black point as a percentile of the stack; 0 = its minimum (default, as in Morgan & Druckmueller 2014); 0.5-2 if the sky looks grey |
| `--gamma-white` | --natural-stretch gamma: white point as a percentile; 100 = the maximum (default, as in Morgan & Druckmueller 2014), 99.9 recommended for eclipses (the brightest 0.1% goes white and the corona comes up) |
| `--flatten` | natural layer: how much of the radial fall-off to remove, 0..1; 0 = off (default), e.g. 0.8 |
| `--saturation` | colour saturation, 0 = monochrome (default 1) |
| `--white` | white balance: inner corona neutral (default), the camera's as-shot, or none |
| `--format` | picture type: 16-bit TIFF (default), 16-bit PNG or 8-bit JPG; all 300 dpi |
| `--keep-sky`, `--keep-background-colour` | do not model and remove the sky background (level, colour and gradient) |
| `--quiet` | print only the result, not each step |

### `totality earthshine`

Moon-aligned stack of the long exposures: a layer for the black lunar disc.

| Option | What it does |
| --- | --- |
| `project` | project file (run 'totality stack' on it first) |
| `--stops` | use frames within this many stops of the longest exposure (default 6) |
| `--no-limb-glare` | do not model the steep glare just inside the limb (shows as a bright rim) |
| `--prominence-glare` | fit and remove the fan of glare (and spikes) round a bright prominence (default off: the fan is shown as the data has it; removing it cannot bring back what is under it) |
| `--no-prominence-glare` | leave the glare fan in (the default) |
| `--limb-shape` | fit the limb glare as light fading inward, round the limb, which keeps maria by the limb (default off) |
| `--no-limb-shape` | no limb shape (the default) |
| `--channel` | colour channels to make the earthshine from (default auto: chosen on the stack; the red channel carries most of the chromosphere glare) |
| `--glare-order` | smoothness of the glare removed from the disc, 0-6 in half steps: higher follows uneven glare better but flattens the maria more (default 3.5; 2 = a gentle gradient, 0 = one even level; a half step averages the two neighbouring polynomial orders) |
| `--smooth` | smoothing of the layer in pixels (default 0 = none) |
| `--brightness` | brightest lunar surface in the layer, 0-1 (default 1) |
| `--keep-glare` | earthshine layer file for Photoshop (<name>_earthshine.tif): how much of the removed glare to keep as a glow inside the limb, 0-1 (default 0.1) |
| `--limb-soften` | earthshine layer file for Photoshop: extra softening of the limb in px, on top of the real one (default 0 = none) |
| `--no-colour` | make the earthshine layer grey (default: natural colour, so a prominence at the limb stays red on the Moon side too) |
| `--limb-colour` | colour of the glare just inside the limb: recorded (red from the chromosphere, as the camera saw it), no-red (the colour of the Moon's middle), neutral (no red, and the whole Moon in the corona's white instead of earthshine's own slightly blue colour) (default recorded) |
| `--edge-fill` | fill at least this % of the Moon's radius just inside the limb from further in (default 0: only where the noise is over 10x the middle's; 2.5 = 10.9 px on a 400 mm Moon) |
| `--no-edge-grain` | leave the filled last pixels by the limb smooth (default: grain like the rest) |
| `--surface-denoise` | edge-preserving smoothing of the lunar surface, px (0 = off; e.g. 4): maria keep crisp edges |
| `--edge-smooth` | most smoothing of the noisier pixels near the limb, px (default 0 = none; e.g. 3) |
| `--include-rejected` | also use frames marked as rejected |
| `--output`, `-o` | folder for the pictures (default: next to the project); the .fit and .json files always go in the fits and json folders inside the project folder |
| `--name` | output file name prefix (default: the project name) |
| `--format` | picture type: 16-bit TIFF (default), 16-bit PNG or 8-bit JPG; all 300 dpi |
| `--profile-dir` | folder of saved own-flats profiles (default: your profile library) |
| `--lensfun-db` | folder holding lensfun XML files (default: the installed database) |
| `--quiet` | print only the result, not each step |

### `totality final`

the finished image: Moon and corona stretched together from the stacks.

| Option | What it does |
| --- | --- |
| `project` | the project file (.json) |
| `--paste` | another way: paste the Moon as Blend draws it into an enhanced layer or --base-file, inside the corona layers' black disc, with a blend mode (the earthshine layer itself before any Enhance layers are written) |
| `--base` | enhanced layer to use as the corona (default: blend, the last one written) |
| `--base-file` | use this image as the corona instead (the size of the HDR stack) |
| `--strength` | brightness of the lunar surface, 1 = as made by the earthshine step (default 1) |
| `--mode` | blend mode (default screen) |
| `--join` | how Moon and corona meet: corona-first (the HDR stack wherever the Blend has it, the earthshine stack elsewhere in the disc, black where neither has data) or rebuilt (the older way: the band just outside the limb rebuilt from both stacks) (default corona-first) |
| `--moon-layer` | write only the Moon as --base draws it (with --strength), transparent elsewhere, for Photoshop: above that layer in Normal mode at 100 % it gives the final image (default <name>_moon.tif; TIFF or PNG) |
| `--name` | output file name prefix (default: the project name) |
| `--output`, `-o` | output file (.tif, .png or .jpg); default <name>_final.tif |
| `--quiet` | print only the result, not each step |

### `totality prominence-stack`

stack the shortest exposures round the Sun; 'totality final' then sharpens the HDR stack's prominences by as much as they show.

| Option | What it does |
| --- | --- |
| `project` | project file (run 'totality stack' on it first) |
| `--longest-stops` | frames from the shortest aligned exposure up to this many stops longer (default 2) |
| `--radii` | the square stacked, in solar radii from the Sun's centre (default 1.6) |
| `--settings-only` | only save the settings below for 'totality final' (no stacking) |
| `--off` | 'totality final' leaves the HDR stack's prominences as they are |
| `--sharpen` | 0.5: as sharp as the short frames (default); 1: also undo the blur measured on the Moon's edge; 0 off |
| `--structure` | bring out strands and loops inside them: 0 off (default), 0.3-0.6 more texture |
| `--brightness` | their brightness against the corona (default 1, as measured; 0.2-0.5 shows their structure) |
| `--min-redness` | how much redder than the corona light must be to count as a prominence (default 1.1) |
| `--no-chromosphere` | leave out the thin pink rim along the limb |
| `--chromosphere-arcsec` | height of the rim left out by --no-chromosphere, arcsec (default 12) |
| `--quiet` | print only the result, not each step |

### `totality prominences`

find the prominences and write them as a layer (transparent elsewhere).

| Option | What it does |
| --- | --- |
| `project` | project file (run 'totality stack' on it first) |
| `--height` | how far above the solar limb to look, in solar radii (default 0.3) |
| `--sensitivity` | red excess needed, in noise standard deviations (default 4; lower finds fainter ones) |
| `--min-redness` | how much redder than the corona light must be (default 1.1) |
| `--no-chromosphere` | leave out the thin pink rim along the limb |
| `--chromosphere-arcsec` | height of the rim left out by --no-chromosphere, arcsec (default 12) |
| `--brightness` | brightness of the layer (1 = as measured) (default 1.0) |
| `--saturation` | colour saturation of the layer (1 = as measured) (default 1.2) |
| `--output`, `-o` | layer file (.tif, .png or .jpg); default <project>_prominences.tif |
| `--quiet` | print only the result, not each step |

### `totality final-prominences`

put the prominence layer on top of the final image.

| Option | What it does |
| --- | --- |
| `project` | the project file (.json) |
| `--base` | final: the image from 'totality final' (default); blend or natural: the last-written layer |
| `--base-file` | put them on this image instead (the size of the HDR stack) |
| `--strength` | opacity (normal) or brightness (screen, lighten) (default 1.0) |
| `--mode` | normal (default) keeps their colour over the bright inner corona |
| `--output`, `-o` | output file (.tif, .png or .jpg); default <project>_final_prominences.tif |
| `--quiet` | print only the result, not each step |

### `totality doctor`

show which optional readers are available.
<!-- cli-reference:end -->

## Tests

For anyone changing the code:

```bash
pip install -e ".[dev]"
pytest
```

The tests use synthetic eclipses with known geometry (a moving Moon, shutter
errors, seeing, earthshine, prominences) and real 2024 data from Lake
Wappapello, binned down to keep the repository small (`tests/data`). The
real data catches what synthetic data missed: non-image edge columns, an
uneven twilight sky, corona reaching the frame edges, dust, stars, and the
glare round the Moon's limb. The filter layers are checked against their
publications' own formulas, and a detail box against the full-size result.
Development uses only the 2024 data; the 2017 set is held back as a blind
test of the finished tool and is never added to the tests.

Set `TSE_SAMPLE_RAWS` to a folder of sample RAW files to also test the
metadata readers on real files. The Siril window is tested without a
display through a small stand-in for PyQt6 (`tests/qtstub`;
`PYTHONPATH=src:tests:tests/qtstub python tests/gui_smoke.py <folder>` drives
it through whole steps). Layout, painting and the mouse are only checked by
running it in Siril.

## Licence

MIT, see `LICENSE`.

The lensfun database is not bundled; it is downloaded from the lensfun
project (CC BY-SA 3.0). `tests/data/lensfun_subset.xml` is a short excerpt
used only by the tests.
