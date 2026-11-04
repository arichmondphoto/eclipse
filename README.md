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
filter and the Sun's centre), Morgan, Habbal & Woo (2006, NRGF) and Morgan &
Druckmüller (2014, multi-scale Gaussian normalization).

Works on any total solar eclipse. Eclipse geometry comes from NASA's Besselian
elements for any eclipse from 1600 to 2400.

## Status

| Step | Scope | State |
| --- | --- | --- |
| 1 Frames | RAW inventory, bracket detection, lensfun or own-flats vignetting, calibration planning, project folders | done |
| 2 Align | Lunar limb, phase correlation alignment, field rotation, the Sun's centre | done |
| 3 Stack | Moon-aware linear HDR stacking with measured brightness matching | done |
| 4 Enhance | Natural, NRGF, FNRGF, FNRGF-N and adaptive layers; Soft Light / Overlay blend | done |
| 5 Earthshine | Moon-aligned stack of the long exposures; glare removed; the Moon put into the HDR stack and stretched with the corona | done (checked on the 2024 data) |
| 6 Prominences | Found by colour at the limb; transparent layer put on top | done (checked on a reduced copy of real data) |

Each step works in the Siril window and on the command line. Tested on the
2024 eclipse (Nikon D850, 400 mm); the 2017 set is kept back as a blind test.
Still to come: the blind test, and help for other cameras and operating
systems before sharing the tool more widely.

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
6. **Prominences:** pick out the prominences and put them on top.

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
2. Run **Scripts → SolarEclipseToolkit**. On first run Siril installs
   PyQt6, numpy, scipy, exifread, rawpy, tifffile, Pillow, certifi and
   truststore into its own Python environment.

The script finds the core package in `../src`, so keep the folders together.

To update, pull the repository (`git pull`) and run the script again. The
window title shows the version and the commit it is running (for example
`Solar Eclipse Toolkit 0.1.0 (8e54fa8)`), so you can check an update took
effect.

The window has one tab per step, and each step saves into the project file:

| Tab | What you do | What you see |
| --- | --- | --- |
| 1 Frames | Read the RAW folder, check the frames and brackets. **Event:** the eclipse, the site (latitude, longitude, elevation), the camera time zone and clock error if known; **Compute contacts** fills in second and third contact. **Calibration:** the lens profile and the calibration folders | Frame table (right-click to exclude a frame), the contacts and the check of your brackets against them, bracket and calibration report |
| 2 Align | Whether the optics mirror the image; the rotation model (the event is shown, from the Frames tab) | Per-frame shift, rotation and peak strength; the Sun centre report |
| 3 Stack | Full frame or a square around the Sun; which frame's Moon to keep; how the flat frames are used (master flat, smooth profile, or master flat + compare) | The stacked preview and each frame's measured brightness |
| 4 Enhance | Layers to write; each layer's blend strength (all 0 % by default) and Soft Light or Overlay; **Natural layer and colour:** flatten, highlight roll-off, evenness, midtone (all off by default), saturation, white balance, sky removal; **Adaptive layer (MGN):** strength (0.7, the paper's) and noise gate (off); **Adaptive (ACHF)** as a layer (not ticked by default: slow) and the size and strength of each of its six scales (0.5, 1, 3, 10 px on the 2024 frames at strength 1; two larger scales at 0); FNRGF order, segments and noise compensation | A live preview on a reduced copy (about 1400 px) that updates as you change settings; then **Write full-size layers** |
| 5 Earthshine | 1 Stack the Moon: frames within N stops, colour channels, frames off the Moon's track. 2 Lunar surface: glare smoothness, limb glare, glare round a bright prominence, maria by the limb, optional albedo map and its guidance, smoothing (overall and by the limb), edge grain, brightness, darkening towards the limb and its shape, natural colour; for pasting only, kept glare and limb softening. 3 Final image: Blend or Natural (stretched with the Moon, using your Enhance settings) or paste into a file, lunar surface strength, blend mode when pasting | The Moon close up, then a preview of the final picture; **Write final image** |
| 6 Prominences | Height, sensitivity, redness, chromosphere on or off, brightness, saturation; then the image to put them on, strength and mode | The prominences over a dim view of the inner corona, then a preview of the picture; **Write final image** |

Every setting has hover help explaining what it does and when to change it.
Every picture view zooms: scroll (or **+** / **-**) to zoom about the pointer,
drag to move round, double-click for 100% at that point (again for fit),
**0** fit, **1** 100%; the zoom shows in the corner. A preview run again with
other settings keeps the zoom and place, so settings can be compared on the
same detail. Previews are reduced copies (about 1400 px), so 100% there is a
preview pixel, not a full-size one.
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
calibration frames from the other subfolders, saves the project file as
`Eclipse 2024/Eclipse 2024.json`, and writes every result (alignment,
stack) into the folder. Opening that folder again reopens the project.
Wherever a picture is written (enhanced layers, earthshine and prominence
layers, final images), the GUI asks where to save, under what name and in
which format, starting in the project folder.

## Command line

The command is `totality` (it was `tse` before 31 Oct 2026; run
`pip install -e .` again after updating so the new name is installed).

A whole event, from a project folder:

```bash
totality inventory "Eclipse 2024"                       # reads lights/, writes "Eclipse 2024/Eclipse 2024.json"
P="Eclipse 2024/Eclipse 2024.json"
totality align "$P" --lat 36.9422 --lon -90.3343 --elev 110
totality stack "$P"
totality enhance "$P" --blend natural=1,fnrgf_n=0.7,adaptive=0.1:overlay
totality earthshine "$P"
totality final "$P" -o "Eclipse 2024/with_moon.tif"
totality prominences "$P"
totality final-prominences "$P" -o "Eclipse 2024/finished.png"
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
   significant. Measuring and correcting repeat until nothing significant is
   left; each frame's note gives the total rotation applied (deg/min and over
   the sequence) and what was left afterwards. Use `--rotation per-frame` or `--any-angle` for mounts that
   do not track.
4. **Finds the Sun's centre** (Druckmüllerová et al. 2011, appendix): the
   Moon's track across the aligned frames is fitted to the eclipse
   geometry computed for your site from NASA Besselian elements. This also
   gives the image scale and where celestial north points. If the camera
   clock was wrong, the clock error is bounded by requiring every frame to
   fall inside totality and the remaining along-track uncertainty is
   reported; pass `--clock-error` if you know it. Without it, the clock
   error used is the middle of that allowed range: it comes from the frames'
   times and the eclipse's times at your site, not from the Moon's track.

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
pixels near the edge of the path. Use `--mirrored` for mirrored optics; the
tool reports both answers and warns when the difference matters.

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
camera's clock. Projects from before November 2026 kept typed contacts with a
separate clock offset; they are converted on opening (same frames inside).
Bracket detection and Moon tracking only use time differences, so a wrong
clock never affects them.

## HDR stacking (Phase 3)

Each frame leaves out the corona just round its own Moon (the limb is
blurred there and the Moon moves between frames). That margin is scaled to
the Moon's size in the image: 0.7% of its radius, at least 2 px, so 3 px for
the 2024 D850 at 400 mm and about 7.5 px at 1000 mm (`moon_margin_px` sets
it by hand). The log names it.

```bash
totality stack eclipse2024.json            # uses the alignment saved by 'totality align'
totality stack eclipse2024.json --square   # quicker: a square 10 solar radii wide
```

Every frame is read again at full resolution and calibrated with the
frames chosen on the Frames tab (see [Calibration rules](#calibration-rules)):
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
are used instead and the log says so.

**Merging.** Each pixel is the inverse-variance weighted mean of the frames'
radiance estimates. Weights fall to zero approaching saturation and inside
each frame's own Moon (plus 3 px), so the moving Moon never darkens the
inner corona. Inside the Moon as it was in the Moon frame (the alignment
reference unless `--moon-frame` says otherwise) the result is that frame's
data, which keeps the lunar limb sharp. Pixels that are saturated or covered
in every frame (rare with 1/3-stop brackets) take the shortest exposure that
shows them and are counted in the report. Rows and columns at the frame edge
that are not image (a D850's last column reads several times too bright) are
detected and given zero weight.

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

A full 45-megapixel frame needs about 3-5 GB of free memory at the peak.

| File | Contents |
| --- | --- |
| `<name>_hdr.fit` | linear 32-bit RGB in units of the reference exposure, camera colour; header has `SUNX/SUNY/SUNR`, `MOONX/MOONY/MOONR`, `PIXSCALE`, `NORTHANG` and the crop origin (0, 0 for the full frame) |
| `<name>_weight.fit` | sum of weights per pixel (where the data came from) |
| `<name>_preview.tif` | 16-bit asinh stretch with the camera white balance, for a quick look only |
| `<name>_stack.json` | gains, offsets and settings |

## Enhancement (Phase 4)

```bash
totality enhance eclipse2024.json          # uses the HDR stack saved by 'totality stack'
```

Writes one picture per layer (16-bit TIFF by default; `--format png|jpg`,
`-o` folder, `--name` prefix), all the same size and lined up pixel for
pixel, ready to stack as layers in Photoshop, plus a suggested blend:

| Layer | What it shows |
| --- | --- |
| `natural` | log stretch; `--flatten` (off by default, e.g. 0.8) removes that much of the radial fall-off |
| `nrgf` | normalizing-radial-graded filter (Morgan et al. 2006): each circle around the Sun normalized to the same mean and spread |
| `fnrgf` | Fourier NRGF (Druckmüllerová et al. 2011): mean and spread follow position angle too, so faint and bright regions at the same height get similar contrast |
| `fnrgf_n` | FNRGF-N: the same with noise compensation (`--noise-m`, default 0.7); the noise is estimated automatically from the stack's weight map |
| `adaptive` | multi-scale Gaussian normalization (Morgan & Druckmüller 2014) of the log image: fine structure at all scales and orientations, the published successor to the adaptive circular high-pass filter |
| `achf` | **Adaptive (ACHF)**: Druckmüller's adaptive circular high-pass filter (Druckmüller, Rušin & Minarovjech 2006; defined in Druckmüllerová 2013): the log image minus its local mean under a kernel that follows the circles round the Sun, at its own six scales (size and strength each), Moon left out. Written only on request (`--layers ...,achf`): a few minutes on a full frame |
| `blend` | suggested mix: the natural layer as the base with the filter layers applied in Soft Light (or Overlay) at the given opacities; by default every filter layer is at 0 (`--blend natural=1`), so the blend is the natural layer until you add layers, e.g. `--blend natural=1,fnrgf_n=0.7,adaptive=0.5`; add `:overlay` to a layer for Overlay, e.g. `--blend natural=1,adaptive=0.1,fnrgf=0.05:overlay` (`--blend none` to skip); the same thing you would do with layers in Photoshop |

**Names used:** NRGF = Normalizing-Radial-Graded Filter; FNRGF = Fourier
Normalizing-Radial-Graded Filter; FNRGF-N = FNRGF with Noise compensation;
MGN = Multi-scale Gaussian Normalization; ACHF = Adaptive Circular
High-pass Filter; HDR = High Dynamic Range (the merged exposures).

The layers are already stretched: each filter maps the linear data to
display brightness itself (a log stretch for `natural` and `adaptive`, the
filter's own normalization and a soft S-curve for the NRGF family). Run
`totality enhance` on the linear HDR stack, never on a stretched copy, and use
only gentle curves or levels on the results.

In Photoshop: `natural` at the bottom in Normal mode, `fnrgf_n` above it in
Soft Light (40-70 %), `adaptive` above that in Soft Light (20-40 %); Overlay
is a stronger alternative at lower opacity. The filter layers are centred on
mid-grey, which both modes treat as "no change"; desaturate them so the
colour comes from the natural layer.

Before filtering, the sky (twilight glow, its gradient across the frame,
vignetting leftovers and black-level offsets) is modelled per colour channel
as a smooth surface fitted to the darkest parts of the frame away from the
inner corona, and removed. Radial terms in the fit take up the faint outer
corona so it is not mistaken for sky. The natural layer then keeps the sky
dark: it stops flattening where the corona sinks into the noise
(`--keep-sky` skips the sky model).

The Sun's centre and radius come from the stack header, so the circles are
centred on the true Sun, not the Moon. Circles that leave a rectangular
frame are completed from further in, a thin border at the frame edge is
left out, and dust shadows and stars beyond 2 solar radii are kept out of
the circle statistics, so none of these show as arcs or rings.

Colour: by default the inner corona (1.2-2.5 solar radii) is made neutral,
since K-corona light is scattered sunlight; `--white camera` uses the
camera's as-shot balance, `--saturation 0` gives monochrome. Colour is faded
to grey where the signal is buried in noise or no brighter than the
uncertainty of the removed sky. The Moon is black in these layers; the
Earthshine step fills it.

Highlights: by default the natural layer and the blend set a white point
at the brightest 0.1%, which clips the inner corona by the limb and bright
prominences to flat white or flat pink. `--shoulder 0.3` (**Highlight
roll-off** on the Enhance tab) rolls the top 30% of the range off smoothly
instead: nothing clips, and colours keep their hue. On the 2024 stack with
`--flatten 1` it took the clipped share of the inner corona (out to 2.5 Moon
radii) from 1.5% to 0%.

Tone: `--even 0.7` (**Evenness**) takes out that fraction of the large
brightness differences round the Sun (streamers against coronal holes, over
about a quarter of the Sun's radius) as `--flatten` does for the fall-off
from the Sun; the strands stay. `--midtone 0.6` (**Midtone**) puts the inner
corona's median (1.05-2.5 solar radii) at that level with a midtones
transfer curve, for a light grey corona. Both off by default.

Noise in the adaptive layer: at its finest scales a stack is mostly noise
(on the 2024 stack, structure stood only 2x and 3.6x above the noise at
2.5 and 5 px, against 9-250x at 10-83 px), and the adaptive filter turns it
into grain. `--noise-gate 3` (**Adaptive noise gate**) weights each scale's
detail by how far it stands above the noise at that scale (a Wiener gain).
The noise at each scale comes from a noise image with the stack's own noise
level (from its weights) and spatial spectrum (measured in the faint corona:
demosaicing and resampling correlate neighbouring pixels), put through the
same steps. A test checks that pure noise gives under 0.4x the texture of
the plain filter and of real strands. Off by default (the filter as
published); the gated output keeps a fixed scale, so where there is little
real structure it stays flat instead of being stretched back up.
`--adaptive-k` (**Strength** in the **Adaptive layer (MGN)** box, steps of
0.01) is k in the paper's arctan(k C): how hard local contrast is brought
out at each scale. The default, 0.7, is the paper's value.

Adaptive (ACHF): Druckmüller's adaptive circular high-pass filter, as
defined in H. Druckmüllerová's PhD thesis (*Application of Adaptive Filters
in Processing of Solar Corona Images*, Brno University of Technology, 2013),
section 5.2. For each scale sigma, each pixel's local mean is an incomplete
convolution (eq. 5.2) with the kernel of eq. 5.1,
exp(-((rho - r)^2 + (r (theta - phi))^2) / 2 sigma^2): a Gaussian of width sigma
across the radius and sigma *along the Sun-centred circle*, so the kernel
bends with the corona; the Moon and pixels outside the image have weight 0,
so nothing rings at the limb. The filtered image is the pixel minus that
mean. It is computed exactly in Sun-centred polar coordinates (rows 1 px
apart in radius, the circle sampled at least 1 px apart; across the radius,
then along each circle by FFT), or as an ordinary weighted Gaussian where the
circle is straight to within 0.1 px over the kernel. Tests check the
circular geometry (a corona that only varies with radius gives exactly the
1-D blur of its profile; a straight kernel does not), that the Moon has no
influence, and the small-scale limit. Not published, so chosen here and
open to change: the filter works on log brightness, sums its scales with
the strengths set (an amplification rising with spatial frequency), and is
centred on mid-grey for the blend with its contrast set
from where every kernel is whole (next to the Moon the large kernels see
only the fainter corona outside, so a bright ring forms there; set from all
pixels, that ring took the whole range and left the rest flat grey on the
2024 crop). Its six scales each have a size and a strength (**Adaptive layer (ACHF):
size and strength of each scale** on the Enhance tab; `--achf-scales` in %
of the Sun's radius and `--achf-weights`, finest first), as the sliders of
Druckmüller's Corona program did. Defaults (Andy's choice): sizes of 0.5, 1,
3 and 10 px on the 2024 frames (Sun radius 413 px; within the thesis's 0.5-64 px)
at strength 1, and 10% and 20% of the Sun's radius at strength 0. Sizes are
kept as fractions of the Sun's radius, so they suit any lens; the window
shows them in pixels of the open stack. 0 leaves a scale out and saves its
time; only the ratios of the strengths matter. The 2006 paper's one
published setting (its Figure 9, rising to 20x for the finest detail)
likewise rests on scales of about 0.5-1 px; on a typical stack the finest
scales are mostly noise, and the thesis says the amplification "is limited
by the amount of noise", so lower them if the layer is grainy. The preview
is a reduced copy (about 1/6 for a full frame), so scales under about 6 px
hardly show there; they show in the full-size layers.
In the blend, the natural-layer base plays the part of the original image's
share (K1 in h = K1 f + K2 g). The later refinements in Druckmüller's Corona
4.1 software ("nonlinearity in the use of filtered images") are not
published and not reproduced. It amplifies detail without normalizing its
contrast, so it keeps more of the difference between bright and faint
structure than Adaptive (MGN), and more noise in the faint corona. On the
2024 crop (2530 px square) it took 61 s against 3 s for Adaptive (MGN); a full
frame takes a few minutes, so it is off unless ticked (Enhance tab) or listed
in `--layers`, and the preview computes it only when it is ticked, blended
or shown.

FNRGF settings: `--order` (10), `--segments` (50), `--mean-att` and
`--std-att` (attenuation factors A0..A10 and S0..S10, default 1, 0.9, ..., 0).

## Earthshine

```bash
totality earthshine eclipse2024.json       # after 'totality stack'
```

The Moon's night side, lit by sunlight reflected from the Earth. The Moon
moves across the Sun during totality (about 40 px in three minutes at
400 mm), so a corona-aligned stack blurs it. This step keeps each frame's
corona alignment and adds a shift that puts its Moon where the Moon is in
the HDR stack's Moon frame. Only exposures within `--stops` (default 6) of
the longest are used. The glare from the inner corona inside the disc, a
gentle gradient, is removed with a smooth surface (`--glare-order`, default
2; higher orders start removing large maria too). Close to the limb the glare
climbs steeply (10-20x the earthshine) because the bright inner corona,
the chromosphere and prominences are only pixels away, and it changes
within a few degrees round the limb. Terms that fall off with distance
inside the limb take out its average shape. The rest is followed round the
limb: the image is divided by the glare surface, resampled onto circles,
and each circle is median-filtered round the limb. The window is about 1°
at the limb and widens quickly further in (13° at 40 px, 50° at 90 px on
a 436 px Moon), because the glare's pattern spreads out with depth. Lunar
features smaller than the window are kept, so maria near the limb such as
Crisium survive; only within ~20 px of the limb, where the glare is many
times the earthshine, is most detail flattened. Before that, in short arcs round the limb, light that fades inward
(a sum of decaying exponentials with strengths of one sign) is fitted and
taken away: glare can only fade inward, while a mare near the limb is a dip
and highlands a bump, which such a fit cannot take (`--no-limb-shape` turns
it off). The earthshine is made from the colour channels that suit the
stack best (`--channel auto`): much of the glare that changes quickly round
the limb is red light from the chromosphere and prominences, and a colour
camera has twice as many green pixels, so all of green, green + blue and the
average of all three are tried and the one with the weakest glare pattern
3-6% of the radius inside the limb (and, among close ones, the least noise)
is used. On the 2024 D850 stack green won: glare pattern 7.1 against 9.3
for the average, noise 0.15 against 0.19 (both relative to the lunar
contrast). `<name>_earthshine_rgb.fit` holds the stack per channel. `--no-limb-glare` turns all of the limb glare
handling off.

The glare is fitted on the disc from a small margin inside the limb, which
is scaled to the Moon's size like the stacking margin: 0.7% of its radius,
at least 2 px (3 px for the 2024 D850 at 400 mm, about 7.5 px at 1000 mm).
Most other limb widths are already fractions of the radius.

**Glare round a bright prominence.** A bright prominence throws a fan of
glare onto the Moon, with diffraction spikes, centred on the prominence
rather than on the Moon, so the limb terms (smooth round the limb) leave a
bright fan with dark streaks beside it. Where the residual just inside the
limb stands well above the range of the lunar surface, the fan is fitted in
polar coordinates about the prominence: a pattern of directions times a
fall-off with distance shared by every direction (fitted robustly, the
fall-off only allowed to decrease, so a mare in the fan is not taken for
glare), which the maria do not follow. The prominence's place on the limb
(on it or just outside) is searched within 2 degrees for the fit that leaves
least behind. Up to four such sources are fitted, brightest first. On the 2024 stack this removed the fan
at 74 degrees; before, it was covered by a smooth lighter patch.
`--no-prominence-glare` (or the checkbox on the tab) turns it off.

**Albedo map (optional).** `--albedo-map lroc_color_2k.jpg` (or **Albedo
map** on the Earthshine tab) uses a map of the Moon's surface brightness to
help tell maria near the limb from glare. Any 2:1 equirectangular map with
0° longitude in the middle works, for example the colour map from NASA's
[CGI Moon Kit](https://svs.gsfc.nasa.gov/4720) (credit: NASA's Scientific
Visualization Studio). The map is turned to match your Moon (rotation,
mirror and libration are searched against the pattern in the middle of your
disc, so no clock or site is needed); if the match is below 0.6 it is not
used. It is blurred, scaled to your data and only subtracted while the glare
is *estimated*; the glare estimate is then subtracted from your data, so
nothing from the map is added to the picture. Where your data can tell the
surface from glare, the map is first pulled towards your data, so a feature
the map has and your Moon does not is mostly rejected (not in the outer 5%
of the radius, where that correction fed the glare back as a string of beads
along the limb). On simulations a
false mare 25 px inside the limb came through at about 15–20% of
`--albedo-strength` (default 0.5), and not at all 45 px or more inside.

**A dark, clean Moon; map texture (optional, a composite).** On the 2024
D850 earthshine stack the lunar surface is mostly noise at every scale up to
about 35 px (structure/noise 1.0-1.4 from 1 to 35 px): the maria are really
there, craters and rays are not. `--surface-denoise 4` (**Edge-preserving
smoothing**) smooths only across similar brightness (a joint bilateral
filter judged on a lightly smoothed guide), so the noise inside the maria
goes and their edges stay crisp; plain `--smooth` blurs the edges too, and
large values of it spread leftover limb glare into beads by the limb. A
darker, flat, neutral Moon: `--brightness 0.35 --limb-darkening 0
--no-colour --surface-denoise 4`.
`--albedo-texture 1` (**Map texture (composite)**, with an albedo map)
adds the map's own fine detail (finer than 4% of the radius: craters, rays)
to the surface, scaled to your Moon's contrast. That detail comes from the
map, not your photographs, so it is off by default, only used when the map
matches your Moon, and the log says "albedo texture ADDED (composite, from
the map, not your data)".

**How the Moon meets the corona.** A perfectly clean disc cut on a circle
looks pasted in, so the layer is built like a photograph of the limb:

* *The real limb.* The Moon-aligned stack reaches `--limb-reach` (6) px past
  the limb, and over the last few pixels and beyond the layer shows that
  data itself, compressed logarithmically so the corona just outside comes
  to full brightness. The edge therefore has the Moon's own outline
  (mountains, valleys) and softness, and runs into the corona; the layer
  hands over to the corona layers 2 px past their black disc.
  `--limb-soften` (1.5 px) adds a little extra softness.
* *A glow.* `--keep-glare` (0.5) of the glare that was taken away is put back
  as a glow fading inward from the limb (the inner corona's light spilling
  over the dark Moon), on the same logarithmic scale, so it cannot clip.
* *Colour.* The layer is in colour (`--no-colour` for grey): the glow and
  the limb keep the colour measured in the stack, white balanced like the
  corona layers (from the Enhance step), so a prominence or the pink
  chromosphere stays red where it spills over the limb; the surface takes
  the earthshine's own tint, measured from how strongly the lunar pattern
  shows in each channel (on the 2024 stack R, G, B = 1.06, 0.97, 0.97).
* *Grain in the filled edge.* The last pixels, which are only noise and are
  filled from just inside, get grain matching the surface next to them
  (`--no-edge-grain` to leave them smooth).

The shown layer gets a little extra smoothing (up to `--edge-smooth`, 3 px)
only where the noise is high: the last pixels by the limb, where the long
exposures are saturated. Real structure such as diffraction spikes from a
bright prominence is left alone. Where the noise (measured on the picture
itself, ring by ring) is still ~10x that of the middle, and at least over
the last 2.5% of the radius (~11 px), the edge is filled out to 1.5 px past the limb
with the brightness just inside, taken round the limb in ~16 px steps
(a median, so single noisy pixels do not show as teeth) and blended in. That is where the corona layers' black disc ends, so no dark
ring shows between them. Removing the glare also removes the disc's overall
shading, so it is put back as a vignette: full brightness in the middle,
darkening gradually all the way to the limb (`--limb-darkening`, default
0.6 = the limb at 40% of the middle; 0 = evenly lit; `--vignette-shape`,
default 1.5: 1 = a straight-line fall from the centre, higher keeps the
middle bright longer). The stretch is set from the
body of the disc. `<name>_earthshine_weight.fit` holds the stack weights
next to the linear stack.

| File | Contents |
| --- | --- |
| `<name>_earthshine.tif` | layer the size of the HDR stack: the lunar surface inside the disc, black outside, so it drops into the corona layers' black disc (Screen or Lighten mode); `--format png|jpg` for another type |
| `<name>_earthshine_crop.tif` | the same, cropped to the Moon |
| `<name>_earthshine.fit` | the Moon-aligned linear stack before glare removal |
| `<name>_earthshine_rgb.fit` | the same, per colour channel (used by the final image) |
| `<name>_earthshine_weight.fit` | the stack weights (how much went into each pixel) |
| `<name>_moon_surface.fit` | the seas and highlands, glare removed, as added to the final image (RGB and where to show them) |
| `<name>_earthshine.json` | where the crop sits, the Moon's centre and radius, and the settings used |

**The final image: Moon and corona stretched together.** `totality final
eclipse2024.json` (or **Final image** on the Earthshine tab, with a preview)
puts the Moon-aligned earthshine stack, linear and in the same units, into
the HDR stack inside the lunar disc, registered on the limb. The settings of
your last Enhance run then stretch Moon and corona together, so the limb is
the stack's own: one curve, one colour treatment, no seam. A prominence is
the same red on both sides of the limb, and nothing at the limb is cut,
pasted, softened or matched. It needs the files the Earthshine step writes
now (`_earthshine_rgb.fit`, `_moon_surface.fit`); with older ones it asks
you to stack the earthshine again. Enhance does not need running again.

* *Where the two stacks meet.* Just outside the limb neither stack is used as
  it is: the earthshine stack's corona is smeared (it is aligned on the Moon)
  and the HDR stack's holds few frames there (the Moon covered it part of the
  time). The earthshine stack's own limb edge hands over to the HDR's corona,
  continued inwards from where the HDR stack is reliable; switching from one
  stack to the other showed as a line round the limb. That distance is
  measured from the HDR stack's weights, not fixed: the outermost ring where
  the weight still climbs steeply away from the limb (a rise of over 0.1 in
  log per px and over 3x the rise a few rings further out; further out the
  weight keeps rising more gently as fewer long frames saturate), at least
  2 px, at most 3% of the radius (on the 2024 stack: 3 px). The log names it
  ("the HDR stack's corona is used from N px outside the limb"). It is wider
  at long focal lengths or small pixels.
* *By a bright prominence.* The long earthshine frames saturate, so towards
  it ever fewer frames are usable and the last are near full well. Below 40%
  of the usual number of frames (fully below 10%) the light is interpolated
  along each radius in log brightness, from the last well-exposed earthshine
  pixel to the limb, where it takes the HDR's corona times the drop across
  the limb seen in the nearest good directions (reaching up to 7% of the
  radius inside). Spreading it sideways from neighbours left a ragged
  dark-red fringe; the stack's own one-frame Moon, clipped white there, is
  never used.
* *Noise and colour.* The two stacks' weights are in different units; each
  is turned into a noise level from its own pixel scatter, so the stretch
  knows how good the Moon's pixels are. (With the HDR's one-frame weights the
  colour inside the limb was faded to grey as if it were noise.)
* *Filter layers.* NRGF, FNRGF and adaptive have no data on the Moon, so in
  the blend they fade out over the 4 px outside the limb.
* *The lunar surface.* The seas and highlands (glare removed, from
  `<name>_moon_surface.fit`) are added inside with screen at `--strength`
  (1 = as made by the earthshine step), so near the limb, where the Moon's
  own stretched light is brighter, that light wins.

`--base natural` uses the natural layer instead of the blend. Writes
`<name>_final.tif` (or `-o`).

The older way, pasting the earthshine layer (`<name>_earthshine.tif`, still
written for Photoshop) into any image with a blend mode, is `--paste` or
`--base-file` (on the tab: **Paste into a file**).

The report gives the lunar detail over the noise; below about 3 the layer
is mostly noise (short exposures, low ISO or a bright sky).

## Prominences

```bash
totality prominences eclipse2024.json              # writes <name>_prominences.tif
totality final-prominences eclipse2024.json        # puts them on top of the final image
```

Prominences glow in hydrogen-alpha (red) and weaker blue hydrogen lines, so
they look pink or magenta once the corona is white. They are found by
colour: pixels clearly redder than the corona at the same height, beyond the
noise (`--sensitivity`, default 4) and the redness ratio (`--min-redness`,
default 1.2), up to `--height` solar radii above the limb (default 0.3). They
come from the HDR stack, so every prominence the moving Moon uncovered at any
time is there, and the layer stops at the lunar limb of the Moon frame, so
nothing covers the Moon or the earthshine. `--no-chromosphere` leaves out the
thin pink rim along the limb (red light lower than `--chromosphere-arcsec`,
default 12), keeping the bases of taller prominences.

The layer is transparent outside the prominences (TIFF and PNG), so it sits
on top in Photoshop in Normal mode. `totality final-prominences` (or tab 6) puts
it on the final image from the Earthshine step, the last-written blend or
natural layer (`--base`), or any image (`--base-file`), in normal mode by
default, which keeps their colour over the bright inner corona.

## Picture formats

Every picture output (layers, earthshine, prominences, final images) can be
written as:

| Format | Bits | Compression | |
| --- | --- | --- | --- |
| TIFF (default) | 16 | none | highest quality, largest |
| PNG | 16 | lossless | same quality, smaller |
| JPG | 8 | quality 100, no colour subsampling, dithered | smallest; JPG cannot hold 16 bits or transparency |

All are tagged 300 dpi (a printing label; the pixels are not changed) and
sRGB. In the GUI the Save dialog's file type chooses; on the command line
`--format tif|png|jpg` or the output file's extension.

## Calibration rules

* **Vignetting**, first match wins and only one is ever applied:
  1. a profile you name with `--profile`;
  2. your own flats (`--flats`, or the flats folder on the Frames tab), when
     they match the frame's aperture and focal length, used in one of three
     ways (**Flat frames** on the Stack tab, `totality stack --flat-mode`):
     - **Master flat** (`master`, the default): the flats are averaged (each
       with its own bias or black level removed and scaled to 1.0 in the
       middle, per CFA position) and every frame is divided by that, pixel by
       pixel. This corrects vignetting *and* dust shadows, provided nothing
       was changed or cleaned between the eclipse and the flats. A straight
       brightness slope across the whole frame (a light panel or sky
       brighter on one side) comes from the light, not the lens: it is
       measured against the smooth profile, taken out of the master flat and
       named in the log ("brighter on the right (3.2% at the edge)").
     - **Smooth profile from the flats** (`profile`): a smooth, radially
       symmetric profile is fitted to the master flat (as `totality
       flat-profile` does) and used instead. Vignetting only; dust in the
       flats is ignored, so dust that moved cannot add dark rings.
     - **Master flat + compare** (`compare`): the master flat, and the log
       reports how far it differs from the smooth profile (rms; the share of
       pixels in dark spots more than 2% deep, judged against the
       surrounding level so a broad difference is not counted as dust; the
       deepest spot), with a warning when the difference is large. The ratio is saved as
       `<name>_flat_vs_profile_<settings>.fit` next to the stack: 1.0 where
       they agree, dark spots where there is dust.

     Flats that match no light frame are reported in the log and not used.
  3. a saved own-flats profile for the same lens, within 5% focal length
     and 1/3 stop;
  4. the lensfun profile;
  5. none, with a warning.

  Profiles of either kind can be written as a synthetic FITS master flat
  for Siril. The eclipse is assumed focused at infinity.
* **Darks:** matched by exposure time and ISO and averaged into a master
  dark, which is subtracted (it includes the bias); without a match, the
  bias is used and hot pixels are found from the light frames.
* **Bias:** matched by ISO and averaged into a master bias; without bias
  frames, each file's black level is used as a synthetic bias.

Stack logs which correction every frame got ("vignetting: 48 frame(s) by
master flat (f/8, 400 mm)", "dark/bias: ..."), how much darker the corners
were ("the corners are 62% as bright as the middle (-0.69 stops)"), and
saves that correction for the reference frame as `<name>_vignetting.fit`
at 1/8 size (1.0 = the middle; open it in Siril to see what was divided
out). When nothing matched it says "NONE applied". The folders chosen on
the Frames tab are saved with the project as soon as a step runs, so
Apply is not needed first. The earthshine stack is
calibrated exactly as the HDR stack was, so the two stay in the same
units; if the HDR stack was made before Stack used the calibration frames,
the earthshine log says to run Stack again. The Align step uses the lens
profile only: vignetting and dust do not affect the alignment, which
works on fine structure.

## Own-flats profiles

When lensfun has no vignetting data for your lens, shoot flats and fit a
profile once:

1. Same lens, focal length, aperture and focus (infinity) as the eclipse
   frames; any ISO; any camera with the same sensor size.
2. An even light source filling the frame: a light panel, a white screen,
   or a twilight sky through a white T-shirt. Expose so the histogram peak
   is near the middle. 20 to 30 frames.
3. `totality flat-profile FLATS` fits a smooth, radially symmetric profile per
   colour channel and saves it to your profile library
   (`~/.solar-eclipse-toolkit/profiles`, or `TSE_PROFILE_DIR`).

The profile keeps only the lens's falloff: dust in the flats is rejected,
and a light source that is brighter on one side is fitted as a tilt and
removed. Because the profile is measured on the sensor in millimetres, it
corrects any camera with the same lens, so one set of flats serves both a
D800 and a D850. `totality inventory` then picks it up automatically. A FITS
master flat stacked in Siril can be used instead of RAW flats.

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
| `--no-exiftool` | do not use exiftool even if installed |
| `--eclipse` | the eclipse, by its date, e.g. 2024-04-08 (default: the frames' date) |
| `--lat` | site latitude in degrees, north positive (saved in the project) |
| `--lon` | site longitude in degrees, east positive (west is negative) |
| `--elev` | site elevation in metres |
| `--camera-utc-offset` | the camera clock's time zone in hours, e.g. -6, when the files do not record one |
| `--clock-error` | seconds the camera clock was behind true time (true minus camera), when known |
| `--c2` | second contact typed in, e.g. 2024-04-08T13:47:33, in the camera's clock (or in true time with --clock-offset); default: computed from the eclipse and site |
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
| `--date` | eclipse date, e.g. 2024-04-08 |
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
| `--output`, `-o` | alignment results file (default: next to the project) |
| `--profile-dir` | folder of saved own-flats profiles |
| `--lensfun-db` | folder holding lensfun XML files (default: the installed database) |
| `--quiet` | print only the result, not each step |

### `totality stack`

merge the aligned frames into one linear HDR image.

| Option | What it does |
| --- | --- |
| `project` | project file (run 'totality align' on it first) |
| `--align` | alignment results file (default: the one saved by 'totality align') |
| `--output`, `-o` | output folder (default: next to the project) |
| `--name` | output file name prefix (default: the project name) |
| `--square` | quicker run: output only a square 10 solar radii wide around the Sun (default: the full frame, the same size as the RAW files) |
| `--radii` | like --square, but this many solar radii from the centre to the edge |
| `--moon-frame` | frame number whose Moon (and earthshine) is kept (default: the alignment reference) |
| `--include-rejected` | also stack frames the Sun-centre fit flagged as off the Moon's track |
| `--no-hot-pixels` | skip the hot-pixel correction |
| `--flat-mode` | how the project's flat frames are used: master = divide by the master flat (vignetting and dust; default), profile = a smooth profile fitted to them (vignetting only), compare = master flat and report how it differs from the profile |
| `--profile-dir` | folder of saved own-flats profiles |
| `--lensfun-db` | folder holding lensfun XML files (default: the installed database) |
| `--quiet` | print only the result, not each step |

### `totality enhance`

reveal coronal structure: NRGF, FNRGF, adaptive filter, natural, blend.

| Option | What it does |
| --- | --- |
| `project` | project file (run 'totality stack' on it first) |
| `--hdr` | HDR FITS file (default: the one saved by 'totality stack') |
| `--output`, `-o` | output folder (default: next to the project) |
| `--name` | output file name prefix (default: the project name) |
| `--layers` | comma-separated layers to write, from natural, nrgf, fnrgf, fnrgf_n, adaptive, achf (default: all but achf, which takes minutes on a full frame) |
| `--blend` | suggested blend as layer=weight pairs, each optionally ':overlay' (default soft light), e.g. natural=1,fnrgf_n=0.7,adaptive=0.1:overlay; or 'none' (default: %(default)s) |
| `--order` | FNRGF Fourier order (default 10) |
| `--segments` | FNRGF angular segments (default 50) |
| `--mean-att` | FNRGF attenuation A0,A1,... for the mean (default 1,0.9,...,0) |
| `--std-att` | FNRGF attenuation S0,S1,... for the deviation (default 1,0.9,...,0) |
| `--noise-m` | FNRGF-N noise compensation, 0..0.99 (default 0.7) |
| `--adaptive-k` | adaptive layer: strength of the local contrast at each scale, k in arctan(k C) (default 0.7, as in Morgan & Druckmueller 2014) |
| `--achf-scales` | Adaptive (ACHF): the size (kernel sigma) of each of its six scales, finest first, in %% of the Sun's radius (default %(default)s: 0.5, 1, 3 and 10 px on the 2024 stack's Sun of 413 px, then 10%% and 20%%) |
| `--achf-weights` | Adaptive (ACHF): strength of each of its six scales, finest first; 0 leaves a scale out (default %(default)s) |
| `--noise-gate` | adaptive layer: each scale's detail counts only as far as it stands above the stack's noise at that scale (e.g. 3); 0 = off |
| `--even` | take out this fraction of the large brightness differences round the Sun (streamers against coronal holes); 0 = off, e.g. 0.7 |
| `--midtone` | put the inner corona's median brightness at this level with a midtones transfer curve; 0 = off, e.g. 0.6 |
| `--shoulder` | highlight roll-off: 0 = hard white point (the brightest 0.1%% clip); e.g. 0.3 = the top 30%% of the range rolls off smoothly to white, nothing clips and colours keep their hue |
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
| `--no-prominence-glare` | do not fit and remove the fan of glare (and spikes) round a bright prominence |
| `--no-limb-shape` | do not fit the limb glare as light fading inward (that fit keeps maria by the limb) |
| `--channel` | colour channels to make the earthshine from (default auto: chosen on the stack; the red channel carries most of the chromosphere glare) |
| `--albedo-map` | lunar albedo map (2:1, 0 deg longitude centred, e.g. lroc_color_2k.jpg from NASA's CGI Moon Kit) to guide the glare removal near the limb |
| `--albedo-strength` | how much the map guides it, 0-1 (default 0.5); nothing from the map is added to the picture |
| `--glare-order` | smoothness of the glare removed from the disc: polynomial order (default 2) |
| `--smooth` | smoothing of the layer in pixels (default 1, 0 = none) |
| `--brightness` | brightest lunar surface in the layer, 0-1 (default 0.6, dimmer than the corona) |
| `--limb-darkening` | vignette: how much darker the limb is than the middle, 0-1 (default 0.6; 0 = evenly lit) |
| `--keep-glare` | how much of the removed glare to keep as a glow inside the limb, 0-1 (default 0.5) |
| `--limb-soften` | extra softening of the limb in px, on top of the real one (default 1.5, 0 = none) |
| `--no-colour` | make the earthshine layer grey (default: natural colour, so a prominence at the limb stays red on the Moon side too) |
| `--no-edge-grain` | leave the filled last pixels by the limb smooth (default: grain like the rest) |
| `--vignette-shape` | how the darkening falls from the centre: 1 = straight line, 1.5 (default) = gentle in the middle, 3 = the middle stays bright longer |
| `--surface-denoise` | edge-preserving smoothing of the lunar surface, px (0 = off; e.g. 4): maria keep crisp edges |
| `--albedo-texture` | COMPOSITE: add the albedo map's own craters and rays at this strength (0 = off, your data only; 1 = at your Moon's contrast); needs --albedo-map |
| `--edge-smooth` | most smoothing of the noisier pixels near the limb, px (default 3) |
| `--include-rejected` | also use frames marked as rejected |
| `--output`, `-o` | output folder (default: next to the project) |
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
| `--paste` | the older way: paste the earthshine layer into an enhanced layer with a blend mode (needed with --base-file) |
| `--base` | enhanced layer to use as the corona (default: blend, the last one written) |
| `--base-file` | use this image as the corona instead (the size of the HDR stack) |
| `--strength` | brightness of the lunar surface, 1 = as made by the earthshine step (default 1) |
| `--mode` | blend mode (default screen) |
| `--name` | output file name prefix (default: the project name) |
| `--output`, `-o` | output file (.tif, .png or .jpg); default <name>_final.tif |
| `--quiet` | print only the result, not each step |

### `totality prominences`

find the prominences and write them as a layer (transparent elsewhere).

| Option | What it does |
| --- | --- |
| `project` | project file (run 'totality stack' on it first) |
| `--height` | how far above the solar limb to look, in solar radii (default 0.3) |
| `--sensitivity` | red excess needed, in noise standard deviations (default 4; lower finds fainter ones) |
| `--min-redness` | red/green ratio needed; the white corona is about 1 (default 1.2) |
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

```bash
pip install -e ".[dev]"
pytest
```

Set `TSE_SAMPLE_RAWS` to a folder of sample RAW files to also test the
metadata readers on real files. Development uses only the 2024 data; the
2017 set is held back as a blind test of the finished tool and must not be
added to the tests.

* Synthetic eclipses with known geometry, shutter errors, seeing, a moving
  Moon, earthshine and prominences check each step against the truth.
* `tests/test_real_data.py` runs the enhancement on real data: the 2024
  stack from Lake Wappapello, binned 8 x 8
  (`tests/data/eclipse2024_bin8.npz`, 3.6 MB). It keeps the problems
  synthetic data missed (non-image edge columns, an uneven twilight sky with
  a negative red level, corona reaching the frame edges, dust and stars) and
  checks for rings, a bright sky and colour casts.
* `tests/test_earthshine_real.py` runs the earthshine glare removal on the
  real 2024 Moon-aligned stack, binned 2 x 2
  (`tests/data/earthshine2024_bin2.npz`): glare round the limb and round the
  bright prominence, maria kept by the limb, no arcs or teeth at the edge.
  Other tests use real figures from that stack too (the HDR weights across
  the limb).
* `tests/test_moonmap.py` checks the albedo map guide on a simulated Moon:
  orientation found, maria by the limb recovered, a false feature hardly
  leaking, a wrong map refused.
* `tests/test_viewzoom.py` checks the zoom and pan of the picture views.
* `tests/test_readme.py` checks that the command-line reference below is
  complete and up to date.
* `tests/test_gui.py` runs every tab of the Siril window's logic end to end
  with a small stand-in for PyQt6 (`tests/qtstub`), so it needs no display;
  layout and rendering are only checked by running it in Siril.

## Licence

MIT, see `LICENSE`.

The lensfun database is not bundled; it is downloaded from the lensfun
project (CC BY-SA 3.0). `tests/data/lensfun_subset.xml` is a short excerpt
used only by the tests.
