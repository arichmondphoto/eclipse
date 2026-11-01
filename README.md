# Solar Eclipse Toolkit

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

Nothing is tied to one eclipse, camera or bracket: every setting is read
from each frame's metadata, and eclipse geometry comes from NASA's Besselian
elements for any eclipse from 1600 to 2400.

## Status

| Step | Scope | State |
| --- | --- | --- |
| 1 Frames | RAW inventory, bracket detection, lensfun or own-flats vignetting, calibration planning, project folders | done |
| 2 Align | Lunar limb, phase correlation alignment, field rotation, the Sun's centre | done |
| 3 Stack | Moon-aware linear HDR stacking with measured brightness matching | done |
| 4 Enhance | Natural, NRGF, FNRGF, FNRGF-N and adaptive layers; Soft Light / Overlay blend | done |
| 5 Earthshine | Moon-aligned stack of the long exposures; blended into the picture | done (synthetic tests only so far) |
| 6 Prominences | Found by colour at the limb; transparent layer put on top | done (checked on a reduced copy of real data) |

Each step works in the Siril window and on the command line. Tested on the
2024 eclipse (Nikon D850, 400 mm); the 2017 set is kept back as a blind test.
Still to come: the blind test, and help for other cameras and operating
systems before sharing the tool more widely.

## Workflow

1. **Frames:** make a project from the event folder (see *Project folders*).
2. **Align:** enter the site; the frames are aligned and the Sun's centre found.
3. **Stack:** the frames are merged into one linear 32-bit HDR image.
4. **Enhance:** tune the layers in a live preview, then write them full size.
5. **Earthshine:** stack the Moon and blend it into the picture.
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

The window has one tab per step, and each step saves into the project file:

| Tab | What you do | What you see |
| --- | --- | --- |
| 1 Frames | Read the RAW folder, check the frames and brackets, choose the lens profile, add calibration folders | Frame table (right-click to exclude a frame), bracket and calibration report |
| 2 Align | Enter the site (latitude, longitude, elevation), the camera time zone and clock error if known; choose the rotation model | Per-frame shift, rotation and peak strength; the Sun centre report |
| 3 Stack | Full frame or a square around the Sun; which frame's Moon to keep | The stacked preview and each frame's measured brightness |
| 4 Enhance | Layers to write; each layer's blend strength and Soft Light or Overlay; flatten, saturation, white balance, sky removal; FNRGF order, segments and noise compensation | A live preview on a reduced copy (about 1400 px) that updates as you change settings; then **Write full-size layers** |
| 5 Earthshine | Frames to use, glare removal, smoothing; then the corona image (last-written blend, natural layer or a file), strength and blend mode | The Moon close up, then a preview of the picture; **Write final image** |
| 6 Prominences | Height, sensitivity, redness, chromosphere on or off, brightness, saturation; then the image to put them on, strength and mode | The prominences over a dim view of the inner corona, then a preview of the picture; **Write final image** |

Every setting has hover help explaining what it does and when to change it.
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

`totality <command> --help` lists every option. More examples:

```bash
totality inventory "D:/Eclipse 2024/RAW" --project eclipse2024.json --csv frames.csv
totality inventory RAW --c2 2024-04-08T13:47:30 --c3 2024-04-08T13:51:45   # mark frames outside totality
totality inventory RAW --flats FLATS --darks DARKS --bias BIAS              # optional calibration frames
totality inventory RAW --lens "Nikon AF-S Nikkor 80-400mm f/4.5-5.6G ED VR" # force a lens profile
totality lens "AF-S Nikkor 80-400mm f/4.5-5.6G ED VR" --focal 400 --aperture 8
totality flat-profile FLATS                       # fit and save a vignetting profile from your own flats
totality inventory RAW --profile my_profile.json  # use one profile for every frame
totality frame DSC_4100.NEF                       # show everything read from one file
totality frames eclipse2024.json --exclude DSC_9108.nef   # leave a frame out (--include to undo)
totality eclipse --date 2024-04-08 --lat 36.9422 --lon -90.3343 --utc-offset -5   # contacts and duration
totality align eclipse2024.json --lat 36.9422 --lon -90.3343 --elev 110           # align and find the Sun
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

Contact times are compared with the camera's clock. If the clock was wrong,
give the correction with `--clock-offset` (added to camera time, for example
`+1:04:08`) and enter the contact times in true time. Bracket detection and
Moon tracking only use time differences, so a wrong clock never affects them.

## HDR stacking (Phase 3)

```bash
totality stack eclipse2024.json            # uses the alignment saved by 'totality align'
totality stack eclipse2024.json --square   # quicker: a square 10 solar radii wide
```

Every frame is read again at full resolution: black level subtracted per
channel (negative values kept), vignetting divided out per colour, isolated
hot pixels replaced, demosaiced with the linear Malvar-He-Cutler method in
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
| `natural` | log stretch with most of the radial fall-off removed (`--flatten`, default 0.8); the eye-view look |
| `nrgf` | normalizing-radial-graded filter (Morgan et al. 2006): each circle around the Sun normalized to the same mean and spread |
| `fnrgf` | Fourier NRGF (Druckmüllerová et al. 2011): mean and spread follow position angle too, so faint and bright regions at the same height get similar contrast |
| `fnrgf_n` | FNRGF-N: the same with noise compensation (`--noise-m`, default 0.7); the noise is estimated automatically from the stack's weight map |
| `adaptive` | multi-scale Gaussian normalization (Morgan & Druckmüller 2014) of the log image: fine structure at all scales and orientations, the published successor to the adaptive circular high-pass filter |
| `blend` | suggested mix: the natural layer as the base with the filter layers applied in Soft Light (or Overlay) at the given opacities, `--blend natural=1,fnrgf_n=0.7,adaptive=0.5` by default; add `:overlay` to a layer for Overlay, e.g. `--blend natural=1,adaptive=0.1,fnrgf=0.05:overlay` (`--blend none` to skip); the same thing you would do with layers in Photoshop |

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

To skip Photoshop, blend it in here: `totality final eclipse2024.json` (or the
**Blend into the image** box on the Earthshine tab, with a preview) puts the
earthshine into the lunar disc of the last-written blend (`--base natural`
for the natural layer, `--base-file` any image the size of the stack, such
as a blend saved earlier), at `--strength` (1 = as stacked) in `--mode` screen
(default), lighten or add, and writes `<name>_final.tif` (or the file given
with `-o`, in the format its extension names).

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
3. `totality flat-profile FLATS` fits a smooth, radially symmetric profile per
   colour channel and saves it to your profile library
   (`~/.solar-eclipse-toolkit/profiles`, or `TSE_PROFILE_DIR`).

The profile keeps only the lens's falloff: dust in the flats is rejected,
and a light source that is brighter on one side is fitted as a tilt and
removed. Because the profile is measured on the sensor in millimetres, it
corrects any camera with the same lens, so one set of flats serves both a
D800 and a D850. `totality inventory` then picks it up automatically. A FITS
master flat stacked in Siril can be used instead of RAW flats.

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
* `tests/test_gui.py` runs every tab of the Siril window's logic end to end
  with a small stand-in for PyQt6 (`tests/qtstub`), so it needs no display;
  layout and rendering are only checked by running it in Siril.

## Licence

MIT, see `LICENSE`.

The lensfun database is not bundled; it is downloaded from the lensfun
project (CC BY-SA 3.0). `tests/data/lensfun_subset.xml` is a short excerpt
used only by the tests.
