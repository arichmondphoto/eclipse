# Future additions, if needed or requested

Ideas that are worked out but not started. Each waits until someone needs it or asks for it.

## Astronomy cameras (ZWO and similar)

Pinned 6 Nov 2026. Support for frames from dedicated astronomy cameras (ZWO ASI and similar),
recorded with ASIAIR, ASICap, SharpCap, N.I.N.A. or FireCapture.

**Where things stand.** `rawio` can already read the pixels of a FITS frame (colour with a
`BAYERPAT` header, or mono). But the Frames tab and `find_raw_files` only pick up camera RAW
files, no FITS header settings are read (exposure, time, gain), and a FITS frame's white level
is taken as its brightest pixel, which misjudges saturation. Frames without lens data already
work (no f-number: the same for every frame; flats matched as the same optics).

**What it needs, in order:**

1. Colour FITS frames found and read: exposure (`EXPTIME`), time (`DATE-OBS`, UTC), gain, offset,
   CFA pattern (`BAYERPAT`), sensor temperature, from each capture program's own header variants.
2. Gain in place of ISO for the nominal brightness (ZWO: 10 gain units = about x1.12, i.e.
   10^(gain/200)). Only needs to be roughly right: Stack measures the ratios.
3. Black and white levels: black from the offset or bias frames; white from the bit depth
   (ZWO writes 12- and 14-bit data left-justified in 16 bits), not the brightest pixel.
4. Darks, flats and bias as FITS, matched by exposure, gain and temperature.
5. SER video: a reader, and a rule for combining a burst (probably averaging each burst into
   one frame before alignment).
6. Mono cameras: grey output, with the colour steps (white balance, earthshine tint) switched
   off. Colour through filters would be a larger project.

**Route decided (11 Nov 2026): read FITS directly; no TIFF.** The user asked whether Siril could
convert astronomy-camera files to TIFF so the toolkit needs no support per manufacturer. Not
recommended: capture programs (ASIAIR, N.I.N.A., SharpCap, FireCapture) already write FITS with
the raw Bayer mosaic, like a camera RAW, and a TIFF keeps the pixels but loses the FITS header
(exposure, time, gain, Bayer pattern), which would then have to come from somewhere else. The
header names are largely standard across brands (`EXPTIME`, `DATE-OBS`, `BAYERPAT`, `GAIN`,
`OFFSET`, `CCD-TEMP`, `XBINNING`), so reading FITS needs a few fallbacks for variants, not code
per manufacturer. Siril's part: converting SER video into FITS sequences with their headers,
so no SER reader of our own is needed (this replaces step 5's reader). RAW from any Bayer camera
was added the same day (37be75c); this is the remaining part of "any camera".

**Before starting:** the user had no astronomy camera on 11 Nov 2026, so there are no real
files yet; do not start without them (no guessing at header variants or bit packing). Ask for a
short test set from his camera, any subject (the Moon or a bright daytime target is enough):
colour or mono lights at two or three exposures, a bias, a dark and a flat, and one short SER
if the capture program records them; and which capture program wrote them.

## Prominence brightness by red weight, as Druckmüller does

Pinned 10 Nov 2026, when the Prominences tab was removed (it stacked the short exposures and
sharpened the HDR stack's prominences after stacking: 0.6-1.3 px on the 2024 stack, too little
for its cost, and found prominences by a colour threshold measured on the 2024 frames). The one
control worth having again is their brightness, so bright prominences do not turn flat pink.
Druckmüller's way does it while the HDR stack is made: the red channel's weight on the
prominences and chromosphere, so they come out of the stack at the brightness chosen and every
later step (Enhance, the earthshine Moon) sees them that way. To be worked out from his papers
(Druckmüller, Rušin & Minarovjech 2006; Druckmüller 2009) before starting, and it must not rely
on a colour threshold tuned to one camera.

## Several cameras and focal lengths in one stack

Pinned 11 Nov 2026 at the user's request. Frames of one eclipse from more than one camera body
and lens, stacked into one HDR stack. For example: three brackets at 400 mm on one body and three
at 800 mm on another. The longer lens would supply the inner corona, the limb and the prominences,
and the shorter lens the outer streamers. The example is hypothetical, and there is no data for it yet.

Image scales in that example: 4.35 µm pixels at 400 mm give 2.24″/px, and 4.88 µm pixels at
800 mm give 1.26″/px. The Sun is therefore 1.78× larger in the second set, which covers about a
third of the first set's field.

**Where things stand.** A project assumes one camera and one lens.
- `workflow.py` looks up the camera and lens from `lights[0]` only. That lookup gives the pixel
  pitch, the expected Moon radius from the focal length, and the lensfun vignetting.
- `align.Transform` already includes scale, and `coarse_rotation` searches any angle and scale
  (log-polar). However, `scaled_params` sets the blur widths from one Sun radius, and the default
  search expects a scale near 1.
- `stack.warp_region` resamples every frame onto the reference frame's crop grid, by point
  sampling (order 1 or 3). The output therefore has the reference's scale and field. This is fine
  for scales near 1, but it would alias when shrinking a frame.
- Brightness matching measures a gain and an offset per frame and colour. It therefore already
  handles sensors of different sensitivity, but it cannot handle different colour responses
  (that needs a 3×3 matrix, not three gains).
- Darks, flats and bias are matched by exposure, ISO and lens, but not by camera body.

**What it needs, in order:**

1. **Frame sets.** Each light keeps its own camera (make, model and body serial when the EXIF has
   it) and its own lens and focal length. The Frames tab groups the lights by set, for example
   "Set A: D850 + 400 mm, 21 frames". Calibration frames are matched within a set: by body, ISO
   and exposure for darks and bias, and by body, lens, focal length and aperture for flats.
   Lensfun vignetting, the pixel pitch and the expected Moon radius are also looked up per set.
   Old projects read as one set.
2. **A clock per set.** Each body has its own clock offset, measured from its own contacts or
   set by hand. Field rotation and the Moon's offset from the Sun are already worked out per
   frame from its time, so only the offset is new.
3. **Image scale measured, not taken from EXIF.** A lens's nominal focal length is approximate,
   and teleconverters, focus breathing and crop modes change it. Each set's arcsec per pixel is
   therefore measured from its fitted Sun or Moon radius, and the EXIF value is used only as the
   starting guess. A log line names the result, for example "Set B: 1.258″/px (EXIF 800 mm:
   1.258); Set A: 2.241″/px".
4. **Aligning sets to each other.** Each set is aligned within itself as now. The sets are then
   aligned to each other: the scale ratio comes from the measured radii, and scale, rotation and
   shift are refined on the corona where the sets overlap. On separate mounts, the cameras' roll
   can differ by any angle, so the rotation between sets is not limited. A log line gives the
   result, for example "Set B to Set A: scale 1.781, rotation 2.3°, residual 0.15 px".
5. **The output grid.** By default, the output has the finest set's scale and a field that
   covers all sets. A setting can choose a coarser scale to keep the file smaller. In the example,
   the 400 mm field at the 800 mm scale has about 3.2× the pixels of one frame, so memory goes
   through `parallel.run_all`'s budget, and the log names the output size.
6. **Resampling that keeps radiance.** Frames enlarged onto a finer grid are interpolated. That
   adds no detail, and step 8 must account for it. Frames shrunk onto a coarser grid are averaged
   over each output pixel (area resampling), not point-sampled, so they do not alias. The stack
   stays in linear radiance per output pixel either way.
7. **Colour.** Two sensors record colour differently. Each set is converted into one linear
   working space with its camera matrix (LibRaw's `rgb_xyz_matrix`, or the DNG `ColorMatrix`)
   before stacking. The existing per-channel matching then removes what is left, measured where
   the sets overlap. A full 3×3 matrix fitted to the overlap is not recommended: the corona has
   too little colour variety to fit one reliably. White balance is applied once, to the result.
   Everything stays linear.
8. **Where each set counts.** Weighting by noise alone would let many coarse frames blur the
   inner corona, because the finer set carries more detail, not just a different noise level.
   - **Recommended:** combine by spatial scale using the à trous layers already in `enhance.py`.
     Coarse scales (larger than the coarse set's resolution) come from every set, weighted by
     inverse variance as now. Fine scales come only from the sets that resolve them.
   - **Where the fine set's field ends:** the detail stops there, but the brightness does not
     jump, because the coarse scales come from both sets and agree after matching. This is a
     real change of resolution in the data, not a join that is blended away, so it keeps to the
     rule "Nothing is pasted, feathered or blended to hide a join."
   - **Checking it:** the log names the residual between the sets' coarse scales in the overlap.
     If they disagree, a set is wrongly calibrated or matched, and the stack is not made.
   - **Saturation:** a set's saturated or low-weight pixels drop out at every scale, as now.
9. **The Moon and earthshine.** The Moon margins are already fractions of the Moon's radius, so
   they scale per set. Stack earthshine needs the same per-set scale, colour conversion and
   alignment, and its output grid must be the HDR stack's.
10. **Project file, GUI and CLI.** A list of sets in `<event>.json`, with the measured scale,
    clock offset and set-to-set transform for each.
    - **Frames tab:** shows the sets and each clock offset.
    - **Stack tab:** an **Output scale** setting (Finest set / Coarsest set), with hover help.
    - **CLI:** matching options, regenerated into README with `tools/cli_reference.py`.
    - **README:** the method section and the weight-file description.

**Tests.**
- **Synthetic:** one synthetic corona rendered at two scales, through two different colour
  matrices, with noise and different fields. The stack must recover the truth:
  - at the fine scale inside the fine set's field;
  - at the coarse scale outside it;
  - with no brightness step in a radial profile across the edge of the fine field.
- **Pipeline test on the 2024 data:** the 2024 frames binned 2× and cropped can stand in for a
  second lens. This tests alignment, resampling and weights, but not colour, since the sensor is
  the same.
- **Real two-camera data:** frames of one eclipse from two bodies or two lenses. The 2017 D800
  set is the blind test and must not be used for this.

**Before starting:** ask the user for real frames from two cameras or two lenses of the same
eclipse, with calibration frames for each. Expect several sessions, in the order above, with
steps 1–4 checkable before any change to stacking.

**Until then:** process each set as its own project, and combine the finished pictures by hand
(the long-lens version for the inner corona and prominences, the short-lens version for the
outer corona), matching their tones where they meet.

## Wavelet-optimized whitening (WOW)

Done 10 Nov 2026: the **WOW** layer on the Enhance tab (see README, "WOW").
