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
6. Mono cameras: grey output, with the colour steps (white balance, prominences by colour,
   earthshine tint) switched off. Colour through filters would be a larger project.

**Before starting:** ask whether the frames are FITS or SER, colour or mono, and for a few real
test frames from the camera (any subject: a couple of lights at different exposures, a dark and
a flat), since every capture program writes its headers a little differently.

## Prominences from the frames left out near the contacts

Built 7 Nov 2026: Stack prominences (the short aligned frames), used to measure how much softer
each prominence is in the HDR stack, which Write final image undoes. Not built: also using the
frames marked as excluded near the contacts (on the 2024 set frames 0-4, 1/6400-1/2000 s, 30-35 s
after second contact), which Align leaves out. They would need aligning to the existing reference
(shift only when no field rotation is applied; a Moon fit for each), and brightness matching of
their own, since the HDR stack has no gains for them.

## Wavelet-optimized whitening (WOW) next to MGN on the Enhance tab

Pinned 7 Nov 2026; Andy wants it added. WOW (Auchère et al., 2023, Astronomy & Astrophysics) splits the
image into wavelet layers (à trous, B3-spline kernel, ln N / ln 2 scales; the layers add up
exactly to the image), divides each layer by its local power (the squared layer smoothed with that
scale's own kernel) and adds them back with equal weight, so there are no scale weights to set.
Noise: each coefficient kept by erf(|w| / (n_s σ_s)), with n_s = 5, 2, 1 on the three finest
scales in the paper; σ_s from the stack's noise map times each scale's response to unit white
noise. Edge-aware version: each smoothing step weights neighbours by exp(-½ (Δc / ν_s)²), ν_s the
local spread at that scale, to keep halos off sharp edges.

**Tested on the 2024 stack (7 Nov 2026, scratch code from the paper's equations):** more
structure and much less grain than MGN with the same settings as Andy's; plain WOW closest to the
present look in his 45 % Average blend; edge-aware WOW the smoothest, with fine rays as thin,
often dark lines and the large-scale brightness flattened. No dark rings round the prominences in
any of them.

**What it needs:**

1. The bright band a few pixels wide that both versions put along the limb (the step from the
   inner corona to the filled Moon) solved first, as for MGN: the layer must see the corona
   continued to the limb, not the Moon's edge (see `MoonInput.corona_gain` / `Frame.filter_lum`).
2. A layer `wow` in `enhance.py` next to Adaptive (MGN), working on the whole image or a box
   (`origin`, a reach for the detail box's margin, like `_mgn_reach`).
3. Settings with `_tip` help: plain or edge-aware, and noise suppression (n_s); the README's
   Enhance section and CLI reference updated.
4. Speed: plain WOW took 3 s and edge-aware 10 s for 1792 px square on 2 cores; full frames
   need tiling or the edge-aware steps in a compiled form.
5. Tests: synthetic (layers add up to the image; white noise comes out flat) and on the binned
   2024 data in `tests/data`.
