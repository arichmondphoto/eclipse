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

## Sharper prominences from the short exposures

Pinned 7 Nov 2026. Andy agreed to all of it ("that's how I'd want it built"); to start when he
decides to go forward.

**What was found on the 2024 frames.** The Prominences tab takes the prominences from the HDR
stack, which at a prominence is built mostly from the longest exposures not saturated there
(about 1/20-1/10 s). Those show the prominences swollen and soft, with a red glow round them.
Measured on the two largest prominences (3 and 5 o'clock), edge width from 80 % to 20 % of the
peak, in 2x2 superpixels (smaller is sharper):

| | 3 o'clock | 5 o'clock |
| --- | --- | --- |
| HDR stack | 3.25 | 2.49 |
| 1/4000-1/1000 s single frames | 2.2-2.6 | 1.9-2.0 |
| 1/500-1/160 s single frames | 2.7-3.0 | 2.0-2.2 |
| 1/20 s | 3.51 | 2.05 (visibly swollen) |
| mean of 1/1600, 1/1000, 1/500, 1/160 s | 2.57 | 2.08 |

Red glow 3-8 superpixels outside the prominence, as a share of its peak, at 5 o'clock: HDR 1.0 %,
short frames 0.4-0.6 %. The layer also carries the white corona behind the prominence (so its
edge fades into the background), grows the outline by 3 px and feathers it by 1.5 px, and has no
sharpness or structure control. At 400 mm the prominences are only 30-35 px tall and an edge
spans about 4 full-size px even in the best frames, so expect a clear but limited gain.

(Comparison script and sheet from the session: scratchpad `prom_cmp2.py`, `prom_cmp2.png`; not
kept in the repository.)

**What to build, all three:**

1. **A prominence stack from the short exposures.** The exposure range is chosen on the tab
   (e.g. 1/4000-1/160 s); optionally also the frames excluded near the contacts, where the
   Baily's beads are on the other side of the Sun (they need aligning first: Align skips
   excluded frames). Linear, brightness-matched to the HDR stack (gains from Stack's matching).
2. **Only the prominence's own light.** The white corona behind it is measured from the green
   channel (the corona's R/G from the stack) and taken off, so the layer is the hydrogen emission
   alone, with no grown outline, over the HDR stack's corona.
3. **Sharpness from the data.** The blur measured on the Moon's edge in the short frames (a sharp
   step) gives the point-spread function; deconvolve the prominence stack with it (strength
   setting). Plus a **Structure** setting that brings out strands with the multi-scale method of
   Adaptive (MGN).

Keep to the project rules: widths as fractions of the Sun's radius or measured from the data,
linear until the stretch, hover help for every new setting, and check on the real 2024 frames.
