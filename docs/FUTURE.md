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

**Before starting:** ask whether the frames are FITS or SER, colour or mono, and for a few real
test frames from the camera (any subject: a couple of lights at different exposures, a dark and
a flat), since every capture program writes its headers a little differently.

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

## Wavelet-optimized whitening (WOW)

Done 10 Nov 2026: the **WOW** layer on the Enhance tab (see README, "WOW").
