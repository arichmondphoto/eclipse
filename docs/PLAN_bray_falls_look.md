# Plan: a final image like Bray Falls' blended eclipse

Status: **steps 1 and 2 in the toolkit as options; step 3 measured, waiting on Andy's decision (albedo texture or not)** (1 Nov 2026).
Code at the time of writing: commit `38830ba` on `main`. Tick items off here as they are done.

## Goal

Andy wants the final image (corona with the earthshine Moon) to look like Bray Falls' blended
2024 image (crop shared in the chat on 1 Nov 2026):

- **Corona:** even mid-to-light grey everywhere, nothing clipped, fine continuous loops and
  filaments at many scales, right down to the limb; pink prominences crisp.
- **Moon:** nearly black, neutral grey, faint fine detail (rays, crater rims) out to the limb,
  no darkening towards the limb, not blotchy.
- **Limb:** a crisp edge, with the corona bright right up to it and no glow ring.

Bray Falls' earthshine before and after his own process (also shared): his raw earthshine had a
blue cast, a bright glow round the edge and a magenta flare; afterwards it was grey with ray
systems and maria sharp to the limb. His HDR weight function: rises 0→1 over 0–~5,000 ADU, flat,
falls smoothly from ~40,000 to ~60,000 ADU (of 65,535).

## Where we are (quarter-size test, 1 Nov)

`bf_cmp1.png` (default, then flatten 1.0 + adaptive 1.0 overlay, then also mgn_k 1.5) from the
30 Oct quarter-size HDR (`hdr_bin4.fit`) and an older earthshine run:

- Large white clipped area in the bright inner corona (the natural stretch's white point at the
  99.9th percentile).
- Stronger settings give coarse, noisy streaks, not fine filaments: noise grows faster than the
  structure.
- Moon brighter than his, blotchy, slightly tinted, darkened to 40% at the limb (default
  `limb_darkening` 0.6).
- Soft glowing limb.

Pushing the existing settings harder does not get there: new processing is needed.

## Data needed (Andy is preparing it)

A **full-size crop of the HDR stack around the Sun**, about 2400 × 2400 px, 32-bit FITS (cropped
and saved in Siril, e.g. `hdr_crop_2400.fit`, ~70 MB), plus **the crop's top-left corner**. That
is needed to line it up with the earthshine files (`Eclipse 2024_earthshine*.fit`, crop at
(3466, 2290), Moon at (3972.25, 2796.13) R 435.92 in full-frame px). The matching weight crop
would help too (the full `Eclipse 2024_weight.fit` was sent on 31 Oct; cut the same box from it).
Fine filaments cannot be judged at quarter size, so do not tune steps 1–2 without this crop.

## Steps

Each step is a new **option** that can be switched on and compared in the zoomable preview, not a
replacement for current behaviour. Show Andy results on his crop before changing any default.
Follow CLAUDE.md: real data first, a failing test before each fix, scale widths with the Sun's or
Moon's radius, log what was measured, README and CLI reference kept current.

### 1. Corona tone without clipping

- [x] Smooth highlight roll-off instead of a hard white point: `EnhanceParams.shoulder` /
      `--shoulder` / **Highlight roll-off** (`enhance.roll_off`), in the natural layer and the
      blend; colours scaled to keep their hue instead of clipping per channel. Default 0 (off).
- [x] Full flattening (`flatten` 1.0) works with it.
- [x] Measured on the crop (blend, out to 2.5 Moon radii): default 1.5% clipped, median 0.34;
      `flatten=1.0, shoulder=0.3`: 0% clipped, median 0.31. The prominence at the right keeps a red
      core instead of flat pink.
- [ ] Still open: a bright band ~20 px wide stays by the limb (lighter, but not even grey). That
      is local contrast, so it belongs to step 2.

Notes from the crop (`hdr_crop_2500.fit`, 2530 px, top-left (2715, 1468) full frame): the crop
alone has no sky, so sky removal and white balance fail on it. Test renders paste it into the
quarter-size stack scaled up to full frame (scratch scripts `load2.canvas()`); a full-frame
Enhance takes ~4 min. At 1:1 the strands are soft and grainy: much of the gap to his image is
signal (data), not stretch. Dust spots show as dark rings in the corona (no dust tool, by rule).

### 2. Fine structure without noise

Done as options of the adaptive layer (`EnhanceParams.noise_gate`, `.strands`; `--noise-gate`,
`--strands`; **Adaptive noise gate** and **Smooth along strands** on the Enhance tab), in
`enhance.mgn_gated`:

- [x] Noise per scale from a noise image with the stack's own level (`noise_map`, from the
      weights) and spatial spectrum (`correlated_noise`: patch power spectrum of the faint
      corona's residual, corrected for the 6 px high-pass below which it is taken as flat). White
      noise understated it about 2x after smoothing: demosaicing and resampling correlate
      neighbours (lag-1 correlation 0.26 on the 2024 crop).
- [x] Wiener gain per scale, with signal and noise judged over 3x the scale (narrow windows let
      noise through by chance). Fixed output scale (not stretched to the image's spread), so
      where there is little structure the layer stays flat.
- [x] Smoothing along the strands (`strand_orientation`, `smooth_along`): direction from the
      structure tensor of the mid-scale detail (minus a blur of 10% R_sun, gradient 0.73%,
      tensor 2.4%). The Moon is filled first by widening normalized blurs (`_masked_fill`; the
      old fill left the Moon's middle at 0 and made a ring). Directions are only followed where
      clearer than noise alone gives. The noise image is smoothed along its *own* ridges, mixed
      like the data.
- [x] Test `test_gated_adaptive_does_not_make_texture_out_of_noise`: pure correlated noise gives
      < 0.4x the texture of the plain filter and of real strands.
- [x] On the crop (adaptive layer alone, real weights): fine texture 0.12 inner / 0.06 outer
      corona. The grain is gone and the strands are continuous; the outer corona between
      streamers is smooth (`step2_final_1to1.png`, `fb_0.8_small.png`: natural + FNRGF-N 0.5 +
      adaptive 0.8 soft light, flatten 1, roll-off 0.3).
- Cost: a full frame takes ~6.5 min for Enhance (was ~4) and peaks at ~5.6 GB in the test
  script (which also holds a full-frame canvas).
- Testing pitfall: the crop pasted into the scaled-up quarter-size stack gives a wrong noise
  level (the smooth scaled-up sky sets `noise_constant`), which shows as mottled texture in the
  outer corona. Test the adaptive layer on the crop alone with its weights cut from
  `Eclipse 2024_weight.fit` (scratch `crop_adapt.py` approach).

- [x] Speckles where two strand families meet: the along/round mix now changes over a strand
      length (`smooth_along` blurs its coherence ramp). Much reduced; some remain round the
      dust rings (real features in the stack; no dust tool, by rule).
- [x] Adaptive strength and skipping the finest scales: `mgn_k` / `--adaptive-k` / **Adaptive
      strength**; `mgn_skip_fine` / `--adaptive-skip-fine` / **Adaptive: skip finest scales**.
- [x] Tone like his (light, even grey): `even` / `--even` / **Evenness** takes out a fraction
      of the large-scale differences round the Sun. It is limited to where the corona is above
      the noise, fades in from 1.15 to 1.4 R_sun (by the limb it pushed the gaps between loops
      to black), and takes the black point before. `midtone` / `--midtone` / **Midtone** is a
      midtones transfer curve (PixInsight-style, finite slope at black, not a gamma) putting the
      inner corona's median at the level asked. On the crop, natural (flatten 1, even 0.7,
      midtone 0.6) + adaptive (gate 3, strands 1) 0.8 soft light gives `step2b_small.png`, close
      to his tone. They also lift the faint outer corona and twilight sky (far sky on the bin8
      stack: 0.19 by default, 0.36 at flatten 1, 0.62 with midtone 0.6).

Still open:
- [ ] The paper's MGN form (gamma global + local, h and γ) as an extra layer for comparison.
- [ ] Look: his corona is lighter, more even grey with finer, denser strands. Ours at 0.8 soft
      light is closer but darker between streamers. Try stronger flattening of the local mean
      (an NRGF-like normalization of the base) before the adaptive layer.

### 3. The Moon in his style

**Finding (1 Nov), on Andy's latest earthshine stack (`Eclipse 2024_earthshine_rgb/weight.fit`):**
the lunar surface is noise-dominated at every scale up to ~35 px. Structure/noise, measured with
a noise model independent of the Moon's own detail (correlated like a demosaiced stack, level from
the weights): 1.1 px 1.02, 2 px 1.10, 4 px 1.13, 9 px 1.17, 17 px 1.22, 35 px 1.36. Pixel noise
is 0.15 of the surface's 5–95% contrast. Only the maria (large scales) are really there; craters
and rays like his are not in this data.

- Noise-gated local contrast on the surface (`mgn_gated` on the glare-free detail) changes almost
  nothing: the gate correctly finds little above the noise. A per-octave Wiener filter leaves
  mid-scale blotches, because those bands are only 10–20% above the noise.
- Pitfall: on the Moon, `correlated_noise` sampled from the Moon itself counts real detail as
  noise (structure/noise came out at exactly 1.00). There is no noise-only region in the
  earthshine crop, so use a fixed correlation model (Gaussian σ 0.45, as measured in the corona's
  faint part) or the corona's measured spectrum.
- What the settings already do (no code): `smooth 3, edge_smooth 4, brightness 0.35,
  limb_darkening 0, colour off` gives a darker, flat, neutral, less blotchy Moon
  (`moon_looks.png`), closer to his. It shows small beads at the limb with that smoothing; to
  look at.

Open decision for Andy (asked 1 Nov): his crater and ray detail can only come from outside
this data, i.e. an albedo map blended in as visible texture (a composite, clearly labelled, off by
default). Until now the map has only guided glare removal and never added to the picture.

- [ ] Preset or documented settings for the dark, flat, neutral Moon; fix the limb beads at
      high smoothing.
- [ ] Only if Andy wants it: albedo-map texture as an explicit, labelled composite option.
- [ ] Edge-preserving smoothing for the maria (instead of Gaussian) so their edges stay crisp.

### 4. A crisp limb

- [ ] With the corona flattened and normalized right up to the limb (steps 1–2), check whether
      the glow goes. The edge should be crisp because the stack is accurate there, not because
      anything is cut or blended (CLAUDE.md rule). Re-check the Moon merge's limb zone
      (`hdr_reliable_from`, 3 px on 2024) and the hand-over.

### 5. Optional: HDR weights like Bray Falls'

- [ ] Make the start of the bright-end roll-off a setting (now a linear ramp from 80% to 95% of
      full scale, `StackParams.saturation_level`/`saturation_taper`). Try his smoother roll-off
      from ~61% to ~90%, which allows for sensor non-linearity before clipping. Compare on the
      2024 stack. This needs a re-stack, so do it last and only if steps 1–4 show non-linearity
      near the bright end.

## Expectations to keep stating

His image appears to have much more signal than one D850 sequence (very smooth strands at full
size). Processing can match the look and the tones; where his data is simply deeper, the faintest
parts of ours will show more grain. Say so where that is the limit.

## How to pick this up

1. Read this file and CLAUDE.md.
2. Get the HDR crop (and corner) from Andy if it is not already attached; check it against the
   Moon position above.
3. Start at step 1. Render comparisons at full size on the crop and send them before changing
   defaults. After each step: tests, GUI smoke, README, commit, and tick the box here.
