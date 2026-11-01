# Plan: a final image like Bray Falls' blended eclipse

Status: **step 1 done (option, not default); step 2 next** (1 Nov 2026).
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

- [ ] Edge-preserving noise reduction first (driven by the stack's own noise map,
      `enhance.noise_map`), so strands stay sharp and grain does not.
- [ ] Fine-structure enhancement in the Druckmüller style (adaptive filtering / multi-scale
      normalization at small scales). Each scale is enhanced only where it rises above the noise
      at that scale, so faint regions are not turned into grain.
- [ ] Consider the paper's MGN form (gamma-stretched global image mixed with the local part,
      settings h and γ) as an extra layer for comparison; Andy asked why it was not used. The
      current choice: the natural layer is the global part, and the adaptive layer is blended in
      Soft Light so the sky stays black.
- [ ] Expose the adaptive filter's scales (now fixed at 0.6–20% of the Sun's radius) and its k.
- [ ] Target: continuous filaments and loops like his, measured as fine-scale contrast against
      noise on Andy's crop.

### 3. The Moon in his style

- [ ] Darker earthshine overall (lower `brightness`), `limb_darkening` near 0 or a new "flat"
      look, neutral grey (`colour` off) as a preset.
- [ ] Multi-scale local contrast on the lunar surface (MGN applied to the Moon) to bring out
      rays and craters.
- [ ] Edge-preserving noise reduction for the surface instead of the current Gaussian-based
      smoothing, so it is not blotchy.
- [ ] Detail near the limb: keep improving the glare removal there (now about 0.66 of the
      middle's contrast 15–60 px in); the albedo map guide already helps.

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
