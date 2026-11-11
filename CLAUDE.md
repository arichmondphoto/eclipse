# CLAUDE.md

Guidance for Claude (and anyone else) working on Totality, a solar eclipse toolkit. README.md is the
user documentation and describes every method in detail; this file covers how to work on the code.

## What this is

Processing for DSLR photographs of total solar eclipses, RAW files to finished pictures:
alignment on the corona, Moon-aware linear HDR stacking, coronal enhancement (natural, NRGF,
FNRGF, FNRGF-N, adaptive and WOW layers) and an earthshine Moon. It runs as a PyQt6 script
inside Siril 1.4+ (`siril/Totality.py`) and from the command line as `totality`
(`src/tse_core/cli.py`). Both drive the same functions in `src/tse_core/workflow.py`, and every
step saves its results into one project file (`<event>/<event>.json`).

## Rules that are not up for discussion

- **The 2017 Nikon D800 set is a blind test.** Never use it for tuning, debugging or tests, and
  never add it, or anything derived from it, to the repository. Develop and test only with the
  2024 data and synthetic data.
- **Any camera, any lens, any eclipse.** Nothing may be tied to one eclipse, camera, bracket or
  focal length. Vignetting comes from lensfun or the user's own calibration frames.
- **No dust-spot tool.** Do not add one.
- **Widths near the limb scale with the image.** Do not hard-code pixel counts. Use fractions of
  the Moon's radius (see `EarthshineParams.margin()` and `stack.moon_margin()`), or measure the
  width from the data (see `earthshine.hdr_reliable_from()`). A 1000 mm lens has 2.5× the pixels
  per arcsecond of the 2024 setup.
- **Everything stays linear until the stretch.** Stacks are linear radiance; the earthshine
  stack is in the same units as the HDR stack. Nothing is pasted, feathered or blended to hide a
  join. If a seam shows, find out which data is wrong there.

## Layout

| Path | What |
| --- | --- |
| `src/tse_core/workflow.py` | Steps as used by the GUI and CLI: `align`, `stack`, `enhance_project`, `earthshine`, `final_with_moon`, `final_moon_crop`, `moon_layer` |
| `src/tse_core/align.py`, `geometry.py`, `eclipse.py` | Phase correlation alignment, field rotation, the Sun's centre, Besselian elements |
| `src/tse_core/stack.py` | Moon-aware HDR stacking with measured brightness matching |
| `src/tse_core/enhance.py` | Stretch and filter layers; `MoonInput` lets the Moon be drawn with the same curve |
| `src/tse_core/earthshine.py` | Moon-aligned stack, glare removal (limb terms, prominence fan), lunar surface, `merge_moon` (Moon into the HDR stack) |
| `src/tse_core/parallel.py` | Threads within half the memory: `run_all` (tasks with a byte cost), `ahead` (frames prepared in the background, used in order); `TOTALITY_THREADS` |
| `src/tse_core/viewzoom.py` | Zoom and pan arithmetic for the GUI's picture views (no Qt) |
| `siril/Totality.py` | The Siril window, one tab per step |
| `tools/cli_reference.py` | Generates README's command-line reference from the parser |
| `tests/` | pytest; `tests/qtstub` is a stand-in for PyQt6; `tests/data` holds binned real 2024 data |

## Commands

```bash
pip install -e ".[dev]"                     # rawpy and PyQt6 are optional extras
PYTHONPATH=src python -m pytest -q          # the whole suite, about 1.5 minutes
PYTHONPATH=src python -m pytest -q tests/test_earthshine.py -k merge
PYTHONPATH=src:tests:tests/qtstub python tests/gui_smoke.py /tmp/guitest   # drives the window with the stub
python tools/cli_reference.py               # regenerate README's command-line reference
```

PyQt6 may not install in a sandbox. The GUI is then only checked through `tests/qtstub`, which
does not render, so code that paints, handles mouse or key events, or does layout must be written
carefully against the PyQt6 API. Say so when handing it over.

## How to work here

- **Check on the real data, not only on synthetic data.** Synthetic tests have repeatedly missed
  what real data showed: saturated prominences, low-weight pixels, a weight profile that keeps
  rising past the Moon's coverage. When the user sends files (stacks, weights, results), look at
  the actual pixels before changing code: radial profiles across the limb, weight fractions, and
  crops scaled up with nearest-neighbour.
- **Test before fixing.** For every bug, first write a test that fails on the old code (check it
  with `git stash -- src`), then fix. Where possible, build the test from real figures; see
  `test_hdr_reliable_zone_on_the_2024_weights`.
- **Don't fit tests to the code.** If a test only passes after its numbers are changed, say why
  the new numbers are realistic (for example "as on the 2024 data: white to +1 px").
- **The user cannot see logs or inner workings unless they paste them.** Results that depend on
  data name what they found in a log line (`say(...)`), for example "the HDR stack's corona is
  used from N px outside the limb". The window title shows the commit (`BUILD`), so the user can
  confirm they are running the latest code.
- **Tell the user what to re-run.** Changes to `merge_moon`/`final_with_moon` need only
  **Write final image**. Changes to the earthshine glare removal or the lunar surface need
  **Stack earthshine** again. Changes to stacking need re-stacking.
- **Keep README.md in step.** It documents methods, defaults and file outputs. When a CLI option
  changes, run `tools/cli_reference.py` and paste its output between the
  `<!-- cli-reference:start/end -->` markers (`tests/test_readme.py` fails otherwise). Every CLI
  option needs help text. The GUI tab table in README is updated by hand.
- **Settings shown in the GUI need hover help** (`_tip(...)`) explaining what they do and when
  to change them.
- **Before pushing:** run the full suite and the GUI smoke script.

## Work in progress

- **The Moon's edge in the final image** (paused 7 Nov 2026 at 386d9b3; the user: "better but not
  perfect"): a 1 px step where the Moon meets the corona, at the filter layers' black disc (on his
  stack 0.757 to 0.858 at 1.5 px, MGN 45 % Weighted Average). His test case, a 3000 px crop of his
  2024 stack with his earthshine files and settings, is on his computer in
  `~/Desktop/Eclipse 2024/limb_test_case` (README.txt, `limb_check.py`). Stage it from there when
  he asks to continue; never add it to the repository.
- **the user's list for the next session (8 Nov 2026, at 8a9388c).** Done before it: Align no longer fails
  short frames for a small overlap (74e5b6b); **Colour by the limb** (5fdd560); **Moon meets corona**,
  the per-pixel rule "corona first" (8a9388c). His frame-9 files were staged as a 3000 px crop in
  `~/Desktop/Eclipse 2024/_claude_trial` (hdr.fit, weight.fit; origin 2466, 1292); the earthshine
  files are in `fits/`. He is short of usage: keep sessions lean and do not re-derive this.
  1. Done (10 Nov 2026): the rule holds for Paste into a file too, and under the rule the earthshine
     stack is used wherever it has data (1119 black px on frame 9 down to 280 that no frame recorded).
  2. Done (10 Nov 2026): **Write Moon layer (for Photoshop)** was right (`_moon.tif`, checked on frame 9); the
     file like Stack earthshine's is `_earthshine.tif`, the old paste layer (its box: item 5). Paste into a file
     removed; **Moon's edge**: Corona as stacked / Corona blended into the Moon (`--moon-edge`).
  3. Done (10 Nov 2026): Earthshine **Preview** is a full-size box round the Moon (`final_moon_crop`,
     1.3 R each side), the very pixels Write final image writes.
  4. Done (10 Nov 2026): **Write full-size layers** writes only the ticked layers and the Blend (unticked
     layers the Blend uses were written too).
  5. Done (10 Nov 2026): save dialogs suggest `workflow.output_name` names (`_earthshine_stack`,
     `_earthshine_blend|natural`, `_earthshine_photoshop` for the Moon layer) in the format chosen last
     (project `image_ext`).
  6. Done (10 Nov 2026, not seen rendered): settings panels open at their content's width plus the scroll
     bar, at most 60 % of the tab (`viewzoom.settings_panel_width`, on first show).
  7. Done (10 Nov 2026): **WOW** layer (Auchère et al. 2023; `enhance.wow`), plain/edge-aware, n_s 5,2,1.
     Departures, all in README: noise per scale measured on the stack (not white), no c_S (units), the
     Moon's disc filled by odd reflection through the limb (`wow_disc_fill`; no white band).
  8. Done (10 Nov 2026): the Prominences tab removed on the user's go-ahead (tab, `prominences.py`, the
     prominence commands, the hook in Write final image). Red-weight brightness: `docs/FUTURE.md`.
  9. In progress (10 Nov 2026). Recommended order: 4 log timings, 1 ACHF polar resampling once for all
     scales and parts (same pixels; about half its time), 2 multi-core, 3 keep whole-image measurements
     between Earthshine previews, 5 GPU only if needed. The user chose 4 and 2 first and wants their gain
     measured before any bigger change (he tests on an M5 MacBook Air, 24 GB, and an M1 Ultra Mac
     Studio, 64 GB). Done: 4 (elapsed time on every log line, "Finished in") and 2 (`parallel.py`:
     frames in Align/Stack/Stack earthshine, Enhance layers, ACHF rings, FFT workers; results identical,
     test_parallel.py). On 2 cores, frame 9 crop: layers 12.7 -> 9.3 s, ACHF 189 -> 108 s. The user (M5 Air):
     stacking at least 5x faster, ACHF "fast enough now". Step 1 declined (10 Nov 2026): sharing the
     polar resampling would only work per part and needs one grid for all scales, so not bit-identical;
     do not propose it again. Step 3 done: `final_with_moon` keeps the last preview's prepared frame
     (`_FINAL_PREVIEW`) while only the strength changes; frame 9: 13.4 s, then 1.8 s per strength.
- `docs/FUTURE.md`: future additions, if needed or requested (astronomy cameras such as ZWO:
  FITS and SER frames, gain, mono; prominence brightness by red weight while stacking). Not to be
  started unless the user asks; read it when they do.

## Commits

- Commit and push to `main` (the remote is `arichmondphoto/eclipse`).
- Author: `Claude <noreply@anthropic.com>`. End the message with the attribution lines given for
  the current session (`Co-Authored-By: ...` and `Claude-Session: ...`).
- Write messages that explain why: what was wrong, what changed, and how it was tested.

## Writing for the user

The user is a photographer, not a programmer, and works in the Siril window. In replies, name the
buttons and settings as they appear on the tab, give the commit to look for in the window title,
and say plainly what was checked on real data and what only on simulated data.
