"""The processing steps as plain functions, shared by the command line and the GUI.

Each step reads and updates the event's project file, reports progress
through ``say`` and raises ``WorkflowError`` with a message meant for the
user when it cannot go on. Passing ``cancel`` (a callable returning True
when the user pressed Cancel) makes the next progress report raise
``Cancelled``; the step stops there without writing a half-finished result.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .enhance import LAYERS
from .lensfun import LensfunDatabase
from .project import EventProject
from .vignetting_profile import load_profile_library

Say = Callable[[str], None]


class WorkflowError(Exception):
    """A problem the user can fix; the message says how."""


class Cancelled(Exception):
    """The user stopped the step."""


def reporter(say: Optional[Say] = None, cancel: Optional[Callable[[], bool]] = None) -> Say:
    def _say(msg: str) -> None:
        if cancel is not None and cancel():
            raise Cancelled()
        if say is not None:
            say(msg)

    return _say


def resolve(project_path, stored: Optional[str]) -> str:
    """Find a file the project refers to, wherever the program was started from.

    Older project files (and the command line run from another folder) store
    paths relative to the folder the command ran in. Try the path as stored,
    then relative to the project file's folder, then the same file name next
    to the project file. Returns "" when nothing exists.
    """
    if not stored:
        return ""
    p = Path(stored)
    base = Path(project_path).resolve().parent
    for cand in ((p,) if p.is_absolute() else ()) + (base / p, base / p.name, p):
        if cand.exists():
            return str(cand.resolve())
    return ""


def _locate_frames(project_path, frames) -> None:
    """Point each frame at its RAW file wherever it is found (see ``resolve``); report the missing."""
    missing = []
    for f in frames:
        found = resolve(project_path, f.path)
        if found:
            f.path = found
        else:
            missing.append(f.file_name)
    if missing:
        raise WorkflowError(f"{len(missing)} RAW file(s) not found, for example {missing[0]}. If the folder moved, "
                            "read it again on the Frames tab (or run 'tse inventory' on the new folder).")


def _db(path: Optional[str]) -> LensfunDatabase:
    return LensfunDatabase.load([path] if path else None)


def ensure_elements(day: str, say: Say) -> None:
    """Besselian elements for this date, fetching NASA's catalogue once if the eclipse is not bundled."""
    from .eclipse import CANON_CACHE, elements_for, update_canon

    try:
        elements_for(day)
        return
    except KeyError:
        if CANON_CACHE.is_file():
            raise
    say("This eclipse is not in the bundled set; fetching NASA's full catalogue once.")
    update_canon()
    elements_for(day)


def _plan(project: EventProject, lights, lensfun_db, profile_dir):
    from .calibration import plan_calibration

    db = _db(lensfun_db)
    plan = plan_calibration(lights, db, None, lens_name=project.lens_profile or None,
                            use_lensfun=project.use_lensfun, profiles=load_profile_library(profile_dir))
    cam = db.find_camera(lights[0].camera_make, lights[0].camera_model) if lights and not db.empty else None
    return plan, (cam.crop_factor if cam else 1.0)


# ---------------------------------------------------------------------------
# Align
# ---------------------------------------------------------------------------


@dataclass
class AlignOptions:
    latitude: Optional[float] = None  # None = keep the project's value
    longitude: Optional[float] = None
    elevation_m: Optional[float] = None
    camera_utc_offset_h: Optional[float] = None
    mirrored: Optional[bool] = None
    clock_error_s: Optional[float] = None
    rotation: str = "drift"  # drift | per-frame | none
    any_angle: bool = False
    rho: Optional[float] = None
    window: Optional[int] = None
    output: Optional[str] = None
    profile_dir: Optional[str] = None
    lensfun_db: Optional[str] = None


def align(project_path, opts: AlignOptions, say: Optional[Say] = None, cancel=None):
    """Align the light frames and find the Sun's centre. Returns (AlignmentResult, path written)."""
    from .align import AlignParams
    from .eclipse import Site
    from .pipeline import default_loader, run_alignment

    say = reporter(say, cancel)
    path = Path(project_path).resolve()
    project = EventProject.load(path)
    for attr in ("latitude", "longitude", "elevation_m", "camera_utc_offset_h", "mirrored", "clock_error_s"):
        value = getattr(opts, attr)
        if value is not None:
            setattr(project, attr, value)
    lights = project.inventory.lights()
    if len(lights) < 2:
        raise WorkflowError("Need at least two light frames: read the RAW folder on the Frames tab first.")
    _locate_frames(path, lights)
    plan, crop = _plan(project, lights, opts.lensfun_db, opts.profile_dir)
    site = None
    if project.latitude is not None and project.longitude is not None:
        site = Site(project.latitude, project.longitude, project.elevation_m)
    if site is not None and lights[0].timestamp:
        try:
            ensure_elements(lights[0].timestamp[:10], say)
        except KeyError as exc:
            say(str(exc.args[0]))
        except OSError as exc:
            say(f"Could not download NASA's eclipse catalogue ({exc}); the Sun centre will be skipped. "
                "Run 'tse elements-update --from <file>' with a copy downloaded in a browser.")
    params = AlignParams(rho=opts.rho, rotation_model=opts.rotation, coarse_rotation=opts.any_angle,
                         solve_rotation=opts.rotation != "none")
    try:
        result = run_alignment(lights, default_loader(plan, crop), params, window=opts.window, site=site,
                               clock_error_s=project.clock_error_s, crop_factor=crop,
                               utc_offset_h=project.camera_utc_offset_h, mirrored=project.mirrored, progress=say)
    except (RuntimeError, ValueError) as exc:
        raise WorkflowError(str(exc)) from exc
    out = Path(opts.output) if opts.output else path.with_name(path.stem + "_align.json")
    result.save(out)
    project.parameters["align"] = {"file": str(out), "rho": opts.rho, "rotation": opts.rotation,
                                   "window": opts.window, "clock_error": opts.clock_error_s}
    project.save(path)
    return result, out


def load_alignment(project_path, align_file: Optional[str] = None) -> tuple[dict, Path]:
    path = Path(project_path).resolve()
    project = EventProject.load(path)
    align_path = Path(resolve(path, align_file or project.parameters.get("align", {}).get("file"))
                      or resolve(path, path.stem + "_align.json") or path.with_name(path.stem + "_align.json"))
    if not align_path.exists():
        raise WorkflowError(f"No alignment results at {align_path}; run the Align step first.")
    return json.loads(align_path.read_text(encoding="utf-8")), align_path


# ---------------------------------------------------------------------------
# Stack
# ---------------------------------------------------------------------------


@dataclass
class StackOptions:
    align: Optional[str] = None
    output: Optional[str] = None
    name: Optional[str] = None
    radii: Optional[float] = None  # None = full frame
    moon_frame: Optional[int] = None
    include_rejected: bool = False
    hot_pixels: bool = True
    profile_dir: Optional[str] = None
    lensfun_db: Optional[str] = None


def _frame_infos(project: EventProject, alignment: dict, include_rejected: bool):
    """Light frames with their alignment, Moon and nominal brightness, for stacking.

    Returns (frames by index, reference index, [FrameInfo], [skipped messages]).
    """
    from .frames import true_exposure
    from .stack import FrameInfo

    frames = {f.index: f for f in project.inventory.lights()}
    ref = alignment["reference_index"]
    rejected = set()
    if alignment.get("sun") and not include_rejected:
        order = [fr["index"] for fr in alignment["frames"]]
        rejected = {order[k] for k in alignment["sun"].get("rejected", [])}

    def _ev(fm):  # light gathered, using the shutter's real time behind its rounded label
        if not (fm.exposure_s and fm.aperture_f and fm.iso):
            return None
        return math.log2(true_exposure(fm.exposure_s) * fm.iso / 100.0 / fm.aperture_f**2)

    ev_ref = _ev(frames[ref]) if ref in frames else None
    infos, skipped = [], []
    for fr in alignment["frames"]:
        i = fr["index"]
        if i not in frames:
            skipped.append(f"frame #{i} ({fr['file_name']}) is no longer a light frame in the project; skipped")
            continue
        if i in rejected:
            skipped.append(f"frame #{i} ({fr['file_name']}) was flagged as off the Moon's track; skipped "
                           "(include rejected frames to keep it)")
            continue
        ev = _ev(frames[i])
        ratio = 2.0 ** (ev - ev_ref) if ev is not None and ev_ref is not None else 1.0
        mx, my = fr["moon_ref"]
        infos.append(FrameInfo(i, fr["file_name"], ratio, fr["matrix"], mx, my, fr["moon_frame"]["radius"]))
    return frames, ref, infos, skipped


def stack(project_path, opts: StackOptions, say: Optional[Say] = None, cancel=None):
    """Merge the aligned frames into the linear HDR stack. Returns (StackResult, skipped messages)."""
    from .rawio import read_mosaic
    from .stack import StackParams, stack_frames

    say = reporter(say, cancel)
    path = Path(project_path).resolve()
    project = EventProject.load(path)
    alignment, align_path = load_alignment(path, opts.align)
    frames, ref, infos, skipped = _frame_infos(project, alignment, opts.include_rejected)
    for msg in skipped:
        say(msg)
    if not infos:
        raise WorkflowError("Nothing to stack: no aligned light frames.")
    lights = [frames[f.index] for f in infos]
    _locate_frames(path, lights)
    plan, crop_factor = _plan(project, lights, opts.lensfun_db, opts.profile_dir)
    calibs = {fc.frame_index: fc for fc in plan.frames}
    params = StackParams(crop_radii=opts.radii, moon_frame=opts.moon_frame, hot_pixels=opts.hot_pixels)
    out_dir = Path(opts.output) if opts.output else path.parent
    name = opts.name or path.stem
    try:
        result, _hdr, _w = stack_frames(infos, lambda f: read_mosaic(frames[f.index].path), calibs,
                                        tuple(alignment["image_size"]), ref, alignment.get("sun"), params,
                                        out_dir=out_dir, name=name, image_crop=crop_factor, progress=say)
    except (RuntimeError, ValueError) as exc:
        raise WorkflowError(str(exc)) from exc
    project.parameters["stack"] = {"align": str(align_path), "radii": opts.radii, "moon_frame": result.moon_frame,
                                   **result.outputs}
    project.save(path)
    return result, skipped


# ---------------------------------------------------------------------------
# Enhance
# ---------------------------------------------------------------------------


@dataclass
class HdrData:
    hdr: np.ndarray  # (h, w, 3) linear
    weight: np.ndarray
    header: dict
    path: str
    notes: list = field(default_factory=list)
    binning: int = 1


def parse_layers(text: str) -> tuple[str, ...]:
    layers = tuple(x.strip() for x in text.split(",") if x.strip())
    bad = [x for x in layers if x not in LAYERS]
    if bad:
        raise WorkflowError(f"unknown layer(s): {', '.join(bad)}; choose from {', '.join(LAYERS)}")
    return layers


def parse_blend(text: Optional[str]) -> dict:
    blend: dict = {}
    if not text or text.strip().lower() == "none":
        return blend
    try:
        for item in text.split(","):
            k, v = item.split("=")
            blend[k.strip()] = float(v)
    except ValueError as exc:
        raise WorkflowError("The blend takes layer=weight pairs, for example natural=1,fnrgf_n=0.7,adaptive=0.5") \
            from exc
    bad = [k for k in blend if k not in LAYERS]
    if bad:
        raise WorkflowError(f"unknown blend layer(s): {', '.join(bad)}; choose from {', '.join(LAYERS)}")
    return blend


def load_hdr(project_path, hdr_file: Optional[str] = None) -> HdrData:
    from .fits import read_fits

    path = Path(project_path).resolve()
    project = EventProject.load(path)
    stack_info = project.parameters.get("stack", {})
    hdr_path = resolve(path, hdr_file or stack_info.get("hdr")) or resolve(path, path.stem + "_hdr.fit")
    if not hdr_path:
        raise WorkflowError(f"No HDR stack found{' at ' + hdr_path if hdr_path else ''}; run the Stack step first.")
    weight_path = resolve(path, str(hdr_path).replace("_hdr.fit", "_weight.fit") if hdr_file
                          else stack_info.get("weight")) or resolve(path, str(hdr_path).replace("_hdr.fit", "_weight.fit"))
    data, header = read_fits(hdr_path)
    hdr = np.ascontiguousarray(np.moveaxis(data, 0, -1)) if data.ndim == 3 else np.repeat(data[..., None], 3, 2)
    del data
    notes = []
    if weight_path and Path(weight_path).exists():
        weight, _ = read_fits(weight_path)
    else:
        notes.append("no weight map found; treating every pixel as equally reliable (FNRGF-N noise estimate is rough)")
        weight = np.ones(hdr.shape[:2], np.float32)
    if "MOONX" not in header:
        raise WorkflowError(f"{hdr_path} has no Moon/Sun geometry in its header; was it made by the Stack step?")
    return HdrData(hdr, weight, header, str(hdr_path), notes)


def reduce_for_preview(data: HdrData, max_side: int = 1400) -> HdrData:
    """A binned copy for quick previews: pixel values averaged, geometry scaled to match.

    Pixels with no data stay at zero weight; the binned weight is the sum
    of the weights (inverse variances add when pixels are averaged).
    """
    h, w = data.hdr.shape[:2]
    b = max(1, int(math.ceil(max(h, w) / max_side)))
    if b == 1:
        return data
    hh, ww = h // b, w // b
    wt = data.weight[: hh * b, : ww * b].reshape(hh, b, ww, b)
    good = (wt > 0).astype(np.float32)
    n = good.sum(axis=(1, 3))
    x = data.hdr[: hh * b, : ww * b].reshape(hh, b, ww, b, 3)
    with np.errstate(invalid="ignore", divide="ignore"):
        hdr = (x * good[..., None]).sum(axis=(1, 3)) / n[..., None]
    hdr = np.nan_to_num(hdr, nan=0.0).astype(np.float32)
    weight = wt.sum(axis=(1, 3)).astype(np.float32) * (n == b * b)  # partly empty bins get no weight
    header = dict(data.header)
    for k in ("SUNX", "SUNY", "MOONX", "MOONY"):
        if k in header:
            header[k] = (float(header[k]) + 0.5) / b - 0.5
    for k in ("SUNR", "MOONR"):
        if k in header:
            header[k] = float(header[k]) / b
    return HdrData(hdr, weight, header, data.path, list(data.notes), data.binning * b)


def enhance_preview(data: HdrData, params, say: Optional[Say] = None, cancel=None) -> tuple:
    """Run the enhancement on (usually reduced) data without writing files. Returns (result, layers)."""
    from .enhance import enhance

    say = reporter(say, cancel)
    scale = 1.0 / data.binning
    p = _scaled_params(params, scale)
    return enhance(data.hdr.copy(), data.weight, data.header, p, out_dir=None, progress=say)


def _scaled_params(params, scale: float):
    """Pixel-sized settings follow the binning so the preview matches the full-size result."""
    from dataclasses import replace

    if params.mgn_scales:
        return replace(params, mgn_scales=[s * scale for s in params.mgn_scales])
    return params


def enhance_project(project_path, params, hdr_file: Optional[str] = None, output: Optional[str] = None,
                    name: Optional[str] = None, say: Optional[Say] = None, cancel=None):
    """Full-resolution enhancement, writing the TIFF layers. Returns (EnhanceResult, notes).

    Loads the HDR stack itself (the enhancement works on it in place to save memory).
    """
    from .enhance import enhance

    say = reporter(say, cancel)
    path = Path(project_path).resolve()
    data = load_hdr(path, hdr_file)
    for n in data.notes:
        say(n)
    out_dir = Path(output) if output else path.parent
    result, _ = enhance(data.hdr, data.weight, data.header, params, out_dir=out_dir, name=name or path.stem,
                        progress=say)
    project = EventProject.load(path)
    project.parameters["enhance"] = {"hdr": data.path, **result.layers}
    project.save(path)
    return result, data.notes


# ---------------------------------------------------------------------------
# Earthshine
# ---------------------------------------------------------------------------


def earthshine(project_path, params=None, include_rejected: bool = False, output: Optional[str] = None,
               name: Optional[str] = None, profile_dir: Optional[str] = None, lensfun_db: Optional[str] = None,
               say: Optional[Say] = None, cancel=None):
    """Moon-aligned stack of the long exposures; writes a layer that fits the corona layers' black disc.

    Needs the alignment and the HDR stack (for the measured frame brightness and the canvas).
    Returns the EarthshineResult.
    """
    from .earthshine import EarthshineParams, stack_earthshine
    from .rawio import read_mosaic
    from .stack import Match

    say = reporter(say, cancel)
    path = Path(project_path).resolve()
    project = EventProject.load(path)
    alignment, _ = load_alignment(path)
    st = project.parameters.get("stack", {})
    info_path = resolve(path, st.get("info"))
    if not info_path:
        raise WorkflowError("Run the Stack step first: earthshine uses the brightness it measures for each frame.")
    info = json.loads(Path(info_path).read_text(encoding="utf-8"))
    matches = {m["index"]: Match(**m) for m in info["matches"]}
    frames, ref, infos, skipped = _frame_infos(project, alignment, include_rejected)
    moon_frame = info.get("moon_frame", ref)
    mf = next((fr for fr in alignment["frames"] if fr["index"] == moon_frame), None)
    if mf is None:
        raise WorkflowError(f"The Moon frame #{moon_frame} is not in the alignment.")
    moon_xy = tuple(mf["moon_ref"])
    moon_r = float(mf["moon_frame"]["radius"])
    canvas = tuple(info["crop"]) if len(info["crop"]) == 4 else (info["crop"][0], info["crop"][1], info["crop"][2],
                                                                 info["crop"][2])
    p = params or EarthshineParams()
    gmax = max((matches[f.index].gain for f in infos if f.index in matches), default=0)
    use = [f for f in infos if f.index in matches and matches[f.index].gain >= gmax * 2.0 ** (-p.stops)]
    if not use:
        raise WorkflowError("No frames to use for earthshine.")
    lights = [frames[f.index] for f in use]
    _locate_frames(path, lights)
    plan, crop_factor = _plan(project, lights, lensfun_db, profile_dir)
    calibs = {fc.frame_index: fc for fc in plan.frames}
    out_dir = Path(output) if output else path.parent
    try:
        result, _lum, _detail = stack_earthshine(use, lambda f: read_mosaic(frames[f.index].path), calibs, matches,
                                                 moon_xy, moon_r, canvas, p, out_dir=out_dir,
                                                 name=name or path.stem, image_crop=crop_factor, progress=say)
    except (RuntimeError, ValueError) as exc:
        raise WorkflowError(str(exc)) from exc
    project.parameters["earthshine"] = {**result.outputs, "frames": result.frames}
    project.save(path)
    return result


def read_image(path: str, step: int = 1) -> np.ndarray:
    """A 16-bit (or 8-bit) TIFF/PNG layer as float RGB in 0..1, optionally every ``step``-th pixel."""
    if str(path).lower().endswith((".tif", ".tiff")):
        import tifffile

        data = tifffile.imread(path)
    else:
        from PIL import Image

        data = np.asarray(Image.open(path).convert("RGB"))
    data = data[::step, ::step]
    if data.ndim == 2:
        data = np.repeat(data[..., None], 3, axis=2)
    scale = 65535.0 if data.dtype == np.uint16 else 255.0
    return data[..., :3].astype(np.float32) / scale


def final_image(project_path, base_layer: str = "blend", strength: float = 1.0, mode: str = "screen",
                preview_side: Optional[int] = None, name: Optional[str] = None,
                say: Optional[Say] = None, cancel=None):
    """The chosen enhanced layer with the earthshine blended into the lunar disc.

    With ``preview_side`` returns a reduced image (no file); otherwise writes
    ``<name>_final.tif`` next to the project and returns its path.
    """
    from .earthshine import composite

    say = reporter(say, cancel)
    path = Path(project_path).resolve()
    project = EventProject.load(path)
    base_path = resolve(path, project.parameters.get("enhance", {}).get(base_layer))
    if not base_path:
        raise WorkflowError(f"No '{base_layer}' layer yet: write the full-size layers on the Enhance tab first.")
    earth_path = resolve(path, project.parameters.get("earthshine", {}).get("earthshine"))
    if not earth_path:
        raise WorkflowError("No earthshine layer yet: click 'Stack earthshine' first.")
    step = 1
    if preview_side:
        import tifffile

        with tifffile.TiffFile(base_path) as t:
            h, w = t.pages[0].shape[:2]
        step = max(1, int(math.ceil(max(h, w) / preview_side)))
    say("reading the layers")
    base = read_image(base_path, step)
    earth = read_image(earth_path, step)
    if base.shape != earth.shape:
        raise WorkflowError("The earthshine layer and the enhanced layer differ in size: re-run the earthshine "
                            "step after the latest stack.")
    out = composite(base, earth, strength, mode)
    if preview_side:
        return out
    say("writing the final image")
    import tifffile

    final = path.parent / f"{name or path.stem}_final.tif"
    tifffile.imwrite(final, (out * 65535 + 0.5).astype(np.uint16), photometric="rgb")
    project.parameters["final"] = {"file": str(final), "base": base_layer, "strength": strength, "mode": mode}
    project.save(path)
    return str(final)
