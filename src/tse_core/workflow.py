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
from dataclasses import asdict, dataclass, field
from datetime import datetime
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
                            "read it again on the Frames tab (or run 'totality inventory' on the new folder).")


def _db(path: Optional[str]) -> LensfunDatabase:
    return LensfunDatabase.load([path] if path else None)


def ensure_elements(day: str, say: Say) -> None:
    """Besselian elements for this date, fetching NASA's catalogue once if the eclipse is not bundled."""
    from .event import ensure_elements as _ensure

    _ensure(day, say)


def _plan(project: EventProject, lights, lensfun_db, profile_dir, calib=None):
    from .calibration import plan_calibration

    db = _db(lensfun_db)
    plan = plan_calibration(lights, db, calib, lens_name=project.lens_profile or None,
                            use_lensfun=project.use_lensfun, profiles=load_profile_library(profile_dir))
    cam = db.find_camera(lights[0].camera_make, lights[0].camera_model) if lights and not db.empty else None
    return plan, (cam.crop_factor if cam else 1.0)


def _calibration_frames(project_path, project: EventProject, say):
    """The bias, dark and flat frames chosen on the Frames tab (or with --flats etc.), read."""
    from . import metadata
    from .calibration import CalibrationFrames

    groups = {}
    for kind, files in (("bias", project.bias_files), ("darks", project.dark_files), ("flats", project.flat_files)):
        found = [resolve(project_path, f) for f in files]
        missing = [Path(f).name for f, g in zip(files, found) if not g]
        if missing:
            say(f"warning: {len(missing)} {kind} file(s) not found (for example {missing[0]}); "
                f"choose the {kind} folder again on the Frames tab")
        found = [g for g in found if g]
        groups[kind] = metadata.read_frames(found) if found else []
    return CalibrationFrames(**groups)


def _calibrated_plan(project_path, project: EventProject, lights, lensfun_db, profile_dir, flat_mode, say,
                     out_dir=None, name="stack"):
    """The calibration plan for Stack and the earthshine stack, with the master frames attached.

    ``flat_mode`` None means the calibration frames are not used (a stack made before they were),
    so that the earthshine stack stays in the same units as that HDR stack.
    """
    from .masters import attach_masters, describe_frames, flat_mismatch
    from . import rawio

    if flat_mode is None:
        plan, crop = _plan(project, lights, lensfun_db, profile_dir)
        info = {}
    else:
        calib = _calibration_frames(project_path, project, say)
        plan, crop = _plan(project, lights, lensfun_db, profile_dir, calib)
        why = flat_mismatch(lights, calib)
        if why:
            say("warning: " + why)
        info = attach_masters(plan, calib, lambda p: rawio.read_mosaic(p), lights, flat_mode, crop, say,
                              out_dir=out_dir, name=name)
    for w in plan.warnings:
        say("warning: " + w)
    for line in describe_frames(plan):
        say(line)
    return plan, crop, info


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
            ensure_elements(project.eclipse_date or lights[0].timestamp[:10], say)
        except KeyError as exc:
            say(str(exc.args[0]))
        except OSError as exc:
            say(f"Could not download NASA's eclipse catalogue ({exc}); the Sun centre will be skipped. "
                "Run 'totality elements-update --from <file>' with a copy downloaded in a browser.")
    params = AlignParams(rho=opts.rho, rotation_model=opts.rotation, coarse_rotation=opts.any_angle,
                         solve_rotation=opts.rotation != "none")
    try:
        result = run_alignment(lights, default_loader(plan, crop), params, window=opts.window, site=site,
                               clock_error_s=project.clock_error_s, crop_factor=crop,
                               utc_offset_h=project.camera_utc_offset_h, mirrored=project.mirrored,
                               eclipse_date=project.eclipse_date, progress=say)
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
    flat_mode: str = "master"  # how flat frames are used: master | profile | compare (see masters.FLAT_MODES)


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
    out_dir = Path(opts.output) if opts.output else path.parent
    name = opts.name or path.stem
    try:
        plan, crop_factor, calib_info = _calibrated_plan(path, project, lights, opts.lensfun_db, opts.profile_dir,
                                                         opts.flat_mode, say, out_dir=out_dir, name=name)
    except ValueError as exc:
        raise WorkflowError(str(exc)) from exc
    calibs = {fc.frame_index: fc for fc in plan.frames}
    # What the reference frame is divided by, small enough to send or open quickly in Siril.
    from .fits import write_fits
    from .masters import correction_preview

    small, text = correction_preview(calibs.get(ref), tuple(alignment["image_size"]), crop_factor)
    say(text)
    if small is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        vpath = out_dir / f"{name}_vignetting.fit"
        write_fits(vpath, small, {"COMMENT": "Correction applied by Stack (frames divided by this), 1/8 size"})
        say(f"  saved as {vpath.name} (1/8 size; 1.0 = the middle)")
        calib_info = {**calib_info, "vignetting": str(vpath)}
    params = StackParams(crop_radii=opts.radii, moon_frame=opts.moon_frame, hot_pixels=opts.hot_pixels)
    try:
        result, _hdr, _w = stack_frames(infos, lambda f: read_mosaic(frames[f.index].path), calibs,
                                        tuple(alignment["image_size"]), ref, alignment.get("sun"), params,
                                        out_dir=out_dir, name=name, image_crop=crop_factor, progress=say)
    except (RuntimeError, ValueError) as exc:
        raise WorkflowError(str(exc)) from exc
    project.parameters["stack"] = {"align": str(align_path), "radii": opts.radii, "moon_frame": result.moon_frame,
                                   **calib_info, **result.outputs}
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


def parse_blend_modes(text: Optional[str]) -> dict:
    """The modes in a blend string: 'fnrgf_n=0.7,adaptive=0.1:overlay' -> {'adaptive': 'overlay'}."""
    modes = {}
    if not text or text.strip().lower() == "none":
        return modes
    for item in text.split(","):
        if ":" in item and "=" in item:
            k, rest = item.split("=", 1)
            from .enhance import blend_mode_name

            mode = blend_mode_name(rest.split(":", 1)[1])
            if mode not in ("soft light", "overlay", "linear"):
                raise WorkflowError(f"unknown blend mode {mode!r}; use soft light, overlay or linear")
            modes[k.strip()] = mode
    return modes


def parse_blend(text: Optional[str]) -> dict:
    """Layer weights in a blend string, e.g. 'natural=1,fnrgf_n=0.7,adaptive=0.1:overlay'."""
    blend: dict = {}
    if not text or text.strip().lower() == "none":
        return blend
    try:
        for item in text.split(","):
            k, v = item.split("=")
            blend[k.strip()] = float(v.split(":")[0])
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


def preview_frame_key(params) -> tuple:
    """What a prepared stack depends on: the sky and white balance settings (and the Moon margin)."""
    return (bool(params.neutral_background), str(params.white), float(params.moon_margin_px))


def enhance_whole_preview(data: HdrData, params, frame=None, say: Optional[Say] = None, cancel=None) -> tuple:
    """The reduced preview of every layer, from ``frame`` (the reduced stack prepared earlier with the
    same :func:`preview_frame_key`) or a newly prepared one. Each layer is drawn again only when its
    own settings changed. Returns (frame, result, layers)."""
    from .enhance import LAYERS, EnhanceResult, _mgn_scales, prepare, render

    say = reporter(say, cancel)
    p = _scaled_params(params, 1.0 / data.binning)
    if frame is None:
        frame = prepare(data.hdr.copy(), data.weight, data.header, p, progress=say)
    layers = render(frame, p, progress=say, cache=True)
    notes = []
    unknown = set(p.blend) - set(LAYERS)
    if unknown:
        notes.append(f"unknown blend layer(s) ignored: {', '.join(sorted(unknown))}")
    result = EnhanceResult({}, frame.wb.tolist(), frame.nlev, frame.bg.tolist(), _mgn_scales(p, frame.g), notes,
                           reduction=int(data.binning))
    return frame, result, layers


def _scaled_params(params, scale: float):
    """Pixel-sized settings follow the binning so the preview matches the full-size result."""
    from dataclasses import replace

    if params.mgn_scales:
        return replace(params, mgn_scales=tuple(s * scale for s in params.mgn_scales))
    return params


# -- Enhance settings files -------------------------------------------------------------------------

SETTINGS_KIND = "totality-enhance-settings"
SETTINGS_VERSION = 1
# Not saved: two internal widths in pixels, and the detail box, which belongs to the project. (The
# MGN sizes are saved in pixels, as the paper gives them; the ACHF sizes as fractions of the Sun.)
SETTINGS_SKIP = ("moon_margin_px", "ring_smooth")
_REMOVED = {"strands": "Smooth along scales (removed Nov 2026)", "mgn_skip_fine": "Skip finest scales (removed Nov 2026)"}


def enhance_settings(params) -> dict:
    """The Enhance settings as saved: every setting but those of one stack (``SETTINGS_SKIP``)."""
    out = {}
    for k, v in asdict(params).items():
        if k in SETTINGS_SKIP:
            continue
        out[k] = list(v) if isinstance(v, tuple) else v
    return out


def save_enhance_settings(params, path) -> Path:
    """Write the Enhance settings to ``path`` (JSON), to load again with :func:`load_enhance_settings`."""
    from . import __version__ as version

    path = Path(path)
    if path.suffix.lower() != ".json":
        path = path.with_suffix(".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {"kind": SETTINGS_KIND, "version": SETTINGS_VERSION, "toolkit": version,
           "saved": datetime.now().isoformat(timespec="seconds"), "settings": enhance_settings(params)}
    path.write_text(json.dumps(doc, indent=2, default=float), encoding="utf-8")
    return path


def load_enhance_settings(path, base=None):
    """Enhance settings from a file written by :func:`save_enhance_settings` or from the
    ``<name>_enhance.json`` written beside full-size layers. Settings missing from the file keep
    ``base``'s values (default: the defaults). Returns (EnhanceParams, notes)."""
    from dataclasses import fields, replace

    from .enhance import EnhanceParams, LAYERS, blend_mode_name

    path = Path(path)
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise WorkflowError(f"Could not read the settings file {path.name}: {exc}") from exc
    if isinstance(doc, dict) and doc.get("kind") == SETTINGS_KIND:
        found = doc.get("settings")
        if int(doc.get("version", 1)) > SETTINGS_VERSION:
            raise WorkflowError(f"{path.name} was written by a newer version of the toolkit; update it to load this file.")
    elif isinstance(doc, dict) and isinstance(doc.get("params"), dict):
        found = doc["params"]  # the info file written with the layers
    else:
        raise WorkflowError(f"{path.name} is not an Enhance settings file (nor the _enhance.json written with layers).")
    if not isinstance(found, dict):
        raise WorkflowError(f"{path.name} holds no settings.")
    base = base or EnhanceParams()
    names = {f.name for f in fields(EnhanceParams)} - set(SETTINGS_SKIP)
    notes, kw = [], {}
    ignored = [_REMOVED.get(k, k) for k in found if k not in names and k not in SETTINGS_SKIP]
    if ignored:
        notes.append("not used (no longer in the toolkit): " + ", ".join(ignored))
    missing = sorted(names - set(found))
    if missing:
        notes.append("not in the file, left as they were: " + ", ".join(missing))

    def bad(k, v):
        return WorkflowError(f"{path.name}: setting {k} = {v!r} is not valid")

    for k in names & set(found):
        v = found[k]
        try:
            if k == "layers":
                v = tuple(str(x) for x in v)
                if any(x not in LAYERS for x in v):
                    raise bad(k, v)
            elif k in ("mgn_scales", "mgn_weights"):
                v = tuple(float(x) for x in v)
                if len(v) != 6 or any(x < 0 for x in v) or (k == "mgn_scales" and min(v) <= 0):
                    raise bad(k, v)
            elif k in ("achf_scales", "achf_weights"):
                v = tuple(float(x) for x in v)
                if len(v) != 6 or any(x < 0 for x in v) or (k == "achf_scales" and min(v) <= 0):
                    raise bad(k, v)
            elif k in ("mean_attenuation", "std_attenuation"):
                v = None if v is None else tuple(float(x) for x in v)
            elif k == "blend":
                v = {str(a): float(b) for a, b in dict(v).items()}
                if any(a not in LAYERS for a in v) or any(b < 0 for b in v.values()):
                    raise bad(k, v)
            elif k == "blend_modes":
                v = {str(a): blend_mode_name(b) for a, b in dict(v).items()}
                if any(m not in ("soft light", "overlay", "linear") for m in v.values()):
                    raise bad(k, v)
            elif k == "white":
                v = str(v)
                if v not in ("corona", "camera", "none"):
                    raise bad(k, v)
            elif k == "neutral_background":
                v = bool(v)
            elif k in ("segments", "order"):
                v = int(v)
            else:
                v = float(v)
        except WorkflowError:
            raise
        except (TypeError, ValueError, AttributeError) as exc:
            raise bad(k, v) from exc
        kw[k] = v
    return replace(base, **kw), notes


# The detail box: at most this many full-size pixels a side, so it renders in seconds.
DETAIL_MAX_SIDE = 2000


def detail_box(box, shape, max_side: int = DETAIL_MAX_SIDE) -> tuple:
    """A box (y0, y1, x0, x1) in full-size pixels inside an image of ``shape``, at most
    ``max_side`` a side (shrunk about its centre)."""
    h, w = int(shape[0]), int(shape[1])
    y0, y1, x0, x1 = (int(round(v)) for v in box)
    y0, y1 = sorted((y0, y1))
    x0, x1 = sorted((x0, x1))
    out = []
    for a, b, n in ((y0, y1, h), (x0, x1, w)):
        if b - a > max_side:
            c = (a + b) // 2
            a, b = c - max_side // 2, c - max_side // 2 + max_side
        a, b = max(0, a), min(n, b)
        if b - a < 1:
            raise WorkflowError("The detail box is outside the picture.")
        out += [a, b]
    return tuple(out)


def detail_frame_key(project_path, params, hdr_file: Optional[str] = None) -> tuple:
    """What a prepared full-size stack depends on (see :func:`detail_frame`)."""
    path = Path(project_path).resolve()
    hdr = hdr_file or EventProject.load(path).parameters.get("stack", {}).get("hdr") or ""
    return (str(path), str(hdr), bool(params.neutral_background), str(params.white), float(params.moon_margin_px))


def detail_frame(project_path, params, hdr_file: Optional[str] = None, say: Optional[Say] = None, cancel=None):
    """The full-size stack prepared for rendering detail boxes (sky, white balance, noise and dust
    spots measured over the whole image, as the full-size write does)."""
    from .enhance import prepare

    say = reporter(say, cancel)
    data = load_hdr(Path(project_path).resolve(), hdr_file)
    say("Reading the full-size stack for the detail box (once; kept while the settings White balance "
        "and Keep the sky stay the same)…")
    frame = prepare(data.hdr, data.weight, data.header, params, progress=say)
    del data
    return frame


def enhance_detail(frame, params, box, say: Optional[Say] = None, cancel=None) -> tuple[tuple, dict]:
    """The layers of ``box`` at full resolution, exactly as the full-size write makes them there.
    Returns (the box used, {layer: (h, w, 3) 0..1})."""
    from .enhance import render, render_margin

    say = reporter(say, cancel)
    b = detail_box(box, frame.shape)
    say(f"detail box {b[3] - b[2]} x {b[1] - b[0]} px at x {b[2]}, y {b[0]} (full size), with "
        f"{render_margin(params, frame.g)} px round it for the filters")
    return b, render(frame, params, box=b, progress=say, cache=True)


def enhance_project(project_path, params, hdr_file: Optional[str] = None, output: Optional[str] = None,
                    name: Optional[str] = None, say: Optional[Say] = None, cancel=None, fmt: str = "tif"):
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
    result, _ = enhance(data.hdr, data.weight, data.header, params, out_dir=out_dir, name=name or path.stem, fmt=fmt,
                        progress=say)
    project = EventProject.load(path)
    project.parameters["enhance"] = {"hdr": data.path, **result.layers}
    project.save(path)
    return result, data.notes


# ---------------------------------------------------------------------------
# Earthshine
# ---------------------------------------------------------------------------


def _corona_white(path, project) -> Optional[list]:
    """The corona layers' white balance (from the Enhance step), so the earthshine's colour matches."""
    info_path = resolve(path, project.parameters.get("enhance", {}).get("info"))
    if info_path:
        try:
            white = json.loads(Path(info_path).read_text(encoding="utf-8")).get("white")
            if white and len(white) == 3:
                return [float(v) for v in white]
        except (OSError, ValueError):
            pass
    return None


def earthshine(project_path, params=None, include_rejected: bool = False, output: Optional[str] = None,
               name: Optional[str] = None, profile_dir: Optional[str] = None, lensfun_db: Optional[str] = None,
               say: Optional[Say] = None, cancel=None, fmt: str = "tif"):
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
    # Calibrate exactly as the HDR stack was, so the two stay in the same units.
    flat_mode = st.get("flat_mode")
    if flat_mode is None and (project.flat_files or project.dark_files or project.bias_files):
        say("note: the HDR stack was made without your calibration frames; run Stack again to use them "
            "(the earthshine stack follows the HDR stack)")
    try:
        plan, crop_factor, _ = _calibrated_plan(path, project, lights, lensfun_db, profile_dir, flat_mode, say)
    except ValueError as exc:
        raise WorkflowError(str(exc)) from exc
    calibs = {fc.frame_index: fc for fc in plan.frames}
    out_dir = Path(output) if output else path.parent
    try:
        result, _lum, _detail = stack_earthshine(use, lambda f: read_mosaic(frames[f.index].path), calibs, matches,
                                                 moon_xy, moon_r, canvas, p, out_dir=out_dir,
                                                 name=name or path.stem, image_crop=crop_factor, progress=say,
                                                 fmt=fmt, white=_corona_white(path, project))
    except (RuntimeError, ValueError) as exc:
        raise WorkflowError(str(exc)) from exc
    project.parameters["earthshine"] = {**result.outputs, "frames": result.frames}
    project.save(path)
    return result


from .imageout import image_size, read_image, with_format, write_image  # noqa: E402


def enhance_params_from_project(path, project):
    """The settings of the last full-size Enhance run (its json), or the defaults."""
    from dataclasses import fields

    from .enhance import EnhanceParams

    info_path = resolve(path, project.parameters.get("enhance", {}).get("info"))
    kw = {}
    if info_path:
        try:
            saved = json.loads(Path(info_path).read_text(encoding="utf-8")).get("params", {})
            names = {f.name for f in fields(EnhanceParams)}
            kw = {k: (tuple(v) if k == "layers" else v) for k, v in saved.items() if k in names}
        except (OSError, ValueError):
            kw = {}
    return EnhanceParams(**kw)


def final_with_moon(project_path, base_layer: str = "blend", strength: float = 1.0,
                    preview_side: Optional[int] = None, name: Optional[str] = None, say: Optional[Say] = None,
                    cancel=None, output_file: Optional[str] = None, enhance_params=None):
    """The final picture with the Moon in it, drawn in one go with the corona.

    The Moon-aligned earthshine stack (linear, same units) replaces the inside of the lunar disc
    in the HDR stack, and the Enhance step's settings stretch Moon and corona together, so the
    limb is the stack's own: same curve, same colour, no seam. The seas and highlands (glare
    removed, from the earthshine step) are then added where the earthshine dominates, ``strength``
    times their brightness. ``base_layer``: blend or natural.
    With ``preview_side`` returns a reduced image (no file); otherwise writes ``output_file``
    (default ``<name>_final.tif``) and returns its path.
    """
    from dataclasses import replace

    from scipy import ndimage

    from .earthshine import merge_moon
    from .enhance import enhance
    from .fits import read_fits

    say = reporter(say, cancel)
    path = Path(project_path).resolve()
    project = EventProject.load(path)
    es = project.parameters.get("earthshine", {})
    info_path, rgb_path = resolve(path, es.get("info")), resolve(path, es.get("rgb"))
    wpath, mpath = resolve(path, es.get("weight")), resolve(path, es.get("moon_surface"))
    if not (info_path and rgb_path and wpath and mpath):
        raise WorkflowError("Stack the earthshine again first: this needs the colour stack and the lunar surface "
                            "it now writes (_earthshine_rgb.fit, _moon_surface.fit).")
    info = json.loads(Path(info_path).read_text(encoding="utf-8"))
    say(f"final image: the Moon put into the stack, stretched with the corona ({base_layer}); "
        f"surface from {Path(mpath).name}")
    params = enhance_params_from_project(path, project) if enhance_params is None else enhance_params
    if base_layer == "natural":
        params = replace(params, layers=("natural",), blend={})
    else:
        params = replace(params, layers=())
    say("reading the HDR stack")
    data = load_hdr(path)
    es_rgb, es_head = read_fits(rgb_path)
    es_w, _ = read_fits(wpath)
    moon_data, _ = read_fits(mpath)
    X0, Y0 = int(info["crop"][0]), int(info["crop"][1])
    cx0, cy0 = int(info["canvas"][0]), int(info["canvas"][1])
    ox, oy = X0 - cx0, Y0 - cy0
    centre = (float(es_head["MOONX"]), float(es_head["MOONY"]))
    es_w = np.asarray(es_w, np.float32)
    b = 1
    if preview_side:
        data = reduce_for_preview(data, max_side=preview_side)
        b = data.binning
        params = _scaled_params(params, 1.0 / b)
    if b > 1:
        def binned(a):
            a = np.asarray(a, np.float32)
            hh, ww = a.shape[-2] // b, a.shape[-1] // b
            return a[..., : hh * b, : ww * b].reshape(a.shape[:-2] + (hh, b, ww, b)).mean(axis=(-3, -1))

        # Bin the crop on the same grid as the HDR stack: start it on a multiple of b.
        sx0, sy0 = (-ox) % b, (-oy) % b
        es_rgb, es_w, moon_data = es_rgb[:, sy0:, sx0:], es_w[sy0:, sx0:], moon_data[:, sy0:, sx0:]
        es_w = np.where(binned(es_w > 0) > 0.5, binned(es_w), 0.0).astype(np.float32)
        es_rgb, moon_data = binned(es_rgb), binned(moon_data)
        ox, oy = (ox + sx0) // b, (oy + sy0) // b
        centre = ((centre[0] - sx0 + 0.5) / b - 0.5, (centre[1] - sy0 + 0.5) / b - 0.5)
    moon, shift = merge_moon(data.hdr, data.weight, data.header, es_rgb, es_w, (ox, oy), centre,
                             params.moon_margin_px, say)
    surface = np.moveaxis(np.asarray(moon_data[:3], np.float32), 0, -1) * float(strength)
    vis = np.asarray(moon_data[3], np.float32)
    if abs(shift[0]) + abs(shift[1]) > 1e-3:
        surface = np.stack([ndimage.shift(surface[..., c], shift[::-1], order=1) for c in range(3)], axis=-1)
        vis = ndimage.shift(vis, shift[::-1], order=1)
    moon.surface, moon.visibility = surface.astype(np.float32), vis.astype(np.float32)
    say("stretching Moon and corona together")
    _, colour = enhance(data.hdr, data.weight, data.header, params, out_dir=None, progress=say, moon=moon,
                        keep={base_layer})
    out = colour[base_layer]
    if preview_side:
        return out
    say("writing the final image")
    final = Path(output_file) if output_file else path.parent / f"{name or path.stem}_final.tif"
    final = write_image(final, out)
    project.parameters["final"] = {"file": str(final), "base": base_layer, "strength": strength,
                                   "mode": "moon in the stack"}
    project.save(path)
    return str(final)


def final_image(project_path, base_layer: str = "blend", strength: float = 1.0, mode: str = "screen",
                preview_side: Optional[int] = None, name: Optional[str] = None,
                say: Optional[Say] = None, cancel=None, output_file: Optional[str] = None,
                base_file: Optional[str] = None):
    """The chosen enhanced layer with the earthshine blended into the lunar disc.

    ``base_file`` uses any image file as the corona instead of an enhanced layer (it must be the
    size of the HDR stack, for example a blend written earlier or one finished elsewhere).
    With ``preview_side`` returns a reduced image (no file); otherwise writes
    ``output_file`` (default ``<name>_final.tif`` next to the project) and returns its path.
    """
    from .earthshine import composite

    say = reporter(say, cancel)
    path = Path(project_path).resolve()
    project = EventProject.load(path)
    if base_file:
        base_path = resolve(path, base_file)
        if not base_path:
            raise WorkflowError(f"Image not found: {base_file}")
    else:
        base_path = resolve(path, project.parameters.get("enhance", {}).get(base_layer))
        if not base_path:
            raise WorkflowError(f"No '{base_layer}' layer yet: write the full-size layers on the Enhance tab first.")
    earth_path = resolve(path, project.parameters.get("earthshine", {}).get("earthshine"))
    if not earth_path:
        raise WorkflowError("No earthshine layer yet: click 'Stack earthshine' first.")
    step = 1
    if preview_side:
        w, h = image_size(base_path)
        step = max(1, int(math.ceil(max(h, w) / preview_side)))
    say(f"final image, older way: pasting {Path(earth_path).name} into {Path(base_path).name}")
    base = read_image(base_path, step)
    earth = read_image(earth_path, step)
    if base.shape != earth.shape:
        raise WorkflowError(f"{Path(base_path).name} is {base.shape[1] * step} x {base.shape[0] * step} px but the "
                            f"earthshine layer is {earth.shape[1] * step} x {earth.shape[0] * step} px. Use an image "
                            "the size of the HDR stack (not cropped or resized), or re-run the earthshine step after "
                            "the latest stack.")
    moon, limb_px, reach = None, 1.5, 6.0
    info_path = resolve(path, project.parameters.get("earthshine", {}).get("info"))
    if info_path:
        try:
            info = json.loads(Path(info_path).read_text(encoding="utf-8"))
            if info.get("moon"):
                mx, my, mr = info["moon"]
                moon = ((mx + 0.5) / step - 0.5, (my + 0.5) / step - 0.5, mr / step)
                limb_px = float(info.get("params", {}).get("limb_px", 1.5)) / step
                reach = float(info.get("params", {}).get("limb_reach", 6.0)) / step
        except (OSError, ValueError):
            moon = None
    out = composite(base, earth, strength, mode, moon=moon, limb_px=limb_px, reach=reach)
    if preview_side:
        return out
    say("writing the final image")
    final = Path(output_file) if output_file else path.parent / f"{name or path.stem}_final.tif"
    final = write_image(final, out)
    project.parameters["final"] = {"file": str(final), "base": base_layer, "base_file": base_file or "",
                                   "strength": strength, "mode": mode}
    project.save(path)
    return str(final)


# ---------------------------------------------------------------------------
# Prominences
# ---------------------------------------------------------------------------


def load_sun_region(project_path):
    """The HDR stack around the Sun (where prominences are), for quick, repeated runs."""
    from .prominences import sun_region

    data = load_hdr(project_path)
    region = sun_region(data.hdr, data.weight, data.header)
    del data
    return region


def prominence_layer(project_path, params=None, output_file: Optional[str] = None, preview: bool = False,
                     region=None, say: Optional[Say] = None, cancel=None):
    """Find the prominences and write a layer the size of the HDR stack, transparent outside them
    (TIFF and PNG; a JPG is on black). With ``preview`` returns (result, layer, alpha) over the
    region around the Sun and writes nothing. Otherwise returns the result."""
    from dataclasses import asdict

    from .prominences import ProminenceParams, find_prominences, place

    say = reporter(say, cancel)
    path = Path(project_path).resolve()
    p = params or ProminenceParams()
    if region is None:
        say("reading the HDR stack")
        region = load_sun_region(path)
    say("finding prominences")
    result, layer, alpha = find_prominences(region, p)
    if preview:
        return result, layer, alpha
    full = place(region, np.concatenate([layer, alpha[..., None]], axis=2))
    out = Path(output_file) if output_file else path.parent / f"{path.stem}_prominences.tif"
    say("writing the layer")
    out = write_image(out, full)
    result.outputs = {"prominences": str(out)}
    project = EventProject.load(path)
    project.parameters["prominences"] = {"layer": str(out), "params": asdict(p)}
    project.save(path)
    return result


def _base_image(path: Path, project: EventProject, base: str, base_file: Optional[str]) -> str:
    if base_file:
        found = resolve(path, base_file)
        if not found:
            raise WorkflowError(f"Image not found: {base_file}")
        return found
    if base == "final":
        found = resolve(path, project.parameters.get("final", {}).get("file"))
        if not found:
            raise WorkflowError("No final image yet: write one on the Earthshine tab first (or pick another base).")
        return found
    found = resolve(path, project.parameters.get("enhance", {}).get(base))
    if not found:
        raise WorkflowError(f"No '{base}' layer yet: write the full-size layers on the Enhance tab first.")
    return found


def final_with_prominences(project_path, base: str = "final", strength: float = 1.0, mode: str = "normal",
                           preview_side: Optional[int] = None, output_file: Optional[str] = None,
                           base_file: Optional[str] = None, say: Optional[Say] = None, cancel=None):
    """The prominence layer on top of an image: by default the final image from the Earthshine tab
    ("final"), or the last-written blend ("blend"), the natural layer ("natural"), or ``base_file``.
    With ``preview_side`` returns a reduced image; otherwise writes ``output_file`` and returns its path."""
    from .prominences import composite_top

    say = reporter(say, cancel)
    path = Path(project_path).resolve()
    project = EventProject.load(path)
    base_path = _base_image(path, project, base, base_file)
    layer_path = resolve(path, project.parameters.get("prominences", {}).get("layer"))
    if not layer_path:
        raise WorkflowError("No prominence layer yet: click 'Write prominence layer' first.")
    step = 1
    if preview_side:
        w, h = image_size(base_path)
        step = max(1, int(math.ceil(max(h, w) / preview_side)))
    say("reading the images")
    img = read_image(base_path, step)
    lay = read_image(layer_path, step, alpha=True)
    if img.shape[:2] != lay.shape[:2]:
        raise WorkflowError(f"{Path(base_path).name} is not the size of the HDR stack, so the prominences would "
                            "not line up. Use an image that is not cropped or resized.")
    rgb, alpha = lay[..., :3], lay[..., 3]
    if Path(layer_path).suffix.lower() in (".jpg", ".jpeg"):
        alpha = np.clip(rgb.max(axis=2) * 20.0, 0.0, 1.0)  # no transparency in a JPG: black is empty
    out = composite_top(img, rgb, alpha, strength, mode)
    if preview_side:
        return out
    say("writing the image")
    final = Path(output_file) if output_file else path.parent / f"{path.stem}_final_prominences.tif"
    final = write_image(final, out)
    project.parameters["final_prominences"] = {"file": str(final), "base": base, "base_file": base_file or "",
                                               "strength": strength, "mode": mode}
    project.save(path)
    return str(final)
