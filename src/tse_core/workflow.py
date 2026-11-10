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


def data_dirs(project_path) -> tuple[Path, Path]:
    """Where the steps put their data files: (.fit files, .json files) in their own folders inside the
    project folder, ``fits`` and ``json``. The project file and the pictures stay beside them."""
    base = Path(project_path).resolve().parent
    return base / "fits", base / "json"


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
    for cand in ((p,) if p.is_absolute() else ()) + (base / p, base / p.name, base / "fits" / p.name,
                                                       base / "json" / p.name, p):
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


VIGNETTING_CHOICES = ("auto", "master", "profile", "compare", "lensfun", "none")


@dataclass
class Calibration:
    """How the light frames are calibrated while stacking (the Stack tab's Calibration box).

    ``vignetting``: auto (master flat if flats are chosen, else the lensfun profile, else none),
    master (flats from the same session: vignetting and dust), profile (a smooth surface fitted to
    the flats, for flats from another time: vignetting only), compare (master, and a report against
    that surface), lensfun (the lens's lensfun profile, ``lens`` or matched from the files), none.
    ``darks``/``bias``: "use" the frames chosen on the Frames tab, or "none" (then the file's black
    level). Hot pixels are fixed from the frames themselves only when darks are not used."""
    vignetting: str = "auto"
    lens: str = ""
    uneven_light: bool = False
    darks: str = "none"
    bias: str = "use"

    def record(self) -> dict:
        return {"calibration": {"vignetting": self.vignetting, "lens": self.lens, "uneven_light": self.uneven_light,
                                "darks": self.darks, "bias": self.bias}}


def lens_choices(project_path, lensfun_db=None, limit: int = 8) -> list[str]:
    """lensfun lenses that match the light frames' lens (for the Stack tab's lens list)."""
    project = EventProject.load(Path(project_path))
    sample = next((f for f in project.inventory.frames if f.lens_model or f.lens_spec), None)
    db = _db(lensfun_db)
    if sample is None or db.empty:
        return []
    maker = sample.lens_make or sample.camera_make
    return [m.lens.label() for m in db.match_lens(sample.lens_model, sample.lens_spec, sample.focal_length_mm,
                                                  maker, limit=limit)]


def calibration_of(stack_params: dict, project: EventProject) -> Optional[Calibration]:
    """The calibration a stack was made with (so the earthshine and prominence stacks repeat it);
    None for a stack made before calibration frames were used."""
    st = stack_params or {}
    c = st.get("calibration")
    if isinstance(c, dict):
        return Calibration(c.get("vignetting", "auto"), c.get("lens", ""), bool(c.get("uneven_light", False)),
                           c.get("darks", "none"), c.get("bias", "use"))
    mode = st.get("flat_mode")
    if mode is None:
        return None
    vign = mode if project.flat_files and mode in VIGNETTING_CHOICES else ("lensfun" if project.use_lensfun else "none")
    return Calibration(vign, project.lens_profile or "", bool(st.get("flats_uneven_light", False)), "use", "use")


def _plan(project: EventProject, lights, lensfun_db, profile_dir, calib=None, vignetting: str = "lensfun",
          lens: str = ""):
    from .calibration import plan_calibration

    db = _db(lensfun_db)
    lensfun = vignetting == "lensfun"
    # Own-flats profiles saved earlier count as a lens profile; flats are used only when chosen.
    profiles = load_profile_library(profile_dir) if lensfun else []
    plan = plan_calibration(lights, db, calib, lens_name=lens or None, use_lensfun=lensfun, profiles=profiles)
    cam = db.find_camera(lights[0].camera_make, lights[0].camera_model) if lights and not db.empty else None
    return plan, (cam.crop_factor if cam else 1.0)


def _crop_factor(lights, lensfun_db) -> float:
    db = _db(lensfun_db)
    cam = db.find_camera(lights[0].camera_make, lights[0].camera_model) if lights and not db.empty else None
    return cam.crop_factor if cam else 1.0


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
    from .frames import fill_optics
    if fill_optics(groups["flats"], project.focal_length_mm, project.aperture_f):
        say("flats: the focal length and f-number typed on the Frames tab are used where the files do not say")
    return CalibrationFrames(**groups)


def _calibrated_plan(project_path, project: EventProject, lights, lensfun_db, profile_dir,
                     cal: Optional[Calibration], say, out_dir=None, name="stack"):
    """The calibration plan for Stack and the earthshine stack, with the master frames attached.

    ``cal`` None means the calibration frames are not used (a stack made before they were), so that
    the earthshine stack stays in the same units as that HDR stack. Returns (plan, crop factor,
    what was done, for the project file)."""
    from .masters import attach_masters, describe_frames, flat_mismatch
    from . import rawio

    if cal is None:
        plan, crop = _plan(project, lights, lensfun_db, profile_dir,
                           vignetting="lensfun" if project.use_lensfun else "none", lens=project.lens_profile)
        info = {}
    else:
        if cal.vignetting not in VIGNETTING_CHOICES:
            raise ValueError(f"vignetting must be one of {', '.join(VIGNETTING_CHOICES)}")
        calib = _calibration_frames(project_path, project, say)
        vign = cal.vignetting
        if vign == "auto":
            vign = "master" if calib.flats else "lensfun"
        uses_flats = vign in ("master", "profile", "compare")
        unused = []
        if not uses_flats and calib.flats:
            unused.append(f"{len(calib.flats)} flat(s)")
            calib.flats = []
        if cal.darks != "use" and calib.darks:
            unused.append(f"{len(calib.darks)} dark(s)")
            calib.darks = []
        if cal.bias != "use" and calib.bias:
            unused.append(f"{len(calib.bias)} bias frame(s)")
            calib.bias = []
        say("calibration: vignetting by " + {"master": "master flat", "profile": "smooth profile from the flats",
                                             "compare": "master flat (compared with the smooth profile)",
                                             "lensfun": "lensfun profile", "none": "nothing (none chosen)"}[vign]
            + ("" if cal.vignetting != "auto" else " (automatic)")
            + f"; darks {'used' if calib.darks else 'not used'}; "
            + ("bias frames used" if calib.bias else "black level of each file for the offset")
            + (f"; not used: {', '.join(unused)}" if unused else ""))
        plan, crop = _plan(project, lights, lensfun_db, profile_dir, calib,
                           vignetting="lensfun" if vign == "lensfun" else "none", lens=cal.lens)
        if uses_flats:
            why = flat_mismatch(lights, calib)
            if why:
                say("warning: " + why)
        # Masters are saved in fits/masters in the project folder and reused while their frames are unchanged.
        info = attach_masters(plan, calib, lambda p: rawio.read_mosaic(p), lights,
                              vign if uses_flats else "master", crop, say,
                              out_dir=data_dirs(project_path)[0], name=name,
                              cache_dir=data_dirs(project_path)[0] / "masters", uneven_light=cal.uneven_light)
        info.pop("flat_mode", None)
        info.pop("flats_uneven_light", None)
        info.update(cal.record())
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
    # No vignetting correction: the lens's darkening is fixed to the sensor, the same in every frame,
    # smooth, and smallest round the Sun, and phase correlation compares the corona's fine structure
    # with the smooth part filtered out. Only the camera's crop factor is needed.
    plan, crop = None, _crop_factor(lights, opts.lensfun_db)
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
    out = Path(opts.output) if opts.output else data_dirs(path)[1] / (path.stem + "_align.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    result.save(out)
    project.parameters["align"] = {"file": str(out), "rho": opts.rho, "rotation": opts.rotation,
                                   "window": opts.window, "clock_error": opts.clock_error_s}
    project.save(path)
    return result, out


def load_alignment(project_path, align_file: Optional[str] = None) -> tuple[dict, Path]:
    path = Path(project_path).resolve()
    project = EventProject.load(path)
    align_path = Path(resolve(path, align_file or project.parameters.get("align", {}).get("file"))
                      or resolve(path, path.stem + "_align.json") or data_dirs(path)[1] / (path.stem + "_align.json"))
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
    hot_pixels: bool = True  # fixed from the frames themselves; only when darks are not used
    profile_dir: Optional[str] = None
    lensfun_db: Optional[str] = None
    vignetting: str = "auto"  # see Calibration
    lens: str = ""
    flats_uneven_light: bool = False  # the light on the flats was brighter on one side: take that slope out
    darks: str = "none"
    bias: str = "use"

    def calibration(self) -> Calibration:
        return Calibration(self.vignetting, self.lens, self.flats_uneven_light, self.darks, self.bias)


def _frame_infos(project: EventProject, alignment: dict, include_rejected: bool):
    """Light frames with their alignment, Moon and nominal brightness, for stacking.

    Returns (frames by index, reference index, [FrameInfo], [skipped messages]).
    """
    from .frames import true_exposure
    from .stack import FrameInfo

    frames = {f.index: f for f in project.inventory.lights()}
    ref = alignment["reference_index"]
    rejected = set()
    order = [fr["index"] for fr in alignment["frames"]]
    if alignment.get("sun") and not include_rejected:
        rejected = {order[k] for k in alignment["sun"].get("rejected", [])}
    resid = (alignment.get("sun") or {}).get("residuals_px") or []
    track_off = {order[k]: float(v) for k, v in enumerate(resid) if k < len(order)}

    def _ev(fm):  # light gathered, using the shutter's real time behind its rounded label
        if not (fm.exposure_s and fm.iso):
            return None
        # No f-number recorded (a telescope): the same for every frame (see FrameMeta.exposure_value).
        return math.log2(true_exposure(fm.exposure_s) * fm.iso / 100.0 / (fm.aperture_f or 1.0) ** 2)

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
        infos.append(FrameInfo(i, fr["file_name"], ratio, fr["matrix"], mx, my, fr["moon_frame"]["radius"],
                               track_off.get(i)))
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
                                                         opts.calibration(), say, out_dir=out_dir, name=name)
    except ValueError as exc:
        raise WorkflowError(str(exc)) from exc
    calibs = {fc.frame_index: fc for fc in plan.frames}
    # What the reference frame is divided by, small enough to send or open quickly in Siril.
    from .fits import write_fits
    from .masters import correction_preview

    small, text = correction_preview(calibs.get(ref), tuple(alignment["image_size"]), crop_factor)
    say(text)
    if small is not None:
        fits_dir = data_dirs(path)[0]
        fits_dir.mkdir(parents=True, exist_ok=True)
        vpath = fits_dir / f"{name}_vignetting.fit"
        write_fits(vpath, small, {"COMMENT": "Correction applied by Stack (frames divided by this), 1/8 size"})
        say(f"  saved as {vpath.name} (1/8 size; 1.0 = the middle)")
        calib_info = {**calib_info, "vignetting": str(vpath)}
    hot = opts.hot_pixels and opts.darks != "use"
    if opts.hot_pixels and not hot:
        say("hot pixels: corrected by the darks (the separate hot-pixel fix is off while darks are used)")
    params = StackParams(crop_radii=opts.radii, moon_frame=opts.moon_frame, hot_pixels=hot)
    try:
        result, _hdr, _w = stack_frames(infos, lambda f: read_mosaic(frames[f.index].path), calibs,
                                        tuple(alignment["image_size"]), ref, alignment.get("sun"), params,
                                        out_dir=out_dir, name=name, image_crop=crop_factor, progress=say,
                                        data_dirs=data_dirs(path))
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
            if mode not in ("soft light", "overlay", "linear", "average"):
                raise WorkflowError(f"unknown blend mode {mode!r}; use soft light, overlay, linear or average")
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
                if any(m not in ("soft light", "overlay", "linear", "average") for m in v.values()):
                    raise bad(k, v)
            elif k == "white":
                v = str(v)
                if v not in ("corona", "camera", "none"):
                    raise bad(k, v)
            elif k == "natural_stretch":
                v = str(v)
                if v not in ("log", "gamma"):
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
                        progress=say, json_dir=data_dirs(path)[1])
    project = EventProject.load(path)
    project.parameters["enhance"] = {"hdr": data.path, **result.layers}
    project.save(path)
    return result, data.notes


# ---------------------------------------------------------------------------
# Earthshine
# ---------------------------------------------------------------------------


def _corona_white(path, project) -> tuple:
    """The corona layers' white balance, so the earthshine's colour matches: (white, where from).

    From the Enhance step's written layers when there are any (their settings), otherwise measured
    on the HDR stack itself the way the Enhance step does it (inner corona white), so the Moon is
    never left in the camera's raw colour. (None, why) if neither can be had."""
    from .enhance import prepare

    info_path = resolve(path, project.parameters.get("enhance", {}).get("info"))
    if info_path:
        try:
            white = json.loads(Path(info_path).read_text(encoding="utf-8")).get("white")
            if white and len(white) == 3:
                return [float(v) for v in white], "from the Enhance step's written layers"
        except (OSError, ValueError):
            pass
    try:
        data = reduce_for_preview(load_hdr(path), max_side=1400)
    except (WorkflowError, OSError, ValueError) as exc:
        return None, f"no white balance ({exc})"
    params = _scaled_params(enhance_params_from_project(path, project), 1.0 / data.binning)
    frame = prepare(data.hdr, data.weight, data.header, params)
    return [float(v) for v in frame.wb], "measured on the HDR stack (Enhance layers not written yet)"


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
    cal = calibration_of(st, project)
    if cal is not None and cal.darks == "use" and p.hot_pixels:
        from dataclasses import replace as _replace

        p = _replace(p, hot_pixels=False)
        say("hot pixels: corrected by the darks, as in the HDR stack")
    if cal is None and (project.flat_files or project.dark_files or project.bias_files):
        say("note: the HDR stack was made without your calibration frames; run Stack again to use them "
            "(the earthshine stack follows the HDR stack)")
    try:
        plan, crop_factor, _ = _calibrated_plan(path, project, lights, lensfun_db, profile_dir, cal, say)
    except ValueError as exc:
        raise WorkflowError(str(exc)) from exc
    calibs = {fc.frame_index: fc for fc in plan.frames}
    out_dir = Path(output) if output else path.parent
    white, where = _corona_white(path, project)
    say("white balance for the earthshine's colour (R, G, B): "
        + (", ".join(f"{v:.3f}" for v in white) + f", {where}" if white is not None
           else f"none, {where}: the colour is the camera's raw colour"))
    try:
        result, _lum, _detail = stack_earthshine(use, lambda f: read_mosaic(frames[f.index].path), calibs, matches,
                                                 moon_xy, moon_r, canvas, p, out_dir=out_dir,
                                                 name=name or path.stem, image_crop=crop_factor, progress=say,
                                                 fmt=fmt, white=white, data_dirs=data_dirs(path))
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
                    cancel=None, output_file: Optional[str] = None, enhance_params=None, moon_only: bool = False,
                    box: Optional[tuple] = None, prominences: Optional[bool] = None, prominence_params=None,
                    join: Optional[str] = None):
    """The final picture with the Moon in it, drawn in one go with the corona.

    ``join``: how Moon and corona meet (see :func:`earthshine.merge_moon`): "corona first" (the
    HDR stack wherever the Blend has it, then the earthshine stack, else black) or "rebuilt" (the
    band by the limb rebuilt from both); None: as last saved in the project (default corona first).

    With ``moon_only`` returns (picture, Moon mask) without writing, for writing the Moon as drawn
    on its own (see :func:`moon_layer`).

    The Moon-aligned earthshine stack (linear, same units) replaces the inside of the lunar disc
    in the HDR stack, and the Enhance step's settings stretch Moon and corona together, so the
    limb is the stack's own: same curve, same colour, no seam. The seas and highlands (glare
    removed, from the earthshine step) are then added where the earthshine dominates, ``strength``
    times their brightness. ``base_layer``: blend or natural.
    With ``preview_side`` returns a reduced image (no file); with ``box`` = (y0, y1, x0, x1) returns
    that part at full size, exactly as the written image has it (no file); otherwise writes
    ``output_file`` (default ``<name>_final.tif``) and returns its path. ``prominences``: use the
    prominence stack (None: as the project says), with ``prominence_params`` (None: the project's).
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
    if prominences is not None:
        project.parameters.setdefault("prominence_stack", {})["use"] = bool(prominences)
    sharpen_project_prominences(path, project, data, say, prominence_params)
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
    no_data = ~(np.asarray(data.weight) > 0) if moon_only else None  # black in the corona layers
    if join is None:
        join = str(project.parameters.get("final", {}).get("join", "corona first"))
    moon, shift = merge_moon(data.hdr, data.weight, data.header, es_rgb, es_w, (ox, oy), centre,
                             params.moon_margin_px, say, join=join)
    surface = np.moveaxis(np.asarray(moon_data[:3], np.float32), 0, -1) * float(strength)
    vis = np.asarray(moon_data[3], np.float32)
    if abs(shift[0]) + abs(shift[1]) > 1e-3:
        surface = np.stack([ndimage.shift(surface[..., c], shift[::-1], order=1) for c in range(3)], axis=-1)
        vis = ndimage.shift(vis, shift[::-1], order=1)
    moon.surface, moon.visibility = surface.astype(np.float32), vis.astype(np.float32)
    say("stretching Moon and corona together")
    if box is not None:  # the whole-image measurements, then only this part drawn
        from .enhance import prepare, render

        frame = prepare(data.hdr, data.weight, data.header, params, moon=moon, progress=say)
        y0, y1, x0, x1 = box
        return render(frame, params, box=(max(0, y0), min(frame.shape[0], y1), max(0, x0), min(frame.shape[1], x1)),
                      keep={base_layer}, progress=say)[base_layer]
    _, colour = enhance(data.hdr, data.weight, data.header, params, out_dir=None, progress=say, moon=moon,
                        keep={base_layer})
    out = colour[base_layer]
    if moon_only:  # for pasting: the Moon as drawn here, cut where the corona layers' black disc ends
        hh, ww = out.shape[:2]
        yy, xx = np.mgrid[0:hh, 0:ww]
        mx, my, mr = (float(data.header[k]) for k in ("MOONX", "MOONY", "MOONR"))
        disc = np.hypot(xx - mx, yy - my) <= mr + params.moon_margin_px
        # Where the corona layers are black: the disc, and pixels the HDR stack had no data for
        # (saturated in every frame, by a bright prominence), which the Moon's stack fills here.
        return out, disc | no_data[:hh, :ww]
    if preview_side:
        return out
    say("writing the final image")
    final = Path(output_file) if output_file else path.parent / f"{name or path.stem}_final.tif"
    final = write_image(final, out)
    project.parameters["final"] = {"file": str(final), "base": base_layer, "strength": strength,
                                   "mode": "moon in the stack", "join": join}
    project.save(path)
    return str(final)


# Half-side of the Earthshine tab's preview box, as a fraction of the Moon's radius: the Moon, the
# limb all round and the inner corona with its prominences
MOON_CROP_HALF = 1.3


def moon_crop_box(project_path) -> tuple:
    """The box (y0, y1, x0, x1, full-size px) round the Moon that the Earthshine tab previews:
    MOON_CROP_HALF Moon radii each side of its centre, cut to the image."""
    from .fits import read_fits_header

    path = Path(project_path).resolve()
    project = EventProject.load(path)
    hdr_path = resolve(path, project.parameters.get("stack", {}).get("hdr"))
    if not hdr_path:
        raise WorkflowError("Stack first: the preview box is placed on the Moon in the HDR stack.")
    h = read_fits_header(hdr_path)
    mx, my, mr = float(h["MOONX"]), float(h["MOONY"]), float(h["MOONR"])
    w, hh = int(h["NAXIS1"]), int(h["NAXIS2"])
    half = int(math.ceil(mr * MOON_CROP_HALF))
    return (max(0, int(my) - half), min(hh, int(my) + half + 1), max(0, int(mx) - half), min(w, int(mx) + half + 1))


def final_moon_crop(project_path, base_layer: str = "blend", strength: float = 1.0, say: Optional[Say] = None,
                    cancel=None, join: Optional[str] = None):
    """The Earthshine tab's preview: the box round the Moon (:func:`moon_crop_box`) of the final
    image at full resolution, exactly as Write final image draws it there. Returns (picture, box)."""
    say = reporter(say, cancel)
    box = moon_crop_box(project_path)
    say(f"preview: {box[3] - box[2]} x {box[1] - box[0]} px round the Moon at full size (x {box[2]}, y {box[0]})")
    return final_with_moon(project_path, base_layer, strength, say=say, cancel=cancel, box=box, join=join), box


# Width of the Moon layer's soft edge outside its cover, as a fraction of the Moon's radius (5 px for a
# 400 mm lens on a 45-megapixel full-frame camera)
MOON_LAYER_FADE = 0.0115


def moon_layer(project_path, base_layer: str = "blend", strength: float = 1.0, name: Optional[str] = None,
               say: Optional[Say] = None, cancel=None, output_file: Optional[str] = None, enhance_params=None):
    """The Moon alone, as the final image draws it (``base_layer`` blend or natural, with the Enhance
    settings and ``strength`` for the seas and highlands), transparent everywhere else.

    Opaque where the corona layers are black (the disc, and pixels the HDR stack had no data for)
    and, when that layer as last written is there, wherever the final image differs from it by more
    than half an 8-bit level: the band just outside the limb, where the final image blends the Moon's
    stack into the corona. Laid over that layer at 100 % in Normal mode it gives the final image (a
    far-off difference under half a level, from the Moon changing the stack's statistics, is left).
    Writes ``output_file`` (TIFF or PNG with transparency; default ``<name>_moon.tif``)."""
    say = reporter(say, cancel)
    path = Path(project_path).resolve()
    img, mask = final_with_moon(path, base_layer, strength, say=say, enhance_params=enhance_params, moon_only=True)
    project = EventProject.load(path)
    written = resolve(path, project.parameters.get("enhance", {}).get(base_layer))
    band = 0
    if written and image_size(written) == (img.shape[1], img.shape[0]):
        base = read_image(written)
        differs = np.abs(img - base).max(axis=2) > 0.5 / 255   # half an 8-bit level
        band = int((differs & ~mask).sum())
        mask = mask | differs
        say(f"the Moon layer also covers {band} px round the limb where the final image differs visibly from "
            f"{Path(written).name}")
    else:
        say(f"no written {base_layer} layer the size of the stack: the Moon layer covers only the black disc, so "
            "the first pixels outside the limb will not match the final image exactly (Write full-size layers "
            "on the Enhance tab first)")
    # A soft edge grown outward over 1.15 % of the Moon's radius (5 px for a 400 mm lens on a
    # 45-megapixel full-frame camera), at least 2 px, so the layer's edge and the corona layer's
    # meet without a line. There the layer holds the final image, within half an 8-bit level of the
    # corona layer, so the fade adds no edge of its own.
    from scipy import ndimage
    from .fits import read_fits_header

    hdr_path = resolve(path, project.parameters.get("stack", {}).get("hdr"))
    moon_r = float(read_fits_header(hdr_path).get("MOONR", 0.0)) if hdr_path else 0.0
    width = max(2.0, MOON_LAYER_FADE * moon_r)
    alpha = np.clip(1.0 - ndimage.distance_transform_edt(~mask) / width, 0.0, 1.0).astype(np.float32)
    say(f"Moon layer edge: soft over {width:.1f} px outside")
    say("writing the Moon layer")
    rgba = np.concatenate([img * (alpha > 0)[..., None], alpha[..., None]], axis=2)
    out = Path(output_file) if output_file else path.parent / f"{name or path.stem}_moon.tif"
    out = write_image(out, rgba)
    project.parameters["moon_layer"] = {"file": str(out), "base": base_layer, "strength": strength}
    project.save(path)
    say(f"Moon layer written: {out.name} (the Moon as {base_layer.capitalize()} draws it, lunar surface "
        f"strength {strength:g}; over the {base_layer} layer at 100 % in Normal mode it gives the final "
        "image)")
    return str(out)


def _write_final(path, project, out, preview_side, output_file, name, say, record: dict):
    if preview_side:
        return out
    say("writing the final image")
    final = Path(output_file) if output_file else path.parent / f"{name or path.stem}_final.tif"
    final = write_image(final, out)
    project.parameters["final"] = {"file": str(final), **record}
    project.save(path)
    return str(final)


# ---------------------------------------------------------------------------
# Prominences
# ---------------------------------------------------------------------------


@dataclass
class ProminenceStackOptions:
    longest_stops: float = 2.0  # frames up to this many stops longer than the shortest
    radii: float = 1.6  # the square stacked, in solar radii from the Sun's centre


def prominence_stack(project_path, opts: Optional[ProminenceStackOptions] = None, say: Optional[Say] = None,
                     cancel=None) -> dict:
    """Stack the shortest aligned exposures around the Sun for the prominences.

    The prominences are sharpest in the shortest frames: in the HDR stack they come mostly from
    the longest frames not saturated there, where they are swollen and soft (2024: edge widths
    2.2-2.6 px in the 1/1600-1/1000 s frames, 3.25 px in the HDR stack). The frames are stacked
    with the HDR stack's own brightness gains, so the result is in its units, each frame's Moon
    left out as usual. On the way, each frame's blur is measured on its own Moon's edge; the
    median is saved for the sharpening. Returns the project's ``prominence_stack`` entry."""
    from .prominences import limb_blur
    from .rawio import read_mosaic
    from .stack import Match, StackParams, stack_frames

    say = reporter(say, cancel)
    o = opts or ProminenceStackOptions()
    path = Path(project_path).resolve()
    project = EventProject.load(path)
    st = project.parameters.get("stack", {})
    info_path = resolve(path, st.get("info"))
    if not info_path:
        raise WorkflowError("Stack the frames first (Stack tab): the prominence stack uses its brightness matching.")
    info = json.loads(Path(info_path).read_text(encoding="utf-8"))
    gains = {}
    for m in info.get("matches", []):
        m = dict(m)
        if m.get("offsets") is not None:
            m["offsets"] = tuple(m["offsets"])
        gains[int(m["index"])] = Match(**m)
    alignment, _ = load_alignment(path, st.get("align"))
    frames, ref, infos, skipped = _frame_infos(project, alignment, False)
    infos = [f for f in infos if f.index in gains and frames[f.index].exposure_s]
    if not infos:
        raise WorkflowError("No aligned frames with an exposure time to stack.")
    t0 = min(frames[f.index].exposure_s for f in infos)
    tmax = t0 * 2.0 ** o.longest_stops * 1.03  # a little over, for rounded shutter speeds
    infos = sorted((f for f in infos if frames[f.index].exposure_s <= tmax), key=lambda f: frames[f.index].exposure_s)
    say(f"prominence stack: {len(infos)} frame(s), {frames[infos[0].index].exposure_text} to "
        f"{frames[infos[-1].index].exposure_text} s (up to {o.longest_stops:g} stops longer than the shortest)")
    lights = [frames[f.index] for f in infos]
    _locate_frames(path, lights)
    try:
        plan, crop_factor, _ = _calibrated_plan(path, project, lights, None, None, calibration_of(st, project),
                                                say, out_dir=path.parent, name=path.stem)
    except ValueError as exc:
        raise WorkflowError(str(exc)) from exc
    calibs = {fc.frame_index: fc for fc in plan.frames}
    blurs = []

    def measure(f, radiance, crop):
        got = limb_blur(radiance, f.moon_x - crop[0], f.moon_y - crop[1], f.moon_r)
        if got is not None:
            blurs.append(got[0])
            say(f"  {f.file_name}: blur at the Moon's edge {got[0]:.2f} px")

    by_t = sorted(infos, key=lambda f: frames[f.index].exposure_s)
    moon_frame = int(st.get("moon_frame")) if st.get("moon_frame") in [f.index for f in infos] else by_t[0].index
    params = StackParams(crop_radii=o.radii, moon_frame=moon_frame)
    fits_dir, json_dir = data_dirs(path)
    name = f"{path.stem}_prominence"
    try:
        result, _h, _w = stack_frames(infos, lambda f: read_mosaic(frames[f.index].path), calibs,
                                      tuple(alignment["image_size"]), ref, alignment.get("sun"), params,
                                      out_dir=path.parent, name=name, image_crop=crop_factor, progress=say,
                                      data_dirs=(fits_dir, json_dir), matches=gains, on_frame=measure)
    except (RuntimeError, ValueError) as exc:
        raise WorkflowError(str(exc)) from exc
    sigma = float(np.median(blurs)) if blurs else None
    if sigma is not None:
        say(f"prominence stack: blur {sigma:.2f} px (median at the Moon's edge in {len(blurs)} frame(s))")
    else:
        say("prominence stack: the Moon's edge could not be measured; no sharpening")
    entry = {**result.outputs, "frames": [f.index for f in infos], "longest_stops": o.longest_stops,
             "radii": o.radii, "blur_px": sigma}
    project.parameters["prominence_stack"] = entry
    project.save(path)
    return entry


def prominence_params_from_project(project: EventProject):
    """The Prominences tab's settings saved in the project (defaults where none)."""
    from dataclasses import fields

    from .prominences import ProminenceParams

    saved = project.parameters.get("prominences", {}).get("params", {}) or {}
    names = {f.name for f in fields(ProminenceParams)}
    return ProminenceParams(**{k: v for k, v in saved.items() if k in names})


def sharpen_project_prominences(path: Path, project: EventProject, data, say: Say, params=None) -> Optional[dict]:
    """The HDR stack's prominences in ``data.hdr`` sharpened in place, as far as the prominence
    stack shows they are softer (see prominences.sharpen_prominences), if there is a prominence
    stack and it is switched on."""
    from .enhance import Geometry
    from .fits import read_fits
    from .prominences import sharpen_prominences

    ps = project.parameters.get("prominence_stack", {})
    phdr, pw = resolve(path, ps.get("hdr")), resolve(path, ps.get("weight"))
    if not (phdr and pw) or not ps.get("use", True):
        return None
    p = params or prominence_params_from_project(project)
    prom, head = read_fits(phdr)
    pweight, _ = read_fits(pw)
    prom = np.moveaxis(np.asarray(prom, np.float32), 0, -1)
    x0 = int(head.get("CROPX0", 0)) - int(data.header.get("CROPX0", 0))
    y0 = int(head.get("CROPY0", 0)) - int(data.header.get("CROPY0", 0))
    H, W = data.hdr.shape[:2]
    ys, xs = slice(max(y0, 0), min(y0 + prom.shape[0], H)), slice(max(x0, 0), min(x0 + prom.shape[1], W))
    if ys.stop <= ys.start or xs.stop <= xs.start:
        say("prominences: the prominence stack does not overlap the HDR stack; not used")
        return None
    py, px = slice(ys.start - y0, ys.stop - y0), slice(xs.start - x0, xs.stop - x0)
    h = data.header
    g = Geometry(float(h["SUNX"]) - xs.start, float(h["SUNY"]) - ys.start, float(h["SUNR"]),
                 float(h["MOONX"]) - xs.start, float(h["MOONY"]) - ys.start, float(h["MOONR"]))
    hdr_sub, w_sub = data.hdr[ys, xs], data.weight[ys, xs]
    return sharpen_prominences(hdr_sub, w_sub, prom[py, px], np.asarray(pweight, np.float32)[py, px], g, p,
                               ps.get("blur_px"), float(h.get("PIXSCALE") or 2.4), say)


def save_prominence_settings(project_path, params, use: Optional[bool] = None) -> None:
    """The Prominences tab's settings into the project, for Write final image (here and on the
    Earthshine tab) and the command line's ``final``."""
    from dataclasses import asdict

    path = Path(project_path).resolve()
    project = EventProject.load(path)
    entry = dict(project.parameters.get("prominences", {}))
    entry["params"] = asdict(params)
    project.parameters["prominences"] = entry
    if use is not None and "prominence_stack" in project.parameters:
        project.parameters["prominence_stack"]["use"] = bool(use)
    project.save(path)


def prominence_preview(project_path, params=None, say: Optional[Say] = None, cancel=None,
                       base: Optional[str] = None, strength: Optional[float] = None):
    """The final image round the Sun, full size, exactly as Write final image draws it (the
    Earthshine tab's base and lunar surface strength, the Enhance settings) with the prominences
    from the prominence stack and ``params``. Nothing is written. Returns (picture, Sun's centre
    and radius in it)."""
    from .fits import read_fits_header

    say = reporter(say, cancel)
    path = Path(project_path).resolve()
    project = EventProject.load(path)
    if not resolve(path, project.parameters.get("prominence_stack", {}).get("hdr")):
        raise WorkflowError("Stack the prominences first (Stack prominences).")
    h = read_fits_header(resolve(path, project.parameters["stack"]["hdr"]))
    sx, sy, sr = float(h["SUNX"]), float(h["SUNY"]), float(h["SUNR"])
    half = int(sr * 1.3)
    box = (int(sy) - half, int(sy) + half, int(sx) - half, int(sx) + half)
    fin = project.parameters.get("final", {})
    if base not in ("blend", "natural"):
        base = fin.get("base") if fin.get("base") in ("blend", "natural") else "blend"
    k = float(fin.get("strength", 1.0)) if strength is None else float(strength)
    es = project.parameters.get("earthshine", {})
    if all(resolve(path, es.get(x)) for x in ("info", "rgb", "weight", "moon_surface")):
        img = final_with_moon(path, base, k, say=say, box=box, prominences=True, prominence_params=params)
    else:  # no earthshine yet: the corona and prominences, as Enhance draws them (no Moon)
        from dataclasses import replace

        from .enhance import prepare, render

        say("no earthshine stack yet: the corona and prominences without the Moon")
        data = load_hdr(path)
        project.parameters.setdefault("prominence_stack", {})["use"] = True
        sharpen_project_prominences(path, project, data, say, params)
        ep = enhance_params_from_project(path, project)
        ep = replace(ep, layers=("natural",), blend={}) if base == "natural" else replace(ep, layers=())
        frame = prepare(data.hdr, data.weight, data.header, ep, progress=say)
        b = (max(0, box[0]), min(frame.shape[0], box[1]), max(0, box[2]), min(frame.shape[1], box[3]))
        img = render(frame, ep, box=b, keep={base}, progress=say)[base]
    return img, (sx - max(0, box[2]), sy - max(0, box[0]), sr)


def load_sun_region(project_path, say: Optional[Say] = None):
    """The stack around the Sun (where prominences are), for quick, repeated runs: the prominence
    stack (Stack prominences) if there is one, else the HDR stack. Either way the Moon is the HDR
    stack's, the one in the final image, which the prominences are cut at."""
    from dataclasses import replace

    from .fits import read_fits, read_fits_header
    from .prominences import sun_region

    say = say or (lambda _m: None)
    path = Path(project_path).resolve()
    project = EventProject.load(path)
    ps = project.parameters.get("prominence_stack", {})
    phdr, pw = resolve(path, ps.get("hdr")), resolve(path, ps.get("weight"))
    hdr_path = resolve(path, project.parameters.get("stack", {}).get("hdr"))
    if phdr and pw and hdr_path:
        main = read_fits_header(hdr_path)
        data, head = read_fits(phdr)
        weight, _ = read_fits(pw)
        x0, y0 = int(head.get("CROPX0", 0)) - int(main.get("CROPX0", 0)), int(head.get("CROPY0", 0)) - int(main.get("CROPY0", 0))
        head = dict(head, MOONX=float(main["MOONX"]) - x0, MOONY=float(main["MOONY"]) - y0, MOONR=float(main["MOONR"]))
        region = sun_region(np.moveaxis(np.asarray(data, np.float32), 0, -1), np.asarray(weight, np.float32), head)
        region = replace(region, origin=(region.origin[0] + x0, region.origin[1] + y0),
                         canvas=(int(main["NAXIS2"]), int(main["NAXIS1"])), blur_px=ps.get("blur_px"),
                         source=f"prominence stack ({len(ps.get('frames', []))} frames)")
        say(f"prominences from the prominence stack ({len(ps.get('frames', []))} short frames"
            + (f", blur {ps['blur_px']:.2f} px)" if ps.get("blur_px") else ")"))
        return region
    data = load_hdr(project_path)
    region = sun_region(data.hdr, data.weight, data.header)
    del data
    say("prominences from the HDR stack (Stack prominences for sharper ones from the short frames)")
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
        say("reading the stack")
        region = load_sun_region(path, say)
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
    project.parameters["final_prominences"] = {"file": str(final), "base": base, "base_file": str(base_file or ""),
                                               "strength": strength, "mode": mode}
    project.save(path)
    return str(final)
