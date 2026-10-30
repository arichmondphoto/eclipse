"""Command-line tools: ``tse inventory``, ``tse lens``, ``tse lensfun-update``."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from .brackets import detect_brackets
from .calibration import CalibrationFrames, plan_calibration
from .frames import format_exposure, parse_duration
from .inventory import Inventory, find_raw_files
from .lensfun import LensfunDatabase, corner_falloff, download_database, import_database
from .metadata import ExifToolReader, rawpy_available, read_frames
from .project import EventProject
from .vignetting_profile import VignettingProfile, fit_profile_from_files, load_profile_library


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tse", description="Solar Eclipse Toolkit")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("inventory", help="read every RAW file in a folder and report brackets and calibration")
    p.add_argument("folder")
    p.add_argument("--recursive", action="store_true")
    p.add_argument("--csv", help="write the frame table to this CSV file")
    p.add_argument("--project", help="create or update this project file (.json)")
    p.add_argument("--lens", help="force this lensfun lens profile")
    p.add_argument("--no-exiftool", action="store_true", help="do not use exiftool even if installed")
    p.add_argument("--c2", help="second contact, e.g. 2024-04-08T13:47:33 (camera time, or true time with --clock-offset)")
    p.add_argument("--c3", help="third contact")
    p.add_argument("--clock-offset", help="add this to the camera clock to get true time, e.g. +1:04:08, -45s, 1h")
    p.add_argument("--bias", help="folder of bias frames")
    p.add_argument("--darks", help="folder of dark frames")
    p.add_argument("--flats", help="folder of flat frames")
    p.add_argument("--lensfun-db", help="folder holding lensfun XML files")
    p.add_argument("--profile", help="use this own-flats vignetting profile (.json) for every frame")
    p.add_argument("--profile-dir", help="folder of saved own-flats profiles (default: your profile library)")

    p = sub.add_parser("flat-profile", help="fit a smooth vignetting profile from your own flat frames")
    p.add_argument("source", help="folder of RAW flats, or a FITS master flat")
    p.add_argument("--name", default="", help="a name for the profile")
    p.add_argument("--output", "-o", help="save to this .json file instead of the profile library")
    p.add_argument("--block", type=int, default=32, help="block size in pixels for the fit (default 32)")
    p.add_argument("--crop", type=float, help="crop factor of the camera that took the flats (default: from lensfun, else 1.0)")
    p.add_argument("--lens", help="lens name to store, if the files do not say")
    p.add_argument("--focal", type=float, help="focal length to store, if the files do not say")
    p.add_argument("--aperture", type=float, help="aperture to store, if the files do not say")
    p.add_argument("--lensfun-db")

    p = sub.add_parser("lens", help="find a lensfun profile and show its vignetting")
    p.add_argument("name", nargs="?", default="")
    p.add_argument("--focal", type=float, required=True)
    p.add_argument("--aperture", type=float, required=True)
    p.add_argument("--crop", type=float, default=1.0, help="camera crop factor (1.0 = full frame)")
    p.add_argument("--lensfun-db")

    p = sub.add_parser("lensfun-update", help="download the lensfun database from GitHub")
    p.add_argument("--dest")
    p.add_argument("--from", dest="source", help="install from a downloaded lensfun .zip or a folder instead")

    p = sub.add_parser("frame", help="show everything read from one RAW file (for troubleshooting)")
    p.add_argument("file")
    p.add_argument("--lensfun-db")

    p = sub.add_parser("eclipse", help="eclipse circumstances (contacts, duration) for a site")
    p.add_argument("--date", required=True, help="eclipse date, e.g. 2024-04-08")
    p.add_argument("--lat", type=float, required=True, help="latitude, degrees north")
    p.add_argument("--lon", type=float, required=True, help="longitude, degrees east (west is negative)")
    p.add_argument("--elev", type=float, default=0.0, help="elevation, metres")
    p.add_argument("--utc-offset", type=float, help="also show local times at this UTC offset, hours")
    p.add_argument("--delta-t", type=float, help="TT-UT in seconds (default: the measured or predicted value)")
    p.add_argument("--elements", help="JSON file with Besselian elements for other eclipses")

    p = sub.add_parser("frames", help="list a project's frames, or exclude / include some")
    p.add_argument("project")
    p.add_argument("--exclude", nargs="+", metavar="FILE", help="leave these frames out (file names)")
    p.add_argument("--include", nargs="+", metavar="FILE", help="use these frames again")

    p = sub.add_parser("elements-update", help="download NASA's Besselian elements for every eclipse (1600-2400)")
    p.add_argument("--from", dest="source", help="use a downloaded copy of NASA's canon CSV instead")

    p = sub.add_parser("align", help="align the light frames of a project on the corona and find the Sun's centre")
    p.add_argument("project", help="project file made by 'tse inventory --project'")
    p.add_argument("--lat", type=float, help="observing site latitude (saved in the project)")
    p.add_argument("--lon", type=float, help="observing site longitude, east positive (saved in the project)")
    p.add_argument("--elev", type=float, help="observing site elevation, metres")
    p.add_argument("--clock-error", type=float,
                   help="seconds to add to the camera's UT to get true UT, if known (saved in the project); "
                        "otherwise bounded from totality")
    p.add_argument("--camera-utc-offset", type=float, help="camera time zone, hours, when the files carry none")
    p.add_argument("--mirrored", action=argparse.BooleanOptionalAction, default=None,
                   help="the optics mirror the image (telescope with a star diagonal); saved in the project")
    p.add_argument("--window", type=int, help="analysis window in superpixels (default: about 8 solar radii)")
    p.add_argument("--rotation", choices=["drift", "per-frame", "none"], default="drift",
                   help="field rotation: smooth drift over time (default), per frame, or none")
    p.add_argument("--any-angle", action="store_true", help="search any rotation and scale (alt-az mounts)")
    p.add_argument("--rho", type=float, help="tangential filter length in superpixels (default: from the Sun's size)")
    p.add_argument("--output", "-o", help="alignment results file (default: next to the project)")
    p.add_argument("--profile-dir", help="folder of saved own-flats profiles")
    p.add_argument("--lensfun-db")
    p.add_argument("--quiet", action="store_true")

    p = sub.add_parser("stack", help="merge the aligned frames into one linear HDR image")
    p.add_argument("project", help="project file (run 'tse align' on it first)")
    p.add_argument("--align", help="alignment results file (default: the one saved by 'tse align')")
    p.add_argument("--output", "-o", help="output folder (default: next to the project)")
    p.add_argument("--name", help="output file name prefix (default: the project name)")
    p.add_argument("--square", action="store_true",
                   help="quicker run: output only a square 10 solar radii wide around the Sun "
                        "(default: the full frame, the same size as the RAW files)")
    p.add_argument("--radii", type=float,
                   help="like --square, but this many solar radii from the centre to the edge")
    p.add_argument("--moon-frame", type=int,
                   help="frame number whose Moon (and earthshine) is kept (default: the alignment reference)")
    p.add_argument("--include-rejected", action="store_true",
                   help="also stack frames the Sun-centre fit flagged as off the Moon's track")
    p.add_argument("--no-hot-pixels", action="store_true", help="skip the hot-pixel correction")
    p.add_argument("--profile-dir", help="folder of saved own-flats profiles")
    p.add_argument("--lensfun-db")
    p.add_argument("--quiet", action="store_true")

    sub.add_parser("doctor", help="show which optional readers are available")

    args = parser.parse_args(argv)
    if args.command == "inventory":
        return cmd_inventory(args)
    if args.command == "lens":
        return cmd_lens(args)
    if args.command == "lensfun-update":
        return cmd_lensfun_update(args)
    if args.command == "doctor":
        return cmd_doctor()
    if args.command == "frame":
        return cmd_frame(args)
    if args.command == "eclipse":
        return cmd_eclipse(args)
    if args.command == "frames":
        return cmd_frames(args)
    if args.command == "elements-update":
        return cmd_elements_update(args.source)
    if args.command == "align":
        return cmd_align(args)
    if args.command == "stack":
        return cmd_stack(args)
    if args.command == "flat-profile":
        return cmd_flat_profile(args)
    return 1


def _db(path: str | None) -> LensfunDatabase:
    return LensfunDatabase.load([path] if path else None)


def cmd_inventory(args) -> int:
    files = find_raw_files(args.folder, args.recursive)
    if not files:
        print(f"No RAW files found in {args.folder}")
        return 1
    print(f"Reading {len(files)} RAW file(s)...")
    inv = Inventory.from_paths(files, use_exiftool=not args.no_exiftool)

    offset = parse_duration(args.clock_offset) if args.clock_offset else None
    if args.project and offset is None and Path(args.project).exists():
        offset = EventProject.load(args.project).camera_clock_offset_s
    if offset:
        print(f"Camera clock offset: {offset:+.0f} s")
    if args.c2 and args.c3:
        changed = inv.mark_outside(
            datetime.fromisoformat(args.c2), datetime.fromisoformat(args.c3), clock_offset_s=offset or 0.0
        )
        print(f"{changed} frame(s) marked as outside totality")

    report = detect_brackets(inv.frames)
    _print_table(inv)
    print()
    print(_summary(inv))
    print()
    print(report.describe())

    calib = CalibrationFrames(
        bias=read_frames(find_raw_files(args.bias)) if args.bias else [],
        darks=read_frames(find_raw_files(args.darks)) if args.darks else [],
        flats=read_frames(find_raw_files(args.flats)) if args.flats else [],
    )
    db = _db(args.lensfun_db)
    forced = VignettingProfile.load(args.profile) if args.profile else None
    library = load_profile_library(args.profile_dir)
    plan = plan_calibration(inv.lights(), db, calib, lens_name=args.lens, profile=forced, profiles=library)
    print()
    print(plan.describe())

    if args.csv:
        inv.to_csv(args.csv)
        print(f"\nFrame table written to {args.csv}")
    if args.project:
        path = Path(args.project)
        project = EventProject.load(path) if path.exists() else EventProject(name=Path(args.folder).name)
        project.raw_folder = str(Path(args.folder).resolve())
        project.lens_profile = args.lens or project.lens_profile
        if args.c2:
            project.second_contact = args.c2
        if args.c3:
            project.third_contact = args.c3
        if args.clock_offset:
            project.camera_clock_offset_s = offset
        project.set_inventory(inv)
        project.save(path)
        print(f"Project saved to {path}")
    return 0


def cmd_flat_profile(args) -> int:
    src = Path(args.source)
    if src.is_dir():
        paths = find_raw_files(src)
        if not paths:
            print(f"No RAW flats found in {src}")
            return 1
        meta = read_frames(paths)
    elif src.is_file():
        paths, meta = [str(src)], []
    else:
        print(f"{src} not found")
        return 1

    if meta:
        settings = {(f.lens_model, f.focal_length_mm, f.aperture_f) for f in meta}
        if len(settings) > 1:
            print("The flats were not all taken with the same lens, focal length and aperture:")
            for s in sorted(settings, key=str):
                print(f"  {s}")
            print("Put each set of settings in its own folder.")
            return 1
    try:
        import rawpy  # noqa: F401
    except ImportError:
        if any(not p.lower().endswith((".fit", ".fits", ".fts")) for p in paths):
            print("Reading RAW flats needs rawpy: pip install rawpy")
            return 1

    crop = args.crop
    if crop is None and meta:
        cam = _db(args.lensfun_db).find_camera(meta[0].camera_make, meta[0].camera_model)
        crop = cam.crop_factor if cam else (meta[0].crop_factor or 1.0)
    print(f"Fitting a vignetting profile from {len(paths)} flat(s)...")
    try:
        profile = fit_profile_from_files(paths, meta, crop_factor=crop or 1.0, block=args.block)
    except ValueError as exc:
        print(exc)
        return 1
    profile.name = args.name
    if args.lens:
        profile.lens_model = args.lens
    if args.focal:
        profile.focal_length_mm = args.focal
    if args.aperture:
        profile.aperture_f = args.aperture
    if not (profile.lens_model and profile.focal_length_mm and profile.aperture_f):
        print("Warning: lens, focal length or aperture unknown; set them with --lens, --focal and --aperture")
    print(profile.describe())
    path = profile.save(args.output)
    print(f"Profile saved to {path}")
    return 0


def cmd_lensfun_update(args) -> int:
    import urllib.error

    try:
        path = import_database(args.source, args.dest) if args.source else download_database(args.dest)
    except urllib.error.URLError as exc:
        from .net import certificate_help

        print(f"Download failed: {exc.reason}")
        if "CERTIFICATE" in str(exc.reason).upper():
            print(certificate_help("https://github.com/lensfun/lensfun/archive/refs/heads/master.zip",
                                   "tse lensfun-update --from <path to the zip>"))
        return 1
    except (FileNotFoundError, ValueError) as exc:
        print(exc)
        return 1
    db = LensfunDatabase.load([path])
    print(f"lensfun database saved to {path} ({len(db.lenses)} lenses)")
    return 0


def cmd_lens(args) -> int:
    db = _db(args.lensfun_db)
    if db.empty:
        print("lensfun database not found; run 'tse lensfun-update' first")
        return 1
    matches = db.match_lens(args.name, focal=args.focal)
    if not matches:
        print("No matching lens profile")
        return 1
    for m in matches:
        terms = m.lens.vignetting_terms(args.focal, args.aperture)
        if terms:
            brightness, stops = corner_falloff(terms, args.crop, m.lens.crop_factor)
            vign = f"corners {brightness:.0%} of centre ({stops:+.2f} stops)"
        else:
            vign = "no vignetting data for these settings"
        print(f"{m.score:5.1f}  {m.lens.label()}  [{m.lens.source_file}]  {vign}")
    return 0


def cmd_frames(args) -> int:
    path = Path(args.project)
    project = EventProject.load(path)
    inv = project.inventory
    by_name = {f.file_name.lower(): f for f in inv.frames}
    changed, unknown = [], []
    for names, role in ((args.exclude or [], "excluded"), (args.include or [], "light")):
        for name in names:
            f = by_name.get(Path(name).name.lower())
            if f is None:
                unknown.append(name)
                continue
            if f.role != role:
                f.role = role
                if "role" not in f.overrides:
                    f.overrides.append("role")
                changed.append(f"{f.file_name}: {role}")
    if changed:
        project.store_inventory(inv)
        project.save(path)
    for c in changed:
        print(c)
    for u in unknown:
        print(f"not in this project: {u}")
    if not (args.exclude or args.include):
        for f in inv.frames:
            print(f"{f.index:>3}  {f.file_name:<16} {f.exposure_text or '':>8}  {f.role}")
    lights = sum(f.role == "light" for f in inv.frames)
    print(f"{lights} of {len(inv.frames)} frames in use")
    return 1 if unknown else 0


def cmd_elements_update(source=None) -> int:
    import urllib.error

    from .eclipse import CANON_URL, update_canon

    if not source:
        print(f"Downloading NASA's eclipse catalogue from {CANON_URL} ...")
    try:
        path, n = update_canon(source)
    except urllib.error.URLError as exc:
        from .net import certificate_help

        print(f"Download failed: {exc.reason}")
        if "CERTIFICATE" in str(exc.reason).upper():
            print(certificate_help(CANON_URL, "tse elements-update --from <path to the .csv file>"))
        else:
            print(f"Download {CANON_URL} in a browser and run: tse elements-update --from <file>")
        return 1
    except (OSError, ValueError) as exc:
        print(exc)
        return 1
    print(f"Saved Besselian elements for {n} eclipses to {path}")
    return 0


def _ensure_elements(day, elements_file=None):
    """Elements for this date, downloading NASA's catalogue once if needed."""
    from .eclipse import CANON_CACHE, elements_for

    try:
        return elements_for(day, elements_file)
    except KeyError:
        if CANON_CACHE.is_file():
            raise
        print("This eclipse is not in the bundled set; fetching NASA's full catalogue once.")
        if cmd_elements_update() != 0:
            raise
        return elements_for(day, elements_file)


def cmd_eclipse(args) -> int:
    from .eclipse import LocalEclipse, Site

    try:
        el = _ensure_elements(args.date, args.elements)
    except KeyError as exc:
        print(exc.args[0])
        return 1
    circ = LocalEclipse(el, Site(args.lat, args.lon, args.elev), args.delta_t).circumstances()
    print(circ.describe(args.utc_offset))
    return 0


def cmd_align(args) -> int:
    from .align import AlignParams
    from .calibration import plan_calibration
    from .eclipse import Site
    from .pipeline import default_loader, run_alignment

    path = Path(args.project)
    project = EventProject.load(path)
    if args.lat is not None:
        project.latitude = args.lat
    if args.lon is not None:
        project.longitude = args.lon
    if args.elev is not None:
        project.elevation_m = args.elev
    if args.camera_utc_offset is not None:
        project.camera_utc_offset_h = args.camera_utc_offset
    if args.mirrored is not None:
        project.mirrored = args.mirrored
    if args.clock_error is not None:
        project.clock_error_s = args.clock_error
    inv = project.inventory
    lights = inv.lights()
    db = _db(args.lensfun_db)
    library = load_profile_library(args.profile_dir)
    plan = plan_calibration(lights, db, None, lens_name=project.lens_profile or None,
                            use_lensfun=project.use_lensfun, profiles=library)
    cam = db.find_camera(lights[0].camera_make, lights[0].camera_model) if lights and not db.empty else None
    crop = cam.crop_factor if cam else 1.0
    site = None
    if project.latitude is not None and project.longitude is not None:
        site = Site(project.latitude, project.longitude, project.elevation_m)
    if site is not None and lights and lights[0].timestamp:
        try:
            _ensure_elements(lights[0].timestamp[:10])
        except KeyError as exc:
            print(exc.args[0])
    params = AlignParams(rho=args.rho, rotation_model=args.rotation, coarse_rotation=args.any_angle,
                         solve_rotation=args.rotation != "none")
    say = (lambda m: None) if args.quiet else print
    try:
        result = run_alignment(lights, default_loader(plan, crop), params, window=args.window, site=site,
                               clock_error_s=project.clock_error_s, crop_factor=crop,
                               utc_offset_h=project.camera_utc_offset_h, mirrored=project.mirrored,
                               progress=say)
    except RuntimeError as exc:
        print(exc)
        return 1
    out = Path(args.output) if args.output else path.with_name(path.stem + "_align.json")
    result.save(out)
    project.parameters["align"] = {"file": str(out), "rho": args.rho, "rotation": args.rotation,
                                   "window": args.window, "clock_error": args.clock_error}
    project.save(path)
    print()
    print(result.describe())
    print(f"\nAlignment saved to {out}")
    return 0


def cmd_stack(args) -> int:
    from .calibration import plan_calibration
    from .rawio import read_mosaic
    from .stack import FrameInfo, StackParams, stack_frames

    path = Path(args.project)
    project = EventProject.load(path)
    align_path = args.align or project.parameters.get("align", {}).get("file") or str(
        path.with_name(path.stem + "_align.json"))
    if not Path(align_path).exists():
        print(f"No alignment results at {align_path}; run 'tse align {path}' first.")
        return 1
    alignment = json.loads(Path(align_path).read_text(encoding="utf-8"))
    frames = {f.index: f for f in project.inventory.lights()}
    ref = alignment["reference_index"]
    rejected = set()
    if alignment.get("sun") and not args.include_rejected:
        order = [fr["index"] for fr in alignment["frames"]]
        rejected = {order[k] for k in alignment["sun"].get("rejected", [])}
    ev_ref = frames[ref].exposure_value() if ref in frames else None
    infos = []
    for fr in alignment["frames"]:
        i = fr["index"]
        if i not in frames:
            print(f"frame #{i} ({fr['file_name']}) is no longer a light frame in the project; skipped")
            continue
        if i in rejected:
            print(f"frame #{i} ({fr['file_name']}) was flagged as off the Moon's track; skipped "
                  "(use --include-rejected to keep it)")
            continue
        ev = frames[i].exposure_value()
        ratio = 2.0 ** (ev - ev_ref) if ev is not None and ev_ref is not None else 1.0
        mx, my = fr["moon_ref"]
        infos.append(FrameInfo(i, fr["file_name"], ratio, fr["matrix"], mx, my, fr["moon_frame"]["radius"]))
    if len(infos) < 1:
        print("nothing to stack")
        return 1
    db = _db(args.lensfun_db)
    library = load_profile_library(args.profile_dir)
    lights = [frames[f.index] for f in infos]
    plan = plan_calibration(lights, db, None, lens_name=project.lens_profile or None,
                            use_lensfun=project.use_lensfun, profiles=library)
    calibs = {fc.frame_index: fc for fc in plan.frames}
    cam = db.find_camera(lights[0].camera_make, lights[0].camera_model) if not db.empty else None
    crop_factor = cam.crop_factor if cam else 1.0
    radii = args.radii if args.radii is not None else (5.0 if args.square else None)
    params = StackParams(crop_radii=radii, moon_frame=args.moon_frame, hot_pixels=not args.no_hot_pixels)
    out_dir = Path(args.output) if args.output else path.parent
    name = args.name or path.stem
    say = (lambda m: None) if args.quiet else print
    try:
        result, _hdr, _w = stack_frames(infos, lambda f: read_mosaic(frames[f.index].path), calibs,
                                        tuple(alignment["image_size"]), ref, alignment.get("sun"), params,
                                        out_dir=out_dir, name=name, image_crop=crop_factor, progress=say)
    except (RuntimeError, ValueError) as exc:
        print(exc)
        return 1
    project.parameters["stack"] = {"align": str(align_path), "radii": radii, "moon_frame": result.moon_frame,
                                   **result.outputs}
    project.save(path)
    print()
    print(result.describe())
    return 0


def cmd_frame(args) -> int:
    """Print the parsed metadata, the raw lens tags and the lens matches."""
    frame = read_frames([args.file])[0]
    print("Parsed metadata:")
    for key, value in frame.to_dict().items():
        if value not in (None, "", []):
            print(f"  {key}: {value}")
    try:
        import exifread

        with open(args.file, "rb") as fh:
            tags = exifread.process_file(fh, details=True)
        print("Lens-related tags (exifread):")
        for key in sorted(tags):
            if "lens" in key.lower() or key.endswith(("Make", "Model", "FocalLength", "FNumber")):
                tag = tags[key]
                print(f"  {key}: {getattr(tag, 'printable', tag)}  (raw: {getattr(tag, 'values', '')})")
    except Exception as exc:
        print(f"exifread could not read the file: {exc}")
    db = _db(args.lensfun_db)
    if not db.empty:
        print("lensfun matches:")
        maker = frame.lens_make or frame.camera_make
        for m in db.match_lens(frame.lens_model, frame.lens_spec, frame.focal_length_mm, maker):
            vign = "vignetting data" if m.lens.has_vignetting else "NO vignetting data"
            print(f"  {m.score:5.1f}  {m.lens.label()}  ({vign}; {', '.join(m.reasons)})")
    return 0


def cmd_doctor() -> int:
    exiftool = ExifToolReader.find()
    print(f"exiftool: {'found: ' + ' '.join(exiftool.command) if exiftool else 'not found (optional; needed for full lens names on some cameras)'}")
    print(f"rawpy:    {'available' if rawpy_available() else 'not installed (pip install rawpy) - needed for exact black/white levels'}")
    try:
        import exifread

        print(f"exifread: {exifread.__version__}")
    except ImportError:
        print("exifread: not installed (pip install exifread)")
    try:
        import truststore  # noqa: F401

        print("certificates: system store (truststore)")
    except ImportError:
        print("certificates: certifi only (pip install truststore to use your system's certificates)")
    db = LensfunDatabase.load()
    print(f"lensfun:  {len(db.lenses)} lenses from {len(db.files)} file(s)" if not db.empty else "lensfun:  database not found (run 'tse lensfun-update')")
    return 0


def _print_table(inv: Inventory) -> None:
    header = f"{'#':>3}  {'file':<18} {'time':<12} {'exposure':>9} {'f':>5} {'ISO':>6} {'focal':>6} {'black':>6} {'white':>6}  {'br':>3}  role"
    print(header)
    print("-" * len(header))
    for f in inv.frames:
        dt = f.datetime()
        time = dt.strftime("%H:%M:%S.%f")[:11] if dt else ""
        print(
            f"{f.index:>3}  {f.file_name[:18]:<18} {time:<12} {f.exposure_text or format_exposure(f.exposure_s):>9} "
            f"{_num(f.aperture_f):>5} {_num(f.iso):>6} {_num(f.focal_length_mm):>6} "
            f"{_num(f.black_level):>6} {_num(f.white_level):>6}  {'' if f.bracket_id is None else f.bracket_id:>3}  {f.role}"
        )


def _summary(inv: Inventory) -> str:
    start, end = inv.time_span()
    lines = [
        f"Frames: {len(inv.frames)} ({len(inv.lights())} light)",
        "Cameras: " + ", ".join(f"{k} ({v})" for k, v in inv.cameras().items()),
        "Lenses: " + ", ".join(f"{k} ({v})" for k, v in inv.lenses().items()),
        "ISO: " + ", ".join(_num(k) for k in sorted(k for k in inv.settings("iso") if k)),
        "Aperture: " + ", ".join("f/" + _num(k) for k in sorted(k for k in inv.settings("aperture_f") if k)),
    ]
    if start and end:
        lines.append(f"Time span: {start:%H:%M:%S} to {end:%H:%M:%S} ({(end - start).total_seconds():.0f} s)")
    warn = sorted({w for f in inv.frames for w in f.warnings})
    for w in warn:
        lines.append(f"Note: {w}")
    return "\n".join(lines)


def _num(x) -> str:
    if x is None:
        return ""
    return f"{x:.1f}".rstrip("0").rstrip(".") if isinstance(x, float) else str(x)


if __name__ == "__main__":
    sys.exit(main())
