"""Command-line tools: ``tse inventory``, ``tse lens``, ``tse lensfun-update``."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from .brackets import detect_brackets
from .calibration import CalibrationFrames, plan_calibration
from .frames import format_exposure
from .inventory import Inventory, find_raw_files
from .lensfun import LensfunDatabase, corner_falloff, download_database, import_database
from .metadata import ExifToolReader, rawpy_available, read_frames
from .project import EventProject


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
    p.add_argument("--c2", help="second contact, camera clock time, e.g. 2024-04-08T13:48:10")
    p.add_argument("--c3", help="third contact, camera clock time")
    p.add_argument("--bias", help="folder of bias frames")
    p.add_argument("--darks", help="folder of dark frames")
    p.add_argument("--flats", help="folder of flat frames")
    p.add_argument("--lensfun-db", help="folder holding lensfun XML files")

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

    if args.c2 and args.c3:
        changed = inv.mark_outside(datetime.fromisoformat(args.c2), datetime.fromisoformat(args.c3))
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
    plan = plan_calibration(inv.lights(), db, calib, lens_name=args.lens)
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
        project.set_inventory(inv)
        project.save(path)
        print(f"Project saved to {path}")
    return 0


def cmd_lensfun_update(args) -> int:
    import urllib.error

    try:
        path = import_database(args.source, args.dest) if args.source else download_database(args.dest)
    except urllib.error.URLError as exc:
        print(f"Download failed: {exc.reason}")
        if "CERTIFICATE" in str(exc.reason).upper():
            print(
                "Python could not verify GitHub's certificate. Fix it with either:\n"
                "  pip install --upgrade certifi      (then run this command again), or\n"
                "  on macOS, run 'Install Certificates.command' in your Python folder in Applications.\n"
                "Or download https://github.com/lensfun/lensfun/archive/refs/heads/master.zip in a browser\n"
                "and run:  tse lensfun-update --from <path to the zip>"
            )
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
