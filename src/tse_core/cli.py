"""Command-line tools: ``totality inventory``, ``totality lens``, ``totality lensfun-update``."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path

from .brackets import detect_brackets
from .calibration import CalibrationFrames, plan_calibration
from .frames import format_exposure, parse_duration
from .inventory import Inventory, find_raw_files
from .lensfun import LensfunDatabase, corner_falloff, download_database, import_database
from .metadata import ExifToolReader, rawpy_available, read_frames
from .enhance import ACHF_SCALES, ACHF_WEIGHTS, DEFAULT_BLEND, DEFAULT_LAYERS, LAYERS, MGN_SCALES, MGN_WEIGHTS
from .project import EventProject
from .vignetting_profile import VignettingProfile, fit_profile_from_files, load_profile_library


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="totality", description="Solar Eclipse Toolkit")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("inventory", help="read every RAW file in a folder and report brackets and calibration")
    p.add_argument("folder", help="the folder of light frames, or a project folder holding lights/ (and "
                                  "optionally darks/, flats/, bias/); with a project folder the project file "
                                  "and every result go in it")
    p.add_argument("--recursive", action="store_true", help="also read sub-folders")
    p.add_argument("--csv", help="write the frame table to this CSV file")
    p.add_argument("--project", help="create or update this project file (.json)")
    p.add_argument("--lens", help="force this lensfun lens profile")
    p.add_argument("--focal-length", type=float,
                   help="focal length in mm, used only for frames (and flats) whose files do not record it, "
                        "e.g. a camera on a telescope; optional, nothing needs it (saved in the project; 0 clears)")
    p.add_argument("--f-number", type=float,
                   help="f-number (focal ratio), used only for frames (and flats) whose files do not record it; "
                        "optional: without it every frame is taken to be at the same f-number (saved in the "
                        "project; 0 clears)")
    p.add_argument("--no-exiftool", action="store_true", help="do not use exiftool even if installed")
    p.add_argument("--eclipse", help="the eclipse, by its date, YYYY-MM-DD (default: the frames' date)")
    p.add_argument("--lat", type=float, help="site latitude in degrees, north positive (saved in the project)")
    p.add_argument("--lon", type=float, help="site longitude in degrees, east positive (west is negative)")
    p.add_argument("--elev", type=float, help="site elevation in metres")
    p.add_argument("--camera-utc-offset", type=float,
                   help="the camera clock's time zone in hours, e.g. -6, when the files do not record one")
    p.add_argument("--clock-error", type=float,
                   help="seconds the camera clock was behind true time (true minus camera), when known")
    p.add_argument("--c2", help="second contact typed in, YYYY-MM-DDThh:mm:ss, in the camera's clock "
                                "(or in true time with --clock-offset); default: computed from the eclipse and site")
    p.add_argument("--c3", help="third contact typed in (with --c2)")
    p.add_argument("--clock-offset", help="with --c2/--c3 typed in another clock: add this to the camera clock to "
                                          "get that clock, e.g. +1:04:08, -45s, 1h")
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
    p.add_argument("--lensfun-db", help="folder holding lensfun XML files (default: the installed database)")

    p = sub.add_parser("lens", help="find a lensfun profile and show its vignetting")
    p.add_argument("name", nargs="?", default="", help="lens name or part of it (empty: list the lenses lensfun knows)")
    p.add_argument("--focal", type=float, required=True, help="focal length, mm")
    p.add_argument("--aperture", type=float, required=True, help="f-number")
    p.add_argument("--crop", type=float, default=1.0, help="camera crop factor (1.0 = full frame)")
    p.add_argument("--lensfun-db", help="folder holding lensfun XML files (default: the installed database)")

    p = sub.add_parser("lensfun-update", help="download the lensfun database from GitHub")
    p.add_argument("--dest", help="folder to put the database in (default: the toolkit's own data folder)")
    p.add_argument("--from", dest="source", help="install from a downloaded lensfun .zip or a folder instead")

    p = sub.add_parser("frame", help="show everything read from one RAW file (for troubleshooting)")
    p.add_argument("file", help="one RAW file")
    p.add_argument("--lensfun-db", help="folder holding lensfun XML files (default: the installed database)")

    p = sub.add_parser("eclipse", help="eclipse circumstances (contacts, duration) for a site")
    p.add_argument("--date", required=True, help="eclipse date, YYYY-MM-DD")
    p.add_argument("--lat", type=float, required=True, help="latitude, degrees north")
    p.add_argument("--lon", type=float, required=True, help="longitude, degrees east (west is negative)")
    p.add_argument("--elev", type=float, default=0.0, help="elevation, metres")
    p.add_argument("--utc-offset", type=float, help="also show local times at this UTC offset, hours")
    p.add_argument("--delta-t", type=float, help="TT-UT in seconds (default: the measured or predicted value)")
    p.add_argument("--elements", help="JSON file with Besselian elements for other eclipses")

    p = sub.add_parser("frames", help="list a project's frames, or exclude / include some")
    p.add_argument("project", help="the project file (.json)")
    p.add_argument("--exclude", nargs="+", metavar="FILE", help="leave these frames out (file names)")
    p.add_argument("--include", nargs="+", metavar="FILE", help="use these frames again")

    p = sub.add_parser("elements-update", help="download NASA's Besselian elements for every eclipse (1600-2400)")
    p.add_argument("--from", dest="source", help="use a downloaded copy of NASA's canon CSV instead")

    p = sub.add_parser("align", help="align the light frames of a project on the corona and find the Sun's centre")
    p.add_argument("project", help="project file made by 'totality inventory --project'")
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
    p.add_argument("--output", "-o", help="alignment results file (default: json/<project>_align.json in the project folder)")
    p.add_argument("--profile-dir", help="folder of saved own-flats profiles")
    p.add_argument("--lensfun-db", help="folder holding lensfun XML files (default: the installed database)")
    p.add_argument("--quiet", action="store_true", help="print only the result, not each step")

    p = sub.add_parser("stack", help="merge the aligned frames into one linear HDR image")
    p.add_argument("project", help="project file (run 'totality align' on it first)")
    p.add_argument("--align", help="alignment results file (default: the one saved by 'totality align')")
    p.add_argument("--output", "-o", help="folder for the pictures (default: next to the project); the .fit and .json files always go in the fits and json folders inside the project folder")
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
    p.add_argument("--vignetting", choices=["auto", "master", "profile", "compare", "lensfun", "none"],
                   default="auto",
                   help="vignetting correction: auto = master flat if flats are chosen, else the lensfun profile "
                        "(default); master = master flat (flats from the same session: vignetting and dust); "
                        "profile = a smooth surface fitted to the flats (flats from another time: vignetting only); "
                        "compare = master flat and report how it differs from that surface; lensfun = the lens's "
                        "lensfun profile; none = no correction")
    p.add_argument("--lens", default="", help="lensfun lens for --vignetting lensfun/auto (default: matched from the files)")
    p.add_argument("--flats-uneven-light", action="store_true",
                   help="the light on the flats was brighter on one side: take a straight slope across them out "
                        "(off by default: an off-centre lens gives the same slope)")
    p.add_argument("--darks", choices=["use", "none"], default="none",
                   help="use the project's dark frames (hot pixels, thermal signal, amp glow; the separate hot-pixel "
                        "fix is then off), or not (default)")
    p.add_argument("--bias", choices=["use", "none"], default="use",
                   help="use the project's bias frames for the offset (default), or the black level in each file")
    p.add_argument("--profile-dir", help="folder of saved own-flats profiles")
    p.add_argument("--lensfun-db", help="folder holding lensfun XML files (default: the installed database)")
    p.add_argument("--quiet", action="store_true", help="print only the result, not each step")

    p = sub.add_parser("enhance", help="reveal coronal structure: NRGF, FNRGF, adaptive filter, natural, blend")
    enhance_parser = p
    p.add_argument("project", help="project file (run 'totality stack' on it first)")
    p.add_argument("--settings", help="Enhance settings file to start from (saved with 'Save settings...' on the "
                                      "Enhance tab, or the <name>_enhance.json in the project's json folder); "
                                      "options given here as well override the file")
    p.add_argument("--hdr", help="HDR FITS file (default: the one saved by 'totality stack')")
    p.add_argument("--output", "-o", help="folder for the pictures (default: next to the project); the .fit and .json files always go in the fits and json folders inside the project folder")
    p.add_argument("--name", help="output file name prefix (default: the project name)")
    p.add_argument("--layers", default=",".join(DEFAULT_LAYERS),
                   help=f"comma-separated layers to write, from {', '.join(LAYERS)} (default: all but achf, which "
                        "takes minutes on a full frame)")
    p.add_argument("--blend", default=",".join(f"{k}={v}" for k, v in DEFAULT_BLEND.items()),
                   help="suggested blend as layer=weight pairs, each optionally ':overlay' or ':linear' (combined as "
                        "each method's own publication does: NRGF, FNRGF and FNRGF-N as Druckmuellerova, Morgan & "
                        "Habbal 2011, typically 0.3-1; adaptive as Morgan & Druckmueller 2014, 0.43 for their mix; "
                        "achf as Druckmuellerova's thesis 2013, added to the linear stack before the natural "
                        "stretch, typically 20-50 with the log stretch, about twice that with gamma) or, for "
                        "adaptive and achf only, ':average' (a gentler look, typically 0.4-0.6); "
                        "default soft light, "
                        "e.g. natural=1,fnrgf_n=0.7,adaptive=0.1:overlay; or 'none' (default: %(default)s)")
    p.add_argument("--order", type=int, default=10, help="FNRGF Fourier order (default 10)")
    p.add_argument("--segments", type=int, default=50, help="FNRGF angular segments (default 50)")
    p.add_argument("--mean-att", help="FNRGF attenuation A0,A1,... for the mean (default 1,0.9,...,0)")
    p.add_argument("--std-att", help="FNRGF attenuation S0,S1,... for the deviation (default 1,0.9,...,0)")
    p.add_argument("--noise-m", type=float, default=0.7, help="FNRGF-N noise compensation, 0..0.99 (default 0.7)")
    p.add_argument("--adaptive-k", type=float, default=0.7,
                   help="adaptive layer: strength of the local contrast at each scale, k in arctan(k C) "
                        "(default 0.7, as in Morgan & Druckmueller 2014)")
    p.add_argument("--achf-scales",
                   help="Adaptive (ACHF): the size (kernel sigma) of each of its six scales, finest first, in %% of "
                        "the Sun's radius (default " + ",".join(f"{100 * v:.4g}" for v in ACHF_SCALES)
                        + "; the last two start at strength 0)")
    p.add_argument("--mgn-scales", default=",".join(f"{v:g}" for v in MGN_SCALES),
                   help="Adaptive (MGN): the width (kernel sigma) of each of its six scales, finest first, in px of "
                        "the full-size stack (default %(default)s, as Morgan & Druckmueller 2014; a frame of reference: a 400 mm lens "
                        "on a 45-megapixel full-frame camera; raise them with a longer lens or smaller pixels)")
    p.add_argument("--mgn-weights", default=",".join(f"{v:g}" for v in MGN_WEIGHTS),
                   help="Adaptive (MGN): the weight g_i of each scale, 0-1, finest first (default %(default)s: the "
                        "plain mean); the layer is 1/6 of the weighted sum, as in Morgan & Druckmueller "
                        "2014, so lowering one weakens it; 0 leaves the scale out")
    p.add_argument("--achf-weights", default=",".join(f"{v:g}" for v in ACHF_WEIGHTS),
                   help="Adaptive (ACHF): strength of each of its six scales, finest first; 0 leaves a scale out "
                        "(default %(default)s)")
    p.add_argument("--achf-part-limit", type=float, default=2.0,
                   help="Adaptive (ACHF): a pixel this many times brighter or darker than the corona at its height "
                        "(a prominence) is a separate part, left out of the corona's kernels and filtered with its "
                        "own pixels only, so no dark fringe forms round it; 0 = off (default 2)")
    p.add_argument("--achf-contrast", type=float, default=1.0,
                   help="Adaptive (ACHF): how strongly the layer shows its detail with soft light, overlay or "
                        "average: at 1, a pixel about 1.1%% brighter than its surroundings shows at 82%% grey; raise "
                        "it (1.5-3) if the layer looks flat. With achf on ':linear' it multiplies that blend weight "
                        "instead: leave it at 1 there (default 1)")
    p.add_argument("--noise-gate", type=float, default=0.0,
                   help="adaptive layer: each scale's detail counts only as far as it stands above the stack's "
                        "noise at that scale (e.g. 3); 0 = off")
    p.add_argument("--even", type=float, default=0.0,
                   help="take out this fraction of the large brightness differences round the Sun (streamers "
                        "against coronal holes); 0 = off, e.g. 0.7")
    p.add_argument("--midtone", type=float, default=0.0,
                   help="put the inner corona's median brightness at this level with a midtones transfer curve; "
                        "0 = off, e.g. 0.6")
    p.add_argument("--shoulder", type=float, default=0.0,
                   help="highlight roll-off: 0 = hard white point (the brightest 0.1%% clip); e.g. 0.3 = the top 30%% of "
                        "the range rolls off smoothly to white, nothing clips and colours keep their hue")
    p.add_argument("--natural-stretch", choices=["log", "gamma"], default="log",
                   help="natural layer: log brightness (default) or gamma, the base Morgan & "
                        "Druckmueller 2014 put under their MGN detail (darker on an eclipse: use --gamma-white 99.9)")
    p.add_argument("--gamma", type=float, default=3.2,
                   help="gamma of --natural-stretch gamma (3.2 as in Morgan & Druckmueller 2014; 2.5-4 suggested)")
    p.add_argument("--gamma-black", type=float, default=0.0,
                   help="--natural-stretch gamma: black point as a percentile of the stack; 0 = its "
                        "minimum (default, as in Morgan & Druckmueller 2014); 0.5-2 if the sky looks grey")
    p.add_argument("--gamma-white", type=float, default=100.0,
                   help="--natural-stretch gamma: white point as a percentile; 100 = the maximum "
                        "(default, as in Morgan & Druckmueller 2014), 99.9 recommended for eclipses (the brightest "
                        "0.1%% goes white and the corona comes up)")
    p.add_argument("--flatten", type=float, default=0.0,
                   help="natural layer: how much of the radial fall-off to remove, 0..1; 0 = off (default), "
                        "e.g. 0.8")
    p.add_argument("--saturation", type=float, default=1.0, help="colour saturation, 0 = monochrome (default 1)")
    p.add_argument("--white", choices=["corona", "camera", "none"], default="corona",
                   help="white balance: inner corona neutral (default), the camera's as-shot, or none")
    p.add_argument("--format", choices=["tif", "png", "jpg"], default="tif",
                   help="picture type: 16-bit TIFF (default), 16-bit PNG or 8-bit JPG; all 300 dpi")
    p.add_argument("--keep-sky", "--keep-background-colour", dest="keep_background_colour", action="store_true",
                   help="do not model and remove the sky background (level, colour and gradient)")
    p.add_argument("--quiet", action="store_true", help="print only the result, not each step")

    p = sub.add_parser("earthshine", help="Moon-aligned stack of the long exposures: a layer for the black lunar disc")
    p.add_argument("project", help="project file (run 'totality stack' on it first)")
    p.add_argument("--stops", type=float, default=6.0,
                   help="use frames within this many stops of the longest exposure (default 6)")
    p.add_argument("--no-limb-glare", action="store_true",
                   help="do not model the steep glare just inside the limb (shows as a bright rim)")
    p.add_argument("--prominence-glare", action="store_true",
                   help="fit and remove the fan of glare (and spikes) round a bright prominence (default off: the "
                        "fan is shown as the data has it; removing it cannot bring back what is under it)")
    p.add_argument("--no-prominence-glare", action="store_true",
                   help="leave the glare fan in (the default)")
    p.add_argument("--limb-shape", action="store_true",
                   help="fit the limb glare as light fading inward, round the limb, which keeps maria by the limb "
                        "(default off)")
    p.add_argument("--no-limb-shape", action="store_true",
                   help="no limb shape (the default)")
    p.add_argument("--channel", default="auto", choices=["auto", "all", "green", "green+blue", "blue"],
                   help="colour channels to make the earthshine from (default auto: chosen on the stack; the red "
                        "channel carries most of the chromosphere glare)")
    p.add_argument("--glare-order", type=float, default=3.5,
                   help="smoothness of the glare removed from the disc, 0-6 in half steps: higher follows uneven "
                        "glare better but flattens the maria more (default 3.5; 2 = a gentle gradient, 0 = one even "
                        "level; a half step averages the two neighbouring polynomial orders)")
    p.add_argument("--smooth", type=float, default=0.0, help="smoothing of the layer in pixels (default 0 = none)")
    p.add_argument("--brightness", type=float, default=1.0,
                   help="brightest lunar surface in the layer, 0-1 (default 1)")
    p.add_argument("--keep-glare", type=float, default=0.1,
                   help="earthshine layer file for Photoshop (<name>_earthshine_stack.tif): how much of the removed glare "
                        "to keep as a glow inside the limb, 0-1 (default 0.1)")
    p.add_argument("--limb-soften", type=float, default=0.0,
                   help="earthshine layer file for Photoshop: extra softening of the limb in px, on top of the real "
                        "one (default 0 = none)")
    p.add_argument("--no-colour", action="store_true",
                   help="make the earthshine layer grey (default: natural colour, so a prominence at the limb "
                        "stays red on the Moon side too)")
    p.add_argument("--limb-colour", default="recorded", choices=["recorded", "no-red", "neutral"],
                   help="colour of the glare just inside the limb: recorded (red from the chromosphere, as the "
                        "camera saw it), no-red (the colour of the Moon's middle), neutral (no red, and the whole "
                        "Moon in the corona's white instead of earthshine's own slightly blue colour)")
    p.add_argument("--edge-fill", type=float, default=0.0,
                   help="fill at least this %% of the Moon's radius just inside the limb from further in (default "
                        "0: only where the noise is over 10x the middle's; 2.5 = 10.9 px on a 400 mm Moon)")
    p.add_argument("--no-edge-grain", action="store_true",
                   help="leave the filled last pixels by the limb smooth (default: grain like the rest)")
    p.add_argument("--surface-denoise", type=float, default=0.0,
                   help="edge-preserving smoothing of the lunar surface, px (0 = off; e.g. 4): maria keep crisp edges")
    p.add_argument("--edge-smooth", type=float, default=0.0,
                   help="most smoothing of the noisier pixels near the limb, px (default 0 = none; e.g. 3)")
    p.add_argument("--include-rejected", action="store_true", help="also use frames marked as rejected")
    p.add_argument("--output", "-o", help="folder for the pictures (default: next to the project); the .fit and .json files always go in the fits and json folders inside the project folder")
    p.add_argument("--name", help="output file name prefix (default: the project name)")
    p.add_argument("--format", choices=["tif", "png", "jpg"], default="tif",
                   help="picture type: 16-bit TIFF (default), 16-bit PNG or 8-bit JPG; all 300 dpi")
    p.add_argument("--profile-dir", help="folder of saved own-flats profiles (default: your profile library)")
    p.add_argument("--lensfun-db", help="folder holding lensfun XML files (default: the installed database)")
    p.add_argument("--quiet", action="store_true", help="print only the result, not each step")

    p = sub.add_parser("final", help="the finished image: Moon and corona stretched together from the stacks")
    p.add_argument("project", help="the project file (.json)")
    p.add_argument("--base", default="blend", choices=["blend", "natural"],
                   help="enhanced layer to use as the corona (default: blend, the last one written)")
    p.add_argument("--strength", type=float, default=1.0, help="brightness of the lunar surface, 1 = as made by the "
                   "earthshine step (default 1)")
    p.add_argument("--moon-edge", default="as-stacked", choices=["as-stacked", "blended"],
                   help="where the Moon meets the corona: as-stacked (the corona exactly as in the Blend without the "
                        "Moon, the Moon only in the dark disc, black where no frame recorded anything; default) or "
                        "blended (the older way: the first few px outside the limb recomputed from both stacks)")
    p.add_argument("--moon-layer", action="store_true",
                   help="write only the Moon as --base draws it (with --strength), transparent elsewhere, for "
                        "Photoshop: above that layer in Normal mode at 100 %% it gives the final image (default "
                        "<name>_earthshine_photoshop.tif; TIFF or PNG)")
    p.add_argument("--name", help="output file name prefix (default: the project name)")
    p.add_argument("--output", "-o", help="output file (.tif, .png or .jpg); default <name>_earthshine_blend.tif or _natural.tif by --base")
    p.add_argument("--quiet", action="store_true", help="print only the result, not each step")

    p = sub.add_parser("prominence-stack",
                       help="stack the shortest exposures round the Sun; 'totality final' then sharpens the HDR stack's "
                            "prominences by as much as they show")
    p.add_argument("project", help="project file (run 'totality stack' on it first)")
    p.add_argument("--longest-stops", type=float, default=2.0,
                   help="frames from the shortest aligned exposure up to this many stops longer (default 2)")
    p.add_argument("--radii", type=float, default=1.6,
                   help="the square stacked, in solar radii from the Sun's centre (default 1.6)")
    p.add_argument("--settings-only", action="store_true",
                   help="only save the settings below for 'totality final' (no stacking)")
    p.add_argument("--off", action="store_true", help="'totality final' leaves the HDR stack's prominences as they are")
    p.add_argument("--sharpen", type=float, default=0.5,
                   help="0.5: as sharp as the short frames (default); 1: also undo the blur measured on the Moon's edge; 0 off")
    p.add_argument("--structure", type=float, default=0.0,
                   help="bring out strands and loops inside them: 0 off (default), 0.3-0.6 more texture")
    p.add_argument("--brightness", type=float, default=1.0,
                   help="their brightness against the corona (default 1, as measured; 0.2-0.5 shows their structure)")
    p.add_argument("--min-redness", type=float, default=1.1,
                   help="how much redder than the corona light must be to count as a prominence (default 1.1)")
    p.add_argument("--no-chromosphere", action="store_true", help="leave out the thin pink rim along the limb")
    p.add_argument("--chromosphere-arcsec", type=float, default=12.0,
                   help="height of the rim left out by --no-chromosphere, arcsec (default 12)")
    p.add_argument("--quiet", action="store_true", help="print only the result, not each step")

    p = sub.add_parser("prominences", help="find the prominences and write them as a layer (transparent elsewhere)")
    p.add_argument("project", help="project file (run 'totality stack' on it first)")
    p.add_argument("--height", type=float, default=0.3, help="how far above the solar limb to look, in solar radii "
                                                            "(default 0.3)")
    p.add_argument("--sensitivity", type=float, default=4.0,
                   help="red excess needed, in noise standard deviations (default 4; lower finds fainter ones)")
    p.add_argument("--min-redness", type=float, default=1.1,
                   help="how much redder than the corona light must be (default 1.1)")
    p.add_argument("--no-chromosphere", action="store_true", help="leave out the thin pink rim along the limb")
    p.add_argument("--chromosphere-arcsec", type=float, default=12.0,
                   help="height of the rim left out by --no-chromosphere, arcsec (default 12)")
    p.add_argument("--brightness", type=float, default=1.0, help="brightness of the layer (1 = as measured)")
    p.add_argument("--saturation", type=float, default=1.2, help="colour saturation of the layer (1 = as measured)")
    p.add_argument("--output", "-o", help="layer file (.tif, .png or .jpg); default <project>_prominences.tif")
    p.add_argument("--quiet", action="store_true", help="print only the result, not each step")

    p = sub.add_parser("final-prominences", help="put the prominence layer on top of the final image")
    p.add_argument("project", help="the project file (.json)")
    p.add_argument("--base", default="final", choices=["final", "blend", "natural"],
                   help="final: the image from 'totality final' (default); blend or natural: the last-written layer")
    p.add_argument("--base-file", help="put them on this image instead (the size of the HDR stack)")
    p.add_argument("--strength", type=float, default=1.0, help="opacity (normal) or brightness (screen, lighten)")
    p.add_argument("--mode", default="normal", choices=["normal", "screen", "lighten"],
                   help="normal (default) keeps their colour over the bright inner corona")
    p.add_argument("--output", "-o", help="output file (.tif, .png or .jpg); default <project>_final_prominences.tif")
    p.add_argument("--quiet", action="store_true", help="print only the result, not each step")

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
    if args.command == "enhance":
        return cmd_enhance(args, given_options(enhance_parser, sys.argv[1:] if argv is None else argv))
    if args.command == "earthshine":
        return cmd_earthshine(args)
    if args.command == "final":
        from .workflow import final_with_moon

        if args.moon_layer:
            from .workflow import moon_layer
            out, code = _run(moon_layer, args.project, args.base, args.strength, name=args.name,
                             output_file=args.output, quiet=args.quiet)
            if not code:
                print(f"Moon layer written to {out}")
            return code
        join = {"as-stacked": "corona first", "blended": "rebuilt"}[args.moon_edge]
        out, code = _run(final_with_moon, args.project, args.base, args.strength, name=args.name,
                         output_file=args.output, quiet=args.quiet, join=join)
        if not code:
            print(f"Final image written to {out}")
        return code
    if args.command == "prominence-stack":
        return cmd_prominence_stack(args)
    if args.command == "prominences":
        return cmd_prominences(args)
    if args.command == "final-prominences":
        from .workflow import final_with_prominences

        out, code = _run(final_with_prominences, args.project, args.base, args.strength, args.mode,
                         output_file=args.output, base_file=args.base_file, quiet=args.quiet)
        if not code:
            print(f"Image written to {out}")
        return code
    if args.command == "flat-profile":
        return cmd_flat_profile(args)
    return 1


def _db(path: str | None) -> LensfunDatabase:
    return LensfunDatabase.load([path] if path else None)


def cmd_inventory(args) -> int:
    from .inventory import ProjectFolder

    layout = ProjectFolder.scan(args.folder)
    if layout.is_project_folder:
        print(f"Project folder {layout.root}: light frames from {layout.lights.name}/"
              + "".join(f", {k} from {getattr(layout, k).name}/" for k in ("darks", "flats", "bias")
                        if getattr(layout, k)))
        args.folder = str(layout.lights)
        args.bias = args.bias or (str(layout.bias) if layout.bias else None)
        args.darks = args.darks or (str(layout.darks) if layout.darks else None)
        args.flats = args.flats or (str(layout.flats) if layout.flats else None)
        args.project = args.project or str(layout.project_file)
    files = find_raw_files(args.folder, args.recursive)
    if not files:
        print(f"No RAW files found in {args.folder}")
        return 1
    print(f"Reading {len(files)} RAW file(s)...")
    inv = Inventory.from_paths(files, use_exiftool=not args.no_exiftool)

    path = Path(args.project) if args.project else None
    project = EventProject.load(path) if path and path.exists() else EventProject(name=Path(args.folder).name)
    for attr, value in (("eclipse_date", args.eclipse), ("latitude", args.lat), ("longitude", args.lon),
                        ("elevation_m", args.elev), ("camera_utc_offset_h", args.camera_utc_offset),
                        ("clock_error_s", args.clock_error), ("focal_length_mm", args.focal_length),
                        ("aperture_f", args.f_number)):
        if value is not None:
            setattr(project, attr, value or None)
    from .frames import fill_optics
    if fill_optics(inv.frames, project.focal_length_mm, project.aperture_f):
        print("The focal length and f-number given are used for frames whose files do not record them.")
    _contacts(inv, project, args)
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
    fill_optics(calib.flats, project.focal_length_mm, project.aperture_f)
    db = _db(args.lensfun_db)
    forced = VignettingProfile.load(args.profile) if args.profile else None
    library = load_profile_library(args.profile_dir)
    plan = plan_calibration(inv.lights(), db, calib, lens_name=args.lens, profile=forced, profiles=library)
    print()
    print(plan.describe())

    if args.csv:
        inv.to_csv(args.csv)
        print(f"\nFrame table written to {args.csv}")
    if path is not None:
        project.raw_folder = str(Path(args.folder).resolve())
        project.lens_profile = args.lens or project.lens_profile
        project.bias_files = find_raw_files(args.bias) if args.bias else project.bias_files
        project.dark_files = find_raw_files(args.darks) if args.darks else project.dark_files
        project.flat_files = find_raw_files(args.flats) if args.flats else project.flat_files
        project.set_inventory(inv)
        project.save(path)
        print(f"Project saved to {path}")
    return 0


def _six(text: str, option: str, positive: bool = False) -> tuple:
    """Six numbers of 0 or more (more than 0 with ``positive``), finest scale first."""
    try:
        vals = tuple(float(v) for v in text.split(","))
    except ValueError:
        vals = ()
    if len(vals) != 6 or min(vals) < 0 or (positive and min(vals) <= 0):
        raise SystemExit(f"{option} takes six numbers{' above 0' if positive else ' of 0 or more'}, finest scale first")
    return vals


def _contacts(inv, project, args) -> None:
    """Mark frames outside totality: contacts typed in, else computed from the eclipse and the site,
    else the ones saved in the project. All are kept in the camera's clock."""
    from datetime import timedelta

    from .event import check_fit, compute_contacts

    c2 = c3 = None
    if args.c2 and args.c3:
        offset = parse_duration(args.clock_offset) if args.clock_offset else 0.0
        c2 = datetime.fromisoformat(args.c2) - timedelta(seconds=offset)
        c3 = datetime.fromisoformat(args.c3) - timedelta(seconds=offset)
        if offset:
            print(f"Contacts moved into the camera's clock ({-offset:+.0f} s): {c2:%H:%M:%S} to {c3:%H:%M:%S}")
    elif project.latitude is not None and project.longitude is not None:
        try:
            con = compute_contacts(inv.frames, project.latitude, project.longitude, project.elevation_m,
                                   project.camera_utc_offset_h, project.clock_error_s, project.eclipse_date, print)
        except ValueError as exc:
            print(f"Contact times not computed: {exc}")
            return
        print(con.describe())
        fit = check_fit(inv.frames, con.c2_camera, con.c3_camera, con.clock_error_s)
        if not fit.ok:
            for w in fit.warnings:
                print(f"Warning: {w}")
            print("No frames were marked as outside totality; fix the above, or type the contacts with --c2/--c3.")
            return
        c2, c3 = con.c2_camera, con.c3_camera
    else:
        c2, c3 = project.contact_times()
    if c2 is None or c3 is None:
        return
    project.second_contact, project.third_contact = c2.isoformat(timespec="seconds"), c3.isoformat(timespec="seconds")
    changed = inv.mark_outside(c2, c3)
    print(f"{changed} frame(s) marked as outside totality")


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
                                   "totality lensfun-update --from <path to the zip>"))
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
        print("lensfun database not found; run 'totality lensfun-update' first")
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
            print(certificate_help(CANON_URL, "totality elements-update --from <path to the .csv file>"))
        else:
            print(f"Download {CANON_URL} in a browser and run: totality elements-update --from <file>")
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


def _run(step, *a, quiet=False, **k):
    """Run a workflow step with printed progress; WorkflowError -> message and exit code 1."""
    from .workflow import WorkflowError

    try:
        return step(*a, say=None if quiet else print, **k), 0
    except WorkflowError as exc:
        print(exc)
        return None, 1


def cmd_align(args) -> int:
    from .workflow import AlignOptions, align

    opts = AlignOptions(latitude=args.lat, longitude=args.lon, elevation_m=args.elev,
                        camera_utc_offset_h=args.camera_utc_offset, mirrored=args.mirrored,
                        clock_error_s=args.clock_error, rotation=args.rotation, any_angle=args.any_angle,
                        rho=args.rho, window=args.window, output=args.output, profile_dir=args.profile_dir,
                        lensfun_db=args.lensfun_db)
    out, code = _run(align, args.project, opts, quiet=args.quiet)
    if code:
        return code
    result, path = out
    print()
    print(result.describe())
    print(f"\nAlignment saved to {path}")
    return 0


def cmd_stack(args) -> int:
    from .workflow import StackOptions, stack

    radii = args.radii if args.radii is not None else (5.0 if args.square else None)
    opts = StackOptions(align=args.align, output=args.output, name=args.name, radii=radii,
                        moon_frame=args.moon_frame, include_rejected=args.include_rejected,
                        hot_pixels=not args.no_hot_pixels, profile_dir=args.profile_dir, lensfun_db=args.lensfun_db,
                        vignetting=args.vignetting, lens=args.lens, flats_uneven_light=args.flats_uneven_light,
                        darks=args.darks, bias=args.bias)
    out, code = _run(stack, args.project, opts, quiet=args.quiet)
    if code:
        return code
    result, skipped = out
    if args.quiet:
        for msg in skipped:
            print(msg)
    print()
    print(result.describe())
    return 0


def _floats(text):
    return [float(v) for v in text.split(",") if v.strip()] if text else None


# Enhance options and the settings they set (the rest, such as --output, are not settings).
ENHANCE_OPTION_SETTINGS = {
    "layers": ("layers",), "segments": ("segments",), "order": ("order",), "mean_att": ("mean_attenuation",),
    "std_att": ("std_attenuation",), "noise_m": ("noise_m",), "flatten": ("flatten",), "shoulder": ("shoulder",),
    "even": ("even",), "midtone": ("midtone",), "natural_stretch": ("natural_stretch",), "gamma": ("gamma",),
    "gamma_black": ("gamma_black",), "gamma_white": ("gamma_white",), "noise_gate": ("noise_gate",), "adaptive_k": ("mgn_k",),
    "achf_weights": ("achf_weights",), "achf_scales": ("achf_scales",),
    "mgn_scales": ("mgn_scales",), "mgn_weights": ("mgn_weights",), "achf_part_limit": ("achf_part_limit",),
    "achf_contrast": ("achf_contrast",), "saturation": ("saturation",), "white": ("white",),
    "blend": ("blend", "blend_modes"), "keep_background_colour": ("neutral_background",),
}
ENHANCE_NOT_SETTINGS = {"help", "project", "hdr", "output", "name", "format", "quiet", "settings"}


def given_options(parser: argparse.ArgumentParser, argv: list) -> set:
    """The ``dest`` of each option typed in ``argv`` (also abbreviated, as argparse allows)."""
    opts = {s: a.dest for a in parser._actions for s in a.option_strings}
    given = set()
    for tok in argv:
        if not tok.startswith("-") or tok == "-":
            continue
        name = tok.split("=", 1)[0]
        if name in opts:
            given.add(opts[name])
        elif name.startswith("--"):
            hits = {d for s, d in opts.items() if s.startswith(name)}
            if len(hits) == 1:
                given |= hits
    return given


def enhance_params(args, given: set | None = None):
    """The Enhance settings from the command line: from ``--settings`` if given, with every option
    typed on the command line (``given``) taking its place; without a file, the options alone.
    Returns (EnhanceParams, notes)."""
    from dataclasses import replace

    from .enhance import EnhanceParams
    from .workflow import load_enhance_settings, parse_blend, parse_blend_modes, parse_layers

    layers = parse_layers(args.layers)
    blend = parse_blend(args.blend)
    blend_modes = parse_blend_modes(args.blend)
    params = EnhanceParams(layers=layers, segments=args.segments, order=args.order,
                           mean_attenuation=_floats(args.mean_att), std_attenuation=_floats(args.std_att),
                           noise_m=args.noise_m, flatten=args.flatten, shoulder=args.shoulder, even=args.even, midtone=args.midtone, noise_gate=args.noise_gate,
                           natural_stretch=args.natural_stretch, gamma=args.gamma,
                           gamma_black=args.gamma_black, gamma_white=args.gamma_white,
                           mgn_k=args.adaptive_k, achf_weights=_six(args.achf_weights, "--achf-weights"),
                           mgn_scales=_six(args.mgn_scales, "--mgn-scales", positive=True),
                           mgn_weights=_six(args.mgn_weights, "--mgn-weights"),
                           achf_scales=(tuple(ACHF_SCALES) if args.achf_scales is None else
                                        tuple(v / 100.0 for v in _six(args.achf_scales, "--achf-scales", positive=True))),
                           achf_part_limit=args.achf_part_limit, achf_contrast=args.achf_contrast,
                           saturation=args.saturation,
                           white=args.white, blend=blend, blend_modes=blend_modes,
                           neutral_background=not args.keep_background_colour)
    if not args.settings:
        return params, []
    base, notes = load_enhance_settings(args.settings)
    over = {f: getattr(params, f) for d in (given or set()) for f in ENHANCE_OPTION_SETTINGS.get(d, ())}
    if over:
        notes.append("from the command line instead of the file: " + ", ".join(sorted(over)))
    return replace(base, **over), notes


def cmd_enhance(args, given: set | None = None) -> int:
    from .workflow import WorkflowError, enhance_project

    try:
        params, loaded = enhance_params(args, given)
    except WorkflowError as exc:
        print(exc)
        return 1
    if args.settings:
        print(f"Settings from {args.settings}")
        for n in loaded:
            print(f"  {n}")
    out, code = _run(enhance_project, args.project, params, hdr_file=args.hdr, output=args.output,
                     name=args.name, fmt=args.format, quiet=args.quiet)
    if code:
        return code
    result, notes = out
    if args.quiet:
        for n in notes:
            print(n)
    print()
    print(result.describe())
    return 0


def cmd_earthshine(args) -> int:
    from .earthshine import EarthshineParams
    from .workflow import earthshine

    params = EarthshineParams(stops=args.stops, glare_order=args.glare_order, smooth=args.smooth,
                              brightness=args.brightness, limb_glare=not args.no_limb_glare,
                              source_glare=args.prominence_glare and not args.no_prominence_glare,
                              keep_glare=args.keep_glare, limb_soften=args.limb_soften,
                              edge_grain=not args.no_edge_grain, edge_fill=args.edge_fill,
                              colour=not args.no_colour, limb_colour=args.limb_colour.replace("-", " "),
                              edge_smooth=args.edge_smooth, surface_denoise=args.surface_denoise,
                              limb_shape=args.limb_shape and not args.no_limb_shape, channel=args.channel)
    result, code = _run(earthshine, args.project, params, include_rejected=args.include_rejected,
                        output=args.output, name=args.name, profile_dir=args.profile_dir,
                        lensfun_db=args.lensfun_db, fmt=args.format, quiet=args.quiet)
    if code:
        return code
    print()
    print(result.describe())
    return 0


def cmd_prominence_stack(args) -> int:
    from .prominences import ProminenceParams
    from .workflow import ProminenceStackOptions, WorkflowError, prominence_stack, save_prominence_settings

    if not args.settings_only:
        entry, code = _run(prominence_stack, args.project,
                           ProminenceStackOptions(longest_stops=args.longest_stops, radii=args.radii), quiet=args.quiet)
        if code:
            return code
        print(f"Prominence stack: {len(entry['frames'])} frame(s)"
              + (f", blur at the Moon's edge {entry['blur_px']:.2f} px" if entry.get("blur_px") else ""))
    params = ProminenceParams(min_redness=args.min_redness, chromosphere=not args.no_chromosphere,
                              chromosphere_arcsec=args.chromosphere_arcsec, brightness=args.brightness,
                              sharpen=args.sharpen, structure=args.structure)
    try:
        save_prominence_settings(args.project, params, use=not args.off)
    except WorkflowError as exc:
        print(exc)
        return 1
    print("Settings saved for 'totality final'" + (" (prominence stack off)" if args.off else ""))
    return 0


def cmd_prominences(args) -> int:
    from .prominences import ProminenceParams
    from .workflow import prominence_layer

    params = ProminenceParams(height=args.height, sensitivity=args.sensitivity, min_redness=args.min_redness,
                              chromosphere=not args.no_chromosphere, chromosphere_arcsec=args.chromosphere_arcsec,
                              brightness=args.brightness, saturation=args.saturation)
    result, code = _run(prominence_layer, args.project, params, output_file=args.output, quiet=args.quiet)
    if code:
        return code
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
    print(f"lensfun:  {len(db.lenses)} lenses from {len(db.files)} file(s)" if not db.empty else "lensfun:  database not found (run 'totality lensfun-update')")
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
