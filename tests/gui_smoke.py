import sys, json, numpy as np
from pathlib import Path
T = Path(sys.argv[1]); T.mkdir(exist_ok=True)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "siril"))
import Totality as G
# The window title: name, version and the commit, so the user can see which code is running.
import re as _re
assert _re.fullmatch(r"Totality - A Solar Eclipse Toolkit \d+\.\d+\.\d+( \([0-9a-f]{7}\))?", G.window_title()), G.window_title()
from PyQt6.QtWidgets import QFileDialog, QMessageBox
from tse_core.fits import write_fits
from tse_core.project import EventProject
from tse_core.inventory import Inventory
sys.path.insert(0, str(Path(__file__).resolve().parent))
from conftest import make_frames

# A project with a finished stack (the real 2024 fixture) and a fake alignment file.
z = np.load(Path(__file__).resolve().parent / "data" / "eclipse2024_bin8.npz")
hdr = z["hdr"].astype(np.float32) * float(z["scale"])
geo = {k: float(z[k]) for k in ("SUNX", "SUNY", "SUNR", "MOONX", "MOONY", "MOONR")}
write_fits(T / "e_hdr.fit", hdr, geo)
write_fits(T / "e_weight.fit", (np.abs(hdr).sum(0) > 0).astype(np.float32))
p = EventProject(name="e")
p.set_inventory(Inventory(make_frames(["1/250", "1/60", "1/15"])))
align = {"reference_index": 1, "image_size": [1036, 690], "frames": [
    {"index": i, "file_name": f"DSC_{i:04d}.NEF", "exposure_text": e, "shift_px": [0.1 * i, -0.2], "angle_deg": 0.0,
     "psr": None if i == 1 else 500.0, "notes": [], "matrix": [[1, 0, 0], [0, 1, 0]], "moon_ref": [1, 1],
     "moon_frame": {"radius": 50}} for i, e in enumerate(["1/250", "1/60", "1/15"])],
    "sun": {"text": "Sun centre (495.0, 348.3)", "rejected": []}, "notes": []}
(T / "e_align.json").write_text(json.dumps(align))
p.parameters["align"] = {"file": str(T / "e_align.json"), "rotation": "drift"}
p.parameters["stack"] = {"hdr": str(T / "e_hdr.fit"), "weight": str(T / "e_weight.fit"), "radii": None,
                         "moon_frame": 1}
path = T / "e.json"
p.save(path)

link = G.SirilLink()
win = G.MainWindow(link)
# Open the project as the menu would.
import os
os.environ["STUB_OPEN"] = str(path)
win.open_project()
assert win.frames_tab.project_path == path, win.frames_tab.project_path
print("align table rows:", win.align_tab.table.rowCount(), "| summary:", win.align_tab.summary.lines[0][:40])
assert win.align_tab.table.rowCount() == 3
print("stack moon-frame choices:", win.stack_tab.moon_frame.count(), "current", win.stack_tab.moon_frame.currentData())
assert win.stack_tab.moon_frame.currentData() == 1
# Calibration on the Stack tab (Andy, Nov 2026): darks off and hot pixels on by default; using darks
# unticks and greys out the hot-pixel box; not using them gives it back.
stt = win.stack_tab
assert stt.darks.currentData() == "none" and stt.hot.isChecked() and stt.hot.isEnabled()
assert stt.vignetting.currentData() == "auto" and stt.bias.currentData() == "use"
stt.darks.setCurrentIndex(stt.darks.findData("use"))
assert not stt.hot.isChecked() and not stt.hot.isEnabled()
stt.darks.setCurrentIndex(stt.darks.findData("none"))
assert stt.hot.isChecked() and stt.hot.isEnabled()
stt.vignetting.setCurrentIndex(stt.vignetting.findData("none"))
assert not stt.lens.isEnabled() and not stt.uneven_light.isEnabled()
stt.vignetting.setCurrentIndex(stt.vignetting.findData("auto"))
for w in (stt.vignetting, stt.lens, stt.uneven_light, stt.darks, stt.hot, stt.bias):
    assert w.toolTip(), w

# Enhance: preview (runs synchronously in the stub), then change a setting, then the full run.
et = win.enhance_tab
et.preview()
print("preview layers:", sorted(et.preview_layers), "binning", et.small.binning if et.small else None)
assert {"natural", "fnrgf_n", "adaptive", "blend"} <= set(et.preview_layers)
# Defaults (Andy, 4 Nov 2026): every option off; adaptive strength the paper's k = 0.7 in 0.01 steps.
d = et.params()
assert (d.flatten, d.shoulder, d.even, d.midtone, d.noise_gate, d.mgn_k) == (0.0, 0.0, 0.0, 0.0, 0.0, 0.7), d
assert d.blend == {"natural": 1.0}, d.blend  # every filter layer at 0 %
# Adaptive (ACHF): six per-scale strengths, all 1 by default, each with hover help; ticked off.
from tse_core.enhance import ACHF_SCALES
assert len(et.achf_w) == 6 and d.achf_weights == (1.0, 1.0, 1.0, 1.0, 0.0, 0.0), d.achf_weights
# Sizes are fractions of the Sun's radius whatever the open stack's Sun; shown in its pixels.
assert tuple(d.achf_scales) == tuple(ACHF_SCALES), d.achf_scales  # not rounded through the px display
assert abs(et.achf_size[1].value() - ACHF_SCALES[1] * et.sun_r_full) < 0.01
assert all(len(sz.toolTip()) > 100 for sz in et.achf_size)
assert d.achf_part_limit == 2.0 and len(et.achf_parts.toolTip()) > 100
et.achf_size[2].setValue(6.0)  # typed in px for this stack: kept as a fraction of its Sun's radius
assert abs(et.params().achf_scales[2] - 6.0 / et.sun_r_full) < 1e-12
et.achf_size[2].setValue(round(ACHF_SCALES[2] * et.sun_r_full, 2)); et.achf_frac[2] = ACHF_SCALES[2]
assert all(len(sp.toolTip()) > 100 for sp in et.achf_w)
assert not et.layer_checks["achf"].isChecked() and "achf" not in d.layers
et.achf_w[0].setValue(0.4)
assert et.params().achf_weights[0] == 0.4
et.achf_w[0].setValue(1.0)
assert "strands" not in vars(et) and "skip_fine" not in vars(et)  # (the stand-in answers any hasattr)
assert "strands" not in d.__dataclass_fields__ and "mgn_skip_fine" not in d.__dataclass_fields__
et.blend["fnrgf_n"].setValue(40)
et.flatten.setValue(0.6)
assert et.params().blend["fnrgf_n"] == 0.4 and et.params().flatten == 0.6
assert et.params().natural_stretch == "log" and et.params().gamma == 3.2   # Stretch and Gamma
et.natural_stretch.setCurrentIndex(1)
et.gamma.setValue(2.5)
assert et.params().natural_stretch == "gamma" and et.params().gamma == 2.5
et.natural_stretch.setCurrentIndex(0)
m = et.blend_mode["adaptive"]
assert m.findData("average") >= 0                       # Weighted average: MGN only
assert et.blend_mode["achf"].findData("average") >= 0      # and ACHF (its earlier look)
assert all(et.blend_mode[k].findData("average") < 0 for k in et.blend_mode if k not in ("adaptive", "achf"))
ma = et.blend_mode["achf"]
ma.setCurrentIndex(ma.findData("linear"))                # the thesis's form: K2 up to 100
et.blend["achf"].setValue(5000)
assert et.blend["achf"].value() == 5000 and et.params().blend["achf"] == 50.0
assert et.params().blend_modes["achf"] == "linear"
ma.setCurrentIndex(ma.findData("soft light"))
et.blend["achf"].setValue(0)
et.blend["adaptive"].setValue(43)              # a mode is kept only for a layer in use
m.setCurrentIndex(m.findData("average"))
assert m.currentText() == "Weighted average" and et.params().blend_modes.get("adaptive") == "average"
m.setCurrentIndex(m.findData("soft light"))
et.layer_checks["nrgf"].setChecked(False)
et.blend["adaptive"].setValue(10)
et.blend_mode["adaptive"].setCurrentIndex(1)  # Overlay
assert et.params().blend_modes["adaptive"] == "overlay"
et.blend_mode["adaptive"].setCurrentIndex(2)  # Linear: as each method's publication combines
assert et.params().blend_modes["adaptive"] == "linear" and et.blend_mode["adaptive"].currentText() == "Linear"
et.blend["adaptive"].setValue(250)  # K2 = 2.5: above 100 % in Linear mode
assert et.params().blend["adaptive"] == 2.5
et.blend["adaptive"].setValue(10)
et.blend_mode["adaptive"].setCurrentIndex(1)
os.environ["STUB_SAVE"] = str(T / "out" / "custom_blend.tif")  # the name the user types
et.run()
assert Path(et.outputs["blend"]).name == "custom_blend.tif" and Path(et.outputs["natural"]).name == "custom_natural.tif"
print("written:", sorted(et.outputs))
assert "nrgf" not in et.outputs and "blend" in et.outputs and all(Path(v).exists() for v in et.outputs.values())
assert "enhance" in EventProject.load(path).parameters
# Changing an ACHF setting redraws only ACHF (and the blend mixes again); the reduced stack is
# prepared once and kept until a setting it depends on (Keep the sky, White balance) changes.
et._view_changed(0)
et.blend["achf"].setValue(100)
et.preview()
small_frame = et.small_frame
assert small_frame is not None
mark = len(et.log.toPlainText())
et.achf_parts.setValue(2.5)
et.preview()
new_log = et.log.toPlainText()[mark:]
assert "layer achf" in new_log and "layer achf (unchanged" not in new_log, new_log
for k in ("natural", "fnrgf_n", "adaptive"):
    assert f"layer {k} (unchanged)" in new_log, (k, new_log)
mark = len(et.log.toPlainText())
et.achf_contrast.setValue(1.6)  # only rescales: nothing filtered again
et.preview()
assert "layer achf (unchanged; Contrast applied)" in et.log.toPlainText()[mark:]
et.achf_parts.setValue(2.0)
assert et.small_frame is small_frame
et.keep_sky.setChecked(not et.keep_sky.isChecked())
et.preview()
assert et.small_frame is not small_frame
et.keep_sky.setChecked(not et.keep_sky.isChecked())
et.achf_contrast.setValue(1.0)
et.blend["achf"].setValue(0)
et.preview()

# Adaptive (MGN): the paper's six scales and weights, adjustable.
assert et.params().mgn_scales == (1.25, 2.5, 5.0, 10.0, 20.0, 40.0) and et.params().mgn_weights == (1.0,) * 6
assert all(len(w.toolTip()) > 100 for w in et.mgn_size + et.mgn_w)
et.mgn_size[0].setValue(2.0)
et.mgn_w[5].setValue(0.0)
assert et.params().mgn_scales[0] == 2.0 and et.params().mgn_weights[5] == 0.0
et.preview()
et.mgn_size[0].setValue(1.25)
et.mgn_w[5].setValue(1.0)

# Save settings... and Load settings...: every control comes back as saved.
import os
from dataclasses import replace as _replace
saved_params = et.params()
_stub_env = {k: os.environ.get(k) for k in ("STUB_SAVE", "STUB_OPEN")}
os.environ["STUB_SAVE"] = str(Path(sys.argv[1]) / "look")
et.save_settings()
assert (Path(sys.argv[1]) / "look.json").exists()
other = _replace(saved_params, mgn_scales=(2.0, 3.0, 6.0, 12.0, 24.0, 48.0), mgn_weights=(0.5, 1, 1, 1, 1, 0),
                 flatten=0.15, saturation=0.4, white="none", neutral_background=not saved_params.neutral_background,
                 achf_contrast=2.2, blend={"natural": 1.0, "nrgf": 0.3}, blend_modes={"nrgf": "overlay"},
                 achf_scales=tuple(v * 1.5 for v in saved_params.achf_scales), layers=("natural",))
assert et.set_params(other) == [] and et.params() == other  # (the stand-in's controls take any value)
os.environ["STUB_OPEN"] = str(Path(sys.argv[1]) / "look.json")
et.load_settings()
assert et.params() == saved_params, (et.params(), saved_params)
assert "Settings loaded from" in et.log.toPlainText()
# The settings beside the layers written above load the same way.
info = G.W.resolve(path, EventProject.load(path).parameters["enhance"]["info"])
os.environ["STUB_OPEN"] = info
et.set_params(other)
et.load_settings()
assert f"Settings loaded from {info}" in et.log.toPlainText()
assert et.params().flatten == json.loads(Path(info).read_text())["params"]["flatten"] != other.flatten
os.environ["STUB_OPEN"] = ""
before = et.params()
et.load_settings()  # no file chosen: nothing happens
assert et.params() == before
for k, v in _stub_env.items():  # as the rest of this script expects
    if v is None:
        os.environ.pop(k, None)
    else:
        os.environ[k] = v
assert len(et.save_settings_btn.toolTip()) > 100 and len(et.load_settings_btn.toolTip()) > 100

# Detail box: Shift-drag on the reduced picture (here the signal the view sends); the box is rendered
# at full resolution and is exactly the same pixels as the full-size blend just written.
import tifffile
assert len(et.achf_contrast.toolTip()) > 100 and et.params().achf_contrast == 1.0
b = et.small.binning
et.view.box_drawn.emit((20, 30, 120, 90))  # x0, y0, x1, y1 in preview px
assert et.detail_mode and et.box == (30 * b, 90 * b, 20 * b, 120 * b), et.box
assert EventProject.load(path).parameters["enhance_box"] == list(et.box)  # kept with the project
written = tifffile.imread(et.outputs["blend"]).astype(np.float64) / 65535.0
y0, y1, x0, x1 = et.box
diff = np.abs(et.detail_layers["blend"] - written[y0:y1, x0:x1]).max()
print("detail box", et.box, "largest difference from the written blend:", diff)
assert diff < 2 / 65535, diff  # the 16-bit rounding of the file only
frame = et.frame
et.flatten.setValue(0.7)
et.preview()  # (the stand-in's timer does not fire)
assert et.frame is frame and et.params().flatten == 0.7  # the full-size stack is not read again
et.white.setCurrentIndex(1)
et.preview()
assert et.frame is not frame  # White balance changes what is measured over the whole stack
et.white.setCurrentIndex(0)
et.flatten.setValue(0.6)
et._view_changed(0)
assert not et.detail_mode

# Align with nothing to align (no real RAW files): a clear message, no crash.
win.align_tab.lat.setText("abc")
win.align_tab.run()
print("messages:", [m[1][:60] for m in QMessageBox.log])
win.align_tab.lat.setText("36.94")
win.align_tab.run()
print("last message:", QMessageBox.log[-1][1][:120])
print("buttons re-enabled:", all(b._enabled for b in win.align_tab.run_buttons))
# Earthshine without a stack brightness file: a clear message.
n = len(QMessageBox.log)
win.earthshine_tab.run()
print("earthshine message:", QMessageBox.log[-1][1][:80])
assert len(QMessageBox.log) == n + 1 and "Stack step" in QMessageBox.log[-1][1]
# Final image: needs an earthshine layer; with a fake one the size of the blend, preview and write work.
import tifffile
blend = tifffile.imread(et.outputs["blend"])
fake = np.zeros_like(blend)
tifffile.imwrite(T / "e_earthshine.tif", fake, photometric="rgb")
pp = EventProject.load(path)
pp.parameters["earthshine"] = {"earthshine": str(T / "e_earthshine.tif")}
pp.save(path)
win.reload_project()
es = win.earthshine_tab
es.strength.setValue(1.5)
# Write Moon layer: the Moon as Blend draws it, for Photoshop (Normal mode over the blend).
assert es.moon_btn in es.run_buttons and "Normal mode at 100 %" in es.moon_btn.toolTip()
assert es.edge_fill.value() == 0.0  # Filled edge, at least: % of the Moon's radius
# Moon and corona together needs the new earthshine outputs: without them, a clear message.
n = len(QMessageBox.log)
es.preview_final()
assert not es._final_shown and len(QMessageBox.log) == n + 1 and "Stack the earthshine again" in QMessageBox.log[-1][1]
# Corona image: Blend or Natural only (pasting into a file was removed, Nov 2026); Moon's edge named
# for the user, with hover help, and the older way still available.
assert es.base.findData("file") < 0 and es.base.findData("blend") >= 0 and es.base.findData("natural") >= 0
assert es.join.findData("corona first") == 0 and es.join.findData("rebuilt") == 1
assert es.join.itemText(0) == "Corona as stacked (recommended)" and "older" in es.join.itemText(1)
assert "Corona as stacked" in es.join.toolTip() and "Corona blended into the Moon" in es.join.toolTip()
# Prominences on the real 2024 data. No RAW files here to stack the short frames, so the HDR stack
# stands in for the prominence stack (the swap's mechanics, settings saved, the final image written).
pt = win.prominences_tab
pp = EventProject.load(path)
st = pp.parameters["stack"]
pp.parameters["prominence_stack"] = {"hdr": st["hdr"], "weight": st["weight"], "frames": [1, 2, 3], "blur_px": 1.4,
                                     "longest_stops": 2.0}
pp.save(path)
win.reload_project()
assert "3 frame(s)" in pt.stack_label.text() and "1.40 px" in pt.stack_label.text()
assert es.use_prom.isEnabled() and es.use_prom.isChecked()     # on the Earthshine tab
es.use_prom.setChecked(False)
assert win.frames_tab.project.parameters["prominence_stack"]["use"] is False
es.use_prom.setChecked(True)
pt.brightness.setValue(0.3)
pt.preview()
assert pt._shown
print("prominences:", pt.log.lines[-1].splitlines()[0])
os.environ["STUB_SAVE"] = str(T / "out" / "with_proms.jpg")
pt.write_final()
# Each step suggests its own name (Andy's list item 5), in the format chosen last (TIFF so far).
assert Path(QFileDialog.last_save[1]).name == f"{Path(path).stem}_earthshine_blend.tif", QFileDialog.last_save
assert "*.tif" in QFileDialog.last_save[3]
# No RAW files here, so no earthshine stack: writing hands over to the Earthshine tab, which asks for it
# (the merge itself is tested in tests/test_earthshine.py).
assert QMessageBox.log[-1][0] == "warn" and "Stack the earthshine" in QMessageBox.log[-1][1]
assert EventProject.load(path).parameters["prominences"]["params"]["brightness"] == 0.3
# The enhance layers as PNG.
os.environ["STUB_SAVE"] = str(T / "out" / "pngs.png")
et.run()
assert et.outputs["blend"].endswith("pngs_blend.png")
assert QFileDialog.last_save[1].endswith(".jpg") and "*.jpg" in QFileDialog.last_save[3]  # JPG chosen last
assert EventProject.load(path).parameters["image_ext"] == ".png"
os.environ["STUB_SAVE"] = ""  # cancelled: only the suggestions are checked
es.write_moon_layer()
assert Path(QFileDialog.last_save[1]).name == f"{Path(path).stem}_earthshine_photoshop.png"
es.base.setCurrentIndex(es.base.findData("natural"))
es.write_final()
assert Path(QFileDialog.last_save[1]).name == f"{Path(path).stem}_earthshine_natural.png"
es.base.setCurrentIndex(es.base.findData("blend"))
es.run()
assert Path(QFileDialog.last_save[1]).name == f"{Path(path).stem}_earthshine_stack.png"
# Opening a project folder with Browse... on the Frames tab: every tab shows that project (the Align
# tab's site came up blank because only the Frames tab was told; seen by Andy, Nov 2026).
root = T / "Folder project"
(root / "lights").mkdir(parents=True)
fp = EventProject(name="Folder project")
fl = make_frames(["1/250", "1/60"])
for f in fl:
    f.path = str(root / "lights" / f.file_name)
    Path(f.path).write_bytes(b"")
fp.set_inventory(Inventory(fl))
fp.latitude, fp.longitude, fp.elevation_m, fp.camera_utc_offset_h = 36.942202, -90.334287, 110.0, -5.0
fp.raw_folder = str(root / "lights")
(root / "Folder project_align.json").write_text(json.dumps({**align, "frames": align["frames"][:2]}))
fp.parameters["align"] = {"file": str(root / "Folder project_align.json"), "rotation": "per-frame"}
fp.save(root / "Folder project.json")
win4 = G.MainWindow(G.SirilLink())
os.environ["STUB_DIR"] = str(root)
win4.frames_tab._pick_folder(win4.frames_tab.folder)
at, ft4 = win4.align_tab, win4.frames_tab
print("Browse... on a project folder: site", ft4.lat.text(), ft4.lon.text(), ft4.elev.text(), ft4.tz.text(),
      "| alignment rows", at.table.rowCount())
assert (ft4.lat.text(), ft4.lon.text(), ft4.elev.text(), ft4.tz.text()) == ("36.942202", "-90.334287", "110", "-5")
assert "36.942202, -90.334287" in at.event_label.text()  # the Align tab shows the event from the Frames tab
assert at.table.rowCount() == 2 and at.rotation.currentText() == "per-frame"
os.environ.pop("STUB_DIR")

# Every control on the Frames tab has hover help (CLAUDE.md: settings shown in the GUI need it).
ft = win4.frames_tab
controls = {"RAW folder": ft.folder, "Browse": ft.browse_btn, "exiftool": ft.use_exiftool, "Read frames": ft.scan_btn,
            "frame table": ft.table, "Eclipse": ft.eclipse_combo,
            "Latitude": ft.lat, "Longitude": ft.lon, "Elevation": ft.elev, "Camera time zone": ft.tz,
            "Clock error": ft.clock_error, "Compute contacts": ft.compute_btn,
            "Second contact": ft.c2, "Third contact": ft.c3, "Bias": ft.bias_dir, "Darks": ft.dark_dir,
            "Focal length": ft.focal, "f-number": ft.fnum,
            "Flats": ft.flat_dir, "report": ft.report, **{f"{k} button": b for k, b in ft.setting_buttons.items()},
            **{f"folder button {i}": b for i, b in enumerate(ft.calib_browse)}}
missing = [k for k, w in controls.items() if len(w.toolTip()) < 30]
print("Frames tab controls without hover help:", missing or "none")
assert not missing

# Focal length and f-number typed for frames whose files do not record them (a telescope): kept in
# the project, used only where a file leaves them out.
for f in ft.inventory.frames:
    f.focal_length_mm = f.aperture_f = None
ft.focal.setText("1000")
ft.fnum.setText("10")
assert ft._store_event()
assert (ft.project.focal_length_mm, ft.project.aperture_f) == (1000.0, 10.0)
assert {(f.focal_length_mm, f.aperture_f) for f in ft.inventory.frames} == {(1000.0, 10.0)}
ft.focal.setText("")
ft.fnum.setText("")
assert ft._store_event() and ft.project.focal_length_mm is None

# Event: contacts computed from the eclipse and site (2024, at the site above, camera on UTC-5).
from datetime import datetime as _dt
QMessageBox.log.clear()
inside = make_frames(["1/250", "1/60", "1/15"], start=_dt(2024, 4, 8, 13, 58, 0), first_index=10)
early = make_frames(["1/250", "1/60"], start=_dt(2024, 4, 8, 13, 40, 0), first_index=20)  # partial phase
ft4.inventory = Inventory(early + inside)
ft4.project.set_inventory(ft4.inventory)
ft4.inventory = ft4.project.inventory
ft4.c2.setText(""); ft4.c3.setText("")
ft4.compute_contacts()
print("computed contacts:", ft4.c2.text(), ft4.c3.text())
assert ft4.c2.text() == "2024-04-08T13:56:37" and ft4.c3.text() == "2024-04-08T14:00:48"
roles = {f.index: f.role for f in ft4.inventory.frames}
assert [roles[i] for i in (20, 21)] == ["partial", "partial"] and all(roles[i] == "light" for i in (10, 11, 12))
assert not any(k == "question" for k, _ in QMessageBox.log)  # fits: nothing to ask
# A bracket cut by second contact (clock 30 s fast): asked first, nothing marked when the answer is No.
cut = make_frames(["1/250", "1/60", "1/15", "1/4", "1"], start=_dt(2024, 4, 8, 13, 56, 33), first_index=30)
ft4.inventory = Inventory(cut)
ft4.project.set_inventory(ft4.inventory)
ft4.inventory = ft4.project.inventory
ft4.compute_contacts()
asked = [t for k, t in QMessageBox.log if k == "question"]
print("asked:", asked[-1].splitlines()[0][:100] if asked else None)
assert asked and "clock error" in asked[-1] and all(f.role == "light" for f in ft4.inventory.frames)
assert "no frames were marked" in ft4.report.toPlainText()
# Saved for the steps, and shown on the Align tab.
win4.ensure_saved()
saved = EventProject.load(root / "Folder project.json")
assert saved.latitude == 36.942202 and saved.second_contact == "2024-04-08T13:56:37"

# A flats folder chosen with "..." reaches the project even without Apply (Stack reads it from there).
(root / "myflats").mkdir()
for k in range(3):
    (root / "myflats" / f"FLAT_{k}.NEF").write_bytes(b"")
ft4.flat_dir.setText(str(root / "myflats"))
win4.ensure_saved()
assert len(EventProject.load(root / "Folder project.json").flat_files) == 3

# An older project: contacts typed in true time plus a clock offset move into the camera clock.
old = EventProject(name="old", second_contact="2024-04-08T13:56:37", third_contact="2024-04-08T14:00:48",
                   camera_clock_offset_s=3600.0)
old.save(T / "old.json")
o = EventProject.load(T / "old.json")
assert (o.second_contact, o.camera_clock_offset_s) == ("2024-04-08T12:56:37", 0.0)

print("GUI SMOKE OK")
