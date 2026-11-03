import sys, json, numpy as np
from pathlib import Path
T = Path(sys.argv[1]); T.mkdir(exist_ok=True)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "siril"))
import SolarEclipseToolkit as G
from PyQt6.QtWidgets import QMessageBox
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

# Enhance: preview (runs synchronously in the stub), then change a setting, then the full run.
et = win.enhance_tab
et.preview()
print("preview layers:", sorted(et.preview_layers), "binning", et.small.binning if et.small else None)
assert {"natural", "fnrgf_n", "adaptive", "blend"} <= set(et.preview_layers)
et.blend["fnrgf_n"].setValue(40)
et.flatten.setValue(0.6)
assert et.params().blend["fnrgf_n"] == 0.4 and et.params().flatten == 0.6
et.layer_checks["nrgf"].setChecked(False)
et.blend["adaptive"].setValue(10)
et.blend_mode["adaptive"].setCurrentIndex(1)  # Overlay
assert et.params().blend_modes["adaptive"] == "overlay"
os.environ["STUB_SAVE"] = str(T / "out" / "custom_blend.tif")  # the name the user types
et.run()
assert Path(et.outputs["blend"]).name == "custom_blend.tif" and Path(et.outputs["natural"]).name == "custom_natural.tif"
print("written:", sorted(et.outputs))
assert "nrgf" not in et.outputs and "blend" in et.outputs and all(Path(v).exists() for v in et.outputs.values())
assert "enhance" in EventProject.load(path).parameters

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
# Moon and corona together needs the new earthshine outputs: without them, a clear message.
n = len(QMessageBox.log)
es.preview_final()
assert not es._final_shown and len(QMessageBox.log) == n + 1 and "Stack the earthshine again" in QMessageBox.log[-1][1]
# The older way (pasting into a file) still works with just the layer.
os.environ["STUB_OPEN"] = et.outputs["blend"]
es.base.setCurrentIndex(es.base.findData("file"))
es._base_chosen()
es.preview_final()
assert es._final_shown
os.environ["STUB_SAVE"] = str(T / "out" / "my picture")
es.write_final()
assert Path(es.final_path).name == "my picture.tif"
assert Path(es.final_path).exists() and EventProject.load(path).parameters["final"]["strength"] == 1.5
print("final image:", Path(es.final_path).name)
es.base.setCurrentIndex(es.base.findData("blend"))
# Albedo map: Browse fills the path, and the earthshine settings carry it.
os.environ["STUB_OPEN"] = str(T / "moon_map.jpg")
es.albedo_path.setText("")
es._pick_albedo()
assert es.albedo_path.text() == str(T / "moon_map.jpg")
es.albedo_path.setText("")
# From file: a blend saved earlier under another name.
os.environ["STUB_OPEN"] = et.outputs["blend"]
es.base.setCurrentIndex(es.base.findData("file"))
es._base_chosen()
assert es._base_args() == ("blend", et.outputs["blend"])
os.environ["STUB_SAVE"] = str(T / "out" / "from_file_final.tif")
es.write_final()
assert Path(es.final_path).name == "from_file_final.tif"
assert EventProject.load(path).parameters["final"]["base_file"] == et.outputs["blend"]
# Prominences on the real 2024 data: preview, layer as PNG, on top of the blend as a JPG.
pt = win.prominences_tab
pt.preview()
assert pt.region is not None and pt._shown
print("prominences:", pt.log.lines[-1].splitlines()[0])
os.environ["STUB_SAVE"] = str(T / "out" / "proms")  # no extension typed: the chosen type (stub: TIFF)
pt.write_layer()
assert pt.layer_path.endswith(".tif") and Path(pt.layer_path).exists()
pt.base.setCurrentIndex(pt.base.findData("blend"))
pt.preview_final()
assert pt._final_shown
os.environ["STUB_SAVE"] = str(T / "out" / "with_proms.jpg")
pt.write_final()
assert pt.final_path.endswith("with_proms.jpg") and Path(pt.final_path).exists()
# The enhance layers as PNG.
os.environ["STUB_SAVE"] = str(T / "out" / "pngs.png")
et.run()
assert et.outputs["blend"].endswith("pngs_blend.png")
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
            "frame table": ft.table, "Lens profile": ft.lens_combo, "Eclipse": ft.eclipse_combo,
            "Latitude": ft.lat, "Longitude": ft.lon, "Elevation": ft.elev, "Camera time zone": ft.tz,
            "Clock error": ft.clock_error, "Compute contacts": ft.compute_btn,
            "Second contact": ft.c2, "Third contact": ft.c3, "Bias": ft.bias_dir, "Darks": ft.dark_dir,
            "Flats": ft.flat_dir, "report": ft.report, **{f"{k} button": b for k, b in ft.setting_buttons.items()},
            **{f"folder button {i}": b for i, b in enumerate(ft.calib_browse)}}
missing = [k for k, w in controls.items() if len(w.toolTip()) < 30]
print("Frames tab controls without hover help:", missing or "none")
assert not missing

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

# An older project: contacts typed in true time plus a clock offset move into the camera clock.
old = EventProject(name="old", second_contact="2024-04-08T13:56:37", third_contact="2024-04-08T14:00:48",
                   camera_clock_offset_s=3600.0)
old.save(T / "old.json")
o = EventProject.load(T / "old.json")
assert (o.second_contact, o.camera_clock_offset_s) == ("2024-04-08T12:56:37", 0.0)

print("GUI SMOKE OK")
