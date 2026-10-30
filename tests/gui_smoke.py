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
et.run()
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
es.preview_final()
assert es._final_shown
es.write_final()
assert Path(es.final_path).exists() and EventProject.load(path).parameters["final"]["strength"] == 1.5
print("final image:", Path(es.final_path).name)
print("GUI SMOKE OK")
