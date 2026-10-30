"""Smoke test of the Siril GUI script's logic.

PyQt6 is replaced by a small stand-in (tests/qtstub) that keeps the state of
the widgets the logic reads and runs background threads synchronously. It
checks every tab's code path end to end (open a project, align, stack
results, enhance preview, full-size layers, error messages); it does not
check Qt layout or rendering, which needs a real run in Siril.
"""

import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def test_gui_logic_with_qt_stand_in(tmp_path):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(HERE / "qtstub"), str(HERE.parent / "src")])
    out = subprocess.run([sys.executable, str(HERE / "gui_smoke.py"), str(tmp_path)], env=env,
                         capture_output=True, text=True, timeout=600, cwd=str(HERE.parent))
    assert out.returncode == 0, out.stdout[-3000:] + out.stderr[-3000:]
    assert "GUI SMOKE OK" in out.stdout
