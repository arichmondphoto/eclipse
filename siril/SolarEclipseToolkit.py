# SPDX-License-Identifier: MIT
"""Solar Eclipse Toolkit: Siril script entry point.

Run from Siril (Scripts menu, Siril 1.4 or later) or on its own with any
Python 3.10+ that has PyQt6:

    python SolarEclipseToolkit.py            # GUI
    python SolarEclipseToolkit.py --help     # command-line options (same as 'tse')

Four tabs, one per step, each saving into the event's project file:

1. Frames: read the RAW folder, review the inventory and brackets, choose
   the lens profile, add optional calibration frames.
2. Align: site and clock, then phase-correlation alignment and the Sun's centre.
3. Stack: the Moon-aware linear HDR stack (full frame or a square).
4. Enhance: natural, NRGF, FNRGF, FNRGF-N and adaptive layers with a live,
   reduced-size preview, then the full-size 16-bit TIFF layers.

Long steps run in the background with progress in the log and a Cancel button.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

VERSION = "0.1.0"

# The core package lives in ../src when the script runs from the repository.
_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# --------------------------------------------------------------- Siril link
try:
    import sirilpy as s

    for _pkg in ("PyQt6", "numpy", "scipy", "exifread", "rawpy", "tifffile", "certifi", "truststore"):
        try:
            s.ensure_installed(_pkg)
        except Exception as exc:  # report and keep going; a missing reader shows up when it is needed
            print(f"Could not install {_pkg}: {exc}")
    HAVE_SIRIL = True
except ImportError:
    s = None
    HAVE_SIRIL = False

from datetime import datetime  # noqa: E402

from PyQt6.QtCore import QThread, QTimer, Qt, pyqtSignal  # noqa: E402
from PyQt6.QtGui import QAction, QFont, QImage, QPixmap  # noqa: E402
from PyQt6.QtWidgets import (  # noqa: E402
    QAbstractItemView, QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMainWindow, QMenu,
    QMessageBox, QPlainTextEdit, QPushButton, QRadioButton, QScrollArea, QSizePolicy, QSpinBox,
    QSplitter, QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget,
)

from tse_core import workflow as W  # noqa: E402
from tse_core.brackets import detect_brackets  # noqa: E402
from tse_core.enhance import LAYERS, EnhanceParams  # noqa: E402
from tse_core.calibration import CalibrationFrames, plan_calibration, write_synthetic_flat  # noqa: E402
from tse_core.frames import format_exposure, parse_duration  # noqa: E402
from tse_core.inventory import Inventory, find_raw_files  # noqa: E402
from tse_core.lensfun import LensfunDatabase, download_database  # noqa: E402
from tse_core.metadata import read_frames  # noqa: E402
from tse_core.project import EventProject  # noqa: E402
from tse_core.vignetting_profile import load_profile_library  # noqa: E402

COLUMNS = ["#", "File", "Time", "Exposure", "f/", "ISO", "Focal", "Black", "White", "Bracket", "Role", "Notes"]
AUTOMATIC = "Automatic (from EXIF)"
NO_PROFILE = "None (no vignetting correction)"


class SirilLink:
    """Thin wrapper so the GUI also runs without Siril."""

    def __init__(self) -> None:
        self.siril = None
        if HAVE_SIRIL:
            try:
                self.siril = s.SirilInterface()
                self.siril.connect()
                self.siril.cmd("requires", "1.4.0")
            except Exception as exc:
                print(f"Not connected to Siril: {exc}")
                self.siril = None

    @property
    def connected(self) -> bool:
        return self.siril is not None

    def log(self, text: str) -> None:
        if self.siril:
            try:
                self.siril.log(text[:1000])
                return
            except Exception:
                pass
        print(text)

    def working_dir(self) -> str:
        if self.siril:
            try:
                return self.siril.get_siril_wd()
            except Exception:
                pass
        return str(Path.cwd())


class FramesTab(QWidget):
    def __init__(self, link: SirilLink, parent=None) -> None:
        super().__init__(parent)
        self.link = link
        self.inventory = Inventory()
        self.project = EventProject()
        self.project_path: Path | None = None
        self.db = LensfunDatabase.load()
        self.plan = None
        self._build()

    # ---------------------------------------------------------------- layout
    def _build(self) -> None:
        layout = QVBoxLayout(self)

        row = QHBoxLayout()
        self.folder = QLineEdit()
        self.folder.setPlaceholderText("Folder of RAW files from totality")
        browse = QPushButton("Browse…")
        browse.clicked.connect(lambda: self._pick_folder(self.folder))
        self.use_exiftool = QCheckBox("Use exiftool if installed")
        self.use_exiftool.setChecked(True)
        scan = QPushButton("Read frames")
        scan.clicked.connect(self.scan)
        row.addWidget(QLabel("RAW folder:"))
        row.addWidget(self.folder, 1)
        row.addWidget(browse)
        row.addWidget(self.use_exiftool)
        row.addWidget(scan)
        layout.addLayout(row)

        splitter = QSplitter(Qt.Orientation.Vertical)
        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._table_menu)
        splitter.addWidget(self.table)

        bottom = QWidget()
        bl = QHBoxLayout(bottom)
        bl.setContentsMargins(0, 0, 0, 0)

        options = QGroupBox("Event settings")
        form = QFormLayout(options)
        self.lens_combo = QComboBox()
        self.lens_combo.addItems([AUTOMATIC, NO_PROFILE])
        self.lens_combo.currentIndexChanged.connect(lambda _: self.replan())
        form.addRow("Lens profile:", self.lens_combo)
        self.c2 = QLineEdit()
        self.c2.setPlaceholderText("e.g. 2024-04-08T13:47:33")
        self.c3 = QLineEdit()
        self.c3.setPlaceholderText("optional")
        self.clock = QLineEdit()
        self.clock.setPlaceholderText("added to camera time, e.g. +1:04:08 (blank = clock was right)")
        form.addRow("Clock offset:", self.clock)
        form.addRow("Second contact:", self.c2)
        form.addRow("Third contact:", self.c3)
        self.bias_dir, self.dark_dir, self.flat_dir = QLineEdit(), QLineEdit(), QLineEdit()
        for label, edit in (("Bias folder:", self.bias_dir), ("Darks folder:", self.dark_dir), ("Flats folder:", self.flat_dir)):
            edit.setPlaceholderText("optional")
            box = QHBoxLayout()
            box.addWidget(edit, 1)
            b = QPushButton("…")
            b.setFixedWidth(28)
            b.clicked.connect(lambda _=False, e=edit: self._pick_folder(e))
            box.addWidget(b)
            form.addRow(label, box)
        buttons = QHBoxLayout()
        for text, slot in (
            ("Apply", self.apply_settings),
            ("Write synthetic flat", self.write_flat),
            ("Update lensfun", self.update_lensfun),
        ):
            b = QPushButton(text)
            b.clicked.connect(slot)
            buttons.addWidget(b)
        form.addRow(buttons)
        bl.addWidget(options, 0)

        self.report = QPlainTextEdit()
        self.report.setReadOnly(True)
        font = QFont("Monospace")
        font.setStyleHint(QFont.StyleHint.TypeWriter)
        self.report.setFont(font)
        bl.addWidget(self.report, 1)
        splitter.addWidget(bottom)
        splitter.setSizes([420, 260])
        layout.addWidget(splitter, 1)

        if self.db.empty:
            self.report.setPlainText(
                "The lensfun database was not found.\nClick 'Update lensfun' to download it (about 15 MB)."
            )

    # --------------------------------------------------------------- actions
    def _pick_folder(self, edit: QLineEdit) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Choose folder", edit.text() or self.link.working_dir())
        if folder:
            edit.setText(folder)

    def scan(self) -> None:
        folder = self.folder.text().strip()
        files = find_raw_files(folder) if folder else []
        if not files:
            QMessageBox.warning(self, "No RAW files", f"No RAW files were found in:\n{folder}")
            return
        self.link.log(f"Solar Eclipse Toolkit: reading {len(files)} RAW files")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            self.inventory = Inventory.from_paths(files, use_exiftool=self.use_exiftool.isChecked())
            self.project.set_inventory(self.inventory)
            self.inventory = self.project.inventory  # keeps earlier overrides
            self.project.raw_folder = folder
            if not self.project.name:
                self.project.name = Path(folder).name
        finally:
            QApplication.restoreOverrideCursor()
        self._refresh_lens_choices()
        self.apply_settings()

    def apply_settings(self) -> None:
        if not self.inventory.frames:
            return
        c2, c3 = self.c2.text().strip(), self.c3.text().strip()
        try:
            self.project.camera_clock_offset_s = parse_duration(self.clock.text()) if self.clock.text().strip() else 0.0
        except ValueError:
            QMessageBox.warning(self, "Clock offset", "Use a form like +1:04:08, -45s or 1h.")
        if c2 and c3:
            try:
                self.inventory.mark_outside(datetime.fromisoformat(c2), datetime.fromisoformat(c3),
                                            clock_offset_s=self.project.camera_clock_offset_s)
            except ValueError:
                QMessageBox.warning(self, "Contact times", "Use the form 2024-04-08T13:48:10 for contact times.")
        self.project.second_contact, self.project.third_contact = c2, c3
        self.project.bias_files = find_raw_files(self.bias_dir.text()) if self.bias_dir.text() else []
        self.project.dark_files = find_raw_files(self.dark_dir.text()) if self.dark_dir.text() else []
        self.project.flat_files = find_raw_files(self.flat_dir.text()) if self.flat_dir.text() else []
        self.replan()

    def replan(self) -> None:
        if not self.inventory.frames:
            return
        self.bracket_report = detect_brackets(self.inventory.frames)
        choice = self.lens_combo.currentText()
        calib = CalibrationFrames(
            bias=read_frames(self.project.bias_files) if self.project.bias_files else [],
            darks=read_frames(self.project.dark_files) if self.project.dark_files else [],
            flats=read_frames(self.project.flat_files) if self.project.flat_files else [],
        )
        self.plan = plan_calibration(
            self.inventory.lights(),
            self.db,
            calib,
            lens_name=None if choice in (AUTOMATIC, NO_PROFILE) else choice,
            use_lensfun=choice != NO_PROFILE,
            profiles=load_profile_library(),
        )
        self.project.lens_profile = "" if choice == AUTOMATIC else choice
        self.project.use_lensfun = choice != NO_PROFILE
        self.project.store_inventory(self.inventory)
        self._fill_table()
        self.report.setPlainText(self.bracket_report.describe() + "\n\n" + self.plan.describe())

    def _refresh_lens_choices(self) -> None:
        current = self.project.lens_profile
        self.lens_combo.blockSignals(True)
        self.lens_combo.clear()
        self.lens_combo.addItems([AUTOMATIC, NO_PROFILE])
        sample = next((f for f in self.inventory.frames if f.lens_model or f.lens_spec), None)
        if sample and not self.db.empty:
            maker = sample.lens_make or sample.camera_make
            for m in self.db.match_lens(sample.lens_model, sample.lens_spec, sample.focal_length_mm, maker, limit=8):
                self.lens_combo.addItem(m.lens.label())
        if current:
            idx = self.lens_combo.findText(current)
            if idx < 0:
                self.lens_combo.addItem(current)
                idx = self.lens_combo.count() - 1
            self.lens_combo.setCurrentIndex(idx)
        self.lens_combo.blockSignals(False)

    def _fill_table(self) -> None:
        frames = self.inventory.frames
        self.table.setRowCount(len(frames))
        for row, f in enumerate(frames):
            dt = f.datetime()
            values = [
                f.index, f.file_name, dt.strftime("%H:%M:%S.%f")[:11] if dt else "",
                f.exposure_text or format_exposure(f.exposure_s), _num(f.aperture_f), _num(f.iso),
                _num(f.focal_length_mm), _num(f.black_level), _num(f.white_level),
                "" if f.bracket_id is None else f.bracket_id, f.role, "; ".join(f.warnings),
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                if col in (0, 3, 4, 5, 6, 7, 8, 9):
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                if f.role != "light":
                    item.setForeground(Qt.GlobalColor.gray)
                self.table.setItem(row, col, item)

    def _table_menu(self, pos) -> None:
        rows = sorted({i.row() for i in self.table.selectedIndexes()})
        if not rows:
            return
        menu = QMenu(self)
        for role in ("light", "partial", "excluded"):
            act = QAction(f"Mark as {role}", menu)
            act.triggered.connect(lambda _=False, r=role: self._set_role(rows, r))
            menu.addAction(act)
        menu.exec(self.table.viewport().mapToGlobal(pos))

    def _set_role(self, rows: list[int], role: str) -> None:
        for r in rows:
            frame = self.inventory.frames[r]
            frame.role = role
        self.replan()

    def write_flat(self) -> None:
        if not self.plan:
            return
        lensfun_frames = [fc for fc in self.plan.frames if fc.vignetting in ("lensfun", "profile")]
        if not lensfun_frames:
            QMessageBox.information(self, "Lensfun flat", "No frame uses a lensfun or own-flats profile.")
            return
        by_index = {f.index: f for f in self.inventory.frames}
        written = set()
        out_dir = Path(self.link.working_dir())
        for fc in lensfun_frames:
            f = by_index[fc.frame_index]
            if not (f.width and f.height):
                QMessageBox.warning(self, "Lensfun flat", "Image size unknown: install rawpy so the flat can match the RAW frames.")
                return
            key = (f.width, f.height, fc.lens_terms)
            if key in written:
                continue
            kind = "profile" if fc.vignetting == "profile" else "lensfun"
            name = f"tse_{kind}_flat_f{f.aperture_f:g}_{f.focal_length_mm:g}mm.fit"
            write_synthetic_flat(out_dir / name, f.width, f.height, fc.lens_terms, f.crop_factor or 1.0,
                                 fc.profile_crop, fc.channel_terms, f.cfa_pattern)
            written.add(key)
            self.link.log(f"Solar Eclipse Toolkit: wrote {name}")
        QMessageBox.information(self, "Lensfun flat", f"Wrote {len(written)} synthetic flat(s) to\n{out_dir}")

    def update_lensfun(self) -> None:
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            path = download_database()
            self.db = LensfunDatabase.load()
        except Exception as exc:
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, "Update lensfun", f"Download failed:\n{exc}")
            return
        QApplication.restoreOverrideCursor()
        self.link.log(f"Solar Eclipse Toolkit: lensfun database updated ({len(self.db.lenses)} lenses) in {path}")
        self._refresh_lens_choices()
        self.replan()

    # -------------------------------------------------------------- project
    def save_project(self, path: Path | None = None) -> None:
        if path is None:
            name, _ = QFileDialog.getSaveFileName(
                self, "Save project", str(self.project_path or Path(self.link.working_dir()) / "eclipse_project.json"),
                "Project (*.json)")
            if not name:
                return
            path = Path(name)
        self.project.store_inventory(self.inventory)
        self.project.save(path)
        self.project_path = path
        self.link.log(f"Solar Eclipse Toolkit: project saved to {path}")

    def open_project(self) -> None:
        name, _ = QFileDialog.getOpenFileName(self, "Open project", self.link.working_dir(), "Project (*.json)")
        if not name:
            return
        self.project = EventProject.load(name)
        self.project_path = Path(name)
        self.inventory = self.project.inventory
        self.folder.setText(self.project.raw_folder)
        self.c2.setText(self.project.second_contact)
        self.c3.setText(self.project.third_contact)
        offset = self.project.camera_clock_offset_s
        self.clock.setText(f"{offset:+.0f}s" if offset else "")
        self._refresh_lens_choices()
        if not self.project.use_lensfun:
            self.lens_combo.setCurrentText(NO_PROFILE)
        self.replan()


# ---------------------------------------------------------------------------
# Running steps in the background
# ---------------------------------------------------------------------------


class Worker(QThread):
    """Runs one workflow step off the GUI thread; progress and the result come back as signals."""

    progress = pyqtSignal(str)
    succeeded = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, fn, parent=None) -> None:
        super().__init__(parent)
        self.fn = fn
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:  # noqa: D401 - Qt override
        try:
            result = self.fn(self.progress.emit, lambda: self._cancel)
        except W.Cancelled:
            self.failed.emit("Cancelled.")
        except W.WorkflowError as exc:
            self.failed.emit(str(exc))
        except MemoryError:
            self.failed.emit("Out of memory. Close other programs, or use a square output on the Stack tab.")
        except Exception:  # anything else: show the details so it can be reported
            self.failed.emit("Unexpected error:\n" + traceback.format_exc())
        else:
            self.succeeded.emit(result)


class StepTab(QWidget):
    """A tab that runs one step: settings on the left, results on the right, a log underneath."""

    def __init__(self, window: "MainWindow", parent=None) -> None:
        super().__init__(parent)
        self.main = window
        self.worker: Worker | None = None
        outer = QVBoxLayout(self)
        self.split = QSplitter(Qt.Orientation.Horizontal)
        self.settings = QWidget()
        self.settings_layout = QVBoxLayout(self.settings)
        self.results = QWidget()
        self.results_layout = QVBoxLayout(self.results)
        self.results_layout.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.settings)
        scroll.setMinimumWidth(330)
        self.split.addWidget(scroll)
        self.split.addWidget(self.results)
        self.split.setSizes([360, 840])
        vsplit = QSplitter(Qt.Orientation.Vertical)
        vsplit.addWidget(self.split)
        logbox = QWidget()
        ll = QVBoxLayout(logbox)
        ll.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        self.status = QLabel("")
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel)
        row.addWidget(self.status, 1)
        row.addWidget(self.cancel_btn)
        ll.addLayout(row)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setFont(_mono())
        ll.addWidget(self.log)
        vsplit.addWidget(logbox)
        vsplit.setSizes([520, 180])
        outer.addWidget(vsplit)
        self.run_buttons: list[QPushButton] = []

    # Subclasses fill these in.
    def load_project(self, project: EventProject, path: Path) -> None:
        pass

    def start(self, label: str, fn, on_done, reload: bool = True) -> None:
        """Run ``fn(project_path, say, cancel)`` in the background; ``reload`` = the step changes the project."""
        if self.worker is not None:
            return
        path = self.main.ensure_saved()
        if path is None:
            return
        self.log.appendPlainText(f"--- {label} ---")
        self.status.setText(f"{label}…")
        self.worker = Worker(lambda say, cancel: fn(path, say, cancel), self)
        self.worker.progress.connect(self._progress)
        self.worker.succeeded.connect(lambda result: self._finish(result, on_done, reload))
        self.worker.failed.connect(self._failed)
        for b in self.run_buttons:
            b.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.main.link.log(f"Solar Eclipse Toolkit: {label}")
        self.worker.start()

    def cancel(self) -> None:
        if self.worker is not None:
            self.worker.cancel()
            self.status.setText("Stopping at the next step…")

    def _progress(self, msg: str) -> None:
        self.log.appendPlainText(msg)
        self.status.setText(msg[:120])

    def _done_common(self) -> None:
        self.worker = None
        for b in self.run_buttons:
            b.setEnabled(True)
        self.cancel_btn.setEnabled(False)

    def _finish(self, result, on_done, reload: bool = True) -> None:
        self._done_common()
        self.status.setText("Done.")
        on_done(result)
        if reload:
            self.main.reload_project()

    def _failed(self, msg: str) -> None:
        self._done_common()
        self.status.setText(msg.splitlines()[0][:120])
        self.log.appendPlainText(msg)
        if msg != "Cancelled.":
            QMessageBox.warning(self, "Solar Eclipse Toolkit", msg[:2000])


# ---------------------------------------------------------------------------
# Align
# ---------------------------------------------------------------------------


class AlignTab(StepTab):
    def __init__(self, window: "MainWindow") -> None:
        super().__init__(window)
        site = QGroupBox("Where and when")
        form = QFormLayout(site)
        self.lat, self.lon, self.elev = QLineEdit(), QLineEdit(), QLineEdit()
        self.lat.setPlaceholderText("e.g. 36.942202 (north positive)")
        self.lon.setPlaceholderText("e.g. -90.334287 (east positive)")
        self.elev.setPlaceholderText("metres, e.g. 110")
        self.tz = QLineEdit()
        self.tz.setPlaceholderText("hours, e.g. -6 (only if the files carry no time zone)")
        self.clock = QLineEdit()
        self.clock.setPlaceholderText("seconds to add to camera time, blank = work it out")
        form.addRow("Latitude:", self.lat)
        form.addRow("Longitude:", self.lon)
        form.addRow("Elevation:", self.elev)
        form.addRow("Camera time zone:", self.tz)
        form.addRow("Camera clock error:", self.clock)
        self.mirrored = QCheckBox("Optics mirror the image (telescope with a star diagonal)")
        form.addRow(self.mirrored)
        self.settings_layout.addWidget(site)

        opts = QGroupBox("Alignment")
        f2 = QFormLayout(opts)
        self.rotation = QComboBox()
        self.rotation.addItems(["drift", "per-frame", "none"])
        self.rotation.setToolTip("Field rotation: a smooth drift over time (tracking mounts), measured per frame, "
                                 "or none")
        f2.addRow("Rotation:", self.rotation)
        self.any_angle = QCheckBox("Search any rotation and scale (alt-az mounts)")
        f2.addRow(self.any_angle)
        self.settings_layout.addWidget(opts)

        run = QPushButton("Align frames")
        run.clicked.connect(self.run)
        self.run_buttons.append(run)
        self.settings_layout.addWidget(run)
        self.settings_layout.addStretch(1)

        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(["#", "File", "Exposure", "dx px", "dy px", "rot °", "PSR", "Notes"])
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.summary = QPlainTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setFont(_mono())
        rs = QSplitter(Qt.Orientation.Vertical)
        rs.addWidget(self.table)
        rs.addWidget(self.summary)
        rs.setSizes([360, 200])
        self.results_layout.addWidget(rs)

    def load_project(self, project: EventProject, path: Path) -> None:
        self.lat.setText(_txt(project.latitude))
        self.lon.setText(_txt(project.longitude))
        self.elev.setText(_txt(project.elevation_m))
        self.tz.setText(_txt(project.camera_utc_offset_h))
        self.clock.setText(_txt(project.clock_error_s))
        self.mirrored.setChecked(bool(project.mirrored))
        a = project.parameters.get("align", {})
        if a.get("rotation"):
            self.rotation.setCurrentText(a["rotation"])
        try:
            alignment, _ = W.load_alignment(path)
            self.show_alignment(alignment)
        except W.WorkflowError:
            self.table.setRowCount(0)
            self.summary.setPlainText("Not aligned yet.")

    def run(self) -> None:
        try:
            opts = W.AlignOptions(latitude=_float(self.lat), longitude=_float(self.lon), elevation_m=_float(self.elev),
                                  camera_utc_offset_h=_float(self.tz), clock_error_s=_float(self.clock),
                                  mirrored=self.mirrored.isChecked(), rotation=self.rotation.currentText(),
                                  any_angle=self.any_angle.isChecked())
        except ValueError as exc:
            QMessageBox.warning(self, "Align", str(exc))
            return
        self.start("Aligning frames", lambda p, say, cancel: W.align(p, opts, say, cancel), self._done)

    def _done(self, out) -> None:
        result, path = out
        self.log.appendPlainText(result.describe())
        self.log.appendPlainText(f"Alignment saved to {path}")
        import json

        self.show_alignment(json.loads(Path(path).read_text(encoding="utf-8")))

    def show_alignment(self, a: dict) -> None:
        frames = a.get("frames", [])
        self.table.setRowCount(len(frames))
        for row, f in enumerate(frames):
            sx, sy = f.get("shift_px", (0, 0))
            values = [f["index"], f["file_name"], f.get("exposure_text", ""), f"{sx:+.2f}", f"{sy:+.2f}",
                      f"{f.get('angle_deg', 0):+.4f}", "ref" if f.get("psr") in (None, float("inf")) else
                      f"{f['psr']:.0f}", "; ".join(f.get("notes", []))]
            for col, v in enumerate(values):
                item = QTableWidgetItem(str(v))
                if col in (0, 3, 4, 5, 6):
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.table.setItem(row, col, item)
        lines = [f"Reference frame: #{a.get('reference_index')}"]
        sun = a.get("sun")
        lines.append(sun.get("text", "") if sun else "Sun centre not computed (enter the site to get it).")
        lines += [f"Note: {n}" for n in a.get("notes", [])]
        self.summary.setPlainText("\n".join(lines))


# ---------------------------------------------------------------------------
# Stack
# ---------------------------------------------------------------------------


class StackTab(StepTab):
    def __init__(self, window: "MainWindow") -> None:
        super().__init__(window)
        size = QGroupBox("Output size")
        sl = QVBoxLayout(size)
        self.full = QRadioButton("Full frame (same size as the RAW files)")
        self.square = QRadioButton("Square around the Sun (quicker)")
        self.full.setChecked(True)
        self.radii = QDoubleSpinBox()
        self.radii.setRange(1.5, 30.0)
        self.radii.setValue(5.0)
        self.radii.setSuffix(" solar radii from centre to edge")
        sl.addWidget(self.full)
        sl.addWidget(self.square)
        sl.addWidget(self.radii)
        self.settings_layout.addWidget(size)

        opts = QGroupBox("Options")
        f = QFormLayout(opts)
        self.moon_frame = QComboBox()
        self.moon_frame.addItem("Alignment reference", None)
        self.moon_frame.setToolTip("The frame whose Moon (and earthshine) is kept inside the lunar disc")
        f.addRow("Moon from frame:", self.moon_frame)
        self.include_rejected = QCheckBox("Include frames flagged as off the Moon's track")
        self.hot = QCheckBox("Fix isolated hot pixels")
        self.hot.setChecked(True)
        f.addRow(self.include_rejected)
        f.addRow(self.hot)
        self.settings_layout.addWidget(opts)

        run = QPushButton("Stack frames")
        run.clicked.connect(self.run)
        self.run_buttons.append(run)
        self.settings_layout.addWidget(run)
        self.open_btn = QPushButton("Open HDR stack in Siril")
        self.open_btn.clicked.connect(lambda: self.main.open_in_siril(self.hdr_path))
        self.settings_layout.addWidget(self.open_btn)
        self.settings_layout.addStretch(1)
        self.hdr_path = ""

        self.preview = ImageView()
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["#", "Gain", "Offset", "vs nominal (EV)", "Pixels"])
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        rs = QSplitter(Qt.Orientation.Horizontal)
        rs.addWidget(self.preview)
        rs.addWidget(self.table)
        rs.setSizes([620, 260])
        self.results_layout.addWidget(rs)

    def load_project(self, project: EventProject, path: Path) -> None:
        self.moon_frame.clear()
        self.moon_frame.addItem("Alignment reference", None)
        for fr in project.inventory.lights():
            self.moon_frame.addItem(f"#{fr.index} {fr.file_name} ({fr.exposure_text})", fr.index)
        st = project.parameters.get("stack", {})
        if st.get("radii"):
            self.square.setChecked(True)
            self.radii.setValue(float(st["radii"]))
        if st.get("moon_frame") is not None:
            i = self.moon_frame.findData(st["moon_frame"])
            if i >= 0:
                self.moon_frame.setCurrentIndex(i)
        self.hdr_path = W.resolve(path, st.get("hdr"))
        self.open_btn.setEnabled(bool(self.hdr_path))
        self.preview.show_file(W.resolve(path, st.get("preview")))
        info = W.resolve(path, st.get("info"))
        if info:
            import json

            self.show_matches(json.loads(Path(info).read_text(encoding="utf-8")).get("matches", []))

    def run(self) -> None:
        opts = W.StackOptions(radii=self.radii.value() if self.square.isChecked() else None,
                              moon_frame=self.moon_frame.currentData(),
                              include_rejected=self.include_rejected.isChecked(), hot_pixels=self.hot.isChecked())
        self.start("Stacking frames", lambda p, say, cancel: W.stack(p, opts, say, cancel), self._done)

    def _done(self, out) -> None:
        result, _skipped = out
        self.log.appendPlainText(result.describe())
        self.hdr_path = result.outputs.get("hdr", "")
        self.open_btn.setEnabled(bool(self.hdr_path))
        self.preview.show_file(result.outputs.get("preview", ""))
        self.show_matches([m.__dict__ for m in result.matches])
        self.main.enhance_tab.forget_preview()

    def show_matches(self, matches: list) -> None:
        self.table.setRowCount(len(matches))
        for row, m in enumerate(matches):
            for col, v in enumerate([m["index"], f"{m['gain']:.5f}", f"{m['offset']:+.5f}",
                                     f"{m['stops_vs_nominal']:+.2f}", m["pixels"]]):
                item = QTableWidgetItem(str(v))
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                if col == 3 and abs(m["stops_vs_nominal"]) > 0.3:
                    item.setForeground(Qt.GlobalColor.red)
                self.table.setItem(row, col, item)


# ---------------------------------------------------------------------------
# Enhance
# ---------------------------------------------------------------------------

LAYER_NAMES = {"natural": "Natural", "nrgf": "NRGF", "fnrgf": "FNRGF", "fnrgf_n": "FNRGF-N",
               "adaptive": "Adaptive", "blend": "Blend"}


class EnhanceTab(StepTab):
    """Settings with a live, reduced-size preview; the full-size run writes the TIFF layers."""

    def __init__(self, window: "MainWindow") -> None:
        super().__init__(window)
        self.small: W.HdrData | None = None
        self.preview_layers: dict = {}
        self._dirty = False
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(700)
        self.timer.timeout.connect(self.preview)

        lay = QGroupBox("Layers to write")
        ll = QVBoxLayout(lay)
        self.layer_checks = {}
        for k in LAYERS:
            c = QCheckBox(LAYER_NAMES[k])
            c.setChecked(True)
            ll.addWidget(c)
            self.layer_checks[k] = c
        self.settings_layout.addWidget(lay)

        blend = QGroupBox("Suggested blend (Soft Light over the natural layer)")
        bf = QFormLayout(blend)
        self.blend = {}
        for k, default in (("fnrgf_n", 70), ("adaptive", 50), ("fnrgf", 0), ("nrgf", 0)):
            sp = self._spin(0, 100, default, " %")
            bf.addRow(f"{LAYER_NAMES[k]}:", sp)
            self.blend[k] = sp
        self.settings_layout.addWidget(blend)

        tone = QGroupBox("Natural layer and colour")
        tf = QFormLayout(tone)
        self.flatten = self._dspin(0.0, 1.0, 0.8, 0.05, "How much of the brightness fall-off from the Sun to remove")
        tf.addRow("Flatten:", self.flatten)
        self.saturation = self._dspin(0.0, 3.0, 1.0, 0.1, "0 = monochrome")
        tf.addRow("Saturation:", self.saturation)
        self.white = QComboBox()
        self.white.addItems(["corona", "camera", "none"])
        self.white.setToolTip("corona: the inner corona is made white; camera: as-shot white balance")
        self.white.currentIndexChanged.connect(self.changed)
        tf.addRow("White balance:", self.white)
        self.keep_sky = QCheckBox("Keep the sky (don't model and remove it)")
        self.keep_sky.toggled.connect(self.changed)
        tf.addRow(self.keep_sky)
        self.settings_layout.addWidget(tone)

        fn = QGroupBox("FNRGF")
        ff = QFormLayout(fn)
        self.order = self._spin(1, 20, 10, "")
        self.order.setToolTip("Fourier order of the circle statistics (paper: 10)")
        ff.addRow("Order:", self.order)
        self.segments = self._spin(12, 200, 50, "")
        self.segments.setToolTip("Angular segments per circle (paper: 50)")
        ff.addRow("Segments:", self.segments)
        self.noise_m = self._dspin(0.0, 0.99, 0.7, 0.05, "FNRGF-N noise compensation (0 = plain FNRGF)")
        ff.addRow("Noise compensation:", self.noise_m)
        self.settings_layout.addWidget(fn)

        btns = QHBoxLayout()
        self.preview_btn = QPushButton("Preview")
        self.preview_btn.clicked.connect(self.preview)
        self.run_btn = QPushButton("Write full-size layers")
        self.run_btn.clicked.connect(self.run)
        btns.addWidget(self.preview_btn)
        btns.addWidget(self.run_btn)
        self.run_buttons += [self.preview_btn, self.run_btn]
        self.settings_layout.addLayout(btns)
        self.auto = QCheckBox("Update the preview when a setting changes")
        self.auto.setChecked(True)
        self.settings_layout.addWidget(self.auto)
        self.settings_layout.addStretch(1)

        top = QHBoxLayout()
        top.addWidget(QLabel("Show:"))
        self.show_layer = QComboBox()
        for k in list(LAYERS) + ["blend"]:
            self.show_layer.addItem(LAYER_NAMES[k], k)
        self.show_layer.setCurrentIndex(self.show_layer.count() - 1)
        self.show_layer.currentIndexChanged.connect(self._show)
        top.addWidget(self.show_layer)
        top.addStretch(1)
        self.open_btn = QPushButton("Open layer in Siril")
        self.open_btn.clicked.connect(self._open_layer)
        top.addWidget(self.open_btn)
        self.results_layout.addLayout(top)
        self.view = ImageView()
        self.results_layout.addWidget(self.view, 1)
        self.outputs: dict = {}

    def _spin(self, lo, hi, val, suffix) -> QSpinBox:
        sp = QSpinBox()
        sp.setRange(lo, hi)
        sp.setValue(val)
        sp.setSuffix(suffix)
        sp.valueChanged.connect(self.changed)
        return sp

    def _dspin(self, lo, hi, val, step, tip) -> QDoubleSpinBox:
        sp = QDoubleSpinBox()
        sp.setRange(lo, hi)
        sp.setSingleStep(step)
        sp.setValue(val)
        sp.setToolTip(tip)
        sp.valueChanged.connect(self.changed)
        return sp

    def load_project(self, project: EventProject, path: Path) -> None:
        self.outputs = {k: W.resolve(path, v) for k, v in project.parameters.get("enhance", {}).items()
                        if k in LAYER_NAMES and W.resolve(path, v)}

    def forget_preview(self) -> None:
        self.small = None

    def params(self) -> EnhanceParams:
        blend = {"natural": 1.0}
        blend.update({k: sp.value() / 100.0 for k, sp in self.blend.items() if sp.value() > 0})
        layers = tuple(k for k, c in self.layer_checks.items() if c.isChecked())
        return EnhanceParams(layers=layers, segments=self.segments.value(), order=self.order.value(),
                             noise_m=self.noise_m.value(), flatten=self.flatten.value(),
                             saturation=self.saturation.value(), white=self.white.currentText(), blend=blend,
                             neutral_background=not self.keep_sky.isChecked())

    def changed(self, *_args) -> None:
        if self.auto.isChecked() and self.small is not None:
            self.timer.start()

    def preview(self) -> None:
        if self.worker is not None:
            self._dirty = True  # run again when the current preview finishes
            return
        params = self.params()
        cached = self.small

        def job(path, say, cancel):
            small = cached
            if small is None:
                say("Reading the HDR stack for the preview…")
                small = W.reduce_for_preview(W.load_hdr(path), max_side=1400)
            # The preview shows every layer the blend needs, whatever is ticked for writing.
            from dataclasses import replace

            p = replace(params, layers=tuple(LAYERS))
            result, layers = W.enhance_preview(small, p, say, cancel)
            return small, result, layers

        self.start("Preview", job, self._preview_done, reload=False)

    def _preview_done(self, out) -> None:
        self.small, result, layers = out
        self.preview_layers = layers
        self.log.appendPlainText(result.describe())
        self._show()
        if self._dirty:
            self._dirty = False
            self.preview()

    def _show(self) -> None:
        k = self.show_layer.currentData()
        img = self.preview_layers.get(k)
        if img is not None:
            self.view.show_array(img)
        elif self.outputs.get(k):
            self.view.show_file(self.outputs[k])

    def run(self) -> None:
        params = self.params()
        if not params.layers:
            QMessageBox.information(self, "Enhance", "Tick at least one layer to write.")
            return
        self.start("Writing full-size layers",
                   lambda p, say, cancel: W.enhance_project(p, params, say=say, cancel=cancel), self._done)

    def _done(self, out) -> None:
        result, _notes = out
        self.log.appendPlainText(result.describe())
        self.outputs = {k: v for k, v in result.layers.items() if k in LAYER_NAMES}
        folder = Path(next(iter(self.outputs.values()))).parent if self.outputs else ""
        QMessageBox.information(self, "Enhance", f"Wrote {len(self.outputs)} layer(s) to\n{folder}")

    def _open_layer(self) -> None:
        path = self.outputs.get(self.show_layer.currentData())
        if not path:
            QMessageBox.information(self, "Enhance", "Write the full-size layers first.")
            return
        self.main.open_in_siril(path)


# ---------------------------------------------------------------------------
# Image display
# ---------------------------------------------------------------------------


class ImageView(QLabel):
    """Shows an image scaled to fit, keeping its aspect ratio."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(320, 240)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet("background: #111; color: #888;")
        self.setText("No image yet.")
        self._pix: QPixmap | None = None

    def show_array(self, img) -> None:
        import numpy as np

        a = np.ascontiguousarray((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8))
        if a.ndim == 2:
            a = np.ascontiguousarray(np.repeat(a[..., None], 3, axis=2))
        h, w = a.shape[:2]
        qimg = QImage(a.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
        self._pix = QPixmap.fromImage(qimg)
        self._rescale()

    def show_file(self, path: str) -> None:
        if not path or not Path(path).exists():
            return
        try:
            import numpy as np

            if path.lower().endswith((".tif", ".tiff")):
                import tifffile

                data = tifffile.imread(path)
                step = max(1, max(data.shape[:2]) // 1600)
                data = data[::step, ::step].astype(np.float32) / (65535.0 if data.dtype == np.uint16 else 255.0)
                self.show_array(data)
            else:
                self._pix = QPixmap(path)
                self._rescale()
        except Exception as exc:  # a preview is a convenience; never fail a step over it
            self.setText(f"Could not show {Path(path).name}: {exc}")

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().resizeEvent(event)
        self._rescale()

    def _rescale(self) -> None:
        if self._pix is not None and not self._pix.isNull():
            self.setPixmap(self._pix.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                            Qt.TransformationMode.SmoothTransformation))


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------


class MainWindow(QMainWindow):
    def __init__(self, link: SirilLink) -> None:
        super().__init__()
        self.link = link
        self.setWindowTitle(f"Solar Eclipse Toolkit {VERSION}" + ("" if link.connected else " (not connected to Siril)"))
        self.resize(1280, 820)
        self.tabs = QTabWidget()
        self.frames_tab = FramesTab(link)
        self.align_tab = AlignTab(self)
        self.stack_tab = StackTab(self)
        self.enhance_tab = EnhanceTab(self)
        self.tabs.addTab(self.frames_tab, "1 Frames")
        self.tabs.addTab(self.align_tab, "2 Align")
        self.tabs.addTab(self.stack_tab, "3 Stack")
        self.tabs.addTab(self.enhance_tab, "4 Enhance")
        self.setCentralWidget(self.tabs)

        menu = self.menuBar().addMenu("&Project")
        for text, slot, key in (("&Open…", self.open_project, "Ctrl+O"),
                                ("&Save", lambda: self.ensure_saved(), "Ctrl+S"),
                                ("Save &as…", self.save_as, None)):
            act = QAction(text, self)
            act.triggered.connect(slot)
            if key:
                act.setShortcut(key)
            menu.addAction(act)

    # The Frames tab owns the project in memory; the steps read and update it on disk.
    def ensure_saved(self) -> Path | None:
        ft = self.frames_tab
        if not ft.inventory.frames and ft.project_path is None:
            QMessageBox.information(self, "Solar Eclipse Toolkit",
                                    "Read the RAW folder on the Frames tab first (or open a saved project).")
            return None
        if ft.project_path is None:
            ft.save_project()
        else:
            ft.save_project(ft.project_path)
        if ft.project_path is not None:
            self._title(ft.project_path)
        return ft.project_path

    def save_as(self) -> None:
        self.frames_tab.save_project()
        if self.frames_tab.project_path:
            self.reload_project()

    def open_project(self) -> None:
        self.frames_tab.open_project()
        if self.frames_tab.project_path:
            self.enhance_tab.forget_preview()
            self.reload_project()

    def reload_project(self) -> None:
        """After a step changed the project on disk: reload it everywhere."""
        path = self.frames_tab.project_path
        if path is None or not Path(path).exists():
            return
        project = EventProject.load(path)
        self.frames_tab.project = project
        self.frames_tab.inventory = project.inventory
        for tab in (self.align_tab, self.stack_tab, self.enhance_tab):
            tab.load_project(project, Path(path))
        self._title(path)

    def _title(self, path) -> None:
        self.setWindowTitle(f"Solar Eclipse Toolkit {VERSION} – {Path(path).name}"
                            + ("" if self.link.connected else " (not connected to Siril)"))

    def open_in_siril(self, path: str) -> None:
        if not path:
            return
        if not self.link.connected:
            QMessageBox.information(self, "Solar Eclipse Toolkit", f"Not connected to Siril. The file is:\n{path}")
            return
        try:
            self.link.siril.cmd("load", f'"{path}"')
        except Exception as exc:
            QMessageBox.warning(self, "Solar Eclipse Toolkit", f"Siril could not open the file:\n{exc}")


def _mono() -> QFont:
    font = QFont("Monospace")
    font.setStyleHint(QFont.StyleHint.TypeWriter)
    return font


def _txt(x) -> str:
    return "" if x is None else f"{x:g}" if isinstance(x, float) else str(x)


def _float(edit: QLineEdit):
    t = edit.text().strip()
    if not t:
        return None
    try:
        return float(t)
    except ValueError:
        raise ValueError(f"Not a number: {t!r}") from None



def _num(x) -> str:
    if x is None:
        return ""
    return f"{x:.1f}".rstrip("0").rstrip(".") if isinstance(x, float) else str(x)

def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] not in ("-platform",):
        # Command-line use, e.g. from Siril's pyscript command.
        from tse_core.cli import main as cli_main

        return cli_main(sys.argv[1:])
    app = QApplication.instance() or QApplication(sys.argv)
    link = SirilLink()
    window = MainWindow(link)
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
