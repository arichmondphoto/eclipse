# SPDX-License-Identifier: MIT
"""Solar Eclipse Toolkit: Siril script entry point (Phase 1: Frames).

Run from Siril (Scripts menu, Siril 1.4 or later) or on its own with any
Python 3.10+ that has PyQt6:

    python SolarEclipseToolkit.py            # GUI
    python SolarEclipseToolkit.py --help     # command-line options

Phase 1 covers the Frames step: read a folder of RAW files, review the
inventory, check the detected brackets, choose the lens profile, add
optional calibration frames, and save the event's project file.
"""

from __future__ import annotations

import sys
from pathlib import Path

VERSION = "0.1.0"

# The core package lives in ../src when the script runs from the repository.
_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# --------------------------------------------------------------- Siril link
try:
    import sirilpy as s

    for _pkg in ("PyQt6", "exifread", "numpy", "rawpy"):
        try:
            s.ensure_installed(_pkg)
        except Exception as exc:  # rawpy is optional; keep going without it
            print(f"Could not install {_pkg}: {exc}")
    HAVE_SIRIL = True
except ImportError:
    s = None
    HAVE_SIRIL = False

from datetime import datetime  # noqa: E402

from PyQt6.QtCore import Qt  # noqa: E402
from PyQt6.QtGui import QAction, QFont  # noqa: E402
from PyQt6.QtWidgets import (  # noqa: E402
    QAbstractItemView, QApplication, QCheckBox, QComboBox, QFileDialog, QFormLayout,
    QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMainWindow, QMenu,
    QMessageBox, QPlainTextEdit, QPushButton, QSplitter, QTableWidget, QTableWidgetItem,
    QTabWidget, QVBoxLayout, QWidget,
)

from tse_core.brackets import detect_brackets  # noqa: E402
from tse_core.calibration import CalibrationFrames, plan_calibration, write_synthetic_flat  # noqa: E402
from tse_core.frames import format_exposure  # noqa: E402
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
        self.c2.setPlaceholderText("camera time, e.g. 2024-04-08T13:48:10")
        self.c3 = QLineEdit()
        self.c3.setPlaceholderText("optional")
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
        if c2 and c3:
            try:
                self.inventory.mark_outside(datetime.fromisoformat(c2), datetime.fromisoformat(c3))
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
        self._refresh_lens_choices()
        if not self.project.use_lensfun:
            self.lens_combo.setCurrentText(NO_PROFILE)
        self.replan()


class MainWindow(QMainWindow):
    def __init__(self, link: SirilLink) -> None:
        super().__init__()
        self.setWindowTitle(f"Solar Eclipse Toolkit {VERSION}" + ("" if link.connected else " (not connected to Siril)"))
        self.resize(1200, 760)
        tabs = QTabWidget()
        self.frames_tab = FramesTab(link)
        tabs.addTab(self.frames_tab, "Frames")
        for name in ("Geometry", "Align", "Stack", "Enhance", "Export"):
            placeholder = QLabel(f"{name}: coming in a later phase.")
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            tabs.addTab(placeholder, name)
            tabs.setTabEnabled(tabs.count() - 1, False)
        self.setCentralWidget(tabs)

        menu = self.menuBar().addMenu("&Project")
        for text, slot in (("&Open…", self.frames_tab.open_project), ("&Save…", lambda: self.frames_tab.save_project())):
            act = QAction(text, self)
            act.triggered.connect(slot)
            menu.addAction(act)


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
