# SPDX-License-Identifier: MIT
"""Solar Eclipse Toolkit: Siril script entry point.

Run from Siril (Scripts menu, Siril 1.4 or later) or on its own with any
Python 3.10+ that has PyQt6:

    python SolarEclipseToolkit.py            # GUI
    python SolarEclipseToolkit.py --help     # command-line options (same as 'totality')

Four tabs, one per step, each saving into the event's project file:

1. Frames: read the RAW folder, review the inventory and brackets, choose
   the lens profile, add optional calibration frames.
2. Align: site and clock, then phase-correlation alignment and the Sun's centre.
3. Stack: the Moon-aware linear HDR stack (full frame or a square).
4. Enhance: natural, NRGF, FNRGF, FNRGF-N and adaptive layers with a live,
   reduced-size preview, then the full-size 16-bit TIFF layers.
5. Earthshine: the lunar surface from a Moon-aligned stack of the long
   exposures, as a layer that fits the corona layers' black disc.
6. Prominences: found by colour at the limb, written as a transparent layer
   and put on top of the final image.

Picture outputs can be 16-bit TIFF, 16-bit PNG or 8-bit JPG, all 300 dpi.

Long steps run in the background with progress in the log and a Cancel button.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

VERSION = "0.1.0"


def _git_version() -> str:
    """The repository commit the script runs from (shown in the title and the log), or ''."""
    try:
        git = Path(__file__).resolve().parent.parent / ".git"
        head = (git / "HEAD").read_text().strip()
        if head.startswith("ref:"):
            ref = head.split(" ", 1)[1]
            p = git / ref
            if p.exists():
                return p.read_text().strip()[:7]
            for line in (git / "packed-refs").read_text().splitlines():
                if line.endswith(ref):
                    return line[:7]
            return ""
        return head[:7]
    except OSError:
        return ""


BUILD = _git_version()

# The core package lives in ../src when the script runs from the repository.
_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# --------------------------------------------------------------- Siril link
try:
    import sirilpy as s

    for _pkg in ("PyQt6", "numpy", "scipy", "exifread", "rawpy", "tifffile", "pillow", "certifi", "truststore"):
        try:
            s.ensure_installed(_pkg)
        except Exception as exc:  # report and keep going; a missing reader shows up when it is needed
            print(f"Could not install {_pkg}: {exc}")
    HAVE_SIRIL = True
except ImportError:
    s = None
    HAVE_SIRIL = False

from datetime import datetime  # noqa: E402

from PyQt6.QtCore import QRectF, QThread, QTimer, Qt, pyqtSignal  # noqa: E402
from PyQt6.QtGui import QAction, QColor, QFont, QImage, QPainter, QPixmap  # noqa: E402
from PyQt6.QtWidgets import (  # noqa: E402
    QAbstractItemView, QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMainWindow, QMenu,
    QMessageBox, QPlainTextEdit, QPushButton, QRadioButton, QScrollArea, QSizePolicy, QSpinBox,
    QSplitter, QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget,
)

from tse_core import workflow as W  # noqa: E402
from tse_core.brackets import detect_brackets  # noqa: E402
from tse_core import imageout as IO  # noqa: E402
from tse_core.viewzoom import ViewZoom  # noqa: E402
from tse_core.enhance import LAYERS, EnhanceParams  # noqa: E402
from tse_core.calibration import CalibrationFrames, plan_calibration, write_synthetic_flat  # noqa: E402
from tse_core.frames import format_exposure  # noqa: E402
from tse_core.inventory import Inventory, ProjectFolder, find_raw_files  # noqa: E402
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
        self.on_project_opened = None  # set by the main window
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
        self.browse_btn = browse
        self.use_exiftool = QCheckBox("Use exiftool if installed")
        self.use_exiftool.setChecked(True)
        scan = QPushButton("Read frames")
        scan.clicked.connect(self.scan)
        self.scan_btn = scan
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

        left = QVBoxLayout()
        event = QGroupBox("Event")
        ef = QFormLayout(event)
        self.eclipse_combo = QComboBox()
        ef.addRow("Eclipse:", self.eclipse_combo)
        self.lat, self.lon, self.elev = QLineEdit(), QLineEdit(), QLineEdit()
        self.lat.setPlaceholderText("e.g. 36.942202 (north positive)")
        self.lon.setPlaceholderText("e.g. -90.334287 (west negative)")
        self.elev.setPlaceholderText("metres, e.g. 110")
        self.tz = QLineEdit()
        self.tz.setPlaceholderText("hours, e.g. -6 (only if the files carry no time zone)")
        self.clock_error = QLineEdit()
        self.clock_error.setPlaceholderText("seconds, true minus camera; blank = not known")
        ef.addRow("Latitude:", self.lat)
        ef.addRow("Longitude:", self.lon)
        ef.addRow("Elevation:", self.elev)
        ef.addRow("Camera time zone:", self.tz)
        ef.addRow("Camera clock error:", self.clock_error)
        self.c2 = QLineEdit()
        self.c2.setPlaceholderText("camera clock, e.g. 2024-04-08T12:56:37")
        self.c3 = QLineEdit()
        self.c3.setPlaceholderText("camera clock, e.g. 2024-04-08T13:00:48")
        ef.addRow("Second contact:", self.c2)
        ef.addRow("Third contact:", self.c3)
        self.compute_btn = QPushButton("Compute contacts")
        self.compute_btn.clicked.connect(self.compute_contacts)
        ef.addRow(self.compute_btn)
        left.addWidget(event)

        options = QGroupBox("Calibration")
        form = QFormLayout(options)
        self.lens_combo = QComboBox()
        self.lens_combo.addItems([AUTOMATIC, NO_PROFILE])
        self.lens_combo.currentIndexChanged.connect(lambda _: self.replan())
        form.addRow("Lens profile:", self.lens_combo)
        self.bias_dir, self.dark_dir, self.flat_dir = QLineEdit(), QLineEdit(), QLineEdit()
        calib_tips = {
            "Bias folder:": "Optional. A folder of bias frames: the shortest exposures with the lens cap on, at "
                            "the same ISO as the eclipse frames. They are averaged into a master bias and "
                            "subtracted from frames that have no matching dark. Without them, the black level "
                            "recorded in each RAW file is used, which is fine for most modern cameras.",
            "Darks folder:": "Optional. A folder of dark frames: lens cap on, the same exposure times and ISO as "
                             "your eclipse frames, ideally right after totality at the same temperature. Each "
                             "eclipse frame gets the master dark with its own exposure and ISO subtracted; that "
                             "removes hot pixels and amp glow, which matters most for the long exposures. Frames "
                             "with no matching dark use the bias instead.",
            "Flats folder:": "Optional. A folder of flat frames: the same lens, focal length, aperture and focus "
                             "(infinity) as the eclipse, aimed at an even light (a light panel, or a white "
                             "T-shirt over the lens against twilight sky), histogram peak near the middle, 20 to "
                             "30 frames. They correct vignetting, and dust shadows if nothing was cleaned or "
                             "changed since the eclipse. How they are used (master flat, smooth profile, or "
                             "both compared) is chosen on the Stack tab. Flats at another aperture or focal length "
                             "are not used; the report below says so.",
        }
        self.calib_browse = []
        for label, edit in (("Bias folder:", self.bias_dir), ("Darks folder:", self.dark_dir), ("Flats folder:", self.flat_dir)):
            edit.setPlaceholderText("optional")
            box = QHBoxLayout()
            box.addWidget(edit, 1)
            b = QPushButton("…")
            b.setFixedWidth(28)
            b.clicked.connect(lambda _=False, e=edit: self._pick_folder(e))
            box.addWidget(b)
            form.addRow(label, box)
            _tip(edit, calib_tips[label] + " Click Apply after changing it.", form)
            _tip(b, "Choose the " + label.rstrip(":").lower() + ".")
            self.calib_browse.append(b)
        buttons = QHBoxLayout()
        self.setting_buttons = {}
        for text, slot in (
            ("Apply", self.apply_settings),
            ("Write synthetic flat", self.write_flat),
            ("Update lensfun", self.update_lensfun),
        ):
            b = QPushButton(text)
            b.clicked.connect(slot)
            buttons.addWidget(b)
            self.setting_buttons[text] = b
        form.addRow(buttons)
        left.addWidget(options)
        bl.addLayout(left, 0)
        self._settings_form = form
        self._event_form = ef
        self._fill_eclipses()

        self.report = QPlainTextEdit()
        self.report.setReadOnly(True)
        font = QFont("Monospace")
        font.setStyleHint(QFont.StyleHint.TypeWriter)
        self.report.setFont(font)
        bl.addWidget(self.report, 1)
        splitter.addWidget(bottom)
        splitter.setSizes([420, 260])
        layout.addWidget(splitter, 1)
        self._frames_tips(self._settings_form)

        if self.db.empty:
            self.report.setPlainText(
                "The lensfun database was not found.\nClick 'Update lensfun' to download it (about 15 MB)."
            )

    def _frames_tips(self, form) -> None:
        """Hover help for the Frames tab."""
        _tip(self.folder, "The folder holding the RAW files from totality (or a project folder holding a "
             "'lights' subfolder, with optional 'darks', 'flats' and 'bias' subfolders). Partial-phase frames "
             "may be in it too: they can be marked 'partial' and are then left out.")
        _tip(self.browse_btn, "Choose the RAW folder. If you choose a project folder (one with a 'lights' "
             "subfolder), its project file is opened, or a new one made, and the calibration folders are filled "
             "in from its 'darks', 'flats' and 'bias' subfolders.")
        _tip(self.use_exiftool, "Read the camera settings with exiftool when it is installed. It reads more "
             "makers' files fully (lens name, exact time) than the built-in reader. Untick only if exiftool "
             "gives wrong values.")
        _tip(self.scan_btn, "Read every RAW file's settings (time, exposure, aperture, ISO, lens) and group the "
             "exposures into brackets. Run it again after adding or removing files. Frames you marked by hand "
             "keep their mark.")
        _tip(self.table, "Every frame found, in capture order. Role: 'light' frames are aligned and stacked; "
             "'partial' frames (outside totality, from the contact times) and 'excluded' frames are left out. "
             "Right-click selected rows to mark them light, partial or excluded, for example a shaken frame or "
             "one with a cloud. Black and White are the sensor levels from the file; Bracket groups the frames "
             "of one exposure sweep.")
        _tip(self.lens_combo, "Where the vignetting (darkening towards the corners) correction comes from when you "
             "have no flats. Automatic: the lensfun profile matching the lens in the files, or a profile you "
             "fitted from your own flats earlier. Choose a lens by name if Automatic picks none or the wrong one, "
             "or 'None' to switch the correction off. Flats in the Flats folder are used instead whenever they "
             "match the frames' aperture and focal length.", form)
        ef = self._event_form
        _tip(self.eclipse_combo, "Which eclipse the frames are of. Automatic uses the eclipse on the date the frames "
             "were taken. The list holds the total and hybrid eclipses on this computer; the first time an "
             "eclipse outside the bundled set is needed, NASA's catalogue of every eclipse from 1600 to 2400 is "
             "downloaded once.", ef)
        _tip(self.lat, "Observing site latitude in degrees, north positive. With the longitude and elevation it "
             "gives the contact times here, and the Moon's path across the Sun, which locates the Sun's true "
             "centre (needed by the enhancement filters).", ef)
        _tip(self.lon, "Observing site longitude in degrees, east positive (west of Greenwich, as in the "
             "Americas, is negative).", ef)
        _tip(self.elev, "Site elevation above sea level in metres. A rough value is fine.", ef)
        _tip(self.tz, "The time zone the camera's clock was set to, in hours (for example -6 if it was on CST, "
             "-5 for CDT), used only when the RAW files do not record it. If the camera was on standard time "
             "while the eclipse was on daylight time, enter the zone the CAMERA was on.", ef)
        _tip(self.clock_error, "Seconds the camera's clock was behind true time (true minus camera), if you know "
             "it: 117 if it ran 117 s slow, -45 if it ran 45 s fast. A wrong time zone setting is not a clock "
             "error: fix the time zone instead. Leave blank if you do not know: the contacts are then computed "
             "as if the clock were right, and the Align step works the error out within the range that keeps "
             "every frame inside totality. Used for the contacts and for the Sun's centre.", ef)
        _tip(self.c2, "Second contact (start of totality) in the CAMERA's clock, as the RAW files record time, "
             "for example 2024-04-08T12:56:37. Compute contacts fills it in; you can also type it. Frames taken "
             "before it are marked 'partial' and left out when you click Apply. Needs the third contact too.", ef)
        _tip(self.c3, "Third contact (end of totality) in the camera's clock. Frames taken after it are marked "
             "'partial' and left out. Frames marked by hand as excluded stay excluded.", ef)
        _tip(self.compute_btn, "Work out second and third contact for the eclipse and site above from NASA's "
             "Besselian elements, convert them to the camera's clock (time zone and clock error), fill them in "
             "and mark the frames outside them. First it checks that your exposure brackets fit: a bracket cut in "
             "two by a contact usually means the clock or time zone is wrong, and you are asked before any frame "
             "is marked, with the clock error that would fit.")
        b = self.setting_buttons
        _tip(b["Apply"], "Use the settings above: mark frames outside the contact times, read the calibration "
             "folders and update the calibration plan in the report. The project is saved.")
        _tip(b["Write synthetic flat"], "Write the lens profile in use (lensfun, or one fitted from your own flats) "
             "as a FITS master flat, one per aperture and focal length, for calibrating the frames in Siril "
             "yourself. The toolkit's own steps do not need it.")
        _tip(b["Update lensfun"], "Download the latest lensfun database of lens profiles (about 15 MB). Do this "
             "once after installing, or when your lens is missing from the Lens profile list.")
        _tip(self.report, "The exposure brackets found and the calibration plan: which vignetting correction, "
             "darks and bias each group of frames will get, and warnings about anything missing.")

    # --------------------------------------------------------------- actions
    def _pick_folder(self, edit: QLineEdit) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Choose folder", edit.text() or self.link.working_dir())
        if not folder:
            return
        if edit is self.folder and ProjectFolder.scan(folder).is_project_folder:
            if self.use_project_folder(folder) and self.on_project_opened is not None:
                self.on_project_opened()  # every other tab shows this project too
            return
        edit.setText(folder)

    def _show_event(self) -> None:
        p = self.project
        self._fill_eclipses(p.eclipse_date)
        self.lat.setText(_txt(p.latitude))
        self.lon.setText(_txt(p.longitude))
        self.elev.setText(_txt(p.elevation_m))
        self.tz.setText(_txt(p.camera_utc_offset_h))
        self.clock_error.setText(_txt(p.clock_error_s))
        self.event_report = self.fit_report = ""

    def use_project_folder(self, folder: str) -> bool:
        """Set up from a project folder (lights/ and optional darks/, flats/, bias/); returns False if it
        has no lights/ subfolder. An existing project file in the folder is opened."""
        layout = ProjectFolder.scan(folder)
        if not layout.is_project_folder:
            return False
        if layout.project_file.exists():
            self.load_project_file(layout.project_file)
        else:
            self.project = EventProject(name=layout.root.name)
            self.project_path = layout.project_file
            self._show_event()
        self.folder.setText(str(layout.lights))
        for edit, sub in ((self.bias_dir, layout.bias), (self.dark_dir, layout.darks), (self.flat_dir, layout.flats)):
            edit.setText(str(sub) if sub else "")
        self.scan()
        self.save_project(self.project_path)
        return True

    def project_dir(self) -> Path:
        """Where the project file and results belong: the project folder if there is one."""
        if self.project_path is not None:
            return Path(self.project_path).parent
        raw = self.folder.text().strip()
        if raw:
            return Path(raw).resolve().parent  # the project folder (above lights/) or beside the frames
        return Path(self.link.working_dir())

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

    def _fill_eclipses(self, selected: str = "") -> None:
        from tse_core.event import known_eclipses

        self.eclipse_combo.blockSignals(True)
        self.eclipse_combo.clear()
        self.eclipse_combo.addItem("Automatic (the date of your frames)", "")
        try:
            items = known_eclipses()
        except Exception:  # a damaged catalogue must not stop the window opening
            items = []
        for day, label in items:
            self.eclipse_combo.addItem(label, day)
        if selected and self.eclipse_combo.findData(selected) < 0:
            self.eclipse_combo.addItem(selected, selected)
        i = self.eclipse_combo.findData(selected)
        self.eclipse_combo.setCurrentIndex(max(i, 0))
        self.eclipse_combo.blockSignals(False)

    def _store_event(self) -> bool:
        """Copy the Event box into the project; False (after a message) if a number is not readable."""
        try:
            lat, lon, elev = _float(self.lat), _float(self.lon), _float(self.elev)
            tz, ce = _float(self.tz), _float(self.clock_error)
        except ValueError as exc:
            QMessageBox.warning(self, "Event", str(exc))
            return False
        p = self.project
        p.eclipse_date = self.eclipse_combo.currentData() or ""
        p.latitude, p.longitude, p.elevation_m = lat, lon, elev if elev is not None else 0.0
        p.camera_utc_offset_h, p.clock_error_s = tz, ce
        return True

    def _store_calibration_folders(self, force: bool = False) -> None:
        """The bias, darks and flats folders shown into the project, where Stack reads them. Only a
        folder that changed is read again, so a project whose folders moved keeps its file list."""
        for edit, attr in ((self.bias_dir, "bias_files"), (self.dark_dir, "dark_files"),
                           (self.flat_dir, "flat_files")):
            text = edit.text().strip()
            files = getattr(self.project, attr)
            current = str(Path(files[0]).parent) if files else ""
            if force or text != current:
                setattr(self.project, attr, find_raw_files(text) if text else [])

    def compute_contacts(self) -> None:
        """Fill in the contacts from the eclipse and the site, then mark frames (after the fit check)."""
        from tse_core.event import compute_contacts

        if not self.inventory.frames:
            QMessageBox.information(self, "Compute contacts", "Read the frames first: their date and time zone "
                                                              "are needed.")
            return
        if not self._store_event():
            return
        p = self.project
        if p.latitude is None or p.longitude is None:
            QMessageBox.warning(self, "Compute contacts", "Enter the site's latitude and longitude first.")
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            con = compute_contacts(self.inventory.frames, p.latitude, p.longitude, p.elevation_m,
                                   p.camera_utc_offset_h, p.clock_error_s, p.eclipse_date,
                                   say=lambda m: self.link.log("Solar Eclipse Toolkit: " + m))
        except ValueError as exc:
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, "Compute contacts", str(exc))
            return
        QApplication.restoreOverrideCursor()
        self.c2.setText(con.c2_camera.isoformat(timespec="seconds"))
        self.c3.setText(con.c3_camera.isoformat(timespec="seconds"))
        self.event_report = con.describe()
        self.link.log("Solar Eclipse Toolkit: " + con.describe().replace("\n", "\n  "))
        self.apply_settings()

    def apply_settings(self) -> None:
        from tse_core.event import check_fit

        if not self.inventory.frames:
            return
        if not self._store_event():
            return
        c2, c3 = self.c2.text().strip(), self.c3.text().strip()
        if c2 and c3:
            try:
                t2, t3 = datetime.fromisoformat(c2), datetime.fromisoformat(c3)
            except ValueError:
                QMessageBox.warning(self, "Contact times", "Use the form 2024-04-08T12:56:37 for contact times.")
                return
            t2, t3 = t2.replace(tzinfo=None), t3.replace(tzinfo=None)
            fit = check_fit(self.inventory.frames, t2, t3, self.project.clock_error_s or 0.0)
            mark = fit.ok
            if not fit.ok:
                answer = QMessageBox.question(
                    self, "Contact times", "\n\n".join(w[:1].upper() + w[1:] for w in fit.warnings)
                    + "\n\nMark the frames outside these contact times as partial anyway?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
                mark = answer == QMessageBox.StandardButton.Yes
            self.fit_report = "" if fit.ok else "Contact check:\n" + "\n".join(f"  {w}" for w in fit.warnings) + (
                "\n  (frames were marked anyway)" if mark else "\n  (no frames were marked)")
            if mark:
                self.inventory.mark_outside(t2, t3)
        elif c2 or c3:
            QMessageBox.warning(self, "Contact times", "Enter both second and third contact (or neither).")
            return
        else:
            self.fit_report = ""
        self.project.second_contact, self.project.third_contact = c2, c3
        self.project.camera_clock_offset_s = 0.0
        self._store_calibration_folders(force=True)  # Apply also picks up files added to the same folder
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
        event = "\n\n".join(t for t in (getattr(self, "event_report", ""), getattr(self, "fit_report", "")) if t)
        self.report.setPlainText((event + "\n\n" if event else "") + self.bracket_report.describe() + "\n\n"
                                 + self.plan.describe())

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
        out_dir = self.project_dir()
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
            default = self.project_path or self.project_dir() / f"{self.project.name or 'eclipse_project'}.json"
            name, _ = QFileDialog.getSaveFileName(self, "Save project", str(default), "Project (*.json)")
            if not name:
                return
            path = Path(name)
        self._store_event()  # typed into the Event box but not applied yet: the steps need it too
        self._store_calibration_folders()  # a folder chosen with "..." counts even without Apply
        self.project.store_inventory(self.inventory)
        self.project.save(path)
        self.project_path = path
        self.link.log(f"Solar Eclipse Toolkit: project saved to {path}")

    def open_project(self) -> None:
        name, _ = QFileDialog.getOpenFileName(self, "Open project", str(self.project_dir()), "Project (*.json)")
        if not name:
            return
        self.load_project_file(name)

    def load_project_file(self, name) -> None:
        self.project = EventProject.load(name)
        self.project_path = Path(name)
        self.inventory = self.project.inventory
        self.folder.setText(self.project.raw_folder)
        self.c2.setText(self.project.second_contact)
        self.c3.setText(self.project.third_contact)
        self._show_event()
        for edit, files in ((self.bias_dir, self.project.bias_files), (self.dark_dir, self.project.dark_files),
                            (self.flat_dir, self.project.flat_files)):
            edit.setText(str(Path(files[0]).parent) if files else "")
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
        site = QGroupBox("Event (set on the Frames tab)")
        sl = QVBoxLayout(site)
        self.event_label = QLabel("")
        self.event_label.setWordWrap(True)
        sl.addWidget(self.event_label)
        self.mirrored = QCheckBox("Optics mirror the image (telescope with a star diagonal)")
        sl.addWidget(self.mirrored)
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

        _tip(self.event_label, "The eclipse, site, camera time zone and clock error used to find the Sun's "
             "true centre. They are entered once, on the Frames tab, where they also give the contact times.")
        _tip(self.mirrored, "Tick for telescopes used with a star diagonal, which mirror the image. Camera lenses "
             "do not. It changes where the Sun's centre is computed to be.")
        _tip(self.rotation, "Field rotation between frames. drift (default): a smooth change over time, right "
             "for tracking mounts. per-frame: measured separately for each frame. none: no rotation.", f2)
        _tip(self.any_angle, "Search for large rotations and scale changes, for example from an alt-az mount. "
             "Slower; leave off for tracking mounts.")
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
        self.event_label.setText(_event_summary(project))
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
            opts = W.AlignOptions(mirrored=self.mirrored.isChecked(), rotation=self.rotation.currentText(),
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
        self.flat_mode = QComboBox()
        self.flat_mode.addItem("Master flat", "master")
        self.flat_mode.addItem("Smooth profile from the flats", "profile")
        self.flat_mode.addItem("Master flat + compare with profile", "compare")
        f.addRow("Flat frames:", self.flat_mode)
        self.settings_layout.addWidget(opts)

        _tip(self.full, "The output is the same size as the RAW files, so it lines up with your originals in "
             "Photoshop and keeps streamers out to the frame edges. Needs about 3-4 GB of memory.")
        _tip(self.square, "Only a square around the Sun: about three times quicker and smaller.")
        _tip(self.radii, "Size of the square: solar radii from the Sun's centre to the edge (5 = a square 10 solar "
             "radii wide).")
        _tip(self.moon_frame, "Inside the lunar disc the stack keeps one frame's data, so the limb stays sharp and "
             "in one place. Default: the alignment reference. A longer exposure shows more earthshine.", f)
        _tip(self.include_rejected, "Frames whose Moon is off the track the others follow are usually shaken or "
             "doubled and are left out. Tick to stack them anyway.")
        _tip(self.flat_mode, "How the flat frames chosen on the Frames tab are used (only flats taken at the same "
             "aperture and focal length as the light frames are used; the log says which correction each frame "
             "got). Master flat: divide by the average of your flats, pixel by pixel, which corrects vignetting "
             "AND dust shadows; best when the flats were taken the same day without changing or cleaning "
             "anything. Smooth profile: fit only the lens's smooth vignetting to the flats and ignore dust; use it "
             "if the dust may have moved since (it can then add dark rings instead of removing them). Master flat "
             "+ compare: the master flat, and the log reports how much it differs from the smooth profile; "
             "a FITS image of the difference is saved next to the stack (dark spots in it are dust). Darks and "
             "bias frames from the Frames tab are always used when they match. Changing this needs a re-stack.",
             f)
        _tip(self.hot, "Replace single hot or dead pixels with their neighbours' median. Real stars and "
             "prominences, which cover several pixels, are left alone.")
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
        i = self.flat_mode.findData(st.get("flat_mode", "master"))
        if i >= 0:
            self.flat_mode.setCurrentIndex(i)
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
                              include_rejected=self.include_rejected.isChecked(), hot_pixels=self.hot.isChecked(),
                              flat_mode=self.flat_mode.currentData())
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
               "adaptive": "Adaptive (MGN)", "achf": "Adaptive (ACHF)", "blend": "Blend"}


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
            c.setChecked(k != "achf")  # Adaptive (ACHF) takes minutes on a full frame: on request
            c.toggled.connect(self.changed)
            ll.addWidget(c)
            self.layer_checks[k] = c
        self.settings_layout.addWidget(lay)

        blend = QGroupBox("Blend (filter layers over the natural layer)")
        bf = QFormLayout(blend)
        self.blend = {}
        self.blend_mode = {}
        for k, default in (("fnrgf_n", 0), ("adaptive", 0), ("achf", 0), ("fnrgf", 0), ("nrgf", 0)):
            sp = self._spin(0, 100, default, " %")
            mode = QComboBox()
            mode.addItems(["Soft Light", "Overlay"])
            mode.currentIndexChanged.connect(self.changed)
            row = QHBoxLayout()
            row.addWidget(sp, 1)
            row.addWidget(mode)
            bf.addRow(f"{LAYER_NAMES[k]}:", row)
            self.blend[k] = sp
            self.blend_mode[k] = mode
        self.settings_layout.addWidget(blend)

        tone = QGroupBox("Natural layer and colour")
        tf = QFormLayout(tone)
        self.flatten = self._dspin(0.0, 1.0, 0.0, 0.05, "How much of the brightness fall-off from the Sun to remove")
        tf.addRow("Flatten:", self.flatten)
        self.shoulder = self._dspin(0.0, 0.6, 0.0, 0.05, "How softly the brightest parts roll off to white")
        tf.addRow("Highlight roll-off:", self.shoulder)
        self.even = self._dspin(0.0, 1.0, 0.0, 0.05, "Even out streamers against coronal holes")
        tf.addRow("Evenness:", self.even)
        self.midtone = self._dspin(0.0, 0.9, 0.0, 0.05, "Where the inner corona's middle brightness sits")
        tf.addRow("Midtone:", self.midtone)
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

        adaptive = QGroupBox("Adaptive layer (MGN)")
        af = QFormLayout(adaptive)
        self.mgn_k = self._dspin(0.05, 3.0, 0.7, 0.01, "Adaptive layer: strength of the local contrast")
        self.mgn_k.setDecimals(2)
        af.addRow("Strength:", self.mgn_k)
        self.noise_gate = self._dspin(0.0, 6.0, 0.0, 0.5, "Adaptive layer: keep noise from becoming texture")
        af.addRow("Noise gate:", self.noise_gate)
        self.settings_layout.addWidget(adaptive)
        self._adaptive_form = af

        fn = QGroupBox("FNRGF (Fourier Normalizing-Radial-Graded Filter)")
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

        layer_tips = {
            "natural": "Natural: a logarithmic (log) stretch with most of the brightness fall-off from the Sun removed. The closest to what "
                       "the eye saw, and the base layer of the blend. Keeps the colour.",
            "nrgf": "NRGF = Normalizing-Radial-Graded Filter (Morgan, Habbal & Woo 2006). 'Radial-graded': it "
                    "works outward from the Sun, circle by circle. 'Normalizing': every circle is brought to "
                    "the same average brightness and contrast. Simple and honest; bright streamers stay brighter than "
                    "coronal holes at the same height.",
            "fnrgf": "FNRGF = Fourier Normalizing-Radial-Graded Filter (Druckmüllerová, Morgan & Habbal 2011). "
                     "Like NRGF, but the average and contrast are measured in pieces around each circle and "
                     "smoothed with a Fourier series (a sum of waves around the circle), so contrast is also "
                     "evened out around each circle, so faint regions get as much contrast as bright streamers at the same height.",
            "fnrgf_n": "FNRGF-N = FNRGF with Noise compensation (from the same 2011 paper): FNRGF that discounts "
                       "the noise, so the faint outer corona is enhanced as strongly "
                       "as the inner corona without turning into grain. Usually the best structure layer.",
            "achf": "Fine detail with Druckmüller's adaptive circular high-pass filter (Druckmüller, Rušin & "
                    "Minarovjech 2006; defined in H. Druckmüllerová's thesis, 2013): at each of the same six scales "
                    "as Adaptive (MGN), each pixel minus its local mean, taken with a kernel that bends along the "
                    "circle round the Sun and leaves the Moon out. Amplifies detail without normalizing its "
                    "contrast, so it keeps more of the difference between bright and faint structure than "
                    "Adaptive (MGN), and more noise in the faint corona. Slow: a few minutes on a full frame, "
                    "about 10 s on the preview, so it is only computed when ticked, blended or shown.",
            "adaptive": "Fine detail (loops, plumes, streamer edges) at several scales and every direction. "
                        "Method: MGN = Multi-scale Gaussian Normalization (Morgan & Druckmüller 2014), the "
                        "published successor to Druckmüller's ACHF = Adaptive Circular High-pass Filter. "
                        "Use lightly; it flattens the overall brightness.",
        }
        for i in range(self.show_layer.count()):
            k = self.show_layer.itemData(i)
            if k in layer_tips:
                self.show_layer.setItemData(i, layer_tips[k], Qt.ItemDataRole.ToolTipRole)
        _tip(self.show_layer, "Which layer the preview shows. Blend = the suggested mix of the layers. "
             "Rest the mouse on a layer name in the list, or on its checkbox, for what it is.")
        for k, c in self.layer_checks.items():
            _tip(c, layer_tips[k] + "\n\nTicked layers are written as 16-bit TIFFs by 'Write full-size layers'. "
                 "The preview always shows every layer.")
        blend_tip = ("How strongly this layer is applied over the natural layer in the blend, like layer "
                     "opacity in Photoshop. 0 % leaves it out. The mode beside it picks Soft Light or Overlay.\n\n"
                     "Both modes treat mid-grey as 'no change', so the filter layers add structure without "
                     "lifting the dark sky. Default 0 % for every layer (the blend is then the natural layer); "
                     "try FNRGF-N 50-70 % and adaptive 30-50 %, Soft Light.")
        for k, sp in self.blend.items():
            _tip(sp, f"{LAYER_NAMES[k]}: " + blend_tip, bf)
            _tip(self.blend_mode[k], "How this layer is applied, as Photoshop's layer modes. Both treat mid-grey "
                 "as 'no change', so the sky stays dark.\n\nSoft Light (default): gentle, keeps the natural "
                 "tones.\n\nOverlay: stronger, adds more contrast, mostly in the mid-tones. Use a lower "
                 "strength with Overlay (for example 5-20 %).")
        af = self._adaptive_form
        _tip(self.mgn_k, "How strongly the adaptive layer brings out local contrast at each scale: k in the "
             "paper's arctan(k x C) (Morgan & Druckmüller 2014). Default 0.7, the paper's value. Higher gives "
             "harder, more graphic strands; lower is gentler. Steps of 0.01.", af)
        _tip(self.noise_gate, "The adaptive filter brings out structure at six scales; at the finest ones a stack is "
             "mostly noise, which it turns into grain. With the gate, each scale's detail counts only as far as it "
             "stands above the noise measured in your stack at that scale (the number is how many times the noise "
             "counts as clearly above it).\n\n0 (default): off, the paper's filter as published. 3 is a good start; "
             "higher keeps more of the faint corona smooth, lower shows finer detail and more grain.", af)
        _tip(self.even, "Takes out this fraction of the large brightness differences round the Sun (bright "
             "streamers against dark coronal holes, over about a quarter of the Sun's radius), as Flatten does for "
             "the fall-off from the Sun. 0 (default): off. 0.6-0.8 gives the even grey corona of a strongly "
             "processed eclipse picture; the strands themselves stay.")
        _tip(self.midtone, "Lightens or darkens the middle tones (a midtones transfer curve; black and white stay) so that the inner "
             "corona's median brightness (1.05-2.5 solar radii) sits at this level. 0 (default): off. 0.6 gives a "
             "light grey corona; 0.4 a darker one.")
        _tip(self.shoulder, "How the brightest parts of the picture (the inner corona by the limb, bright "
             "prominences) reach white.\n\n0 (default): a hard white point, so the brightest 0.1% clip to flat "
             "white. Above 0 that top part of the range rolls off smoothly instead, so it keeps its gradations and "
             "nothing clips; prominences keep their colour instead of turning flat pink. 0.3 is a good start; "
             "higher compresses more of the bright range.")
        _tip(self.flatten, "How much of the corona's brightness fall-off from the Sun the natural layer removes, "
             "from 0 (none: a plain log stretch, the inner corona dominates) to 1 (all of it: equal brightness at "
             "every height).\n\n0 (default): off. 0.5-0.7 keeps much of the natural fall-off; 0.8-1 shows more "
             "of the outer corona. It stops where the corona sinks into the noise, so the sky stays "
             "dark.", tf)
        _tip(self.saturation, "Colour strength in every layer: 0 = black and white, 1 = the measured colour, "
             "above 1 = stronger. Colour fades to grey on its own where the corona is too faint for its colour to "
             "be trusted.", tf)
        _tip(self.white, "corona (default): the inner corona, 1.2-2.5 solar radii, is made white. Its light is "
             "sunlight scattered by electrons, so it really is white, which makes this the most reliable "
             "choice.\n\ncamera: the camera's as-shot white balance.\n\nnone: raw camera colour (looks green).",
             tf)
        _tip(self.keep_sky, "By default the sky (twilight glow, its gradient across the frame and small "
             "black-level offsets) is modelled for each colour and removed before filtering; otherwise the "
             "filters turn it into arcs and colour casts. Tick this to compare without it.")
        _tip(self.order, "How closely FNRGF's 'local average' follows brightness changes around each circle.\n\n"
             "Lower (3-6): bright streamers stay clearly brighter than dark coronal holes; closer to plain NRGF, "
             "with more of the corona's overall shape.\n\nHigher (14-20): faint regions are boosted to match "
             "bright ones. Too high and the streamers count as part of the average and get subtracted, so they "
             "look flat and blotchy.\n\nDefault 10 (the value in the FNRGF paper). Lower it if the result looks "
             "too uniform; raise it if dim regions beside bright streamers still look empty. If you go above "
             "about 15, raise Segments too.", ff)
        _tip(self.segments, "How finely each circle is divided to measure its average and contrast. 50 means "
             "pieces of 7.2 degrees.\n\nMore segments: finer but noisier measurements. Fewer: smoother.\n\n"
             "Default 50 (the FNRGF paper). Rarely needs changing, but it must stay well above 2 x Order + 1 "
             "(21 at order 10): at order 20, use about 80-100.", ff)
        _tip(self.noise_m, "How much of the noise is discounted when FNRGF-N measures contrast, 0 to 0.99.\n\n"
             "Far from the Sun much of the measured variation is just noise, so plain FNRGF under-enhances the "
             "faint outer corona. This subtracts the estimated noise first.\n\n0: plain FNRGF, smooth but weak "
             "far out. Higher (0.8-0.95): more faint streamers and plumes, and more grain.\n\nDefault 0.7. This "
             "is the main setting to try: compare 0.5 and 0.9 in the preview. Stacks of many frames can go "
             "higher; noisier ones may need 0.3-0.5.", ff)
        _tip(self.preview_btn, "Runs every layer on a reduced copy of the stack (about 1400 px wide), in seconds. "
             "Nothing is written.")
        _tip(self.run_btn, "Runs the ticked layers at full resolution and writes them, plus the blend, as "
             "16-bit TIFFs next to the project file. Takes a few minutes for a full frame.")

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
        modes = {k: c.currentText().lower() for k, c in self.blend_mode.items() if k in blend}
        return EnhanceParams(layers=layers, segments=self.segments.value(), order=self.order.value(),
                             noise_m=self.noise_m.value(), flatten=self.flatten.value(), shoulder=self.shoulder.value(), even=self.even.value(), midtone=self.midtone.value(), noise_gate=self.noise_gate.value(), mgn_k=self.mgn_k.value(),
                             saturation=self.saturation.value(), white=self.white.currentText(), blend=blend,
                             blend_modes=modes, neutral_background=not self.keep_sky.isChecked())

    def changed(self, *_args) -> None:
        if self.auto.isChecked() and self.small is not None:
            self.timer.start()

    def preview(self) -> None:
        if self.worker is not None:
            self._dirty = True  # run again when the current preview finishes
            return
        params = self.params()
        cached = self.small
        # Adaptive (ACHF) is slow (about 10 s even on the preview): only when it is wanted.
        want_achf = (self.layer_checks["achf"].isChecked() or self.blend["achf"].value() > 0
                     or self.show_layer.currentData() == "achf")
        preview_layers = tuple(k for k in LAYERS if k != "achf" or want_achf)

        def job(path, say, cancel):
            small = cached
            if small is None:
                say("Reading the HDR stack for the preview…")
                small = W.reduce_for_preview(W.load_hdr(path), max_side=1400)
            # The preview shows every layer the blend needs, whatever is ticked for writing.
            from dataclasses import replace

            p = replace(params, layers=preview_layers)
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
        elif k == "achf" and self.small is not None:
            self.preview()  # Adaptive (ACHF) is computed on request: chosen here, so now
        elif self.outputs.get(k):
            self.view.show_file(self.outputs[k])

    def run(self) -> None:
        params = self.params()
        if not params.layers:
            QMessageBox.information(self, "Enhance", "Tick at least one layer to write.")
            return
        target = self._ask_layers_name()
        if target is None:
            return
        out_dir, name, fmt = target
        self.start("Writing full-size layers",
                   lambda p, say, cancel: W.enhance_project(p, params, output=out_dir, name=name, say=say,
                                                            cancel=cancel, fmt=fmt), self._done)

    def _ask_layers_name(self):
        """Where to write the layers: a folder and a name; each layer is <name>_<layer>.tif."""
        proj = self.main.ensure_saved()
        if proj is None:
            return None
        last = self.outputs.get("blend") or next(iter(self.outputs.values()), "")
        if last:
            default = Path(last).with_name(Path(last).name.rsplit("_", 1)[0] + ".tif")
        else:
            default = Path(proj).with_suffix(".tif")
        fname, filt = QFileDialog.getSaveFileName(
            self, "Save layers as (each layer is saved as <name>_<layer>)", str(default), IO.FILTER)
        if not fname:
            return None
        f = _with_ext(fname, filt)
        stem = f.stem
        for k in list(LAYER_NAMES):
            if stem.endswith("_" + k):
                stem = stem[: -len(k) - 1]
        return str(f.parent), stem, f.suffix.lstrip(".")

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
# Earthshine
# ---------------------------------------------------------------------------


class EarthshineTab(StepTab):
    def __init__(self, window: "MainWindow") -> None:
        super().__init__(window)
        def spin(lo, hi, step, val, suffix=""):
            w = QDoubleSpinBox()
            w.setRange(lo, hi)
            w.setSingleStep(step)
            w.setValue(val)
            if suffix:
                w.setSuffix(suffix)
            return w

        # 1. Which frames and channels make the Moon-aligned stack.
        stack_box = QGroupBox("1  Stack the Moon")
        fs = QFormLayout(stack_box)
        self.stops = spin(0.5, 15.0, 0.5, 6.0, " stops")
        fs.addRow("Use frames within:", self.stops)
        self.channel = QComboBox()
        for label, key in (("Automatic", "auto"), ("Green", "green"), ("Green + blue", "green+blue"),
                           ("All (average)", "all"), ("Blue", "blue")):
            self.channel.addItem(label, key)
        fs.addRow("Colour channels:", self.channel)
        self.include_rejected = QCheckBox("Include frames flagged as off the Moon's track")
        fs.addRow(self.include_rejected)

        # 2. The seas and highlands: glare taken away, then stretched and shaded.
        box = QGroupBox("2  Lunar surface (seas and highlands)")
        f = QFormLayout(box)
        self.glare = QSpinBox()
        self.glare.setRange(1, 6)
        self.glare.setValue(2)
        f.addRow("Glare smoothness:", self.glare)
        self.limb_glare = QCheckBox("Remove the bright glare along the limb")
        self.limb_glare.setChecked(True)
        f.addRow(self.limb_glare)
        self.source_glare = QCheckBox("Remove the glare fanning out from a bright prominence")
        self.source_glare.setChecked(True)
        f.addRow(self.source_glare)
        self.limb_shape = QCheckBox("Keep maria by the limb (glare only fades inward)")
        self.limb_shape.setChecked(True)
        f.addRow(self.limb_shape)
        arow = QHBoxLayout()
        self.albedo_path = QLineEdit()
        self.albedo_path.setPlaceholderText("none (optional)")
        abrowse = QPushButton("Browse…")
        abrowse.clicked.connect(self._pick_albedo)
        arow.addWidget(self.albedo_path, 1)
        arow.addWidget(abrowse)
        f.addRow("Albedo map:", arow)
        self.albedo_strength = spin(0.0, 1.0, 0.1, 0.5)
        f.addRow("Map guidance:", self.albedo_strength)
        self.smooth = spin(0.0, 5.0, 0.5, 1.0, " px")
        f.addRow("Smoothing:", self.smooth)
        self.edge_smooth = spin(0.0, 12.0, 0.5, 3.0, " px")
        f.addRow("Smoothing near the limb:", self.edge_smooth)
        self.surface_denoise = spin(0.0, 10.0, 0.5, 0.0, " px")
        f.addRow("Edge-preserving smoothing:", self.surface_denoise)
        self.albedo_texture = spin(0.0, 1.5, 0.1, 0.0)
        f.addRow("Map texture (composite):", self.albedo_texture)
        self.edge_grain = QCheckBox("Grain in the filled edge, like the rest of the surface")
        self.edge_grain.setChecked(True)
        f.addRow(self.edge_grain)
        self.brightness = spin(0.1, 1.0, 0.05, 0.6)
        f.addRow("Brightness:", self.brightness)
        self.limb_dark = spin(0.0, 1.0, 0.05, 0.6)
        f.addRow("Darker towards the limb:", self.limb_dark)
        self.vignette_shape = spin(1.0, 4.0, 0.25, 1.5)
        f.addRow("Gradient shape:", self.vignette_shape)
        self.colour = QCheckBox("Natural colour (the earthshine's own tint)")
        self.colour.setChecked(True)
        f.addRow(self.colour)

        # Only for the separate earthshine layer (pasting it in, or Photoshop).
        layer_box = QGroupBox("Earthshine layer for Photoshop / pasting only")
        fl = QFormLayout(layer_box)
        self.keep_glare = spin(0.0, 1.0, 0.05, 0.5)
        fl.addRow("Keep glare (glow):", self.keep_glare)
        self.limb_soften = spin(0.0, 6.0, 0.5, 1.5, " px")
        fl.addRow("Soften the limb:", self.limb_soften)
        fl.addRow(QLabel("Not used by the final image below: there the limb is the stack's own."))
        self.layer_box = layer_box

        self.settings_layout.addWidget(stack_box)
        self.settings_layout.addWidget(box)
        run = QPushButton("Stack earthshine (makes the Moon and its surface)")
        run.clicked.connect(self.run)
        self.run_buttons.append(run)
        self.settings_layout.addWidget(run)
        self.open_btn = QPushButton("Open earthshine layer in Siril")
        self.open_btn.clicked.connect(lambda: self.main.open_in_siril(self.layer_path))
        self.settings_layout.addWidget(self.open_btn)

        final = QGroupBox("3  Final image: Moon and corona together")
        ff = QFormLayout(final)
        self.base = QComboBox()
        self.base.addItem("Blend (Enhance settings)", "blend")
        self.base.addItem("Natural (Enhance settings)", "natural")
        self.base.addItem("Paste into a file… (older way)", "file")
        self.base_file = ""
        self.base.activated.connect(self._base_chosen)
        ff.addRow("Corona image:", self.base)
        self.base_label = QLabel("")
        self.base_label.setWordWrap(True)
        ff.addRow(self.base_label)
        self.strength = QDoubleSpinBox()
        self.strength.setRange(0.0, 3.0)
        self.strength.setSingleStep(0.1)
        self.strength.setValue(1.0)
        ff.addRow("Lunar surface strength:", self.strength)
        self.mode = QComboBox()
        self.mode.addItems(["screen", "lighten", "add"])
        ff.addRow("Blend mode (pasting only):", self.mode)
        self.mode.setEnabled(False)
        _tip(self.base, "Blend or Natural: the Moon-aligned stack goes into the HDR stack inside the lunar disc, "
             "and Moon and corona are stretched together with the settings of your last Enhance run, so the limb "
             "is the stack's own: one curve, one colour, no seam. The seas and highlands are then added inside. "
             "'Paste into a file' is the older way: the earthshine layer pasted into any image with a blend mode.",
             ff)
        _tip(self.strength, "Brightness of the seas and highlands added inside the Moon, 1 = as made by the "
             "earthshine step (its Brightness setting).", ff)
        fb = QHBoxLayout()
        self.final_preview_btn = QPushButton("Preview")
        self.final_preview_btn.clicked.connect(self.preview_final)
        self.final_btn = QPushButton("Write final image")
        self.final_btn.clicked.connect(self.write_final)
        fb.addWidget(self.final_preview_btn)
        fb.addWidget(self.final_btn)
        ff.addRow(fb)
        self.open_final_btn = QPushButton("Open final image in Siril")
        self.open_final_btn.clicked.connect(lambda: self.main.open_in_siril(self.final_path))
        ff.addRow(self.open_final_btn)
        self.settings_layout.addWidget(final)
        self.settings_layout.addWidget(self.layer_box)
        self.run_buttons += [self.final_preview_btn, self.final_btn]
        self.final_path = ""
        self.final_timer = QTimer(self)
        self.final_timer.setSingleShot(True)
        self.final_timer.setInterval(500)
        self.final_timer.timeout.connect(self.preview_final)
        self._final_shown = False
        for w in (self.strength,):
            w.valueChanged.connect(self._final_changed)
        for w in (self.base, self.mode):
            w.currentIndexChanged.connect(self._final_changed)
        _tip(final, "Puts the earthshine into the lunar disc of one of the enhanced corona layers and writes the "
             "finished picture as a 16-bit TIFF (<project>_final.tif), so Photoshop is optional. Needs the "
             "full-size layers from the Enhance tab and the earthshine stack above.")
        _tip(self.base, "The corona picture the earthshine goes into.\n\nLast-written blend (default): the blend "
             "most recently written on the Enhance tab.\n\nLast-written Natural: the natural layer alone, for a picture "
             "without the filter layers.\n\nFrom file…: pick any "
             "image, for example a blend you saved earlier under another name, or one you finished elsewhere. "
             "It must be the full size of the HDR stack (not cropped or resized), so the Moon lines up.", ff)
        _tip(self.strength, "How bright the earthshine is in the final picture: 1 = as stacked, 0 = a black "
             "Moon, above 1 brightens a faint Moon (and its noise).", ff)
        _tip(self.mode, "How the two are combined, as Photoshop's layer modes. The corona is black inside the "
             "disc and the earthshine black outside it, so they differ only at the soft edge of the limb. "
             "screen (default): smooth and never clips; lighten: the brighter of the two; add: the sum "
             "(can clip at the limb).", ff)
        self.settings_layout.addStretch(1)
        self.layer_path = ""
        self.view = ImageView()
        self.results_layout.addWidget(self.view, 1)

        _tip(box, "Earthshine is the Moon's night side lit by sunlight reflected from the Earth. The Moon moves "
             "across the Sun during totality, so this stacks the frames aligned on the Moon instead of the "
             "corona, removes the glare from the bright inner corona, and writes a layer the same size as the "
             "corona layers: the lunar surface inside the disc, black outside. In Photoshop put it under the "
             "corona layers in Screen or Lighten mode, or paste it into the black disc.")
        _tip(self.stops, "Only the longer exposures show earthshine; shorter ones add noise only. Frames more than "
             "this many stops below the longest exposure are skipped. Default 6.", fs)
        _tip(self.limb_glare, "Close to the edge of the Moon, light scattered from the bright inner corona rises "
             "steeply, far faster than the glare surface above can follow. Left in, it shows as a bright ring "
             "inside the limb and leaves the middle of the Moon dark. Leave this on unless you want to see that "
             "raw glare.")
        _tip(self.source_glare, "A bright prominence throws a fan of glare, with diffraction spikes, onto the "
             "Moon. It is fitted from the prominence itself (a pattern of directions times a fall-off with "
             "distance), which the lunar surface does not follow, and taken away. Off, the fan was covered with a "
             "smooth patch instead.")
        _tip(self.albedo_path, "Optional: a map of the Moon's surface brightness, in the usual 2:1 world-map "
             "layout with 0 degrees longitude in the middle (for example lroc_color_2k.jpg from NASA's CGI Moon "
             "Kit, svs.gsfc.nasa.gov/4720). It helps tell maria near the limb from glare. It is turned to match your "
             "Moon automatically and only guides the glare removal: nothing from the map is added to your picture.",
             f)
        _tip(self.albedo_strength, "How much the albedo map guides the glare removal near the limb, from 0 (not at "
             "all) to 1. Where your data can tell maria from glare, it overrides the map; the map only decides what "
             "the data cannot, in the outer ~30 px. On tests a mare the map has but the Moon does not came through "
             "at about 15-20% of this strength 25 px inside the limb, and not at all further in.", f)
        _tip(self.limb_shape, "Glare from just outside the Moon can only get fainter going inward, while a sea "
             "(mare) near the edge is a dark dip and highlands a bright bump. With this on, only light that "
             "fades inward is taken away around the edge, so seas near the limb such as Crisium survive.")
        _tip(self.channel, "Which colour channels the earthshine is made from. Much of the glare that changes "
             "quickly around the edge is deep red light from the pink chromosphere and prominences, and a colour "
             "camera has twice as many green pixels, so green usually gives a cleaner Moon than the average of all "
             "three. Automatic tries all, green and green + blue on your stack and picks the cleanest; the log "
             "shows the scores.", fs)
        _tip(self.glare, "How flexible the glare surface removed from the disc is (polynomial order). Glare from "
             "the corona is a gentle gradient, so 2 (default) is usually right. Higher removes uneven glare but "
             "also starts removing large maria; 1 removes a straight gradient only.", f)
        _tip(self.smooth, "Softens the noise in the layer, in pixels. Earthshine is faint, so a little smoothing "
             "(default 1) helps; 0 for none, 2-3 for very noisy data.", f)
        _tip(self.limb_dark, "A vignette across the Moon: the middle stays at full brightness and it darkens "
             "gradually all the way out to the edge. 0 = evenly lit, 0.6 = the edge at 40% of the middle (the "
             "default), 1 = fading to black at the edge. Removing the glare also removes the Moon's overall "
             "shading; this gives back a brighter middle and keeps the noisier last pixels by the edge dim.", f)
        _tip(self.keep_glare, "How much of the glare that was taken away is put back as a soft glow just inside "
             "the edge of the Moon, fading inward: the inner corona's light spilling over the dark Moon, as the eye "
             "sees it at totality. 0 = none, 0.5 = a soft glow (default), 1 = all of it (compressed so it cannot "
             "clip). The edge itself always comes from your data: the Moon's own outline, with its mountains and "
             "softness, not a drawn circle.", fl)
        _tip(self.limb_soften, "Extra softening of the Moon's edge, in pixels, on top of the softness it already "
             "has in your frames. 0 = none.", fl)
        _tip(self.colour, "Keep the colour of the light at the edge of the Moon, so a prominence or the pink "
             "chromosphere stays red where it spills over the limb, and give the lunar surface the earthshine's "
             "own measured tint (white balanced like the corona). Off: a grey layer.")
        _tip(self.edge_grain, "The last few pixels by the edge are only noise and are filled from just inside; "
             "this gives them grain like the rest of the surface, so they do not look smoother.")
        _tip(self.vignette_shape, "How the darkening is spread from the middle to the edge. 1 = a straight-line "
             "fall from the centre; 1.5 (the default) = gentle in the middle, steady further out; 3 = the middle "
             "stays bright longer and the darkening gathers towards the edge.", f)
        _tip(self.surface_denoise, "Smooths the lunar surface only across similar brightness, so the noise inside "
             "the maria goes while their edges stay crisp (unlike Smoothing, which blurs edges too). Size in pixels; "
             "0 (default) = off. 3-5 suits a 400 mm frame. Needs Stack earthshine again.")
        _tip(self.albedo_texture, "Adds the albedo map's own craters and rays to the lunar surface, scaled to your "
             "Moon's contrast. This is a COMPOSITE: that fine detail comes from the map (a spacecraft mosaic), not "
             "from your photographs; a typical earthshine stack is mostly noise at that scale. Needs an albedo map "
             "above, and is only used if the map matches your Moon; the log says when it is added.\n\n"
             "0 (default) = off, your data only. 1 = the map's detail at your Moon's own contrast.")
        _tip(self.edge_smooth, "The last pixels by the limb are much noisier (the long exposures are saturated "
             "there), so they get extra smoothing, up to this many pixels where the noise is highest. The rest "
             "of the Moon only gets the Smoothing above. Lower it to keep more grain at the edge.", f)
        _tip(self.brightness, "How bright the brightest lunar surface is in the layer, from 0 to 1. Earthshine is "
             "much fainter than the corona, so the default 0.6 keeps it subdued; adjust to taste in Photoshop.", f)
        _tip(self.include_rejected, "Frames whose Moon is off the track the others follow are usually shaken; "
             "they would blur the lunar surface.")

    def _base_chosen(self, *_a) -> None:
        if self.base.currentData() == "file":
            start = self.base_file or str(self.main.frames_tab.project_dir())
            fname, _ = QFileDialog.getOpenFileName(self, "Corona image to blend the earthshine into", start,
                                                   "Images (*.tif *.tiff *.png)")
            if fname:
                self.base_file = fname
            elif not self.base_file:
                self.base.setCurrentIndex(0)  # nothing chosen: back to the last-written blend
        self._show_base()
        self.mode.setEnabled(self.base.currentData() == "file")  # the blend mode only matters when pasting
        self._final_changed()

    def _pick_albedo(self) -> None:
        start = self.albedo_path.text().strip() or str(self.main.frames_tab.project_dir())
        fname, _ = QFileDialog.getOpenFileName(self, "Lunar albedo map", start, "Images (*.jpg *.jpeg *.tif *.tiff *.png)")
        if fname:
            self.albedo_path.setText(fname)

    def _show_base(self) -> None:
        self.base_label.setText(f"File: {Path(self.base_file).name}" if self.base.currentData() == "file"
                                and self.base_file else "")

    def _base_args(self):
        if self.base.currentData() == "file":
            return "blend", self.base_file or None
        return self.base.currentData(), None

    def _final_changed(self, *_a) -> None:
        if self._final_shown:
            self.final_timer.start()

    def preview_final(self) -> None:
        if self.worker is not None:
            return
        (base, bfile), k, mode = self._base_args(), self.strength.value(), self.mode.currentText()
        if bfile:  # an image from elsewhere: the older way, pasting the earthshine layer in
            fn = lambda p, say, cancel: W.final_image(p, base, k, mode, preview_side=1400, say=say, cancel=cancel,
                                                      base_file=bfile)
        else:  # Moon and corona stretched together from the stacks
            fn = lambda p, say, cancel: W.final_with_moon(p, base, k, preview_side=1400, say=say, cancel=cancel)
        self.start("Preview of the final image", fn, self._final_preview_done, reload=False)

    def _final_preview_done(self, img) -> None:
        self._final_shown = True
        self.view.show_array(img)

    def write_final(self) -> None:
        (base, bfile), k, mode = self._base_args(), self.strength.value(), self.mode.currentText()
        proj = self.main.ensure_saved()
        if proj is None:
            return
        default = self.final_path or str(Path(proj).with_name(Path(proj).stem + "_final.tif"))
        fname, filt = QFileDialog.getSaveFileName(self, "Save final image as", default, IO.FILTER)
        if not fname:
            return
        fname = str(_with_ext(fname, filt))
        if bfile:
            fn = lambda p, say, cancel: W.final_image(p, base, k, mode, say=say, cancel=cancel, output_file=fname,
                                                      base_file=bfile)
        else:
            fn = lambda p, say, cancel: W.final_with_moon(p, base, k, say=say, cancel=cancel, output_file=fname)
        self.start("Writing the final image", fn, self._final_done)

    def _final_done(self, path) -> None:
        self.final_path = path
        self.open_final_btn.setEnabled(True)
        self.log.appendPlainText(f"Final image written to {path}")
        QMessageBox.information(self, "Final image", f"Wrote\n{path}")

    def load_project(self, project: EventProject, path: Path) -> None:
        es = project.parameters.get("earthshine", {})
        if es.get("albedo_map"):
            self.albedo_path.setText(W.resolve(path, es["albedo_map"]) or es["albedo_map"])
        if es.get("albedo_strength") is not None:
            self.albedo_strength.setValue(float(es["albedo_strength"]))
        fin = project.parameters.get("final", {})
        self.final_path = W.resolve(path, fin.get("file"))
        self.open_final_btn.setEnabled(bool(self.final_path))
        self.base_file = W.resolve(path, fin.get("base_file")) if fin.get("base_file") else ""
        key = "file" if self.base_file else fin.get("base")
        if key:
            i = self.base.findData(key)
            if i >= 0:
                self.base.setCurrentIndex(i)
        self._show_base()
        if fin.get("mode"):
            self.mode.setCurrentText(fin["mode"])
        if fin.get("strength") is not None:
            self.strength.setValue(float(fin["strength"]))
        self._final_shown = False
        e = project.parameters.get("earthshine", {})
        self.layer_path = W.resolve(path, e.get("earthshine"))
        self.open_btn.setEnabled(bool(self.layer_path))
        crop = W.resolve(path, e.get("crop"))
        if crop:
            self.view.show_file(crop)

    def run(self) -> None:
        from tse_core.earthshine import EarthshineParams

        params = EarthshineParams(stops=self.stops.value(), glare_order=self.glare.value(),
                                  smooth=self.smooth.value(), brightness=self.brightness.value(),
                                  limb_glare=self.limb_glare.isChecked(),
                                  source_glare=self.source_glare.isChecked(), limb_darkening=self.limb_dark.value(),
                                  vignette_shape=self.vignette_shape.value(),
                                  keep_glare=self.keep_glare.value(), limb_soften=self.limb_soften.value(),
                                  edge_grain=self.edge_grain.isChecked(), colour=self.colour.isChecked(),
                                  edge_smooth=self.edge_smooth.value(),
                                  surface_denoise=self.surface_denoise.value(),
                                  albedo_texture=self.albedo_texture.value(), limb_shape=self.limb_shape.isChecked(),
                                  channel=self.channel.currentData(),
                                  albedo_map=self.albedo_path.text().strip(),
                                  albedo_strength=self.albedo_strength.value())
        inc = self.include_rejected.isChecked()
        proj = self.main.ensure_saved()
        if proj is None:
            return
        default = self.layer_path or str(Path(proj).with_name(Path(proj).stem + "_earthshine.tif"))
        fname, filt = QFileDialog.getSaveFileName(self, "Save earthshine layer as", default, IO.FILTER)
        if not fname:
            return
        f = _with_ext(fname, filt)
        name = f.stem[: -len("_earthshine")] if f.stem.endswith("_earthshine") else f.stem
        out_dir, fmt = str(f.parent), f.suffix.lstrip(".")
        self.start("Stacking earthshine",
                   lambda p, say, cancel: W.earthshine(p, params, include_rejected=inc, output=out_dir, name=name,
                                                       fmt=fmt, say=say, cancel=cancel),
                   self._done)

    def _done(self, result) -> None:
        self.log.appendPlainText(result.describe())
        self.layer_path = result.outputs.get("earthshine", "")
        self.open_btn.setEnabled(bool(self.layer_path))
        self.view.show_file(result.outputs.get("crop", ""))


# ---------------------------------------------------------------------------
# Prominences
# ---------------------------------------------------------------------------


class ProminencesTab(StepTab):
    """Find the prominences by colour, preview them, write them as a layer and put them on top."""

    def __init__(self, window: "MainWindow") -> None:
        super().__init__(window)
        self.region = None
        self.layer_path = ""
        self.final_path = ""
        self.base_file = ""
        self._shown = False
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(600)
        self.timer.timeout.connect(self.preview)

        box = QGroupBox("Find prominences")
        f = QFormLayout(box)
        self.height_spin = self._dspin(0.05, 1.0, 0.3, 0.05, " solar radii")
        f.addRow("Look up to:", self.height_spin)
        self.sensitivity = self._dspin(1.0, 15.0, 4.0, 0.5, " × noise")
        f.addRow("Red excess needed:", self.sensitivity)
        self.min_redness = self._dspin(1.0, 3.0, 1.2, 0.05, "")
        f.addRow("Redness needed:", self.min_redness)
        self.chromo = QCheckBox("Include the chromosphere (thin pink rim)")
        self.chromo.setChecked(True)
        self.chromo.toggled.connect(self.changed)
        f.addRow(self.chromo)
        self.chromo_h = self._dspin(2.0, 40.0, 12.0, 1.0, " arcsec")
        f.addRow("Rim height:", self.chromo_h)
        self.brightness = self._dspin(0.1, 3.0, 1.0, 0.1, "")
        f.addRow("Brightness:", self.brightness)
        self.saturation = self._dspin(0.0, 3.0, 1.2, 0.1, "")
        f.addRow("Saturation:", self.saturation)
        self.settings_layout.addWidget(box)
        row = QHBoxLayout()
        self.preview_btn = QPushButton("Preview")
        self.preview_btn.clicked.connect(self.preview)
        self.write_btn = QPushButton("Write prominence layer")
        self.write_btn.clicked.connect(self.write_layer)
        row.addWidget(self.preview_btn)
        row.addWidget(self.write_btn)
        self.settings_layout.addLayout(row)

        top = QGroupBox("Put them on top")
        tf = QFormLayout(top)
        self.base = QComboBox()
        self.base.addItem("Final image (Earthshine tab)", "final")
        self.base.addItem("Last-written blend", "blend")
        self.base.addItem("Last-written Natural", "natural")
        self.base.addItem("From file…", "file")
        self.base.activated.connect(self._base_chosen)
        tf.addRow("Image:", self.base)
        self.base_label = QLabel("")
        self.base_label.setWordWrap(True)
        tf.addRow(self.base_label)
        self.strength = self._dspin(0.0, 1.0, 1.0, 0.05, "", final=True)
        tf.addRow("Strength:", self.strength)
        self.mode = QComboBox()
        self.mode.addItems(["normal", "screen", "lighten"])
        self.mode.currentIndexChanged.connect(self._final_changed)
        tf.addRow("Blend mode:", self.mode)
        r2 = QHBoxLayout()
        self.final_preview_btn = QPushButton("Preview")
        self.final_preview_btn.clicked.connect(self.preview_final)
        self.final_btn = QPushButton("Write final image")
        self.final_btn.clicked.connect(self.write_final)
        r2.addWidget(self.final_preview_btn)
        r2.addWidget(self.final_btn)
        tf.addRow(r2)
        self.open_btn = QPushButton("Open final image in Siril")
        self.open_btn.clicked.connect(lambda: self.main.open_in_siril(self.final_path))
        tf.addRow(self.open_btn)
        self.settings_layout.addWidget(top)
        self.settings_layout.addStretch(1)
        self.run_buttons += [self.preview_btn, self.write_btn, self.final_preview_btn, self.final_btn]
        self.view = ImageView()
        self.results_layout.addWidget(self.view, 1)
        self._final_shown = False

        _tip(box, "Prominences glow in the red light of hydrogen (with some blue), so they look pink or magenta "
             "once the corona is white. They are found by colour: pixels clearly redder than the white corona "
             "around them. They come from the HDR stack, so every prominence the moving Moon uncovered at any "
             "time during totality is included; the layer stops at the lunar limb, so nothing covers the Moon "
             "or the earthshine.")
        _tip(self.height_spin, "How far above the solar limb to look, in solar radii (the Sun's radius = 1). Most "
             "prominences reach 0.05-0.2; raise it for a tall or erupting one. Default 0.3.", f)
        _tip(self.sensitivity, "How far above the noise the red excess must be. Lower (2-3) finds fainter "
             "threads but may pick up noise specks; higher (6-10) keeps only the clear ones. Default 4.", f)
        _tip(self.min_redness, "How much redder than green a pixel must be (the white corona is about 1.0). Lower "
             "(1.1) catches pale prominences; higher (1.4+) keeps only strongly coloured ones. Default 1.2.", f)
        _tip(self.chromo, "The chromosphere is the thin pink rim along the edge of the Sun, seen near second and "
             "third contact. Untick to leave it out: red light lower than 'Rim height' is removed, except the "
             "bases of taller prominences.")
        _tip(self.chromo_h, "How high the rim reaches above the solar limb when it is left out, in arcseconds. "
             "Default 12; raise it if bits of rim remain, lower it if small prominences disappear.", f)
        _tip(self.brightness, "Brightness of the prominences in the layer (1 = as measured).", f)
        _tip(self.saturation, "Colour strength of the prominences: 1 = as measured, higher = more vivid pink.", f)
        _tip(self.write_btn, "Writes the prominence layer the full size of the stack. TIFF and PNG keep "
             "transparency around the prominences, so in Photoshop it sits on top in Normal mode; a JPG has none, "
             "so it is on black (use Screen or Lighten then).")
        _tip(self.base, "The picture the prominences go on.\n\nFinal image (default): the one written on the "
             "Earthshine tab, with the earthshine.\n\nLast-written blend / Natural: the layers from the Enhance "
             "tab.\n\nFrom file…: any image the full size of the stack.", tf)
        _tip(self.strength, "Opacity of the prominences (Normal mode), or their brightness (Screen, Lighten).", tf)
        _tip(self.mode, "normal (default): the prominences cover what is under them, so their colour shows even "
             "over the bright inner corona. screen and lighten: as in Photoshop; they tend to wash the colour "
             "out where the corona is bright.", tf)

    def _dspin(self, lo, hi, val, step, suffix, final=False) -> QDoubleSpinBox:
        sp = QDoubleSpinBox()
        sp.setRange(lo, hi)
        sp.setSingleStep(step)
        sp.setValue(val)
        sp.setSuffix(suffix)
        sp.valueChanged.connect(self._final_changed if final else self.changed)
        return sp

    def params(self):
        from tse_core.prominences import ProminenceParams

        return ProminenceParams(height=self.height_spin.value(), sensitivity=self.sensitivity.value(),
                                min_redness=self.min_redness.value(), chromosphere=self.chromo.isChecked(),
                                chromosphere_arcsec=self.chromo_h.value(), brightness=self.brightness.value(),
                                saturation=self.saturation.value())

    def load_project(self, project: EventProject, path: Path) -> None:
        self.region = None  # the stack may have changed
        pr = project.parameters.get("prominences", {})
        self.layer_path = W.resolve(path, pr.get("layer"))
        fp = project.parameters.get("final_prominences", {})
        self.final_path = W.resolve(path, fp.get("file"))
        self.open_btn.setEnabled(bool(self.final_path))
        self.base_file = W.resolve(path, fp.get("base_file")) if fp.get("base_file") else ""
        self._shown = self._final_shown = False

    def changed(self, *_a) -> None:
        if self._shown:
            self.timer.start()

    def preview(self) -> None:
        if self.worker is not None:
            return
        params, cached = self.params(), self.region

        def job(path, say, cancel):
            import numpy as np

            from tse_core.prominences import composite_top

            region = cached or W.load_sun_region(path)
            result, layer, alpha = W.prominence_layer(path, params, preview=True, region=region, say=say,
                                                      cancel=cancel)
            # Show them over a dim view of the inner corona, for placement.
            lum = np.nan_to_num(region.rgb).mean(axis=2)
            top = float(np.percentile(lum[lum > 0], 99.5)) if (lum > 0).any() else 1.0
            back = (np.arcsinh(20 * np.clip(lum, 0, None) / top) / np.arcsinh(20) * 0.5)[..., None].repeat(3, 2)
            return region, result, composite_top(back.astype(np.float32), layer, alpha)

        self.start("Finding prominences", job, self._preview_done, reload=False)

    def _preview_done(self, out) -> None:
        self.region, result, img = out
        self._shown = True
        self.log.appendPlainText(result.describe())
        self.view.show_array(img)

    def write_layer(self) -> None:
        proj = self.main.ensure_saved()
        if proj is None:
            return
        default = self.layer_path or str(Path(proj).with_name(Path(proj).stem + "_prominences.tif"))
        fname, filt = QFileDialog.getSaveFileName(self, "Save prominence layer as", default, IO.FILTER)
        if not fname:
            return
        out = str(_with_ext(fname, filt))
        params, cached = self.params(), self.region
        self.start("Writing the prominence layer",
                   lambda p, say, cancel: W.prominence_layer(p, params, output_file=out, region=cached, say=say,
                                                             cancel=cancel), self._layer_done)

    def _layer_done(self, result) -> None:
        self.log.appendPlainText(result.describe())
        self.layer_path = result.outputs.get("prominences", "")

    # --- on top of the final image
    def _base_chosen(self, *_a) -> None:
        if self.base.currentData() == "file":
            start = self.base_file or str(self.main.frames_tab.project_dir())
            fname, _ = QFileDialog.getOpenFileName(self, "Image to put the prominences on", start,
                                                   "Images (*.tif *.tiff *.png *.jpg *.jpeg)")
            if fname:
                self.base_file = fname
            elif not self.base_file:
                self.base.setCurrentIndex(0)
        self.base_label.setText(f"File: {Path(self.base_file).name}" if self.base.currentData() == "file"
                                and self.base_file else "")
        self._final_changed()

    def _base_args(self):
        if self.base.currentData() == "file":
            return "final", self.base_file or None
        return self.base.currentData(), None

    def _final_changed(self, *_a) -> None:
        if self._final_shown:
            self.timer.stop()
            QTimer.singleShot(500, self.preview_final)

    def preview_final(self) -> None:
        if self.worker is not None:
            return
        (base, bfile), k, mode = self._base_args(), self.strength.value(), self.mode.currentText()
        self.start("Preview with prominences",
                   lambda p, say, cancel: W.final_with_prominences(p, base, k, mode, preview_side=1400,
                                                                   base_file=bfile, say=say, cancel=cancel),
                   self._final_preview_done, reload=False)

    def _final_preview_done(self, img) -> None:
        self._final_shown = True
        self.view.show_array(img)

    def write_final(self) -> None:
        (base, bfile), k, mode = self._base_args(), self.strength.value(), self.mode.currentText()
        proj = self.main.ensure_saved()
        if proj is None:
            return
        default = self.final_path or str(Path(proj).with_name(Path(proj).stem + "_final_prominences.tif"))
        fname, filt = QFileDialog.getSaveFileName(self, "Save final image as", default, IO.FILTER)
        if not fname:
            return
        out = str(_with_ext(fname, filt))
        self.start("Writing the final image",
                   lambda p, say, cancel: W.final_with_prominences(p, base, k, mode, output_file=out,
                                                                   base_file=bfile, say=say, cancel=cancel),
                   self._final_done)

    def _final_done(self, path) -> None:
        self.final_path = path
        self.open_btn.setEnabled(True)
        self.log.appendPlainText(f"Final image written to {path}")
        QMessageBox.information(self, "Final image", f"Wrote\n{path}")


# ---------------------------------------------------------------------------
# Image display
# ---------------------------------------------------------------------------


class ImageView(QWidget):
    """Shows an image, fitted to the view, with zoom and pan.

    Mouse wheel (or + and -): zoom about the pointer. Drag: move round. Double-click: 100% at
    that point, or back to fit. 0: fit, 1: 100%. A refreshed preview of the same size keeps the
    zoom and place, so settings can be compared on the same detail.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumSize(320, 240)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.setMouseTracking(False)
        self.setToolTip("Scroll to zoom, drag to move round, double-click for 100% (again to fit). "
                        "Keys: + and - zoom, 0 fit, 1 100%.")
        self._pix: QPixmap | None = None
        self._text = "No image yet."
        self._zoom = ViewZoom()
        self._drag = None

    def setText(self, text: str) -> None:  # noqa: N802 - Qt-style name, as before
        self._text = text
        self._pix = None
        self.update()

    def show_array(self, img) -> None:
        import numpy as np

        a = np.ascontiguousarray((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8))
        if a.ndim == 2:
            a = np.ascontiguousarray(np.repeat(a[..., None], 3, axis=2))
        h, w = a.shape[:2]
        qimg = QImage(a.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
        self._pix = QPixmap.fromImage(qimg)
        self._zoom.set_view(self.width(), self.height())
        self._zoom.set_image(w, h)
        self.update()

    def show_file(self, path: str) -> None:
        if not path or not Path(path).exists():
            return
        try:
            w, h = IO.image_size(path)
            # Read finely enough to zoom in on (a reduced copy above ~3200 px).
            self.show_array(IO.read_image(path, step=max(1, max(w, h) // 3200)))
        except Exception as exc:  # a preview is a convenience; never fail a step over it
            self.setText(f"Could not show {Path(path).name}: {exc}")

    # -- drawing -----------------------------------------------------------------------------

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().resizeEvent(event)
        self._zoom.set_view(self.width(), self.height())

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt override
        p = QPainter(self)
        try:
            p.fillRect(self.rect(), QColor("#111111"))
            if self._pix is None or self._pix.isNull():
                p.setPen(QColor("#888888"))
                p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter.value | Qt.TextFlag.TextWordWrap.value, self._text)
                return
            self._zoom.set_view(self.width(), self.height())
            rects = self._zoom.rects()
            if rects is None:
                return
            (tx, ty, tw, th), (sx, sy, sw, sh) = rects
            # Smooth when shrinking; sharp pixels when enlarged well past 100%.
            p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, self._zoom.scale() < 2.0)
            p.drawPixmap(QRectF(tx, ty, tw, th), self._pix, QRectF(sx, sy, sw, sh))
            label = self._zoom.label()
            p.setPen(QColor("#bbbbbb"))
            box = QRectF(6, self.height() - 24, 70, 18)
            p.fillRect(box, QColor(0, 0, 0, 150))
            p.drawText(box, Qt.AlignmentFlag.AlignCenter, label)
        finally:
            p.end()

    # -- mouse and keys ----------------------------------------------------------------------

    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt override
        if self._pix is None:
            return
        steps = event.angleDelta().y() / 120.0
        if steps:
            pos = event.position()
            self._zoom.zoom_at(1.25 ** steps, pos.x(), pos.y())
            self.update()
        event.accept()

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt override
        if event.button() == Qt.MouseButton.LeftButton and self._zoom.zoom is not None:
            self._drag = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt override
        if self._drag is not None:
            pos = event.position()
            self._zoom.pan(pos.x() - self._drag.x(), pos.y() - self._drag.y())
            self._drag = pos
            self.update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 - Qt override
        if self._drag is not None:
            self._drag = None
            self.unsetCursor()
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802 - Qt override
        if self._pix is not None:
            pos = event.position()
            self._zoom.toggle(pos.x(), pos.y())
            self.update()

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt override
        key = event.key()
        if self._pix is None:
            return super().keyPressEvent(event)
        if key in (Qt.Key.Key_Plus, Qt.Key.Key_Equal):
            self._zoom.zoom_at(1.25)
        elif key == Qt.Key.Key_Minus:
            self._zoom.zoom_at(0.8)
        elif key == Qt.Key.Key_0:
            self._zoom.fit()
        elif key == Qt.Key.Key_1:
            self._zoom.set_zoom(1.0)
        else:
            return super().keyPressEvent(event)
        self.update()


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------


class MainWindow(QMainWindow):
    def __init__(self, link: SirilLink) -> None:
        super().__init__()
        self.link = link
        self.setWindowTitle(f"Solar Eclipse Toolkit {VERSION}" + (f" ({BUILD})" if BUILD else "")
                            + ("" if link.connected else " (not connected to Siril)"))
        self.resize(1280, 820)
        self.tabs = QTabWidget()
        self.frames_tab = FramesTab(link)
        self.frames_tab.on_project_opened = self._project_opened
        self.align_tab = AlignTab(self)
        self.stack_tab = StackTab(self)
        self.enhance_tab = EnhanceTab(self)
        self.earthshine_tab = EarthshineTab(self)
        self.prominences_tab = ProminencesTab(self)
        self.tabs.addTab(self.frames_tab, "1 Frames")
        self.tabs.addTab(self.align_tab, "2 Align")
        self.tabs.addTab(self.stack_tab, "3 Stack")
        self.tabs.addTab(self.enhance_tab, "4 Enhance")
        self.tabs.addTab(self.earthshine_tab, "5 Earthshine")
        self.tabs.addTab(self.prominences_tab, "6 Prominences")
        self.setCentralWidget(self.tabs)

        menu = self.menuBar().addMenu("&Project")
        for text, slot, key in (("&New from folder…", self.new_from_folder, "Ctrl+N"),
                                ("&Open…", self.open_project, "Ctrl+O"),
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

    def new_from_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Choose the project folder (holding lights/, and "
                                                        "optionally darks/, flats/, bias/)", str(self.frames_tab.project_dir()))
        if not folder:
            return
        if not self.frames_tab.use_project_folder(folder):
            QMessageBox.information(self, "New project",
                                    "That folder has no 'lights' subfolder. Put the light frames in <folder>/lights "
                                    "(and any calibration frames in darks, flats and bias), or choose the folder of "
                                    "light frames on the Frames tab.")
            return
        self.tabs.setCurrentIndex(0)
        self._project_opened()

    def _project_opened(self) -> None:
        """A project was opened on the Frames tab: show it on every tab."""
        self.enhance_tab.forget_preview()
        self.reload_project()

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
        for tab in (self.align_tab, self.stack_tab, self.enhance_tab, self.earthshine_tab, self.prominences_tab):
            tab.load_project(project, Path(path))
        self._title(path)

    def _title(self, path) -> None:
        self.setWindowTitle(f"Solar Eclipse Toolkit {VERSION}" + (f" ({BUILD})" if BUILD else "") + f" – {Path(path).name}"
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


def _with_ext(fname: str, filt: str = "") -> Path:
    """The chosen file with an extension: the one typed, or the one of the chosen file type."""
    f = Path(fname)
    if f.suffix.lower() in IO.FORMATS:
        return f
    for ext in (".png", ".jpg", ".tif"):
        if ext in (filt or ""):
            return f.with_suffix(ext) if f.suffix else Path(str(f) + ext)
    return Path(str(f) + ".tif")


def _tip(widget, text: str, form=None) -> None:
    """Hover help on a control and on its label. Rich text so Qt wraps long tips."""
    html = "<p>" + text.replace("\n\n", "</p><p>") + "</p>"
    widget.setToolTip(html)
    if form is not None:
        label = form.labelForField(widget)
        if label is not None:
            label.setToolTip(html)


def _mono() -> QFont:
    font = QFont("Monospace")
    font.setStyleHint(QFont.StyleHint.TypeWriter)
    return font


def _event_summary(p: EventProject) -> str:
    """One look at the event settings, for the Align tab."""
    if p.latitude is None or p.longitude is None:
        return ("No site entered: enter the latitude and longitude in the Event box on the Frames tab. Without "
                "them the frames are still aligned, but the Sun's true centre is not found.")
    eclipse = p.eclipse_date or "the eclipse on the frames' date"
    zone = f"UTC{p.camera_utc_offset_h:+g} h" if p.camera_utc_offset_h is not None else "from the files"
    clock = f"{p.clock_error_s:+g} s" if p.clock_error_s is not None else "not known (worked out)"
    return (f"Eclipse: {eclipse}\nSite: {p.latitude:.6f}, {p.longitude:.6f}, {p.elevation_m:g} m\n"
            f"Camera time zone: {zone}; clock error: {clock}")


def _txt(x) -> str:
    # 15 significant digits: shows what was typed (36.942202, not 36.9422), so running Align again does
    # not save a rounded copy back into the project.
    return "" if x is None else f"{x:.15g}" if isinstance(x, float) else str(x)


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
