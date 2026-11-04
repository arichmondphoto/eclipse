"""Minimal stand-in for PyQt6 to smoke-test GUI logic without Qt (no rendering)."""
import os


class _Meta(type):
    def __getattr__(cls, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return Dummy()


class Dummy(metaclass=_Meta):
    def __init__(self, *a, **k):
        pass

    def setToolTip(self, text):
        self._tooltip = str(text)

    def toolTip(self):
        return getattr(self, "_tooltip", "")

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return Dummy()

    def __call__(self, *a, **k):
        return Dummy()

    def __or__(self, other):
        return Dummy()

    def __iter__(self):
        return iter(())


class _Signal:
    def __init__(self):
        self.slots = []

    def connect(self, fn):
        self.slots.append(fn)

    def emit(self, *a):
        for fn in list(self.slots):
            fn(*a)


class pyqtSignal:
    def __init__(self, *types):
        self.name = None

    def __set_name__(self, owner, name):
        self.name = "_sig_" + name

    def __get__(self, obj, owner):
        if obj is None:
            return self
        if self.name not in obj.__dict__:
            obj.__dict__[self.name] = _Signal()
        return obj.__dict__[self.name]


class QObject(Dummy):
    pass


class QWidget(Dummy):
    def size(self):
        return Dummy()

    def width(self):
        return 640

    def height(self):
        return 480


class QThread(QObject):
    def start(self):
        self.run()


class QTimer(QObject):
    def __init__(self, *a):
        self.timeout = _Signal()

    def start(self, *a):
        pass


class _Clickable(QWidget):
    def __init__(self, *a, **k):
        self.clicked = _Signal()
        self.toggled = _Signal()
        self._checked = False
        self._enabled = True

    def setChecked(self, v):
        self._checked = bool(v)
        self.toggled.emit(self._checked)

    def isChecked(self):
        return self._checked

    def setEnabled(self, v):
        self._enabled = bool(v)


class QPushButton(_Clickable):
    pass


class QCheckBox(_Clickable):
    pass


class QRadioButton(_Clickable):
    pass


class QLineEdit(QWidget):
    def __init__(self, *a):
        self._t = ""

    def setText(self, t):
        self._t = t

    def text(self):
        return self._t


class _Spin(QWidget):
    def __init__(self, *a):
        self._v = 0
        self._blocked = False
        self.valueChanged = _Signal()

    def blockSignals(self, b):
        # As Qt: while blocked, changes do not signal; returns the previous state.
        was, self._blocked = self._blocked, bool(b)
        return was

    def setValue(self, v):
        changed = v != self._v
        self._v = v
        if changed and not self._blocked:  # Qt signals only a real change
            self.valueChanged.emit(v)

    def value(self):
        return self._v


class QSpinBox(_Spin):
    pass


class QDoubleSpinBox(_Spin):
    pass


class QComboBox(QWidget):
    def __init__(self, *a):
        self.items = []
        self.i = -1
        self.currentIndexChanged = _Signal()
        self.activated = _Signal()

    def addItem(self, text, data=None):
        self.items.append((text, data))
        if self.i < 0:
            self.i = 0

    def addItems(self, texts):
        for t in texts:
            self.addItem(t)

    def clear(self):
        self.items, self.i = [], -1

    def count(self):
        return len(self.items)

    def setCurrentIndex(self, i):
        self.i = i
        self.currentIndexChanged.emit(i)

    def currentIndex(self):
        return self.i

    def currentText(self):
        return self.items[self.i][0] if self.i >= 0 else ""

    def currentData(self):
        return self.items[self.i][1] if self.i >= 0 else None

    def findText(self, t):
        return next((k for k, it in enumerate(self.items) if it[0] == t), -1)

    def findData(self, d):
        return next((k for k, it in enumerate(self.items) if it[1] == d), -1)

    def setCurrentText(self, t):
        k = self.findText(t)
        if k >= 0:
            self.setCurrentIndex(k)

    def blockSignals(self, b):
        pass


class QPlainTextEdit(QWidget):
    def __init__(self, *a):
        self.lines = []

    def appendPlainText(self, t):
        self.lines.append(t)

    def setPlainText(self, t):
        self.lines = [t]

    def toPlainText(self):
        return "\n".join(self.lines)


class QLabel(QWidget):
    def __init__(self, *a):
        self._t = a[0] if a else ""

    def setText(self, t):
        self._t = t

    def text(self):
        return self._t


class QTableWidget(QWidget):
    def __init__(self, rows=0, cols=0):
        self.rows = rows
        self.cells = {}

    def setRowCount(self, n):
        self.rows = n

    def rowCount(self):
        return self.rows

    def setItem(self, r, c, item):
        self.cells[(r, c)] = item


class QTableWidgetItem(Dummy):
    def __init__(self, text=""):
        self.t = text


class QMessageBox(Dummy):
    log = []

    class StandardButton:
        Yes = 16384
        No = 65536

    @staticmethod
    def question(parent, title, text, *a, **k):
        QMessageBox.log.append(("question", text))
        return QMessageBox.StandardButton.Yes if os.environ.get("STUB_YES") else QMessageBox.StandardButton.No

    @staticmethod
    def information(parent, title, text):
        QMessageBox.log.append(("info", text))

    @staticmethod
    def warning(parent, title, text):
        QMessageBox.log.append(("warn", text))


class QFileDialog(Dummy):
    @staticmethod
    def getSaveFileName(*a, **k):
        return os.environ.get("STUB_SAVE", ""), ""

    @staticmethod
    def getOpenFileName(*a, **k):
        return os.environ.get("STUB_OPEN", ""), ""

    @staticmethod
    def getExistingDirectory(*a, **k):
        return os.environ.get("STUB_DIR", "")


class QImage(Dummy):
    pass


class QPixmap(Dummy):
    def isNull(self):
        return False


class QApplication(Dummy):
    @staticmethod
    def setOverrideCursor(*a):
        pass

    @staticmethod
    def restoreOverrideCursor(*a):
        pass
