"""Small reusable Qt widgets (GUI layer only), with the TensorSpec interface conventions
(copied from TensorSpec gui/components/peem_map_common.py and gui/main_browser.py): numbered QGroupBox steps with
bold titles, tidy left-aligned forms, step buttons in the suites' light action colours, one blue main button,
<b>bold</b> section labels, non-collapsible splitters; no global theme (the system light / dark mode)."""
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6 import QtCore, QtWidgets

# Lighter shades of the TensorSpec suites' main-button colours, dark text (peem_map_common.ACTION)
ACTION = {
    "load": ("#a9dcab", "#1d3d1f", "#d6e6d6"),
    "map": ("#f2a9ae", "#5a1418", "#ead3d5"),
    "domains": ("#f7c896", "#5a3208", "#eadccd"),
    "handover": ("#a3dbe5", "#0c3f48", "#d2e5e8"),
}
PRIMARY = "font-weight: bold; padding: 10px; background-color: #2b5c8f; color: white;"   # main_browser launch button


def action_style(button, kind):
    """Colour a step's main button like the TensorSpec suites (bold, light shade, greyed when disabled)."""
    bg, fg, off = ACTION[kind]
    button.setStyleSheet(f"QPushButton {{ background-color: {bg}; color: {fg}; font-weight: bold; padding: 5px; }}"
                         f"QPushButton:disabled {{ background-color: {off}; color: #999999; }}")
    return button


def primary_style(button):
    button.setStyleSheet(PRIMARY)
    return button


def bold_titles(group):
    """Bold section title, as the TensorSpec QE / Wannier90 / PEEM groups."""
    group.setStyleSheet("QGroupBox { font-weight: bold; }")
    return group


def heading(text):
    """A bold section label (<b>...</b>), as in the TensorSpec workspace browser."""
    return QtWidgets.QLabel(f"<b>{text}</b>")


def subtitle(text):
    lab = QtWidgets.QLabel(text)
    lab.setStyleSheet("font-weight: bold; padding-top: 6px;")
    return lab

def not_ready_note(ws, keys):
    """Text for a view whose images are not calculated (or out of date), else None."""
    todo = [k for k in keys if not ws.is_ready(k)]
    return (f"Not calculated or out of date: {', '.join(todo)}.\nPress Calculate / Calculate All on the Domains page "
            f"(step 5).") if todo else None


def tidy_forms(widget):
    """The TensorSpec form look on every platform: labels left, fields grow to the full width."""
    for form in widget.findChildren(QtWidgets.QFormLayout):
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        form.setLabelAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        form.setFormAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        form.setRowWrapPolicy(QtWidgets.QFormLayout.DontWrapRows)


def spin(lo, hi, step, decimals, tip="", value=None):
    s = QtWidgets.QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setSingleStep(step)
    s.setDecimals(decimals)
    s.setKeyboardTracking(False)
    s.setToolTip(tip)
    if value is not None:
        s.setValue(value)
    return s


def ispin(lo, hi, tip="", value=None):
    s = QtWidgets.QSpinBox()
    s.setRange(lo, hi)
    s.setKeyboardTracking(False)
    s.setToolTip(tip)
    if value is not None:
        s.setValue(value)
    return s


def button(text, fn, tip="", kind=None):
    """kind: an ACTION colour ('load', 'map', 'domains', 'handover') or 'primary' (blue main button)."""
    b = QtWidgets.QPushButton(text)
    b.clicked.connect(fn)
    b.setToolTip(tip)
    if kind == "primary":
        primary_style(b)
    elif kind:
        action_style(b, kind)
    return b


def combo(items, tip="", current=None):
    """items: list of labels or (label, value) pairs; value() of the result returns the value."""
    c = QtWidgets.QComboBox()
    for it in items:
        label, val = it if isinstance(it, tuple) else (it, it)
        c.addItem(label, val)
    c.setToolTip(tip)
    c.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)   # long labels never widen
    c.setMinimumContentsLength(12)                                                     # the 400-px column
    if current is not None:
        i = c.findData(current)
        c.setCurrentIndex(max(i, 0))
    return c


def row(*items):
    h = QtWidgets.QHBoxLayout()
    for w in items:
        if isinstance(w, QtWidgets.QWidget):
            h.addWidget(w)
        elif isinstance(w, QtWidgets.QLayout):
            h.addLayout(w)
        elif isinstance(w, str):
            h.addWidget(QtWidgets.QLabel(w))
    return h


def group(title, layout):
    g = QtWidgets.QGroupBox(title)
    g.setLayout(layout)
    return bold_titles(g)


def scroll(widget, width=400):
    s = QtWidgets.QScrollArea()
    s.setWidget(widget)
    s.setWidgetResizable(True)
    s.setMinimumWidth(width)
    return s


def splitter(*widgets, sizes=None, vertical=False):
    sp = QtWidgets.QSplitter(QtCore.Qt.Vertical if vertical else QtCore.Qt.Horizontal)
    sp.setChildrenCollapsible(False)
    for w in widgets:
        sp.addWidget(w)
    if sizes:
        sp.setSizes(sizes)
    return sp


class CompactToolbar(NavigationToolbar2QT):
    """Matplotlib's navigation toolbar, sized like a toolbar (TensorSpec plotting/viewers/raman_canvas.py).

    On a Retina display Qt hands this toolbar pixmaps scaled by the device pixel ratio without shrinking the
    buttons to match, so the icon row ends up taller than the plot it controls. Pinning the icon size and the
    widget height keeps it to one slim strip on every display."""

    ICON_PX = 16

    def __init__(self, canvas, parent=None):
        super().__init__(canvas, parent)
        self.setIconSize(QtCore.QSize(self.ICON_PX, self.ICON_PX))
        self.setFixedHeight(self.ICON_PX + 12)
        self.setStyleSheet("QToolBar { border: 0px; padding: 0px; spacing: 0px; }"
                           "QToolButton { padding: 1px; margin: 0px; }")
        for lbl in self.findChildren(QtWidgets.QLabel):   # the coordinate readout is large on macOS
            lbl.setStyleSheet("font-size: 10px; color: #666;")
        self.addSeparator()
        act = self.addAction("Export…")
        act.setToolTip("Save this figure as vector PDF or SVG (editable in Inkscape or Illustrator), or 600-dpi PNG")
        act.triggered.connect(self._export)

    def _export(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export figure", "figure.pdf",
                                                        "PDF (*.pdf);;SVG (*.svg);;PNG (*.png)")
        if path:
            import matplotlib
            with matplotlib.rc_context({"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
                                        "pdf.compression": 0}):
                self.canvas.figure.savefig(path, dpi=600, bbox_inches="tight", facecolor="white")


class Canvas(QtWidgets.QWidget):
    """A matplotlib figure plus the compact navigation toolbar in one widget (TensorSpec layout)."""

    def __init__(self, parent=None, size=(6, 4.5)):
        super().__init__(parent)
        self.fig = Figure(figsize=size, layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.setFocusPolicy(QtCore.Qt.StrongFocus)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        self.toolbar = CompactToolbar(self.canvas, self)
        lay.addWidget(self.toolbar, 0)
        lay.addWidget(self.canvas, 1)

    def draw(self):
        self.canvas.draw_idle()

    def busy_mode(self):
        """True while the toolbar's zoom / pan is active (clicks then belong to the toolbar)."""
        return bool(self.toolbar.mode)


class Slider(QtWidgets.QWidget):
    """Slider + spin box for one float value; emits changed(value)."""
    changed = QtCore.Signal(float)

    def __init__(self, label, lo, hi, step, decimals, value, tip=""):
        super().__init__()
        self.lo, self.step = lo, step
        self.sl = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sl.setRange(0, int(round((hi - lo) / step)))
        self.sp = spin(lo, hi, step, decimals, tip)
        lab = QtWidgets.QLabel(label)
        lab.setMinimumWidth(95)
        lab.setToolTip(tip)
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(lab)
        lay.addWidget(self.sl, 1)
        lay.addWidget(self.sp)
        self.sl.valueChanged.connect(lambda i: self.sp.setValue(self.lo + i * self.step))
        self.sp.valueChanged.connect(self._sync)
        self.set(value)

    def _sync(self, v):
        self.sl.blockSignals(True)
        self.sl.setValue(int(round((v - self.lo) / self.step)))
        self.sl.blockSignals(False)
        self.changed.emit(float(v))

    def set(self, v, silent=False):
        self.sp.blockSignals(silent)
        self.sp.setValue(v)
        self.sp.blockSignals(False)
        if silent:
            self.sl.blockSignals(True)
            self.sl.setValue(int(round((v - self.lo) / self.step)))
            self.sl.blockSignals(False)

    def value(self):
        return float(self.sp.value())
