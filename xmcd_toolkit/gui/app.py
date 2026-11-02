"""Main window, laid out like the TensorSpec suites (gui/components/peem_panel.py, gui/main_browser.py):
File menu; a non-collapsible splitter with the numbered control steps on the left (scrollable, the steps of the
current view only, step 1 always) and the views on the right (Domains | Align | Compare, document-mode tabs).
GUI layer only: one shared services.workspace.Workspace holds the data.

Run:  sda-gui [session.nc]      or      python -m xmcd_toolkit.gui.app [session.nc]"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("QtAgg")
from PySide6 import QtCore, QtGui, QtWidgets  # noqa: E402

from ..services.workspace import Workspace  # noqa: E402
from .align_tab import AlignTab  # noqa: E402
from .compare_tab import CompareTab  # noqa: E402
from .domains_tab import DomainsTab  # noqa: E402
from .widgets import tidy_forms  # noqa: E402

IMAGE_FILTER = "Images (*.png *.tif *.tiff *.jpg *.jpeg *.bmp *.gif *.webp);;All files (*)"
TITLE = "TensorSpec - XMCD Domains"


class Main(QtWidgets.QMainWindow):
    def __init__(self, session=None):
        super().__init__()
        self.setWindowTitle(TITLE)
        self.ws = Workspace()
        self.domains, self.align, self.compare = DomainsTab(self), AlignTab(self), CompareTab(self)
        self._menu()

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        # --- LEFT: numbered steps (forms only; plots live in the view tabs) ---
        controls = QtWidgets.QWidget()
        cl = QtWidgets.QVBoxLayout(controls)
        self.step_groups = {}
        for tab in (self.domains, self.align, self.compare):
            for g in tab.groups:
                cl.addWidget(g)
                self.step_groups.setdefault(id(tab), []).append(g)
        self.load_group = self.domains.groups[0]          # step 1 stays visible on every view
        self.lStatus = QtWidgets.QLabel("No images loaded")
        self.lStatus.setWordWrap(True)
        self.load_group.layout().addRow(self.lStatus)
        cl.addStretch()
        tidy_forms(controls)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(controls)
        scroll.setMinimumWidth(400)
        splitter.addWidget(scroll)
        # --- RIGHT: Domains | Align | Compare ---
        self.view_tabs = QtWidgets.QTabWidget()
        self.view_tabs.setDocumentMode(True)
        for tab, name in ((self.domains, "Domains"), (self.align, "Align"), (self.compare, "Compare")):
            self.view_tabs.addTab(tab.view, name)
        self.view_tabs.currentChanged.connect(self.tab_changed)
        splitter.addWidget(self.view_tabs)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([420, 980])
        self.setCentralWidget(splitter)
        self.setAcceptDrops(True)          # drop image files onto the window
        self.tab_changed(0)
        self.refresh()
        if session:
            self._load(session)

    def _menu(self):
        file_menu = self.menuBar().addMenu("&File")
        for text, slot, key in (("Add Images…", lambda: self.add_images(), QtGui.QKeySequence.Open),
                                ("Load Session…", self.load, None), ("Save Session…", self.save, QtGui.QKeySequence.Save),
                                ("Clear Workspace", self.clear, None)):
            a = QtGui.QAction(text, self)
            a.triggered.connect(slot)
            if key is not None:
                a.setShortcut(key)
            file_menu.addAction(a)
        file_menu.addSeparator()
        a = QtGui.QAction("&Exit", self)
        a.triggered.connect(self.close)
        file_menu.addAction(a)

    # ---- shared state
    @property
    def tabs(self):            # used by the tabs to know which view is shown
        return self.view_tabs

    def status(self, txt):
        self.lStatus.setText(txt)

    def refresh(self):
        n = len(self.ws.images)
        self.status(f"{n} image(s), common grid {self.ws.shape[1]}×{self.ws.shape[0]} px" if n
                    else "No images loaded: Add Images… (raw XMCD, coded maps, Igor .tif / .png exports) or drop files here")
        self.domains.populate()
        self.align.populate()

    def changed(self, key=None):
        """An image's analysis changed: the other views redraw when shown."""
        self.align.schedule()

    def tab_changed(self, i):
        tab = (self.domains, self.align, self.compare)[i]
        for t in (self.domains, self.align, self.compare):          # the steps of the current view (+ step 1)
            for g in self.step_groups.get(id(t), []):
                g.setVisible(t is tab or g is self.load_group)
        if tab is self.align:
            self.align.schedule()
        elif tab is self.compare:
            self.compare.schedule()

    # ---- files
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        self.add_images([u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()])

    def add_images(self, fs=None):
        if not fs:
            fs, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Add images", "", IMAGE_FILTER)
        if not fs:
            return
        mode = {"input_mode": "asymmetry_map", "map_mode": "auto"}   # every picture as it is, colours detected
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            for i, f in enumerate(fs):
                self.status(f"reading {Path(f).name} ({i + 1}/{len(fs)})…")
                QtWidgets.QApplication.processEvents()
                try:
                    self.ws.add_image(f, mode, run=False)     # analysed when Calculate is pressed
                except Exception as e:  # noqa: BLE001 - show any load / analysis error
                    QtWidgets.QMessageBox.warning(self, "Cannot add", f"{Path(f).name}: {e}")
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self.refresh()
        self.status(self.lStatus.text() + "; draw the active area (step 2), then press Calculate (step 5)")

    def save(self):
        if not self.ws.images:
            return
        f, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save session", "session.nc", "Session (*.nc)")
        if f:
            self.ws.save(f)
            self.status(f"saved {Path(f).name}: images, parameters, regions, alignment, comparisons, history")

    def load(self):
        f, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load session", "", "Session (*.nc)")
        if f:
            self._load(f)

    def _load(self, f):
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            self.ws = Workspace.load(f)
        except Exception as e:  # noqa: BLE001
            QtWidgets.QMessageBox.warning(self, "Cannot load", str(e))
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self.refresh()
        self.status(f"loaded {Path(f).name}")

    def clear(self):
        self.ws = Workspace()
        self.refresh()


def _show_errors(kind, value, tb):
    """An action that cannot run yet (e.g. an image not calculated) is reported in a dialog, not a traceback."""
    if isinstance(value, RuntimeError) and QtWidgets.QApplication.instance():
        QtWidgets.QMessageBox.warning(QtWidgets.QApplication.activeWindow(), "Not ready", str(value))
    else:
        sys.__excepthook__(kind, value, tb)


def main():
    sys.excepthook = _show_errors
    app = QtWidgets.QApplication(sys.argv)
    win = Main(sys.argv[1] if len(sys.argv) > 1 else None)
    win.resize(1400, 900)   # the PEEM suite opens at 1100 x 750; three views need a little more room
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
