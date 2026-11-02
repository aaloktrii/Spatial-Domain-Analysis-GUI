"""Tab 1 · Domains: the spatial domain analysis of each image.

Load images, draw the active area (and optionally a background region) on each, set calibration, denoising and
classification; the three-state map and the distributions update live; Export all writes the full set of
figures / tables / TIFFs next to the image (services.export.run_export). GUI layer only: numbers come from
services.workspace / services.preview, figures from plotting/."""
import os

import matplotlib
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from PySide6 import QtCore, QtGui, QtWidgets

from ..colors import CMAP3_HEX, COLOR_DOWN, COLOR_NEUTRAL, COLOR_UP
from ..core.methods.boundary import BOUNDARIES
from ..core.methods.denoise import DENOISERS
from ..core.methods.width import WIDTH_METHOD_LABELS, WIDTH_METHODS
from ..core.regions import calibration
from ..defaults import DEFAULTS
from ..notation import DICHROISM, relabel
from ..plotting.compare_figures import screen_sample
from ..plotting.distribution_panels import draw_distributions
from ..services.export import run_export
from ..services.preview import summary_text
from . import region_editor as re_
from .widgets import Canvas, Slider, button, combo, group, heading, ispin, row, spin, splitter

CMAP3 = ListedColormap(CMAP3_HEX)
NORM3 = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], 3)
CAL_MODES = [("Pixels (no calibration)", "none"), ("Whole image (known size)", "image"),
             ("Active-area box (known size)", "shape")]


class DomainsTab(QtWidgets.QWidget):
    def __init__(self, main):
        super().__init__()
        self.main = main
        self.editor = None
        self._axp = None
        self.timer = QtCore.QTimer(self, singleShot=True, interval=250)
        self.timer.timeout.connect(self.mark_changed)

        self.groups = []           # numbered steps for the main window's control column
        # 1. load
        self.list = QtWidgets.QListWidget()
        self.list.setMaximumHeight(110)
        how = QtWidgets.QLabel("Any picture (.tif, .png, .jpg, …), read as it is, every pixel. "
                               "Grey: white +1, black −1. Colour: blue +1, white 0, red −1.")
        how.setWordWrap(True)
        load = QtWidgets.QFormLayout()
        load.addRow(row(button("Add Images…", lambda: self.main.add_images(), "or drop files on the window", "load"),
                        button("Remove", self.remove)))
        load.addRow("Images:", self.list)
        load.addRow(how)
        self.cInvert = QtWidgets.QCheckBox("Invert colours (swap +ve / −ve)")
        self.cInvert.setToolTip("for a map whose colours mean the opposite sign; the raw "
                                "panel still shows the file as it is")
        self.cDichro = combo([(lab, key) for key, lab in DICHROISM], "order-parameter name only: XMCD ⟨m⟩, XLD "
                                                                    "(XMLD + XNLD) ⟨v_N⟩; states are +ve / −ve "
                                                                    "for both, and the numbers are the same")
        load.addRow("Dichroism:", self.cDichro)
        load.addRow(self.cInvert)
        self.cSetFrom = combo([], "an image whose settings (invert, calibration, denoising, threshold, domain "
                                  "size, methods; not the outlines) you want to reuse")
        load.addRow("Settings from:", self.cSetFrom)
        load.addRow(row(button("Apply to This Image", self.copy_settings, "copy the settings of the chosen image to "
                                                                          "the selected one"),
                        button("Apply to All", lambda: self.copy_settings(every=True),
                               "copy the settings of the chosen image to every other image")))
        self.groups.append(group("1. Load", load))
        # regions
        self.cTarget = combo([("active area", "polygon"), ("background (zero)", "bg_polygon")],
                             "which region you draw / edit")
        self.cTool = combo(list(re_.TOOLS))
        self.sSides = ispin(3, 12, "sides of a regular polygon", 6)
        self.sStep = spin(0.5, 50, 1, 1, "move step (px)", 2)
        self.sAngle = spin(0.1, 45, 0.5, 1, "rotate step (°)", 1)
        rg = QtWidgets.QFormLayout()
        rg.addRow("Draw:", self.cTarget)
        rg.addRow("Shape:", self.cTool)
        rg.addRow("Sides:", self.sSides)
        rg.addRow(row(button("Whole Image", self.whole_image), button("Clear", self.clear_region)))
        rg.addRow(row(button("Save Region…", self.save_region), button("Load Region…", self.load_region)))
        self.cCentre = combo([("0", "0"), ("median", "median"), ("mean", "mean")],
                             "zero of the map, as TensorSpec's Sym ±: 0 (or the background region's mean if drawn), "
                             "or the median / mean in the background region, else of the map without its edge "
                             "drift strips")
        rg.addRow("Sym ± centre:", self.cCentre)
        self.cFrom = combo([], "an image whose active area and background you want to reuse")
        rg.addRow("Outlines from:", self.cFrom)
        rg.addRow(row(button("Apply to This Image", self.copy_regions, "copy both outlines of the chosen image "
                                                                         "to the selected image"),
                      button("Apply to All", lambda: self.copy_regions(every=True),
                             "copy both outlines of the chosen image to every other image")))
        rg.addRow("Move step (px):", self.sStep)
        rg.addRow("Rotate:", row(button("Rot +", lambda: self.turn(1)), button("Rot −", lambda: self.turn(-1))))
        rg.addRow("Angle (°):", self.sAngle)
        self.lRegion = QtWidgets.QLabel("Click the picture, then arrow keys move the outline (SHIFT + arrow: 10 "
                                        "steps). SHIFT + drag moves it; drag a corner to reshape it")
        self.lRegion.setWordWrap(True)
        rg.addRow(self.lRegion)
        self.groups.append(group("2. Active area / background", rg))
        # calibration
        self.cCal = combo(CAL_MODES)
        self.sW, self.sH = spin(0, 1e4, 0.1, 3, "width (µm)"), spin(0, 1e4, 0.1, 3, "height (µm); 0 = use width only")
        self.lCal = QtWidgets.QLabel("")
        self.groups.append(group("3. Calibration", self._form(("Mode:", self.cCal), ("W / H (µm):", row(self.sW, self.sH)),
                                                              ("", self.lCal))))
        # denoising
        self.cDen = combo([(m.label, m.name) for m in DENOISERS.values()], current=DEFAULTS["denoise_method"])
        self.slTV = Slider("Strength:", 0.0, 1.0, 0.01, 2, DEFAULTS["tv_weight"], "TV / NLM strength")
        self.sFlat = spin(1, 2000, 5, 0, "flattening scale (px); raise to ~150 for zoomed frames", DEFAULTS["flatten_sigma"])
        dn = QtWidgets.QFormLayout()
        dn.addRow("Denoiser:", self.cDen)
        dn.addRow(self.slTV)
        dn.addRow("Flatten σ (px):", self.sFlat)
        self.groups.append(group("4. Denoise", dn))
        # classification
        self.slM = Slider("m threshold:", 0.0, 0.5, 0.005, 3, DEFAULTS["m_threshold"], "|m| ≤ threshold → neutral")
        self.sMin = ispin(0, 1000, "smallest domain kept (px)", DEFAULTS["min_px"])
        self.cConn = combo([("4-way", 1), ("8-way", 2)], current=DEFAULTS["connectivity"])
        self.cWidth = combo([(WIDTH_METHOD_LABELS[k], k) for k in WIDTH_METHODS], current=DEFAULTS["width_method"])
        self.cBound = combo([(m.label, m.name) for m in BOUNDARIES.values()], current=DEFAULTS["boundary_method"])
        self.cShowB = QtWidgets.QCheckBox("Show boundary")
        self.cMeasure = QtWidgets.QCheckBox("Measure domain widths")
        self.cMeasure.setToolTip("widths, periods and correlation length of every domain; on large images this "
                                 "takes ~30 s per update, so it is off until ticked (the export always measures)")
        self.cShowC = QtWidgets.QCheckBox("Picture under the processed map")
        cf = QtWidgets.QFormLayout()
        cf.addRow(self.slM)
        cf.addRow("Min domain (px):", self.sMin)
        cf.addRow("Connectivity:", self.cConn)
        cf.addRow("Width:", self.cWidth)
        cf.addRow("Boundary:", self.cBound)
        cf.addRow(self.cShowB)
        cf.addRow(self.cMeasure)
        cf.addRow(self.cShowC)
        cf.addRow(row(button("Calculate", self.calculate, "run the analysis of this image with the settings above "
                                                          "(changes are not calculated until you press it)",
                             "primary"),
                      button("Calculate All", self.calculate_all, "run the analysis of every image that is new or "
                                                                  "out of date, each with its own settings")))
        self.groups.append(group("5. Classify", cf))
        # distributions
        self.cQty = combo([("width", "width"), ("period", "period")])
        self.cQQ = QtWidgets.QCheckBox("Q-Q plot (vs. log-normal)")
        self.cGauss = QtWidgets.QCheckBox("log-normal vs. Gaussian")
        di = QtWidgets.QFormLayout()
        di.addRow("Panels for:", self.cQty)
        di.addRow(self.cQQ)
        di.addRow(self.cGauss)
        self.groups.append(group("6. Distributions", di))
        # export
        self.cFmt = combo([("PNG (raster)", "png"), ("EPS (vector)", "eps"), ("PDF (vector)", "pdf")],
                          current=DEFAULTS["fig_format"])
        self.cDetail = QtWidgets.QCheckBox("Detailed figures (diagnostics/)")
        # export folder: empty (default) = the folder the raw image is in; another folder only when chosen
        self.eOut = QtWidgets.QLineEdit()
        self.eOut.setPlaceholderText("the raw image's folder")
        self.eOut.setToolTip("each image is written to <folder>/<image name>_analysis/")
        self.groups.append(group("7. Export", self._vbox(
            row("Figure format:", self.cFmt), self.cDetail,
            row("Folder:", self.eOut),
            row(button("Choose Folder…", self.choose_out), button("Raw Data Folder", self.clear_out)),
            button("Export All Files", self.export, "figures, distributions, TIFFs and tables of the selected image "
                                                    "into the export folder", "primary"))))

        self.cv = Canvas(self, (9, 7))
        # arrow keys move the outline: take them from matplotlib's toolbar history (back / forward)
        for name in ("keymap.back", "keymap.forward"):
            matplotlib.rcParams[name] = [k for k in matplotlib.rcParams[name] if k not in ("left", "right")]
        self.cv.canvas.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.cv.canvas.mpl_connect("key_press_event", self._key)
        self.results = QtWidgets.QTextEdit()
        self.results.setReadOnly(True)
        self.results.setPlaceholderText("Draw the active area to see the results…")
        side = QtWidgets.QWidget()
        sl = QtWidgets.QVBoxLayout(side)
        sl.setContentsMargins(4, 4, 4, 4)
        sl.addWidget(heading("Results"))
        sl.addWidget(self.results)
        side.setMinimumWidth(220)
        self.results.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont))
        self.view = splitter(self.cv, side, sizes=[700, 260])

        self.list.currentTextChanged.connect(self.show_image)
        for w in (self.cDen, self.cConn, self.cWidth, self.cBound, self.cQty, self.cCal):
            w.currentIndexChanged.connect(self.schedule)
        for w in (self.slTV, self.slM):
            w.changed.connect(self.schedule)
        for w in (self.sFlat, self.sMin, self.sW, self.sH):
            w.valueChanged.connect(self.schedule)
        for w in (self.cShowB, self.cQQ, self.cGauss, self.cMeasure):
            w.toggled.connect(self.schedule)
        self.cShowC.toggled.connect(self.redraw)
        self.cInvert.toggled.connect(self.schedule)
        self.cDichro.currentIndexChanged.connect(self._dichroism_changed)
        self.cCentre.currentIndexChanged.connect(self.schedule)
        self.cTarget.currentIndexChanged.connect(self.redraw)
        self.cTool.currentIndexChanged.connect(self.redraw)

    # ---- small layout helpers
    @staticmethod
    def _vbox(*items):
        v = QtWidgets.QVBoxLayout()
        for it in items:
            v.addWidget(it) if isinstance(it, QtWidgets.QWidget) else v.addLayout(it)
        return v

    @staticmethod
    def _form(*rows):
        f = QtWidgets.QFormLayout()
        for lab, w in rows:
            f.addRow(lab, w) if isinstance(w, QtWidgets.QWidget) else f.addRow(lab, w)
        return f

    # ---- state
    @property
    def ws(self):
        return self.main.ws

    @property
    def key(self):
        it = self.list.currentItem()
        return it.text() if it else None

    def populate(self):
        cur = self.key
        self.list.blockSignals(True)
        self.list.clear()
        self.list.addItems(list(self.ws.images))
        items = self.list.findItems(cur, QtCore.Qt.MatchExactly) if cur else []
        self.list.setCurrentItem(items[0] if items else self.list.item(self.list.count() - 1))
        self.list.blockSignals(False)
        self._refresh_sources()
        self.show_image()

    def _refresh_sources(self):
        """'Outlines from': the images that have an active area or background drawn; 'Settings from': every image."""
        src = self.cFrom.currentText()
        have = [k for k, im in self.ws.images.items() if im["params"].get("polygon") or im["params"].get("bg_polygon")]
        self.cFrom.blockSignals(True)
        self.cFrom.clear()
        self.cFrom.addItems(have)
        if src in have:
            self.cFrom.setCurrentText(src)
        elif self.key in have:
            self.cFrom.setCurrentText(self.key)
        self.cFrom.blockSignals(False)
        cur = self.cSetFrom.currentText()
        self.cSetFrom.blockSignals(True)
        self.cSetFrom.clear()
        self.cSetFrom.addItems(list(self.ws.images))
        self.cSetFrom.setCurrentText(cur if cur in self.ws.images else (self.key or ""))
        self.cSetFrom.blockSignals(False)

    def show_image(self, *_):
        """Put the selected image's parameters into the controls, then draw."""
        k = self.key
        if not k:
            self.cv.fig.clear()
            self.cv.draw()
            self.results.clear()
            return
        self._show_out()
        p = self.ws.images[k]["params"]
        widgets = (self.cDichro, self.cInvert, self.cCentre, self.cDen, self.cConn, self.cWidth, self.cBound, self.cQty, self.cCal,
                   self.slTV, self.slM, self.sFlat, self.sMin, self.sW, self.sH, self.cShowB, self.cQQ, self.cGauss,
                   self.cMeasure)
        for w in widgets:
            w.blockSignals(True)
        for c, key in ((self.cDen, "denoise_method"),
                       (self.cConn, "connectivity"), (self.cWidth, "width_method"), (self.cBound, "boundary_method"),
                       (self.cQty, "dist_quantity"), (self.cCal, "cal_mode")):
            i = c.findData(p.get(key))
            if i >= 0:
                c.setCurrentIndex(i)
        self.slTV.set(p["tv_weight"], silent=True)
        self.slM.set(p["m_threshold"], silent=True)
        self.sFlat.setValue(p["flatten_sigma"])
        self.sMin.setValue(p["min_px"])
        self.sW.setValue(p.get("cal_w") or 0)
        self.sH.setValue(p.get("cal_h") or 0)
        self.cShowB.setChecked(bool(p.get("show_boundary")))
        self.cMeasure.setChecked(p.get("measure_widths") is not False)
        self.cInvert.setChecked(bool(p.get("invert")))
        self.cDichro.setCurrentIndex(max(self.cDichro.findData(p.get("dichroism", "XMCD")), 0))
        self.cCentre.setCurrentIndex(max(self.cCentre.findData(p.get("sym_centre", "0")), 0))
        self.cQQ.setChecked(bool(p.get("qq_plot")))
        self.cGauss.setChecked(bool(p.get("gauss_compare")))
        for w in widgets:
            w.blockSignals(False)
        self.redraw()

    def params(self):
        """The parameter dictionary of the domain pipeline."""
        p = {"input_mode": "asymmetry_map", "map_mode": "auto",       # every picture as it is, colours detected
             "denoise_method": self.cDen.currentData(), "tv_weight": self.slTV.value(),
             "flatten_sigma": self.sFlat.value(), "m_threshold": self.slM.value(), "min_px": self.sMin.value(),
             "connectivity": self.cConn.currentData(), "width_method": self.cWidth.currentData(),
             "boundary_method": self.cBound.currentData(), "show_boundary": self.cShowB.isChecked(),
             "measure_widths": self.cMeasure.isChecked(),
             "dist_quantity": self.cQty.currentData(), "qq_plot": self.cQQ.isChecked(),
             "gauss_compare": self.cGauss.isChecked(), "cal_mode": self.cCal.currentData(),
             "cal_w": self.sW.value() or None, "cal_h": self.sH.value() or None,
             "invert": self.cInvert.isChecked(), "dichroism": self.cDichro.currentData(), "sym_centre": self.cCentre.currentData()}
        p["um_per_px"] = self._um(p)
        return p

    def _um(self, p):
        k = self.key
        if not k or p["cal_mode"] == "none" or not (p["cal_w"] or p["cal_h"]):
            self.lCal.setText("pixels (no calibration)")
            return None
        um, warn = calibration({**p, "polygon": self.ws.images[k]["params"].get("polygon")}, self.ws.size(k))
        self.lCal.setText(f"{um * 1000:.2f} nm/px" + (f"  ({warn})" if warn else "") if um else "—")
        return um

    # ---- analysis
    def schedule(self, *_):
        self.timer.start()

    def mark_changed(self):
        """A setting or outline changed: store it; the result is out of date until Calculate."""
        k = self.key
        if not k:
            return
        self.ws.set_params(k, run=False, **self.params())
        self._refresh_sources()
        self._show_state()

    def _show_state(self):
        """Title of the processed panel and the status line: calculated / out of date / not calculated."""
        k = self.key
        if not k:
            return
        if self._axp is not None:
            self._axp.set_title(self._processed_title(k), fontsize=8,
                                color="#b00" if not self.ws.is_ready(k) else "black")
            self.cv.draw()
        if not self.ws.is_ready(k):
            self.main.status(f"{k}: changed, press Calculate (step 5)")

    def _dichroism_changed(self, *_):
        """Name only (XMCD <m> / XLD <v_N>): stored at once, nothing to recalculate."""
        k = self.key
        if k:
            self.ws.images[k]["params"]["dichroism"] = self.cDichro.currentData()
            self.redraw()

    def kind(self, k=None):
        k = k or self.key
        return self.ws.images[k]["params"].get("dichroism", "XMCD") if k in self.ws.images else "XMCD"

    def _processed_title(self, k):
        im = self.ws.images[k]
        if im["result"] is None:
            return "processed: not calculated yet (press Calculate)"
        if im.get("stale"):
            return "processed: out of date (press Calculate)"
        return relabel("processed: three-state m_z inside the active area", self.kind(k))

    def calculate(self, *_):
        k = self.key
        if not k:
            return
        self.timer.stop()
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        self.main.status(f"calculating {k}…")
        QtWidgets.QApplication.processEvents()
        try:
            self.ws.set_params(k, **self.params())
            self.main.status(f"{k}: calculated")
        except Exception as e:  # noqa: BLE001 - show the message
            QtWidgets.QMessageBox.warning(self, "Analysis", str(e))
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self.main.changed(k)
        self._refresh_sources()
        self.redraw()

    def calculate_all(self, *_):
        """Every image that is new or out of date (the selected one with the settings shown)."""
        if self.key:
            self.timer.stop()
            self.ws.set_params(self.key, run=False, **self.params())
        todo = [k for k in self.ws.images if not self.ws.is_ready(k)]
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            for i, k in enumerate(todo):
                self.main.status(f"calculating {k} ({i + 1}/{len(todo)})…")
                QtWidgets.QApplication.processEvents()
                try:
                    self.ws.calculate(k)
                except Exception as e:  # noqa: BLE001
                    QtWidgets.QMessageBox.warning(self, "Analysis", f"{k}: {e}")
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self.main.status(f"calculated {len(todo)} image(s)" if todo else "all images are up to date")
        self.main.changed(self.key)
        self._refresh_sources()
        self.redraw()

    def redraw(self, *_):
        k = self.key
        f = self.cv.fig
        if self.editor:
            self.editor.disconnect()
            self.editor = None
        f.clear()
        self._axp = None
        if not k:
            self.cv.draw()
            return
        im = self.ws.images[k]
        res, p = im["result"], im["params"]
        gs = GridSpec(2, 10, height_ratios=[3, 1.2], figure=f)
        # left: the file as it is (regions are drawn here); right: the processed three-state map. Zoom / pan together.
        ax = f.add_subplot(gs[0, :5])
        axp = f.add_subplot(gs[0, 5:], sharex=ax, sharey=ax)
        self._axp = axp
        # drawn at screen resolution (~1600 px across) so zooming / drawing regions stays quick; the analysis
        # uses every pixel. Axes stay in full-image pixels.
        pic, ext = screen_sample(im["picture"])
        ax.imshow(pic, cmap="gray", interpolation="nearest", extent=ext)   # the file's own colours
        if self.cShowC.isChecked() or res is None:           # optional: the picture under the classification
            axp.imshow(pic, cmap="gray", interpolation="nearest", extent=ext, alpha=1.0 if res else 0.35)
        if res is not None:
            proc = np.where(screen_sample(res["roi"])[0], screen_sample(res["mz"])[0], np.nan)
            axp.imshow(proc, cmap=CMAP3, norm=NORM3, alpha=0.8 if self.cShowC.isChecked() else 1.0,
                       interpolation="nearest", extent=ext)
        h, w = self.ws.size(k)
        ax.set_xlim(-0.5, w - 0.5)
        ax.set_ylim(h - 0.5, -0.5)
        target = self.cTarget.currentData()
        other = "bg_polygon" if target == "polygon" else "polygon"
        for key, a in ((other, ax), ("polygon", axp), ("bg_polygon", axp)):
            if p.get(key):
                v = np.asarray(list(p[key]) + [p[key][0]])
                a.plot(v[:, 0], v[:, 1], "--" if key == "bg_polygon" else "-",
                       color="#e69500" if key == "bg_polygon" else "#0a0", lw=1.2)
        b = (res or {}).get("boundary") or {}
        if p.get("show_boundary"):
            if b.get("contours"):
                for cc in b["contours"]:
                    axp.plot(cc[:, 1], cc[:, 0], color="#00e5ff", lw=0.6)
            elif b.get("edge") is not None:                    # edge-mask methods: points
                ys, xs = np.nonzero(b["edge"])
                axp.plot(xs, ys, ".", color="#00e5ff", ms=0.5)
        axp.legend(handles=[Patch(facecolor=COLOR_UP, label=relabel("up (+1)", self.kind(k))),
                            Patch(facecolor=COLOR_NEUTRAL, label="neutral (0)"),
                            Patch(facecolor=COLOR_DOWN, label=relabel("down (−1)", self.kind(k)))],
                   loc="upper right", fontsize=7)
        ax.set_title(f"{k}: raw file (editing: {self.cTarget.currentText()})", fontsize=8)
        axp.set_title(self._processed_title(k), fontsize=8, color="#b00" if not self.ws.is_ready(k) else "black")
        for a in (ax, axp):
            a.set_xticks([])
            a.set_yticks([])
        n = 5 if p.get("gauss_compare") else (4 if p.get("qq_plot") else 3)   # only the panels that are shown
        edges = np.linspace(0, 10, n + 1).round().astype(int)
        axd = [f.add_subplot(gs[1, edges[i]:edges[i + 1]]) for i in range(n)]
        for a in axd:
            a.axis("off")
        if res is not None:
            draw_distributions(axd, res, self.kind(k))
        self.editor = re_.RegionEditor(ax, self._region_changed, self.cTool.currentText(), self.sSides.value(),
                                       p.get(target))
        self.cv.draw()
        if res is None:
            self.results.setPlainText("Not calculated yet.\nDraw the active area (and a background region), "
                                      "then press Calculate (step 5).")
        else:
            self.results.setPlainText(("OUT OF DATE: press Calculate\n\n" if im.get("stale") else "")
                                      + summary_text(res, self.kind(k)))

    # ---- regions
    def _region_changed(self, verts):
        k = self.key
        if k:
            self.ws.images[k]["params"][self.cTarget.currentData()] = list(verts)
            self.schedule()

    def _set_region(self, verts):
        k = self.key
        if k:
            self.ws.images[k]["params"][self.cTarget.currentData()] = list(verts) if verts else None
            self.ws.images[k]["stale"] = True
            self.redraw()
            self._refresh_sources()
            self._show_state()

    def whole_image(self):
        k = self.key
        if k:
            h, w = self.ws.size(k)
            self._set_region([(0, 0), (w - 1, 0), (w - 1, h - 1), (0, h - 1)])

    def clear_region(self):
        self._set_region(None)

    KEYS = {"left": (-1, 0), "right": (1, 0), "up": (0, -1), "down": (0, 1)}

    def _key(self, event):
        """Arrow keys move the outline being edited (SHIFT: 10 steps); the outline moves at once, the analysis
        follows when the keys are released (debounced)."""
        key = (event.key or "").replace("shift+", "")
        if key not in self.KEYS:
            return
        k = self.key
        target = self.cTarget.currentData()
        v = self.ws.images[k]["params"].get(target) if k else None
        if not v:
            return
        n = 10 if "shift" in (event.key or "") else 1
        dx, dy = self.KEYS[key]
        new = re_.move(v, dx * n * self.sStep.value(), dy * n * self.sStep.value())
        self.ws.images[k]["params"][target] = list(new)
        if self.editor:
            self.editor.set_verts(new)
        self.schedule()

    def turn(self, sign):
        k = self.key
        v = self.ws.images[k]["params"].get(self.cTarget.currentData()) if k else None
        if v:
            self._set_region(re_.rotate(v, sign * self.sAngle.value()))

    def copy_regions(self, *_, every=False):
        """Reuse the outlines (active area + background) of the image chosen in 'Outlines from'."""
        src = self.cFrom.currentText()
        if not src or src not in self.ws.images:
            QtWidgets.QMessageBox.warning(self, "Outlines", "Draw an active area on an image first, then pick it "
                                                            "in 'Outlines from'.")
            return
        targets = [k for k in self.ws.images if k != src] if every else [self.key]
        targets = [k for k in targets if k and k != src]
        scaled = []
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            for i, k in enumerate(targets):
                self.main.status(f"outlines of {src} -> {k} ({i + 1}/{len(targets)})…")
                QtWidgets.QApplication.processEvents()
                if self.ws.copy_regions(src, k, run=False):
                    scaled.append(k)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        msg = f"outlines of {src} applied to {len(targets)} image(s); press Calculate All"
        if scaled:
            msg += f"; scaled to the image size for {', '.join(scaled)} (check them)"
        self.main.status(msg)
        self.main.changed(self.key)
        self.populate()

    def copy_settings(self, *_, every=False):
        """Reuse the settings (not the outlines) of the image chosen in 'Settings from'."""
        src = self.cSetFrom.currentText()
        if not src or src not in self.ws.images:
            return
        if self.key == src:                         # keep the shown (maybe not yet stored) settings of the source
            self.ws.set_params(src, run=False, **self.params())
        targets = [k for k in self.ws.images if k != src] if every else [self.key]
        targets = [k for k in targets if k and k != src]
        for k in targets:
            self.ws.copy_settings(src, k)
        self.main.status(f"settings of {src} applied to {len(targets)} image(s); press Calculate All")
        self.populate()

    def save_region(self):
        k = self.key
        v = self.ws.images[k]["params"].get(self.cTarget.currentData()) if k else None
        if not v:
            return
        f, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save region", f"{k}_region.json", "Polygon (*.json)")
        if f:
            p = self.ws.images[k]["params"]
            re_.save_json(f, v, image=os.path.basename(self.ws.images[k]["path"]), m_threshold=p["m_threshold"],
                          tv_weight=p["tv_weight"], cal_mode=p.get("cal_mode"), cal_w=p.get("cal_w"), cal_h=p.get("cal_h"))

    def load_region(self):
        if not self.key:
            return
        f, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load region", "", "Polygon (*.json)")
        if f:
            verts, _ = re_.load_json(f)
            self._set_region(verts)

    # ---- images / export
    def remove(self):
        k = self.key
        if k:
            self.ws.remove(k)
            self.main.refresh()

    def export_dir(self):
        """The chosen export folder, or None (= the folder of the raw image)."""
        return self.eOut.text().strip() or None

    def raw_dir(self, k=None):
        k = k or self.key
        return os.path.dirname(os.path.abspath(self.ws.images[k]["path"])) if k in self.ws.images else ""

    def _show_out(self):
        """Placeholder = where the selected image would be exported by default."""
        d = self.raw_dir()
        self.eOut.setPlaceholderText(f"raw data folder: {d}" if d else "the raw image's folder")
        self.eOut.setToolTip(f"written to <folder>/<image name>_analysis/; empty = the raw data folder"
                             + (f"\n({d})" if d else ""))

    def choose_out(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Export folder", self.export_dir() or self.raw_dir())
        if d:
            self.eOut.setText(d)

    def clear_out(self):
        self.eOut.clear()

    def export(self):
        k = self.key
        if not k:
            return
        p = {**self.ws.images[k]["params"], "fig_format": self.cFmt.currentData(), "detailed": self.cDetail.isChecked(),
             "rigorous": DEFAULTS["rigorous"], "tiff": DEFAULTS["tiff"], "n_boot": DEFAULTS["n_boot"],
             "dpi": DEFAULTS["dpi"], "outdir": self.export_dir(), "shape": self.ws.size(k)}
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            out = run_export(self.ws.images[k]["path"], p)      # default: next to the raw image
            self.main.status(f"exported to {out}")
        except Exception as e:  # noqa: BLE001
            QtWidgets.QMessageBox.warning(self, "Export", str(e))
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
