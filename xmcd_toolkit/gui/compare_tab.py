"""Tab 3 · Compare: pixel-by-pixel comparison of the pair chosen in 2 · Align, with the population changes
(GUI layer). Uses the three-state maps of 1 · Domains, so the flip map and the population numbers rest on the
same classification (same m threshold) as the domain analysis."""
import os
import csv
import json
from pathlib import Path

from PySide6 import QtCore, QtWidgets

from ..colors import SWITCH_PALETTES
from ..core import populations
from ..plotting import compare_figures as cf
from . import region_editor as re_
from .widgets import Canvas, button, combo, group, heading, ispin, row, spin, splitter, not_ready_note

METRICS = [("roi_px", "ROI (px)", "{:d}"), ("classified_pct", "classified (% of ROI)", "{:.1f}"),
           ("n_stayed_up", "stayed up (px)", "{:d}"), ("n_stayed_down", "stayed down (px)", "{:d}"),
           ("n_up_to_down", "up → down (px)", "{:d}"), ("n_down_to_up", "down → up (px)", "{:d}"),
           ("up_to_down_pct", "up → down (% of up)", "{:.1f}"), ("down_to_up_pct", "down → up (% of down)", "{:.1f}"),
           ("switched_pct", "switched (% of classified)", "{:.1f}"), ("memory_from_flips", "memory 1 − Σ / 100", "{:+.3f}"),
           ("r", "memory r (global)", "{:+.3f}"), ("r_ci_lo", "r 95 % low", "{:+.3f}"), ("r_ci_hi", "r 95 % high", "{:+.3f}"),
           ("r_local_median", "local r median", "{:+.3f}"),
           ("wall_shift_B_to_A", "wall shift B→A", "{:.1f}"), ("wall_shift_A_to_B", "wall shift A→B", "{:.1f}")]
POP_ROWS = [("up", "f↑"), ("neutral", "f₀"), ("down", "f↓"), ("mz", "⟨m_z⟩"), ("mz_hat", "⟨m_z⟩ without f₀")]
CHANGE_ROWS = [("d_up", "Δf↑"), ("d_neutral", "Δf₀"), ("d_down", "Δf↓"), ("d_mz", "Δ⟨m_z⟩"),
               ("d_mz_direct", "  from up ↔ down (2 × switched)"), ("d_mz_via_neutral", "  through neutral"),
               ("changed_lower", "changed class, at least"), ("changed_upper", "changed class, at most"),
               ("d_mz_hat", "Δ⟨m_z⟩ without f₀"), ("relative_up", "up area, relative change"),
               ("relative_down", "down area, relative change")]


def _f(v, fmt):
    return "—" if v is None or (isinstance(v, float) and v != v) else fmt.format(v)


class CompareTab(QtWidgets.QWidget):
    def __init__(self, main):
        super().__init__()
        self.main = main
        self.result = None
        self.editor = None
        self.timer = QtCore.QTimer(self, singleShot=True, interval=150)
        self.timer.timeout.connect(self.update_all)

        self.groups = []
        self.lPair = QtWidgets.QLabel("")
        self.lPair.setWordWrap(True)
        self.sWin = spin(2, 60, 0.5, 1, "Gaussian window of the local correlation (px)", 10)
        self.sBoot = ispin(100, 10000, "bootstrap resamples of r", 1000)
        self.sBlock = spin(0.2, 5, 0.1, 1, "bootstrap block size (µm; 32 px if uncalibrated)", 1.0)
        f = QtWidgets.QFormLayout()
        f.addRow(self.lPair)
        note = QtWidgets.QLabel("Up / down from step 5 (same m threshold).")
        note.setWordWrap(True)
        f.addRow(note)
        f.addRow("Local r window σ (px):", self.sWin)
        f.addRow("Bootstrap n:", self.sBoot)
        f.addRow("Block (µm):", self.sBlock)
        self.groups.append(group("12. Compare", f))

        self.cPalette = combo(list(SWITCH_PALETTES), "colours of switched pixels; stayed up / down keep the map colours")
        self.cCrop = QtWidgets.QCheckBox("Crop to the ROI")
        self.cCrop.setChecked(True)
        self.cPop = combo([("common ROI (same pixels)", "common"), ("each image's own active area", "own")])
        f = QtWidgets.QFormLayout()
        f.addRow("Switched colours:", self.cPalette)
        f.addRow("Populations from:", self.cPop)
        f.addRow(self.cCrop)
        self.groups.append(group("13. Display", f))

        self.cDraw = QtWidgets.QCheckBox("Draw an extra comparison region on A")
        v = QtWidgets.QVBoxLayout()
        lab = QtWidgets.QLabel("ROI = active area of A ∩ active area of B (after alignment)")
        lab.setWordWrap(True)
        v.addWidget(lab)
        v.addWidget(self.cDraw)
        v.addWidget(button("Clear Region", self.clear_region))
        v.addLayout(row(button("Save Region…", self.save_region), button("Load Region…", self.load_region)))
        self.groups.append(group("14. Comparison region", v))
        self.lStatus = QtWidgets.QLabel("")
        self.lStatus.setWordWrap(True)
        v = QtWidgets.QVBoxLayout()
        v.addWidget(button("Export PDF + CSV…", self.export, "editable PDF (vector maps) and a CSV of every number", "handover"))
        v.addWidget(button("Δ⟨m_z⟩ Between All Images…", self.mz_matrix))
        v.addWidget(self.lStatus)
        self.groups.append(group("15. Export", v))

        self.cv = Canvas(self, (9, 3.6))
        self.cvPop = Canvas(self, (6, 3))
        self.table = QtWidgets.QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["comparison", "A → B"])
        self.pop = QtWidgets.QTableWidget(0, 4)
        self.pop.setHorizontalHeaderLabels(["population", "A", "B", "change"])
        for t in (self.table, self.pop):
            t.verticalHeader().setVisible(False)
            t.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
            t.horizontalHeader().setStretchLastSection(True)
            t.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        def titled(text, w):
            box = QtWidgets.QWidget()
            lay = QtWidgets.QVBoxLayout(box)
            lay.setContentsMargins(4, 4, 4, 4)
            lay.addWidget(heading(text))
            lay.addWidget(w)
            return box

        pop_tabs = QtWidgets.QTabWidget()
        pop_tabs.setDocumentMode(True)
        pop_tabs.addTab(self.pop, "Table")
        pop_tabs.addTab(self.cvPop, "Net change")
        bottom = splitter(titled("Comparison", self.table), titled("Populations", pop_tabs), sizes=[380, 600])
        self.view = splitter(self.cv, bottom, vertical=True, sizes=[470, 380])

        for w in (self.sWin, self.sBlock):
            w.valueChanged.connect(self.schedule)
        self.sBoot.valueChanged.connect(self.schedule)
        for w in (self.cPalette, self.cPop):
            w.currentIndexChanged.connect(self.schedule)
        self.cCrop.toggled.connect(self.schedule)
        self.cDraw.toggled.connect(self.schedule)

    @property
    def ws(self):
        return self.main.ws

    def schedule(self, *_):
        tabs = getattr(self.main, "view_tabs", None)
        if tabs is not None and tabs.currentWidget() is self.view:   # hidden: redrawn when its tab is opened
            self.timer.start()

    def update_all(self):
        a, b = self.main.align.pair()
        if a not in self.ws.images or b not in self.ws.images or a == b:
            self.lPair.setText("choose two different images in step 8")
            return
        note = not_ready_note(self.ws, (a, b))
        if note:
            self.lPair.setText(note)
            self.cv.fig.clear()
            self.cv.fig.text(0.5, 0.5, note, ha="center", va="center", fontsize=9)
            self.cv.draw()
            return
        self.lPair.setText(f"<b>A</b> {a}  →  <b>B</b> {b}")
        self.lPair.setToolTip("pair and alignment: steps 8–10")
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            self.result = self.ws.compare(a, b, window_px=self.sWin.value(), block_um=self.sBlock.value(),
                                          n_boot=self.sBoot.value())
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        m, L = self.result["metrics"], self.result["layers"]
        crop = self.cCrop.isChecked() and not self.cDraw.isChecked()
        ax = cf.draw_compare(self.cv.fig, self.ws.final(a, "mz"), self.ws.final(b, "mz"), L["flip"], L["r_local"],
                             L["roi"], m, (a, b), self.cPalette.currentText(), crop=crop)
        if self.editor:
            self.editor.disconnect()
            self.editor = None
        if self.cDraw.isChecked():
            self.editor = re_.RegionEditor(ax[0], self._region_changed, "polygon", verts=self.ws.compare_polygon)
        self.cv.draw()
        self._fill(m)

    def _fill(self, m):
        unit = m.get("wall_shift_unit", "px")
        self.table.setRowCount(len(METRICS))
        for i, (k, lab, fmt) in enumerate(METRICS):
            lab = lab + (f" ({unit})" if k.startswith("wall") else "")
            for j, v in enumerate((lab, _f(m.get(k), fmt))):
                self.table.setItem(i, j, QtWidgets.QTableWidgetItem(v))
        p = m["populations_common_roi" if self.cPop.currentData() == "common" else "populations_own_roi"]
        rows = [(lab, f"{100 * p['A'][k]:.2f} %", f"{100 * p['B'][k]:.2f} %",
                 f"{100 * (p['B'][k] - p['A'][k]):+.2f} %") for k, lab in POP_ROWS]
        rows += [(lab, "", "", (f"{100 * p['change'][k]:+.2f} %" if not k.startswith("changed")
                                else f"{100 * p['change'][k]:.2f} %")) for k, lab in CHANGE_ROWS]
        self.pop.setRowCount(len(rows))
        for i, r in enumerate(rows):
            for j, v in enumerate(r):
                self.pop.setItem(i, j, QtWidgets.QTableWidgetItem(v))
        cf.draw_populations(self.cvPop.fig, p["change"], "net change (% of the region)")
        self.cvPop.draw()
        self.lStatus.setText("Populations are net changes of the counts; the flip map shows the actual switching.")

    # ---- comparison region
    def _region_changed(self, verts):
        self.ws.compare_polygon = list(verts)
        self.schedule()

    def clear_region(self):
        self.ws.compare_polygon = None
        self.schedule()

    def save_region(self):
        if self.ws.compare_polygon:
            f, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save region", "compare_region.json", "Polygon (*.json)")
            if f:
                re_.save_json(f, self.ws.compare_polygon, frame="common (aligned) grid", view_rot=self.ws.view_rot)

    def load_region(self):
        f, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load region", "", "Polygon (*.json)")
        if f:
            self.ws.compare_polygon = re_.load_json(f)[0]
            self.cDraw.setChecked(True)
            self.schedule()

    # ---- export
    def export(self):
        if not self.result:
            return
        a0, _ = self.main.align.pair()        # the export folder of step 7, else the raw data folder of image A
        start = os.path.join(self.main.domains.export_dir() or self.main.domains.raw_dir(a0), "compare_A_B.pdf")
        f, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export", start, "PDF (*.pdf)")
        if not f:
            return
        f = Path(f)
        cf.save_pdf(self.cv.fig, f)
        cf.save_pdf(self.cvPop.fig, f.with_name(f.stem + "_populations.pdf"))
        a, b = self.main.align.pair()
        m = self.result["metrics"]
        with open(f.with_suffix(".csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["A", a])
            w.writerow(["B", b])
            w.writerow(["view_rotation_deg", self.ws.view_rot])
            for k in (a, b):
                w.writerow([f"transform_{k}", json.dumps(self.ws.images[k]["tf"])])
                w.writerow([f"m_threshold_{k}", self.ws.images[k]["params"]["m_threshold"]])
            w.writerow(["settings", json.dumps(self.result["settings"])])
            w.writerow([])
            for k, lab, _ in METRICS:
                w.writerow([k, m.get(k)])
            for which in ("populations_common_roi", "populations_own_roi"):
                p = m[which]
                for side in ("A", "B"):
                    for k in ("up", "neutral", "down", "mz", "up_hat", "mz_hat", "n"):
                        w.writerow([f"{which}.{side}.{k}", p[side][k]])
                for k, v in p["change"].items():
                    w.writerow([f"{which}.change.{k}", json.dumps(v) if isinstance(v, dict) else v])
        self.main.status(f"exported {f.name}, {f.stem}_populations.pdf and {f.with_suffix('.csv').name}")

    def mz_matrix(self):
        keys = list(self.ws.images)
        if len(keys) < 2:
            return
        common = self.cPop.currentData() == "common"
        names, M = populations.mz_matrix(self.ws.population_table(keys, common=common))
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Δ⟨m_z⟩ (% of the region), row = from, column = to")
        t = QtWidgets.QTableWidget(len(names), len(names))
        t.setHorizontalHeaderLabels(names)
        t.setVerticalHeaderLabels(names)
        for i in range(len(names)):
            for j in range(len(names)):
                t.setItem(i, j, QtWidgets.QTableWidgetItem(f"{100 * M[i, j]:+.2f}"))
        lay = QtWidgets.QVBoxLayout(dlg)
        lay.addWidget(QtWidgets.QLabel("common ROI of all images" if common else "each image's own active area"))
        lay.addWidget(t)
        dlg.resize(120 + 110 * len(names), 120 + 32 * len(names))
        dlg.exec()
