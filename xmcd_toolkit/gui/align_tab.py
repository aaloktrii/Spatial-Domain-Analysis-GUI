"""Tab 2 · Align: put two analysed images on top of each other (GUI layer).

Neither image is fixed: edit A, B or both (rotation, shift, flips), rotate the whole view, align automatically
(first guess) or from clicked point pairs (reference method), and check with the correlation-vs-shift map."""
import numpy as np
from PySide6 import QtCore, QtWidgets

from ..core import transform
from ..plotting import compare_figures as cf
from .widgets import Canvas, button, combo, group, not_ready_note, row, spin


VIEW_PX = 1600   # the Align panels draw about this many pixels across (the analysis uses every pixel)

class AlignTab(QtWidgets.QWidget):
    def __init__(self, main):
        super().__init__()
        self.main = main
        self.pts_a, self.pts_b, self._axes, self._shown = [], [], None, None
        self.timer = QtCore.QTimer(self, singleShot=True, interval=60)
        self.timer.timeout.connect(self.redraw)

        self.groups = []
        self.cA, self.cB = QtWidgets.QComboBox(), QtWidgets.QComboBox()
        f = QtWidgets.QFormLayout()
        f.addRow("A (before):", self.cA)
        f.addRow("B (after):", self.cB)
        f.addRow(button("Swap A ↔ B", self.swap))
        self.groups.append(group("8. Pair (A before, B after)", f))

        self.eA, self.eB, self.eBoth = (QtWidgets.QRadioButton(t) for t in ("A", "B", "both"))
        self.eB.setChecked(True)
        self.sRot = spin(-180, 180, 0.1, 2, "+ = counter-clockwise")
        self.sDy, self.sDx = spin(-2000, 2000, 0.5, 1, "down (px)"), spin(-2000, 2000, 0.5, 1, "right (px)")
        self.cFlipH, self.cFlipV = QtWidgets.QCheckBox("Flip ↔"), QtWidgets.QCheckBox("Flip ↕")
        f = QtWidgets.QFormLayout()
        f.addRow("Edit:", row(self.eA, self.eB, self.eBoth))
        f.addRow("Rotation (°):", self.sRot)
        f.addRow("Shift dy / dx (px):", row(self.sDy, self.sDx))
        f.addRow(row(self.cFlipH, self.cFlipV, button("Reset", self.reset_tf)))
        self.groups.append(group("9. Move an image", f))

        self.cPick = QtWidgets.QCheckBox("Pick points (A, then B)")
        self.cPick.setToolTip("Click a feature in A, then the same feature in B; repeat, then Fit Points")
        self.cScale = QtWidgets.QCheckBox("Allow scale")
        self.lPts = QtWidgets.QLabel("0 pairs")
        self.cAngles = QtWidgets.QCheckBox("Also rotations ±2°")
        v = QtWidgets.QVBoxLayout()
        v.addWidget(button("Auto Align", self.auto, "rotation ± 3° + shift; moves the edited image onto the other", "map"))
        v.addWidget(self.cPick)
        v.addLayout(row(button("Fit Points", self.fit_points, "", "map"), self.cScale, self.lPts))
        v.addLayout(row(button("Undo Point", self.undo_point), button("Clear Points", self.clear_points)))
        v.addWidget(button("Check: Correlation vs Shift…", self.shift_check))
        v.addWidget(self.cAngles)
        self.lInfo = QtWidgets.QLabel("")
        self.lInfo.setWordWrap(True)
        v.addWidget(self.lInfo)
        self.groups.append(group("10. Align", v))

        self.sView = spin(-180, 180, 0.1, 2, "rotates every image (and the comparison region) together")
        self.cOverlay = combo(["magenta / green", "walls"])
        self.cFlat = QtWidgets.QCheckBox("Flatten + smooth the overlay (display only)")
        self.cFlat.setChecked(False)          # pictures as they are; the option only affects the overlay
        f = QtWidgets.QFormLayout()
        f.addRow("View rotation (°):", self.sView)
        f.addRow("Overlay:", self.cOverlay)
        f.addRow(self.cFlat)
        self.groups.append(group("11. View", f))

        self.cv = Canvas(self, (9, 4))
        self.cv.canvas.mpl_connect("button_press_event", self.click)
        self.view = self.cv

        for w in (self.cA, self.cB):
            w.currentIndexChanged.connect(self.load_tf)
        for r in (self.eA, self.eB, self.eBoth):
            r.toggled.connect(self.load_tf)
        for s in (self.sRot, self.sDy, self.sDx):
            s.valueChanged.connect(self.store_tf)
        for ch in (self.cFlipH, self.cFlipV):
            ch.toggled.connect(self.store_tf)
        self.sView.valueChanged.connect(self.view_changed)
        self.cOverlay.currentIndexChanged.connect(self.schedule)
        self.cFlat.toggled.connect(self.schedule)

    @property
    def ws(self):
        return self.main.ws

    def pair(self):
        return self.cA.currentText(), self.cB.currentText()

    def populate(self):
        keys = list(self.ws.images)
        for i, c in enumerate((self.cA, self.cB)):
            prev = c.currentText()
            c.blockSignals(True)
            c.clear()
            c.addItems(keys)
            if prev in keys:
                c.setCurrentText(prev)
            elif keys:
                c.setCurrentIndex(min(i, len(keys) - 1))
            c.blockSignals(False)
        self.sView.blockSignals(True)
        self.sView.setValue(self.ws.view_rot)
        self.sView.blockSignals(False)
        self.load_tf()

    def swap(self):
        a, b = self.pair()
        for c, t in ((self.cA, b), (self.cB, a)):
            c.blockSignals(True)
            c.setCurrentText(t)
            c.blockSignals(False)
        self.load_tf()

    def targets(self):
        a, b = self.pair()
        return [a] if self.eA.isChecked() else [b] if self.eB.isChecked() else list(dict.fromkeys((a, b)))

    def _values(self):
        return dict(rot=self.sRot.value(), dy=self.sDy.value(), dx=self.sDx.value(),
                    fliph=self.cFlipH.isChecked(), flipv=self.cFlipV.isChecked())

    def load_tf(self, *_):
        tg = [k for k in self.targets() if k in self.ws.images]
        if tg:
            t = self.ws.images[tg[0]]["tf"]
            sh = transform.shift_final(t, self.ws.view_rot)
            for w, v in ((self.sRot, t["rot"]), (self.sDy, sh[0]), (self.sDx, sh[1]),
                         (self.cFlipH, t["fliph"]), (self.cFlipV, t["flipv"])):
                w.blockSignals(True)
                w.setChecked(v) if isinstance(w, QtWidgets.QCheckBox) else w.setValue(v)
                w.blockSignals(False)
            self._shown = self._values()
        self.clear_points()
        self.schedule()

    def store_tf(self, *_):
        tg = [k for k in self.targets() if k in self.ws.images]
        if not tg:
            return
        new, old = self._values(), self._shown or self._values()
        for k in tg:   # one image: set the values; both: apply the change, so their relative alignment is kept
            t = dict(self.ws.images[k]["tf"])
            sh = transform.shift_final(t, self.ws.view_rot)
            if len(tg) == 1:
                t.update(rot=new["rot"], fliph=new["fliph"], flipv=new["flipv"])
                t = transform.with_shift_final(t, self.ws.view_rot, new["dy"], new["dx"])
            else:
                t.update(rot=(t["rot"] + new["rot"] - old["rot"] + 180) % 360 - 180,
                         fliph=t["fliph"] ^ (new["fliph"] != old["fliph"]), flipv=t["flipv"] ^ (new["flipv"] != old["flipv"]))
                t = transform.with_shift_final(t, self.ws.view_rot, sh[0] + new["dy"] - old["dy"], sh[1] + new["dx"] - old["dx"])
            self.ws.set_transform(k, **t)
        self._shown = new
        self.schedule()

    def reset_tf(self):
        for k in self.targets():
            if k in self.ws.images:
                self.ws.set_transform(k, **transform.TF_DEFAULT)
        self.load_tf()

    def view_changed(self, v):
        self.ws.set_view_rotation(v)
        self.load_tf()

    def _single(self, what):
        if self.eBoth.isChecked():
            QtWidgets.QMessageBox.information(self, what, "Edit A or B: that image is moved onto the other.")
            return None
        return self.targets()[0]

    def _progress(self, title):
        prog = QtWidgets.QProgressDialog(title, None, 0, 100, self)
        prog.setWindowModality(QtCore.Qt.WindowModal)
        prog.show()

        def cb(fr):
            prog.setValue(int(100 * fr))
            QtWidgets.QApplication.processEvents()
        return prog, cb

    def auto(self):
        k = self._single("Auto align")
        a, b = self.pair()
        if not k or a == b:
            return
        prog, cb = self._progress("Aligning…")
        rot, sh, _ = self.ws.auto_align(k, b if k == a else a, progress=cb)
        prog.close()
        self.lInfo.setText(f"auto: rotation {rot:+.2f}°, shift added ({sh[0]:+.1f}, {sh[1]:+.1f}) px. Check the overlay: "
                           "changed domains can pull it off; point pairs are more reliable.")
        self.load_tf()

    def click(self, ev):
        if not self.cPick.isChecked() or self._axes is None or ev.button != 1 or self.cv.busy_mode():
            return
        p = (ev.ydata, ev.xdata)
        if ev.inaxes is self._axes[0] and len(self.pts_a) == len(self.pts_b):
            self.pts_a.append(p)
        elif ev.inaxes is self._axes[1] and len(self.pts_b) < len(self.pts_a):
            self.pts_b.append(p)
        else:
            self.lInfo.setText("click the feature in A first, then the same feature in B")
            return
        self._label()
        self.redraw()

    def _label(self):
        self.lPts.setText(f"{len(self.pts_b)} pairs" + (" +1" if len(self.pts_a) > len(self.pts_b) else ""))

    def undo_point(self):
        if len(self.pts_a) > len(self.pts_b):
            self.pts_a.pop()
        elif self.pts_b:
            self.pts_a.pop()
            self.pts_b.pop()
        self._label()
        self.redraw()

    def clear_points(self):
        self.pts_a, self.pts_b = [], []
        self._label()

    def fit_points(self):
        k = self._single("Fit points")
        n = len(self.pts_b)
        if not k or n < 1:
            return
        src, dst = (self.pts_a[:n], self.pts_b) if k == self.pair()[0] else (self.pts_b, self.pts_a[:n])
        th, s, t, rms = self.ws.fit_points(k, src, dst, self.cScale.isChecked())
        self.lInfo.setText(f"fit ({n} pairs): rotation {th:+.3f}°, scale {s:.4f}, shift ({t[0]:+.2f}, {t[1]:+.2f}) px, "
                           f"residual {rms:.2f} px")
        self.load_tf()

    def shift_check(self):
        a, b = self.pair()
        if a not in self.ws.images or b not in self.ws.images or a == b:
            return
        prog, cb = self._progress("Correlating…")
        res = self.ws.shift_check(a, b, angles=(-2.0, -1.0, 0.0, 1.0, 2.0) if self.cAngles.isChecked() else (0.0,),
                                  progress=cb)
        prog.close()
        ShiftDialog(self, res, b).exec()

    def schedule(self, *_):
        tabs = getattr(self.main, "view_tabs", None)
        if tabs is not None and tabs.currentWidget() is self.view:   # hidden: redrawn when its tab is opened
            self.timer.start()

    def redraw(self):
        a, b = self.pair()
        if a not in self.ws.images or b not in self.ws.images:
            self.cv.fig.clear()
            self.cv.draw()
            return
        note = not_ready_note(self.ws, (a, b))
        if note:
            self.cv.fig.clear()
            self._axes = None
            self.cv.fig.text(0.5, 0.5, note, ha="center", va="center", fontsize=9)
            self.cv.draw()
            return
        keep = (self._axes[0].get_xlim(), self._axes[0].get_ylim()) if self._axes is not None else None
        fl = 40.0 if self.cFlat.isChecked() else 0.0
        sm = 1.5 if self.cFlat.isChecked() else 0.0
        ed = self.targets()
        step = max(1, max(self.ws.shape) // VIEW_PX)       # drawing only: ~VIEW_PX screen pixels across
        self._axes = cf.draw_align(self.cv.fig, cf.display(self.ws.final_view(a, "m", step), smooth=sm, flatten=fl),
                                   cf.display(self.ws.final_view(b, "m", step), smooth=sm, flatten=fl),
                                   (f"A: {cf.short(a)}" + (" [edit]" if a in ed else ""),
                                    f"B: {cf.short(b)}" + (" [edit]" if b in ed else "")),
                                   self.cOverlay.currentText(), self.pts_a, self.pts_b, self.ws.compare_roi(a, b),
                                   pictures=(self.ws.final_view(a, "picture", step),
                                             self.ws.final_view(b, "picture", step)),
                                   step=step, shape=self.ws.shape)
        if keep:
            self._axes[0].set_xlim(keep[0])
            self._axes[0].set_ylim(keep[1])
        self.cv.draw()


class ShiftDialog(QtWidgets.QDialog):
    """Correlation-vs-shift result, with the option to move B to the best offset."""

    def __init__(self, tab, res, keyB):
        super().__init__(tab)
        self.setWindowTitle("Alignment check: correlation vs shift")
        self.resize(900 if len(res["angles"]) > 1 else 620, 560)
        cv = Canvas(self, (7, 4.5))
        cf.draw_shift_check(cv.fig, res)
        ang, dy, dx, rb = res["best"]
        spread = float(np.nanstd(res["r"]))
        verdict = ("the best offset is the current alignment (within 1 px)" if abs(dy) <= 1 and abs(dx) <= 1 and ang == 0
                   else f"B is best moved by dy {dy:+d}, dx {dx:+d} px" + (f" and {ang:+.1f}°" if ang else ""))
        if rb < 0.1:
            verdict += "; r stays below 0.1 at every offset: no common pattern (not a misalignment)"
        txt = QtWidgets.QLabel(f"r at the current alignment {res['r0']:+.3f}; best {rb:+.3f}; spread {spread:.3f}. {verdict}.")
        txt.setWordWrap(True)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        apply = bb.addButton("Move B to the best offset", QtWidgets.QDialogButtonBox.ApplyRole)
        apply.setEnabled(not (dy == 0 and dx == 0 and ang == 0))

        def do_apply():
            ws = tab.ws
            t = dict(ws.images[keyB]["tf"])
            sh = transform.shift_final(t, ws.view_rot)
            t["rot"] += ang
            ws.set_transform(keyB, **transform.with_shift_final(t, ws.view_rot, sh[0] + dy, sh[1] + dx))
            tab.load_tf()
            self.accept()

        apply.clicked.connect(do_apply)
        bb.rejected.connect(self.reject)
        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(cv, 1)
        lay.addWidget(txt)
        lay.addWidget(bb)
