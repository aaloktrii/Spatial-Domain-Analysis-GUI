#!/usr/bin/env python3
"""
gui.py -- desktop GUI for the XMCD-PEEM domain toolkit (polygon workflow).

  1. Browse to load an image  -> it displays immediately.
  2. Draw a POLYGON over the active area (click vertices; close it).
     -> the three-state map and the three distributions update live.
  3. Adjust the m-threshold (slider) -> preview updates.
  4. Export all files (figures, distributions, TIFFs, table) next to the image.

The field of view corresponds to the whole image:  um/px = FoV_width / width_px.

Run:  python gui.py     (drag-drop optional; Browse always works.)
"""
import os
import sys
import json
import queue
import threading
import subprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from matplotlib.widgets import PolygonSelector, RectangleSelector, EllipseSelector

from PIL import Image
import gui_core as core
from defaults import DEFAULTS
from xmcd_toolkit import COLOR_UP, COLOR_DOWN, CMAP3_HEX

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    _HAS_DND = True
except Exception:
    _HAS_DND = False

CMAP3 = ListedColormap(CMAP3_HEX)
NORM3 = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], CMAP3.N)


class _ToolTip:
    """Small popup with `text`, shown while the mouse is over `widget`."""
    def __init__(self, widget, text, wraplength=280):
        self.widget = widget
        self.text = text
        self.wraplength = wraplength
        self.tip = None
        widget.bind("<Enter>", self._show)
        widget.bind("<Leave>", self._hide)
        widget.bind("<ButtonPress>", self._hide)

    def _show(self, event=None):
        if self.tip is not None or not self.text:
            return
        x = self.widget.winfo_rootx() + 14
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        self.tip = tk.Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        try:
            self.tip.wm_attributes("-topmost", True)
        except Exception:
            pass
        self.tip.wm_geometry(f"+{x}+{y}")
        # explicit foreground: on macOS dark mode tk.Label defaults to light
        # text, which is invisible on the pale-yellow tooltip background.
        tk.Label(self.tip, text=self.text, justify="left",
                background="#ffffe0", foreground="#111111",
                relief="solid", borderwidth=1, wraplength=self.wraplength,
                font=("", 9)).pack(ipadx=5, ipady=3)

    def _hide(self, event=None):
        if self.tip is not None:
            self.tip.destroy()
            self.tip = None


def _help_row(parent, label_text, help_text, **pack_kw):
    """A label + a small '?' hover-help icon on one row. Returns the row Frame."""
    row = ttk.Frame(parent)
    row.pack(fill="x", **pack_kw)
    ttk.Label(row, text=label_text).pack(side="left")
    q = ttk.Label(row, text=" (?)", foreground="#06c", cursor="question_arrow")
    q.pack(side="left")
    _ToolTip(q, help_text)
    return row


class App:
    def __init__(self, root):
        self.root = root
        self.path = None
        self.img = None          # displayed grayscale image
        self.poly = None         # active-area polygon vertices [(x,y), ...]
        self.bg_poly = None      # background-region polygon vertices [(x,y), ...]
        self.res = None          # last analysis result
        self.q = queue.Queue()
        self._busy = False       # a preview/export thread is running
        self._preview_after = None   # pending debounced preview id
        root.title("XMCD-PEEM domain analysis")
        root.geometry("1240x780")

        main = ttk.Frame(root, padding=6); main.pack(fill="both", expand=True)

        # Left sidebar is scrollable: the stacked controls (calibration,
        # sliders, regions, export, status, metrics) can add up to more
        # height than the window on a short screen, leaving the bottom of
        # the sidebar (status/metrics) unreachable. Canvas + Scrollbar
        # wrapping a plain Frame is the standard Tk pattern for this; `left`
        # stays the parent every control below packs into, so nothing else
        # in this method needs to change.
        left_container = ttk.Frame(main); left_container.pack(side="left", fill="y", padx=(0, 8))
        left_canvas = tk.Canvas(left_container, width=260, highlightthickness=0)
        left_scroll = ttk.Scrollbar(left_container, orient="vertical", command=left_canvas.yview)
        left_canvas.configure(yscrollcommand=left_scroll.set)
        left_canvas.pack(side="left", fill="y", expand=True)
        left_scroll.pack(side="right", fill="y")
        left = ttk.Frame(left_canvas)
        left_window = left_canvas.create_window((0, 0), window=left, anchor="nw")
        left.bind("<Configure>",
                 lambda e: left_canvas.configure(scrollregion=left_canvas.bbox("all")))
        left_canvas.bind("<Configure>",
                         lambda e: left_canvas.itemconfig(left_window, width=e.width))

        def _scroll_left(direction):
            left_canvas.yview_scroll(direction, "units")
        def _bind_wheel(event):
            left_canvas.bind_all("<MouseWheel>", lambda e: _scroll_left(int(-1 * (e.delta / 120))))
            left_canvas.bind_all("<Button-4>", lambda e: _scroll_left(-1))   # Linux scroll up
            left_canvas.bind_all("<Button-5>", lambda e: _scroll_left(1))    # Linux scroll down
        def _unbind_wheel(event):
            left_canvas.unbind_all("<MouseWheel>")
            left_canvas.unbind_all("<Button-4>")
            left_canvas.unbind_all("<Button-5>")
        left_canvas.bind("<Enter>", _bind_wheel)
        left_canvas.bind("<Leave>", _unbind_wheel)

        # ---- dedicated Results sidebar -------------------------------------
        # Its own column between the controls and the figure, so the numeric
        # readout has a fixed home: not buried at the bottom of the scrollable
        # control stack, and never drawn over the image.
        results_col = ttk.Frame(main)
        results_col.pack(side="left", fill="y", padx=(0, 8))
        results_box = ttk.LabelFrame(results_col, text="", padding=6)
        results_box.pack(fill="both", expand=True)
        # hover-help header explaining the correlation length (and how it
        # relates to the other readout quantities), since xi is the robust
        # domain-size measure now shown in the readout below.
        _xi_help = (
            "Correlation length ξ (domain size)\n"
            "\n"
            "ξ is the distance over which the magnetization stays "
            "correlated -- i.e. the characteristic SIZE of a domain. It is "
            "the lag at which the 2-D spatial autocorrelation of a state's "
            "map falls to 1/e.\n"
            "\n"
            "Computed by FFT autocorrelation of the up (or down) region -- "
            "NO threshold on a continuous field and NO skeleton -- so it is "
            "far more robust to the denoising (tv_weight) and m-threshold "
            "choice than the skeleton width (ξ drifts ~10% across "
            "tv_weight; the skeleton width drifts 2-3x). Report ξ as the "
            "characteristic domain scale; treat the skeleton width as a "
            "tv-tagged local-thickness measure.\n"
            "\n"
            "xi_up vs xi_down are the up/down domain sizes: equal ⇒ "
            "balanced (demagnetized); unequal ⇒ the majority (net-"
            "magnetized) state has the larger domains -- an independent "
            "structural check on the net-magnetization direction."
        )
        _hdr = ttk.Label(results_col, text="  what is ξ (corr. length)?  (?)",
                         foreground="#06c", cursor="question_arrow")
        _hdr.pack(anchor="w", before=results_box)
        _ToolTip(_hdr, _xi_help, wraplength=360)
        # "Results (?)" title row with the copyright tooltip, above everything.
        _rhdr = ttk.Frame(results_col); _rhdr.pack(anchor="w", before=_hdr)
        ttk.Label(_rhdr, text="Results", font=("", 11, "bold")).pack(side="left")
        _rq = ttk.Label(_rhdr, text=" (?)", foreground="#06c", cursor="question_arrow")
        _rq.pack(side="left")
        _ToolTip(_rq, "© Aalok Tiwari")
        # plain tk.Label (not ttk): ttk styling ignores bg/fg on macOS, where
        # the themed default can render dark and swallow the text. Explicit
        # white background + near-black text keeps it readable in both light
        # and dark system appearance.
        self.metrics = tk.Label(results_box,
                                text="(load an image and draw an\nactive area to see results)",
                                font=("Menlo", 10), justify="left", anchor="nw",
                                bg="#ffffff", fg="#111111",
                                relief="solid", borderwidth=1, padx=8, pady=8)
        self.metrics.pack(anchor="nw", fill="both", expand=True)

        right = ttk.Frame(main); right.pack(side="right", fill="both", expand=True)

        # ---- file / drop ----
        self.drop = tk.Label(left, text="Drop image here\nor click Browse",
                             relief="ridge", borderwidth=2, width=32, height=3, bg="#eef")
        self.drop.pack(pady=(0, 4))
        if _HAS_DND:
            try:
                self.drop.drop_target_register(DND_FILES)
                self.drop.dnd_bind("<<Drop>>", self._on_drop)
            except Exception:
                pass
        _brow = ttk.Frame(left); _brow.pack(fill="x")
        ttk.Button(_brow, text="Browse image...", command=self._browse).pack(
            side="left", fill="x", expand=True)
        _bq = ttk.Label(_brow, text=" (?)", foreground="#06c", cursor="question_arrow")
        _bq.pack(side="left")
        _ToolTip(_bq,
                 "Accepted image formats\n\n"
                 "Any single image PIL can read: PNG, TIFF/TIF (8-bit, 16-bit, "
                 "and float), JPG/JPEG, BMP. 16-bit / float TIFFs are read at "
                 "full depth. For colour PNG/JPG screenshots, acquisition-"
                 "software chrome (title bar, labels) is auto-cropped by "
                 "keeping the largest near-grayscale region.\n\n"
                 "Not supported: multi-frame stacks, PDFs, .npy arrays, or "
                 "raw x,y,mz text (use the library loader for text).")
        self.file_lbl = ttk.Label(left, text="(no file)", wraplength=230, foreground="#555")
        self.file_lbl.pack(pady=(2, 6))

        self._build_controls(left)

        a = ttk.LabelFrame(left, text="Regions", padding=6); a.pack(fill="x", pady=(8, 2))
        ttk.Button(a, text="Use whole image (active)", command=self._whole_image).pack(fill="x", pady=1)
        r = ttk.Frame(a); r.pack(fill="x", pady=1)
        ttk.Button(r, text="Clear active", command=self._clear_poly).pack(side="left", expand=True, fill="x", padx=(0, 2))
        ttk.Button(r, text="Clear background", command=self._clear_bg).pack(side="left", expand=True, fill="x")
        r = ttk.Frame(a); r.pack(fill="x", pady=1)
        ttk.Button(r, text="Save polygon...", command=self._save_poly).pack(side="left", expand=True, fill="x", padx=(0, 2))
        ttk.Button(r, text="Load polygon...", command=self._load_poly).pack(side="left", expand=True, fill="x")

        # publication figure format for the export (png / eps / pdf)
        self._FIGFMT_CHOICES = [("PNG (raster)", "png"),
                                ("EPS (vector, journals)", "eps"),
                                ("PDF (vector)", "pdf")]
        _ff_disp = {v: d for d, v in self._FIGFMT_CHOICES}
        _ffrow = ttk.Frame(left); _ffrow.pack(fill="x", pady=(8, 0))
        ttk.Label(_ffrow, text="Figure format", width=13).pack(side="left")
        self.fig_format_var = tk.StringVar(value=_ff_disp[DEFAULTS["fig_format"]])
        _ffc = ttk.Combobox(_ffrow, textvariable=self.fig_format_var, state="readonly",
                            values=[d for d, _v in self._FIGFMT_CHOICES])
        _ffc.pack(side="left", fill="x", expand=True, padx=(4, 0))
        _ToolTip(_ffc,
                 "Format of the publication images in figures/ (and the\n"
                 "12-panel pipeline). EPS/PDF are vector for journal submission\n"
                 "-- image data is rasterized inside, text/boundary lines stay\n"
                 "vector. PNG is smaller for quick viewing.")

        self.detailed_var = tk.BooleanVar(value=DEFAULTS["detailed"])
        _dchk = ttk.Checkbutton(
            left, text="Detailed figures (diagnostics/)",
            variable=self.detailed_var)
        _dchk.pack(anchor="w", pady=(6, 0))
        _ToolTip(_dchk,
                 "Off (default): export writes statistics.txt + the clean\n"
                 "publication images in figures/.\n"
                 "On: also writes a diagnostics/ folder -- the 12-panel\n"
                 "step-by-step pipeline, width-method comparison, size/fit\n"
                 "distributions, fits.txt, and (if TIFF on) publication TIFFs.")

        ttk.Button(left, text="Export all files", command=self._export).pack(fill="x", pady=(4, 0))
        self.status = ttk.Label(left, text="load an image to begin",
                                foreground="#070", wraplength=230)
        self.status.pack(pady=6)

        # ---- figure: top image (polygon) + up to 5 distribution axes ----
        # (width, period, up/down probability, an optional Q-Q plot, and an
        # optional log-normal-vs-Gaussian comparison). The 4th/5th axes
        # always exist in the grid but are hidden (axis("off")) when their
        # checkboxes are unticked, so toggling them doesn't require
        # rebuilding the canvas/toolbar.
        self.fig = Figure(figsize=(8.5, 7.0), dpi=100)
        gs = GridSpec(2, 5, height_ratios=[3, 1.2], hspace=0.32, wspace=0.35,
                      figure=self.fig)
        self.ax_img = self.fig.add_subplot(gs[0, :]); self.ax_img.axis("off")
        self.ax_d = [self.fig.add_subplot(gs[1, i]) for i in range(5)]
        for a in self.ax_d:
            a.axis("off")
        self.ax_img.set_title("load an image, then draw a polygon over the domains")
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        tb = NavigationToolbar2Tk(self.canvas, right, pack_toolbar=False)
        tb.update(); tb.pack(side="bottom", fill="x")

        bottom_row = ttk.Frame(right); bottom_row.pack(side="bottom", fill="x")
        ttk.Label(bottom_row, text="Tip: use the toolbar to zoom/pan for precise "
                                   "drawing; turn zoom/pan OFF before clicking "
                                   "polygon vertices.",
                  foreground="#555", wraplength=520, justify="left"
                  ).pack(side="left", anchor="sw", pady=(4, 0))
        # packed between the tip (left) and transform panel (right) so it
        # sits centered along the bottom of the plot area
        self._build_dist_toggles_panel(bottom_row)
        self._build_transform_panel(bottom_row)
        self.selector = None

    # ------------------------------------------------------------------
    def _build_transform_panel(self, parent):
        """Translate/rotate controls for the loaded/drawn shape, anchored to
        the bottom-right corner of the window."""
        adj = ttk.LabelFrame(parent, text="Translate / Rotate", padding=6)
        adj.pack(side="right", anchor="se")

        r = ttk.Frame(adj); r.pack(fill="x")
        ttk.Label(r, text="step").pack(side="left")
        self.move_step = tk.StringVar(value="5")
        ttk.Entry(r, textvariable=self.move_step, width=5).pack(side="left")
        ttk.Label(r, text="px").pack(side="left", padx=(2, 10))
        ttk.Label(r, text="angle").pack(side="left")
        self.rot_step = tk.StringVar(value="5")
        ttk.Entry(r, textvariable=self.rot_step, width=5).pack(side="left")
        ttk.Label(r, text="deg").pack(side="left", padx=(2, 0))

        xy = ttk.Frame(adj); xy.pack(pady=(4, 2))
        ttk.Button(xy, text="+X", width=4,
                   command=lambda: self._translate_poly(self._move_step(), 0)
                   ).grid(row=0, column=0, padx=1)
        ttk.Button(xy, text="−X", width=4,
                   command=lambda: self._translate_poly(-self._move_step(), 0)
                   ).grid(row=0, column=1, padx=1)
        ttk.Button(xy, text="+Y", width=4,
                   command=lambda: self._translate_poly(0, self._move_step())
                   ).grid(row=0, column=2, padx=1)
        ttk.Button(xy, text="−Y", width=4,
                   command=lambda: self._translate_poly(0, -self._move_step())
                   ).grid(row=0, column=3, padx=1)

        rot = ttk.Frame(adj); rot.pack(pady=(2, 0))
        ttk.Button(rot, text="↺ rotate", width=9,
                   command=lambda: self._rotate_poly(-self._rot_step())
                   ).pack(side="left", padx=(0, 2))
        ttk.Button(rot, text="rotate ↻", width=9,
                   command=lambda: self._rotate_poly(self._rot_step())
                   ).pack(side="left")

        ttk.Label(adj, text="+X = right, +Y = down (pixel coords). Applies to "
                            "whichever region 'draw' targets above.",
                  foreground="#555", wraplength=210, font=("", 8), justify="left"
                  ).pack(anchor="w", pady=(3, 0))

    # ------------------------------------------------------------------
    def _build_dist_toggles_panel(self, parent):
        """Distribution-fit toggles (Q-Q plot, log-normal vs. Gaussian
        comparison) plus a width/period selector for the live-preview
        panels, anchored bottom-center of the plot area (between the
        toolbar tip on the left and the Translate/Rotate panel on the
        right).

        Export always writes BOTH width and period versions of the Q-Q and
        Gaussian-comparison figures whenever their checkbox is ticked --
        the radio buttons here only pick which one the small in-GUI panels
        (d)/(e) show live, since there isn't room to show both at once."""
        qq = ttk.LabelFrame(parent, text="Distributions", padding=6)
        qq.pack(side="left", expand=True, anchor="center")

        qr = ttk.Frame(qq); qr.pack(anchor="center")
        ttk.Label(qr, text="show panels (d)/(e) for:").pack(side="left", padx=(0, 4))
        self.dist_quantity_var = tk.StringVar(value="width")
        ttk.Radiobutton(qr, text="width", variable=self.dist_quantity_var,
                        value="width", command=self._repreview).pack(side="left")
        ttk.Radiobutton(qr, text="period", variable=self.dist_quantity_var,
                        value="period", command=self._repreview).pack(side="left")

        self.qq_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(qq, text="Show Q-Q plot (vs. log-normal)",
                        variable=self.qq_var,
                        command=self._repreview).pack(anchor="center", pady=(4, 0))
        ttk.Label(qq, text="Up and down samples, EACH vs. their own fitted\n"
                           "log-normal quantiles (fit separately) -- a straight\n"
                           "line supports the log-normal assumption. Export\n"
                           "writes both width and period versions.",
                  foreground="#555", font=("", 8), justify="center"
                  ).pack(anchor="center", pady=(2, 0))

        self.gauss_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(qq, text="Show log-normal vs. Gaussian comparison",
                        variable=self.gauss_var,
                        command=self._repreview).pack(anchor="center", pady=(6, 0))
        ttk.Label(qq, text="Fits both distributions to each state's samples\n"
                           "and reports AIC/BIC/KS -- the lower AIC/BIC model\n"
                           "is better supported. Export writes both width and\n"
                           "period versions.",
                  foreground="#555", font=("", 8), justify="center"
                  ).pack(anchor="center", pady=(2, 0))

    # ------------------------------------------------------------------

    def _build_controls(self, parent):
        # --- INPUT TYPE: raw grayscale XMCD vs. an already-coded signed map ---
        inp = ttk.LabelFrame(parent, text="Input", padding=6); inp.pack(fill="x")
        self._INPUT_CHOICES = [("Grayscale XMCD (raw)", "grayscale"),
                               ("Asymmetry map (pre-coded)", "asymmetry_map")]
        self._MAP_CHOICES = [("blue +1 / red −1 (RB)", "rb"),
                             ("white +1 / black −1 (BW)", "bw")]
        _inp_disp = {v: d for d, v in self._INPUT_CHOICES}
        _map_disp = {v: d for d, v in self._MAP_CHOICES}
        _help_row(
            inp, "Input type:",
            "Grayscale XMCD = raw PEEM intensity; the pipeline flattens the "
            "illumination and derives the asymmetry itself.\n"
            "Asymmetry map = an ALREADY-coloured signed map (blue/red or "
            "black/white up/down). It is decoded straight to m in [-1,1] and "
            "flattening/background subtraction are skipped (the map is already "
            "background-corrected). Pick the colour scheme below.",
        )
        self.input_mode_var = tk.StringVar(value=_inp_disp[DEFAULTS["input_mode"]])
        _ic = ttk.Combobox(inp, textvariable=self.input_mode_var, state="readonly",
                           values=[d for d, _v in self._INPUT_CHOICES])
        _ic.pack(fill="x", padx=5)
        self._map_row = ttk.Frame(inp)
        ttk.Label(self._map_row, text="colour scheme", width=13).pack(side="left")
        self.map_mode_var = tk.StringVar(value=_map_disp[DEFAULTS["map_mode"]])
        _mc = ttk.Combobox(self._map_row, textvariable=self.map_mode_var, state="readonly",
                           values=[d for d, _v in self._MAP_CHOICES])
        _mc.pack(side="left", fill="x", expand=True, padx=(4, 0))
        _mc.bind("<<ComboboxSelected>>", lambda _ev: self._repreview())

        def _map_row_visibility():
            # colour-scheme row only matters for a pre-coded map
            if dict(self._INPUT_CHOICES)[self.input_mode_var.get()] == "asymmetry_map":
                self._map_row.pack(fill="x", padx=5, pady=(4, 0))
            else:
                self._map_row.pack_forget()

        def _on_input_mode(_ev=None):
            _map_row_visibility()
            self._repreview()
        _ic.bind("<<ComboboxSelected>>", _on_input_mode)
        _map_row_visibility()   # initial visibility only (no repreview yet)

        sh = ttk.LabelFrame(parent, text="Selection", padding=6); sh.pack(fill="x")
        r = ttk.Frame(sh); r.pack(fill="x")
        ttk.Label(r, text="draw", width=6).pack(side="left")
        self.draw_target = tk.StringVar(value="Active area")
        ttk.Combobox(r, textvariable=self.draw_target,
                     values=["Active area", "Background"], width=12,
                     state="readonly").pack(side="left")
        self.draw_target.trace_add("write", lambda *a: self._retarget())
        r = ttk.Frame(sh); r.pack(fill="x", pady=(3, 0))
        ttk.Label(r, text="shape", width=6).pack(side="left")
        self.shape = tk.StringVar(value="Polygon")
        ttk.Combobox(r, textvariable=self.shape,
                     values=["Polygon", "Rectangle", "Square", "Circle",
                             "Hexagon", "Regular N-gon"], width=13,
                     state="readonly").pack(side="left")
        self.shape.trace_add("write", lambda *a: self._change_shape())
        self.sides_row = ttk.Frame(sh)
        ttk.Label(self.sides_row, text="sides", width=6).pack(side="left")
        self.n_sides = tk.IntVar(value=6)
        ttk.Spinbox(self.sides_row, from_=3, to=20, width=5,
                    textvariable=self.n_sides,
                    command=self._change_shape).pack(side="left")
        # packed/unpacked dynamically in _change_shape(); only shown for
        # "Regular N-gon"
        self.dim_lbl = ttk.Label(sh, text="box: -", foreground="#036")
        self.dim_lbl.pack(anchor="w", pady=(3, 0))
        self.bg_lbl = ttk.Label(sh, text="background: none", foreground="#960")
        self.bg_lbl.pack(anchor="w")

        f = ttk.LabelFrame(parent, text="Calibration", padding=6); f.pack(fill="x", pady=4)
        self.cal_mode = tk.StringVar(value="Whole image (W x H)")
        ttk.Combobox(f, textvariable=self.cal_mode,
                     values=["Whole image (W x H)", "Drawn shape (W x H)",
                             "Pixels (none)"],
                     width=22, state="readonly").pack(anchor="w")
        self.cal_mode.trace_add("write", lambda *a: (self._cal_ui(), self._repreview()))

        self.cal_frame = ttk.Frame(f); self.cal_frame.pack(fill="x")
        self.cal_hint = ttk.Label(self.cal_frame, text="", wraplength=230,
                                  foreground="#555")
        self.cal_hint.pack(anchor="w")
        self.cal_w = tk.StringVar(value="14.4")
        self.cal_h = tk.StringVar(value="18.3")
        rw = ttk.Frame(self.cal_frame); rw.pack(fill="x", pady=1)
        self.cal_w_lbl = ttk.Label(rw, text="width (um)", width=12)
        self.cal_w_lbl.pack(side="left")
        ew = ttk.Entry(rw, textvariable=self.cal_w, width=8); ew.pack(side="left")
        ew.bind("<Return>", lambda ev: self._repreview())
        rh = ttk.Frame(self.cal_frame); rh.pack(fill="x", pady=1)
        self.cal_h_lbl = ttk.Label(rh, text="height (um)", width=12)
        self.cal_h_lbl.pack(side="left")
        eh = ttk.Entry(rh, textvariable=self.cal_h, width=8); eh.pack(side="left")
        eh.bind("<Return>", lambda ev: self._repreview())
        ttk.Label(self.cal_frame, text="(leave one blank to use a single axis)",
                  foreground="#777").pack(anchor="w")
        self.cal_lbl = ttk.Label(f, text="", foreground="#070", wraplength=230)
        self.cal_lbl.pack(anchor="w", pady=(3, 0))
        self._cal_ui()

        # --- DENOISING (runs BEFORE classification; this is the expensive,
        # cached step -- changing it forces a full recompute, not the fast
        # threshold-only preview path) ---
        d = ttk.LabelFrame(parent, text="Denoising", padding=6); d.pack(fill="x", pady=4)
        # Denoiser method (registry-driven: options come from
        # methods/denoise.py, so adding a denoiser needs no GUI change).
        from xmcd_toolkit import denoise_methods as _dn
        self._DENOISE_CHOICES = [(m.name, m.label) for m in _dn.DENOISERS.values()]
        _help_row(
            d, "Denoiser:",
            "Which edge-preserving denoiser runs before classification. TV is "
            "the default; NLM (non-local means) is the alternative used by "
            "recent FGT/STXM domain papers and often gives a flatter "
            "width-vs-denoising response. The strength slider below drives "
            "both. Changing this re-runs the full denoise.",
            pady=(0, 2),
        )
        _dn_disp = {n: l for n, l in self._DENOISE_CHOICES}
        self.denoise_method_var = tk.StringVar(value=_dn_disp[DEFAULTS["denoise_method"]])
        _dnc = ttk.Combobox(d, textvariable=self.denoise_method_var, state="readonly",
                            values=[l for _n, l in self._DENOISE_CHOICES])
        _dnc.pack(fill="x", padx=5, pady=(0, 4))
        _dnc.bind("<<ComboboxSelected>>", lambda _ev: self._repreview())
        _help_row(
            d, "TV denoise weight (tv_weight):",
            "Total-variation smoothing applied to the raw image before "
            "up/down classification. Higher = suppresses more per-pixel "
            "noise (fewer speckled false domains) but can blur real domain "
            "walls if pushed too far. Low tv_weight on a noisy frame is the "
            "usual cause of salt-and-pepper dots inside otherwise-clean "
            "domains. Typical range: ~0.1 for clean screenshots/simulations, "
            "~0.6-1.0 for noisy experimental frames (e.g. 0 mA).",
        )
        self.tv_weight_var = tk.DoubleVar(value=DEFAULTS['tv_weight'])
        self.tv_weight_slider = ttk.Scale(
            d, from_=0.02, to=1.5, orient="horizontal",
            variable=self.tv_weight_var,
        )
        self.tv_weight_slider.pack(fill="x", padx=5)
        # Recompute only on release: this slider invalidates the cached
        # denoise/flatten step (see gui_core._prep), so firing on every drag
        # tick like the fast sliders below would re-run the slow TV-Chambolle
        # pass continuously. The label still updates live via the trace.
        self.tv_weight_slider.bind("<ButtonRelease-1>", lambda e: self._repreview())
        self.tv_weight_lbl = ttk.Label(d, text="0.100")
        self.tv_weight_lbl.pack(anchor="w")

        def _update_tv_weight_lbl(*args):
            self.tv_weight_lbl.config(text=f"{self.tv_weight_var.get():.3f}")
        self.tv_weight_var.trace_add("write", _update_tv_weight_lbl)

        g = ttk.LabelFrame(parent, text="Classification", padding=6); g.pack(fill="x", pady=4)

        # Threshold Slider
        _help_row(
            g, "m-threshold:",
            "Sets the gray/neutral band on the normalized asymmetry signal: "
            "|m| <= threshold -> neutral (0, gray/wall); m > threshold -> up "
            "(+1, blue); m < -threshold -> down (-1, red). Raising it widens "
            "the neutral band, turning more borderline pixels gray instead "
            "of blue/red -- useful for punching noise out, but too high and "
            "it eats into real domain interiors. Lowering it does the "
            "opposite: less gray, but noisy pixels get classified as real "
            "up/down speckle instead. Default 0.05; noisy frames often look "
            "cleaner around 0.03-0.06 once tv_weight is also turned up.",
        )
        self.m_thresh_var = tk.DoubleVar(value=DEFAULTS['m_threshold'])
        self.m_thresh = ttk.Scale(g, from_=0.0, to=0.8, orient="horizontal",
                                  variable=self.m_thresh_var,
                                  command=lambda _: self._repreview())
        self.m_thresh.pack(fill="x", padx=5)
        self.m_lbl = ttk.Label(g, text="0.050")
        self.m_lbl.pack(anchor="w")
        
        def _update_m_lbl(*args):
            self.m_lbl.config(text=f"{self.m_thresh_var.get():.3f}")
        self.m_thresh_var.trace_add("write", _update_m_lbl)

        # --- MIN DOMAIN SIZE SLIDER (THE NEW BOUNCER!) ---
        _help_row(
            g, "Min Domain Size (px):",
            "Connected fragments smaller than this many pixels are treated "
            "as noise, not real domains. It does three things at once: "
            "drops tiny specks from the area histogram, drops them from the "
            "width calculation (a real domain's width, not noise-fragment "
            "width), and reverts small isolated up/down flecks back to "
            "neutral in the displayed map (despeckling). Raise it if you "
            "still see small dots after tv_weight/m-threshold are "
            "reasonable; too high and it starts eating real small domains.",
            pady=(10, 0),
        )
        self.min_px_var = tk.IntVar(value=DEFAULTS['min_px'])
        self.min_px_slider = ttk.Scale(
            g,
            from_=1,
            to=100,
            orient="horizontal",
            variable=self.min_px_var,
            command=lambda _: self._repreview()
        )
        self.min_px_slider.pack(fill="x", padx=5)
        self.min_px_label = ttk.Label(g, text=f"{DEFAULTS['min_px']} px")
        self.min_px_label.pack(anchor="w")

        def _update_min_px_label(*args):
            self.min_px_label.config(text=f"{self.min_px_var.get()} px")
        self.min_px_var.trace_add("write", _update_min_px_label)

        # Flatten sigma (background scale). Image-dependent: the default suits
        # wide-field device frames; raise it (~150) for zoomed large-domain
        # frames so the Gaussian background stays SMOOTHER than the domains
        # (otherwise it tracks them and hollows their interiors). Entry (not a
        # slider) since it re-runs the expensive denoise/flatten on change.
        _help_row(
            g, "Flatten sigma (px):",
            "Scale of the Gaussian background that is subtracted to remove "
            "illumination. It must be LARGER than your domains, or it will "
            "follow the domains and hollow out their interiors (leaving only "
            "walls -- widths collapse). Default suits standard device frames; "
            "for a zoomed / large-domain field of view raise it to ~150-250. "
            "Press Enter to apply.",
            pady=(10, 0),
        )
        self.flatten_sigma_var = tk.DoubleVar(value=DEFAULTS['flatten_sigma'])
        _fs_entry = ttk.Entry(g, textvariable=self.flatten_sigma_var, width=8)
        _fs_entry.pack(anchor="w", padx=5)
        _fs_entry.bind("<Return>", lambda ev: self._repreview())

        # Connectivities
        _help_row(
            g, "Connectivity rule:",
            "How touching pixels of the same state get merged into one "
            "domain when counting/measuring. 4-way: only edge-touching "
            "pixels count as connected (stricter, more separate domains). "
            "8-way: pixels touching only at a corner also count as one "
            "domain (fewer, larger domains). The opposite rule is always "
            "used for the neutral/background state, per the digital "
            "Jordan-curve theorem, so foreground and background can't both "
            "claim the same boundary.",
            pady=(10, 0),
        )
        self.conn_var = tk.IntVar(value=DEFAULTS['connectivity'])
        c_frame = ttk.Frame(g); c_frame.pack(fill="x")
        ttk.Radiobutton(c_frame, text="4-way", variable=self.conn_var, value=1,
                        command=self._repreview).pack(side="left", padx=(0, 10))
        ttk.Radiobutton(c_frame, text="8-way", variable=self.conn_var, value=2,
                        command=self._repreview).pack(side="left")

        # Width method: which estimator gives the reported w+/w-. All methods
        # are always shown in the readout for comparison; this picks the
        # primary one. Default 2*Area/Perimeter (agrees with skeleton-mean,
        # no neck bias). See the exported width_comparison figure.
        _help_row(
            g, "Width method:",
            "How the reported domain width w+/w- is computed. 2*Area/Perimeter "
            "and skeleton-mean are direct local widths and usually agree; "
            "skeleton-median reads ~30% low (thin-neck bias); correlation "
            "length xi is a robust domain SIZE (~1.5x the width), not the "
            "width itself. The readout always lists all four so you can "
            "compare on your data.",
            pady=(10, 0),
        )
        # display label -> method key
        self._WIDTH_CHOICES = [
            ("2*Area/Perimeter (recommended)", "perimeter_area"),
            ("skeleton mean", "skeleton_mean"),
            ("skeleton median", "skeleton_median"),
            ("correlation length xi", "correlation_length"),
        ]
        _disp_by_key = {k: d for d, k in self._WIDTH_CHOICES}
        self.width_method_var = tk.StringVar(value=_disp_by_key[DEFAULTS["width_method"]])
        wm = ttk.Combobox(g, textvariable=self.width_method_var, state="readonly",
                          values=[d for d, _ in self._WIDTH_CHOICES])
        wm.pack(fill="x", padx=5)
        wm.bind("<<ComboboxSelected>>", lambda _ev: self._repreview())

        # Boundary method: which domain-boundary line is overlaid on the image
        # (registry-driven from methods/boundary.py).
        from xmcd_toolkit import boundary_methods as _bd
        self._BOUNDARY_CHOICES = [(m.name, m.label) for m in _bd.BOUNDARIES.values()]
        _help_row(
            g, "Boundary (overlay):",
            "Which up/down boundary line is drawn on the image. m=0 sub-pixel "
            "contour is a sharp threshold-free line (follows fine detail and "
            "some noise); threshold band shows the |m|<=t neutral zone; "
            "Chan-Vese is a smooth, noise-robust region interface (slower). "
            "This only draws the overlay -- it does not change f+/f- or width.",
            pady=(10, 0),
        )
        _bd_disp = {n: l for n, l in self._BOUNDARY_CHOICES}
        self.boundary_method_var = tk.StringVar(value=_bd_disp[DEFAULTS["boundary_method"]])
        self.show_boundary_var = tk.BooleanVar(value=False)
        _bframe = ttk.Frame(g); _bframe.pack(fill="x", padx=5)
        ttk.Checkbutton(_bframe, text="show", variable=self.show_boundary_var,
                        command=self._repreview).pack(side="left")
        _bc = ttk.Combobox(_bframe, textvariable=self.boundary_method_var, state="readonly",
                           values=[l for _n, l in self._BOUNDARY_CHOICES])
        _bc.pack(side="left", fill="x", expand=True, padx=(6, 0))
        _bc.bind("<<ComboboxSelected>>", lambda _ev: self._repreview())
        # Q-Q plot / Gaussian-comparison toggles live in
        # _build_dist_toggles_panel(), bottom-center of the plot area --
        # not here, so they aren't duplicated.

    def _params(self):
        """Build the parameter dictionary sent to the backend."""
        um_per_px = self._resolve_um()[0]
        return {
            "polygon": self.poly, "bg_polygon": self.bg_poly,
            "fov": None, "um_per_px": um_per_px,
            "m_threshold": float(self.m_thresh_var.get()),
            "tv_weight": float(self.tv_weight_var.get()),
            "min_px": int(self.min_px_var.get()),
            "connectivity": int(self.conn_var.get()),
            "width_method": dict((d, k) for d, k in self._WIDTH_CHOICES)[
                self.width_method_var.get()],
            "denoise_method": dict((l, n) for n, l in self._DENOISE_CHOICES)[
                self.denoise_method_var.get()],
            "boundary_method": dict((l, n) for n, l in self._BOUNDARY_CHOICES)[
                self.boundary_method_var.get()],
            "input_mode": dict(self._INPUT_CHOICES)[self.input_mode_var.get()],
            "map_mode": dict(self._MAP_CHOICES)[self.map_mode_var.get()],
            "show_boundary": bool(self.show_boundary_var.get()),
            "fig_format": dict(self._FIGFMT_CHOICES)[self.fig_format_var.get()],
            "qq_plot": bool(self.qq_var.get()),
            "gauss_compare": bool(self.gauss_var.get()),
            "dist_quantity": self.dist_quantity_var.get(),
            "states": DEFAULTS["states"], "z_neutral": DEFAULTS["z_neutral"],
            "clahe": DEFAULTS["clahe"],
            "flatten": True, "flatten_sigma": float(self.flatten_sigma_var.get()),
            "rigorous": DEFAULTS["rigorous"], "tiff": DEFAULTS["tiff"],
            "detailed": bool(self.detailed_var.get()),
            "n_boot": DEFAULTS["n_boot"],
            "dpi": DEFAULTS["dpi"], "outdir": None,
            "shape": (self.img.shape if self.img is not None else None),
        }

    # ---- calibration / box helpers -----------------------------------
    def _cal_ui(self):
        """Adapt the calibration entries to the current mode."""
        mode = self.cal_mode.get()
        if mode == "Pixels (none)":
            self.cal_frame.pack_forget()
            return
        self.cal_frame.pack(fill="x")
        if mode == "Whole image (W x H)":
            self.cal_hint.config(text="Enter the physical size of the whole image:")
            self.cal_w_lbl.config(text="image W (um)")
            self.cal_h_lbl.config(text="image H (um)")
        else:  # Drawn shape (W x H)
            self.cal_hint.config(text="Enter the known size of the drawn shape "
                                      "(its bounding box):")
            self.cal_w_lbl.config(text="shape W (um)")
            self.cal_h_lbl.config(text="shape H (um)")

    def _ref_box_px(self):
        """Pixel (W, H) of the calibration reference for the current mode."""
        mode = self.cal_mode.get()
        if mode == "Whole image (W x H)":
            if self.img is None:
                return None
            H, W = self.img.shape
            return (float(W), float(H))
        if mode == "Drawn shape (W x H)":
            return self._box_px()
        return None

    def _box_px(self):
        """(W_px, H_px) bounding box of the current selection, or None."""
        if not self.poly:
            return None
        xs = [p[0] for p in self.poly]; ys = [p[1] for p in self.poly]
        return (max(xs) - min(xs), max(ys) - min(ys))

    def _resolve_um(self):
        """Return (um_per_px, warn) for the current calibration mode."""
        if self.cal_mode.get() == "Pixels (none)":
            return None, None
        ref = self._ref_box_px()
        if ref is None:
            return None, None
        def fnum(v):
            try:
                return float(v)
            except ValueError:
                return None
        return core.um_per_px_from_dims(ref, (fnum(self.cal_w.get()),
                                              fnum(self.cal_h.get())))

    def _update_dims(self, um):
        """Live readout of the drawn box in px and (if calibrated) um, plus the
        derived scale and any non-square warning."""
        box = self._box_px()
        if box is None:
            self.dim_lbl.config(text="box: -")
        else:
            w, h = box
            if um:
                self.dim_lbl.config(text=f"box: {w:.0f} x {h:.0f} px  =  "
                                         f"{w*um:.2f} x {h*um:.2f} um")
            else:
                self.dim_lbl.config(text=f"box: {w:.0f} x {h:.0f} px")
        um2, warn = self._resolve_um()
        if um2 and warn:
            self.cal_lbl.config(text=f"scale = {um2:.4f} um/px  ({warn})",
                                foreground="#a60")
        elif um2:
            src = "image" if self.cal_mode.get().startswith("Whole") else "shape"
            self.cal_lbl.config(text=f"scale = {um2:.4f} um/px (from {src})",
                                foreground="#070")
        else:
            self.cal_lbl.config(text="")

    # ------------------------------------------------------------------
    def _browse(self):
        p = filedialog.askopenfilename(
            filetypes=[("Images", "*.tif *.tiff *.png *.jpg *.jpeg"), ("All", "*.*")])
        if p:
            self._load(p)

    def _on_drop(self, event):
        self._load(event.data.strip().strip("{}"))

    def _load(self, p):
        if not os.path.isfile(p):
            messagebox.showerror("File", f"Not a file:\n{p}"); return
        try:
            self.img = np.asarray(Image.open(p).convert("F"))
        except Exception as e:
            messagebox.showerror("Open", str(e)); return
        core.clear_cache(p)                     # fresh fields for this image
        self.path = p
        self.poly = None; self.bg_poly = None; self.res = None
        self.file_lbl.config(text=os.path.basename(p))
        self.drop.config(text=os.path.basename(p), bg="#dfd")
        self.dim_lbl.config(text="box: -")
        self.bg_lbl.config(text="background: none")
        self._show_image()
        self._enable_selector()
        self.status.config(text=f"draw a {self.shape.get().lower()} over the active area",
                           foreground="#a60")

    def _show_image(self):
        self.ax_img.clear(); self.ax_img.axis("off")
        self.ax_img.imshow(self.img, cmap="gray")
        self.ax_img.set_title("draw a polygon over the domains (click vertices, close it)")
        for a in self.ax_d:
            a.clear(); a.axis("off")
        self.canvas.draw()

    def _enable_selector(self):
        if self.selector is not None:
            try:
                self.selector.disconnect_events()
            except Exception:
                pass
            self.selector = None
        if self.img is None:
            return
        shape = self.shape.get()
        if shape == "Polygon":
            try:
                self.selector = PolygonSelector(self.ax_img, self._on_poly, useblit=False)
            except TypeError:
                self.selector = PolygonSelector(self.ax_img, self._on_poly)
        elif shape in ("Rectangle", "Square"):
            try:
                self.selector = RectangleSelector(
                    self.ax_img, self._on_rect, useblit=False, interactive=True,
                    button=[1])
            except TypeError:
                self.selector = RectangleSelector(self.ax_img, self._on_rect)
        else:  # Circle, Hexagon, Regular N-gon: drag a bounding box, then
               # inscribe a regular polygon (or fine-stepped circle) in it.
            try:
                self.selector = EllipseSelector(
                    self.ax_img, self._on_regular, useblit=False, interactive=True,
                    button=[1])
            except TypeError:
                self.selector = EllipseSelector(self.ax_img, self._on_regular)

    def _change_shape(self):
        """Change the draw shape; keep existing regions, just re-arm the selector."""
        if self.shape.get() == "Regular N-gon":
            self.sides_row.pack(anchor="w", pady=(3, 0))
        else:
            self.sides_row.pack_forget()
        if self.img is not None:
            self._enable_selector()
            self.status.config(
                text=f"draw the {self.draw_target.get().lower()} "
                     f"({self.shape.get().lower()})", foreground="#a60")

    def _on_rect(self, eclick, erelease):
        x0, y0 = eclick.xdata, eclick.ydata
        x1, y1 = erelease.xdata, erelease.ydata
        if None in (x0, y0, x1, y1):
            return
        xlo, xhi = sorted((x0, x1)); ylo, yhi = sorted((y0, y1))
        if self.shape.get() == "Square":          # force equal sides from the corner
            side = min(xhi - xlo, yhi - ylo)
            xhi, yhi = xlo + side, ylo + side
        if (xhi - xlo) < 2 or (yhi - ylo) < 2:
            return
        self._store_verts([(xlo, ylo), (xhi, ylo), (xhi, yhi), (xlo, yhi)])
        self.root.after(0, self._commit_selector)
        self._schedule_preview()

    def _on_regular(self, eclick, erelease):
        """Circle / Hexagon / Regular N-gon: drag a bounding box, inscribe a
        regular polygon of the requested number of sides (a circle is just a
        64-gon, which renders smoothly at typical image scales)."""
        x0, y0 = eclick.xdata, eclick.ydata
        x1, y1 = erelease.xdata, erelease.ydata
        if None in (x0, y0, x1, y1):
            return
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        r = min(abs(x1 - x0), abs(y1 - y0)) / 2
        if r < 2:
            return
        shape = self.shape.get()
        n = {"Circle": 64, "Hexagon": 6}.get(shape, max(3, self.n_sides.get()))
        # start at the top (-90 deg) so e.g. a hexagon/pentagon sits flat-ish
        # and looks deliberate rather than randomly rotated
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False) - np.pi / 2
        verts = [(cx + r * np.cos(a), cy + r * np.sin(a)) for a in angles]
        self._store_verts(verts)
        self.root.after(0, self._commit_selector)
        self._schedule_preview()

    def _on_poly(self, verts):
        if verts and len(verts) >= 3:
            self._store_verts([(float(x), float(y)) for x, y in verts])
            self.root.after(0, self._commit_selector)   # lock it (defer past callback)
            self._schedule_preview()

    def _store_verts(self, verts):
        """Send completed vertices to the current draw target."""
        if self.draw_target.get() == "Background":
            self.bg_poly = verts
            self.bg_lbl.config(text=f"background: {len(verts)}-vertex region")
        else:
            self.poly = verts

    def _retarget(self):
        """Switch between drawing the active area and the background region."""
        if self.img is not None:
            self._enable_selector()
            self.status.config(
                text=f"draw the {self.draw_target.get().lower()} "
                     f"({self.shape.get().lower()})", foreground="#a60")

    def _commit_selector(self):
        """Finish the active polygon: drop the live selector so the committed
        outline stays stable. 'Clear polygon' re-enables drawing."""
        if self.selector is not None:
            try:
                self.selector.set_visible(False)
            except Exception:
                pass
            try:
                self.selector.disconnect_events()
            except Exception:
                pass
            self.selector = None

    def _schedule_preview(self):
        """Debounce: coalesce rapid vertex edits into one preview."""
        if self._preview_after is not None:
            try:
                self.root.after_cancel(self._preview_after)
            except Exception:
                pass
        self._preview_after = self.root.after(400, self._repreview)

    def _clear_poly(self):
        self.poly = None; self.res = None
        if self.img is not None:
            self._show_image(); self._enable_selector()
            self.dim_lbl.config(text="box: -"); self.cal_lbl.config(text="")
            self.status.config(text=f"draw a {self.shape.get().lower()} over the active area",
                               foreground="#a60")

    def _clear_bg(self):
        self.bg_poly = None
        self.bg_lbl.config(text="background: none")
        if self.img is not None and self.poly:
            self._repreview()                 # recompute without a background zero
        elif self.img is not None:
            self._show_image(); self._enable_selector()

    def _whole_image(self):
        if self.img is None:
            messagebox.showwarning("Image", "Load an image first."); return
        H, W = self.img.shape
        self.poly = [(0.0, 0.0), (W - 1.0, 0.0), (W - 1.0, H - 1.0), (0.0, H - 1.0)]
        self._commit_selector()
        self._repreview()

    def _save_poly(self):
        if not self.poly:
            messagebox.showwarning("Polygon", "Draw a polygon first."); return
        p = filedialog.asksaveasfilename(
            defaultextension=".json", filetypes=[("Polygon JSON", "*.json")])
        if not p:
            return
        data = {"vertices": self.poly,
                "cal_mode": self.cal_mode.get(),
                "cal_w": self.cal_w.get(), "cal_h": self.cal_h.get(),
                "m_threshold": float(self.m_thresh_var.get()),
                "tv_weight": float(self.tv_weight_var.get()),
                "shape": self.shape.get(),
                "image": os.path.basename(self.path) if self.path else None}
        try:
            with open(p, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            messagebox.showerror("Save polygon", str(e)); return
        self.status.config(text=f"polygon saved: {os.path.basename(p)}", foreground="#070")

    def _load_poly(self):
        p = filedialog.askopenfilename(
            filetypes=[("Polygon JSON", "*.json"), ("All", "*.*")])
        if not p:
            return
        try:
            with open(p) as f:
                data = json.load(f)
            self.poly = [(float(x), float(y)) for x, y in data["vertices"]]
            if data.get("cal_mode"):
                self.cal_mode.set(data["cal_mode"])
            if data.get("cal_w") is not None:
                self.cal_w.set(str(data["cal_w"]))
            if data.get("cal_h") is not None:
                self.cal_h.set(str(data["cal_h"]))
            elif data.get("fov"):                      # backward compat
                self.cal_w.set(str(data["fov"][0])); self.cal_h.set(str(data["fov"][1]))
            if data.get("shape"):
                self.shape.set(data["shape"])
            if data.get("m_threshold") is not None:
                self.m_thresh_var.set(float(data["m_threshold"]))
            if data.get("tv_weight") is not None:
                self.tv_weight_var.set(float(data["tv_weight"]))
        except Exception as e:
            messagebox.showerror("Load polygon", str(e)); return
        if self.img is not None:
            self._commit_selector()
            self._repreview()
        else:
            self.status.config(text="polygon loaded -- now load an image", foreground="#a60")

    def _move_step(self):
        try:
            return float(self.move_step.get())
        except (ValueError, tk.TclError):
            return 5.0

    def _rot_step(self):
        try:
            return float(self.rot_step.get())
        except (ValueError, tk.TclError):
            return 5.0

    def _transform_target(self):
        """Which region ('poly' = active area, or 'bg_poly' = background)
        the translate/rotate controls act on, following the 'draw' target
        combobox so the same control moves whichever shape you're editing."""
        return "bg_poly" if self.draw_target.get() == "Background" else "poly"

    def _translate_poly(self, dx, dy):
        attr = self._transform_target()
        verts = getattr(self, attr)
        if not verts:
            messagebox.showwarning("Move", "Nothing to move -- draw or load a shape first.")
            return
        setattr(self, attr, [(x + dx, y + dy) for x, y in verts])
        self._after_transform()

    def _rotate_poly(self, deg):
        attr = self._transform_target()
        verts = getattr(self, attr)
        if not verts:
            messagebox.showwarning("Rotate", "Nothing to rotate -- draw or load a shape first.")
            return
        cx = sum(p[0] for p in verts) / len(verts)
        cy = sum(p[1] for p in verts) / len(verts)
        a = np.deg2rad(deg)
        ca, sa = np.cos(a), np.sin(a)
        setattr(self, attr, [
            (cx + (x - cx) * ca - (y - cy) * sa,
             cy + (x - cx) * sa + (y - cy) * ca) for x, y in verts])
        self._after_transform()

    def _after_transform(self):
        self._commit_selector()      # drop any live selector so it can't fight the redraw
        if self.bg_poly:
            self.bg_lbl.config(text=f"background: {len(self.bg_poly)}-vertex region")
        if self.img is None:
            return
        if self.poly:
            self._repreview()
        else:
            self._show_image()

    # ------------------------------------------------------------------
    def _repreview(self):
        self._preview_after = None
        if self.img is None or not self.poly:
            return
        if self._busy:                       # a job is running; try again shortly
            self._schedule_preview(); return
        self._busy = True
        self.status.config(text="computing preview...", foreground="#a60")
        self.root.update_idletasks()
        params = self._params(); path = self.path
        threading.Thread(target=self._work_preview, args=(path, params), daemon=True).start()
        self._poll()

    def _work_preview(self, path, params):
        try:
            self.q.put(("preview", core.preview_polygon(path, params)))
        except Exception as e:
            self.q.put(("error", str(e)))

    def _poll(self):
        try:
            kind, payload = self.q.get_nowait()
        except queue.Empty:
            self.root.after(100, self._poll); return
        self._busy = False
        if kind == "error":
            self.status.config(text="error", foreground="#a00")
            messagebox.showerror("Error", payload)
        elif kind == "preview":
            self.res = payload
            self._draw_preview(payload)
            self.status.config(text="preview ready", foreground="#070")
        elif kind == "export":
            self.status.config(text="export complete", foreground="#070")
            if messagebox.askyesno("Done", f"Saved to:\n{payload}\n\nOpen folder?"):
                self._open(payload)

    def _redraw_only(self):
        """Redraw the last result without recomputing -- for toggling the
        boundary overlay on/off (no need to re-run the analysis)."""
        if self.res is not None:
            self._draw_preview(self.res)

    def _draw_preview(self, res):
        self.ax_img.clear(); self.ax_img.axis("off")
        self.ax_img.imshow(self.img, cmap="gray")
        self.ax_img.imshow(np.where(res["roi"], res["mz"], np.nan),
                           cmap=CMAP3, norm=NORM3, alpha=0.8)
        if self.poly:
            xs = [p[0] for p in self.poly] + [self.poly[0][0]]
            ys = [p[1] for p in self.poly] + [self.poly[0][1]]
            self.ax_img.plot(xs, ys, "-", color="#0a0", lw=1.5)
        if self.bg_poly:
            xs = [p[0] for p in self.bg_poly] + [self.bg_poly[0][0]]
            ys = [p[1] for p in self.bg_poly] + [self.bg_poly[0][1]]
            self.ax_img.plot(xs, ys, "--", color="#e69500", lw=1.5)
        # optional domain-boundary overlay (the "show" checkbox); contours as
        # thin cyan lines, edge masks as cyan points.
        if self.show_boundary_var.get() and res.get("boundary"):
            b = res["boundary"]
            if b.get("contours"):
                for c in b["contours"]:
                    self.ax_img.plot(c[:, 1], c[:, 0], color="#00e5ff", lw=0.6)
            elif b.get("edge") is not None:
                ys2, xs2 = np.where(b["edge"])
                self.ax_img.plot(xs2, ys2, ".", color="#00e5ff", ms=0.5)
        ttl = "three-state $m_z$ inside active area"
        if self.bg_poly:
            ttl += " (bg = amber dashed -> zero)"
        if self.show_boundary_var.get():
            ttl += "  (boundary = cyan)"
        self.ax_img.set_title(ttl)
        self.ax_img.legend(handles=[
            Patch(facecolor=COLOR_UP, label="up (+1)"),
            Patch(facecolor="#dddddd", label="neutral (0)"),
            Patch(facecolor=COLOR_DOWN, label="down (-1)"),
        ], loc="upper right", fontsize=7, framealpha=0.85)
        core.draw_distributions(self.ax_d, res)
        self.fig.tight_layout()
        self.canvas.draw()
        # readout goes to the dedicated Results sidebar, never onto the image
        self.metrics.config(text=core.summary_text(res))
        self._update_dims(res["um"])

    # ------------------------------------------------------------------
    def _export(self):
        if self.img is None or not self.poly:
            messagebox.showwarning("Polygon", "Load an image and draw a polygon first.")
            return
        if self._busy:
            messagebox.showinfo("Busy", "A computation is already running; please wait.")
            return
        self._busy = True
        self.status.config(text="exporting (a few seconds)...", foreground="#a60")
        self.root.update_idletasks()
        params = self._params(); path = self.path
        threading.Thread(target=self._work_export, args=(path, params), daemon=True).start()
        self._poll()

    def _work_export(self, path, params):
        try:
            self.q.put(("export", core.run_export(path, params)))
        except Exception as e:
            self.q.put(("error", str(e)))

    def _open(self, folder):
        try:
            if sys.platform == "darwin":
                subprocess.run(["open", folder])
            elif os.name == "nt":
                os.startfile(folder)  # noqa
            else:
                subprocess.run(["xdg-open", folder])
        except Exception:
            pass


def main():
    global _HAS_DND
    root = None
    if _HAS_DND:
        try:
            root = TkinterDnD.Tk()
        except Exception as e:
            print(f"note: drag-and-drop unavailable ({e}); use Browse.")
            _HAS_DND = False
    if root is None:
        root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
