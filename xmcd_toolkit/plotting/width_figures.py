"""Figures for core/methods/width.py (plotting layer, no analysis)."""
from .labels import relabel_figure
import numpy as np
from skimage import measure as _measure
from ..core.morphometry import per_state_morphometry
from ..core.magnetization import correlation_length_state
from ..core.methods.width import (
    WIDTH_METHODS,
    WIDTH_METHOD_LABELS,
    domain_widths,
)


def make_width_comparison_figure(mz, roi, um_per_px, out_png, min_px=12, mm=None, title=""):
    """Horizontal bar chart of w_up/w_down across all width methods."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ..core.morphometry import _PUB_RC
    from ..colors import COLOR_UP_LINE, COLOR_DOWN_LINE

    dw = domain_widths(mz, roi, um_per_px=um_per_px, min_px=min_px, mm=mm)
    unit = "nm" if um_per_px else "px"
    methods = list(WIDTH_METHODS)
    labels = [WIDTH_METHOD_LABELS[m] for m in methods]
    wu = [dw[m]["up"] for m in methods]
    wd = [dw[m]["down"] for m in methods]
    with plt.rc_context(_PUB_RC):
        fig, ax = plt.subplots(figsize=(8.5, 4.2))
        y = np.arange(len(methods)); h = 0.38
        ax.barh(y - h / 2, wu, h, color=COLOR_UP_LINE, label="w_up")
        ax.barh(y + h / 2, wd, h, color=COLOR_DOWN_LINE, label="w_down")
        for i, (a, b) in enumerate(zip(wu, wd)):
            if a == a: ax.text(a, y[i] - h / 2, f" {a:.0f}", va="center", fontsize=7)
            if b == b: ax.text(b, y[i] + h / 2, f" {b:.0f}", va="center", fontsize=7)
        ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis(); ax.set_xlabel(f"domain width ({unit})")
        ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="x")
        ax.set_title((title + " -- " if title else "") + "domain width by method")
        relabel_figure(plt.gcf())
        plt.tight_layout(); plt.savefig(out_png, dpi=130); plt.close(fig)
    return out_png, dw
