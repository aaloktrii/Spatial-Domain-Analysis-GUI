"""
width.py -- swappable per-state domain-WIDTH estimators.

Each returns w_up / w_down for one method. On a clean stripe pattern the
direct-width methods (perimeter_area, skeleton_mean) and the correlation
length agree; the period/FFT family instead measures the coarse pattern
WAVELENGTH (not the domain thickness), so it is deliberately NOT offered here.

    domain_widths(mz, roi, um_per_px, min_px, mm) -> {method: {up, down, unit}}
    make_width_comparison_figure(...) -> (png, domain_widths dict)

Registry: WIDTHS (name -> Method). Add a method = write w_up/w_down for one
state and register it; the GUI dropdown updates automatically.
"""
import numpy as np
from skimage import measure as _measure

from . import make_registry
from ..morphometry import per_state_morphometry
from ..magnetization import correlation_length_state

WIDTHS, _register = make_registry()


def _perim_area_width(mask):
    """2*Area/Perimeter: hydraulic width of a stripe/labyrinth state. A stripe
    w x L has A=wL, P~2L, so 2A/P = w. Robust, threshold-only, no skeleton."""
    A = float(mask.sum()); P = float(_measure.perimeter(mask))
    return (2.0 * A / P) if P > 0 else float("nan")


# Each registered fn takes a small context dict and returns (w_up, w_down) in
# PX (the dispatcher scales to nm). ctx has: mz, roi, mm, um_per_px.
@_register("perimeter_area", "2*Area/Perimeter (hydraulic mean width)",
           "Direct mean stripe width from area and perimeter. Robust, "
           "threshold-only, no skeleton. Recommended primary.")
def _perimeter_area(ctx):
    return (_perim_area_width(ctx["mz"] == 1), _perim_area_width(ctx["mz"] == -1))


@_register("skeleton_mean", "skeleton mean (2*DT)",
           "Mean local thickness (2*distance-transform on the medial axis). "
           "A direct width; usually agrees with 2*Area/Perimeter.")
def _skeleton_mean(ctx):
    mm = ctx["mm"]
    up = mm["up"]["widths_px"]; dn = mm["down"]["widths_px"]
    return (float(up.mean()) if up.size else np.nan,
            float(dn.mean()) if dn.size else np.nan)


@_register("skeleton_median", "skeleton median (2*DT; neck-biased low)",
           "Median local thickness. Biased ~30% low by thin necks in a "
           "labyrinth -- reads smaller than the true width.")
def _skeleton_median(ctx):
    mm = ctx["mm"]
    up = mm["up"]["widths_px"]; dn = mm["down"]["widths_px"]
    return (float(np.median(up)) if up.size else np.nan,
            float(np.median(dn)) if dn.size else np.nan)


@_register("correlation_length", "correlation length xi (a SIZE, ~1.5x width)",
           "1/e autocorrelation length -- a domain SIZE, the most "
           "denoising-robust scale, but ~1.5x the width. Report as a size.")
def _corr_len(ctx):
    # correlation_length is an FFT autocorrelation -- the most expensive width
    # method. The GUI live readout already computes xi_up/xi_down (in px) for
    # its own per-state size line, so it passes them in via ctx["xi_px"] to
    # avoid recomputing the identical numbers here (halves the FFT cost per
    # preview). Falls back to computing them when not supplied.
    if ctx.get("xi_px") is not None:
        return tuple(ctx["xi_px"])
    xu = correlation_length_state(ctx["mz"], ctx["roi"], 1, um_per_px=None)
    xd = correlation_length_state(ctx["mz"], ctx["roi"], -1, um_per_px=None)
    return (xu, xd)


# public ordering for tables/figures/GUI
WIDTH_METHODS = ("perimeter_area", "skeleton_mean", "skeleton_median",
                 "correlation_length")
WIDTH_METHOD_LABELS = {k: WIDTHS[k].label for k in WIDTH_METHODS}


def domain_widths(mz, roi, um_per_px=None, min_px=12, mm=None, xi_px=None):
    """Per-state width by every registered method. Widths in nm if calibrated,
    else px. Pass an existing per_state_morphometry() dict as `mm` to reuse
    the skeleton, and a precomputed (xi_up_px, xi_down_px) as `xi_px` to reuse
    the correlation-length FFT instead of recomputing it. Returns
    {method: {"up", "down", "unit"}}."""
    nm = (um_per_px * 1000.0) if um_per_px else 1.0
    unit = "nm" if um_per_px else "px"
    if mm is None:
        mm = per_state_morphometry(mz, um_per_px=um_per_px, min_px=min_px)
    ctx = {"mz": mz, "roi": roi, "mm": mm, "um_per_px": um_per_px, "xi_px": xi_px}
    out = {}
    for k in WIDTH_METHODS:
        wu, wd = WIDTHS[k].fn(ctx)
        out[k] = {"up": wu * nm, "down": wd * nm, "unit": unit}
    return out


def compute(mz, roi, method="perimeter_area", um_per_px=None, min_px=12, mm=None):
    """w_up/w_down for a SINGLE named method (GUI primary readout)."""
    return domain_widths(mz, roi, um_per_px=um_per_px, min_px=min_px, mm=mm)[method]


def make_width_comparison_figure(mz, roi, um_per_px, out_png, min_px=12, mm=None, title=""):
    """Horizontal bar chart of w_up/w_down across all width methods."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ..morphometry import _PUB_RC
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
        plt.tight_layout(); plt.savefig(out_png, dpi=130); plt.close(fig)
    return out_png, dw
