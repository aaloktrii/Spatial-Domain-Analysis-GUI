"""Figures for core/magnetization.py (plotting layer, no analysis)."""
import numpy as np
from scipy import optimize, stats
from ..colors import COLOR_UP, COLOR_DOWN
from ..core.magnetization import (
    _three_gauss,
    asymmetry_field,
    correlation_length,
    local_xi_map,
)


def make_xi_map_figure(layers, out_png, um_per_px=None, title=""):
    """
    Two-panel figure of the local correlation length xi(x,y): (a) the ximap
    over the flake image, (b) the histogram of local xi across windows.
    Uses the same continuous asymmetry field the scalar xi comes from.

    Returns (out_png, stats) where stats has n_windows, window_px, and the
    median/mean/SD/min/max of local xi (in nm if calibrated, else px). NaN
    stats if the flake is too small for a single valid window.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m, roi = asymmetry_field(layers)
    img = layers["img"]
    nm = (um_per_px * 1000.0) if um_per_px else 1.0
    U = "nm" if um_per_px else "px"
    xi_g, _, _, _ = correlation_length(m, roi)
    xim, yc, xc, win = local_xi_map(m, roi)
    xim_u = xim * nm
    vals = xim_u[np.isfinite(xim_u)]

    def _s(a): return float(a) if a == a else float("nan")
    if vals.size:
        stats_d = {"n_windows": int(vals.size), "window_px": int(win),
                   "median": _s(np.median(vals)), "mean": _s(vals.mean()),
                   "std": _s(vals.std()), "min": _s(vals.min()), "max": _s(vals.max()),
                   "global_xi": _s(xi_g * nm), "unit": U}
    else:
        stats_d = {"n_windows": 0, "window_px": int(win), "median": float("nan"),
                   "mean": float("nan"), "std": float("nan"), "min": float("nan"),
                   "max": float("nan"), "global_xi": _s(xi_g * nm), "unit": U}

    H, W = m.shape
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].imshow(img, cmap="gray")
    im = ax[0].imshow(xim_u, cmap="viridis", alpha=0.8,
                      extent=[0, W, H, 0], interpolation="bilinear")
    ax[0].set_title(f"(a) local correlation length xi(x,y)\n(window={win}px ~ 7x global xi)")
    ax[0].axis("off")
    fig.colorbar(im, ax=ax[0], fraction=0.046, pad=0.04, label=f"local xi ({U})")
    if vals.size:
        ax[1].hist(vals, bins=24, color="#3a7d44", edgecolor="white", alpha=0.9)
        ax[1].axvline(stats_d["median"], color="k", ls="--", lw=1.4,
                      label=f"median={stats_d['median']:.0f}")
        ax[1].axvline(stats_d["global_xi"], color="#d12b2b", ls=":", lw=1.6,
                      label=f"global xi={stats_d['global_xi']:.0f}")
        ax[1].legend(fontsize=9)
    ax[1].set_xlabel(f"local correlation length xi ({U})")
    ax[1].set_ylabel("number of windows")
    ax[1].set_title(f"(b) distribution of local xi  (n={stats_d['n_windows']} windows)")
    ax[1].grid(alpha=0.3)
    if title:
        fig.suptitle(title, fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95] if title else None)
    plt.savefig(out_png, dpi=130); plt.close(fig)
    return out_png, stats_d

# ----------------------------------------------------------------------
def make_magnetization_figure(res, out_png, title=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m, roi, C_radial, rr, boot = res["_field"]
    fig, ax = plt.subplots(2, 2, figsize=(12, 9))

    # (a) asymmetry histogram + mixture
    fit = res["mixture"]
    ax[0, 0].bar(fit["centers"], fit["hist"], width=fit["centers"][1] - fit["centers"][0],
                 color="#bbb", edgecolor="none", label="data")
    xs = np.linspace(-1, 1, 400)
    ax[0, 0].plot(xs, _three_gauss(xs, *fit["popt"]), "r-", lw=2, label="3-Gaussian")
    ax[0, 0].set_xlabel("normalized asymmetry m"); ax[0, 0].set_ylabel("density")
    ax[0, 0].set_title("(a) magnetization distribution + mixture fit")
    ax[0, 0].legend(fontsize=8)

    # (b) autocorrelation
    ax[0, 1].plot(rr, C_radial, "k-")
    ax[0, 1].axhline(np.exp(-1), color="r", ls="--", lw=1, label="1/e")
    ax[0, 1].axvline(res["corr_length_px"], color="b", ls=":", lw=1,
                     label=f"xi={res['corr_length_px']:.1f}px")
    ax[0, 1].set_xlim(0, max(8 * res["corr_length_px"], 20))
    ax[0, 1].set_xlabel("lag r (px)"); ax[0, 1].set_ylabel("C(r)")
    ax[0, 1].set_title("(b) spatial autocorrelation"); ax[0, 1].legend(fontsize=8)

    # (c) bootstrap distribution of <m_z>
    ax[1, 0].hist(boot, bins=40, color="#6a9", edgecolor="white")
    for q, c in [(res["ci95_bootstrap"][0], "r"), (res["ci95_bootstrap"][1], "r"),
                 (res["mean_mz"], "k")]:
        ax[1, 0].axvline(q, color=c, lw=1.5)
    ax[1, 0].set_xlabel("<m_z> (block-bootstrap)"); ax[1, 0].set_ylabel("count")
    ax[1, 0].set_title("(c) bootstrap sampling distribution")

    # (d) threshold robustness
    sw = res["threshold_sweep"]
    ax[1, 1].plot(sw["t"], sw["net_m"], "o-", color="k", label="net <m_z>")
    ax[1, 1].plot(sw["t"], sw["frac_up"], "s--", color=COLOR_UP, label="frac up")
    ax[1, 1].plot(sw["t"], sw["frac_down"], "v--", color=COLOR_DOWN, label="frac down")
    ax[1, 1].set_xlabel("threshold |m|"); ax[1, 1].set_ylabel("value")
    ax[1, 1].set_title("(d) threshold robustness"); ax[1, 1].legend(fontsize=8)

    if title:
        fig.suptitle(title, fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_png, dpi=130); plt.close(fig)
    return out_png
