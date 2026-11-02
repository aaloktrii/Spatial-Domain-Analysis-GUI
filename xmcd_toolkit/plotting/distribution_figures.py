"""Figures for core/distributions.py (plotting layer, no analysis)."""
from .labels import relabel_figure
import numpy as np
from scipy import stats
from ..core.distributions import (
    fit_lognormal,
)


# ----------------------------------------------------------------------
def make_distribution_figure(sizes, out_png, unit="px", title=""):
    """Histogram of domain sizes with the fitted log-normal overlaid."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sizes = np.asarray(sizes, float)
    sizes = sizes[sizes > 0]
    fit = fit_lognormal(sizes)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    # linear histogram + log-normal pdf
    ax[0].hist(sizes, bins=24, density=True, color="#777", edgecolor="white",
               alpha=0.85, label="data")
    if fit:
        ax[0].plot(fit["pdf_x"], fit["pdf_y"], "r-", lw=2,
                   label=f"log-normal\n  median={fit['median']:.1f} {unit}\n"
                         f"  $\\sigma_{{\\ln}}$={fit['sigma_log']:.2f}\n"
                         f"  KS={fit['ks_stat']:.2f}")
    ax[0].set_xlabel(f"domain equiv. diameter ({unit})")
    ax[0].set_ylabel("probability density")
    ax[0].set_title("(a) size distribution + log-normal fit")
    ax[0].legend(fontsize=8)

    # log-log survival to eyeball power-law tail (avalanche-style view)
    s = np.sort(sizes)
    surv = 1.0 - np.arange(1, s.size + 1) / s.size
    ax[1].loglog(s, surv + 1e-9, ".", ms=4, color="#333")
    ax[1].set_xlabel(f"domain area / size ({unit})")
    ax[1].set_ylabel("P(>S)")
    ax[1].set_title("(b) log-log survival (power-law check)")

    if title:
        fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    relabel_figure(plt.gcf())
    plt.savefig(out_png, dpi=130)
    plt.close(fig)
    return out_png, fit
