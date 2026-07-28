"""
distributions.py -- probability distribution fits used in the magnetic-domain
literature.

Three quantities, three standard functions:

1. Per-pixel m_z (magnetization) histogram  -> BIMODAL (two Gaussians at +-1)
   The XMCD-asymmetry histogram of a saturated PMA two-level domain image has
   peaks at the up/down values; a central peak appears from walls/background.
   (This is what stats.pdf_three_state summarizes discretely.)

2. Domain / feature SIZE distribution        -> LOG-NORMAL
   Grain and magnetic-domain sizes in thin films are routinely log-normal.

3. Current/field-step AVALANCHE size          -> POWER LAW  P(S) ~ S^(-tau)
   The reversed-cluster area between two consecutive images (Barkhausen
   avalanche) follows a power law; tau ~ 1.0-1.8 depending on regime.

This module provides log-normal and power-law fits plus a plotting helper.
"""

import numpy as np
from scipy import stats


# ----------------------------------------------------------------------
def fit_lognormal(values):
    """
    Fit a log-normal to positive values (e.g. domain equivalent diameters).
    Returns shape parameters, a goodness-of-fit (KS) statistic, and
    log-likelihood/AIC/BIC so this can be compared directly against
    fit_gaussian() on the same sample (see compare_lognormal_gaussian).
    """
    v = np.asarray(values, float)
    v = v[v > 0]
    if v.size < 5:
        return None
    shape, loc, scale = stats.lognorm.fit(v, floc=0)  # floc=0 -> 2-parameter
    ks = stats.kstest(v, "lognorm", args=(shape, loc, scale))
    loglik = float(np.sum(stats.lognorm.logpdf(v, shape, loc, scale)))
    k = 2  # free params: shape (sigma_log), scale (median) -- loc fixed at 0
    n = int(v.size)
    aic = 2 * k - 2 * loglik
    bic = k * np.log(n) - 2 * loglik
    xs = np.linspace(v.min(), v.max(), 200)
    pdf = stats.lognorm.pdf(xs, shape, loc, scale)
    # mode of a log-normal = exp(mu_log - sigma_log^2) = median * exp(-sigma_log^2)
    # -- the peak of the fitted PDF, distinct from the median (scale) and the
    # mean, and always <= median for a log-normal (right-skewed).
    mode = float(scale * np.exp(-shape ** 2))
    return {"dist": "lognormal", "n": n,
            "sigma_log": float(shape),          # shape = sigma of ln(x)
            "median": float(scale),             # median = exp(mu)
            "mu_log": float(np.log(scale)),
            "mode": mode,
            "loglik": loglik, "aic": float(aic), "bic": float(bic),
            "ks_stat": float(ks.statistic), "ks_p": float(ks.pvalue),
            "pdf_x": xs, "pdf_y": pdf}


def fit_gaussian(values):
    """
    Fit a Gaussian (normal) to the same kind of sample fit_lognormal() takes
    (e.g. domain widths), for a direct comparison against the log-normal
    model. Returns the same style of fields (n, loglik, AIC, BIC, KS
    stat/p, pdf_x/pdf_y) so the two fits can be compared like-for-like --
    see compare_lognormal_gaussian().
    """
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if v.size < 5:
        return None
    mu, sigma = stats.norm.fit(v)
    if sigma <= 0:
        return None
    # stats.kstest(v, "norm", args=(mu, sigma)) crashes on scipy >= 1.18
    # (an ndtr() argument-count regression in its internal CDF dispatch for
    # "norm" specifically -- "lognorm" and other named dists are unaffected).
    # A frozen-distribution .cdf callable sidesteps that dispatch entirely
    # and gives the identical KS statistic.
    ks = stats.kstest(v, stats.norm(loc=mu, scale=sigma).cdf)
    loglik = float(np.sum(stats.norm.logpdf(v, mu, sigma)))
    k = 2  # free params: mean, sd
    n = int(v.size)
    aic = 2 * k - 2 * loglik
    bic = k * np.log(n) - 2 * loglik
    xs = np.linspace(v.min(), v.max(), 200)
    pdf = stats.norm.pdf(xs, mu, sigma)
    return {"dist": "gaussian", "n": n,
            "mean": float(mu), "sd": float(sigma),
            "loglik": loglik, "aic": float(aic), "bic": float(bic),
            "ks_stat": float(ks.statistic), "ks_p": float(ks.pvalue),
            "pdf_x": xs, "pdf_y": pdf}


def compare_lognormal_gaussian(values):
    """
    Fit both a log-normal and a Gaussian to the same sample (e.g. one
    state's domain widths) and compare via AIC/BIC (Burnham & Anderson,
    2002): the model with the LOWER AIC/BIC is better supported by the
    data given its complexity; both models have the same number of free
    parameters (2) here, so AIC/BIC differences reduce to a log-likelihood
    comparison. A |delta AIC| > ~10 is usually treated as decisive.

    Returns {"lognormal": fit_or_None, "gaussian": fit_or_None,
             "delta_aic": aic_gaussian - aic_lognormal (positive favors
                          log-normal, negative favors Gaussian),
             "delta_bic": same, using BIC,
             "preferred": "lognormal" | "gaussian" | None (None if either
                          fit failed, e.g. < 5 samples)}.
    """
    ln = fit_lognormal(values)
    ga = fit_gaussian(values)
    out = {"lognormal": ln, "gaussian": ga,
           "delta_aic": None, "delta_bic": None, "preferred": None}
    if ln and ga:
        out["delta_aic"] = ga["aic"] - ln["aic"]
        out["delta_bic"] = ga["bic"] - ln["bic"]
        out["preferred"] = "lognormal" if out["delta_aic"] > 0 else "gaussian"
    return out


def fit_powerlaw(values, xmin=None):
    """
    Maximum-likelihood power-law exponent for P(S) ~ S^(-tau), continuous form
    (Clauset-Shalizi-Newman):  tau = 1 + n / sum(ln(x/xmin)).
    If xmin is None, use the smallest value.
    """
    v = np.asarray(values, float)
    v = v[v > 0]
    if xmin is None:
        xmin = v.min()
    tail = v[v >= xmin]
    if tail.size < 5:
        return None
    tau = 1.0 + tail.size / np.sum(np.log(tail / xmin))
    err = (tau - 1.0) / np.sqrt(tail.size)
    return {"dist": "powerlaw", "tau": float(tau), "tau_err": float(err),
            "xmin": float(xmin), "n_tail": int(tail.size)}


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
    plt.savefig(out_png, dpi=130)
    plt.close(fig)
    return out_png, fit
