"""
magnetization.py -- rigorous statistical analysis of the magnetization order
parameter from an XMCD-PEEM image.

The order parameter is the spatial mean of the per-pixel asymmetry over the
flake:
        <m_z> = (1/N) * sum_i  m_i ,   m_i in [-1, 1]   (m_i ~ M_z / M_s)

Why a naive error bar is wrong
------------------------------
The pixels are NOT independent: every pixel inside a domain carries the same
sign, so the number of *independent* samples is ~ (flake area)/(domain area),
not the pixel count. Using SEM = sd/sqrt(N_pixels) underestimates the
uncertainty by a large factor. This module corrects for spatial correlation.

What it computes
----------------
1. <m_z> with a SPATIAL-CORRELATION-CORRECTED standard error:
       Var(<m_z>) = sigma^2 / N_eff ,  N_eff = N_pixels / tau_A
   where tau_A is the integrated 2-D autocorrelation area (in pixels), and the
   correlation length xi (C(xi)=1/e) is reported as the domain length scale.
2. A BLOCK-BOOTSTRAP 95% confidence interval (resamples xi-sized blocks, so it
   respects the correlation without assuming a parametric form).
3. A SIGNIFICANCE test of H0: <m_z> = 0 (is the flake net-magnetized?).
4. A GAUSSIAN-MIXTURE fit of the per-pixel asymmetry histogram (down / zero /
   up components): peak positions = saturated levels, weights = area fractions.
5. THRESHOLD ROBUSTNESS: <m_z> and fractions vs the classification threshold.

Depends only on numpy / scipy / matplotlib.
"""

import numpy as np
from scipy import optimize, stats

from .colors import COLOR_UP, COLOR_DOWN


# ----------------------------------------------------------------------
def asymmetry_field(layers, norm_percentile=99):
    """
    Continuous normalized asymmetry m(x,y) in [-1,1] over the whole frame,
    plus the flake mask. Saturated domains -> ~ +-1.
    """
    flat, roi = layers["flat"], layers["roi"]
    scale = np.percentile(np.abs(flat[roi]), norm_percentile) + 1e-12
    m = np.clip(flat / scale, -1.0, 1.0)
    return m, roi


# ----------------------------------------------------------------------
def area_averaged_mz(layers, norm_percentile=99):
    """
    Area-averaged magnetization order parameter, BEFORE any up/neutral/down
    thresholding:
        <m_z> = (1/N) * sum_{i in ROI} a_i
    where a_i is the continuous, background-subtracted, percentile-normalized
    asymmetry (asymmetry_field() -- the same field magnetization_statistics
    uses, just without the correlation-length/bootstrap/mixture-fit machinery
    on top). Cheap: one percentile + one mean over the ROI, no bootstrap.

    Note: this is independent of three_state_field()'s classification mode --
    in that function's z-score mode, layers["asymmetry"] is mean-centered by
    construction (its ROI average is ~0 by definition, not a physically
    meaningful order parameter), so this recomputes the percentile-normalized
    asymmetry from layers["flat"] directly rather than reusing that field.

    Returns {"mean_mz", "sd", "N_pixels"}.
    """
    m, roi = asymmetry_field(layers, norm_percentile=norm_percentile)
    v = m[roi]
    return {"mean_mz": float(v.mean()),
            "sd": float(v.std(ddof=1)) if v.size > 1 else float("nan"),
            "N_pixels": int(v.size)}


# ----------------------------------------------------------------------
def net_mz_raw(mz, layers):
    """
    Area-averaged magnetization using the RAW (unnormalized) asymmetry
    signal, over only the pixels that were classified up/down (excludes the
    neutral/gray-band pixels, same SELECTION as pdf_three_state's discrete
    net_m) -- but averages each pixel's actual signal magnitude instead of
    snapping it to +-1 and counting, and does so on layers["flat"] directly,
    without area_averaged_mz's percentile-normalize-and-clip step:

        <m_z> = (1/N') * sum_{i: mz_i != 0} flat_i

    This sits between the toolkit's other two <m_z> definitions and combines
    the useful half of each:
      - pdf_three_state's discrete net_m: same pixel SELECTION (drops the
        gray band) but discards how strongly magnetized each included pixel
        actually was, snapping it to +-1 before averaging.
      - area_averaged_mz's continuous mean: keeps each pixel's real signal,
        but averages over EVERY roi pixel (including the gray band, which
        dilutes it toward 0) and rescales by a percentile-based saturation
        clip first.

    `mz` is whatever three_state_field() produced (m_threshold VALUE mode or
    the legacy z-score mode) -- reusing its inside/outside decision keeps
    this consistent with whichever classification mode was actually used,
    rather than re-deriving a separate threshold test here.

    Returns {"mean_mz", "sd", "N_pixels" (mz != 0 count), "N_excluded"
    (gray-band pixel count)}; mean_mz/sd are NaN if no pixel cleared the
    threshold.
    """
    flat, roi = layers["flat"], layers["roi"]
    outside = roi & (mz != 0)
    v = flat[outside]
    n_excluded = int(roi.sum()) - int(v.size)
    if v.size == 0:
        return {"mean_mz": float("nan"), "sd": float("nan"),
                "N_pixels": 0, "N_excluded": n_excluded}
    return {"mean_mz": float(v.mean()),
            "sd": float(v.std(ddof=1)) if v.size > 1 else float("nan"),
            "N_pixels": int(v.size), "N_excluded": n_excluded}


# ----------------------------------------------------------------------
def correlation_length(m, roi):
    """
    Masked 2-D autocorrelation of the (de-meaned) asymmetry on the flake.
    Returns (xi, tau_A, C_radial, r):
        xi     -- correlation length in px, where radial C(r) falls to 1/e
        tau_A  -- integrated autocorrelation AREA in px (sum of C over lags),
                  i.e. the effective area of one independent domain
        C_radial, r -- radial autocorrelation for plotting
    """
    mu = m[roi].mean()
    f = np.where(roi, m - mu, 0.0)
    mask = roi.astype(float)

    # FFT autocorrelations; divide signal autocorr by overlap count per lag
    F = np.fft.rfft2(f)
    ac = np.fft.irfft2(F * np.conj(F), s=f.shape)
    M = np.fft.rfft2(mask)
    acm = np.fft.irfft2(M * np.conj(M), s=mask.shape)
    ac = np.fft.fftshift(ac)
    acm = np.fft.fftshift(acm)
    C = ac / np.maximum(acm, 1.0)
    cy, cx = np.array(C.shape) // 2
    C0 = C[cy, cx]
    if C0 <= 0:
        return np.nan, 1.0, np.array([1.0]), np.array([0.0])
    C = C / C0

    # radial average
    yy, xx = np.indices(C.shape)
    r = np.round(np.hypot(xx - cx, yy - cy)).astype(int)
    rmax = min(cy, cx)
    C_radial = np.array([C[r == k].mean() for k in range(rmax)])
    rr = np.arange(rmax)

    # correlation length: first crossing of 1/e
    below = np.where(C_radial < np.exp(-1))[0]
    xi = float(below[0]) if below.size else float(rmax)

    # integrated autocorrelation area: sum of positive radial lobe, 2-D weighted
    pos = C_radial.copy()
    cut = below[0] if below.size else rmax
    # integrate C over 2-D lags out to first zero-ish crossing
    zero_cross = np.where(C_radial <= 0)[0]
    cut = int(zero_cross[0]) if zero_cross.size else rmax
    tau_A = float(np.sum(2 * np.pi * rr[:cut] * np.clip(C_radial[:cut], 0, None)))
    tau_A = max(tau_A, 1.0)
    return xi, tau_A, C_radial, rr


# ----------------------------------------------------------------------
def block_bootstrap_mean(m, roi, block, n_boot=2000, seed=0):
    """
    Block bootstrap of <m_z>: tile the flake into `block`-sized squares,
    resample tiles (with replacement), pool their pixels, recompute the mean.
    Respects spatial correlation when block >~ correlation length.
    Returns (lo95, hi95, boot_means).
    """
    rng = np.random.default_rng(seed)
    H, W = m.shape
    ny, nx = -(-H // block), -(-W // block)   # ceil division

    # Each tile's sum/count is fixed -- the bootstrap only resamples WHICH
    # tiles are drawn, not their contents -- so get every tile's sum/count
    # in one pass via bincount instead of building a per-tile pixel list and
    # re-summing picked tiles from scratch on every one of n_boot draws.
    tile_id = ((np.arange(H) // block)[:, None] * nx
               + (np.arange(W) // block)[None, :])
    n_tiles_grid = ny * nx
    counts_grid = np.bincount(tile_id[roi], minlength=n_tiles_grid)
    sums_grid = np.bincount(tile_id[roi], weights=m[roi], minlength=n_tiles_grid)
    nonempty = counts_grid > 0
    tile_sums = sums_grid[nonempty]
    counts = counts_grid[nonempty]

    if tile_sums.size < 3:
        v = m[roi]
        return float(v.mean()), float(v.mean()), np.array([v.mean()])

    idx = np.arange(tile_sums.size)
    picks = rng.choice(idx, size=(n_boot, tile_sums.size), replace=True)
    boot = tile_sums[picks].sum(axis=1) / counts[picks].sum(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(lo), float(hi), boot


# ----------------------------------------------------------------------
def _three_gauss(x, a1, m1, s1, a2, m2, s2, a3, m3, s3):
    g = lambda a, mu, s: a * np.exp(-0.5 * ((x - mu) / s) ** 2)
    return g(a1, m1, s1) + g(a2, m2, s2) + g(a3, m3, s3)


def asymmetry_mixture_fit(m, roi, bins=80):
    """
    Fit a 3-Gaussian mixture (down / zero / up) to the per-pixel asymmetry
    histogram. Returns component (weight, mean, sigma) and the histogram.
    """
    v = m[roi]
    hist, edges = np.histogram(v, bins=bins, range=(-1, 1), density=True)
    ctr = 0.5 * (edges[:-1] + edges[1:])
    p0 = [hist.max(), -0.6, 0.2, hist.max() * 0.5, 0.0, 0.1, hist.max(), 0.6, 0.2]
    bounds = ([0, -1, 0.02] * 3, [np.inf, 1, 1] * 3)
    try:
        popt, _ = optimize.curve_fit(_three_gauss, ctr, hist, p0=p0,
                                     bounds=bounds, maxfev=20000)
    except Exception:
        popt = p0
    comps = []
    for k in range(3):
        a, mu, s = popt[3 * k:3 * k + 3]
        comps.append({"weight_area": float(a * s * np.sqrt(2 * np.pi)),
                      "mean": float(mu), "sigma": float(s)})
    comps.sort(key=lambda c: c["mean"])
    tot = sum(c["weight_area"] for c in comps) + 1e-12
    for c in comps:
        c["weight"] = c["weight_area"] / tot
    return {"components": comps, "hist": hist, "centers": ctr, "popt": popt}


# ----------------------------------------------------------------------
def magnetization_statistics(layers, m_threshold=0.1, n_boot=2000):
    """
    Full rigorous magnetization analysis. Returns a dict of results.
    """
    m, roi = asymmetry_field(layers)
    v = m[roi]
    aa = area_averaged_mz(layers)
    N, mean, sd = aa["N_pixels"], aa["mean_mz"], aa["sd"]

    xi, tau_A, C_radial, rr = correlation_length(m, roi)
    N_eff = max(N / tau_A, 1.0)
    sem_naive = sd / np.sqrt(N)
    sem_corr = sd / np.sqrt(N_eff)

    # analytic 95% CI using corrected SE and t with N_eff-1 dof
    tcrit = stats.t.ppf(0.975, max(N_eff - 1, 1))
    ci_lo, ci_hi = mean - tcrit * sem_corr, mean + tcrit * sem_corr

    # block bootstrap CI (block ~ 2*xi, at least 8 px)
    block = int(max(8, round(2 * xi)))
    blo, bhi, boot = block_bootstrap_mean(m, roi, block, n_boot=n_boot)

    # significance vs 0 (correlation-corrected)
    z = mean / (sem_corr + 1e-12)
    pval = 2 * (1 - stats.norm.cdf(abs(z)))

    fit = asymmetry_mixture_fit(m, roi)

    # threshold robustness sweep
    ts = np.linspace(0.02, 0.30, 15)
    nets, fup, fdn = [], [], []
    for t in ts:
        up = (v > t).sum(); dn = (v < -t).sum()
        nets.append((up - dn) / max(up + dn, 1))
        fup.append(up / N); fdn.append(dn / N)

    return {
        "mean_mz": mean, "sd": sd, "N_pixels": N,
        "corr_length_px": xi, "tau_area_px": tau_A, "N_eff": N_eff,
        "sem_naive": sem_naive, "sem_corrected": sem_corr,
        "ci95_analytic": (ci_lo, ci_hi),
        "ci95_bootstrap": (blo, bhi), "bootstrap_block_px": block,
        "z_vs_zero": z, "p_value_vs_zero": pval,
        "mixture": fit,
        "threshold_sweep": {"t": ts, "net_m": np.array(nets),
                            "frac_up": np.array(fup), "frac_down": np.array(fdn)},
        "_field": (m, roi, C_radial, rr, boot),
    }


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


def print_magnetization_report(res):
    print("=" * 64)
    print("  RIGOROUS MAGNETIZATION STATISTICS")
    print("=" * 64)
    print(f"  <m_z>                 = {res['mean_mz']:+.4f}")
    print(f"  pixel sd              = {res['sd']:.4f}   (N_pixels = {res['N_pixels']})")
    print(f"  correlation length xi = {res['corr_length_px']:.1f} px")
    print(f"  indep. domain area    = {res['tau_area_px']:.0f} px  ->  N_eff = {res['N_eff']:.0f}")
    print(f"  SE (naive, wrong)     = {res['sem_naive']:.5f}")
    print(f"  SE (corrected)        = {res['sem_corrected']:.5f}   "
          f"(x{res['sem_corrected']/max(res['sem_naive'],1e-9):.1f} larger)")
    lo, hi = res["ci95_analytic"]; blo, bhi = res["ci95_bootstrap"]
    print(f"  95% CI (analytic)     = [{lo:+.4f}, {hi:+.4f}]")
    print(f"  95% CI (bootstrap)    = [{blo:+.4f}, {bhi:+.4f}]  block={res['bootstrap_block_px']}px")
    print(f"  H0: <m_z>=0           z={res['z_vs_zero']:+.2f}  p={res['p_value_vs_zero']:.3g}")
    sig = "net-magnetized" if res["p_value_vs_zero"] < 0.05 else "consistent with demagnetized"
    print(f"  -> {sig}")
    print("  mixture components (weight, mean, sigma):")
    for c in res["mixture"]["components"]:
        print(f"     w={c['weight']:.2f}  mean={c['mean']:+.3f}  sigma={c['sigma']:.3f}")
    print("=" * 64)
