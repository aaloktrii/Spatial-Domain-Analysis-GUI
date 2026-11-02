"""Figures for core/morphometry.py (plotting layer, no analysis)."""
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
from ..core.distributions import fit_lognormal, fit_gaussian, compare_lognormal_gaussian
from ..core.loaders import neighbor_structure, _remove_small
from ..core.magnetization import area_averaged_mz
from ..colors import (
    COLOR_UP_FILL as _COLOR_UP_FILL, COLOR_UP_LINE as _COLOR_UP_LINE,
    COLOR_DOWN_FILL as _COLOR_DOWN_FILL, COLOR_DOWN_LINE as _COLOR_DOWN_LINE,
    CMAP3_HEX,
)
from ..core.morphometry import (
    _PUB_RC,
    _save_fit_comparison,
    _save_lognormal_fits,
    per_domain_width_area,
    per_state_morphometry,
    qq_series_per_state,
)


def _overlay_lognormal(ax, values, edges, color, label):
    """
    Fit a log-normal to `values` and draw it as a thin line scaled to match
    a COUNT histogram (not a density one) that used `edges` as its bin
    edges. Returns the fit dict (median, sigma_log, mu_log, KS stat/p, n) so
    the caller can save it, or None if there weren't enough points (<5) to
    fit.
    """
    fit = fit_lognormal(values)
    if fit is None:
        return None
    bin_width = edges[1] - edges[0]
    y = fit["pdf_y"] * values.size * bin_width       # density -> expected count/bin
    ax.plot(fit["pdf_x"], y, color=color, linewidth=0.6, label=label)
    return fit

def _overlay_gaussian(ax, values, edges, color, label):
    """
    Same as _overlay_lognormal but for a Gaussian fit, drawn dashed so the
    two overlaid model curves stay visually distinct on the same axes.
    Returns the fit dict (mean, sd, AIC, BIC, KS stat/p, n), or None if
    there weren't enough points (<5) to fit.
    """
    fit = fit_gaussian(values)
    if fit is None:
        return None
    bin_width = edges[1] - edges[0]
    y = fit["pdf_y"] * values.size * bin_width
    ax.plot(fit["pdf_x"], y, color=color, linewidth=0.6, linestyle="--", label=label)
    return fit

def export_panels_tiff(img, mz, roi, um_per_px, outdir, stem, dpi=300):
    """
    Write the three domain panels as SEPARATE .tiff files, each with its own
    colorbar, in microns when calibrated. Returns the list of file paths:
        <stem>_three_state.tiff, <stem>_domain_width.tiff, <stem>_skeletons.tiff
    """
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm

    mm = per_state_morphometry(mz, um_per_px=um_per_px)
    unit = "nm" if um_per_px else "px"
    sc = (um_per_px * 1000.0) if um_per_px else 1.0   # width panel only: um -> nm
    cmap3 = ListedColormap(CMAP3_HEX)
    norm3 = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap3.N)
    masked = np.where(roi, mz, np.nan)
    paths = []

    with plt.rc_context(_PUB_RC):
        fig, ax = plt.subplots(figsize=(5.2, 6.2))
        ax.imshow(img, cmap="gray")
        im = ax.imshow(masked, cmap=cmap3, norm=norm3, alpha=0.85)
        ax.set_title("flake three-state $m_z$"); ax.axis("off")
        cb = fig.colorbar(im, ax=ax, ticks=[-1, 0, 1], fraction=0.046, pad=0.04)
        cb.ax.set_yticklabels(["down ($-1$)", "none ($0$)", "up ($+1$)"])
        plt.tight_layout()
        f1 = os.path.join(outdir, f"{stem}_three_state.tiff")
        plt.savefig(f1, dpi=dpi); plt.close(fig); paths.append(f1)

        wmap = np.zeros(mz.shape)
        for st in (mm["up"], mm["down"]):
            if st["dt"] is not None:
                wmap += 2.0 * st["dt"]
        wmap = np.where(mz != 0, wmap * sc, np.nan)
        fig, ax = plt.subplots(figsize=(5.6, 6.2))
        im = ax.imshow(wmap, cmap="viridis")
        ax.set_title("local domain width"); ax.axis("off")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(f"domain width ({unit})")
        plt.tight_layout()
        f2 = os.path.join(outdir, f"{stem}_domain_width.tiff")
        plt.savefig(f2, dpi=dpi); plt.close(fig); paths.append(f2)

        fig, ax = plt.subplots(figsize=(5.2, 6.2))
        ax.imshow(masked, cmap=cmap3, norm=norm3)
        sk = np.zeros((*mz.shape, 4))
        for st, col in ((mm["up"], [1, 1, 0, 1]), (mm["down"], [0, 1, 1, 1])):
            if st["skeleton"] is not None:
                sk[st["skeleton"]] = col
        ax.imshow(sk)
        ax.set_title("domain skeletons"); ax.axis("off")
        cb = fig.colorbar(im, ax=ax, ticks=[-1, 0, 1], fraction=0.046, pad=0.04)
        cb.ax.set_yticklabels(["down", "none", "up"])
        plt.tight_layout()
        f3 = os.path.join(outdir, f"{stem}_skeletons.tiff")
        plt.savefig(f3, dpi=dpi); plt.close(fig); paths.append(f3)
    return paths

def make_morphometry_figure(mz, roi, um_per_px, out_png, title="", min_px=0, mm=None):
    """
    `min_px` is forwarded to per_state_morphometry (dropping speckle
    fragments before measuring width, same "Min Domain Size" control used
    elsewhere) -- pass the SAME min_px other consumers of this mz (e.g.
    summarize_domains) use, or width_median/the log-normal fit here won't
    match theirs (different-sized skeleton-pixel samples).

    `mm` lets a caller that already ran per_state_morphometry(mz, ...) pass
    that result in directly instead of this function recomputing it (and
    refitting every log-normal a second time) from scratch; when given,
    `min_px` is ignored (mm was already built with whatever min_px the
    caller used).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if mm is None:
        mm = per_state_morphometry(mz, um_per_px=um_per_px, min_px=min_px)
    # width is shown in nm when calibrated (natural scale for these domains);
    # um_per_px itself is unchanged, this only affects display scaling below.
    unit = "nm" if um_per_px else "px"
    w_scale = 1000.0 if um_per_px else 1.0   # um -> nm, applied on top of um_per_px
    up, dn = mm["up"], mm["down"]

    with plt.rc_context(_PUB_RC):
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))

        # (a) width map: 2*DT over up+down domains
        wmap = np.zeros(mz.shape)
        for st in (up, dn):
            if st["dt"] is not None:
                wmap += 2.0 * st["dt"]
        wmap = np.where(mz != 0, wmap, np.nan) * (um_per_px or 1.0) * w_scale
        im = ax[0].imshow(wmap, cmap="viridis")
        ax[0].set_title(f"(a) local domain width ({unit})"); ax[0].axis("off")
        fig.colorbar(im, ax=ax[0], fraction=0.046, pad=0.04)

        # (b) skeletons over the three-state map
        cmap3 = matplotlib.colors.ListedColormap(CMAP3_HEX)
        ax[1].imshow(np.where(roi, mz, np.nan), cmap=cmap3, vmin=-1, vmax=1)
        sk = np.zeros((*mz.shape, 4))
        for st, col in ((up, [1, 1, 0, 1]), (dn, [0, 1, 1, 1])):
            if st["skeleton"] is not None:
                sk[st["skeleton"]] = col
        ax[1].imshow(sk)
        ax[1].set_title("(b) domain skeletons"); ax[1].axis("off")

        # (c) width distribution, lighter fill + thin log-normal fit overlay
        up_w = up["widths_px"] * (um_per_px or 1.0) * w_scale
        dn_w = dn["widths_px"] * (um_per_px or 1.0) * w_scale
        datasets = [w for w in (up_w, dn_w) if w.size]
        if datasets:
            edges = np.histogram_bin_edges(np.concatenate(datasets), bins=24)
            for w, col, lab in ((up_w, _COLOR_UP_FILL, "up"), (dn_w, _COLOR_DOWN_FILL, "down")):
                if w.size:
                    ax[2].hist(w, bins=edges, alpha=0.85, color=col, label=lab)
            if up_w.size >= 5:
                _overlay_lognormal(ax[2], up_w, edges, _COLOR_UP_LINE, "up fit")
            if dn_w.size >= 5:
                _overlay_lognormal(ax[2], dn_w, edges, _COLOR_DOWN_LINE, "down fit")
        ax[2].set_xlabel(f"domain width ({unit})"); ax[2].set_ylabel("count")
        per = mm["domain_period"] * w_scale if np.isfinite(mm["domain_period"]) else mm["domain_period"]
        ttl = "(c) domain-width distribution"
        if np.isfinite(per):
            ttl += f"\nperiod ~ {per:.2f} {unit}"
        ax[2].set_title(ttl); ax[2].legend(fontsize=8)

        if title:
            fig.suptitle(title, fontsize=12)
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        plt.savefig(out_png, dpi=130); plt.close(fig)
    return out_png, mm

def make_size_distribution_figure(mz, um_per_px=None, min_px=0, bins=24,
                                   out_stem="size_distribution",
                                   formats=("svg",), dpi=300, title="",
                                   compare_gaussian=False, quantity="width",
                                   fractions=None):
    """
    Standalone version of make_morphometry_figure's panel (c): the
    skeleton-pixel WIDTH or PERIOD distribution for up vs. down, on its own
    (not merged into the 3-panel figure). quantity="width" uses each
    state's widths_px (2*DT on its own skeleton); quantity="period" uses
    periods_px (each skeleton pixel's own width + the nearest OTHER state's
    width there -- see _local_period_px).

    up and down are drawn as SIDE-BY-SIDE bars per bin (matplotlib's
    multi-dataset ax.hist([up, down], ...)), not overlaid with alpha -- so
    the two colors never blend into a third color where they overlap. If you
    want the overlaid/blended look back, use make_morphometry_figure's panel
    (c) or draw_distributions in plotting/distribution_panels.py instead.

    A thin log-normal fit is overlaid for each of the up/down distributions
    (scaled to match the count histogram, not a density plot); the fitted
    parameters (median, sigma_log, AIC, BIC, KS stat/p, n) are written to
    "<out_stem>_lognormal_fit.json" alongside the figure, and that path is
    appended to the returned list.

    If compare_gaussian=True, a Gaussian fit is ALSO overlaid for each state
    (dashed, same color as that state's log-normal line) and the AIC/BIC/KS
    comparison between the two models is written to
    "<out_stem>_fit_comparison.json" (see compare_lognormal_gaussian).

    `fractions`, if given as (f_up, f_down, f_dw), annotates the panel with
    the up / down / domain-wall AREA FRACTIONS ("f_+ = .., f_- = .., f_dw =
    ..") -- the three-state PDF (pdf_three_state's p_up/p_down/p_zero) the
    caller already computed, so the number on the figure matches the rest of
    the analysis rather than being re-derived here. None omits it.

    Values are in nm when calibrated (um_per_px given), else px -- same
    convention as the rest of this module. Writes one file per entry in
    `formats` to "<out_stem>.<ext>". Returns (list_of_paths, per_state_morphometry_dict).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mm = per_state_morphometry(mz, um_per_px=um_per_px, min_px=min_px)
    unit = "nm" if um_per_px else "px"
    w_scale = 1000.0 if um_per_px else 1.0
    key = "widths_px" if quantity == "width" else "periods_px"
    up_w = mm["up"][key] * (um_per_px or 1.0) * w_scale
    dn_w = mm["down"][key] * (um_per_px or 1.0) * w_scale

    fits = {}
    comparisons = {}
    with plt.rc_context(_PUB_RC):
        fig, ax = plt.subplots(figsize=(6.5, 4.6))
        datasets = [w for w in (up_w, dn_w) if w.size]
        labels = [lab for lab, w in (("up", up_w), ("down", dn_w)) if w.size]
        fill_colors = [c for c, w in ((_COLOR_UP_FILL, up_w), (_COLOR_DOWN_FILL, dn_w))
                       if w.size]

        if datasets:
            edges = np.histogram_bin_edges(np.concatenate(datasets), bins=bins)
            ax.hist(datasets, bins=edges, color=fill_colors, label=labels)
            if up_w.size >= 5:
                fits["up"] = _overlay_lognormal(ax, up_w, edges, _COLOR_UP_LINE,
                                                f"up log-normal fit")
                if compare_gaussian:
                    _overlay_gaussian(ax, up_w, edges, _COLOR_UP_LINE, "up Gaussian fit")
                    comparisons["up"] = compare_lognormal_gaussian(up_w)
            if dn_w.size >= 5:
                fits["down"] = _overlay_lognormal(ax, dn_w, edges, _COLOR_DOWN_LINE,
                                                   f"down log-normal fit")
                if compare_gaussian:
                    _overlay_gaussian(ax, dn_w, edges, _COLOR_DOWN_LINE, "down Gaussian fit")
                    comparisons["down"] = compare_lognormal_gaussian(dn_w)

        ax.set_xlabel(f"domain {quantity} ({unit})")
        ax.set_ylabel("count (skeleton pixels)")
        per = mm.get("domain_period_nm") if um_per_px else mm["domain_period"]
        ttl = f"domain-{quantity} distribution (skeleton pixels)"
        if per == per:  # not NaN
            ttl += f"\nperiod ~ {per:.2f} {unit}"
        ax.set_title(ttl)
        ax.legend(fontsize=9)

        if fractions is not None:
            f_up, f_dn, f_dw = fractions
            ax.text(0.97, 0.55,
                    f"$f_+$ = {f_up:.2f}, $f_-$ = {f_dn:.2f}, $f_{{dw}}$ = {f_dw:.2f}",
                    transform=ax.transAxes, ha="right", va="top", fontsize=10)

        if title:
            fig.suptitle(title, fontsize=12)
        plt.tight_layout(rect=[0, 0, 1, 0.94] if title else None)

        out_paths = []
        for ext in formats:
            path = f"{out_stem}.{ext}"
            fig.savefig(path, dpi=dpi, format=("tiff" if ext == "tiff" else ext))
            out_paths.append(path)
        plt.close(fig)

    if fits:
        out_paths.append(_save_lognormal_fits(fits, f"{out_stem}_lognormal_fit.json", unit))
    if compare_gaussian and comparisons:
        out_paths.append(_save_fit_comparison(comparisons, f"{out_stem}_fit_comparison.json", unit))
    return out_paths, mm

def make_width_distribution_figure(mz, um_per_px=None, min_px=0, bins=24,
                                    out_stem="width_distribution",
                                    formats=("svg",), dpi=300, title="",
                                    compare_gaussian=False, fractions=None):
    """Backward-compatible name: make_size_distribution_figure(..., quantity="width")."""
    return make_size_distribution_figure(mz, um_per_px=um_per_px, min_px=min_px,
                                          bins=bins, out_stem=out_stem, formats=formats,
                                          dpi=dpi, title=title,
                                          compare_gaussian=compare_gaussian, quantity="width",
                                          fractions=fractions)

def make_period_distribution_figure(mz, um_per_px=None, min_px=0, bins=24,
                                     out_stem="period_distribution",
                                     formats=("svg",), dpi=300, title="",
                                     compare_gaussian=False, fractions=None):
    """Same as make_width_distribution_figure but for the local domain-PERIOD
    sample (see per_state_morphometry / _local_period_px for how a period
    population is derived from the up/down skeletons)."""
    return make_size_distribution_figure(mz, um_per_px=um_per_px, min_px=min_px,
                                          bins=bins, out_stem=out_stem, formats=formats,
                                          dpi=dpi, title=title,
                                          compare_gaussian=compare_gaussian, quantity="period",
                                          fractions=fractions)

def make_qq_plot_figure(mz, um_per_px=None, min_px=0, out_stem="width_qq",
                         formats=("svg",), dpi=300, title="", quantity="width"):
    """
    Standalone Q-Q plot of domain-width (or domain-period, quantity="period")
    samples against a fitted log-normal -- a visual goodness-of-fit check for
    the log-normal assumption used everywhere else in this module (the
    median/sigma_log statistics). A straight line supports log-normal;
    systematic curvature, especially in the tails, argues against it.

    Up and down are fit and plotted SEPARATELY (each against its own
    log-normal, in its own color), not pooled into one combined sample --
    same convention as make_size_distribution_figure's histogram, since up
    and down distributions can genuinely differ (e.g. under applied current).

    Same GUI toggle/export pairing as make_size_distribution_figure: writes
    "<out_stem>.<ext>" for each entry in `formats`, plus
    "<out_stem>_lognormal_fit.json" with the fitted up/down parameters for
    whichever state(s) had enough samples to fit.

    Returns (list_of_paths, fits_dict_or_None). fits_dict has "up"/"down"
    keys (only present where there were >= 5 samples); None if neither
    state had enough.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mm = per_state_morphometry(mz, um_per_px=um_per_px, min_px=min_px)
    unit = "nm" if um_per_px else "px"
    w_scale = (um_per_px * 1000.0) if um_per_px else 1.0
    qq = qq_series_per_state(mm, scale=w_scale, quantity=quantity)

    fits = {}
    with plt.rc_context(_PUB_RC):
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        any_series = False
        for name, col_pt, col_line in (
            ("up", _COLOR_UP_FILL, _COLOR_UP_LINE),
            ("down", _COLOR_DOWN_FILL, _COLOR_DOWN_LINE),
        ):
            series = qq[name]
            if series is None:
                continue
            any_series = True
            fits[name] = series["fit"]
            ax.plot(series["osm"], series["osr"], "o", color=col_pt,
                    markersize=4, label=f"{name} (n={series['fit']['n']})")
            fit_line = series["slope"] * series["osm"] + series["intercept"]
            ax.plot(series["osm"], fit_line, "-", color=col_line, linewidth=1)

        if any_series:
            ax.set_title(f"domain-{quantity} Q-Q vs. log-normal (fit per state)")
            ax.legend(fontsize=9)
        else:
            ax.text(0.5, 0.5, f"not enough {quantity} samples\n(need >= 5 per state)",
                    ha="center", va="center", transform=ax.transAxes, fontsize=10)
            ax.set_title(f"domain-{quantity} Q-Q vs. log-normal")
        ax.set_xlabel(f"theoretical log-normal quantiles ({unit})")
        ax.set_ylabel(f"ordered domain-{quantity} samples ({unit})")

        if title:
            fig.suptitle(title, fontsize=12)
        plt.tight_layout(rect=[0, 0, 1, 0.94] if title else None)

        out_paths = []
        for ext in formats:
            path = f"{out_stem}.{ext}"
            fig.savefig(path, dpi=dpi, format=("tiff" if ext == "tiff" else ext))
            out_paths.append(path)
        plt.close(fig)

    if fits:
        out_paths.append(_save_lognormal_fits(fits, f"{out_stem}_lognormal_fit.json", unit))
    return out_paths, (fits or None)

def make_lognormal_gaussian_comparison_figure(mz, um_per_px=None, min_px=0,
                                               out_stem="width_lognorm_vs_gauss",
                                               formats=("svg",), dpi=300, title="",
                                               quantity="width"):
    """
    Dedicated comparison figure: for each state (up, down) the domain-width
    (or domain-period, quantity="period") sample is fit BOTH to a log-normal
    and to a Gaussian, and both PDFs are drawn over the same
    density-normalized histogram so their shapes can be compared directly.
    AIC/BIC (lower = better-supported model given equal complexity -- both
    are 2-parameter fits here) and the KS goodness-of-fit statistic for each
    are annotated on the panel and written to "<out_stem>_fit_comparison.json".

    Up and down get their own panel -- (a) up, (b) down -- rather than being
    pooled, same per-state convention used throughout this module (see
    qq_series_per_state, make_size_distribution_figure), since the two
    states' distributions can genuinely differ (e.g. under applied current).

    Same GUI toggle/export pairing as make_qq_plot_figure: writes
    "<out_stem>.<ext>" for each entry in `formats`, plus the comparison JSON.

    Returns (list_of_paths, comparisons_dict). comparisons_dict has
    "up"/"down" keys holding compare_lognormal_gaussian() output (or None
    if that state had < 5 samples).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mm = per_state_morphometry(mz, um_per_px=um_per_px, min_px=min_px)
    unit = "nm" if um_per_px else "px"
    w_scale = (um_per_px * 1000.0) if um_per_px else 1.0
    key = "widths_px" if quantity == "width" else "periods_px"
    widths = {"up": mm["up"][key] * w_scale,
              "down": mm["down"][key] * w_scale}

    comparisons = {}
    with plt.rc_context(_PUB_RC):
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
        for a, (name, col_fill, col_line, letter) in zip(
            ax, (("up", _COLOR_UP_FILL, _COLOR_UP_LINE, "a"),
                 ("down", _COLOR_DOWN_FILL, _COLOR_DOWN_LINE, "b"))):
            w = widths[name]
            w = w[w > 0]
            if w.size < 5:
                a.text(0.5, 0.5, "not enough samples\n(need >= 5)",
                       ha="center", va="center", transform=a.transAxes, fontsize=9)
                a.set_title(f"({letter}) {name}")
                comparisons[name] = None
                continue
            cmp = compare_lognormal_gaussian(w)
            comparisons[name] = cmp
            a.hist(w, bins=24, density=True, color=col_fill, alpha=0.6, label="data")
            if cmp["lognormal"]:
                a.plot(cmp["lognormal"]["pdf_x"], cmp["lognormal"]["pdf_y"],
                       "-", color=col_line, linewidth=1.4, label="log-normal")
            if cmp["gaussian"]:
                a.plot(cmp["gaussian"]["pdf_x"], cmp["gaussian"]["pdf_y"],
                       "--", color=col_line, linewidth=1.4, label="Gaussian")
            txt = []
            if cmp["lognormal"]:
                txt.append(f"log-normal: AIC={cmp['lognormal']['aic']:.1f}, "
                           f"KS p={cmp['lognormal']['ks_p']:.3f}")
            if cmp["gaussian"]:
                txt.append(f"Gaussian:   AIC={cmp['gaussian']['aic']:.1f}, "
                           f"KS p={cmp['gaussian']['ks_p']:.3f}")
            if cmp["preferred"]:
                txt.append(f"delta AIC={cmp['delta_aic']:+.1f} -> "
                           f"{cmp['preferred']} favored")
            a.text(0.02, 0.98, "\n".join(txt), transform=a.transAxes,
                   fontsize=7.5, va="top", ha="left",
                   bbox=dict(boxstyle="round", fc="white", ec="#999", alpha=0.9))
            a.set_xlabel(f"domain {quantity} ({unit})")
            a.set_ylabel("probability density")
            a.set_title(f"({letter}) {name}")
            a.legend(fontsize=8)

        if title:
            fig.suptitle(title, fontsize=12)
        plt.tight_layout(rect=[0, 0, 1, 0.94] if title else None)

        out_paths = []
        for ext in formats:
            path = f"{out_stem}.{ext}"
            fig.savefig(path, dpi=dpi, format=("tiff" if ext == "tiff" else ext))
            out_paths.append(path)
        plt.close(fig)

    if any(v is not None for v in comparisons.values()):
        out_paths.append(_save_fit_comparison(comparisons, f"{out_stem}_fit_comparison.json", unit))
    return out_paths, comparisons

def make_width_area_count_figure(mz, um_per_px=None, min_px=12, bins=24,
                                 out_stem="width_area_counts", formats=("svg", "tiff"),
                                 dpi=300, title=""):
    """
    Two histograms: number of domains vs. width, number of domains vs. area.
    (x = domain width / domain area, y = number of domains in that bin.)

    Writes one file per entry in `formats` (default both .svg and .tiff),
    named "<out_stem>.<ext>". A thin log-normal fit is overlaid on the WIDTH
    panel only (area isn't fit here) for up and down, with the fitted
    parameters written to "<out_stem>_lognormal_fit.json" (appended to the
    returned path list). Returns (list_of_paths, per_domain_width_area_dict).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = per_domain_width_area(mz, min_px=min_px, um_per_px=um_per_px)
    # width is reported in nm when calibrated (the natural scale for XMCD-PEEM
    # domains -- um values here would mostly read as 0.0xx); area stays in
    # um^2/px since that wasn't asked to change.
    unit_l = "nm" if um_per_px else "px"
    unit_a = "um^2" if um_per_px else "px"
    width_key = "width_nm" if um_per_px else "width"

    fits = {}
    with plt.rc_context(_PUB_RC):
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
        for name, col in (("up", _COLOR_UP_FILL), ("down", _COLOR_DOWN_FILL)):
            if d[name][width_key].size:
                ax[0].hist(d[name][width_key], bins=bins, color=col, alpha=0.85, label=name)
            if d[name]["area"].size:
                ax[1].hist(d[name]["area"], bins=bins, color=col, alpha=0.85, label=name)

        w_up, w_dn = d["up"][width_key], d["down"][width_key]
        if w_up.size or w_dn.size:
            all_w = np.concatenate([w for w in (w_up, w_dn) if w.size])
            edges = np.histogram_bin_edges(all_w, bins=bins)
            if w_up.size >= 5:
                fits["up"] = _overlay_lognormal(ax[0], w_up, edges, _COLOR_UP_LINE,
                                                "up log-normal fit")
            if w_dn.size >= 5:
                fits["down"] = _overlay_lognormal(ax[0], w_dn, edges, _COLOR_DOWN_LINE,
                                                   "down log-normal fit")

        ax[0].set_xlabel(f"domain width ({unit_l})")
        ax[0].set_ylabel("number of domains")
        ax[0].set_title("(a) domain count vs width")
        ax[0].legend(fontsize=8)

        ax[1].set_xlabel(f"domain area ({unit_a})")
        ax[1].set_ylabel("number of domains")
        ax[1].set_title("(b) domain count vs area")
        ax[1].legend(fontsize=8)

        if title:
            fig.suptitle(title, fontsize=12)
        plt.tight_layout(rect=[0, 0, 1, 0.92])

        out_paths = []
        for ext in formats:
            path = f"{out_stem}.{ext}"
            # .tiff needs a raster dpi; .svg is vector and ignores dpi at draw time
            # but we pass it anyway so text/marker sizing stays consistent.
            fig.savefig(path, dpi=dpi, format=("tiff" if ext == "tiff" else ext))
            out_paths.append(path)
        plt.close(fig)

    if fits:
        out_paths.append(_save_lognormal_fits(fits, f"{out_stem}_lognormal_fit.json", unit_l))
    return out_paths, d
