"""Width / period distribution panels of the preview and export.
Plotting layer."""
from ..colors import COLOR_UP, COLOR_DOWN
from ..core import compare_lognormal_gaussian, qq_series_per_state


# ----------------------------------------------------------------------
def _draw_gauss_compare(ax, mm, um, quantity="width"):
    """Draw the log-normal-vs-Gaussian comparison panel: for each state
    (up/down) with >= 5 samples of the chosen quantity ("width" or
    "period"), a density histogram plus both fitted PDFs (log-normal solid,
    Gaussian dashed), with delta-AIC annotated per state. Same per-state
    convention as the rest of this module -- up and down are fit and shown
    independently, never pooled."""
    ax.clear()
    key = "widths_px" if quantity == "width" else "periods_px"
    w_scale = (um * 1000.0) if um else 1.0   # nm when calibrated (see draw_distributions)
    any_series = False
    notes = []
    for name, col in (("up", COLOR_UP), ("down", COLOR_DOWN)):
        w = mm[name][key] * w_scale
        w = w[w > 0]
        if w.size < 5:
            continue
        any_series = True
        cmp = compare_lognormal_gaussian(w)
        ax.hist(w, bins=20, density=True, alpha=0.25, color=col)
        if cmp["lognormal"]:
            ax.plot(cmp["lognormal"]["pdf_x"], cmp["lognormal"]["pdf_y"],
                    "-", color=col, linewidth=1.2, label=f"{name} log-normal")
        if cmp["gaussian"]:
            ax.plot(cmp["gaussian"]["pdf_x"], cmp["gaussian"]["pdf_y"],
                    "--", color=col, linewidth=1.2, label=f"{name} Gaussian")
        if cmp["preferred"]:
            notes.append(f"{name}: dAIC={cmp['delta_aic']:+.1f} "
                         f"({cmp['preferred']} favored)")
    if any_series:
        ax.set_title(f"(e) log-normal vs. Gaussian ({quantity})", fontsize=9)
        ax.legend(fontsize=6)
        if notes:
            ax.text(0.02, 0.98, "\n".join(notes), transform=ax.transAxes,
                    fontsize=6.5, va="top", ha="left",
                    bbox=dict(boxstyle="round", fc="white", ec="#999", alpha=0.85))
    else:
        ax.text(0.5, 0.5, f"not enough {quantity}\nsamples (need >= 5/state)",
                ha="center", va="center", transform=ax.transAxes, fontsize=8)
        ax.set_title(f"(e) log-normal vs. Gaussian ({quantity})", fontsize=9)
    ax.set_xlabel(f"domain {quantity} ({'nm' if um else 'px'})")
    ax.set_ylabel("probability density")


def draw_distributions(ax, res):
    """Draw the within-polygon distributions onto a list of axes:
    [width, period, probability] or [..., Q-Q] or [..., Q-Q, log-normal-vs-Gaussian].
    Shared by the GUI preview and the file export.

    Panel (b) is the domain-PERIOD distribution (local skeleton-pixel period,
    see per_state_morphometry/_local_period_px), NOT domain area -- area is
    still available via res["counts"] (domain_counts_areas) and the
    exported <stem>_summary.txt's domain table, just no longer as a
    live-preview histogram here.

    The optional 4th axis, if present in `ax`, gets a log-normal Q-Q plot,
    and the optional 5th a log-normal-vs-Gaussian comparison -- for
    whichever quantity res["params"]["dist_quantity"] selects ("width" or
    "period", default "width"), only if res["params"]["qq_plot"] /
    ["gauss_compare"] is truthy; otherwise hidden (axis("off")). Export
    (run_export) always writes BOTH width and period versions of these to
    file regardless of this live-preview selector -- it only controls what
    the small in-GUI panels show.

    This lets the GUI always allocate 5 subplots and just toggle the last
    two on/off without rebuilding the canvas.
    """
    um = res["um"]
    # nm when calibrated -- matches the text readout and every exported figure
    # (previously these panels alone were in um, so the same width read ~0.32
    # here and ~316 everywhere else).
    unit = "nm" if um else "px"
    w_scale = (um * 1000.0) if um else 1.0
    mm = res["morphometry"]; pl = res["pdf"]
    dist_quantity = res.get("params", {}).get("dist_quantity", "width")
    for a in ax:
        a.clear()
    if mm is None:      # widths not measured (large image, live preview): only the probabilities
        for a, t in ((ax[0], "(a) width distribution"), (ax[1], "(b) period distribution")):
            a.text(0.5, 0.5, "not measured\n(tick 'Measure domain\nwidths', step 5)", ha="center", va="center",
                   transform=a.transAxes, fontsize=8)
            a.set_title(t, fontsize=9)
            a.set_xticks([]); a.set_yticks([])
        for a in ax[3:]:
            a.axis("off")

    for st, col, lab in (() if mm is None else ((mm["up"], COLOR_UP, "up"), (mm["down"], COLOR_DOWN, "down"))):
        w = st["widths_px"] * w_scale
        if w.size:
            ax[0].hist(w, bins=24, alpha=0.55, color=col, label=lab)
    if mm is not None:
        ax[0].set_xlabel(f"domain width ({unit})"); ax[0].set_ylabel("count")
        per = (mm.get("domain_period_nm") if um else mm["domain_period"])
        ax[0].set_title("(a) width distribution"
                        + (f"\nperiod ~ {per:.1f} {unit}" if per == per else ""), fontsize=9)
        ax[0].legend(fontsize=7)

        for st, col, lab in ((mm["up"], COLOR_UP, "up"), (mm["down"], COLOR_DOWN, "down")):
            p = st["periods_px"] * w_scale
            if p.size:
                ax[1].hist(p, bins=24, alpha=0.55, color=col, label=lab)
        ax[1].set_xlabel(f"domain period ({unit})"); ax[1].set_ylabel("count")
        ax[1].set_title("(b) period distribution", fontsize=9); ax[1].legend(fontsize=7)

    ax[2].bar(["P(up)", "P(0)", "P(dn)"],
              [pl["p_up"], pl["p_zero"], pl["p_down"]],
              color=[COLOR_UP, "#bbbbbb", COLOR_DOWN])
    ax[2].set_ylim(0, 1); ax[2].set_ylabel("probability")
    ax[2].set_title(f"(c) +/- probability\nnet <m_z>={pl['net_m']:+.3f}", fontsize=9)
    for i, v in enumerate([pl["p_up"], pl["p_zero"], pl["p_down"]]):
        ax[2].text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)

    # (d) optional Q-Q plot: up and down samples of the selected quantity,
    # EACH vs. their own fitted log-normal (not pooled). Only drawn if the
    # caller asked for it (GUI checkbox / qq_plot param) and there's a 4th
    # axis to draw into.
    if len(ax) > 3 and mm is not None:
        want_qq = bool(res.get("params", {}).get("qq_plot", False))
        if want_qq:
            ax[3].axis("on")
            qq = qq_series_per_state(mm, scale=w_scale, quantity=dist_quantity)
            any_series = False
            for name, col in (("up", COLOR_UP), ("down", COLOR_DOWN)):
                series = qq[name]
                if series is None:
                    continue
                any_series = True
                ax[3].plot(series["osm"], series["osr"], "o", color=col,
                           markersize=3, label=f"{name} (n={series['fit']['n']})")
                fit_line = series["slope"] * series["osm"] + series["intercept"]
                ax[3].plot(series["osm"], fit_line, "-", color=col, linewidth=1)
            if any_series:
                ax[3].set_title(f"(d) Q-Q: {dist_quantity} vs. log-normal\n(fit per state)",
                                fontsize=9)
                ax[3].legend(fontsize=7)
            else:
                ax[3].text(0.5, 0.5, f"not enough {dist_quantity}\nsamples (need >= 5/state)",
                           ha="center", va="center", transform=ax[3].transAxes,
                           fontsize=8)
                ax[3].set_title("(d) Q-Q plot", fontsize=9)
        else:
            ax[3].axis("off")

    # (e) optional log-normal-vs-Gaussian comparison: same "always allocate,
    # toggle visibility" pattern as the Q-Q panel above.
    if len(ax) > 4 and mm is not None:
        want_gc = bool(res.get("params", {}).get("gauss_compare", False))
        if want_gc:
            ax[4].axis("on")
            _draw_gauss_compare(ax[4], mm, um, quantity=dist_quantity)
        else:
            ax[4].axis("off")


def make_distributions_figure(res, out_png):
    """Within-polygon distributions: domain WIDTH, domain AREA, up/down P,
    and (if toggled on via res["params"]) a log-normal Q-Q plot and/or a
    log-normal-vs-Gaussian comparison panel. Always allocates 5 axes, same
    as the GUI's ax_d, and hides whichever of the last two aren't toggled
    on -- keeps this one shared code path with the live preview instead of
    reproducing the ax[3]/ax[4] index mapping here."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 5, figsize=(23, 4.4))
    draw_distributions(list(ax), res)
    plt.tight_layout()
    plt.savefig(out_png, dpi=130); plt.close(fig)
    return out_png
