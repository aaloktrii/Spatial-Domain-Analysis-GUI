"""Batch flake analysis of one image -> statistics, tables, figures (analyze_one and helpers; used by
sda-analyze and the GUI export). Qt-free service layer: calls core/ for numbers and plotting/ for figures."""
import contextlib
import io

from ..notation import column, relabel, using
import os
import re
import glob


from ..defaults import DEFAULTS
from ..core import (
    analyze_xmcd,
    three_state_field,
    despeckle_mz,
    pdf_three_state,
    number_density_local,
    per_state_domain_stats,
    fit_lognormal,
    magnetization_statistics,
    area_averaged_mz,
    net_mz_raw,
    summarize_domains,
    export_fit_summary,
    bootstrap_fraction_cis,
    fraction_threshold_sweep,
    correlation_length_state,
    domain_widths,
    per_state_morphometry,
)
from ..core.walls import subpixel_wall_density
from ..plotting import (
    make_distribution_figure, make_magnetization_figure, make_domain_wall_figure, make_xi_map_figure,
    make_width_comparison_figure, make_morphometry_figure,
)
from ..plotting.flake_figures import flake_stats_figure, _detailed_pipeline_figure


# columns written to the combined summary table
SUMMARY_COLS = [
    "image", "current_mA", "temp_K", "calibrated", "um_per_px",
    "image_w_px", "image_h_px",
    "flake_area_px", "flake_area_um2",
    "net_m", "frac_up", "frac_down",
    "p_up", "p_zero", "p_down",
    "n_domains_total", "n_domains_up", "n_domains_down",
    "mean_domain_area_px", "domain_wall_density_per_px",
    "up_area_mean", "up_area_sd", "up_area_median", "up_ecc", "up_circ",
    "down_area_mean", "down_area_sd", "down_area_median", "down_ecc", "down_circ",
    "zero_area_mean", "zero_area_sd", "zero_area_median",
    "area_units",
    "area_avg_mz", "area_avg_mz_sd", "area_avg_mz_N",
    "net_mz_raw", "net_mz_raw_sd", "net_mz_raw_N",
    "mz_mean", "mz_ci_lo", "mz_ci_hi", "mz_sem_corrected",
    "mz_corr_len_px", "mz_N_eff", "mz_p_value",
    "up_width_median", "down_width_median", "domain_period",
    "up_width_median_nm", "down_width_median_nm", "domain_period_nm",
    "up_skeleton_len", "down_skeleton_len",
    # per-skeleton-pixel WIDTH/PERIOD log-normal fits (the same ones fits.txt
    # and summary.txt report). up/down width_fit_median = e^mu = the
    # manuscript's w~. Replaces the old Otsu per-domain-diameter lognorm_*
    # columns, which were a different (unreliable-for-labyrinth) quantity.
    "up_width_fit_median_nm", "down_width_fit_median_nm",
    "up_width_fit_mode_nm", "down_width_fit_mode_nm",
    "width_fit_sigma_ln", "width_fit_KS",
    "period_fit_median_nm", "period_fit_mode_nm",
]


def parse_filename(stem):
    """Best-effort pull of current (mA) and temperature (K) from a filename."""
    cur = re.search(r"([+-]?\d+)\s*mA", stem, re.IGNORECASE)
    tmp = re.search(r"(\d+)\s*K", stem, re.IGNORECASE)
    return (int(cur.group(1)) if cur else None,
            int(tmp.group(1)) if tmp else None)


def analyze_one(path, args, outroot):
    """Analyse one image and write its statistics, tables and figures. State names in every output are +ve / -ve
    and the order parameter that of args.dichroism (notation.py)."""
    buf = io.StringIO()
    with using(getattr(args, "dichroism", "XMCD")), contextlib.redirect_stdout(buf):
        row = _analyze_one(path, args, outroot)
    print(relabel(buf.getvalue(), getattr(args, "dichroism", "XMCD")), end="")
    return row


def _analyze_one(path, args, outroot):
    stem = os.path.splitext(os.path.basename(path))[0]
    # default: save results NEXT TO the source image; --outdir overrides
    base = outroot if outroot else os.path.dirname(os.path.abspath(path))
    outdir = os.path.join(base, f"{stem}_analysis")
    # Tiered output: statistics.txt at the top; Priority-2 publication images
    # in figures/; opt-in Priority-3 diagnostics (12-panel, comparisons, fits)
    # in diagnostics/, only written when --detailed / the GUI checkbox is on.
    detailed = getattr(args, "detailed", False)
    figdir = os.path.join(outdir, "figures")
    diagdir = os.path.join(outdir, "diagnostics")
    os.makedirs(figdir, exist_ok=True)
    if detailed:
        os.makedirs(diagdir, exist_ok=True)
    p = lambda s: os.path.join(outdir, f"{stem}_{s}")           # top level (statistics/run_info)
    # publication figures honor the chosen vector/raster format (png/eps/pdf);
    # `pf` takes a bare name and appends the extension. Diagnostics stay png.
    fext = getattr(args, "fig_format", "png")
    pf = lambda s: os.path.join(figdir, f"{stem}_{s}.{fext}")   # figures/ (formatted)
    pd = lambda s: os.path.join(diagdir, f"{stem}_{s}")         # diagnostics/
    um = args.um_per_px

    # per-image calibration from field of view (handles FoV varying per image):
    #   um_per_px = FoV_width / image_width_px  (computed for THIS image)
    if args.fov:
        from ..core.igor_import import analysis_shape
        H_px, W_px = analysis_shape(path, getattr(args, "input_mode", "grayscale"))   # the analysed grid
        um_w = args.fov[0] / W_px
        um_h = args.fov[1] / H_px
        um = um_w
        if abs(um_w - um_h) / max(um_w, um_h) > 0.05:
            print(f"  WARN [{stem}]: non-square pixels "
                  f"({um_w:.4f} vs {um_h:.4f} um/px from W/H) -- check FoV/aspect")

    # gray-control options
    roi_box = tuple(args.roi) if args.roi else None
    bg_box = tuple(args.bg) if args.bg else None
    clip = tuple(args.clip) if args.clip else (0.5, 99.5)
    # classification mode:
    #   --m-threshold t  -> gray band = [-t, +t] on the normalized asymmetry
    #   --states 2       -> no gray (z_neutral = 0)
    #   else             -> z-score band of half-width --z-neutral
    z_neutral = 0.0 if args.states == 2 else args.z_neutral

    # core analysis (metrics are already flake-restricted)
    conn = getattr(args, "connectivity", DEFAULTS["connectivity"])
    tv_weight = getattr(args, "tv_weight", DEFAULTS["tv_weight"])
    # min domain size: honored from args (GUI slider / --min-px) instead of
    # being hardcoded, so the export uses the same value the GUI shows and the
    # despeckle / width / count statistics are all consistent with it.
    min_px = getattr(args, "min_px", DEFAULTS["min_px"])
    # flatten_sigma must be honored here too: the GUI export (run_export ->
    # build_args) sets it per image (e.g. ~150 for zoomed large-domain frames),
    # and the live preview already uses it -- without threading it through,
    # the export silently fell back to the 25px default and hollowed out large
    # domains, disagreeing with the preview. Default 25 matches analyze_xmcd.
    flatten_sigma = getattr(args, "flatten_sigma", 25.0)
    src, input_mode = path, getattr(args, "input_mode", "grayscale")
    # Igor export decoded; a large picture drawn as pixel blocks read on its own pixel grid (any input type)
    from ..core.igor_import import source_for
    src, input_mode = source_for(path, input_mode, getattr(args, "map_mode", "auto"), getattr(args, "invert", False))
    metrics, layers = analyze_xmcd(src, um_per_px=um, clahe=args.clahe,
                                   roi_box=roi_box, bg_box=bg_box, clip=clip,
                                   connectivity=conn, tv_weight=tv_weight,
                                   flatten_sigma=flatten_sigma,
                                   denoise_method=getattr(args, "denoise_method", "tv"),
                                   input_mode=input_mode,
                                   sym_centre=getattr(args, "sym_centre", "0"),
                                   map_mode=getattr(args, "map_mode", "rb"))
    mz, roi = three_state_field(layers, z_neutral=z_neutral,
                                m_threshold=args.m_threshold)
    # revert sub-12px +/-1 speckle to neutral BEFORE any statistic, so every
    # number here matches the GUI path (the GUI preview also despeckles at min_px=12)
    # and the min-domain-size filter is applied consistently across the whole
    # report -- pdf, morphometry, and the figures all see one cleaned field.
    # (12 = the same min_px used by summarize_domains/per_state_morphometry below.)
    mz = despeckle_mz(mz, min_px=min_px)
    pl = pdf_three_state(mz, region=roi)

    # area-averaged <m_z> from the continuous asymmetry field, BEFORE
    # thresholding -- cheap (no bootstrap), so computed unconditionally even
    # when --no-rigorous skips the full magnetization_statistics() pipeline.
    aa = area_averaged_mz(layers)
    # raw signal averaged over only the classified (non-gray-band) pixels --
    # see net_mz_raw's docstring for why this sits between the discrete and
    # continuous definitions above.
    hy = net_mz_raw(mz, layers)
    # threshold robustness of the (inherently threshold-defined) fractions:
    # how f+/f-/D and the ratio move as t is swept. Cheap, always computed.
    # A flat quantity is not a threshold artifact; a drifting one's range is
    # its systematic uncertainty. (A threshold-FREE mixture decomposition was
    # tried but is unreliable for bimodal stripe data -- no distinct central
    # peak -- so the sweep is the honest tool here.)
    fsweep = fraction_threshold_sweep(layers)
    nd = number_density_local(mz, tile=args.tile, region=roi)   # FLAKE-only

    from xmcd_toolkit import pubfigures, asymmetry_field, domain_boundary
    unit = "um" if metrics["calibrated"] else "px"
    sizes = layers["eq_diam_px"] * (um or 1.0)
    fit = fit_lognormal(sizes)   # equiv-diam fit for summary_table columns

    # --- statistics (Priority 1): computed ALWAYS (numbers feed statistics.txt
    #     + summary_table.csv). Only the corresponding FIGURES are gated. ---
    mm = per_state_morphometry(mz, um_per_px=um, min_px=min_px)
    width_method = getattr(args, "width_method", "perimeter_area")
    # correlation length (FFT) once, in px -> reused by the width comparison
    # (correlation_length method) AND the per-state xi line below.
    xu_px = correlation_length_state(mz, roi, 1, um_per_px=None)
    xd_px = correlation_length_state(mz, roi, -1, um_per_px=None)
    widths_cmp = domain_widths(mz, roi, um_per_px=um, min_px=min_px, mm=mm,
                               xi_px=(xu_px, xd_px))
    domain_summary = summarize_domains(mz, roi, um_per_px=um, min_px=min_px,
                                       connectivity=conn, layers=layers, mm=mm)
    mzs = None
    fracs_ci = None
    if not args.no_rigorous:
        mzs = magnetization_statistics(layers, n_boot=args.n_boot)
        if args.m_threshold is not None:
            fracs_ci = bootstrap_fraction_cis(layers, m_threshold=args.m_threshold,
                                              n_boot=args.n_boot)

    # --- Priority-2 publication figures -> figures/ (always) ---
    img0 = layers["img"]
    m_field, _ = asymmetry_field(layers)
    boundary = domain_boundary(m_field, roi,
                               method=getattr(args, "boundary_method", "zero_contour"),
                               m_threshold=(args.m_threshold or 0.05))
    pubfigures.up_down_states(img0, mz, roi, pf(column("up_down_states")),
                              title=f"{stem} -- up/down states")
    pubfigures.asymmetry_map(m_field, roi, pf("asymmetry"),
                             title=f"{stem} -- asymmetry m")
    pubfigures.width_map(img0, mz, mm, um, pf("width_map"),
                         title=f"{stem} -- local domain width")
    pubfigures.domain_boundary(img0, boundary, pf("domain_boundary"),
                               title=f"{stem} -- domain boundary")
    pubfigures.net_mz_map(nd, pf("net_mz"), title=f"{stem} -- local net m_z")
    pubfigures.key_montage(img0, m_field, mz, roi, mm, boundary, um,
                           pf("key_montage"), title=stem)
    # D and xi maps compute stats used below; their figures are Priority-2.
    _, dwd = make_domain_wall_figure(img0, roi, nd, pf("domain_wall_D"),
                                     title=f"{stem} -- domain-wall fraction D")
    _, xistats = make_xi_map_figure(layers, pf("xi_map"), um_per_px=um,
                                    title=f"{stem} -- correlation length xi(x,y)")

    # --- Priority-3 diagnostics -> diagnostics/ (only when --detailed) ---
    if detailed:
        make_distribution_figure(sizes, pd(f"size_distribution.{fext}"), unit=unit, title=stem)
        flake_stats_figure(img0, mz, roi, nd, pd(f"flake_stats.{fext}"),
                           title=f"{stem} (flake-area statistics)")
        if mzs:
            make_magnetization_figure(mzs, pd(f"magnetization.{fext}"),
                                      title=f"{stem} -- magnetization statistics")
        make_morphometry_figure(mz, roi, um, pd(f"morphometry.{fext}"), mm=mm,
                                title=f"{stem} -- domain width / skeleton")
        make_width_comparison_figure(mz, roi, um, pd(f"width_comparison.{fext}"),
                                     min_px=min_px, mm=mm, title=stem)
        export_fit_summary(mm, um_per_px=um, out_path=pd("fits.txt"),
                           title=stem, min_px=min_px)
        _detailed_pipeline_figure(layers, mz, mm, m_field, um,
                                  pd(f"pipeline_steps.{fext}"), stem, args,
                                  boundary=boundary)
        if args.tiff:
            from xmcd_toolkit import export_panels_tiff
            export_panels_tiff(img0, mz, roi, um, diagdir, stem, dpi=args.dpi)

    # per-state domain area + shape (mean / SD / median, eccentricity, etc.)
    ss = per_state_domain_stats(mz, um_per_px=um, connectivity=conn, roi=roi)
    au = ss["up"]["units"]   # "px" or "um^2"
    exact_wall_density = subpixel_wall_density(mz, roi, um)
    # scalar flake metrics table
    cur, tmp = parse_filename(stem)
    # helpers for the per-skeleton-pixel width/period fit columns: fit values
    # (from domain_summary) are in px; _wfnm scales a length to nm when
    # calibrated (None otherwise), _wfval passes a scale-invariant value
    # (sigma_ln, KS) through. Both tolerate a missing/failed fit (None).
    _sfit_nm = (um * 1000.0) if um else None
    _wfnm = lambda fitd, k: (round(fitd[k] * _sfit_nm, 1)
                             if (fitd and _sfit_nm) else None)
    _wfval = lambda fitd, k: (round(fitd[k], 3) if fitd else None)
    flake = {
        "image": os.path.basename(path), "current_mA": cur, "temp_K": tmp,
        "calibrated": metrics["calibrated"], "um_per_px": um,
        "image_w_px": int(layers["img"].shape[1]),
        "image_h_px": int(layers["img"].shape[0]),
        "flake_area_px": metrics["flake_area_px"],
        "flake_area_um2": metrics["flake_area_um2"],
        "net_m": round(pl["net_m"], 5),
        "frac_up": round(metrics["frac_up"], 4),
        "frac_down": round(metrics["frac_down"], 4),
        "p_up": round(pl["p_up"], 4), "p_zero": round(pl["p_zero"], 4),
        "p_down": round(pl["p_down"], 4),
        "n_domains_total": metrics["n_domains_total"],
        "n_domains_up": metrics["n_domains_up"],
        "n_domains_down": metrics["n_domains_down"],
        "mean_domain_area_px": round(metrics["mean_domain_area_px"], 2),
        "domain_wall_density_per_px": round(exact_wall_density, 5),
        "up_area_mean": round(ss["up"]["area_mean"], 2),
        "up_area_sd": round(ss["up"]["area_sd"], 2),
        "up_area_median": round(ss["up"]["area_median"], 2),
        "up_ecc": round(ss["up"]["ecc_mean"], 3),
        "up_circ": round(ss["up"]["circularity_mean"], 3),
        "down_area_mean": round(ss["down"]["area_mean"], 2),
        "down_area_sd": round(ss["down"]["area_sd"], 2),
        "down_area_median": round(ss["down"]["area_median"], 2),
        "down_ecc": round(ss["down"]["ecc_mean"], 3),
        "down_circ": round(ss["down"]["circularity_mean"], 3),
        "zero_area_mean": round(ss["zero"]["area_mean"], 2),
        "zero_area_sd": round(ss["zero"]["area_sd"], 2),
        "zero_area_median": round(ss["zero"]["area_median"], 2),
        "area_units": au,
        "area_avg_mz": round(aa["mean_mz"], 5),
        "area_avg_mz_sd": round(aa["sd"], 5) if aa["sd"] == aa["sd"] else None,
        "area_avg_mz_N": aa["N_pixels"],
        "net_mz_raw": round(hy["mean_mz"], 5) if hy["mean_mz"] == hy["mean_mz"] else None,
        "net_mz_raw_sd": round(hy["sd"], 5) if hy["sd"] == hy["sd"] else None,
        "net_mz_raw_N": hy["N_pixels"],
        "mz_mean": round(mzs["mean_mz"], 5) if mzs else None,
        "mz_ci_lo": round(mzs["ci95_bootstrap"][0], 5) if mzs else None,
        "mz_ci_hi": round(mzs["ci95_bootstrap"][1], 5) if mzs else None,
        "mz_sem_corrected": round(mzs["sem_corrected"], 5) if mzs else None,
        "mz_corr_len_px": round(mzs["corr_length_px"], 2) if mzs else None,
        "mz_N_eff": round(mzs["N_eff"], 1) if mzs else None,
        "mz_p_value": float(f"{mzs['p_value_vs_zero']:.3g}") if mzs else None,
        "up_width_median": round(mm["up"]["width_median"], 3),
        "down_width_median": round(mm["down"]["width_median"], 3),
        "domain_period": round(mm["domain_period"], 3) if mm["domain_period"] == mm["domain_period"] else None,
        # nm is the natural scale for these domains; present only when
        # calibrated (um_per_px given), since px has no physical size.
        "up_width_median_nm": round(mm["up"]["width_median_nm"], 1) if um else None,
        "down_width_median_nm": round(mm["down"]["width_median_nm"], 1) if um else None,
        "domain_period_nm": (round(mm["domain_period_nm"], 1)
                             if um and mm["domain_period"] == mm["domain_period"] else None),
        "up_skeleton_len": round(mm["up"].get("skeleton_len", mm["up"]["skeleton_len_px"]), 2),
        "down_skeleton_len": round(mm["down"].get("skeleton_len", mm["down"]["skeleton_len_px"]), 2),
        # per-skeleton-pixel width/period log-normal fits (from domain_summary,
        # i.e. the same mm the figures + fits.txt use). Fit values are in px;
        # scale to nm when calibrated. median = e^mu = manuscript w~.
        "up_width_fit_median_nm": _wfnm(domain_summary["up"]["width_fit"], "median"),
        "down_width_fit_median_nm": _wfnm(domain_summary["down"]["width_fit"], "median"),
        "up_width_fit_mode_nm": _wfnm(domain_summary["up"]["width_fit"], "mode"),
        "down_width_fit_mode_nm": _wfnm(domain_summary["down"]["width_fit"], "mode"),
        "width_fit_sigma_ln": _wfval(domain_summary["total"]["width_fit"], "sigma_log"),
        "width_fit_KS": _wfval(domain_summary["total"]["width_fit"], "ks_stat"),
        "period_fit_median_nm": _wfnm(domain_summary["total"]["period_fit"], "median"),
        "period_fit_mode_nm": _wfnm(domain_summary["total"]["period_fit"], "mode"),
    }
    # Everything below folds into ONE physics/statistics report -- no more
    # separate flake_metrics.csv, flake_pixels.csv, number_density.csv, or
    # domain_table.csv (those were raw per-pixel/per-domain dumps, not
    # summary statistics, and were cluttering the per-image output folder).
    # Figures (size_distribution.png, flake_stats.png, magnetization.png,
    # morphometry.png[, TIFFs]) are unaffected -- still written above.
    # `flake` (the SUMMARY_COLS dict) still feeds the run-level
    # summary_table.csv only, via this function's return value.

    s_nm = um * 1000.0 if um else None

    def _fit_line(label, fit):
        """One line of full log-normal fit parameters (median, mode,
        sigma_log, KS goodness-of-fit) for a per-skeleton-pixel width or
        period sample -- NOT the per-domain equivalent-diameter fit
        (unreliable for a fused labyrinth maze; see METHODS.md Sec 11/14).
        median/mode are scaled to nm when calibrated; sigma_log and the KS
        statistic/p-value are scale-invariant (same in px or nm)."""
        if not fit:
            return f"        {label}: not enough samples (need >= 5)"
        sc = s_nm if s_nm else 1.0
        return (f"        {label}: n={fit['n']}  median={fit['median']*sc:.2f}  "
                f"mode={fit['mode']*sc:.2f}  sigma_ln={fit['sigma_log']:.3f}  "
                f"KS={fit['ks_stat']:.3f} (p={fit['ks_p']:.2g})")

    def _domain_row(name):
        s = domain_summary[name]
        area = s["area_um2"] if um else s["area_px"]
        area_unit = "um^2" if um else "px"
        w = s["width_median_nm"] if um else s["width_median"]
        row = (f"  {name:<5} n={s['n_domains']:<4} area={area:.1f} {area_unit}  "
               f"width_median={w:.2f}  mz={s['mz']:+.4f}")
        return "\n".join([row,
                          _fit_line("width  fit", s["width_fit"]),
                          _fit_line("period fit", s["period_fit"])])

    lines = [
        f"Image: {os.path.basename(path)}",
        f"Parsed: current={cur} mA, temp={tmp} K" if (cur or tmp) else "Parsed: (no current/temp in name)",
        f"Calibration: {f'{um:.4f} um/px' if um else 'uncalibrated (px)'}",
        f"Flake area: {metrics['flake_area_px']} px"
        + (f" ({metrics['flake_area_um2']:.2f} um^2)" if metrics["calibrated"] else ""),
        "",
        "--- Magnetization ---",
        f"Discrete net <m_z> (thresholded up/down area): {pl['net_m']:+.4f}",
        (f"Net <m_z>, raw signal outside gray band (N={hy['N_pixels']}, "
         f"{hy['N_excluded']} px excluded): {hy['mean_mz']:+.4f} +/- {hy['sd']:.4f}"
         if hy["mean_mz"] == hy["mean_mz"] else
         "Net <m_z>, raw signal outside gray band: unavailable (no pixels cleared threshold)"),
        f"Continuous area-avg <m_z> (before thresholding, N={aa['N_pixels']}): "
        f"{aa['mean_mz']:+.4f} +/- {aa['sd']:.4f} (pixel sd)",
        f"Flake PDF: p(+1)={pl['p_up']:.3f}  p(0)={pl['p_zero']:.3f}  p(-1)={pl['p_down']:.3f}",
    ]
    if mzs:
        lines += [
            f"Rigorous <m_z> = {mzs['mean_mz']:+.4f}  SE corrected={mzs['sem_corrected']:.5f} "
            f"(naive={mzs['sem_naive']:.5f}, x{mzs['sem_corrected']/max(mzs['sem_naive'],1e-9):.1f})",
            f"  correlation length xi={mzs['corr_length_px']:.1f} px, "
            f"N_eff={mzs['N_eff']:.0f} (vs {mzs['N_pixels']} px)",
            f"  95% CI analytic  = [{mzs['ci95_analytic'][0]:+.4f}, {mzs['ci95_analytic'][1]:+.4f}]",
            f"  95% CI bootstrap = [{mzs['ci95_bootstrap'][0]:+.4f}, {mzs['ci95_bootstrap'][1]:+.4f}]",
            f"  H0 <m_z>=0: z={mzs['z_vs_zero']:+.2f}, p={mzs['p_value_vs_zero']:.3g}  -> "
            f"{'net-magnetized' if mzs['p_value_vs_zero'] < 0.05 else 'consistent with demagnetized'}",
        ]

    # --- State fractions at the working threshold, with robustness ----------
    # (the threshold-vs-continuous <m_z> comparison is already in the
    # Magnetization block above: discrete net vs continuous area-avg.)
    # f+ + f- + D = 1. These ARE threshold-defined (a wall is the |m|<=t band),
    # so robustness is shown by the sweep range over t, not a "continuous"
    # value. Bootstrap CIs (rigorous) are correlation-corrected.
    r_thr = (pl["p_up"] / pl["p_down"]) if pl["p_down"] > 0 else float("nan")
    def _ci(key):
        if not fracs_ci:
            return ""
        lo, hi = fracs_ci[key]
        return f"   95%CI[{lo:.3f}, {hi:.3f}]"
    def _rng(key):
        a = fsweep[key]
        return f"[{a.min():.3f}, {a.max():.3f}]"
    lines += [
        "",
        f"--- State fractions at m-threshold={args.m_threshold} (f+ + f- + D = 1) ---",
        f"  f+ (up)   = {pl['p_up']:.3f}{_ci('f_up')}",
        f"  f- (down) = {pl['p_down']:.3f}{_ci('f_down')}",
        f"  D  (wall) = {pl['p_zero']:.3f}{_ci('f_dw')}",
        f"  f+/f-     = {r_thr:.3f}{_ci('ratio')}",
        f"  threshold robustness (t swept 0.02-0.30):  f+/f- in {_rng('ratio')}, "
        f"D in {_rng('f_dw')}, <m_z>_disc in {_rng('net_m')}",
        "    (flat over t => not a threshold artifact; the range is the systematic)",
        f"  local D distribution (tile={args.tile}px): mean={dwd['mean']:.3f} "
        f"median={dwd['median']:.3f} SD={dwd['std']:.3f} "
        f"range[{dwd['min']:.3f}, {dwd['max']:.3f}] over {dwd['n_tiles']} tiles",
    ]

    period = domain_summary["total"]["period_nm"] if um else domain_summary["total"]["period_px"]
    lines += [
        "",
        "--- Domains (up / down / total) ---",
        f"(width/period in {'nm' if um else 'px'}; mz = area fraction for up/down, net <m_z> for total)",
        _domain_row("up"),
        _domain_row("down"),
        _domain_row("total"),
        (f"  period ~ {period:.2f} {'nm' if um else 'px'}" if (period is not None and period == period)
         else "  period unavailable"),
        # per-state correlation length = up/down domain SIZE, robust to the
        # tv_weight / m-threshold choice (autocorrelation; no skeleton). Equal
        # up=down => balanced; unequal => the net-magnetized (majority) state
        # has the larger domains.
        (lambda xu, xd: f"  correlation length xi (domain size, robust): "
                        f"up={xu:.1f} down={xd:.1f} {'nm' if um else 'px'}")(
            xu_px * (um * 1000.0 if um else 1.0),
            xd_px * (um * 1000.0 if um else 1.0)),
        f"  local xi map (window={xistats['window_px']}px, n={xistats['n_windows']}): "
        f"median={xistats['median']:.1f} SD={xistats['std']:.1f} "
        f"range[{xistats['min']:.1f}, {xistats['max']:.1f}] {xistats['unit']}  "
        f"(see figures/..._xi_map.png)",
        "",
        f"--- Domain WIDTH by method ({'nm' if um else 'px'}; * = selected primary) ---",
        *[f"  {'*' if k == width_method else ' '} {k:18} up={v['up']:.1f} down={v['down']:.1f}"
          for k, v in widths_cmp.items()],
        "  (2A/P & skeleton-mean = direct width and usually agree; skeleton-median "
        "reads ~30% low; xi is a domain SIZE ~1.5x width."
        + (" see diagnostics/..._width_comparison.png)" if detailed else
           " run with --detailed for the side-by-side width_comparison figure)"),
        f"Domain-wall density: {metrics['domain_wall_density_per_px']:.4f} /px",
        "",
        f"Area per domain ({au}), shape (mean):",
        f"  +1 up:   mean {ss['up']['area_mean']:.1f} +/- {ss['up']['area_sd']:.1f}, "
        f"median {ss['up']['area_median']:.1f}  (ecc {ss['up']['ecc_mean']:.2f}, "
        f"solidity {ss['up']['solidity_mean']:.2f}, circ {ss['up']['circularity_mean']:.2f})",
        f"   0 gray: mean {ss['zero']['area_mean']:.1f} +/- {ss['zero']['area_sd']:.1f}, "
        f"median {ss['zero']['area_median']:.1f}",
        f"  -1 down: mean {ss['down']['area_mean']:.1f} +/- {ss['down']['area_sd']:.1f}, "
        f"median {ss['down']['area_median']:.1f}  (ecc {ss['down']['ecc_mean']:.2f}, "
        f"solidity {ss['down']['solidity_mean']:.2f}, circ {ss['down']['circularity_mean']:.2f})",
        "  NOTE: for labyrinth patterns mean >> median (one merged maze); "
        "report median + area fractions, not mean domain area.",
    ]
    with open(p("statistics.txt"), "w") as f:
        f.write(relabel("\n".join(lines)) + "\n")

    print(f"[{stem}] <m_z>={pl['net_m']:+.3f}  "
          f"p(+/0/-)={pl['p_up']:.2f}/{pl['p_zero']:.2f}/{pl['p_down']:.2f}  "
          f"domains={metrics['n_domains_total']}"
          + (f"  I={cur}mA" if cur is not None else ""))
    return flake


def collect_targets(paths, pattern, recursive):
    out = []
    for path in paths:
        if os.path.isfile(path):
            out.append(path)
        elif os.path.isdir(path):
            if recursive:
                for root, _, _ in os.walk(path):
                    out += sorted(glob.glob(os.path.join(root, pattern)))
            else:
                out += sorted(glob.glob(os.path.join(path, pattern)))
        else:
            print(f"  skip (not found): {path}")
    # de-dup, keep order
    seen, uniq = set(), []
    for t in out:
        if t not in seen:
            seen.add(t); uniq.append(t)
    return uniq
