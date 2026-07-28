
"""
analyze.py -- flake-area statistical analysis of XMCD-PEEM images.

           one row per image -> plot stats vs current
"""
import os
import re
import sys
import csv
import glob
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage import measure
from scipy.ndimage import gaussian_filter

from xmcd_toolkit import (
    analyze_xmcd, three_state_field, despeckle_mz, pdf_three_state,
    number_density_local,
    per_state_domain_stats,
    make_distribution_figure, fit_lognormal,
    magnetization_statistics, make_magnetization_figure,
    area_averaged_mz, net_mz_raw, summarize_domains, export_fit_summary,
    make_morphometry_figure, per_state_morphometry,
    COLOR_UP, COLOR_DOWN, CMAP3_HEX,
)

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
    "lognorm_median", "lognorm_sigma_ln", "lognorm_KS",
]


def parse_filename(stem):
    """Best-effort pull of current (mA) and temperature (K) from a filename."""
    cur = re.search(r"([+-]?\d+)\s*mA", stem, re.IGNORECASE)
    tmp = re.search(r"(\d+)\s*K", stem, re.IGNORECASE)
    return (int(cur.group(1)) if cur else None,
            int(tmp.group(1)) if tmp else None)


def flake_stats_figure(img, mz, roi, nd, out_png, title=""):
    masked = np.where(roi, mz, np.nan)
    pl = pdf_three_state(mz, region=roi)

    fig = plt.figure(figsize=(15, 4.4))
    cmap3 = matplotlib.colors.ListedColormap(CMAP3_HEX)

    ax = fig.add_subplot(1, 3, 1)
    ax.imshow(img, cmap="gray")
    ax.imshow(masked, cmap=cmap3, vmin=-1, vmax=1, alpha=0.85)
    ax.set_title("(a) flake three-state m_z"); ax.axis("off")

    ax = fig.add_subplot(1, 3, 2)
    ax.bar(["+1\nup", "0\nnone", "-1\ndown"],
           [pl["p_up"], pl["p_zero"], pl["p_down"]],
           color=[COLOR_UP, "#bbbbbb", COLOR_DOWN], edgecolor="k")
    ax.set_ylim(0, 1); ax.set_ylabel("probability (flake)")
    ax.set_title(f"(b) flake PDF   <m_z>={pl['net_m']:+.3f}")

    ax = fig.add_subplot(1, 3, 3)
    im = ax.imshow(nd["net"], cmap="seismic", vmin=-1, vmax=1)
    ax.set_title("(c) local net m_z (flake tiles)"); ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if title:
        fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(out_png, dpi=130); plt.close(fig)

def subpixel_wall_density(mz_field, roi, um_per_px=None):
    """
    Computes exact domain wall density (length / area) using sub-pixel 
    Marching Squares contour extraction at the mz=0.0 boundary.
    """
    # 1. Slightly smooth the field so contours can be interpolated smoothly 
    # between the +1 (up) and -1 (down) domains.
    m_smooth = gaussian_filter(mz_field.astype(float), sigma=1.0)
    
    # 2. Extract sub-pixel coordinates of all domain walls (level=0.0)
    contours = measure.find_contours(m_smooth, level=0.0)
    
    total_length_px = 0.0
    for contour in contours:
        # contour is an (N, 2) array of (row, col) coordinates
        # Calculate the Euclidean length of each sub-pixel segment
        diffs = np.diff(contour, axis=0)
        segment_lengths = np.linalg.norm(diffs, axis=1)
        
        # Only count wall segments that are actually inside our active flake ROI
        r_idx = np.clip(contour[:-1, 0].astype(int), 0, roi.shape[0]-1)
        c_idx = np.clip(contour[:-1, 1].astype(int), 0, roi.shape[1]-1)
        valid_mask = roi[r_idx, c_idx]
        
        total_length_px += np.sum(segment_lengths[valid_mask])
        
    area_px = np.sum(roi)
    if area_px == 0:
        return 0.0

    # 3. Scale to physical units if calibrated
    if um_per_px:
        total_length_um = total_length_px * um_per_px
        area_um2 = area_px * (um_per_px ** 2)
        return total_length_um / area_um2  # units: um / um^2
    else:
        return total_length_px / area_px   # units: px / px^2
    
def analyze_one(path, args, outroot):
    stem = os.path.splitext(os.path.basename(path))[0]
    # default: save results NEXT TO the source image; --outdir overrides
    base = outroot if outroot else os.path.dirname(os.path.abspath(path))
    outdir = os.path.join(base, f"{stem}_analysis")
    os.makedirs(outdir, exist_ok=True)
    p = lambda s: os.path.join(outdir, f"{stem}_{s}")
    um = args.um_per_px

    # per-image calibration from field of view (handles FoV varying per image):
    #   um_per_px = FoV_width / image_width_px  (computed for THIS image)
    if args.fov:
        from PIL import Image as _PILImage
        with _PILImage.open(path) as _im:
            W_px, H_px = _im.size           # PIL: (width, height)
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
    conn = getattr(args, "connectivity", 1)
    tv_weight = getattr(args, "tv_weight", 0.12)
    metrics, layers = analyze_xmcd(path, um_per_px=um, clahe=args.clahe,
                                   roi_box=roi_box, bg_box=bg_box, clip=clip,
                                   connectivity=conn, tv_weight=tv_weight)
    mz, roi = three_state_field(layers, z_neutral=z_neutral,
                                m_threshold=args.m_threshold)
    # revert sub-12px +/-1 speckle to neutral BEFORE any statistic, so every
    # number here matches the GUI path (gui_core also despeckles at min_px=12)
    # and the min-domain-size filter is applied consistently across the whole
    # report -- pdf, morphometry, and the figures all see one cleaned field.
    # (12 = the same min_px used by summarize_domains/per_state_morphometry below.)
    mz = despeckle_mz(mz, min_px=12)
    pl = pdf_three_state(mz, region=roi)

    # area-averaged <m_z> from the continuous asymmetry field, BEFORE
    # thresholding -- cheap (no bootstrap), so computed unconditionally even
    # when --no-rigorous skips the full magnetization_statistics() pipeline.
    aa = area_averaged_mz(layers)
    # raw signal averaged over only the classified (non-gray-band) pixels --
    # see net_mz_raw's docstring for why this sits between the discrete and
    # continuous definitions above.
    hy = net_mz_raw(mz, layers)
    nd = number_density_local(mz, tile=args.tile, region=roi)   # FLAKE-only

    unit = "um" if metrics["calibrated"] else "px"
    sizes = layers["eq_diam_px"] * (um or 1.0)
    _, fit = make_distribution_figure(sizes, p("size_distribution.png"),
                                      unit=unit, title=stem)
    flake_stats_figure(layers["img"], mz, roi, nd, p("flake_stats.png"),
                       title=f"{stem} (flake-area statistics)")

    # rigorous magnetization statistics (correlation-corrected error, CI, etc.)
    mzs = None
    if not args.no_rigorous:
        mzs = magnetization_statistics(layers, n_boot=args.n_boot)
        make_magnetization_figure(mzs, p("magnetization.png"),
                                  title=f"{stem} -- magnetization statistics")

    # domain width / skeleton (robust size for labyrinth patterns) -- computed
    # ONCE here (min_px=12, the despeckle threshold used everywhere else in
    # this pipeline) and reused for both the figure and domain_summary below,
    # instead of each independently refitting every log-normal on its own
    # differently-filtered pass (which used to disagree with each other).
    mm = per_state_morphometry(mz, um_per_px=um, min_px=12)
    make_morphometry_figure(mz, roi, um, p("morphometry.png"), mm=mm,
                            title=f"{stem} -- domain width / skeleton")

    # consolidated up/down/total summary: n_domains, area, width median +
    # mode, mz, period -- folded into summary.txt below (single source of
    # truth) instead of a separate CSV.
    domain_summary = summarize_domains(mz, roi, um_per_px=um, min_px=12,
                                       connectivity=conn, layers=layers, mm=mm)

    # standalone fitting summary: log-normal AND Gaussian parameters for the
    # width and period samples (the same fits the distribution figures plot),
    # from the same `mm` everything else here used.
    export_fit_summary(mm, um_per_px=um, out_path=p("fits.txt"),
                       title=stem, min_px=12)

    # optional publication TIFFs (three separate panels with colorbars)
    if args.tiff:
        from xmcd_toolkit import export_panels_tiff
        export_panels_tiff(layers["img"], mz, roi, um, outdir, stem, dpi=args.dpi)

    # per-state domain area + shape (mean / SD / median, eccentricity, etc.)
    ss = per_state_domain_stats(mz, um_per_px=um, connectivity=conn, roi=roi)
    au = ss["up"]["units"]   # "px" or "um^2"
    exact_wall_density = subpixel_wall_density(mz, roi, um)
    # scalar flake metrics table
    cur, tmp = parse_filename(stem)
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
        "lognorm_median": round(fit["median"], 3) if fit else None,
        "lognorm_sigma_ln": round(fit["sigma_log"], 3) if fit else None,
        "lognorm_KS": round(fit["ks_stat"], 3) if fit else None,
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
    period = domain_summary["total"]["period_nm"] if um else domain_summary["total"]["period_px"]
    lines += [
        "",
        "--- Domains (up / down / total) ---",
        f"(width/period in {'nm' if um else 'px'}; mz = area fraction for up/down, net <m_z> for total)",
        _domain_row("up"),
        _domain_row("down"),
        _domain_row("total"),
        f"  period ~ {period:.2f} {'nm' if um else 'px'}" if period == period else "  period unavailable",
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
    with open(p("summary.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")

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


def main():
    ap = argparse.ArgumentParser(description="Flake-area statistical analysis of XMCD-PEEM images.")
    ap.add_argument("paths", nargs="*", default=["."],
                    help="image files and/or folders (default: current dir)")
    ap.add_argument("--recursive", action="store_true", help="descend into sub-folders")
    ap.add_argument("--pattern", default="*.png", help="filename glob (default *.png)")
    ap.add_argument("--outdir", default=None,
                    help="output directory (default: next to each source image)")
    ap.add_argument("--um-per-px", type=float, default=None, help="scale-bar calibration (fixed for all images)")
    ap.add_argument("--fov", type=float, nargs=2, metavar=("WIDTH_UM", "HEIGHT_UM"),
                    default=None, help="field of view in um; um/px is computed "
                                       "per image from its own pixel size")
    ap.add_argument("--tile", type=int, default=16, help="number-density tile size (px)")
    ap.add_argument("--z-neutral", type=float, default=0.5, help="neutral-band half-width (z)")
    ap.add_argument("--m-threshold", type=float, default=None,
                    help="value-based gray band on normalized asymmetry: "
                         "|m|<=t -> 0, m>t -> +1, m<-t -> -1 (e.g. 0.1)")
    ap.add_argument("--tv-weight", type=float, default=0.12, dest="tv_weight",
                    help="total-variation denoise strength applied BEFORE "
                         "classification (default 0.12, tuned for clean "
                         "screenshots; noisy experimental frames typically "
                         "need ~0.6-1.0)")
    # --- controls for "too much gray inside the active area" ---
    ap.add_argument("--states", type=int, choices=(2, 3), default=3,
                    help="2 = hard up/down (no gray neutral); 3 = up/none/down")
    ap.add_argument("--clahe", action="store_true",
                    help="local contrast enhancement to pull weak domains out of gray")
    ap.add_argument("--roi", type=int, nargs=4, metavar=("X0", "Y0", "X1", "Y1"),
                    default=None, help="manual ACTIVE AREA box (skips auto-detect)")
    ap.add_argument("--bg", type=int, nargs=4, metavar=("X0", "Y0", "X1", "Y1"),
                    default=None, help="BACKGROUND region box (sets magnetic zero)")
    ap.add_argument("--clip", type=float, nargs=2, metavar=("LO", "HI"),
                    default=None, help="contrast percentiles (default 0.5 99.5)")
    # --- rigorous magnetization analysis ---
    ap.add_argument("--no-rigorous", action="store_true",
                    help="skip correlation-corrected magnetization stats (faster)")
    ap.add_argument("--n-boot", type=int, default=1000,
                    help="block-bootstrap iterations for the m_z CI (default 1000)")
    # --- extra outputs ---
    ap.add_argument("--tiff", action="store_true",
                    help="also export the 3 publication TIFF panels per image")
    ap.add_argument("--dpi", type=int, default=300, help="TIFF resolution (default 300)")
    ap.add_argument("--connectivity", type=int, choices=(4, 8), default=4,
                    help="domain connectivity: 4 = edges only (default), "
                         "8 = edges + diagonals/corners")
    args = ap.parse_args()
    args.connectivity = 1 if args.connectivity == 4 else 2   # -> scipy rank

    paths = args.paths if args.paths else ["."]
    targets = collect_targets(paths, args.pattern, args.recursive)
    if not targets:
        print(f"No images matching '{args.pattern}' found."); sys.exit(1)

    # where the combined table / run-info go: --outdir, else the source folder
    if args.outdir:
        table_dir = args.outdir
        os.makedirs(table_dir, exist_ok=True)
    else:
        table_dir = os.path.dirname(os.path.abspath(targets[0]))
    print(f"Analyzing {len(targets)} image(s); results saved "
          + (f"in {args.outdir}/" if args.outdir else "next to each source image"))

    rows = []
    for t in targets:
        try:
            rows.append(analyze_one(t, args, args.outdir))
        except Exception as e:
            print(f"  ERROR on {t}: {e}")

    # combined table across all images
    if rows:
        out_csv = os.path.join(table_dir, "summary_table.csv")
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=SUMMARY_COLS)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k) for k in SUMMARY_COLS})

        # run-info / calibration provenance record
        import datetime
        cal = (f"{args.um_per_px} um/px (fixed)" if args.um_per_px is not None
               else f"FoV {args.fov[0]}x{args.fov[1]} um (per image)" if args.fov
               else "uncalibrated (pixels) -- use calibrate.py later")
        info = [
            f"run date        : {datetime.datetime.now().isoformat(timespec='seconds')}",
            f"images analyzed : {len(rows)}",
            f"calibration     : {cal}",
            f"m-threshold     : {args.m_threshold if args.m_threshold is not None else f'z-score band {args.z_neutral}'}",
            f"tv-weight       : {getattr(args, 'tv_weight', 0.12)}",
            f"states          : {args.states}",
            f"tile (px)       : {args.tile}",
            f"bootstrap iters : {0 if args.no_rigorous else args.n_boot}",
            f"tiff panels     : {'yes' if args.tiff else 'no'}",
            f"summary table   : {out_csv}",
            "",
            "calibrated table: run  python calibrate.py "
            f"{os.path.basename(out_csv)} --fov WIDTH HEIGHT  (or --um-per-px)",
        ]
        with open(os.path.join(table_dir, "run_info.txt"), "w") as f:
            f.write("\n".join(info) + "\n")

        print(f"\nWrote combined table: {out_csv}  ({len(rows)} rows)")
        print(f"Wrote run info / calibration record: {os.path.join(table_dir, 'run_info.txt')}")
        if any(r.get("current_mA") is not None for r in rows):
            print("  current_mA column populated -> ready to plot stats vs current.")


if __name__ == "__main__":
    main()
