"""Export everything for one image and polygon: figures, distributions, TIFFs, tables.
Qt-free service layer."""
from ..notation import column, using
import os
import csv
import argparse
import datetime


from ..defaults import DEFAULTS
from . import batch as ANALYZE
from ..core.regions import polygon_to_mask, resolve_um
from ..plotting import (
    make_width_area_count_figure, make_width_distribution_figure, make_period_distribution_figure,
    make_qq_plot_figure, make_lognormal_gaussian_comparison_figure,
)
from ..plotting.distribution_panels import make_distributions_figure
from .preview import analyze_with_polygon


# ----------------------------------------------------------------------
def build_args(params):
    """Build the analyze_one args Namespace from the GUI params, so the EXPORT
    (analyze_one) uses exactly the values set in the GUI -- including min_px,
    flatten_sigma, and tv_weight, which the export previously ignored or
    hardcoded. Fallbacks come from the shared DEFAULTS (defaults.py), so the
    GUI, CLI, and export all start from identical values."""
    g = lambda k: params.get(k, DEFAULTS[k])
    cal_um, _ = resolve_um(params, params["shape"]) if params.get("shape") else (None, None)
    return argparse.Namespace(
        um_per_px=cal_um, fov=None,
        roi=None, bg=None, clip=None,
        states=g("states"),
        z_neutral=g("z_neutral"),
        m_threshold=g("m_threshold"),
        tv_weight=g("tv_weight"),
        flatten_sigma=(g("flatten_sigma") if params.get("flatten", True) else 999.0),
        min_px=g("min_px"),
        width_method=g("width_method"),
        denoise_method=g("denoise_method"),
        boundary_method=g("boundary_method"),
        input_mode=g("input_mode"),
        map_mode=g("map_mode"),
        invert=bool(params.get("invert", DEFAULTS["invert"])),
        sym_centre=str(params.get("sym_centre", DEFAULTS["sym_centre"])),
        dichroism=str(params.get("dichroism", DEFAULTS["dichroism"])),
        fig_format=g("fig_format"),
        clahe=g("clahe"),
        tile=g("tile"),
        no_rigorous=(not params.get("rigorous", DEFAULTS["rigorous"])),
        n_boot=g("n_boot"),
        tiff=g("tiff"),
        detailed=params.get("detailed", DEFAULTS["detailed"]),
        dpi=g("dpi"),
        connectivity=g("connectivity"),
    )


def run_export(path, params):
    """Full export (see _run_export); state names +ve / -ve, order parameter of params["dichroism"]."""
    with using(params.get("dichroism", "XMCD")):
        return _run_export(path, params)


def _run_export(path, params):
    """Full export restricted to the polygon. Writes figures/tables/TIFFs plus a
    polygon-distributions figure, next to the source image. Returns output dir."""
    import matplotlib
    matplotlib.use("Agg")

    poly = params.get("polygon")
    from ..core.igor_import import analysis_shape
    shape = analysis_shape(path, params.get("input_mode", "grayscale"))
    params = dict(params); params["shape"] = shape
    roi_mask = polygon_to_mask(poly, shape) if poly else None
    bg_poly = params.get("bg_polygon")
    bg_mask = polygon_to_mask(bg_poly, shape) if bg_poly else None

    # build_args already sets flatten_sigma from the GUI (999.0 when flatten
    # is off), min_px, tv_weight, etc. -- so the export uses the GUI values.
    args = build_args(params)
    flatten = params.get("flatten", True)

    base = params.get("outdir") or os.path.dirname(os.path.abspath(path))
    stem = os.path.splitext(os.path.basename(path))[0]
    outdir = os.path.join(base, f"{stem}_analysis")
    os.makedirs(outdir, exist_ok=True)

    def _drop_json_sidecars(paths):
        """The make_*_figure helpers below also write small
        *_lognormal_fit.json / *_fit_comparison.json sidecars alongside
        their SVG/TIFF plot -- the same numbers (median, mode, sigma_log,
        AIC/BIC/KS) are already in the exported summary.txt's domain table,
        so these standalone JSON copies are dropped to keep the export
        folder to figures + one physics/statistics report, not figures +
        report + a JSON file per figure."""
        for fp in paths:
            if fp.endswith(".json"):
                try:
                    os.remove(fp)
                except OSError:
                    pass

    # The GUI's distribution / Q-Q / log-normal-vs-Gaussian figures are
    # DIAGNOSTICS -- written to diagnostics/ only when "detailed" is on (the
    # default export is statistics.txt + figures/, kept lean).
    if params.get("detailed", DEFAULTS["detailed"]):
        # all diagnostic figures follow the chosen figure format (png/eps/pdf);
        # the multi-format make_*_figure helpers get it as their single format.
        fext = params.get("fig_format", DEFAULTS["fig_format"])
        _dg = os.path.join(outdir, "diagnostics"); os.makedirs(_dg, exist_ok=True)
        _dp = lambda s: os.path.join(_dg, f"{stem}_{s}")
        res = analyze_with_polygon(path, params)
        make_distributions_figure(res, _dp(f"distributions.{fext}"))
        paths, _ = make_width_area_count_figure(
            res["mz"], um_per_px=res["um"], min_px=params.get("min_px", 12),
            out_stem=_dp("width_area_counts"), formats=(fext,), title=stem)
        _drop_json_sidecars(paths)
        pl = res["pdf"]; fractions = (pl["p_up"], pl["p_down"], pl["p_zero"])
        paths, _ = make_width_distribution_figure(
            res["mz"], um_per_px=res["um"], min_px=params.get("min_px", 12),
            out_stem=_dp("width_distribution"), formats=(fext,), title=stem,
            compare_gaussian=True, fractions=fractions)
        _drop_json_sidecars(paths)
        paths, _ = make_period_distribution_figure(
            res["mz"], um_per_px=res["um"], min_px=params.get("min_px", 12),
            out_stem=_dp("period_distribution"), formats=(fext,), title=stem,
            compare_gaussian=True, fractions=fractions)
        _drop_json_sidecars(paths)
        if params.get("qq_plot", False):
            for q in ("width", "period"):
                paths, _ = make_qq_plot_figure(
                    res["mz"], um_per_px=res["um"], min_px=params.get("min_px", 12),
                    out_stem=_dp(f"{q}_qq"), formats=(fext,), title=stem, quantity=q)
                _drop_json_sidecars(paths)
        if params.get("gauss_compare", False):
            for q in ("width", "period"):
                paths, _ = make_lognormal_gaussian_comparison_figure(
                    res["mz"], um_per_px=res["um"], min_px=params.get("min_px", 12),
                    out_stem=_dp(f"{q}_lognorm_vs_gauss"), formats=(fext,), title=stem, quantity=q)
                _drop_json_sidecars(paths)

    # Domain-count/area/width/period/mz statistics -- previously also
    # written here as export_domain_stats' raw per-domain-area /
    # per-skeleton-pixel width & period CSVs plus its own summary JSON, and
    # a separate domain_summary.csv. All of that is now folded into the
    # single <stem>_summary.txt written by analyze_one() below (which
    # includes the same up/down/total table via summarize_domains()), so
    # this export folder isn't drowning in small redundant data files.
    row = _analyze_one_polygon(path, args, base, roi_mask, bg_mask)

    out_csv = os.path.join(base, "summary_table.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[column(k) for k in ANALYZE.SUMMARY_COLS])   # up / down -> pos / neg
        w.writeheader(); w.writerow({column(k): row.get(k) for k in ANALYZE.SUMMARY_COLS})
        
    cal = f"{args.um_per_px} um/px (FoV = whole image)" if args.um_per_px else "uncalibrated (px)"
    
    # --- FIX 2: Update the metadata log for your MRS records ---
    if bg_poly:
        bg_desc = f"polygon ({len(bg_poly)} vertices)"
    else:
        if flatten:
            bg_desc = f"Gaussian flatten (sigma = {args.flatten_sigma} px)"
        else:
            bg_desc = "none (disabled / global flat average)"
    # -----------------------------------------------------------

    with open(os.path.join(base, "run_info.txt"), "w") as f:
        f.write(f"run date       : {datetime.datetime.now().isoformat(timespec='seconds')}\n"
                f"image          : {os.path.basename(path)}\n"
                f"active area    : polygon ({len(poly) if poly else 0} vertices)\n"
                f"background     : {bg_desc}\n"
                f"calibration    : {cal}\n"
                f"m-threshold    : {args.m_threshold}\n"
                f"input mode     : {getattr(args, 'input_mode', DEFAULTS['input_mode'])}"
                + (f" ({getattr(args, 'map_mode', DEFAULTS['map_mode'])})\n"
                   if getattr(args, 'input_mode', 'grayscale') == 'asymmetry_map' else "\n")
                + f"denoise method : {getattr(args, 'denoise_method', DEFAULTS['denoise_method'])}\n"
                f"tv-weight      : {args.tv_weight}\n"
                f"flatten-sigma  : {getattr(args, 'flatten_sigma', DEFAULTS['flatten_sigma'])}\n"
                f"min domain (px): {getattr(args, 'min_px', DEFAULTS['min_px'])}\n"
                f"width method   : {getattr(args, 'width_method', DEFAULTS['width_method'])}\n"
                f"boundary method: {getattr(args, 'boundary_method', DEFAULTS['boundary_method'])}\n"
                f"detailed figs  : {'yes (diagnostics/)' if params.get('detailed', DEFAULTS['detailed']) else 'no'}\n"
                f"states         : {args.states}\n")
    return outdir


def _analyze_one_polygon(path, args, outroot, roi_mask, bg_mask=None):
    """Run the standard analyze_one pipeline but force the polygon ROI (and an
    optional background region) by shimming analyze_xmcd to inject the masks."""
    orig = ANALYZE.analyze_xmcd
    def shim(image, **kw):
        kw.setdefault("roi_mask", roi_mask)
        if bg_mask is not None:
            kw.setdefault("bg_mask", bg_mask)
        return orig(image, **kw)
    try:
        ANALYZE.analyze_xmcd = shim
        return ANALYZE.analyze_one(path, args, outroot)
    finally:
        ANALYZE.analyze_xmcd = orig
