"""
gui_core.py -- processing logic for the GUI (no Tk dependency, testable headless).

Workflow the GUI implements:
  1. load image  -> show immediately
  2. draw a polygon over the ACTIVE AREA (the region containing domains)
  3. three-state classification (up / neutral / down) INSIDE the polygon,
     controlled by the m-threshold
  4. within the polygon: distributions of domain WIDTH, domain PERIOD, and
     the up/down PROBABILITY (area fractions); domain counts/areas
     (up/down/neutral) via domain_counts_areas, summarized into
     <stem>_summary.txt (see analyze.analyze_one) on export
The field of view corresponds to the whole image, so
     um_per_px = FoV_width / image_width_px.
"""
import os
import csv
import argparse
import datetime

import numpy as np

import analyze as ANALYZE
from scipy import ndimage as ndi
from xmcd_toolkit import (
    analyze_xmcd, three_state_field, despeckle_mz, pdf_three_state, per_state_morphometry,
    make_width_area_count_figure, make_width_distribution_figure, make_period_distribution_figure,
    make_qq_plot_figure, qq_series_per_state, make_lognormal_gaussian_comparison_figure,
    compare_lognormal_gaussian, domain_counts_areas,
    area_averaged_mz, net_mz_raw, fit_lognormal,
    COLOR_UP, COLOR_DOWN,
)


def _domain_areas(mz, state, min_px=12, connectivity=1):
    """Per-domain pixel areas for a given state (+1 up / -1 down)."""
    from xmcd_toolkit import neighbor_structure
    lbl, n = ndi.label(mz == state, structure=neighbor_structure(connectivity))
    if not n:
        return np.array([])
    sizes = np.bincount(lbl.ravel())[1:]
    return sizes[sizes >= min_px].astype(float)


# ----------------------------------------------------------------------
def polygon_to_mask(vertices, shape):
    """Boolean mask for a polygon given (x, y) vertices and image shape (H, W)."""
    from skimage.draw import polygon as sk_polygon
    mask = np.zeros(shape, bool)
    if not vertices or len(vertices) < 3:
        return mask
    xs = np.array([v[0] for v in vertices], float)
    ys = np.array([v[1] for v in vertices], float)
    rr, cc = sk_polygon(ys, xs, shape)
    mask[rr, cc] = True
    return mask


def um_per_px_from_fov(shape, fov_wh):
    """FoV (whole image) -> um/px from the image width; checks square pixels."""
    if not fov_wh:
        return None, None
    H, W = shape
    s_w = fov_wh[0] / W
    s_h = fov_wh[1] / H
    warn = None
    if abs(s_w - s_h) / max(s_w, s_h) > 0.05:
        warn = f"non-square pixels ({s_w:.4f} vs {s_h:.4f} um/px) -- check FoV order"
    return s_w, warn


def um_per_px_from_box(box_px, known_um, axis="width"):
    """Calibrate from a drawn box of known physical size along one axis:
        um/px = known_um / box_length_px. box_px = (W_px, H_px)."""
    if not box_px or not known_um:
        return None
    w_px, h_px = box_px
    if axis == "height":
        ref = h_px
    elif axis == "diagonal":
        ref = float(np.hypot(w_px, h_px))
    else:
        ref = w_px
    return (known_um / ref) if ref > 0 else None


def um_per_px_from_dims(box_px, known_wh):
    """Calibrate from a reference rectangle of known width AND/OR height.
    box_px = (W_px, H_px); known_wh = (W_um or None, H_um or None).
    The reference may be the whole image or the bounding box of any drawn
    shape (polygon, rectangle, or square). Uses whichever dimensions are given;
    if both are given, averages them and warns on non-square pixels.
    Returns (um_per_px, warn)."""
    if not box_px:
        return None, None
    w_px, h_px = box_px
    W_um, H_um = (known_wh + (None, None))[:2]
    scales = []
    if W_um and w_px > 0:
        scales.append(W_um / w_px)
    if H_um and h_px > 0:
        scales.append(H_um / h_px)
    if not scales:
        return None, None
    warn = None
    if len(scales) == 2 and abs(scales[0] - scales[1]) / max(scales) > 0.05:
        warn = (f"non-square pixels ({scales[0]:.4f} vs {scales[1]:.4f} um/px) "
                f"-- check dimensions")
    return float(np.mean(scales)), warn


def resolve_um(params, shape):
    """Pixel size from an explicit um_per_px (e.g. box calibration) if present,
    otherwise from the whole-image field of view. Returns (um_per_px, warn)."""
    if params.get("um_per_px"):
        return float(params["um_per_px"]), None
    return um_per_px_from_fov(shape, params.get("fov"))


# ----- image-level cache: the expensive denoise/flatten is polygon- and
#       threshold-independent, so compute it once per (path, clahe). -----
_PREP_CACHE = {}
_PREP_ORDER = []

def _prep(path, clahe, flatten=True, flatten_sigma=25.0, tv_weight=0.12):
    # The cache key tracks the state of every toggle/slider that feeds this
    # expensive step. tv_weight belongs here (not in preview_polygon) because
    # it drives the TV-Chambolle denoise inside analyze_xmcd itself -- the
    # same "compute once, cache" step as flatten_sigma. Changing it is a cache
    # miss, so it re-runs the full (slow) denoise, same as flatten_sigma does.
    key = (os.path.abspath(path), bool(clahe), bool(flatten), float(flatten_sigma),
           float(tv_weight))
    if key not in _PREP_CACHE:

        # --- SAFE FALLBACK: If disabled, use a massive sigma ---
        # On a 512x512 grid, a 999px sigma produces a perfectly uniform matrix,
        # effectively disabling localized background flattening without crashing.
        sigma_val = float(flatten_sigma) if flatten else 999.0
        # --------------------------------------------------------

        _, layers = analyze_xmcd(path, clahe=clahe, flatten_sigma=sigma_val,
                                 tv_weight=float(tv_weight))

        _PREP_CACHE[key] = {"img": layers["img"], "flat": layers["flat"],
                            "den": layers["denoised"]}
        _PREP_ORDER.append(key)
        while len(_PREP_ORDER) > 6:                       # cap memory
            _PREP_CACHE.pop(_PREP_ORDER.pop(0), None)
    return _PREP_CACHE[key]
def clear_cache(path=None):
    """Drop cached fields (call when an image is (re)loaded)."""
    global _PREP_CACHE, _PREP_ORDER
    if path is None:
        _PREP_CACHE, _PREP_ORDER = {}, []
    else:
        ap = os.path.abspath(path)
        for k in [k for k in _PREP_CACHE if k[0] == ap]:
            _PREP_CACHE.pop(k, None)
        _PREP_ORDER[:] = [k for k in _PREP_ORDER if k[0] != ap]


def preview_polygon(path, params):
    """FAST preview: reuse the cached flattened field and recompute only the
    threshold-dependent three-state map, distributions, and morphometry.
    Consistent with analyze_with_polygon (same flat/roi -> same result)."""
    poly = params.get("polygon")
    m_threshold = params.get("m_threshold", 0.1)
    states = params.get("states", 3)
    z_neutral = 0.0 if states == 2 else params.get("z_neutral", 0.5)

    prep = _prep(path,
                 params.get("clahe", False),
                 params.get("flatten", True),
                 params.get("flatten_sigma", 25.0),
                 params.get("tv_weight", 0.12))
    shape = prep["flat"].shape
    roi_mask = polygon_to_mask(poly, shape) if poly else np.ones(shape, bool)
    um, warn = resolve_um(params, shape)

    # background region: set the magnetic zero from its mean (same rule as
    # analyze_xmcd's bg path), computed on the cached denoised image so the
    # preview matches the full export exactly.
    bg_poly = params.get("bg_polygon")
    if bg_poly:
        bg = polygon_to_mask(bg_poly, shape)
        flat = (prep["den"] - float(prep["den"][bg].mean())) if bg.any() else prep["flat"]
    else:
        flat = prep["flat"]

    layers = {"img": prep["img"], "flat": flat, "roi": roi_mask}
    mz, roi = three_state_field(layers, z_neutral=z_neutral, m_threshold=m_threshold)
    mz = despeckle_mz(mz, min_px=params.get("min_px", 12))
    pl = pdf_three_state(mz, region=roi)
    mm = per_state_morphometry(mz, um_per_px=um, min_px=params.get("min_px", 12))
    counts = domain_counts_areas(mz, roi=roi, min_px=params.get("min_px", 12),
                                 connectivity=int(params.get("connectivity", 1)), um_per_px=um)
    # area-averaged <m_z> from the continuous asymmetry field, BEFORE
    # thresholding -- cheap (no bootstrap), so always available live.
    aa = area_averaged_mz(layers)
    # raw signal averaged over only the classified (non-gray-band) pixels --
    # see net_mz_raw's docstring for why this sits between the discrete
    # (pl["net_m"]) and continuous (aa) definitions.
    hy = net_mz_raw(mz, layers)
    return {
        "img": prep["img"], "mz": mz, "roi": roi, "layers": layers,
        "um": um, "fov_warn": warn, "shape": shape,
        "pdf": pl, "morphometry": mm, "counts": counts, "params": params,
        "area_avg_mz": aa, "net_mz_raw": hy,
    }


# ----------------------------------------------------------------------
def analyze_with_polygon(path, params):
    """Full analysis restricted to the polygon (used for export). Returns a
    results dict with the three-state field, morphometry, and distributions."""
    poly = params.get("polygon")
    clahe = params.get("clahe", False)
    m_threshold = params.get("m_threshold", 0.1)
    states = params.get("states", 3)
    z_neutral = 0.0 if states == 2 else params.get("z_neutral", 0.5)

    from PIL import Image
    with Image.open(path) as im:
        W, H = im.size
    shape = (H, W)
    roi_mask = polygon_to_mask(poly, shape) if poly else None
    bg_poly = params.get("bg_polygon")
    bg_mask = polygon_to_mask(bg_poly, shape) if bg_poly else None

    um, fov_warn = resolve_um(params, shape)

    if roi_mask is not None:
        # Reuse the same (path, clahe, flatten, flatten_sigma, tv_weight)
        # cache preview_polygon uses. Export always has a drawn polygon by
        # this point (the GUI blocks Export without one), and the normal
        # workflow is preview repeatedly then Export with the same settings
        # -- so this skips redoing the expensive TV-denoise + illumination
        # flatten from scratch right after the live preview just computed
        # it. connectivity/min_domain_px don't affect img/flat/roi inside
        # analyze_xmcd, so they're irrelevant to this cached step. Falls
        # back to a full analyze_xmcd() call (with its own auto-ROI
        # detection) when there's no polygon to key the reused flat field
        # against.
        prep = _prep(path, clahe,
                     params.get("flatten", True),
                     params.get("flatten_sigma", 25.0),
                     params.get("tv_weight", 0.12))
        if bg_mask is not None and bg_mask.any():
            flat = prep["den"] - float(prep["den"][bg_mask].mean())
        else:
            flat = prep["flat"]
        layers = {"img": prep["img"], "flat": flat, "roi": roi_mask}
    else:
        _, layers = analyze_xmcd(path, clahe=clahe, roi_mask=roi_mask, bg_mask=bg_mask,
                                 connectivity=params.get("connectivity", 1),
                                 tv_weight=params.get("tv_weight", 0.12))
    mz, roi = three_state_field(layers, z_neutral=z_neutral, m_threshold=m_threshold)
    mz = despeckle_mz(mz, min_px=params.get("min_px", 12))

    pl = pdf_three_state(mz, region=roi)
    mm = per_state_morphometry(mz, um_per_px=um, min_px=params.get("min_px", 12))
    counts = domain_counts_areas(mz, roi=roi, min_px=params.get("min_px", 12),
                                 connectivity=params.get("connectivity", 1), um_per_px=um)
    # area-averaged <m_z> from the continuous asymmetry field, BEFORE
    # thresholding -- cheap (no bootstrap), so always available on export too.
    aa = area_averaged_mz(layers)
    # raw signal averaged over only the classified (non-gray-band) pixels --
    # see net_mz_raw's docstring for why this sits between the discrete
    # (pl["net_m"]) and continuous (aa) definitions.
    hy = net_mz_raw(mz, layers)

    return {
        "img": layers["img"], "mz": mz, "roi": roi, "layers": layers,
        "um": um, "fov_warn": fov_warn, "shape": shape,
        "pdf": pl, "morphometry": mm, "counts": counts, "params": params,
        "area_avg_mz": aa, "net_mz_raw": hy,
    }


def summary_text(res):
    um = res["um"]; unit = "um" if um else "px"
    # domain width/period display in nm when calibrated -- the natural scale
    # for XMCD-PEEM domains; um_per_px itself (the actual calibration) is
    # untouched, this only picks which pre-computed field to print.
    w_unit = "nm" if um else "px"
    pl = res["pdf"]; mm = res["morphometry"]; aa = res["area_avg_mz"]; hy = res["net_mz_raw"]
    npx = int(res["roi"].sum())
    area = npx * (um * um) if um else npx
    up_width = mm["up"]["width_median_nm"] if um else mm["up"]["width_median"]
    dn_width = mm["down"]["width_median_nm"] if um else mm["down"]["width_median"]
    period = mm.get("domain_period_nm", mm["domain_period"]) if um else mm["domain_period"]
    # fitted log-normal median/mode on the SAME samples fits.txt reports, so
    # the live readout and the exported fit summary agree (the raw sample
    # median above is a different statistic and can differ by >20%).
    w_scale = (um * 1000.0) if um else 1.0
    fit_up = fit_lognormal(mm["up"]["widths_px"] * w_scale)
    fit_dn = fit_lognormal(mm["down"]["widths_px"] * w_scale)
    _v = lambda fit, k: f"{fit[k]:.1f}" if fit else "n/a"
    # Compact, fixed-width block -- this is rendered as a corner overlay on the
    # GUI image panel, so line LENGTH matters: long trailing parentheticals
    # push the box over the image and into the legend.
    lines = [
        f"active area  = {area:.1f} {'um^2' if um else 'px'} ({npx} px)",
        f"P(up/0/dn)   = {pl['p_up']:.3f} / {pl['p_zero']:.3f} / {pl['p_down']:.3f}",
        f"net <m_z>    = {pl['net_m']:+.3f}  (discrete)",
        (f"net <m_z> raw= {hy['mean_mz']:+.3f}  (excl. gray, N={hy['N_pixels']})"
         if hy["mean_mz"] == hy["mean_mz"] else
         "net <m_z> raw= n/a (no px cleared threshold)"),
        f"area-avg<m_z>= {aa['mean_mz']:+.3f}  (pre-thr, N={aa['N_pixels']})",
        f"width [{w_unit}] raw med : up={up_width:.1f} dn={dn_width:.1f}",
        f"          fit med : up={_v(fit_up,'median')} dn={_v(fit_dn,'median')}",
        f"          fit mode: up={_v(fit_up,'mode')} dn={_v(fit_dn,'mode')}",
        f"period ~ {period:.1f} {w_unit}",
        f"calibration  = {f'{um:.4f} um/px' if um else 'uncalibrated (px)'}",
    ]
    counts = res.get("counts")
    if counts:
        lines.append(f"# domains: up={counts['up']['n_domains']} "
                     f"dn={counts['down']['n_domains']} "
                     f"neu={counts['neutral']['n_domains']}")
    if res.get("fov_warn"):
        lines.append("WARN: " + res["fov_warn"])
    return "\n".join(lines)


# ----------------------------------------------------------------------
def build_args(params):
    cal_um, _ = resolve_um(params, params["shape"]) if params.get("shape") else (None, None)
    return argparse.Namespace(
        um_per_px=cal_um, fov=None,
        roi=None, bg=None, clip=None,
        states=params.get("states", 3),
        z_neutral=params.get("z_neutral", 0.5),
        m_threshold=params.get("m_threshold", 0.1),
        tv_weight=params.get("tv_weight", 0.12),
        clahe=params.get("clahe", False),
        tile=params.get("tile", 16),
        no_rigorous=(not params.get("rigorous", True)),
        n_boot=params.get("n_boot", 1000),
        tiff=params.get("tiff", True),
        dpi=params.get("dpi", 300),
        connectivity=params.get("connectivity", 1),
    )


def run_export(path, params):
    """Full export restricted to the polygon. Writes figures/tables/TIFFs plus a
    polygon-distributions figure, next to the source image. Returns output dir."""
    import matplotlib
    matplotlib.use("Agg")

    poly = params.get("polygon")
    from PIL import Image
    with Image.open(path) as im:
        W, H = im.size
    shape = (H, W)
    params = dict(params); params["shape"] = shape
    roi_mask = polygon_to_mask(poly, shape) if poly else None
    bg_poly = params.get("bg_polygon")
    bg_mask = polygon_to_mask(bg_poly, shape) if bg_poly else None

    args = build_args(params)
    
    # --- FIX 1: Safely inject the flatten parameters into the args object ---
    # This guarantees the core analysis engine gets the 999.0 fallback if disabled
    flatten = params.get("flatten", True)
    args.flatten_sigma = float(params.get("flatten_sigma", 25.0)) if flatten else 999.0
    # ------------------------------------------------------------------------

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

    res = analyze_with_polygon(path, params)
    make_distributions_figure(res, os.path.join(outdir, f"{stem}_distributions.png"))

    # per-domain width/area vs. domain-count histograms (svg + tiff), same
    # min_px the rest of this export used, so the counts agree.
    paths, _ = make_width_area_count_figure(
        res["mz"], um_per_px=res["um"], min_px=params.get("min_px", 12),
        out_stem=os.path.join(outdir, f"{stem}_width_area_counts"),
        formats=("svg", "tiff"), title=stem,
    )
    _drop_json_sidecars(paths)

    # The up/down area fractions (f_+, f_-, f_dw) annotated on the width/period
    # distribution figures -- the SAME three-state PDF summary.txt reports, so
    # the figure numbers match the report exactly.
    pl = res["pdf"]
    fractions = (pl["p_up"], pl["p_down"], pl["p_zero"])

    # standalone skeleton-pixel width distribution: up/down histograms with
    # BOTH the log-normal AND Gaussian fits overlaid (compare_gaussian=True,
    # always -- not gated on the GUI checkbox, which only drives the live
    # preview panel), plus the f_+/f_-/f_dw fraction annotation.
    paths, _ = make_width_distribution_figure(
        res["mz"], um_per_px=res["um"], min_px=params.get("min_px", 12),
        out_stem=os.path.join(outdir, f"{stem}_width_distribution"),
        formats=("svg",), title=stem,
        compare_gaussian=True, fractions=fractions,
    )
    _drop_json_sidecars(paths)

    # standalone skeleton-pixel PERIOD distribution -- same treatment as width
    # above (both fits, fraction annotation), always exported.
    paths, _ = make_period_distribution_figure(
        res["mz"], um_per_px=res["um"], min_px=params.get("min_px", 12),
        out_stem=os.path.join(outdir, f"{stem}_period_distribution"),
        formats=("svg",), title=stem,
        compare_gaussian=True, fractions=fractions,
    )
    _drop_json_sidecars(paths)

    # standalone Q-Q plots (width AND period vs. fitted log-normal), same
    # on/off state as the GUI's "Show Q-Q plot" checkbox -- both are
    # exported together whenever it's ticked, regardless of which one the
    # live-preview panel is currently showing.
    if params.get("qq_plot", False):
        paths, _ = make_qq_plot_figure(
            res["mz"], um_per_px=res["um"], min_px=params.get("min_px", 12),
            out_stem=os.path.join(outdir, f"{stem}_width_qq"),
            formats=("svg",), title=stem, quantity="width",
        )
        _drop_json_sidecars(paths)
        paths, _ = make_qq_plot_figure(
            res["mz"], um_per_px=res["um"], min_px=params.get("min_px", 12),
            out_stem=os.path.join(outdir, f"{stem}_period_qq"),
            formats=("svg",), title=stem, quantity="period",
        )
        _drop_json_sidecars(paths)

    # OPTIONAL two-panel (a=up, b=down) log-normal-vs-Gaussian comparison --
    # a density-normalized alternate view of the same comparison the
    # width/period distribution figures above already show overlaid on one
    # axes. Redundant with those, so only written when the GUI's "Show
    # log-normal vs. Gaussian comparison" checkbox is ticked, to keep the
    # default export folder lean.
    if params.get("gauss_compare", False):
        paths, _ = make_lognormal_gaussian_comparison_figure(
            res["mz"], um_per_px=res["um"], min_px=params.get("min_px", 12),
            out_stem=os.path.join(outdir, f"{stem}_width_lognorm_vs_gauss"),
            formats=("svg",), title=stem, quantity="width",
        )
        _drop_json_sidecars(paths)
        paths, _ = make_lognormal_gaussian_comparison_figure(
            res["mz"], um_per_px=res["um"], min_px=params.get("min_px", 12),
            out_stem=os.path.join(outdir, f"{stem}_period_lognorm_vs_gauss"),
            formats=("svg",), title=stem, quantity="period",
        )
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
        w = csv.DictWriter(f, fieldnames=ANALYZE.SUMMARY_COLS)
        w.writeheader(); w.writerow({k: row.get(k) for k in ANALYZE.SUMMARY_COLS})
        
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
        f.write(f"run date   : {datetime.datetime.now().isoformat(timespec='seconds')}\n"
                f"image      : {os.path.basename(path)}\n"
                f"active area: polygon ({len(poly) if poly else 0} vertices)\n"
                f"background : {bg_desc}\n"
                f"calibration: {cal}\n"
                f"m-threshold: {args.m_threshold}\n"
                f"tv-weight  : {args.tv_weight}\n"
                f"states     : {args.states}\n")
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

    for st, col, lab in ((mm["up"], COLOR_UP, "up"), (mm["down"], COLOR_DOWN, "down")):
        w = st["widths_px"] * w_scale
        if w.size:
            ax[0].hist(w, bins=24, alpha=0.55, color=col, label=lab)
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
    if len(ax) > 3:
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
    if len(ax) > 4:
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
