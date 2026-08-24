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
from defaults import DEFAULTS
from scipy import ndimage as ndi
from xmcd_toolkit import (
    analyze_xmcd, three_state_field, despeckle_mz, pdf_three_state, per_state_morphometry,
    make_width_area_count_figure, make_width_distribution_figure, make_period_distribution_figure,
    make_qq_plot_figure, qq_series_per_state, make_lognormal_gaussian_comparison_figure,
    compare_lognormal_gaussian, domain_counts_areas,
    area_averaged_mz, net_mz_raw, fit_lognormal, correlation_length_state,
    domain_widths, WIDTH_METHOD_LABELS,
    domain_boundary, asymmetry_field,
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
_POLY_CACHE = {}   # (vertices, shape) -> mask; rasterizing is ~20ms, and the
                   # polygon is unchanged across the many threshold-slider
                   # previews between edits, so cache the last few.

def polygon_to_mask(vertices, shape):
    """Boolean mask for a polygon given (x, y) vertices and image shape (H, W).
    Cached on (vertices, shape): dragging the m-threshold slider re-previews
    with the same polygon, so the rasterization is reused instead of redone."""
    from skimage.draw import polygon as sk_polygon
    if not vertices or len(vertices) < 3:
        return np.zeros(shape, bool)
    key = (tuple(map(tuple, vertices)), tuple(shape))
    hit = _POLY_CACHE.get(key)
    if hit is not None:
        return hit
    mask = np.zeros(shape, bool)
    xs = np.array([v[0] for v in vertices], float)
    ys = np.array([v[1] for v in vertices], float)
    rr, cc = sk_polygon(ys, xs, shape)
    mask[rr, cc] = True
    _POLY_CACHE[key] = mask
    if len(_POLY_CACHE) > 8:                      # cap memory (LRU-ish)
        _POLY_CACHE.pop(next(iter(_POLY_CACHE)))
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

def _prep(path, clahe, flatten=True, flatten_sigma=25.0, tv_weight=0.12,
          denoise_method="tv", input_mode="grayscale", map_mode="rb"):
    # The cache key tracks the state of every toggle/slider that feeds this
    # expensive step. tv_weight/denoise_method belong here (not in
    # preview_polygon) because they drive the denoise inside analyze_xmcd --
    # the same "compute once, cache" step as flatten_sigma. Changing any is a
    # cache miss, so it re-runs the full (slow) denoise. input_mode/map_mode
    # change how the image is decoded, so they belong in the key too.
    key = (os.path.abspath(path), bool(clahe), bool(flatten), float(flatten_sigma),
           float(tv_weight), str(denoise_method), str(input_mode), str(map_mode))
    if key not in _PREP_CACHE:

        # --- SAFE FALLBACK: If disabled, use a massive sigma ---
        # On a 512x512 grid, a 999px sigma produces a perfectly uniform matrix,
        # effectively disabling localized background flattening without crashing.
        sigma_val = float(flatten_sigma) if flatten else 999.0
        # --------------------------------------------------------

        # This cached step only needs img/flat/den, and the flattened field
        # (Gaussian high-pass, or the map decode) does NOT depend on the ROI.
        # So skip analyze_xmcd's expensive auto-ROI segmentation (morphological
        # closing + fill-holes on a multi-MP frame, ~seconds) by handing it a
        # whole-image box -- the real ROI is the polygon, applied downstream in
        # preview_polygon / analyze_with_polygon. Saves several seconds per
        # cold prep (each tv_weight / denoiser / sigma change).
        _, layers = analyze_xmcd(path, clahe=clahe, flatten_sigma=sigma_val,
                                 tv_weight=float(tv_weight),
                                 denoise_method=str(denoise_method),
                                 input_mode=str(input_mode), map_mode=str(map_mode),
                                 roi_box=(0, 0, 10**9, 10**9))

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


def _compute_boundary(layers, roi, params):
    """Domain-boundary overlay from the selected method (methods/boundary.py).
    zero_contour/threshold are fast; chan_vese is slower (only when picked)."""
    method = params.get("boundary_method", "zero_contour")
    m, _ = asymmetry_field(layers)
    return domain_boundary(m, roi, method=method,
                           m_threshold=params.get("m_threshold", 0.05))


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
                 params.get("tv_weight", 0.12),
                 params.get("denoise_method", "tv"),
                 params.get("input_mode", "grayscale"),
                 params.get("map_mode", "rb"))
    shape = prep["flat"].shape
    roi_mask = polygon_to_mask(poly, shape) if poly else np.ones(shape, bool)
    um, warn = resolve_um(params, shape)

    # background region: set the magnetic zero from its mean (same rule as
    # analyze_xmcd's bg path), computed on the cached denoised image so the
    # preview matches the full export exactly.
    # A pre-coded asymmetry map is already background-corrected and centered at
    # 0, so a background-region zero doesn't apply -- use the decoded field as-is.
    bg_poly = params.get("bg_polygon")
    if bg_poly and params.get("input_mode", "grayscale") != "asymmetry_map":
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
    # per-state correlation length: the up/down domain SIZE, robust to the
    # tv_weight/threshold choice (FFT autocorrelation, no skeleton).
    # compute the FFT correlation length ONCE (in px) and reuse it for both the
    # per-state xi readout and the correlation_length width method, instead of
    # running the same FFT autocorrelation 4x per preview.
    xu_px = correlation_length_state(mz, roi, 1, um_per_px=None)
    xd_px = correlation_length_state(mz, roi, -1, um_per_px=None)
    _sc = (um * 1000.0) if um else 1.0            # px -> nm when calibrated
    xi = {"up": xu_px * _sc, "down": xd_px * _sc}
    # all width methods (reuse mm for the skeleton, xi_px for the FFT) ->
    # readout shows the one picked by params["width_method"], export writes
    # the full comparison.
    widths = domain_widths(mz, roi, um_per_px=um, min_px=params.get("min_px", 12),
                           mm=mm, xi_px=(xu_px, xd_px))
    # the boundary overlay (find_contours / chan_vese) is only drawn when the
    # "show boundary" checkbox is on, so only compute it then -- it is ~0.4s on
    # a multi-MP frame and would otherwise slow every threshold-slider preview.
    if params.get("show_boundary", False):
        boundary = _compute_boundary(layers, roi, params)
    else:
        boundary = {"edge": None, "contours": None}
    return {
        "img": prep["img"], "mz": mz, "roi": roi, "layers": layers,
        "um": um, "fov_warn": warn, "shape": shape,
        "pdf": pl, "morphometry": mm, "counts": counts, "params": params,
        "area_avg_mz": aa, "net_mz_raw": hy, "xi": xi, "widths": widths,
        "boundary": boundary,
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
                     params.get("tv_weight", 0.12),
                     params.get("denoise_method", "tv"),
                     params.get("input_mode", "grayscale"),
                     params.get("map_mode", "rb"))
        # a pre-coded map is already centered -> ignore any background region
        if (bg_mask is not None and bg_mask.any()
                and params.get("input_mode", "grayscale") != "asymmetry_map"):
            flat = prep["den"] - float(prep["den"][bg_mask].mean())
        else:
            flat = prep["flat"]
        layers = {"img": prep["img"], "flat": flat, "roi": roi_mask}
    else:
        _, layers = analyze_xmcd(path, clahe=clahe, roi_mask=roi_mask, bg_mask=bg_mask,
                                 connectivity=params.get("connectivity", 1),
                                 tv_weight=params.get("tv_weight", 0.12),
                                 denoise_method=params.get("denoise_method", "tv"),
                                 input_mode=params.get("input_mode", "grayscale"),
                                 map_mode=params.get("map_mode", "rb"))
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
    # per-state correlation length (up/down domain size), robust to tv_weight.
    xi = {"up": correlation_length_state(mz, roi, 1, um_per_px=um),
          "down": correlation_length_state(mz, roi, -1, um_per_px=um)}
    widths = domain_widths(mz, roi, um_per_px=um, min_px=params.get("min_px", 12), mm=mm)
    boundary = _compute_boundary(layers, roi, params)

    return {
        "img": layers["img"], "mz": mz, "roi": roi, "layers": layers,
        "um": um, "fov_warn": fov_warn, "shape": shape,
        "pdf": pl, "morphometry": mm, "counts": counts, "params": params,
        "area_avg_mz": aa, "net_mz_raw": hy, "xi": xi, "widths": widths,
        "boundary": boundary,
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
    period = mm.get("domain_period_nm", mm["domain_period"]) if um else mm["domain_period"]
    # primary width comes from the SELECTED method (params["width_method"]);
    # all methods are shown compactly beneath it for side-by-side comparison.
    widths = res.get("widths", {})
    method = res.get("params", {}).get("width_method", "perimeter_area")
    _abbr = {"perimeter_area": "2A/P", "skeleton_mean": "sk-mean",
             "skeleton_median": "sk-med", "correlation_length": "xi"}
    _f = lambda v: f"{v:.1f}" if v == v else "n/a"   # NaN-safe
    sel = widths.get(method, {"up": float("nan"), "down": float("nan")})
    lines = [
        f"active area  = {area:.1f} {'um^2' if um else 'px'} ({npx} px)",
        f"P(up/0/dn)   = {pl['p_up']:.3f} / {pl['p_zero']:.3f} / {pl['p_down']:.3f}",
        f"net <m_z>    = {pl['net_m']:+.3f}  (discrete)",
        (f"net <m_z> raw= {hy['mean_mz']:+.3f}  (excl. gray, N={hy['N_pixels']})"
         if hy["mean_mz"] == hy["mean_mz"] else
         "net <m_z> raw= n/a (no px cleared threshold)"),
        f"area-avg<m_z>= {aa['mean_mz']:+.3f}  (pre-thr, N={aa['N_pixels']})",
        f"w+/w- [{_abbr.get(method, method)}] = {_f(sel['up'])} / {_f(sel['down'])} {w_unit}  (selected)",
    ]
    if widths:
        lines.append("  all methods [" + w_unit + "]:")
        for m in ("perimeter_area", "skeleton_mean", "skeleton_median", "correlation_length"):
            w = widths.get(m)
            if w:
                mark = " *" if m == method else "  "
                lines.append(f"   {_abbr[m]:>7}{mark} up={_f(w['up'])} dn={_f(w['down'])}")
    lines.append(f"period ~ {period:.1f} {w_unit}")
    lines += [
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
