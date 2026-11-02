"""Live preview and analysis inside a drawn polygon. Qt-free
service layer used by the GUI and the CLI."""
import os

import numpy as np
from scipy import ndimage as ndi

from ..core import (
    analyze_xmcd,
    three_state_field,
    despeckle_mz,
    pdf_three_state,
    area_averaged_mz,
    net_mz_raw,
    mz_four_ways,
    correlation_length_state,
    domain_widths,
    per_state_morphometry,
    domain_counts_areas,
    domain_boundary,
    asymmetry_field,
)
from ..core.offset import sym_centre
from ..core.regions import polygon_to_mask, resolve_um


def _domain_areas(mz, state, min_px=12, connectivity=1):
    """Per-domain pixel areas for a given state (+1 up / -1 down)."""
    from xmcd_toolkit import neighbor_structure
    lbl, n = ndi.label(mz == state, structure=neighbor_structure(connectivity))
    if not n:
        return np.array([])
    sizes = np.bincount(lbl.ravel())[1:]
    return sizes[sizes >= min_px].astype(float)


# ----- image-level cache: the expensive denoise/flatten is polygon- and
#       threshold-independent, so compute it once per (path, clahe). -----
_PREP_CACHE = {}


_PREP_ORDER = []


def _prep(path, clahe, flatten=True, flatten_sigma=25.0, tv_weight=0.12,
          denoise_method="tv", input_mode="grayscale", map_mode="rb", invert=False):
    # The cache key tracks the state of every toggle/slider that feeds this
    # expensive step. tv_weight/denoise_method belong here (not in
    # preview_polygon) because they drive the denoise inside analyze_xmcd --
    # the same "compute once, cache" step as flatten_sigma. Changing any is a
    # cache miss, so it re-runs the full (slow) denoise. input_mode/map_mode
    # change how the image is decoded, so they belong in the key too.
    key = (os.path.abspath(path), bool(clahe), bool(flatten), float(flatten_sigma),
           float(tv_weight), str(denoise_method), str(input_mode), str(map_mode), bool(invert))
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
        # Igor export decoded; invert swaps up / down
        from ..core.igor_import import source_for
        picture = picture_of(path, input_mode)          # the file as it is (never inverted)
        src, input_mode = source_for(path, input_mode, map_mode, invert)
        _, layers = analyze_xmcd(src, clahe=clahe, flatten_sigma=sigma_val,
                                 tv_weight=float(tv_weight),
                                 denoise_method=str(denoise_method),
                                 input_mode=str(input_mode), map_mode=str(map_mode),
                                 roi_box=(0, 0, 10**9, 10**9))

        if picture.shape[:2] != layers["img"].shape:   # raw grayscale with the acquisition frame cut off
            picture = np.repeat((np.clip(layers["img"], 0, 1) * 255).astype(np.uint8)[..., None], 3, -1)
        _PREP_CACHE[key] = {"img": layers["img"], "flat": layers["flat"], "picture": picture,
                            "den": layers["denoised"],
                            "is_map": input_mode == "asymmetry_map"}  # after the Igor decode
        _PREP_ORDER.append(key)
        while len(_PREP_ORDER) > 2:      # cap memory: ~1 GB per entry for a 41 MP image
            _PREP_CACHE.pop(_PREP_ORDER.pop(0), None)
    return _PREP_CACHE[key]


def picture_of(path, input_mode="asymmetry_map"):
    """The picture as the file shows it (uint8 RGB), on the grid the analysis uses; no analysis."""
    if input_mode == "igor_export":
        from ..core.igor_import import sample_grid
        return sample_grid(path)[0]
    from ..core.loaders import load_picture
    return load_picture(path)


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


def _bbox(mask, pad=1):
    """Slices of the bounding box of `mask` grown by `pad` pixels (the whole frame if the mask is empty)."""
    rows, cols = np.nonzero(mask.any(1))[0], np.nonzero(mask.any(0))[0]
    if not rows.size:
        return (slice(None), slice(None))
    return (slice(max(rows[0] - pad, 0), rows[-1] + pad + 1), slice(max(cols[0] - pad, 0), cols[-1] + pad + 1))


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
                 params.get("map_mode", "rb"),
                 params.get("invert", False))
    shape = prep["flat"].shape
    roi_mask = polygon_to_mask(poly, shape) if poly else np.ones(shape, bool)
    um, warn = resolve_um(params, shape)

    # background region: set the magnetic zero from its mean (same rule as
    # analyze_xmcd's bg path), computed on the cached denoised image so the
    # preview matches the full export exactly.
    # The same for a map: a drawn background region sets its zero (otherwise the
    # map's own zero, mid-grey / white, is used).
    bg_poly = params.get("bg_polygon")
    bg = polygon_to_mask(bg_poly, shape) if bg_poly else None
    centre = params.get("sym_centre", "0")
    zero = {"source": "file zero", "value": 0.0}
    if centre in ("median", "mean") and prep["is_map"]:          # TensorSpec Sym ± centre
        c, _, where = sym_centre(prep["den"], centre, bg)
        zero = {"source": f"{centre}, {where}", "value": c}
        flat = prep["den"] - c
    elif bg is not None and bg.any():
        zero = {"source": "background region mean", "value": float(prep["den"][bg].mean())}
        flat = prep["den"] - zero["value"]
    else:
        flat = prep["flat"]

    layers = {"img": prep["img"], "flat": flat, "roi": roi_mask}
    mz, roi = three_state_field(layers, z_neutral=z_neutral, m_threshold=m_threshold)
    # up / down exist only inside the active area, so cleaning and counting domains on its bounding box (+1 px)
    # gives the same numbers as on the whole frame, much faster on a large image
    sl = _bbox(roi)
    mz_c = despeckle_mz(mz[sl], min_px=params.get("min_px", 12))
    mz = np.zeros_like(mz)
    mz[sl] = mz_c
    pl = pdf_three_state(mz, region=roi)
    # widths / periods / correlation length are the slow part on a large image (skeleton + distance transform
    # of every pixel: ~30 s at 6415 x 6415); measured only when asked (params["measure_widths"]); the export
    # always measures them.
    measure = params.get("measure_widths", True) is not False
    mm = per_state_morphometry(mz, um_per_px=um, min_px=params.get("min_px", 12)) if measure else None
    counts = domain_counts_areas(mz[sl], roi=roi[sl], min_px=params.get("min_px", 12),
                                 connectivity=int(params.get("connectivity", 1)), um_per_px=um)
    # area-averaged <m_z> from the continuous asymmetry field, BEFORE
    # thresholding -- cheap (no bootstrap), so always available live.
    aa = area_averaged_mz(layers)
    # raw signal averaged over only the classified (non-gray-band) pixels --
    # see net_mz_raw's docstring for why this sits between the discrete
    # (pl["net_m"]) and continuous (aa) definitions.
    hy = net_mz_raw(mz, layers)
    four = mz_four_ways(mz, layers, roi)
    # per-state correlation length: the up/down domain SIZE, robust to the
    # tv_weight/threshold choice (FFT autocorrelation, no skeleton).
    # compute the FFT correlation length ONCE (in px) and reuse it for both the
    # per-state xi readout and the correlation_length width method, instead of
    # running the same FFT autocorrelation 4x per preview.
    if measure:
        xu_px = correlation_length_state(mz, roi, 1, um_per_px=None)
        xd_px = correlation_length_state(mz, roi, -1, um_per_px=None)
        _sc = (um * 1000.0) if um else 1.0            # px -> nm when calibrated
        xi = {"up": xu_px * _sc, "down": xd_px * _sc}
    else:
        xi = None
    # all width methods (reuse mm for the skeleton, xi_px for the FFT) ->
    # readout shows the one picked by params["width_method"], export writes
    # the full comparison.
    widths = domain_widths(mz, roi, um_per_px=um, min_px=params.get("min_px", 12),
                           mm=mm, xi_px=(xu_px, xd_px)) if measure else {}
    # the boundary overlay (find_contours / chan_vese) is only drawn when the
    # "show boundary" checkbox is on, so only compute it then -- it is ~0.4s on
    # a multi-MP frame and would otherwise slow every threshold-slider preview.
    if params.get("show_boundary", False):
        boundary = _compute_boundary(layers, roi, params)
    else:
        boundary = {"edge": None, "contours": None}
    return {
        "img": prep["img"], "picture": prep["picture"], "mz": mz, "roi": roi, "layers": layers,
        "um": um, "fov_warn": warn, "shape": shape,
        "pdf": pl, "morphometry": mm, "counts": counts, "params": params,
        "area_avg_mz": aa, "net_mz_raw": hy, "mz_four": four, "zero": zero, "xi": xi, "widths": widths,
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

    from ..core.igor_import import analysis_shape
    shape = analysis_shape(path, params.get("input_mode", "grayscale"))
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
                     params.get("map_mode", "rb"),
                     params.get("invert", False))
        # a pre-coded map is already centered -> ignore any background region
        if params.get("sym_centre", "0") in ("median", "mean") and prep["is_map"]:
            flat = prep["den"] - sym_centre(prep["den"], params["sym_centre"], bg_mask)[0]
        elif bg_mask is not None and bg_mask.any():
            flat = prep["den"] - float(prep["den"][bg_mask].mean())
        else:
            flat = prep["flat"]
        layers = {"img": prep["img"], "flat": flat, "roi": roi_mask}
    else:
        from ..core.igor_import import source_for
        src, input_mode = source_for(path, params.get("input_mode", "grayscale"), params.get("map_mode", "auto"),
                                     params.get("invert", False))
        _, layers = analyze_xmcd(src, clahe=clahe, roi_mask=roi_mask, bg_mask=bg_mask,
                                 connectivity=params.get("connectivity", 1),
                                 tv_weight=params.get("tv_weight", 0.12),
                                 denoise_method=params.get("denoise_method", "tv"),
                                 input_mode=input_mode,
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
    if mm is None:
        period = float("nan")
    else:
        period = mm.get("domain_period_nm", mm["domain_period"]) if um else mm["domain_period"]
    # primary width comes from the SELECTED method (params["width_method"]);
    # all methods are shown compactly beneath it for side-by-side comparison.
    widths = res.get("widths", {})
    method = res.get("params", {}).get("width_method", "perimeter_area")
    _abbr = {"perimeter_area": "2A/P", "skeleton_mean": "sk-mean",
             "skeleton_median": "sk-med", "correlation_length": "xi"}
    _f = lambda v: f"{v:.1f}" if v == v else "n/a"   # NaN-safe
    sel = widths.get(method, {"up": float("nan"), "down": float("nan")})
    four = res.get("mz_four")
    m0 = res.get("params", {}).get("m_threshold")
    lines = [f"active area  = {area:.1f} {'um^2' if um else 'px'} ({npx} px)"]
    if res.get("zero"):
        lines.append(f"zero         = {res['zero']['value']:+.4f} ({res['zero']['source']})")
    if four:
        _s = lambda v: f"{v:+.3f}" if v == v else "  n/a "
        lines += [
            f"pixels  up   = {four['N_up']}",
            f"        0    = {four['N_0']}",
            f"        down = {four['N_down']}",
            f"P(up/0/dn)   = {pl['p_up']:.3f} / {pl['p_zero']:.3f} / {pl['p_down']:.3f}",
            f"<m_z>  (m_0 = {m0})" if m0 is not None else "<m_z>",
            "            m_z(m_0)    m_z",
            f"discrete    {_s(four['discrete_m0'])}     {_s(four['discrete'])}",
            f"continuous  {_s(four['continuous_m0'])}     {_s(four['continuous'])}",
            f"up vs down  {_s(four['up_vs_down'])}  (no 0 px)",
        ]
    else:
        lines += [
            f"P(up/0/dn)   = {pl['p_up']:.3f} / {pl['p_zero']:.3f} / {pl['p_down']:.3f}",
            f"net <m_z>    = {pl['net_m']:+.3f}  (discrete)",
            (f"net <m_z> raw= {hy['mean_mz']:+.3f}  (excl. gray, N={hy['N_pixels']})"
             if hy["mean_mz"] == hy["mean_mz"] else
             "net <m_z> raw= n/a (no px cleared threshold)"),
            f"area-avg<m_z>= {aa['mean_mz']:+.3f}  (pre-thr, N={aa['N_pixels']})",
        ]
    lines += [
        (f"w+/w- [{_abbr.get(method, method)}] = {_f(sel['up'])} / {_f(sel['down'])} {w_unit}  (selected)"
         if mm is not None else "widths       = not measured (tick 'Measure domain widths', step 5)"),
    ]
    if widths:
        lines.append("  all methods [" + w_unit + "]:")
        for m in ("perimeter_area", "skeleton_mean", "skeleton_median", "correlation_length"):
            w = widths.get(m)
            if w:
                mark = " *" if m == method else "  "
                lines.append(f"   {_abbr[m]:>7}{mark} up={_f(w['up'])} dn={_f(w['down'])}")
    if mm is not None:
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
