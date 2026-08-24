"""
morphometry.py -- domain WIDTH and LENGTH from distance transform + skeleton.

Why this exists
---------------
For labyrinthine / stripe domains, connected-component "area per domain" is
unreliable: the maze fuses into one blob, so mean area is meaningless
(mean >> median). The physically robust size of a stripe pattern is its
WIDTH, which is independent of how the maze connects.

Method (HIPR2: Distance Transform + Skeletonization; the "local thickness"
construction):
    1. distance transform DT(x) = distance from each domain pixel to the
       nearest domain boundary.
    2. skeleton (medial axis) of the domain.
    3. local half-width at a skeleton pixel = DT there; WIDTH = 2 * DT.
       The distribution of 2*DT over the skeleton is the domain-width
       distribution; its median is the characteristic domain width.

Also:
    - skeleton length  = number of skeleton pixels (diagonal-corrected),
      i.e. the total "spine" length of the domains of a given sign.
    - domain period    ~ up_width + down_width  (for a two-phase stripe state).

Depends only on numpy / scipy / scikit-image.
"""

import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

from .distributions import fit_lognormal, fit_gaussian, compare_lognormal_gaussian
from .loaders import neighbor_structure, _remove_small
from .magnetization import area_averaged_mz
from .colors import (
    COLOR_UP_FILL as _COLOR_UP_FILL, COLOR_UP_LINE as _COLOR_UP_LINE,
    COLOR_DOWN_FILL as _COLOR_DOWN_FILL, COLOR_DOWN_LINE as _COLOR_DOWN_LINE,
    CMAP3_HEX,
)


# diagonal vs orthogonal skeleton-step length correction
_DIAG = np.sqrt(2.0)


# ----------------------------------------------------------------------
# Shared publication style for the width-related export figures (the width
# distribution, the width/area-vs-count histograms, and the morphometry
# panels). Kept in one place so all of them stay consistent.
_PUB_RC = {
    "font.family": "Arial",
    "font.size": 12,
    "font.weight": "normal",
    "axes.linewidth": 0.25,
    "xtick.major.width": 0.25,
    "ytick.major.width": 0.25,
    "xtick.minor.width": 0.25,
    "ytick.minor.width": 0.25,
    # keep text as real, editable <text> elements tagged font-family:Arial
    # in the SVG, instead of matplotlib's default of baking each character
    # into a vector outline drawn from whatever font this machine actually
    # has (this sandbox doesn't have Arial, so without this the SVG would
    # silently render in DejaVu Sans instead).
    "svg.fonttype": "none",
}

# _COLOR_UP_FILL / _COLOR_UP_LINE / _COLOR_DOWN_FILL / _COLOR_DOWN_LINE are
# imported from .colors above (single source of truth for the whole
# toolkit) -- up/down bars use a lighter fill so overlapping/adjacent bars
# stay easy to read; the log-normal fit line uses a darker shade of the
# same hue so it's still visible sitting on top of the lighter fill.


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


def _save_lognormal_fits(fits, out_path, unit):
    """Save the fitted log-normal parameters next to the figure -- the plot
    itself only shows them to 1-2 significant figures in the legend."""
    import json
    payload = {"unit": unit}
    for name, fit in fits.items():
        if fit is None:
            continue
        payload[name] = {k: fit[k] for k in
                         ("n", "median", "mode", "sigma_log", "mu_log",
                          "aic", "bic", "ks_stat", "ks_p")}
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    return out_path


def _save_fit_comparison(comparisons, out_path, unit):
    """Save a log-normal vs. Gaussian fit comparison (AIC/BIC/KS per state)
    as JSON. `comparisons` is {"up": cmp_or_None, "down": cmp_or_None} as
    returned by compare_lognormal_gaussian() for each state."""
    import json
    payload = {"unit": unit}
    for name, cmp in comparisons.items():
        if not cmp:
            continue
        entry = {}
        ln, ga = cmp.get("lognormal"), cmp.get("gaussian")
        if ln:
            entry["lognormal"] = {k: ln[k] for k in
                                  ("n", "median", "mode", "sigma_log", "mu_log",
                                   "aic", "bic", "ks_stat", "ks_p")}
        if ga:
            entry["gaussian"] = {k: ga[k] for k in
                                 ("n", "mean", "sd", "aic", "bic",
                                  "ks_stat", "ks_p")}
        entry["delta_aic"] = cmp.get("delta_aic")
        entry["delta_bic"] = cmp.get("delta_bic")
        entry["preferred"] = cmp.get("preferred")
        payload[name] = entry
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    return out_path


def _skeleton_length_px(skel):
    """Approximate skeleton length: count steps between adjacent skeleton
    pixels, weighting diagonal neighbours by sqrt(2)."""
    if skel.sum() < 2:
        return float(skel.sum())
    ortho = 0
    diag = 0
    s = skel.astype(bool)
    # orthogonal adjacencies
    ortho += np.logical_and(s[:, :-1], s[:, 1:]).sum()
    ortho += np.logical_and(s[:-1, :], s[1:, :]).sum()
    # diagonal adjacencies
    diag += np.logical_and(s[:-1, :-1], s[1:, 1:]).sum()
    diag += np.logical_and(s[:-1, 1:], s[1:, :-1]).sum()
    # each adjacency is half an edge per endpoint; length ~ sum of edge lengths
    return float(ortho + _DIAG * diag)


def domain_width(mask, um_per_px=None, min_px=0):
    """
    Width statistics of a binary domain mask via distance-transform-on-skeleton.

    min_px: drop connected fragments smaller than this before measuring width
    (same "Min Domain Size" the GUI/CLI use for area filtering elsewhere).
    Real experimental frames carry per-pixel classification noise -- isolated
    specks a few pixels across that the neutral-state threshold missed --
    and left unfiltered they contribute spurious near-zero-width skeleton
    points that drag the whole distribution down. 0 (default) disables this
    and reproduces the previous behavior exactly.

    Returns dict with width mean/median/SD (px and um if calibrated), the
    skeleton length, the area, and the raw width samples; plus the DT and
    skeleton arrays for plotting.
    """
    mask = mask.astype(bool)
    if min_px:
        # _remove_small (numpy/scipy bincount) instead of skimage's
        # remove_small_objects: skimage 0.26 deprecated the min_size
        # parameter (removal planned for 2.0) in favor of max_size, whose
        # semantics are subtly different at the boundary (keeps area >
        # max_size, not >= min_size as here) -- a mechanical min_size ->
        # max_size rename would silently drop domains of exactly min_px
        # pixels instead of keeping them. This keeps the >= convention
        # used everywhere else in the toolkit and doesn't depend on
        # skimage's parameter naming at all. connectivity=1 matches
        # remove_small_objects' own default (4-connected).
        mask = _remove_small(mask, min_px, connectivity=1)
    out = {"n_px": int(mask.sum())}
    if mask.sum() < 4:
        out.update({"width_mean": np.nan, "width_median": np.nan,
                    "width_sd": np.nan, "skeleton_len_px": 0.0,
                    "widths_px": np.array([]), "dt": None, "skeleton": None,
                    "units": "um" if um_per_px else "px"})
        return out

    # Crop to the mask's bounding box (pad 1 so a background border remains,
    # keeping the distance transform identical) before the distance-transform
    # and skeletonize. Both are O(pixels); when the domains occupy only part of
    # a large frame (a polygon ROI in a multi-MP image) this avoids processing
    # the empty margin. dt/skel are scattered back to full size so callers
    # (width map, skeleton overlay) are unaffected.
    ys, xs = np.where(mask)
    y0, y1 = max(ys.min() - 1, 0), min(ys.max() + 2, mask.shape[0])
    x0, x1 = max(xs.min() - 1, 0), min(xs.max() + 2, mask.shape[1])
    sub = mask[y0:y1, x0:x1]
    dt_sub = ndi.distance_transform_edt(sub)
    skel_sub = skeletonize(sub)
    dt = np.zeros(mask.shape); dt[y0:y1, x0:x1] = dt_sub
    skel = np.zeros(mask.shape, bool); skel[y0:y1, x0:x1] = skel_sub
    widths_px = 2.0 * dt[skel]                     # local full width on the spine
    skel_len = _skeleton_length_px(skel)

    s = (um_per_px or 1.0)
    out.update({
        "width_mean": float(widths_px.mean() * s),
        "width_median": float(np.median(widths_px) * s),
        "width_sd": float(widths_px.std() * s),
        "skeleton_len_px": skel_len,
        "skeleton_len": float(skel_len * s),
        "area_px": int(mask.sum()),
        "area": float(mask.sum() * s * s) if um_per_px else float(mask.sum()),
        "widths_px": widths_px,
        "dt": dt, "skeleton": skel,
        "units": "um" if um_per_px else "px",
    })
    if um_per_px:
        # nm is the natural unit for XMCD-PEEM domain widths (typically tens
        # to hundreds of nm) -- um numbers here would mostly read as 0.0xx.
        s_nm = um_per_px * 1000.0
        out.update({
            "width_mean_nm": float(widths_px.mean() * s_nm),
            "width_median_nm": float(np.median(widths_px) * s_nm),
            "width_sd_nm": float(widths_px.std() * s_nm),
            "widths_nm": widths_px * s_nm,
        })
    return out


def _local_period_px(up, dn):
    """
    Per-skeleton-pixel LOCAL domain period, as a genuine sample (not the
    single scalar up_median + down_median that domain_period is).

    At every up-skeleton pixel: period = that pixel's own local up-width
    (2*DT there) + the local down-width at the NEAREST down-skeleton pixel
    (found via a distance-transform nearest-index lookup). Symmetrically,
    every down-skeleton pixel gets its own down-width + the nearest
    up-skeleton pixel's width. This gives a spatially local period sample
    at every point along the stripe/maze, without assuming a 1D scanline
    direction or a 1:1 domain pairing (which don't generalize to a
    labyrinthine, non-stripe pattern).

    Returns {"up": array_from_up_skeleton_pixels,
             "down": array_from_down_skeleton_pixels} (px, empty array if
    either state has no skeleton).
    """
    out = {"up": np.array([]), "down": np.array([])}
    up_skel, dn_skel = up.get("skeleton"), dn.get("skeleton")
    up_dt, dn_dt = up.get("dt"), dn.get("dt")
    if up_skel is None or dn_skel is None:
        return out
    if not up_skel.any() or not dn_skel.any():
        return out

    # distance_transform_edt(~skel, return_indices=True) gives, for every
    # pixel, the (row, col) of the nearest True pixel in `skel`.
    _, (dn_iy, dn_ix) = ndi.distance_transform_edt(~dn_skel, return_indices=True)
    _, (up_iy, up_ix) = ndi.distance_transform_edt(~up_skel, return_indices=True)

    uy, ux = np.nonzero(up_skel)
    out["up"] = 2.0 * up_dt[uy, ux] + 2.0 * dn_dt[dn_iy[uy, ux], dn_ix[uy, ux]]

    dy, dx = np.nonzero(dn_skel)
    out["down"] = 2.0 * dn_dt[dy, dx] + 2.0 * up_dt[up_iy[dy, dx], up_ix[dy, dx]]

    return out


def per_state_morphometry(mz, um_per_px=None, min_px=0):
    """
    Domain width + skeleton length for up (+1) and down (-1) states, plus:
      - the characteristic scalar domain period ~ up_median_width +
        down_median_width (domain_period / domain_period_nm, as before), and
      - a full per-skeleton-pixel period SAMPLE for each state (periods_px /
        periods_nm inside up/down, see _local_period_px), so the period can
        be fit/Q-Q'd/compared the same way widths are, not just summarized
        by one number.

    min_px is forwarded to domain_width (see there) so the same "Min Domain
    Size" control used for area filtering also governs the width statistic.
    """
    up = domain_width(mz == 1, um_per_px=um_per_px, min_px=min_px)
    dn = domain_width(mz == -1, um_per_px=um_per_px, min_px=min_px)

    per_px = _local_period_px(up, dn)
    up["periods_px"] = per_px["up"]
    dn["periods_px"] = per_px["down"]
    if um_per_px:
        s_nm = um_per_px * 1000.0
        up["periods_nm"] = per_px["up"] * s_nm
        dn["periods_nm"] = per_px["down"] * s_nm

    period = np.nan
    if np.isfinite(up["width_median"]) and np.isfinite(dn["width_median"]):
        period = up["width_median"] + dn["width_median"]
    out = {"up": up, "down": dn, "domain_period": period,
            "units": "um" if um_per_px else "px"}
    if um_per_px and np.isfinite(period):
        out["domain_period_nm"] = period * 1000.0
    return out


def _distribution_stats(values):
    """
    mean / median / mode / variance for a positive-valued sample (domain
    width or period). Mode is taken from the FITTED log-normal
    (exp(mu_log - sigma_log^2)) rather than a histogram peak, so it isn't
    sensitive to bin width/count and stays consistent with the median and
    sigma_log already reported by fit_lognormal elsewhere in this module.
    variance is of the raw (native-scale) sample, i.e. Var(X) not Var(ln X)
    -- sigma_log (= SD of ln X) is included alongside it so both are
    available.

    Returns None if there are fewer than 5 positive samples (same floor as
    fit_lognormal/fit_gaussian).
    """
    v = np.asarray(values, float)
    v = v[v > 0]
    if v.size < 5:
        return None
    fit = fit_lognormal(v)
    if fit is None:
        return None
    return {"n": int(v.size), "mean": float(v.mean()),
            "median": float(np.median(v)), "mode": fit["mode"],
            "variance": float(v.var()), "sigma_log": fit["sigma_log"]}


def domain_counts_areas(mz, roi=None, min_px=0, connectivity=1, um_per_px=None):
    """
    Per-state (up / down / neutral) domain COUNT and AREA.

    'neutral' (mz==0) is restricted to `roi` when given: three_state_field
    only ever sets +1/-1 inside the ROI, but leaves everything else --
    including the region OUTSIDE the ROI -- at 0. Without the roi mask,
    labeling mz==0 directly would merge genuine in-ROI neutral/wall pixels
    into one giant connected component with the entire exterior background,
    corrupting both the neutral domain count and its area. up/down are
    unaffected by roi (they're already ROI-confined by construction).

    min_px filters which domains COUNT toward n_domains and appear in
    areas_px (same "Min Domain Size" control used elsewhere), but
    total_area_px is always the full, unfiltered pixel count of that state
    -- so "how much of the flake is up/down/neutral" doesn't shrink just
    because some small speckle domains get excluded from the count.

    Returns {"up": {...}, "down": {...}, "neutral": {...}}, each with:
        n_domains, total_area_px, areas_px (per-domain, filtered by min_px),
        and (if um_per_px given) total_area_um2, areas_um2.
    """
    a2 = (um_per_px * um_per_px) if um_per_px else None
    fg = neighbor_structure(connectivity)
    bg = neighbor_structure(2 if connectivity == 1 else 1)
    out = {}
    for name, s in (("up", 1), ("down", -1), ("neutral", 0)):
        mask = (mz == s)
        if s == 0 and roi is not None:
            mask = mask & roi
        struct = bg if s == 0 else fg
        lbl, _ = ndi.label(mask, structure=struct)
        sizes = np.bincount(lbl.ravel())[1:].astype(float) if lbl.max() else np.array([])
        sizes = sizes[sizes >= (min_px or 1)]
        entry = {"n_domains": int(sizes.size),
                 "total_area_px": int(mask.sum()),
                 "areas_px": sizes}
        if a2:
            entry["total_area_um2"] = float(mask.sum() * a2)
            entry["areas_um2"] = sizes * a2
        out[name] = entry
    return out


def export_domain_stats(mz, roi=None, um_per_px=None, min_px=0, connectivity=1,
                         out_stem="domain_stats"):
    """
    Consolidated per-state summary -- counts, areas, and (for up/down)
    mean/median/mode/variance of both the domain-WIDTH and local
    domain-PERIOD samples, in px and (if calibrated) nm/um^2.

    Writes:
        <out_stem>.json               -- the summary dict below
        <out_stem>_areas_{up,down,neutral}.csv   -- per-domain area (px[, um^2])
        <out_stem>_widths_{up,down}.csv          -- per-skeleton-pixel width (px[, nm])
        <out_stem>_periods_{up,down}.csv         -- per-skeleton-pixel period (px[, nm])

    Returns (summary_dict, list_of_paths).
    """
    import json, csv

    mm = per_state_morphometry(mz, um_per_px=um_per_px, min_px=min_px)
    ca = domain_counts_areas(mz, roi=roi, min_px=min_px, connectivity=connectivity,
                              um_per_px=um_per_px)
    s_nm = (um_per_px * 1000.0) if um_per_px else None

    summary = {"unit_len": "nm" if um_per_px else "px",
               "unit_area": "um^2" if um_per_px else "px"}
    paths = []

    def _write_csv(path, cols_px, name_px, arr_nm=None, name_nm=None):
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            header = [name_px] + ([name_nm] if arr_nm is not None else [])
            w.writerow(header)
            for i, v in enumerate(cols_px):
                row = [v] + ([arr_nm[i]] if arr_nm is not None else [])
                w.writerow(row)
        paths.append(path)

    for name in ("up", "down"):
        entry = {"n_domains": ca[name]["n_domains"],
                 "total_area_px": ca[name]["total_area_px"]}
        if "total_area_um2" in ca[name]:
            entry["total_area_um2"] = ca[name]["total_area_um2"]

        w_px = mm[name]["widths_px"]
        entry["width_px"] = _distribution_stats(w_px)
        if s_nm is not None:
            entry["width_nm"] = _distribution_stats(w_px * s_nm)

        p_px = mm[name]["periods_px"]
        entry["period_px"] = _distribution_stats(p_px)
        if s_nm is not None:
            entry["period_nm"] = _distribution_stats(p_px * s_nm)

        summary[name] = entry

        _write_csv(f"{out_stem}_areas_{name}.csv", ca[name]["areas_px"], "area_px",
                   ca[name].get("areas_um2"), "area_um2")
        _write_csv(f"{out_stem}_widths_{name}.csv", w_px, "width_px",
                   (w_px * s_nm) if s_nm is not None else None, "width_nm")
        _write_csv(f"{out_stem}_periods_{name}.csv", p_px, "period_px",
                   (p_px * s_nm) if s_nm is not None else None, "period_nm")

    summary["neutral"] = {"n_domains": ca["neutral"]["n_domains"],
                          "total_area_px": ca["neutral"]["total_area_px"]}
    if "total_area_um2" in ca["neutral"]:
        summary["neutral"]["total_area_um2"] = ca["neutral"]["total_area_um2"]
    _write_csv(f"{out_stem}_areas_neutral.csv", ca["neutral"]["areas_px"], "area_px",
               ca["neutral"].get("areas_um2"), "area_um2")

    out_json = f"{out_stem}.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    paths.append(out_json)

    return summary, paths


def summarize_domains(mz, roi, um_per_px=None, min_px=12, connectivity=1, layers=None, mm=None):
    """
    One consolidated table -- up / down / total -- instead of hunting the
    same numbers across domain_stats.json, summary.txt, and
    flake_metrics.csv separately. Each row has:

        n_domains    -- domain_counts_areas() count (3-state, m_threshold
                        classification; NOT analyze_xmcd's separate 2-state
                        Otsu count used for metrics["n_domains_*"] --
                        the two are different classifications and don't
                        agree by design, see per_state_domain_stats).
        area_px / area_um2 (if calibrated) -- total area of that state
        width_median / width_median_nm  -- median of the per-skeleton-pixel
                        width sample (domain_width's robust size measure)
        width_mode / width_mode_nm      -- log-normal-fitted mode of the
                        same sample (fit_lognormal; None if < 5 skeleton
                        pixels)
        width_fit / period_fit  -- the FULL log-normal fit on that state's
                        per-skeleton-pixel width / local-period sample (see
                        domain_width, _local_period_px) -- NOT the per-domain
                        equivalent-diameter fit make_distribution_figure
                        shows (that one is unreliable for a fused labyrinth
                        maze, see METHODS.md Sec 11/14). {n, median, mode,
                        sigma_log, mu_log, aic, bic, ks_stat, ks_p} in px, or
                        None if < 5 samples.
        mz           -- up/down rows: that state's AREA FRACTION of the
                        total up+down area (f_up, f_down; sums to 1).
                        total row: the discrete net <m_z> = (A_up-A_down)/
                        (A_up+A_down), same definition as pdf_three_state's
                        net_m.
        period_px / period_nm  -- the single combined domain_period
                        (up_width_median + down_width_median), repeated on
                        every row for convenience since it isn't a
                        per-state quantity.

    "total" width_median/width_mode/width_fit/period_fit are fit to the
    POOLED up+down sample (not derived from the two per-state values), so
    they're a genuine combined statistic, not up_median + down_median
    (that's what period_px/period_nm already is).

    If `layers` is given (needs layers["flat"]/["roi"]), the total row also
    gets "area_avg_mz": the continuous, pre-threshold area-averaged
    magnetization (area_averaged_mz) -- absent (not None) if layers isn't
    passed, since it isn't derivable from mz/roi alone.

    `mm`: pass a per_state_morphometry(mz, um_per_px=um_per_px, min_px=...)
    result the caller already computed (e.g. for make_morphometry_figure)
    to reuse it here instead of refitting every log-normal a second time on
    a second min_px-filtered pass -- the two MUST use the same min_px or
    the width/period numbers here won't match whatever the caller's own mm
    reports (this is the exact mismatch this parameter exists to prevent).
    Computed internally if not given.
    """
    def _fit_summary(fit):
        """The scalar fields of a fit_lognormal() result, dropping the
        pdf_x/pdf_y plotting arrays -- what's worth persisting to a summary,
        not what's worth persisting to redraw the curve."""
        if fit is None:
            return None
        return {k: fit[k] for k in
                ("n", "median", "mode", "sigma_log", "mu_log", "aic", "bic",
                 "ks_stat", "ks_p")}

    counts = domain_counts_areas(mz, roi=roi, min_px=min_px, connectivity=connectivity,
                                 um_per_px=um_per_px)
    if mm is None:
        mm = per_state_morphometry(mz, um_per_px=um_per_px, min_px=min_px)
    cu, cd = counts["up"], counts["down"]

    tot_area = cu["total_area_px"] + cd["total_area_px"]
    f_up = cu["total_area_px"] / tot_area if tot_area else float("nan")
    f_down = cd["total_area_px"] / tot_area if tot_area else float("nan")
    net_m = ((cu["total_area_px"] - cd["total_area_px"]) / tot_area
             if tot_area else float("nan"))

    s_nm = um_per_px * 1000.0 if um_per_px else None
    period_px = mm["domain_period"]
    period_nm = mm.get("domain_period_nm")

    out = {}
    for name, c, st, frac in (("up", cu, mm["up"], f_up), ("down", cd, mm["down"], f_down)):
        width_fit = fit_lognormal(st["widths_px"])
        period_fit = fit_lognormal(st["periods_px"])
        entry = {
            "n_domains": c["n_domains"],
            "area_px": c["total_area_px"], "area_um2": c.get("total_area_um2"),
            "width_median": st["width_median"], "width_median_nm": st.get("width_median_nm"),
            "width_mode": width_fit["mode"] if width_fit else None,
            "width_mode_nm": (width_fit["mode"] * s_nm) if (width_fit and s_nm) else None,
            "width_fit": _fit_summary(width_fit),
            "period_fit": _fit_summary(period_fit),
            "mz": frac,
            "period_px": period_px, "period_nm": period_nm,
        }
        out[name] = entry

    pooled_w = np.concatenate([mm["up"]["widths_px"], mm["down"]["widths_px"]])
    pooled_p = np.concatenate([mm["up"]["periods_px"], mm["down"]["periods_px"]])
    fit_tot_w = fit_lognormal(pooled_w)
    fit_tot_p = fit_lognormal(pooled_p)
    out["total"] = {
        "n_domains": cu["n_domains"] + cd["n_domains"],
        "area_px": tot_area,
        "area_um2": (cu.get("total_area_um2", 0.0) + cd.get("total_area_um2", 0.0)) if um_per_px else None,
        "width_median": float(np.median(pooled_w)) if pooled_w.size else float("nan"),
        "width_median_nm": (float(np.median(pooled_w)) * s_nm) if (pooled_w.size and s_nm) else None,
        "width_mode": fit_tot_w["mode"] if fit_tot_w else None,
        "width_mode_nm": (fit_tot_w["mode"] * s_nm) if (fit_tot_w and s_nm) else None,
        "width_fit": _fit_summary(fit_tot_w),
        "period_fit": _fit_summary(fit_tot_p),
        "mz": net_m,
        "period_px": period_px, "period_nm": period_nm,
    }
    if layers is not None:
        out["total"]["area_avg_mz"] = area_averaged_mz(layers)["mean_mz"]
    return out


def export_domain_summary(mz, roi, um_per_px=None, min_px=12, connectivity=1,
                          layers=None, out_path="domain_summary.csv"):
    """Write summarize_domains()'s up/down/total table as one CSV (one row
    per state). Returns (summary_dict, out_path)."""
    import csv
    summary = summarize_domains(mz, roi, um_per_px=um_per_px, min_px=min_px,
                                connectivity=connectivity, layers=layers)
    cols = ["state", "n_domains", "area_px", "area_um2",
            "width_median", "width_median_nm", "width_mode", "width_mode_nm",
            "mz", "period_px", "period_nm", "area_avg_mz"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for state in ("up", "down", "total"):
            row = {"state": state, **summary[state]}
            w.writerow({c: row.get(c) for c in cols})
    return summary, out_path



def export_fit_summary(mm, um_per_px=None, out_path="fits.txt", title="", min_px=None):
    """
    Standalone fitting-summary text file: the log-normal AND Gaussian fits of
    the per-skeleton-pixel domain WIDTH and local PERIOD samples, for up,
    down, and the pooled total -- the same fits the *_width_distribution /
    *_period_distribution figures plot, written out as numbers.

    For each sample it reports the log-normal (median, mode, sigma_ln, mu_ln)
    and the Gaussian (mean, sd), each with AIC / BIC / KS-statistic and
    p-value, plus the AIC-preferred model. Lengths are in nm when calibrated,
    else px; sigma_ln and the KS statistic are scale-invariant.

    These are the PER-SKELETON-PIXEL samples (distance-transform local
    thickness on the medial axis, see domain_width / _local_period_px) -- NOT
    the per-domain equivalent-diameter distribution, which is unreliable for
    a fused labyrinth (METHODS.md Sec 11/14).

    `mm` is a per_state_morphometry() result; pass the SAME one the rest of
    the report used so these numbers match it. Returns out_path.
    """
    scale = (um_per_px * 1000.0) if um_per_px else 1.0
    unit = "nm" if um_per_px else "px"

    def _block(values):
        """Formatted lines for one sample's log-normal vs Gaussian fit."""
        v = np.asarray(values, float)
        v = v[v > 0]
        if v.size < 5:
            return [f"    (not enough samples to fit: n={v.size}, need >= 5)"]
        cmp = compare_lognormal_gaussian(v)
        ln, ga = cmp["lognormal"], cmp["gaussian"]
        out = [f"    n = {v.size}"]
        if ln:
            out.append(f"    log-normal : median={ln['median']:.3f}  mode={ln['mode']:.3f}  "
                       f"sigma_ln={ln['sigma_log']:.4f}  mu_ln={ln['mu_log']:.4f}")
            out.append(f"                 AIC={ln['aic']:.1f}  BIC={ln['bic']:.1f}  "
                       f"KS={ln['ks_stat']:.4f} (p={ln['ks_p']:.3g})")
        if ga:
            out.append(f"    gaussian   : mean={ga['mean']:.3f}  sd={ga['sd']:.3f}")
            out.append(f"                 AIC={ga['aic']:.1f}  BIC={ga['bic']:.1f}  "
                       f"KS={ga['ks_stat']:.4f} (p={ga['ks_p']:.3g})")
        if cmp["preferred"]:
            out.append(f"    preferred  : {cmp['preferred']}  "
                       f"(dAIC={cmp['delta_aic']:+.1f}, dBIC={cmp['delta_bic']:+.1f}; "
                       f"positive favors log-normal)")
        return out

    lines = [
        "Distribution fit summary" + (f" -- {title}" if title else ""),
        "=" * 66,
        f"samples : per-skeleton-pixel local thickness (2*DT on the medial axis)",
        f"          and local period; lengths in {unit}"
        + (f"; min_px={min_px}" if min_px is not None else ""),
        "note    : NOT the per-domain equivalent-diameter fit (unreliable for a",
        "          fused labyrinth -- see METHODS.md Sec 11/14)",
        "",
    ]
    for label, key in (("WIDTH", "widths_px"), ("PERIOD", "periods_px")):
        up = mm["up"][key] * scale
        dn = mm["down"][key] * scale
        tot = np.concatenate([up, dn]) if (up.size or dn.size) else np.array([])
        lines.append(f"=== domain {label} ({unit}) ===")
        for state, vals in (("up", up), ("down", dn), ("total (pooled up+down)", tot)):
            lines.append(f"  -- {state} --")
            lines.extend(_block(vals))
            lines.append("")
    with open(out_path, "w") as f:
        f.write("\n".join(lines).rstrip() + "\n")
    return out_path


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
    (c) or draw_distributions in gui_core.py instead.

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


def _qq_series(values, min_n=5):
    """Log-normal Q-Q series (theoretical vs. ordered sample values) for a
    single array of width samples, fit to its OWN log-normal (not pooled
    with any other state). Returns a dict with osm/osr/slope/intercept/fit,
    or None if there are fewer than `min_n` positive samples."""
    from scipy import stats as _stats
    v = np.asarray(values, float)
    v = v[v > 0]
    if v.size < min_n:
        return None
    fit = fit_lognormal(v)
    if fit is None:
        return None
    (osm, osr), (slope, intercept, r) = _stats.probplot(
        v, sparams=(fit["sigma_log"],), dist="lognorm", plot=None)
    return {"osm": osm, "osr": osr, "slope": slope, "intercept": intercept,
            "r": r, "fit": fit}


def qq_series_per_state(mm, scale=1.0, min_n=5, quantity="width"):
    """Up/down log-normal Q-Q series from a per_state_morphometry() dict,
    with the samples scaled by `scale` (e.g. um_per_px*1000 for nm).
    quantity="width" uses widths_px, quantity="period" uses periods_px.
    Each state is fit to ITS OWN log-normal, independently of the other --
    up and down are never pooled. Returns
    {"up": series_or_None, "down": series_or_None}."""
    key = "widths_px" if quantity == "width" else "periods_px"
    return {
        name: _qq_series(mm[name][key] * scale, min_n=min_n)
        for name in ("up", "down")
    }


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


# ----------------------------------------------------------------------
# Per-domain (not per-skeleton-pixel) width + area, with a true domain-count
# histogram on y. make_morphometry_figure's panel (c) above histograms every
# *skeleton pixel's* local width, so its bin heights are pixel counts, not
# domain counts; this pair of functions gives one width + one area value per
# individual domain instead.

def per_domain_width_area(mz, min_px=12, um_per_px=None):
    """
    One width + one area value per individual connected domain (not per
    skeleton pixel), separately for up (+1) and down (-1) states.

        width = 2 * max(distance-transform) inside that domain's own mask
                (the domain's local half-width at its thickest point)
        area  = pixel count of that domain (um^2 if calibrated)

    Returns {"up": {"area": arr, "width": arr, "width_nm": arr (if calibrated),
                    "n": int}, "down": {...}}
    """
    s = (um_per_px or 1.0)
    s_nm = (um_per_px * 1000.0) if um_per_px else None
    a2 = (um_per_px * um_per_px) if um_per_px else None
    out = {}
    for name, val in (("up", 1), ("down", -1)):
        lbl, n = ndi.label(mz == val)
        if n == 0:
            entry = {"area": np.array([]), "width": np.array([]), "n": 0}
            if s_nm:
                entry["width_nm"] = np.array([])
            out[name] = entry
            continue

        # Per-domain EDT only needs that domain's own pixels, so crop to its
        # (1px-padded) bounding box instead of running distance_transform_edt
        # on the full frame for every domain -- for a labyrinth pattern with
        # hundreds/thousands of domains that full-frame-per-domain EDT was
        # the dominant cost of this function. A single pixel of padding
        # reproduces the full-frame result exactly: a domain never extends
        # past its own bounding box, so the nearest background pixel outside
        # it is the same one the full-frame version would have found.
        sizes = np.bincount(lbl.ravel())[1:]       # sizes[i-1] == area of label i
        objects = ndi.find_objects(lbl)            # objects[i-1] == slice for label i
        H, W = lbl.shape
        areas_px, widths_px = [], []
        for i in range(1, n + 1):
            if sizes[i - 1] < min_px:
                continue
            slc = objects[i - 1]
            y0, y1 = max(slc[0].start - 1, 0), min(slc[0].stop + 1, H)
            x0, x1 = max(slc[1].start - 1, 0), min(slc[1].stop + 1, W)
            dom = lbl[y0:y1, x0:x1] == i
            dt = ndi.distance_transform_edt(dom)
            widths_px.append(2.0 * float(dt.max()))
            areas_px.append(int(sizes[i - 1]))

        areas_px = np.array(areas_px, dtype=float)
        widths_px = np.array(widths_px, dtype=float)
        entry = {"area": areas_px * (a2 if a2 else 1.0),
                 "width": widths_px * s, "n": areas_px.size}
        if s_nm:
            entry["width_nm"] = widths_px * s_nm
        out[name] = entry
    return out


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


# --- call site ---
# from xmcd_toolkit.stats import three_state_field
# from xmcd_toolkit.domain import analyze_xmcd
# from xmcd_toolkit.morphometry import make_width_area_count_figure
#
# metrics, layers = analyze_xmcd("0mA.png")          # or your 0 mA file
# mz, roi = three_state_field(layers)
# make_width_area_count_figure(mz, um_per_px=metrics["um_per_px"],
#                              out_stem="outputs/0mA_width_area_counts",
#                              title="0 mA")
