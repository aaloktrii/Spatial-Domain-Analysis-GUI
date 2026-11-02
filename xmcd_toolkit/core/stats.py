"""
stats.py -- three-state m_z statistics.

States (the three probability cases):
    m_z = +1  up domain   |  0  no domain (wall/background)  |  -1  down domain

Provides:
    three_state_field        build the (-1/0/+1) field from a domain analysis
    pdf_three_state          discrete PDF p(+1), p(0), p(-1) over a region
    number_density_local     LOCAL number density n_s = N_s(area)/N_total(area)
    net_change_mz            d<m_z> between two states (e.g. before/after pulse)
    domain_count_distribution    number-of-domains statistic per state
    make_stats_figure        4/6-panel summary figure
"""

import numpy as np
from scipy import ndimage as ndi
from skimage import measure

from .domain import analyze_xmcd
from .loaders import neighbor_structure, _remove_small


# ----------------------------------------------------------------------
def per_state_domain_stats(mz, min_px=12, um_per_px=None, connectivity=1, roi=None):
    """
    For each state s in {+1, 0, -1}, label connected domains and compute their
    area and shape, then summarize.

    Per-domain measures:
        area_px, area_um2 (if calibrated), equiv_diam, eccentricity (0=circle,
        1=line), solidity (1=convex), circularity (4*pi*A/P^2; 1=circle).

    Returns dict with, per state:
        'domains'  -> list of per-domain dicts (the table)
        'n', 'area_mean', 'area_sd', 'area_median'  (px or um^2 if calibrated)
        'ecc_mean', 'solidity_mean', 'circularity_mean'

    NOTE on labyrinthine patterns: connected-component labeling merges an
    interconnected stripe maze into one giant domain, so area_mean >> area_median
    and area_sd is huge. For such states report area FRACTIONS and the median,
    not the mean domain area.

    `roi`: the flake mask, restricting the 'zero' (neutral) state before
    labeling. three_state_field only ever sets +1/-1 inside the ROI, leaving
    everything else -- including the entire region outside the ROI -- at 0.
    Without this restriction, in-flake wall pixels are directly adjacent to
    that huge off-flake background and get merged into it as one connected
    component, making 'zero' n/area_mean/area_median meaningless (dominated
    by the off-flake area instead of describing actual wall fragments). up
    and down are unaffected (they're already ROI-confined by construction).
    """
    a2 = (um_per_px * um_per_px) if um_per_px else None
    fg = neighbor_structure(connectivity)                    # up / down domains
    bg = neighbor_structure(2 if connectivity == 1 else 1)   # neutral = opposite rule
    out = {}
    for name, s in [("up", 1), ("zero", 0), ("down", -1)]:
        struct = bg if s == 0 else fg
        mask = (mz == s)
        if s == 0 and roi is not None:
            mask = mask & roi
        lbl, _ = ndi.label(mask, structure=struct)
        rows = []
        for p in measure.regionprops(lbl):
            if p.area < min_px:
                continue
            circ = 4 * np.pi * p.area / max(p.perimeter, 1.0) ** 2
            rows.append({
                "state": s, "area_px": int(p.area),
                "area_um2": (p.area * a2) if a2 else None,
                "equiv_diam_px": float(p.equivalent_diameter_area),
                "eccentricity": float(p.eccentricity),
                "solidity": float(p.solidity),
                "circularity": float(min(circ, 1.0)),
            })
        A = np.array([r["area_px"] for r in rows], float)
        ecc = np.array([r["eccentricity"] for r in rows])
        sol = np.array([r["solidity"] for r in rows])
        cir = np.array([r["circularity"] for r in rows])
        scale = a2 if a2 else 1.0
        out[name] = {
            "domains": rows, "n": int(A.size),
            "area_mean": float(A.mean() * scale) if A.size else float("nan"),
            "area_sd": float(A.std() * scale) if A.size else float("nan"),
            "area_median": float(np.median(A) * scale) if A.size else float("nan"),
            "ecc_mean": float(ecc.mean()) if A.size else float("nan"),
            "solidity_mean": float(sol.mean()) if A.size else float("nan"),
            "circularity_mean": float(cir.mean()) if A.size else float("nan"),
            "units": "um^2" if a2 else "px",
        }
    return out


# ----------------------------------------------------------------------
def three_state_field(image_or_layers, z_neutral=0.5, m_threshold=None,
                      norm_percentile=99):
    """
    Return (m_z, roi) with m_z in {-1, 0, +1} over the whole frame, plus the
    continuous normalized magnetization (XMCD asymmetry) used to classify.

    Two thresholding modes:

    * VALUE mode (m_threshold given) -- the one to use for "gray = [-t, +t]":
        the background-subtracted signal is normalized to a symmetric
        asymmetry m in [-1, 1] (saturated domains ~ +-1), then
            m > +m_threshold  -> +1
            m < -m_threshold  -> -1
            |m| <= m_threshold -> 0   (gray)
        e.g. m_threshold=0.1 makes the gray band exactly [-0.1, +0.1].

    * Z-SCORE mode (default, m_threshold=None) -- legacy behavior:
        |z| < z_neutral -> 0, using the z-score of the flake signal.

    Returns (mz, roi).  The continuous asymmetry is also stored on the
    `layers` dict as layers["asymmetry"] when a layers dict is passed in.
    """
    if isinstance(image_or_layers, dict) and "flat" in image_or_layers:
        layers = image_or_layers
    else:
        _, layers = analyze_xmcd(image_or_layers)
    flat, roi = layers["flat"], layers["roi"]
    mz = np.zeros(flat.shape, dtype=np.int8)       # -1 / 0 / +1: one byte per pixel

    if m_threshold is not None:
        # normalize to a symmetric asymmetry: saturated domains -> ~ +-1
        scale = np.percentile(np.abs(flat[roi]), norm_percentile) + 1e-12
        m = np.clip(flat / scale, -1.0, 1.0)
        layers["asymmetry"] = m
        mz[roi & (m > m_threshold)] = +1
        mz[roi & (m < -m_threshold)] = -1
    else:
        vals = flat[roi]
        z = (flat - vals.mean()) / (vals.std() + 1e-12)
        layers["asymmetry"] = z
        mz[roi & (z >= z_neutral)] = +1
        mz[roi & (z <= -z_neutral)] = -1
    return mz, roi


# ----------------------------------------------------------------------
def despeckle_mz(mz, min_px=12):
    """
    Strip isolated small blobs of a state (a handful of pixels, well below a
    real domain) that per-pixel classification noise scatters inside
    otherwise-clean up/down regions -- the "small dots" salt-and-pepper look.

    Any +1 or -1 fragment with fewer than min_px connected pixels reverts to
    neutral (0) instead of keeping its (almost certainly spurious) sign.
    min_px=0 is a no-op, so nothing changes unless this is called.

    Apply this ONCE, right after three_state_field, so the overlay, the
    three-case PDF, and per_state_morphometry all see the same cleaned field
    -- rather than each consumer quietly filtering it a different way.
    """
    if not min_px:
        return mz
    # _remove_small (numpy/scipy bincount, not skimage's remove_small_objects)
    # so this doesn't depend on skimage's deprecated min_size parameter --
    # see morphometry.domain_width for the full story. connectivity=1
    # matches remove_small_objects' own default (4-connected).
    up = _remove_small(mz == 1, min_px, connectivity=1)
    dn = _remove_small(mz == -1, min_px, connectivity=1)
    out = np.zeros_like(mz)
    out[up] = 1
    out[dn] = -1
    return out


# ----------------------------------------------------------------------
def pdf_three_state(mz, region=None):
    """Discrete PDF over {-1,0,+1}. region: boolean mask or None (whole frame)."""
    sel = mz if region is None else mz[region]
    n = sel.size
    Np, Nz, Nm = int((sel == 1).sum()), int((sel == 0).sum()), int((sel == -1).sum())
    return {"N_total": n, "N_up": Np, "N_zero": Nz, "N_down": Nm,
            "p_up": Np / n, "p_zero": Nz / n, "p_down": Nm / n,
            "net_m": (Np - Nm) / (Np + Nm) if (Np + Nm) else float("nan")}


# ----------------------------------------------------------------------
def number_density_local(mz, tile=16, region=None):
    """
    LOCAL number density per the working definition:
        n_s(cell) = N_s(cell) / N_total(cell)
    i.e. the fraction of each state inside a defined area (tile). The three
    fractions sum to 1 in every cell.

    If `region` (a boolean flake mask) is given, the denominator counts only
    flake pixels in each tile, and tiles with no flake pixels are set to NaN
    (so background tiles are excluded from the flake-only statistics).

    Returns full-resolution fields plus a per-tile table.
    """
    H, W = mz.shape
    ys = np.arange(0, H, tile)
    xs = np.arange(0, W, tile)
    ny, nx = len(ys), len(xs)
    n_tiles = ny * nx

    # Per-pixel tile id (row-major, matching the (ny, nx) grid below), so
    # every state's per-tile pixel count is one bincount over the whole
    # frame instead of a Python loop that re-slices + re-sums each tile
    # individually -- same result, no per-tile Python/indexing overhead.
    tile_id = ((np.arange(H) // tile)[:, None] * nx
               + (np.arange(W) // tile)[None, :])

    def _tile_counts(mask):
        return np.bincount(tile_id[mask], minlength=n_tiles)[:n_tiles].reshape(ny, nx).astype(float)

    sel = region if region is not None else np.ones((H, W), bool)
    tot = _tile_counts(sel)
    up_c = _tile_counts((mz == 1) & sel)
    dn_c = _tile_counts((mz == -1) & sel)
    zr_c = _tile_counts((mz == 0) & sel)

    bg_tile = (tot == 0)
    tot_safe = np.where(bg_tile, 1.0, tot)   # avoid 0/0 warnings; result masked below anyway
    n_up = np.where(bg_tile, np.nan, up_c / tot_safe)
    n_dn = np.where(bg_tile, np.nan, dn_c / tot_safe)
    n_zr = np.where(bg_tile, np.nan, zr_c / tot_safe)
    net = n_up - n_dn   # local net magnetization fraction

    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    valid = ~bg_tile
    rows = [
        {"x_center": int(x + tile / 2), "y_center": int(y + tile / 2),
         "n_up": float(u), "n_zero": float(z), "n_down": float(d), "net_mz": float(m)}
        for x, y, u, z, d, m in zip(xx[valid], yy[valid], n_up[valid],
                                     n_zr[valid], n_dn[valid], net[valid])
    ]
    return {"n_up": n_up, "n_down": n_dn, "n_zero": n_zr, "net": net,
            "tile": tile, "tiles": rows}


# ----------------------------------------------------------------------
def domain_wall_distribution(nd):
    """
    Distribution of the LOCAL domain-wall fraction D across the flake, from a
    number_density_local() result. D is the zero/neutral state fraction in
    each tile (nd["n_zero"]); its scalar frame value is one number, but its
    SPATIAL distribution over tiles reveals whether walls are uniform or
    clustered/pinned -- fragmentation heterogeneity.

    Returns {"values": per-tile D array, "n_tiles", "mean", "median", "std",
             "min", "max"} (empty/NaN if no flake tiles).
    """
    vals = np.array([t["n_zero"] for t in nd["tiles"]], float)
    if vals.size == 0:
        return {"values": vals, "n_tiles": 0, "mean": float("nan"),
                "median": float("nan"), "std": float("nan"),
                "min": float("nan"), "max": float("nan")}
    return {"values": vals, "n_tiles": int(vals.size),
            "mean": float(vals.mean()), "median": float(np.median(vals)),
            "std": float(vals.std()), "min": float(vals.min()),
            "max": float(vals.max())}


# ----------------------------------------------------------------------
def net_change_mz(mz_before, mz_after, region=None):
    """
    Net change in magnetization between two conditions (e.g. two pulses):
        d<m_z> = <m_z>_after - <m_z>_before
    where <m_z> = (N_up - N_down)/(N_up + N_down) over `region`
    (boolean mask) or the whole frame.
    """
    a = pdf_three_state(mz_before, region)
    b = pdf_three_state(mz_after, region)
    return {"m_before": a["net_m"], "m_after": b["net_m"],
            "delta_m": b["net_m"] - a["net_m"]}


# ----------------------------------------------------------------------
def domain_count_distribution(mz, min_px=12):
    """Number-of-domains statistic and size list per state."""
    out = {}
    for name, s in [("up", 1), ("down", -1)]:
        lbl, _ = ndi.label(mz == s)
        sizes = np.bincount(lbl.ravel())[1:]
        sizes = sizes[sizes >= min_px]
        out[name] = {"n_domains": int(sizes.size), "sizes_px": sizes,
                     "mean_px": float(sizes.mean()) if sizes.size else 0.0}
    return out


