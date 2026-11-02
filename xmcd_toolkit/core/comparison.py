"""Pixel-by-pixel comparison of two aligned states (no plotting, no GUI).

Inputs (final frame, same grid): the three-state maps s_A, s_B of the domain analysis (+1 up, -1 down, 0 neutral,
i.e. |m| <= m0 or a removed speck), the continuous asymmetry fields m_A, m_B, and one ROI.

  flip map   per pixel: stayed up / stayed down / up->down / down->up / uncertain (neutral in A or B)
  switched   f_up->down = N(up->down) / N(up in A), f_down->up = N(down->up) / N(down in A), both over pixels
             classified in both; switched = (N(up->down) + N(down->up)) / N(classified in both)
  memory r   Pearson r of m_A, m_B over the ROI, 95 % block bootstrap (blocks ~ 1 um)
  local r    the same coefficient in a Gaussian window around every pixel (local means removed)
  walls      wall pixels = up pixels touching a down pixel (and the reverse); wall shift = median distance from
             the walls of B to those of A (and back), in nm when the pixel size is known (else px)
"""
import numpy as np
from scipy import ndimage

FLIP_CODES = {"stayed_down": 0, "stayed_up": 1, "up_to_down": 2, "down_to_up": 3, "uncertain": 4}


def flip_map(sA, sB, roi):
    """Category map (float, NaN outside the ROI) and pixel counts."""
    both = roi & (sA != 0) & (sB != 0)
    flip = np.full(sA.shape, np.nan)
    flip[roi] = FLIP_CODES["uncertain"]
    flip[both & (sA == 1) & (sB == 1)] = FLIP_CODES["stayed_up"]
    flip[both & (sA == -1) & (sB == -1)] = FLIP_CODES["stayed_down"]
    flip[both & (sA == 1) & (sB == -1)] = FLIP_CODES["up_to_down"]
    flip[both & (sA == -1) & (sB == 1)] = FLIP_CODES["down_to_up"]
    n = {k: int((flip == v).sum()) for k, v in FLIP_CODES.items()}
    n_up, n_dn, n_cl = n["stayed_up"] + n["up_to_down"], n["stayed_down"] + n["down_to_up"], int(both.sum())
    m = {"roi_px": int(roi.sum()), "classified_px": n_cl, "classified_pct": 100 * n_cl / max(int(roi.sum()), 1),
         **{f"n_{k}": v for k, v in n.items()},
         "up_to_down_pct": 100 * n["up_to_down"] / max(n_up, 1),
         "down_to_up_pct": 100 * n["down_to_up"] / max(n_dn, 1),
         "switched_pct": 100 * (n["up_to_down"] + n["down_to_up"]) / max(n_cl, 1)}
    m["memory_from_flips"] = 1 - (m["up_to_down_pct"] + m["down_to_up_pct"]) / 100
    return flip, m


def global_r(a, b, roi, block_px, n_boot=1000, seed=0):
    """Pearson r of a, b over the ROI and its 95 % block-bootstrap interval (block_px x block_px blocks drawn with
    replacement, so neighbouring, correlated pixels are resampled together)."""
    v = roi & np.isfinite(a) & np.isfinite(b)
    if v.sum() < 3:
        return float("nan"), float("nan"), float("nan")
    av, bv = a[v] - a[v].mean(), b[v] - b[v].mean()
    r = float((av * bv).sum() / np.sqrt((av * av).sum() * (bv * bv).sum()))
    ys, xs = np.nonzero(v)
    _, inv = np.unique((ys // block_px) * (xs.max() // block_px + 1) + xs // block_px, return_inverse=True)
    k = inv.max() + 1
    sums = [np.bincount(inv, w, k) for w in (np.ones(v.sum()), av, bv, av * bv, av ** 2, bv ** 2)]
    pick = np.random.default_rng(seed).integers(0, k, (n_boot, k))
    n, sa, sb, sab, saa, sbb = (s[pick].sum(1) for s in sums)
    rb = (sab - sa * sb / n) / np.sqrt(np.maximum((saa - sa ** 2 / n) * (sbb - sb ** 2 / n), 1e-30))
    lo, hi = np.percentile(rb, (2.5, 97.5))
    return r, float(lo), float(hi)


def local_r(a, b, roi, sigma_px):
    """Windowed Pearson r (Gaussian window sigma_px, restricted to the ROI, renormalised at its edge)."""
    w = (roi & np.isfinite(a) & np.isfinite(b)).astype(float)
    if not w.any():
        return np.full(a.shape, np.nan)
    sel = w > 0
    av = np.where(sel, a - a[sel].mean(), 0.0)
    bv = np.where(sel, b - b[sel].mean(), 0.0)
    gw = np.maximum(ndimage.gaussian_filter(w, sigma_px, truncate=3.0), 1e-6)

    def g(v):
        return ndimage.gaussian_filter(v * w, sigma_px, truncate=3.0) / gw

    ma, mb = g(av), g(bv)
    r = (g(av * bv) - ma * mb) / np.sqrt(np.maximum((g(av * av) - ma ** 2) * (g(bv * bv) - mb ** 2), 1e-30))
    return np.where(roi, np.clip(r, -1, 1), np.nan)


def walls(s, roi):
    """Up pixels with a down neighbour (4-connectivity) and vice versa, inside the ROI."""
    up, dn = (s == 1) & roi, (s == -1) & roi
    return (up & ndimage.binary_dilation(dn)) | (dn & ndimage.binary_dilation(up))


def wall_shift(wa, wb, um_per_px=None):
    """Median distance from the walls wb to the nearest wall in wa: nm if um_per_px is known, else px."""
    if not wa.any() or not wb.any():
        return float("nan")
    d = float(np.median(ndimage.distance_transform_edt(~wa)[wb]))
    return d * um_per_px * 1000 if um_per_px else d


def compare_states(sA, sB, mA, mB, roi, um_per_px=None, window_px=10.0, block_um=1.0, n_boot=1000):
    """Everything for one pair. Returns (layers, metrics)."""
    roi = roi & np.isfinite(mA) & np.isfinite(mB)
    flip, m = flip_map(sA, sB, roi)
    block = max(int(round(block_um / um_per_px)), 4) if um_per_px else 32
    r, lo, hi = global_r(mA, mB, roi, block, n_boot)
    rl = local_r(mA, mB, roi, window_px)
    wA, wB = walls(sA, roi), walls(sB, roi)
    m.update(r=r, r_ci_lo=lo, r_ci_hi=hi, r_local_median=float(np.nanmedian(rl)) if roi.any() else float("nan"),
             wall_shift_B_to_A=wall_shift(wA, wB, um_per_px), wall_shift_A_to_B=wall_shift(wB, wA, um_per_px),
             wall_shift_unit="nm" if um_per_px else "px")
    return {"flip": flip, "r_local": rl, "roi": roi, "walls_A": wA, "walls_B": wB}, m
