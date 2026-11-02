"""Centre of a map (the level subtracted so that 0 is the magnetic zero), as TensorSpec's "Sym ±" centre
(gui/components/data_viewer_panel.py: sym_center) and "Subtract offset" (gui/services/peem_service.py:
map_offset): 0, or the median / mean of the map inside a drawn region, else of the map without the drift strips
at the field's edges (core/peem/domain_analysis.py: edge_strips). No plotting, no GUI."""
import numpy as np


def edge_strips(a, k=2.0, max_frac=0.2, margin=5):
    """Mask of the field without drift strips: rows / columns at an edge whose scatter is above k times the
    median row / column scatter (the area left uncovered by the drift shift, filled from the edge).
    At most max_frac of the field is cut on each side; margin px more are cut where a strip is found.
    (Copied from TensorSpec core/peem/domain_analysis.py.)"""
    a = np.asarray(a, float)
    keep = np.isfinite(a)
    ny, nx = a.shape
    for axis, n in ((0, nx), (1, ny)):  # axis 0: scatter of each column; axis 1: of each row
        sd = np.nanstd(a, axis=axis)
        ref = float(np.nanmedian(sd))
        hi = sd > k * ref
        lim = int(max_frac * n)
        for side in (0, 1):
            idx = range(lim) if side == 0 else range(n - 1, n - 1 - lim, -1)
            cut = 0
            for i in idx:
                if not hi[i]:
                    break
                cut += 1
            if cut:
                cut = min(cut + margin, lim)
                sl = slice(0, cut) if side == 0 else slice(n - cut, n)
                if axis == 0:
                    keep[:, sl] = False
                else:
                    keep[sl, :] = False
    return keep


CENTRES = ("0", "median", "mean")


def sym_centre(a, how="median", region=None):
    """Centre c of the map: "0", or the "median" / "mean" inside `region` (bool mask) if given and not empty,
    else of the map without its edge drift strips -> (c, number of pixels used, where)."""
    if how not in CENTRES:
        raise ValueError(f"centre must be one of {CENTRES}")
    a = np.asarray(a, float)
    if how == "0":
        return 0.0, 0, "0"
    if region is not None and np.any(region):
        vals, where = a[np.asarray(region, bool) & np.isfinite(a)], "background region"
    else:
        vals, where = a[edge_strips(a) & np.isfinite(a)], "map, edges left out"
    if not vals.size:
        raise ValueError("no finite pixels to take the centre from")
    return (float(np.median(vals)) if how == "median" else float(vals.mean())), int(vals.size), where
