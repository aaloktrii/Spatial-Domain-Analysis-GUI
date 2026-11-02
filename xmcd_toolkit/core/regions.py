"""Regions and calibration: polygon -> mask, um per pixel from the field of view or a drawn box
No plotting, no GUI."""
import numpy as np


# ----------------------------------------------------------------------
_POLY_CACHE = {}   # (vertices, shape) -> mask; rasterizing is ~20ms, and the


def _scanline_fill(xs, ys, shape):
    """Pixels whose centre lies inside the polygon (even-odd rule), one row at a time: for each pixel row the edge
    crossings are computed and the columns between pairs filled. Identical to skimage.draw.polygon when no vertex
    lies exactly on a pixel centre."""
    H, W = shape
    mask = np.zeros(shape, bool)
    r0, r1 = max(int(np.ceil(ys.min())), 0), min(int(np.floor(ys.max())), H - 1)
    if r1 < r0:
        return mask
    x1, y1 = np.roll(xs, -1), np.roll(ys, -1)
    r = np.arange(r0, r1 + 1, dtype=float)[:, None]
    cross = ((ys <= r) & (y1 > r)) | ((y1 <= r) & (ys > r))
    with np.errstate(divide="ignore", invalid="ignore"):
        xi = np.where(cross, xs + (r - ys) * (x1 - xs) / (y1 - ys), np.inf)
    xi.sort(axis=1)
    for k, n in enumerate(cross.sum(1)):
        v = xi[k, :n]
        for a, b in zip(v[0::2], v[1::2]):
            c0, c1 = max(int(np.ceil(a)), 0), min(int(np.floor(b)), W - 1)
            if c1 >= c0:
                mask[r0 + k, c0:c1 + 1] = True
    return mask


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
    xs = np.array([v[0] for v in vertices], float)
    ys = np.array([v[1] for v in vertices], float)
    H, W = shape
    if (len(xs) == 4 and set(zip(xs, ys)) == {(0, 0), (W - 1, 0), (W - 1, H - 1), (0, H - 1)}):
        mask = np.ones(shape, bool)               # the whole image (what skimage gives for this rectangle)
    elif np.any(xs == np.round(xs)) or np.any(ys == np.round(ys)):
        # a vertex exactly on a pixel centre: skimage's tie rule decides the edge pixels
        mask = np.zeros(shape, bool)
        rr, cc = sk_polygon(ys, xs, shape)
        mask[rr, cc] = True
    else:
        mask = _scanline_fill(xs, ys, shape)      # same pixels, ~200x faster on a 41 MP image
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


def calibration(params, shape):
    """um/px from the calibration settings: cal_mode "none" (pixels), "image" (the whole image is cal_w x cal_h
    um) or "shape" (the active area's bounding box is cal_w x cal_h um). -> (um_per_px or None, warning)."""
    if params.get("cal_mode", "none") == "none" or not (params.get("cal_w") or params.get("cal_h")):
        return None, None
    poly = params.get("polygon")
    if params.get("cal_mode") == "shape" and poly:
        v = np.asarray(poly, float)
        box = (float(np.ptp(v[:, 0])), float(np.ptp(v[:, 1])))
    else:
        box = (float(shape[1]), float(shape[0]))
    return um_per_px_from_dims(box, (params.get("cal_w"), params.get("cal_h")))


def resolve_um(params, shape):
    """Pixel size from an explicit um_per_px (e.g. box calibration) if present,
    otherwise from the whole-image field of view. Returns (um_per_px, warn)."""
    if params.get("um_per_px"):
        return float(params["um_per_px"]), None
    return um_per_px_from_fov(shape, params.get("fov"))
