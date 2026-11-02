"""Per-image transforms for one-to-one comparison of maps (no plotting, no GUI).

Frames
  raw    the image as analysed (its own pixel grid)
  final  what is shown and compared: per image rotation theta, scale s, flips F, shift t, then one view
         rotation phi for the whole set:  final = c + R(phi) [ s R(theta) F (raw - c) + t ],
         c = image centre, (row, col) coordinates, R(a) = [[cos a, -sin a], [sin a, cos a]] on (row, col),
         so + angle = counter-clockwise on screen. Polygons are (x, y) vertices.
"""
import numpy as np
from scipy import ndimage

TF_DEFAULT = {"rot": 0.0, "scale": 1.0, "dy": 0.0, "dx": 0.0, "fliph": False, "flipv": False}


def rot_rc(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def centre(shape):
    return np.array([(shape[0] - 1) / 2, (shape[1] - 1) / 2])


def is_identity(tf, view_rot=0.0):
    return (abs(tf["rot"]) < 1e-9 and abs(tf["scale"] - 1) < 1e-12 and abs(tf["dy"]) < 1e-9 and abs(tf["dx"]) < 1e-9
            and not tf["fliph"] and not tf["flipv"] and abs(view_rot) < 1e-9)


def warp(img, tf, view_rot=0.0, order=1, out_shape=None):
    """`img` (raw frame) in the final frame. order 1: values (NaN outside); order 0: labels / masks (cval 0)."""
    a = np.asarray(img)
    shape = out_shape or a.shape
    if is_identity(tf, view_rot) and shape == a.shape:
        return a.copy()
    F = np.diag([-1.0 if tf["flipv"] else 1.0, -1.0 if tf["fliph"] else 1.0])
    Minv = F @ rot_rc(-tf["rot"]) @ rot_rc(-view_rot) / tf["scale"]
    c_in, c_out = centre(a.shape), centre(shape)
    off = c_in - Minv @ c_out - F @ rot_rc(-tf["rot"]) @ np.array([tf["dy"], tf["dx"]]) / tf["scale"]
    if order == 0:
        return ndimage.affine_transform(a, Minv, off, output_shape=shape, order=0, cval=0)
    v = np.asarray(a, float)
    out = ndimage.affine_transform(np.nan_to_num(v), Minv, off, output_shape=shape, order=1, cval=0.0)
    inside = ndimage.affine_transform(np.isfinite(v).astype(float), Minv, off, output_shape=shape, order=1, cval=0.0)
    out[inside < 0.999] = np.nan
    return out


def warp_view(img, tf, view_rot, order, common_shape, step):
    """For drawing only: `img` (raw frame, any size) in the final frame on every `step`-th pixel of the common
    grid, read straight from the raw pixels (no full-size intermediate). Same geometry as warp(); order 0 for
    labels and colours (cval 0), 1 for values (NaN outside)."""
    a = np.asarray(img)
    H, W = common_shape
    z = np.array([H / a.shape[0], W / a.shape[1]])                    # raw -> common grid scale
    F = np.diag([-1.0 if tf["flipv"] else 1.0, -1.0 if tf["fliph"] else 1.0])
    Minv = F @ rot_rc(-tf["rot"]) @ rot_rc(-view_rot) / tf["scale"]
    c = centre(common_shape)
    off = c - Minv @ c - F @ rot_rc(-tf["rot"]) @ np.array([tf["dy"], tf["dx"]]) / tf["scale"]
    h = (step - 1) / 2.0
    out_shape = (int(np.ceil((H - h) / step)), int(np.ceil((W - h) / step)))   # samples at step * o + h
    M = np.diag(1 / z) @ Minv * step                                   # small output -> raw input
    o = (off + Minv @ np.array([h, h]) + 0.5) / z - 0.5
    if order == 0:
        return ndimage.affine_transform(a, M, o, output_shape=out_shape, order=0, cval=0)
    v = np.asarray(a, float)
    out = ndimage.affine_transform(np.nan_to_num(v), M, o, output_shape=out_shape, order=1, cval=0.0)
    inside = ndimage.affine_transform(np.isfinite(v).astype(float), M, o, output_shape=out_shape, order=1, cval=0.0)
    out[inside < 0.999] = np.nan
    return out


def shift_final(tf, view_rot):
    """The image's shift as displayed (final frame)."""
    return rot_rc(view_rot) @ np.array([tf["dy"], tf["dx"]])


def with_shift_final(tf, view_rot, dy, dx):
    t = rot_rc(-view_rot) @ np.array([dy, dx])
    return {**tf, "dy": float(t[0]), "dx": float(t[1])}


def compose(tf, view_rot, dtheta, ds, dt_final):
    """A delta similarity (final frame, about the centre) applied after `tf`; returns the new transform."""
    new_t = ds * rot_rc(dtheta) @ np.array([tf["dy"], tf["dx"]]) + rot_rc(-view_rot) @ np.asarray(dt_final, float)
    return {**tf, "rot": (tf["rot"] + dtheta + 180) % 360 - 180, "scale": tf["scale"] * ds,
            "dy": float(new_t[0]), "dx": float(new_t[1])}


def rotate_xy(verts, deg, centre_rc):
    """Rotate (x, y) vertices about the image centre like the maps (+ = counter-clockwise on screen)."""
    v = np.asarray(verts, float)
    rc = (np.c_[v[:, 1], v[:, 0]] - centre_rc) @ rot_rc(deg).T + centre_rc
    return [(float(x), float(y)) for y, x in rc]


def fit_similarity(p_from, p_to, centre_rc, allow_scale=False):
    """Delta D(p) = c + s R(theta) (p - c) + t taking points p_from onto p_to ((row, col), final frame); rigid unless
    allow_scale; one pair = translation. Returns (theta_deg, s, t (2,), rms px)."""
    pm, pr = np.asarray(p_from, float), np.asarray(p_to, float)
    zm = (pm[:, 1] - centre_rc[1]) + 1j * (pm[:, 0] - centre_rc[0])
    zr = (pr[:, 1] - centre_rc[1]) + 1j * (pr[:, 0] - centre_rc[0])
    a = 1.0 + 0j
    if len(pm) >= 2:
        dm, dr = zm - zm.mean(), zr - zr.mean()
        a = (dr * np.conj(dm)).sum() / max((np.abs(dm) ** 2).sum(), 1e-12)
        if not allow_scale:
            a = a / abs(a)
    b = zr.mean() - a * zm.mean()
    rms = float(np.sqrt(np.mean(np.abs(zr - (a * zm + b)) ** 2)))
    # z = col + i row: multiplying by e^{-i theta} is R(theta) on (row, col)
    return float(-np.degrees(np.angle(a))), float(abs(a)), np.array([b.imag, b.real]), rms
