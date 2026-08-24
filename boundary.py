"""
boundary.py -- swappable domain-boundary (wall line) detectors. Given the
continuous asymmetry m in [-1,1] and the flake ROI, return a boolean EDGE
mask marking the up/down boundary (and, for zero_contour, the sub-pixel
contours themselves).

    domain_boundary(m, roi, method="zero_contour", **params)
        -> {"edge": bool mask, "contours": [Nx2 arrays] or None}

Registry keys:
  "threshold"    the |m|<=t neutral band (a zone, not a line) -- the current
                 3-state wall. t-dependent.
  "zero_contour" the m=0 sub-pixel iso-contour (marching squares on a lightly
                 smoothed m). Threshold-free, sharp line. Default.
  "chan_vese"    region-based level-set two-phase segmentation; boundary =
                 the interface. No hard threshold, smooth, noise-robust.
"""
import numpy as np
from scipy import ndimage as ndi
from skimage import measure as _measure

from . import make_registry

BOUNDARIES, _register = make_registry()


def _edge_of(mask):
    m = np.asarray(mask, bool)
    return m ^ ndi.binary_erosion(m)


@_register("zero_contour", "m=0 sub-pixel contour (sharp)",
           "The m=0 iso-contour (marching squares on lightly smoothed m). "
           "Threshold-free, sub-pixel, sharp line. Follows fine structure "
           "(and some noise). Default.")
def zero_contour(m, roi, smooth=1.0, **_):
    ms = ndi.gaussian_filter(np.asarray(m, float), float(smooth))
    contours = _measure.find_contours(ms, 0.0)
    edge = np.zeros(m.shape, bool)
    H, W = m.shape
    for c in contours:
        r = np.clip(c[:, 0].astype(int), 0, H - 1)
        cc = np.clip(c[:, 1].astype(int), 0, W - 1)
        edge[r, cc] = True
    edge &= np.asarray(roi, bool)
    return {"edge": edge, "contours": contours}


@_register("threshold", "Threshold band (|m|<=t neutral zone)",
           "The neutral gray band -- a zone, not a line. Depends on the "
           "m-threshold. This is the current 3-state wall.")
def threshold(m, roi, m_threshold=0.05, **_):
    band = np.asarray(roi, bool) & (np.abs(np.asarray(m, float)) <= float(m_threshold))
    return {"edge": band, "contours": None}


@_register("chan_vese", "Chan-Vese region level-set (smooth, robust)",
           "Region-based two-phase segmentation; the up/down interface is the "
           "boundary. No hard threshold; smooth and noise-robust. Slower.")
def chan_vese(m, roi, cv_mu=0.15, cv_iter=80, cv_downsample=2, **_):
    from skimage.segmentation import chan_vese as _cv
    d = int(cv_downsample)
    md = np.asarray(m, float)[::d, ::d]
    seg = _cv(md, mu=float(cv_mu), max_num_iter=int(cv_iter),
              init_level_set="checkerboard")
    if d > 1:
        seg = np.repeat(np.repeat(seg, d, 0), d, 1)[:m.shape[0], :m.shape[1]]
    edge = _edge_of(seg) & np.asarray(roi, bool)
    return {"edge": edge, "contours": None}


BOUNDARY_METHODS = ("zero_contour", "threshold", "chan_vese")
BOUNDARY_METHOD_LABELS = {k: BOUNDARIES[k].label for k in BOUNDARY_METHODS}


def domain_boundary(m, roi, method="zero_contour", **params):
    """Dispatch to the named boundary detector."""
    if method not in BOUNDARIES:
        raise ValueError(f"unknown boundary method {method!r}; "
                         f"choices: {list(BOUNDARIES)}")
    return BOUNDARIES[method].fn(m, roi, **params)
