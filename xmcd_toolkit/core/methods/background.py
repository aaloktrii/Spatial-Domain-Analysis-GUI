"""
background.py -- swappable background / illumination correction. Takes the
denoised field and returns a FLATTENED field whose magnetic zero is well
defined, so the up/down threshold is unbiased.

    subtract_background(field, roi=None, bg_mask=None, method="gaussian",
                        flatten_sigma=25.0) -> flattened field

Registry keys:
  "gaussian"  subtract a large-scale Gaussian (needs no bg region). Default.
              flatten_sigma must be LARGER than the domains or it tracks them
              and hollows their interiors (raise to ~150 for zoomed frames).
  "bg_region" subtract the MEAN of a drawn non-magnetic region (bg_mask) --
              one constant DC level. Correct only if the background is flat.
  "plane"     subtract a fitted plane a*x+b*y+c to the bg_mask region (removes
              a linear gradient). Opt-in; sensitive to bg-region purity, so
              use only with a clean, well-spread background region.
  "none"      no correction (field passed through).
"""
import numpy as np
from scipy import ndimage as ndi

from . import make_registry

BACKGROUNDS, _register = make_registry()


def _fast_gaussian_lowpass(field, sigma):
    """Large-sigma Gaussian low-pass (the background estimate) computed on a
    downsampled copy and bilinearly upsampled. A sigma~150 Gaussian is an
    extremely smooth field, so blurring at 1/d resolution (working sigma
    sigma/d ~ 8) and zooming back is visually identical to the full-kernel
    convolution but ~d^2 cheaper -- on a multi-MP frame the exact filter uses
    a ~1200-tap kernel and costs seconds. Below sigma~30 the exact filter is
    already cheap, so use it (and keep results identical there)."""
    sigma = float(sigma)
    if sigma <= 30:
        return ndi.gaussian_filter(field, sigma=sigma)
    d = max(1, int(sigma // 8))                       # downsample factor
    small = field[::d, ::d]
    sb = ndi.gaussian_filter(small, sigma=sigma / d)
    bg = ndi.zoom(sb, (field.shape[0] / sb.shape[0],
                       field.shape[1] / sb.shape[1]), order=1)
    return bg


@_register("gaussian", "Gaussian high-pass (no region needed)",
           "Subtract a large-scale Gaussian background (sigma = flatten_sigma). "
           "Sigma must exceed the domain size or it follows the domains and "
           "hollows them out -- raise it (~150) for zoomed / large-domain frames.")
def gaussian(field, roi=None, bg_mask=None, flatten_sigma=25.0, **_):
    return field - _fast_gaussian_lowpass(field, flatten_sigma)


@_register("bg_region", "Background-region mean (constant)",
           "Subtract the mean of a drawn non-magnetic region as the magnetic "
           "zero. One constant DC level -- correct only if the background is flat.")
def bg_region(field, roi=None, bg_mask=None, flatten_sigma=25.0, **_):
    if bg_mask is None or not np.asarray(bg_mask, bool).any():
        return gaussian(field, flatten_sigma=flatten_sigma)   # graceful fallback
    return field - float(field[np.asarray(bg_mask, bool)].mean())


@_register("plane", "Background-region plane (linear gradient)",
           "Fit a tilted plane a*x+b*y+c to the drawn background region and "
           "subtract it -- removes a linear gradient. Sensitive to bg-region "
           "purity; use only with a clean, full-width background region.")
def plane(field, roi=None, bg_mask=None, flatten_sigma=25.0, **_):
    bg = None if bg_mask is None else np.asarray(bg_mask, bool)
    if bg is None or not bg.any():
        return gaussian(field, flatten_sigma=flatten_sigma)
    ys, xs = np.nonzero(bg)
    if xs.size < 10:
        return field - float(field[bg].mean())
    A = np.c_[xs.astype(float), ys.astype(float), np.ones(xs.size)]
    coef, *_2 = np.linalg.lstsq(A, field[ys, xs], rcond=None)
    H, W = field.shape
    yy, xx = np.mgrid[0:H, 0:W]
    return field - (coef[0] * xx + coef[1] * yy + coef[2])


@_register("none", "None (no flattening)", "Pass the field through unchanged.")
def none(field, roi=None, bg_mask=None, flatten_sigma=25.0, **_):
    return np.asarray(field, float)


def subtract_background(field, roi=None, bg_mask=None, method=None,
                        flatten_sigma=25.0):
    """
    Dispatch background correction. If `method` is None it is inferred for
    backward compatibility: "bg_region" when a bg_mask is given, else
    "gaussian" -- exactly the original analyze_xmcd behavior.
    """
    if method is None:
        method = "bg_region" if (bg_mask is not None and
                                 np.asarray(bg_mask, bool).any()) else "gaussian"
    if method not in BACKGROUNDS:
        raise ValueError(f"unknown background method {method!r}; "
                         f"choices: {list(BACKGROUNDS)}")
    return BACKGROUNDS[method].fn(field, roi=roi, bg_mask=bg_mask,
                                  flatten_sigma=flatten_sigma)
