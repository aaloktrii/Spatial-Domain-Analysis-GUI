"""
denoise.py -- swappable image denoisers, applied to the normalized [0,1]
grayscale BEFORE segmentation. All must be EDGE-PRESERVING (they may not blur
domain walls, which would corrupt widths) -- so a plain Gaussian is not
offered here (that is background removal, see background.py).

    denoise(img, method="tv", **params) -> denoised image (same shape/range)

Registry keys: "tv" (default), "nlm", "median", "none".
"""
import numpy as np
from skimage import restoration
from scipy import ndimage as ndi

from . import make_registry

DENOISERS, _register = make_registry()


@_register("tv", "TV-Chambolle (edge-preserving)",
           "Total-variation denoising. Flattens shot-noise inside domains "
           "while keeping walls sharp. `tv_weight` sets strength (higher = "
           "smoother). The toolkit default and the single biggest knob on "
           "measured width.")
def tv(img, tv_weight=0.20, **_):
    return restoration.denoise_tv_chambolle(img, weight=tv_weight)


@_register("nlm", "Non-local means",
           "Non-local means denoising (the method used by recent FGT/STXM "
           "domain papers). Edge-preserving; often gives a flatter "
           "width-vs-denoising curve than TV. `nlm_h` sets strength.")
def nlm(img, nlm_h=None, tv_weight=0.20, **_):
    # robust high-frequency noise estimate (MAD of the median-filter residual)
    # -- avoids skimage.estimate_sigma, which needs PyWavelets. If nlm_h is not
    # given, h scales with that noise sigma.
    resid = img - ndi.median_filter(img, size=3)
    sigma = max(1.4826 * float(np.median(np.abs(resid - np.median(resid)))), 1e-3)
    h = float(nlm_h) if nlm_h is not None else sigma
    return restoration.denoise_nl_means(img, h=h, sigma=sigma, fast_mode=True,
                                        patch_size=5, patch_distance=6)


@_register("median", "Median filter",
           "Median blur -- removes salt-and-pepper speckle, edge-preserving "
           "for step edges. `median_size` is the window (px).")
def median(img, median_size=3, **_):
    return ndi.median_filter(img, size=int(median_size))


@_register("none", "None (no denoising)",
           "Skip denoising entirely -- classify the raw normalized image.")
def none(img, **_):
    return np.asarray(img, float)


def denoise(img, method="tv", **params):
    """Dispatch to the named denoiser. Unknown method -> ValueError."""
    if method not in DENOISERS:
        raise ValueError(f"unknown denoise method {method!r}; "
                         f"choices: {list(DENOISERS)}")
    return DENOISERS[method].fn(img, **params)
