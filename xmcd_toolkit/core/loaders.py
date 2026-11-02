"""
loaders.py -- input handling.

Two entry points:
  load_image(path)        -> grayscale float array in [0,1], chrome stripped
  load_xy_mz_txt(path)    -> integer m_z grid (H,W) from a 3-column x,y,mz file
"""

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage import measure


def neighbor_structure(connectivity=1):
    """Structuring element for connected-component labelling.
    connectivity 1 -> 4-neighbours (edges only);
    connectivity 2 -> 8-neighbours (edges + diagonals/corners)."""
    return ndi.generate_binary_structure(2, 1 if int(connectivity) == 1 else 2)


# ----------------------------------------------------------------------
def _keep_largest(mask):
    lbl = measure.label(mask)
    if lbl.max() < 1:
        return mask
    sizes = np.bincount(lbl.ravel())
    sizes[0] = 0
    return lbl == sizes.argmax()


def _remove_small(mask, min_px, connectivity=None):
    if connectivity is not None:
        lbl, _ = ndi.label(mask, structure=neighbor_structure(connectivity))
    else:
        lbl = measure.label(mask)
    if lbl.max() < 1:
        return mask
    sizes = np.bincount(lbl.ravel())
    keep = np.where(sizes >= min_px)[0]
    keep = keep[keep != 0]
    return np.isin(lbl, keep)


# ----------------------------------------------------------------------
def load_image(path, strip_chrome=True, chroma_tol=12):
    """
    Load an XMCD-PEEM image as a normalized grayscale float array.

    If strip_chrome is True, colored acquisition-software UI (title bar,
    filename strip, buttons) is discarded by keeping only the largest
    near-grayscale (R==G==B) region -- the actual data frame.
    """
def load_image(path, strip_chrome=True, chroma_tol=12, clip=(0.5, 99.5)):
    """
    Load an XMCD-PEEM image as a normalized grayscale float array.

    Robust to 16-bit / float TIFFs and to hot/dead pixels: contrast is
    normalized between the `clip` percentiles (default 0.5-99.5) rather than
    raw min/max, so a few saturated pixels can no longer crush all the real
    domain contrast into a narrow gray band.

    For colour screenshots (RGB/RGBA), if strip_chrome is True the colored
    acquisition-software UI is discarded by keeping the largest near-grayscale
    region. Grayscale TIFFs are loaded directly (no chrome logic).
    """
    im = Image.open(path)
    is_color = im.mode in ("RGB", "RGBA", "P")
    if is_color:
        rgb = np.asarray(im.convert("RGB")).astype(float)
        R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        gray = (R + G + B) / 3.0
        if strip_chrome:
            chroma = np.maximum(np.abs(R - G), np.abs(G - B))
            data_area = _keep_largest(
                _remove_small(ndi.binary_fill_holes(chroma < chroma_tol), 2000)
            )
            if data_area.any():
                ys, xs = np.where(data_area)
                gray = gray[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    else:
        # 16-bit / float / 8-bit grayscale TIFF or PNG -> load as float directly
        gray = np.asarray(im.convert("F")).astype(float)

    lo, hi = np.percentile(gray, clip)
    if hi <= lo:                      # degenerate (flat) image
        lo, hi = gray.min(), gray.max()
    img = np.clip((gray - lo) / (hi - lo + 1e-12), 0.0, 1.0)
    return img


def _full_scale(a):
    """Largest value of the file's number format: 255 (8-bit), 65535 (16-bit), 1 (float in 0..1)."""
    if np.issubdtype(a.dtype, np.integer):
        return float(np.iinfo(a.dtype).max) if a.max() > 255 else 255.0
    return 1.0 if a.max() <= 1.0 else float(a.max())


def load_picture(path):
    """The picture as the file shows it: uint8 RGB (n_y, n_x, 3), colours unchanged (alpha dropped; a one-channel
    image repeated in R, G, B; 16-bit / float scaled by the format's full scale, not by its contrast)."""
    im = Image.open(path)
    if im.mode in ("RGB", "RGBA", "P", "CMYK", "YCbCr", "LAB", "HSV", "LA", "L", "1"):
        return np.asarray(im.convert("RGB"))
    raw = np.asarray(im)
    g = np.clip(raw.astype(float) / _full_scale(raw), 0, 1)
    return np.repeat((g * 255).round().astype(np.uint8)[..., None], 3, -1)


def detect_map_mode(path, chroma=30, min_fraction=0.005):
    """'rb' if the picture has colour (red / white / blue map: more than `min_fraction` of the pixels have a
    channel spread above `chroma`), else 'bw' (black / white map; also any one-channel or 16-bit image)."""
    im = Image.open(path)
    if im.mode not in ("RGB", "RGBA", "P", "CMYK", "YCbCr", "LAB", "HSV"):
        return "bw"
    rgb = np.asarray(im.convert("RGB"))[::4, ::4].astype(np.int16)
    spread = rgb.max(-1) - rgb.min(-1)
    return "rb" if (spread > chroma).mean() > min_fraction else "bw"


def load_asymmetry_map(path, mode="rb", clip=(0.5, 99.5)):
    """
    Load an ALREADY-CODED signed asymmetry map and decode it to the continuous
    order parameter m(x,y) in [-1, 1] -- red/blue up/down or black/white
    up/down -- WITHOUT the grayscale luminance path (which destroys the sign:
    pure red and pure blue have the same (R+G+B)/3, and strip_chrome would
    delete the colored domains as if they were UI).

    mode:
      "rb"  blue-red diverging map (neutral = white or black):
            m = (B - R) / (B + R + eps)   -> blue -> +1, red -> -1, white/black -> 0
            This matches the toolkit's OWN convention (blue = +1), so no sign
            flip is needed and the decoded field renders identically on export.
      "bw"  black-white grayscale map (neutral = mid-gray):
            m = 2*I/I_full - 1            -> white -> +1, black -> -1, mid -> 0
            on the file's own scale (I_full = 255, 65535 or 1): the picture is read as it is,
            no contrast stretch (`clip` is kept for the call signature only).

    Unlike load_image this does NOT strip chrome (colored domains ARE the data)
    and does NOT flatten -- a finished asymmetry map is already background-
    corrected, so the caller feeds the result straight in as the flattened
    field and skips illumination flattening. Restrict to the flake with a
    polygon/box ROI (or the local-contrast auto-detector) downstream.

      "auto" "rb" for a coloured picture, "bw" for a grey one (detect_map_mode).

    Returns a float array m in [-1, 1], shape (H, W).
    """
    mode = str(mode).lower()
    if mode == "auto":
        mode = detect_map_mode(path)
    im = Image.open(path)
    if mode == "bw":        # the file's own grey scale: black -1, mid-grey 0, white +1 (no contrast stretch)
        if im.mode in ("I;16", "I;16B", "I;16L", "I", "F"):
            raw = np.asarray(im)
            gray = raw.astype(float) / _full_scale(raw)
        else:
            gray = np.asarray(im.convert("RGB")).astype(float).mean(-1) / 255.0
        return np.clip(2.0 * gray - 1.0, -1.0, 1.0)
    if mode == "rb":
        rgb = np.asarray(im.convert("RGB")).astype(float)
        R, B = rgb[..., 0], rgb[..., 2]
        m = (B - R) / (B + R + 1e-6)          # blue -> +1, red -> -1, white/black -> ~0
        return np.clip(m, -1.0, 1.0)
    raise ValueError(f"unknown asymmetry-map mode {mode!r} (expected 'auto', 'rb' or 'bw')")


def load_xy_mz_txt(path, mz_round=True, delimiter=None):
    """
    Build an (H, W) integer m_z grid from a whitespace/CSV text file with
    three columns: x, y, mz.

    x, y are treated as integer pixel coordinates (0-indexed, gaps filled
    with 0). mz is mapped to {-1, 0, +1} by sign if mz_round is True;
    otherwise the raw values are kept as a float grid.
    """
    try:
        data = np.loadtxt(path, delimiter=delimiter, comments=("#", "%"))
    except ValueError:
        # tolerate a single non-numeric header row (e.g. "x,y,mz")
        data = np.loadtxt(path, delimiter=delimiter, comments=("#", "%"), skiprows=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    x = data[:, 0].astype(int)
    y = data[:, 1].astype(int)
    mz = data[:, 2]
    x -= x.min(); y -= y.min()
    H, W = y.max() + 1, x.max() + 1
    grid = np.zeros((H, W), dtype=(int if mz_round else float))
    grid[y, x] = np.sign(mz).astype(int) if mz_round else mz
    return grid
