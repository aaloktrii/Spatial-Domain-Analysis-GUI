"""Igor graph exports (.tif / .png) read back into the map they show (no plotting, no GUI).

Read an Igor graph export (.tif: RGB/RGBA or CMYK, uint8, image upsampled to screen pixels, maybe a black frame)
back into the map it shows: one value per original image pixel.

Steps:
  1. memory-map the file (uncompressed TIFF: nothing is read until used)
  2. plot area = the largest block of rows / columns that are not black frame
  3. original pixel grid: each image pixel is drawn as a b x b block of one colour (b need not be an integer);
     b and the grid origin are fitted to the colour changes (largest b with coherent edges: b/2, b/3 fit too)
  4. read one screen pixel per block on a square grid (step = the smaller block size in x and y, so the picture
     keeps its shape in the file; exact colours, no mixing; ~0.1 % of the file)
  5. colour -> value: 'grey' (luminance), 'blue-red' (sign(B - R) * (1 - min(R, G, B) / 255): blue / navy +1,
     white 0, red -1; equal to (B - R) / 255 for a pure blue-white-red table), or a colour bar
     image (nearest colour along the bar, top = +1, bottom = -1); 'auto' picks grey or blue-red
The value is in colour-scale units (the Igor colour range, white = its zero), not XMCD, unless the range is given.
Results are cached (.npz, keyed by path, size and modification time), so a second load is instant.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import tifffile

CACHE = Path.home() / ".cache" / "xmcd_compare_tiff"


def _open(path):
    if Path(path).suffix.lower() not in (".tif", ".tiff"):   # PNG, JPG, BMP, ...: decoded once (cached afterwards)
        from PIL import Image
        with Image.open(path) as im:
            return np.asarray(im.convert("RGB")), "RGB"
    try:
        a = tifffile.memmap(path, mode="r")
    except Exception:   # compressed or tiled: read it all once (cached afterwards)
        a = tifffile.imread(path)
    with tifffile.TiffFile(path) as t:
        photometric = t.pages[0].photometric.name
    return a, photometric


def _rgb(px, photometric):
    """uint8 pixels (..., C) -> float RGB 0-255."""
    px = np.asarray(px, float)
    if px.ndim == 2 or px.shape[-1] == 1:
        g = px.reshape(px.shape[:2] + (1,)) if px.ndim == 2 else px
        return np.repeat(g, 3, -1)
    if photometric == "SEPARATED":   # CMYK
        c, m, y, k = (px[..., i] / 255 for i in range(4))
        return np.stack([(1 - c) * (1 - k), (1 - m) * (1 - k), (1 - y) * (1 - k)], -1) * 255
    return px[..., :3]


def _longest_run(mask):
    best, start, cur = (0, 0), None, 0
    for i, v in enumerate(np.append(mask, False)):
        if v and start is None:
            start = i
        if not v and start is not None:
            if i - start > best[1] - best[0]:
                best = (start, i)
            start = None
    return best


def plot_area(a, photometric, step=8):
    """(y0, y1, x0, x1) of the image inside a black frame (rows / columns that are mostly not black)."""
    sub = _rgb(a[::step, ::step], photometric)
    black = sub.max(-1) < 12
    y0, y1 = _longest_run(black.mean(1) < 0.5)
    x0, x1 = _longest_run(black[y0:y1].mean(0) < 0.5)
    y0, y1, x0, x1 = y0 * step, min(y1 * step, a.shape[0]), x0 * step, min(x1 * step, a.shape[1])

    # refine each edge to the exact pixel (first / last line that is mostly not black)
    def edge(lines, forward):
        idx = range(len(lines)) if forward else range(len(lines) - 1, -1, -1)
        for i in idx:
            if lines[i] < 0.5:
                return i
        return 0 if forward else len(lines) - 1

    mid_y, mid_x = slice(y0 + (y1 - y0) // 4, y1 - (y1 - y0) // 4), slice(x0 + (x1 - x0) // 4, x1 - (x1 - x0) // 4)
    ry = slice(max(y0 - step, 0), min(y1 + step, a.shape[0]))
    rx = slice(max(x0 - step, 0), min(x1 + step, a.shape[1]))
    col_black = (_rgb(a[ry, mid_x][:, ::4], photometric).max(-1) < 12).mean(1)
    row_black = (_rgb(a[mid_y, rx][::4], photometric).max(-1) < 12).mean(0)
    Y0, Y1 = ry.start + edge(col_black, True), ry.start + edge(col_black, False) + 1
    X0, X1 = rx.start + edge(row_black, True), rx.start + edge(row_black, False) + 1
    return int(Y0), int(Y1), int(X0), int(X1)


def _grid(lines, start, b_range=(1.5, 60.0)):
    """Block size b and phase p (edges at p + i*b, screen px) along one axis, from the colour changes in sampled
    lines (k, length, 3) starting at screen position `start`. Coherence |<exp(2 pi i x / b)>| is ~1 at the block
    size and at every b / m, so take the largest b whose coherence is within 15 % of the best."""
    ch = np.any(np.abs(np.diff(lines, axis=1)) > 0, axis=-1)
    pos = np.nonzero(ch)[1].astype(float) + 1.0 + start
    if pos.size < 50:
        raise ValueError("too few colour changes to find the pixel grid")
    rng = np.random.default_rng(0)
    p = pos if pos.size <= 1500 else rng.choice(pos, 1500, replace=False)
    bs = np.exp(np.arange(np.log(b_range[0]), np.log(b_range[1]), 5e-4))
    coh = np.empty(bs.size)
    for s0 in range(0, bs.size, 512):
        ph = np.exp(2j * np.pi * p[None, :] / bs[s0:s0 + 512, None])
        coh[s0:s0 + 512] = np.abs(ph.mean(1))
    good = np.nonzero(coh >= 0.85 * coh.max())[0]
    i = good[-1]
    # the peak around the largest good b
    lo, hi = max(i - 16, 0), min(i + 16, bs.size)
    i = lo + int(np.argmax(coh[lo:hi]))
    b = float(bs[i])
    for _ in range(3):   # refine on all edges
        fine = b * (1 + np.linspace(-6e-4, 6e-4, 121))
        c = np.abs(np.exp(2j * np.pi * pos[None, :] / fine[:, None]).mean(1))
        b = float(fine[np.argmax(c)])
    z = np.exp(2j * np.pi * pos / b).mean()
    phase = float(np.angle(z) / (2 * np.pi) * b) % b
    return b, phase, float(np.abs(z))


def _centres(lo, hi, b, phase):
    """Screen positions of the centres of the blocks whose centre lies inside [lo, hi) (edge blocks may be cut)."""
    k0 = int(np.ceil((lo - phase) / b - 0.5))
    k1 = int(np.floor((hi - 1 - phase) / b - 0.5)) + 1
    return (phase + (np.arange(k0, k1) + 0.5) * b).astype(int)


def to_value(rgb, colour="auto", colorbar=None):
    """RGB (..., 3) 0-255 -> value."""
    if colour == "auto":
        spread = np.nanmedian(rgb.max(-1) - rgb.min(-1))
        colour = "grey" if spread < 3 else "blue-red"
    if colour == "grey":
        return rgb.mean(-1) / 127.5 - 1.0, colour
    if colour == "blue-red":   # sign from blue vs red, size = distance from white (works for dark navy / dark red ends)
        return np.sign(rgb[..., 2] - rgb[..., 0]) * (1.0 - rgb.min(-1) / 255.0), colour
    if colour == "colorbar":
        bar = np.asarray(colorbar, float)                    # (m, 3), index 0 = +1 (top), last = -1 (bottom)
        vals = np.linspace(1, -1, len(bar))
        flat = rgb.reshape(-1, 3)
        out = np.empty(len(flat))
        for s in range(0, len(flat), 65536):
            d = ((flat[s:s + 65536, None, :] - bar[None]) ** 2).sum(-1)
            out[s:s + 65536] = vals[np.argmin(d, 1)]
        return out.reshape(rgb.shape[:-1]), colour
    raise ValueError("colour must be auto, grey, blue-red or colorbar")


def colorbar_palette(path, m=256):
    """Colours along a colour-bar image (the long axis, top / left = +1), black frame and labels skipped."""
    a, ph = _open(path)
    y0, y1, x0, x1 = plot_area(a, ph, step=2)
    rgb = _rgb(a[y0:y1, x0:x1], ph)
    vertical = (y1 - y0) >= (x1 - x0)
    line = np.median(rgb[:, (x1 - x0) // 3:2 * (x1 - x0) // 3], axis=1) if vertical else \
        np.median(rgb[(y1 - y0) // 3:2 * (y1 - y0) // 3], axis=0)
    idx = np.linspace(0, len(line) - 1, m).round().astype(int)
    return line[idx]


def sample_grid(path, use_cache=True):
    """-> (RGB uint8 (n_y, n_x, 3), info): one screen pixel per image-pixel block (steps 1-4), on a SQUARE grid:
    the same step (the smaller block size) along y and x, so the picture keeps the shape it has in the file (the file
    is true to scale; blocks may be wider than tall). Along the axis with the larger blocks some blocks are read
    twice; every value is a colour of the file (nothing blended or interpolated).
    Cached (.npz, keyed by path, size and modification time), so a second call is instant."""
    path = Path(path)
    st = path.stat()
    key = hashlib.sha1(f"grid2|{path.resolve()}|{st.st_size}|{st.st_mtime_ns}".encode()).hexdigest()[:16]
    cf = CACHE / f"{path.stem[:60]}_{key}.npz"
    if use_cache and cf.exists():
        z = np.load(cf, allow_pickle=False)
        return z["rgb"], json.loads(str(z["info"]))
    a, ph = _open(path)
    y0, y1, x0, x1 = plot_area(a, ph)
    H, W = y1 - y0, x1 - x0
    ys = np.linspace(y0 + H * 0.1, y1 - H * 0.1, 24).astype(int)
    xs = np.linspace(x0 + W * 0.1, x1 - W * 0.1, 24).astype(int)
    try:
        bx, px, sx = _grid(_rgb(a[ys, x0:x1], ph), x0)
        by, py, sy = _grid(np.swapaxes(_rgb(a[y0:y1, xs], ph), 0, 1), y0)
        upsampled = min(sx, sy) > 0.5 and min(bx, by) >= 1.6
    except ValueError:
        upsampled = False
    if upsampled:          # square grid: step = smaller block, on the block centres of that axis
        step = min(bx, by)
        if by <= bx:
            cy = _centres(y0, y1, by, py)
            cx = (x0 + (np.arange(int((x1 - x0) / step)) + 0.5) * step).astype(int)
        else:
            cx = _centres(x0, x1, bx, px)
            cy = (y0 + (np.arange(int((y1 - y0) / step)) + 0.5) * step).astype(int)
    else:                 # not drawn as blocks (e.g. a plain image): one value per file pixel
        bx = by = 1.0
        sx = sy = 0.0
        cy, cx = np.arange(y0, y1), np.arange(x0, x1)
    rgb = np.clip(np.rint(_rgb(np.asarray(a[cy][:, cx]), ph)), 0, 255).astype(np.uint8)
    info = {"source": str(path), "photometric": ph, "file_shape": list(a.shape), "plot_area_yx": [y0, y1, x0, x1],
            "n_yx": [int(cy.size), int(cx.size)], "block_px_yx": [by, bx], "grid_fit_yx": [sy, sx]}
    if use_cache:
        CACHE.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cf, rgb=rgb, info=json.dumps(info))
    return rgb, info


def load_igor_tiff(path, colour="auto", colorbar=None, use_cache=True):
    """-> (value map float32 (n_y, n_x), info dict). colorbar: path of a colour-bar image (colour='colorbar')."""
    rgb, info = sample_grid(path, use_cache)
    pal = colorbar_palette(colorbar) if colour == "colorbar" else None
    val, used = to_value(rgb.astype(float), colour, pal)
    return val.astype(np.float32), {**info, "colour": used, "colorbar": str(colorbar) if colorbar else None}

def source_for(path, input_mode, map_mode="auto", invert=False):
    """What analyze_xmcd should read for this file and input type -> (image path or array, input_mode).
    Only an explicitly chosen Igor export is decoded; every other file is read as it is, every pixel.
    invert=True swaps up and down (a map whose colours mean the opposite sign): the decoded map is negated
    (m -> -m), a raw grey image inverted (I -> 1 - I)."""
    if input_mode == "igor_export":
        val, mode = decode_for_analysis(path)
        return (-val if invert else val), mode
    if not invert:
        return path, input_mode
    from .loaders import load_asymmetry_map, load_image
    if input_mode == "asymmetry_map":
        return -load_asymmetry_map(path, mode=map_mode), "asymmetry_map"
    return 1.0 - load_image(path), input_mode

def analysis_shape(path, input_mode="grayscale"):
    """(H, W) of the pixel grid the analysis works on: the original grid of an Igor export or of a large picture
    drawn as pixel blocks, else the file size. Polygons, masks and the FoV calibration refer to this grid."""
    if input_mode == "igor_export":
        return tuple(sample_grid(path)[1]["n_yx"])
    from PIL import Image
    with Image.open(path) as im:
        return im.height, im.width

def decode_for_analysis(path):
    """Igor export -> (signed map on the original pixel grid, "asymmetry_map"): the colour scale read back
    (grey: white +1 / black -1; blue-red: blue +1 / white 0 / red -1)."""
    return load_igor_tiff(path)[0], "asymmetry_map"


def is_rendered_tiff(path):
    """True for 8-bit RGB / RGBA / CMYK TIFFs and 8-bit PNGs (pictures / graph exports), False for data images
    (float / 16-bit / one channel)."""
    if Path(path).suffix.lower() not in (".tif", ".tiff"):
        from PIL import Image
        with Image.open(path) as im:
            return im.mode in ("RGB", "RGBA", "P", "L", "LA", "CMYK")
    with tifffile.TiffFile(path) as t:
        p = t.pages[0]
        return p.dtype == np.uint8 and p.samplesperpixel >= 3


if __name__ == "__main__":
    import sys
    import time
    for f in sys.argv[1:]:
        t = time.time()
        v, info = load_igor_tiff(f, use_cache=False)
        print(f"{Path(f).name}: {info['n_yx']} px, block {info['block_px_yx'][0]:.3f}x{info['block_px_yx'][1]:.3f}, "
              f"grid fit {info['grid_fit_yx'][0]:.2f}/{info['grid_fit_yx'][1]:.2f}, plot {info['plot_area_yx']}, "
              f"{info['colour']}, range {np.nanmin(v):+.2f}..{np.nanmax(v):+.2f}, {time.time() - t:.2f} s")
