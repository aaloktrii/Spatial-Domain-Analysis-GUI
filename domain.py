"""
domain.py -- segmentation and domain metrics.

Pipeline:
  0. (load) chrome-strip handled by loaders.load_image
  1. ROI segmentation via local-contrast texture
  2. illumination flattening
  3. up/down classification (Otsu on flattened contrast)
  4. metrics: areas, net magnetization, domain counts, size distribution,
     domain-wall length and density (fragmentation metric)
"""

import numpy as np
from scipy import ndimage as ndi
from skimage import filters, morphology, measure, restoration, exposure

from .loaders import load_image, _keep_largest, _remove_small, neighbor_structure
from .colors import OVERLAY_UP, OVERLAY_DOWN


def _box_mask(shape, box):
    """box = (x0, y0, x1, y1) in pixels -> boolean mask."""
    H, W = shape
    x0, y0, x1, y1 = box
    x0, x1 = sorted((max(0, int(x0)), min(W, int(x1))))
    y0, y1 = sorted((max(0, int(y0)), min(H, int(y1))))
    m = np.zeros(shape, bool)
    m[y0:y1, x0:x1] = True
    return m


def analyze_xmcd(
    image,
    um_per_px=None,        # calibration; set from the scale bar (TODO per image)
    tv_weight=0.12,        # total-variation denoising strength
    contrast_window=9,     # px window for the local-contrast ROI detector
    flatten_sigma=25,      # px scale for illumination flattening
    min_roi_px=800,        # discard ROI blobs smaller than this
    min_domain_px=12,      # discard domains smaller than this when counting
    connectivity=1,        # domain labelling: 1 = 4-conn (edges), 2 = 8-conn (+diag)
    clahe=False,           # local contrast enhancement (reduces flat gray)
    clahe_clip=0.02,       # CLAHE clip limit (higher = stronger)
    roi_box=None,          # (x0,y0,x1,y1) manual ACTIVE AREA; overrides auto
    roi_mask=None,         # boolean array: arbitrary ACTIVE AREA (e.g. polygon);
                           #   highest priority, overrides roi_box and auto-detect
    bg_box=None,           # (x0,y0,x1,y1) BACKGROUND region for zero reference
    bg_mask=None,          # boolean array: arbitrary BACKGROUND region (polygon);
                           #   highest priority, overrides bg_box
    clip=(0.5, 99.5),      # percentile contrast clip at load (hot-pixel robust)
):
    """
    Analyze one XMCD-PEEM image.

    `image` may be a file path (str/Path) or a 2-D grayscale float array.

    Controls for "too much gray inside the active area":
      - clahe=True             sharpen weak domain contrast before classifying
      - roi_box=(x0,y0,x1,y1)  define the active area by hand (skips the
                               auto-detector, which can grab flat gray substrate)
      - bg_box=(x0,y0,x1,y1)   set the magnetic zero from a non-magnetic region,
                               so up/down are thresholded symmetrically about it
      - clip=(lo,hi)           percentile contrast normalization (hot-pixel safe)

    Returns (metrics: dict, layers: dict).
    """
    is_path = isinstance(image, (str, bytes)) or hasattr(image, "__fspath__")
    img = load_image(image, clip=clip) if is_path else np.asarray(image, float)

    # optional local contrast enhancement -- pulls weak domains out of the gray
    if clahe:
        img = exposure.equalize_adapthist(img, clip_limit=clahe_clip)

    # 1. active area (ROI): polygon mask > manual box > local-contrast auto-detect
    den = restoration.denoise_tv_chambolle(img, weight=tv_weight)

    # fast local standard deviation (separable; ~100x faster than generic_filter)
    _m = ndi.uniform_filter(den, size=contrast_window)
    _sq = ndi.uniform_filter(den * den, size=contrast_window)
    local_std = np.sqrt(np.maximum(_sq - _m * _m, 0.0))

    if roi_mask is not None:
        roi = np.asarray(roi_mask, bool)
        if roi.shape != img.shape:
            raise ValueError(f"roi_mask shape {roi.shape} != image {img.shape}")
    elif roi_box is not None:
        roi = _box_mask(img.shape, roi_box)
    else:
        roi = local_std > filters.threshold_otsu(local_std)
        roi = morphology.closing(roi, morphology.disk(5))
        roi = ndi.binary_fill_holes(roi)
        roi = _remove_small(roi, min_roi_px)
        roi = _keep_largest(roi)

    # 2. flatten illumination. If a background region is given (polygon mask or
    #    box), use its mean as the magnetic zero; otherwise subtract a
    #    large-scale Gaussian.
    if bg_mask is not None:
        bg = np.asarray(bg_mask, bool)
        if bg.shape != img.shape:
            raise ValueError(f"bg_mask shape {bg.shape} != image {img.shape}")
    elif bg_box is not None:
        bg = _box_mask(img.shape, bg_box)
    else:
        bg = None

    if bg is not None and bg.any():
        flat = den - float(den[bg].mean())
    else:
        flat = den - ndi.gaussian_filter(den, sigma=flatten_sigma)

    # 3. up / down classification.
    #    With a background zero, threshold symmetrically about 0; otherwise Otsu.
    if bg is not None and bg.any():
        up = roi & (flat > 0)
        down = roi & (flat <= 0)
    else:
        thr = filters.threshold_otsu(flat[roi])
        up = roi & (flat > thr)
        down = roi & (flat <= thr)

    # 4. metrics
    A_up, A_down = int(up.sum()), int(down.sum())
    A_tot = A_up + A_down
    m_net = (A_up - A_down) / A_tot if A_tot else float("nan")

    up_c = _remove_small(up, min_domain_px, connectivity=connectivity)
    down_c = _remove_small(down, min_domain_px, connectivity=connectivity)
    struct = neighbor_structure(connectivity)
    up_lbl, n_up = ndi.label(up_c, structure=struct)
    down_lbl, n_down = ndi.label(down_c, structure=struct)
    up_sizes = np.bincount(up_lbl.ravel())[1:] if n_up else np.array([])
    down_sizes = np.bincount(down_lbl.ravel())[1:] if n_down else np.array([])
    all_sizes = np.concatenate([up_sizes, down_sizes]) if (n_up + n_down) else np.array([])
    eq_diam_px = np.sqrt(4 * all_sizes / np.pi) if all_sizes.size else np.array([])

    roi_interior = ndi.binary_erosion(roi, iterations=1)
    wall = (ndi.binary_dilation(up) & down) & roi_interior
    wall_len_px = int(wall.sum())

    px = um_per_px
    a2 = (lambda a: a * px * px if px else None)
    l1 = (lambda l: l * px if px else None)
    roi_area_um2 = a2(A_tot)

    metrics = {
        "calibrated": px is not None, "um_per_px": px,
        "frame_shape_px": img.shape,
        "flake_area_px": A_tot, "flake_area_um2": roi_area_um2,
        "A_up_px": A_up, "A_down_px": A_down,
        "A_up_um2": a2(A_up), "A_down_um2": a2(A_down),
        "frac_up": A_up / A_tot if A_tot else float("nan"),
        "frac_down": A_down / A_tot if A_tot else float("nan"),
        "net_magnetization": m_net,
        "n_domains_up": int(n_up), "n_domains_down": int(n_down),
        "n_domains_total": int(n_up + n_down),
        "mean_domain_area_px": float(all_sizes.mean()) if all_sizes.size else float("nan"),
        "mean_domain_area_um2": a2(float(all_sizes.mean())) if (px and all_sizes.size) else None,
        "median_domain_diam_px": float(np.median(eq_diam_px)) if eq_diam_px.size else float("nan"),
        "median_domain_diam_um": l1(float(np.median(eq_diam_px))) if (px and eq_diam_px.size) else None,
        "domain_wall_len_px": wall_len_px, "domain_wall_len_um": l1(wall_len_px),
        "domain_wall_density_per_px": wall_len_px / A_tot if A_tot else float("nan"),
        "domain_wall_density_per_um": (wall_len_px / roi_area_um2) if (px and roi_area_um2) else None,
    }
    layers = {"img": img, "denoised": den, "local_std": local_std,
              "roi": roi, "flat": flat, "up": up, "down": down,
              "wall": wall, "eq_diam_px": eq_diam_px}
    return metrics, layers


def make_domain_figure(metrics, layers, out_png, title=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img, roi, up, down = layers["img"], layers["roi"], layers["up"], layers["down"]
    px = metrics["um_per_px"]
    unit = "um" if metrics["calibrated"] else "px"

    fig, ax = plt.subplots(1, 4, figsize=(17, 4.4))
    ax[0].imshow(img, cmap="gray"); ax[0].set_title("(a) XMCD-PEEM input")
    ax[1].imshow(img, cmap="gray"); ax[1].contour(roi, colors="#39FF14", linewidths=1.4)
    ax[1].set_title("(b) segmented flake")
    ov = np.zeros((*img.shape, 4)); ov[up] = OVERLAY_UP; ov[down] = OVERLAY_DOWN
    ax[2].imshow(img, cmap="gray"); ax[2].imshow(ov); ax[2].set_title("(c) up (blue) / down (red)")
    d = layers["eq_diam_px"]
    if metrics["calibrated"] and d.size:
        d = d * px
    if d.size:
        ax[3].hist(d, bins=24, color="#555", edgecolor="white")
    ax[3].set_title("(d) domain size distribution")
    ax[3].set_xlabel(f"equiv. diameter ({unit})"); ax[3].set_ylabel("count")
    for a in ax[:3]:
        a.axis("off")
    m = metrics
    area_str = f"{m['flake_area_um2']:.1f} um^2" if m["calibrated"] else f"{m['flake_area_px']} px"
    wall_str = (f"{m['domain_wall_density_per_um']:.3f} /um" if m["calibrated"]
                else f"{m['domain_wall_density_per_px']:.4f} /px")
    txt = (f"net m = {m['net_magnetization']:+.3f}   up/down = {m['frac_up']:.2f}/{m['frac_down']:.2f}\n"
           f"flake area = {area_str}   domains = {m['n_domains_total']}   wall density = {wall_str}")
    fig.suptitle((title + "\n" if title else "") + txt, fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.90]); plt.savefig(out_png, dpi=130); plt.close(fig)
    return out_png


def print_report(metrics):
    m = metrics
    print("=" * 56); print("  XMCD-PEEM DOMAIN ANALYSIS"); print("=" * 56)
    print(f"  calibrated ............. {'yes' if m['calibrated'] else 'NO (pixels only)'}")
    if m["calibrated"]:
        print(f"  scale .................. {m['um_per_px']:.4f} um/px")
    print(f"  flake area ............. {m['flake_area_px']} px"
          + (f"  ({m['flake_area_um2']:.2f} um^2)" if m["calibrated"] else ""))
    print(f"  up / down fraction ..... {m['frac_up']:.3f} / {m['frac_down']:.3f}")
    print(f"  NET MAGNETIZATION m .... {m['net_magnetization']:+.4f}")
    print(f"  number of domains ...... {m['n_domains_total']}  (up {m['n_domains_up']}, down {m['n_domains_down']})")
    md = m["mean_domain_area_um2"] if m["calibrated"] else m["mean_domain_area_px"]
    print(f"  mean domain area ....... {md:.2f} {'um^2' if m['calibrated'] else 'px'}")
    wd = m["domain_wall_density_per_um"] if m["calibrated"] else m["domain_wall_density_per_px"]
    print(f"  domain-wall density .... {wd:.4f} {'/um' if m['calibrated'] else '/px'}   <-- fragmentation")
    print("=" * 56)
