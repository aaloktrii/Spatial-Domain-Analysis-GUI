"""Figures for core/domain.py (plotting layer, no analysis)."""
import numpy as np
from scipy import ndimage as ndi
from skimage import filters, morphology, measure, restoration, exposure
from ..core.loaders import (load_image, load_asymmetry_map,
                      _keep_largest, _remove_small, neighbor_structure)
from ..colors import OVERLAY_UP, OVERLAY_DOWN


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
