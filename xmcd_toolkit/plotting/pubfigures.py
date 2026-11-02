"""
pubfigures.py -- clean, single-purpose PUBLICATION figures for the export
(Priority-2 images): up/down states, local width map, domain boundary,
asymmetry map, local net-m_z map, and a combined key montage.

Deliberately minimal: no scale bars, one panel per file (plus the montage),
saved to whatever paths the caller gives. Diagnostic/multi-panel figures
(the step-by-step pipeline, method comparisons, fits) live elsewhere.
"""
from .labels import relabel_figure
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

from ..colors import CMAP3_HEX

_CMAP3 = ListedColormap(CMAP3_HEX)
_NORM3 = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], _CMAP3.N)


def _save(fig, out_path, dpi=200):
    # Rasterize the image (imshow) artists but leave text, colorbars, contour
    # and boundary lines vector. This keeps vector formats (EPS/PDF) faithful:
    # the PostScript backend cannot blend stacked alpha overlays (the gray base
    # + coloured state map), so those must be baked to a raster -- while labels
    # and boundary contours stay crisp vector. Harmless for PNG.
    for ax in fig.axes:
        for im in ax.get_images():
            im.set_rasterized(True)
    relabel_figure(fig)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _width_map_array(mz, mm, um_per_px):
    """local domain width = 2*DT over up+down, in nm if calibrated else px."""
    scale = (um_per_px * 1000.0) if um_per_px else 1.0
    wmap = np.zeros(mz.shape)
    for st in (mm["up"], mm["down"]):
        if st.get("dt") is not None:
            wmap += 2.0 * st["dt"]
    return np.where(mz != 0, wmap * scale, np.nan)


def up_down_states(img, mz, roi, out_path, title=""):
    """Three-state up/neutral/down map over the flake image."""
    fig, ax = plt.subplots(figsize=(5.4, 5.4))
    ax.imshow(img, cmap="gray")
    ax.imshow(np.where(roi, mz, np.nan), cmap=_CMAP3, norm=_NORM3, alpha=0.85)
    ax.set_title(title or "up (+1) / neutral (0) / down (-1)"); ax.axis("off")
    return _save(fig, out_path)


def asymmetry_map(m, roi, out_path, title=""):
    """Continuous normalized asymmetry m in [-1,1] (the order parameter)."""
    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    im = ax.imshow(np.where(roi, m, np.nan), cmap="seismic", vmin=-1, vmax=1)
    ax.set_title(title or "normalized asymmetry m"); ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="m")
    return _save(fig, out_path)


def width_map(img, mz, mm, um_per_px, out_path, title=""):
    """Local domain-width map (2*DT) with colorbar."""
    unit = "nm" if um_per_px else "px"
    wmap = _width_map_array(mz, mm, um_per_px)
    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    ax.imshow(img, cmap="gray")
    im = ax.imshow(wmap, cmap="viridis")
    ax.set_title(title or f"local domain width ({unit})"); ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=f"width ({unit})")
    return _save(fig, out_path)


def domain_boundary(img, boundary, out_path, title=""):
    """Domain boundary overlaid on the image. `boundary` is a
    methods.boundary.domain_boundary() result ({edge, contours})."""
    fig, ax = plt.subplots(figsize=(5.4, 5.4))
    ax.imshow(img, cmap="gray")
    if boundary.get("contours"):
        for c in boundary["contours"]:
            ax.plot(c[:, 1], c[:, 0], color="#00e5ff", lw=0.6)
    elif boundary.get("edge") is not None:
        ys, xs = np.where(boundary["edge"])
        ax.plot(xs, ys, ".", color="#00e5ff", ms=0.5)
    ax.set_title(title or "domain boundary"); ax.axis("off")
    ax.set_xlim(0, img.shape[1]); ax.set_ylim(img.shape[0], 0)
    return _save(fig, out_path)


def net_mz_map(nd, out_path, title=""):
    """Local net-m_z tile map (n_up - n_down per tile)."""
    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    im = ax.imshow(nd["net"], cmap="seismic", vmin=-1, vmax=1)
    ax.set_title(title or f"local net m_z (tile={nd.get('tile','?')}px)"); ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="net m_z")
    return _save(fig, out_path)


def key_montage(img, m, mz, roi, mm, boundary, um_per_px, out_path, title=""):
    """One-page 2x2: asymmetry | up/down | width map | boundary."""
    unit = "nm" if um_per_px else "px"
    wmap = _width_map_array(mz, mm, um_per_px)
    fig, ax = plt.subplots(2, 2, figsize=(11, 10.4))
    a = ax[0, 0]; im = a.imshow(np.where(roi, m, np.nan), cmap="seismic", vmin=-1, vmax=1)
    a.set_title("(a) asymmetry m"); a.axis("off")
    fig.colorbar(im, ax=a, fraction=0.046, pad=0.04)
    a = ax[0, 1]; a.imshow(img, cmap="gray")
    a.imshow(np.where(roi, mz, np.nan), cmap=_CMAP3, norm=_NORM3, alpha=0.85)
    a.set_title("(b) up / neutral / down"); a.axis("off")
    a = ax[1, 0]; a.imshow(img, cmap="gray"); im = a.imshow(wmap, cmap="viridis")
    a.set_title(f"(c) local width ({unit})"); a.axis("off")
    fig.colorbar(im, ax=a, fraction=0.046, pad=0.04)
    a = ax[1, 1]; a.imshow(img, cmap="gray")
    if boundary.get("contours"):
        for c in boundary["contours"]:
            a.plot(c[:, 1], c[:, 0], color="#00e5ff", lw=0.5)
    elif boundary.get("edge") is not None:
        ys, xs = np.where(boundary["edge"]); a.plot(xs, ys, ".", color="#00e5ff", ms=0.4)
    a.set_title("(d) domain boundary"); a.axis("off")
    a.set_xlim(0, img.shape[1]); a.set_ylim(img.shape[0], 0)
    if title:
        fig.suptitle(title, fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.97] if title else None)
    return _save(fig, out_path)
