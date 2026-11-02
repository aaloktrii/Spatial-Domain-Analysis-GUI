"""Flake statistics and pipeline-step figures (used by the batch export). Plotting layer."""
from .labels import relabel_figure
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from ..colors import COLOR_UP, COLOR_DOWN, CMAP3_HEX
from ..core import pdf_three_state


def flake_stats_figure(img, mz, roi, nd, out_png, title=""):
    masked = np.where(roi, mz, np.nan)
    pl = pdf_three_state(mz, region=roi)

    fig = plt.figure(figsize=(15, 4.4))
    cmap3 = matplotlib.colors.ListedColormap(CMAP3_HEX)

    ax = fig.add_subplot(1, 3, 1)
    ax.imshow(img, cmap="gray")
    ax.imshow(masked, cmap=cmap3, vmin=-1, vmax=1, alpha=0.85)
    ax.set_title("(a) flake three-state m_z"); ax.axis("off")

    ax = fig.add_subplot(1, 3, 2)
    ax.bar(["+1\nup", "0\nnone", "-1\ndown"],
           [pl["p_up"], pl["p_zero"], pl["p_down"]],
           color=[COLOR_UP, "#bbbbbb", COLOR_DOWN], edgecolor="k")
    ax.set_ylim(0, 1); ax.set_ylabel("probability (flake)")
    ax.set_title(f"(b) flake PDF   <m_z>={pl['net_m']:+.3f}")

    ax = fig.add_subplot(1, 3, 3)
    im = ax.imshow(nd["net"], cmap="seismic", vmin=-1, vmax=1)
    ax.set_title("(c) local net m_z (flake tiles)"); ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if title:
        fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    relabel_figure(plt.gcf())
    plt.savefig(out_png, dpi=130); plt.close(fig)


def _detailed_pipeline_figure(layers, mz, mm, m_field, um, out_png, stem, args,
                              boundary=None):
    """The opt-in 12-panel step-by-step pipeline diagnostic, built from the
    already-computed arrays (no recompute) and labelled with the ACTUAL
    settings used (denoiser, flatten_sigma, tv_weight). If `boundary` (a
    methods.boundary result) is given, its m=0 contour is overlaid (cyan) on
    the three-state panel. Saved in whatever format out_png's extension names
    (png/eps/pdf); image artists are rasterized so vector output stays faithful."""
    from matplotlib.colors import ListedColormap, BoundaryNorm
    cmap3 = ListedColormap(CMAP3_HEX); norm3 = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap3.N)
    img, den, lstd = layers["img"], layers["denoised"], layers["local_std"]
    roi, flat = layers["roi"], layers["flat"]
    bg = den - flat
    nm = (um * 1000.0) if um else 1.0
    U = "nm" if um else "px"
    wmap = np.zeros(mz.shape)
    for st in (mm["up"], mm["down"]):
        if st.get("dt") is not None:
            wmap += 2.0 * st["dt"]
    wmap = np.where(mz != 0, wmap * nm, np.nan)
    skel = np.zeros((*mz.shape, 4))
    for st, col in ((mm["up"], [0, 0, 1, 1]), (mm["down"], [1, 0, 0, 1])):
        if st.get("skeleton") is not None:
            skel[st["skeleton"]] = col
    ov = np.zeros((*img.shape, 4)); ov[layers["up"]] = [0, 0, 1, .55]; ov[layers["down"]] = [1, 0, 0, .55]
    dn_method = getattr(args, "denoise_method", "tv")
    tvw = getattr(args, "tv_weight", 0.2); fsig = getattr(args, "flatten_sigma", 25.0)

    fig, ax = plt.subplots(3, 4, figsize=(20, 15))
    def show(a, im, ttl, **kw):
        a.imshow(im, **kw); a.set_title(ttl, fontsize=12); a.axis("off")
    show(ax[0, 0], img, "(1) load + percentile normalize", cmap="gray")
    _d2 = f"(2) denoise: {dn_method} (w={tvw})"
    if boundary and boundary.get("contours"):
        _d2 = f"(2) denoise: {dn_method} + m=0 boundary"
    show(ax[0, 1], den, _d2, cmap="gray")
    if boundary and boundary.get("contours"):
        for c in boundary["contours"]:
            ax[0, 1].plot(c[:, 1], c[:, 0], color="#00e5ff", lw=0.35)
        ax[0, 1].set_xlim(0, img.shape[1]); ax[0, 1].set_ylim(img.shape[0], 0)
    show(ax[0, 2], lstd, "(3) local-std texture map", cmap="magma")
    ax[0, 3].imshow(img, cmap="gray"); ax[0, 3].contour(roi, colors="#39FF14", linewidths=1.3)
    ax[0, 3].set_title("(4) segmented flake ROI", fontsize=12); ax[0, 3].axis("off")
    show(ax[1, 0], bg, f"(5) background (sigma={fsig:g})", cmap="gray")
    show(ax[1, 1], flat, "(6) flattened field", cmap="gray")
    show(ax[1, 2], np.where(roi, m_field, np.nan), "(7) asymmetry m [-1,1]", cmap="seismic", vmin=-1, vmax=1)
    ax[1, 3].imshow(img, cmap="gray")
    ax[1, 3].imshow(np.where(roi, mz, np.nan), cmap=cmap3, norm=norm3, alpha=.85)
    _bt = "(8) three-state m_z"
    if boundary and boundary.get("contours"):
        for c in boundary["contours"]:
            ax[1, 3].plot(c[:, 1], c[:, 0], color="#00e5ff", lw=0.35)
        ax[1, 3].set_xlim(0, img.shape[1]); ax[1, 3].set_ylim(img.shape[0], 0)
        _bt = "(8) three-state m_z + m=0 boundary"
    ax[1, 3].set_title(_bt, fontsize=12); ax[1, 3].axis("off")
    ax[2, 0].imshow(img, cmap="gray"); ax[2, 0].imshow(ov)
    ax[2, 0].set_title("(9) up (blue) / down (red)", fontsize=12); ax[2, 0].axis("off")
    im = ax[2, 1].imshow(wmap, cmap="viridis")
    ax[2, 1].set_title(f"(10) local width map ({U})", fontsize=12); ax[2, 1].axis("off")
    fig.colorbar(im, ax=ax[2, 1], fraction=0.046, pad=0.04)
    ax[2, 2].imshow(np.where(roi, mz, np.nan), cmap=cmap3, norm=norm3); ax[2, 2].imshow(skel)
    ax[2, 2].set_title("(11) skeletons / medial axis", fontsize=12); ax[2, 2].axis("off")
    show(ax[2, 3], layers["wall"], "(12) domain walls", cmap="gray")
    fig.suptitle(f"{stem} -- pipeline step by step  "
                 f"(denoise={dn_method}, tv={tvw}, sigma={fsig:g})", fontsize=15, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    # rasterize image artists (keep text/lines vector) so EPS/PDF stay faithful
    for a in ax.ravel():
        for im in a.get_images():
            im.set_rasterized(True)
    relabel_figure(plt.gcf())
    plt.savefig(out_png, dpi=110); plt.close(fig)
    return out_png
