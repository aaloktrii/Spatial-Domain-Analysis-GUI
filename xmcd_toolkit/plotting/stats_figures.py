"""Figures for core/stats.py (plotting layer, no analysis)."""
import numpy as np
from scipy import ndimage as ndi
from skimage import measure
from ..core.domain import analyze_xmcd
from ..core.loaders import neighbor_structure, _remove_small
from ..colors import COLOR_UP, COLOR_DOWN, COLOR_NEUTRAL, CMAP3_HEX
from ..core.stats import (
    domain_wall_distribution,
    number_density_local,
    pdf_three_state,
)


def make_domain_wall_figure(img, roi, nd, out_png, title="", tile=None):
    """
    Two-panel figure of the local domain-wall fraction D:
      (a) spatial map of D (nd["n_zero"]) over the flake image, and
      (b) histogram of the per-tile D values (its spatial distribution).
    Returns (out_png, domain_wall_distribution(nd) dict).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dist = domain_wall_distribution(nd)
    tile = tile if tile is not None else nd.get("tile")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))

    ax[0].imshow(img, cmap="gray")
    # upsample the per-tile map to image extent so it overlays the frame
    Dmap = nd["n_zero"]
    im = ax[0].imshow(Dmap, cmap="magma", vmin=0, vmax=1, alpha=0.85,
                      extent=[0, img.shape[1], img.shape[0], 0],
                      interpolation="nearest")
    ax[0].set_title(f"(a) local domain-wall fraction D  (tile={tile}px)")
    ax[0].axis("off")
    fig.colorbar(im, ax=ax[0], fraction=0.046, pad=0.04, label="D")

    if dist["n_tiles"]:
        ax[1].hist(dist["values"], bins=24, color="#8a4fa0", edgecolor="white", alpha=0.9)
        ax[1].axvline(dist["median"], color="k", ls="--", lw=1.4,
                      label=f"median={dist['median']:.3f}")
        ax[1].axvline(dist["mean"], color="#b23a3a", ls=":", lw=1.4,
                      label=f"mean={dist['mean']:.3f}")
        ax[1].legend(fontsize=9)
    ax[1].set_xlabel("local domain-wall fraction D (per tile)")
    ax[1].set_ylabel("number of tiles")
    ax[1].set_title(f"(b) spatial distribution of D  (n={dist['n_tiles']} tiles, "
                    f"SD={dist['std']:.3f})")

    if title:
        fig.suptitle(title, fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95] if title else None)
    plt.savefig(out_png, dpi=130); plt.close(fig)
    return out_png, dist

# ----------------------------------------------------------------------
def make_stats_figure(mz, roi, out_png, tile=16, title=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pf = pdf_three_state(mz)
    pl = pdf_three_state(mz, region=roi)
    nd = number_density_local(mz, tile=tile)

    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(2, 3, 1)
    cmap3 = matplotlib.colors.ListedColormap(CMAP3_HEX)
    ax.imshow(mz, cmap=cmap3, vmin=-1, vmax=1)
    ax.set_title("(a) three-state m_z"); ax.axis("off")

    ax = fig.add_subplot(2, 3, 2)
    cases = ["+1\nup", "0\nnone", "-1\ndown"]
    ax.bar(cases, [pf["p_up"], pf["p_zero"], pf["p_down"]],
           color=[COLOR_UP, COLOR_NEUTRAL, COLOR_DOWN], alpha=0.85, label="whole frame")
    ax.bar(cases, [pl["p_up"], pl["p_zero"], pl["p_down"]],
           fill=False, edgecolor="k", linewidth=1.8, label="flake only")
    ax.set_ylabel("probability"); ax.set_title("(b) three-case PDF"); ax.legend(fontsize=8)

    ax = fig.add_subplot(2, 3, 3)
    ax.text(0.02, 0.5,
            f"flake <m_z> = {pl['net_m']:+.3f}\n"
            f"flake p(+1/0/-1) =\n  {pl['p_up']:.2f} / {pl['p_zero']:.2f} / {pl['p_down']:.2f}\n\n"
            f"tile = {tile}px\nlocal n_s = N_s/N_total",
            fontsize=11, va="center", family="monospace")
    ax.axis("off"); ax.set_title("(c) summary")

    for k, (key, ttl, cm) in enumerate([
            ("n_up", "(d) local n_up", "viridis"),
            ("n_down", "(e) local n_down", "viridis"),
            ("net", "(f) local net m_z", "seismic")]):
        ax = fig.add_subplot(2, 3, 4 + k)
        if key == "net":
            im = ax.imshow(nd[key], cmap=cm, vmin=-1, vmax=1)
        else:
            im = ax.imshow(nd[key], cmap=cm, vmin=0, vmax=1)
        ax.set_title(ttl); ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if title:
        fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(out_png, dpi=130); plt.close(fig)
    return out_png
