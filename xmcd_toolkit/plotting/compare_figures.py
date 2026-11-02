"""Figures for the comparison of states (plotting layer: matplotlib only, no analysis, no GUI).

Categorical maps (three-state, flip map) are drawn as vector shapes, one compound path per category with
pixel-exact outlines: they stay editable in Illustrator, and no raster is stored as indexed colour (which
Illustrator renders wrongly). Save with save_pdf() for an editable, Illustrator-safe PDF."""
import numpy as np
import matplotlib
from matplotlib.patches import Patch, PathPatch
from matplotlib.path import Path as MPath
from scipy import ndimage
from skimage.measure import find_contours

from ..colors import COLOR_UP, COLOR_DOWN, COLOR_UNCERTAIN, SWITCH_PALETTES

FLIP_LABELS = ("stayed down", "stayed up", "up → down", "down → up", "uncertain")
MAP_CMAP = "bwr_r"      # signed maps: positive (up) blue, negative (down) red


def flip_colours(palette="purple / magenta"):
    """stayed down, stayed up, up→down, down→up, uncertain"""
    a, b = SWITCH_PALETTES[palette]
    return [COLOR_DOWN, COLOR_UP, a, b, COLOR_UNCERTAIN]


def short(name, width=30):
    return name if len(name) <= width else name[:width - 1] + "…"


def display(v, smooth=1.5, flatten=40.0):
    """Display copy: Gaussian high-pass (large background gradients) and light smoothing; NaN kept."""
    fin = np.isfinite(v)
    out = np.nan_to_num(np.asarray(v, float))
    for s, subtract in ((flatten, True), (smooth, False)):
        if s > 0:
            w = np.maximum(ndimage.gaussian_filter(fin.astype(float), s), 1e-6)
            lp = ndimage.gaussian_filter(np.where(fin, out, 0.0), s) / w
            out = out - lp if subtract else lp
    return np.where(fin, out, np.nan)


def _norm01(v, mask):
    x = v[mask]
    if not x.size:
        return np.zeros_like(v)
    lo, hi = np.percentile(x, (1, 99))
    return np.clip((v - lo) / max(hi - lo, 1e-12), 0, 1)


def signed_map(ax, v, title="", extent=None):
    x = np.abs(v[np.isfinite(v)])
    lim = float(np.percentile(x, 99)) if x.size else 1.0
    im = ax.imshow(v, cmap=MAP_CMAP, vmin=-lim, vmax=lim, interpolation="nearest", extent=extent)
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


VECTOR_MAX_PX = 4_000_000    # larger maps are drawn as true-colour RGB pixels (vector outlines would be huge)


def categorical(ax, img, colours, title=""):
    """Integer-category image as vector filled shapes, one compound path per category. colours: {value: colour}.
    Above VECTOR_MAX_PX pixels: an RGB image, one pixel per map pixel (true colour, so still Illustrator-safe)."""
    h, w = img.shape
    if img.size > VECTOR_MAX_PX:
        rgb = np.ones((h, w, 3))
        v = np.nan_to_num(img, nan=-999)
        for val, col in colours.items():
            rgb[v == val] = matplotlib.colors.to_rgb(col)
        ax.imshow(rgb, interpolation="nearest")
        ax.set_title(title, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        return
    for val, col in colours.items():
        m = np.nan_to_num(img, nan=-999) == val
        if not m.any():
            continue
        up = np.kron(m, np.ones((2, 2), bool))                     # 2x: contours run along the pixel edges
        verts, codes = [], []
        for c in find_contours(np.pad(up, 1).astype(float), 0.5):
            xy = np.round(np.c_[(c[:, 1] - 1) / 2 - 0.25, (c[:, 0] - 1) / 2 - 0.25] * 2) / 2
            verts += list(xy) + [xy[0]]
            codes += [MPath.MOVETO] + [MPath.LINETO] * (len(xy) - 1) + [MPath.CLOSEPOLY]
        ax.add_patch(PathPatch(MPath(verts, codes), facecolor=col, edgecolor=col, lw=0.15))
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])


def outline(ax, mask, colour="k", lw=0.6, ls="-", step=1):
    if mask is not None and mask.any() and not mask.all():
        if step > 1:      # mask on every step-th pixel: contour drawn in full-grid coordinates
            h = (step - 1) / 2
            ys, xs = np.arange(mask.shape[0]) * step + h, np.arange(mask.shape[1]) * step + h
            ax.contour(xs, ys, mask.astype(float), levels=[0.5], colors=colour, linewidths=lw, linestyles=ls)
        else:
            ax.contour(mask.astype(float), levels=[0.5], colors=colour, linewidths=lw, linestyles=ls)


def screen_sample(a, max_px=1600):
    """For drawing only: every n-th pixel of a large image so that ~max_px are drawn across (a screen shows no
    more), and the extent that keeps the axes in full-image pixels. -> (array, extent or None)."""
    step = max(1, int(max(a.shape[:2]) // max_px))
    if step == 1:
        return a, None
    h = (step - 1) // 2
    s = a[h::step, h::step]
    return s, (-0.5, s.shape[1] * step - 0.5, s.shape[0] * step - 0.5, -0.5)


def picture(ax, rgb, title="", extent=None):
    """An image in its own colours (no colour map, no contrast change)."""
    ax.imshow(rgb, interpolation="nearest", extent=extent)
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])


def draw_align(fig, A, B, titles, overlay="magenta / green", pts_a=(), pts_b=(), roi=None, pictures=None,
               step=1, shape=None):
    """A | B | overlay (A magenta + B green, grey where equal; or zero crossings of both). Returns the axes.
    pictures: (RGB A, RGB B) to show A and B in their own colours instead of the signed maps.
    step > 1: the arrays hold every step-th pixel of a grid of `shape`; axes stay in full-grid pixels."""
    fig.clear()
    ax = fig.subplots(1, 3, sharex=True, sharey=True)
    # sample o covers full-grid pixels o*step .. o*step + step - 1
    ext = None if step <= 1 else (-0.5, A.shape[1] * step - 0.5, A.shape[0] * step - 0.5, -0.5)
    if pictures is not None:
        picture(ax[0], pictures[0], titles[0], ext)
        picture(ax[1], pictures[1], titles[1], ext)
    else:
        signed_map(ax[0], A, titles[0], ext)
        signed_map(ax[1], B, titles[1], ext)
    both = np.isfinite(A) & np.isfinite(B)
    if overlay.startswith("magenta"):
        nA, nB = _norm01(A, both), _norm01(B, both)
        rgb = np.dstack([nA, nB, nA])
        rgb[~both] = 1
        ax[2].imshow(rgb, interpolation="nearest", extent=ext)
        ax[2].set_title("A magenta, B green (aligned → grey)", fontsize=8)
    else:
        signed_map(ax[2], np.where(both, A, np.nan), "walls: A black, B grey (on A)", ext)
        h = (step - 1) / 2
        ys, xs = np.arange(A.shape[0]) * step + h, np.arange(A.shape[1]) * step + h
        for img, col in ((A, "black"), (B, "0.45")):
            ax[2].contour(xs, ys, np.where(both, img, np.nan), levels=[0], colors=[col], linewidths=0.8)
    ax[2].set_xticks([])
    ax[2].set_yticks([])
    h = (step - 1) // 2
    for a in ax:
        outline(a, roi if (roi is None or step <= 1) else roi[h::step, h::step], "k", 0.8, "--", step)
    if step > 1:
        ax[0].set_xlim(-0.5, shape[1] - 0.5)
        ax[0].set_ylim(shape[0] - 0.5, -0.5)
    for i, p in enumerate(pts_a):
        ax[0].plot(p[1], p[0], "+", color="k", ms=10, mew=1.5)
        ax[0].annotate(str(i + 1), (p[1], p[0]), xytext=(4, 4), textcoords="offset points", fontsize=8)
    for i, p in enumerate(pts_b):
        ax[1].plot(p[1], p[0], "x", color="k", ms=9, mew=1.5)
        ax[1].annotate(str(i + 1), (p[1], p[0]), xytext=(4, 4), textcoords="offset points", fontsize=8)
    return ax


def draw_compare(fig, sA, sB, flip, r_local, roi, m, titles, palette="purple / magenta", crop=True):
    """A and B (three states), flip map, local memory. sA, sB, flip, r_local on the common grid."""
    fig.clear()
    ys, xs = np.nonzero(roi)
    sl = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1)) if crop and ys.size else (slice(None),) * 2
    ax = fig.subplots(1, 4, sharex=True, sharey=True)
    for a, s, t in ((ax[0], sA, f"A: {short(titles[0])}"), (ax[1], sB, f"B: {short(titles[1])}")):
        categorical(a, np.where(roi, s, -9)[sl], {1: COLOR_UP, -1: COLOR_DOWN}, t)
        outline(a, roi[sl], "0.3", 0.4)
    fc = flip_colours(palette)
    categorical(ax[2], np.where(roi, flip, -9)[sl], {i: fc[i] for i in range(4)},
                f"switched {m['switched_pct']:.0f}% of classified\n"
                f"{m['up_to_down_pct']:.0f}% of up → down · {m['down_to_up_pct']:.0f}% of down → up")
    outline(ax[2], roi[sl], "k", 0.6)
    ax[2].legend(handles=[Patch(facecolor=fc[i], edgecolor="0.4", lw=0.4, label=l) for i, l in enumerate(FLIP_LABELS)],
                 fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3, frameon=False)
    im = ax[3].imshow(np.where(roi, r_local, np.nan)[sl], cmap="PiYG", vmin=-1, vmax=1, interpolation="nearest")
    ax[3].set_title(f"local memory (r = {m['r']:+.2f} [{m['r_ci_lo']:+.2f}, {m['r_ci_hi']:+.2f}])", fontsize=8)
    ax[3].set_xticks([])
    ax[3].set_yticks([])
    fig.colorbar(im, ax=ax[3], shrink=0.7)
    return ax


def draw_populations(fig, change, title=""):
    """Bars: Delta f_up, Delta f_0, Delta f_down and Delta<m_z> (direct up<->down part and via neutral), in %."""
    fig.clear()
    ax = fig.subplots(1, 2)
    keys = (("d_up", "Δf↑", COLOR_UP), ("d_neutral", "Δf₀", "0.6"), ("d_down", "Δf↓", COLOR_DOWN))
    vals = [100 * change[k] for k, _, _ in keys]
    ax[0].bar([l for _, l, _ in keys], vals, color=[c for _, _, c in keys])
    ax[0].axhline(0, color="k", lw=0.6)
    ax[0].set_ylabel("% of the region")
    for i, v in enumerate(vals):
        ax[0].annotate(f"{v:+.2f}", (i, v), ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
    d, dd, dn = 100 * change["d_mz"], 100 * change["d_mz_direct"], 100 * change["d_mz_via_neutral"]
    ax[1].bar(["direct", "via f₀", "Δ⟨m_z⟩"], [dd, dn, d], color=["0.35", "0.7", "k"])
    ax[1].axhline(0, color="k", lw=0.6)
    for i, v in enumerate((dd, dn, d)):
        ax[1].annotate(f"{v:+.2f}", (i, v), ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
    for a in ax:
        a.tick_params(labelsize=7)
    fig.suptitle(title, fontsize=8)
    return ax


def draw_shift_check(fig, res):
    """r(dy, dx) at the best angle; + = current alignment, ★ = best; best r against rotation if several angles."""
    fig.clear()
    ang, dy, dx, rb = res["best"]
    ia = res["angles"].index(ang)
    sh = res["shifts"]
    ax = fig.add_subplot(1, 2 if len(res["angles"]) > 1 else 1, 1)
    lim = max(0.1, float(np.nanmax(np.abs(res["r"]))))
    im = ax.imshow(res["r"][ia], cmap="PiYG", vmin=-lim, vmax=lim,
                   extent=(sh[0] - 0.5, sh[-1] + 0.5, sh[-1] + 0.5, sh[0] - 0.5))
    ax.plot(0, 0, "+", color="k", ms=14, mew=1.5, label=f"current: r = {res['r0']:+.3f}")
    ax.plot(dx, dy, "*", color="k", ms=12, label=f"best: r = {rb:+.3f} at dy {dy:+d}, dx {dx:+d} px"
            + (f", {ang:+.1f}°" if ang else ""))
    ax.set_xlabel("dx, B moved right (px)")
    ax.set_ylabel("dy, B moved down (px)")
    ax.set_title(f"r between A and shifted B (rotation {ang:+.1f}°)", fontsize=8)
    ax.legend(fontsize=7, frameon=False, loc="lower left")
    fig.colorbar(im, ax=ax, shrink=0.8, label="r")
    if len(res["angles"]) > 1:
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.plot(res["angles"], [np.nanmax(r) for r in res["r"]], "o-", color="k")
        ax2.set_xlabel("extra rotation of B (°)")
        ax2.set_ylabel("best r over shifts")


def save_pdf(fig, path):
    """Editable PDF: text as text; uncompressed, so no image is stored as indexed colour."""
    with matplotlib.rc_context({"pdf.fonttype": 42, "ps.fonttype": 42, "pdf.compression": 0}):
        fig.savefig(path)
