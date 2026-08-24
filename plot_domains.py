"""
Two plots vs I, each with left/right y-axes, saved as separate .svg files:

  Plot 1 (domain_plot_fratio_mz.svg):
      left axis  = f+/f- (ratio, unitless)
      right axis = m_z (net magnetization)
  Plot 2 (domain_plot_widths.svg):
      left axis  = w~+ (median domain width, nm)
      right axis = w~- (median domain width, nm)

EDIT THE NUMBERS BELOW, then just run this file -- no prompts.
    I        : current, mA
    f_ratio  : f_plus / f_minus, entered directly (unitless)
    m_z      : net magnetization, e.g. (N_up - N_down)/(N_up + N_down)
    w_plus, w_minus : median domain width, nm

All five lists must be the same length as I.
"""

import shutil
import subprocess

import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# EDIT THESE -- one entry per current value, in the same order.
I = [-40, -30, -20, 0, 20, 30, 40]                   # mA
f_ratio = [0.57,0.58,0.66,1.07,1.47,1.50, 1.85]  # f_plus / f_minus, unitless
m_z = [0.33,0.26,0.23,-0.04,-0.21,-0.23,-0.34]            # EDIT: net magnetization
w_plus = [96,95,111,149,128,91,90]             # nm
w_minus = [75,79,87,152,175,106,110]            # nm
# ----------------------------------------------------------------------
# Okabe-Ito colourblind-safe palette -- distinct in print/greyscale, the
# usual choice for APL/PRL figures.
COLOR_F_RATIO = "#000000"  # black          -- f+/f-
COLOR_M_Z = "#E69F00"      # orange         -- m_z
COLOR_W_PLUS = "#0072B2"   # blue           -- w~+ (spin up)
COLOR_W_MINUS = "#D55E00"  # vermillion     -- w~- (spin down)


def _simplify_svg(path):
    """Best-effort: matplotlib wraps almost every tick, marker, and line in
    its own <g> layer, which makes the .svg needlessly deep/cluttered to
    open in Illustrator/Inkscape. Flatten that with svgo (via npx), which
    only collapses groups that don't change the rendered output -- lossless,
    just less nested. No-op (with a note, not an error) if npx/svgo isn't
    available; the plot itself is unaffected either way.
    """
    if not path.lower().endswith(".svg"):
        return                       # svgo is SVG-only; nothing to do for .eps/.pdf
    npx = shutil.which("npx")
    if not npx:
        print(f"  (svgo not found -- skipped layer simplification for {path})")
        return
    try:
        subprocess.run(
            [npx, "--yes", "svgo@3", "-i", path, "-o", path, "--multipass", "-q"],
            check=True, capture_output=True, timeout=60,
        )
        print(f"  simplified SVG layers: {path}")
    except Exception as e:
        print(f"  (svgo simplification failed for {path}: {e})")


def _add_mz_top_axis(ax1):
    """Add a secondary x-axis on TOP showing <m_z> aligned to each current.
    <m_z> is paired one-to-one with I (not a linear rescale of it), so a tick
    is placed at every current position and labelled with that point's <m_z>.
    Lets the reader read the same data against both the drive current (bottom)
    and the resulting net magnetization (top)."""
    axtop = ax1.twiny()
    axtop.set_xlim(ax1.get_xlim())          # share the current axis range
    axtop.set_xticks(I)
    axtop.set_xticklabels([f"{v:+.2f}" for v in m_z], fontsize=9)
    axtop.set_xlabel(r"$\langle m_z \rangle$")
    return axtop


def _check_lengths():
    lengths = {len(I), len(f_ratio), len(m_z), len(w_plus), len(w_minus)}
    if len(lengths) != 1:
        raise ValueError(
            f"All inputs must be the same length. Got lengths: "
            f"I={len(I)}, f+/f-={len(f_ratio)}, m_z={len(m_z)}, "
            f"w+={len(w_plus)}, w-={len(w_minus)}"
        )


def plot_fratio_mz(out_path="domain_plot_fratio_mz.eps"):
    """I on x; f+/f- on the left axis, m_z on the right axis."""
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    l1, = ax1.plot(I, f_ratio, "o", color=COLOR_F_RATIO, markersize=10,
                   linestyle=":", linewidth=0.5, label=r"$f_+/f_-$")
    l2, = ax2.plot(I, m_z, "o", color=COLOR_M_Z, markersize=10,
                   linestyle=":", linewidth=0.5, label=r"$m_z$")

    ax1.set_xlabel("I (mA)")
    ax1.set_ylabel(r"$f_+/f_-$")
    ax2.set_ylabel(r"$m_z$")
    ax1.grid(False)
    ax2.grid(False)
    ax1.legend(handles=[l1, l2], frameon=False)
    _add_mz_top_axis(ax1)
    fig.tight_layout()

    fig.savefig(out_path)
    _simplify_svg(out_path)
    print(f"Saved plot to {out_path}")
    return fig


def plot_widths(out_path="domain_plot_widths.eps"):
    """I on x; BOTH w~+ and w~- on the same left axis (one nm scale, so the
    two widths are directly comparable)."""
    fig, ax1 = plt.subplots(figsize=(8, 5))

    l1, = ax1.plot(I, w_plus, "s", color=COLOR_W_PLUS, markersize=10,
                   linestyle=":", linewidth=0.5, label=r"$\tilde{w}_+$")
    l2, = ax1.plot(I, w_minus, "s", color=COLOR_W_MINUS, markersize=10,
                   linestyle=":", linewidth=0.5, label=r"$\tilde{w}_-$")

    ax1.set_xlabel("I (mA)")
    ax1.set_ylabel(r"$\tilde{w}_\pm$ (nm)")
    ax1.grid(False)
    ax1.legend(handles=[l1, l2], frameon=False)
    _add_mz_top_axis(ax1)
    fig.tight_layout()

    fig.savefig(out_path)
    _simplify_svg(out_path)
    print(f"Saved plot to {out_path}")
    return fig


def plot_ratio_mz2(out_path="domain_plot_ratio_mz2.eps"):
    """Standalone second dataset: f+/f- on the LEFT axis, <m_z> on the RIGHT.
    <m_z> is entered in units of 1e-2 (as given), so the right axis is
    labelled x10^-2 and the raw numbers are plotted directly."""
    I2 = [-40, -20, 0, 20, 40]                       # mA
    fr2 = [1.5, 0.8, 0.75, 0.53, 0.56]               # f+/f-, unitless
    mz2 = [1.9, -4.9, -9.2, -12.9, -7.0]             # <m_z>, units of 1e-2

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    l1, = ax1.plot(I2, fr2, "o", color=COLOR_F_RATIO, markersize=10,
                   linestyle=":", linewidth=0.5, label=r"$f_+/f_-$")
    l2, = ax2.plot(I2, mz2, "o", color=COLOR_M_Z, markersize=10,
                   linestyle=":", linewidth=0.5, label=r"$\langle m_z \rangle$")

    ax1.set_xlabel("I (mA)")
    ax1.set_ylabel(r"$f_+/f_-$")
    ax2.set_ylabel(r"$\langle m_z \rangle$ ($\times 10^{-2}$)")
    ax2.invert_yaxis()           # <m_z> axis runs positive (bottom) -> negative (top)
    ax1.grid(False)
    ax2.grid(False)
    ax1.legend(handles=[l1, l2], frameon=False)
    fig.tight_layout()

    fig.savefig(out_path)
    _simplify_svg(out_path)
    print(f"Saved plot to {out_path}")
    return fig


def plot_new_ab(out_path="domain_plot_new_ab.eps"):
    """Two-panel (a, b) figure in the reference-manuscript style, for the
    second (5-point) dataset:
      (a) f+/f- on the LEFT (black triangles), <m_z> on the RIGHT (red squares)
      (b) w~+ on the LEFT (blue), w~- on the RIGHT (red), both circles
    Dashed connectors throughout. <m_z> is in units of 1e-2 (as entered)."""
    I2 = [-40, -20, 0, 20, 40]                       # mA
    fr = [1.5, 0.8, 0.75, 0.53, 0.56]                # f+/f-, unitless
    mz = [1.9, -4.9, -9.2, -12.9, -7.0]              # <m_z>, units of 1e-2
    wp = [145, 134, 158, 193, 137]                   # <w+>, nm
    wm = [114, 141, 193, 304, 167]                   # <w->, nm
    # colours chosen to match the reference manuscript figure:
    c_m = "#c0392b"       # brick red     -- <m_z> (panel a, right)
    c_wp = "#a9c7e9"      # pale blue     -- w~+ (panel b, left)
    c_wm = "#f2a6a6"      # pale red      -- w~- (panel b, right)

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(13, 5))

    # --- panel (a): f+/f- (left) + <m_z> (right) ---
    axa2 = axa.twinx()
    la, = axa.plot(I2, fr, "^--", color=COLOR_F_RATIO, markersize=9,
                   linewidth=1.0, label=r"$f_+/f_-$")
    lm, = axa2.plot(I2, mz, "s--", color=c_m, markersize=9,
                    linewidth=1.0, label=r"$\langle m_z \rangle$")
    axa.set_xlabel("I (mA)")
    axa.set_ylabel(r"$f_+/f_-$")
    axa2.set_ylabel(r"$\langle m_z \rangle$ ($\times 10^{-2}$)")
    axa2.invert_yaxis()          # <m_z> axis runs positive (bottom) -> negative (top)
    axa.grid(False); axa2.grid(False)
    axa.legend(handles=[la, lm], frameon=False, loc="upper center")
    axa.text(-0.16, 1.02, "a", transform=axa.transAxes,
             fontsize=16, fontweight="bold")

    # --- panel (b): w~+ (left) + w~- (right) ---
    axb2 = axb.twinx()
    lp, = axb.plot(I2, wp, "o--", color=c_wp, markersize=9,
                   linewidth=1.0, label=r"$\tilde{w}_+$")
    ln, = axb2.plot(I2, wm, "o--", color=c_wm, markersize=9,
                    linewidth=1.0, label=r"$\tilde{w}_-$")
    axb.set_xlabel("I (mA)")
    axb.set_ylabel(r"$\tilde{w}_+$ (nm)")
    axb2.set_ylabel(r"$\tilde{w}_-$ (nm)")
    axb.grid(False); axb2.grid(False)
    axb.legend(handles=[lp, ln], frameon=False, loc="upper left")
    axb.text(-0.16, 1.02, "b", transform=axb.transAxes,
             fontsize=16, fontweight="bold")

    fig.tight_layout()
    fig.savefig(out_path)
    _simplify_svg(out_path)
    print(f"Saved plot to {out_path}")
    return fig


def main():
    _check_lengths()
    fig1 = plot_fratio_mz()
    fig2 = plot_widths()
    fig3 = plot_ratio_mz2()
    fig4 = plot_new_ab()
    plt.show()


if __name__ == "__main__":
    main()
