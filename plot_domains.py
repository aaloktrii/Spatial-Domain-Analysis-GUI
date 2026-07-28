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
COLOR_F_RATIO = "#000000"  # black
COLOR_M_Z = "#555555"      # dark gray
COLOR_W_PLUS = "#a9c7e9"   # blue -- same as spin up (COLOR_UP)
COLOR_W_MINUS = "#f2a6a6"  # red  -- same as spin down (COLOR_DOWN)


def _simplify_svg(path):
    """Best-effort: matplotlib wraps almost every tick, marker, and line in
    its own <g> layer, which makes the .svg needlessly deep/cluttered to
    open in Illustrator/Inkscape. Flatten that with svgo (via npx), which
    only collapses groups that don't change the rendered output -- lossless,
    just less nested. No-op (with a note, not an error) if npx/svgo isn't
    available; the plot itself is unaffected either way.
    """
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


def _check_lengths():
    lengths = {len(I), len(f_ratio), len(m_z), len(w_plus), len(w_minus)}
    if len(lengths) != 1:
        raise ValueError(
            f"All inputs must be the same length. Got lengths: "
            f"I={len(I)}, f+/f-={len(f_ratio)}, m_z={len(m_z)}, "
            f"w+={len(w_plus)}, w-={len(w_minus)}"
        )


def plot_fratio_mz(out_path="domain_plot_fratio_mz.svg"):
    """I on x; f+/f- on the left axis, m_z on the right axis."""
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    l1, = ax1.plot(I, f_ratio, "s", color=COLOR_F_RATIO, markersize=10,
                   linestyle=":", linewidth=0.5, label=r"$f_+/f_-$")
    l2, = ax2.plot(I, m_z, "^", color=COLOR_M_Z, markersize=10,
                   linestyle=":", linewidth=0.5, label=r"$m_z$")

    ax1.set_xlabel("I (mA)")
    ax1.set_ylabel(r"$f_-/f+-$")
    ax2.set_ylabel(r"$m_z$")
    ax1.grid(False)
    ax2.grid(False)
    ax1.legend(handles=[l1, l2], frameon=False)
    fig.tight_layout()

    fig.savefig(out_path)
    _simplify_svg(out_path)
    print(f"Saved plot to {out_path}")
    return fig


def plot_widths(out_path="domain_plot_widths.svg"):
    """I on x; w~+ on the left axis, w~- on the right axis."""
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    l1, = ax1.plot(I, w_plus, "o", color=COLOR_W_PLUS, markersize=10,
                   linestyle=":", linewidth=0.5, label=r"$\tilde{w}_+$")
    l2, = ax2.plot(I, w_minus, "o", color=COLOR_W_MINUS, markersize=10,
                   linestyle=":", linewidth=0.5, label=r"$\tilde{w}_-$")

    ax1.set_xlabel("I (mA)")
    ax1.set_ylabel(r"$\tilde{w}_+$ (nm)")
    ax2.set_ylabel(r"$\tilde{w}_-$ (nm)")
    ax1.grid(False)
    ax2.grid(False)
    ax1.legend(handles=[l1, l2], frameon=False)
    fig.tight_layout()

    fig.savefig(out_path)
    _simplify_svg(out_path)
    print(f"Saved plot to {out_path}")
    return fig


def main():
    _check_lengths()
    fig1 = plot_fratio_mz()
    fig2 = plot_widths()
    plt.show()


if __name__ == "__main__":
    main()
