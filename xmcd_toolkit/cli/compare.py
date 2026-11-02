"""sda-compare -- test whether a current pulse changed the magnetization.

Command line (thin layer over services/ and core/)."""
import os
import sys
import re
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..core import analyze_xmcd, three_state_field, magnetization_statistics
from ..colors import CMAP3_HEX
from ..defaults import DEFAULTS
from ..core.statistics_compare import difference_bootstrap


def label_of(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    cur = re.search(r"([+-]?\d+)\s*mA", stem, re.IGNORECASE)
    return stem, (f"{int(cur.group(1))} mA" if cur else stem)


def analyze(path, m_threshold, clahe, n_boot):
    from ..core.igor_import import source_for
    src, mode = source_for(path, DEFAULTS["input_mode"])   # a large picture drawn as pixel blocks: its own grid
    metrics, layers = analyze_xmcd(src, clahe=clahe, input_mode=mode, map_mode=DEFAULTS["map_mode"])
    mz, roi = three_state_field(layers, m_threshold=m_threshold)
    res = magnetization_statistics(layers, m_threshold=m_threshold, n_boot=n_boot)
    return metrics, layers, mz, roi, res


def main():
    ap = argparse.ArgumentParser(description="Before/after magnetization-change test.")
    ap.add_argument("before"); ap.add_argument("after")
    ap.add_argument("--m-threshold", type=float, default=0.1)
    ap.add_argument("--um-per-px", type=float, default=None)
    ap.add_argument("--clahe", action="store_true")
    ap.add_argument("--n-boot", type=int, default=1500)
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    for f in (args.before, args.after):
        if not os.path.isfile(f):
            print(f"not found: {f}"); sys.exit(1)

    sb, lb = label_of(args.before)
    sa, la = label_of(args.after)
    outdir = args.outdir or f"compare_{sb}_vs_{sa}"
    os.makedirs(outdir, exist_ok=True)

    mb, Lb, mzb, roib, rb = analyze(args.before, args.m_threshold, args.clahe, args.n_boot)
    ma, La, mza, roia, ra = analyze(args.after, args.m_threshold, args.clahe, args.n_boot)

    m0, m1 = rb["mean_mz"], ra["mean_mz"]
    se0, se1 = rb["sem_corrected"], ra["sem_corrected"]
    delta = m1 - m0
    se_delta = np.hypot(se0, se1)                       # independent measurements
    z = delta / (se_delta + 1e-12)
    from scipy import stats
    pval = 2 * (1 - stats.norm.cdf(abs(z)))
    ci_lo, ci_hi = delta - 1.96 * se_delta, delta + 1.96 * se_delta
    blo, bhi, diff = difference_bootstrap(rb, ra)

    significant = not (blo <= 0 <= bhi)
    verdict = ("SIGNIFICANT change in magnetization"
               if significant else
               "NO significant change (CI includes 0)")

    lines = [
        f"Before : {os.path.basename(args.before)}  ({lb})",
        f"After  : {os.path.basename(args.after)}  ({la})",
        "",
        f"<m_z>_before = {m0:+.4f} +/- {se0:.4f}   (N_eff={rb['N_eff']:.0f}, xi={rb['corr_length_px']:.1f}px)",
        f"<m_z>_after  = {m1:+.4f} +/- {se1:.4f}   (N_eff={ra['N_eff']:.0f}, xi={ra['corr_length_px']:.1f}px)",
        "",
        f"d<m_z>       = {delta:+.4f}",
        f"combined SE  = {se_delta:.4f}",
        f"95% CI (analytic)  = [{ci_lo:+.4f}, {ci_hi:+.4f}]",
        f"95% CI (bootstrap) = [{blo:+.4f}, {bhi:+.4f}]",
        f"z = {z:+.2f}   p = {pval:.3g}",
        "",
        f"VERDICT: {verdict}",
    ]
    report = "\n".join(lines)
    print(report)
    with open(os.path.join(outdir, "comparison.txt"), "w") as f:
        f.write(report + "\n")

    # figure
    cmap3 = matplotlib.colors.ListedColormap(CMAP3_HEX)
    fig = plt.figure(figsize=(14, 4.6))
    for k, (img, mz, roi, lab, mm) in enumerate([
            (Lb["img"], mzb, roib, lb, m0), (La["img"], mza, roia, la, m1)]):
        ax = fig.add_subplot(1, 3, k + 1)
        ax.imshow(img, cmap="gray")
        ax.imshow(np.where(roi, mz, np.nan), cmap=cmap3, vmin=-1, vmax=1, alpha=0.85)
        ax.set_title(f"{lab}\n<m_z>={mm:+.3f}"); ax.axis("off")

    ax = fig.add_subplot(1, 3, 3)
    ax.hist(diff, bins=40, color="#6a9", edgecolor="white")
    ax.axvline(0, color="k", lw=2, label="no change")
    ax.axvline(blo, color="r", lw=1.4, ls="--")
    ax.axvline(bhi, color="r", lw=1.4, ls="--", label="95% CI")
    ax.axvline(delta, color="b", lw=2, label=f"d<m_z>={delta:+.3f}")
    ax.set_title(f"change: {'SIGNIFICANT' if significant else 'not significant'}")
    ax.set_xlabel("d<m_z> (bootstrap)"); ax.set_ylabel("count"); ax.legend(fontsize=8)

    fig.suptitle(f"{lb}  ->  {la}", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig_path = os.path.join(outdir, "comparison.png")
    plt.savefig(fig_path, dpi=130); plt.close(fig)
    print(f"\nwrote {outdir}/comparison.png and comparison.txt")


if __name__ == "__main__":
    main()
