"""sda-analyze -- flake-area statistical analysis of X-PEEM images (XMCD, XLD).

           one row per image -> plot stats vs current

Command line (thin layer over services/ and core/)."""
from ..notation import column
import os
import csv
import sys
import argparse

import matplotlib
matplotlib.use("Agg")   # headless: the command line only writes files

from ..defaults import DEFAULTS
from ..services.batch import SUMMARY_COLS, analyze_one, collect_targets


def main():
    ap = argparse.ArgumentParser(description="Flake-area statistical analysis of X-PEEM images (XMCD, XLD).")
    ap.add_argument("paths", nargs="*", default=["."],
                    help="image files and/or folders (default: current dir)")
    ap.add_argument("--recursive", action="store_true", help="descend into sub-folders")
    ap.add_argument("--pattern", default="*.png", help="filename glob (default *.png)")
    ap.add_argument("--outdir", default=None,
                    help="output directory (default: next to each source image)")
    ap.add_argument("--um-per-px", type=float, default=None, help="scale-bar calibration (fixed for all images)")
    ap.add_argument("--fov", type=float, nargs=2, metavar=("WIDTH_UM", "HEIGHT_UM"),
                    default=None, help="field of view in um; um/px is computed "
                                       "per image from its own pixel size")
    # Defaults come from defaults.DEFAULTS so the CLI, GUI, and export all
    # share one source of truth.
    ap.add_argument("--tile", type=int, default=DEFAULTS["tile"],
                    help=f"number-density tile size (px, default {DEFAULTS['tile']})")
    ap.add_argument("--z-neutral", type=float, default=DEFAULTS["z_neutral"],
                    help="neutral-band half-width (z)")
    ap.add_argument("--m-threshold", type=float, default=DEFAULTS["m_threshold"],
                    help="value-based gray band on normalized asymmetry: "
                         "|m|<=t -> 0, m>t -> +1, m<-t -> -1 "
                         f"(default {DEFAULTS['m_threshold']})")
    ap.add_argument("--tv-weight", type=float, default=DEFAULTS["tv_weight"], dest="tv_weight",
                    help="total-variation denoise strength applied BEFORE "
                         f"classification (default {DEFAULTS['tv_weight']}; noisy "
                         "experimental frames may need more)")
    ap.add_argument("--flatten-sigma", type=float, default=DEFAULTS["flatten_sigma"],
                    dest="flatten_sigma",
                    help="Gaussian background scale in px (default "
                         f"{DEFAULTS['flatten_sigma']}; raise to ~150 for zoomed "
                         "large-domain frames)")
    ap.add_argument("--min-px", type=int, default=DEFAULTS["min_px"], dest="min_px",
                    help=f"smallest domain kept (default {DEFAULTS['min_px']})")
    ap.add_argument("--denoise-method", default=DEFAULTS["denoise_method"], dest="denoise_method",
                    choices=("tv", "nlm", "median", "none"),
                    help=f"image denoiser (default {DEFAULTS['denoise_method']}; "
                         "nlm = non-local means, edge-preserving alternative to TV)")
    ap.add_argument("--width-method", default=DEFAULTS["width_method"], dest="width_method",
                    choices=("perimeter_area", "skeleton_mean", "skeleton_median",
                             "correlation_length"),
                    help=f"primary w+/w- estimator (default {DEFAULTS['width_method']}; "
                         "all methods are compared in statistics.txt regardless)")
    ap.add_argument("--boundary-method", default=DEFAULTS["boundary_method"],
                    dest="boundary_method",
                    choices=("zero_contour", "threshold", "chan_vese"),
                    help=f"domain-boundary/wall overlay method "
                         f"(default {DEFAULTS['boundary_method']})")
    ap.add_argument("--input-mode", default=DEFAULTS["input_mode"], dest="input_mode",
                    choices=("grayscale", "asymmetry_map", "igor_export"),
                    help="asymmetry_map = an already-coded signed map (default; decoded "
                         "directly, flatten skipped; colours with --map-mode); grayscale = "
                         "raw XMCD intensity; igor_export = Igor graph read back via its colour scale")
    ap.add_argument("--map-mode", default=DEFAULTS["map_mode"], dest="map_mode",
                    choices=("auto", "rb", "bw"),
                    help="asymmetry_map decode: auto = rb for a colour picture, bw for a grey one "
                         "(default); rb = blue(+1)/white(0)/red(-1); bw = white(+1)/black(-1)")
    ap.add_argument("--sym-centre", default=DEFAULTS["sym_centre"], dest="sym_centre",
                    choices=("0", "median", "mean"),
                    help="map zero, as TensorSpec's Sym ±: 0 (the --bg mean if given), or the median / mean "
                         "in --bg, else of the map without its edge drift strips")
    ap.add_argument("--invert", action="store_true", default=DEFAULTS["invert"],
                    help="swap +ve and -ve (a map whose colours mean the opposite sign)")
    ap.add_argument("--dichroism", default=DEFAULTS["dichroism"], choices=("XMCD", "XLD"),
                    help="order parameter in the outputs: XMCD <m>, XLD (XMLD + XNLD) <v_N>; states are +ve / -ve")
    ap.add_argument("--fig-format", default=DEFAULTS["fig_format"], dest="fig_format",
                    choices=("png", "eps", "pdf"),
                    help="publication figure format for figures/ and the pipeline "
                         f"(default {DEFAULTS['fig_format']}; eps/pdf are vector, "
                         "for journal submission)")
    # --- controls for "too much gray inside the active area" ---
    ap.add_argument("--states", type=int, choices=(2, 3), default=DEFAULTS["states"],
                    help="2 = +ve / -ve only (no neutral band); 3 = +ve / neutral / -ve")
    ap.add_argument("--clahe", action="store_true",
                    help="local contrast enhancement to pull weak domains out of gray")
    ap.add_argument("--roi", type=int, nargs=4, metavar=("X0", "Y0", "X1", "Y1"),
                    default=None, help="manual ACTIVE AREA box (skips auto-detect)")
    ap.add_argument("--bg", type=int, nargs=4, metavar=("X0", "Y0", "X1", "Y1"),
                    default=None, help="BACKGROUND region box (sets magnetic zero)")
    ap.add_argument("--clip", type=float, nargs=2, metavar=("LO", "HI"),
                    default=None, help="contrast percentiles (default 0.5 99.5)")
    # --- rigorous magnetization analysis ---
    ap.add_argument("--no-rigorous", action="store_true",
                    help="skip correlation-corrected magnetization stats (faster)")
    ap.add_argument("--n-boot", type=int, default=DEFAULTS["n_boot"],
                    help=f"block-bootstrap iterations for the m_z CI (default {DEFAULTS['n_boot']})")
    # --- extra outputs ---
    ap.add_argument("--detailed", action="store_true",
                    help="also write the diagnostics/ folder: 12-panel pipeline "
                         "figure, method comparisons, distribution/fit figures")
    ap.add_argument("--tiff", action="store_true",
                    help="also export publication TIFF panels (into diagnostics/)")
    ap.add_argument("--dpi", type=int, default=DEFAULTS["dpi"], help="TIFF resolution (default 300)")
    ap.add_argument("--connectivity", type=int, choices=(4, 8), default=4,
                    help="domain connectivity: 4 = edges only (default), "
                         "8 = edges + diagonals/corners")
    args = ap.parse_args()
    args.connectivity = 1 if args.connectivity == 4 else 2   # -> scipy rank

    paths = args.paths if args.paths else ["."]
    targets = collect_targets(paths, args.pattern, args.recursive)
    if not targets:
        print(f"No images matching '{args.pattern}' found."); sys.exit(1)

    # where the combined table / run-info go: --outdir, else the source folder
    if args.outdir:
        table_dir = args.outdir
        os.makedirs(table_dir, exist_ok=True)
    else:
        table_dir = os.path.dirname(os.path.abspath(targets[0]))
    print(f"Analyzing {len(targets)} image(s); results saved "
          + (f"in {args.outdir}/" if args.outdir else "next to each source image"))

    rows = []
    for t in targets:
        try:
            rows.append(analyze_one(t, args, args.outdir))
        except Exception as e:
            print(f"  ERROR on {t}: {e}")

    # combined table across all images
    if rows:
        out_csv = os.path.join(table_dir, "summary_table.csv")
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[column(k) for k in SUMMARY_COLS])     # up / down -> pos / neg
            w.writeheader()
            for r in rows:
                w.writerow({column(k): r.get(k) for k in SUMMARY_COLS})

        # run-info / calibration provenance record
        import datetime
        cal = (f"{args.um_per_px} um/px (fixed)" if args.um_per_px is not None
               else f"FoV {args.fov[0]}x{args.fov[1]} um (per image)" if args.fov
               else "uncalibrated (pixels) -- use sda-calibrate later")
        info = [
            f"run date        : {datetime.datetime.now().isoformat(timespec='seconds')}",
            f"images analyzed : {len(rows)}",
            f"calibration     : {cal}",
            f"m-threshold     : {args.m_threshold if args.m_threshold is not None else f'z-score band {args.z_neutral}'}",
            f"input mode      : {getattr(args, 'input_mode', DEFAULTS['input_mode'])}"
            + (f" ({getattr(args, 'map_mode', DEFAULTS['map_mode'])})"
               if getattr(args, 'input_mode', 'grayscale') == 'asymmetry_map' else ""),
            f"denoise method  : {getattr(args, 'denoise_method', DEFAULTS['denoise_method'])}",
            f"tv-weight       : {getattr(args, 'tv_weight', DEFAULTS['tv_weight'])}",
            f"flatten-sigma   : {getattr(args, 'flatten_sigma', DEFAULTS['flatten_sigma'])}",
            f"min domain (px) : {getattr(args, 'min_px', DEFAULTS['min_px'])}",
            f"width method    : {getattr(args, 'width_method', DEFAULTS['width_method'])}",
            f"boundary method : {getattr(args, 'boundary_method', DEFAULTS['boundary_method'])}",
            f"figure format   : {getattr(args, 'fig_format', DEFAULTS['fig_format'])}",
            f"states          : {args.states}",
            f"tile (px)       : {args.tile}",
            f"bootstrap iters : {0 if args.no_rigorous else args.n_boot}",
            f"detailed figs   : {'yes (diagnostics/)' if getattr(args, 'detailed', False) else 'no'}",
            f"tiff panels     : {'yes' if args.tiff else 'no'}",
            f"summary table   : {out_csv}",
            "",
            "calibrated table: run  sda-calibrate "
            f"{os.path.basename(out_csv)} --fov WIDTH HEIGHT  (or --um-per-px)",
        ]
        with open(os.path.join(table_dir, "run_info.txt"), "w") as f:
            f.write("\n".join(info) + "\n")

        print(f"\nWrote combined table: {out_csv}  ({len(rows)} rows)")
        print(f"Wrote run info / calibration record: {os.path.join(table_dir, 'run_info.txt')}")
        if any(r.get("current_mA") is not None for r in rows):
            print("  current_mA column populated -> ready to plot stats vs current.")


if __name__ == "__main__":
    main()
