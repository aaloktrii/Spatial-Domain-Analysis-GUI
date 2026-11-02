#!/usr/bin/env python3
"""
sda-calibrate -- apply a physical calibration to results AFTER analysis.

Run sda-analyze once WITHOUT calibration (everything comes out in pixels), then
use this to convert to microns whenever you know the field of view. No images
are reprocessed -- calibration is just a multiply, so this is instant and can
be redone with different numbers any time.

Usage
-----
    # same field of view for the whole folder:
    sda-calibrate xmcd_analysis/summary_table.csv --fov 14.4 18.3

    # or a fixed pixel size directly:
    sda-calibrate xmcd_analysis/summary_table.csv --um-per-px 0.1333

Writes summary_table_um.csv next to the input (or use --out).

Notes
-----
* --fov WIDTH HEIGHT (um): um/px is computed PER IMAGE as WIDTH/image_w_px,
  using the image_w_px column written by sda-analyze. Square-pixel consistency
  (WIDTH/w vs HEIGHT/h) is checked and warned.
* Dimensionless quantities (net m, fractions, eccentricity, p-values, N_eff)
  are unchanged. Lengths are scaled by s, areas by s^2, the per-pixel wall
  density by 1/s. New microns columns are ADDED; pixel columns are kept.
* The input must be an UNCALIBRATED table (area_units = px). If it is already
  calibrated, the tool warns and refuses (to avoid double-scaling).
"""
import os
import csv
import sys
import argparse

# columns to convert: (source_px_column, new_um_column, kind)
LENGTH = "length"     # x s
AREA = "area"         # x s^2
INVLEN = "invlength"  # / s
CONVERSIONS = [
    ("flake_area_px", "flake_area_um2", AREA),
    ("mean_domain_area_px", "mean_domain_area_um2", AREA),
    ("up_area_mean", "up_area_mean_um2", AREA),
    ("up_area_sd", "up_area_sd_um2", AREA),
    ("up_area_median", "up_area_median_um2", AREA),
    ("down_area_mean", "down_area_mean_um2", AREA),
    ("down_area_sd", "down_area_sd_um2", AREA),
    ("down_area_median", "down_area_median_um2", AREA),
    ("zero_area_mean", "zero_area_mean_um2", AREA),
    ("zero_area_sd", "zero_area_sd_um2", AREA),
    ("zero_area_median", "zero_area_median_um2", AREA),
    ("up_width_median", "up_width_median_um", LENGTH),
    ("down_width_median", "down_width_median_um", LENGTH),
    ("domain_period", "domain_period_um", LENGTH),
    ("up_skeleton_len", "up_skeleton_len_um", LENGTH),
    ("down_skeleton_len", "down_skeleton_len_um", LENGTH),
    ("lognorm_median", "lognorm_median_um", LENGTH),
    ("domain_wall_density_per_px", "domain_wall_density_per_um", INVLEN),
]


def to_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def scale_for_row(row, args):
    """microns-per-pixel for this image."""
    if args.um_per_px is not None:
        return args.um_per_px, None
    # --fov: per image from its own pixel width
    w = to_float(row.get("image_w_px"))
    h = to_float(row.get("image_h_px"))
    if not w:
        return None, "no image_w_px column (re-run sda-analyze to record it)"
    s_w = args.fov[0] / w
    warn = None
    if h:
        s_h = args.fov[1] / h
        if abs(s_w - s_h) / max(s_w, s_h) > 0.05:
            warn = f"non-square pixels ({s_w:.4f} vs {s_h:.4f} um/px)"
    return s_w, warn


def main():
    ap = argparse.ArgumentParser(description="Apply FoV/pixel calibration to a summary table.")
    ap.add_argument("summary_csv")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--um-per-px", type=float, help="fixed pixel size for all rows")
    g.add_argument("--fov", type=float, nargs=2, metavar=("WIDTH_UM", "HEIGHT_UM"),
                   help="field of view; um/px computed per image from image_w_px")
    ap.add_argument("--out", default=None, help="output CSV (default: *_um.csv)")
    args = ap.parse_args()

    if not os.path.isfile(args.summary_csv):
        print(f"not found: {args.summary_csv}"); sys.exit(1)

    with open(args.summary_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print("empty table"); sys.exit(1)

    if str(rows[0].get("area_units", "px")).strip() == "um^2":
        print("WARNING: this table is already calibrated (area_units = um^2).")
        print("Re-run sda-analyze uncalibrated (pixels) before calibrating. Aborting.")
        sys.exit(1)

    out_cols = list(rows[0].keys())
    for _, new, _ in CONVERSIONS:
        if new not in out_cols:
            out_cols.append(new)

    n_warn = 0
    for row in rows:
        s, warn = scale_for_row(row, args)
        if warn:
            print(f"  WARN [{row.get('image','?')}]: {warn}"); n_warn += 1
        if s is None:
            continue
        row["um_per_px"] = round(s, 6)
        row["calibrated"] = True
        row["area_units"] = "um^2"
        for src, new, kind in CONVERSIONS:
            v = to_float(row.get(src))
            if v is None:
                row[new] = None; continue
            if kind == AREA:
                row[new] = round(v * s * s, 5)
            elif kind == LENGTH:
                row[new] = round(v * s, 5)
            elif kind == INVLEN:
                row[new] = round(v / s, 5) if s else None

    out_path = args.out or os.path.splitext(args.summary_csv)[0] + "_um.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_cols)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k) for k in out_cols})

    s0, _ = scale_for_row(rows[0], args)
    print(f"calibrated {len(rows)} row(s): {os.path.basename(out_path)}")
    if args.um_per_px is not None:
        print(f"  um/px = {args.um_per_px} (fixed)")
    else:
        print(f"  um/px from FoV {args.fov[0]}x{args.fov[1]} um (per image); "
              f"first image -> {s0:.4f} um/px")
    if n_warn:
        print(f"  {n_warn} non-square-pixel warning(s) -- check FoV order (WIDTH HEIGHT) / aspect")


if __name__ == "__main__":
    main()
