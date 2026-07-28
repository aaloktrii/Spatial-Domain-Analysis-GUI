#!/usr/bin/env python3
"""
export_figures.py -- export the three domain panels as SEPARATE .tiff images,
each with its own colorbar, in physical units (um) when calibrated.

Options: --m-threshold (default 0.1), --outdir, --dpi (default 300).
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

from xmcd_toolkit import analyze_xmcd, three_state_field, export_panels_tiff
from PIL import Image


def resolve_um(path, args):
    if args.um_per_px is not None:
        return args.um_per_px
    if args.fov:
        with Image.open(path) as im:
            W, H = im.size
        sw, sh = args.fov[0] / W, args.fov[1] / H
        if abs(sw - sh) / max(sw, sh) > 0.05:
            print(f"  WARN: non-square pixels ({sw:.4f} vs {sh:.4f} um/px) "
                  f"-- check --fov order (WIDTH HEIGHT)")
        return sw
    return None


def main():
    ap = argparse.ArgumentParser(description="Export three domain panels as separate calibrated TIFFs.")
    ap.add_argument("image")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--um-per-px", type=float, default=None)
    g.add_argument("--fov", type=float, nargs=2, metavar=("WIDTH_UM", "HEIGHT_UM"))
    ap.add_argument("--m-threshold", type=float, default=0.1)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    if not os.path.isfile(args.image):
        print(f"not found: {args.image}"); sys.exit(1)

    stem = os.path.splitext(os.path.basename(args.image))[0]
    outdir = args.outdir or f"{stem}_figures"
    os.makedirs(outdir, exist_ok=True)

    um = resolve_um(args.image, args)
    unit = "um" if um else "px"
    sc = um or 1.0

    _, layers = analyze_xmcd(args.image)
    mz, roi = three_state_field(layers, m_threshold=args.m_threshold)
    paths = export_panels_tiff(layers["img"], mz, roi, um, outdir, stem, dpi=args.dpi)

    unit = "um" if um else "px"
    print(f"wrote {len(paths)} TIFFs to {outdir}/ ({unit}):")
    for f in paths:
        print(f"  {os.path.basename(f)}")
    if um:
        print(f"  calibration: {um:.4f} um/px")
    else:
        print("  uncalibrated (pixels) -- pass --fov or --um-per-px for microns")


if __name__ == "__main__":
    main()
