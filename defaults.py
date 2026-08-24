"""
defaults.py -- single source of truth for pipeline parameter defaults.

Imported by gui.py (initial control values), gui_core.py (params.get
fallbacks + build_args), and analyze.py (CLI argparse defaults), so the GUI,
the CLI, and the export all start from identical values and stay in sync.
Change a default here once and it changes in all three at the same time.

flatten_sigma is the one genuinely image-dependent knob (raise it well above
the default for zoomed / large-domain frames, e.g. ~150 for a 5 um field of
view); everything else is a sane cross-image default.
"""

DEFAULTS = {
    "m_threshold":  0.05,   # value-mode gray band: |m| <= t -> neutral
    "tv_weight":    0.20,   # TV-Chambolle denoise strength
    "flatten_sigma": 25.0,  # Gaussian background scale (px); raise for zoomed frames
    "min_px":       12,     # smallest domain kept (despeckle / counts / widths)
    "width_method": "perimeter_area",  # primary w+/w- estimator (see WIDTH_METHODS)
    "denoise_method": "tv",  # image denoiser: tv (default) / nlm / median / none
    "boundary_method": "zero_contour",  # wall overlay: zero_contour / threshold / chan_vese
    "connectivity": 1,      # 1 = 4-connected (edges), 2 = 8-connected (+diagonals)
    "tile":         16,     # number-density / D-map / local tile size (px)
    "z_neutral":    0.5,    # z-score neutral band half-width (legacy mode)
    "states":       3,      # 3 = up/neutral/down, 2 = no gray band
    "clahe":        False,  # local contrast enhancement
    "flatten":      True,   # illumination flattening on/off
    "rigorous":     True,   # correlation-corrected magnetization stats + CIs
    "tiff":         True,   # also write publication TIFFs
    "n_boot":       1000,   # block-bootstrap iterations
    "dpi":          300,    # raster export dpi
    "detailed":     False,  # also write the diagnostics/ folder (12-panel, fits, comparisons)
    "input_mode":   "grayscale",  # "grayscale" (raw XMCD) or "asymmetry_map" (pre-coded)
    "map_mode":     "rb",   # decode for input_mode="asymmetry_map": rb (blue/red) or bw (white/black)
    "fig_format":   "png",  # publication figure format for the export: png / eps / pdf
}


def d(key):
    """Convenience accessor: defaults.d('tv_weight')."""
    return DEFAULTS[key]
