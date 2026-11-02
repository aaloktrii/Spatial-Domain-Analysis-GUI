"""
colors.py -- single source of truth for the up (+1) / neutral (0) / down (-1)
color scheme used across every figure this toolkit produces: the GUI's live
preview and exports, the command-line figures (sda-analyze, sda-compare) and the
morphometry width plots.

Why this file exists
---------------------
Every consuming module used to hardcode its own copy of the same hex
strings, so a color-scheme change meant hunting down 9+ duplicated
literals across the GUI, the command-line tools and the
xmcd_toolkit package -- easy to miss one and end up with an inconsistent
mix. Import from here instead of hardcoding a hex string; changing the
scheme (as we just did -- up/down were reversed) then takes effect
everywhere at once.
"""

# ---- primary (saturated) scheme -------------------------------------
# up is BLUE, down is RED. (Reversed from the original red-up/blue-down.)
COLOR_UP = "#2b5dd1"
COLOR_DOWN = "#d12b2b"
COLOR_NEUTRAL = "#dddddd"

# ordered [down(-1), neutral(0), up(+1)] -- this is the index order
# ListedColormap + BoundaryNorm(-1.5,-0.5,0.5,1.5) need for the three-state
# maps (mz in {-1, 0, +1}).
CMAP3_HEX = [COLOR_DOWN, COLOR_NEUTRAL, COLOR_UP]

# ---- lighter fill / darker line shades, for the width histograms -----
# (make_width_distribution_figure, make_width_area_count_figure,
# make_morphometry_figure panel c)
COLOR_UP_FILL, COLOR_UP_LINE = "#a8c8ef", "#2f5fa3"      # light / dark blue
COLOR_DOWN_FILL, COLOR_DOWN_LINE = "#f2a6a6", "#b23a3a"  # light / dark red

# ---- RGBA overlay tuples for domain.py's up/down mask overlay --------
OVERLAY_UP = (0.25, 0.45, 1.0, 0.55)     # blue-ish
OVERLAY_DOWN = (0.95, 0.25, 0.25, 0.55)  # red-ish

# ---- comparison of two states (flip map) -----------------------------
# stayed up / stayed down keep COLOR_UP / COLOR_DOWN; switched pixels get one of two pairs. Every colour pair is
# distinct (CIE dE >= 19) also under simulated deuteranopia, protanopia and tritanopia.
SWITCH_PALETTES = {"purple / magenta": ("#6A3D9A", "#E36FA8"),   # up->down, down->up
                   "orange / green": ("#E69F00", "#004D3A")}
COLOR_UNCERTAIN = "#FFFFFF"
