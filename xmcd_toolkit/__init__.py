"""
xmcd_toolkit
============
Quantitative analysis of magnetic domains in X-PEEM images (XMCD and XLD) of
Graphene/Fe3GeTe2/WTe2 (and related) SOT heterostructures.

Layers (TensorSpec rules): core/ = analysis (numpy / scipy / scikit-image, no plotting, no GUI);
plotting/ = matplotlib figures; gui/ = PySide6 widgets only. Every name below is re-exported here, so
`from xmcd_toolkit import ...` works as before.

Public API
----------
    load_image, load_xy_mz_txt          (loaders)
    analyze_xmcd, make_domain_figure, print_report   (domain)
    three_state_field, pdf_three_state,
    number_density_local, net_change_mz,
    domain_count_distribution      (stats)

Definitions
-----------
Signal (local):          dichroism asymmetry A = (I1 - I2)/(I1 + I2): XMCD (circular) or XLD (linear)
Order parameter:         <m> = (A_+ve - A_-ve)/(A_+ve + A_-ve) in [-1, 1]   (shown as <v_N> for XLD)
Number density (local):  n_s(area) = N_s(area) / N_total(area)     (a fraction)
Net change:              d<m> = <m>_after - <m>_before
"""

__version__ = "2.1.0"
__author__ = "Aalok Tiwari"
__license__ = "MIT"

from .colors import (
    COLOR_UP, COLOR_DOWN, COLOR_NEUTRAL, CMAP3_HEX,
    COLOR_UP_FILL, COLOR_UP_LINE, COLOR_DOWN_FILL, COLOR_DOWN_LINE,
    OVERLAY_UP, OVERLAY_DOWN,
)
from .core.loaders import load_image, load_asymmetry_map, load_xy_mz_txt, neighbor_structure
from .core.domain import analyze_xmcd, print_report
from .plotting.domain_figures import make_domain_figure
from .core.stats import (
    three_state_field,
    despeckle_mz,
    pdf_three_state,
    number_density_local,
    net_change_mz,
    domain_count_distribution,
    per_state_domain_stats,
    domain_wall_distribution,
)
from .core.distributions import (
    fit_lognormal,
    fit_gaussian,
    compare_lognormal_gaussian,
    fit_powerlaw,
)
from .core.magnetization import (
    magnetization_statistics,
    print_magnetization_report,
    area_averaged_mz,
    asymmetry_field,
    net_mz_raw,
    mz_four_ways,
    continuous_fractions,
    bootstrap_fraction_cis,
    fraction_threshold_sweep,
    correlation_length,
    correlation_length_state,
    local_xi_map,
)
from .core.morphometry import (
    domain_width,
    per_state_morphometry,
    per_domain_width_area,
    qq_series_per_state,
    domain_counts_areas,
    export_domain_stats,
    summarize_domains,
    export_domain_summary,
    export_fit_summary,
)
# swappable methods (registries) -- width/boundary/denoise/background
from .core.methods.width import (
    domain_widths,
    WIDTH_METHODS, WIDTH_METHOD_LABELS,
)
from .core.methods.boundary import (
    domain_boundary, BOUNDARY_METHODS, BOUNDARY_METHOD_LABELS,
)
from .core.methods import denoise as denoise_methods
from .core.methods import background as background_methods
from .core.methods import boundary as boundary_methods
from .plotting import pubfigures
# figures (plotting layer)
from .plotting.stats_figures import make_stats_figure, make_domain_wall_figure
from .plotting.distribution_figures import make_distribution_figure
from .plotting.magnetization_figures import make_magnetization_figure, make_xi_map_figure
from .plotting.morphometry_figures import make_morphometry_figure, export_panels_tiff, make_width_area_count_figure, make_size_distribution_figure, make_width_distribution_figure, make_period_distribution_figure, make_qq_plot_figure, make_lognormal_gaussian_comparison_figure
from .plotting.width_figures import make_width_comparison_figure


__all__ = [
    "COLOR_UP", "COLOR_DOWN", "COLOR_NEUTRAL", "CMAP3_HEX",
    "COLOR_UP_FILL", "COLOR_UP_LINE", "COLOR_DOWN_FILL", "COLOR_DOWN_LINE",
    "OVERLAY_UP", "OVERLAY_DOWN",
    "load_image", "load_asymmetry_map", "load_xy_mz_txt", "neighbor_structure",
    "analyze_xmcd", "make_domain_figure", "print_report",
    "three_state_field", "despeckle_mz", "pdf_three_state", "number_density_local",
    "net_change_mz", "domain_count_distribution", "make_stats_figure",
    "per_state_domain_stats", "domain_wall_distribution", "make_domain_wall_figure",
    "fit_lognormal", "fit_gaussian", "compare_lognormal_gaussian",
    "fit_powerlaw", "make_distribution_figure",
    "magnetization_statistics", "make_magnetization_figure",
    "print_magnetization_report", "area_averaged_mz", "asymmetry_field",
    "net_mz_raw", "mz_four_ways", "continuous_fractions", "bootstrap_fraction_cis",
    "fraction_threshold_sweep", "correlation_length", "correlation_length_state",
    "local_xi_map", "make_xi_map_figure",
    "domain_width", "per_state_morphometry", "make_morphometry_figure",
    "export_panels_tiff", "per_domain_width_area", "make_width_area_count_figure",
    "make_size_distribution_figure", "make_width_distribution_figure",
    "make_period_distribution_figure",
    "make_qq_plot_figure", "qq_series_per_state",
    "make_lognormal_gaussian_comparison_figure",
    "domain_counts_areas", "export_domain_stats",
    "summarize_domains", "export_domain_summary", "export_fit_summary",
    "domain_widths", "make_width_comparison_figure", "WIDTH_METHODS", "WIDTH_METHOD_LABELS",
    "domain_boundary", "BOUNDARY_METHODS", "BOUNDARY_METHOD_LABELS",
    "denoise_methods", "background_methods", "boundary_methods",
]

