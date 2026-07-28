"""
xmcd_toolkit
============
Quantitative analysis of magnetic domains in XMCD-PEEM images of
Graphene/Fe3GeTe2/WTe2 (and related) SOT heterostructures.

Public API
----------
    load_image, load_xy_mz_txt          (loaders)
    analyze_xmcd, make_domain_figure, print_report   (domain)
    three_state_field, pdf_three_state,
    number_density_local, net_change_mz,
    domain_count_distribution, make_stats_figure      (stats)

Definitions
-----------
Magnetization (local):   XMCD asymmetry A = (I+ - I-)/(I+ + I-)  ~  M_z
Order parameter:         <m_z> = (A_up - A_down)/(A_up + A_down)   in [-1, 1]
Number density (local):  n_s(area) = N_s(area) / N_total(area)     (a fraction)
Net change:              d<m_z> = <m_z>_after - <m_z>_before
"""

from .colors import (
    COLOR_UP, COLOR_DOWN, COLOR_NEUTRAL, CMAP3_HEX,
    COLOR_UP_FILL, COLOR_UP_LINE, COLOR_DOWN_FILL, COLOR_DOWN_LINE,
    OVERLAY_UP, OVERLAY_DOWN,
)
from .loaders import load_image, load_xy_mz_txt, neighbor_structure
from .domain import analyze_xmcd, make_domain_figure, print_report
from .stats import (
    three_state_field,
    despeckle_mz,
    pdf_three_state,
    number_density_local,
    net_change_mz,
    domain_count_distribution,
    per_state_domain_stats,
    make_stats_figure,
)
from .distributions import (
    fit_lognormal,
    fit_gaussian,
    compare_lognormal_gaussian,
    fit_powerlaw,
    make_distribution_figure,
)
from .magnetization import (
    magnetization_statistics,
    make_magnetization_figure,
    print_magnetization_report,
    area_averaged_mz,
    asymmetry_field,
    net_mz_raw,
)
from .morphometry import (
    domain_width,
    per_state_morphometry,
    make_morphometry_figure,
    export_panels_tiff,
    per_domain_width_area,
    make_width_area_count_figure,
    make_size_distribution_figure,
    make_width_distribution_figure,
    make_period_distribution_figure,
    make_qq_plot_figure,
    qq_series_per_state,
    make_lognormal_gaussian_comparison_figure,
    domain_counts_areas,
    export_domain_stats,
    summarize_domains,
    export_domain_summary,
    export_fit_summary,
)

__all__ = [
    "COLOR_UP", "COLOR_DOWN", "COLOR_NEUTRAL", "CMAP3_HEX",
    "COLOR_UP_FILL", "COLOR_UP_LINE", "COLOR_DOWN_FILL", "COLOR_DOWN_LINE",
    "OVERLAY_UP", "OVERLAY_DOWN",
    "load_image", "load_xy_mz_txt", "neighbor_structure",
    "analyze_xmcd", "make_domain_figure", "print_report",
    "three_state_field", "despeckle_mz", "pdf_three_state", "number_density_local",
    "net_change_mz", "domain_count_distribution", "make_stats_figure",
    "per_state_domain_stats",
    "fit_lognormal", "fit_gaussian", "compare_lognormal_gaussian",
    "fit_powerlaw", "make_distribution_figure",
    "magnetization_statistics", "make_magnetization_figure",
    "print_magnetization_report", "area_averaged_mz", "asymmetry_field",
    "net_mz_raw",
    "domain_width", "per_state_morphometry", "make_morphometry_figure",
    "export_panels_tiff", "per_domain_width_area", "make_width_area_count_figure",
    "make_size_distribution_figure", "make_width_distribution_figure",
    "make_period_distribution_figure",
    "make_qq_plot_figure", "qq_series_per_state",
    "make_lognormal_gaussian_comparison_figure",
    "domain_counts_areas", "export_domain_stats",
    "summarize_domains", "export_domain_summary", "export_fit_summary",
]

__version__ = "0.1.0"
