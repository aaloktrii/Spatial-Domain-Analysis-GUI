"""Plotting layer (TensorSpec rule: matplotlib only, no analysis, no GUI)."""
from .domain_figures import make_domain_figure
from .stats_figures import make_stats_figure, make_domain_wall_figure
from .distribution_figures import make_distribution_figure
from .magnetization_figures import make_magnetization_figure, make_xi_map_figure
from .morphometry_figures import (
    make_morphometry_figure, export_panels_tiff, make_width_area_count_figure, make_size_distribution_figure,
    make_width_distribution_figure, make_period_distribution_figure, make_qq_plot_figure,
    make_lognormal_gaussian_comparison_figure,
)
from .width_figures import make_width_comparison_figure
from . import pubfigures
