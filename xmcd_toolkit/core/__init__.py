"""Analysis layer (TensorSpec rule: no plotting, no GUI): numpy / scipy / scikit-image only."""
from .loaders import load_image, load_asymmetry_map, load_xy_mz_txt, neighbor_structure
from .domain import analyze_xmcd, print_report
from .stats import (
    three_state_field, despeckle_mz, pdf_three_state, number_density_local, net_change_mz,
    domain_count_distribution, per_state_domain_stats, domain_wall_distribution,
)
from .distributions import fit_lognormal, fit_gaussian, compare_lognormal_gaussian, fit_powerlaw
from .magnetization import (
    magnetization_statistics, print_magnetization_report, area_averaged_mz, asymmetry_field, net_mz_raw, mz_four_ways,
    continuous_fractions, bootstrap_fraction_cis, fraction_threshold_sweep, correlation_length,
    correlation_length_state, local_xi_map,
)
from .morphometry import (
    domain_width, per_state_morphometry, per_domain_width_area, qq_series_per_state, domain_counts_areas,
    export_domain_stats, summarize_domains, export_domain_summary, export_fit_summary,
)
from .methods.width import domain_widths, WIDTH_METHODS, WIDTH_METHOD_LABELS
from .methods.boundary import domain_boundary, BOUNDARY_METHODS, BOUNDARY_METHOD_LABELS

__all__ = [
    "load_image",
    "load_asymmetry_map",
    "load_xy_mz_txt",
    "neighbor_structure",
    "analyze_xmcd",
    "print_report",
    "three_state_field",
    "despeckle_mz",
    "pdf_three_state",
    "number_density_local",
    "net_change_mz",
    "domain_count_distribution",
    "per_state_domain_stats",
    "domain_wall_distribution",
    "fit_lognormal",
    "fit_gaussian",
    "compare_lognormal_gaussian",
    "fit_powerlaw",
    "magnetization_statistics",
    "print_magnetization_report",
    "area_averaged_mz",
    "asymmetry_field",
    "net_mz_raw",
    "mz_four_ways",
    "continuous_fractions",
    "bootstrap_fraction_cis",
    "fraction_threshold_sweep",
    "correlation_length",
    "correlation_length_state",
    "local_xi_map",
    "domain_width",
    "per_state_morphometry",
    "per_domain_width_area",
    "qq_series_per_state",
    "domain_counts_areas",
    "export_domain_stats",
    "summarize_domains",
    "export_domain_summary",
    "export_fit_summary",
    "domain_widths",
    "WIDTH_METHODS",
    "WIDTH_METHOD_LABELS",
    "domain_boundary",
    "BOUNDARY_METHODS",
    "BOUNDARY_METHOD_LABELS",
]
