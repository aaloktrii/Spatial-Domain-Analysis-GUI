"""
methods/ -- pluggable, swappable analysis methods, grouped by role.

Each module holds a small REGISTRY of interchangeable methods plus a
`compute(...)` dispatcher, so callers (and the GUI) pick a method by name and
never hard-code the list. Adding a method = write one function + register it;
the GUI dropdowns read the registry, so they update automatically.

  denoise.py     image denoisers        (tv, nlm, median, none)
  background.py  background correction  (gaussian, bg_region, plane, none)
  width.py       per-state domain width (perimeter_area, skeleton_*, correlation_length)
  boundary.py    domain-boundary/wall   (threshold, zero_contour, chan_vese)

Each registry entry is a Method(name, label, help, fn). The physics core
(magnetization, fractions, distributions) is NOT here -- those are not
swappable "methods", they are the analysis itself.
"""
from collections import namedtuple

Method = namedtuple("Method", ["name", "label", "help", "fn"])


def make_registry():
    """Return (REGISTRY dict, register decorator). The decorator registers a
    function under `name` with a display `label` and `help` string."""
    registry = {}

    def register(name, label, help=""):
        def deco(fn):
            registry[name] = Method(name, label, help, fn)
            return fn
        return deco

    return registry, register


def options(registry):
    """[(name, label), ...] for building a GUI dropdown from a registry."""
    return [(m.name, m.label) for m in registry.values()]


# Submodules are imported explicitly by their consumers (domain.py imports
# denoise/background; the width/boundary registries by their callers) rather
# than eagerly here, so import order never risks a cycle.
__all__ = ["Method", "make_registry", "options"]
