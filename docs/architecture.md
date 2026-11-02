# Architecture

The toolkit is one Python package, `xmcd_toolkit`, in layers. Analysis, plotting, workflows and the GUI live in
separate sub-packages, and imports only flow downward (gui → services → plotting / core; plotting → core), so
each layer can be used on its own: `core/` in a script or notebook without matplotlib or Qt, `services/` in a batch
job without a GUI. `tests/test_structure_and_math.py` enforces the layer rules.

```
.
├── README.md  LICENSE  CITATION.cff  CHANGELOG.md  pyproject.toml
├── docs/                methods.md (the analysis, step by step), gui.md (the program), architecture.md (this file)
├── examples/            domain_example.py, stats_example.py, make_synthetic.py, data/ (two synthetic images)
├── tests/               layer rules + maths (python -m pytest)
└── xmcd_toolkit/
    ├── __init__.py      public API: every name importable as before (from xmcd_toolkit import ...)
    ├── colors.py        +ve / −ve / neutral colours, flip-map palettes (single source of truth)
    ├── defaults.py      pipeline defaults shared by GUI, CLI and export (single source of truth)
    │
    ├── core/            ANALYSIS: numpy / scipy / scikit-image only; no matplotlib, no Qt
    │   ├── loaders.py           images (chrome strip), coded maps, x y mz text
    │   ├── igor_import.py       Igor graph exports (.tif / .png) -> the map they show (frame cropped,
    │   │                        original pixel grid recovered, cached)
    │   ├── domain.py            analyze_xmcd: segmentation, flattening, classification, scalar metrics
    │   ├── stats.py             three-state field, PDF, number density, net change, domain counts
    │   ├── distributions.py     log-normal / Gaussian / power-law fits
    │   ├── magnetization.py     <m> statistics, correlation length, bootstrap CIs
    │   ├── morphometry.py       widths, periods, domain counts / areas, summaries (CSV / text export)
    │   ├── methods/             swappable denoise / background / boundary / width methods (registries)
    │   ├── regions.py           polygon -> mask, um / px calibration
    │   ├── offset.py            zero of a map (Sym ± centre: 0 / median / mean; edge drift strips)
    │   ├── walls.py             sub-pixel wall density
    │   ├── statistics_compare.py  bootstrap difference of two images' magnetization statistics
    │   ├── transform.py         per-image rotation / scale / flips / shift + view rotation; point-pair fit
    │   ├── registration.py      automatic alignment; correlation vs shift
    │   ├── comparison.py        flip map, switched fractions, memory r (+ bootstrap), local r, wall shift
    │   ├── populations.py       Δf₊ / Δf₀ / Δf₋, Δ<m>, minimal redistribution, bounds, matrix
    │   └── session.py           session file (xarray DataTree: /images, /analysis/compare, /history)
    │
    ├── plotting/        FIGURES: matplotlib only; no Qt; numbers come from core/
    │   ├── domain_figures.py  stats_figures.py  distribution_figures.py  magnetization_figures.py
    │   ├── morphometry_figures.py  width_figures.py  pubfigures.py     (figures of the core modules)
    │   ├── flake_figures.py         flake statistics and pipeline-step figures (CLI export)
    │   ├── distribution_panels.py   the preview / export distribution panels
    │   └── compare_figures.py       align overlay, flip map (vector, Illustrator-safe), local memory,
    │                                population bars, correlation vs shift
    │
    ├── services/        WORKFLOWS: Qt-free; combine core/ and plotting/; used by GUI, CLI and scripts
    │   ├── preview.py       cached preparation + live preview / analysis inside a polygon
    │   ├── export.py        "Export all files" for one image and polygon
    │   ├── batch.py         flake analysis of one image (tables, figures) for the CLI and the export
    │   └── workspace.py     several images: analysis, common grid, alignment, comparison, sessions
    │
    ├── gui/             PySide6 WIDGETS only: layout, signals, drawing what services / plotting return
    │   ├── app.py           main window (Domains | Align | Compare), sessions, drag-and-drop
    │   ├── domains_tab.py   domain analysis per image
    │   ├── align_tab.py     alignment of a pair; correlation-vs-shift dialog
    │   ├── compare_tab.py   comparison, population tables, export
    │   ├── region_editor.py editable polygon / rectangle / ellipse / regular polygon; JSON save / load
    │   └── widgets.py       small reusable widgets
    │
    └── cli/             COMMAND LINE: argument parsing over services/ and core/
        └── analyze.py  compare.py  calibrate.py  export_figures.py   (sda-analyze, sda-compare, ...)
```

## Conventions

- **Definitions** (identical in the docstrings and docs/methods.md):
  - Signal: dichroism asymmetry `A = (I₁ − I₂)/(I₁ + I₂)`, XMCD (circular) or XLD (linear: XMLD + XNLD).
  - Three states: `m ∈ {+1 +ve, 0 neutral (|m| ≤ m₀) / wall, −1 −ve}`.
  - Area fractions `f_k = N_k / N` (k = +ve, neutral, −ve), `<m> = f₊ − f₋`.
  - Net change: `Δ<m> = <m>_after − <m>_before`.
- **Colours**: +ve blue, −ve red everywhere (colors.py); switched pixels purple / magenta or orange / dark green.
- **Calibration** is one parameter `um_per_px` (None = pixels).
- **Tunable knobs** are arguments with defaults from `defaults.py`, never hard-coded constants.
- **Where new code goes**: a new number → `core/`; its figure → `plotting/`; a new workflow → `services/`; a new
  control → `gui/`; a new command → `cli/` (+ an entry in `[project.scripts]` of `pyproject.toml`).
