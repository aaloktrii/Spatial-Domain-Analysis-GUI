# Changelog

## 2.0.0 – 2026-11-01

Restructured release (the first, SDA_v1 of 2026-08-24, is [10.5281/zenodo.22087857](https://doi.org/10.5281/zenodo.22087857)).

- One package, `xmcd_toolkit`, in layers: `core/` (analysis, no plotting or GUI), `plotting/`, `services/`
  (workflows), `gui/` (PySide6), `cli/`; installable with `pip`, commands `sda-gui`, `sda-analyze`, `sda-compare`,
  `sda-export-figures`, `sda-calibrate`.
- Domain analysis of one image: images read as they are (grey or red / white / blue asymmetry maps detected
  automatically, Igor graph exports, raw grey XMCD), invert, zero from a background region or a Sym ± centre,
  scale from the active area, three-state classification, ⟨m_z⟩ four ways, domain statistics and morphometry,
  correlation-corrected statistics, figures, TIFFs and tables.
- Comparison of two images: alignment, flip map, switched fractions, memory r, local memory, wall shift,
  population changes; sessions (`.nc`).
- Graphical program: raw and processed side by side, analysis on request (Calculate / Calculate All), outlines and
  settings copied between images, arrow keys move outlines, export to the raw-data folder; responsive on
  40-megapixel images.
