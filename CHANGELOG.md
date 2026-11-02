# Changelog

## Unreleased

- **+ve / −ve everywhere**: the two states are called +ve and −ve (0 = neutral band) on screen, in every exported
  figure, report and table, for XMCD and XLD alike, since the domains may be out-of-plane or in-plane; the physical
  meaning is the user's to interpret (`xmcd_toolkit/notation.py`).
- **Order parameter** ⟨m⟩ for XMCD, ⟨v_N⟩ for XLD (XMLD + XNLD): *Dichroism* in step 1, `--dichroism` on the
  command line.
- **Table columns renamed** `up` / `down` → `pos` / `neg` (e.g. `frac_up` → `frac_pos`, `n_domains_down` →
  `n_domains_neg`), also in the per-state CSV and JSON files; `sda-calibrate` still reads older tables.
- Fixed: the colour bar of the "domain skeletons" TIFF panel used the width map of the previous panel.
- **No measured data in the repository**: the example image is replaced by two synthetic labyrinthine domain images
  (`examples/make_synthetic.py`, `examples/data/labyrinth_A.png`, `labyrinth_B.png`).

## 2.0.0 – 2026-11-01

Restructured release (the first, SDA_v1 of 2026-08-24, is [10.5281/zenodo.22087857](https://doi.org/10.5281/zenodo.22087857)).

- X-PEEM scope: XMCD and XLD (XMLD + XNLD) maps; states +ve / −ve for both, order parameter ⟨m⟩ (XMCD) or ⟨v_N⟩
  (XLD), one click per image (docs/methods.md, section 2).
- One package, `xmcd_toolkit`, in layers: `core/` (analysis, no plotting or GUI), `plotting/`, `services/`
  (workflows), `gui/` (PySide6), `cli/`; installable with `pip`, commands `sda-gui`, `sda-analyze`, `sda-compare`,
  `sda-export-figures`, `sda-calibrate`.
- Domain analysis of one image: images read as they are (grey or red / white / blue asymmetry maps detected
  automatically, Igor graph exports, raw grey XMCD), invert, zero from a background region or a Sym ± centre,
  scale from the active area, three-state classification, ⟨m⟩ four ways, domain statistics and morphometry,
  correlation-corrected statistics, figures, TIFFs and tables.
- Comparison of two images: alignment, flip map, switched fractions, memory r, local memory, wall shift,
  population changes; sessions (`.nc`).
- Graphical program: raw and processed side by side, analysis on request (Calculate / Calculate All), outlines and
  settings copied between images, arrow keys move outlines, export to the raw-data folder; responsive on
  40-megapixel images.
