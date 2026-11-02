# Spatial Domain Analysis Toolkit

[![tests](https://github.com/aaloktrii/Spatial-Domain-Analysis-GUI/actions/workflows/tests.yml/badge.svg)](https://github.com/aaloktrii/Spatial-Domain-Analysis-GUI/actions/workflows/tests.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22087856.svg)](https://doi.org/10.5281/zenodo.22087856)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Quantitative analysis and comparison of magnetic domains in X-PEEM images (XMCD and XLD).
This toolkit is part of a broader general-purpose framework for N-dimensional spectroscopic analysis
([TensorSpec](https://github.com/fawkesdx/TensorSpec)).

The toolkit turns an X-ray dichroism (XMCD or XLD) asymmetry map into +ve / neutral / −ve domains and measures them: area
fractions and the order parameter (⟨m⟩ or ⟨v_N⟩), domain counts, sizes, widths, periods and walls, with correlation-corrected statistics. Two images (for
example before and after a current pulse) can be aligned and compared pixel by pixel: which domains switched, how
much of the pattern is remembered, and how the populations changed.

It can be used three ways: a **graphical program**, **command-line tools** for batch work, or a **Python library**.

## Features

- **Any image, read as it is** – .tif, .png, .jpg, .bmp, …; grey maps (white +1 / black −1) and red / white /
  blue maps detected automatically; Igor graph exports decoded to their original pixel grid.
- **Three-state classification** with a neutral band m₀, inside a drawn active area; the zero from a background
  region or a Sym ± centre (0 / median / mean); the scale from the active area only, so defects elsewhere do not
  distort it.
- **Domain statistics** – ⟨m⟩ (XMCD) or ⟨v_N⟩ (XLD), discrete and continuous, with and without m₀; pixel counts; domain counts and areas;
  widths (four methods), periods, wall density; log-normal / Gaussian fits; block-bootstrap confidence intervals.
- **Comparison of two states** – alignment (automatic or from point pairs, checked by correlation vs shift), flip
  map, switched fractions, memory r, local memory, wall shift, population changes.
- **Publication output** – figures (vector PDF / EPS or PNG, Illustrator-safe), TIFF panels, CSV tables, sessions.
- **Large images** – 40-megapixel frames stay responsive (analysis on request, screen-resolution drawing).

## XMCD and XLD

The analysis works on any signed dichroism map and computes the same numbers for both. The two states are called
**+ve** and **−ve** (with 0 the neutral band); what they mean physically is for the user to interpret: the two magnetization
directions (out-of-plane or in-plane) for **XMCD**, or the two axis populations for **XLD** (XMLD + XNLD, e.g. antiferromagnets). The
order parameter is written **⟨m⟩** for XMCD and **⟨v_N⟩** (Néel vector) for XLD: one click per image
(*Dichroism*, step 1). See [docs/methods.md](docs/methods.md), section 2.

## Installation

Python 3.10 or newer.

```bash
git clone https://github.com/aaloktrii/Spatial-Domain-Analysis-GUI.git
cd Spatial-Domain-Analysis-GUI
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[gui]"                                 # without ".[gui]": library and command line only
```

## Quick start

**Graphical program**

```bash
sda-gui
```

Add images, draw the active area (and a background region), press **Calculate**. See [docs/gui.md](docs/gui.md).

**Command line**

```bash
sda-analyze images/ --um-per-px 0.02 --tiff             # statistics, tables and figures for every image
sda-compare before.png after.png                         # before / after magnetization-change test
sda-export-figures image.png --um-per-px 0.02            # three calibrated domain panels as TIFFs
sda-calibrate summary_table.csv --um-per-px 0.02         # apply a calibration to a summary table
```

Each command has `--help`. `sda-analyze` writes next to each image (`<image>_analysis/`), `sda-compare` into
`compare_<before>_vs_<after>/`; `--outdir` sends the results elsewhere.

**Python**

```python
from xmcd_toolkit import analyze_xmcd, three_state_field, pdf_three_state

metrics, layers = analyze_xmcd("map.png", input_mode="asymmetry_map", map_mode="auto", um_per_px=0.02)
mz, roi = three_state_field(layers, m_threshold=0.1)        # +1 +ve, 0 neutral, -1 −ve
p = pdf_three_state(mz, region=roi)
print(p["p_up"], p["p_zero"], p["p_down"])
```

Runnable examples on the bundled image: `python examples/domain_example.py` and `python examples/stats_example.py`.

## Documentation

- [docs/methods.md](docs/methods.md) – the analysis step by step, every definition and default (for a methods section)
- [docs/gui.md](docs/gui.md) – the graphical program
- [docs/architecture.md](docs/architecture.md) – code layout and where new code goes

## Tests

```bash
pip install -e ".[test]"
python -m pytest
```

## How to cite

If you use this toolkit in published work, please cite it:

> Aalok Tiwari, *Spatial Domain Analysis Toolkit* (2026). Zenodo. [https://doi.org/10.5281/zenodo.22087856](https://doi.org/10.5281/zenodo.22087856)

This DOI always resolves to the latest version; each release also has its own DOI on Zenodo (e.g. SDA_v1:
[10.5281/zenodo.22087857](https://doi.org/10.5281/zenodo.22087857)).

```bibtex
@software{tiwari_spatial_domain_analysis,
  author  = {Tiwari, Aalok},
  title   = {Spatial Domain Analysis Toolkit},
  year    = {2026},
  url     = {https://github.com/aaloktrii/Spatial-Domain-Analysis-GUI},
  doi     = {10.5281/zenodo.22087856}
}
```

The citation details are also in [CITATION.cff](CITATION.cff) (GitHub's "Cite this repository" button).

## License

MIT License. Copyright © 2026 Aalok Tiwari. See [LICENSE](LICENSE).
