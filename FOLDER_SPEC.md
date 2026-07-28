# Folder specification — `xmcd_domain_toolkit`

Quantitative analysis of magnetic domains in XMCD-PEEM images of
Graphene/Fe₃GeTe₂/WTe₂ SOT heterostructures. This document defines the
layout, the role of each file, and the conventions the code follows so the
toolkit and the manuscript stay in sync.

```
xmcd_domain_toolkit/
├── README.md                  Quick start, definitions, example commands
├── FOLDER_SPEC.md             This file — layout + conventions
├── requirements.txt           Python dependencies
├── setup_env.sh               Build a .venv and install all dependencies
├── analyze.py                 CLI: flake-area statistics on any file/folder
│
├── xmcd_toolkit/              Importable package (the method)
│   ├── __init__.py            Public API surface + definitions
│   ├── loaders.py             Input: image (chrome-strip) and .txt (x,y,mz)
│   ├── domain.py              Segmentation, up/down classification, metrics
│   ├── stats.py               Three-state PDF, local number density,
│   │                          net-change m_z, domain-count distribution
│   └── distributions.py       Log-normal (size) and power-law (avalanche) fits
│
├── examples/                 Worked examples (runnable)
│   ├── run_domain_example.py  Domain-area pipeline on the bundled image
│   ├── run_stats_example.py   Three-state statistics on the bundled image
│   └── data/
│       └── device_25_7_example.png   Example XMCD-PEEM frame
│
├── outputs/                  Generated artifacts (figures, CSVs) — git-ignored
│   └── .gitkeep
│
└── manuscript/               The paper this method supports
    ├── main.tex               PRB Letters draft (revtex4-2)
    └── references.bib         Bibliography
```

## Module responsibilities (single source of truth)

- **loaders.py** — All input parsing. `load_image` returns a normalized
  grayscale array with acquisition-software chrome removed. `load_xy_mz_txt`
  builds an (H,W) m_z grid from a three-column `x y mz` text file. No
  analysis logic lives here.
- **domain.py** — `analyze_xmcd(image, ...)` is the core entry point and
  accepts a path **or** an array. It owns segmentation, illumination
  flattening, up/down classification, and the scalar metrics (areas, net
  magnetization, domain counts, size distribution, wall length/density).
  Returns `(metrics, layers)`; `layers` is reused by `stats.py`.
- **stats.py** — Everything probabilistic. Builds the three-state field,
  the PDF, the local number-density tiles, the net-change, and the
  domain-count distribution. Imports from `domain.py`; never the reverse.

## Conventions

- **Definitions** (kept identical in code docstrings and the manuscript):
  - Magnetization: XMCD asymmetry `A = (I+ - I-)/(I+ + I-) ~ M_z`.
  - Order parameter: `<m_z> = (A_up - A_down)/(A_up + A_down)`, range [-1, 1].
  - Number density (local): `n_s(area) = N_s(area) / N_total(area)`; the three
    fractions sum to 1 in every defined area.
  - Net change: `d<m_z> = <m_z>_after - <m_z>_before`.
  - Three states: `m_z ∈ {+1 up, 0 none/wall, -1 down}`.
- **Calibration** is a single parameter `um_per_px`, defaulting to `None`
  (results stay in pixels). Set it from the image scale bar. Every TODO that
  blocks physical units is tagged `# TODO` in the source.
- **Tunable knobs** are function arguments with defaults, not hard-coded
  constants: `tv_weight`, `contrast_window`, `flatten_sigma`, `z_neutral`,
  `tile`, `min_domain_px`.
- **Outputs** are written only to `outputs/`. The package never writes inside
  `xmcd_toolkit/` or `examples/data/`.
- **Imports** flow one way: `loaders → domain → stats → examples`. Add new
  input formats in `loaders.py`, new scalar metrics in `domain.py`, new
  probabilistic measures in `stats.py`.

## Planned (GUI) extensions, and where they attach

1. Image processing with user-defined **background region** and **active
   area**, three-color identification + surface-area distribution →
   extends `domain.py` (accept explicit ROI/background masks instead of the
   automatic detector).
2. `.txt` → xy m_z map → probability distribution vs current → already
   supported via `loaders.load_xy_mz_txt` + `stats`; the GUI layer would call
   these and add interactive ROI selection.
