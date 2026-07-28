# Spatial Domain Analysis Toolkit

Quantitative analysis of spatial domains, written to analysze XMCD/PEEM dataset.
© Aalok Tiwari

See `FOLDER_SPEC.md` for the full layout and conventions.

## Install

```bash
pip install -r requirements.txt
```

## Quick start

```bash
# domain-area pipeline (segmentation, up/down areas, net m, wall density)
python examples/run_domain_example.py

# three-state m_z statistics (PDF, local number density, domain counts)
python examples/run_stats_example.py
```

Both write figures and CSVs into `outputs/`.

## Use as a library

```python
from xmcd_toolkit import analyze_xmcd, three_state_field, net_change_mz

# one image -> scalar metrics
metrics, layers = analyze_xmcd("my_image.png", um_per_px=0.05)
print(metrics["net_magnetization"], metrics["domain_wall_density_per_um"])

# before/after a current pulse -> net change in magnetization
mz0, roi = three_state_field("before.png")
mz1, _   = three_state_field("after.png")
print(net_change_mz(mz0, mz1, region=roi)["delta_m"])
```

## Working from a text dataset

```python
from xmcd_toolkit import load_xy_mz_txt, pdf_three_state, number_density_local
mz = load_xy_mz_txt("scan.txt")            # 3 columns: x  y  mz
pdf = pdf_three_state(mz)                   # p(+1), p(0), p(-1)
nd  = number_density_local(mz, tile=16)     # local fractions per defined area
```

## Definitions

- Magnetization: XMCD asymmetry `A = (I⁺ − I⁻)/(I⁺ + I⁻) ∝ M_z`
- Order parameter: `⟨m_z⟩ = (A↑ − A↓)/(A↑ + A↓)` ∈ [−1, 1]
- Local number density: `n_s(area) = N_s(area)/N_total(area)` (sums to 1)
- Net change: `Δ⟨m_z⟩ = ⟨m_z⟩_after − ⟨m_z⟩_before`

## Calibration

Areas are reported in pixels until you set `um_per_px` from the image scale
bar (e.g. a 2 µm bar spanning N px → `um_per_px = 2 / N`). All such gaps are
marked `# TODO` in the source.

## Graphical interface

A desktop GUI wraps the whole pipeline:

```bash
python gui.py
```

Workflow: load an image (it displays at once), choose an **active-area shape**
(Polygon, Rectangle, or Square), and draw it over the domains. The three-state
map and the three within-area distributions (domain width, domain area, and
up/down probability) update live as you draw and as you move the m-threshold.

Set the **draw** target to *Background* to outline a non-magnetic region; its
mean becomes the magnetic zero (symmetric thresholding) instead of the Gaussian
high-pass. **Connectivity** selects how domains are counted: `4` counts only
edge-touching pixels as one domain (default), `8` also merges pixels that touch
at a diagonal corner. When `8` is chosen for the domains, the neutral background
is counted with the opposite (`4`) rule to avoid the connectivity paradox. The
CLI exposes the same choice via `--connectivity {4,8}`.

**Calibration** sets the pixel size from a reference rectangle's known
width and/or height, in one of three modes:

- *Whole image (W x H)* -- enter the physical width and height of the whole
  image.
- *Drawn shape (W x H)* -- enter the known width and height of the drawn
  polygon/rectangle/square (its bounding box); a rectangle can carry two
  different dimensions. Leave one field blank to calibrate from a single axis.
- *Pixels (none)* -- work uncalibrated.

If both width and height are given the pixel size is the mean of the two axes,
with a warning if they disagree (non-square pixels or a wrong dimension). The
live readout shows the drawn box in both px and um and the derived um/px.

Rectangles and squares are stored as polygons, so all downstream analysis is
identical. Polygons/boxes can be saved and loaded as JSON to reuse the same
active area across a pulse series, and **Use whole image** selects the full frame.

Performance: the expensive denoise/flatten is computed once per image and cached,
so moving the m-threshold or redrawing re-runs only the fast three-state and
morphometry steps (a threshold change is ~5-7x faster than a cold analysis).

Drag-and-drop needs `tkinterdnd2` (`pip install tkinterdnd2`); without it the
Browse button still works. The GUI also needs Tk, which ships with the
python.org macOS installer; with Homebrew Python run `brew install python-tk`.
