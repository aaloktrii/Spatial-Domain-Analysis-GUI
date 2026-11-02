# The graphical program

```bash
sda-gui                 # or: sda-gui session.nc      (python -m xmcd_toolkit.gui.app also works)
```

One window: a **File** menu (Add Images…, Load Session…, Save Session…, Clear Workspace), the numbered steps on
the left, and three views on the right: **Domains | Align | Compare**. The steps of the current view are shown;
step 1 is always visible. Figures have a compact toolbar whose **Export…** saves a vector PDF / SVG or a
600-dpi PNG.

Nothing is recalculated while you edit: outlines, sliders and menus are stored at once and the image is marked
*out of date* (red title of the processed panel); press **Calculate** when you are ready.

## Domains (steps 1–7): one image at a time

The view shows the **raw file** (left, exactly as the file is) next to the **processed three-state map**
(right: up blue, neutral grey, down red, inside the active area). Both zoom and pan together. The Results box
lists the numbers; the panels below show the width, period and up / neutral / down distributions.

1. **Load** – *Add Images…* or drop files on the window. Any picture works (.tif, .png, .jpg, .bmp, …); it is
   read as it is, every pixel, as an asymmetry map: a grey picture as white +1 / black −1, a coloured one as
   blue +1 / white 0 / red −1 (detected automatically). *Invert colours* swaps up and down for a map whose colours
   mean the opposite sign. *Settings from* copies the settings of another image (everything except the outlines)
   to this image or to all images.
2. **Active area / background** – draw the **active area** (the flake: polygon, rectangle, ellipse or regular
   polygon) and, optionally, a **background** region on bare substrate. Drag a corner to reshape, SHIFT + drag to
   move; click the picture, then the arrow keys move the outline (SHIFT + arrow: 10 steps). *Save / Load Region…*
   keep outlines as JSON. **Sym ± centre** sets the zero of the map: 0 (or the background region's mean if one
   is drawn), or the median / mean in the background region, else of the map without its edge drift strips.
   *Outlines from* copies the outlines of one image to this image or to all images (scaled if the image size
   differs).
3. **Calibration** – pixels, the whole image of known size, or the active area's box of known size (µm).
4. **Denoise** – none (default: the image as it is), TV, non-local means or median.
5. **Classify** – the neutral band m₀, the smallest domain kept, 4- or 8-connectivity, the width and boundary
   methods. *Measure domain widths* (off by default above 4 MP, where it is slow) and *Picture under the processed
   map*. **Calculate** (this image) and **Calculate All** (every image that is new or out of date).
6. **Distributions** – which quantity the panels show; optional Q-Q and log-normal-vs-Gaussian panels.
7. **Export** – *Export All Files* writes figures, distributions, TIFFs and tables into
   `<raw data folder>/<image>_analysis/`; *Choose Folder…* sends them elsewhere.

### The numbers in the Results box

The map is put on −1 … +1 by dividing by the 99th percentile of |signal| inside the active area (so defects
outside it do not set the scale). Then, inside the active area:

| | with the neutral band, m_z(m₀) | without it, m_z |
|---|---|---|
| discrete | (N↑ − N↓) / N, neutral pixels count as 0 | mean of sign(m) |
| continuous | mean of m over the up and down pixels | mean of m over all pixels |

plus *up vs down* = (N↑ − N↓)/(N↑ + N↓), the pixel counts N↑, N₀, N↓, the zero used, domain widths, period
and domain counts. See [methods.md](methods.md) for every definition.

## Align (steps 8–11): put two images on top of each other

Choose a pair, A (before) and B (after). Rotate, shift or flip A, B or both, rotate the whole view, then
**Auto Align** (a first guess) or click matching points in A and B and **Fit Points**. *Check: Correlation vs
Shift…* tests the alignment: the correlation should peak at zero shift.

## Compare (steps 12–15): pixel by pixel

Uses the three-state maps of the Domains view (same m₀). Shows the flip map (stayed up, stayed down, up→down,
down→up, uncertain; colour-blind-safe switch colours), the switched fractions, the memory r with a block-bootstrap
interval, the local memory map, the wall shift and the population changes Δf↑ / Δf₀ / Δf↓ / Δ⟨m_z⟩.
A comparison region can be drawn, saved and loaded. *Export PDF + CSV…* writes an editable vector PDF and a table.

## Sessions

*Save Session…* stores the images, settings, outlines, alignment, comparisons and history in one `.nc` file;
*Load Session…* re-runs the analysis from the original files.
