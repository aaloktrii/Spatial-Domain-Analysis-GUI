"""Three-state m_z statistics of the bundled example image: probabilities, local number density, domain counts.

    python examples/stats_example.py

Writes a figure, the per-pixel (x, y, mz) table and the number-density tiles into ./outputs/.
"""
import csv
from pathlib import Path

import numpy as np

from xmcd_toolkit import (analyze_xmcd, domain_count_distribution, make_stats_figure, number_density_local,
                          pdf_three_state, three_state_field)

IMAGE = Path(__file__).parent / "data" / "device_25_7_example.png"
OUT = Path("outputs")
M_THRESHOLD = 0.1         # neutral band |m| <= m0
TILE = 16                 # px per tile for the local number density

OUT.mkdir(exist_ok=True)
_, layers = analyze_xmcd(str(IMAGE))
mz, roi = three_state_field(layers, m_threshold=M_THRESHOLD)

p = pdf_three_state(mz, region=roi)
nd = number_density_local(mz, tile=TILE)
dc = domain_count_distribution(mz)
print(f"p(+1 / 0 / -1) in the flake = {p['p_up']:.3f} / {p['p_zero']:.3f} / {p['p_down']:.3f}")
print(f"up vs down (N_up - N_dn) / (N_up + N_dn) = {p['net_m']:+.4f}")
print(f"domains: up {dc['up']['n_domains']}, down {dc['down']['n_domains']}")

make_stats_figure(mz, roi, str(OUT / "example_statistics.png"), tile=TILE, title="Three-state m_z statistics")
ys, xs = np.mgrid[0:mz.shape[0], 0:mz.shape[1]]
with open(OUT / "mz_pixels.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["x", "y", "mz"])
    w.writerows(zip(xs.ravel().tolist(), ys.ravel().tolist(), mz.ravel().tolist()))
with open(OUT / "number_density_tiles.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(nd["tiles"][0].keys()))
    w.writeheader()
    w.writerows(nd["tiles"])
print(f"wrote {OUT}/example_statistics.png, mz_pixels.csv, number_density_tiles.csv")
