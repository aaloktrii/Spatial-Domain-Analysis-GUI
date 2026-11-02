"""Domain analysis of an image: scalar metrics, a figure and a CSV.

    python examples/domain_example.py                 # the synthetic example (examples/data/labyrinth_A.png)
    python examples/domain_example.py my_map.png      # your own image

Writes into ./outputs/.
"""
import csv
import sys
from pathlib import Path

from xmcd_toolkit import analyze_xmcd, make_domain_figure, print_report

IMAGE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "data" / "labyrinth_A.png"
OUT = Path("outputs")
UM_PER_PX = None          # set from the scale bar (e.g. a 2 um bar spanning N px -> 2 / N); None = pixels

OUT.mkdir(exist_ok=True)
metrics, layers = analyze_xmcd(str(IMAGE), um_per_px=UM_PER_PX, input_mode="asymmetry_map", map_mode="auto")
print_report(metrics)
make_domain_figure(metrics, layers, str(OUT / "example_domain_analysis.png"), title="Example image")
with open(OUT / "example_metrics.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["metric", "value"])
    w.writerows(metrics.items())
print(f"wrote {OUT / 'example_domain_analysis.png'} and {OUT / 'example_metrics.csv'}")
