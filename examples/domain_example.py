"""Domain analysis of the bundled example image: scalar metrics, a figure and a CSV.

    python examples/domain_example.py

Writes into ./outputs/.
"""
import csv
from pathlib import Path

from xmcd_toolkit import analyze_xmcd, make_domain_figure, print_report

IMAGE = Path(__file__).parent / "data" / "device_25_7_example.png"
OUT = Path("outputs")
UM_PER_PX = None          # set from the scale bar (e.g. a 2 um bar spanning N px -> 2 / N); None = pixels

OUT.mkdir(exist_ok=True)
metrics, layers = analyze_xmcd(str(IMAGE), um_per_px=UM_PER_PX)
print_report(metrics)
make_domain_figure(metrics, layers, str(OUT / "example_domain_analysis.png"), title="Example image")
with open(OUT / "example_metrics.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["metric", "value"])
    w.writerows(metrics.items())
print(f"wrote {OUT / 'example_domain_analysis.png'} and {OUT / 'example_metrics.csv'}")
