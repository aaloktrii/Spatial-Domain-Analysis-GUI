"""Two synthetic labyrinthine domain images (no measured data): a "before" and an "after" state of the same flake.

    python examples/make_synthetic.py            # writes examples/data/labyrinth_A.png and labyrinth_B.png

Each image is a grey asymmetry map (white +1, mid-grey 0, black -1): a labyrinth of +ve / -ve domains inside an
irregular flake on flat substrate, with domain walls of finite width and pixel noise. The labyrinth is white noise
filtered to one stripe period (a ring in Fourier space); B keeps most of A's pattern and changes the rest, so the
pair can be aligned and compared. Fixed seeds: the files are the same on every run.
"""
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

SIZE = 512            # px
PERIOD = 24.0         # stripe period (px): domain width ~ PERIOD / 2
WALL = 0.35           # wall sharpness (smaller = sharper walls)
NOISE = 0.12          # pixel noise (in units of the full contrast)
MEMORY = 0.75         # how much of A's pattern B keeps (1 = identical, 0 = new pattern)


def labyrinth(rng, size=SIZE, period=PERIOD):
    """Zero-mean field whose sign forms a labyrinth: white noise kept only near |k| = 1 / period."""
    k = np.hypot(*np.meshgrid(np.fft.fftfreq(size), np.fft.fftfreq(size)))
    ring = np.exp(-0.5 * ((k - 1.0 / period) / (0.15 / period)) ** 2)
    f = np.real(np.fft.ifft2(np.fft.fft2(rng.standard_normal((size, size))) * ring))
    return f / f.std()


def flake(rng, size=SIZE):
    """Irregular flake mask: a blurred random blob, thresholded and kept as one piece."""
    yy, xx = np.mgrid[0:size, 0:size] / size - 0.5
    blob = 0.35 - np.hypot(xx * 1.1, yy * 0.9) + 0.06 * ndi.gaussian_filter(rng.standard_normal((size, size)), 25) * 25
    lab, n = ndi.label(blob > 0)
    return lab == (1 + np.argmax(np.bincount(lab.ravel())[1:])) if n else blob > 0


def to_png(m, roi, rng, path):
    m = np.where(roi, np.tanh(m / WALL), 0.0) + NOISE * rng.standard_normal(m.shape)
    Image.fromarray(np.clip((m + 1) * 127.5, 0, 255).astype(np.uint8)).save(path)


def main(out=Path(__file__).parent / "data"):
    out.mkdir(exist_ok=True)
    rng = np.random.default_rng(2026)
    roi = flake(rng)
    a = labyrinth(rng)
    b = MEMORY * a + np.sqrt(1 - MEMORY ** 2) * labyrinth(rng)      # same statistics, partly new pattern
    to_png(a, roi, rng, out / "labyrinth_A.png")
    to_png(b, roi, rng, out / "labyrinth_B.png")
    print(f"wrote {out / 'labyrinth_A.png'} and {out / 'labyrinth_B.png'}")


if __name__ == "__main__":
    main()
