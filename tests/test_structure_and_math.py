"""Layering rules and the comparison maths (run: python -m pytest tests)."""
import ast
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PKG = ROOT / "xmcd_toolkit"


def _imports(path):
    tree = ast.parse(path.read_text())
    mods = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            mods |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module and n.level == 0:
            mods.add(n.module.split(".")[0])
    return mods


@pytest.mark.parametrize("path", sorted((PKG / "core").rglob("*.py")), ids=lambda p: p.name)
def test_core_has_no_plotting_or_gui(path):
    assert not _imports(path) & {"matplotlib", "PySide6", "PyQt5", "tkinter"}, path


@pytest.mark.parametrize("path", sorted((PKG / "plotting").rglob("*.py")) + sorted((PKG / "services").rglob("*.py")),
                         ids=lambda p: p.name)
def test_plotting_and_services_have_no_gui(path):
    assert not _imports(path) & {"PySide6", "PyQt5", "tkinter"}, path


def test_populations_match_hand_calculation():
    from xmcd_toolkit.core import populations as P
    gs, ia = P.fractions(97114, 9177, 91090), P.fractions(100610, 14388, 82383)
    c = P.change(gs, ia)
    assert round(100 * c["d_up"], 2) == 1.77 and round(100 * c["d_neutral"], 2) == 2.64
    assert round(100 * c["d_down"], 2) == -4.41 and round(100 * c["d_mz"], 2) == 6.18
    assert round(100 * c["d_mz_direct"], 2) == 3.54 and round(100 * c["d_mz_via_neutral"], 2) == 2.64
    assert round(100 * c["changed_lower"], 1) == 4.4


def test_flip_map_same_and_inverted():
    from xmcd_toolkit.core.comparison import flip_map
    rng = np.random.default_rng(0)
    s = rng.choice([-1, 1], (40, 40))
    roi = np.ones_like(s, bool)
    assert flip_map(s, s, roi)[1]["switched_pct"] == 0
    m = flip_map(s, -s, roi)[1]
    assert m["switched_pct"] == 100 and m["memory_from_flips"] == -1


def test_transform_fit_recovers_known_motion():
    from xmcd_toolkit.core import transform as T
    c = T.centre((512, 512))
    tf = {**T.TF_DEFAULT, "rot": 4.0, "dy": 7.0, "dx": -5.0, "scale": 1.02}
    pts = np.array([[150, 150], [150, 350], [350, 200], [300, 380]], float)
    moved = np.array([c + tf["scale"] * T.rot_rc(tf["rot"]) @ (p - c) + [tf["dy"], tf["dx"]] for p in pts])
    th, s, t, rms = T.fit_similarity(moved, pts, c, allow_scale=True)
    back = T.compose(tf, 0.0, th, s, t)
    assert abs(back["rot"]) < 1e-6 and abs(back["scale"] - 1) < 1e-9 and abs(back["dy"]) < 1e-6 and rms < 1e-6


def test_shift_correlation_finds_known_offset():
    from scipy import ndimage
    from xmcd_toolkit.core.registration import shift_correlation
    rng = np.random.default_rng(1)
    a = ndimage.gaussian_filter(rng.normal(size=(160, 160)), 3)
    b = ndimage.shift(a, (5, -3), order=1)          # B is A moved by (+5, -3)
    roi = np.zeros_like(a, bool)
    roi[30:130, 30:130] = True
    res = shift_correlation(a, lambda ang: b, roi, max_shift=8)
    assert res["best"][1:3] == (-5, 3)              # move B back by (-5, +3)



def test_files_are_read_as_they_are_only_igor_is_decoded(tmp_path, monkeypatch):
    """Every input type except an explicitly chosen Igor export reads the file itself, every pixel."""
    from PIL import Image
    from xmcd_toolkit.core import igor_import
    from xmcd_toolkit.core.loaders import detect_map_mode
    monkeypatch.setattr(igor_import, "CACHE", tmp_path / "cache")
    rng = np.random.default_rng(1)
    small = rng.integers(40, 220, (60, 50)).astype(np.uint8)
    big = np.kron(small, np.ones((6, 6), np.uint8))
    f = tmp_path / "grey_export.png"
    Image.fromarray(np.stack([big] * 3, -1)).save(f)
    for mode in ("asymmetry_map", "grayscale"):
        assert igor_import.source_for(f, mode) == (f, mode)
        assert igor_import.analysis_shape(f, mode) == big.shape
    val, used = igor_import.source_for(f, "igor_export")
    assert used == "asymmetry_map" and val.shape == small.shape
    assert np.corrcoef(val.ravel(), small.ravel().astype(float))[0, 1] > 0.99
    assert detect_map_mode(f) == "bw"

def test_any_format_and_automatic_map_colours(tmp_path):
    """Grey -> black/white map, colour -> red/white/blue map; PNG, JPG, TIFF, BMP all load."""
    from PIL import Image
    from xmcd_toolkit.core.loaders import detect_map_mode, load_asymmetry_map
    m = np.tile(np.linspace(-1, 1, 40), (30, 1))
    rwb = np.stack([np.where(m < 0, 255, 255 * (1 - m)), 255 * (1 - abs(m)), np.where(m > 0, 255, 255 * (1 + m))],
                   -1).astype(np.uint8)
    grey = ((m + 1) * 127.5).astype(np.uint8)
    for ext in ("png", "jpg", "tif", "bmp"):
        Image.fromarray(rwb).save(tmp_path / f"c.{ext}")
        Image.fromarray(grey).save(tmp_path / f"g.{ext}")
        assert detect_map_mode(tmp_path / f"c.{ext}") == "rb"
        assert detect_map_mode(tmp_path / f"g.{ext}") == "bw"
        for name in ("c", "g"):
            out = load_asymmetry_map(tmp_path / f"{name}.{ext}", mode="auto")
            assert np.corrcoef(out.ravel(), m.ravel())[0, 1] > 0.95, (name, ext)


def test_non_square_blocks_keep_the_file_shape(tmp_path, monkeypatch):
    """Blocks 6 tall x 9 wide in a 360 x 360 picture: read on a square 60 x 60 grid (the file's shape), with values
    taken from the file only (no interpolation)."""
    from PIL import Image
    from xmcd_toolkit.core import igor_import
    monkeypatch.setattr(igor_import, "CACHE", tmp_path / "cache")
    rng = np.random.default_rng(2)
    small = rng.integers(40, 220, (60, 40)).astype(np.uint8)
    big = np.kron(small, np.ones((6, 9), np.uint8))
    f = tmp_path / "wide_blocks.png"
    Image.fromarray(np.stack([big] * 3, -1)).save(f)
    rgb, info = igor_import.sample_grid(f)
    assert rgb.shape[:2] == (60, 60)
    assert set(np.unique(rgb[..., 0])) <= set(np.unique(small))


def test_maps_are_read_as_they_are(tmp_path):
    """No contrast stretch: a low-contrast grey map keeps its values (mid-grey 0, white +1, black -1), and the
    picture shown is the file's colours."""
    from PIL import Image
    from xmcd_toolkit.core.loaders import load_asymmetry_map, load_picture
    g = np.full((20, 20), 128, np.uint8)
    g[:, :10] = 160                      # weak contrast, would be stretched to +-1 by a percentile normalisation
    f = tmp_path / "weak.png"
    Image.fromarray(g).save(f)
    m = load_asymmetry_map(f, mode="auto")
    assert np.allclose(m[:, :10], 2 * 160 / 255 - 1) and np.allclose(m[:, 10:], 2 * 128 / 255 - 1)
    rgb = np.random.default_rng(3).integers(0, 256, (12, 15, 3)).astype(np.uint8)
    Image.fromarray(rgb).save(tmp_path / "c.png")
    assert np.array_equal(load_picture(tmp_path / "c.png"), rgb)


def test_view_warp_is_the_full_warp_on_every_step_th_pixel():
    """The Align drawing (every step-th pixel) shows exactly the pixels the full-size transform gives there."""
    from xmcd_toolkit.core import transform
    rng = np.random.default_rng(4)
    img = rng.integers(1, 9, (91, 91)).astype(np.int16)
    tf = {**transform.TF_DEFAULT, "rot": 7.0, "dy": 3.2, "dx": -5.1}
    step = 3
    full = transform.warp(img, tf, 12.0, order=0, out_shape=img.shape)
    view = transform.warp_view(img, tf, 12.0, 0, img.shape, step)
    h = (step - 1) // 2
    sub = full[h::step, h::step]
    assert view.shape == sub.shape
    assert (view == sub).mean() > 0.97        # equal except where a sample lands exactly on a pixel edge



def test_fast_polygon_mask_equals_skimage():
    from skimage.draw import polygon as sk_polygon
    from xmcd_toolkit.core.regions import _scanline_fill
    rng = np.random.default_rng(5)
    for _ in range(100):
        n, H, W = rng.integers(3, 25), 120, 97
        xs, ys = rng.uniform(-10, W + 10, n), rng.uniform(-10, H + 10, n)
        ref = np.zeros((H, W), bool)
        rr, cc = sk_polygon(ys, xs, (H, W))
        ref[rr, cc] = True
        assert np.array_equal(_scanline_fill(xs, ys, (H, W)), ref)



def test_calculate_only_on_request(tmp_path):
    """Loading reads the picture only; changes mark the image out of date; calculate() runs the analysis."""
    from PIL import Image
    from xmcd_toolkit.services.workspace import Workspace
    g = np.full((60, 80), 128, np.uint8)
    g[:, :40] = 230
    g[:, 40:] = 30
    f = tmp_path / "map.png"
    Image.fromarray(g).save(f)
    ws = Workspace()
    k = ws.add_image(f, run=False)
    assert ws.images[k]["result"] is None and not ws.is_ready(k) and ws.size(k) == (60, 80)
    ws.calculate(k)
    assert ws.is_ready(k)
    ws.set_params(k, run=False, m_threshold=ws.images[k]["params"]["m_threshold"])   # unchanged value
    assert ws.is_ready(k)
    ws.set_params(k, run=False, m_threshold=0.3)
    assert not ws.is_ready(k)
    ws.calculate(k)
    mz = ws.images[k]["result"]["mz"]
    assert (mz[:, :40] == 1).all() and (mz[:, 40:] == -1).all()



def test_mz_four_ways():
    from xmcd_toolkit.core import mz_four_ways
    m = np.array([[0.9, 0.5, 0.05, -0.05, -0.6, -0.2]])
    mz = np.array([[1, 1, 0, 0, -1, -1]], np.int8)
    roi = np.ones_like(mz, bool)
    r = mz_four_ways(mz, {"asymmetry": m, "roi": roi, "flat": m})
    assert (r["N_up"], r["N_0"], r["N_down"]) == (2, 2, 2)
    assert np.isclose(r["discrete_m0"], 0.0) and np.isclose(r["discrete"], 0.0)
    assert np.isclose(r["continuous_m0"], (0.9 + 0.5 - 0.6 - 0.2) / 4)
    assert np.isclose(r["continuous"], m.mean())
    assert np.isclose(r["up_vs_down"], 0.0)



def test_invert_and_recall_settings(tmp_path):
    """Invert swaps up and down; settings and outlines copy between images (outlines scaled to the size)."""
    from PIL import Image
    from xmcd_toolkit.services.workspace import Workspace
    g = np.full((60, 80), 128, np.uint8)
    g[:, :40], g[:, 40:] = 230, 30
    f1, f2 = tmp_path / "a.png", tmp_path / "b.png"
    Image.fromarray(g).save(f1)
    Image.fromarray(np.kron(g, np.ones((2, 2), np.uint8))).save(f2)      # same picture at twice the size
    ws = Workspace()
    a, b = ws.add_image(f1, run=False), ws.add_image(f2, run=False)
    ws.set_params(a, run=False, invert=True, m_threshold=0.2, polygon=[(5.2, 5.2), (70.3, 5.2), (70.3, 50.1)])
    ws.calculate(a)
    mz = ws.images[a]["result"]["mz"]
    assert (mz[10, 30] == -1) and (mz[10, 50] == 1)                     # white -> down, black -> up
    ws.copy_settings(a, b)
    assert ws.images[b]["params"]["invert"] and ws.images[b]["params"]["m_threshold"] == 0.2
    assert ws.images[b]["params"]["polygon"] is None                      # outlines not part of settings
    assert not ws.is_ready(b)
    ws.copy_regions(a, b, run=False)
    assert np.allclose(ws.images[b]["params"]["polygon"][0], (10.9, 10.9))   # scaled x2 (pixel centres)



def test_sym_centre():
    from xmcd_toolkit.core.offset import edge_strips, sym_centre
    rng = np.random.default_rng(6)
    a = 0.2 + 0.01 * rng.standard_normal((100, 120))
    a[:, :6] = rng.standard_normal((100, 6))             # noisy drift strip at the left edge
    keep = edge_strips(a)
    assert not keep[:, :6].any() and keep[:, 20:].all()
    assert sym_centre(a, "0")[0] == 0.0
    c, n, where = sym_centre(a, "median")
    assert abs(c - 0.2) < 0.005 and n < a.size and "edges" in where
    region = np.zeros_like(a, bool)
    region[50:60, 50:60] = True
    a[region] = -0.5
    assert np.isclose(sym_centre(a, "mean", region)[0], -0.5)


def test_calibrated_widths_with_an_empty_state():
    """A state with (almost) no pixels still returns the nm keys when calibrated (no KeyError downstream)."""
    from xmcd_toolkit.core import per_state_morphometry
    mz = np.zeros((40, 40), int)
    mz[5:30, 5:30] = 1                       # only up domains, no down
    mm = per_state_morphometry(mz, um_per_px=0.02)
    assert np.isnan(mm["down"]["width_median_nm"]) and mm["up"]["width_median_nm"] > 0
