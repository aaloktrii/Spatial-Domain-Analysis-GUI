"""Workspace: several images analysed with the domain pipeline, aligned onto one grid and compared pairwise.
Qt-free service layer (front ends: gui/, scripts). All numbers come from core/; this module only keeps the state
and calls the right functions in order.

  add_image(path, params)      domain analysis (services.preview.preview_polygon) of one image
  set_params(key, **params)    re-run it (threshold, polygon, denoising, ...)
  final(key, what)             'mz' (three-state), 'm' (asymmetry), 'roi', 'img' on the common grid
  auto_align / fit_points      alignment of one image onto another (transforms are not destructive)
  shift_check                  correlation vs shift (alignment test)
  compare(a, b)                flip map, switched fractions, memory r, local r, wall shift, population changes
  save / load                  session file (core/session.py, DataTree .nc)
"""
import datetime
import os

import numpy as np
from scipy import ndimage

from ..defaults import DEFAULTS
from ..core import asymmetry_field
from ..core import comparison, populations, registration, session, transform
from ..core.regions import calibration, polygon_to_mask
from .preview import picture_of, preview_polygon

DEFAULT_PARAMS = {**{k: DEFAULTS[k] for k in ("m_threshold", "tv_weight", "flatten_sigma", "min_px", "width_method",
                                                 "denoise_method", "boundary_method", "connectivity", "z_neutral",
                                                 "states", "clahe", "flatten", "input_mode", "map_mode", "invert",
                                                 "sym_centre", "dichroism")},
                  "polygon": None, "bg_polygon": None, "um_per_px": None, "fov": None, "show_boundary": False,
                  "measure_widths": None}
LIVE_WIDTHS_MAX_PX = 4_000_000   # above this, domain widths are measured on request (or at export)
COMPARE_DEFAULTS = {"window_px": 10.0, "block_um": 1.0, "n_boot": 1000, "roi": "intersection"}


class Workspace:
    def __init__(self):
        self.images = {}          # key -> dict(path, params, tf, result)
        self.compares = {}        # (a, b) -> dict(flip, r_local, metrics, settings, layers)
        self.view_rot = 0.0
        self.reference = None     # key whose grid is the common grid
        self.compare_polygon = None   # optional drawn comparison ROI (final frame, (x, y))
        self.history = []

    # ------------------------------------------------------------------ images
    def _log(self, action, **details):
        self.history.append({"time": datetime.datetime.now().isoformat(timespec="seconds"), "action": action,
                             "details": details})

    def add_image(self, path, params=None, key=None, run=True):
        """Add an image. run=False only reads the picture (the analysis waits for calculate())."""
        key = key or os.path.splitext(os.path.basename(str(path)))[0]
        while key in self.images:
            key += "_2"
        p = {**DEFAULT_PARAMS, **(params or {})}
        if p.get("measure_widths") is None:      # live widths on small images; large ones only when asked
            from PIL import Image
            with Image.open(path) as im:
                p["measure_widths"] = im.width * im.height <= LIVE_WIDTHS_MAX_PX
        self.images[key] = {"path": str(path), "params": p, "tf": dict(transform.TF_DEFAULT), "result": None,
                            "picture": picture_of(path, p["input_mode"]), "stale": True}
        self.reference = self.reference or key
        if run:
            self._run(key)
        self._log("add_image", key=key, path=str(path))
        return key

    def _run(self, key):
        im = self.images[key]
        res = preview_polygon(im["path"], im["params"])
        if im.get("picture") is None:
            im["picture"] = res["picture"]
        res["picture"] = im["picture"]                   # one copy of the file's colours per image
        layers = res.pop("layers")
        m = layers["asymmetry"] if "asymmetry" in layers else asymmetry_field(layers)[0]   # same formula
        # keep per image only what the views and the comparison use, compactly: the file's colours (uint8),
        # the three states (int8), the area (bool) and the continuous asymmetry (float32, ~1e-7 rounding);
        # the full-precision working layers are dropped (a large image would otherwise hold ~2 GB)
        im["m"] = np.asarray(m, np.float32)
        res["img"] = None
        im["result"] = res
        im["stale"] = False
        self._invalidate(key)

    def calculate(self, key):
        """Run the domain analysis of `key` with its current parameters."""
        self._run(key)
        self._log("calculate", key=key)

    def is_ready(self, key):
        """Calculated with the current parameters."""
        im = self.images.get(key)
        return bool(im) and im["result"] is not None and not im.get("stale")

    def size(self, key):
        """(H, W) of the image's pixel grid."""
        return tuple(self.images[key]["picture"].shape[:2])

    def set_params(self, key, run=True, **params):
        """Change parameters; run=False only stores them (the result is then out of date until calculate())."""
        old = self.images[key]["params"]
        changed = any(old.get(k) != v for k, v in params.items())
        old.update(params)
        if run:
            self._run(key)
        elif changed:
            self.images[key]["stale"] = True
        self._log("set_params", key=key, **{k: v for k, v in params.items() if k not in ("polygon", "bg_polygon")})

    def copy_regions(self, src, dst, run=True):
        """Give image `dst` the active area and background of image `src` (same pixel positions; scaled by the
        size ratio if the sizes differ by more than 1 %, e.g. the same field of view saved at another size) and
        re-run its analysis. -> True if scaled."""
        ps, pd = self.images[src]["params"], self.images[dst]["params"]
        hs, ws_ = self.size(src)
        hd, wd = self.size(dst)
        sx, sy = wd / ws_, hd / hs
        scaled = abs(sx - 1) > 0.01 or abs(sy - 1) > 0.01   # a pixel or two of difference: same positions
        new = {}
        for name in ("polygon", "bg_polygon"):
            v = ps.get(name)
            new[name] = [((x + 0.5) * sx - 0.5, (y + 0.5) * sy - 0.5) for x, y in v] if (v and scaled) else \
                (list(v) if v else None)
        pd.update(new)
        if run:
            self._run(dst)
        else:
            self.images[dst]["stale"] = True
        self._log("copy_regions", src=src, dst=dst, scaled=scaled)
        return scaled

    SETTINGS_EXCLUDE = ("polygon", "bg_polygon", "um_per_px")    # per image: outlines and the derived pixel size

    def settings(self, key):
        """The analysis settings of `key` (no outlines, no derived pixel size)."""
        return {k: v for k, v in self.images[key]["params"].items() if k not in self.SETTINGS_EXCLUDE}

    def _apply_settings(self, dst, settings):
        p = self.images[dst]["params"]
        old = dict(p)
        p.update({k: v for k, v in settings.items() if k not in self.SETTINGS_EXCLUDE})
        p["um_per_px"] = calibration(p, self.size(dst))[0]      # recomputed for this image's size and outline
        if p != old:
            self.images[dst]["stale"] = True

    def copy_settings(self, src, dst):
        """Give `dst` the settings of `src` (threshold, denoising, calibration, invert, ...); outlines unchanged.
        The result of `dst` is then out of date until calculate()."""
        self._apply_settings(dst, self.settings(src))
        self._log("copy_settings", src=src, dst=dst)

    def remove(self, key):
        self.images.pop(key)
        self._invalidate(key)
        if self.reference == key:
            self.reference = next(iter(self.images), None)

    def _invalidate(self, key):
        self.compares = {k: v for k, v in self.compares.items() if key not in k}

    # ------------------------------------------------------------------ common grid
    @property
    def shape(self):
        return self.size(self.reference) if self.reference else None

    def raw(self, key, what):
        res, im = self.images[key]["result"], self.images[key]
        if res is None:
            raise RuntimeError(f"{key} is not calculated yet: press Calculate (Domains, step 5)")
        if what == "picture":
            return im["picture"]
        return {"mz": res["mz"], "roi": res["roi"], "m": im["m"], "picture": res["picture"]}[what]

    def final_view(self, key, what, step):
        """Like final(), for drawing only: every `step`-th pixel of the common grid (a screen shows ~1000 px
        across; the analysis always uses final())."""
        if step <= 1:
            return self.final(key, what)
        a, tf = self.raw(key, what), self.images[key]["tf"]
        if what == "picture":
            ch = [transform.warp_view(np.asarray(a[..., i], np.int16) + 1, tf, self.view_rot, 0, self.shape, step)
                  for i in range(3)]
            return np.where(np.dstack(ch) == 0, 255, np.dstack(ch) - 1).astype(np.uint8)
        cat = what in ("mz", "roi")
        out = transform.warp_view(a.astype(np.int8) if cat else a, tf, self.view_rot, 0 if cat else 1, self.shape,
                                  step)
        return out.astype(bool) if what == "roi" else out

    def final_channel(self, key, c):
        """One uint8 colour channel on the common grid, nearest pixel only (no new colours); 255 outside."""
        a = np.asarray(c, np.int16) + 1                      # 0 marks "outside"
        if a.shape != self.shape:
            z = (self.shape[0] / a.shape[0], self.shape[1] / a.shape[1])
            a = ndimage.zoom(a, z, order=0, grid_mode=True, mode="nearest")[:self.shape[0], :self.shape[1]]
        out = transform.warp(a, self.images[key]["tf"], self.view_rot, order=0, out_shape=self.shape)
        return np.where(out == 0, 255, out - 1)

    def final(self, key, what):
        """Layer of `key` on the common grid (resampled if its pixel grid differs, then transformed)."""
        a = self.raw(key, what)
        if what == "picture":      # the file's colours, moved like the image: nearest pixel, white outside
            ch = [self.final_channel(key, a[..., i]) for i in range(3)]
            return np.dstack(ch).astype(np.uint8)
        cat = what in ("mz", "roi")
        if a.shape != self.shape:
            z = (self.shape[0] / a.shape[0], self.shape[1] / a.shape[1])
            a = ndimage.zoom(np.asarray(a, float if not cat else np.int8), z, order=0 if cat else 1, grid_mode=True,
                             mode="nearest")[:self.shape[0], :self.shape[1]]
        out = transform.warp(a.astype(np.int8) if cat else a, self.images[key]["tf"], self.view_rot,
                             order=0 if cat else 1, out_shape=self.shape)
        return out.astype(bool) if what == "roi" else out

    @property
    def centre(self):
        return transform.centre(self.shape)

    # ------------------------------------------------------------------ alignment
    def set_transform(self, key, **tf):
        self.images[key]["tf"].update(tf)
        self._invalidate(key)

    def set_view_rotation(self, deg):
        delta = deg - self.view_rot
        self.view_rot = float(deg)
        if self.compare_polygon and abs(delta) > 1e-9:
            self.compare_polygon = transform.rotate_xy(self.compare_polygon, delta, self.centre)
        self.compares = {}

    def auto_align(self, moving, ref, rot_range=3.0, progress=None):
        """Rotation (+- rot_range) and shift of `moving` onto `ref`, from their asymmetry maps. A first guess."""
        tf0 = dict(self.images[moving]["tf"])
        A = self.final(ref, "m")

        def moving_at(ang):
            self.images[moving]["tf"] = {**tf0, "rot": tf0["rot"] + ang}
            return self.final(moving, "m")

        ang, peak = registration.best_rotation(A, moving_at, rot_range, progress=progress)
        self.images[moving]["tf"] = {**tf0, "rot": tf0["rot"] + ang}
        sh = registration.best_shift(A, self.final(moving, "m"))
        self.images[moving]["tf"] = transform.compose(self.images[moving]["tf"], self.view_rot, 0.0, 1.0, sh)
        self._invalidate(moving)
        self._log("auto_align", moving=moving, ref=ref, rotation=tf0["rot"] + ang, shift=list(map(float, sh)))
        return tf0["rot"] + ang, sh, peak

    def fit_points(self, moving, pts_moving, pts_target, allow_scale=False):
        """Move `moving` so that its points (final frame, (row, col)) land on the target points."""
        th, s, t, rms = transform.fit_similarity(pts_moving, pts_target, self.centre, allow_scale)
        self.images[moving]["tf"] = transform.compose(self.images[moving]["tf"], self.view_rot, th, s, t)
        self._invalidate(moving)
        self._log("fit_points", moving=moving, n=len(pts_moving), rotation=th, scale=s, rms_px=rms)
        return th, s, t, rms

    def shift_check(self, a, b, max_shift=20, angles=(0.0,), progress=None):
        tf0 = dict(self.images[b]["tf"])

        def b_at(ang):
            self.images[b]["tf"] = {**tf0, "rot": tf0["rot"] + ang}
            return self.final(b, "m")

        try:
            return registration.shift_correlation(self.final(a, "m"), b_at, self.compare_roi(a, b), max_shift, angles,
                                                  progress=progress)
        finally:
            self.images[b]["tf"] = tf0

    # ------------------------------------------------------------------ comparison
    def compare_roi(self, a, b):
        roi = self.final(a, "roi") & self.final(b, "roi")
        if self.compare_polygon:
            roi &= polygon_to_mask(self.compare_polygon, self.shape)
        return roi

    def compare(self, a, b, **settings):
        st = {**COMPARE_DEFAULTS, **settings}
        sA, sB = self.final(a, "mz"), self.final(b, "mz")
        roi = self.compare_roi(a, b)
        um = self.images[a]["result"].get("um")
        layers, m = comparison.compare_states(sA, sB, self.final(a, "m"), self.final(b, "m"), roi, um,
                                              st["window_px"], st["block_um"], st["n_boot"])
        fa = populations.fractions(*populations.counts_from_states(sA, layers["roi"]))
        fb = populations.fractions(*populations.counts_from_states(sB, layers["roi"]))
        m["populations_common_roi"] = {"A": fa, "B": fb, "change": populations.change(fa, fb)}
        own = {k: populations.fractions(*populations.counts_from_states(self.raw(k, "mz"), self.raw(k, "roi")))
               for k in (a, b)}
        m["populations_own_roi"] = {"A": own[a], "B": own[b], "change": populations.change(own[a], own[b])}
        self.compares[(a, b)] = {"flip": layers["flip"], "r_local": layers["r_local"], "metrics": m,
                                 "settings": st, "layers": layers}
        self._log("compare", A=a, B=b, switched_pct=m["switched_pct"], r=m["r"])
        return self.compares[(a, b)]

    def population_table(self, keys=None, common=False):
        """Per image: fractions in its own ROI (or, common=True, in the ROI shared by all `keys`)."""
        keys = list(keys or self.images)
        if common:
            roi = np.logical_and.reduce([self.final(k, "roi") for k in keys])
            return {k: populations.fractions(*populations.counts_from_states(self.final(k, "mz"), roi)) for k in keys}
        return {k: populations.fractions(*populations.counts_from_states(self.raw(k, "mz"), self.raw(k, "roi")))
                for k in keys}

    # ------------------------------------------------------------------ session
    def save(self, path):
        imgs = {k: {"path": v["path"], "params": v["params"], "tf": v["tf"], "raw": v["m"]}
                for k, v in self.images.items()}
        self._log("save", path=str(path))
        session.save(path, imgs, self.compares, self.history, self.view_rot, self.reference)

    @classmethod
    def load(cls, path):
        d = session.load(path)
        ws = cls()
        ws.view_rot, ws.history = d["view_rot"], d["history"]
        for k, im in d["images"].items():       # re-run the analysis from the original files and parameters
            ws.images[k] = {"path": im["path"], "params": im["params"], "tf": im["tf"], "result": None}
            ws._run(k)
        ws.reference = d["reference"] if d["reference"] in ws.images else next(iter(ws.images), None)
        ws.compares = {k: v for k, v in d["compares"].items() if k[0] in ws.images and k[1] in ws.images}
        return ws
