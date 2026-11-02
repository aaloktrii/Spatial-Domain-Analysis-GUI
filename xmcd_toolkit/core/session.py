"""Session file: the workspace as a hierarchical xarray DataTree (.nc), following the TensorSpec layout
(no plotting, no GUI):

    /                      attrs: format, view_rot, reference image, created
    /images/image_<i>/raw  image   the analysed image (float32), attrs: key, path, params (JSON), transform (JSON)
    /analysis/compare/pair_<i>   flip (int8), r_local (float32), attrs: A, B, metrics (JSON), settings (JSON)
    /history               attrs: entries (JSON list of {time, action, details})
Group names are generic because netCDF forbids many characters in them; the real names are attributes.
"""
import datetime
import json

import numpy as np
import xarray as xr

FORMAT = "xmcd_toolkit-session-1"


def _js(v):
    return json.dumps(v, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else float(o))


def to_datatree(images, compares, history, view_rot, reference):
    """images: {key: dict(path, params, tf, raw)}; compares: {(a, b): dict(flip, r_local, metrics, settings)}."""
    nodes = {"/": xr.Dataset(attrs={"format": FORMAT, "view_rot": float(view_rot), "reference": reference or "",
                                    "created": datetime.datetime.now().isoformat(timespec="seconds")})}
    for i, (k, im) in enumerate(images.items()):
        nodes[f"/images/image_{i}/raw"] = xr.Dataset(
            {"image": (("y", "x"), np.asarray(im["raw"], np.float32))},
            attrs={"key": k, "path": str(im["path"]), "params": _js(im["params"]), "transform": _js(im["tf"])})
    for i, ((a, b), c) in enumerate(compares.items()):
        nodes[f"/analysis/compare/pair_{i}"] = xr.Dataset(
            {"flip": (("y", "x"), np.nan_to_num(c["flip"], nan=-1).astype(np.int8)),
             "r_local": (("y", "x"), np.asarray(c["r_local"], np.float32))},
            attrs={"A": a, "B": b, "metrics": _js(c["metrics"]), "settings": _js(c.get("settings", {}))})
    nodes["/history"] = xr.Dataset(attrs={"entries": _js(history)})
    return xr.DataTree.from_dict(nodes)


def save(path, *args, **kwargs):
    to_datatree(*args, **kwargs).to_netcdf(path)


def load(path):
    """-> dict(images={key: dict(path, params, tf, raw)}, compares={...}, history=[...], view_rot, reference)."""
    t = xr.open_datatree(path)
    if t.attrs.get("format") != FORMAT:
        raise ValueError(f"{path} is not a {FORMAT} file")
    out = {"images": {}, "compares": {}, "history": json.loads(t["history"].attrs.get("entries", "[]")),
           "view_rot": float(t.attrs.get("view_rot", 0.0)), "reference": t.attrs.get("reference") or None}
    for _, node in (t["images"].children.items() if "images" in t.children else []):
        raw = node["raw"]
        out["images"][raw.attrs["key"]] = {"path": raw.attrs["path"], "params": json.loads(raw.attrs["params"]),
                            "tf": json.loads(raw.attrs["transform"]), "raw": raw["image"].values.astype(float)}
    cmp_root = t["analysis"]["compare"] if "analysis" in t.children and "compare" in t["analysis"].children else None
    for _, node in (cmp_root.children.items() if cmp_root is not None else []):
        flip = node["flip"].values.astype(float)
        flip[flip < 0] = np.nan
        out["compares"][(node.attrs["A"], node.attrs["B"])] = {
            "flip": flip, "r_local": node["r_local"].values.astype(float),
            "metrics": json.loads(node.attrs["metrics"]), "settings": json.loads(node.attrs["settings"])}
    t.close()
    return out
