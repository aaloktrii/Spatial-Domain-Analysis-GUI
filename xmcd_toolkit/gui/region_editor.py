"""Draw / edit one polygon on a matplotlib axis (GUI layer). Used for the active area and the background region
(Domains) and for an optional comparison region (Compare).

Tools: polygon (click corners, close on the first), rectangle, ellipse and regular polygon (drag a box). Every
shape becomes an editable polygon: drag a corner, SHIFT + drag to move it all, or call move() / rotate().
Polygon files are JSON with "vertices" [[x, y], ...]."""
import json

import numpy as np
from matplotlib.widgets import PolygonSelector, RectangleSelector

EDIT_COLOUR = "#ff8c00"
TOOLS = ("polygon", "rectangle", "ellipse", "regular polygon")


def shape_from_box(tool, x0, y0, x1, y1, sides=6):
    """Vertices of a rectangle / ellipse / regular polygon inscribed in the dragged box."""
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    if tool == "rectangle":
        return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    n = 48 if tool == "ellipse" else max(int(sides), 3)
    t = np.linspace(0, 2 * np.pi, n, endpoint=False) - (np.pi / 2 if tool != "ellipse" else 0)
    cx, cy, rx, ry = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2, (y1 - y0) / 2
    return [(float(cx + rx * np.cos(a)), float(cy + ry * np.sin(a))) for a in t]


def move(verts, dx, dy):
    return [(x + dx, y + dy) for x, y in verts]


def rotate(verts, deg):
    """About the centroid; + = counter-clockwise on screen (y points down)."""
    v = np.asarray(verts, float)
    c = v.mean(0)
    a = np.radians(deg)
    R = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
    return [tuple(map(float, p)) for p in (v - c) @ R.T + c]


def save_json(path, verts, **extra):
    with open(path, "w") as f:
        json.dump({"vertices": [list(map(float, p)) for p in verts], **extra}, f, indent=2)


def load_json(path):
    with open(path) as f:
        d = json.load(f)
    return [(float(x), float(y)) for x, y in d["vertices"]], d


class RegionEditor:
    """One editable shape on one axis. on_change(verts) is called when it is finished or edited."""

    def __init__(self, ax, on_change, tool="polygon", sides=6, verts=None):
        self.ax, self.on_change, self.tool, self.sides = ax, on_change, tool, sides
        self.verts = list(verts) if verts else None
        self.selector = None
        self._build()

    def _build(self):
        self.disconnect()
        props = dict(color=EDIT_COLOUR, lw=1.6)
        if self.verts or self.tool == "polygon":
            self.selector = PolygonSelector(self.ax, self._poly_done, useblit=True, props=props,
                                            handle_props=dict(markersize=6, markerfacecolor=EDIT_COLOUR))
            if self.verts:       # placed with blitting off: one ordinary (merged) redraw instead of a forced one
                blit = getattr(self.selector, "_useblit", None)       # matplotlib's internal switch
                if blit is not None:
                    self.selector._useblit = False
                self.selector.verts = self.verts
                if blit is not None:
                    self.selector._useblit = blit
        else:
            self.selector = RectangleSelector(self.ax, self._box_done, useblit=True,
                                              props=dict(edgecolor=EDIT_COLOUR, facecolor="none", lw=1.6))

    def _poly_done(self, verts):
        self.verts = [tuple(map(float, p)) for p in verts]
        self.on_change(self.verts)

    def _box_done(self, e, r):
        if abs(r.xdata - e.xdata) < 2 or abs(r.ydata - e.ydata) < 2:
            return
        self.verts = shape_from_box(self.tool, e.xdata, e.ydata, r.xdata, r.ydata, self.sides)
        self._build()                    # now an editable polygon
        self.ax.figure.canvas.draw_idle()
        self.on_change(self.verts)

    def set_verts(self, verts):
        self.verts = list(verts) if verts else None
        self._build()
        self.ax.figure.canvas.draw_idle()

    def disconnect(self):
        if self.selector is not None:
            try:
                self.selector.disconnect_events()
                self.selector.set_visible(False)
            except Exception:
                pass
            self.selector = None
