"""State names in a finished figure (plotting layer): every text, legend and category tick label is shown in
+ve / -ve and the order parameter of the current dichroism (notation.relabel), just before the figure is saved."""
from ..notation import relabel


def relabel_figure(fig, kind=None):
    fig.canvas.draw()                                     # tick labels exist only after a draw
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            texts = [t.get_text() for t in axis.get_ticklabels()]
            new = [relabel(t, kind) for t in texts]
            if new != texts:                              # categorical ticks (e.g. P(up), P(0), P(dn))
                axis.set_ticks(axis.get_ticklocs(), labels=new)
    for t in fig.findobj(lambda a: hasattr(a, "get_text") and hasattr(a, "set_text")):
        s = t.get_text()
        if s:
            t.set_text(relabel(s, kind))
    return fig
