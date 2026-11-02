"""Changes of the up / neutral / down populations between states, from pixel counts (no plotting, no GUI).

For each state  f_k = N_k / N  (k = up, 0, down; N = pixels in the region), <m_z> = f_up - f_down.
Between A and B  Delta f_k = f_k(B) - f_k(A)  (sum 0),  Delta<m_z> = Delta f_up - Delta f_down.

Minimal redistribution (the fewest pixels that must change class): the classes that shrink feed those that grow.
A pixel moving directly between up and down changes f_up - f_down by 2/N, one entering or leaving the neutral class
by 1/N, so Delta<m_z> = 2 (flow down->up - flow up->down) + (flows via neutral, signed).

Bounds on the fraction of the region that changed class (counts fix only the marginals):
  lower  1/2 sum_k |Delta f_k|                (maximal overlap of each class)
  upper  1 - sum_k max(0, f_k(A) + f_k(B) - 1) (minimal overlap)
Neutral-independent version: hat f_up = N_up / (N_up + N_down), hat m = (N_up - N_down) / (N_up + N_down).
These are net quantities: switching in opposite directions that cancels is invisible here (see comparison.py).
"""
import numpy as np

CLASSES = ("up", "neutral", "down")


def fractions(n_up, n_neutral, n_down):
    """Area fractions (0-1) and <m_z> from pixel counts."""
    n = n_up + n_neutral + n_down
    f = {"up": n_up / n, "neutral": n_neutral / n, "down": n_down / n}
    return {**f, "mz": f["up"] - f["down"], "n": n,
            "up_hat": n_up / max(n_up + n_down, 1), "mz_hat": (n_up - n_down) / max(n_up + n_down, 1)}


def counts_from_states(s, roi):
    """(n_up, n_neutral, n_down) of a three-state map inside the ROI."""
    return int(((s == 1) & roi).sum()), int(((s == 0) & roi).sum()), int(((s == -1) & roi).sum())


def minimal_flows(d):
    """Fewest class changes consistent with the changes d = {class: Delta f}. Returns {(from, to): fraction}."""
    shrink = {k: -v for k, v in d.items() if v < 0}
    grow = {k: v for k, v in d.items() if v > 0}
    flows = {}
    if len(shrink) == 1:
        (s, _), = shrink.items()
        flows = {(s, g): v for g, v in grow.items()}
    elif len(grow) == 1:
        (g, _), = grow.items()
        flows = {(s, g): v for s, v in shrink.items()}
    return flows


def change(fa, fb):
    """Everything between two states (fractions dicts from `fractions`); all values as fractions (0-1)."""
    d = {k: fb[k] - fa[k] for k in CLASSES}
    flows = minimal_flows(d)
    direct = flows.get(("down", "up"), 0.0) - flows.get(("up", "down"), 0.0)
    via_neutral = (flows.get(("neutral", "up"), 0.0) + flows.get(("down", "neutral"), 0.0)
                   - flows.get(("neutral", "down"), 0.0) - flows.get(("up", "neutral"), 0.0))
    return {"d_up": d["up"], "d_neutral": d["neutral"], "d_down": d["down"],
            "d_mz": fb["mz"] - fa["mz"], "d_mz_direct": 2 * direct, "d_mz_via_neutral": via_neutral,
            "flows": {f"{a}->{b}": v for (a, b), v in flows.items()},
            "changed_lower": 0.5 * sum(abs(v) for v in d.values()),
            "changed_upper": 1 - sum(max(0.0, fa[k] + fb[k] - 1) for k in CLASSES),
            "d_up_hat": fb["up_hat"] - fa["up_hat"], "d_mz_hat": fb["mz_hat"] - fa["mz_hat"],
            "relative_up": d["up"] / fa["up"] if fa["up"] else float("nan"),
            "relative_down": d["down"] / fa["down"] if fa["down"] else float("nan"),
            "relative_neutral": d["neutral"] / fa["neutral"] if fa["neutral"] else float("nan")}


def mz_matrix(states):
    """Delta<m_z> between every pair: states = {name: fractions dict}; returns (names, matrix[to, from] ... as
    matrix[i, j] = mz(j) - mz(i), i = from, j = to)."""
    names = list(states)
    mz = np.array([states[k]["mz"] for k in names])
    return names, mz[None, :] - mz[:, None]
