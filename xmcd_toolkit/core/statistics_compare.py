"""Bootstrap difference of the magnetization statistics of two images (used by sda-compare)."""
import numpy as np


def difference_bootstrap(res_b, res_a, seed=0):
    """95% CI for d<m_z> from the difference of the two independent
    block-bootstrap distributions."""
    rng = np.random.default_rng(seed)
    bb = res_b["_field"][4]
    ba = res_a["_field"][4]
    n = min(len(bb), len(ba))
    if n < 2:
        d = res_a["mean_mz"] - res_b["mean_mz"]
        return d, d, np.array([d])
    diff = rng.choice(ba, 4000) - rng.choice(bb, 4000)
    lo, hi = np.percentile(diff, [2.5, 97.5])
    return float(lo), float(hi), diff
