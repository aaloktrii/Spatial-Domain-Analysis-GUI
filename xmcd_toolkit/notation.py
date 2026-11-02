"""Names used in every output (single source of truth): screen, figures, reports, tables.

The two states are +ve (+1) and -ve (-1) contrast, with 0 the neutral band, for XMCD and XLD (XMLD + XNLD) maps
alike: the domains may be out-of-plane or in-plane, so what +ve / -ve mean physically is for the user to interpret.
The order parameter is <m> for XMCD and <v_N> (Néel vector) for XLD.

The code keeps the short internal names "up" / "down" (dictionary keys); they are translated where text leaves the
program: relabel() for words, column() for column names and JSON keys (pos / neg). using(kind) sets the dichroism
for a block of output (figures and reports written by the export). No plotting, no GUI."""
import contextlib
import contextvars
import re

DICHROISM = (("XMCD", "XMCD"), ("XLD", "XLD (XMLD + XNLD)"))
ORDER = {"XMCD": "m", "XLD": "v_N"}

_KIND = contextvars.ContextVar("dichroism", default="XMCD")
_STATES = ((re.compile(r"\bup\b"), "+ve"), (re.compile(r"\bdown\b"), "−ve"), (re.compile(r"\bdn\b"), "−ve"),
           (re.compile(r"\bUp\b"), "+ve"), (re.compile(r"\bDown\b"), "−ve"), (re.compile(r"\bUP\b"), "+VE"),
           (re.compile(r"\bDOWN\b"), "−VE"), (re.compile("↑"), "₊"), (re.compile("↓"), "₋"))
_COLUMN = {"up": "pos", "down": "neg", "dn": "neg"}


@contextlib.contextmanager
def using(kind):
    """Dichroism (XMCD / XLD) of the output written inside this block."""
    token = _KIND.set(kind or "XMCD")
    try:
        yield
    finally:
        _KIND.reset(token)


def relabel(text, kind=None):
    """Words: up / down -> +ve / -ve, m_z -> m (XMCD) or v_N (XLD). kind None: the current using() block."""
    if not text:
        return text
    for pat, new in _STATES:
        text = pat.sub(new, text)
    return text.replace("m_z", ORDER.get(kind or _KIND.get(), "m"))


def column(name):
    """Column name / JSON key: the tokens up, down, dn -> pos, neg (e.g. frac_up -> frac_pos)."""
    return "_".join(_COLUMN.get(t, t) for t in str(name).split("_"))


def keys(obj):
    """column() applied to every dictionary key, recursively (for JSON output)."""
    if isinstance(obj, dict):
        return {column(k) if isinstance(k, str) else k: keys(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(keys(v) for v in obj)
    return obj
