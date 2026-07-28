#!/usr/bin/env bash
# Create and populate a virtual environment for the XMCD domain toolkit.
#
#   bash setup_env.sh
#
# Then, in each new terminal:
#   source .venv/bin/activate
#
set -euo pipefail

# Resolve to the directory this script lives in, so it works from anywhere.
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"

echo ">> Using interpreter: $($PYTHON --version)"
echo ">> Creating virtual environment in ./$VENV_DIR"
"$PYTHON" -m venv "$VENV_DIR"

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo ">> Upgrading pip"
python -m pip install --upgrade pip

echo ">> Installing dependencies"
if [ -f requirements.txt ]; then
  pip install -r requirements.txt
else
  # Fallback if requirements.txt is missing. NOTE: the package is
  # 'scikit-image' (it imports as 'skimage') -- NOT 'skimage'.
  pip install "numpy>=1.24" "scipy>=1.10" "scikit-image>=0.21" \
              "matplotlib>=3.7" "pillow>=9.5"
fi

echo ">> Verifying imports"
python - <<'PY'
import numpy, scipy, skimage, matplotlib, PIL
print("  numpy        ", numpy.__version__)
print("  scipy        ", scipy.__version__)
print("  scikit-image ", skimage.__version__, "(imports as 'skimage')")
print("  matplotlib   ", matplotlib.__version__)
print("  pillow       ", PIL.__version__)
print("OK: all imports succeeded.")
PY

echo
echo ">> Done. To use the environment:"
echo "     source $VENV_DIR/bin/activate"
echo "     python examples/run_domain_example.py"
