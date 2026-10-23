#!/usr/bin/env bash
# Install the two isolated Python environments required by the training suite.
set -euo pipefail

PROJECT_DIR="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

cd "$PROJECT_DIR"

if ! command -v sudo >/dev/null 2>&1; then
  echo "sudo is required to install GRF system packages." >&2
  exit 1
fi

sudo apt-get update
sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y \
  git cmake build-essential pkg-config python3-dev python3-venv \
  libgl1-mesa-dev libsdl2-dev libsdl2-image-dev libsdl2-ttf-dev \
  libsdl2-gfx-dev libboost-all-dev libdirectfb-dev libst-dev \
  mesa-utils xvfb x11vnc rsync tmux

"$PYTHON_BIN" - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit(f"Python 3.10+ is required, found {sys.version}")
PY

if [[ ! -d .venv-methods ]]; then
  "$PYTHON_BIN" -m venv .venv-methods
fi
.venv-methods/bin/python -m pip install --upgrade pip wheel setuptools
.venv-methods/bin/python -m pip install -r requirements-project.txt -r requirements-methods.txt

if [[ ! -d .venv-grf ]]; then
  "$PYTHON_BIN" -m venv .venv-grf
fi
.venv-grf/bin/python -m pip install pip==24.0 setuptools==65.5.0 wheel==0.38.4
.venv-grf/bin/python -m pip install -r requirements-grf.txt -r football/requirements.txt -r requirements-project.txt
GFOOTBALL_PYTHON="$PROJECT_DIR/.venv-grf/bin/python" \
  .venv-grf/bin/python -m pip install -e football --no-deps

export SDL_VIDEODRIVER=dummy
export SDL_AUDIODRIVER=dummy
.venv-grf/bin/python -m thesis_experiments.scripts.check_environment --target grf
.venv-methods/bin/python -m thesis_experiments.scripts.check_environment --target methods
.venv-grf/bin/python main.py --steps 2
MPLCONFIGDIR=/tmp/thesis-matplotlib .venv-methods/bin/python -m pytest thesis_experiments/tests -q

mkdir -p artifacts/vm_runs
.venv-methods/bin/python -m pip freeze > artifacts/vm_runs/environment-methods.txt
.venv-grf/bin/python -m pip freeze > artifacts/vm_runs/environment-grf.txt

echo "VM setup and verification completed in $PROJECT_DIR"
