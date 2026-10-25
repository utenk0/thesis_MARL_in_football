# Google Research Football local setup

This project uses Homebrew Python 3.11 and a project virtual environment:

```bash
source .venv/bin/activate
python --version
python main.py --steps 10
```

Expected Python version: `3.11.15`.

GRF is installed from the official source checkout in `football/`. The source
contains small macOS compatibility changes for CMake 4, current Homebrew SDL2,
Python interpreter selection, Boost 1.90, and bundled Boost.Python libraries.

Install the Python dependencies and local package with:

```bash
.venv/bin/python -m pip install -r requirements-grf.txt
.venv/bin/python -m pip install -r football/requirements.txt
.venv/bin/python -m pip install -r requirements-project.txt
GFOOTBALL_USE_PREBUILT_SO=1 .venv/bin/python -m pip install -e football --no-deps
```

The prebuilt flag reuses the local macOS arm64 Python 3.11 engine. This avoids
linking against Homebrew's current `boost-python3`, which may target a newer
Python version. Rebuild the engine only when a Python 3.11-compatible
Boost.Python library is available.

The legacy Python packaging tools must remain pinned:

```text
pip==24.0
setuptools==65.5.0
wheel==0.38.4
gym==0.21.0
six==1.17.0
```

Do not upgrade pip in this environment: newer pip versions reject Gym 0.21's
package metadata.

The first simulation run may print warnings about SDL2 being loaded by both
GRF and OpenCV. The headless environment still runs successfully.

## Verified setup

Stage 1 was verified on 2026-08-31 with:

```text
Python 3.11.15
pip 24.0
setuptools 65.5.0
wheel 0.38.4
gfootball 2.10.3 (editable local install)
```

The required Homebrew packages (`cmake`, `sdl2`, `sdl2_image`, `sdl2_ttf`,
`sdl2_gfx`, `boost`, and `boost-python3`) were present. Running
`.venv/bin/python main.py --steps 10` reset the environment with a
`simple115v2` observation of shape `(115,)`, executed all ten actions, and
finished without an error.

Official API documentation:
https://github.com/google-research/football/blob/master/gfootball/doc/api.md
