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

Official API documentation:
https://github.com/google-research/football/blob/master/gfootball/doc/api.md
