# Environment Setup

This project needs two Python environments because Google Research Football
and the modern learning libraries depend on incompatible Gym generations.

```text
.venv-grf       GRF replay and live simulation
.venv-methods   Gymnasium method framework and offline training
```

Do not install `requirements-methods.txt` into the GRF environment. `d3rlpy`
depends on `gym>=0.26`, while vendored Google Research Football expects
`gym<=0.21`.

## GRF Environment

Use this environment for:

```text
main.py
audit_metrica_grf_100.py --simulate/--live
audit_dataset_a_grf_100.py --simulate/--live
track_trajectory.py
play_grf.py
```

Setup:

```bash
python3.11 -m venv .venv-grf
.venv-grf/bin/python -m pip install pip==24.0 setuptools==65.5.0 wheel==0.38.4
.venv-grf/bin/python -m pip install -r requirements-grf.txt
.venv-grf/bin/python -m pip install -r football/requirements.txt
.venv-grf/bin/python -m pip install -r requirements-project.txt
GFOOTBALL_USE_PREBUILT_SO=1 .venv-grf/bin/python -m pip install -e football --no-deps
```

Verify:

```bash
.venv-grf/bin/python -m thesis_experiments.scripts.check_environment --target grf
.venv-grf/bin/python main.py --steps 10
```

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

## Method Environment

Use this environment for:

```text
build_transitions.py
run_method.py
BC, GAIL, PPO, CQL, DT, Ghosting, CTDE
offline tests and method smoke runs
```

Setup:

```bash
python3.11 -m venv .venv-methods
.venv-methods/bin/python -m pip install -r requirements-project.txt
.venv-methods/bin/python -m pip install -r requirements-methods.txt
```

Verify:

```bash
.venv-methods/bin/python -m thesis_experiments.scripts.check_environment --target methods
.venv-methods/bin/python -m pytest thesis_experiments/tests -q
```

The method-facing simulation layer uses `FootballGRFEnv` (Gymnasium) and a
JSON-lines worker running under `.venv-grf`. PPO and GAIL now train in live GRF
with 22-player joint actions through a shared-policy vector wrapper. The GRF
package never imports into the method interpreter. The worker is closed when
training/evaluation finishes. `--grf-python` overrides the worker interpreter.

BC, CQL, DT, ghosting and CTDE train offline. All seven saved policy types can
be evaluated through the same live simulator. `FootballReplayEnv` is retained
for explicit `--env-backend replay` diagnostics; it cannot test action-driven
football dynamics. See [the training report](thesis_plan/TRAINING_STAGE_STATUS.md)
for commands and retained artifacts.

## Previously Verified GRF Setup

The split GRF setup was verified on 2026-09-15 with:

```text
.venv-grf
Python 3.11.9
pip 24.0
setuptools 65.5.0
wheel 0.38.4
gym 0.21.0
gfootball 2.10.3 (editable local install)
```

Running `.venv-grf/bin/python main.py --steps 1` reset the environment with a
`simple115v2` observation of shape `(115,)`, executed one action, and finished
without an error.

The original local GRF setup was verified on 2026-08-31 with:

```text
Python 3.11.15
pip 24.0
setuptools 65.5.0
wheel 0.38.4
gfootball 2.10.3 (editable local install)
```

The required Homebrew packages were present:

```text
cmake
sdl2
sdl2_image
sdl2_ttf
sdl2_gfx
boost
boost-python3
```

Official GRF API documentation:
https://github.com/google-research/football/blob/master/gfootball/doc/api.md
