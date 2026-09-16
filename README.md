# Dataset-to-GRF conversion project

The thesis uses exactly two football datasets: **Dataset A (BASSEK)** and
**Metrica Sports sample data**. StatsBomb and other datasets are out of scope.

## Quick start

Google Research Football is vendored in `football/`. Follow
[`GRF_SETUP.md`](GRF_SETUP.md) for the two-environment setup. Use
`.venv-grf` for live GRF simulation and `.venv-methods` for offline method
training plus the Gymnasium-compatible replay backends.

Run the headless GRF smoke test from the repository root with:

```bash
.venv-grf/bin/python -m thesis_experiments.scripts.check_environment --target grf
.venv-grf/bin/python main.py --steps 10
```

Put source datasets under `data/raw/` and generated, normalized datasets under
`data/processed/`. Experiment outputs are written under `artifacts/`; both data
locations are ignored except for their placeholder files. The staged project
plan and repository inventory live under `thesis_plan/`.

The current raw inputs are `data/raw/dataset_a/datasetA.zip` and
`data/raw/metrica_sample/`. Dataset provenance, licensing, citation, formats,
and coordinate conventions are recorded in
[`thesis_plan/DATASET_NOTES.md`](thesis_plan/DATASET_NOTES.md).

The validated data layer does:

```text
Metrica or Dataset A (BASSEK) tracking/events
  -> convert all 22 players and the ball to GRF coordinates
  -> infer one GRF action per player per transition
  -> validate 100 source frames
  -> render the joint action replay in GRF
```

The focused learning layer now continues with:

```text
unified 22-player transitions
  -> Gymnasium replay environment for env-based backends
  -> common method registry
  -> BC / GAIL / PPO / CQL / DT placeholder / Ghosting / CTDE
  -> comparable result JSON and checkpoints under artifacts/methods/
```

Environment split:

```text
.venv-grf       vendored Google Research Football, gym 0.21
.venv-methods   Gymnasium/SB3/imitation/d3rlpy method framework
```

The split is required because `d3rlpy` depends on `gym>=0.26`, while GRF
expects `gym<=0.21`.

## Project-owned code

```text
thesis_experiments/
├── datasets/
│   ├── coordinates.py       # Metrica/Dataset A <-> GRF coordinates
│   ├── action_mapping.py    # source events -> GRF Discrete(19)
│   └── movement.py          # position changes -> directional actions
├── data/
│   ├── metrica_grf_audit.py
│   └── dataset_a_grf_audit.py
├── scripts/
│   ├── audit_metrica_grf_100.py
│   └── audit_dataset_a_grf_100.py
├── methods/                 # unified BC/GAIL/PPO/CQL/DT/Ghosting/CTDE adapters
├── tests/                   # coordinate, transition, and method contract checks
└── env.py                   # locate the local GRF dependency
```

Learning-specific code is intentionally layered:

```text
transitions/    unified local/global state contract and builder
policies/       shared actor and centralized critic
simulation/     Gymnasium replay env plus GRF validation utilities
training/       BC, offline CTDE, and ghosting PyTorch helpers
methods/        method interface, registry, and backend adapters
scripts/        build_transitions.py, run_method.py, train_bc.py, train_ctde.py
```

Dataset A extraction helpers also live in `thesis_experiments/scripts/`. Run
the complete batch pipeline as a module:

```bash
python -m thesis_experiments.scripts.build_dataset_a_batch --help
```

Audit every raw match and regenerate the Stage 3 report with:

```bash
python -m thesis_experiments.scripts.audit_datasets
```

The report is written to `artifacts/reports/dataset_audit.md`.

Build provider-independent Stage 4 frame JSONL with:

```bash
python -m thesis_experiments.scripts.build_unified_data \
  --dataset metrica \
  --input data/raw/metrica_sample/data/Sample_Game_1 \
  --output data/processed/metrica_game_1_unified.jsonl
```

For Dataset A, first run `build_dataset_a_batch`, then pass one generated
`data/processed/dataset_a/frames_jsonl/*.jsonl` file to `build_unified_data`
with `--dataset dataset_a`.

Convert provider frames directly into GRF world coordinates with:

```bash
python -m thesis_experiments.scripts.convert_frames_to_grf \
  --dataset metrica \
  --input data/raw/metrica_sample/data/Sample_Game_1 \
  --output data/processed/metrica_game_1_grf.jsonl
```

Coordinate formulas and team mirroring are documented in
[`thesis_plan/COORDINATE_SYSTEMS.md`](thesis_plan/COORDINATE_SYSTEMS.md).

Open a live keyboard-controlled GRF match without recording:

```bash
.venv-grf/bin/python -m thesis_experiments.scripts.play_grf
```

Provider events are normalized by `events/schema.py` and
`events/converters.py`. Export them with:

```bash
python -m thesis_experiments.scripts.convert_events \
  --dataset metrica \
  --input data/raw/metrica_sample/data/Sample_Game_1 \
  --output data/processed/metrica_events.jsonl
```

Every JSONL row uses one vocabulary and preserves its original event in the
`raw` field for traceability.

## Closed-loop trajectory validation

This controller observes GRF positions and repeatedly corrects all 22 players
toward the next converted ground-truth frame. It is a validation baseline with
future target access, not an autonomous policy.

```bash
.venv-grf/bin/python -m thesis_experiments.scripts.track_trajectory \
  --dataset metrica \
  --input data/raw/metrica_sample/data/Sample_Game_1 \
  --output-dir artifacts/metrica_closed_loop \
  --start-frame 1 --frames 100 --live --record --fps 10
```

Other top-level folders:

```text
data/raw/       local source datasets
artifacts/      retained validation reports, arrays and videos
football/       vendored Google Research Football dependency
```

## Coordinate systems

```text
Metrica:  x,y in [0,1], origin at top-left
Dataset A: centred metres, x in [-52.5,52.5], y in [-34,34]
GRF:       centred, x in [-1,1], y in about [-0.42,0.42]
```

## Run Dataset A

```bash
.venv-grf/bin/python -m thesis_experiments.scripts.audit_dataset_a_grf_100 \
  --input /Users/liza/PycharmProjects/PythonProject/outputs/frames_jsonl/dataset_a_frames_DFL-MAT-J03WN1.jsonl \
  --output-dir artifacts/dataset_a_grf_live \
  --start-frame 10000 --frames 100 \
  --simulate --live --fps 10
```

## Run Metrica

```bash
.venv-grf/bin/python -m thesis_experiments.scripts.audit_metrica_grf_100 \
  --game-dir data/raw/metrica_sample/data/Sample_Game_1 \
  --output-dir artifacts/metrica_grf_live \
  --start-frame 1 --frames 100 \
  --simulate --live --fps 10
```

## Verify coordinates

```bash
.venv-methods/bin/python -m thesis_experiments.scripts.check_environment --target methods
.venv-methods/bin/python -m pytest thesis_experiments/tests -q
```

## Unified method commands

Build Dataset A and Metrica transitions with `build_transitions.py`, then run:

```bash
.venv-methods/bin/python -m thesis_experiments.scripts.run_method \
  --method bc \
  --data data/processed/dataset_a_ctde_100.npz data/processed/metrica_ctde_100.npz \
  --output-dir artifacts/methods \
  --backend torch

.venv-methods/bin/python -m thesis_experiments.scripts.run_method \
  --method ghosting \
  --data data/processed/dataset_a_ctde_100.npz data/processed/metrica_ctde_100.npz \
  --output-dir artifacts/methods

.venv-methods/bin/python -m thesis_experiments.scripts.run_method \
  --method ctde \
  --data data/processed/dataset_a_ctde_100.npz data/processed/metrica_ctde_100.npz \
  --bc-checkpoint artifacts/methods/bc/shared_actor_bc.pt \
  --output-dir artifacts/methods
```

The registry currently contains seven entries:

```text
bc, gail, ppo, cql, dt, ghosting, ctde
```

Method backend environment use:

```text
BC       uses imitation with Gymnasium spaces and offline demonstrations
GAIL     uses imitation + stable-baselines3 PPO on FootballReplayEnv
PPO      uses stable-baselines3 PPO on FootballReplayEnv
CQL      uses d3rlpy offline data; d3rlpy depends on gym>=0.26 internally
DT       placeholder manifest, no simulator backend yet
Ghosting uses project PyTorch code
CTDE     uses project PyTorch code
```

`bc`, `ghosting`, and `ctde` can use the project-owned PyTorch code. `gail`,
`ppo`, and `cql` require the optional method stack:

```bash
.venv-methods/bin/python -m pip install -r requirements-methods.txt
```

`dt` writes a placeholder manifest until the external Decision Transformer
backend is selected.

Stage 9 smoke results for the two 100-frame transition files are summarized in
[`thesis_plan/METHOD_SMOKE_RESULTS.md`](thesis_plan/METHOD_SMOKE_RESULTS.md).
The final Stage 9 environment conclusion is recorded in
[`thesis_plan/STAGE_9_CONCLUSION.md`](thesis_plan/STAGE_9_CONCLUSION.md).
The optional method stack installs `gym>=0.26` through d3rlpy, which conflicts
with the vendored GRF package's `gym<=0.21` expectation. Keep GRF replay in
`.venv-grf` and method training in `.venv-methods`.
