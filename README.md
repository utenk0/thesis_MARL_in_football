# Dataset-to-GRF conversion project

The validated data layer does:

```text
Metrica or Dataset A tracking/events
  -> convert all 22 players and the ball to GRF coordinates
  -> infer one GRF action per player per transition
  -> validate 100 source frames
  -> render the joint action replay in GRF
```

The focused learning layer now continues with:

```text
unified 22-player transitions
  -> shared-actor behavioural cloning
  -> centralized two-team value critic
  -> critic-weighted actor fine-tuning with a BC anchor
```

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
├── tests/                   # coordinate and CTDE contract checks
└── env.py                   # locate the local GRF dependency
```

CTDE-specific code is intentionally small:

```text
transitions/    unified local/global state contract and builder
policies/       shared actor and centralized critic
training/       BC and offline CTDE optimization
scripts/        build_transitions.py, train_bc.py, train_ctde.py
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
python -m thesis_experiments.scripts.track_trajectory \
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
source .venv/bin/activate
python -m thesis_experiments.scripts.audit_dataset_a_grf_100 \
  --input /Users/liza/PycharmProjects/PythonProject/outputs/frames_jsonl/dataset_a_frames_DFL-MAT-J03WN1.jsonl \
  --output-dir artifacts/dataset_a_grf_live \
  --start-frame 10000 --frames 100 \
  --simulate --live --fps 10
```

## Run Metrica

```bash
source .venv/bin/activate
python -m thesis_experiments.scripts.audit_metrica_grf_100 \
  --game-dir data/raw/metrica_sample/data/Sample_Game_1 \
  --output-dir artifacts/metrica_grf_live \
  --start-frame 1 --frames 100 \
  --simulate --live --fps 10
```

## Verify coordinates

```bash
source .venv/bin/activate
python -m pytest thesis_experiments/tests -q
```

## Focused CTDE commands

Build Dataset A and Metrica transitions with `build_transitions.py`, then run:

```bash
python -m thesis_experiments.scripts.train_bc \
  --data data/processed/dataset_a_ctde_100.npz data/processed/metrica_ctde_100.npz \
  --output artifacts/ctde/shared_actor_bc.pt

python -m thesis_experiments.scripts.train_ctde \
  --data data/processed/dataset_a_ctde_100.npz data/processed/metrica_ctde_100.npz \
  --bc-checkpoint artifacts/ctde/shared_actor_bc.pt \
  --output artifacts/ctde/shared_actor_ctde.pt
```

GAIL, CQL, PPO and unrelated experimental methods remain intentionally absent.
