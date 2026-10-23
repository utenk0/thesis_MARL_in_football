# Direct displacement BC experiment

Date: 17 October 2026

## Purpose

The frozen discrete-action BC imitates tracking-derived movement labels but
does not reconstruct held-out trajectories. This experiment introduces a
separate supervised target without changing or overwriting that baseline:

\[
f_\theta(o_t^i) = (\Delta x_t^i, \Delta y_t^i)
= (x_{t+1}^i-x_t^i, y_{t+1}^i-y_t^i).
\]

The target is expressed in metres. The model is shared across the eleven
players of team `DFL-CLU-00000P` and uses the same
`temporal_context_no_slot` observation mask as discrete BC: 31 of the stored
61 causal features. Previous-action and sticky-action features remain masked;
the causal previous player velocity is retained.

## Model and training contract

The regressor is a two-layer MLP:

\[
61 \rightarrow 128 \rightarrow 128 \rightarrow 2.
\]

Excluded inputs are zeroed by the stored feature mask. Training minimizes
Smooth L1 (Huber) loss between predicted and recorded next displacement.
Checkpoints are selected by minimum validation mean Euclidean displacement
error, never by test performance.

The match-disjoint contract is unchanged:

- train: `DFL-MAT-J03WOH`, `DFL-MAT-J03WOY`, `DFL-MAT-J03WPY`;
- validation: `DFL-MAT-J03WQQ`;
- test: `DFL-MAT-J03WR9`.

Three seeds use 4,650,514 focal-player training samples. Validation selected
epochs 8, 8, and 5.

## Metrics and baselines

The primary one-step metric is mean Euclidean displacement error in metres:

\[
\frac{1}{N}\sum_n \lVert \hat{\Delta p}_n-\Delta p_n \rVert_2.
\]

Two baselines are required:

1. stationary predicts zero displacement;
2. previous velocity predicts the last observed displacement again.

The 100-step cumulative diagnostic integrates per-step predictions. Each
prediction still receives its recorded observation at that future step, so
this is teacher-forced and is not an autonomous rollout.

## Result

| Split | Policy error (m) | Stationary (m) | Previous velocity (m) | Policy RMSE (m) |
|---|---:|---:|---:|---:|
| train | 0.0153 ± 0.0021 | 0.0748 | 0.0129 | 0.0871 ± 0.0004 |
| validation | 0.0144 ± 0.0013 | 0.0792 | 0.0126 | 0.0670 ± 0.0007 |
| test | 0.0156 ± 0.0036 | 0.0710 | 0.0125 | 0.0597 ± 0.0010 |

| Test 100-step predictor | ADE (m) | FDE (m) |
|---|---:|---:|
| direct displacement BC | 0.4107 ± 0.3442 | 0.7927 ± 0.6944 |
| stationary | 3.2989 | 6.1834 |
| previous velocity | 0.0575 | 0.0886 |

The model passes finite-output and stationary-baseline checks but fails the
predeclared validation gate against previous velocity. It also has high seed
variance under cumulative integration. The direct target is implemented and
learnable, but this version is not accepted as the final reconstruction model.

An important secondary observation is that the policy has lower RMSE than
previous velocity despite worse mean error. It reduces some large errors but
adds small error to many smooth frames. Future work must be chosen from
validation only and could model acceleration/residual displacement or predict
multiple future steps explicitly.

## Relationship to discrete-action BC

The discrete and continuous formulations answer different questions:

- `single_team_bc_full` is frozen as an inferred discrete-action baseline;
- `single_team_displacement_bc_full` is a separate reconstruction-target
  experiment;
- neither current result supports autonomous full-game reconstruction.

The final discrete-action diagnostic also shows why ordinary action accuracy
must not be used alone: all three seeds have zero recall for every
unambiguously event-derived test action (passes, shot, tackle, and dribble),
while movement-direction classes dominate the score.

## Reproduce

Run one seed:

```bash
.venv/bin/python -m thesis_experiments.scripts.run_team_displacement_bc \
  --manifest data/processed/final_stage_v1/team_00000P_manifest.json \
  --output-dir artifacts/single_team_displacement_bc_full/seed_0 \
  --epochs 10 --batch-size 2048 --hidden-size 128 --seed 0 \
  --torch-threads 6 --early-stopping-patience 3 \
  --cumulative-horizon 100
```

Aggregate the three locked seeds:

```bash
.venv/bin/python -m thesis_experiments.scripts.summarize_team_displacement_bc \
  --results \
    artifacts/single_team_displacement_bc_full/seed_0/result.json \
    artifacts/single_team_displacement_bc_full/seed_1/result.json \
    artifacts/single_team_displacement_bc_full/seed_2/result.json \
  --output-dir artifacts/single_team_displacement_bc_full
```

The durable generated result is
`artifacts/single_team_displacement_bc_full/DISPLACEMENT_BC_REPORT.md`.

## Residual displacement and autonomous multi-horizon follow-up

The direct model's validation failure triggered a new, separately versioned
residual experiment. It predicts a correction to previous velocity from the
current 31-feature observation and ten causal player/ball velocity steps. The
final layer is initialized to zero, making epoch 0 exactly equal to previous
velocity and eligible for selection.

Before any test data were loaded, all three residual seeds improved validation
mean displacement error: `0.008497 ± 0.000094 m` versus `0.012610 m`. Selected
epochs were 8, 9, and 3. The pretest configuration, source, checkpoints, and
validation outputs were then frozen with hashes.

Because the residual gate passed, a multi-horizon model was trained at steps
`1, 5, 10, 25, 50, 100`. It predicts corrections to constant velocity and
receives no observation after the forecast origin. Before test evaluation it
improved validation ADE from `1.3546 m` to `1.0716 ± 0.0079 m` and 100-step
FDE from `5.4746 m` to `4.2380 ± 0.0304 m`. All three seeds improved both
metrics, and their checkpoints were frozen at epochs 6, 9, and 10.

Only after verifying both pretest freezes was the test match loaded:

| Frozen test result | BC | Required baseline |
|---|---:|---:|
| Residual one-step error | 0.008612 ± 0.000311 m | previous velocity: 0.012538 m |
| Multi-horizon ADE | 0.9654 ± 0.0128 m | constant velocity: 1.1946 m |
| Multi-horizon 100-step FDE | 3.8036 ± 0.0493 m | constant velocity: 4.7974 m |

At step 100, the learned model's centroid error is `2.4505 ± 0.0684 m`,
width MAE is `2.9587 ± 0.0868 m`, and depth MAE is
`2.6464 ± 0.0243 m`.

This passes the autonomous held-out trajectory forecasting gate. It does not
yet prove closed-loop simulator control or complete interactive game
reconstruction. The frozen common contract for later methods is documented in
`thesis_plan/TRAJECTORY_EVALUATION_CONTRACT.md`.
