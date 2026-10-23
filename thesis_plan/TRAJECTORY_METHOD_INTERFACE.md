# Shared trajectory-method interface

The action-oriented `FootballMethod` registry is retained for reproducibility.
New comparisons governed by `single_team_autonomous_trajectory_v1` use the
parallel `TrajectoryMethod` interface in
`thesis_experiments/methods/trajectory.py`.

## Information and output boundary

Each loaded method receives:

- one current 61-feature observation per focal player;
- only the causal motion-history vector available at the forecast origin.

It returns relative player positions in metres with shape
`(samples, 6, 2)` at steps `1, 5, 10, 25, 50, 100`.  No observation, action,
velocity, event, or simulator state after the forecast origin is permitted.

The shared surface is:

```text
train(dataset, output_dir, config)
load(checkpoint, device)
predict(current_observations, causal_history)
evaluate(paths, team_id)
```

`TrajectoryDatasetSplit` validates non-empty, match-disjoint splits and can
bind a manifest to the frozen trajectory contract.  A method's `train`
implementation receives train and validation paths through that object, but
must not load `test_paths`.  The common CLI additionally requires an explicit
`--allow-test` flag before test evaluation.

## Current registry

`bc` maps to `MultiHorizonBCMethod`, an adapter around the frozen accepted BC.
`cql` maps to `TrajectoryCQLMethod`, the conservative offline Q-learning
adapter over train-only trajectory-plan actions. `ctde` maps to
`TrajectoryCTDEMethod`: a BC-initialized local trajectory actor trained with a
centralized, training-only team-value critic. The BC adapter does not alter the
original training implementation or checkpoint format. Its real-data
validation result is byte-for-byte numerically equal to the existing evaluator
result.

The earlier 19-action CQL and CTDE remain in the action-oriented registry for
reproducibility but are not used by the trajectory contract. The new
formulations and losses are documented in `TRAJECTORY_CQL_EXPERIMENT.md` and
`TRAJECTORY_CTDE_EXPERIMENT.md`.

List or invoke registered methods through
`thesis_experiments.methods.trajectory_registry`.  For example, validation of
a saved BC checkpoint is:

```bash
.venv/bin/python -m thesis_experiments.scripts.run_trajectory_method \
  --method bc \
  --manifest data/processed/final_stage_v1/team_00000P_manifest.json \
  --contract thesis_plan/trajectory_evaluation_contract_v1.json \
  evaluate \
  --checkpoint artifacts/single_team_multi_horizon_bc_full/seed_0/single_team_multi_horizon_bc.pt \
  --split validation \
  --output artifacts/trajectory_interface_validation/seed_0.json
```

Every later method must join this registry and pass the same shape, causality,
split, reload, and evaluation tests before entering a full comparison.
