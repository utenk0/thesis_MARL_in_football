# Shared trajectory evaluation contract

Version: `single_team_autonomous_trajectory_v1`
Frozen: 17 October 2026

This contract is the required trajectory interface for BC and every subsequent
method. It prevents later methods from receiving easier inputs or being judged
on different horizons.

## Information boundary

At a forecast origin \(t\), a method may use the current
`temporal_context_no_slot` observation and ten causal history steps. It may not
use any observation, action, velocity, event, or simulator state after \(t\).

The focal team is `DFL-CLU-00000P`. The fixed match-disjoint split is:

- train: `DFL-MAT-J03WOH`, `DFL-MAT-J03WOY`, `DFL-MAT-J03WPY`;
- validation: `DFL-MAT-J03WQQ`;
- test: `DFL-MAT-J03WR9`.

Test is loaded once, only after the method's architecture, hyperparameters,
validation-selected checkpoints, and pretest file hashes are frozen.

## Outputs and metrics

Each method predicts the eleven focal-player positions relative to the
forecast origin at steps `1, 5, 10, 25, 50, 100`. Coordinates and errors are
in metres.

Primary metrics are mean position error over the six direct horizons (ADE)
and position error at step 100 (FDE). Secondary metrics at every horizon are
player position error, team centroid error, team-width MAE, and team-depth
MAE. Three seeds (`0, 1, 2`) are reported as mean and sample standard
deviation.

Every method is compared with stationary, constant velocity, and constant
acceleration computed only at the forecast origin. Teacher-forced baselines
that ingest later recorded velocities are excluded from this contract.

The validation gate requires mean ADE and FDE to beat constant velocity, at
least two seeds to improve both, and no seed to regress on either metric.

## Scope

This is an autonomous offline forecasting contract because no information
after the forecast origin is supplied. It is not a closed-loop GRF controller:
passing it supports a held-out team-trajectory forecasting claim, not a claim
that an entire interactive match has been recreated.

The machine-readable contract is
`thesis_plan/trajectory_evaluation_contract_v1.json`.
