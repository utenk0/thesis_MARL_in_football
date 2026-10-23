# Trajectory CQL flow

## What is new and what remains legacy

The old `FootballMethod` CQL is retained for reproducibility. It uses d3rlpy,
predicts one of 19 GRF action labels, and is not part of the accepted
trajectory comparison.

The current CQL is `TrajectoryCQLMethod`. It obeys the same
`single_team_autonomous_trajectory_v1` contract as the accepted BC:

- focal team `DFL-CLU-00000P`;
- the same match-disjoint train, validation, and held-out test matches;
- the same current 61-feature `temporal_context_no_slot` observation;
- the same ten causal motion-history steps;
- direct predictions at steps `1, 5, 10, 25, 50, 100`;
- output shape `(players/origins, 6, 2)` in metres;
- no observation, action, velocity, or simulator state after the origin.

## Why CQL needs trajectory-plan actions

BC is continuous regression: its network directly predicts twelve correction
values. Discrete CQL instead needs a finite action set and learns a Q-value for
each action. The trajectory CQL therefore builds its action set from the
training demonstrations only:

1. Compute each demonstrated six-horizon trajectory.
2. Subtract the causal constant-velocity forecast.
3. Fit k-means prototypes to a bounded train-only sample of these residuals.
4. Reserve action `0` as an exact all-zero residual.

Each CQL action is consequently a complete residual motion plan with shape
`(6, 2)`. At inference, the chosen plan is added to constant velocity. The
plan is interpreted in receding-horizon form: a new plan is selected at every
recorded frame.

## Training transition and reward

For every focal player and forecast origin, the offline transition is:

```text
state       = current masked observation + causal history
action      = nearest train-only residual-trajectory prototype
reward      = focal-team ball progress plus recorded goal reward
next_state  = the same player's next causal state
done        = recorded termination or the current file boundary
```

The reward already exists in Dataset A's processed transition files. It is
positive for ball progress in the focal team's attacking direction, negative
for reverse progress, and includes goals. The eleven focal players share the
team reward, matching the shared-policy/team-objective assumption.

## Network and loss

One shared MLP maps every player's causal state to one Q-value per trajectory
plan. The final layer is zero-initialized. All actions therefore tie at epoch
zero, `argmax` selects action `0`, and the complete epoch-zero policy is
exactly constant velocity.

For a demonstrated action `a`, CQL minimizes:

```text
Huber(Q(s,a), r + gamma * Q_target(s', argmax Q(s',.)))
+ alpha * (logsumexp(Q(s,.)) - Q(s,a))
```

The first term is the Bellman temporal-difference loss. The second is the
conservative penalty: it pushes down unsupported actions relative to actions
seen in Dataset A. This is neither cross-entropy nor Euclidean regression.
Euclidean distance is used only to build the action prototypes and to measure
the final trajectory accuracy.

## Selection and final evaluation

Prototype fitting and Q-learning use train matches only. After each epoch,
validation ADE selects the checkpoint; epoch zero is eligible. Early stopping
uses validation only. The test paths are declared in the manifest but their
contents are not loaded during training.

The loaded checkpoint implements the shared `predict` and `evaluate` methods.
Evaluation reports the same policy, stationary, constant-velocity, and
constant-acceleration ADE/FDE and formation metrics as BC. Test evaluation
still requires `--allow-test` and must wait until three seeds and pretest
configuration are frozen.

## Commands

One local pretest seed is trained through the common interface with:

```bash
.venv/bin/python -m thesis_experiments.scripts.run_trajectory_method \
  --method cql \
  --manifest data/processed/final_stage_v1/team_00000P_manifest.json \
  --contract thesis_plan/trajectory_evaluation_contract_v1.json \
  --device cpu \
  train \
  --output-dir artifacts/trajectory_cql_pretest_v1/seed_0 \
  --seed 0
```

After the configuration is fixed, repeat it for seeds `0`, `1`, and `2`, then
aggregate and hash-freeze the validation runs:

```bash
.venv/bin/python -m thesis_experiments.scripts.summarize_trajectory_cql_pretest \
  --results \
    artifacts/trajectory_cql_pretest_v1/seed_0/trajectory_result.json \
    artifacts/trajectory_cql_pretest_v1/seed_1/trajectory_result.json \
    artifacts/trajectory_cql_pretest_v1/seed_2/trajectory_result.json \
  --manifest data/processed/final_stage_v1/team_00000P_manifest.json \
  --contract thesis_plan/trajectory_evaluation_contract_v1.json \
  --output-dir artifacts/trajectory_cql_pretest_v1
```

Only if that validation gate passes, verify the freeze and open the test split
once:

```bash
.venv/bin/python -m thesis_experiments.scripts.evaluate_trajectory_cql_test \
  --experiment-dir artifacts/trajectory_cql_pretest_v1 \
  --manifest data/processed/final_stage_v1/team_00000P_manifest.json \
  --contract thesis_plan/trajectory_evaluation_contract_v1.json
```

The Docker service uses the same code, read-only data, and external writable
results directory:

```bash
docker compose --env-file docker/server.env \
  -f docker/compose.yaml -f docker/compose.server.yaml \
  --profile trajectory-cql run --rm trajectory-cql-train
```

Its default output is
`artifacts/container_runs/trajectory_cql_pretest_v1/seed_0` on the host.
Set `CQL_SEEDS="0 1 2"` only after the CQL hyperparameters have been selected
on validation and are ready for the frozen three-seed pretest.
