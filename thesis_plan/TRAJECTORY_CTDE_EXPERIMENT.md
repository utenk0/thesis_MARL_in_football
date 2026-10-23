# Trajectory CTDE flow

## What CTDE means here

CTDE means **centralized training with decentralized execution**. The new
`TrajectoryCTDEMethod` uses more team-wide information to estimate which
demonstrated moments are useful during training, but its saved actor can run
from exactly the same local information as the accepted BC.

The older `FootballMethod` CTDE is retained for reproducibility. It predicts
one of 19 GRF action labels and is not part of the accepted trajectory
comparison. The current trajectory CTDE obeys
`single_team_autonomous_trajectory_v1`:

- focal team `DFL-CLU-00000P`;
- the same match-disjoint train, validation, and held-out test matches;
- the same current 61-feature `temporal_context_no_slot` actor observation;
- the same ten causal motion-history steps;
- direct displacement forecasts at steps `1, 5, 10, 25, 50, 100`;
- actor output shape `(players/origins, 6, 2)` in metres;
- no observation, action, velocity, or simulator state after the origin at
  execution.

## Actor and centralized critic

Each seed starts from the accepted, frozen trajectory-BC checkpoint with the
same seed. The shared actor therefore has the BC architecture and initially
predicts the same twelve continuous residual values: an `(x, y)` correction
to constant velocity at every one of the six horizons.

During training only, one centralized value critic receives:

```text
all 22 player positions + ball position + focal-team side = 47 values
```

It estimates the focal team's value. It does not select actions and is never
called by `predict` or by validation/test evaluation. At execution, every
focal player is controlled by the same actor and only that player's local
observation plus causal history enters the network.

## Reward and losses

The critic uses the Dataset A focal-team reward already stored in each
processed transition: ball progress in the focal team's attacking direction
plus the recorded goal reward. For current global state `s` and next global
state `s'`, it minimizes Bellman mean-squared error:

```text
target       = reward + gamma * (1 - done) * V_target(s')
critic loss  = mean((V(s) - target)^2)
advantage    = reward + gamma * (1 - done) * V(s') - V(s)
```

The advantage is converted to a bounded positive weight. The actor continues
to imitate the real six-horizon trajectories with Euclidean position error,
but gives more weight to demonstrations with higher centralized advantage:

```text
weight      = clip(exp(advantage / temperature), max=max_weight)
actor loss  = mean(weight * Euclidean trajectory error)
              + bc_coefficient * mean(Euclidean trajectory error)
```

The second term anchors the update to BC and prevents the team reward from
discarding ordinary football motion. Thus CTDE is not a cross-entropy model:
the actor is continuous trajectory regression, while the critic is scalar
value regression.

## Selection and held-out evaluation

The critic and actor use train matches only. Validation ADE selects the actor
checkpoint after every epoch. Epoch zero—the unmodified accepted BC—is also a
candidate, so unsuccessful CTDE fine-tuning falls back to the BC actor.
Neither the test file contents nor test metrics participate in selection.

After one fixed configuration has run for seeds `0`, `1`, and `2`, the
pretest summarizer verifies the split, actor information boundary, shared
hyperparameters, per-seed BC dependencies, and the validation gate. It then
hashes the implementation, configuration, BC initializations, CTDE
checkpoints, and reports. The held-out evaluator refuses to open test data if
that freeze changed or the validation gate failed.

The final report uses the same policy, stationary, constant-velocity, and
constant-acceleration ADE/FDE and formation metrics as BC and CQL. The critic
is absent from all reported forecasts.

## Commands

Train one local pretest seed through the shared interface:

```bash
.venv/bin/python -m thesis_experiments.scripts.run_trajectory_method \
  --method ctde \
  --manifest data/processed/final_stage_v1/team_00000P_manifest.json \
  --contract thesis_plan/trajectory_evaluation_contract_v1.json \
  --device cpu \
  train \
  --output-dir artifacts/trajectory_ctde_pretest_v1/seed_0 \
  --seed 0 \
  --ctde-bc-checkpoint artifacts/single_team_multi_horizon_bc_full/seed_0/single_team_multi_horizon_bc.pt
```

Once the hyperparameters are fixed, run seeds `0`, `1`, and `2` and freeze
the validation results:

```bash
.venv/bin/python -m thesis_experiments.scripts.summarize_trajectory_ctde_pretest \
  --results \
    artifacts/trajectory_ctde_pretest_v1/seed_0/trajectory_result.json \
    artifacts/trajectory_ctde_pretest_v1/seed_1/trajectory_result.json \
    artifacts/trajectory_ctde_pretest_v1/seed_2/trajectory_result.json \
  --manifest data/processed/final_stage_v1/team_00000P_manifest.json \
  --contract thesis_plan/trajectory_evaluation_contract_v1.json \
  --output-dir artifacts/trajectory_ctde_pretest_v1
```

Only after the reported validation gate passes, verify the freeze and open the
held-out test split once:

```bash
.venv/bin/python -m thesis_experiments.scripts.evaluate_trajectory_ctde_test \
  --experiment-dir artifacts/trajectory_ctde_pretest_v1 \
  --manifest data/processed/final_stage_v1/team_00000P_manifest.json \
  --contract thesis_plan/trajectory_evaluation_contract_v1.json
```

The Docker service mounts processed data and frozen BC checkpoints read-only
and writes CTDE artifacts to the external results directory:

```bash
docker compose --env-file docker/server.env \
  -f docker/compose.yaml -f docker/compose.server.yaml \
  --profile trajectory-ctde run --rm trajectory-ctde-train
```

Its default output is
`artifacts/container_runs/trajectory_ctde_pretest_v1/seed_0` on the host. Set
`CTDE_SEEDS="0 1 2"` only after the configuration has been selected on
validation and is ready for the frozen three-seed run.
