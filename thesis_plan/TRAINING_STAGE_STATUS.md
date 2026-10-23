# Policy creation and training stage: audit and completed initial run

Date: 2026-09-16

The requested implementation and initial-training stage is complete. All seven
selected entries have trainable policies, saved checkpoints, held-out action
evaluation, and autonomous execution in Google Research Football. This does
not establish convergence, tactical realism, or final comparative conclusions.

## What the audit found

The earlier Stage 9 report marked a smoke stage complete, but it included a DT
placeholder. PPO/GAIL trained on teacher-forced replay, with actions unable to
change the state. Replay advanced through different players instead of through
one player's trajectory. CQL implicitly used the next player's observation as
a next state. There was no common reload-and-run evaluation for all methods.
The existing checkpoints covered just two 100-frame clips.

## What was completed

- Implemented a causal, return-conditioned Decision Transformer and stateful
  policy inference. Future/current action labels cannot leak into predictions.
- Added a Gymnasium environment over a separate legacy GRF worker, with all 22
  players controlled by learned actions. PPO/GAIL now default to live GRF.
- Added a shared-policy SB3 vector adapter with team rewards, time-limit
  bootstrapping, terminal observations, joint resets and seeded initialization.
- Corrected replay and CQL trajectory ordering, file boundaries and truncation
  handling. CQL saves a reloadable model including its configuration.
- Added loaders for BC, GAIL, PPO, CQL, DT, ghosting and CTDE checkpoints.
- Corrected right-team ghosting directions, added residual displacement
  prediction, and retained loading compatibility for older ghosting checkpoints.
- Corrected goal rewards: a shot at goal or goal kick is not a scored goal.
  Dataset A successful-shot outcomes are now preserved during extraction.
- Prepared finite, stable-roster real-data clips with disjoint match splits.
  No missing coordinates are silently replaced by the pitch centre for training.
- Added validation/test evaluation, autonomous trajectory artifacts, a common
  comparison table, incremental suite manifests and nonzero failure exits.
  Placeholder output is explicitly rejected as a completed training run.
- Documented each method's mechanism and implementation scope in
  [METHOD_MECHANISMS.md](METHOD_MECHANISMS.md).

## Actual data and training budget

Ten real matches contribute one 512-frame clip each (511 transitions per
match). Dataset A uses frames 10000–10511. Metrica uses the first complete,
contiguous 512-frame segment with a visible ball and stable player roster.

| Split | Dataset A matches | Metrica matches | Joint transitions | Player transitions |
|---|---|---|---:|---:|
| Train | J03WMX, J03WN1, J03WOH, J03WOY, J03WPY | Sample Game 1 | 3,066 | 67,452 |
| Validation | J03WQQ | Sample Game 2 | 1,022 | 22,484 |
| Test | J03WR9 | Sample Game 3 | 1,022 | 22,484 |

The split manifest records source, match ID, frame range and SHA-256 for every
NPZ. All methods use seed 0 and the same split. No selection or tuning was
performed using the test results.

BC, DT and ghosting: 10 epochs. CTDE: the trained BC initialization, 10 critic
epochs and 10 actor epochs. CQL: 11,264 gradient updates. PPO and GAIL: 11,264
agent transitions each (512 joint simulator steps, four 128-step rollouts).
Hidden size is 128; the CTDE critic uses its existing 256-unit default.
CPU execution uses one PyTorch thread. These budgets verify the initial stage;
CQL gradient steps and PPO/GAIL agent steps are different units.

## Measured results

| Method | Status | Validation accuracy | Test accuracy | Test balanced accuracy | Mean GRF position error |
|---|---|---:|---:|---:|---:|
| BC | completed | 20.84% | 9.27% | 5.88% | 0.1228 |
| GAIL | completed | 5.47% | 2.96% | 7.81% | 0.1261 |
| PPO | completed | 0.89% | 1.96% | 1.66% | 0.1195 |
| CQL | completed | 22.14% | 10.00% | 6.34% | 0.0902 |
| DT | completed | 3.14% | 5.68% | 4.45% | 0.1020 |
| GHOSTING | completed | 12.36% | 11.62% | 8.30% | 0.0944 |
| CTDE | completed | 20.72% | 9.83% | 6.39% | 0.1059 |

Every policy was reloaded in a fresh evaluation process. Each controlled all
22 players for 100 simulator steps on the Dataset A test match and another
100 steps on Metrica Game 3: **14 autonomous rollouts, 1,400 joint steps**.
Every rollout produced finite observations and positions. All scores remained
0–0 during these four-second clips. Position error is Euclidean distance in
GRF coordinates; the x and y axes have different metric scales, so these are
not distances in metres. Rollouts are aligned at 25 Hz and use current
observations, without a future-position controller.

**Prediction quality remains poor.** The training-majority action achieves
16.59% on the test set, higher than every learned policy in this initial run.
BC reaches 51.07% training accuracy and CQL 63.21%, but held-out accuracy falls
to 9.27% and 10.00%. DT training cross-entropy decreases from 1.443 in the first
epoch to 0.441 in the last. This proves optimization runs, not useful
cross-match generalization. PPO/GAIL have very small online budgets and optimize
rewards rather than supervised action accuracy.

## Reproduce and inspect

From the repository root:

```bash
.venv-methods/bin/python -m thesis_experiments.scripts.prepare_training_data
.venv-methods/bin/python -m thesis_experiments.scripts.run_training_suite \
  --split-manifest data/processed/training_stage/split_manifest.json \
  --output-dir artifacts/training_stage_20260916 \
  --methods all --seeds 0 --epochs 10 --actor-epochs 10 \
  --batch-size 256 --hidden-size 128 --total-timesteps 11264 \
  --rollout-steps 128 --episode-steps 128 --context-length 16 \
  --bc-backend torch --env-backend grf --eval-grf-steps 100
.venv-methods/bin/python -m pytest thesis_experiments/tests -q
```

Retained evidence:

- [Run manifest](../artifacts/training_stage_20260916/training_manifest.json)
- [Comparison CSV](../artifacts/training_stage_20260916/comparison.csv)
- [Comparison JSON](../artifacts/training_stage_20260916/comparison.json)
- [Split manifest](../data/processed/training_stage/split_manifest.json)
- Checkpoints, configurations, training metrics, held-out evaluation JSON and
  rollout NPZ arrays: `artifacts/training_stage_20260916/seed_0/methods/<method>/`
- Complete per-method training/evaluation logs: `artifacts/training_stage_20260916/logs/`
- Installed versions: `environment-methods.txt` and `environment-grf.txt` in
  the run directory. The original Stage 9 artifacts were preserved.

Generated data and artifacts remain ignored by Git, as the project requires.
This report and source code are the durable repository deliverables.

## Verification and remaining experimental work

The full project suite passes: **39 tests**, including trajectory-order regression tests,
CQL next-state/last-transition checks, a DT causal-mask test, checkpoint
roundtrips, goal-reward and right-team action checks, split-leakage rejection,
a placeholder-failure exit test, Gymnasium's environment checker, and a live
GRF test proving different actions produce different trajectories.

The remaining work is the full research experiment stage: broader clips and
complete matches, larger online training budgets, repeated seeds, validation-
based model selection, richer tactical metrics and unseen-opponent evaluation.
The 24-feature observation lacks velocities, possession and sticky-action
state; the GRF scenario initializes positions but cannot reconstruct all of
those hidden variables. Action labels are inferred and technical-event labels
are sparse. Ghosting is a feed-forward baseline and CTDE is a project
advantage-weighted offline baseline, not exact reproductions of all papers
cited in the thesis. These limits must accompany any future result comparison.
