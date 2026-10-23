# Single-team experiment: behavioural cloning first

## Research question

Can a policy learned from several matches of one identified team reproduce that
team's player-level actions and short-horizon team trajectories in a held-out
match?

This experiment deliberately starts with behavioural cloning (BC). Other
methods are not run until the BC data contract, split, baselines, checkpoint
reload and reconstruction metrics pass.

## Checked focal team and split

The repeated Dataset A team `DFL-CLU-00000P` occurs in five matches in the
prepared `final_stage_v1` data:

| Role | Match IDs |
|---|---|
| Train | `DFL-MAT-J03WOH`, `DFL-MAT-J03WOY`, `DFL-MAT-J03WPY` |
| Validation | `DFL-MAT-J03WQQ` |
| Test | `DFL-MAT-J03WR9` |

The policy therefore sees the same club identity during training and testing,
but the test game is never used for optimization or model choice. Opponents,
match context and player availability can differ. The manifest records whether
the focal team occupies the left or right GRF slots in every chunk.

If only one match is available, use a chronological 60/20/20 split and purge
one complete chunk at each boundary. This is a weaker fallback: it measures
within-match temporal reconstruction, not generalization to a new game.

## What BC learns

For focal-team player (i) at decision time (t), BC approximates the expert
policy

\[
\pi_\theta(a_t^i \mid o_t^i)
= \operatorname{softmax}(f_\theta(o_t^i)),
\]

where (f_\theta) is a shared multilayer perceptron reused for all eleven focal
players. Training minimizes categorical negative log likelihood:

\[
\mathcal L_{BC}(\theta)
= -\frac{1}{N}\sum_{n=1}^{N}\log \pi_\theta(a_n^* \mid o_n).
\]

Implementation: `61 -> 128 -> 128 -> 19`, ReLU after each hidden layer, Adam,
learning rate `3e-4`, minibatches of `512`, and deterministic `argmax` at test
time. The output is 19 unnormalized action logits. Softmax is applied by the
cross-entropy loss; it does not need to be stored in the network.

The 19 targets are the GRF discrete actions: idle; eight movement directions;
long, high and short pass; shot; sprint; release direction; release sprint;
sliding; dribble; and release dribble. Movement labels are inferred from the
next tracking displacement. Sparse technical actions override movement labels
when a source event can be mapped to a player. These are approximate GRF
controls, not observed real-player button presses.

## Input representation

Each sample is one focal player's causal local observation at time (t). It
contains 61 stored values:

| Indices | Values |
|---|---|
| `0:24` | Player and ball position, relative ball position, side, player slot, four nearest teammates and four nearest opponents |
| `24:28` | Player and ball displacement from `t-1` to `t` |
| `28:47` | One-hot previous action |
| `47:57` | Reconstructed sticky direction/sprint/dribble state |
| `57:61` | Team possession state and individual ball ownership |

The primary full-data BC policy uses `temporal_context_no_slot`: spatial
context, velocities and possession, but with the roster-order value at feature
7 removed (31 retained inputs). The data audit showed that this value is not a
tactical role: 38 players change it within a match, and every value mixes
multiple `PlayingPosition` categories across games. Previous-action and sticky
features are also zeroed. This prevents a high score obtained mainly by
repeating the previous action. A `full` 61-feature policy remains a diagnostic
and must always be compared with the previous-action persistence baseline.

The network receives no future position, future action, match result or test
game data. Team side is retained because GRF exposes right-team controls in a
rotated attacking frame.

## Inputs and outputs at each stage

| Stage | Input | Output |
|---|---|---|
| Conversion | Raw tracking, ball and event records | Valid 22-player transition chunks |
| Team selection | Transition chunk and focal `team_id` | Eleven focal player slots; opponent remains context only |
| Training sample | `o_t^i` with 32 active/61 stored features | Expert action class `a_t^i` |
| Actor | Tensor `[batch,61]` | Logits `[batch,19]` |
| Offline test | Recorded held-out observations | Predicted actions and class metrics |
| Conditional GRF test | Held-out initial state, live focal observations, recorded opponent actions | Simulated focal positions and shape errors |

One shared actor is used rather than eleven separate networks. Parameter
sharing increases data per role and expresses the hypothesis that the team's
players use a common behavioral policy conditioned on position, local context
and player slot. It does not explicitly model communication or a joint action
distribution.

## Whole experiment flow

```text
Raw Dataset A tracking + events
  -> normalize coordinates to GRF
  -> keep finite, stable 11-v-11 runs
  -> infer 22 GRF actions per transition
  -> preserve team_id, player_id, match_id and event provenance
  -> find all matches containing DFL-CLU-00000P
  -> split entire matches 3 train / 1 validation / 1 test
  -> select only the focal team's 11 supervised samples
  -> audit train action support
  -> train shared BC MLP, streaming one chunk at a time
  -> save and reload checkpoint
  -> offline evaluation on train / validation / test
       * ordinary accuracy
       * balanced accuracy
       * per-action support and recall
       * majority baseline
       * previous-action baseline
       * accuracy only at action-change frames
  -> conditional GRF reconstruction on held-out test chunks
       * learned actions for focal team
       * recorded open-loop actions for opponent
       * previous-action focal baseline
       * recorded focal-action replay as a simulator-mismatch reference
       * average/final player displacement error in metres
       * centroid, team width and team depth error
  -> inspect trajectories and decide whether BC passes the gate
  -> only then prepare the next method
```

Offline evaluation is teacher-forced: every prediction receives the real
recorded observation. It measures action reconstruction without compounding
errors. GRF reconstruction is autoregressive for the focal team and therefore
tests covariate shift, but it is conditional because the opponent replays its
held-out recorded action sequence. The same initial state and opponent actions
are run with the BC policy, a previous-action focal baseline, and recorded focal
actions. The recorded-action rollout is not a learned result; it exposes the
irreducible initialization and simulator mismatch in this reconstruction. A
future fully autonomous test should use a separately trained/frozen opponent
and be reported as a different experiment.

## Reproduce the setup

Build the full and pilot same-team manifests:

```bash
.venv/bin/python -m thesis_experiments.scripts.prepare_team_experiment \
  --source-manifest data/processed/final_stage_v1/split_manifest.json \
  --team-id DFL-CLU-00000P \
  --train-matches DFL-MAT-J03WOH DFL-MAT-J03WOY DFL-MAT-J03WPY \
  --validation-matches DFL-MAT-J03WQQ \
  --test-matches DFL-MAT-J03WR9 \
  --output data/processed/final_stage_v1/team_00000P_manifest.json \
  --pilot-output data/processed/final_stage_v1/team_00000P_pilot_manifest.json \
  --pilot-chunks-per-match 2
```

Run the bounded BC pilot first:

```bash
.venv/bin/python -m thesis_experiments.scripts.run_team_bc \
  --manifest data/processed/final_stage_v1/team_00000P_pilot_manifest.json \
  --output-dir artifacts/single_team_bc_pilot \
  --observation-variant temporal_context_no_slot \
  --epochs 10 --batch-size 512 --hidden-size 128 --seed 0 \
  --grf-steps 100 --grf-python .venv-grf/bin/python
```

After the pilot passes, use the full manifest, repeat seeds `0`, `1`, and `2`
as separate runs, choose epochs using validation only, lock the choice, then
evaluate the test match once.

## BC gate before the next method

Proceed only when all of the following hold:

1. Team identity and match separation checks pass.
2. Every evaluated action has reported training support; unsupported actions
   are not interpreted as learnable failures.
3. The checkpoint reload reproduces predictions.
4. The primary temporal-context model beats the training-majority baseline and
   is reported beside previous-action persistence.
5. Action-change and supported event-action recall are non-trivial across
   repeated seeds.
6. Conditional rollouts remain finite and improve on a declared baseline for
   displacement and team-shape errors.
7. Validation determines all modeling choices before test evaluation.

BC cannot by itself prove that tactical behavior was learned. High frame-level
accuracy may reflect action persistence, and low GRF trajectory error mixes
policy error with simulator initialization, hidden velocity, possession,
sticky-state and player-ability mismatch. These limitations must accompany the
result.

## Completed full-data result (17 October 2026)

The data audit passed match separation, side coverage and train-to-evaluation
action support. It found two label classes with fewer than 100 training samples
(`long_pass`: 10; `shot`: 42) and confirmed that the stored slot value was not
a tactical role. The full runs therefore used
`temporal_context_no_slot`, retaining 31 of 61 stored features.

Three full-data seeds used 4,650,514 focal-player training samples. Validation
loss selected epochs 8, 9 and 10. Mean test action accuracy was
`0.7980 ± 0.0219`, compared with the `0.2013` training-majority baseline.
Action-change accuracy was `0.1956 ± 0.0308`, compared with the `0.0859`
change-point majority baseline. This passes the inferred-action imitation gate.

Conditional GRF reconstruction used five evenly spaced held-out test chunks
per seed and 100 predicted steps per chunk. BC obtained `6.817 ± 1.784 m` ADE;
autoregressive previous-action persistence obtained `3.636 m`; replaying the
recorded inferred actions obtained `6.307 m`. A validation-only warm-up audit
showed that recorded-action warm-up worsens every controller because simulator
mismatch accumulates before evaluation.

The resulting conclusion is deliberately split:

- BC is a defensible baseline for imitating the inferred discrete action
  labels.
- BC does not reconstruct the held-out team trajectory better than persistence
  and cannot support a game-reconstruction claim.

The durable generated reports are `artifacts/single_team_bc_full_audit/` and
`artifacts/single_team_bc_full/FULL_BC_REPORT.md` when local artifacts are
available. The next reconstruction-specific experiment should predict player
displacement/trajectory directly or introduce a dynamics-aware policy; the
discrete BC checkpoint remains the action-imitation baseline for later method
comparisons.

## Final diagnostics and frozen baseline (17 October 2026)

The three existing checkpoints were re-evaluated without retraining. The final
diagnostic package adds target-by-prediction confusion matrices, per-action
precision/recall/support, player-, match-, and chunk-level metrics, and
five-frame examples around correct/incorrect action changes and
unambiguously event-derived labels. All headline numbers reproduce the saved
full-data report.

The confusion matrices refine the action-imitation conclusion. Movement
directions (actions 0–8) account for the useful performance, while every
unambiguously event-derived test action has zero recall in every seed. The
model should therefore be described as a discrete movement-action baseline,
not as a complete football action policy.

The baseline is logically frozen by
`artifacts/single_team_bc_full/frozen_discrete_action_bc.json`, which records
SHA-256 hashes for the defining manifest, three checkpoints, configurations,
evaluations, results, and aggregate reports. It verifies validation-based
checkpoint selection and records test as report-only. Displacement experiments
must use a separate artifact directory.

The diagnostic report is
`artifacts/single_team_bc_full/diagnostics/BC_DIAGNOSTICS.md`.

## Direct displacement follow-up (17 October 2026)

A separate shared MLP now predicts current-to-next player displacement in
metres from the same 31-feature causal observation. Three seeds were selected
by validation mean displacement error at epochs 8, 8, and 5. Mean test error
is `0.0156 ± 0.0036 m`, better than the stationary baseline (`0.0710 m`) but
worse than previous-velocity persistence (`0.0125 m`). Validation gives the
same decision: `0.0144 ± 0.0013 m` versus `0.0126 m`.

The 100-step teacher-forced cumulative result is `0.4107 ± 0.3442 m` ADE and
`0.7927 ± 0.6944 m` FDE, compared with `0.0575 m` and `0.0886 m` for previous
velocity. The direct target is functional, but it fails the declared
reconstruction gate and has substantial seed-dependent drift. These numbers
are not autonomous rollouts because each step uses its recorded observation.

Technical details and reproduction commands are in
`thesis_plan/DIRECT_DISPLACEMENT_BC_EXPERIMENT.md`; the generated report is
`artifacts/single_team_displacement_bc_full/DISPLACEMENT_BC_REPORT.md`.

## Accepted autonomous trajectory BC (17 October 2026)

Residual displacement BC, initialized exactly at previous-velocity
persistence and selected with epoch 0 eligible, passes the validation and test
one-step gates. A subsequent direct multi-horizon model predicts steps
`1, 5, 10, 25, 50, 100` without future observations. Its frozen held-out test
result is `0.9654 ± 0.0128 m` ADE and `3.8036 ± 0.0493 m` 100-step FDE,
compared with `1.1946 m` and `4.7974 m` for constant velocity.

This is the accepted BC trajectory-forecasting formulation for later method
comparisons. The exact information boundary, split, horizons, seeds, metrics,
and baselines are frozen in
`thesis_plan/trajectory_evaluation_contract_v1.json`. The claim remains
offline autonomous forecasting, not closed-loop match reconstruction.
