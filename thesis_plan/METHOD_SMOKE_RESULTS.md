# Stage 9 Method Smoke Results

Date: 2026-09-15

These runs verify that the unified method framework can consume the same
football transition inputs and write method artifacts. They are short smoke
runs, not final thesis results.

The Stage 9 environment fix and method-backend conclusion are recorded in
[`STAGE_9_CONCLUSION.md`](STAGE_9_CONCLUSION.md).

## Inputs

```text
data/processed/dataset_a_ctde_100.npz
  source: Dataset A provider sample
  frames: 10000..10098
  transitions: 99
  local observations: [99, 22, 24]
  global states: [99, 46]

data/processed/metrica_ctde_100.npz
  source: Metrica Sample Game 1
  frames: 1..99
  transitions: 99
  local observations: [99, 22, 24]
  global states: [99, 46]
```

Flattened player transitions for single-agent backends:

```text
samples: 4356
action space: GRF Discrete(19)
```

## Dependency Note

The method stack belongs in `.venv-methods`:

```text
torch 2.14.0
gymnasium 0.29.1
stable-baselines3 2.2.1
imitation 1.0.1
d3rlpy 2.6.1
gym 0.26.2
```

This creates a known environment conflict with vendored Google Research
Football, which expects `gym<=0.21.0`. Keep method training in `.venv-methods`
and live GRF replay in `.venv-grf`.

Current local note: the first smoke run was executed in the existing `.venv`
after installing the method stack there. `.venv-methods` now points to that
working method environment, while `.venv-grf` is a separate GRF-pinned
environment.

## Results

| Method | Backend | Status | Smoke metric | Artifact |
|---|---|---:|---:|---|
| BC | imitation | trained | action accuracy 0.3067 | `artifacts/methods/bc/imitation_bc_policy.zip` |
| GAIL | imitation + stable-baselines3 PPO | trained | action accuracy 0.0771 | `artifacts/methods/gail/gail_ppo_generator.zip` |
| PPO | stable-baselines3 | trained | action accuracy 0.1444 | `artifacts/methods/ppo/ppo_replay_policy.zip` |
| CQL | d3rlpy | trained | action accuracy 0.0611 | `artifacts/methods/cql/cql_policy.pt` |
| DT | external placeholder | placeholder | 4356 samples checked | `artifacts/methods/dt/decision_transformer_placeholder.json` |
| Ghosting | PyTorch | trained | movement-action accuracy 0.1573 | `artifacts/methods/ghosting/ghosting_baseline.pt` |
| CTDE | PyTorch | trained | critic loss 0.000207 | `artifacts/methods/ctde/shared_actor_ctde.pt` |

Additional PyTorch BC checkpoint for CTDE initialization:

```text
artifacts/methods/bc/shared_actor_bc.pt
```

## Interpretation

These metrics are only framework smoke checks on two 100-frame windows. They
should not be reported as final performance. Stage 10 must define train/valid/test
splits, longer training budgets, repeated seeds, and evaluation metrics before
the thesis comparison table is produced.

## Conclusion

The Gym/Gymnasium caveat is fixed by using two environments, not by ignoring the
version conflict: `.venv-grf` keeps GRF pinned to `gym 0.21`, while
`.venv-methods` keeps Gymnasium, SB3, imitation, d3rlpy, and PyTorch together
for offline method training.

Gymnasium is the standard interface for environment-based method backends, but
not every method directly steps through a Gymnasium environment. BC's backend is
`imitation`; Gymnasium is only used there to provide observation/action spaces:

```text
Direct FootballReplayEnv users: PPO, GAIL
imitation BC with Gymnasium spaces: BC
Offline d3rlpy dataset API: CQL
Project PyTorch trainers: Ghosting, CTDE
Placeholder only: Decision Transformer
```
