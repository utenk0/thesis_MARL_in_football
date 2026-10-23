# Stage 9 Conclusion

Date: 2026-09-15

Historical smoke-stage snapshot. The 2026-09-16 audit found that the broader
training stage was incomplete. DT, live Gymnasium training and saved-policy
evaluation are now implemented; see [TRAINING_STAGE_STATUS.md](TRAINING_STAGE_STATUS.md).

Stage 9 is complete as a smoke-tested implementation stage. The project now has
one shared method interface, one common transition-data contract, and runnable
adapters for:

```text
BC
GAIL
PPO
CQL
Decision Transformer placeholder
Ghosting
CTDE
```

## Environment Fix

The Gym/Gymnasium conflict is fixed by environment separation, not by ignoring
the issue.

```text
.venv-grf
  purpose: live Google Research Football replay and validation
  key packages: gfootball 2.10.3, gym 0.21.0
  does not install d3rlpy

.venv-methods
  purpose: offline method training and smoke tests
  key packages: gymnasium 0.29.1, gym 0.26.2, d3rlpy 2.6.1,
                stable-baselines3 2.2.1, imitation 1.0.1, torch 2.14.0
  does not install gfootball
```

Why this is necessary:

```text
Google Research Football requires gym<=0.21.
d3rlpy requires gym>=0.26.
Those requirements cannot be satisfied safely in one environment.
```

Verification:

```text
.venv-grf/bin/python -m thesis_experiments.scripts.check_environment --target grf
.venv-methods/bin/python -m thesis_experiments.scripts.check_environment --target methods
.venv-grf/bin/python main.py --steps 1
.venv-methods/bin/python -m pytest thesis_experiments/tests -q
```

All checks passed locally.

## Gymnasium Use By Method

The project uses Gymnasium as the method-facing environment standard, but not
every method directly steps through a Gymnasium environment.

```text
PPO
  Directly uses Gymnasium through FootballReplayEnv.

GAIL
  Directly uses Gymnasium through FootballReplayEnv wrapped for stable-baselines3.

BC
  Uses the imitation backend. Gymnasium only supplies observation/action spaces
  for offline demonstrations; BC does not run environment rollouts.

CQL
  Does not use FootballReplayEnv directly. It uses d3rlpy's offline dataset API
  built from the same flattened player transitions.

Ghosting
  Does not use Gymnasium. It is a project-owned PyTorch next-position baseline.

CTDE
  Does not use Gymnasium. It is a project-owned PyTorch centralized-critic and
  shared-actor trainer.

Decision Transformer
  Does not use Gymnasium yet. It is only a placeholder manifest until the
  external backend is selected.
```

So the accurate statement is:

```text
All methods share the same project transition data contract.
PPO and GAIL directly use the Gymnasium replay environment.
BC uses the imitation backend with Gymnasium spaces for offline demonstrations.
CQL, Ghosting, CTDE, and the DT placeholder do not directly use Gymnasium.
```

In other words, the split is not "Gymnasium for every method except CQL and
Ghosting." CTDE and the Decision Transformer placeholder also do not directly
step a Gymnasium environment in the current implementation.

## Next Stage

Stage 10 is the next active stage. The smoke metrics prove that the framework
runs, but they are not final thesis results. Stage 10 should define the final
train/validation/test split, repeated seeds, longer training budgets, and the
offline/simulation evaluation metrics.
