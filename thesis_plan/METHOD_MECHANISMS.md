# Implemented method mechanisms

The selected methods in `theory/utenko_irp.pdf`, sections 3–4, are BC,
GAIL, PPO, CQL, Decision Transformer and ghosting. CTDE is an additional
project baseline. IRL is explicitly a conceptual contrast class in the paper,
not a selected experimental method. Citations in the literature review do not
imply that every cited algorithm is a separate experiment.

All policies consume the same 24 local features: own and ball positions,
ball-relative position, team and player-slot indicators, and the four nearest
teammates/opponents. There are 19 discrete GRF actions. World positions remain
in a shared coordinate system; right-team direction actions use GRF's rotated
control frame. The global critic receives 46 features (22 positions and ball).

| Method | Training mechanism | Implemented variant and execution |
|---|---|---|
| BC | Minimize action cross-entropy on demonstration state/action pairs. | Shared two-layer MLP, PyTorch or `imitation` backend. Each player acts from local observations. |
| GAIL | Train a discriminator to separate expert and generated state/action samples; optimize the policy against its learned reward. | `imitation` GAIL with an SB3 PPO generator, actual GRF interaction, one shared policy for all 22 players. A state/action reward network avoids learning recording-cutoff labels. |
| PPO | Collect current-policy rollouts and optimize the clipped policy-ratio objective with value and entropy terms. | Parameter-sharing independent PPO, with local actor and local value function. The 22 rollout streams are coupled players in one game. This is not MAPPO or a reproduction of Yu et al.'s experiments. |
| CQL | Fit Bellman targets while penalizing high Q values for actions absent from demonstrations. | d3rlpy discrete CQL: discrete conservative penalty plus DQN-style learning, not continuous-action SAC-CQL. Chronological per-player episodes preserve next states and bootstrap at recording cutoffs. |
| DT | Predict actions using causal sequences of return-to-go, state and previous actions. | Two-layer, four-head PyTorch Transformer, 16-step context by default. Tokens are `(R_t, s_t, a_t)`; action logits come from the state token, so the label at time t is hidden. Right padding is excluded from loss. GRF inference maintains a separate history per player and subtracts observed rewards from the desired return. |
| Ghosting | Regress next positions from demonstration observations, then map displacement to a movement action. | A small feed-forward residual displacement baseline. It uses nearby players but does not reproduce the recurrent role-assignment model of Le et al.; it is labelled a baseline throughout the results. |
| CTDE | Fit a centralized two-team value critic, then use TD advantages to weight demonstration-action likelihood with a BC anchor. | Project offline advantage-weighted imitation baseline. Training sees global state; execution uses only the shared local actor. CTDE is a training architecture, not one uniquely defined published algorithm. |

PPO maximizes simulator scoring reward plus signed ball-x progression. CQL,
DT and CTDE use the same formula on recorded transitions. GAIL replaces the
environment reward with its discriminator reward during generator training.
BC and ghosting do not optimize rewards. These are different objectives, so
action accuracy alone cannot establish which method is best.

The simulator runs four physics steps per observation (25 Hz), matching the
tracking data. Initial scenarios reproduce recorded positions, but tracking
does not specify all GRF state: velocity, possession, sticky controls and
player abilities are not reconstructed. Trajectory errors therefore include
initialization and simulator mismatch as well as policy error.

The legacy `dones` array records clip boundaries, not terminal match states.
Replay exposes them as truncations. CQL appends the final recorded next state
and uses nonterminal episodes; CTDE bootstraps at clip ends. DT computes a
finite-clip return-to-go. That return is a short-horizon training target, not
the full match outcome.

Sources: [PPO](https://arxiv.org/abs/1707.06347),
[GAIL](https://arxiv.org/abs/1606.03476),
[CQL](https://arxiv.org/abs/2006.04779),
[Decision Transformer](https://arxiv.org/abs/2106.01345),
[Gymnasium environment contract](https://gymnasium.farama.org/api/env/), and
[SB3 vector environment contract](https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html).
Local algorithm papers are retained under `theory/`.
