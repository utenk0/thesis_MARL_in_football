# Current BC, CQL, and CTDE implementation

This document describes the **current trajectory implementations** used by the
shared `single_team_autonomous_trajectory_v1` comparison. It also separates
them from the older 19-action implementations that remain in the repository
for reproducibility.

## 1. What is being predicted

All three current methods solve the same offline forecasting problem. At a
forecast origin $t$, a method predicts the displacement of each of the 11
focal-team players from its current position at six direct horizons:

\[
H = \{1,5,10,25,50,100\}.
\]

For player $i$, the target at horizon $h$ is

\[
y_{i,t,h}=p_{i,t+h}-p_{i,t}\in\mathbb R^2,
\]

in metres. A prediction therefore has shape `(samples, 6, 2)`. The methods do
not receive observations, actions, velocities, events, or simulator state from
after $t$. These are direct forecasts, not teacher-forced rollouts and not
closed-loop Google Research Football controllers.

The frozen experiment uses focal team `DFL-CLU-00000P`, three match-disjoint
training matches, one validation match, one held-out test match, and seeds
`0`, `1`, and `2`. Training and checkpoint selection never open the test data.

## 2. Shared input and data path

`JointTransitionDataset` is the stored transition container. Each transition
contains:

- `local_observations`: one 61-feature observation for each of 22 players;
- `global_states`: 22 player positions plus the ball position, or 46 values;
- recorded 19-class GRF actions;
- a two-team reward;
- the corresponding next observations/states and a terminal flag;
- frame, player, team, source, and match identifiers.

The current trajectory actor is shared by all focal players. Each player is a
separate training example, but every player uses the same network parameters.
Its input consists of:

1. A 61-value current local observation.
2. Ten causal history steps, each containing player `(dx, dy)` and ball
   `(dx, dy)`, giving 40 additional values.

The physical input tensor consequently has 101 values. The
`temporal_context_no_slot` mask retains 31 of the 61 current features and all
40 history values, so 71 values are active. The retained current features are
the spatial context, player and ball velocity, and possession information.
The roster-slot feature, previous-action one-hot vector, and sticky actions are
masked out.

For every horizon, all three methods start from the same causal
constant-velocity forecast

\[
b_{i,t,h}=h\,v_{i,t}.
\]

They differ in how they produce or select the correction added to this
baseline.

## 3. Shared interface classes

### `TrajectoryDatasetSplit`

Loads train, validation, and test paths from the manifest, verifies that the
match sets are disjoint, and optionally verifies the frozen trajectory
contract. Its `paths(split)` method is the common split accessor.

### `TrajectoryTrainingConfig`

Contains shared training options and the CQL- and CTDE-specific
hyperparameters. It validates horizons, positive sizes and learning rates,
discount factors, early-stopping settings, and method-specific parameters.

### `TrajectoryTrainingResult`

Records the method name, selected checkpoint, training report, validation
report, and the two information-boundary flags. For a valid pretest run,
`test_loaded` and `future_observations_used` remain `False`.

### `TrajectoryMethod`

The abstract base class for the current comparison. Every implementation must
provide:

- `horizons` and `history_steps`;
- `train(dataset, output_dir, config)`;
- `load(checkpoint, device)`;
- `predict(observations, causal_history)` returning `(N, H, 2)` metres;
- `evaluate(paths, team_id)` using the common trajectory metrics.

The registry maps `bc`, `cql`, and `ctde` to the three concrete adapter
classes below.

## 4. Behavioural cloning (BC)

### Idea

BC directly imitates the demonstrated future player trajectories. It does not
use rewards or learn values. The model predicts one continuous `(x, y)`
correction for every horizon:

\[
\hat y_{i,t,h}=h\,v_{i,t}+c_\theta(o_{i,t},m_{i,\leq t})_h.
\]

Here $o_{i,t}$ is the current local observation and $m_{i,\leq t}$ is the
causal motion history.

### Classes

#### `MultiHorizonResidualPredictor`

The PyTorch actor is a shared MLP:

```text
101 inputs -> hidden -> ReLU -> hidden -> ReLU -> 12 outputs
```

The default hidden size is 128. The 12 outputs are reshaped to `(6, 2)`. The
last layer is initialized to zero, so before any update the correction is
exactly zero and epoch 0 exactly reproduces constant velocity.

#### `MultiHorizonBCMethod`

This is the `TrajectoryMethod` adapter. It trains and loads the predictor,
validates inference shapes, adds the learned correction to constant velocity,
and delegates evaluation to the shared trajectory evaluator.

#### `_MetricAccumulator`

This private evaluation helper is shared by BC, CQL, and CTDE. It streams
player-position and formation errors without retaining every prediction.

### Training objective

For a player-origin example $n$, define the mean trajectory error

\[
e_n(\theta)=\frac{1}{|H|}\sum_{h\in H}
\left\|\hat y_{n,h}-y_{n,h}\right\|_2.
\]

BC minimizes

\[
L_{BC}(\theta)=\frac{1}{N}\sum_{n=1}^{N}e_n(\theta).
\]

Thus the actual loss is mean Euclidean position error in metres across all
players, forecast origins, and horizons. It is not action cross-entropy and it
is not coordinate-wise MSE.

After every epoch, validation autonomous ADE is computed. The checkpoint with
the lowest validation ADE is retained. Epoch 0 is a valid candidate, and early
stopping uses validation only.

### Inference

Only the local actor is needed. It emits six corrections in one forward pass;
the code adds them to the six constant-velocity displacements. No predicted
horizon is fed back to produce a later horizon.

## 5. Conservative Q-learning (CQL)

### Idea

Discrete CQL must choose among a finite set of actions, whereas the shared
contract requires a continuous six-horizon trajectory. The implementation
therefore turns demonstrated residual trajectories into discrete
**trajectory-plan actions**.

For every training example, it first computes

\[
c_{i,t}=y_{i,t}-b_{i,t}\in\mathbb R^{6\times2}.
\]

A deterministic bounded sample of training corrections is clustered with
k-means. With the default `action_count=32`, 31 centres are learned and action
0 is explicitly reserved as an all-zero correction. Validation and test data
are not used to fit these prototypes.

Every demonstrated correction is labelled with its nearest prototype using
squared Euclidean distance over the flattened 12-dimensional plan. At each
recorded frame, the plan is reselected, so the offline transition semantics
are receding-horizon.

### Classes

#### `TrajectoryCQLQNetwork`

The shared Q-network is

```text
101 inputs -> hidden -> ReLU -> hidden -> ReLU -> K Q-values
```

where $K$ is the number of trajectory prototypes. The last layer is zero
initialized. At epoch 0 all actions tie; PyTorch `argmax` chooses action 0,
which makes the complete policy exactly constant velocity.

#### `TrajectoryCQLMethod`

This adapter trains and loads the Q-network and prototypes. At inference it
selects `argmax_a Q(s,a)`, retrieves that prototype, adds it to constant
velocity, and returns the same `(N, 6, 2)` output as BC.

### Offline transition

For each player and forecast origin:

```text
state      = masked current local observation + causal history
action     = nearest train-only residual-plan prototype
reward     = focal-team ball progress + goal reward
next_state = the same player's state at the next recorded frame
done       = recorded terminal flag, with every file boundary forced terminal
```

All 11 focal players receive the same team reward at a time step. In the data
builder, left-team ball progress is `ball_x[t+1] - ball_x[t]`; the right-team
reward has the opposite sign. A detected goal adds `+1` for the scoring team
and `-1` for the opponent under the default settings.

### CQL objective

The implementation uses a Double-DQN-style target action with a periodically
copied target network:

\[
a^*=\arg\max_a Q_\theta(s',a),
\]

\[
z=r+\gamma(1-d)Q_{\bar\theta}(s',a^*).
\]

For demonstrated action $a_D$, the loss is

\[
L_{CQL}=
\operatorname{Huber}(Q_\theta(s,a_D),z)
+\alpha\left[
\log\sum_a\exp Q_\theta(s,a)-Q_\theta(s,a_D)
\right].
\]

The first term performs temporal-difference learning. The second is the
conservative penalty: it lowers unsupported actions relative to the action
present in the offline data. Defaults are `gamma=0.99`, `alpha=1.0`, and a
target-network copy every 500 optimizer updates. Gradients are clipped to norm
10.

This is genuine reward-based offline Q-learning. Euclidean distance is used
to quantize demonstrated plans and to evaluate forecasts, but it is not the Q
network's training loss.

As with BC, validation ADE chooses the saved epoch and epoch 0 is eligible.
This means RL trains the network, while the common forecasting metric selects
the checkpoint.

## 6. Centralized training with decentralized execution (CTDE)

### Idea

CTDE is an architecture rather than one uniquely defined algorithm. Here it
means a BC-initialized local trajectory actor plus a training-only centralized
value critic. The critic identifies higher-advantage demonstrated moments;
those moments receive more weight during actor imitation.

### Classes

#### `MultiHorizonResidualPredictor` as actor

CTDE reuses the exact BC actor class and requires a frozen BC checkpoint with
the same seed, focal team, horizons, history length, observation mask, and
hidden size. Epoch 0 of CTDE is consequently the selected frozen BC actor, not
constant velocity.

#### `CentralizedTrajectoryCritic`

The critic is a scalar value MLP:

```text
47 inputs -> hidden -> ReLU -> hidden -> ReLU -> 1 value
```

The default hidden size is 256. Its 47 inputs are the current positions of all
22 players (44), the ball position (2), and a focal-side indicator (1). It is
used only during training.

#### `TrajectoryCTDEMethod`

This adapter enforces the BC-checkpoint compatibility checks, trains and saves
the actor and critic, and exposes only local-actor predictions through the
shared interface. Evaluation loads the critic from the checkpoint but never
calls it to make forecasts.

### Critic training

The critic minimizes Bellman MSE:

\[
z_t=r_t+\gamma(1-d_t)V_{\bar\phi}(s_{t+1}),
\]

\[
L_V(\phi)=\mathbb E[(V_\phi(s_t)-z_t)^2].
\]

The target critic begins as a copy of the critic and is updated after every
critic epoch. The default is 20 critic epochs, `gamma=0.99`, and gradient-norm
clipping at 10.

### Actor training

After critic training, a temporal-difference advantage is computed per team
time step:

\[
A_t=r_t+\gamma(1-d_t)V_\phi(s_{t+1})-V_\phi(s_t).
\]

It becomes a positive bounded weight shared by the 11 focal players:

\[
w_t=\min\left(\exp(A_t/\tau),w_{max}\right).
\]

For each player-origin example, $e_n(\theta)$ is the same mean six-horizon
Euclidean error used by BC. The actor minimizes

\[
L_{actor}=\mathbb E[w_t e_n(\theta)]
+\lambda_{BC}\mathbb E[e_n(\theta)].
\]

Defaults are `temperature=1.0`, `max_weight=20`, and
`bc_coefficient=0.5`.

The second term is an unweighted BC imitation term. It is useful to call it a
**BC-loss anchor**, but it is not an L2 penalty between the current actor
parameters and a frozen copy of the initial BC parameters.

Validation ADE selects the actor after every epoch. Since epoch 0 is eligible,
CTDE falls back to the original BC actor if no fine-tuned actor improves
validation ADE. The saved critic is the final trained critic; it has no role in
validation, test evaluation, or deployment.

## 7. What “accuracy” means in the current comparison

The current trajectory methods do **not** define an `accuracy` percentage.
They perform continuous regression, where lower error is better.

Let $\hat p_{n,h}$ and $p_{n,h}$ be the predicted and true absolute player
positions for player-origin example $n$ and horizon $h$. The per-horizon
mean position error is

\[
E_h=\frac{1}{N}\sum_{n=1}^{N}
\left\|\hat p_{n,h}-p_{n,h}\right\|_2.
\]

### Primary metrics

- **Autonomous ADE:** the unweighted mean of the six direct-horizon errors,
  \(\text{ADE}=\frac{1}{6}\sum_{h\in H}E_h\). “Unweighted” matters because
  the horizons are not equally spaced.
- **Autonomous FDE:** the mean player-position error at the final, 100-step
  horizon, \(\text{FDE}=E_{100}\).

“Autonomous” means that every horizon uses only information available at the
forecast origin. It does not mean an interactive simulator rollout.

### Secondary metrics

- **RMSE:** square root of the mean squared radial position error.
- **Centroid error:** Euclidean distance between the predicted and true means
  of the 11 player positions.
- **Team-width MAE:** absolute error in `max(y) - min(y)` over the 11 players.
- **Team-depth MAE:** absolute error in `max(x) - min(x)` over the 11 players.

Metrics are also reported separately at all six horizons and per match. The
three seeds are aggregated using their arithmetic mean and sample standard
deviation (`ddof=1`).

### Baselines and validation gate

Every method is compared against:

- stationary: zero displacement;
- constant velocity: $h v_t$;
- constant acceleration:
  $h v_t + \frac{h(h+1)}{2}(v_t-v_{t-1})$.

The validation gate passes only if:

1. mean three-seed ADE strictly beats constant velocity;
2. mean three-seed FDE strictly beats constant velocity;
3. at least two of three seeds strictly improve both ADE and FDE; and
4. no seed regresses on either ADE or FDE.

Therefore, a statement such as “90% trajectory accuracy” is not defined by
this code. A percentage would require an additional explicit rule, such as
“fraction of predictions within 1 metre,” which is not currently implemented.

## 8. Legacy 19-action implementations

The repository retains an older method framework, but it is not the accepted
trajectory comparison:

- `FootballMethod` is the old common base class, with `MethodConfig` and
  `MethodResult` as its configuration/result containers.
- `BehavioralCloningMethod` predicts one of 19 GRF action labels with the
  `imitation` package or a local PyTorch `SharedActor`.
- `CQLMethod` uses d3rlpy `DiscreteCQL` over the 19 GRF actions.
- `CTDEMethod` uses the older `CentralizedCritic` and a BC-initialized
  19-action `SharedActor`.

In this legacy path:

\[
\text{action accuracy}=
\frac{\#\{\hat a=a_{expert}\}}{\#\text{samples}}.
\]

The detailed discrete-BC diagnostics additionally define:

- **balanced action accuracy:** macro-average recall over action classes that
  occur in the evaluated split;
- **action-change accuracy:** exact action accuracy only where the current
  expert action differs from the previous expert action.

Those classification metrics must not be compared numerically with trajectory
ADE/FDE. The legacy classes answer “which discrete action label was recorded?”;
the current classes answer “where will each player be at the six horizons?”

## 9. End-to-end execution flow

```text
manifest + frozen contract
        |
        v
TrajectoryDatasetSplit + TrajectoryTrainingConfig
        |
        v
trajectory registry selects BC / CQL / CTDE adapter
        |
        +-- BC: continuous residual imitation
        +-- CQL: Q-learning over train-only residual-plan prototypes
        +-- CTDE: BC actor + training-only global value critic
        |
        v
validation ADE selects an epoch (epoch 0 included)
        |
        v
three seeds are aggregated and checked against constant velocity
        |
        v
code/config/checkpoints are hash-frozen
        |
        v
held-out test is opened once and reported with the common metrics
```

At the time represented by the repository artifacts, the accepted BC has a
frozen held-out result of `0.9654 +/- 0.0128 m` ADE and
`3.8036 +/- 0.0493 m` 100-step FDE, versus constant velocity at `1.1946 m`
ADE and `4.7974 m` FDE. No frozen trajectory-CQL or trajectory-CTDE result
artifacts are present in the repository, so their performance must not be
inferred from the existence of their implementations.
