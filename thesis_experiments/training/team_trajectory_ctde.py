"""Trajectory CTDE with a centralized critic and decentralized shared actor.

The actor is initialized from the accepted trajectory BC and remains callable
from the same local causal input.  During offline training only, a centralized
critic sees the current 22-player/ball state and the focal-team side.  Its TD
advantage weights the demonstrated trajectory-regression updates.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from thesis_experiments.training.team_multi_horizon_bc import (
    DEFAULT_HORIZONS,
    MultiHorizonResidualPredictor,
    _MetricAccumulator,
    _load_model as _load_bc_model,
    _predict_corrections,
    _validation_metrics,
    load_team_multi_horizon_examples,
)
from thesis_experiments.training.team_residual_displacement_bc import (
    residual_input_mask,
)
from thesis_experiments.transitions.observations import (
    LOCAL_FEATURES,
    OBSERVATION_VERSION,
)


class CentralizedTrajectoryCritic(nn.Module):
    """Estimate focal-team value from current global state and team side."""

    def __init__(self, state_size: int = 47, hidden_size: int = 256) -> None:
        super().__init__()
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.network(states).squeeze(-1)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_trajectory_ctde_transitions(
    path: Path,
    team_id: str,
    *,
    history_steps: int,
    horizons: tuple[int, ...],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, object],
]:
    """Load aligned critic transitions and decentralized actor examples."""

    inputs, targets, velocity, _, metadata = load_team_multi_horizon_examples(
        path,
        team_id,
        history_steps=history_steps,
        horizons=horizons,
    )
    starts = int(metadata["time_starts"])
    with np.load(path, allow_pickle=False) as data:
        team_ids = np.asarray(data["team_ids"]).astype(str)
        selected = np.flatnonzero(team_ids == team_id)
        if len(selected) != 11:
            raise ValueError(f"{path}: expected 11 slots for team {team_id!r}.")
        sides = np.unique(selected // 11)
        if len(sides) != 1:
            raise ValueError(f"{path}: focal team crosses left/right team slots.")
        side = int(sides[0])
        global_states = np.asarray(data["global_states"][:starts], dtype=np.float32)
        next_global_states = np.asarray(
            data["next_global_states"][:starts], dtype=np.float32
        )
        rewards = np.asarray(data["team_rewards"][:starts, side], dtype=np.float32)
        dones = np.asarray(data["dones"][:starts], dtype=np.float32)
    if global_states.shape != next_global_states.shape or global_states.ndim != 2:
        raise ValueError(f"{path}: invalid current/next centralized states.")
    side_feature = np.full(
        (starts, 1), 1.0 if side == 0 else -1.0, dtype=np.float32
    )
    critic_states = np.concatenate([global_states, side_feature], axis=1)
    next_critic_states = np.concatenate([next_global_states, side_feature], axis=1)
    metadata = dict(metadata)
    metadata.update(
        {
            "team_side": "left" if side == 0 else "right",
            "reward": "focal_team_ball_progress_plus_goals",
            "centralized_state_features": int(critic_states.shape[1]),
        }
    )
    return (
        critic_states,
        next_critic_states,
        rewards,
        dones,
        inputs,
        targets,
        velocity,
        metadata,
    )


def _load_model(checkpoint: Path, device: str):
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    actor = MultiHorizonResidualPredictor(
        input_size=int(saved["input_size"]),
        horizons=tuple(int(value) for value in saved["horizons"]),
        hidden_size=int(saved["actor_hidden_size"]),
    ).to(device)
    actor.load_state_dict(saved["actor_state_dict"])
    actor.eval()
    critic = CentralizedTrajectoryCritic(
        state_size=int(saved["critic_state_size"]),
        hidden_size=int(saved["critic_hidden_size"]),
    ).to(device)
    critic.load_state_dict(saved["critic_state_dict"])
    critic.eval()
    return actor, critic, np.asarray(saved["input_mask"], dtype=np.float32), saved


def evaluate_team_trajectory_ctde(
    checkpoint: Path,
    paths: list[Path],
    *,
    team_id: str,
    device: str = "cpu",
    prediction_batch_size: int = 16_384,
) -> dict[str, object]:
    """Evaluate only the decentralized actor through the shared contract."""

    if not paths:
        raise ValueError("At least one evaluation path is required.")
    actor, _, input_mask, saved = _load_model(Path(checkpoint), device)
    horizons = tuple(int(value) for value in saved["horizons"])
    history_steps = int(saved["history_steps"])
    names = ("policy", "stationary", "constant_velocity", "constant_acceleration")
    accumulators = {name: _MetricAccumulator(horizons) for name in names}
    matches: dict[str, dict[str, _MetricAccumulator]] = {}
    for path in paths:
        inputs, targets, velocity, acceleration, metadata = load_team_multi_horizon_examples(
            path,
            team_id,
            history_steps=history_steps,
            horizons=horizons,
        )
        corrections = _predict_corrections(
            actor,
            inputs,
            input_mask,
            batch_size=prediction_batch_size,
            device=device,
        )
        predictors = {
            "policy": velocity + corrections,
            "stationary": np.zeros_like(targets),
            "constant_velocity": velocity,
            "constant_acceleration": acceleration,
        }
        match_id = str(metadata["match_id"])
        match_accumulators = matches.setdefault(
            match_id,
            {name: _MetricAccumulator(horizons) for name in names},
        )
        for name, prediction in predictors.items():
            accumulators[name].update(
                prediction, targets, metadata["current_positions_meters"]
            )
            match_accumulators[name].update(
                prediction, targets, metadata["current_positions_meters"]
            )
    return {
        "method": "trajectory_ctde",
        "team_id": team_id,
        "checkpoint": str(checkpoint),
        "selected_epoch": int(saved["selected_epoch"]),
        "epoch_zero": str(saved["epoch_zero"]),
        "horizons": list(horizons),
        "history_steps": history_steps,
        "selection_metric": str(saved["selection_metric"]),
        "test_used_for_selection": bool(saved["test_used_for_selection"]),
        "future_observations_used": False,
        "centralized_information_used_at_execution": False,
        "overall": {name: values.result() for name, values in accumulators.items()},
        "matches": {
            match_id: {name: values.result() for name, values in rows.items()}
            for match_id, rows in sorted(matches.items())
        },
        "scope": (
            "The centralized critic is training-only. Every reported forecast is "
            "produced by the shared actor from the origin observation and causal history."
        ),
    }


def train_team_trajectory_ctde(
    paths: list[Path],
    bc_checkpoint: Path,
    output: Path,
    *,
    team_id: str,
    validation_paths: list[Path],
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    actor_epochs: int = 10,
    critic_epochs: int = 20,
    batch_size: int = 2048,
    learning_rate: float = 3e-4,
    actor_hidden_size: int = 128,
    critic_hidden_size: int = 256,
    history_steps: int = 10,
    seed: int = 0,
    device: str = "cpu",
    observation_variant: str = "temporal_context_no_slot",
    early_stopping_patience: int = 3,
    minimum_epochs: int = 5,
    gamma: float = 0.99,
    temperature: float = 1.0,
    max_weight: float = 20.0,
    bc_coefficient: float = 0.5,
) -> dict[str, object]:
    """Fit a training-only critic and validation-select the local actor."""

    if not paths or not validation_paths:
        raise ValueError("Training and validation paths must both be non-empty.")
    if min(
        actor_epochs,
        critic_epochs,
        batch_size,
        actor_hidden_size,
        critic_hidden_size,
        history_steps,
        early_stopping_patience,
    ) < 1:
        raise ValueError("CTDE training parameters must be positive.")
    if learning_rate <= 0 or not 0 <= gamma <= 1:
        raise ValueError("Invalid CTDE learning rate or gamma.")
    if min(temperature, max_weight, bc_coefficient) <= 0:
        raise ValueError("CTDE weighting parameters must be positive.")
    if minimum_epochs < 0 or minimum_epochs > actor_epochs:
        raise ValueError("minimum_epochs must be between zero and actor_epochs.")
    bc_checkpoint = Path(bc_checkpoint)
    if not bc_checkpoint.is_file():
        raise FileNotFoundError(bc_checkpoint)

    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    actor, bc_input_mask, bc_saved = _load_bc_model(bc_checkpoint, device)
    if int(bc_saved.get("seed", -1)) != seed:
        raise ValueError("CTDE and BC initialization seeds do not match.")
    if str(bc_saved.get("team_id", "")) != team_id:
        raise ValueError("BC initialization belongs to a different focal team.")
    if bc_saved.get("future_observations_used") is not False:
        raise ValueError("BC initialization does not guarantee causal observations.")
    if bc_saved.get("test_used_for_selection") is not False:
        raise ValueError("BC initialization used test data for model selection.")
    bc_horizons = tuple(int(value) for value in bc_saved["horizons"])
    if bc_horizons != horizons:
        raise ValueError("BC initialization horizons do not match CTDE horizons.")
    if int(bc_saved["history_steps"]) != history_steps:
        raise ValueError("BC initialization history does not match CTDE history.")
    expected_mask = residual_input_mask(observation_variant, history_steps)
    if not np.array_equal(bc_input_mask, expected_mask):
        raise ValueError("BC initialization observation mask does not match CTDE.")
    if int(bc_saved["hidden_size"]) != actor_hidden_size:
        raise ValueError("BC initialization hidden size does not match CTDE actor.")
    input_mask = bc_input_mask.astype(np.float32)
    mask = torch.as_tensor(input_mask, dtype=torch.float32, device=device)

    first = load_trajectory_ctde_transitions(
        paths[0],
        team_id,
        history_steps=history_steps,
        horizons=horizons,
    )
    critic_state_size = first[0].shape[1]
    critic = CentralizedTrajectoryCritic(
        state_size=critic_state_size, hidden_size=critic_hidden_size
    ).to(device)
    target_critic = deepcopy(critic).to(device).eval()
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=learning_rate)
    critic_epoch_losses = []
    for _ in range(critic_epochs):
        critic.train()
        loss_sum = 0.0
        seen = 0
        for file_index in rng.permutation(len(paths)):
            states, next_states, rewards, dones, _, _, _, _ = (
                load_trajectory_ctde_transitions(
                    paths[int(file_index)],
                    team_id,
                    history_steps=history_steps,
                    horizons=horizons,
                )
            )
            order = rng.permutation(len(rewards))
            for start in range(0, len(order), batch_size):
                indices = order[start : start + batch_size]
                state = torch.as_tensor(states[indices], dtype=torch.float32, device=device)
                next_state = torch.as_tensor(
                    next_states[indices], dtype=torch.float32, device=device
                )
                reward = torch.as_tensor(rewards[indices], dtype=torch.float32, device=device)
                done = torch.as_tensor(dones[indices], dtype=torch.float32, device=device)
                with torch.no_grad():
                    target = reward + gamma * (1.0 - done) * target_critic(next_state)
                prediction = critic(state)
                loss = F.mse_loss(prediction, target)
                critic_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), max_norm=10.0)
                critic_optimizer.step()
                count = len(indices)
                loss_sum += float(loss.detach().cpu()) * count
                seen += count
        critic_epoch_losses.append(loss_sum / seen)
        target_critic.load_state_dict(critic.state_dict())
    critic.eval()

    sample_count = 0
    matches = set()
    for path in paths:
        _, _, _, _, inputs, _, _, metadata = load_trajectory_ctde_transitions(
            path,
            team_id,
            history_steps=history_steps,
            horizons=horizons,
        )
        sample_count += len(inputs)
        matches.add(str(metadata["match_id"]))

    validation_batch_size = max(batch_size, 8192)
    initial = _validation_metrics(
        actor,
        validation_paths,
        team_id=team_id,
        history_steps=history_steps,
        horizons=horizons,
        input_mask=input_mask,
        batch_size=validation_batch_size,
        device=device,
    )
    initial["epoch"] = 0
    validation_history = [initial]
    best_error = float(initial["policy_autonomous_ade_meters"])
    best_epoch = 0
    best_actor_state = {
        name: value.detach().cpu().clone() for name, value in actor.state_dict().items()
    }
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=learning_rate)
    actor_epoch_losses = []
    advantage_history = []
    without_improvement = 0
    for epoch in range(1, actor_epochs + 1):
        actor.train()
        loss_sum = 0.0
        seen = 0
        epoch_advantages = []
        epoch_weights = []
        for file_index in rng.permutation(len(paths)):
            states, next_states, rewards, dones, inputs, targets, velocity, _ = (
                load_trajectory_ctde_transitions(
                    paths[int(file_index)],
                    team_id,
                    history_steps=history_steps,
                    horizons=horizons,
                )
            )
            with torch.no_grad():
                state_tensor = torch.as_tensor(states, dtype=torch.float32, device=device)
                next_state_tensor = torch.as_tensor(
                    next_states, dtype=torch.float32, device=device
                )
                reward_tensor = torch.as_tensor(
                    rewards, dtype=torch.float32, device=device
                )
                done_tensor = torch.as_tensor(dones, dtype=torch.float32, device=device)
                advantage = (
                    reward_tensor
                    + gamma * (1.0 - done_tensor) * critic(next_state_tensor)
                    - critic(state_tensor)
                )
                weights_by_time = torch.exp(advantage / temperature).clamp(
                    max=max_weight
                )
            weights = weights_by_time[:, None].expand(-1, 11).reshape(-1)
            epoch_advantages.append(advantage.detach().cpu().numpy())
            epoch_weights.append(weights_by_time.detach().cpu().numpy())
            order = rng.permutation(len(inputs))
            for start in range(0, len(order), batch_size):
                indices = order[start : start + batch_size]
                batch = torch.as_tensor(
                    inputs[indices], dtype=torch.float32, device=device
                )
                target = torch.as_tensor(
                    targets[indices], dtype=torch.float32, device=device
                )
                baseline = torch.as_tensor(
                    velocity[indices], dtype=torch.float32, device=device
                )
                prediction = baseline + actor(batch * mask)
                errors = torch.sqrt(
                    torch.sum(torch.square(prediction - target), dim=-1) + 1e-12
                ).mean(dim=1)
                selected_weights = weights[torch.as_tensor(indices, device=device)]
                loss = (
                    (selected_weights * errors).mean()
                    + bc_coefficient * errors.mean()
                )
                actor_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), max_norm=10.0)
                actor_optimizer.step()
                count = len(indices)
                loss_sum += float(loss.detach().cpu()) * count
                seen += count
        actor_epoch_losses.append(loss_sum / seen)
        all_advantages = np.concatenate(epoch_advantages)
        all_weights = np.concatenate(epoch_weights)
        advantage_history.append(
            {
                "epoch": epoch,
                "mean_advantage": float(all_advantages.mean()),
                "std_advantage": float(all_advantages.std()),
                "mean_weight": float(all_weights.mean()),
                "maximum_weight": float(all_weights.max()),
            }
        )
        validation = _validation_metrics(
            actor,
            validation_paths,
            team_id=team_id,
            history_steps=history_steps,
            horizons=horizons,
            input_mask=input_mask,
            batch_size=validation_batch_size,
            device=device,
        )
        validation["epoch"] = epoch
        validation_history.append(validation)
        error = float(validation["policy_autonomous_ade_meters"])
        if error < best_error - 1e-8:
            best_error = error
            best_epoch = epoch
            best_actor_state = {
                name: value.detach().cpu().clone()
                for name, value in actor.state_dict().items()
            }
            without_improvement = 0
        else:
            without_improvement += 1
        if epoch >= minimum_epochs and without_improvement >= early_stopping_patience:
            break

    actor.load_state_dict(best_actor_state)
    selected_validation = validation_history[best_epoch]
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "actor_state_dict": actor.state_dict(),
            "critic_state_dict": critic.state_dict(),
            "input_size": len(input_mask),
            "actor_hidden_size": actor_hidden_size,
            "critic_state_size": critic_state_size,
            "critic_hidden_size": critic_hidden_size,
            "history_steps": history_steps,
            "horizons": list(horizons),
            "input_mask": torch.from_numpy(input_mask),
            "stage": "single_team_trajectory_ctde",
            "team_id": team_id,
            "observation_version": OBSERVATION_VERSION,
            "observation_variant": observation_variant,
            "actor": "constant_velocity_plus_shared_local_residual_predictor",
            "critic": "training_only_current_global_state_value",
            "reward": "focal_team_ball_progress_plus_goals",
            "actor_loss": "advantage_weighted_euclidean_bc_plus_bc_anchor",
            "critic_loss": "bellman_mean_squared_error",
            "gamma": gamma,
            "temperature": temperature,
            "max_weight": max_weight,
            "bc_coefficient": bc_coefficient,
            "bc_initialization_checkpoint": str(bc_checkpoint),
            "bc_initialization_sha256": _sha256(bc_checkpoint),
            "bc_initialization_seed": int(bc_saved["seed"]),
            "bc_initialization_team_id": str(bc_saved["team_id"]),
            "selection_metric": "validation_autonomous_ade_meters",
            "epoch_zero": "frozen_bc_initialization",
            "future_observations_used": False,
            "centralized_information_used_at_execution": False,
            "training_matches": sorted(matches),
            "selected_epoch": best_epoch,
            "validation_selection": True,
            "test_used_for_selection": False,
            "seed": seed,
        },
        output,
    )
    return {
        "team_id": team_id,
        "seed": seed,
        "matches": sorted(matches),
        "forecast_player_origins": sample_count,
        "critic_epochs": critic_epochs,
        "critic_epoch_losses": critic_epoch_losses,
        "maximum_actor_epochs": actor_epochs,
        "actor_epochs_run": len(actor_epoch_losses),
        "selected_epoch": best_epoch,
        "actor_epoch_losses": actor_epoch_losses,
        "advantage_history": advantage_history,
        "validation_history": validation_history,
        "selected_validation": selected_validation,
        "best_validation_autonomous_ade_meters": best_error,
        "initial_bc_validation_autonomous_ade_meters": float(
            initial["policy_autonomous_ade_meters"]
        ),
        "constant_velocity_validation_autonomous_ade_meters": float(
            initial["constant_velocity_autonomous_ade_meters"]
        ),
        "strictly_beats_constant_velocity_ade_on_validation": bool(
            best_error
            < float(initial["constant_velocity_autonomous_ade_meters"]) - 1e-8
        ),
        "strictly_beats_constant_velocity_fde_on_validation": bool(
            float(selected_validation["policy_autonomous_fde_meters"])
            < float(initial["constant_velocity_autonomous_fde_meters"]) - 1e-8
        ),
        "horizons": list(horizons),
        "history_steps": history_steps,
        "observation_variant": observation_variant,
        "retained_current_features": int(input_mask[:LOCAL_FEATURES].sum()),
        "bc_initialization_checkpoint": str(bc_checkpoint),
        "bc_initialization_sha256": _sha256(bc_checkpoint),
        "bc_initialization_seed": int(bc_saved["seed"]),
        "bc_initialization_team_id": str(bc_saved["team_id"]),
        "gamma": gamma,
        "temperature": temperature,
        "max_weight": max_weight,
        "bc_coefficient": bc_coefficient,
        "centralized_information_used_at_execution": False,
        "future_observations_used": False,
        "selection_metric": "validation_autonomous_ade_meters",
    }
