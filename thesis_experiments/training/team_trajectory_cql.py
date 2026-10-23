"""Conservative offline Q-learning over causal trajectory-plan actions.

The accepted comparison contract asks every method to return six direct
future displacements, while classic discrete CQL selects one action.  This
module reconciles the two without turning CQL into supervised regression:
training-only demonstrated residual trajectories are clustered into discrete
motion-plan prototypes, and a Q-network selects a prototype from the current
causal state.  Action zero is always the zero correction, so an untrained
network exactly reproduces the constant-velocity baseline.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from thesis_experiments.training.team_displacement_bc import METERS_PER_GRF
from thesis_experiments.training.team_multi_horizon_bc import (
    DEFAULT_HORIZONS,
    _MetricAccumulator,
    load_team_multi_horizon_examples,
)
from thesis_experiments.training.team_residual_displacement_bc import (
    residual_input_mask,
)
from thesis_experiments.transitions.observations import (
    LOCAL_FEATURES,
    LOCAL_FEATURE_SLICES,
    OBSERVATION_VERSION,
)


class TrajectoryCQLQNetwork(nn.Module):
    """Shared Q-network over discrete multi-horizon motion plans."""

    def __init__(
        self,
        input_size: int,
        action_count: int,
        hidden_size: int = 128,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.action_count = action_count
        self.hidden_size = hidden_size
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_count),
        )
        # All actions tie at epoch zero. torch.argmax selects action zero,
        # which is reserved for the constant-velocity plan.
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


def _scalar(value: np.ndarray) -> str:
    return str(np.asarray(value).item())


def _validate_horizons(horizons: tuple[int, ...]) -> None:
    if not horizons or tuple(sorted(set(horizons))) != horizons:
        raise ValueError("Horizons must be unique and increasing.")
    if any(value < 1 for value in horizons):
        raise ValueError("Horizons must be positive.")


def _squared_distances(values: np.ndarray, centers: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    centers = np.asarray(centers, dtype=np.float32)
    distances = (
        np.square(values).sum(axis=1, keepdims=True)
        + np.square(centers).sum(axis=1)[None, :]
        - 2.0 * values @ centers.T
    )
    return np.maximum(distances, 0.0)


def nearest_trajectory_prototypes(
    corrections: np.ndarray,
    prototypes: np.ndarray,
    *,
    chunk_size: int = 16_384,
) -> np.ndarray:
    """Assign each demonstrated residual trajectory to its nearest plan."""

    corrections = np.asarray(corrections, dtype=np.float32)
    prototypes = np.asarray(prototypes, dtype=np.float32)
    if corrections.ndim != 3 or corrections.shape[-1] != 2:
        raise ValueError("corrections must have shape (samples, horizons, 2)")
    if prototypes.ndim != 3 or prototypes.shape[1:] != corrections.shape[1:]:
        raise ValueError("prototypes must match correction horizon and coordinate shapes")
    flat_values = corrections.reshape(len(corrections), -1)
    flat_centers = prototypes.reshape(len(prototypes), -1)
    labels = []
    for start in range(0, len(flat_values), chunk_size):
        distances = _squared_distances(
            flat_values[start : start + chunk_size], flat_centers
        )
        labels.append(np.argmin(distances, axis=1).astype(np.int64))
    return np.concatenate(labels) if labels else np.empty(0, dtype=np.int64)


def _training_corrections(
    path: Path,
    team_id: str,
    *,
    history_steps: int,
    horizons: tuple[int, ...],
) -> np.ndarray:
    _, targets, velocity, _, _ = load_team_multi_horizon_examples(
        path,
        team_id,
        history_steps=history_steps,
        horizons=horizons,
    )
    return (targets - velocity).astype(np.float32)


def sample_training_corrections(
    paths: list[Path],
    team_id: str,
    *,
    history_steps: int,
    horizons: tuple[int, ...],
    maximum_samples: int,
    seed: int,
) -> np.ndarray:
    """Uniformly retain a bounded, deterministic sample from train only."""

    if not paths or maximum_samples < 1:
        raise ValueError("Training paths and a positive sample limit are required.")
    rng = np.random.default_rng(seed)
    retained_values: np.ndarray | None = None
    retained_keys = np.empty(0, dtype=np.float64)
    for path in paths:
        values = _training_corrections(
            path,
            team_id,
            history_steps=history_steps,
            horizons=horizons,
        )
        keys = rng.random(len(values))
        if retained_values is None:
            candidates = values
        else:
            candidates = np.concatenate([retained_values, values], axis=0)
            keys = np.concatenate([retained_keys, keys])
        if len(candidates) > maximum_samples:
            indices = np.argpartition(keys, -maximum_samples)[-maximum_samples:]
            retained_values = candidates[indices]
            retained_keys = keys[indices]
        else:
            retained_values = candidates
            retained_keys = keys
    if retained_values is None or not len(retained_values):
        raise ValueError("No train-only trajectory corrections were available.")
    return retained_values.astype(np.float32)


def fit_trajectory_prototypes(
    samples: np.ndarray,
    *,
    action_count: int,
    iterations: int,
    seed: int,
    chunk_size: int = 16_384,
) -> tuple[np.ndarray, dict[str, object]]:
    """Fit train-only k-means plans while reserving action zero as no correction."""

    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim != 3 or samples.shape[-1] != 2:
        raise ValueError("samples must have shape (samples, horizons, 2)")
    if action_count < 2 or iterations < 1:
        raise ValueError("At least two actions and one k-means iteration are required.")
    if len(samples) < action_count - 1:
        raise ValueError("Not enough trajectory samples for the requested action count.")

    rng = np.random.default_rng(seed)
    values = samples.reshape(len(samples), -1)
    cluster_count = action_count - 1
    centers = np.empty((cluster_count, values.shape[1]), dtype=np.float32)
    centers[0] = values[int(rng.integers(len(values)))]
    minimum_distances = _squared_distances(values, centers[:1])[:, 0]
    for index in range(1, cluster_count):
        total = float(minimum_distances.sum())
        if total <= 1e-12:
            chosen = int(rng.integers(len(values)))
        else:
            chosen = int(rng.choice(len(values), p=minimum_distances / total))
        centers[index] = values[chosen]
        candidate = _squared_distances(values, centers[index : index + 1])[:, 0]
        minimum_distances = np.minimum(minimum_distances, candidate)

    completed_iterations = 0
    for iteration in range(iterations):
        sums = np.zeros_like(centers, dtype=np.float64)
        counts = np.zeros(cluster_count, dtype=np.int64)
        farthest_value = values[0]
        farthest_distance = -1.0
        for start in range(0, len(values), chunk_size):
            chunk = values[start : start + chunk_size]
            distances = _squared_distances(chunk, centers)
            labels = np.argmin(distances, axis=1)
            np.add.at(sums, labels, chunk)
            counts += np.bincount(labels, minlength=cluster_count)
            nearest = distances[np.arange(len(chunk)), labels]
            local = int(np.argmax(nearest))
            if float(nearest[local]) > farthest_distance:
                farthest_distance = float(nearest[local])
                farthest_value = chunk[local]
        updated = centers.copy()
        occupied = counts > 0
        updated[occupied] = (sums[occupied] / counts[occupied, None]).astype(np.float32)
        updated[~occupied] = farthest_value
        movement = float(np.max(np.linalg.norm(updated - centers, axis=1)))
        centers = updated
        completed_iterations = iteration + 1
        if movement < 1e-5:
            break

    prototypes = np.concatenate(
        [np.zeros((1, values.shape[1]), dtype=np.float32), centers], axis=0
    ).reshape(action_count, samples.shape[1], 2)
    labels = nearest_trajectory_prototypes(samples, prototypes)
    quantization = np.linalg.norm(prototypes[labels] - samples, axis=-1)
    counts = np.bincount(labels, minlength=action_count)
    metrics = {
        "samples": int(len(samples)),
        "action_count": action_count,
        "iterations_run": completed_iterations,
        "action_counts": counts.tolist(),
        "mean_quantization_error_meters": float(quantization.mean()),
        "per_horizon_quantization_error_meters": quantization.mean(axis=0).tolist(),
        "action_zero": "exact_zero_correction_constant_velocity",
    }
    return prototypes.astype(np.float32), metrics


def load_trajectory_cql_transitions(
    path: Path,
    team_id: str,
    *,
    history_steps: int,
    horizons: tuple[int, ...],
    prototypes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    """Load one file as per-player, receding-horizon offline transitions."""

    inputs, targets, velocity, _, metadata = load_team_multi_horizon_examples(
        path,
        team_id,
        history_steps=history_steps,
        horizons=horizons,
    )
    starts = int(metadata["time_starts"])
    states = inputs.reshape(starts, 11, -1)
    corrections = (targets - velocity).reshape(starts, 11, len(horizons), 2)
    actions = nearest_trajectory_prototypes(
        corrections.reshape(-1, len(horizons), 2), prototypes
    ).reshape(starts, 11)

    with np.load(path, allow_pickle=False) as data:
        team_ids = np.asarray(data["team_ids"]).astype(str)
        selected = np.flatnonzero(team_ids == team_id)
        if len(selected) != 11:
            raise ValueError(f"{path}: expected 11 slots for team {team_id!r}.")
        sides = np.unique(selected // 11)
        if len(sides) != 1:
            raise ValueError(f"{path}: focal team crosses left/right team slots.")
        side = int(sides[0])
        rewards_by_time = np.asarray(data["team_rewards"][:starts, side], dtype=np.float32)
        recorded_dones = np.asarray(data["dones"][:starts], dtype=bool)

    next_states = np.empty_like(states)
    if starts > 1:
        next_states[:-1] = states[1:]
    next_states[-1] = states[-1]
    dones_by_time = recorded_dones.copy()
    dones_by_time[-1] = True  # a file boundary never bootstraps into another clip
    rewards = np.repeat(rewards_by_time[:, None], 11, axis=1)
    dones = np.repeat(dones_by_time[:, None], 11, axis=1)
    metadata = dict(metadata)
    metadata.update(
        {
            "team_side": "left" if side == 0 else "right",
            "reward": "focal_team_ball_progress_plus_goals",
            "decision_semantics": "receding_horizon_motion_plan_reselected_each_frame",
        }
    )
    return (
        states.reshape(-1, states.shape[-1]).astype(np.float32),
        actions.reshape(-1).astype(np.int64),
        rewards.reshape(-1).astype(np.float32),
        next_states.reshape(-1, states.shape[-1]).astype(np.float32),
        dones.reshape(-1).astype(np.float32),
        metadata,
    )


def _load_model(checkpoint: Path, device: str):
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    model = TrajectoryCQLQNetwork(
        input_size=int(saved["input_size"]),
        action_count=int(saved["action_count"]),
        hidden_size=int(saved["hidden_size"]),
    ).to(device)
    model.load_state_dict(saved["model_state_dict"])
    model.eval()
    return (
        model,
        np.asarray(saved["input_mask"], dtype=np.float32),
        np.asarray(saved["prototypes"], dtype=np.float32),
        saved,
    )


def _predict_actions(
    model: TrajectoryCQLQNetwork,
    inputs: np.ndarray,
    input_mask: np.ndarray,
    *,
    batch_size: int,
    device: str,
) -> np.ndarray:
    mask = torch.as_tensor(input_mask, dtype=torch.float32, device=device)
    actions = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(inputs), batch_size):
            batch = torch.as_tensor(
                inputs[start : start + batch_size],
                dtype=torch.float32,
                device=device,
            )
            actions.append(model(batch * mask).argmax(dim=1).cpu().numpy())
    return np.concatenate(actions).astype(np.int64)


def predict_trajectory_cql_corrections(
    model: TrajectoryCQLQNetwork,
    inputs: np.ndarray,
    input_mask: np.ndarray,
    prototypes: np.ndarray,
    *,
    batch_size: int,
    device: str,
) -> np.ndarray:
    actions = _predict_actions(
        model,
        inputs,
        input_mask,
        batch_size=batch_size,
        device=device,
    )
    return prototypes[actions].astype(np.float32)


def _validation_metrics(
    model: TrajectoryCQLQNetwork,
    paths: list[Path],
    *,
    team_id: str,
    history_steps: int,
    horizons: tuple[int, ...],
    input_mask: np.ndarray,
    prototypes: np.ndarray,
    batch_size: int,
    device: str,
) -> dict[str, object]:
    policy_sum = np.zeros(len(horizons), dtype=np.float64)
    velocity_sum = np.zeros(len(horizons), dtype=np.float64)
    counts = np.zeros(len(horizons), dtype=np.int64)
    action_counts = np.zeros(len(prototypes), dtype=np.int64)
    for path in paths:
        inputs, targets, velocity, _, _ = load_team_multi_horizon_examples(
            path,
            team_id,
            history_steps=history_steps,
            horizons=horizons,
        )
        actions = _predict_actions(
            model,
            inputs,
            input_mask,
            batch_size=batch_size,
            device=device,
        )
        action_counts += np.bincount(actions, minlength=len(prototypes))
        prediction = velocity + prototypes[actions]
        policy_errors = np.linalg.norm(prediction - targets, axis=-1)
        velocity_errors = np.linalg.norm(velocity - targets, axis=-1)
        policy_sum += policy_errors.sum(axis=0)
        velocity_sum += velocity_errors.sum(axis=0)
        counts += len(policy_errors)
    policy = policy_sum / counts
    velocity = velocity_sum / counts
    return {
        "forecast_player_origins": int(counts[0]),
        "policy_autonomous_ade_meters": float(policy.mean()),
        "policy_autonomous_fde_meters": float(policy[-1]),
        "constant_velocity_autonomous_ade_meters": float(velocity.mean()),
        "constant_velocity_autonomous_fde_meters": float(velocity[-1]),
        "policy_per_horizon_meters": policy.tolist(),
        "constant_velocity_per_horizon_meters": velocity.tolist(),
        "selected_action_counts": action_counts.tolist(),
    }


def evaluate_team_trajectory_cql(
    checkpoint: Path,
    paths: list[Path],
    *,
    team_id: str,
    device: str = "cpu",
    prediction_batch_size: int = 16_384,
) -> dict[str, object]:
    """Evaluate CQL through the same autonomous trajectory metrics as BC."""

    if not paths:
        raise ValueError("At least one evaluation path is required.")
    model, input_mask, prototypes, saved = _load_model(Path(checkpoint), device)
    horizons = tuple(int(value) for value in saved["horizons"])
    history_steps = int(saved["history_steps"])
    names = ("policy", "stationary", "constant_velocity", "constant_acceleration")
    accumulators = {name: _MetricAccumulator(horizons) for name in names}
    matches: dict[str, dict[str, _MetricAccumulator]] = {}
    action_counts = np.zeros(len(prototypes), dtype=np.int64)
    for path in paths:
        inputs, targets, velocity, acceleration, metadata = load_team_multi_horizon_examples(
            path,
            team_id,
            history_steps=history_steps,
            horizons=horizons,
        )
        actions = _predict_actions(
            model,
            inputs,
            input_mask,
            batch_size=prediction_batch_size,
            device=device,
        )
        action_counts += np.bincount(actions, minlength=len(prototypes))
        predictors = {
            "policy": velocity + prototypes[actions],
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
        "method": "trajectory_cql",
        "team_id": team_id,
        "checkpoint": str(checkpoint),
        "selected_epoch": int(saved["selected_epoch"]),
        "epoch_zero": str(saved["epoch_zero"]),
        "horizons": list(horizons),
        "history_steps": history_steps,
        "selection_metric": str(saved["selection_metric"]),
        "test_used_for_selection": bool(saved["test_used_for_selection"]),
        "future_observations_used": False,
        "action_count": int(saved["action_count"]),
        "selected_action_counts": action_counts.tolist(),
        "overall": {name: values.result() for name, values in accumulators.items()},
        "matches": {
            match_id: {name: values.result() for name, values in rows.items()}
            for match_id, rows in sorted(matches.items())
        },
        "scope": (
            "CQL selects a train-only residual-trajectory prototype from the "
            "forecast-origin observation and causal history. No future input is supplied."
        ),
    }


def train_team_trajectory_cql(
    paths: list[Path],
    output: Path,
    *,
    team_id: str,
    validation_paths: list[Path],
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    epochs: int = 10,
    batch_size: int = 2048,
    learning_rate: float = 3e-4,
    hidden_size: int = 128,
    history_steps: int = 10,
    seed: int = 0,
    device: str = "cpu",
    observation_variant: str = "temporal_context_no_slot",
    early_stopping_patience: int = 3,
    minimum_epochs: int = 5,
    action_count: int = 32,
    conservative_alpha: float = 1.0,
    gamma: float = 0.99,
    target_update_interval: int = 500,
    prototype_samples: int = 100_000,
    kmeans_iterations: int = 15,
) -> dict[str, object]:
    """Fit train-only prototypes and validation-select a conservative Q-policy."""

    _validate_horizons(horizons)
    if not paths or not validation_paths:
        raise ValueError("Training and validation paths must both be non-empty.")
    if min(
        epochs,
        batch_size,
        hidden_size,
        history_steps,
        early_stopping_patience,
        target_update_interval,
        prototype_samples,
        kmeans_iterations,
    ) < 1:
        raise ValueError("CQL training parameters must be positive.")
    if action_count < 2 or learning_rate <= 0 or conservative_alpha < 0:
        raise ValueError("Invalid CQL action, learning-rate, or alpha setting.")
    if not 0 <= gamma <= 1:
        raise ValueError("gamma must be between zero and one.")
    if minimum_epochs < 0 or minimum_epochs > epochs:
        raise ValueError("minimum_epochs must be between zero and epochs.")

    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    sampled_corrections = sample_training_corrections(
        paths,
        team_id,
        history_steps=history_steps,
        horizons=horizons,
        maximum_samples=prototype_samples,
        seed=seed,
    )
    prototypes, prototype_metrics = fit_trajectory_prototypes(
        sampled_corrections,
        action_count=action_count,
        iterations=kmeans_iterations,
        seed=seed,
    )
    input_mask = residual_input_mask(observation_variant, history_steps)
    mask = torch.as_tensor(input_mask, dtype=torch.float32, device=device)
    model = TrajectoryCQLQNetwork(
        len(input_mask), action_count, hidden_size=hidden_size
    ).to(device)
    target_model = deepcopy(model).to(device).eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    sample_count = 0
    matches = set()
    for path in paths:
        inputs, _, _, _, metadata = load_team_multi_horizon_examples(
            path,
            team_id,
            history_steps=history_steps,
            horizons=horizons,
        )
        sample_count += len(inputs)
        matches.add(str(metadata["match_id"]))

    validation_batch_size = max(batch_size, 8192)
    initial = _validation_metrics(
        model,
        validation_paths,
        team_id=team_id,
        history_steps=history_steps,
        horizons=horizons,
        input_mask=input_mask,
        prototypes=prototypes,
        batch_size=validation_batch_size,
        device=device,
    )
    initial["epoch"] = 0
    validation_history = [initial]
    best_error = float(initial["policy_autonomous_ade_meters"])
    best_epoch = 0
    best_state = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }
    without_improvement = 0
    update_count = 0
    epoch_losses = []

    for epoch in range(1, epochs + 1):
        model.train()
        totals = {"loss": 0.0, "td_loss": 0.0, "conservative_loss": 0.0}
        seen = 0
        for file_index in rng.permutation(len(paths)):
            states, actions, rewards, next_states, dones, _ = load_trajectory_cql_transitions(
                paths[int(file_index)],
                team_id,
                history_steps=history_steps,
                horizons=horizons,
                prototypes=prototypes,
            )
            order = rng.permutation(len(actions))
            for start in range(0, len(order), batch_size):
                indices = order[start : start + batch_size]
                state = torch.as_tensor(states[indices], dtype=torch.float32, device=device)
                action = torch.as_tensor(actions[indices], dtype=torch.long, device=device)
                reward = torch.as_tensor(rewards[indices], dtype=torch.float32, device=device)
                next_state = torch.as_tensor(
                    next_states[indices], dtype=torch.float32, device=device
                )
                done = torch.as_tensor(dones[indices], dtype=torch.float32, device=device)

                q_values = model(state * mask)
                data_q = q_values.gather(1, action[:, None]).squeeze(1)
                with torch.no_grad():
                    next_action = model(next_state * mask).argmax(dim=1)
                    next_q = target_model(next_state * mask).gather(
                        1, next_action[:, None]
                    ).squeeze(1)
                    target = reward + gamma * (1.0 - done) * next_q
                td_loss = F.smooth_l1_loss(data_q, target)
                conservative_loss = (
                    torch.logsumexp(q_values, dim=1).mean() - data_q.mean()
                )
                loss = td_loss + conservative_alpha * conservative_loss
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                optimizer.step()
                update_count += 1
                if update_count % target_update_interval == 0:
                    target_model.load_state_dict(model.state_dict())

                count = len(indices)
                totals["loss"] += float(loss.detach().cpu()) * count
                totals["td_loss"] += float(td_loss.detach().cpu()) * count
                totals["conservative_loss"] += (
                    float(conservative_loss.detach().cpu()) * count
                )
                seen += count
        epoch_losses.append(
            {name: value / seen for name, value in totals.items()}
        )
        validation = _validation_metrics(
            model,
            validation_paths,
            team_id=team_id,
            history_steps=history_steps,
            horizons=horizons,
            input_mask=input_mask,
            prototypes=prototypes,
            batch_size=validation_batch_size,
            device=device,
        )
        validation["epoch"] = epoch
        validation_history.append(validation)
        error = float(validation["policy_autonomous_ade_meters"])
        if error < best_error - 1e-8:
            best_error = error
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            without_improvement = 0
        else:
            without_improvement += 1
        if epoch >= minimum_epochs and without_improvement >= early_stopping_patience:
            break

    model.load_state_dict(best_state)
    selected_validation = validation_history[best_epoch]
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_size": len(input_mask),
            "hidden_size": hidden_size,
            "history_steps": history_steps,
            "horizons": list(horizons),
            "input_mask": torch.from_numpy(input_mask),
            "prototypes": torch.from_numpy(prototypes),
            "action_count": action_count,
            "stage": "single_team_trajectory_cql",
            "team_id": team_id,
            "observation_version": OBSERVATION_VERSION,
            "observation_variant": observation_variant,
            "action_semantics": "train_only_residual_trajectory_prototypes",
            "reward": "focal_team_ball_progress_plus_goals",
            "loss": "bellman_huber_plus_alpha_logsumexp_conservative_penalty",
            "conservative_alpha": conservative_alpha,
            "gamma": gamma,
            "target_update_interval": target_update_interval,
            "selection_metric": "validation_autonomous_ade_meters",
            "epoch_zero": "exact_constant_velocity_baseline",
            "future_observations_used": False,
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
        "maximum_epochs": epochs,
        "epochs_run": len(epoch_losses),
        "selected_epoch": best_epoch,
        "epoch_losses": epoch_losses,
        "validation_history": validation_history,
        "selected_validation": selected_validation,
        "best_validation_autonomous_ade_meters": best_error,
        "epoch_zero_validation_autonomous_ade_meters": float(
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
        "action_count": action_count,
        "prototype_fit": prototype_metrics,
        "conservative_alpha": conservative_alpha,
        "gamma": gamma,
        "target_update_interval": target_update_interval,
        "future_observations_used": False,
        "selection_metric": "validation_autonomous_ade_meters",
    }
