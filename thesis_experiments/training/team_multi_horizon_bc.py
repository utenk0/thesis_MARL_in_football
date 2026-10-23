"""Autonomous multi-horizon team trajectory BC.

Each prediction uses only the observation and causal motion history available
at the forecast origin.  The network predicts corrections to a constant-
velocity forecast at fixed horizons.  Its zero-initialized output makes epoch
zero exactly equal to that baseline and eligible for validation selection.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from thesis_experiments.training.team_displacement_bc import METERS_PER_GRF
from thesis_experiments.training.team_residual_displacement_bc import (
    causal_motion_history,
    residual_input_mask,
)
from thesis_experiments.transitions.observations import (
    LOCAL_FEATURES,
    LOCAL_FEATURE_SLICES,
    OBSERVATION_VERSION,
)


DEFAULT_HORIZONS = (1, 5, 10, 25, 50, 100)


class MultiHorizonResidualPredictor(nn.Module):
    """Predict corrections to constant velocity at several future horizons."""

    def __init__(self, input_size: int, horizons: tuple[int, ...], hidden_size: int = 128):
        super().__init__()
        self.input_size = input_size
        self.horizons = tuple(horizons)
        self.hidden_size = hidden_size
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, len(horizons) * 2),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs).reshape(-1, len(self.horizons), 2)


def _scalar(value: np.ndarray) -> str:
    return str(np.asarray(value).item())


def _validate_horizons(horizons: tuple[int, ...]) -> None:
    if not horizons or any(value < 1 for value in horizons):
        raise ValueError("Horizons must be positive.")
    if tuple(sorted(set(horizons))) != horizons:
        raise ValueError("Horizons must be unique and increasing.")


def load_team_multi_horizon_examples(
    path: Path,
    team_id: str,
    *,
    history_steps: int,
    horizons: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    """Load causal forecast origins and future relative-position targets."""

    _validate_horizons(horizons)
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        team_ids = np.asarray(data["team_ids"]).astype(str)
        selected = np.flatnonzero(team_ids == team_id)
        if len(selected) != 11:
            raise ValueError(f"{path}: expected 11 slots for team {team_id!r}.")
        version = (
            _scalar(data["observation_version"])
            if "observation_version" in data.files
            else "local_v1_spatial_24"
        )
        if version != OBSERVATION_VERSION:
            raise ValueError(
                f"{path}: multi-horizon BC requires {OBSERVATION_VERSION!r}, found {version!r}."
            )
        observations = np.asarray(
            data["local_observations"][:, selected], dtype=np.float32
        )
        next_observations = np.asarray(
            data["next_local_observations"][:, selected], dtype=np.float32
        )
        frames = np.asarray(data["frames"], dtype=np.int64)
        player_ids = np.asarray(data["player_ids"])[selected].astype(str)
        match_id = _scalar(data["match_id"])
        source = _scalar(data["source"])

    max_horizon = horizons[-1]
    starts = len(observations) - max_horizon + 1
    if starts < 1:
        raise ValueError(f"{path}: fewer than {max_horizon} forecastable transitions.")
    current_observations = observations[:starts]
    history = causal_motion_history(observations, history_steps)[:starts]
    inputs = np.concatenate([current_observations, history], axis=-1)
    current_positions = current_observations[..., :2] * METERS_PER_GRF
    velocity = (
        current_observations[..., LOCAL_FEATURE_SLICES["player_velocity"]]
        * METERS_PER_GRF
    )
    previous_velocity = np.zeros_like(velocity)
    if starts > 1:
        previous_velocity[1:] = velocity[:-1]
    acceleration = velocity - previous_velocity

    target_displacements = []
    constant_velocity = []
    constant_acceleration = []
    for horizon in horizons:
        future_positions = (
            next_observations[horizon - 1 : horizon - 1 + starts, ..., :2]
            * METERS_PER_GRF
        )
        target_displacements.append(future_positions - current_positions)
        constant_velocity.append(float(horizon) * velocity)
        discrete_acceleration_factor = horizon * (horizon + 1) / 2.0
        constant_acceleration.append(
            float(horizon) * velocity
            + float(discrete_acceleration_factor) * acceleration
        )

    targets = np.stack(target_displacements, axis=2).astype(np.float32)
    velocity_baseline = np.stack(constant_velocity, axis=2).astype(np.float32)
    acceleration_baseline = np.stack(constant_acceleration, axis=2).astype(np.float32)
    metadata = {
        "path": str(path),
        "source": source,
        "match_id": match_id,
        "frames": frames[:starts],
        "player_ids": player_ids,
        "current_positions_meters": current_positions,
        "target_positions_meters": current_positions[:, :, None, :] + targets,
        "time_starts": starts,
    }
    flat_inputs = inputs.reshape(-1, inputs.shape[-1]).astype(np.float32)
    shape = (-1, len(horizons), 2)
    return (
        flat_inputs,
        targets.reshape(shape),
        velocity_baseline.reshape(shape),
        acceleration_baseline.reshape(shape),
        metadata,
    )


def _load_model(checkpoint: Path, device: str):
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    horizons = tuple(int(value) for value in saved["horizons"])
    model = MultiHorizonResidualPredictor(
        input_size=int(saved["input_size"]),
        horizons=horizons,
        hidden_size=int(saved["hidden_size"]),
    ).to(device)
    model.load_state_dict(saved["model_state_dict"])
    model.eval()
    return model, np.asarray(saved["input_mask"], dtype=np.float32), saved


def _predict_corrections(
    model: MultiHorizonResidualPredictor,
    inputs: np.ndarray,
    input_mask: np.ndarray,
    *,
    batch_size: int,
    device: str,
) -> np.ndarray:
    mask = torch.as_tensor(input_mask, dtype=torch.float32, device=device)
    parts = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(inputs), batch_size):
            batch = torch.as_tensor(
                inputs[start : start + batch_size],
                dtype=torch.float32,
                device=device,
            )
            parts.append(model(batch * mask).cpu().numpy())
    return np.concatenate(parts).astype(np.float32)


class _MetricAccumulator:
    def __init__(self, horizons: tuple[int, ...]):
        self.horizons = horizons
        self.player_count = np.zeros(len(horizons), dtype=np.int64)
        self.error_sum = np.zeros(len(horizons), dtype=np.float64)
        self.error_square_sum = np.zeros(len(horizons), dtype=np.float64)
        self.team_count = np.zeros(len(horizons), dtype=np.int64)
        self.centroid_sum = np.zeros(len(horizons), dtype=np.float64)
        self.width_sum = np.zeros(len(horizons), dtype=np.float64)
        self.depth_sum = np.zeros(len(horizons), dtype=np.float64)

    def update(
        self,
        predicted_displacements: np.ndarray,
        target_displacements: np.ndarray,
        current_positions: np.ndarray,
    ) -> None:
        starts = len(current_positions)
        predicted = predicted_displacements.reshape(starts, 11, len(self.horizons), 2)
        targets = target_displacements.reshape(starts, 11, len(self.horizons), 2)
        errors = np.linalg.norm(predicted - targets, axis=-1)
        self.player_count += starts * 11
        self.error_sum += errors.sum(axis=(0, 1))
        self.error_square_sum += np.square(errors).sum(axis=(0, 1))

        predicted_positions = current_positions[:, :, None, :] + predicted
        target_positions = current_positions[:, :, None, :] + targets
        predicted_centroid = predicted_positions.mean(axis=1)
        target_centroid = target_positions.mean(axis=1)
        centroid = np.linalg.norm(predicted_centroid - target_centroid, axis=-1)
        predicted_depth = np.ptp(predicted_positions[..., 0], axis=1)
        target_depth = np.ptp(target_positions[..., 0], axis=1)
        predicted_width = np.ptp(predicted_positions[..., 1], axis=1)
        target_width = np.ptp(target_positions[..., 1], axis=1)
        self.team_count += starts
        self.centroid_sum += centroid.sum(axis=0)
        self.depth_sum += np.abs(predicted_depth - target_depth).sum(axis=0)
        self.width_sum += np.abs(predicted_width - target_width).sum(axis=0)

    def result(self) -> dict[str, object]:
        mean_errors = self.error_sum / self.player_count
        rmse = np.sqrt(self.error_square_sum / self.player_count)
        per_horizon = []
        for index, horizon in enumerate(self.horizons):
            per_horizon.append(
                {
                    "horizon_steps": horizon,
                    "player_samples": int(self.player_count[index]),
                    "mean_position_error_meters": float(mean_errors[index]),
                    "rmse_meters": float(rmse[index]),
                    "centroid_error_meters": float(
                        self.centroid_sum[index] / self.team_count[index]
                    ),
                    "team_width_mae_meters": float(
                        self.width_sum[index] / self.team_count[index]
                    ),
                    "team_depth_mae_meters": float(
                        self.depth_sum[index] / self.team_count[index]
                    ),
                }
            )
        return {
            "forecast_origins": int(self.team_count[0]),
            "player_forecasts": int(self.player_count.sum()),
            "autonomous_ade_meters": float(mean_errors.mean()),
            "autonomous_fde_meters": float(mean_errors[-1]),
            "per_horizon": per_horizon,
        }


def evaluate_team_multi_horizon_bc(
    checkpoint: Path,
    paths: list[Path],
    *,
    team_id: str,
    device: str = "cpu",
    prediction_batch_size: int = 16384,
) -> dict[str, object]:
    """Evaluate direct forecasts that receive no observations after their origin."""

    if not paths:
        raise ValueError("At least one evaluation path is required.")
    model, input_mask, saved = _load_model(Path(checkpoint), device)
    horizons = tuple(int(value) for value in saved["horizons"])
    history_steps = int(saved["history_steps"])
    accumulators = {
        name: _MetricAccumulator(horizons)
        for name in ("policy", "stationary", "constant_velocity", "constant_acceleration")
    }
    matches: dict[str, dict[str, _MetricAccumulator]] = {}
    for path in paths:
        inputs, targets, velocity, acceleration, metadata = load_team_multi_horizon_examples(
            path,
            team_id,
            history_steps=history_steps,
            horizons=horizons,
        )
        corrections = _predict_corrections(
            model,
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
            {name: _MetricAccumulator(horizons) for name in accumulators},
        )
        for name, prediction in predictors.items():
            accumulators[name].update(
                prediction, targets, metadata["current_positions_meters"]
            )
            match_accumulators[name].update(
                prediction, targets, metadata["current_positions_meters"]
            )
    return {
        "team_id": team_id,
        "checkpoint": str(checkpoint),
        "selected_epoch": int(saved["selected_epoch"]),
        "epoch_zero": str(saved["epoch_zero"]),
        "horizons": list(horizons),
        "history_steps": history_steps,
        "selection_metric": str(saved["selection_metric"]),
        "test_used_for_selection": bool(saved["test_used_for_selection"]),
        "future_observations_used": False,
        "overall": {name: values.result() for name, values in accumulators.items()},
        "matches": {
            match_id: {name: values.result() for name, values in rows.items()}
            for match_id, rows in sorted(matches.items())
        },
        "scope": (
            "Every horizon is predicted directly from the forecast-origin observation "
            "and causal past motion history. No future observation, future velocity, "
            "target action, or simulator state is supplied."
        ),
    }


def _validation_metrics(
    model: MultiHorizonResidualPredictor,
    paths: list[Path],
    *,
    team_id: str,
    history_steps: int,
    horizons: tuple[int, ...],
    input_mask: np.ndarray,
    batch_size: int,
    device: str,
) -> dict[str, float | int]:
    policy_sum = np.zeros(len(horizons), dtype=np.float64)
    velocity_sum = np.zeros(len(horizons), dtype=np.float64)
    counts = np.zeros(len(horizons), dtype=np.int64)
    mask = torch.as_tensor(input_mask, dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        for path in paths:
            inputs, targets, velocity, _, _ = load_team_multi_horizon_examples(
                path,
                team_id,
                history_steps=history_steps,
                horizons=horizons,
            )
            for start in range(0, len(inputs), batch_size):
                stop = min(start + batch_size, len(inputs))
                batch = torch.as_tensor(
                    inputs[start:stop], dtype=torch.float32, device=device
                )
                correction = model(batch * mask).cpu().numpy()
                policy_errors = np.linalg.norm(
                    velocity[start:stop] + correction - targets[start:stop], axis=-1
                )
                velocity_errors = np.linalg.norm(
                    velocity[start:stop] - targets[start:stop], axis=-1
                )
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
    }


def train_team_multi_horizon_bc(
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
) -> dict[str, object]:
    """Train direct multi-horizon corrections with validation-only selection."""

    _validate_horizons(horizons)
    if not paths or not validation_paths:
        raise ValueError("Training and validation paths must both be non-empty.")
    if min(epochs, batch_size, hidden_size, history_steps, early_stopping_patience) < 1:
        raise ValueError("Training parameters must be positive.")
    if learning_rate <= 0 or minimum_epochs < 0 or minimum_epochs > epochs:
        raise ValueError("Invalid learning rate or minimum_epochs.")
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    input_mask = residual_input_mask(observation_variant, history_steps)
    mask = torch.as_tensor(input_mask, dtype=torch.float32, device=device)
    model = MultiHorizonResidualPredictor(
        input_size=len(input_mask), horizons=horizons, hidden_size=hidden_size
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    sample_count = 0
    matches = set()
    for path in paths:
        inputs, _, _, _, metadata = load_team_multi_horizon_examples(
            path, team_id, history_steps=history_steps, horizons=horizons
        )
        sample_count += len(inputs)
        matches.add(str(metadata["match_id"]))

    initial = _validation_metrics(
        model,
        validation_paths,
        team_id=team_id,
        history_steps=history_steps,
        horizons=horizons,
        input_mask=input_mask,
        batch_size=max(batch_size, 8192),
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
    epoch_losses = []

    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        seen = 0
        for file_index in rng.permutation(len(paths)):
            inputs, targets, velocity, _, _ = load_team_multi_horizon_examples(
                paths[int(file_index)],
                team_id,
                history_steps=history_steps,
                horizons=horizons,
            )
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
                prediction = baseline + model(batch * mask)
                errors = torch.sqrt(
                    torch.sum(torch.square(prediction - target), dim=-1) + 1e-12
                )
                loss = errors.mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                count = len(indices)
                loss_sum += float(loss.detach().cpu()) * count
                seen += count
        epoch_losses.append(loss_sum / seen)
        validation = _validation_metrics(
            model,
            validation_paths,
            team_id=team_id,
            history_steps=history_steps,
            horizons=horizons,
            input_mask=input_mask,
            batch_size=max(batch_size, 8192),
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
            "stage": "single_team_multi_horizon_bc",
            "team_id": team_id,
            "observation_version": OBSERVATION_VERSION,
            "observation_variant": observation_variant,
            "target": "future_player_positions_relative_to_forecast_origin_meters",
            "prediction": "constant_velocity_plus_learned_horizon_corrections",
            "loss": "mean_euclidean_position_error_across_horizons_meters",
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
        "future_observations_used": False,
        "selection_metric": "validation_autonomous_ade_meters",
    }
