"""Residual next-displacement BC with a persistence-safe epoch zero.

The model predicts a correction to the previous observed player velocity.  A
zero-initialized output head therefore exactly reproduces the declared
previous-velocity baseline before training.  Model selection includes epoch 0
and uses validation mean Euclidean displacement error only.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from thesis_experiments.policies.networks import TemporalResidualDisplacementPredictor
from thesis_experiments.training.team_displacement_bc import (
    METERS_PER_GRF,
    displacement_metrics,
)
from thesis_experiments.transitions.observations import (
    LOCAL_FEATURES,
    LOCAL_FEATURE_SLICES,
    OBSERVATION_VERSION,
    observation_feature_mask,
)


MOTION_FEATURES_PER_STEP = 4  # player dx/dy and ball dx/dy


def _scalar(value: np.ndarray) -> str:
    return str(np.asarray(value).item())


def causal_motion_history(
    observations: np.ndarray, history_steps: int
) -> np.ndarray:
    """Append causal player/ball velocity history in metres, oldest to newest."""

    observations = np.asarray(observations, dtype=np.float32)
    if observations.ndim != 3 or observations.shape[-1] != LOCAL_FEATURES:
        raise ValueError("Expected observations with shape [T,players,61].")
    if history_steps < 1:
        raise ValueError("history_steps must be positive.")
    time_steps, players, _ = observations.shape
    motion = observations[..., 24:28].copy()
    motion[..., :2] *= METERS_PER_GRF
    motion[..., 2:4] *= METERS_PER_GRF
    history = np.zeros(
        (time_steps, players, history_steps, MOTION_FEATURES_PER_STEP),
        dtype=np.float32,
    )
    for history_index in range(history_steps):
        lag = history_steps - 1 - history_index
        if lag < time_steps:
            history[lag:, :, history_index] = motion[: time_steps - lag]
    return history.reshape(time_steps, players, -1)


def load_team_residual_examples(
    path: Path, team_id: str, *, history_steps: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    """Load causal inputs, absolute displacement targets, and persistence."""

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
                f"{path}: residual BC requires {OBSERVATION_VERSION!r}, found {version!r}."
            )
        observations_3d = np.asarray(
            data["local_observations"][:, selected], dtype=np.float32
        )
        next_observations = np.asarray(
            data["next_local_observations"][:, selected], dtype=np.float32
        )
        targets = (next_observations[..., :2] - observations_3d[..., :2]) * METERS_PER_GRF
        persistence = (
            observations_3d[..., LOCAL_FEATURE_SLICES["player_velocity"]]
            * METERS_PER_GRF
        )
        history = causal_motion_history(observations_3d, history_steps)
        inputs = np.concatenate([observations_3d, history], axis=-1)
        metadata = {
            "path": str(path),
            "source": _scalar(data["source"]),
            "match_id": _scalar(data["match_id"]),
            "frames": np.asarray(data["frames"], dtype=np.int64),
            "player_ids": np.asarray(data["player_ids"])[selected].astype(str),
        }
    flat_inputs = inputs.reshape(-1, inputs.shape[-1]).astype(np.float32)
    flat_targets = targets.reshape(-1, 2).astype(np.float32)
    flat_persistence = persistence.reshape(-1, 2).astype(np.float32)
    if not all(
        np.isfinite(values).all()
        for values in (flat_inputs, flat_targets, flat_persistence)
    ):
        raise ValueError(f"{path}: residual examples contain non-finite values.")
    return flat_inputs, flat_targets, flat_persistence, metadata


def residual_input_mask(observation_variant: str, history_steps: int) -> np.ndarray:
    current = observation_feature_mask(observation_variant).astype(np.float32)
    history = np.ones(
        history_steps * MOTION_FEATURES_PER_STEP, dtype=np.float32
    )
    return np.concatenate([current, history])


def _load_model(checkpoint: Path, device: str):
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    model = TemporalResidualDisplacementPredictor(
        input_size=int(saved["input_size"]),
        hidden_size=int(saved["hidden_size"]),
    ).to(device)
    model.load_state_dict(saved["model_state_dict"])
    model.eval()
    return model, np.asarray(saved["input_mask"], dtype=np.float32), saved


def _predict_corrections(
    model: TemporalResidualDisplacementPredictor,
    inputs: np.ndarray,
    input_mask: np.ndarray,
    *,
    batch_size: int,
    device: str,
) -> np.ndarray:
    mask = torch.as_tensor(input_mask, dtype=torch.float32, device=device)
    output = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(inputs), batch_size):
            batch = torch.as_tensor(
                inputs[start : start + batch_size],
                dtype=torch.float32,
                device=device,
            )
            output.append(model(batch * mask).cpu().numpy())
    return np.concatenate(output).astype(np.float32)


def _streaming_metrics(
    model: TemporalResidualDisplacementPredictor,
    paths: list[Path],
    *,
    team_id: str,
    history_steps: int,
    input_mask: np.ndarray,
    batch_size: int,
    device: str,
) -> dict[str, object]:
    policy_errors = []
    baseline_errors = []
    correction_norms = []
    mask = torch.as_tensor(input_mask, dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        for path in paths:
            inputs, targets, persistence, _ = load_team_residual_examples(
                path, team_id, history_steps=history_steps
            )
            for start in range(0, len(targets), batch_size):
                stop = min(start + batch_size, len(targets))
                batch = torch.as_tensor(
                    inputs[start:stop], dtype=torch.float32, device=device
                )
                correction = model(batch * mask).cpu().numpy()
                prediction = persistence[start:stop] + correction
                policy_errors.append(np.linalg.norm(prediction - targets[start:stop], axis=-1))
                baseline_errors.append(
                    np.linalg.norm(persistence[start:stop] - targets[start:stop], axis=-1)
                )
                correction_norms.append(np.linalg.norm(correction, axis=-1))
    policy = np.concatenate(policy_errors)
    baseline = np.concatenate(baseline_errors)
    corrections = np.concatenate(correction_norms)
    return {
        "samples": int(len(policy)),
        "policy_mean_displacement_error_meters": float(policy.mean()),
        "policy_rmse_meters": float(np.sqrt(np.mean(np.square(policy)))),
        "policy_p90_meters": float(np.quantile(policy, 0.9)),
        "previous_velocity_mean_displacement_error_meters": float(baseline.mean()),
        "previous_velocity_rmse_meters": float(np.sqrt(np.mean(np.square(baseline)))),
        "previous_velocity_p90_meters": float(np.quantile(baseline, 0.9)),
        "mean_correction_meters": float(corrections.mean()),
        "maximum_correction_meters": float(corrections.max()),
    }


def train_team_residual_displacement_bc(
    paths: list[Path],
    output: Path,
    *,
    team_id: str,
    validation_paths: list[Path],
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
    """Train residual BC; epoch zero is a selectable persistence checkpoint."""

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
    model = TemporalResidualDisplacementPredictor(
        input_size=len(input_mask), hidden_size=hidden_size
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    sample_count = 0
    matches = set()
    for path in paths:
        _, targets, _, metadata = load_team_residual_examples(
            path, team_id, history_steps=history_steps
        )
        sample_count += len(targets)
        matches.add(str(metadata["match_id"]))

    initial_validation = _streaming_metrics(
        model,
        validation_paths,
        team_id=team_id,
        history_steps=history_steps,
        input_mask=input_mask,
        batch_size=max(batch_size, 8192),
        device=device,
    )
    initial_validation["epoch"] = 0
    validation_history = [initial_validation]
    best_error = float(initial_validation["policy_mean_displacement_error_meters"])
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
            inputs, targets, persistence, _ = load_team_residual_examples(
                paths[int(file_index)], team_id, history_steps=history_steps
            )
            order = rng.permutation(len(targets))
            for start in range(0, len(order), batch_size):
                indices = order[start : start + batch_size]
                batch = torch.as_tensor(
                    inputs[indices], dtype=torch.float32, device=device
                )
                target = torch.as_tensor(
                    targets[indices], dtype=torch.float32, device=device
                )
                baseline = torch.as_tensor(
                    persistence[indices], dtype=torch.float32, device=device
                )
                prediction = baseline + model(batch * mask)
                # Optimize the declared selection metric directly.  The tiny
                # epsilon keeps the derivative finite at an exact match.
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
        validation = _streaming_metrics(
            model,
            validation_paths,
            team_id=team_id,
            history_steps=history_steps,
            input_mask=input_mask,
            batch_size=max(batch_size, 8192),
            device=device,
        )
        validation["epoch"] = epoch
        validation_history.append(validation)
        error = float(validation["policy_mean_displacement_error_meters"])
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
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_size": len(input_mask),
            "hidden_size": hidden_size,
            "history_steps": history_steps,
            "input_mask": torch.from_numpy(input_mask),
            "stage": "single_team_residual_displacement_bc",
            "team_id": team_id,
            "observation_version": OBSERVATION_VERSION,
            "observation_variant": observation_variant,
            "target": "correction_to_previous_velocity_meters",
            "prediction": "previous_velocity_plus_learned_correction",
            "loss": "mean_euclidean_displacement_error_meters",
            "selection_metric": "validation_mean_displacement_error_meters",
            "epoch_zero": "exact_previous_velocity_baseline",
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
        "samples": sample_count,
        "maximum_epochs": epochs,
        "epochs_run": len(epoch_losses),
        "selected_epoch": best_epoch,
        "epoch_losses": epoch_losses,
        "validation_history": validation_history,
        "best_validation_mean_displacement_error_meters": best_error,
        "epoch_zero_validation_mean_displacement_error_meters": float(
            initial_validation["policy_mean_displacement_error_meters"]
        ),
        "strictly_beats_previous_velocity_on_validation": bool(
            best_error
            < float(
                initial_validation[
                    "previous_velocity_mean_displacement_error_meters"
                ]
            )
            - 1e-8
        ),
        "observation_variant": observation_variant,
        "retained_current_features": int(input_mask[:LOCAL_FEATURES].sum()),
        "history_steps": history_steps,
        "temporal_history_features": history_steps * MOTION_FEATURES_PER_STEP,
        "target": "correction_to_previous_velocity_meters",
        "selection_metric": "validation_mean_displacement_error_meters",
    }


def evaluate_team_residual_displacement_bc(
    checkpoint: Path,
    paths: list[Path],
    *,
    team_id: str,
    device: str = "cpu",
    prediction_batch_size: int = 16384,
) -> dict[str, object]:
    """Evaluate a selected residual checkpoint on recorded causal inputs."""

    if not paths:
        raise ValueError("At least one evaluation path is required.")
    model, input_mask, saved = _load_model(Path(checkpoint), device)
    if str(saved["team_id"]) != team_id:
        raise ValueError("Checkpoint and requested team do not match.")
    history_steps = int(saved["history_steps"])
    combined: dict[str, list[np.ndarray]] = defaultdict(list)
    by_match: dict[str, dict[str, list[np.ndarray]]] = defaultdict(
        lambda: defaultdict(list)
    )
    by_player: dict[str, dict[str, list[np.ndarray]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for path in paths:
        inputs, targets, persistence, metadata = load_team_residual_examples(
            path, team_id, history_steps=history_steps
        )
        corrections = _predict_corrections(
            model,
            inputs,
            input_mask,
            batch_size=prediction_batch_size,
            device=device,
        )
        predictions = persistence + corrections
        stationary = np.zeros_like(targets)
        match_id = str(metadata["match_id"])
        for name, values in (
            ("predictions", predictions),
            ("targets", targets),
            ("persistence", persistence),
            ("stationary", stationary),
            ("corrections", corrections),
        ):
            combined[name].append(values)
            by_match[match_id][name].append(values)
        time_steps = len(metadata["frames"])
        reshaped = {
            name: values.reshape(time_steps, 11, 2)
            for name, values in (
                ("predictions", predictions),
                ("targets", targets),
                ("persistence", persistence),
                ("stationary", stationary),
                ("corrections", corrections),
            )
        }
        for player_index, player_id in enumerate(metadata["player_ids"].tolist()):
            for name, values in reshaped.items():
                by_player[str(player_id)][name].append(values[:, player_index])

    def summarize(rows: dict[str, list[np.ndarray]]) -> dict[str, object]:
        targets = np.concatenate(rows["targets"])
        corrections = np.concatenate(rows["corrections"])
        result = {
            "policy": displacement_metrics(np.concatenate(rows["predictions"]), targets),
            "previous_velocity_baseline": displacement_metrics(
                np.concatenate(rows["persistence"]), targets
            ),
            "stationary_baseline": displacement_metrics(
                np.concatenate(rows["stationary"]), targets
            ),
            "mean_correction_meters": float(
                np.linalg.norm(corrections, axis=-1).mean()
            ),
        }
        return result

    return {
        "team_id": team_id,
        "checkpoint": str(checkpoint),
        "selected_epoch": int(saved["selected_epoch"]),
        "epoch_zero": str(saved["epoch_zero"]),
        "selection_metric": str(saved["selection_metric"]),
        "test_used_for_selection": bool(saved["test_used_for_selection"]),
        "history_steps": history_steps,
        "target": str(saved["target"]),
        "overall": summarize(combined),
        "matches": {
            match_id: summarize(rows) for match_id, rows in sorted(by_match.items())
        },
        "players": {
            player_id: summarize(rows) for player_id, rows in sorted(by_player.items())
        },
    }
