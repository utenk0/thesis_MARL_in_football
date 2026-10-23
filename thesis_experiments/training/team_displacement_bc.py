"""Direct next-displacement behavioural cloning for one football team.

This formulation is deliberately separate from discrete-action BC.  It uses
the same causal observation contract, but regresses the tracked displacement
from the current frame to the next frame in metres.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from thesis_experiments.policies.networks import SharedDisplacementPredictor
from thesis_experiments.transitions.observations import (
    LOCAL_FEATURES,
    LOCAL_FEATURE_SLICES,
    OBSERVATION_VERSION,
    observation_feature_mask,
)


METERS_PER_GRF = np.asarray([52.5, 68.0 / 0.84], dtype=np.float32)


def _scalar(value: np.ndarray) -> str:
    return str(np.asarray(value).item())


def load_team_displacement_examples(
    path: Path, team_id: str
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Return observations and causal current-to-next displacement targets."""

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
                f"{path}: displacement BC requires {OBSERVATION_VERSION!r}, "
                f"found {version!r}."
            )
        observations_3d = np.asarray(
            data["local_observations"][:, selected], dtype=np.float32
        )
        next_observations = np.asarray(
            data["next_local_observations"][:, selected], dtype=np.float32
        )
        displacement = (
            next_observations[..., :2] - observations_3d[..., :2]
        ) * METERS_PER_GRF
        metadata = {
            "path": str(path),
            "source": _scalar(data["source"]),
            "match_id": _scalar(data["match_id"]),
            "player_ids": np.asarray(data["player_ids"])[selected].astype(str),
            "frames": np.asarray(data["frames"], dtype=np.int64),
            "positions_meters": observations_3d[..., :2] * METERS_PER_GRF,
            "next_positions_meters": next_observations[..., :2] * METERS_PER_GRF,
        }
    observations = observations_3d.reshape(-1, LOCAL_FEATURES)
    targets = displacement.reshape(-1, 2).astype(np.float32)
    if not np.isfinite(observations).all() or not np.isfinite(targets).all():
        raise ValueError(f"{path}: non-finite displacement examples.")
    return observations, targets, metadata


def displacement_metrics(
    predictions: np.ndarray, targets: np.ndarray
) -> dict[str, float | int]:
    predictions = np.asarray(predictions, dtype=np.float32).reshape(-1, 2)
    targets = np.asarray(targets, dtype=np.float32).reshape(-1, 2)
    if predictions.shape != targets.shape or not len(targets):
        raise ValueError("Predictions and non-empty targets must have shape [N,2].")
    errors = np.linalg.norm(predictions - targets, axis=-1)
    return {
        "samples": int(len(targets)),
        "mean_displacement_error_meters": float(errors.mean()),
        "median_displacement_error_meters": float(np.median(errors)),
        "p90_displacement_error_meters": float(np.quantile(errors, 0.9)),
        "root_mean_squared_displacement_error_meters": float(
            np.sqrt(np.mean(np.square(errors)))
        ),
        "mean_target_displacement_meters": float(
            np.linalg.norm(targets, axis=-1).mean()
        ),
        "mean_predicted_displacement_meters": float(
            np.linalg.norm(predictions, axis=-1).mean()
        ),
    }


def _load_model(checkpoint: Path, device: str):
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    model = SharedDisplacementPredictor(
        observation_size=int(saved["observation_size"]),
        hidden_size=int(saved["hidden_size"]),
    ).to(device)
    model.load_state_dict(saved["model_state_dict"])
    model.eval()
    return model, np.asarray(saved["feature_mask"], dtype=np.float32), saved


def _predict(
    model: SharedDisplacementPredictor,
    observations: np.ndarray,
    feature_mask: np.ndarray,
    *,
    batch_size: int,
    device: str,
) -> np.ndarray:
    mask = torch.as_tensor(feature_mask, dtype=torch.float32, device=device)
    output = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(observations), batch_size):
            batch = torch.as_tensor(
                observations[start : start + batch_size],
                dtype=torch.float32,
                device=device,
            )
            output.append(model(batch * mask).cpu().numpy())
    return np.concatenate(output).astype(np.float32)


def _validation_metrics(
    model: SharedDisplacementPredictor,
    paths: list[Path],
    *,
    team_id: str,
    feature_mask: np.ndarray,
    batch_size: int,
    device: str,
) -> dict[str, float | int]:
    error_sum = 0.0
    squared_error_sum = 0.0
    huber_sum = 0.0
    samples = 0
    mask = torch.as_tensor(feature_mask, dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        for path in paths:
            observations, targets, _ = load_team_displacement_examples(path, team_id)
            for start in range(0, len(targets), batch_size):
                stop = min(start + batch_size, len(targets))
                batch = torch.as_tensor(
                    observations[start:stop], dtype=torch.float32, device=device
                )
                target = torch.as_tensor(
                    targets[start:stop], dtype=torch.float32, device=device
                )
                prediction = model(batch * mask)
                errors = torch.linalg.vector_norm(prediction - target, dim=-1)
                error_sum += float(errors.sum().cpu())
                squared_error_sum += float(torch.square(errors).sum().cpu())
                huber_sum += float(
                    F.smooth_l1_loss(prediction, target, reduction="sum").cpu()
                )
                samples += len(target)
    return {
        "samples": samples,
        "huber_loss_per_sample": huber_sum / samples,
        "mean_displacement_error_meters": error_sum / samples,
        "root_mean_squared_displacement_error_meters": (
            squared_error_sum / samples
        ) ** 0.5,
    }


def train_team_displacement_bc(
    paths: list[Path],
    output: Path,
    *,
    team_id: str,
    validation_paths: list[Path],
    epochs: int = 10,
    batch_size: int = 2048,
    learning_rate: float = 3e-4,
    hidden_size: int = 128,
    seed: int = 0,
    device: str = "cpu",
    observation_variant: str = "temporal_context_no_slot",
    early_stopping_patience: int = 3,
) -> dict[str, object]:
    """Train with Huber loss and select by validation displacement error."""

    if not paths or not validation_paths:
        raise ValueError("Training and validation paths must both be non-empty.")
    if epochs < 1 or batch_size < 1 or learning_rate <= 0 or hidden_size < 1:
        raise ValueError("Training hyperparameters must be positive.")
    if early_stopping_patience < 1:
        raise ValueError("early_stopping_patience must be positive.")
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    feature_mask = observation_feature_mask(observation_variant).astype(np.float32)
    mask = torch.as_tensor(feature_mask, dtype=torch.float32, device=device)
    model = SharedDisplacementPredictor(
        observation_size=LOCAL_FEATURES, hidden_size=hidden_size
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    sample_count = 0
    matches = set()
    for path in paths:
        _, targets, metadata = load_team_displacement_examples(path, team_id)
        sample_count += len(targets)
        matches.add(str(metadata["match_id"]))

    epoch_losses = []
    validation_history = []
    best_error = float("inf")
    best_epoch = 0
    best_state = None
    without_improvement = 0
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        seen = 0
        for file_index in rng.permutation(len(paths)):
            observations, targets, _ = load_team_displacement_examples(
                paths[int(file_index)], team_id
            )
            order = rng.permutation(len(targets))
            for start in range(0, len(order), batch_size):
                indices = order[start : start + batch_size]
                batch = torch.as_tensor(
                    observations[indices], dtype=torch.float32, device=device
                )
                target = torch.as_tensor(
                    targets[indices], dtype=torch.float32, device=device
                )
                loss = F.smooth_l1_loss(model(batch * mask), target)
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
            feature_mask=feature_mask,
            batch_size=max(batch_size, 8192),
            device=device,
        )
        validation["epoch"] = epoch
        validation_history.append(validation)
        error = float(validation["mean_displacement_error_meters"])
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
        if without_improvement >= early_stopping_patience:
            break

    assert best_state is not None
    model.load_state_dict(best_state)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "observation_size": LOCAL_FEATURES,
            "hidden_size": hidden_size,
            "stage": "single_team_displacement_bc",
            "team_id": team_id,
            "observation_version": OBSERVATION_VERSION,
            "observation_variant": observation_variant,
            "feature_mask": torch.from_numpy(feature_mask),
            "target": "next_player_displacement_meters",
            "target_scale_meters_per_grf": torch.from_numpy(METERS_PER_GRF),
            "loss": "smooth_l1",
            "selection_metric": "validation_mean_displacement_error_meters",
            "training_matches": sorted(matches),
            "selected_epoch": best_epoch,
            "validation_selection": True,
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
        "early_stopping_patience": early_stopping_patience,
        "observation_variant": observation_variant,
        "retained_features": int(feature_mask.sum()),
        "target": "next_player_displacement_meters",
        "loss": "smooth_l1",
        "selection_metric": "validation_mean_displacement_error_meters",
    }


def _cumulative_window_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    starts: np.ndarray,
    *,
    horizon: int,
) -> dict[str, float | int]:
    ade = []
    fde = []
    for start in starts.tolist():
        predicted_positions = np.cumsum(predictions[start : start + horizon], axis=0)
        target_positions = np.cumsum(targets[start : start + horizon], axis=0)
        errors = np.linalg.norm(predicted_positions - target_positions, axis=-1)
        ade.append(float(errors.mean()))
        fde.append(float(errors[-1].mean()))
    return {
        "windows": len(ade),
        "horizon_steps": horizon,
        "ade_meters": float(np.mean(ade)),
        "fde_meters": float(np.mean(fde)),
    }


def evaluate_team_displacement_bc(
    checkpoint: Path,
    paths: list[Path],
    *,
    team_id: str,
    device: str = "cpu",
    prediction_batch_size: int = 16384,
    cumulative_horizon: int = 100,
) -> dict[str, object]:
    """Evaluate one-step displacement and teacher-forced cumulative trajectories."""

    if not paths:
        raise ValueError("At least one evaluation path is required.")
    model, feature_mask, saved = _load_model(Path(checkpoint), device)
    if str(saved["team_id"]) != team_id:
        raise ValueError("Checkpoint and requested team do not match.")
    combined: dict[str, list[np.ndarray]] = defaultdict(list)
    by_match: dict[str, dict[str, list[np.ndarray]]] = defaultdict(
        lambda: defaultdict(list)
    )
    cumulative = {"policy": [], "stationary": [], "previous_velocity": []}
    by_player: dict[str, dict[str, list[np.ndarray]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for path in paths:
        observations, targets, metadata = load_team_displacement_examples(path, team_id)
        predictions = _predict(
            model,
            observations,
            feature_mask,
            batch_size=prediction_batch_size,
            device=device,
        )
        previous_velocity = (
            observations[:, LOCAL_FEATURE_SLICES["player_velocity"]] * METERS_PER_GRF
        )
        stationary = np.zeros_like(targets)
        match_id = str(metadata["match_id"])
        for name, values in (
            ("predictions", predictions),
            ("targets", targets),
            ("stationary", stationary),
            ("previous_velocity", previous_velocity),
        ):
            combined[name].append(values)
            by_match[match_id][name].append(values)

        time_steps = len(metadata["frames"])
        predictions_3d = predictions.reshape(time_steps, 11, 2)
        targets_3d = targets.reshape(time_steps, 11, 2)
        stationary_3d = stationary.reshape(time_steps, 11, 2)
        velocity_3d = previous_velocity.reshape(time_steps, 11, 2)
        for player_index, player_id in enumerate(metadata["player_ids"].tolist()):
            for name, values in (
                ("predictions", predictions_3d[:, player_index]),
                ("targets", targets_3d[:, player_index]),
                ("stationary", stationary_3d[:, player_index]),
                ("previous_velocity", velocity_3d[:, player_index]),
            ):
                by_player[str(player_id)][name].append(values)

        if cumulative_horizon > 0 and time_steps >= cumulative_horizon:
            starts = np.arange(0, time_steps - cumulative_horizon + 1, cumulative_horizon)
            for name, values in (
                ("policy", predictions_3d),
                ("stationary", stationary_3d),
                ("previous_velocity", velocity_3d),
            ):
                cumulative[name].append(
                    _cumulative_window_metrics(
                        values, targets_3d, starts, horizon=cumulative_horizon
                    )
                )

    def summarize(rows: dict[str, list[np.ndarray]]) -> dict[str, object]:
        targets = np.concatenate(rows["targets"])
        return {
            "policy": displacement_metrics(np.concatenate(rows["predictions"]), targets),
            "stationary_baseline": displacement_metrics(
                np.concatenate(rows["stationary"]), targets
            ),
            "previous_velocity_baseline": displacement_metrics(
                np.concatenate(rows["previous_velocity"]), targets
            ),
        }

    cumulative_summary = {}
    for name, rows in cumulative.items():
        if rows:
            window_count = sum(int(row["windows"]) for row in rows)
            cumulative_summary[name] = {
                "windows": window_count,
                "horizon_steps": cumulative_horizon,
                "ade_meters": float(
                    sum(row["ade_meters"] * row["windows"] for row in rows)
                    / window_count
                ),
                "fde_meters": float(
                    sum(row["fde_meters"] * row["windows"] for row in rows)
                    / window_count
                ),
            }
    return {
        "team_id": team_id,
        "checkpoint": str(checkpoint),
        "selected_epoch": int(saved["selected_epoch"]),
        "selection_metric": str(saved["selection_metric"]),
        "observation_variant": str(saved["observation_variant"]),
        "target": str(saved["target"]),
        "overall": summarize(combined),
        "matches": {
            match_id: summarize(rows) for match_id, rows in sorted(by_match.items())
        },
        "players": {
            player_id: summarize(rows) for player_id, rows in sorted(by_player.items())
        },
        "teacher_forced_cumulative": cumulative_summary,
        "scope": (
            "One-step metrics use recorded causal observations. Cumulative metrics "
            "integrate predictions made from each recorded future observation and are "
            "therefore teacher-forced diagnostics, not autonomous closed-loop rollouts."
        ),
    }
