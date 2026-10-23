"""Streaming behavioural cloning for one identified football team.

The general project BC trainer flattens both teams into one shared-policy
dataset.  This module keeps the same shared actor, but selects only the eleven
players belonging to one stable ``team_id``.  Files are loaded one at a time so
complete-match experiments do not require all observations in memory at once.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from thesis_experiments.policies.networks import SharedActor
from thesis_experiments.transitions.observations import (
    ACTION_COUNT,
    LOCAL_FEATURES,
    LOCAL_FEATURE_SLICES,
    OBSERVATION_VERSION,
    observation_feature_mask,
)


def _scalar(value: np.ndarray) -> str:
    return str(np.asarray(value).item())


def inspect_team_chunk(path: Path, team_id: str) -> dict[str, object]:
    """Return identity metadata without decompressing the large observations."""

    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        ids = np.asarray(data["team_ids"]).astype(str)
        selected = np.flatnonzero(ids == team_id)
        if len(selected) != 11:
            raise ValueError(
                f"{path}: expected team {team_id!r} to have 11 player slots, "
                f"found {len(selected)}."
            )
        version = (
            _scalar(data["observation_version"])
            if "observation_version" in data.files
            else "local_v1_spatial_24"
        )
        return {
            "path": str(path),
            "source": _scalar(data["source"]),
            "match_id": _scalar(data["match_id"]),
            "team_id": team_id,
            "player_indices": selected.tolist(),
            "side": "left" if bool(np.all(selected < 11)) else "right",
            "transitions": int(data["actions"].shape[0]),
            "observation_version": version,
        }


def load_team_examples(path: Path, team_id: str) -> tuple[np.ndarray, np.ndarray]:
    """Load flattened observations/actions for the requested team's 11 slots."""

    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        ids = np.asarray(data["team_ids"]).astype(str)
        selected = np.flatnonzero(ids == team_id)
        if len(selected) != 11:
            raise ValueError(
                f"{path}: expected team {team_id!r} to have 11 player slots, "
                f"found {len(selected)}."
            )
        version = (
            _scalar(data["observation_version"])
            if "observation_version" in data.files
            else "local_v1_spatial_24"
        )
        if version != OBSERVATION_VERSION:
            raise ValueError(
                f"{path}: one-team BC requires {OBSERVATION_VERSION!r}, found {version!r}."
            )
        observations = np.asarray(
            data["local_observations"][:, selected], dtype=np.float32
        ).reshape(-1, LOCAL_FEATURES)
        actions = np.asarray(data["actions"][:, selected], dtype=np.int64).reshape(-1)
    if len(observations) != len(actions):
        raise ValueError(f"{path}: team observations and actions do not align.")
    if not np.isfinite(observations).all():
        raise ValueError(f"{path}: team observations contain non-finite values.")
    if np.any((actions < 0) | (actions >= ACTION_COUNT)):
        raise ValueError(f"{path}: team actions must be in [0,{ACTION_COUNT - 1}].")
    return observations, actions


def _action_metrics(predictions: np.ndarray, labels: np.ndarray) -> dict[str, object]:
    predictions = np.asarray(predictions, dtype=np.int64).reshape(-1)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if predictions.shape != labels.shape or not len(labels):
        raise ValueError("Predictions and non-empty labels must have identical shapes.")
    recalls: list[float] = []
    per_action = []
    for action in range(ACTION_COUNT):
        mask = labels == action
        recall = float((predictions[mask] == action).mean()) if mask.any() else None
        if recall is not None:
            recalls.append(recall)
        per_action.append(
            {
                "action": action,
                "support": int(mask.sum()),
                "predicted": int((predictions == action).sum()),
                "recall": recall,
            }
        )
    return {
        "action_accuracy": float((predictions == labels).mean()),
        "balanced_action_accuracy": float(np.mean(recalls)),
        "samples": int(len(labels)),
        "expert_action_counts": np.bincount(labels, minlength=ACTION_COUNT).tolist(),
        "predicted_action_counts": np.bincount(
            predictions, minlength=ACTION_COUNT
        ).tolist(),
        "per_action": per_action,
    }


def _load_actor(checkpoint: Path, device: str):
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    actor = SharedActor(
        observation_size=int(saved["observation_size"]),
        action_size=int(saved["action_size"]),
        hidden_size=int(saved["hidden_size"]),
    ).to(device)
    actor.load_state_dict(saved["actor_state_dict"])
    actor.eval()
    feature_mask = np.asarray(saved["feature_mask"], dtype=np.float32)
    return actor, feature_mask, saved


def _streaming_actor_metrics(
    actor: SharedActor,
    paths: list[Path],
    *,
    team_id: str,
    feature_mask: np.ndarray,
    batch_size: int,
    device: str,
) -> dict[str, float | int]:
    """Compute validation loss/accuracy without retaining all predictions."""

    confusion = np.zeros((ACTION_COUNT, ACTION_COUNT), dtype=np.int64)
    loss_sum = 0.0
    samples = 0
    change_correct = 0
    change_samples = 0
    mask_tensor = torch.as_tensor(feature_mask, dtype=torch.float32, device=device)
    actor.eval()
    with torch.no_grad():
        for path in paths:
            observations, labels = load_team_examples(path, team_id)
            previous = observations[:, LOCAL_FEATURE_SLICES["previous_action"]].argmax(-1)
            for start in range(0, len(labels), batch_size):
                stop = min(start + batch_size, len(labels))
                observation_batch = torch.as_tensor(
                    observations[start:stop], dtype=torch.float32, device=device
                )
                label_batch = torch.as_tensor(
                    labels[start:stop], dtype=torch.int64, device=device
                )
                logits = actor(observation_batch * mask_tensor)
                loss_sum += float(
                    F.cross_entropy(logits, label_batch, reduction="sum").detach()
                )
                predictions = logits.argmax(-1).cpu().numpy()
                batch_labels = labels[start:stop]
                np.add.at(confusion, (batch_labels, predictions), 1)
                changed = batch_labels != previous[start:stop]
                change_samples += int(changed.sum())
                change_correct += int((predictions[changed] == batch_labels[changed]).sum())
                samples += len(batch_labels)
    support = confusion.sum(axis=1)
    observed = support > 0
    recalls = np.divide(
        np.diag(confusion),
        support,
        out=np.zeros(ACTION_COUNT, dtype=np.float64),
        where=observed,
    )
    return {
        "loss": loss_sum / samples,
        "action_accuracy": float(np.trace(confusion) / samples),
        "balanced_action_accuracy": float(recalls[observed].mean()),
        "action_change_accuracy": (
            float(change_correct / change_samples) if change_samples else 0.0
        ),
        "samples": samples,
        "action_change_samples": change_samples,
    }


def train_team_bc(
    paths: list[Path],
    output: Path,
    *,
    team_id: str,
    epochs: int = 20,
    batch_size: int = 512,
    learning_rate: float = 3e-4,
    hidden_size: int = 128,
    seed: int = 0,
    device: str = "cpu",
    observation_variant: str = "temporal_context",
    class_weighting: str = "none",
    max_class_weight: float = 20.0,
    validation_paths: list[Path] | None = None,
    early_stopping_patience: int = 3,
) -> dict[str, object]:
    """Train one shared 11-player team policy with chunk-streamed SGD."""

    if not paths:
        raise ValueError("At least one training chunk is required.")
    if epochs < 1 or batch_size < 1 or learning_rate <= 0 or hidden_size < 1:
        raise ValueError("Epochs, batch size, learning rate and hidden size must be positive.")
    if class_weighting not in {"none", "inverse_frequency"}:
        raise ValueError("class_weighting must be 'none' or 'inverse_frequency'.")
    if max_class_weight <= 0:
        raise ValueError("max_class_weight must be positive.")
    if validation_paths is not None and not validation_paths:
        raise ValueError("validation_paths must be non-empty when provided.")
    if early_stopping_patience < 1:
        raise ValueError("early_stopping_patience must be positive.")

    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    feature_mask = observation_feature_mask(observation_variant).astype(np.float32)

    class_counts = np.zeros(ACTION_COUNT, dtype=np.int64)
    matches: set[str] = set()
    sample_count = 0
    for path in paths:
        metadata = inspect_team_chunk(path, team_id)
        matches.add(str(metadata["match_id"]))
        _, actions = load_team_examples(path, team_id)
        class_counts += np.bincount(actions, minlength=ACTION_COUNT)
        sample_count += len(actions)

    class_weights = np.ones(ACTION_COUNT, dtype=np.float32)
    if class_weighting == "inverse_frequency":
        observed = class_counts > 0
        class_weights.fill(0.0)
        class_weights[observed] = sample_count / (
            int(observed.sum()) * class_counts[observed]
        )
        class_weights = np.minimum(class_weights, max_class_weight)

    actor = SharedActor(LOCAL_FEATURES, hidden_size=hidden_size).to(device)
    optimizer = torch.optim.Adam(actor.parameters(), lr=learning_rate)
    mask_tensor = torch.as_tensor(feature_mask, dtype=torch.float32, device=device)
    weight_tensor = torch.as_tensor(class_weights, dtype=torch.float32, device=device)
    epoch_losses: list[float] = []
    validation_history: list[dict[str, float | int]] = []
    best_validation_loss = float("inf")
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0

    actor.train()
    for epoch in range(1, epochs + 1):
        loss_sum = 0.0
        seen = 0
        for file_index in rng.permutation(len(paths)):
            observations, actions = load_team_examples(paths[int(file_index)], team_id)
            order = rng.permutation(len(actions))
            for start in range(0, len(order), batch_size):
                indices = order[start : start + batch_size]
                observation_batch = torch.as_tensor(
                    observations[indices], dtype=torch.float32, device=device
                )
                observation_batch = observation_batch * mask_tensor
                action_batch = torch.as_tensor(
                    actions[indices], dtype=torch.int64, device=device
                )
                loss = F.cross_entropy(
                    actor(observation_batch),
                    action_batch,
                    weight=weight_tensor if class_weighting == "inverse_frequency" else None,
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                count = len(indices)
                loss_sum += float(loss.detach()) * count
                seen += count
        epoch_losses.append(loss_sum / seen)
        if validation_paths is not None:
            validation_metrics = _streaming_actor_metrics(
                actor,
                validation_paths,
                team_id=team_id,
                feature_mask=feature_mask,
                batch_size=max(batch_size, 4096),
                device=device,
            )
            validation_metrics["epoch"] = epoch
            validation_history.append(validation_metrics)
            if float(validation_metrics["loss"]) < best_validation_loss - 1e-8:
                best_validation_loss = float(validation_metrics["loss"])
                best_epoch = epoch
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in actor.state_dict().items()
                }
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            actor.train()
            if epochs_without_improvement >= early_stopping_patience:
                break

    if validation_paths is None:
        best_epoch = len(epoch_losses)
        best_state = {
            name: value.detach().cpu().clone() for name, value in actor.state_dict().items()
        }
    assert best_state is not None
    actor.load_state_dict(best_state)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "actor_state_dict": actor.state_dict(),
            "observation_size": LOCAL_FEATURES,
            "action_size": ACTION_COUNT,
            "hidden_size": hidden_size,
            "stage": "single_team_bc",
            "team_id": team_id,
            "observation_version": OBSERVATION_VERSION,
            "observation_variant": observation_variant,
            "feature_mask": torch.from_numpy(feature_mask),
            "class_weighting": class_weighting,
            "max_class_weight": float(max_class_weight),
            "class_counts": torch.from_numpy(class_counts),
            "class_weights": torch.from_numpy(class_weights),
            "training_matches": sorted(matches),
            "training_chunks": [str(Path(path)) for path in paths],
            "selected_epoch": best_epoch,
            "validation_selection": validation_paths is not None,
            "seed": seed,
        },
        output,
    )
    return {
        "team_id": team_id,
        "seed": seed,
        "matches": sorted(matches),
        "chunks": len(paths),
        "samples": sample_count,
        "maximum_epochs": epochs,
        "epochs_run": len(epoch_losses),
        "selected_epoch": best_epoch,
        "epoch_losses": epoch_losses,
        "selected_training_epoch_loss": epoch_losses[best_epoch - 1],
        "validation_history": validation_history,
        "best_validation_loss": (
            best_validation_loss if validation_paths is not None else None
        ),
        "early_stopping_patience": early_stopping_patience,
        "observation_variant": observation_variant,
        "retained_features": int(feature_mask.sum()),
        "class_weighting": class_weighting,
        "class_counts": class_counts.tolist(),
        "observed_action_classes": int((class_counts > 0).sum()),
    }


def evaluate_team_bc(
    checkpoint: Path,
    paths: list[Path],
    *,
    team_id: str,
    training_majority_action: int,
    device: str = "cpu",
    prediction_batch_size: int = 4096,
) -> dict[str, object]:
    """Evaluate one team, per match and overall, without loading all files together."""

    if not paths:
        raise ValueError("At least one evaluation chunk is required.")
    actor, feature_mask, saved = _load_actor(Path(checkpoint), device)
    if str(saved["team_id"]) != team_id:
        raise ValueError(
            f"Checkpoint team {saved['team_id']!r} does not match requested team {team_id!r}."
        )
    mask_tensor = torch.as_tensor(feature_mask, dtype=torch.float32, device=device)
    aggregate: dict[str, list[np.ndarray]] = defaultdict(list)
    by_match: dict[str, dict[str, list[np.ndarray]]] = defaultdict(
        lambda: defaultdict(list)
    )
    sources: dict[str, str] = {}

    with torch.no_grad():
        for path in paths:
            metadata = inspect_team_chunk(path, team_id)
            match_id = str(metadata["match_id"])
            sources[match_id] = str(metadata["source"])
            observations, labels = load_team_examples(path, team_id)
            predicted_batches = []
            for start in range(0, len(labels), prediction_batch_size):
                batch = torch.as_tensor(
                    observations[start : start + prediction_batch_size],
                    dtype=torch.float32,
                    device=device,
                )
                predicted_batches.append(
                    actor(batch * mask_tensor).argmax(-1).cpu().numpy()
                )
            predictions = np.concatenate(predicted_batches).astype(np.int64)
            previous = observations[:, LOCAL_FEATURE_SLICES["previous_action"]].argmax(-1)
            changes = labels != previous
            for key, value in (
                ("predictions", predictions),
                ("labels", labels),
                ("previous", previous),
                ("changes", changes),
            ):
                aggregate[key].append(value)
                by_match[match_id][key].append(value)

    def summarize(rows: dict[str, list[np.ndarray]]) -> dict[str, object]:
        predictions = np.concatenate(rows["predictions"])
        labels = np.concatenate(rows["labels"])
        previous = np.concatenate(rows["previous"])
        changes = np.concatenate(rows["changes"]).astype(bool)
        majority = np.full_like(labels, training_majority_action)
        result = {
            "policy": _action_metrics(predictions, labels),
            "previous_action_baseline": _action_metrics(previous, labels),
            "training_majority_baseline": _action_metrics(majority, labels),
            "action_change_samples": int(changes.sum()),
            "action_change_policy": (
                _action_metrics(predictions[changes], labels[changes])
                if changes.any()
                else None
            ),
            "action_change_previous_action_baseline": (
                _action_metrics(previous[changes], labels[changes])
                if changes.any()
                else None
            ),
            "action_change_training_majority_baseline": (
                _action_metrics(majority[changes], labels[changes])
                if changes.any()
                else None
            ),
        }
        return result

    return {
        "team_id": team_id,
        "checkpoint": str(checkpoint),
        "observation_variant": str(saved["observation_variant"]),
        "training_majority_action": int(training_majority_action),
        "overall": summarize(aggregate),
        "matches": {
            match_id: dict(source=sources[match_id], **summarize(rows))
            for match_id, rows in sorted(by_match.items())
        },
    }
