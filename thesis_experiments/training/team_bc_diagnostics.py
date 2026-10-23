"""Detailed, streaming diagnostics for a frozen single-team BC policy.

The normal evaluator intentionally stores only the headline metrics.  This
module adds confusion matrices, precision/recall/support, player-, match-, and
chunk-level results, plus small qualitative examples without retaining every
prediction from a complete match in memory.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from thesis_experiments.datasets.action_mapping import GRF_ACTIONS
from thesis_experiments.training.team_bc import _load_actor
from thesis_experiments.transitions.observations import (
    ACTION_COUNT,
    LOCAL_FEATURE_SLICES,
)


ACTION_NAMES = {identifier: name for name, identifier in GRF_ACTIONS.items()}
# These labels can be unambiguously attributed to an event mapping.  Action 0
# is omitted because it mixes stationary tracking with some event overrides.
EVENT_ACTION_IDS = (9, 10, 11, 12, 16, 17, 18)


def confusion_metrics(confusion: np.ndarray) -> dict[str, object]:
    """Return class and aggregate metrics from a target-by-prediction matrix."""

    confusion = np.asarray(confusion, dtype=np.int64)
    if confusion.shape != (ACTION_COUNT, ACTION_COUNT):
        raise ValueError(
            f"Expected a {(ACTION_COUNT, ACTION_COUNT)} confusion matrix, "
            f"got {confusion.shape}."
        )
    support = confusion.sum(axis=1)
    predicted = confusion.sum(axis=0)
    true_positive = np.diag(confusion)
    recall = np.divide(
        true_positive,
        support,
        out=np.full(ACTION_COUNT, np.nan, dtype=np.float64),
        where=support > 0,
    )
    precision = np.divide(
        true_positive,
        predicted,
        out=np.full(ACTION_COUNT, np.nan, dtype=np.float64),
        where=predicted > 0,
    )
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.full(ACTION_COUNT, np.nan, dtype=np.float64),
        where=(precision + recall) > 0,
    )
    samples = int(confusion.sum())
    observed = support > 0
    per_action = []
    for action in range(ACTION_COUNT):
        per_action.append(
            {
                "action": action,
                "name": ACTION_NAMES[action],
                "support": int(support[action]),
                "predicted": int(predicted[action]),
                "true_positive": int(true_positive[action]),
                "precision": None if np.isnan(precision[action]) else float(precision[action]),
                "recall": None if np.isnan(recall[action]) else float(recall[action]),
                "f1": None if np.isnan(f1[action]) else float(f1[action]),
            }
        )
    return {
        "samples": samples,
        "action_accuracy": float(true_positive.sum() / samples) if samples else None,
        "balanced_action_accuracy": (
            float(recall[observed].mean()) if observed.any() else None
        ),
        "confusion_matrix": confusion.tolist(),
        "per_action": per_action,
    }


def _empty_confusion() -> np.ndarray:
    return np.zeros((ACTION_COUNT, ACTION_COUNT), dtype=np.int64)


def _add_confusion(
    confusion: np.ndarray, labels: np.ndarray, predictions: np.ndarray
) -> None:
    np.add.at(
        confusion,
        (
            np.asarray(labels, dtype=np.int64).reshape(-1),
            np.asarray(predictions, dtype=np.int64).reshape(-1),
        ),
        1,
    )


def _predict(
    actor: torch.nn.Module,
    observations: np.ndarray,
    feature_mask: np.ndarray,
    *,
    device: str,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    mask = torch.as_tensor(feature_mask, dtype=torch.float32, device=device)
    predictions = []
    confidences = []
    actor.eval()
    with torch.no_grad():
        for start in range(0, len(observations), batch_size):
            values = torch.as_tensor(
                observations[start : start + batch_size],
                dtype=torch.float32,
                device=device,
            )
            probabilities = torch.softmax(actor(values * mask), dim=-1)
            confidence, prediction = probabilities.max(dim=-1)
            predictions.append(prediction.cpu().numpy())
            confidences.append(confidence.cpu().numpy())
    return (
        np.concatenate(predictions).astype(np.int64),
        np.concatenate(confidences).astype(np.float32),
    )


def _chunk_values(path: Path, team_id: str) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as data:
        team_ids = np.asarray(data["team_ids"]).astype(str)
        selected = np.flatnonzero(team_ids == team_id)
        if len(selected) != 11:
            raise ValueError(f"{path}: expected 11 slots for team {team_id!r}.")
        observations = np.asarray(
            data["local_observations"][:, selected], dtype=np.float32
        )
        labels = np.asarray(data["actions"][:, selected], dtype=np.int64)
        return {
            "observations": observations,
            "labels": labels,
            "previous": observations[..., LOCAL_FEATURE_SLICES["previous_action"]].argmax(-1),
            "frames": np.asarray(data["frames"], dtype=np.int64),
            "player_ids": np.asarray(data["player_ids"])[selected].astype(str),
            "match_id": str(np.asarray(data["match_id"]).item()),
            "source": str(np.asarray(data["source"]).item()),
        }


def _summary_row(confusion: np.ndarray, change_confusion: np.ndarray) -> dict[str, object]:
    metrics = confusion_metrics(confusion)
    change = confusion_metrics(change_confusion)
    return {
        "samples": metrics["samples"],
        "action_accuracy": metrics["action_accuracy"],
        "balanced_action_accuracy": metrics["balanced_action_accuracy"],
        "action_change_samples": change["samples"],
        "action_change_accuracy": change["action_accuracy"],
    }


def _context(
    labels: np.ndarray,
    predictions: np.ndarray,
    player_index: int,
    transition_index: int,
) -> list[dict[str, object]]:
    start = max(0, transition_index - 2)
    stop = min(len(labels), transition_index + 3)
    return [
        {
            "relative_step": int(index - transition_index),
            "target": int(labels[index, player_index]),
            "target_name": ACTION_NAMES[int(labels[index, player_index])],
            "prediction": int(predictions[index, player_index]),
            "prediction_name": ACTION_NAMES[int(predictions[index, player_index])],
        }
        for index in range(start, stop)
    ]


def diagnose_split(
    checkpoint: Path,
    paths: Iterable[Path],
    *,
    team_id: str,
    training_majority_action: int,
    device: str = "cpu",
    prediction_batch_size: int = 16384,
    max_examples_per_kind: int = 12,
) -> dict[str, object]:
    """Evaluate one checkpoint on one split and retain diagnostic aggregates."""

    paths = [Path(path) for path in paths]
    if not paths:
        raise ValueError("At least one path is required.")
    actor, feature_mask, saved = _load_actor(Path(checkpoint), device)
    if str(saved["team_id"]) != team_id:
        raise ValueError("Checkpoint and requested team do not match.")

    policy = _empty_confusion()
    previous_baseline = _empty_confusion()
    majority_baseline = _empty_confusion()
    change_policy = _empty_confusion()
    change_previous = _empty_confusion()
    change_majority = _empty_confusion()
    by_player: dict[str, dict[str, np.ndarray]] = defaultdict(
        lambda: {"all": _empty_confusion(), "change": _empty_confusion()}
    )
    by_match: dict[str, dict[str, np.ndarray]] = defaultdict(
        lambda: {"all": _empty_confusion(), "change": _empty_confusion()}
    )
    chunks = []
    examples: dict[str, list[dict[str, object]]] = {
        "action_change_correct": [],
        "action_change_error": [],
        "event_action_label": [],
    }

    for chunk_index, path in enumerate(paths):
        values = _chunk_values(path, team_id)
        observations = values["observations"]
        labels = values["labels"]
        previous = values["previous"]
        frames = values["frames"]
        player_ids = values["player_ids"]
        flat_predictions, flat_confidences = _predict(
            actor,
            observations.reshape(-1, observations.shape[-1]),
            feature_mask,
            device=device,
            batch_size=prediction_batch_size,
        )
        predictions = flat_predictions.reshape(labels.shape)
        confidences = flat_confidences.reshape(labels.shape)
        majority = np.full_like(labels, training_majority_action)
        changed = labels != previous

        chunk_confusion = _empty_confusion()
        chunk_change = _empty_confusion()
        _add_confusion(chunk_confusion, labels, predictions)
        _add_confusion(chunk_change, labels[changed], predictions[changed])
        _add_confusion(policy, labels, predictions)
        _add_confusion(previous_baseline, labels, previous)
        _add_confusion(majority_baseline, labels, majority)
        _add_confusion(change_policy, labels[changed], predictions[changed])
        _add_confusion(change_previous, labels[changed], previous[changed])
        _add_confusion(change_majority, labels[changed], majority[changed])
        match_id = str(values["match_id"])
        _add_confusion(by_match[match_id]["all"], labels, predictions)
        _add_confusion(by_match[match_id]["change"], labels[changed], predictions[changed])
        for player_index, player_id in enumerate(player_ids.tolist()):
            player_changed = changed[:, player_index]
            _add_confusion(
                by_player[player_id]["all"],
                labels[:, player_index],
                predictions[:, player_index],
            )
            _add_confusion(
                by_player[player_id]["change"],
                labels[player_changed, player_index],
                predictions[player_changed, player_index],
            )

        chunks.append(
            {
                "chunk_index": chunk_index,
                "path": str(path),
                "source": values["source"],
                "match_id": match_id,
                "first_frame": int(frames[0]),
                "last_frame": int(frames[-1]),
                **_summary_row(chunk_confusion, chunk_change),
            }
        )

        masks = {
            "action_change_correct": changed & (predictions == labels),
            "action_change_error": changed & (predictions != labels),
            "event_action_label": np.isin(labels, EVENT_ACTION_IDS),
        }
        for kind, candidate_mask in masks.items():
            if len(examples[kind]) >= max_examples_per_kind:
                continue
            candidate = np.argwhere(candidate_mask)
            if not len(candidate):
                continue
            # One representative per chunk distributes examples through the
            # match instead of taking an adjacent block of highly correlated frames.
            transition_index, player_index = candidate[len(candidate) // 2]
            target = int(labels[transition_index, player_index])
            prediction = int(predictions[transition_index, player_index])
            examples[kind].append(
                {
                    "chunk": str(path),
                    "match_id": match_id,
                    "frame": int(frames[transition_index]),
                    "player_id": str(player_ids[player_index]),
                    "previous_action": int(previous[transition_index, player_index]),
                    "previous_action_name": ACTION_NAMES[
                        int(previous[transition_index, player_index])
                    ],
                    "target": target,
                    "target_name": ACTION_NAMES[target],
                    "prediction": prediction,
                    "prediction_name": ACTION_NAMES[prediction],
                    "correct": bool(prediction == target),
                    "prediction_confidence": float(
                        confidences[transition_index, player_index]
                    ),
                    "context": _context(
                        labels, predictions, int(player_index), int(transition_index)
                    ),
                }
            )

    return {
        "checkpoint": str(checkpoint),
        "selected_epoch": int(saved["selected_epoch"]),
        "validation_selected": bool(saved["validation_selection"]),
        "observation_variant": str(saved["observation_variant"]),
        "event_frame_definition": (
            "Targets with unambiguously event-derived action IDs "
            f"{list(EVENT_ACTION_IDS)}. Mixed idle/event action 0 is excluded."
        ),
        "policy": confusion_metrics(policy),
        "previous_action_baseline": confusion_metrics(previous_baseline),
        "training_majority_baseline": confusion_metrics(majority_baseline),
        "action_change_policy": confusion_metrics(change_policy),
        "action_change_previous_action_baseline": confusion_metrics(change_previous),
        "action_change_training_majority_baseline": confusion_metrics(change_majority),
        "by_player": {
            player_id: _summary_row(values["all"], values["change"])
            for player_id, values in sorted(by_player.items())
        },
        "by_match": {
            match_id: _summary_row(values["all"], values["change"])
            for match_id, values in sorted(by_match.items())
        },
        "by_chunk": chunks,
        "examples": examples,
    }


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_split_artifacts(result: dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix = np.asarray(result["policy"]["confusion_matrix"], dtype=np.int64)
    header = ["target_id", "target_name"] + [
        f"pred_{action}_{ACTION_NAMES[action]}" for action in range(ACTION_COUNT)
    ]
    count_rows = []
    normalized_rows = []
    for action in range(ACTION_COUNT):
        count_rows.append(
            dict(
                zip(
                    header,
                    [action, ACTION_NAMES[action], *matrix[action].tolist()],
                )
            )
        )
        denominator = int(matrix[action].sum())
        values = matrix[action] / denominator if denominator else np.zeros(ACTION_COUNT)
        normalized_rows.append(
            dict(zip(header, [action, ACTION_NAMES[action], *values.tolist()]))
        )
    _write_rows(output_dir / "confusion_counts.csv", count_rows)
    _write_rows(output_dir / "confusion_row_normalized.csv", normalized_rows)
    _write_rows(output_dir / "per_action.csv", result["policy"]["per_action"])
    _write_rows(
        output_dir / "per_player.csv",
        [dict(player_id=key, **value) for key, value in result["by_player"].items()],
    )
    _write_rows(
        output_dir / "per_match.csv",
        [dict(match_id=key, **value) for key, value in result["by_match"].items()],
    )
    _write_rows(output_dir / "per_chunk.csv", result["by_chunk"])
    (output_dir / "examples.json").write_text(json.dumps(result["examples"], indent=2))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
