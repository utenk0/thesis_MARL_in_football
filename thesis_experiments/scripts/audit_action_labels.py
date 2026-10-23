"""Audit GRF action-label support, imbalance, and temporal persistence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from thesis_experiments.datasets.action_mapping import GRF_ACTIONS
from thesis_experiments.transitions.observations import (
    ACTION_COUNT,
    LOCAL_FEATURE_SLICES,
    OBSERVATION_VERSION,
)
from thesis_experiments.transitions.schema import JointTransitionDataset

COMMON_ACTION_IDS = tuple(range(13)) + (14, 16)

ACTION_NAMES = {identifier: name for name, identifier in GRF_ACTIONS.items()}
ACTION_PROVENANCE = {
    0: "mixed stationary tracking / selected event overrides",
    **{identifier: "tracking-derived displacement" for identifier in range(1, 9)},
    9: "event-derived pass override",
    10: "event-derived pass/set-piece override",
    11: "event-derived pass/set-piece override",
    12: "event-derived shot override",
    13: "not emitted by current converters",
    14: "tracking-derived direction release",
    15: "not emitted by current converters",
    16: "event-derived tackle override",
    17: "event-derived ball-action override",
    18: "event-derived ball-lost override",
}


def _resolve_path(manifest_path: Path, value: str) -> Path:
    path = Path(value)
    candidates = [path, *(parent / path for parent in manifest_path.parents)]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(path)


def _counts(labels: np.ndarray) -> list[int]:
    return np.bincount(np.asarray(labels).reshape(-1), minlength=ACTION_COUNT).tolist()


def _accuracy(predictions: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    predictions = np.asarray(predictions).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    recalls = [
        float((predictions[labels == action] == action).mean())
        for action in np.unique(labels)
    ]
    return float((predictions == labels).mean()), float(np.mean(recalls))


def audit(manifest_path: Path, output_dir: Path, *, low_support: int = 100) -> dict:
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text())
    split_rows = manifest["splits"]
    report: dict[str, object] = {
        "split_manifest": str(manifest_path),
        "low_support_threshold": low_support,
        "provenance_scope": (
            "NPZ files store the final labels. The final-stage manifest links each chunk to "
            "an event-level provenance sidecar for every mapped event override; movement "
            "and release labels are deterministically derived from tracking."
        ),
        "splits": {},
        "sources": {},
        "matches": [],
    }
    source_labels: dict[str, list[np.ndarray]] = defaultdict(list)
    match_parts: dict[tuple[str, str, str], dict[str, object]] = {}
    for split, rows in split_rows.items():
        labels_parts: list[np.ndarray] = []
        previous_parts: list[np.ndarray] = []
        for row in rows:
            path = _resolve_path(manifest_path, row["path"])
            expected_sha = row.get("sha256")
            actual_sha = hashlib.sha256(path.read_bytes()).hexdigest()
            if expected_sha and expected_sha != actual_sha:
                raise ValueError(f"Checksum mismatch for {path}")
            dataset = JointTransitionDataset.load(path)
            if dataset.observation_version != OBSERVATION_VERSION:
                raise ValueError(
                    f"{path} uses {dataset.observation_version}; the persistence audit "
                    f"requires {OBSERVATION_VERSION}."
                )
            labels = dataset.actions.reshape(-1)
            previous = dataset.local_observations[
                ..., LOCAL_FEATURE_SLICES["previous_action"]
            ].argmax(axis=-1).reshape(-1)
            labels_parts.append(labels)
            previous_parts.append(previous)
            source_labels[dataset.source].append(labels)
            key = (split, dataset.source, dataset.match_id)
            aggregate = match_parts.setdefault(key, {
                "labels": [], "previous": [], "paths": [], "event_overrides": 0,
            })
            aggregate["labels"].append(labels)
            aggregate["previous"].append(previous)
            aggregate["paths"].append(str(path))
            aggregate["event_overrides"] += int(row.get("event_overrides", 0))
        combined = np.concatenate(labels_parts)
        previous = np.concatenate(previous_parts)
        persistence_accuracy, persistence_balanced = _accuracy(previous, combined)
        report["splits"][split] = {
            "matches": len({(row["source"], row["match_id"]) for row in rows}),
            "chunks": len(rows),
            "player_samples": int(combined.size),
            "action_counts": _counts(combined),
            "observed_actions": int(len(np.unique(combined))),
            "persistence_accuracy": persistence_accuracy,
            "persistence_balanced_accuracy": persistence_balanced,
        }

    for (split, source, match_id), values in sorted(match_parts.items()):
        labels = np.concatenate(values["labels"])
        previous = np.concatenate(values["previous"])
        persistence_accuracy, persistence_balanced = _accuracy(previous, labels)
        report["matches"].append({
            "split": split,
            "source": source,
            "match_id": match_id,
            "chunks": len(values["paths"]),
            "joint_transitions": int(len(labels) // 22),
            "player_samples": int(labels.size),
            "action_counts": _counts(labels),
            "event_overrides": int(values["event_overrides"]),
            "persistence_accuracy": persistence_accuracy,
            "persistence_balanced_accuracy": persistence_balanced,
        })

    for source, parts in sorted(source_labels.items()):
        labels = np.concatenate(parts)
        report["sources"][source] = {
            "player_samples": int(labels.size),
            "action_counts": _counts(labels),
            "observed_actions": int(len(np.unique(labels))),
        }

    train_counts = report["splits"]["train"]["action_counts"]
    action_rows = []
    for action in range(ACTION_COUNT):
        row = {
            "action": action,
            "name": ACTION_NAMES[action],
            "provenance_rule": ACTION_PROVENANCE[action],
            **{
                f"{split}_support": int(report["splits"][split]["action_counts"][action])
                for split in split_rows
            },
        }
        row["train_percentage"] = row["train_support"] / report["splits"]["train"]["player_samples"]
        row["low_train_support"] = 0 < row["train_support"] < low_support
        row["absent_from_train"] = row["train_support"] == 0
        row["unseen_in_train_but_held_out"] = row["train_support"] == 0 and any(
            row.get(f"{split}_support", 0) > 0
            for split in split_rows if split != "train"
        )
        action_rows.append(row)
    report["actions"] = action_rows
    report["findings"] = {
        "absent_from_train": [row["name"] for row in action_rows if row["absent_from_train"]],
        "low_train_support": [row["name"] for row in action_rows if row["low_train_support"]],
        "unseen_in_train_but_held_out": [
            row["name"] for row in action_rows if row["unseen_in_train_but_held_out"]
        ],
        "majority_action": ACTION_NAMES[int(np.argmax(train_counts))],
        "common_action_ids": list(COMMON_ACTION_IDS),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "action_audit.json").write_text(json.dumps(report, indent=2))
    with (output_dir / "action_support.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(action_rows[0]))
        writer.writeheader()
        writer.writerows(action_rows)

    lines = [
        "# Action-label audit",
        "",
        "This audit describes all eligible final-stage chunks (continuous valid runs of at least the configured minimum length).",
        "",
        "## Split summary",
        "",
        "| Split | Matches | Chunks | Player samples | Observed actions | Previous-action accuracy | Previous-action balanced accuracy |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split in split_rows:
        values = report["splits"][split]
        lines.append(
            f"| {split} | {values['matches']} | {values['chunks']} | {values['player_samples']} | {values['observed_actions']}/19 | "
            f"{values['persistence_accuracy']:.4f} | "
            f"{values['persistence_balanced_accuracy']:.4f} |"
        )
    lines.extend([
        "",
        "## Per-action support",
        "",
        "| ID | Action | " + " | ".join(split.replace("_", " ").title() for split in split_rows) + " | Converter-level origin |",
        "|---:|---|" + "---:|" * len(split_rows) + "---|",
    ])
    for row in action_rows:
        lines.append(
            f"| {row['action']} | {row['name']} | "
            + " | ".join(str(row.get(f"{split}_support", 0)) for split in split_rows)
            + f" | {row['provenance_rule']} |"
        )
    findings = report["findings"]
    lines.extend([
        "",
        "## Findings",
        "",
        f"- Majority training action: `{findings['majority_action']}`.",
        f"- Actions absent from training: {', '.join(findings['absent_from_train']) or 'none'}.",
        f"- Actions with 1–{low_support - 1} training samples: "
        f"{', '.join(findings['low_train_support']) or 'none'}.",
        "- High previous-action accuracy indicates temporal persistence and must be shown "
        "beside learned-model accuracy.",
        "- Event-derived labels are traceable through the per-chunk provenance sidecars; "
        "tracking-derived labels are reproducible from the documented deterministic rule.",
        "",
    ])
    (output_dir / "ACTION_AUDIT.md").write_text("\n".join(lines))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--low-support", type=int, default=100)
    args = parser.parse_args()
    report = audit(args.split_manifest, args.output_dir, low_support=args.low_support)
    print(json.dumps({"outputs": str(args.output_dir), "findings": report["findings"]}, indent=2))


if __name__ == "__main__":
    main()
