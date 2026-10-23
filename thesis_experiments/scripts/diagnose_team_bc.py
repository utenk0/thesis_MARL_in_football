"""Create the final diagnostics and logical freeze for discrete-action team BC."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from thesis_experiments.training.team_bc_diagnostics import (
    ACTION_NAMES,
    diagnose_split,
    file_sha256,
    write_split_artifacts,
)


def _stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _resolve(value: str, base: Path) -> Path:
    path = Path(value)
    for candidate in (path, base / path, *[parent / path for parent in base.parents]):
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(value)


def _headline(rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "action_accuracy": _stats([row["policy"]["action_accuracy"] for row in rows]),
        "balanced_action_accuracy": _stats(
            [row["policy"]["balanced_action_accuracy"] for row in rows]
        ),
        "action_change_accuracy": _stats(
            [row["action_change_policy"]["action_accuracy"] for row in rows]
        ),
        "majority_baseline_accuracy": rows[0]["training_majority_baseline"][
            "action_accuracy"
        ],
        "change_majority_baseline_accuracy": rows[0][
            "action_change_training_majority_baseline"
        ]["action_accuracy"],
        "previous_action_baseline_accuracy": rows[0]["previous_action_baseline"][
            "action_accuracy"
        ],
        "samples": rows[0]["policy"]["samples"],
        "action_change_samples": rows[0]["action_change_policy"]["samples"],
    }


def _write_report(summary: dict[str, object], path: Path) -> None:
    lines = [
        "# Frozen discrete-action BC diagnostics",
        "",
        f"- Team: `{summary['team_id']}`",
        f"- Seeds: {', '.join(map(str, summary['seeds']))}",
        f"- Selected epochs: {', '.join(map(str, summary['selected_epochs']))}",
        "- Checkpoint selection: minimum validation cross-entropy only",
        "- Test-set role: final report only; never model selection",
        "- Status: frozen discrete-action imitation baseline",
        "",
        "## Primary action-policy metrics",
        "",
        "| Split | Accuracy | Balanced accuracy | Action-change accuracy | Majority | Change majority | Previous action |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split in ("train", "validation", "test"):
        row = summary["headline"][split]
        lines.append(
            f"| {split} | {row['action_accuracy']['mean']:.4f} ± {row['action_accuracy']['std']:.4f} | "
            f"{row['balanced_action_accuracy']['mean']:.4f} ± {row['balanced_action_accuracy']['std']:.4f} | "
            f"{row['action_change_accuracy']['mean']:.4f} ± {row['action_change_accuracy']['std']:.4f} | "
            f"{row['majority_baseline_accuracy']:.4f} | "
            f"{row['change_majority_baseline_accuracy']:.4f} | "
            f"{row['previous_action_baseline_accuracy']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Test per-action diagnostics",
            "",
            "Values are mean ± sample standard deviation over seeds. Support is fixed because every seed uses the same held-out match.",
            "",
            "| ID | Action | Support | Precision | Recall | F1 |",
            "|---:|---|---:|---:|---:|---:|",
        ]
    )
    for row in summary["test_per_action"]:
        precision = row["precision"]
        recall = row["recall"]
        f1 = row["f1"]
        precision_text = "n/a" if precision is None else f"{precision['mean']:.4f} ± {precision['std']:.4f}"
        recall_text = "n/a" if recall is None else f"{recall['mean']:.4f} ± {recall['std']:.4f}"
        f1_text = "n/a" if f1 is None else f"{f1['mean']:.4f} ± {f1['std']:.4f}"
        lines.append(
            f"| {row['action']} | {row['name']} | {row['support']:,} | "
            f"{precision_text} | {recall_text} | {f1_text} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The action-imitation result is retained as a baseline: it beats the training-majority baseline and has non-trivial action-change accuracy. Ordinary accuracy remains below previous-action persistence, so it cannot by itself demonstrate strong decision modeling. Event-action rows with very low training support are descriptive only.",
            "",
            "The confusion matrices, row-normalized matrices, per-player, per-match, per-chunk tables, and qualitative examples are stored under each `seed_<n>/<split>/` directory. Event examples include only unambiguously event-derived action IDs; action 0 is excluded because it mixes stationary tracking and event overrides.",
            "",
            "This frozen model predicts inferred discrete actions. It is not a trajectory-reconstruction model, and its existing GRF rollout result must continue to be reported separately.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def _per_action(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output = []
    for action in range(19):
        action_rows = [row["policy"]["per_action"][action] for row in rows]
        values = {}
        for metric in ("precision", "recall", "f1"):
            observed = [float(row[metric]) for row in action_rows if row[metric] is not None]
            values[metric] = _stats(observed) if observed else None
        output.append(
            {
                "action": action,
                "name": ACTION_NAMES[action],
                "support": action_rows[0]["support"],
                **values,
            }
        )
    return output


def _freeze(experiment_dir: Path, manifest_path: Path, diagnostics: Path) -> dict:
    files = [
        manifest_path,
        experiment_dir / "summary.json",
        experiment_dir / "FULL_BC_REPORT.md",
        diagnostics / "diagnostics.json",
        diagnostics / "BC_DIAGNOSTICS.md",
    ]
    selected_epochs = []
    validation_flags = []
    for seed_dir in sorted(experiment_dir.glob("seed_*")):
        if not seed_dir.is_dir() or not (seed_dir / "single_team_bc.pt").is_file():
            continue
        files.extend(
            seed_dir / name
            for name in (
                "single_team_bc.pt",
                "config.json",
                "result.json",
                "train_evaluation.json",
                "validation_evaluation.json",
                "test_evaluation.json",
            )
        )
        checkpoint = torch.load(
            seed_dir / "single_team_bc.pt", map_location="cpu", weights_only=True
        )
        selected_epochs.append(int(checkpoint["selected_epoch"]))
        validation_flags.append(bool(checkpoint["validation_selection"]))
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Cannot freeze; missing files: {missing}")
    result = {
        "status": "frozen_discrete_action_bc",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "Shared single-team policy for inferred 19-class actions. This freeze does "
            "not assert trajectory or game reconstruction."
        ),
        "selection_contract": {
            "criterion": "minimum validation cross-entropy",
            "validation_selection_verified_for_all_checkpoints": all(validation_flags),
            "selected_epochs": selected_epochs,
            "test_used_for_selection": False,
            "test_role": "final report only",
        },
        "files": [
            {"path": str(path), "sha256": file_sha256(path), "bytes": path.stat().st_size}
            for path in files
        ],
    }
    (experiment_dir / "frozen_discrete_action_bc.json").write_text(
        json.dumps(result, indent=2)
    )
    (experiment_dir / "DISCRETE_ACTION_BC_FROZEN.md").write_text(
        "# Discrete-action BC freeze\n\n"
        "This experiment is the frozen inferred-action imitation baseline. "
        "Its checkpoints were selected using validation loss only; test data was "
        "used only for final reporting. File hashes are recorded in "
        "`frozen_discrete_action_bc.json`.\n\n"
        "Do not overwrite these checkpoints when developing displacement or "
        "trajectory targets. Those models belong in a separate artifact directory.\n"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--prediction-batch-size", type=int, default=16384)
    args = parser.parse_args()
    torch.set_num_threads(args.torch_threads)
    experiment_dir = args.experiment_dir.resolve()
    first_config = json.loads((experiment_dir / "seed_0" / "config.json").read_text())
    manifest_path = (args.manifest or Path(first_config["manifest"])).resolve()
    manifest = json.loads(manifest_path.read_text())
    team_id = str(manifest["team_id"])
    split_paths = {
        split: [_resolve(row["path"], manifest_path.parent) for row in rows]
        for split, rows in manifest["splits"].items()
    }
    output_dir = (args.output_dir or experiment_dir / "diagnostics").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_results: dict[str, dict[str, object]] = {}
    for seed_dir in sorted(experiment_dir.glob("seed_*")):
        checkpoint = seed_dir / "single_team_bc.pt"
        if not checkpoint.is_file():
            continue
        seed = seed_dir.name.split("_", 1)[1]
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        majority = int(torch.as_tensor(saved["class_counts"]).argmax().item())
        seed_results[seed] = {}
        for split in ("train", "validation", "test"):
            result = diagnose_split(
                checkpoint,
                split_paths[split],
                team_id=team_id,
                training_majority_action=majority,
                device=args.device,
                prediction_batch_size=args.prediction_batch_size,
            )
            seed_results[seed][split] = result
            write_split_artifacts(result, output_dir / f"seed_{seed}" / split)
            print(
                f"seed={seed} split={split} "
                f"accuracy={result['policy']['action_accuracy']:.6f}",
                flush=True,
            )

    if len(seed_results) < 2:
        raise ValueError("Expected at least two completed seed checkpoints.")
    seeds = sorted(seed_results, key=int)
    split_rows = {
        split: [seed_results[seed][split] for seed in seeds]
        for split in ("train", "validation", "test")
    }
    summary = {
        "team_id": team_id,
        "seeds": [int(seed) for seed in seeds],
        "selected_epochs": [
            int(seed_results[seed]["validation"]["selected_epoch"]) for seed in seeds
        ],
        "model_selection": {
            "criterion": "minimum validation cross-entropy",
            "test_used_for_selection": False,
        },
        "headline": {split: _headline(rows) for split, rows in split_rows.items()},
        "test_per_action": _per_action(split_rows["test"]),
        "seed_results": seed_results,
    }
    diagnostics_path = output_dir / "diagnostics.json"
    diagnostics_path.write_text(json.dumps(summary, indent=2))
    _write_report(summary, output_dir / "BC_DIAGNOSTICS.md")
    freeze = _freeze(experiment_dir, manifest_path, output_dir)
    print(
        json.dumps(
            {
                "diagnostics": str(diagnostics_path),
                "report": str(output_dir / "BC_DIAGNOSTICS.md"),
                "freeze": str(experiment_dir / "frozen_discrete_action_bc.json"),
                "files_frozen": len(freeze["files"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
