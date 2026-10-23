"""Run the controlled BC observation/class-balance experiment."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


EXPERIMENTS = (
    {"name": "spatial", "observation_variant": "spatial", "class_weighting": "none"},
    {
        "name": "temporal_context",
        "observation_variant": "temporal_context",
        "class_weighting": "none",
    },
    {"name": "full", "observation_variant": "full", "class_weighting": "none"},
    {
        "name": "full_balanced",
        "observation_variant": "full",
        "class_weighting": "inverse_frequency",
    },
)


def _resolve_data_paths(manifest_path: Path) -> dict[str, list[Path]]:
    manifest = json.loads(manifest_path.read_text())
    result: dict[str, list[Path]] = {}
    for split, rows in manifest["splits"].items():
        result[split] = []
        for row in rows:
            raw = Path(row["path"])
            candidates = (raw, *(parent / raw for parent in manifest_path.parents))
            path = next((item for item in candidates if item.is_file()), None)
            if path is None:
                raise FileNotFoundError(raw)
            result[split].append(path.resolve())
    return result


def _run(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=False)
    return int(process.returncode)


def _mean_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=0))


def summarize(output_dir: Path) -> list[dict]:
    manifest_path = output_dir / "ablation_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    rows: list[dict] = []
    for run in manifest:
        row = {
            "experiment": run["experiment"],
            "observation_variant": run["observation_variant"],
            "class_weighting": run["class_weighting"],
            "seed": run["seed"],
            "status": run["status"],
        }
        method_dir = Path(run["method_dir"])
        for split in ("validation", "test", "external_test"):
            evaluation_path = method_dir / f"{split}_evaluation.json"
            if not evaluation_path.exists():
                continue
            evaluation = json.loads(evaluation_path.read_text())
            offline = evaluation["offline"]
            row[f"{split}_action_accuracy"] = offline["action_accuracy"]
            row[f"{split}_balanced_accuracy"] = offline["balanced_action_accuracy"]
            row[f"{split}_movement_accuracy"] = offline["movement_action_accuracy"]
            row[f"{split}_rare_accuracy"] = offline["rare_action_accuracy"]
            row[f"{split}_rare_samples"] = offline["rare_action_samples"]
            row[f"{split}_predicted_rare_actions"] = offline["predicted_rare_actions"]
            row[f"{split}_common_accuracy"] = offline["common_action_accuracy"]
            row[f"{split}_common_balanced_accuracy"] = offline[
                "common_balanced_action_accuracy"
            ]
            row[f"{split}_common_samples"] = offline["common_action_samples"]
            row[f"{split}_majority_accuracy"] = evaluation["majority_baseline"][
                "action_accuracy"
            ]
            row[f"{split}_majority_balanced_accuracy"] = evaluation[
                "majority_baseline"
            ]["balanced_action_accuracy"]
            if evaluation.get("action_change_points"):
                row[f"{split}_action_change_accuracy"] = evaluation[
                    "action_change_points"
                ]["action_accuracy"]
            if "previous_action_baseline" in evaluation:
                row[f"{split}_persistence_accuracy"] = evaluation[
                    "previous_action_baseline"
                ]["action_accuracy"]
                row[f"{split}_persistence_balanced_accuracy"] = evaluation[
                    "previous_action_baseline"
                ]["balanced_action_accuracy"]
            rollouts = [dataset["grf"] for dataset in evaluation["datasets"] if "grf" in dataset]
            if rollouts:
                total_steps = sum(item["steps"] for item in rollouts)
                row["grf_joint_steps"] = total_steps
                row["grf_mean_position_error"] = sum(
                    item["steps"] * item["mean_position_error_grf"] for item in rollouts
                ) / total_steps
                row["grf_all_finite"] = all(item["finite"] for item in rollouts)
        rows.append(row)

    (output_dir / "comparison.json").write_text(json.dumps(rows, indent=2))
    with (output_dir / "comparison.csv").open("w", newline="") as handle:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    completed = [row for row in rows if row["status"] == "completed"]
    lines = [
        "# BC observation and class-balance ablation",
        "",
        "All configurations use the same data splits, architecture, optimiser settings, "
        "and seeds. Excluded observation features are zeroed, preserving the common "
        "61-dimensional policy interface.",
        "",
        "| Configuration | Test accuracy | Test balanced | Common balanced | Action-change accuracy | Rare-action accuracy |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for experiment in [item["name"] for item in EXPERIMENTS]:
        selected = [row for row in completed if row["experiment"] == experiment]
        if not selected:
            continue
        statistics = []
        for key in (
            "test_action_accuracy",
            "test_balanced_accuracy",
            "test_common_balanced_accuracy",
            "test_action_change_accuracy",
            "test_rare_accuracy",
        ):
            values = [float(row[key]) for row in selected if row.get(key) is not None]
            statistics.append(_mean_std(values) if values else (float("nan"), float("nan")))
        formatted = [f"{mean:.4f} ± {std:.4f}" for mean, std in statistics]
        lines.append(f"| {experiment} | {' | '.join(formatted)} |")
    if completed:
        persistence = [
            float(row["test_persistence_accuracy"])
            for row in completed
            if "test_persistence_accuracy" in row
        ]
        persistence_balanced = [
            float(row["test_persistence_balanced_accuracy"])
            for row in completed
            if "test_persistence_balanced_accuracy" in row
        ]
        majority = [
            float(row["test_majority_accuracy"])
            for row in completed
            if "test_majority_accuracy" in row
        ]
        majority_balanced = [
            float(row["test_majority_balanced_accuracy"])
            for row in completed
            if "test_majority_balanced_accuracy" in row
        ]
        lines.extend([
            "",
            "## Interpretation guardrails",
            "",
            f"- Training-majority baseline: {_mean_std(majority)[0]:.4f} ordinary and "
            f"{_mean_std(majority_balanced)[0]:.4f} balanced accuracy.",
            f"- Previous-action baseline: {_mean_std(persistence)[0]:.4f} ordinary and "
            f"{_mean_std(persistence_balanced)[0]:.4f} balanced accuracy.",
            "- Compare `spatial` with `temporal_context` to test causal motion/context features.",
            "- Compare `temporal_context` with `full` to measure the contribution of previous/sticky actions.",
            "- Compare `full` with `full_balanced` to measure the accuracy/rare-action trade-off.",
            "- These are bounded diagnostic runs, not final thesis results or hyperparameter optimisation.",
        ])
    (output_dir / "ABLATION_REPORT.md").write_text("\n".join(lines) + "\n")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--max-class-weight", type=float, default=20.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--grf-python", type=Path)
    parser.add_argument("--eval-grf-steps", type=int, default=50)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    splits = _resolve_data_paths(args.split_manifest.resolve())
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "ablation_manifest.json"
    if manifest_path.exists() and not args.resume:
        parser.error(f"{manifest_path} exists; choose another directory or pass --resume")
    runs = json.loads(manifest_path.read_text()) if manifest_path.exists() else []
    by_key = {(run["experiment"], run["seed"]): run for run in runs}

    def persist() -> None:
        temporary = manifest_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(runs, indent=2))
        temporary.replace(manifest_path)

    for experiment in EXPERIMENTS:
        for seed in args.seeds:
            key = (experiment["name"], seed)
            run_root = output_dir / experiment["name"] / f"seed_{seed}"
            method_output = run_root / "methods"
            method_dir = method_output / "bc"
            result_path = method_dir / "result.json"
            run = by_key.get(key)
            evaluation_splits = tuple(
                split for split in ("validation", "test", "external_test")
                if splits.get(split)
            )
            evaluations_exist = all(
                (method_dir / f"{split}_evaluation.json").is_file()
                for split in evaluation_splits
            )
            if (
                run
                and run.get("status") == "completed"
                and result_path.is_file()
                and evaluations_exist
            ):
                print(f"Skipping completed {experiment['name']}, seed {seed}", flush=True)
                continue
            if run is None:
                run = {
                    "experiment": experiment["name"],
                    "observation_variant": experiment["observation_variant"],
                    "class_weighting": experiment["class_weighting"],
                    "seed": seed,
                    "status": "pending",
                    "method_dir": str(method_dir),
                }
                runs.append(run)
                by_key[key] = run
            train_command = [
                args.python,
                "-m",
                "thesis_experiments.scripts.run_method",
                "--method", "bc",
                "--backend", "torch",
                "--output-dir", str(method_output),
                "--seed", str(seed),
                "--epochs", str(args.epochs),
                "--batch-size", str(args.batch_size),
                "--learning-rate", str(args.learning_rate),
                "--hidden-size", str(args.hidden_size),
                "--device", args.device,
                "--observation-variant", experiment["observation_variant"],
                "--class-weighting", experiment["class_weighting"],
                "--max-class-weight", str(args.max_class_weight),
                "--data", *map(str, splits["train"]),
            ]
            run["train_command"] = train_command
            if args.dry_run:
                run["status"] = "dry-run"
                persist()
                continue
            run["status"] = "training"
            persist()
            print(f"Training {experiment['name']}, seed {seed}", flush=True)
            returncode = _run(train_command, run_root / "train.log")
            if returncode:
                run["status"] = "training-failed"
                run["returncode"] = returncode
                persist()
                raise SystemExit(returncode)
            run["status"] = "evaluating"
            persist()
            for split in evaluation_splits:
                evaluation_command = [
                    args.python,
                    "-m",
                    "thesis_experiments.scripts.evaluate_policy",
                    "--result", str(result_path),
                    "--data", *map(str, splits[split]),
                    "--output", str(method_dir / f"{split}_evaluation.json"),
                    "--grf-steps", str(args.eval_grf_steps if split == "test" else 0),
                    "--seed", str(seed),
                    "--device", args.device,
                ]
                if args.grf_python:
                    evaluation_command.extend(["--grf-python", str(args.grf_python)])
                returncode = _run(evaluation_command, run_root / f"{split}_evaluation.log")
                if returncode:
                    run["status"] = f"{split}-evaluation-failed"
                    run["returncode"] = returncode
                    persist()
                    raise SystemExit(returncode)
            run["status"] = "completed"
            run["returncode"] = 0
            persist()
    rows = summarize(output_dir) if not args.dry_run else runs
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
