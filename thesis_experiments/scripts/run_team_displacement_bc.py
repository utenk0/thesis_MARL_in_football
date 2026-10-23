"""Train and evaluate direct next-displacement BC for one identified team."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from thesis_experiments.training.team_displacement_bc import (
    evaluate_team_displacement_bc,
    train_team_displacement_bc,
)


def _paths(manifest: dict, split: str) -> list[Path]:
    return [Path(row["path"]) for row in manifest["splits"][split]]


def _matches(manifest: dict, split: str) -> list[str]:
    return sorted({str(row["match_id"]) for row in manifest["splits"][split]})


def _write_report(result: dict[str, object], path: Path) -> None:
    lines = [
        "# Single-team direct displacement BC",
        "",
        f"- Team: `{result['team_id']}`",
        f"- Target: `{result['training']['target']}`",
        f"- Observation: `{result['training']['observation_variant']}` ({result['training']['retained_features']} / 61 features)",
        f"- Training samples: {result['training']['samples']:,}",
        f"- Validation-selected epoch: {result['training']['selected_epoch']} of {result['training']['epochs_run']} run",
        "- Selection metric: validation mean displacement error",
        "- Test role: report only",
        "",
        "## One-step displacement",
        "",
        "| Split | Policy error (m) | Stationary (m) | Previous velocity (m) | Policy RMSE (m) |",
        "|---|---:|---:|---:|---:|",
    ]
    for split in ("train", "validation", "test"):
        overall = result["evaluations"][split]["overall"]
        policy = overall["policy"]
        stationary = overall["stationary_baseline"]
        velocity = overall["previous_velocity_baseline"]
        lines.append(
            f"| {split} | {policy['mean_displacement_error_meters']:.4f} | "
            f"{stationary['mean_displacement_error_meters']:.4f} | "
            f"{velocity['mean_displacement_error_meters']:.4f} | "
            f"{policy['root_mean_squared_displacement_error_meters']:.4f} |"
        )
    test_cumulative = result["evaluations"]["test"]["teacher_forced_cumulative"]
    if test_cumulative:
        lines.extend(
            [
                "",
                "## Test 100-step teacher-forced cumulative displacement",
                "",
                "| Predictor | ADE (m) | FDE (m) | Windows |",
                "|---|---:|---:|---:|",
            ]
        )
        for name, values in test_cumulative.items():
            lines.append(
                f"| {name} | {values['ade_meters']:.4f} | "
                f"{values['fde_meters']:.4f} | {values['windows']} |"
            )
    lines.extend(
        [
            "",
            "Cumulative results use recorded future observations at each step. They test whether predicted displacements accumulate coherently, but they are not autonomous reconstruction. A closed-loop or multi-step dynamics model remains necessary for a full game-reconstruction claim.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--cumulative-horizon", type=int, default=100)
    parser.add_argument(
        "--observation-variant",
        choices=["spatial", "temporal_context", "temporal_context_no_slot", "full"],
        default="temporal_context_no_slot",
    )
    args = parser.parse_args()
    torch.set_num_threads(args.torch_threads)
    manifest = json.loads(args.manifest.read_text())
    team_id = str(manifest["team_id"])
    split_paths = {split: _paths(manifest, split) for split in ("train", "validation", "test")}
    split_sets = [set(_matches(manifest, split)) for split in ("train", "validation", "test")]
    if any(split_sets[i] & split_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Displacement BC requires match-disjoint splits.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "single_team_displacement_bc.pt"
    training = train_team_displacement_bc(
        split_paths["train"],
        checkpoint,
        team_id=team_id,
        validation_paths=split_paths["validation"],
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hidden_size=args.hidden_size,
        seed=args.seed,
        device=args.device,
        observation_variant=args.observation_variant,
        early_stopping_patience=args.early_stopping_patience,
    )
    evaluations = {}
    for split in ("train", "validation", "test"):
        evaluation = evaluate_team_displacement_bc(
            checkpoint,
            split_paths[split],
            team_id=team_id,
            device=args.device,
            cumulative_horizon=args.cumulative_horizon,
        )
        evaluations[split] = evaluation
        (args.output_dir / f"{split}_evaluation.json").write_text(
            json.dumps(evaluation, indent=2)
        )

    result = {
        "method": "direct_displacement_bc",
        "team_id": team_id,
        "split_matches": {
            split: _matches(manifest, split) for split in ("train", "validation", "test")
        },
        "training": training,
        "evaluations": evaluations,
        "selection_contract": {
            "selected_on": "validation_mean_displacement_error_meters",
            "test_used_for_selection": False,
        },
    }
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2))
    config = {
        "manifest": str(args.manifest),
        "team_id": team_id,
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "hidden_size": args.hidden_size,
            "seed": args.seed,
            "observation_variant": args.observation_variant,
            "early_stopping_patience": args.early_stopping_patience,
            "cumulative_horizon": args.cumulative_horizon,
        },
        "data_paths": {
            split: [str(path) for path in paths] for split, paths in split_paths.items()
        },
    }
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2))
    _write_report(result, args.output_dir / "RUN_REPORT.md")
    print(
        json.dumps(
            {
                "result": str(args.output_dir / "result.json"),
                "checkpoint": str(checkpoint),
                "selected_epoch": training["selected_epoch"],
                "test_mean_displacement_error_meters": evaluations["test"]["overall"][
                    "policy"
                ]["mean_displacement_error_meters"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
