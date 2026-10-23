"""Train residual displacement BC using train/validation data only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from thesis_experiments.training.team_residual_displacement_bc import (
    evaluate_team_residual_displacement_bc,
    train_team_residual_displacement_bc,
)


def _paths(manifest: dict, split: str) -> list[Path]:
    return [Path(row["path"]) for row in manifest["splits"][split]]


def _matches(manifest: dict, split: str) -> list[str]:
    return sorted({str(row["match_id"]) for row in manifest["splits"][split]})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--history-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--minimum-epochs", type=int, default=5)
    parser.add_argument(
        "--observation-variant",
        choices=["spatial", "temporal_context", "temporal_context_no_slot", "full"],
        default="temporal_context_no_slot",
    )
    args = parser.parse_args()
    torch.set_num_threads(args.torch_threads)
    manifest = json.loads(args.manifest.read_text())
    team_id = str(manifest["team_id"])
    train_paths = _paths(manifest, "train")
    validation_paths = _paths(manifest, "validation")
    split_sets = [set(_matches(manifest, split)) for split in ("train", "validation", "test")]
    if any(split_sets[i] & split_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Residual BC requires match-disjoint splits.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "single_team_residual_bc.pt"
    training = train_team_residual_displacement_bc(
        train_paths,
        checkpoint,
        team_id=team_id,
        validation_paths=validation_paths,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hidden_size=args.hidden_size,
        history_steps=args.history_steps,
        seed=args.seed,
        device=args.device,
        observation_variant=args.observation_variant,
        early_stopping_patience=args.early_stopping_patience,
        minimum_epochs=args.minimum_epochs,
    )
    evaluations = {}
    for split, paths in (("train", train_paths), ("validation", validation_paths)):
        evaluation = evaluate_team_residual_displacement_bc(
            checkpoint, paths, team_id=team_id, device=args.device
        )
        evaluations[split] = evaluation
        (args.output_dir / f"{split}_evaluation.json").write_text(
            json.dumps(evaluation, indent=2)
        )
    result = {
        "method": "residual_displacement_bc",
        "phase": "pretest",
        "team_id": team_id,
        "split_matches": {
            split: _matches(manifest, split) for split in ("train", "validation", "test")
        },
        "training": training,
        "evaluations": evaluations,
        "selection_contract": {
            "selected_on": "validation_mean_displacement_error_meters",
            "epoch_zero_included": True,
            "test_loaded": False,
            "test_used_for_selection": False,
        },
    }
    (args.output_dir / "pretest_result.json").write_text(json.dumps(result, indent=2))
    config = {
        "manifest": str(args.manifest),
        "team_id": team_id,
        "architecture": "two_layer_mlp_residual_correction",
        "prediction": "previous_velocity_plus_learned_correction",
        "temporal_input": "causal_player_and_ball_velocity_history_meters",
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "hidden_size": args.hidden_size,
            "history_steps": args.history_steps,
            "seed": args.seed,
            "observation_variant": args.observation_variant,
            "early_stopping_patience": args.early_stopping_patience,
            "minimum_epochs": args.minimum_epochs,
        },
        "data_contract": {
            "train_matches": _matches(manifest, "train"),
            "validation_matches": _matches(manifest, "validation"),
            "test_matches_declared_but_not_loaded": _matches(manifest, "test"),
        },
    }
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(
        json.dumps(
            {
                "checkpoint": str(checkpoint),
                "pretest_result": str(args.output_dir / "pretest_result.json"),
                "selected_epoch": training["selected_epoch"],
                "validation_error_meters": training[
                    "best_validation_mean_displacement_error_meters"
                ],
                "validation_previous_velocity_meters": training[
                    "epoch_zero_validation_mean_displacement_error_meters"
                ],
                "strictly_beats_previous_velocity": training[
                    "strictly_beats_previous_velocity_on_validation"
                ],
                "test_loaded": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
