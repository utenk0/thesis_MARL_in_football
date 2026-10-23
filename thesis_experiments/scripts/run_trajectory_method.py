"""Train or evaluate one method through the shared trajectory interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from thesis_experiments.evaluation.trajectory_contract import load_contract
from thesis_experiments.methods.trajectory import (
    TrajectoryDatasetSplit,
    TrajectoryTrainingConfig,
)
from thesis_experiments.methods.trajectory_registry import (
    available_trajectory_methods,
    load_trajectory_method,
    train_trajectory_method,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        choices=available_trajectory_methods(),
        default="bc",
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=4)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train")
    train.add_argument("--output-dir", required=True, type=Path)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--epochs", type=int, default=10)
    train.add_argument("--batch-size", type=int, default=2048)
    train.add_argument("--learning-rate", type=float, default=3e-4)
    train.add_argument("--hidden-size", type=int, default=128)
    train.add_argument("--early-stopping-patience", type=int, default=3)
    train.add_argument("--minimum-epochs", type=int, default=5)
    train.add_argument(
        "--cql-action-count",
        type=int,
        default=32,
        help="Number of train-only trajectory-plan actions, including constant velocity.",
    )
    train.add_argument(
        "--cql-alpha",
        type=float,
        default=1.0,
        help="Weight of the conservative log-sum-exp Q penalty.",
    )
    train.add_argument("--cql-gamma", type=float, default=0.99)
    train.add_argument("--cql-target-update-interval", type=int, default=500)
    train.add_argument("--cql-prototype-samples", type=int, default=100_000)
    train.add_argument("--cql-kmeans-iterations", type=int, default=15)
    train.add_argument(
        "--ctde-bc-checkpoint",
        type=Path,
        help="Frozen trajectory-BC checkpoint used to initialize the local actor.",
    )
    train.add_argument("--ctde-critic-epochs", type=int, default=20)
    train.add_argument("--ctde-gamma", type=float, default=0.99)
    train.add_argument("--ctde-temperature", type=float, default=1.0)
    train.add_argument("--ctde-max-weight", type=float, default=20.0)
    train.add_argument("--ctde-bc-coefficient", type=float, default=0.5)
    train.add_argument("--ctde-critic-hidden-size", type=int, default=256)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--checkpoint", required=True, type=Path)
    evaluate.add_argument("--split", choices=["validation", "test"], default="validation")
    evaluate.add_argument("--allow-test", action="store_true")
    evaluate.add_argument("--output", required=True, type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.torch_threads < 1:
        raise ValueError("--torch-threads must be positive")
    torch.set_num_threads(args.torch_threads)
    contract = load_contract(args.contract)
    dataset = TrajectoryDatasetSplit.from_manifest(
        args.manifest,
        contract_path=args.contract,
    )

    if args.command == "train":
        config = TrajectoryTrainingConfig(
            seed=args.seed,
            horizons=tuple(int(value) for value in contract["forecast"]["horizons_steps"]),
            history_steps=int(contract["information"]["causal_history_steps"]),
            observation_variant=str(contract["information"]["observation_variant"]),
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            hidden_size=args.hidden_size,
            early_stopping_patience=args.early_stopping_patience,
            minimum_epochs=args.minimum_epochs,
            device=args.device,
            cql_action_count=args.cql_action_count,
            cql_alpha=args.cql_alpha,
            cql_gamma=args.cql_gamma,
            cql_target_update_interval=args.cql_target_update_interval,
            cql_prototype_samples=args.cql_prototype_samples,
            cql_kmeans_iterations=args.cql_kmeans_iterations,
            ctde_bc_checkpoint=args.ctde_bc_checkpoint,
            ctde_critic_epochs=args.ctde_critic_epochs,
            ctde_gamma=args.ctde_gamma,
            ctde_temperature=args.ctde_temperature,
            ctde_max_weight=args.ctde_max_weight,
            ctde_bc_coefficient=args.ctde_bc_coefficient,
            ctde_critic_hidden_size=args.ctde_critic_hidden_size,
        )
        result = train_trajectory_method(
            args.method,
            dataset,
            args.output_dir,
            config,
        )
        print(json.dumps(result.to_dict(), indent=2))
        return

    if args.split == "test" and not args.allow_test:
        raise ValueError("Test evaluation requires the explicit --allow-test flag.")
    method = load_trajectory_method(args.method, args.checkpoint, device=args.device)
    if list(method.horizons) != list(contract["forecast"]["horizons_steps"]):
        raise ValueError("Checkpoint horizons do not match the trajectory contract.")
    evaluation = method.evaluate(
        dataset.paths(args.split),
        team_id=dataset.team_id,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evaluation, indent=2) + "\n")
    print(
        json.dumps(
            {
                "method": args.method,
                "split": args.split,
                "output": str(args.output),
                "ade_meters": evaluation["overall"]["policy"][
                    "autonomous_ade_meters"
                ],
                "fde_meters": evaluation["overall"]["policy"][
                    "autonomous_fde_meters"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
