"""Pretrain the shared 22-player actor with behavioural cloning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from thesis_experiments.training.bc import train_bc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--observation-variant",
        choices=["spatial", "temporal_context", "full"],
        default="full",
    )
    parser.add_argument(
        "--class-weighting",
        choices=["none", "inverse_frequency"],
        default="none",
    )
    parser.add_argument("--max-class-weight", type=float, default=20.0)
    args = parser.parse_args()
    print(json.dumps(train_bc(
        args.data,
        args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hidden_size=args.hidden_size,
        seed=args.seed,
        device=args.device,
        observation_variant=args.observation_variant,
        class_weighting=args.class_weighting,
        max_class_weight=args.max_class_weight,
    ), indent=2))


if __name__ == "__main__":
    main()
