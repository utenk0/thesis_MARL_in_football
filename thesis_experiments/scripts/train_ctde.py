"""Train the centralized critic and fine-tune the shared actor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from thesis_experiments.training.ctde import train_ctde


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", nargs="+", required=True, type=Path)
    parser.add_argument("--bc-checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--critic-epochs", type=int, default=30)
    parser.add_argument("--actor-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-weight", type=float, default=20.0)
    parser.add_argument("--bc-coefficient", type=float, default=0.5)
    parser.add_argument("--critic-hidden-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(train_ctde(args.data, args.bc_checkpoint, args.output, critic_epochs=args.critic_epochs, actor_epochs=args.actor_epochs, batch_size=args.batch_size, learning_rate=args.learning_rate, gamma=args.gamma, temperature=args.temperature, max_weight=args.max_weight, bc_coefficient=args.bc_coefficient, critic_hidden_size=args.critic_hidden_size, seed=args.seed), indent=2))


if __name__ == "__main__":
    main()
