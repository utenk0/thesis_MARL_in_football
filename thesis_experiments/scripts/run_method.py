"""Train or prepare one unified football learning method."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from thesis_experiments.methods import MethodConfig, available_methods, build_method


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=available_methods())
    parser.add_argument("--data", nargs="+", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/methods"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total-timesteps", type=int, default=10_000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--backend", default="auto")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--env-backend", choices=["grf", "replay"], default="grf")
    parser.add_argument("--grf-python", type=Path)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--episode-steps", type=int, default=128)
    parser.add_argument("--context-length", type=int, default=16)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--bc-checkpoint", type=Path)
    parser.add_argument("--actor-epochs", type=int, default=10, help="CTDE actor fine-tuning epochs.")
    parser.add_argument("--disc-updates", type=int, default=4, help="GAIL discriminator updates per round.")
    parser.add_argument(
        "--observation-variant",
        choices=["spatial", "temporal_context", "temporal_context_no_slot", "full"],
        default="full",
        help="BC feature ablation; excluded features remain present but are zeroed.",
    )
    parser.add_argument(
        "--class-weighting",
        choices=["none", "inverse_frequency"],
        default="none",
        help="BC loss weighting strategy.",
    )
    parser.add_argument("--max-class-weight", type=float, default=20.0)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(args.torch_threads)
    config = MethodConfig(
        method=args.method,
        data_paths=args.data,
        output_dir=args.output_dir,
        seed=args.seed,
        total_timesteps=args.total_timesteps,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hidden_size=args.hidden_size,
        backend=args.backend,
        device=args.device,
        env_backend=args.env_backend,
        grf_python=args.grf_python,
        rollout_steps=args.rollout_steps,
        episode_steps=args.episode_steps,
        context_length=args.context_length,
        bc_checkpoint=args.bc_checkpoint,
        extra={
            "actor_epochs": args.actor_epochs,
            "disc_updates": args.disc_updates,
            "observation_variant": args.observation_variant,
            "class_weighting": args.class_weighting,
            "max_class_weight": args.max_class_weight,
        },
    )
    result = build_method(config).train()
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
