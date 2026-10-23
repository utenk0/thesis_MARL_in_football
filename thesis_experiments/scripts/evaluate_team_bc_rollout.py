"""Evaluate a saved single-team BC checkpoint in conditional GRF rollouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from thesis_experiments.scripts.run_team_bc import conditional_reconstruction


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--split", choices=["validation", "test"], required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--evaluation-start", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--grf-python", type=Path)
    parser.add_argument("--chunks-per-match", type=int, default=1)
    args = parser.parse_args()
    if args.steps < 1 or args.warmup_steps < 0 or args.chunks_per_match < 1:
        parser.error("Steps/chunks must be positive and warmup non-negative")
    manifest = json.loads(args.manifest.read_text())
    team_id = str(manifest["team_id"])
    by_match: dict[str, list[Path]] = {}
    for row in manifest["splits"][args.split]:
        by_match.setdefault(str(row["match_id"]), []).append(Path(row["path"]))
    reports = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for match_id, paths in sorted(by_match.items()):
        if len(paths) <= args.chunks_per_match:
            selected_paths = paths
        else:
            indices = np.linspace(
                0, len(paths) - 1, args.chunks_per_match, dtype=np.int64
            )
            selected_paths = [paths[int(index)] for index in np.unique(indices)]
        for chunk_index, path in enumerate(selected_paths):
            for controller in ("bc_policy", "previous_action", "recorded_expert_actions"):
                artifact = (
                    args.output_dir
                    / f"{match_id}_{chunk_index:02d}_{controller}_warmup_{args.warmup_steps}.npz"
                )
                reports.append(
                    conditional_reconstruction(
                        args.checkpoint,
                        path,
                        artifact,
                        team_id=team_id,
                        steps=args.steps,
                        grf_python=args.grf_python,
                        seed=args.seed + chunk_index,
                        device=args.device,
                        focal_controller=controller,
                        warmup_steps=args.warmup_steps,
                        evaluation_start_transition=args.evaluation_start,
                    )
                )
    output = args.output_dir / "rollout_evaluation.json"
    output.write_text(json.dumps(reports, indent=2))
    print(json.dumps({"evaluation": str(output)}, indent=2))


if __name__ == "__main__":
    main()
