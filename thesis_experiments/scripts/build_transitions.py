"""Build unified CTDE transitions from Dataset A or Metrica source data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from thesis_experiments.data.dataset_a_grf_audit import build_dataset_a_grf_audit
from thesis_experiments.data.metrica_grf_audit import build_metrica_grf_audit
from thesis_experiments.transitions.builder import build_joint_transitions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["dataset_a", "metrica"])
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--start-frame", type=int)
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--progress-scale", type=float, default=1.0)
    parser.add_argument("--goal-reward", type=float, default=1.0)
    args = parser.parse_args()
    if args.dataset == "dataset_a":
        audit = build_dataset_a_grf_audit(args.input, start_frame=args.start_frame, frame_count=args.frames)
    else:
        audit = build_metrica_grf_audit(args.input, start_frame=args.start_frame or 1, frame_count=args.frames)
    dataset = build_joint_transitions(audit, source=args.dataset, progress_scale=args.progress_scale, goal_reward=args.goal_reward)
    dataset.save(args.output)
    print(json.dumps({"output": str(args.output), "source": dataset.source, "match_id": dataset.match_id, "transitions": len(dataset.actions), "local_shape": list(dataset.local_observations.shape), "global_shape": list(dataset.global_states.shape), "actions_shape": list(dataset.actions.shape), "team_rewards_shape": list(dataset.team_rewards.shape)}, indent=2))


if __name__ == "__main__":
    main()
