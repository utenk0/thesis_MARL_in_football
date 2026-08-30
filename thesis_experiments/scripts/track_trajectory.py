"""Follow a real 22-player trajectory with a closed-loop GRF controller."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from thesis_experiments.data.dataset_a_grf_audit import build_dataset_a_grf_audit
from thesis_experiments.data.metrica_grf_audit import build_metrica_grf_audit
from thesis_experiments.simulation.closed_loop import run_closed_loop


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["dataset_a", "metrica"])
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--start-frame", type=int)
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--stop-tolerance", type=float, default=.008)
    parser.add_argument("--sprint-threshold", type=float, default=.08)
    parser.add_argument("--sprint-release-threshold", type=float, default=.04)
    args = parser.parse_args()
    audit = (
        build_dataset_a_grf_audit(args.input, start_frame=args.start_frame, frame_count=args.frames)
        if args.dataset == "dataset_a"
        else build_metrica_grf_audit(args.input, start_frame=args.start_frame or 1, frame_count=args.frames)
    )
    report = run_closed_loop(audit, args.output_dir, live=args.live, record=args.record, fps=args.fps, stop_tolerance=args.stop_tolerance, sprint_threshold=args.sprint_threshold, sprint_release_threshold=args.sprint_release_threshold)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
