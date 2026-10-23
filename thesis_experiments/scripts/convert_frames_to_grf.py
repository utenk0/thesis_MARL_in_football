"""Convert provider frames directly to validated GRF-world JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from thesis_experiments.frames.grf import convert_frame_to_grf
from thesis_experiments.frames.parsers import parse_dataset_a_frames, parse_metrica_frames
from thesis_experiments.frames.validation import FrameStreamValidator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["dataset_a", "metrica"])
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    source = parse_dataset_a_frames(args.input) if args.dataset == "dataset_a" else parse_metrica_frames(args.input)
    validator = FrameStreamValidator()
    points = clipped = 0
    x_min, x_max, y_min, y_max = 1.0, -1.0, 0.42, -0.42
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for index, frame in enumerate(source):
            if args.limit is not None and index >= args.limit:
                break
            converted = convert_frame_to_grf(frame)
            validator.add(converted)
            positions = [(p.x, p.y) for p in converted.left_players + converted.right_players]
            if converted.ball is not None:
                positions.append((converted.ball.x, converted.ball.y))
            for x, y in positions:
                points += 1
                x_min, x_max = min(x_min, x), max(x_max, x)
                y_min, y_max = min(y_min, y), max(y_max, y)
                clipped += int(abs(x) == 1.0 or abs(y) == 0.42)
            handle.write(json.dumps(converted.to_dict(), ensure_ascii=False) + "\n")
    report = {
        "dataset": args.dataset,
        **validator.summary(),
        "finite_points": points,
        "points_on_clipping_boundary": clipped,
        "grf_x_min_max": [x_min, x_max] if points else None,
        "grf_y_min_max": [y_min, y_max] if points else None,
        "output": str(args.output),
    }
    report_path = args.report or args.output.with_suffix(".validation.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
