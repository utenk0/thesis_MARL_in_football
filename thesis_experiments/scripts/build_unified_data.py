"""Build and validate Stage 4 provider-independent frame/event JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from thesis_experiments.frames.parsers import parse_dataset_a_frames, parse_metrica_frames
from thesis_experiments.frames.validation import FrameStreamValidator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["dataset_a", "metrica"])
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, help="Optional validation/export frame limit.")
    args = parser.parse_args()
    source = parse_dataset_a_frames(args.input) if args.dataset == "dataset_a" else parse_metrica_frames(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    validator = FrameStreamValidator()
    with args.output.open("w", encoding="utf-8") as handle:
        for index, frame in enumerate(source):
            if args.limit is not None and index >= args.limit:
                break
            validator.add(frame)
            handle.write(json.dumps(frame.to_dict(), ensure_ascii=False) + "\n")
    summary = validator.summary()
    print(json.dumps({**summary, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
