"""Convert provider events to the unified JSONL event schema."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from thesis_experiments.events.converters import (
    convert_dataset_a_events,
    convert_metrica_events,
    write_events_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["dataset_a", "metrica"])
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--start-frame", type=int)
    parser.add_argument("--end-frame", type=int)
    args = parser.parse_args()
    if args.dataset == "dataset_a":
        events = convert_dataset_a_events(args.input, start_frame=args.start_frame, end_frame=args.end_frame)
    else:
        events = convert_metrica_events(args.input, start_frame=args.start_frame, end_frame=args.end_frame)
    count = write_events_jsonl(events, args.output)
    print(json.dumps({
        "dataset": args.dataset, "output": str(args.output), "events": count,
        "event_types": dict(sorted(Counter(event.event_type for event in events).items())),
        "events_with_grf_action": sum(event.grf_action is not None for event in events),
        "events_with_position": sum(event.start_grf is not None for event in events),
    }, indent=2))


if __name__ == "__main__":
    main()
