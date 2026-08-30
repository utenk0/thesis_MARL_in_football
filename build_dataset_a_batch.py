from __future__ import annotations

import argparse
from pathlib import Path

from build_dataset_a_frame_jsonl import build_frame_jsonl
from extract_dataset_a_positions import (
    DEFAULT_ZIP_PATH,
    iter_complete_matches,
    parse_match_metadata,
    write_positions_csv,
)
from sort_positions_by_frame import chunked_sort


BASE_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Dataset A outputs for all matches: positions CSV, frame-sorted CSV, and frame JSONL with events."
    )
    parser.add_argument(
        "--zip-path",
        type=Path,
        default=DEFAULT_ZIP_PATH,
        help=f"Path to datasetA.zip (default: {DEFAULT_ZIP_PATH})",
    )
    parser.add_argument(
        "--positions-dir",
        type=Path,
        default=BASE_DIR / "outputs" / "positions",
        help="Directory for unsorted positions CSVs.",
    )
    parser.add_argument(
        "--sorted-dir",
        type=Path,
        default=BASE_DIR / "outputs" / "positions_by_frame",
        help="Directory for frame-sorted positions CSVs.",
    )
    parser.add_argument(
        "--frames-dir",
        type=Path,
        default=BASE_DIR / "outputs" / "frames_jsonl",
        help="Directory for frame-level JSONL outputs.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=200_000,
        help="Rows per in-memory chunk for external frame sorting.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.positions_dir.mkdir(parents=True, exist_ok=True)
    args.sorted_dir.mkdir(parents=True, exist_ok=True)
    args.frames_dir.mkdir(parents=True, exist_ok=True)

    total_position_rows = 0
    total_frame_objects = 0
    matches = iter_complete_matches(args.zip_path)

    for match_id, matchinfo_xml, positions_xml in matches:
        positions_csv = args.positions_dir / f"dataset_a_positions_{match_id}.csv"
        sorted_csv = args.sorted_dir / f"dataset_a_positions_{match_id}_by_frame.csv"
        frames_jsonl = args.frames_dir / f"dataset_a_frames_{match_id}.jsonl"

        team_names, players = parse_match_metadata(args.zip_path, matchinfo_xml)
        position_rows = write_positions_csv(
            zip_path=args.zip_path,
            match_id=match_id,
            positions_xml=positions_xml,
            output_csv=positions_csv,
            team_names=team_names,
            players=players,
        )
        total_position_rows += position_rows
        print(f"{match_id}: wrote {position_rows} position rows")

        chunked_sort(positions_csv, sorted_csv, args.chunk_size)
        print(f"{match_id}: wrote frame-sorted CSV to {sorted_csv}")

        frame_count = build_frame_jsonl(sorted_csv, frames_jsonl, args.zip_path, match_id)
        total_frame_objects += frame_count
        print(f"{match_id}: wrote {frame_count} frame objects")

    print(
        f"Finished {len(matches)} matches with {total_position_rows} position rows and {total_frame_objects} frame objects."
    )


if __name__ == "__main__":
    main()
