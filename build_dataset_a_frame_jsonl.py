from __future__ import annotations

import argparse
import csv
import json
import re
import xml.etree.ElementTree as ET
import zipfile
from bisect import bisect_left
from datetime import datetime
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_ZIP_PATH = BASE_DIR / ".venv" / "datasets" / "datasetA.zip"


def parse_iso_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value)


def build_match_file_map(zip_path: Path) -> dict[str, dict[str, str]]:
    match_map: dict[str, dict[str, str]] = {}

    with zipfile.ZipFile(zip_path) as archive:
        for name in archive.namelist():
            if not name.endswith(".xml"):
                continue

            stem = Path(name).stem
            if "_DFL-MAT-" not in stem:
                continue

            prefix, match_id = stem.rsplit("_", 1)
            slots = match_map.setdefault(match_id, {})

            if "events_raw" in prefix:
                slots["events"] = name

    return match_map


def infer_match_id_from_path(path: Path) -> str:
    match = re.search(r"(DFL-MAT-[A-Z0-9]+)", path.name)
    if not match:
        raise SystemExit(f"Could not infer match id from {path.name}")
    return match.group(1)


def build_frame_index(sorted_positions_csv: Path, explicit_match_id: str | None) -> tuple[str, list[int], list[datetime]]:
    match_id = explicit_match_id or ""
    frames: list[int] = []
    timestamps: list[datetime] = []
    last_frame: int | None = None

    with sorted_positions_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if not match_id and row.get("match_id"):
                match_id = row["match_id"]

            frame = int(row["frame"])
            if last_frame is not None and frame < last_frame:
                raise SystemExit(f"{sorted_positions_csv} is not frame-sorted.")

            if frame != last_frame:
                frames.append(frame)
                timestamps.append(parse_iso_datetime(row["timestamp"]))
                last_frame = frame

    if not frames:
        raise SystemExit(f"No rows found in {sorted_positions_csv}")

    if not match_id:
        match_id = infer_match_id_from_path(sorted_positions_csv)

    return match_id, frames, timestamps


def nearest_frame(event_time: datetime, frames: list[int], timestamps: list[datetime]) -> int:
    index = bisect_left(timestamps, event_time)
    if index <= 0:
        return frames[0]
    if index >= len(frames):
        return frames[-1]

    before_delta = abs((event_time - timestamps[index - 1]).total_seconds())
    after_delta = abs((timestamps[index] - event_time).total_seconds())
    return frames[index - 1] if before_delta <= after_delta else frames[index]


def parse_events_by_frame(
    zip_path: Path,
    events_xml: str,
    frames: list[int],
    timestamps: list[datetime],
) -> dict[int, list[dict[str, object]]]:
    by_frame: dict[int, list[dict[str, object]]] = {}

    with zipfile.ZipFile(zip_path) as archive:
        with archive.open(events_xml) as xml_file:
            root = ET.parse(xml_file).getroot()

    for event in root:
        child = next(iter(event), None)
        event_time_raw = event.attrib.get("EventTime", "")
        anchor_frame = (
            int(event.attrib["StartFrame"])
            if "StartFrame" in event.attrib
            else nearest_frame(parse_iso_datetime(event_time_raw), frames, timestamps)
        )

        payload = {
            "event_id": event.attrib.get("EventId", ""),
            "event_type": child.tag if child is not None else "Unknown",
            "event_time": event_time_raw,
            "anchor_frame": anchor_frame,
            "start_frame": int(event.attrib["StartFrame"]) if "StartFrame" in event.attrib else None,
            "end_frame": int(event.attrib["EndFrame"]) if "EndFrame" in event.attrib else None,
            "x_position": event.attrib.get("X-Position", ""),
            "y_position": event.attrib.get("Y-Position", ""),
            "details": dict(child.attrib) if child is not None else {},
        }
        by_frame.setdefault(anchor_frame, []).append(payload)

    return by_frame


def flush_frame(out_handle, current_frame: dict[str, object] | None) -> None:
    if current_frame is None:
        return
    out_handle.write(json.dumps(current_frame, ensure_ascii=False) + "\n")


def build_frame_jsonl(
    sorted_positions_csv: Path,
    output_jsonl: Path,
    zip_path: Path,
    explicit_match_id: str | None = None,
) -> int:
    match_id, frames, timestamps = build_frame_index(sorted_positions_csv, explicit_match_id)
    match_files = build_match_file_map(zip_path)
    events_xml = match_files.get(match_id, {}).get("events")
    if not events_xml:
        raise SystemExit(f"No events XML found for {match_id} in {zip_path}")

    events_by_frame = parse_events_by_frame(zip_path, events_xml, frames, timestamps)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    frame_count = 0
    with (
        sorted_positions_csv.open(newline="", encoding="utf-8") as in_handle,
        output_jsonl.open("w", encoding="utf-8") as out_handle,
    ):
        reader = csv.DictReader(in_handle)
        current_frame_id: str | None = None
        current_frame: dict[str, object] | None = None

        for row in reader:
            frame_id = row["frame"]
            if frame_id != current_frame_id:
                flush_frame(out_handle, current_frame)
                current_frame_id = frame_id
                frame_number = int(frame_id)
                current_frame = {
                    "match_id": row.get("match_id", match_id),
                    "frame": frame_number,
                    "timestamp": row["timestamp"],
                    "context": {
                        "game_section": row["game_section"],
                        "match_time_minute": row["match_time_minute"],
                        "events": events_by_frame.get(frame_number, []),
                    },
                    "state": {
                        "ball": None,
                        "players": [],
                    },
                }
                frame_count += 1

            if row["entity_type"] == "ball":
                current_frame["state"]["ball"] = {
                    "person_id": row["person_id"],
                    "x": row["x"],
                    "y": row["y"],
                    "z": row["z"],
                    "speed": row["speed"],
                    "acceleration": row["acceleration"],
                    "direction": row["direction"],
                    "ball_possession": row["ball_possession"],
                    "ball_status": row["ball_status"],
                }
            else:
                current_frame["state"]["players"].append(
                    {
                        "person_id": row["person_id"],
                        "short_name": row["short_name"],
                        "shirt_number": row["shirt_number"],
                        "team_id": row["team_id"],
                        "team_name": row["team_name"],
                        "x": row["x"],
                        "y": row["y"],
                        "speed": row["speed"],
                        "acceleration": row["acceleration"],
                        "direction": row["direction"],
                        "motion_status": row["motion_status"],
                    }
                )

        flush_frame(out_handle, current_frame)

    return frame_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build one JSON object per frame and attach Dataset A events.")
    parser.add_argument("sorted_positions_csv", type=Path, help="Frame-sorted positions CSV")
    parser.add_argument("output_jsonl", type=Path, help="Output JSONL path")
    parser.add_argument("--match-id", help="Optional override if the CSV does not contain match_id.")
    parser.add_argument(
        "--zip-path",
        type=Path,
        default=DEFAULT_ZIP_PATH,
        help=f"Path to datasetA.zip (default: {DEFAULT_ZIP_PATH})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame_count = build_frame_jsonl(args.sorted_positions_csv, args.output_jsonl, args.zip_path, args.match_id)
    print(f"Wrote {frame_count} frame objects to {args.output_jsonl}")


if __name__ == "__main__":
    main()
