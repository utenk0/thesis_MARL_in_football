"""Build the fixed match-disjoint thesis dataset and a laptop-sized pilot subset.

Dataset A supplies the train/validation/in-domain-test matches.  Metrica is kept
entirely out of training and is written as an external-test split.  Long valid
runs are stored in bounded files so training jobs can select or stream chunks
without loading a complete match conversion at once.
"""

from __future__ import annotations

import argparse
from array import array
from collections import defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET
import zipfile

import numpy as np

from thesis_experiments.data.dataset_a_grf_audit import (
    _events_and_overrides,
    _infer_actions,
)
from thesis_experiments.frames.grf import convert_frame_to_grf
from thesis_experiments.frames.parsers import (
    parse_metrica_csv_frames,
    parse_metrica_epts_frames,
)
from thesis_experiments.scripts.build_dataset_a_frame_jsonl import parse_events_by_frame
from thesis_experiments.scripts.extract_dataset_a_positions import (
    build_match_file_map,
    parse_match_metadata,
)
from thesis_experiments.transitions.builder import build_joint_transitions
from thesis_experiments.transitions.observations import OBSERVATION_VERSION


DATASET_A_SPLIT = {
    "train": (
        "DFL-MAT-J03WN1",
        "DFL-MAT-J03WOH",
        "DFL-MAT-J03WOY",
        "DFL-MAT-J03WPY",
        "DFL-MAT-J03WQQ",
    ),
    "validation": ("DFL-MAT-J03WR9",),
    "test": ("DFL-MAT-J03WMX",),
}
COMMON_ACTION_IDS = tuple(range(13)) + (14, 16)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _action_counts(actions: np.ndarray) -> list[int]:
    return np.bincount(np.asarray(actions).reshape(-1), minlength=19).tolist()


def _partitions(start: int, end: int, chunk_frames: int, min_chunk_frames: int):
    """Partition [start,end) without leaving an unusably small final chunk."""

    cursor = start
    while end - cursor > chunk_frames + min_chunk_frames:
        yield cursor, cursor + chunk_frames
        cursor += chunk_frames
    if end - cursor >= min_chunk_frames:
        yield cursor, end


def _convert_dataset_a_xy(values: np.ndarray) -> np.ndarray:
    converted = np.empty_like(values, dtype=np.float32)
    converted[..., 0] = np.clip(values[..., 0] / 52.5, -1.0, 1.0)
    converted[..., 1] = np.clip(0.42 * values[..., 1] / 34.0, -0.42, 0.42)
    return converted


def _player_key(player_id: str, metadata: dict[str, dict[str, str]]) -> tuple[bool, str]:
    return str(metadata.get(player_id, {}).get("shirt_number")) != "1", player_id


def _dataset_a_tracks(archive_path: Path, files: dict[str, str]):
    """Read one entity-grouped tracking XML into compact per-entity NumPy tracks."""

    teams, metadata = parse_match_metadata(archive_path, files["matchinfo"])
    segments: dict[tuple[str, str], list[tuple[str, np.ndarray, np.ndarray, list[str]]]] = defaultdict(list)
    with zipfile.ZipFile(archive_path) as archive, archive.open(files["positions"]) as stream:
        context = None
        frame_values: array | None = None
        xy_values: array | None = None
        timestamps: list[str] | None = None
        selected = False
        for kind, element in ET.iterparse(stream, events=("start", "end")):
            if kind == "start" and element.tag == "FrameSet":
                context = dict(element.attrib)
                team_id = context.get("TeamId", "")
                selected = team_id == "BALL" or team_id in teams
                if selected:
                    frame_values = array("q")
                    xy_values = array("f")
                    timestamps = [] if team_id == "BALL" else None
            elif kind == "end" and element.tag == "Frame" and selected:
                assert frame_values is not None and xy_values is not None and context is not None
                frame_values.append(int(element.attrib["N"]))
                xy_values.extend((float(element.attrib["X"]), float(element.attrib["Y"])))
                if timestamps is not None:
                    timestamps.append(element.attrib["T"])
                element.clear()
            elif kind == "end" and element.tag == "FrameSet":
                if selected and context is not None and frame_values is not None and xy_values is not None:
                    entity = context.get("PersonId", "") if context.get("TeamId") != "BALL" else "BALL"
                    segments[(context.get("TeamId", ""), entity)].append(
                        (
                            context.get("GameSection", ""),
                            np.asarray(frame_values, dtype=np.int64),
                            np.asarray(xy_values, dtype=np.float32).reshape(-1, 2),
                            timestamps or [],
                        )
                    )
                context = frame_values = xy_values = timestamps = None
                selected = False
                element.clear()
    return teams, metadata, segments


def _dataset_a_match_arrays(archive_path: Path, match_id: str, files: dict[str, str]):
    teams, metadata, segments = _dataset_a_tracks(archive_path, files)
    ball_segments = segments.get(("BALL", "BALL"), [])
    if not ball_segments:
        raise ValueError(f"{match_id}: no ball track")
    frames = np.concatenate([item[1] for item in ball_segments])
    ball = np.concatenate([item[2] for item in ball_segments])
    timestamps = [value for item in ball_segments for value in item[3]]
    periods = np.concatenate([
        np.repeat(item[0], len(item[1])) for item in ball_segments
    ])
    order = np.argsort(frames, kind="stable")
    frames, ball, periods = frames[order], ball[order], periods[order]
    timestamps = [timestamps[index] for index in order]
    if len(np.unique(frames)) != len(frames):
        raise ValueError(f"{match_id}: duplicate ball frames")

    team_players = {
        team: sorted(
            [person for person, item in metadata.items() if item.get("team_id") == team],
            key=lambda person: _player_key(person, metadata),
        )
        for team in teams
    }
    all_players = [(team, player) for team in sorted(teams) for player in team_players[team]]
    positions = np.full((len(frames), len(all_players), 2), np.nan, dtype=np.float32)
    frame_lookup = {int(frame): index for index, frame in enumerate(frames)}
    for column, (team, player) in enumerate(all_players):
        for _, entity_frames, entity_xy, _ in segments.get((team, player), []):
            indices = np.fromiter(
                (frame_lookup.get(int(frame), -1) for frame in entity_frames),
                dtype=np.int64,
                count=len(entity_frames),
            )
            keep = indices >= 0
            positions[indices[keep], column] = entity_xy[keep]

    event_file = files.get("events")
    if not event_file:
        with zipfile.ZipFile(archive_path) as archive:
            event_file = next(
                name for name in archive.namelist()
                if "events_raw" in name and match_id in name and name.endswith(".xml")
            )
    events = parse_events_by_frame(
        archive_path,
        event_file,
        frames.tolist(),
        [datetime.fromisoformat(value) for value in timestamps],
    )

    teams_sorted = sorted(teams)
    initial = next(
        index for index in range(len(frames))
        if all(
            np.isfinite(positions[index, [i for i, item in enumerate(all_players) if item[0] == team]]).all(axis=1).sum() == 11
            for team in teams_sorted
        )
    )
    side_from_event = None
    for frame_events in events.values():
        for event in frame_events:
            details = event.get("details") or {}
            if details.get("TeamLeft") in teams and details.get("TeamRight") in teams:
                side_from_event = (str(details["TeamLeft"]), str(details["TeamRight"]))
                break
        if side_from_event:
            break
    if side_from_event is None:
        medians = {}
        for team in teams_sorted:
            columns = [i for i, item in enumerate(all_players) if item[0] == team]
            finite = np.isfinite(positions[initial, columns]).all(axis=1)
            medians[team] = float(np.median(positions[initial, np.asarray(columns)[finite], 0]))
        side_from_event = (min(teams_sorted, key=medians.get), max(teams_sorted, key=medians.get))
    left_team, right_team = side_from_event
    player_columns = {
        team: [index for index, item in enumerate(all_players) if item[0] == team]
        for team in teams
    }
    return {
        "frames": frames,
        "periods": periods,
        "ball": ball,
        "positions": positions,
        "all_players": all_players,
        "player_columns": player_columns,
        "left_team": left_team,
        "right_team": right_team,
        "events": events,
    }


def _dataset_a_runs(data: dict):
    frames = data["frames"]
    positions = data["positions"]
    team_columns = data["player_columns"]
    teams = (data["left_team"], data["right_team"])
    run_start = None
    previous_signature = None
    for index, frame in enumerate(frames):
        active = []
        for team in teams:
            columns = np.asarray(team_columns[team], dtype=np.int64)
            finite = np.isfinite(positions[index, columns]).all(axis=1)
            active.append(tuple(columns[finite].tolist()))
        valid = all(len(side) == 11 for side in active) and np.isfinite(data["ball"][index]).all()
        signature = (str(data["periods"][index]), *active) if valid else None
        contiguous = (
            run_start is not None
            and int(frame) == int(frames[index - 1]) + 1
            and signature == previous_signature
        )
        if run_start is not None and not contiguous:
            yield run_start, index, previous_signature
            run_start = None
        if valid and run_start is None:
            run_start = index
        previous_signature = signature
    if run_start is not None:
        yield run_start, len(frames), previous_signature


def _provenance(events: list[dict]) -> list[dict]:
    return [
        {
            "event_id": event.get("event_id", event.get("Event ID")),
            "event_type": event.get("event_type", event.get("Type")),
            "anchor_frame": event.get("anchor_frame", event.get("Start Frame")),
            "grf_action": event.get("grf_action"),
            "actor_id": event.get("resolved_actor_id", event.get("From")),
            "actor_team_id": event.get("resolved_actor_team_id", event.get("Team")),
            "joint_action_player_index": event.get("joint_action_player_index"),
            "status": event.get("action_override_status"),
        }
        for event in events
        if event.get("grf_action") is not None
    ]


def _dataset_a_chunks(archive_path: Path, match_id: str, files: dict[str, str], chunk_frames: int, min_chunk_frames: int):
    data = _dataset_a_match_arrays(archive_path, match_id, files)
    for run_start, run_end, signature in _dataset_a_runs(data):
        if signature is None:
            continue
        for start, end in _partitions(run_start, run_end, chunk_frames, min_chunk_frames):
            left_columns, right_columns = signature[1], signature[2]
            left_ids = [data["all_players"][index][1] for index in left_columns]
            right_ids = [data["all_players"][index][1] for index in right_columns]
            raw_players = np.concatenate(
                [data["positions"][start:end, left_columns], data["positions"][start:end, right_columns]],
                axis=1,
            )
            players = _convert_dataset_a_xy(raw_players)
            actions = _infer_actions(players, 0.0005)
            records = [
                {
                    "frame": int(frame),
                    "context": {"events": data["events"].get(int(frame), [])},
                }
                for frame in data["frames"][start:end]
            ]
            events = _events_and_overrides(records, actions, left_ids, right_ids)
            audit = SimpleNamespace(
                frames=data["frames"][start:end],
                match_id=match_id,
                home_grf_xy=players[:, :11],
                away_grf_xy=players[:, 11:],
                ball_grf_xy=_convert_dataset_a_xy(data["ball"][start:end]),
                joint_actions=np.asarray(actions, dtype=np.int64),
                events=events,
                home_player_ids=left_ids,
                away_player_ids=right_ids,
                left_team_id=data["left_team"],
                right_team_id=data["right_team"],
            )
            yield build_joint_transitions(audit, source="dataset_a"), _provenance(events)


def _metrica_dataset(frames):
    converted = [convert_frame_to_grf(frame) for frame in frames]
    first = converted[0]
    players = np.asarray(
        [[[player.x, player.y] for player in frame.left_players + frame.right_players] for frame in converted],
        dtype=np.float32,
    )
    actions = _infer_actions(players, 0.0005)
    player_ids = [player.player_id for player in first.left_players + first.right_players]
    events = []
    for index, frame in enumerate(frames):
        for event in frame.events_at_frame:
            raw = dict(event.raw)
            raw["Start Frame"] = frame.frame_id
            raw["Team"] = "Home" if event.team_id == first.left_team_id else "Away"
            raw["grf_action"] = event.grf_action
            raw["resolved_actor_id"] = event.player_id
            raw["resolved_actor_team_id"] = event.team_id
            action = event.grf_action
            if action is not None and index < len(actions) and event.player_id in player_ids:
                player_index = player_ids.index(event.player_id)
                actions[index][player_index] = action
                raw["joint_action_player_index"] = player_index
                raw["action_override_status"] = "applied"
            elif action is not None:
                raw["action_override_status"] = "unmatched_actor"
            events.append(raw)
    audit = SimpleNamespace(
        frames=np.asarray([frame.frame_id for frame in converted], dtype=np.int64),
        match_id=first.match_id,
        home_grf_xy=players[:, :11],
        away_grf_xy=players[:, 11:],
        ball_grf_xy=np.asarray([[frame.ball.x, frame.ball.y] for frame in converted], dtype=np.float32),
        joint_actions=np.asarray(actions, dtype=np.int64),
        events=events,
        home_player_ids=player_ids[:11],
        away_player_ids=player_ids[11:],
        left_team_id=first.left_team_id,
        right_team_id=first.right_team_id,
    )
    return build_joint_transitions(audit, source="metrica"), _provenance(events)


def _metrica_chunks(game_dir: Path, chunk_frames: int, min_chunk_frames: int):
    iterator = parse_metrica_epts_frames(game_dir) if game_dir.name.endswith("3") else parse_metrica_csv_frames(game_dir)
    buffer = []
    signature = None

    def flush():
        nonlocal buffer
        if len(buffer) >= min_chunk_frames:
            result = _metrica_dataset(buffer)
            buffer = []
            return result
        buffer = []
        return None

    for frame in iterator:
        ids = tuple(player.player_id for player in frame.left_players + frame.right_players)
        current = (frame.raw.get("period"), frame.segment_id, ids)
        valid = frame.valid and frame.ball is not None
        contiguous = (
            bool(buffer)
            and valid
            and current == signature
            and frame.frame_id == buffer[-1].frame_id + 1
        )
        if buffer and not contiguous:
            result = flush()
            if result is not None:
                yield result
        if not valid:
            signature = None
            continue
        if not buffer:
            signature = current
        buffer.append(frame)
        if len(buffer) > chunk_frames + min_chunk_frames:
            selected, buffer = buffer[:chunk_frames], buffer[chunk_frames:]
            yield _metrica_dataset(selected)
    result = flush()
    if result is not None:
        yield result


def _save_chunk(dataset, provenance, output_dir: Path, split: str, chunk_index: int):
    stem = f"{dataset.source}_{dataset.match_id}_chunk_{chunk_index:04d}"
    path = output_dir / split / f"{stem}.npz"
    dataset.save(path)
    provenance_path = output_dir / "label_provenance" / f"{stem}.json"
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    provenance_path.write_text(json.dumps(provenance, indent=2))
    counts = _action_counts(dataset.actions)
    record = {
        "path": str(path),
        "source": dataset.source,
        "match_id": dataset.match_id,
        "chunk_index": chunk_index,
        "transitions": int(len(dataset.actions)),
        "start_frame": int(dataset.frames[0]),
        "end_frame": int(dataset.frames[-1]),
        "sha256": _sha256(path),
        "action_counts": counts,
        "event_overrides": sum(item.get("status") == "applied" for item in provenance),
        "mapped_events": len(provenance),
        "label_provenance": str(provenance_path),
    }
    print(json.dumps(dict(record, split=split)), flush=True)
    return record


def _pilot_manifest(manifest: dict, chunks_per_match: int) -> dict:
    pilot = {
        key: value for key, value in manifest.items()
        if key not in {"splits", "scope"}
    }
    pilot["scope"] = (
        "Laptop-sized, event-rich diagnostic subset selected only for pipeline/data-quality checks; "
        "not valid for final thesis comparison."
    )
    pilot["parent_manifest"] = "split_manifest.json"
    pilot["splits"] = {}
    for split, rows in manifest["splits"].items():
        selected = []
        for match_id in sorted({row["match_id"] for row in rows}):
            candidates = [row for row in rows if row["match_id"] == match_id]
            candidates.sort(
                key=lambda row: (
                    row["event_overrides"],
                    sum(row["action_counts"][9:]),
                    row["transitions"],
                ),
                reverse=True,
            )
            selected.extend(candidates[:chunks_per_match])
        pilot["splits"][split] = sorted(
            selected, key=lambda row: (row["match_id"], row["chunk_index"])
        )
    return pilot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-a", type=Path, default=Path("data/raw/dataset_a/datasetA.zip"))
    parser.add_argument("--metrica", type=Path, default=Path("data/raw/metrica_sample/data"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/final_stage_v1"))
    parser.add_argument("--chunk-frames", type=int, default=4096)
    parser.add_argument("--min-chunk-frames", type=int, default=512)
    parser.add_argument("--pilot-chunks-per-match", type=int, default=2)
    args = parser.parse_args()
    if args.chunk_frames < 2 or not 2 <= args.min_chunk_frames <= args.chunk_frames:
        parser.error("Require chunk-frames >= min-chunk-frames >= 2")
    if args.pilot_chunks_per_match < 1:
        parser.error("--pilot-chunks-per-match must be positive")
    manifest_path = args.output_dir / "split_manifest.json"
    if manifest_path.exists():
        parser.error(f"{manifest_path} already exists; use a new output directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    match_files = build_match_file_map(args.dataset_a)
    # The generic match map does not currently retain event XML names.
    with zipfile.ZipFile(args.dataset_a) as archive:
        for name in archive.namelist():
            if "events_raw" not in name or not name.endswith(".xml"):
                continue
            match_id = Path(name).stem.rsplit("_", 1)[-1]
            match_files.setdefault(match_id, {})["events"] = name

    manifest = {
        "version": "final_stage_v1",
        "scope": (
            "All continuous valid match runs meeting min_chunk_frames, with a fixed Dataset-A "
            "match split and a provider-held-out Metrica external test. Shorter disconnected "
            "runs are excluded and chunk boundaries do not cross rosters or periods."
        ),
        "observation_version": OBSERVATION_VERSION,
        "chunk_frames": args.chunk_frames,
        "min_chunk_frames": args.min_chunk_frames,
        "common_action_ids": list(COMMON_ACTION_IDS),
        "dataset_a_match_assignments": {key: list(value) for key, value in DATASET_A_SPLIT.items()},
        "splits": {key: [] for key in ("train", "validation", "test", "external_test")},
    }
    for split, matches in DATASET_A_SPLIT.items():
        for match_id in matches:
            if match_id not in match_files:
                raise ValueError(f"Dataset A archive is missing {match_id}")
            print(f"Preparing Dataset A {match_id} -> {split}", flush=True)
            for index, (dataset, provenance) in enumerate(
                _dataset_a_chunks(
                    args.dataset_a,
                    match_id,
                    match_files[match_id],
                    args.chunk_frames,
                    args.min_chunk_frames,
                )
            ):
                manifest["splits"][split].append(
                    _save_chunk(dataset, provenance, args.output_dir, split, index)
                )
    for game in (1, 2, 3):
        game_dir = args.metrica / f"Sample_Game_{game}"
        print(f"Preparing Metrica {game_dir.name} -> external_test", flush=True)
        for index, (dataset, provenance) in enumerate(
            _metrica_chunks(game_dir, args.chunk_frames, args.min_chunk_frames)
        ):
            manifest["splits"]["external_test"].append(
                _save_chunk(dataset, provenance, args.output_dir, "external_test", index)
            )
    manifest_path.write_text(json.dumps(manifest, indent=2))
    pilot_path = args.output_dir / "pilot_split_manifest.json"
    pilot_path.write_text(json.dumps(_pilot_manifest(manifest, args.pilot_chunks_per_match), indent=2))
    print(json.dumps({"manifest": str(manifest_path), "pilot_manifest": str(pilot_path)}, indent=2))


if __name__ == "__main__":
    main()
