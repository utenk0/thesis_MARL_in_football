"""Prepare bounded real-data clips with disjoint train/validation/test matches."""

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET
import zipfile

import numpy as np

from thesis_experiments.scripts.extract_dataset_a_positions import build_match_file_map, parse_match_metadata
from thesis_experiments.scripts.build_dataset_a_frame_jsonl import parse_events_by_frame
from thesis_experiments.data.dataset_a_grf_audit import build_dataset_a_grf_audit, _infer_actions
from thesis_experiments.frames.parsers import parse_metrica_csv_frames, parse_metrica_epts_frames
from thesis_experiments.frames.grf import convert_frame_to_grf
from thesis_experiments.transitions.builder import build_joint_transitions


def dataset_a_clip(archive_path, match_id, files, output, start, count):
    teams, metadata = parse_match_metadata(archive_path, files["matchinfo"])
    records = {}
    with zipfile.ZipFile(archive_path) as archive, archive.open(files["positions"]) as stream:
        context = None
        for kind, element in ET.iterparse(stream, events=("start", "end")):
            if kind == "start" and element.tag == "FrameSet":
                context = dict(element.attrib)
            elif kind == "end" and element.tag == "Frame" and context is not None:
                frame = int(element.attrib["N"])
                team = context.get("TeamId")
                if start <= frame < start + count and (team in teams or team == "BALL"):
                    record = records.setdefault(frame, {
                        "match_id": match_id, "frame": frame, "timestamp": element.attrib["T"],
                        "context": {"game_section": context.get("GameSection"), "events": []},
                        "state": {"players": [], "ball": None},
                    })
                    xy = {"x": float(element.attrib["X"]), "y": float(element.attrib["Y"])}
                    if team == "BALL":
                        record["state"]["ball"] = xy
                    else:
                        person = context.get("PersonId")
                        record["state"]["players"].append(dict(xy, person_id=person, team_id=team,
                            shirt_number=metadata.get(person, {}).get("shirt_number")))
                element.clear()
            elif kind == "end" and element.tag == "FrameSet":
                context = None
                element.clear()
        frames = sorted(records)
        if len(frames) != count:
            raise ValueError(f"{match_id}: requested {count} frames at {start}, found {len(frames)}")
        event_file = next(n for n in archive.namelist() if "events_raw" in n and match_id in n and n.endswith(".xml"))
    events = parse_events_by_frame(archive_path, event_file, frames,
                                  [datetime.fromisoformat(records[f]["timestamp"]) for f in frames])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        for frame in frames:
            records[frame]["context"]["events"] = events.get(frame, [])
            handle.write(json.dumps(records[frame]) + "\n")
    return build_joint_transitions(build_dataset_a_grf_audit(output, start_frame=start, frame_count=count), source="dataset_a")


def metrica_clip(game_dir, count):
    iterator = parse_metrica_epts_frames(game_dir) if game_dir.name.endswith("3") else parse_metrica_csv_frames(game_dir)
    window = []
    roster = None
    for frame in iterator:
        current = tuple(p.player_id for p in frame.left_players + frame.right_players)
        if not frame.valid or frame.ball is None:
            window, roster = [], None
            continue
        if window and (current != roster or frame.frame_id != window[-1].frame_id + 1
                       or frame.raw.get("period") != window[-1].raw.get("period")):
            window = []
        roster = current
        window.append(frame)
        if len(window) == count:
            break
    if len(window) != count:
        raise ValueError(f"No complete {count}-frame clip in {game_dir}")
    frames = [convert_frame_to_grf(frame) for frame in window]
    first = frames[0]
    players = np.asarray([[[p.x, p.y] for p in f.left_players + f.right_players] for f in frames], dtype=np.float32)
    actions = _infer_actions(players, 0.0005)
    ids = [p.player_id for p in first.left_players + first.right_players]
    events = []
    for index, frame in enumerate(window[:-1]):
        for event in frame.events_at_frame:
            raw = dict(event.raw)
            raw["Start Frame"] = frame.frame_id
            raw["Team"] = "Home" if event.team_id == first.left_team_id else "Away"
            events.append(raw)
            if event.player_id in ids:
                action = event.grf_action
                if action is not None:
                    actions[index][ids.index(event.player_id)] = action
    audit = SimpleNamespace(
        frames=np.asarray([f.frame_id for f in frames]), match_id=first.match_id,
        home_grf_xy=players[:, :11], away_grf_xy=players[:, 11:],
        ball_grf_xy=np.asarray([[f.ball.x, f.ball.y] for f in frames], dtype=np.float32),
        joint_actions=np.asarray(actions), events=events,
        home_player_ids=ids[:11], away_player_ids=ids[11:],
        left_team_id=first.left_team_id, right_team_id=first.right_team_id,
    )
    return build_joint_transitions(audit, source="metrica")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-a", type=Path, default=Path("data/raw/dataset_a/datasetA.zip"))
    parser.add_argument("--metrica", type=Path, default=Path("data/raw/metrica_sample/data"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/training_stage"))
    parser.add_argument("--frames", type=int, default=512)
    parser.add_argument("--start-frame", type=int, default=10000)
    args = parser.parse_args()
    if args.frames < 2:
        parser.error("--frames must be at least 2")
    match_files = build_match_file_map(args.dataset_a)
    manifest = {"scope": "bounded training-stage clips; not full-match experiments", "splits": {k: [] for k in ("train", "validation", "test")}}

    def save(dataset, split):
        path = args.output_dir / split / f"{dataset.source}_{dataset.match_id}.npz"
        dataset.save(path)
        record = {"path": str(path), "source": dataset.source, "match_id": dataset.match_id,
                  "transitions": len(dataset.actions), "start_frame": int(dataset.frames[0]),
                  "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        manifest["splits"][split].append(record)
        print(json.dumps(dict(record, split=split)), flush=True)

    for index, match in enumerate(sorted(match_files)):
        split = "train" if index < len(match_files) - 2 else "validation" if index == len(match_files) - 2 else "test"
        dataset = dataset_a_clip(args.dataset_a, match, match_files[match],
            args.output_dir / "provider_clips" / f"{match}.jsonl", args.start_frame, args.frames)
        save(dataset, split)
    for game, split in [(1, "train"), (2, "validation"), (3, "test")]:
        save(metrica_clip(args.metrica / f"Sample_Game_{game}", args.frames), split)
    path = args.output_dir / "split_manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    print(path)


if __name__ == "__main__":
    main()
