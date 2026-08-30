"""Traceable Dataset A-to-GRF conversion for validation and replay."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from thesis_experiments.datasets.action_mapping import dataset_a_event_to_grf_action
from thesis_experiments.datasets.coordinates import dataset_a_to_grf, grf_to_dataset_a
from thesis_experiments.datasets.movement import movement_actions


@dataclass(slots=True)
class DatasetAGRFAudit:
    match_id: str
    frames: np.ndarray
    periods: np.ndarray
    times_seconds: np.ndarray
    left_team_id: str
    right_team_id: str
    left_player_ids: list[str]
    right_player_ids: list[str]
    left_raw_xy: np.ndarray
    right_raw_xy: np.ndarray
    ball_raw_xy: np.ndarray
    left_grf_xy: np.ndarray
    right_grf_xy: np.ndarray
    ball_grf_xy: np.ndarray
    joint_actions: np.ndarray
    events: list[dict[str, object]]

    # Aliases let the common GRF simulator consume either provider audit.
    @property
    def home_grf_xy(self):
        return self.left_grf_xy

    @property
    def away_grf_xy(self):
        return self.right_grf_xy

    def save(self, output_dir: Path) -> dict[str, object]:
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = "dataset_a_grf_100_frames"
        npz_path = output_dir / f"{stem}.npz"
        np.savez_compressed(
            npz_path, match_id=self.match_id, frames=self.frames, periods=self.periods,
            times_seconds=self.times_seconds, left_team_id=self.left_team_id,
            right_team_id=self.right_team_id, left_player_ids=np.asarray(self.left_player_ids),
            right_player_ids=np.asarray(self.right_player_ids), left_raw_xy=self.left_raw_xy,
            right_raw_xy=self.right_raw_xy, ball_raw_xy=self.ball_raw_xy,
            left_grf_xy=self.left_grf_xy, right_grf_xy=self.right_grf_xy,
            ball_grf_xy=self.ball_grf_xy, joint_actions=self.joint_actions,
        )
        csv_path = output_dir / f"{stem}_comparison.csv"
        _write_comparison_csv(self, csv_path)
        events_path = output_dir / f"{stem}_events.json"
        events_path.write_text(json.dumps(self.events, indent=2), encoding="utf-8")
        report = self.validation_report()
        report.update({"npz": str(npz_path), "comparison_csv": str(csv_path), "events_json": str(events_path)})
        (output_dir / f"{stem}_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report

    def validation_report(self) -> dict[str, object]:
        source = np.concatenate([self.left_raw_xy, self.right_raw_xy, self.ball_raw_xy[:, None]], axis=1)
        converted = np.concatenate([self.left_grf_xy, self.right_grf_xy, self.ball_grf_xy[:, None]], axis=1)
        roundtrip = np.empty_like(source)
        for index in np.ndindex(converted.shape[:2]):
            roundtrip[index] = grf_to_dataset_a(*converted[index])
        finite = np.isfinite(source).all(axis=-1)
        errors = np.abs(roundtrip - source)[finite]
        outside = finite & ((np.abs(source[..., 0]) > 52.5) | (np.abs(source[..., 1]) > 34.0))
        return {
            "dataset": "dataset_a_bassek_2025", "match_id": self.match_id,
            "frame_count": int(len(self.frames)), "transition_count": int(len(self.joint_actions)),
            "left_team_id": self.left_team_id, "right_team_id": self.right_team_id,
            "left_players": len(self.left_player_ids), "right_players": len(self.right_player_ids),
            "events_in_window": len(self.events),
            "source_coordinate_system": {"units": "meters", "x_range": [-52.5, 52.5], "y_range": [-34.0, 34.0], "origin": "pitch center", "pitch_meters": [105.0, 68.0]},
            "grf_coordinate_system": {"x_range": [-1.0, 1.0], "y_range": [-0.42, 0.42], "origin": "pitch center"},
            "formula": {"x_grf": "x_meters/52.5", "y_grf": "0.42*y_meters/34"},
            "finite_source_points": int(finite.sum()), "missing_source_points": int((~finite).sum()),
            "source_points_clipped_to_grf_pitch": int(outside.sum()),
            "roundtrip_max_abs_error_meters": float(errors.max()) if errors.size else None,
            "roundtrip_mean_abs_error_meters": float(errors.mean()) if errors.size else None,
            "frame_steps": sorted(set(np.diff(self.frames).tolist())),
            "action_counts": {str(a): int(c) for a, c in zip(*np.unique(self.joint_actions, return_counts=True))},
        }


def build_dataset_a_grf_audit(path: Path, *, start_frame: int | None = None, frame_count: int = 100, movement_threshold: float = 0.0005) -> DatasetAGRFAudit:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if start_frame is None or int(record["frame"]) >= start_frame:
                records.append(record)
                if len(records) == frame_count:
                    break
    if len(records) != frame_count:
        raise ValueError(f"Requested {frame_count} frames, found {len(records)} at or after {start_frame}.")
    frames = np.asarray([int(r["frame"]) for r in records], dtype=np.int64)
    if np.any(np.diff(frames) != 1):
        raise ValueError("Dataset A audit window is not frame-contiguous.")
    periods = np.asarray([str(r.get("context", {}).get("game_section", "")) for r in records])
    if len(set(periods.tolist())) != 1:
        raise ValueError("Audit window crosses a match-period boundary.")
    first_players = records[0]["state"]["players"]
    teams = sorted({str(p["team_id"]) for p in first_players})
    if len(teams) != 2:
        raise ValueError(f"Expected two teams, found {teams}.")
    left_team, right_team = _team_sides(records, teams)
    left_ids = _ordered_player_ids(first_players, left_team)
    right_ids = _ordered_player_ids(first_players, right_team)
    if len(left_ids) != 11 or len(right_ids) != 11:
        raise ValueError(f"Expected 11 players per team, found {len(left_ids)} and {len(right_ids)}.")
    left_raw = _positions(records, left_ids)
    right_raw = _positions(records, right_ids)
    ball_raw = np.asarray([_xy(r["state"].get("ball", {})) for r in records], dtype=np.float32)
    left_grf, right_grf, ball_grf = map(_convert_array, (left_raw, right_raw, ball_raw))
    players = np.concatenate([left_grf, right_grf], axis=1)
    actions = _infer_actions(players, movement_threshold)
    events = _events_and_overrides(records, actions, left_ids, right_ids)
    timestamps = [datetime.fromisoformat(r["timestamp"]) for r in records]
    times = np.asarray([(t - timestamps[0]).total_seconds() for t in timestamps], dtype=np.float32)
    return DatasetAGRFAudit(str(records[0]["match_id"]), frames, periods, times, left_team, right_team, left_ids, right_ids, left_raw, right_raw, ball_raw, left_grf, right_grf, ball_grf, np.asarray(actions, dtype=np.int64), events)


def _team_sides(records, teams):
    for record in records:
        for event in record.get("context", {}).get("events", []):
            details = event.get("details") or {}
            if details.get("TeamLeft") in teams and details.get("TeamRight") in teams:
                return str(details["TeamLeft"]), str(details["TeamRight"])
    medians = {team: np.median([float(p["x"]) for p in records[0]["state"]["players"] if p["team_id"] == team]) for team in teams}
    return min(teams, key=medians.get), max(teams, key=medians.get)


def _ordered_player_ids(players, team):
    selected = [p for p in players if str(p["team_id"]) == team]
    selected.sort(key=lambda p: (str(p.get("shirt_number")) != "1", str(p.get("person_id"))))
    return [str(p["person_id"]) for p in selected]


def _positions(records, ids):
    output = np.full((len(records), len(ids), 2), np.nan, dtype=np.float32)
    for frame_index, record in enumerate(records):
        by_id = {str(p["person_id"]): p for p in record["state"]["players"]}
        for player_index, player_id in enumerate(ids):
            if player_id in by_id:
                output[frame_index, player_index] = _xy(by_id[player_id])
    return output


def _xy(item):
    try:
        return [float(item["x"]), float(item["y"])]
    except (KeyError, TypeError, ValueError):
        return [np.nan, np.nan]


def _convert_array(values):
    output = np.empty_like(values, dtype=np.float32)
    for index in np.ndindex(values.shape[:-1]):
        output[index] = dataset_a_to_grf(*values[index])
    return output


def _infer_actions(players, threshold):
    actions, previous = [], np.zeros(22, dtype=bool)
    for current, following in zip(players[:-1], players[1:]):
        movement = movement_actions(current, following, threshold)
        moving = (movement >= 1) & (movement <= 8)
        movement[(~moving) & previous] = 14
        mirrored = movement_actions(-current[11:], -following[11:], threshold)
        right_moving = (mirrored >= 1) & (mirrored <= 8)
        mirrored[(~right_moving) & previous[11:]] = 14
        movement[11:] = mirrored
        previous = moving
        previous[11:] = right_moving
        actions.append(movement)
    return actions


def _events_and_overrides(records, actions, left_ids, right_ids):
    events = []
    for frame_index, record in enumerate(records):
        for event in record.get("context", {}).get("events", []):
            copied = dict(event)
            action = dataset_a_event_to_grf_action(event.get("event_type", ""), event.get("details"))
            copied["grf_action"] = action
            details = event.get("details") or {}
            player_id = str(details.get("Player") or details.get("Winner") or "")
            team_id = str(details.get("Team") or "")
            ids, offset = (left_ids, 0) if team_id else ([], 0)
            if team_id and team_id != "":
                ids, offset = (left_ids, 0) if player_id in left_ids else (right_ids, 11)
            if action is not None and frame_index < len(actions) and player_id in ids:
                index = offset + ids.index(player_id)
                actions[frame_index][index] = action
                copied["joint_action_player_index"] = index
            events.append(copied)
    return events


def _write_comparison_csv(audit, path):
    fields = ["frame", "period", "time_seconds", "side", "team_id", "player_id", "meters_x", "meters_y", "grf_x", "grf_y", "roundtrip_meters_x", "roundtrip_meters_y", "action_to_next_frame"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for fi, frame in enumerate(audit.frames):
            for side, team, ids, raw, grf, offset in (("Left", audit.left_team_id, audit.left_player_ids, audit.left_raw_xy, audit.left_grf_xy, 0), ("Right", audit.right_team_id, audit.right_player_ids, audit.right_raw_xy, audit.right_grf_xy, 11)):
                for pi, player_id in enumerate(ids):
                    back = grf_to_dataset_a(*grf[fi, pi])
                    writer.writerow({"frame": int(frame), "period": audit.periods[fi], "time_seconds": float(audit.times_seconds[fi]), "side": side, "team_id": team, "player_id": player_id, "meters_x": float(raw[fi, pi, 0]), "meters_y": float(raw[fi, pi, 1]), "grf_x": float(grf[fi, pi, 0]), "grf_y": float(grf[fi, pi, 1]), "roundtrip_meters_x": float(back[0]), "roundtrip_meters_y": float(back[1]), "action_to_next_frame": int(audit.joint_actions[fi, offset + pi]) if fi < len(audit.joint_actions) else ""})
