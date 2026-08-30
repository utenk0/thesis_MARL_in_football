"""Traceable 100-frame Metrica-to-GRF conversion for validation and replay."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from thesis_experiments.datasets.action_mapping import metrica_event_to_grf_action
from thesis_experiments.datasets.coordinates import (
    grf_to_metrica,
    metrica_to_grf,
)
from thesis_experiments.datasets.movement import movement_actions


@dataclass(slots=True)
class MetricaGRFAudit:
    frames: np.ndarray
    periods: np.ndarray
    times_seconds: np.ndarray
    home_player_ids: list[str]
    away_player_ids: list[str]
    home_raw_xy: np.ndarray
    away_raw_xy: np.ndarray
    ball_raw_xy: np.ndarray
    home_grf_xy: np.ndarray
    away_grf_xy: np.ndarray
    ball_grf_xy: np.ndarray
    joint_actions: np.ndarray
    events: list[dict[str, object]]

    def save(self, output_dir: Path) -> dict[str, object]:
        output_dir.mkdir(parents=True, exist_ok=True)
        npz_path = output_dir / "metrica_grf_100_frames.npz"
        np.savez_compressed(
            npz_path,
            frames=self.frames,
            periods=self.periods,
            times_seconds=self.times_seconds,
            home_player_ids=np.asarray(self.home_player_ids),
            away_player_ids=np.asarray(self.away_player_ids),
            home_raw_xy=self.home_raw_xy,
            away_raw_xy=self.away_raw_xy,
            ball_raw_xy=self.ball_raw_xy,
            home_grf_xy=self.home_grf_xy,
            away_grf_xy=self.away_grf_xy,
            ball_grf_xy=self.ball_grf_xy,
            joint_actions=self.joint_actions,
        )
        csv_path = output_dir / "metrica_grf_100_frames_comparison.csv"
        _write_comparison_csv(self, csv_path)
        event_path = output_dir / "metrica_grf_100_frames_events.json"
        event_path.write_text(json.dumps(self.events, indent=2), encoding="utf-8")
        report = self.validation_report()
        report.update(
            {
                "npz": str(npz_path),
                "comparison_csv": str(csv_path),
                "events_json": str(event_path),
            }
        )
        report_path = output_dir / "metrica_grf_100_frames_validation.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report

    def validation_report(self) -> dict[str, object]:
        source = np.concatenate([self.home_raw_xy, self.away_raw_xy, self.ball_raw_xy[:, None]], axis=1)
        converted = np.concatenate([self.home_grf_xy, self.away_grf_xy, self.ball_grf_xy[:, None]], axis=1)
        roundtrip = np.empty_like(converted)
        for index in np.ndindex(converted.shape[:2]):
            roundtrip[index] = grf_to_metrica(*converted[index])
        finite = np.isfinite(source).all(axis=-1)
        errors = np.abs(roundtrip - source)[finite]
        outside_nominal_pitch = finite & (
            (source[..., 0] < 0.0)
            | (source[..., 0] > 1.0)
            | (source[..., 1] < 0.0)
            | (source[..., 1] > 1.0)
        )
        return {
            "frame_count": int(len(self.frames)),
            "transition_count": int(len(self.joint_actions)),
            "home_players": len(self.home_player_ids),
            "away_players": len(self.away_player_ids),
            "events_in_window": len(self.events),
            "source_coordinate_system": {
                "x_range": [0.0, 1.0],
                "y_range": [0.0, 1.0],
                "origin": "top-left",
                "center": [0.5, 0.5],
                "pitch_meters": [105.0, 68.0],
            },
            "grf_coordinate_system": {
                "x_range": [-1.0, 1.0],
                "y_range": [-0.42, 0.42],
                "origin": "pitch center",
            },
            "formula": {"x_grf": "2*x_metrica-1", "y_grf": "0.42*(2*y_metrica-1)"},
            "finite_source_points": int(finite.sum()),
            "missing_source_points": int((~finite).sum()),
            "source_points_clipped_to_grf_pitch": int(outside_nominal_pitch.sum()),
            "roundtrip_max_abs_error": float(errors.max()) if errors.size else None,
            "roundtrip_mean_abs_error": float(errors.mean()) if errors.size else None,
            "grf_x_min_max": [float(np.nanmin(converted[..., 0])), float(np.nanmax(converted[..., 0]))],
            "grf_y_min_max": [float(np.nanmin(converted[..., 1])), float(np.nanmax(converted[..., 1]))],
            "action_counts": {
                str(action): int(count)
                for action, count in zip(*np.unique(self.joint_actions, return_counts=True))
            },
        }


def build_metrica_grf_audit(
    game_dir: Path,
    *,
    start_frame: int = 1,
    frame_count: int = 100,
    movement_threshold: float = 0.0005,
) -> MetricaGRFAudit:
    game_name = game_dir.name
    home = _read_tracking(
        game_dir / f"{game_name}_RawTrackingData_Home_Team.csv", start_frame, frame_count
    )
    away = _read_tracking(
        game_dir / f"{game_name}_RawTrackingData_Away_Team.csv", start_frame, frame_count
    )
    if not np.array_equal(home[0], away[0]):
        raise ValueError("Home and Away tracking frame numbers are not aligned.")
    frames, periods, times, home_ids, home_raw, ball_raw = home
    _away_frames, _away_periods, _away_times, away_ids, away_raw, _away_ball = away
    if len(frames) != frame_count:
        raise ValueError(f"Requested {frame_count} frames, found {len(frames)}.")
    home_grf = _convert_array(home_raw)
    away_grf = _convert_array(away_raw)
    ball_grf = _convert_array(ball_raw)
    players = np.concatenate([home_grf, away_grf], axis=1)
    actions = []
    previous_directional = np.zeros(22, dtype=bool)
    for current, following in zip(players[:-1], players[1:]):
        movement = movement_actions(current, following, movement_threshold)
        movement = movement.copy()
        moving = (movement >= 1) & (movement <= 8)
        stopping = (~moving) & previous_directional
        movement[stopping] = 14  # GRF release_direction; idle does not stop sticky motion.
        # Right-team observations/actions are mirrored into their own attacking frame.
        right_mirrored = movement_actions(-current[11:], -following[11:], movement_threshold)
        right_moving = (right_mirrored >= 1) & (right_mirrored <= 8)
        right_stopping = (~right_moving) & previous_directional[11:]
        right_mirrored[right_stopping] = 14
        previous_directional = moving
        previous_directional[11:] = right_moving
        movement[11:] = right_mirrored
        actions.append(movement)
    events = _events_in_window(
        game_dir / f"{game_name}_RawEventsData.csv",
        int(frames[0]),
        int(frames[-1]),
        home_ids,
        away_ids,
        actions,
        frames,
    )
    return MetricaGRFAudit(
        frames=frames,
        periods=periods,
        times_seconds=times,
        home_player_ids=home_ids,
        away_player_ids=away_ids,
        home_raw_xy=home_raw,
        away_raw_xy=away_raw,
        ball_raw_xy=ball_raw,
        home_grf_xy=home_grf,
        away_grf_xy=away_grf,
        ball_grf_xy=ball_grf,
        joint_actions=np.asarray(actions, dtype=np.int64),
        events=events,
    )


def _read_tracking(path: Path, start_frame: int, frame_count: int):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        headers = [next(csv.reader(handle)) for _ in range(3)]
    columns = []
    for column in range(3, len(headers[2]) - 2, 2):
        name = headers[2][column].strip().replace(" ", "")
        if name and name.lower() not in {"nan", "ball"}:
            columns.append((name, column, column + 1))
    columns = columns[:11]
    data = pd.read_csv(path, skiprows=3, header=None)
    data = data[data.iloc[:, 1] >= start_frame].head(frame_count)
    raw = np.stack(
        [data.iloc[:, [x_column, y_column]].to_numpy(dtype=np.float32) for _, x_column, y_column in columns],
        axis=1,
    )
    ball = data.iloc[:, [-2, -1]].to_numpy(dtype=np.float32)
    return (
        data.iloc[:, 1].to_numpy(dtype=np.int64),
        data.iloc[:, 0].to_numpy(dtype=np.int64),
        data.iloc[:, 2].to_numpy(dtype=np.float32),
        [name for name, _x, _y in columns],
        raw,
        ball,
    )


def _convert_array(values: np.ndarray) -> np.ndarray:
    output = np.empty_like(values, dtype=np.float32)
    for index in np.ndindex(values.shape[:-1]):
        output[index] = metrica_to_grf(*values[index])
    return output


def _events_in_window(path, first_frame, last_frame, home_ids, away_ids, actions, frames):
    data = pd.read_csv(path)
    data = data[(data["Start Frame"] >= first_frame) & (data["Start Frame"] <= last_frame)]
    frame_to_transition = {int(frame): index for index, frame in enumerate(frames[:-1])}
    events: list[dict[str, object]] = []
    for _, row in data.iterrows():
        record = {str(key): _json_scalar(value) for key, value in row.to_dict().items()}
        action = metrica_event_to_grf_action(row.get("Type"), row.get("Subtype"))
        record["grf_action"] = int(action)
        frame = int(row["Start Frame"])
        player = str(row.get("From", "")).replace(" ", "")
        team = str(row.get("Team", ""))
        ids = home_ids if team == "Home" else away_ids
        if frame in frame_to_transition and player in ids:
            player_index = ids.index(player) + (0 if team == "Home" else 11)
            actions[frame_to_transition[frame]][player_index] = action
            record["joint_action_player_index"] = player_index
        events.append(record)
    return events


def _write_comparison_csv(audit: MetricaGRFAudit, path: Path) -> None:
    fields = ["frame", "period", "time_seconds", "team", "player_id", "raw_x", "raw_y", "grf_x", "grf_y", "roundtrip_x", "roundtrip_y", "action_to_next_frame"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for frame_index, frame in enumerate(audit.frames):
            for team, ids, raw, grf, offset in (
                ("Home", audit.home_player_ids, audit.home_raw_xy, audit.home_grf_xy, 0),
                ("Away", audit.away_player_ids, audit.away_raw_xy, audit.away_grf_xy, 11),
            ):
                for player_index, player_id in enumerate(ids):
                    roundtrip = grf_to_metrica(*grf[frame_index, player_index])
                    writer.writerow({
                        "frame": int(frame), "period": int(audit.periods[frame_index]),
                        "time_seconds": float(audit.times_seconds[frame_index]), "team": team,
                        "player_id": player_id, "raw_x": float(raw[frame_index, player_index, 0]),
                        "raw_y": float(raw[frame_index, player_index, 1]),
                        "grf_x": float(grf[frame_index, player_index, 0]),
                        "grf_y": float(grf[frame_index, player_index, 1]),
                        "roundtrip_x": float(roundtrip[0]), "roundtrip_y": float(roundtrip[1]),
                        "action_to_next_frame": (
                            int(audit.joint_actions[frame_index, offset + player_index])
                            if frame_index < len(audit.joint_actions) else ""
                        ),
                    })


def _json_scalar(value):
    if pd.isna(value):
        return None
    return value.item() if isinstance(value, np.generic) else value
