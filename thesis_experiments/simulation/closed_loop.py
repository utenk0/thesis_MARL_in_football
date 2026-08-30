"""Execute converted ground-truth trajectories using GRF feedback control."""

from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path

import numpy as np

from thesis_experiments.env import ensure_local_football_on_path
from thesis_experiments.simulation.trajectory_controller import TrajectoryController


def run_closed_loop(audit, output_dir: Path, *, live: bool = False, record: bool = False, fps: float = 25.0, stop_tolerance: float = 0.008, sprint_threshold: float = 0.08, sprint_release_threshold: float = 0.04) -> dict[str, object]:
    if fps <= 0:
        raise ValueError("fps must be greater than zero.")
    ensure_local_football_on_path()
    import gfootball.env as football_env

    targets = np.concatenate([audit.home_grf_xy, audit.away_grf_xy], axis=1).astype(np.float32)
    initial = {"home": targets[0, :11].tolist(), "away": targets[0, 11:].tolist(), "ball": np.nan_to_num(audit.ball_grf_xy[0], nan=0.0).tolist(), "game_duration": len(targets) - 1}
    is_dataset_a = hasattr(audit, "left_team_id")
    env_var = "GRF_DATASET_A_REPLAY_INITIAL_STATE" if is_dataset_a else "GRF_METRICA_REPLAY_INITIAL_STATE"
    scenario = "dataset_a_100_replay" if is_dataset_a else "metrica_100_replay"
    os.environ[env_var] = json.dumps(initial)
    output_dir.mkdir(parents=True, exist_ok=True)
    logdir = output_dir / "grf_recording"; logdir.mkdir(exist_ok=True)
    before_videos = set(logdir.rglob("*.avi"))
    env = football_env.create_environment(
        env_name=scenario, representation="raw", rewards="scoring",
        render=live or record, write_goal_dumps=False,
        write_full_episode_dumps=record, write_video=record,
        dump_frequency=1, logdir=str(logdir),
        number_of_left_players_agent_controls=11,
        number_of_right_players_agent_controls=11,
    )
    controller = TrajectoryController(stop_tolerance=stop_tolerance, sprint_threshold=sprint_threshold, sprint_release_threshold=sprint_release_threshold)
    simulated_history, action_history, error_history = [], [], []
    try:
        observations = env.reset()
        simulated = _positions(observations)
        simulated_history.append(simulated)
        error_history.append(np.linalg.norm(simulated - targets[0], axis=1))
        for target_index in range(1, len(targets)):
            started = time.monotonic()
            actions = controller.actions(simulated, targets[target_index])
            observations, _reward, done, _info = env.step(actions)
            simulated = _positions(observations)
            simulated_history.append(simulated)
            action_history.append(actions)
            error_history.append(np.linalg.norm(simulated - targets[target_index], axis=1))
            if live:
                remaining = 1.0 / fps - (time.monotonic() - started)
                if remaining > 0: time.sleep(remaining)
            if done: break
    finally:
        env.close()
        os.environ.pop(env_var, None)
    simulated_array = np.asarray(simulated_history, dtype=np.float32)
    target_array = targets[:len(simulated_array)]
    errors = np.asarray(error_history, dtype=np.float32)
    actions_array = np.asarray(action_history, dtype=np.int64)
    npz_path = output_dir / "closed_loop_trajectories.npz"
    np.savez_compressed(npz_path, frames=np.asarray(audit.frames[:len(simulated_array)]), target_positions=target_array, simulated_positions=simulated_array, position_errors=errors, joint_actions=actions_array)
    csv_path = output_dir / "closed_loop_errors.csv"
    _write_errors(csv_path, audit, errors)
    videos = sorted(str(path) for path in set(logdir.rglob("*.avi")) - before_videos)
    report = _report(errors, actions_array)
    report.update({"source_frames": int(len(targets)), "executed_frames": int(len(simulated_array)), "controlled_players": 22, "live": live, "record": record, "fps": fps if live else None, "trajectory_npz": str(npz_path), "error_csv": str(csv_path), "videos": videos, "interpretation": "Feedback controller uses future target positions and is a validation baseline, not an autonomous policy."})
    report_path = output_dir / "closed_loop_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _positions(observations) -> np.ndarray:
    reference = observations[0]
    return np.concatenate([reference["left_team"], reference["right_team"]]).astype(np.float32)


def _report(errors, actions):
    flattened = errors.reshape(-1)
    return {
        "mean_error_grf": float(flattened.mean()), "median_error_grf": float(np.median(flattened)),
        "p95_error_grf": float(np.percentile(flattened, 95)), "max_error_grf": float(flattened.max()),
        "final_mean_error_grf": float(errors[-1].mean()),
        "left_mean_error_grf": float(errors[:, :11].mean()), "right_mean_error_grf": float(errors[:, 11:].mean()),
        "per_player_mean_error_grf": errors.mean(axis=0).tolist(),
        "action_counts": {str(a): int(c) for a, c in zip(*np.unique(actions, return_counts=True))},
    }


def _write_errors(path, audit, errors):
    left_ids = list(getattr(audit, "home_player_ids", getattr(audit, "left_player_ids", [])))
    right_ids = list(getattr(audit, "away_player_ids", getattr(audit, "right_player_ids", [])))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["frame", "player_index", "side", "player_id", "position_error_grf"]); writer.writeheader()
        for frame_index, values in enumerate(errors):
            for player_index, value in enumerate(values):
                side = "left" if player_index < 11 else "right"
                ids = left_ids if player_index < 11 else right_ids
                local_index = player_index if player_index < 11 else player_index - 11
                writer.writerow({"frame": int(audit.frames[frame_index]), "player_index": player_index, "side": side, "player_id": ids[local_index], "position_error_grf": float(value)})
