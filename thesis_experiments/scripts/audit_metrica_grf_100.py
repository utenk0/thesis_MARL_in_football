"""Convert, validate, visualize, and optionally simulate 100 Metrica frames."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from thesis_experiments.data.metrica_grf_audit import build_metrica_grf_audit
from thesis_experiments.env import ensure_local_football_on_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, default=1)
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--movement-threshold", type=float, default=0.0005)
    parser.add_argument("--simulate", action="store_true")
    parser.add_argument("--record", action="store_true")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Open the GRF game window while executing the joint actions.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=25.0,
        help="Maximum live playback speed; Metrica tracking is normally 25 FPS.",
    )
    args = parser.parse_args()

    audit = build_metrica_grf_audit(
        args.game_dir,
        start_frame=args.start_frame,
        frame_count=args.frames,
        movement_threshold=args.movement_threshold,
    )
    report = audit.save(args.output_dir)
    if args.simulate:
        report["simulation"] = _simulate(
            audit,
            args.output_dir,
            record=args.record,
            live=args.live,
            fps=args.fps,
        )
    source_video = args.output_dir / "metrica_source_and_grf_coordinates.mp4"
    _write_coordinate_video(audit, source_video)
    report["coordinate_video"] = str(source_video)
    report_path = args.output_dir / "metrica_grf_100_frames_validation.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def _write_coordinate_video(audit, path: Path) -> None:
    # Import OpenCV only after GRF simulation has closed. On macOS, OpenCV and
    # GRF may bundle different SDL2 builds whose GUI classes conflict.
    import cv2

    width, height = 1600, 540
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create {path}.")
    try:
        for index, frame in enumerate(audit.frames):
            canvas = np.full((height, width, 3), 35, dtype=np.uint8)
            _draw_pitch(cv2, canvas, 20, 60, 760, 480, "Metrica source [0,1]")
            _draw_pitch(cv2, canvas, 820, 60, 1560, 480, "Converted GRF coordinates")
            _draw_team_raw(cv2, canvas, audit.home_raw_xy[index], 20, 60, 740, 420, (0, 220, 255))
            _draw_team_raw(cv2, canvas, audit.away_raw_xy[index], 20, 60, 740, 420, (255, 100, 40))
            _draw_ball_raw(cv2, canvas, audit.ball_raw_xy[index], 20, 60, 740, 420)
            _draw_team_grf(cv2, canvas, audit.home_grf_xy[index], 820, 60, 740, 420, (0, 220, 255))
            _draw_team_grf(cv2, canvas, audit.away_grf_xy[index], 820, 60, 740, 420, (255, 100, 40))
            _draw_ball_grf(cv2, canvas, audit.ball_grf_xy[index], 820, 60, 740, 420)
            cv2.putText(canvas, f"frame={int(frame)} time={audit.times_seconds[index]:.2f}s", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            writer.write(canvas)
    finally:
        writer.release()


def _draw_pitch(cv2, canvas, x0, y0, x1, y1, label):
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (55, 130, 55), -1)
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (230, 230, 230), 2)
    middle = (x0 + x1) // 2
    cv2.line(canvas, (middle, y0), (middle, y1), (230, 230, 230), 1)
    cv2.circle(canvas, (middle, (y0 + y1) // 2), 45, (230, 230, 230), 1)
    cv2.putText(canvas, label, (x0, y0 - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)


def _draw_team_raw(cv2, canvas, positions, x0, y0, width, height, color):
    for position in positions:
        if np.isfinite(position).all():
            cv2.circle(canvas, (x0 + int(position[0] * width), y0 + int(position[1] * height)), 6, color, -1)


def _draw_ball_raw(cv2, canvas, position, x0, y0, width, height):
    if np.isfinite(position).all():
        cv2.circle(canvas, (x0 + int(position[0] * width), y0 + int(position[1] * height)), 4, (255, 255, 255), -1)


def _draw_team_grf(cv2, canvas, positions, x0, y0, width, height, color):
    for position in positions:
        if np.isfinite(position).all():
            x = x0 + int((position[0] + 1.0) * 0.5 * width)
            y = y0 + int((position[1] / 0.42 + 1.0) * 0.5 * height)
            cv2.circle(canvas, (x, y), 6, color, -1)


def _draw_ball_grf(cv2, canvas, position, x0, y0, width, height):
    if np.isfinite(position).all():
        x = x0 + int((position[0] + 1.0) * 0.5 * width)
        y = y0 + int((position[1] / 0.42 + 1.0) * 0.5 * height)
        cv2.circle(canvas, (x, y), 4, (255, 255, 255), -1)


def _simulate(
    audit,
    output_dir: Path,
    *,
    record: bool,
    live: bool,
    fps: float,
    scenario_name: str = "metrica_100_replay",
    state_env_var: str = "GRF_METRICA_REPLAY_INITIAL_STATE",
) -> dict[str, object]:
    if fps <= 0:
        raise ValueError("--fps must be greater than zero.")
    ensure_local_football_on_path()
    import gfootball.env as football_env

    initial = {
        "home": np.nan_to_num(audit.home_grf_xy[0], nan=0.0).tolist(),
        "away": np.nan_to_num(audit.away_grf_xy[0], nan=0.0).tolist(),
        "ball": np.nan_to_num(audit.ball_grf_xy[0], nan=0.0).tolist(),
        "game_duration": len(audit.joint_actions),
    }
    os.environ[state_env_var] = json.dumps(initial)
    logdir = output_dir / "grf_simulation"
    logdir.mkdir(parents=True, exist_ok=True)
    before = set(logdir.rglob("*.avi"))
    env = football_env.create_environment(
        env_name=scenario_name,
        representation="raw",
        rewards="scoring",
        render=live or record,
        write_goal_dumps=False,
        write_full_episode_dumps=record,
        write_video=record,
        dump_frequency=1,
        logdir=str(logdir),
        number_of_left_players_agent_controls=11,
        number_of_right_players_agent_controls=11,
    )
    errors = []
    steps = 0
    try:
        observations = env.reset()
        for index, actions in enumerate(audit.joint_actions):
            started_at = time.monotonic()
            observations, _reward, done, _info = env.step(actions)
            reference = observations[0]
            simulated = np.concatenate([reference["left_team"], reference["right_team"]])
            target = np.concatenate([audit.home_grf_xy[index + 1], audit.away_grf_xy[index + 1]])
            finite = np.isfinite(target).all(axis=1)
            errors.append(float(np.linalg.norm(simulated[finite] - target[finite], axis=1).mean()))
            steps = index + 1
            if live:
                remaining = (1.0 / fps) - (time.monotonic() - started_at)
                if remaining > 0:
                    time.sleep(remaining)
            if done:
                break
    finally:
        env.close()
        os.environ.pop(state_env_var, None)
    videos = sorted(str(path) for path in set(logdir.rglob("*.avi")) - before)
    return {
        "steps": steps,
        "controlled_players": 22,
        "live_rendering": live,
        "playback_fps": fps if live else None,
        "mean_position_error_grf_units": float(np.mean(errors)) if errors else None,
        "final_position_error_grf_units": errors[-1] if errors else None,
        "videos": videos,
        "interpretation": "Action replay is approximate because GRF applies its own physics; coordinate round-trip validation is exact.",
    }


if __name__ == "__main__":
    main()
