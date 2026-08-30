"""Convert, validate, visualize, and simulate Dataset A tracking in GRF."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from thesis_experiments.data.dataset_a_grf_audit import build_dataset_a_grf_audit
from thesis_experiments.scripts.audit_metrica_grf_100 import _simulate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-frame", type=int)
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--movement-threshold", type=float, default=0.0005)
    parser.add_argument("--simulate", action="store_true")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--fps", type=float, default=25.0)
    args = parser.parse_args()
    audit = build_dataset_a_grf_audit(args.input, start_frame=args.start_frame, frame_count=args.frames, movement_threshold=args.movement_threshold)
    report = audit.save(args.output_dir)
    if args.simulate:
        report["simulation"] = _simulate(audit, args.output_dir, record=args.record, live=args.live, fps=args.fps, scenario_name="dataset_a_100_replay", state_env_var="GRF_DATASET_A_REPLAY_INITIAL_STATE")
    video_path = args.output_dir / "dataset_a_source_and_grf_coordinates.mp4"
    _write_video(audit, video_path)
    report["coordinate_video"] = str(video_path)
    report_path = args.output_dir / "dataset_a_grf_100_frames_validation.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def _write_video(audit, path):
    import cv2
    width, height = 1600, 540
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create {path}.")
    try:
        for i, frame in enumerate(audit.frames):
            canvas = np.full((height, width, 3), 35, dtype=np.uint8)
            for x0, title in ((20, "Dataset A source (metres)"), (820, "Converted GRF coordinates")):
                cv2.rectangle(canvas, (x0, 60), (x0 + 740, 480), (55, 130, 55), -1)
                cv2.rectangle(canvas, (x0, 60), (x0 + 740, 480), (230, 230, 230), 2)
                cv2.line(canvas, (x0 + 370, 60), (x0 + 370, 480), (230, 230, 230), 1)
                cv2.circle(canvas, (x0 + 370, 270), 45, (230, 230, 230), 1)
                cv2.putText(canvas, title, (x0, 48), cv2.FONT_HERSHEY_SIMPLEX, .65, (255,255,255), 2)
            _draw(cv2, canvas, audit.left_raw_xy[i], 20, (0,220,255), meters=True)
            _draw(cv2, canvas, audit.right_raw_xy[i], 20, (255,100,40), meters=True)
            _draw(cv2, canvas, audit.ball_raw_xy[i:i+1], 20, (255,255,255), meters=True, radius=4)
            _draw(cv2, canvas, audit.left_grf_xy[i], 820, (0,220,255))
            _draw(cv2, canvas, audit.right_grf_xy[i], 820, (255,100,40))
            _draw(cv2, canvas, audit.ball_grf_xy[i:i+1], 820, (255,255,255), radius=4)
            cv2.putText(canvas, f"frame={int(frame)} t={audit.times_seconds[i]:.2f}s", (20, 25), cv2.FONT_HERSHEY_SIMPLEX, .7, (255,255,255), 2)
            writer.write(canvas)
    finally:
        writer.release()


def _draw(cv2, canvas, positions, x0, color, *, meters=False, radius=6):
    for x, y in positions:
        if np.isfinite([x, y]).all():
            gx, gy = (x / 52.5, .42 * y / 34.0) if meters else (x, y)
            px = x0 + int((gx + 1) * .5 * 740)
            py = 60 + int((gy / .42 + 1) * .5 * 420)
            cv2.circle(canvas, (px, py), radius, color, -1)


if __name__ == "__main__":
    main()
