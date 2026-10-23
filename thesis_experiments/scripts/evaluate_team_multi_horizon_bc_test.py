"""Verify the multi-horizon pretest freeze, then evaluate held-out forecasts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from thesis_experiments.training.team_multi_horizon_bc import (
    evaluate_team_multi_horizon_bc,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_freeze(path: Path) -> None:
    freeze = json.loads(path.read_text())
    if freeze["status"] != "multi_horizon_bc_pretest_frozen":
        raise ValueError("Multi-horizon experiment is not pretest-frozen.")
    mismatches = [
        row["path"]
        for row in freeze["files"]
        if not Path(row["path"]).is_file()
        or _sha256(Path(row["path"])) != row["sha256"]
    ]
    if mismatches:
        raise ValueError(f"Pretest freeze hash mismatch: {mismatches}")


def _stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "min": float(array.min()),
        "max": float(array.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.torch_threads)
    _verify_freeze(args.experiment_dir / "pretest_freeze.json")
    pretest = json.loads((args.experiment_dir / "pretest_summary.json").read_text())
    manifest = json.loads(args.manifest.read_text())
    team_id = str(manifest["team_id"])
    test_paths = [Path(row["path"]) for row in manifest["splits"]["test"]]
    evaluations = []
    for seed in pretest["seeds"]:
        seed_dir = args.experiment_dir / f"seed_{seed}"
        evaluation = evaluate_team_multi_horizon_bc(
            seed_dir / "single_team_multi_horizon_bc.pt",
            test_paths,
            team_id=team_id,
            device=args.device,
        )
        (seed_dir / "test_evaluation.json").write_text(json.dumps(evaluation, indent=2))
        evaluations.append(evaluation)

    predictors = ("policy", "stationary", "constant_velocity", "constant_acceleration")
    aggregate = {}
    for predictor in predictors:
        aggregate[predictor] = {
            "ade_meters": _stats(
                [row["overall"][predictor]["autonomous_ade_meters"] for row in evaluations]
            ),
            "fde_meters": _stats(
                [row["overall"][predictor]["autonomous_fde_meters"] for row in evaluations]
            ),
            "per_horizon": [],
        }
        for index, horizon in enumerate(pretest["horizons"]):
            horizon_rows = [row["overall"][predictor]["per_horizon"][index] for row in evaluations]
            aggregate[predictor]["per_horizon"].append(
                {
                    "horizon_steps": horizon,
                    "mean_position_error_meters": _stats(
                        [row["mean_position_error_meters"] for row in horizon_rows]
                    ),
                    "centroid_error_meters": _stats(
                        [row["centroid_error_meters"] for row in horizon_rows]
                    ),
                    "team_width_mae_meters": _stats(
                        [row["team_width_mae_meters"] for row in horizon_rows]
                    ),
                    "team_depth_mae_meters": _stats(
                        [row["team_depth_mae_meters"] for row in horizon_rows]
                    ),
                }
            )
    result = {
        "method": "multi_horizon_trajectory_bc",
        "phase": "frozen_test_report",
        "team_id": team_id,
        "test_matches": sorted(
            {str(row["match_id"]) for row in manifest["splits"]["test"]}
        ),
        "pretest_freeze_verified": True,
        "future_observations_used": False,
        "test_used_for_selection": False,
        "trajectory_gate_passed_before_test": pretest["trajectory_gate"]["passed"],
        "seeds": pretest["seeds"],
        "horizons": pretest["horizons"],
        "test": aggregate,
    }
    (args.experiment_dir / "test_summary.json").write_text(json.dumps(result, indent=2))
    lines = [
        "# Frozen autonomous multi-horizon BC test report",
        "",
        "The pretest hashes were verified before loading the held-out match. Every horizon uses only information at or before its forecast origin.",
        "",
        "| Predictor | ADE (m) | 100-step FDE (m) |",
        "|---|---:|---:|",
    ]
    for predictor in predictors:
        row = aggregate[predictor]
        lines.append(
            f"| {predictor} | {row['ade_meters']['mean']:.4f} ± {row['ade_meters']['std']:.4f} | "
            f"{row['fde_meters']['mean']:.4f} ± {row['fde_meters']['std']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Policy error by direct horizon",
            "",
            "| Horizon | Position error (m) | Centroid error (m) | Width MAE (m) | Depth MAE (m) |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for row in aggregate["policy"]["per_horizon"]:
        lines.append(
            f"| {row['horizon_steps']} | {row['mean_position_error_meters']['mean']:.4f} ± {row['mean_position_error_meters']['std']:.4f} | "
            f"{row['centroid_error_meters']['mean']:.4f} ± {row['centroid_error_meters']['std']:.4f} | "
            f"{row['team_width_mae_meters']['mean']:.4f} ± {row['team_width_mae_meters']['std']:.4f} | "
            f"{row['team_depth_mae_meters']['mean']:.4f} ± {row['team_depth_mae_meters']['std']:.4f} |"
        )
    (args.experiment_dir / "TEST_REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
