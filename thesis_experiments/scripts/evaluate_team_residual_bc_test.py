"""Verify the pretest freeze, then evaluate residual BC on the held-out test."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from thesis_experiments.training.team_residual_displacement_bc import (
    evaluate_team_residual_displacement_bc,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_freeze(path: Path) -> dict[str, object]:
    freeze = json.loads(path.read_text())
    if freeze["status"] != "residual_bc_pretest_frozen":
        raise ValueError("Residual experiment is not in pretest-frozen state.")
    mismatches = [
        row["path"]
        for row in freeze["files"]
        if not Path(row["path"]).is_file()
        or _sha256(Path(row["path"])) != row["sha256"]
    ]
    if mismatches:
        raise ValueError(f"Pretest freeze hash mismatch: {mismatches}")
    return freeze


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
    rows = []
    for seed in pretest["seeds"]:
        seed_dir = args.experiment_dir / f"seed_{seed}"
        evaluation = evaluate_team_residual_displacement_bc(
            seed_dir / "single_team_residual_bc.pt",
            test_paths,
            team_id=team_id,
            device=args.device,
        )
        (seed_dir / "test_evaluation.json").write_text(json.dumps(evaluation, indent=2))
        rows.append(evaluation["overall"])
    policy = [row["policy"]["mean_displacement_error_meters"] for row in rows]
    persistence = [
        row["previous_velocity_baseline"]["mean_displacement_error_meters"]
        for row in rows
    ]
    result = {
        "method": "residual_displacement_bc",
        "phase": "frozen_test_report",
        "pretest_freeze_verified": True,
        "test_used_for_selection": False,
        "multi_horizon_gate_passed_before_test": pretest["multi_horizon_gate"]["passed"],
        "seeds": pretest["seeds"],
        "test": {
            "policy_mean_displacement_error_meters": _stats(policy),
            "previous_velocity_mean_displacement_error_meters": _stats(persistence),
            "per_seed_policy": policy,
            "per_seed_previous_velocity": persistence,
        },
    }
    (args.experiment_dir / "test_summary.json").write_text(json.dumps(result, indent=2))
    policy_stats = result["test"]["policy_mean_displacement_error_meters"]
    baseline_stats = result["test"]["previous_velocity_mean_displacement_error_meters"]
    (args.experiment_dir / "TEST_REPORT.md").write_text(
        "# Frozen residual BC test report\n\n"
        "The pretest file hashes were verified before loading the held-out test. "
        "Test data were not used for checkpoint or architecture selection.\n\n"
        "| Predictor | Test mean displacement error (m) |\n"
        "|---|---:|\n"
        f"| Residual BC | {policy_stats['mean']:.6f} ± {policy_stats['std']:.6f} |\n"
        f"| Previous velocity | {baseline_stats['mean']:.6f} ± {baseline_stats['std']:.6f} |\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
