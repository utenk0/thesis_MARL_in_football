"""Verify and execute the frozen trajectory BC on a relocated machine."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import torch

from thesis_experiments.evaluation.trajectory_contract import (
    load_contract,
    validate_test_summary,
)
from thesis_experiments.training.team_multi_horizon_bc import (
    evaluate_team_multi_horizon_bc,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _project_path(project_dir: Path, value: str) -> Path:
    path = (project_dir / value).resolve()
    try:
        path.relative_to(project_dir)
    except ValueError as error:
        raise ValueError(f"Frozen path escapes the project directory: {value}") from error
    return path


def _verify_files(project_dir: Path, records: list[dict[str, object]]) -> list[dict[str, object]]:
    verified = []
    failures = []
    for record in records:
        path = _project_path(project_dir, str(record["path"]))
        if not path.is_file():
            failures.append({"path": str(record["path"]), "reason": "missing"})
            continue
        actual_size = path.stat().st_size
        actual_hash = _sha256(path)
        expected_size = int(record["bytes"])
        expected_hash = str(record["sha256"])
        if actual_size != expected_size or actual_hash != expected_hash:
            failures.append(
                {
                    "path": str(record["path"]),
                    "reason": "hash_or_size_mismatch",
                    "expected_bytes": expected_size,
                    "actual_bytes": actual_size,
                    "expected_sha256": expected_hash,
                    "actual_sha256": actual_hash,
                }
            )
            continue
        verified.append(
            {
                "role": record.get("role"),
                "path": str(record["path"]),
                "bytes": actual_size,
                "sha256": actual_hash,
            }
        )
    if failures:
        raise ValueError(f"Frozen BC file verification failed: {failures}")
    return verified


def _compare_metric(name: str, actual: float, expected: float, tolerance: float) -> dict[str, float | str]:
    difference = abs(actual - expected)
    if difference > tolerance:
        raise ValueError(
            f"Portability metric {name} differs by {difference:.12g} m "
            f"(expected {expected:.12g}, actual {actual:.12g}, tolerance {tolerance:.12g})."
        )
    return {
        "metric": name,
        "expected": expected,
        "actual": actual,
        "absolute_difference": difference,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", required=True, type=Path)
    parser.add_argument("--project-dir", type=Path, default=Path.cwd())
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=1)
    args = parser.parse_args()

    project_dir = args.project_dir.resolve()
    freeze_path = args.freeze.resolve()
    freeze = json.loads(freeze_path.read_text())
    if freeze.get("status") != "frozen":
        raise ValueError("Trajectory BC portability manifest is not frozen.")
    if args.torch_threads < 1:
        raise ValueError("--torch-threads must be positive.")
    if str(freeze["portability"]["device"]) != args.device:
        raise ValueError(
            f"The frozen portability check requires device={freeze['portability']['device']!r}."
        )

    torch.set_num_threads(args.torch_threads)
    verified_files = _verify_files(project_dir, list(freeze["files"]))
    contract_path = _project_path(project_dir, str(freeze["contract_file"]))
    test_summary_path = _project_path(project_dir, str(freeze["test_summary"]))
    contract = load_contract(contract_path)
    test_summary = json.loads(test_summary_path.read_text())
    validate_test_summary(contract, test_summary)
    if contract["contract"] != freeze["contract"]:
        raise ValueError("The trajectory contract name differs from the portable freeze.")
    if contract["team_id"] != freeze["team_id"]:
        raise ValueError("The trajectory contract team differs from the portable freeze.")
    if contract["forecast"]["horizons_steps"] != freeze["horizons"]:
        raise ValueError("The trajectory horizons differ from the portable freeze.")
    if contract["information"]["causal_history_steps"] != freeze["history_steps"]:
        raise ValueError("The causal history length differs from the portable freeze.")

    tolerance = float(freeze["portability"]["absolute_tolerance_meters"])
    data_path = _project_path(project_dir, str(freeze["portability"]["reference_data"]))
    model_reports = []
    for model in freeze["models"]:
        checkpoint = _project_path(project_dir, str(model["checkpoint"]))
        evaluation = evaluate_team_multi_horizon_bc(
            checkpoint,
            [data_path],
            team_id=str(freeze["team_id"]),
            device=args.device,
        )
        if evaluation["selected_epoch"] != int(model["selected_epoch"]):
            raise ValueError(f"Seed {model['seed']} selected epoch differs from the freeze.")
        if evaluation["horizons"] != freeze["horizons"]:
            raise ValueError(f"Seed {model['seed']} horizons differ from the freeze.")
        if evaluation["future_observations_used"]:
            raise ValueError(f"Seed {model['seed']} reports future-observation use.")
        policy = evaluation["overall"]["policy"]
        expected = model["expected"]
        comparisons = [
            _compare_metric(
                "ade_meters",
                float(policy["autonomous_ade_meters"]),
                float(expected["ade_meters"]),
                tolerance,
            ),
            _compare_metric(
                "fde_meters",
                float(policy["autonomous_fde_meters"]),
                float(expected["fde_meters"]),
                tolerance,
            ),
        ]
        actual_horizons = [
            float(row["mean_position_error_meters"])
            for row in policy["per_horizon"]
        ]
        for horizon, actual, expected_value in zip(
            freeze["horizons"], actual_horizons, expected["per_horizon_meters"], strict=True
        ):
            comparisons.append(
                _compare_metric(
                    f"horizon_{horizon}_mean_position_error_meters",
                    actual,
                    float(expected_value),
                    tolerance,
                )
            )
        model_reports.append(
            {
                "seed": int(model["seed"]),
                "checkpoint": str(model["checkpoint"]),
                "selected_epoch": int(evaluation["selected_epoch"]),
                "forecast_origins": int(policy["forecast_origins"]),
                "comparisons": comparisons,
            }
        )

    report = {
        "status": "passed",
        "freeze": str(freeze_path),
        "freeze_version": freeze["version"],
        "contract": freeze["contract"],
        "team_id": freeze["team_id"],
        "device": args.device,
        "tolerance_meters": tolerance,
        "verified_file_count": len(verified_files),
        "verified_files": verified_files,
        "reference_data": str(freeze["portability"]["reference_data"]),
        "models": model_reports,
        "environment": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "contract": report["contract"],
                "verified_file_count": report["verified_file_count"],
                "verified_seeds": [row["seed"] for row in model_reports],
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
