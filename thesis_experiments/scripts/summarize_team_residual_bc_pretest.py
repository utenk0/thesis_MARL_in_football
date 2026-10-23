"""Aggregate and freeze residual BC before any test data are loaded."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def _stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize(paths: list[Path]) -> dict[str, object]:
    runs = [json.loads(path.read_text()) for path in paths]
    if len(runs) != 3:
        raise ValueError("The locked residual protocol requires exactly three seeds.")
    teams = {run["team_id"] for run in runs}
    splits = {json.dumps(run["split_matches"], sort_keys=True) for run in runs}
    if len(teams) != 1 or len(splits) != 1:
        raise ValueError("Residual runs do not share team and split contracts.")
    if any(run["phase"] != "pretest" for run in runs):
        raise ValueError("Only pretest runs may enter the pretest freeze.")
    if any(run["selection_contract"]["test_loaded"] for run in runs):
        raise ValueError("A purported pretest run loaded test data.")

    validation = [run["evaluations"]["validation"]["overall"] for run in runs]
    policy_values = [
        float(row["policy"]["mean_displacement_error_meters"])
        for row in validation
    ]
    baseline_values = [
        float(row["previous_velocity_baseline"]["mean_displacement_error_meters"])
        for row in validation
    ]
    improved = [
        policy < baseline - 1e-8
        for policy, baseline in zip(policy_values, baseline_values)
    ]
    noninferior = [
        policy <= baseline + 1e-8
        for policy, baseline in zip(policy_values, baseline_values)
    ]
    policy_stats = _stats(policy_values)
    baseline_stats = _stats(baseline_values)
    gate = bool(
        policy_stats["mean"] < baseline_stats["mean"] - 1e-8
        and sum(improved) >= 2
        and all(noninferior)
    )
    return {
        "method": "residual_displacement_bc",
        "phase": "pretest_frozen",
        "team_id": next(iter(teams)),
        "seeds": [int(run["training"]["seed"]) for run in runs],
        "selected_epochs": [int(run["training"]["selected_epoch"]) for run in runs],
        "architecture": {
            "prediction": "previous_velocity_plus_learned_correction",
            "zero_initialized_output": True,
            "history_steps": int(runs[0]["training"]["history_steps"]),
            "observation_variant": runs[0]["training"]["observation_variant"],
        },
        "selection": {
            "metric": "validation_mean_displacement_error_meters",
            "epoch_zero_included": True,
            "test_loaded": False,
            "test_used_for_selection": False,
        },
        "validation": {
            "policy_mean_displacement_error_meters": policy_stats,
            "previous_velocity_mean_displacement_error_meters": baseline_stats,
            "per_seed_policy": policy_values,
            "per_seed_previous_velocity": baseline_values,
            "strict_improvements": improved,
            "noninferior": noninferior,
        },
        "multi_horizon_gate": {
            "rule": (
                "Mean residual validation error must be strictly lower than persistence, "
                "at least two of three seeds must strictly improve, and no seed may be worse."
            ),
            "passed": gate,
        },
        "result_paths": [str(path) for path in paths],
    }


def _write_report(result: dict[str, object], path: Path) -> None:
    policy = result["validation"]["policy_mean_displacement_error_meters"]
    baseline = result["validation"]["previous_velocity_mean_displacement_error_meters"]
    lines = [
        "# Residual displacement BC — frozen pretest decision",
        "",
        f"- Team: `{result['team_id']}`",
        f"- Seeds: {', '.join(map(str, result['seeds']))}",
        f"- Selected epochs (epoch 0 eligible): {', '.join(map(str, result['selected_epochs']))}",
        "- Test data loaded: no",
        "- Test used for selection: no",
        "",
        "| Validation metric | Mean ± std (m) |",
        "|---|---:|",
        f"| Residual BC | {policy['mean']:.6f} ± {policy['std']:.6f} |",
        f"| Previous velocity | {baseline['mean']:.6f} ± {baseline['std']:.6f} |",
        "",
        f"Multi-horizon gate: **{'PASS' if result['multi_horizon_gate']['passed'] else 'FAIL'}**.",
        "",
        result["multi_horizon_gate"]["rule"],
    ]
    path.write_text("\n".join(lines) + "\n")


def _freeze(
    experiment_dir: Path,
    manifest: Path,
    summary_path: Path,
    report_path: Path,
) -> dict[str, object]:
    root = Path(__file__).resolve().parents[2]
    files = [
        manifest.resolve(),
        summary_path.resolve(),
        report_path.resolve(),
        (root / "thesis_experiments/policies/networks.py").resolve(),
        (root / "thesis_experiments/training/team_residual_displacement_bc.py").resolve(),
        (root / "thesis_experiments/scripts/train_team_residual_bc.py").resolve(),
    ]
    for seed_dir in sorted(experiment_dir.glob("seed_*")):
        files.extend(
            (seed_dir / name).resolve()
            for name in (
                "single_team_residual_bc.pt",
                "config.json",
                "pretest_result.json",
                "train_evaluation.json",
                "validation_evaluation.json",
            )
        )
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Cannot freeze residual BC: {missing}")
    result = {
        "status": "residual_bc_pretest_frozen",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "test_evaluation_authorized_after_hash_verification": True,
        "files": [
            {"path": str(path), "sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in files
        ],
    }
    (experiment_dir / "pretest_freeze.json").write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, nargs="+", type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    result = summarize(args.results)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "pretest_summary.json"
    report_path = args.output_dir / "PRETEST_REPORT.md"
    summary_path.write_text(json.dumps(result, indent=2))
    _write_report(result, report_path)
    freeze = _freeze(args.output_dir, args.manifest, summary_path, report_path)
    print(
        json.dumps(
            {
                "summary": str(summary_path),
                "report": str(report_path),
                "freeze": str(args.output_dir / "pretest_freeze.json"),
                "frozen_files": len(freeze["files"]),
                "multi_horizon_gate_passed": result["multi_horizon_gate"]["passed"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
