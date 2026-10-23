"""Aggregate and freeze autonomous multi-horizon BC before held-out test use."""

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
        raise ValueError("The multi-horizon protocol requires exactly three seeds.")
    contracts = {
        json.dumps(
            {
                "team": run["team_id"],
                "splits": run["split_matches"],
                "horizons": run["training"]["horizons"],
                "history": run["training"]["history_steps"],
                "variant": run["training"]["observation_variant"],
            },
            sort_keys=True,
        )
        for run in runs
    }
    if len(contracts) != 1:
        raise ValueError("Multi-horizon runs do not share one experiment contract.")
    if any(run["phase"] != "pretest" for run in runs):
        raise ValueError("Only pretest runs may enter this freeze.")
    if any(run["selection_contract"]["test_loaded"] for run in runs):
        raise ValueError("A multi-horizon pretest run loaded test data.")
    if any(run["selection_contract"]["future_observations_used"] for run in runs):
        raise ValueError("Future observations were used by an autonomous forecast.")

    validation = [run["validation_evaluation"]["overall"] for run in runs]
    policy_ade = [float(row["policy"]["autonomous_ade_meters"]) for row in validation]
    policy_fde = [float(row["policy"]["autonomous_fde_meters"]) for row in validation]
    baseline_ade = [
        float(row["constant_velocity"]["autonomous_ade_meters"])
        for row in validation
    ]
    baseline_fde = [
        float(row["constant_velocity"]["autonomous_fde_meters"])
        for row in validation
    ]
    improved = [
        policy_a < baseline_a - 1e-8 and policy_f < baseline_f - 1e-8
        for policy_a, baseline_a, policy_f, baseline_f in zip(
            policy_ade, baseline_ade, policy_fde, baseline_fde
        )
    ]
    noninferior = [
        policy_a <= baseline_a + 1e-8 and policy_f <= baseline_f + 1e-8
        for policy_a, baseline_a, policy_f, baseline_f in zip(
            policy_ade, baseline_ade, policy_fde, baseline_fde
        )
    ]
    policy_ade_stats = _stats(policy_ade)
    policy_fde_stats = _stats(policy_fde)
    baseline_ade_stats = _stats(baseline_ade)
    baseline_fde_stats = _stats(baseline_fde)
    passed = bool(
        policy_ade_stats["mean"] < baseline_ade_stats["mean"] - 1e-8
        and policy_fde_stats["mean"] < baseline_fde_stats["mean"] - 1e-8
        and sum(improved) >= 2
        and all(noninferior)
    )
    first = runs[0]
    return {
        "method": "multi_horizon_trajectory_bc",
        "phase": "pretest_frozen",
        "team_id": first["team_id"],
        "seeds": [int(run["training"]["seed"]) for run in runs],
        "selected_epochs": [int(run["training"]["selected_epoch"]) for run in runs],
        "horizons": first["training"]["horizons"],
        "history_steps": first["training"]["history_steps"],
        "future_observations_used": False,
        "selection": {
            "metric": "validation_autonomous_ade_meters",
            "epoch_zero_included": True,
            "test_loaded": False,
            "test_used_for_selection": False,
        },
        "validation": {
            "policy_ade_meters": policy_ade_stats,
            "policy_fde_meters": policy_fde_stats,
            "constant_velocity_ade_meters": baseline_ade_stats,
            "constant_velocity_fde_meters": baseline_fde_stats,
            "per_seed_policy_ade": policy_ade,
            "per_seed_policy_fde": policy_fde,
            "per_seed_constant_velocity_ade": baseline_ade,
            "per_seed_constant_velocity_fde": baseline_fde,
            "strict_ade_and_fde_improvements": improved,
            "noninferior_ade_and_fde": noninferior,
        },
        "trajectory_gate": {
            "rule": (
                "Mean validation ADE and 100-step FDE must both beat constant velocity; "
                "at least two of three seeds must improve both and no seed may regress."
            ),
            "passed": passed,
        },
        "result_paths": [str(path) for path in paths],
    }


def _write_report(result: dict[str, object], path: Path) -> None:
    row = result["validation"]
    lines = [
        "# Multi-horizon trajectory BC — frozen pretest decision",
        "",
        f"- Team: `{result['team_id']}`",
        f"- Horizons: {result['horizons']}",
        f"- Seeds: {result['seeds']}",
        f"- Selected epochs: {result['selected_epochs']}",
        "- Future observations used: no",
        "- Test data loaded: no",
        "",
        "| Validation forecast | ADE (m) | 100-step FDE (m) |",
        "|---|---:|---:|",
        f"| Multi-horizon BC | {row['policy_ade_meters']['mean']:.4f} ± {row['policy_ade_meters']['std']:.4f} | {row['policy_fde_meters']['mean']:.4f} ± {row['policy_fde_meters']['std']:.4f} |",
        f"| Constant velocity | {row['constant_velocity_ade_meters']['mean']:.4f} | {row['constant_velocity_fde_meters']['mean']:.4f} |",
        "",
        f"Trajectory gate: **{'PASS' if result['trajectory_gate']['passed'] else 'FAIL'}**.",
        "",
        result["trajectory_gate"]["rule"],
    ]
    path.write_text("\n".join(lines) + "\n")


def _freeze(
    experiment_dir: Path,
    manifest: Path,
    residual_gate: Path,
    summary_path: Path,
    report_path: Path,
) -> dict[str, object]:
    root = Path(__file__).resolve().parents[2]
    files = [
        manifest.resolve(),
        residual_gate.resolve(),
        summary_path.resolve(),
        report_path.resolve(),
        (root / "thesis_experiments/training/team_residual_displacement_bc.py").resolve(),
        (root / "thesis_experiments/training/team_multi_horizon_bc.py").resolve(),
        (root / "thesis_experiments/scripts/train_team_multi_horizon_bc.py").resolve(),
    ]
    for seed_dir in sorted(experiment_dir.glob("seed_*")):
        files.extend(
            (seed_dir / name).resolve()
            for name in (
                "single_team_multi_horizon_bc.pt",
                "config.json",
                "pretest_result.json",
                "validation_evaluation.json",
            )
        )
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Cannot freeze multi-horizon BC: {missing}")
    result = {
        "status": "multi_horizon_bc_pretest_frozen",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "future_observations_used": False,
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
    parser.add_argument("--residual-gate", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    residual = json.loads(args.residual_gate.read_text())
    if not residual["multi_horizon_gate"]["passed"]:
        raise ValueError("Residual validation gate did not authorize multi-horizon BC.")
    result = summarize(args.results)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "pretest_summary.json"
    report_path = args.output_dir / "PRETEST_REPORT.md"
    summary_path.write_text(json.dumps(result, indent=2))
    _write_report(result, report_path)
    freeze = _freeze(
        args.output_dir,
        args.manifest,
        args.residual_gate,
        summary_path,
        report_path,
    )
    print(
        json.dumps(
            {
                "summary": str(summary_path),
                "report": str(report_path),
                "freeze": str(args.output_dir / "pretest_freeze.json"),
                "frozen_files": len(freeze["files"]),
                "trajectory_gate_passed": result["trajectory_gate"]["passed"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
