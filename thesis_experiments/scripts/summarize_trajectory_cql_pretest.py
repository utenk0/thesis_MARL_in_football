"""Aggregate three trajectory-CQL seeds and freeze them before test use."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from thesis_experiments.evaluation.trajectory_contract import load_contract


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


def summarize(
    paths: list[Path],
    *,
    contract: dict[str, object],
) -> dict[str, object]:
    runs = [json.loads(path.read_text()) for path in paths]
    required_seeds = list(contract["repetitions"]["seeds"])
    if len(runs) != len(required_seeds):
        raise ValueError(f"Trajectory CQL requires exactly {len(required_seeds)} seeds.")
    if any(run.get("method") != "cql" for run in runs):
        raise ValueError("Only trajectory-CQL results may enter the CQL freeze.")
    if any(run.get("test_loaded") is not False for run in runs):
        raise ValueError("A trajectory-CQL pretest run loaded test data.")
    if any(run.get("future_observations_used") is not False for run in runs):
        raise ValueError("A trajectory-CQL run used future observations.")
    if any(run["training"]["team_id"] != contract["team_id"] for run in runs):
        raise ValueError("A trajectory-CQL run uses the wrong focal team.")
    expected_splits = {
        "train": list(contract["split"]["train_matches"]),
        "validation": list(contract["split"]["validation_matches"]),
        "test_declared_but_not_loaded": list(contract["split"]["test_matches"]),
    }
    if any(run["split_matches"] != expected_splits for run in runs):
        raise ValueError("A trajectory-CQL run violates the frozen match split.")
    expected_horizons = list(contract["forecast"]["horizons_steps"])
    if any(run["training"]["horizons"] != expected_horizons for run in runs):
        raise ValueError("A trajectory-CQL run violates the frozen horizons.")

    runs.sort(key=lambda row: int(row["training"]["seed"]))
    seeds = [int(run["training"]["seed"]) for run in runs]
    if seeds != required_seeds:
        raise ValueError(f"Expected CQL seeds {required_seeds}, found {seeds}.")
    configurations = {
        json.dumps(
            {
                "team_id": run["training"]["team_id"],
                "split_matches": run["split_matches"],
                "config": {
                    key: value
                    for key, value in run["config"].items()
                    if key != "seed"
                },
                "horizons": run["training"]["horizons"],
                "history_steps": run["training"]["history_steps"],
                "action_count": run["training"]["action_count"],
            },
            sort_keys=True,
        )
        for run in runs
    }
    if len(configurations) != 1:
        raise ValueError("Trajectory-CQL seeds do not share one configuration.")

    validation = [run["validation"]["overall"] for run in runs]
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
    gate = contract["validation_gate"]
    passed = bool(
        policy_ade_stats["mean"] < baseline_ade_stats["mean"] - 1e-8
        and policy_fde_stats["mean"] < baseline_fde_stats["mean"] - 1e-8
        and sum(improved) >= int(gate["minimum_improving_seeds"])
        and sum(not value for value in noninferior)
        <= int(gate["maximum_regressing_seeds"])
    )
    first = runs[0]
    return {
        "method": "trajectory_cql",
        "phase": "pretest_frozen",
        "team_id": first["training"]["team_id"],
        "seeds": seeds,
        "selected_epochs": [int(run["training"]["selected_epoch"]) for run in runs],
        "horizons": first["training"]["horizons"],
        "history_steps": first["training"]["history_steps"],
        "action_count": first["training"]["action_count"],
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
                "Mean validation ADE and 100-step FDE must beat constant velocity; "
                "at least two seeds must improve both and no seed may regress."
            ),
            "passed": passed,
        },
        "result_paths": [str(path) for path in paths],
    }


def _write_report(result: dict[str, object], path: Path) -> None:
    row = result["validation"]
    lines = [
        "# Trajectory CQL — frozen pretest decision",
        "",
        f"- Team: `{result['team_id']}`",
        f"- Horizons: {result['horizons']}",
        f"- Trajectory-plan actions: {result['action_count']}",
        f"- Seeds: {result['seeds']}",
        f"- Selected epochs: {result['selected_epochs']}",
        "- Future observations used: no",
        "- Test data loaded: no",
        "",
        "| Validation forecast | ADE (m) | 100-step FDE (m) |",
        "|---|---:|---:|",
        f"| Trajectory CQL | {row['policy_ade_meters']['mean']:.4f} ± {row['policy_ade_meters']['std']:.4f} | {row['policy_fde_meters']['mean']:.4f} ± {row['policy_fde_meters']['std']:.4f} |",
        f"| Constant velocity | {row['constant_velocity_ade_meters']['mean']:.4f} | {row['constant_velocity_fde_meters']['mean']:.4f} |",
        "",
        f"Trajectory gate: **{'PASS' if result['trajectory_gate']['passed'] else 'FAIL'}**.",
        "",
        result["trajectory_gate"]["rule"],
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _freeze(
    output_dir: Path,
    result_paths: list[Path],
    manifest: Path,
    contract_path: Path,
    summary_path: Path,
    report_path: Path,
) -> dict[str, object]:
    root = Path(__file__).resolve().parents[2]
    files = [
        manifest.resolve(),
        contract_path.resolve(),
        summary_path.resolve(),
        report_path.resolve(),
        (root / "thesis_experiments/training/team_trajectory_cql.py").resolve(),
        (root / "thesis_experiments/methods/trajectory.py").resolve(),
        (root / "thesis_experiments/methods/trajectory_cql.py").resolve(),
        (root / "thesis_experiments/methods/trajectory_registry.py").resolve(),
        (root / "thesis_experiments/scripts/run_trajectory_method.py").resolve(),
    ]
    for result_path in result_paths:
        result = json.loads(result_path.read_text())
        files.extend([result_path.resolve(), Path(result["checkpoint"]).resolve()])
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Cannot freeze trajectory CQL: {missing}")
    freeze = {
        "status": "trajectory_cql_pretest_frozen",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "future_observations_used": False,
        "test_evaluation_authorized_after_hash_verification": True,
        "files": [
            {"path": str(path), "sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in files
        ],
    }
    (output_dir / "pretest_freeze.json").write_text(
        json.dumps(freeze, indent=2) + "\n", encoding="utf-8"
    )
    return freeze


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, nargs="+", type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    contract = load_contract(args.contract)
    result = summarize(args.results, contract=contract)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "pretest_summary.json"
    report_path = args.output_dir / "PRETEST_REPORT.md"
    summary_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    _write_report(result, report_path)
    freeze = _freeze(
        args.output_dir,
        args.results,
        args.manifest,
        args.contract,
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
