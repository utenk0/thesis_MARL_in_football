"""Aggregate three trajectory-CTDE seeds and freeze them before test use."""

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
    paths: list[Path], *, contract: dict[str, object]
) -> dict[str, object]:
    runs = [json.loads(path.read_text()) for path in paths]
    expected_seeds = list(contract["repetitions"]["seeds"])
    if len(runs) != len(expected_seeds):
        raise ValueError(f"Trajectory CTDE requires exactly {len(expected_seeds)} seeds.")
    if any(run.get("method") != "ctde" for run in runs):
        raise ValueError("Only trajectory-CTDE results may enter the CTDE freeze.")
    if any(run.get("test_loaded") is not False for run in runs):
        raise ValueError("A trajectory-CTDE pretest run loaded test data.")
    if any(run.get("future_observations_used") is not False for run in runs):
        raise ValueError("A trajectory-CTDE actor used future observations.")
    if any(
        run["training"].get("centralized_information_used_at_execution") is not False
        for run in runs
    ):
        raise ValueError("A trajectory-CTDE actor requires centralized execution data.")

    expected_splits = {
        "train": list(contract["split"]["train_matches"]),
        "validation": list(contract["split"]["validation_matches"]),
        "test_declared_but_not_loaded": list(contract["split"]["test_matches"]),
    }
    if any(run["split_matches"] != expected_splits for run in runs):
        raise ValueError("A trajectory-CTDE run violates the frozen match split.")
    expected_horizons = list(contract["forecast"]["horizons_steps"])
    if any(run["training"]["horizons"] != expected_horizons for run in runs):
        raise ValueError("A trajectory-CTDE run violates the frozen horizons.")
    if any(run["training"]["team_id"] != contract["team_id"] for run in runs):
        raise ValueError("A trajectory-CTDE run uses the wrong focal team.")

    runs.sort(key=lambda row: int(row["training"]["seed"]))
    seeds = [int(run["training"]["seed"]) for run in runs]
    if seeds != expected_seeds:
        raise ValueError(f"Expected CTDE seeds {expected_seeds}, found {seeds}.")
    bc_seeds = [int(run["training"]["bc_initialization_seed"]) for run in runs]
    if bc_seeds != seeds:
        raise ValueError("Each CTDE seed must use the BC checkpoint with the same seed.")
    if any(
        run["training"]["bc_initialization_team_id"] != contract["team_id"]
        for run in runs
    ):
        raise ValueError("A CTDE actor was initialized from another team's BC.")
    configurations = {
        json.dumps(
            {
                key: value
                for key, value in run["config"].items()
                if key not in {"seed", "ctde_bc_checkpoint"}
            },
            sort_keys=True,
        )
        for run in runs
    }
    if len(configurations) != 1:
        raise ValueError("Trajectory-CTDE seeds do not share one configuration.")

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
        "method": "trajectory_ctde",
        "phase": "pretest_frozen",
        "team_id": first["training"]["team_id"],
        "seeds": seeds,
        "selected_epochs": [int(run["training"]["selected_epoch"]) for run in runs],
        "horizons": first["training"]["horizons"],
        "history_steps": first["training"]["history_steps"],
        "bc_initialization_sha256": [
            run["training"]["bc_initialization_sha256"] for run in runs
        ],
        "bc_initialization_seeds": bc_seeds,
        "future_observations_used": False,
        "centralized_information_used_at_execution": False,
        "selection": {
            "metric": "validation_autonomous_ade_meters",
            "epoch_zero_included": True,
            "epoch_zero": "frozen_bc_initialization",
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
        "# Trajectory CTDE — frozen pretest decision",
        "",
        f"- Team: `{result['team_id']}`",
        f"- Horizons: {result['horizons']}",
        f"- Seeds: {result['seeds']}",
        f"- Selected actor epochs: {result['selected_epochs']}",
        "- Centralized information used at execution: no",
        "- Test data loaded: no",
        "",
        "| Validation forecast | ADE (m) | 100-step FDE (m) |",
        "|---|---:|---:|",
        f"| Trajectory CTDE | {row['policy_ade_meters']['mean']:.4f} ± {row['policy_ade_meters']['std']:.4f} | {row['policy_fde_meters']['mean']:.4f} ± {row['policy_fde_meters']['std']:.4f} |",
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
        (root / "thesis_experiments/training/team_trajectory_ctde.py").resolve(),
        (root / "thesis_experiments/training/team_multi_horizon_bc.py").resolve(),
        (
            root
            / "thesis_experiments/training/team_residual_displacement_bc.py"
        ).resolve(),
        (root / "thesis_experiments/training/team_displacement_bc.py").resolve(),
        (root / "thesis_experiments/transitions/observations.py").resolve(),
        (root / "thesis_experiments/policies/networks.py").resolve(),
        (
            root / "thesis_experiments/evaluation/trajectory_contract.py"
        ).resolve(),
        (root / "thesis_experiments/methods/trajectory.py").resolve(),
        (root / "thesis_experiments/methods/trajectory_ctde.py").resolve(),
        (root / "thesis_experiments/methods/trajectory_registry.py").resolve(),
        (root / "thesis_experiments/scripts/run_trajectory_method.py").resolve(),
        (
            root
            / "thesis_experiments/scripts/summarize_trajectory_ctde_pretest.py"
        ).resolve(),
        (
            root / "thesis_experiments/scripts/evaluate_trajectory_ctde_test.py"
        ).resolve(),
        (root / "requirements-trajectory.txt").resolve(),
    ]
    for result_path in result_paths:
        result = json.loads(result_path.read_text())
        files.extend(
            [
                result_path.resolve(),
                Path(result["checkpoint"]).resolve(),
                Path(result["config"]["ctde_bc_checkpoint"]).resolve(),
            ]
        )
    unique_files = list(dict.fromkeys(files))
    missing = [str(path) for path in unique_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Cannot freeze trajectory CTDE: {missing}")
    freeze = {
        "status": "trajectory_ctde_pretest_frozen",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "future_observations_used": False,
        "centralized_information_used_at_execution": False,
        "test_evaluation_authorized_after_hash_verification": True,
        "files": [
            {"path": str(path), "sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in unique_files
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
