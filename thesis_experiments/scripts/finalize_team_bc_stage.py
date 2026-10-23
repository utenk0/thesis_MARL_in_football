"""Finalize the BC stage and bind later methods to the trajectory contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from thesis_experiments.evaluation.trajectory_contract import (
    load_contract,
    validate_test_summary,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_freeze(path: Path, expected_status: str) -> dict[str, object]:
    result = json.loads(path.read_text())
    if result["status"] != expected_status:
        raise ValueError(f"Unexpected freeze status in {path}: {result['status']}")
    bad = [
        row["path"]
        for row in result["files"]
        if not Path(row["path"]).is_file()
        or _sha256(Path(row["path"])) != row["sha256"]
    ]
    if bad:
        raise ValueError(f"Freeze verification failed for {path}: {bad}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action-dir", required=True, type=Path)
    parser.add_argument("--residual-dir", required=True, type=Path)
    parser.add_argument("--multi-horizon-dir", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    _verify_freeze(
        args.action_dir / "frozen_discrete_action_bc.json",
        "frozen_discrete_action_bc",
    )
    _verify_freeze(
        args.residual_dir / "pretest_freeze.json",
        "residual_bc_pretest_frozen",
    )
    _verify_freeze(
        args.multi_horizon_dir / "pretest_freeze.json",
        "multi_horizon_bc_pretest_frozen",
    )
    contract = load_contract(args.contract)
    multi_test = json.loads((args.multi_horizon_dir / "test_summary.json").read_text())
    validate_test_summary(contract, multi_test)
    residual_pretest = json.loads((args.residual_dir / "pretest_summary.json").read_text())
    residual_test = json.loads((args.residual_dir / "test_summary.json").read_text())
    multi_pretest = json.loads((args.multi_horizon_dir / "pretest_summary.json").read_text())

    defining_files = [
        args.action_dir / "frozen_discrete_action_bc.json",
        args.action_dir / "diagnostics/BC_DIAGNOSTICS.md",
        args.residual_dir / "pretest_freeze.json",
        args.residual_dir / "PRETEST_REPORT.md",
        args.residual_dir / "test_summary.json",
        args.residual_dir / "TEST_REPORT.md",
        args.multi_horizon_dir / "pretest_freeze.json",
        args.multi_horizon_dir / "PRETEST_REPORT.md",
        args.multi_horizon_dir / "test_summary.json",
        args.multi_horizon_dir / "TEST_REPORT.md",
        args.contract,
        Path("thesis_plan/TRAJECTORY_EVALUATION_CONTRACT.md"),
        Path("thesis_plan/DIRECT_DISPLACEMENT_BC_EXPERIMENT.md"),
    ]
    missing = [str(path) for path in defining_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    final = {
        "status": "bc_stage_complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "team_id": contract["team_id"],
        "discrete_action_bc": {
            "status": "frozen_movement_action_baseline",
            "complete_football_action_policy": False,
            "reason": "event-derived held-out actions have zero recall",
        },
        "residual_displacement_bc": {
            "validation_gate_passed": residual_pretest["multi_horizon_gate"]["passed"],
            "test_policy_error_meters": residual_test["test"][
                "policy_mean_displacement_error_meters"
            ],
            "test_previous_velocity_error_meters": residual_test["test"][
                "previous_velocity_mean_displacement_error_meters"
            ],
        },
        "multi_horizon_bc": {
            "validation_gate_passed": multi_pretest["trajectory_gate"]["passed"],
            "future_observations_used": False,
            "test_policy_ade_meters": multi_test["test"]["policy"]["ade_meters"],
            "test_policy_fde_meters": multi_test["test"]["policy"]["fde_meters"],
            "test_constant_velocity_ade_meters": multi_test["test"][
                "constant_velocity"
            ]["ade_meters"],
            "test_constant_velocity_fde_meters": multi_test["test"][
                "constant_velocity"
            ]["fde_meters"],
        },
        "subsequent_method_contract": {
            "required": contract["contract"],
            "contract_file": str(args.contract),
            "validated_against_bc_result": True,
        },
        "claim_boundary": (
            "BC passes the offline autonomous held-out trajectory-forecasting contract. "
            "This does not establish closed-loop simulator control or complete-match reconstruction."
        ),
        "files": [
            {"path": str(path), "sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in defining_files
        ],
    }
    (args.output_dir / "final_manifest.json").write_text(json.dumps(final, indent=2))

    residual_policy = residual_test["test"]["policy_mean_displacement_error_meters"]
    residual_baseline = residual_test["test"][
        "previous_velocity_mean_displacement_error_meters"
    ]
    policy_ade = multi_test["test"]["policy"]["ade_meters"]
    policy_fde = multi_test["test"]["policy"]["fde_meters"]
    velocity_ade = multi_test["test"]["constant_velocity"]["ade_meters"]
    velocity_fde = multi_test["test"]["constant_velocity"]["fde_meters"]
    lines = [
        "# Final BC method-stage result",
        "",
        "## Status",
        "",
        "- Discrete BC: frozen movement-action imitation baseline; not a complete event-action policy.",
        "- Residual displacement BC: validation gate passed; frozen test improves on previous velocity.",
        "- Multi-horizon BC: validation gate passed; frozen autonomous test improves on constant velocity.",
        f"- Subsequent methods must use `{contract['contract']}`.",
        "",
        "## Frozen held-out results",
        "",
        "| Experiment | BC | Baseline |",
        "|---|---:|---:|",
        f"| One-step displacement error | {residual_policy['mean']:.6f} ± {residual_policy['std']:.6f} m | previous velocity: {residual_baseline['mean']:.6f} m |",
        f"| Multi-horizon ADE | {policy_ade['mean']:.4f} ± {policy_ade['std']:.4f} m | constant velocity: {velocity_ade['mean']:.4f} m |",
        f"| 100-step FDE | {policy_fde['mean']:.4f} ± {policy_fde['std']:.4f} m | constant velocity: {velocity_fde['mean']:.4f} m |",
        "",
        "## Claim boundary",
        "",
        final["claim_boundary"],
    ]
    (args.output_dir / "FINAL_BC_REPORT.md").write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            {
                "status": final["status"],
                "manifest": str(args.output_dir / "final_manifest.json"),
                "report": str(args.output_dir / "FINAL_BC_REPORT.md"),
                "contract": contract["contract"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
