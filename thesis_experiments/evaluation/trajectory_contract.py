"""Validation for the frozen autonomous trajectory comparison contract."""

from __future__ import annotations

import json
from pathlib import Path


def load_contract(path: Path) -> dict[str, object]:
    contract = json.loads(Path(path).read_text())
    required = {"contract", "team_id", "split", "information", "forecast", "repetitions"}
    missing = required - set(contract)
    if missing:
        raise ValueError(f"Trajectory contract is missing fields: {sorted(missing)}")
    if contract["information"]["future_observations_used"]:
        raise ValueError("An autonomous contract cannot allow future observations.")
    return contract


def validate_test_summary(
    contract: dict[str, object], summary: dict[str, object]
) -> None:
    """Reject a method summary that violates the shared comparison interface."""

    errors = []
    if summary.get("team_id") != contract["team_id"]:
        errors.append("team_id does not match the contract")
    if list(summary.get("test_matches", [])) != list(
        contract["split"]["test_matches"]
    ):
        errors.append("test matches do not match the contract")
    if summary.get("future_observations_used") is not False:
        errors.append("future_observations_used must be false")
    if summary.get("test_used_for_selection") is not False:
        errors.append("test_used_for_selection must be false")
    if list(summary.get("horizons", [])) != list(
        contract["forecast"]["horizons_steps"]
    ):
        errors.append("forecast horizons do not match the contract")
    if list(summary.get("seeds", [])) != list(contract["repetitions"]["seeds"]):
        errors.append("seed repetitions do not match the contract")
    test = summary.get("test", {})
    for predictor in ("policy", "stationary", "constant_velocity", "constant_acceleration"):
        if predictor not in test:
            errors.append(f"missing predictor report: {predictor}")
            continue
        values = test[predictor]
        if "ade_meters" not in values or "fde_meters" not in values:
            errors.append(f"missing ADE/FDE for {predictor}")
        horizon_rows = values.get("per_horizon", [])
        reported_horizons = [row.get("horizon_steps") for row in horizon_rows]
        if reported_horizons != list(contract["forecast"]["horizons_steps"]):
            errors.append(f"per-horizon rows do not match for {predictor}")
    if errors:
        raise ValueError("Trajectory contract violation: " + "; ".join(errors))
