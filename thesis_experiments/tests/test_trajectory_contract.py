import json

import pytest

from thesis_experiments.evaluation.trajectory_contract import (
    load_contract,
    validate_test_summary,
)


def test_current_multi_horizon_report_satisfies_contract():
    contract = load_contract("thesis_plan/trajectory_evaluation_contract_v1.json")
    summary = json.loads(
        __import__("pathlib").Path(
            "artifacts/single_team_multi_horizon_bc_full/test_summary.json"
        ).read_text()
    )
    validate_test_summary(contract, summary)


def test_contract_rejects_future_observations():
    contract = load_contract("thesis_plan/trajectory_evaluation_contract_v1.json")
    summary = {
        "team_id": contract["team_id"],
        "test_matches": contract["split"]["test_matches"],
        "future_observations_used": True,
        "test_used_for_selection": False,
        "horizons": contract["forecast"]["horizons_steps"],
        "seeds": contract["repetitions"]["seeds"],
        "test": {},
    }
    with pytest.raises(ValueError, match="future_observations_used"):
        validate_test_summary(contract, summary)
