import json
from pathlib import Path

import numpy as np
import torch

from thesis_experiments.methods.trajectory import (
    TrajectoryDatasetSplit,
    TrajectoryTrainingConfig,
)
from thesis_experiments.methods.trajectory_registry import (
    available_trajectory_methods,
    load_trajectory_method,
    train_trajectory_method,
)
from thesis_experiments.training.team_multi_horizon_bc import (
    MultiHorizonResidualPredictor,
)
from thesis_experiments.training.team_residual_displacement_bc import (
    residual_input_mask,
)
from thesis_experiments.training.team_trajectory_cql import (
    TrajectoryCQLQNetwork,
    fit_trajectory_prototypes,
    nearest_trajectory_prototypes,
)
from thesis_experiments.training.team_trajectory_ctde import (
    CentralizedTrajectoryCritic,
)
from thesis_experiments.scripts.summarize_trajectory_cql_pretest import (
    summarize as summarize_cql_pretest,
)
from thesis_experiments.scripts.summarize_trajectory_ctde_pretest import (
    summarize as summarize_ctde_pretest,
)
from thesis_experiments.transitions.observations import LOCAL_FEATURES


def _checkpoint(path: Path) -> Path:
    horizons = (1, 3)
    history_steps = 2
    input_mask = residual_input_mask("temporal_context_no_slot", history_steps)
    model = MultiHorizonResidualPredictor(len(input_mask), horizons, hidden_size=8)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_size": len(input_mask),
            "hidden_size": 8,
            "history_steps": history_steps,
            "horizons": list(horizons),
            "input_mask": torch.from_numpy(input_mask),
            "selected_epoch": 0,
            "epoch_zero": "exact_constant_velocity_baseline",
            "selection_metric": "validation_autonomous_ade_meters",
            "seed": 0,
            "team_id": "focus",
            "future_observations_used": False,
            "test_used_for_selection": False,
        },
        path,
    )
    return path


def test_trajectory_registry_loads_bc_and_predicts_shared_shape(tmp_path):
    method = load_trajectory_method("bc", _checkpoint(tmp_path / "bc.pt"))
    observations = np.zeros((5, LOCAL_FEATURES), dtype=np.float32)
    observations[:, 24] = 0.01
    history = np.zeros((5, 8), dtype=np.float32)
    prediction = method.predict(observations, history)
    assert available_trajectory_methods() == ["bc", "cql", "ctde"]
    assert prediction.shape == (5, 2, 2)
    np.testing.assert_allclose(prediction[:, 0, 0], 0.525, atol=1e-6)
    np.testing.assert_allclose(prediction[:, 1, 0], 1.575, atol=1e-6)


def test_trajectory_split_validates_manifest_against_contract(tmp_path):
    paths = {}
    for split, match in (("train", "a"), ("validation", "b"), ("test", "c")):
        path = tmp_path / f"{split}.npz"
        path.touch()
        paths[split] = [{"path": str(path), "match_id": match}]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"team_id": "focus", "splits": paths}))
    contract = tmp_path / "contract.json"
    contract.write_text(
        json.dumps(
            {
                "contract": "test",
                "team_id": "focus",
                "split": {
                    "train_matches": ["a"],
                    "validation_matches": ["b"],
                    "test_matches": ["c"],
                },
                "information": {
                    "future_observations_used": False,
                    "causal_history_steps": 2,
                    "observation_variant": "temporal_context_no_slot",
                },
                "forecast": {"horizons_steps": [1, 3]},
                "repetitions": {"seeds": [0]},
            }
        )
    )
    dataset = TrajectoryDatasetSplit.from_manifest(
        manifest,
        contract_path=contract,
    )
    assert dataset.team_id == "focus"
    assert dataset.train_matches == ("a",)
    assert dataset.paths("test") == (tmp_path / "test.npz",)


def test_trajectory_training_config_rejects_invalid_horizons():
    try:
        TrajectoryTrainingConfig(horizons=(1, 1, 3))
    except ValueError as error:
        assert "unique and increasing" in str(error)
    else:
        raise AssertionError("duplicate horizons were accepted")


def _cql_checkpoint(path: Path) -> Path:
    horizons = (1, 3)
    history_steps = 2
    input_mask = residual_input_mask("temporal_context_no_slot", history_steps)
    model = TrajectoryCQLQNetwork(
        len(input_mask), action_count=2, hidden_size=8
    )
    with torch.no_grad():
        model.network[-1].bias[1] = 1.0
    prototypes = np.zeros((2, 2, 2), dtype=np.float32)
    prototypes[1, :, 1] = [0.25, 0.75]
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_size": len(input_mask),
            "hidden_size": 8,
            "history_steps": history_steps,
            "horizons": list(horizons),
            "input_mask": torch.from_numpy(input_mask),
            "prototypes": torch.from_numpy(prototypes),
            "action_count": 2,
            "selected_epoch": 1,
            "epoch_zero": "exact_constant_velocity_baseline",
            "selection_metric": "validation_autonomous_ade_meters",
            "test_used_for_selection": False,
        },
        path,
    )
    return path


def test_trajectory_registry_loads_cql_and_returns_shared_shape(tmp_path):
    method = load_trajectory_method("cql", _cql_checkpoint(tmp_path / "cql.pt"))
    observations = np.zeros((4, LOCAL_FEATURES), dtype=np.float32)
    observations[:, 24] = 0.01
    history = np.zeros((4, 8), dtype=np.float32)
    prediction = method.predict(observations, history)
    assert prediction.shape == (4, 2, 2)
    np.testing.assert_allclose(prediction[:, :, 0], [[0.525, 1.575]] * 4)
    np.testing.assert_allclose(prediction[:, :, 1], [[0.25, 0.75]] * 4)


def test_cql_prototypes_reserve_constant_velocity_action():
    samples = np.asarray(
        [
            [[0.1, 0.0], [0.2, 0.0]],
            [[0.2, 0.0], [0.4, 0.0]],
            [[-0.1, 0.0], [-0.2, 0.0]],
            [[-0.2, 0.0], [-0.4, 0.0]],
        ],
        dtype=np.float32,
    )
    prototypes, metrics = fit_trajectory_prototypes(
        samples,
        action_count=3,
        iterations=5,
        seed=0,
    )
    np.testing.assert_array_equal(prototypes[0], np.zeros((2, 2)))
    labels = nearest_trajectory_prototypes(np.zeros((1, 2, 2)), prototypes)
    np.testing.assert_array_equal(labels, [0])
    assert metrics["action_zero"] == "exact_zero_correction_constant_velocity"


def _trajectory_file(path: Path, match_id: str) -> Path:
    transitions = 8
    observations = np.zeros((transitions, 22, LOCAL_FEATURES), dtype=np.float32)
    next_observations = np.zeros_like(observations)
    global_states = np.zeros((transitions, 46), dtype=np.float32)
    next_global_states = np.zeros_like(global_states)
    for step in range(transitions):
        observations[step, :, 0] = step * 0.01
        observations[step, :, 24] = 0.01
        next_observations[step, :, 0] = (step + 1) * 0.01
        next_observations[step, :, 24] = 0.01
        global_states[step, 0] = step * 0.01
        next_global_states[step, 0] = (step + 1) * 0.01
    np.savez_compressed(
        path,
        local_observations=observations,
        next_local_observations=next_observations,
        global_states=global_states,
        next_global_states=next_global_states,
        team_ids=np.asarray(["focus"] * 11 + ["other"] * 11),
        player_ids=np.asarray([f"p{index}" for index in range(22)]),
        team_rewards=np.tile(np.asarray([[0.01, -0.01]], dtype=np.float32), (transitions, 1)),
        dones=np.asarray([False] * (transitions - 1) + [True]),
        frames=np.arange(transitions),
        match_id=np.asarray(match_id),
        source=np.asarray("synthetic"),
        observation_version=np.asarray("local_v2_temporal_61"),
    )
    return path


def test_trajectory_cql_trains_without_loading_test(tmp_path):
    train_path = _trajectory_file(tmp_path / "train.npz", "train_match")
    validation_path = _trajectory_file(tmp_path / "validation.npz", "validation_match")
    missing_test = tmp_path / "must_not_be_loaded.npz"
    dataset = TrajectoryDatasetSplit(
        team_id="focus",
        train_paths=(train_path,),
        validation_paths=(validation_path,),
        test_paths=(missing_test,),
        train_matches=("train_match",),
        validation_matches=("validation_match",),
        test_matches=("test_match",),
    )
    config = TrajectoryTrainingConfig(
        seed=0,
        horizons=(1, 3),
        history_steps=2,
        epochs=1,
        batch_size=16,
        hidden_size=8,
        early_stopping_patience=1,
        minimum_epochs=0,
        cql_action_count=3,
        cql_target_update_interval=2,
        cql_prototype_samples=128,
        cql_kmeans_iterations=2,
    )
    result = train_trajectory_method("cql", dataset, tmp_path / "output", config)
    assert result.checkpoint.is_file()
    assert result.test_loaded is False
    assert not missing_test.exists()
    assert result.training["future_observations_used"] is False
    assert result.validation["overall"]["policy"]["autonomous_ade_meters"] < 1e-5


def test_trajectory_ctde_trains_with_global_critic_and_local_actor(tmp_path):
    train_path = _trajectory_file(tmp_path / "ctde_train.npz", "train_match")
    validation_path = _trajectory_file(
        tmp_path / "ctde_validation.npz", "validation_match"
    )
    missing_test = tmp_path / "ctde_test_must_remain_unopened.npz"
    dataset = TrajectoryDatasetSplit(
        team_id="focus",
        train_paths=(train_path,),
        validation_paths=(validation_path,),
        test_paths=(missing_test,),
        train_matches=("train_match",),
        validation_matches=("validation_match",),
        test_matches=("test_match",),
    )
    bc_checkpoint = _checkpoint(tmp_path / "initial_bc.pt")
    config = TrajectoryTrainingConfig(
        seed=0,
        horizons=(1, 3),
        history_steps=2,
        epochs=1,
        batch_size=16,
        hidden_size=8,
        early_stopping_patience=1,
        minimum_epochs=0,
        ctde_bc_checkpoint=bc_checkpoint,
        ctde_critic_epochs=1,
        ctde_critic_hidden_size=8,
    )
    result = train_trajectory_method("ctde", dataset, tmp_path / "ctde_output", config)
    assert result.checkpoint.is_file()
    assert result.test_loaded is False
    assert not missing_test.exists()
    assert result.training["centralized_information_used_at_execution"] is False
    assert result.validation["centralized_information_used_at_execution"] is False
    loaded = load_trajectory_method("ctde", result.checkpoint)
    observations = np.zeros((3, LOCAL_FEATURES), dtype=np.float32)
    history = np.zeros((3, 8), dtype=np.float32)
    assert loaded.predict(observations, history).shape == (3, 2, 2)
    assert CentralizedTrajectoryCritic(state_size=47, hidden_size=8)(
        torch.zeros(2, 47)
    ).shape == (2,)


def test_cql_pretest_summary_enforces_three_seed_validation_gate(tmp_path):
    contract = {
        "team_id": "focus",
        "split": {
            "train_matches": ["a"],
            "validation_matches": ["b"],
            "test_matches": ["c"],
        },
        "forecast": {"horizons_steps": [1, 3]},
        "repetitions": {"seeds": [0, 1, 2]},
        "validation_gate": {
            "minimum_improving_seeds": 2,
            "maximum_regressing_seeds": 0,
        },
    }
    paths = []
    for seed in range(3):
        result = {
            "method": "cql",
            "test_loaded": False,
            "future_observations_used": False,
            "checkpoint": str(tmp_path / f"seed_{seed}.pt"),
            "split_matches": {
                "train": ["a"],
                "validation": ["b"],
                "test_declared_but_not_loaded": ["c"],
            },
            "config": {"seed": seed, "horizons": [1, 3], "cql_alpha": 1.0},
            "training": {
                "team_id": "focus",
                "seed": seed,
                "selected_epoch": seed + 1,
                "horizons": [1, 3],
                "history_steps": 2,
                "action_count": 3,
            },
            "validation": {
                "overall": {
                    "policy": {
                        "autonomous_ade_meters": 0.8,
                        "autonomous_fde_meters": 1.8,
                    },
                    "constant_velocity": {
                        "autonomous_ade_meters": 1.0,
                        "autonomous_fde_meters": 2.0,
                    },
                }
            },
        }
        path = tmp_path / f"result_{seed}.json"
        path.write_text(json.dumps(result))
        paths.append(path)
    summary = summarize_cql_pretest(paths, contract=contract)
    assert summary["seeds"] == [0, 1, 2]
    assert summary["trajectory_gate"]["passed"] is True


def test_ctde_pretest_summary_enforces_local_execution_and_gate(tmp_path):
    contract = {
        "team_id": "focus",
        "split": {
            "train_matches": ["a"],
            "validation_matches": ["b"],
            "test_matches": ["c"],
        },
        "forecast": {"horizons_steps": [1, 3]},
        "repetitions": {"seeds": [0, 1, 2]},
        "validation_gate": {
            "minimum_improving_seeds": 2,
            "maximum_regressing_seeds": 0,
        },
    }
    paths = []
    for seed in range(3):
        result = {
            "method": "ctde",
            "test_loaded": False,
            "future_observations_used": False,
            "checkpoint": str(tmp_path / f"ctde_seed_{seed}.pt"),
            "split_matches": {
                "train": ["a"],
                "validation": ["b"],
                "test_declared_but_not_loaded": ["c"],
            },
            "config": {
                "seed": seed,
                "horizons": [1, 3],
                "ctde_bc_checkpoint": str(tmp_path / f"bc_seed_{seed}.pt"),
                "ctde_gamma": 0.99,
            },
            "training": {
                "team_id": "focus",
                "seed": seed,
                "selected_epoch": seed,
                "horizons": [1, 3],
                "history_steps": 2,
                "bc_initialization_sha256": f"bc-hash-{seed}",
                "bc_initialization_seed": seed,
                "bc_initialization_team_id": "focus",
                "centralized_information_used_at_execution": False,
            },
            "validation": {
                "overall": {
                    "policy": {
                        "autonomous_ade_meters": 0.8,
                        "autonomous_fde_meters": 1.8,
                    },
                    "constant_velocity": {
                        "autonomous_ade_meters": 1.0,
                        "autonomous_fde_meters": 2.0,
                    },
                }
            },
        }
        path = tmp_path / f"ctde_result_{seed}.json"
        path.write_text(json.dumps(result))
        paths.append(path)
    summary = summarize_ctde_pretest(paths, contract=contract)
    assert summary["seeds"] == [0, 1, 2]
    assert summary["trajectory_gate"]["passed"] is True
    assert summary["centralized_information_used_at_execution"] is False
