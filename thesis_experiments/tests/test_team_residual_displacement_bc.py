from pathlib import Path

import numpy as np
import torch

from thesis_experiments.policies.networks import TemporalResidualDisplacementPredictor
from thesis_experiments.training.team_residual_displacement_bc import (
    causal_motion_history,
    evaluate_team_residual_displacement_bc,
    load_team_residual_examples,
    train_team_residual_displacement_bc,
)
from thesis_experiments.transitions.observations import LOCAL_FEATURES, OBSERVATION_VERSION
from thesis_experiments.transitions.schema import JointTransitionDataset
from thesis_experiments.scripts.summarize_team_residual_bc_pretest import summarize


def _dataset(path: Path, *, match_id: str) -> Path:
    transitions = 6
    observations = np.zeros((transitions, 22, LOCAL_FEATURES), dtype=np.float32)
    observations[..., 24] = 0.01
    observations[..., 26] = np.arange(transitions, dtype=np.float32)[:, None]
    next_observations = observations.copy()
    next_observations[..., 0] += 0.01
    dataset = JointTransitionDataset(
        local_observations=observations,
        global_states=np.zeros((transitions, 46), dtype=np.float32),
        actions=np.zeros((transitions, 22), dtype=np.int64),
        team_rewards=np.zeros((transitions, 2), dtype=np.float32),
        next_local_observations=next_observations,
        next_global_states=np.zeros((transitions, 46), dtype=np.float32),
        dones=np.asarray([False] * (transitions - 1) + [True]),
        frames=np.arange(transitions, dtype=np.int64),
        player_ids=np.asarray([f"p{i}" for i in range(22)]),
        team_ids=np.asarray(["focus"] * 11 + ["opponent"] * 11),
        source="test",
        match_id=match_id,
        observation_version=OBSERVATION_VERSION,
    )
    dataset.save(path)
    return path


def test_residual_head_initially_returns_exact_zero():
    model = TemporalResidualDisplacementPredictor(input_size=7, hidden_size=8)
    prediction = model(torch.randn(5, 7))
    torch.testing.assert_close(prediction, torch.zeros(5, 2))


def test_motion_history_is_causal_and_zero_padded():
    observations = np.zeros((4, 2, LOCAL_FEATURES), dtype=np.float32)
    observations[:, :, 24] = np.arange(1, 5, dtype=np.float32)[:, None]
    history = causal_motion_history(observations, history_steps=3).reshape(4, 2, 3, 4)
    assert history[0, 0, 0, 0] == 0.0
    assert history[0, 0, 1, 0] == 0.0
    assert np.isclose(history[0, 0, 2, 0], 52.5)
    np.testing.assert_allclose(history[2, 0, :, 0], [52.5, 105.0, 157.5])


def test_epoch_zero_is_selectable_and_equals_persistence(tmp_path):
    train = _dataset(tmp_path / "train.npz", match_id="train")
    validation = _dataset(tmp_path / "validation.npz", match_id="validation")
    checkpoint = tmp_path / "residual.pt"
    training = train_team_residual_displacement_bc(
        [train],
        checkpoint,
        team_id="focus",
        validation_paths=[validation],
        epochs=1,
        batch_size=16,
        hidden_size=8,
        history_steps=3,
        minimum_epochs=0,
    )
    assert training["validation_history"][0]["epoch"] == 0
    initial = training["validation_history"][0]
    assert np.isclose(
        initial["policy_mean_displacement_error_meters"],
        initial["previous_velocity_mean_displacement_error_meters"],
    )
    evaluation = evaluate_team_residual_displacement_bc(
        checkpoint, [validation], team_id="focus"
    )
    assert evaluation["test_used_for_selection"] is False


def test_residual_loader_shapes(tmp_path):
    path = _dataset(tmp_path / "data.npz", match_id="m1")
    inputs, targets, persistence, metadata = load_team_residual_examples(
        path, "focus", history_steps=3
    )
    assert inputs.shape == (66, LOCAL_FEATURES + 12)
    assert targets.shape == persistence.shape == (66, 2)
    assert metadata["match_id"] == "m1"


def test_pretest_gate_requires_two_improvements_and_no_regression(tmp_path):
    paths = []
    for seed, error in enumerate((0.9, 0.95, 1.0)):
        run = {
            "method": "residual_displacement_bc",
            "phase": "pretest",
            "team_id": "focus",
            "split_matches": {"train": ["a"], "validation": ["b"], "test": ["c"]},
            "training": {
                "seed": seed,
                "selected_epoch": seed,
                "history_steps": 10,
                "observation_variant": "temporal_context_no_slot",
            },
            "evaluations": {
                "validation": {
                    "overall": {
                        "policy": {"mean_displacement_error_meters": error},
                        "previous_velocity_baseline": {
                            "mean_displacement_error_meters": 1.0
                        },
                    }
                }
            },
            "selection_contract": {"test_loaded": False},
        }
        path = tmp_path / f"seed_{seed}.json"
        path.write_text(__import__("json").dumps(run))
        paths.append(path)
    result = summarize(paths)
    assert result["multi_horizon_gate"]["passed"]
