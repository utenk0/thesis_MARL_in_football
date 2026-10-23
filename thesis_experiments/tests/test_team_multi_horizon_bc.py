from pathlib import Path

import numpy as np
import torch

from thesis_experiments.training.team_multi_horizon_bc import (
    MultiHorizonResidualPredictor,
    evaluate_team_multi_horizon_bc,
    load_team_multi_horizon_examples,
    train_team_multi_horizon_bc,
)
from thesis_experiments.transitions.observations import LOCAL_FEATURES, OBSERVATION_VERSION
from thesis_experiments.transitions.schema import JointTransitionDataset
from thesis_experiments.scripts.summarize_team_multi_horizon_bc_pretest import summarize


def _dataset(path: Path, *, match_id: str) -> Path:
    transitions = 8
    positions = np.arange(transitions, dtype=np.float32)[:, None, None] * 0.01
    observations = np.zeros((transitions, 22, LOCAL_FEATURES), dtype=np.float32)
    observations[..., :2] = np.concatenate(
        [np.broadcast_to(positions, (transitions, 22, 1)), np.zeros((transitions, 22, 1))],
        axis=-1,
    )
    observations[..., 24] = 0.01
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


def test_multi_horizon_head_starts_at_zero_correction():
    model = MultiHorizonResidualPredictor(9, (1, 3, 5), hidden_size=8)
    torch.testing.assert_close(model(torch.randn(4, 9)), torch.zeros(4, 3, 2))


def test_multi_horizon_targets_and_velocity_are_causal(tmp_path):
    path = _dataset(tmp_path / "data.npz", match_id="m")
    inputs, targets, velocity, _, metadata = load_team_multi_horizon_examples(
        path, "focus", history_steps=2, horizons=(1, 3)
    )
    assert inputs.shape == (66, LOCAL_FEATURES + 8)
    assert targets.shape == velocity.shape == (66, 2, 2)
    np.testing.assert_allclose(targets[:, 0, 0], 0.525, atol=1e-5)
    np.testing.assert_allclose(targets[:, 1, 0], 1.575, atol=1e-5)
    np.testing.assert_allclose(targets, velocity, atol=1e-5)
    assert metadata["time_starts"] == 6


def test_train_and_evaluate_multi_horizon_epoch_zero(tmp_path):
    train = _dataset(tmp_path / "train.npz", match_id="train")
    validation = _dataset(tmp_path / "validation.npz", match_id="validation")
    checkpoint = tmp_path / "model.pt"
    training = train_team_multi_horizon_bc(
        [train],
        checkpoint,
        team_id="focus",
        validation_paths=[validation],
        horizons=(1, 3),
        epochs=1,
        batch_size=16,
        hidden_size=8,
        history_steps=2,
        minimum_epochs=0,
    )
    assert training["validation_history"][0]["epoch"] == 0
    result = evaluate_team_multi_horizon_bc(
        checkpoint, [validation], team_id="focus"
    )
    assert result["future_observations_used"] is False
    assert result["horizons"] == [1, 3]


def test_multi_horizon_pretest_gate_uses_ade_and_fde(tmp_path):
    paths = []
    for seed, policy_ade, policy_fde in ((0, 0.8, 1.8), (1, 0.9, 1.9), (2, 1.0, 2.0)):
        def row(ade, fde):
            return {"autonomous_ade_meters": ade, "autonomous_fde_meters": fde}

        run = {
            "phase": "pretest",
            "team_id": "focus",
            "split_matches": {"train": ["a"], "validation": ["b"], "test": ["c"]},
            "training": {
                "seed": seed,
                "selected_epoch": seed,
                "horizons": [1, 3],
                "history_steps": 2,
                "observation_variant": "temporal_context_no_slot",
            },
            "selection_contract": {"test_loaded": False, "future_observations_used": False},
            "validation_evaluation": {
                "overall": {
                    "policy": row(policy_ade, policy_fde),
                    "constant_velocity": row(1.0, 2.0),
                }
            },
        }
        path = tmp_path / f"seed_{seed}.json"
        path.write_text(__import__("json").dumps(run))
        paths.append(path)
    result = summarize(paths)
    assert result["trajectory_gate"]["passed"]
