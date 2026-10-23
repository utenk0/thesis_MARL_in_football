from pathlib import Path

import numpy as np

from thesis_experiments.training.team_displacement_bc import (
    METERS_PER_GRF,
    displacement_metrics,
    evaluate_team_displacement_bc,
    load_team_displacement_examples,
    train_team_displacement_bc,
)
from thesis_experiments.scripts.summarize_team_displacement_bc import summarize
from thesis_experiments.transitions.observations import LOCAL_FEATURES, OBSERVATION_VERSION
from thesis_experiments.transitions.schema import JointTransitionDataset


def _dataset(path: Path, *, match_id: str) -> Path:
    transitions = 5
    observations = np.zeros((transitions, 22, LOCAL_FEATURES), dtype=np.float32)
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


def test_displacement_loader_uses_current_to_next_position(tmp_path):
    path = _dataset(tmp_path / "data.npz", match_id="m1")
    observations, targets, metadata = load_team_displacement_examples(path, "focus")
    assert observations.shape == (55, LOCAL_FEATURES)
    assert targets.shape == (55, 2)
    np.testing.assert_allclose(targets[:, 0], 0.01 * METERS_PER_GRF[0])
    np.testing.assert_allclose(targets[:, 1], 0.0)
    assert metadata["match_id"] == "m1"


def test_displacement_metrics_are_in_meters():
    targets = np.asarray([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)
    predictions = np.zeros_like(targets)
    metrics = displacement_metrics(predictions, targets)
    assert metrics["samples"] == 2
    assert np.isclose(metrics["mean_displacement_error_meters"], 2.5)
    assert np.isclose(metrics["root_mean_squared_displacement_error_meters"], np.sqrt(12.5))


def test_train_and_evaluate_displacement_bc(tmp_path):
    train = _dataset(tmp_path / "train.npz", match_id="train")
    validation = _dataset(tmp_path / "validation.npz", match_id="validation")
    test = _dataset(tmp_path / "test.npz", match_id="test")
    checkpoint = tmp_path / "model.pt"
    training = train_team_displacement_bc(
        [train],
        checkpoint,
        team_id="focus",
        validation_paths=[validation],
        epochs=2,
        batch_size=16,
        hidden_size=8,
    )
    evaluation = evaluate_team_displacement_bc(
        checkpoint,
        [test],
        team_id="focus",
        cumulative_horizon=0,
    )
    assert training["selected_epoch"] >= 1
    assert training["retained_features"] == 31
    assert evaluation["overall"]["policy"]["samples"] == 55
    assert evaluation["selection_metric"] == "validation_mean_displacement_error_meters"


def test_repeated_seed_summary_requires_identical_contracts(tmp_path):
    paths = []
    for seed in (0, 1):
        run_dir = tmp_path / f"seed_{seed}"
        run_dir.mkdir()
        metric = {
            "policy": {
                "samples": 10,
                "mean_displacement_error_meters": 0.2 + seed * 0.01,
                "root_mean_squared_displacement_error_meters": 0.3,
            },
            "stationary_baseline": {
                "samples": 10,
                "mean_displacement_error_meters": 0.5,
                "root_mean_squared_displacement_error_meters": 0.6,
            },
            "previous_velocity_baseline": {
                "samples": 10,
                "mean_displacement_error_meters": 0.1,
                "root_mean_squared_displacement_error_meters": 0.2,
            },
        }
        cumulative = {
            name: {"ade_meters": value, "fde_meters": value * 2, "windows": 3, "horizon_steps": 100}
            for name, value in (("policy", 1.0), ("stationary", 2.0), ("previous_velocity", 0.5))
        }
        run = {
            "team_id": "focus",
            "split_matches": {"train": ["a"], "validation": ["b"], "test": ["c"]},
            "training": {
                "seed": seed,
                "selected_epoch": 2,
                "observation_variant": "temporal_context_no_slot",
                "target": "next_player_displacement_meters",
            },
            "evaluations": {
                split: {"overall": metric, "teacher_forced_cumulative": cumulative}
                for split in ("train", "validation", "test")
            },
        }
        path = run_dir / "result.json"
        path.write_text(__import__("json").dumps(run))
        paths.append(path)
    result = summarize(paths)
    assert result["gates"]["validation_beats_stationary"]
    assert not result["gates"]["validation_beats_previous_velocity"]
