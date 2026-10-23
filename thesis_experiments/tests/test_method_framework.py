"""Contract tests for the unified method framework."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from thesis_experiments.methods import MethodConfig, available_methods, build_method
from thesis_experiments.simulation.gymnasium_replay import flatten_player_transitions
from thesis_experiments.transitions.schema import JointTransitionDataset
from thesis_experiments.transitions.observations import LOCAL_FEATURES
from thesis_experiments.policies.loading import load_method_policy


def _sample_dataset(path: Path) -> Path:
    local = np.zeros((3, 22, LOCAL_FEATURES), dtype=np.float32)
    next_local = np.zeros((3, 22, LOCAL_FEATURES), dtype=np.float32)
    for transition in range(3):
        for player in range(22):
            local[transition, player, 0] = transition * 0.01
            local[transition, player, 1] = player * 0.001
            next_local[transition, player, 0] = transition * 0.01 + 0.001
            next_local[transition, player, 1] = player * 0.001
    dataset = JointTransitionDataset(
        local_observations=local,
        global_states=np.zeros((3, 46), dtype=np.float32),
        actions=np.zeros((3, 22), dtype=np.int64),
        team_rewards=np.zeros((3, 2), dtype=np.float32),
        next_local_observations=next_local,
        next_global_states=np.zeros((3, 46), dtype=np.float32),
        dones=np.asarray([False, False, True]),
        frames=np.arange(3),
        player_ids=np.asarray([f"p{i}" for i in range(22)]),
        team_ids=np.asarray(["left"] * 11 + ["right"] * 11),
        source="test",
        match_id="test-match",
    )
    dataset.save(path)
    return path


def test_registry_exposes_selected_methods() -> None:
    assert available_methods() == ["bc", "cql", "ctde", "dt", "gail", "ghosting", "ppo"]


def test_flatten_player_transitions_shapes() -> None:
    arrays = {
        "local_observations": np.zeros((2, 22, LOCAL_FEATURES), dtype=np.float32),
        "next_local_observations": np.ones((2, 22, LOCAL_FEATURES), dtype=np.float32),
        "actions": np.zeros((2, 22), dtype=np.int64),
        "team_rewards": np.asarray([[1.0, -1.0], [2.0, -2.0]], dtype=np.float32),
        "dones": np.asarray([False, True]),
        "frames": np.asarray([10, 11]),
    }
    batch = flatten_player_transitions(arrays)
    assert batch.observations.shape == (44, LOCAL_FEATURES)
    assert batch.next_observations.shape == (44, LOCAL_FEATURES)
    assert batch.actions.shape == (44,)
    assert batch.rewards[:11].tolist() == [1.0] * 11
    assert batch.rewards[11:22].tolist() == [-1.0] * 11
    assert batch.dones[-1]


def test_bc_method_uses_local_fallback_without_optional_backends(tmp_path: Path) -> None:
    if sys.version_info >= (3, 13):
        pytest.skip("PyTorch is not stable in the local Python 3.13 environment.")
    pytest.importorskip("torch")
    data_path = _sample_dataset(tmp_path / "sample.npz")
    config = MethodConfig(
        method="bc",
        data_paths=[data_path],
        output_dir=tmp_path / "methods",
        epochs=1,
        batch_size=16,
        backend="torch",
    )
    result = build_method(config).train()
    assert result.status == "trained"
    assert result.artifacts["checkpoint"].endswith("shared_actor_bc.pt")


def test_weighted_bc_checkpoint_preserves_ablation_metadata(tmp_path: Path) -> None:
    if sys.version_info >= (3, 13):
        pytest.skip("PyTorch is not stable in the local Python 3.13 environment.")
    torch = pytest.importorskip("torch")
    data_path = _sample_dataset(tmp_path / "sample.npz")
    config = MethodConfig(
        method="bc",
        data_paths=[data_path],
        output_dir=tmp_path / "methods",
        epochs=1,
        batch_size=16,
        backend="torch",
        extra={
            "observation_variant": "spatial",
            "class_weighting": "inverse_frequency",
            "max_class_weight": 10.0,
        },
    )
    result = build_method(config).train()
    checkpoint = torch.load(result.artifacts["checkpoint"], weights_only=True)
    assert checkpoint["observation_variant"] == "spatial"
    assert checkpoint["class_weighting"] == "inverse_frequency"
    assert int(checkpoint["feature_mask"].sum()) == 24
    policy = load_method_policy(tmp_path / "methods/bc/result.json")
    first = np.zeros((1, LOCAL_FEATURES), dtype=np.float32)
    changed_excluded_features = first.copy()
    changed_excluded_features[:, 24:] = 1000.0
    np.testing.assert_array_equal(
        policy.act(first), policy.act(changed_excluded_features)
    )


def test_imitation_bc_checkpoint_roundtrip(tmp_path: Path) -> None:
    if sys.version_info >= (3, 13):
        pytest.skip("PyTorch is not stable in the local Python 3.13 environment.")
    pytest.importorskip("imitation")
    data_path = _sample_dataset(tmp_path / "sample.npz")
    config = MethodConfig(
        method="bc",
        data_paths=[data_path],
        output_dir=tmp_path / "methods",
        epochs=1,
        batch_size=16,
        backend="imitation",
    )
    build_method(config).train()
    policy = load_method_policy(tmp_path / "methods/bc/result.json")
    predicted = policy.act(np.zeros((2, LOCAL_FEATURES), dtype=np.float32))
    assert predicted.shape == (2,)
    assert np.all((0 <= predicted) & (predicted < 19))


def test_ghosting_method_trains_baseline(tmp_path: Path) -> None:
    if sys.version_info >= (3, 13):
        pytest.skip("PyTorch is not stable in the local Python 3.13 environment.")
    pytest.importorskip("torch")
    data_path = _sample_dataset(tmp_path / "sample.npz")
    config = MethodConfig(
        method="ghosting",
        data_paths=[data_path],
        output_dir=tmp_path / "methods",
        epochs=1,
        batch_size=16,
    )
    result = build_method(config).train()
    assert result.status == "trained"
    assert "position_mse" in result.metrics
    assert result.artifacts["checkpoint"].endswith("ghosting_baseline.pt")
