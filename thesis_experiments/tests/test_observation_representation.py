"""Causality, semantics, and versioning checks for local observations."""

from types import SimpleNamespace

import numpy as np
import pytest

from thesis_experiments.training.data import load_arrays
from thesis_experiments.transitions.builder import build_joint_transitions
from thesis_experiments.transitions.observations import (
    LEGACY_OBSERVATION_VERSION,
    LOCAL_FEATURES,
    LOCAL_FEATURE_NAMES,
    LOCAL_FEATURE_SLICES,
    OBSERVATION_VERSION,
    apply_observation_variant,
    infer_possession,
    observation_feature_mask,
    update_sticky_actions,
)
from thesis_experiments.transitions.schema import JointTransitionDataset


def _audit() -> SimpleNamespace:
    players = np.zeros((4, 22, 2), dtype=np.float32)
    players[:, :, 0] = np.linspace(-0.9, 0.9, 22)
    players[1:, 0, 0] += np.asarray([0.01, 0.03, 0.06])
    ball = np.asarray([[0.0, 0.0], [0.02, 0.0], [0.05, 0.0], [0.09, 0.0]], dtype=np.float32)
    actions = np.zeros((3, 22), dtype=np.int64)
    actions[0, 0] = 5
    actions[1, 0] = 14
    actions[1, 1] = 13
    return SimpleNamespace(
        home_grf_xy=players[:, :11],
        away_grf_xy=players[:, 11:],
        ball_grf_xy=ball,
        joint_actions=actions,
        frames=np.arange(4),
        home_player_ids=[f"h{i}" for i in range(11)],
        away_player_ids=[f"a{i}" for i in range(11)],
        events=[],
        match_id="observation-test",
    )


def test_temporal_observation_is_causal_and_semantically_aligned() -> None:
    data = build_joint_transitions(_audit(), source="test")
    slices = LOCAL_FEATURE_SLICES
    assert data.local_observations.shape == (3, 22, LOCAL_FEATURES)
    assert len(LOCAL_FEATURE_NAMES) == LOCAL_FEATURES
    assert data.observation_version == OBSERVATION_VERSION

    first = data.local_observations[0, 0]
    second = data.local_observations[1, 0]
    third = data.local_observations[2, 0]
    np.testing.assert_array_equal(first[slices["player_velocity"]], [0.0, 0.0])
    np.testing.assert_allclose(second[slices["player_velocity"]], [0.01, 0.0], atol=1e-7)
    np.testing.assert_allclose(second[slices["ball_velocity"]], [0.02, 0.0], atol=1e-7)
    assert first[slices["previous_action"]][0] == 1.0
    assert second[slices["previous_action"]][5] == 1.0
    assert second[slices["sticky_actions"]][4] == 1.0
    assert third[slices["previous_action"]][14] == 1.0
    assert third[slices["sticky_actions"]][:8].sum() == 0.0


def test_changing_future_data_does_not_change_earlier_observations() -> None:
    original = _audit()
    changed = _audit()
    changed.home_grf_xy[-1] = 0.75
    changed.ball_grf_xy[-1] = [-0.8, 0.3]
    changed.joint_actions[-1] = 18
    original_data = build_joint_transitions(original, source="test")
    changed_data = build_joint_transitions(changed, source="test")
    np.testing.assert_allclose(
        original_data.local_observations[:2], changed_data.local_observations[:2]
    )


def test_sticky_action_updates_direction_sprint_and_dribble() -> None:
    state = np.zeros((22, 10), dtype=np.float32)
    actions = np.zeros(22, dtype=np.int64)
    actions[:3] = [3, 13, 17]
    state = update_sticky_actions(state, actions)
    assert state[0, 2] == state[1, 8] == state[2, 9] == 1.0
    actions[:3] = [14, 15, 18]
    state = update_sticky_actions(state, actions)
    assert state[:3].sum() == 0.0


def test_possession_uses_metric_radius() -> None:
    players = np.full((22, 2), [0.9, 0.4], dtype=np.float32)
    players[3] = [0.0, 0.0]
    assert infer_possession(players, np.asarray([0.01, 0.0]), radius_meters=2.0) == (0, 3)
    assert infer_possession(players, np.asarray([0.2, 0.0]), radius_meters=2.0) == (-1, -1)


def test_observation_ablation_masks_keep_common_shape() -> None:
    values = np.ones((2, LOCAL_FEATURES), dtype=np.float32)
    spatial = apply_observation_variant(values, "spatial")
    temporal = apply_observation_variant(values, "temporal_context")
    temporal_no_slot = apply_observation_variant(values, "temporal_context_no_slot")
    full = apply_observation_variant(values, "full")
    assert spatial.shape == temporal.shape == full.shape == values.shape
    assert int(observation_feature_mask("spatial").sum()) == 24
    assert int(observation_feature_mask("temporal_context").sum()) == 32
    assert int(observation_feature_mask("temporal_context_no_slot").sum()) == 31
    assert np.all(temporal_no_slot[:, 7] == 0.0)
    np.testing.assert_array_equal(temporal_no_slot[:, :7], temporal[:, :7])
    np.testing.assert_array_equal(temporal_no_slot[:, 8:], temporal[:, 8:])
    assert int(observation_feature_mask("full").sum()) == LOCAL_FEATURES
    assert spatial[:, 24:].sum() == 0
    assert temporal[:, LOCAL_FEATURE_SLICES["previous_action"]].sum() == 0
    assert temporal[:, LOCAL_FEATURE_SLICES["sticky_actions"]].sum() == 0
    np.testing.assert_array_equal(full, values)


def test_observation_version_roundtrip_and_mixed_version_rejection(tmp_path) -> None:
    current = build_joint_transitions(_audit(), source="test")
    current_path = tmp_path / "current.npz"
    current.save(current_path)
    assert JointTransitionDataset.load(current_path).observation_version == OBSERVATION_VERSION

    legacy_path = tmp_path / "legacy.npz"
    np.savez_compressed(
        legacy_path,
        local_observations=np.zeros((1, 22, 24), dtype=np.float32),
        global_states=np.zeros((1, 46), dtype=np.float32),
        actions=np.zeros((1, 22), dtype=np.int64),
        team_rewards=np.zeros((1, 2), dtype=np.float32),
        next_local_observations=np.zeros((1, 22, 24), dtype=np.float32),
        next_global_states=np.zeros((1, 46), dtype=np.float32),
        dones=np.ones(1, dtype=bool),
        frames=np.zeros(1, dtype=np.int64),
        player_ids=np.asarray([f"p{i}" for i in range(22)]),
        team_ids=np.asarray(["left"] * 11 + ["right"] * 11),
        source="legacy",
        match_id="legacy",
    )
    assert JointTransitionDataset.load(legacy_path).observation_version == LEGACY_OBSERVATION_VERSION
    with pytest.raises(ValueError, match="same observation representation"):
        load_arrays([current_path, legacy_path])
