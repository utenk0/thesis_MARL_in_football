"""Contract checks for team-level policy composition."""

from __future__ import annotations

import numpy as np
import pytest

from thesis_experiments.policies.networks import SharedActor
from thesis_experiments.policies.team import TeamPolicy, TorchActorPolicy, TwoTeamPolicy


class ConstantPlayerPolicy:
    def __init__(self, action: int):
        self.action = action

    def act(self, observations: np.ndarray, *, deterministic: bool = True) -> np.ndarray:
        return np.full(len(observations), self.action, dtype=np.int64)


def test_two_team_policy_concatenates_left_and_right_actions() -> None:
    left = TeamPolicy.shared("left", ConstantPlayerPolicy(3))
    right = TeamPolicy.shared("right", ConstantPlayerPolicy(7))
    policy = TwoTeamPolicy(left=left, right=right)

    actions = policy.act(np.zeros((22, 24), dtype=np.float32))

    assert actions.shape == (22,)
    assert actions[:11].tolist() == [3] * 11
    assert actions[11:].tolist() == [7] * 11


def test_team_policy_rejects_wrong_player_count() -> None:
    policy = TeamPolicy.shared("left", ConstantPlayerPolicy(0))

    with pytest.raises(ValueError, match="observations"):
        policy.act(np.zeros((10, 24), dtype=np.float32))


def test_torch_actor_policy_returns_one_action_per_player() -> None:
    actor = SharedActor(observation_size=24, action_size=19, hidden_size=16)
    policy = TorchActorPolicy(actor)

    actions = policy.act(np.zeros((11, 24), dtype=np.float32))

    assert actions.shape == (11,)
    assert actions.dtype == np.int64
