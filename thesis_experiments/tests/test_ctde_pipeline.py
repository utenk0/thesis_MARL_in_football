"""Shape and contract checks for the focused CTDE pipeline."""

from types import SimpleNamespace

import numpy as np
import torch

from thesis_experiments.policies.networks import CentralizedCritic, SharedActor
from thesis_experiments.transitions.builder import build_joint_transitions


def test_joint_transition_shapes() -> None:
    left = np.zeros((3, 11, 2), dtype=np.float32)
    right = np.zeros((3, 11, 2), dtype=np.float32)
    ball = np.asarray([[0, 0], [.1, 0], [.2, 0]], dtype=np.float32)
    audit = SimpleNamespace(
        home_grf_xy=left, away_grf_xy=right, ball_grf_xy=ball,
        joint_actions=np.zeros((2, 22), dtype=np.int64), frames=np.arange(3),
        home_player_ids=[f"h{i}" for i in range(11)], away_player_ids=[f"a{i}" for i in range(11)],
        events=[], match_id="test",
    )
    data = build_joint_transitions(audit, source="test")
    assert data.local_observations.shape == (2, 22, 24)
    assert data.global_states.shape == (2, 46)
    assert data.actions.shape == (2, 22)
    assert data.team_rewards.shape == (2, 2)
    assert data.team_rewards[0, 0] == -data.team_rewards[0, 1]
    assert data.dones.tolist() == [False, True]


def test_actor_and_critic_shapes() -> None:
    assert SharedActor()(torch.zeros(22, 24)).shape == (22, 19)
    assert CentralizedCritic()(torch.zeros(4, 46)).shape == (4, 2)
