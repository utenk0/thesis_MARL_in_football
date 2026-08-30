"""Feedback-controller direction, mirroring, and sticky-action tests."""

import numpy as np

from thesis_experiments.simulation.trajectory_controller import TrajectoryController


def test_left_and_right_agents_mirror_world_direction() -> None:
    controller = TrajectoryController()
    simulated = np.zeros((22, 2), dtype=np.float32)
    target = simulated.copy()
    target[0, 0] = .02
    target[11, 0] = .02
    actions = controller.actions(simulated, target)
    assert actions[0] == 5       # left agent: world right -> controller right
    assert actions[11] == 1      # right agent: world right -> controller left


def test_positive_world_y_uses_bottom_action() -> None:
    controller = TrajectoryController()
    simulated = np.zeros((22, 2), dtype=np.float32)
    target = simulated.copy(); target[0, 1] = .02
    assert controller.actions(simulated, target)[0] == 7


def test_far_target_enables_sprint_after_direction_is_set() -> None:
    controller = TrajectoryController()
    simulated = np.zeros((22, 2), dtype=np.float32)
    target = simulated.copy(); target[0, 0] = .2
    assert controller.actions(simulated, target)[0] == 5
    assert controller.actions(simulated, target)[0] == 13


def test_arrival_releases_sprint_then_direction() -> None:
    controller = TrajectoryController()
    simulated = np.zeros((22, 2), dtype=np.float32)
    far = simulated.copy(); far[0, 0] = .2
    controller.actions(simulated, far)
    controller.actions(simulated, far)
    assert controller.actions(simulated, simulated)[0] == 15
    assert controller.actions(simulated, simulated)[0] == 14
