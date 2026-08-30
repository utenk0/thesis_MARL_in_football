"""Stateful closed-loop controller for 22-player target trajectories."""

from __future__ import annotations

import numpy as np

from thesis_experiments.datasets.action_mapping import GRF_ACTIONS
from thesis_experiments.datasets.movement import movement_actions


class TrajectoryController:
    """Choose GRF actions that reduce current-to-target position errors.

    Right-team action directions are mirrored because GRF exposes right-agent
    controls in their own attacking coordinate frame.
    """

    def __init__(self, *, stop_tolerance: float = 0.008, sprint_threshold: float = 0.08, sprint_release_threshold: float = 0.04):
        if not 0 <= stop_tolerance < sprint_release_threshold < sprint_threshold:
            raise ValueError("Expected stop_tolerance < sprint_release_threshold < sprint_threshold.")
        self.stop_tolerance = stop_tolerance
        self.sprint_threshold = sprint_threshold
        self.sprint_release_threshold = sprint_release_threshold
        self.direction_actions = np.zeros(22, dtype=np.int64)
        self.sprinting = np.zeros(22, dtype=bool)

    def reset(self) -> None:
        self.direction_actions.fill(0)
        self.sprinting.fill(False)

    def actions(self, simulated: np.ndarray, target: np.ndarray) -> np.ndarray:
        simulated = np.asarray(simulated, dtype=np.float32)
        target = np.asarray(target, dtype=np.float32)
        if simulated.shape != (22, 2) or target.shape != (22, 2):
            raise ValueError(f"Expected simulated and target shapes (22,2), got {simulated.shape} and {target.shape}.")
        world_delta = target - simulated
        control_delta = world_delta.copy()
        control_delta[11:] *= -1.0
        desired = movement_actions(np.zeros_like(control_delta), control_delta, threshold=0.0)
        distances = np.linalg.norm(world_delta, axis=1)
        output = np.zeros(22, dtype=np.int64)
        for index, distance in enumerate(distances):
            if not np.isfinite(distance):
                output[index] = GRF_ACTIONS["idle"]
                continue
            if distance <= self.stop_tolerance:
                if self.sprinting[index]:
                    output[index] = GRF_ACTIONS["release_sprint"]
                    self.sprinting[index] = False
                elif self.direction_actions[index] != 0:
                    output[index] = GRF_ACTIONS["release_direction"]
                    self.direction_actions[index] = 0
                else:
                    output[index] = GRF_ACTIONS["idle"]
                continue
            desired_direction = int(desired[index])
            if desired_direction != self.direction_actions[index]:
                output[index] = desired_direction
                self.direction_actions[index] = desired_direction
            elif distance >= self.sprint_threshold and not self.sprinting[index]:
                output[index] = GRF_ACTIONS["sprint"]
                self.sprinting[index] = True
            elif distance <= self.sprint_release_threshold and self.sprinting[index]:
                output[index] = GRF_ACTIONS["release_sprint"]
                self.sprinting[index] = False
            else:
                output[index] = desired_direction
        return output
