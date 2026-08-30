"""Infer approximate GRF directional actions from consecutive positions."""

from __future__ import annotations

import numpy as np

from thesis_experiments.datasets.action_mapping import GRF_ACTIONS

# GRF pitch y grows toward the bottom of the rendered pitch. Consequently a
# negative y delta is a top action and a positive y delta is a bottom action.
MOVEMENT_ACTIONS = {
    (-1, 0): 1, (-1, -1): 2, (0, -1): 3, (1, -1): 4,
    (1, 0): 5, (1, 1): 6, (0, 1): 7, (-1, 1): 8,
}


def movement_actions(current: np.ndarray, following: np.ndarray, threshold: float) -> np.ndarray:
    """Return one directional or idle action for every available player."""
    limit = min(len(current), len(following))
    actions = np.full(limit, GRF_ACTIONS["idle"], dtype=np.int64)
    for index in range(limit):
        delta = np.asarray(following[index], dtype=np.float32) - np.asarray(current[index], dtype=np.float32)
        if np.isfinite(delta).all() and float(np.linalg.norm(delta)) >= threshold:
            actions[index] = MOVEMENT_ACTIONS[_quantize(delta)]
    return actions


def _quantize(delta: np.ndarray) -> tuple[int, int]:
    x, y = float(delta[0]), float(delta[1])
    sx = 0 if abs(x) < abs(y) * 0.4142 else (1 if x > 0 else -1)
    sy = 0 if abs(y) < abs(x) * 0.4142 else (1 if y > 0 else -1)
    if sx == sy == 0:
        sx = 1 if abs(x) >= abs(y) and x > 0 else -1 if abs(x) >= abs(y) else 0
        sy = 1 if abs(y) > abs(x) and y > 0 else -1 if abs(y) > abs(x) else 0
    return sx, sy
