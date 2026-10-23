"""Gymnasium replay environment backed by project transition datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from thesis_experiments.training.data import load_arrays

try:  # Keep this module importable in the minimal data-prep environment.
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - exercised only when optional deps are missing.
    gym = None
    spaces = None


class MissingGymnasiumDependency(RuntimeError):
    """Raised when a Gymnasium-backed method is used without Gymnasium."""


@dataclass(slots=True)
class PlayerTransitionBatch:
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    dones: np.ndarray
    frames: np.ndarray
    player_indices: np.ndarray

    @property
    def observation_size(self) -> int:
        return int(self.observations.shape[-1])


def flatten_player_transitions(arrays: dict[str, np.ndarray]) -> PlayerTransitionBatch:
    """Flatten [T,22,*] joint transitions into single-player transitions."""

    local = arrays["local_observations"].astype(np.float32)
    next_local = arrays["next_local_observations"].astype(np.float32)
    actions = arrays["actions"].astype(np.int64)
    team_rewards = arrays["team_rewards"].astype(np.float32)
    dones = arrays["dones"].astype(bool)
    frames = arrays["frames"].astype(np.int64)
    if local.shape[:2] != actions.shape or next_local.shape[:2] != actions.shape:
        raise ValueError("local observations, next observations, and actions must share [T,22].")
    transition_indices = np.repeat(np.arange(len(actions)), 22)
    player_indices = np.tile(np.arange(22), len(actions))
    team_indices = (player_indices >= 11).astype(np.int64)
    return PlayerTransitionBatch(
        observations=local.reshape(-1, local.shape[-1]),
        actions=actions.reshape(-1),
        rewards=team_rewards[transition_indices, team_indices],
        next_observations=next_local.reshape(-1, next_local.shape[-1]),
        dones=dones[transition_indices],
        frames=frames[transition_indices],
        player_indices=player_indices,
    )


def load_player_transitions(paths: list[Path]) -> PlayerTransitionBatch:
    return flatten_player_transitions(load_arrays(paths))


def player_episode_indices(arrays: dict[str, np.ndarray]) -> list[np.ndarray]:
    """Indices into frame-major arrays, grouped by clip and stable player.

    `dones` in the legacy tracking format marks a clip boundary, not a goal or
    a terminal football state. Consumers must treat it as a time limit.
    """
    ends = np.flatnonzero(arrays["dones"]).tolist()
    if len(arrays["actions"]) and (not ends or ends[-1] != len(arrays["actions"]) - 1):
        ends.append(len(arrays["actions"]) - 1)
    episodes = []
    start = 0
    for end in ends:
        for player in range(22):
            episodes.append(np.arange(start, end + 1) * 22 + player)
        start = end + 1
    return episodes


class FootballReplayEnv(gym.Env if gym is not None else object):
    """Teacher-forced football replay environment for single-player policies.

    The action does not alter recorded football dynamics. This environment is a
    learning interface for imitation/offline-RL libraries: each step returns the
    next recorded local observation and rewards actions that match the expert.
    GRF remains the validation simulator for closed-loop rollouts.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        data_paths: list[Path],
        *,
        random_starts: bool = False,
        imitation_bonus: float = 1.0,
        mismatch_penalty: float = 0.0,
        reward_scale: float = 1.0,
        max_episode_steps: int | None = None,
    ):
        if gym is None or spaces is None:
            raise MissingGymnasiumDependency(
                "Gymnasium is required for FootballReplayEnv. Install requirements-methods.txt."
            )
        arrays = load_arrays(data_paths)
        self.batch = flatten_player_transitions(arrays)
        self.episodes = player_episode_indices(arrays)
        if len(self.batch.actions) == 0:
            raise ValueError("At least one player transition is required.")
        self.random_starts = random_starts
        self.imitation_bonus = imitation_bonus
        self.mismatch_penalty = mismatch_penalty
        self.reward_scale = reward_scale
        self.max_episode_steps = max_episode_steps
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.batch.observation_size,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(19)
        self._index = 0
        self._episode_steps = 0
        self._needs_reset = True
        self._episode = self.episodes[0]

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        episode_id = int((options or {}).get("episode_index", self.np_random.integers(len(self.episodes)) if self.random_starts else 0))
        self._episode = self.episodes[episode_id]
        self._index = int(self._episode[0])
        self._episode_steps = 0
        self._needs_reset = False
        return self.batch.observations[self._index].copy(), self._info()

    def step(self, action: int):
        if self._needs_reset:
            raise RuntimeError("Call reset() before stepping or after an episode ends.")
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid GRF action: {action}")
        action_value = int(action)
        expert_action = int(self.batch.actions[self._index])
        matched = action_value == expert_action
        reward = float(self.batch.rewards[self._index] * self.reward_scale)
        reward += self.imitation_bonus if matched else -self.mismatch_penalty
        observation = self.batch.next_observations[self._index].copy()
        terminated = False
        self._episode_steps += 1
        truncated = self._episode_steps == len(self._episode) or bool(self.max_episode_steps and self._episode_steps >= self.max_episode_steps)
        info = self._info()
        info.update({"expert_action": expert_action, "matched_expert": matched})
        self._needs_reset = truncated
        if not truncated:
            self._index = int(self._episode[self._episode_steps])
        return observation, reward, terminated, truncated, info

    def _info(self) -> dict[str, int]:
        return {
            "frame": int(self.batch.frames[self._index]),
            "player_index": int(self.batch.player_indices[self._index]),
            "transition_index": int(self._index),
        }
