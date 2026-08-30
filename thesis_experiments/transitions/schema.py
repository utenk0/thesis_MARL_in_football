"""Array-backed, provider-independent 22-player transition dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(slots=True)
class JointTransitionDataset:
    local_observations: np.ndarray       # [T, 22, L]
    global_states: np.ndarray            # [T, G]
    actions: np.ndarray                  # [T, 22]
    team_rewards: np.ndarray             # [T, 2]
    next_local_observations: np.ndarray  # [T, 22, L]
    next_global_states: np.ndarray       # [T, G]
    dones: np.ndarray                    # [T]
    frames: np.ndarray                   # [T]
    player_ids: np.ndarray               # [22]
    team_ids: np.ndarray                 # [22]
    source: str
    match_id: str

    def validate(self) -> None:
        transitions = len(self.actions)
        expected = {
            "local_observations": (transitions, 22),
            "next_local_observations": (transitions, 22),
            "actions": (transitions, 22),
            "team_rewards": (transitions, 2),
        }
        for name, prefix in expected.items():
            if getattr(self, name).shape[: len(prefix)] != prefix:
                raise ValueError(f"{name} must start with shape {prefix}, got {getattr(self, name).shape}.")
        if self.global_states.shape != self.next_global_states.shape or len(self.global_states) != transitions:
            raise ValueError("Current and next global states must align with transitions.")
        if self.dones.shape != (transitions,) or self.frames.shape != (transitions,):
            raise ValueError("dones and frames must contain one value per transition.")
        if self.player_ids.shape != (22,) or self.team_ids.shape != (22,):
            raise ValueError("Exactly 22 stable player and team identifiers are required.")
        if not np.isfinite(self.local_observations).all() or not np.isfinite(self.global_states).all():
            raise ValueError("Observations must be finite.")
        if np.any((self.actions < 0) | (self.actions > 18)):
            raise ValueError("GRF actions must be in [0,18].")

    def save(self, path: Path) -> None:
        self.validate()
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **{name: getattr(self, name) for name in (
            "local_observations", "global_states", "actions", "team_rewards",
            "next_local_observations", "next_global_states", "dones", "frames",
            "player_ids", "team_ids", "source", "match_id",
        )})

    @classmethod
    def load(cls, path: Path) -> "JointTransitionDataset":
        with np.load(path, allow_pickle=False) as data:
            result = cls(
                **{name: data[name].copy() for name in (
                    "local_observations", "global_states", "actions", "team_rewards",
                    "next_local_observations", "next_global_states", "dones", "frames",
                    "player_ids", "team_ids",
                )},
                source=str(data["source"]), match_id=str(data["match_id"]),
            )
        result.validate()
        return result
