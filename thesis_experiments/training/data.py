"""Loading and concatenation helpers for transition files."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from thesis_experiments.transitions.schema import JointTransitionDataset


def load_arrays(paths: list[Path]) -> dict[str, np.ndarray]:
    datasets = [JointTransitionDataset.load(path) for path in paths]
    if not datasets:
        raise ValueError("At least one transition dataset is required.")
    fields = ("local_observations", "global_states", "actions", "team_rewards", "next_local_observations", "next_global_states", "dones")
    return {field: np.concatenate([getattr(item, field) for item in datasets], axis=0) for field in fields}
