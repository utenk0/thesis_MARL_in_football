"""Loading and concatenation helpers for transition files."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from thesis_experiments.transitions.schema import JointTransitionDataset


def load_arrays(paths: list[Path]) -> dict[str, np.ndarray]:
    datasets = [JointTransitionDataset.load(path) for path in paths]
    if not datasets:
        raise ValueError("At least one transition dataset is required.")
    versions = {dataset.observation_version for dataset in datasets}
    if len(versions) != 1:
        raise ValueError(
            "All training files must use the same observation representation; "
            f"found {sorted(versions)}. Rebuild old transition files before combining them."
        )
    fields = ("local_observations", "global_states", "actions", "team_rewards", "next_local_observations", "next_global_states", "dones", "frames")
    arrays = {field: np.concatenate([getattr(item, field) for item in datasets], axis=0) for field in fields}
    # A file/clip boundary must never join two matches in a temporal learner.
    offset = 0
    for dataset in datasets:
        if len(dataset.actions):
            arrays["dones"][offset + len(dataset.actions) - 1] = True
        offset += len(dataset.actions)
    arrays["observation_version"] = next(iter(versions))
    return arrays
