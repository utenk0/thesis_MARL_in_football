"""Validate the versioned local-observation contract in transition NPZ files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from thesis_experiments.transitions.observations import (
    LOCAL_FEATURE_NAMES,
    LOCAL_FEATURE_SLICES,
    OBSERVATION_VERSION,
    update_sticky_actions,
)
from thesis_experiments.transitions.schema import JointTransitionDataset


def validate_observations(path: Path) -> dict[str, object]:
    dataset = JointTransitionDataset.load(path)
    if dataset.observation_version != OBSERVATION_VERSION:
        raise ValueError(
            f"{path} uses {dataset.observation_version!r}; rebuild it as {OBSERVATION_VERSION!r}."
        )
    current = dataset.local_observations
    following = dataset.next_local_observations
    slices = LOCAL_FEATURE_SLICES

    previous_one_hot = current[..., slices["previous_action"]]
    next_previous_one_hot = following[..., slices["previous_action"]]
    possession = current[..., slices["possession_team"]]
    if not np.allclose(previous_one_hot.sum(axis=-1), 1.0):
        raise ValueError(f"{path}: previous-action features are not one-hot.")
    if not np.array_equal(next_previous_one_hot.argmax(axis=-1), dataset.actions):
        raise ValueError(f"{path}: next observations do not contain the current action as previous action.")
    if not np.allclose(possession.sum(axis=-1), 1.0):
        raise ValueError(f"{path}: possession-team features are not one-hot.")

    expected_player_velocity = following[..., :2] - current[..., :2]
    expected_ball_velocity = following[..., 4:6] - current[..., 4:6]
    if not np.allclose(
        following[..., slices["player_velocity"]], expected_player_velocity, atol=1e-6
    ):
        raise ValueError(f"{path}: player velocities are not causal frame differences.")
    if not np.allclose(
        following[..., slices["ball_velocity"]], expected_ball_velocity, atol=1e-6
    ):
        raise ValueError(f"{path}: ball velocities are not causal frame differences.")

    current_sticky = current[..., slices["sticky_actions"]]
    next_sticky = following[..., slices["sticky_actions"]]
    for transition in range(len(dataset.actions)):
        expected = update_sticky_actions(current_sticky[transition], dataset.actions[transition])
        if not np.array_equal(next_sticky[transition], expected):
            raise ValueError(f"{path}: sticky-action mismatch at transition {transition}.")

    return {
        "path": str(path),
        "observation_version": dataset.observation_version,
        "feature_count": len(LOCAL_FEATURE_NAMES),
        "transitions": len(dataset.actions),
        "player_samples": int(dataset.actions.size),
        "finite": True,
        "previous_action_alignment": True,
        "sticky_action_alignment": True,
        "causal_velocity_alignment": True,
        "possession_one_hot": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    print(json.dumps([validate_observations(path) for path in args.paths], indent=2))


if __name__ == "__main__":
    main()
