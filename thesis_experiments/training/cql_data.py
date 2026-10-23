"""Build chronological, time-limited d3rlpy episodes from explicit transitions."""

import numpy as np

from thesis_experiments.training.data import load_arrays
from thesis_experiments.simulation.gymnasium_replay import flatten_player_transitions, player_episode_indices


def build_cql_dataset(paths):
    from d3rlpy.constants import ActionSpace
    from d3rlpy.dataset import Episode, InfiniteBuffer, ReplayBuffer

    arrays = load_arrays(paths)
    batch = flatten_player_transitions(arrays)
    episodes = []
    for indices in player_episode_indices(arrays):
        observations = batch.observations[indices]
        next_observations = batch.next_observations[indices]
        if not np.allclose(next_observations[:-1], observations[1:], atol=1e-6):
            raise ValueError("Discontinuous player trajectory; split the clip at the discontinuity")
        # Append the final next state so d3rlpy retains the last transition and
        # bootstraps at this recording cutoff rather than inventing a terminal.
        episodes.append(Episode(
            observations=np.concatenate([observations, next_observations[-1:]]),
            actions=np.concatenate([batch.actions[indices], [0]]).astype(np.int64)[:, None],
            rewards=np.concatenate([batch.rewards[indices], [0]]).astype(np.float32)[:, None],
            terminated=False,
        ))
    return ReplayBuffer(InfiniteBuffer(), episodes=episodes,
                        action_space=ActionSpace.DISCRETE, action_size=19)
