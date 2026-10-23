"""Expose coupled GRF players as 22 rollout streams for a shared SB3 policy."""

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv

from thesis_experiments.transitions.observations import LOCAL_FEATURES


class SharedPlayerVecEnv(VecEnv):
    """All players step together; each receives its own team's reward.

    Streams are coupled agents in one match, not 22 independent matches.
    PPO uses local observations/values (parameter-sharing independent PPO).
    """

    def __init__(self, env):
        self.env = env
        super().__init__(
            22,
            spaces.Box(-np.inf, np.inf, shape=(LOCAL_FEATURES,), dtype=np.float32),
            spaces.Discrete(19),
        )

    def reset(self):
        obs, info = self.env.reset(seed=self._seeds[0], options=self._options[0])
        self.reset_infos = [dict(info) for _ in range(22)]
        self._reset_seeds()
        self._reset_options()
        return obs

    def step_async(self, actions):
        self._actions = np.asarray(actions).reshape(22)

    def step_wait(self):
        obs, _, terminated, truncated, info = self.env.step(self._actions)
        rewards = np.repeat(info["team_rewards"], 11)
        done = terminated or truncated
        infos = [{"score": info["score"], "player_index": i,
                  "TimeLimit.truncated": truncated and not terminated} for i in range(22)]
        if done:
            for i in range(22):
                infos[i]["terminal_observation"] = obs[i].copy()
            obs = self.reset()
        return obs, rewards, np.full(22, done, dtype=bool), infos

    def close(self):
        self.env.close()

    def get_attr(self, attr_name, indices=None):
        return [getattr(self.env, attr_name) for _ in self._get_indices(indices)]

    def set_attr(self, attr_name, value, indices=None):
        setattr(self.env, attr_name, value)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        result = getattr(self.env, method_name)(*method_args, **method_kwargs)
        return [result for _ in self._get_indices(indices)]

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False for _ in self._get_indices(indices)]
