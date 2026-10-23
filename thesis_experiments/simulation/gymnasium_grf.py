"""Gymnasium API over an isolated legacy GRF process, with joint 22-player control."""

import json
from pathlib import Path
import select
import subprocess
import tempfile
import time

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from thesis_experiments.training.data import load_arrays
from thesis_experiments.transitions.observations import (
    LOCAL_FEATURES,
    OBSERVATION_VERSION,
    local_observations,
    update_sticky_actions,
)


class FootballGRFEnv(gym.Env):
    metadata = {"render_modes": []}
    render_mode = None

    def __init__(self, data_paths, *, grf_python=None, max_episode_steps=128, progress_scale=1.0):
        root = Path(__file__).resolve().parents[2]
        self.grf_python = Path(grf_python or root / ".venv-grf/bin/python").absolute()
        if not self.grf_python.is_file():
            raise FileNotFoundError(f"GRF interpreter not found: {self.grf_python}")
        if max_episode_steps < 1:
            raise ValueError("max_episode_steps must be positive")
        self.arrays = load_arrays(data_paths)
        if not len(self.arrays["actions"]):
            raise ValueError("At least one initialization state is required")
        if self.arrays["observation_version"] != OBSERVATION_VERSION:
            raise ValueError(
                "Live GRF training requires the current observation representation "
                f"{OBSERVATION_VERSION!r}; rebuild the transition files."
            )
        self.max_episode_steps = max_episode_steps
        self.progress_scale = progress_scale
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(22, LOCAL_FEATURES), dtype=np.float32
        )
        self.action_space = spaces.MultiDiscrete([19] * 22)
        self._process = None
        self._stderr = None
        self._root = root
        self._needs_reset = True
        self._last_players = None
        self._last_ball = None
        self._previous_actions = np.zeros(22, dtype=np.int64)
        self._sticky_actions = np.zeros((22, 10), dtype=np.float32)

    def _request(self, payload):
        if self._process is None:
            self._stderr = tempfile.TemporaryFile(mode="w+")
            self._process = subprocess.Popen(
                [str(self.grf_python), "-u", "-m", "thesis_experiments.scripts.grf_worker"],
                cwd=self._root, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=self._stderr, text=True, bufsize=1,
            )
        self._process.stdin.write(json.dumps(payload) + "\n")
        self._process.stdin.flush()
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline:
            ready, _, _ = select.select([self._process.stdout], [], [], max(0, deadline - time.monotonic()))
            if not ready:
                break
            line = self._process.stdout.readline()
            if not line:
                self._stderr.seek(0)
                raise RuntimeError("GRF worker exited: " + self._stderr.read()[-4000:])
            if line.startswith("GRF_RESPONSE "):
                response = json.loads(line[len("GRF_RESPONSE "):])
                if "error" in response:
                    raise RuntimeError(response["error"])
                return response
        raise TimeoutError("GRF worker did not respond within 45 seconds")

    def _decode(self, response, applied_actions=None):
        players = np.asarray(response["players"], dtype=np.float32)
        ball = np.asarray(response["ball"], dtype=np.float32)
        if self._last_players is None:
            player_velocities = np.zeros_like(players)
            ball_velocity = np.zeros_like(ball)
        else:
            player_velocities = players - self._last_players
            ball_velocity = ball - self._last_ball
        if applied_actions is not None:
            self._previous_actions = np.asarray(applied_actions, dtype=np.int64).copy()
            self._sticky_actions = update_sticky_actions(
                self._sticky_actions, self._previous_actions
            )
        owner_team = int(response.get("ball_owned_team", -1))
        owner_within_team = int(response.get("ball_owned_player", -1))
        owner_global = (
            owner_within_team + (11 if owner_team == 1 else 0)
            if owner_team in (0, 1) and owner_within_team >= 0
            else -1
        )
        obs = local_observations(
            players,
            ball,
            player_velocities=player_velocities,
            ball_velocity=ball_velocity,
            previous_player_actions=self._previous_actions,
            sticky_actions=self._sticky_actions,
            ball_owned_team=owner_team,
            ball_owned_player=owner_global,
        )
        self._last_players = players.copy()
        self._last_ball = ball.copy()
        if not np.isfinite(obs).all():
            raise RuntimeError("GRF returned non-finite observations")
        return obs, {"positions": players, "ball": ball, "score": response["score"],
                     "team_rewards": np.asarray(response["team_rewards"], dtype=np.float32)}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        index = int((options or {}).get("transition_index", self.np_random.integers(len(self.arrays["actions"]))))
        state = self.arrays["global_states"][index]
        players = state[:44].reshape(22, 2)
        initial = {"home": players[:11].tolist(), "away": players[11:].tolist(),
                   "ball": state[44:46].tolist(), "game_duration": self.max_episode_steps + 10}
        self._last_players = None
        self._last_ball = None
        self._previous_actions.fill(0)
        self._sticky_actions.fill(0)
        response = self._request({"op": "reset", "initial": initial,
                                  "seed": int(self.np_random.integers(0, 2**31 - 1))})
        observation, info = self._decode(response)
        self._ball_x = float(info["ball"][0])
        self._steps = 0
        self._needs_reset = False
        return observation, info

    def step(self, actions):
        if self._needs_reset:
            raise RuntimeError("Call reset() before stepping or after an episode ends")
        if not self.action_space.contains(actions):
            raise ValueError("Expected 22 integer GRF actions in [0, 18]")
        applied_actions = np.asarray(actions, dtype=np.int64)
        response = self._request({"op": "step", "actions": applied_actions.tolist()})
        observation, info = self._decode(response, applied_actions=applied_actions)
        progress = (float(info["ball"][0]) - self._ball_x) * self.progress_scale
        self._ball_x = float(info["ball"][0])
        info["team_rewards"] += np.asarray([progress, -progress], dtype=np.float32)
        self._steps += 1
        terminated = bool(response["done"])
        truncated = self._steps >= self.max_episode_steps and not terminated
        self._needs_reset = terminated or truncated
        return observation, float(info["team_rewards"][0]), terminated, truncated, info

    def close(self):
        if self._process is not None:
            if self._process.poll() is None:
                try:
                    self._process.stdin.write('{"op":"close"}\n')
                    self._process.stdin.flush()
                    self._process.wait(timeout=5)
                except (BrokenPipeError, subprocess.TimeoutExpired):
                    self._process.terminate()
                    self._process.wait(timeout=5)
            self._process.stdin.close()
            self._process.stdout.close()
            self._stderr.close()
            self._process = None
        self._needs_reset = True
