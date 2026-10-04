"""Build shared-actor/centralized-critic inputs from a validated audit."""

from __future__ import annotations

import numpy as np

from thesis_experiments.transitions.observations import (
    LOCAL_FEATURES,
    OBSERVATION_VERSION,
    causal_deltas,
    local_observations,
    previous_actions,
    sticky_action_history,
)
from thesis_experiments.transitions.schema import JointTransitionDataset

GLOBAL_FEATURES = 46


def build_joint_transitions(
    audit,
    *,
    source: str,
    progress_scale: float = 1.0,
    goal_reward: float = 1.0,
    possession_radius_meters: float = 2.0,
) -> JointTransitionDataset:
    left = np.asarray(audit.home_grf_xy, dtype=np.float32)
    right = np.asarray(audit.away_grf_xy, dtype=np.float32)
    ball = np.asarray(audit.ball_grf_xy, dtype=np.float32)
    if not all(np.isfinite(value).all() for value in (left, right, ball)):
        raise ValueError("Training clips require finite positions for all 22 players and the ball")
    players = np.concatenate([left, right], axis=1)
    actions = np.asarray(audit.joint_actions, dtype=np.int64)
    if actions.shape != (len(players) - 1, 22):
        raise ValueError(
            f"Joint actions must have shape {(len(players) - 1, 22)}, got {actions.shape}."
        )
    player_velocities = causal_deltas(players)
    ball_velocities = causal_deltas(ball)
    previous = previous_actions(actions, len(players))
    sticky = sticky_action_history(actions, len(players))
    locals_all = np.stack([
        local_observations(
            frame_players,
            frame_ball,
            player_velocities=frame_player_velocities,
            ball_velocity=frame_ball_velocity,
            previous_player_actions=frame_previous,
            sticky_actions=frame_sticky,
            possession_radius_meters=possession_radius_meters,
        )
        for frame_players, frame_ball, frame_player_velocities, frame_ball_velocity, frame_previous, frame_sticky
        in zip(players, ball, player_velocities, ball_velocities, previous, sticky)
    ])
    globals_all = np.concatenate([players.reshape(len(players), -1), ball], axis=1).astype(np.float32)
    rewards = np.zeros((len(players) - 1, 2), dtype=np.float32)
    progress = np.diff(ball[:, 0]) * progress_scale
    rewards[:, 0], rewards[:, 1] = progress, -progress
    _apply_goal_rewards(audit, rewards, goal_reward)
    dones = np.zeros(len(rewards), dtype=bool)
    if len(dones):
        dones[-1] = True
    left_ids = list(getattr(audit, "home_player_ids", getattr(audit, "left_player_ids", [])))
    right_ids = list(getattr(audit, "away_player_ids", getattr(audit, "right_player_ids", [])))
    left_team = str(getattr(audit, "left_team_id", "Home"))
    right_team = str(getattr(audit, "right_team_id", "Away"))
    result = JointTransitionDataset(
        local_observations=locals_all[:-1], global_states=globals_all[:-1],
        actions=actions, team_rewards=rewards,
        next_local_observations=locals_all[1:], next_global_states=globals_all[1:],
        dones=dones, frames=np.asarray(audit.frames[:-1], dtype=np.int64),
        player_ids=np.asarray(left_ids + right_ids),
        team_ids=np.asarray([left_team] * 11 + [right_team] * 11),
        source=source, match_id=str(getattr(audit, "match_id", "metrica_sample")),
        observation_version=OBSERVATION_VERSION,
    )
    result.validate()
    return result


def _local_observations(players: np.ndarray, ball: np.ndarray, **kwargs) -> np.ndarray:
    """Backward-compatible import path for the versioned observation builder."""

    return local_observations(players, ball, **kwargs)


def _apply_goal_rewards(audit, rewards: np.ndarray, magnitude: float) -> None:
    frame_index = {int(frame): index for index, frame in enumerate(audit.frames[:-1])}
    for event in audit.events:
        event_type = str(event.get("event_type", event.get("Type", ""))).lower()
        subtype = str(event.get("Subtype", "")).upper().replace("-", " ").split()
        scored = (event.get("outcome") == "SuccessfulShot" or event_type == "goal"
                  or (event_type == "shot" and "GOAL" in subtype))
        if not scored:
            continue
        frame = event.get("Start Frame", event.get("anchor_frame"))
        if frame is None or int(frame) not in frame_index:
            continue
        team = str(event.get("Team", (event.get("details") or {}).get("Team", "")))
        left_team = str(getattr(audit, "left_team_id", "Home"))
        left_scored = team in {"Home", left_team}
        rewards[frame_index[int(frame)]] += [magnitude, -magnitude] if left_scored else [-magnitude, magnitude]
