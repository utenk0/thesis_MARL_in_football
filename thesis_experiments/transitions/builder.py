"""Build shared-actor/centralized-critic inputs from a validated audit."""

from __future__ import annotations

import numpy as np

from thesis_experiments.transitions.schema import JointTransitionDataset

LOCAL_FEATURES = 24
GLOBAL_FEATURES = 46


def build_joint_transitions(audit, *, source: str, progress_scale: float = 1.0, goal_reward: float = 1.0) -> JointTransitionDataset:
    left = np.nan_to_num(np.asarray(audit.home_grf_xy, dtype=np.float32))
    right = np.nan_to_num(np.asarray(audit.away_grf_xy, dtype=np.float32))
    ball = np.nan_to_num(np.asarray(audit.ball_grf_xy, dtype=np.float32))
    players = np.concatenate([left, right], axis=1)
    locals_all = np.stack([_local_observations(p, b) for p, b in zip(players, ball)])
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
        actions=np.asarray(audit.joint_actions, dtype=np.int64), team_rewards=rewards,
        next_local_observations=locals_all[1:], next_global_states=globals_all[1:],
        dones=dones, frames=np.asarray(audit.frames[:-1], dtype=np.int64),
        player_ids=np.asarray(left_ids + right_ids),
        team_ids=np.asarray([left_team] * 11 + [right_team] * 11),
        source=source, match_id=str(getattr(audit, "match_id", "metrica_sample")),
    )
    result.validate()
    return result


def _local_observations(players: np.ndarray, ball: np.ndarray) -> np.ndarray:
    output = np.zeros((22, LOCAL_FEATURES), dtype=np.float32)
    for index, own in enumerate(players):
        same_start = 0 if index < 11 else 11
        teammates = players[same_start:same_start + 11]
        opponents = players[11:22] if index < 11 else players[0:11]
        own_in_team = index - same_start
        teammate_rel = np.delete(teammates - own, own_in_team, axis=0)
        teammate_rel = teammate_rel[np.argsort(np.linalg.norm(teammate_rel, axis=1))[:4]]
        opponent_rel = opponents - own
        opponent_rel = opponent_rel[np.argsort(np.linalg.norm(opponent_rel, axis=1))[:4]]
        side = 1.0 if index < 11 else -1.0
        role_index = own_in_team / 10.0
        output[index] = np.concatenate([
            own, ball - own, ball, [side, role_index], teammate_rel.reshape(-1), opponent_rel.reshape(-1)
        ])
    return output


def _apply_goal_rewards(audit, rewards: np.ndarray, magnitude: float) -> None:
    frame_index = {int(frame): index for index, frame in enumerate(audit.frames[:-1])}
    for event in audit.events:
        label = f"{event.get('Type', '')} {event.get('Subtype', '')} {event.get('event_type', '')}".lower()
        if "goal" not in label or "goal kick" in label:
            continue
        frame = event.get("Start Frame", event.get("anchor_frame"))
        if frame is None or int(frame) not in frame_index:
            continue
        team = str(event.get("Team", (event.get("details") or {}).get("Team", "")))
        left_team = str(getattr(audit, "left_team_id", "Home"))
        left_scored = team in {"Home", left_team}
        rewards[frame_index[int(frame)]] += [magnitude, -magnitude] if left_scored else [-magnitude, magnitude]
