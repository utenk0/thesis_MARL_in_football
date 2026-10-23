"""Versioned, causal player-local observation representation.

The representation deliberately uses only information available at the current
decision time.  In particular, motion is measured from the previous frame and
the action features contain the *previous* action, never the action being
predicted for the current transition.
"""

from __future__ import annotations

import numpy as np

ACTION_COUNT = 19
SPATIAL_FEATURES = 24
STICKY_FEATURES = 10
LOCAL_FEATURES = 61

LEGACY_OBSERVATION_VERSION = "local_v1_spatial_24"
OBSERVATION_VERSION = "local_v2_temporal_61"

LOCAL_FEATURE_NAMES = (
    "player_x", "player_y",
    "ball_relative_x", "ball_relative_y",
    "ball_x", "ball_y",
    "side", "role_index",
    *(f"teammate_{rank}_{axis}" for rank in range(1, 5) for axis in ("relative_x", "relative_y")),
    *(f"opponent_{rank}_{axis}" for rank in range(1, 5) for axis in ("relative_x", "relative_y")),
    "player_velocity_x", "player_velocity_y",
    "ball_velocity_x", "ball_velocity_y",
    *(f"previous_action_{action}" for action in range(ACTION_COUNT)),
    *(f"sticky_direction_{action}" for action in range(1, 9)),
    "sticky_sprint", "sticky_dribble",
    "possession_own_team", "possession_opponent", "possession_loose",
    "player_owns_ball",
)

LOCAL_FEATURE_SLICES = {
    "spatial": slice(0, 24),
    "player_velocity": slice(24, 26),
    "ball_velocity": slice(26, 28),
    "previous_action": slice(28, 47),
    "sticky_actions": slice(47, 57),
    "possession_team": slice(57, 60),
    "player_owns_ball": slice(60, 61),
}

# Ablations keep the public 61-feature interface unchanged and zero excluded
# features.  This lets every checkpoint use the same dataset and GRF wrapper.
OBSERVATION_VARIANTS = (
    "spatial",
    "temporal_context",
    "temporal_context_no_slot",
    "full",
)

if len(LOCAL_FEATURE_NAMES) != LOCAL_FEATURES:  # pragma: no cover - import-time invariant
    raise RuntimeError("Observation feature names and size are inconsistent.")


def observation_feature_mask(variant: str = "full") -> np.ndarray:
    """Return the features retained by a named BC observation ablation.

    ``temporal_context`` adds causal player/ball velocity and possession to the
    spatial state, but deliberately excludes previous-action and sticky-action
    history. ``temporal_context_no_slot`` additionally removes feature 7,
    because Dataset A's stored player index is roster ordering rather than a
    stable tactical role. ``full`` retains all features, including history.
    """

    if variant not in OBSERVATION_VARIANTS:
        raise ValueError(
            f"Unknown observation variant {variant!r}; expected one of "
            f"{OBSERVATION_VARIANTS}."
        )
    mask = np.zeros(LOCAL_FEATURES, dtype=bool)
    mask[LOCAL_FEATURE_SLICES["spatial"]] = True
    if variant in {"temporal_context", "temporal_context_no_slot", "full"}:
        mask[LOCAL_FEATURE_SLICES["player_velocity"]] = True
        mask[LOCAL_FEATURE_SLICES["ball_velocity"]] = True
        mask[LOCAL_FEATURE_SLICES["possession_team"]] = True
        mask[LOCAL_FEATURE_SLICES["player_owns_ball"]] = True
    if variant == "full":
        mask[LOCAL_FEATURE_SLICES["previous_action"]] = True
        mask[LOCAL_FEATURE_SLICES["sticky_actions"]] = True
    if variant == "temporal_context_no_slot":
        mask[7] = False
    return mask


def apply_observation_variant(values: np.ndarray, variant: str = "full") -> np.ndarray:
    """Zero features excluded by ``variant`` without changing array shape."""

    values = np.asarray(values, dtype=np.float32)
    if values.shape[-1] != LOCAL_FEATURES:
        raise ValueError(
            f"Observation ablations require {LOCAL_FEATURES} features, "
            f"got {values.shape[-1]}."
        )
    return values * observation_feature_mask(variant).astype(np.float32)


def causal_deltas(values: np.ndarray) -> np.ndarray:
    """Return per-step motion using current minus previous values.

    The first frame has no history and therefore receives zero motion.  The
    values are displacement per recorded decision step in GRF coordinates.
    """

    values = np.asarray(values, dtype=np.float32)
    output = np.zeros_like(values, dtype=np.float32)
    if len(values) > 1:
        output[1:] = values[1:] - values[:-1]
    return output


def previous_actions(actions: np.ndarray, frame_count: int) -> np.ndarray:
    """Align transition actions as previous-action features for each frame."""

    actions = np.asarray(actions, dtype=np.int64)
    if actions.shape != (frame_count - 1, 22):
        raise ValueError(
            f"Expected actions with shape {(frame_count - 1, 22)}, got {actions.shape}."
        )
    output = np.zeros((frame_count, 22), dtype=np.int64)
    if frame_count > 1:
        output[1:] = actions
    return output


def sticky_action_history(actions: np.ndarray, frame_count: int) -> np.ndarray:
    """Reconstruct GRF's sticky direction/sprint/dribble state causally."""

    actions = np.asarray(actions, dtype=np.int64)
    if actions.shape != (frame_count - 1, 22):
        raise ValueError(
            f"Expected actions with shape {(frame_count - 1, 22)}, got {actions.shape}."
        )
    output = np.zeros((frame_count, 22, STICKY_FEATURES), dtype=np.float32)
    state = np.zeros((22, STICKY_FEATURES), dtype=np.float32)
    for frame_index in range(1, frame_count):
        state = update_sticky_actions(state, actions[frame_index - 1])
        output[frame_index] = state
    return output


def update_sticky_actions(state: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """Apply one joint GRF action to a copy of a sticky-action state."""

    result = np.asarray(state, dtype=np.float32).copy()
    actions = np.asarray(actions, dtype=np.int64)
    if result.shape != (22, STICKY_FEATURES) or actions.shape != (22,):
        raise ValueError("Sticky state and actions must have shapes [22,10] and [22].")
    if np.any((actions < 0) | (actions >= ACTION_COUNT)):
        raise ValueError("GRF actions must be in [0,18].")
    for player_index, action in enumerate(actions.tolist()):
        if 1 <= action <= 8:
            result[player_index, :8] = 0.0
            result[player_index, action - 1] = 1.0
        elif action == 14:
            result[player_index, :8] = 0.0
        elif action == 13:
            result[player_index, 8] = 1.0
        elif action == 15:
            result[player_index, 8] = 0.0
        elif action == 17:
            result[player_index, 9] = 1.0
        elif action == 18:
            result[player_index, 9] = 0.0
    return result


def infer_possession(
    players: np.ndarray,
    ball: np.ndarray,
    *,
    radius_meters: float = 2.0,
) -> tuple[int, int]:
    """Infer a conservative ball owner from current tracking positions.

    Returns ``(team, global_player_index)`` where team is 0 (left), 1
    (right), or -1 (loose).  GRF's anisotropic pitch coordinates are converted
    to metres before applying the ownership radius.
    """

    if radius_meters <= 0:
        raise ValueError("possession radius must be positive")
    players = np.asarray(players, dtype=np.float32)
    ball = np.asarray(ball, dtype=np.float32)
    if players.shape != (22, 2) or ball.shape != (2,):
        raise ValueError("players and ball must have shapes [22,2] and [2].")
    differences = players - ball
    differences_meters = differences * np.asarray([52.5, 68.0 / 0.84], dtype=np.float32)
    distances = np.linalg.norm(differences_meters, axis=1)
    owner = int(np.argmin(distances))
    if float(distances[owner]) > radius_meters:
        return -1, -1
    return (0 if owner < 11 else 1), owner


def local_observations(
    players: np.ndarray,
    ball: np.ndarray,
    *,
    player_velocities: np.ndarray | None = None,
    ball_velocity: np.ndarray | None = None,
    previous_player_actions: np.ndarray | None = None,
    sticky_actions: np.ndarray | None = None,
    ball_owned_team: int | None = None,
    ball_owned_player: int | None = None,
    possession_radius_meters: float = 2.0,
) -> np.ndarray:
    """Build one 61-feature local observation for each of 22 players."""

    players = np.asarray(players, dtype=np.float32)
    ball = np.asarray(ball, dtype=np.float32)
    if players.shape != (22, 2) or ball.shape != (2,):
        raise ValueError("players and ball must have shapes [22,2] and [2].")
    velocities = (
        np.zeros_like(players) if player_velocities is None
        else np.asarray(player_velocities, dtype=np.float32)
    )
    ball_delta = (
        np.zeros(2, dtype=np.float32) if ball_velocity is None
        else np.asarray(ball_velocity, dtype=np.float32)
    )
    previous = (
        np.zeros(22, dtype=np.int64) if previous_player_actions is None
        else np.asarray(previous_player_actions, dtype=np.int64)
    )
    sticky = (
        np.zeros((22, STICKY_FEATURES), dtype=np.float32) if sticky_actions is None
        else np.asarray(sticky_actions, dtype=np.float32)
    )
    if velocities.shape != (22, 2) or ball_delta.shape != (2,):
        raise ValueError("velocity inputs must have shapes [22,2] and [2].")
    if previous.shape != (22,) or sticky.shape != (22, STICKY_FEATURES):
        raise ValueError("previous actions and sticky state must have shapes [22] and [22,10].")
    if np.any((previous < 0) | (previous >= ACTION_COUNT)):
        raise ValueError("Previous GRF actions must be in [0,18].")
    if ball_owned_team is None or ball_owned_player is None:
        ball_owned_team, ball_owned_player = infer_possession(
            players, ball, radius_meters=possession_radius_meters
        )
    if ball_owned_team not in (-1, 0, 1):
        raise ValueError("ball_owned_team must be -1, 0, or 1.")
    if ball_owned_player != -1 and not 0 <= int(ball_owned_player) < 22:
        raise ValueError("ball_owned_player must be -1 or a global player index in [0,21].")

    output = np.zeros((22, LOCAL_FEATURES), dtype=np.float32)
    previous_one_hot = np.eye(ACTION_COUNT, dtype=np.float32)[previous]
    for index, own in enumerate(players):
        same_start = 0 if index < 11 else 11
        own_team = 0 if index < 11 else 1
        teammates = players[same_start:same_start + 11]
        opponents = players[11:22] if index < 11 else players[0:11]
        own_in_team = index - same_start
        teammate_rel = np.delete(teammates - own, own_in_team, axis=0)
        teammate_rel = teammate_rel[np.argsort(np.linalg.norm(teammate_rel, axis=1))[:4]]
        opponent_rel = opponents - own
        opponent_rel = opponent_rel[np.argsort(np.linalg.norm(opponent_rel, axis=1))[:4]]
        side = 1.0 if index < 11 else -1.0
        role_index = own_in_team / 10.0
        if ball_owned_team == -1:
            possession = (0.0, 0.0, 1.0)
        elif ball_owned_team == own_team:
            possession = (1.0, 0.0, 0.0)
        else:
            possession = (0.0, 1.0, 0.0)
        output[index] = np.concatenate([
            own,
            ball - own,
            ball,
            [side, role_index],
            teammate_rel.reshape(-1),
            opponent_rel.reshape(-1),
            velocities[index],
            ball_delta,
            previous_one_hot[index],
            sticky[index],
            possession,
            [float(index == ball_owned_player)],
        ])
    if not np.isfinite(output).all():
        raise ValueError("Local observations must be finite.")
    return output
