"""Approximate mappings from football event taxonomies to GRF actions."""

from __future__ import annotations

GRF_ACTIONS = {
    "idle": 0,
    "left": 1,
    "top_left": 2,
    "top": 3,
    "top_right": 4,
    "right": 5,
    "bottom_right": 6,
    "bottom": 7,
    "bottom_left": 8,
    "long_pass": 9,
    "high_pass": 10,
    "short_pass": 11,
    "shot": 12,
    "sprint": 13,
    "release_direction": 14,
    "release_sprint": 15,
    "sliding": 16,
    "dribble": 17,
    "release_dribble": 18,
}


def metrica_event_to_grf_action(event_type: str, subtype: str | float | None) -> int:
    """Map a Metrica event label to a single-agent GRF action id.

    This is intentionally conservative and should be treated as an explicit
    thesis assumption, not ground truth motor control.
    """

    event = str(event_type or "").upper()
    subevent = "" if subtype is None else str(subtype).upper()

    if event == "SHOT":
        return GRF_ACTIONS["shot"]
    if event == "PASS":
        if "CROSS" in subevent or "HEAD" in subevent:
            return GRF_ACTIONS["high_pass"]
        if "DEEP" in subevent or "GOAL KICK" in subevent:
            return GRF_ACTIONS["long_pass"]
        return GRF_ACTIONS["short_pass"]
    if event == "SET PIECE":
        if "CORNER" in subevent or "FREE KICK" in subevent:
            return GRF_ACTIONS["high_pass"]
        if "GOAL KICK" in subevent:
            return GRF_ACTIONS["long_pass"]
        if "THROW" in subevent or "KICK OFF" in subevent:
            return GRF_ACTIONS["short_pass"]
    if event == "CHALLENGE":
        return GRF_ACTIONS["sliding"] if "TACKLE" in subevent else GRF_ACTIONS["idle"]
    if event == "BALL LOST":
        return GRF_ACTIONS["release_dribble"]
    if event == "RECOVERY":
        return GRF_ACTIONS["idle"]
    return GRF_ACTIONS["idle"]


def dataset_a_event_to_grf_action(event_type: str, details: dict[str, str] | None) -> int | None:
    """Map Dataset A event labels to the single-agent GRF discrete action set.

    The source data describes football events, not low-level GRF button presses.
    Returning ``None`` means the event is skipped for policy imitation.
    """

    event = str(event_type or "")
    details = details or {}

    if event == "Play":
        height = str(details.get("Height", "")).lower()
        distance = str(details.get("Distance", "")).lower()
        flat_cross = str(details.get("FlatCross", "")).lower() == "true"
        if flat_cross or height == "high":
            return GRF_ACTIONS["high_pass"]
        if distance == "long":
            return GRF_ACTIONS["long_pass"]
        return GRF_ACTIONS["short_pass"]
    if event == "ShotAtGoal":
        return GRF_ACTIONS["shot"]
    if event in {"CornerKick", "FreeKick"}:
        return GRF_ACTIONS["high_pass"]
    if event == "GoalKick":
        return GRF_ACTIONS["long_pass"]
    if event in {"ThrowIn", "KickOff"}:
        return GRF_ACTIONS["short_pass"]
    if event == "TacklingGame":
        return GRF_ACTIONS["sliding"]
    if event == "BallClaiming":
        return GRF_ACTIONS["idle"]
    if event == "OtherBallAction":
        return GRF_ACTIONS["dribble"]
    return None
