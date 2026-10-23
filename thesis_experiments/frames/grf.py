"""Convert unified provider-native frames into shared GRF world coordinates."""

from __future__ import annotations

from dataclasses import replace
from statistics import median

from thesis_experiments.datasets.coordinates import (
    provider_to_grf,
    provider_velocity_to_grf,
    to_attacking_frame,
)
from thesis_experiments.events.schema import UnifiedEvent
from thesis_experiments.frames.schema import UnifiedBall, UnifiedFrame, UnifiedPlayer


def infer_attacking_directions(frame: UnifiedFrame) -> dict[str, int]:
    """Infer team directions from the teams' median provider/world x positions.

    The team occupying the lower-x half is treated as attacking positive x.
    This is recorded per frame and naturally handles half-time side changes.
    """
    if not frame.left_players or not frame.right_players:
        return {frame.left_team_id: 1, frame.right_team_id: -1}
    left_x = median(player.x for player in frame.left_players)
    right_x = median(player.x for player in frame.right_players)
    if left_x <= right_x:
        return {frame.left_team_id: 1, frame.right_team_id: -1}
    return {frame.left_team_id: -1, frame.right_team_id: 1}


def _player_to_grf(source: str, player: UnifiedPlayer) -> UnifiedPlayer:
    position = provider_to_grf(source, player.x, player.y)
    velocity = None
    if player.velocity_x is not None:
        velocity = provider_velocity_to_grf(source, player.velocity_x, player.velocity_y)
    return UnifiedPlayer(
        player_id=player.player_id,
        x=float(position[0]),
        y=float(position[1]),
        velocity_x=float(velocity[0]) if velocity is not None else None,
        velocity_y=float(velocity[1]) if velocity is not None else None,
    )


def _ball_to_grf(source: str, ball: UnifiedBall | None) -> UnifiedBall | None:
    if ball is None:
        return None
    position = provider_to_grf(source, ball.x, ball.y)
    velocity = None
    if ball.velocity_x is not None:
        velocity = provider_velocity_to_grf(source, ball.velocity_x, ball.velocity_y)
    return UnifiedBall(
        x=float(position[0]),
        y=float(position[1]),
        velocity_x=float(velocity[0]) if velocity is not None else None,
        velocity_y=float(velocity[1]) if velocity is not None else None,
    )


def _event_to_grf(event: UnifiedEvent) -> UnifiedEvent:
    position = event.start_grf
    if position is None and event.x is not None:
        if event.source == "dataset_a":
            position = provider_to_grf(event.source, event.x - 52.5, event.y - 34.0)
        else:
            position = provider_to_grf(event.source, event.x, event.y)
    return replace(
        event,
        x=float(position[0]) if position is not None else None,
        y=float(position[1]) if position is not None else None,
        start_grf=(float(position[0]), float(position[1])) if position is not None else None,
    )


def convert_frame_to_grf(frame: UnifiedFrame) -> UnifiedFrame:
    """Return a frame in shared GRF world space, preserving identity and joins."""
    directions = infer_attacking_directions(frame)
    raw = dict(frame.raw)
    raw.update({
        "coordinate_system": "grf_world",
        "source_coordinate_system": frame.source,
        "attacking_directions": directions,
    })
    converted = UnifiedFrame(
        source=frame.source,
        match_id=frame.match_id,
        frame_id=frame.frame_id,
        timestamp=frame.timestamp,
        left_team_id=frame.left_team_id,
        right_team_id=frame.right_team_id,
        left_players=[_player_to_grf(frame.source, player) for player in frame.left_players],
        right_players=[_player_to_grf(frame.source, player) for player in frame.right_players],
        ball=_ball_to_grf(frame.source, frame.ball),
        events_at_frame=[_event_to_grf(event) for event in frame.events_at_frame],
        valid=frame.valid,
        segment_id=frame.segment_id,
        raw=raw,
    )
    converted.validate()
    return converted


def team_attacking_view(frame: UnifiedFrame, team_id: str) -> dict[str, object]:
    """Return all world positions rotated into ``team_id``'s attacking frame."""
    if frame.raw.get("coordinate_system") != "grf_world":
        raise ValueError("team_attacking_view requires a converted GRF frame.")
    directions = frame.raw.get("attacking_directions") or {}
    if team_id not in directions:
        raise ValueError(f"Unknown team ID: {team_id}.")
    positive = int(directions[team_id]) == 1
    players = frame.left_players + frame.right_players
    return {
        "players": {
            player.player_id: to_attacking_frame((player.x, player.y), attacks_positive_x=positive).tolist()
            for player in players
        },
        "ball": (
            to_attacking_frame((frame.ball.x, frame.ball.y), attacks_positive_x=positive).tolist()
            if frame.ball is not None else None
        ),
    }
