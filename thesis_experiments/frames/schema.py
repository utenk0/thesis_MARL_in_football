"""Stage 4 provider-independent football frame schema."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

from thesis_experiments.events.schema import UnifiedEvent


@dataclass(frozen=True, slots=True)
class UnifiedPlayer:
    player_id: str
    x: float
    y: float
    velocity_x: float | None = None
    velocity_y: float | None = None

    def validate(self) -> None:
        if not self.player_id:
            raise ValueError("player_id is required.")
        if not (math.isfinite(self.x) and math.isfinite(self.y)):
            raise ValueError(f"Player {self.player_id} has non-finite coordinates.")
        if (self.velocity_x is None) != (self.velocity_y is None):
            raise ValueError("Player velocity components must both be present or missing.")
        if self.velocity_x is not None and not (
            math.isfinite(self.velocity_x) and math.isfinite(self.velocity_y)
        ):
            raise ValueError("Player velocity must be finite.")


@dataclass(frozen=True, slots=True)
class UnifiedBall:
    x: float
    y: float
    velocity_x: float | None = None
    velocity_y: float | None = None

    def validate(self) -> None:
        if not (math.isfinite(self.x) and math.isfinite(self.y)):
            raise ValueError("Ball coordinates must be finite.")
        if (self.velocity_x is None) != (self.velocity_y is None):
            raise ValueError("Ball velocity components must both be present or missing.")


@dataclass(slots=True)
class UnifiedFrame:
    source: str
    match_id: str
    frame_id: int
    timestamp: str
    left_team_id: str
    right_team_id: str
    left_players: list[UnifiedPlayer]
    right_players: list[UnifiedPlayer]
    ball: UnifiedBall | None
    events_at_frame: list[UnifiedEvent] = field(default_factory=list)
    valid: bool = False
    segment_id: int = 0
    raw: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.source or not self.match_id or not self.timestamp:
            raise ValueError("source, match_id, and timestamp are required.")
        if self.frame_id < 0:
            raise ValueError("frame_id must be non-negative.")
        if self.left_team_id == self.right_team_id:
            raise ValueError("Left and right team IDs must differ.")
        for player in self.left_players + self.right_players:
            player.validate()
        ids = [player.player_id for player in self.left_players + self.right_players]
        if len(ids) != len(set(ids)):
            raise ValueError("Player IDs must be unique within a frame.")
        if self.valid and (len(self.left_players) != 11 or len(self.right_players) != 11):
            raise ValueError("A valid frame must contain exactly 11 players per team.")
        if self.ball is not None:
            self.ball.validate()
        for event in self.events_at_frame:
            event.validate()
            if event.frame_id is not None and event.frame_id != self.frame_id:
                raise ValueError("events_at_frame contains an event joined to another frame.")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        result = asdict(self)
        result["events_at_frame"] = [event.to_dict() for event in self.events_at_frame]
        return result
