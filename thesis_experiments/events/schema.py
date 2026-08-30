"""Unified event record shared by Metrica, Dataset A, and training code."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


EVENT_TYPES = frozenset({
    "PASS", "SHOT", "GOAL", "CARRY", "DRIBBLE", "TACKLE", "INTERCEPTION",
    "BALL_RECOVERY", "BALL_LOSS", "BALL_CLAIM", "CHALLENGE", "CORNER",
    "FREE_KICK", "PENALTY", "GOAL_KICK", "THROW_IN", "KICK_OFF", "FOUL", "OFFSIDE",
    "PERIOD_START", "PERIOD_END", "OTHER",
})


@dataclass(slots=True)
class UnifiedEvent:
    source: str
    match_id: str
    event_id: str
    event_type: str
    source_event_type: str
    frame: int | None = None
    end_frame: int | None = None
    period: str | None = None
    timestamp_seconds: float | None = None
    timestamp: str | None = None
    subtype: str | None = None
    team_id: str | None = None
    player_id: str | None = None
    recipient_id: str | None = None
    start_grf: tuple[float, float] | None = None
    end_grf: tuple[float, float] | None = None
    outcome: str | None = None
    grf_action: int | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.event_type not in EVENT_TYPES:
            raise ValueError(f"Unsupported unified event type: {self.event_type}.")
        if self.grf_action is not None and not 0 <= self.grf_action <= 18:
            raise ValueError("GRF action must be in [0,18].")
        for name in ("start_grf", "end_grf"):
            position = getattr(self, name)
            if position is not None and not (-1.0 <= position[0] <= 1.0 and -0.42 <= position[1] <= 0.42):
                raise ValueError(f"{name} is outside the GRF pitch: {position}.")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)
