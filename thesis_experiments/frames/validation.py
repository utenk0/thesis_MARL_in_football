"""Validation helpers for streams of unified frames."""

from __future__ import annotations

from collections.abc import Iterable

from thesis_experiments.frames.schema import UnifiedFrame


class FrameStreamValidator:
    def __init__(self) -> None:
        self.previous_by_match: dict[str, int] = {}
        self.roster_by_segment: dict[tuple[str, int], tuple[str, ...]] = {}
        self.count = self.valid = self.events = self.missing_ball = 0

    def add(self, frame: UnifiedFrame) -> None:
        frame.validate()
        self.count += 1
        self.valid += int(frame.valid)
        self.events += len(frame.events_at_frame)
        self.missing_ball += int(frame.ball is None)
        previous = self.previous_by_match.get(frame.match_id)
        if previous is not None and frame.frame_id <= previous:
            raise ValueError(f"Frame IDs are not monotonic in {frame.match_id}.")
        self.previous_by_match[frame.match_id] = frame.frame_id
        if frame.valid:
            roster = tuple(player.player_id for player in frame.left_players + frame.right_players)
            key = frame.match_id, frame.segment_id
            if key in self.roster_by_segment and self.roster_by_segment[key] != roster:
                raise ValueError(f"Player IDs changed inside segment {key}.")
            self.roster_by_segment[key] = roster

    def summary(self) -> dict[str, int]:
        return {"frames": self.count, "valid_frames": self.valid, "events_joined": self.events, "missing_ball_frames": self.missing_ball}


def validate_frame_stream(frames: Iterable[UnifiedFrame]) -> dict[str, int]:
    validator = FrameStreamValidator()
    for frame in frames:
        validator.add(frame)
    return validator.summary()
