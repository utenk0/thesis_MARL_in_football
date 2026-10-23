"""Provider-independent frame-to-GRF conversion tests."""

import numpy as np

from thesis_experiments.events.schema import UnifiedEvent
from thesis_experiments.frames.grf import convert_frame_to_grf, team_attacking_view
from thesis_experiments.frames.schema import UnifiedBall, UnifiedFrame, UnifiedPlayer


def test_metrica_frame_and_event_convert_to_grf_world() -> None:
    frame = UnifiedFrame(
        source="metrica", match_id="game", frame_id=1, timestamp="0.04",
        left_team_id="Home", right_team_id="Away",
        left_players=[UnifiedPlayer(f"H{i}", 0.25, 0.5) for i in range(11)],
        right_players=[UnifiedPlayer(f"A{i}", 0.75, 0.5) for i in range(11)],
        ball=UnifiedBall(0.5, 0.5), valid=True,
        events_at_frame=[UnifiedEvent(
            source="metrica", match_id="game", event_id="e", event_type="PASS",
            source_event_type="PASS", frame_id=1, x=0.75, y=0.5,
        )],
    )
    converted = convert_frame_to_grf(frame)
    np.testing.assert_allclose((converted.left_players[0].x, converted.left_players[0].y), (-0.5, 0))
    np.testing.assert_allclose((converted.ball.x, converted.ball.y), (0, 0))
    np.testing.assert_allclose((converted.events_at_frame[0].x, converted.events_at_frame[0].y), (0.5, 0))
    assert converted.raw["attacking_directions"] == {"Home": 1, "Away": -1}
    away_view = team_attacking_view(converted, "Away")
    np.testing.assert_allclose(away_view["players"]["H0"], (0.5, 0))


def test_dataset_a_center_maps_to_grf_center() -> None:
    frame = UnifiedFrame(
        source="dataset_a", match_id="game", frame_id=1, timestamp="time",
        left_team_id="L", right_team_id="R",
        left_players=[UnifiedPlayer(f"L{i}", -10, 0) for i in range(11)],
        right_players=[UnifiedPlayer(f"R{i}", 10, 0) for i in range(11)],
        ball=UnifiedBall(0, 0), valid=True,
    )
    converted = convert_frame_to_grf(frame)
    np.testing.assert_allclose((converted.ball.x, converted.ball.y), (0, 0))
