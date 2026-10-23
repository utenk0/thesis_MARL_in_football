"""Stage 4 frame schema and stream validation tests."""

import json

import pytest

from thesis_experiments.events.schema import UnifiedEvent
from thesis_experiments.frames.parsers import parse_metrica_epts_frames
from thesis_experiments.frames.schema import UnifiedBall, UnifiedFrame, UnifiedPlayer
from thesis_experiments.frames.validation import validate_frame_stream


def players(prefix: str, count: int = 11) -> list[UnifiedPlayer]:
    return [UnifiedPlayer(f"{prefix}{index}", index / 20, 0.5) for index in range(count)]


def frame(frame_id: int, *, valid: bool = True, ball: bool = True) -> UnifiedFrame:
    event = UnifiedEvent(
        source="test", match_id="match", event_id=str(frame_id), event_type="PASS",
        source_event_type="pass", frame_id=frame_id, x=0.5, y=0.5,
    )
    return UnifiedFrame(
        source="test", match_id="match", frame_id=frame_id, timestamp=str(frame_id),
        left_team_id="left", right_team_id="right", left_players=players("L"),
        right_players=players("R"), ball=UnifiedBall(0.5, 0.5) if ball else None,
        events_at_frame=[event], valid=valid,
    )


def test_valid_frame_stream_contract() -> None:
    summary = validate_frame_stream([frame(1), frame(2, ball=False)])
    assert summary == {"frames": 2, "valid_frames": 2, "events_joined": 2, "missing_ball_frames": 1}


def test_valid_frame_requires_eleven_players_per_team() -> None:
    invalid = frame(1)
    invalid.left_players.pop()
    with pytest.raises(ValueError, match="exactly 11"):
        invalid.validate()


def test_frame_ids_must_be_monotonic() -> None:
    with pytest.raises(ValueError, match="not monotonic"):
        validate_frame_stream([frame(2), frame(1)])


def test_epts_slots_resolve_to_real_players_across_substitution(tmp_path) -> None:
    game_dir = tmp_path / "Sample_Game_3"
    game_dir.mkdir()
    left = [f"A{index}" for index in range(1, 13)]
    right = [f"B{index}" for index in range(1, 12)]
    all_players = left + right
    player_xml = "".join(
        f'<Player id="{player}" teamId="{player[0]}"/>'
        for player in all_players
    )
    channel_xml = "".join(
        f'<PlayerChannel channelId="x" id="{player}_x" playerId="{player}"/>'
        for player in all_players
    )
    first_roster = left[:11] + right
    second_roster = [left[11], *left[1:11], *right]

    def references(roster):
        return "".join(f'<PlayerChannelRef playerChannelId="{player}_x"/>' for player in roster)

    metadata = (
        '<Metadata><FrameRate>25</FrameRate><Teams><Team id="A"/><Team id="B"/></Teams>'
        f'<Players>{player_xml}</Players><Channels>{channel_xml}</Channels>'
        '<DataFormatSpecifications>'
        f'<DataFormatSpecification startFrame="1" endFrame="1">{references(first_roster)}</DataFormatSpecification>'
        f'<DataFormatSpecification startFrame="2" endFrame="2">{references(second_roster)}</DataFormatSpecification>'
        '</DataFormatSpecifications></Metadata>'
    )
    (game_dir / "Sample_Game_3_metadata.xml").write_text(metadata, encoding="utf-8")
    positions = ";".join("0.1,0.2" for _ in range(22))
    (game_dir / "Sample_Game_3_tracking.txt").write_text(
        f"1:{positions}:0.5,0.5\n2:{positions}:0.5,0.5\n", encoding="utf-8"
    )
    (game_dir / "Sample_Game_3_events.json").write_text(json.dumps({"data": [{
        "index": 1,
        "type": {"name": "PASS"},
        "start": {"frame": 2, "time": 0.08, "x": 0.5, "y": 0.5},
        "end": {"frame": 2, "x": 0.6, "y": 0.5},
        "team": {"id": "A"},
        "from": {"id": "A12"},
        "to": {"id": "A2"},
        "period": "1",
        "subtypes": {},
    }]}), encoding="utf-8")

    frames = list(parse_metrica_epts_frames(game_dir))

    assert [player.player_id for player in frames[0].left_players] == left[:11]
    assert [player.player_id for player in frames[1].left_players] == second_roster[:11]
    assert [item.segment_id for item in frames] == [0, 1]
    assert frames[1].events_at_frame[0].player_id == "A12"
    assert frames[1].events_at_frame[0].player_id in {
        player.player_id for player in frames[1].left_players
    }
