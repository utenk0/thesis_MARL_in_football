"""Provider event mappings to the unified vocabulary."""

import numpy as np

from thesis_experiments.events.converters import (
    dataset_a_event_to_unified,
    metrica_event_to_unified,
)


def test_metrica_pass_event() -> None:
    event = metrica_event_to_unified({
        "Type": "PASS", "Subtype": "HEAD", "Start Frame": 10,
        "Start X": .25, "Start Y": .5, "End X": .5, "End Y": .75,
        "Team": "Home", "From": "Player1", "To": "Player2",
    }, match_id="game")
    assert event.event_type == "PASS"
    assert event.grf_action == 10
    np.testing.assert_allclose(event.start_grf, [-.5, 0])
    np.testing.assert_allclose(event.end_grf, [0, .21])


def test_metrica_goal_is_not_only_a_shot() -> None:
    event = metrica_event_to_unified({"Type": "SHOT", "Subtype": "GOAL", "Start X": .8, "Start Y": .5}, match_id="game")
    assert event.event_type == "GOAL"
    assert event.outcome == "success"
    assert event.grf_action == 12


def test_dataset_a_event_midfield_position_and_actor() -> None:
    event = dataset_a_event_to_unified({
        "event_id": "1", "event_type": "Play", "anchor_frame": 100,
        "x_position": "52.5", "y_position": "34",
        "details": {"Team": "T1", "Player": "P1", "Recipient": "P2", "Height": "low", "Evaluation": "successful"},
    }, match_id="match")
    assert event.event_type == "PASS"
    assert event.player_id == "P1" and event.recipient_id == "P2"
    assert event.grf_action == 11
    np.testing.assert_allclose(event.start_grf, [0, 0])


def test_dataset_a_kickoff_side_metadata() -> None:
    event = dataset_a_event_to_unified({
        "event_id": "2", "event_type": "KickOff", "anchor_frame": 56,
        "x_position": "52.5", "y_position": "34",
        "details": {"TeamLeft": "left", "TeamRight": "right", "GameSection": "firstHalf"},
    }, match_id="match")
    assert event.event_type == "KICK_OFF"
    assert event.team_id == "left"
    assert event.period == "firstHalf"


def test_unmapped_metrica_event_does_not_override_movement() -> None:
    event = metrica_event_to_unified({"Type": "FAULT RECEIVED", "Start X": .5, "Start Y": .5}, match_id="game")
    assert event.event_type == "FOUL"
    assert event.grf_action is None
