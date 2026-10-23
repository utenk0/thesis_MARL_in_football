from pathlib import Path

import numpy as np

from thesis_experiments.training.team_bc import (
    evaluate_team_bc,
    inspect_team_chunk,
    load_team_examples,
    train_team_bc,
)
from thesis_experiments.training.team_bc_diagnostics import confusion_metrics
from thesis_experiments.transitions.observations import LOCAL_FEATURES, OBSERVATION_VERSION
from thesis_experiments.transitions.schema import JointTransitionDataset


def _dataset(path: Path, *, match_id: str, actions: np.ndarray) -> Path:
    transitions = len(actions)
    observations = np.zeros((transitions, 22, LOCAL_FEATURES), dtype=np.float32)
    observations[..., 0] = np.arange(22, dtype=np.float32)
    observations[:, :, 28] = 1.0
    dataset = JointTransitionDataset(
        local_observations=observations,
        global_states=np.zeros((transitions, 46), dtype=np.float32),
        actions=actions.astype(np.int64),
        team_rewards=np.zeros((transitions, 2), dtype=np.float32),
        next_local_observations=observations.copy(),
        next_global_states=np.zeros((transitions, 46), dtype=np.float32),
        dones=np.asarray([False] * (transitions - 1) + [True]),
        frames=np.arange(transitions, dtype=np.int64),
        player_ids=np.asarray([f"p{i}" for i in range(22)]),
        team_ids=np.asarray(["focus"] * 11 + ["opponent"] * 11),
        source="test",
        match_id=match_id,
        observation_version=OBSERVATION_VERSION,
    )
    dataset.save(path)
    return path


def test_team_examples_select_only_focal_slots(tmp_path):
    actions = np.tile(np.arange(22) % 9, (3, 1))
    path = _dataset(tmp_path / "chunk.npz", match_id="m1", actions=actions)
    metadata = inspect_team_chunk(path, "focus")
    observations, selected_actions = load_team_examples(path, "focus")
    assert metadata["side"] == "left"
    assert metadata["player_indices"] == list(range(11))
    assert observations.shape == (33, LOCAL_FEATURES)
    np.testing.assert_array_equal(selected_actions, actions[:, :11].reshape(-1))


def test_streaming_team_bc_train_and_evaluate(tmp_path):
    actions = np.zeros((4, 22), dtype=np.int64)
    train = _dataset(tmp_path / "train.npz", match_id="train", actions=actions)
    test = _dataset(tmp_path / "test.npz", match_id="test", actions=actions)
    checkpoint = tmp_path / "bc.pt"
    training = train_team_bc(
        [train],
        checkpoint,
        team_id="focus",
        epochs=1,
        batch_size=16,
        hidden_size=8,
        observation_variant="temporal_context_no_slot",
    )
    evaluation = evaluate_team_bc(
        checkpoint,
        [test],
        team_id="focus",
        training_majority_action=0,
    )
    assert training["samples"] == 44
    assert training["retained_features"] == 31
    assert evaluation["overall"]["policy"]["samples"] == 44
    assert evaluation["matches"]["test"]["source"] == "test"


def test_confusion_metrics_reports_precision_recall_and_support():
    confusion = np.zeros((19, 19), dtype=np.int64)
    confusion[0, 0] = 8
    confusion[0, 1] = 2
    confusion[1, 0] = 1
    confusion[1, 1] = 4
    metrics = confusion_metrics(confusion)
    assert metrics["samples"] == 15
    assert np.isclose(metrics["action_accuracy"], 12 / 15)
    assert np.isclose(metrics["balanced_action_accuracy"], (0.8 + 0.8) / 2)
    assert np.isclose(metrics["per_action"][0]["precision"], 8 / 9)
    assert np.isclose(metrics["per_action"][0]["recall"], 8 / 10)
    assert metrics["per_action"][2]["recall"] is None
