"""Regression tests for temporal learning, saved policies and simulator control."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from thesis_experiments.tests.test_method_framework import _sample_dataset
from thesis_experiments.transitions.schema import JointTransitionDataset
from thesis_experiments.simulation.gymnasium_replay import FootballReplayEnv
from thesis_experiments.training.decision_transformer import DecisionTransformer, TrajectoryWindows
from thesis_experiments.methods import MethodConfig, build_method
from thesis_experiments.policies.loading import load_method_policy
from thesis_experiments.transitions.observations import LOCAL_FEATURES, OBSERVATION_VERSION


def coherent_dataset(path):
    _sample_dataset(path)
    dataset = JointTransitionDataset.load(path)
    dataset.next_local_observations[:-1] = dataset.local_observations[1:]
    dataset.next_local_observations[-1, :, 0] = 0.03
    dataset.team_rewards[:] = [1, -1]
    dataset.save(path)
    return path


def test_replay_preserves_player_and_requires_reset(tmp_path):
    env = FootballReplayEnv([coherent_dataset(tmp_path / "data.npz")])
    observation, info = env.reset(options={"episode_index": 12})
    for step in range(3):
        assert info["player_index"] == 12
        following, _, terminated, truncated, info = env.step(0)
        assert not terminated
        assert truncated == (step == 2)
        if step < 2:
            np.testing.assert_allclose(following, env.batch.observations[env._index])
    with pytest.raises(RuntimeError, match="reset"):
        env.step(0)
    env.reset()
    with pytest.raises(ValueError):
        env.step(19)


def test_cql_keeps_player_next_states_and_last_transition(tmp_path):
    pytest.importorskip("d3rlpy")
    from thesis_experiments.training.cql_data import build_cql_dataset
    data = build_cql_dataset([coherent_dataset(tmp_path / "data.npz")])
    assert data.transition_count == 66
    for episode in data.episodes:
        assert not episode.terminated
        for step in range(3):
            transition = data.transition_picker(episode, step)
            assert transition.terminal == 0
            assert transition.observation[1] == transition.next_observation[1]
            assert transition.next_observation[0] == pytest.approx((step + 1) * .01)


def test_dt_causal_mask_hides_current_action_and_future_tokens():
    torch.manual_seed(0)
    model = DecisionTransformer(hidden_size=16, context_length=4).eval()
    observations = torch.randn(1, 4, LOCAL_FEATURES)
    actions = torch.zeros(1, 4, dtype=torch.long)
    returns = torch.zeros(1, 4, 1)
    times = torch.arange(4).unsqueeze(0)
    with torch.no_grad():
        original = model(observations, actions, returns, times)
        actions[:, 1:] = 18
        observations[:, 2:] = 999
        returns[:, 2:] = 999
        altered = model(observations, actions, returns, times)
    torch.testing.assert_close(original[:, :2], altered[:, :2])


def test_dt_windows_returns_and_checkpoint_roundtrip(tmp_path):
    path = coherent_dataset(tmp_path / "data.npz")
    windows = TrajectoryWindows([path], context_length=2)
    obs, actions, returns, times, valid = windows[0]
    np.testing.assert_allclose(returns[:, 0], [3, 2])
    assert obs[0, 1] == obs[1, 1]
    config = MethodConfig(method="dt", data_paths=[path], output_dir=tmp_path,
                          hidden_size=16, epochs=2, batch_size=16, context_length=2)
    result = build_method(config).train()
    assert result.status == "trained"
    policy = load_method_policy(tmp_path / "dt/result.json")
    checkpoint = torch.load(
        tmp_path / "dt/decision_transformer.pt",
        map_location="cpu",
        weights_only=True,
    )
    for name, value in checkpoint["model_state_dict"].items():
        torch.testing.assert_close(policy.model.state_dict()[name], value)
    sample = np.zeros((22, LOCAL_FEATURES), dtype=np.float32)
    first = policy.act(sample)
    policy.reset()
    np.testing.assert_array_equal(first, policy.act(sample))
    assert first.shape == (22,)
    assert np.all((0 <= first) & (first < 19))


def test_ghosting_right_team_actions_are_mirrored():
    from thesis_experiments.training.ghosting import ghosting_actions
    obs = np.zeros((2, LOCAL_FEATURES), dtype=np.float32)
    obs[:, 6] = [1, -1]
    np.testing.assert_array_equal(ghosting_actions(obs, np.array([[.1, 0], [.1, 0]])), [5, 1])


def test_shots_and_goal_kicks_are_not_goals():
    from thesis_experiments.transitions.builder import _apply_goal_rewards
    audit = SimpleNamespace(frames=np.arange(4), left_team_id="A", events=[
        {"event_type": "ShotAtGoal", "anchor_frame": 0, "details": {"Team": "A"}},
        {"event_type": "GoalKick", "anchor_frame": 1, "details": {"Team": "A"}},
        {"event_type": "ShotAtGoal", "outcome": "SuccessfulShot", "anchor_frame": 2, "details": {"Team": "A"}},
    ])
    rewards = np.zeros((3, 2))
    _apply_goal_rewards(audit, rewards, 1)
    np.testing.assert_array_equal(rewards, [[0, 0], [0, 0], [1, -1]])


def test_dataset_a_tackle_override_uses_player_without_ball():
    from thesis_experiments.data.dataset_a_grf_audit import _events_and_overrides

    left_ids = [f"L{index}" for index in range(11)]
    right_ids = [f"R{index}" for index in range(11)]
    records = [{"context": {"events": [
        {
            "event_type": "TacklingGame",
            "details": {
                "Winner": "L0", "WinnerTeam": "left", "WinnerRole": "withBallControl",
                "Loser": "R0", "LoserTeam": "right", "LoserRole": "withoutBallControl",
            },
        },
        {
            "event_type": "TacklingGame",
            "details": {
                "Winner": "L1", "WinnerTeam": "left", "WinnerRole": "withoutBallControl",
                "Loser": "R1", "LoserTeam": "right", "LoserRole": "withBallControl",
            },
        },
    ]}}]
    actions = [np.zeros(22, dtype=np.int64)]

    events = _events_and_overrides(records, actions, left_ids, right_ids)

    assert actions[0][11] == 16
    assert actions[0][1] == 16
    assert [event["resolved_actor_id"] for event in events] == ["R0", "L1"]
    assert [event["joint_action_player_index"] for event in events] == [11, 1]
    assert all(event["action_override_status"] == "applied" for event in events)


def test_action_metrics_separate_movement_and_rare_actions():
    from thesis_experiments.scripts.evaluate_policy import action_metrics

    metrics = action_metrics([1, 2, 1, 5], [1, 2, 11, 14])
    assert metrics["action_accuracy"] == 0.5
    assert metrics["movement_action_accuracy"] == 1.0
    assert metrics["rare_action_accuracy"] == 0.0
    assert metrics["rare_action_samples"] == 2
    assert metrics["predicted_rare_actions"] == 0


def test_action_metrics_exclude_provider_specific_labels_from_common_metric():
    from thesis_experiments.scripts.evaluate_policy import action_metrics

    metrics = action_metrics([1, 0, 0, 16], [1, 13, 17, 16])
    assert metrics["action_accuracy"] == 0.5
    assert metrics["common_action_samples"] == 2
    assert metrics["common_action_accuracy"] == 1.0
    assert 13 not in metrics["common_action_ids"]
    assert 17 not in metrics["common_action_ids"]


def test_final_dataset_partitions_do_not_leave_short_tail():
    from thesis_experiments.scripts.prepare_final_dataset import _partitions

    partitions = list(_partitions(0, 8_500, 4_096, 512))
    assert partitions == [(0, 4_096), (4_096, 8_500)]
    assert all(end - start >= 512 for start, end in partitions)
    assert partitions[-1][1] == 8_500


def test_manifest_validator_enforces_selected_team(tmp_path):
    from thesis_experiments.scripts.validate_split_manifest import validate_manifest

    data_path = coherent_dataset(tmp_path / "data.npz")
    record = {
        "path": str(data_path),
        "focal_team_id": "DFL-CLU-00000P",
        "transitions": 4,
    }
    manifest_path = tmp_path / "team_manifest.json"
    manifest_path.write_text(json.dumps({
        "team_id": "DFL-CLU-00000P",
        "splits": {
            "train": [record],
            "validation": [record],
            "test": [record],
        },
    }))

    result = validate_manifest(
        manifest_path,
        tmp_path,
        expected_team_id="DFL-CLU-00000P",
    )
    assert result["ok"]
    assert result["team_id"] == "DFL-CLU-00000P"

    with pytest.raises(ValueError, match="expected team_id"):
        validate_manifest(
            manifest_path,
            tmp_path,
            expected_team_id="DFL-CLU-other",
        )

    document = json.loads(manifest_path.read_text())
    document["splits"]["test"][0] = {**record, "focal_team_id": "DFL-CLU-other"}
    manifest_path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="does not belong"):
        validate_manifest(manifest_path, tmp_path)


def test_evaluation_rejects_match_leakage_even_in_different_files(tmp_path):
    from thesis_experiments.scripts.evaluate_policy import validate_split
    first = coherent_dataset(tmp_path / "train.npz")
    second = coherent_dataset(tmp_path / "test.npz")
    with pytest.raises(ValueError, match="also occur"):
        validate_split([first], [second])


def test_suite_rejects_placeholder_success_and_persists_failure(tmp_path, monkeypatch):
    import sys
    from thesis_experiments.scripts import run_training_suite
    path = coherent_dataset(tmp_path / "data.npz")
    directory = tmp_path / "suite"

    def placeholder(command, log):
        result_dir = directory / "seed_0/methods/dt"
        result_dir.mkdir(parents=True)
        (result_dir / "result.json").write_text(json.dumps({"status": "placeholder", "artifacts": {}}))
        return 0

    monkeypatch.setattr(run_training_suite, "_run_command", placeholder)
    monkeypatch.setattr(sys, "argv", ["suite", "--data", str(path), "--methods", "dt", "--output-dir", str(directory)])
    with pytest.raises(SystemExit) as exit_info:
        run_training_suite.main()
    assert exit_info.value.code == 1
    manifest = json.loads((directory / "training_manifest.json").read_text())
    assert manifest[0]["status"] == "failed"


def test_suite_binds_every_method_to_one_team_contract(tmp_path, monkeypatch):
    import sys
    from thesis_experiments.scripts import run_training_suite

    split_paths = {}
    for split in ("train", "validation", "test"):
        path = coherent_dataset(tmp_path / f"{split}.npz")
        dataset = JointTransitionDataset.load(path)
        dataset.match_id = f"selected-team-{split}"
        dataset.save(path)
        split_paths[split] = path

    manifest_path = tmp_path / "team_manifest.json"
    manifest_path.write_text(json.dumps({
        "team_id": "DFL-CLU-00000P",
        "splits": {
            split: [{
                "path": str(path),
                "match_id": f"selected-team-{split}",
                "focal_team_id": "DFL-CLU-00000P",
            }]
            for split, path in split_paths.items()
        },
    }))
    output_dir = tmp_path / "suite"
    monkeypatch.setattr(sys, "argv", [
        "suite",
        "--split-manifest", str(manifest_path),
        "--expected-team-id", "DFL-CLU-00000P",
        "--output-dir", str(output_dir),
        "--methods", "all",
        "--dry-run",
    ])
    run_training_suite.main()

    contract = json.loads((output_dir / "dataset_contract.json").read_text())
    assert contract["team_id"] == "DFL-CLU-00000P"
    assert contract["splits"]["train"]["matches"] == ["selected-team-train"]
    runs = json.loads((output_dir / "training_manifest.json").read_text())
    assert {run["method"] for run in runs} == {
        "bc", "gail", "ppo", "cql", "dt", "ghosting", "ctde",
    }
    assert all(str(split_paths["train"]) in run["command"] for run in runs)
    assert all(run["status"] == "dry-run" for run in runs)

    (output_dir / "dataset_contract.json").unlink()
    monkeypatch.setattr(sys, "argv", [
        "suite",
        "--split-manifest", str(manifest_path),
        "--expected-team-id", "DFL-CLU-00000P",
        "--output-dir", str(output_dir),
        "--methods", "all",
        "--dry-run",
        "--resume",
    ])
    with pytest.raises(SystemExit) as exit_info:
        run_training_suite.main()
    assert exit_info.value.code == 2


def test_live_grf_actions_change_dynamics():
    root = Path(__file__).resolve().parents[2]
    if not (root / ".venv-grf/bin/python").exists() or not (root / "data/processed/metrica_ctde_100.npz").exists():
        pytest.skip("Local GRF installation and initialization data required")
    if JointTransitionDataset.load(root / "data/processed/metrica_ctde_100.npz").observation_version != OBSERVATION_VERSION:
        pytest.skip("Live GRF test initialization data must be rebuilt with the current observation version")
    from gymnasium.utils.env_checker import check_env
    from thesis_experiments.simulation.gymnasium_grf import FootballGRFEnv
    env = FootballGRFEnv([root / "data/processed/metrica_ctde_100.npz"], max_episode_steps=10)
    try:
        check_env(env, skip_render_check=True)
        endpoints = []
        for action in (1, 5):
            env.reset(seed=3, options={"transition_index": 0})
            for _ in range(10):
                obs, reward, terminated, truncated, info = env.step([action] * 22)
            assert obs.shape == (22, LOCAL_FEATURES) and np.isfinite(obs).all()
            assert truncated and not terminated
            endpoints.append(info["positions"])
        assert np.linalg.norm(endpoints[0] - endpoints[1]) > .001
    finally:
        env.close()
