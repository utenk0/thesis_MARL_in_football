"""Evaluate a reloaded policy on held-out actions and autonomous GRF trajectories."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from thesis_experiments.datasets.action_mapping import GRF_ACTIONS
from thesis_experiments.policies.loading import load_method_policy
from thesis_experiments.transitions.observations import (
    LOCAL_FEATURE_SLICES,
    OBSERVATION_VERSION,
)
from thesis_experiments.transitions.schema import JointTransitionDataset


# These labels have the same operational meaning in both Dataset A and Metrica.
# Provider-specific sticky-state labels (sprint/release-sprint and the two
# dribble encodings) remain visible in the full 19-action report, but are not
# used for the primary cross-provider comparison.
COMMON_ACTION_IDS = tuple(range(13)) + (14, 16)


def validate_split(train_paths, evaluation_paths):
    train = {(d.source, d.match_id) for d in map(JointTransitionDataset.load, train_paths)}
    evaluation = {(d.source, d.match_id) for d in map(JointTransitionDataset.load, evaluation_paths)}
    overlap = train & evaluation
    if overlap:
        raise ValueError(f"Evaluation matches also occur in training: {sorted(overlap)}")


def _reset(policy):
    if hasattr(policy, "reset"):
        policy.reset()


def _reward(policy, rewards):
    if hasattr(policy, "observe_rewards"):
        policy.observe_rewards(np.repeat(rewards, 11))


def action_metrics(predictions, labels):
    predictions, labels = np.asarray(predictions).reshape(-1), np.asarray(labels).reshape(-1)
    if len(labels) == 0:
        raise ValueError("At least one action label is required.")
    names = {identifier: name for name, identifier in GRF_ACTIONS.items()}
    per_action = []
    recalls = []
    for action in range(19):
        selected = labels == action
        recall = float((predictions[selected] == action).mean()) if selected.any() else None
        if recall is not None:
            recalls.append(recall)
        per_action.append({
            "action": action,
            "name": names[action],
            "support": int(selected.sum()),
            "predicted": int((predictions == action).sum()),
            "recall": recall,
        })
    movement = labels <= 8
    rare = labels > 8
    common = np.isin(labels, COMMON_ACTION_IDS)
    common_recalls = [
        float((predictions[labels == action] == action).mean())
        for action in COMMON_ACTION_IDS
        if np.any(labels == action)
    ]
    return {"action_accuracy": float((predictions == labels).mean()),
            "balanced_action_accuracy": float(np.mean(recalls)), "samples": int(len(labels)),
            "movement_action_accuracy": float((predictions[movement] == labels[movement]).mean()) if movement.any() else None,
            "rare_action_accuracy": float((predictions[rare] == labels[rare]).mean()) if rare.any() else None,
            "rare_action_samples": int(rare.sum()),
            "predicted_rare_actions": int((predictions > 8).sum()),
            "common_action_accuracy": float((predictions[common] == labels[common]).mean()) if common.any() else None,
            "common_balanced_action_accuracy": float(np.mean(common_recalls)) if common_recalls else None,
            "common_action_samples": int(common.sum()),
            "common_action_ids": list(COMMON_ACTION_IDS),
            "predicted_action_counts": np.bincount(predictions, minlength=19).tolist(),
            "expert_action_counts": np.bincount(labels, minlength=19).tolist(),
            "per_action": per_action}


def _masked_metrics(predictions, labels, mask):
    predictions = np.asarray(predictions).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    if mask.shape != labels.shape:
        raise ValueError("Metric mask must align with labels.")
    return action_metrics(predictions[mask], labels[mask]) if mask.any() else None


def evaluate(result_path, data_paths, output, *, grf_steps=100, grf_python=None, seed=0, device="cpu"):
    from thesis_experiments.simulation.gymnasium_grf import FootballGRFEnv

    result_path, output = Path(result_path), Path(output)
    result = json.loads(result_path.read_text())
    config = json.loads((result_path.parent / "config.json").read_text())
    train_paths = [Path(path) for path in config["data_paths"]]
    validate_split(train_paths, data_paths)
    train_labels = np.concatenate([JointTransitionDataset.load(path).actions.reshape(-1) for path in train_paths])
    majority = int(np.bincount(train_labels, minlength=19).argmax())
    policy = load_method_policy(result_path, device=device)
    output.parent.mkdir(parents=True, exist_ok=True)
    reports, all_predictions, all_labels, all_previous_predictions = [], [], [], []
    by_source = {}
    for path in data_paths:
        dataset = JointTransitionDataset.load(path)
        _reset(policy)
        if result["method"] == "dt":
            predictions = []
            for index, observations in enumerate(dataset.local_observations):
                predictions.append(policy.act(observations))
                _reward(policy, dataset.team_rewards[index])
                if dataset.dones[index]:
                    _reset(policy)
            predictions = np.asarray(predictions)
        else:
            flat = dataset.local_observations.reshape(-1, dataset.local_observations.shape[-1])
            predictions = np.concatenate([policy.act(flat[start:start + 4096]) for start in range(0, len(flat), 4096)])
        all_predictions.append(predictions.reshape(-1))
        all_labels.append(dataset.actions.reshape(-1))
        source_rows = by_source.setdefault(dataset.source, {"predictions": [], "labels": [], "previous": []})
        source_rows["predictions"].append(predictions.reshape(-1))
        source_rows["labels"].append(dataset.actions.reshape(-1))
        previous_predictions = None
        if dataset.observation_version == OBSERVATION_VERSION:
            previous_predictions = dataset.local_observations[
                ..., LOCAL_FEATURE_SLICES["previous_action"]
            ].argmax(axis=-1)
            all_previous_predictions.append(previous_predictions.reshape(-1))
            source_rows["previous"].append(previous_predictions.reshape(-1))
        report = dict(source=dataset.source, match_id=dataset.match_id,
                      offline=action_metrics(predictions, dataset.actions))
        if previous_predictions is not None:
            report["previous_action_baseline"] = action_metrics(
                previous_predictions, dataset.actions
            )
            report["action_change_points"] = _masked_metrics(
                predictions,
                dataset.actions,
                dataset.actions != previous_predictions,
            )
        if grf_steps:
            steps = min(grf_steps, len(dataset.actions))
            env = FootballGRFEnv([path], grf_python=grf_python, max_episode_steps=steps)
            positions, actions, rewards = [], [], []
            _reset(policy)
            try:
                observations, info = env.reset(seed=seed, options={"transition_index": 0})
                initial_positions = info["positions"].copy()
                for _ in range(steps):
                    chosen = policy.act(observations)
                    observations, _, terminated, truncated, info = env.step(chosen)
                    _reward(policy, info["team_rewards"])
                    positions.append(info["positions"].copy())
                    actions.append(chosen.copy())
                    rewards.append(info["team_rewards"].copy())
                    if terminated or truncated:
                        break
            finally:
                env.close()
            target = dataset.next_global_states[:len(positions), :44].reshape(-1, 22, 2)
            errors = np.linalg.norm(np.asarray(positions) - target, axis=-1)
            trajectory = output.parent / f"{output.stem}_{dataset.source}_{dataset.match_id}_rollout.npz"
            np.savez_compressed(trajectory, positions=positions, actions=actions, team_rewards=rewards,
                                target_positions=target, initial_positions=initial_positions)
            report["grf"] = {"steps": len(positions), "controlled_players": 22, "score": info["score"],
                             "team_returns": np.sum(rewards, axis=0).tolist(),
                             "mean_position_error_grf": float(errors.mean()),
                             "final_position_error_grf": float(errors[-1].mean()),
                             "mean_displacement_grf": float(np.linalg.norm(positions[-1] - initial_positions, axis=-1).mean()),
                             "finite": bool(np.isfinite(positions).all()), "trajectory": str(trajectory)}
        reports.append(report)
    labels = np.concatenate(all_labels)
    flat_predictions = np.concatenate(all_predictions)
    majority_predictions = np.full_like(labels, majority)
    source_reports = {}
    for source, rows in by_source.items():
        source_predictions = np.concatenate(rows["predictions"])
        source_labels = np.concatenate(rows["labels"])
        source_report = action_metrics(source_predictions, source_labels)
        if rows["previous"]:
            source_previous = np.concatenate(rows["previous"])
            source_report["previous_action_baseline"] = action_metrics(source_previous, source_labels)
            source_report["action_change_points"] = _masked_metrics(
                source_predictions, source_labels, source_labels != source_previous
            )
        source_reports[source] = source_report
    summary = {"method": result["method"], "checkpoint_reloaded": True,
               "evaluation_matches_disjoint_from_training": True,
               "offline": action_metrics(flat_predictions, labels),
               "training_majority_action": majority,
               "majority_baseline_accuracy": float((labels == majority).mean()),
               "majority_baseline": action_metrics(majority_predictions, labels),
               "sources": source_reports,
               "datasets": reports,
               "scope": (
                   "Action labels are inferred. Full 19-action metrics are diagnostic; common-action metrics "
                   "are the primary provider-aware comparison. GRF runs use current observations only."
               )}
    if all_previous_predictions:
        previous = np.concatenate(all_previous_predictions)
        summary["previous_action_baseline"] = action_metrics(
            previous, labels
        )
        summary["action_change_points"] = _masked_metrics(
            flat_predictions, labels, labels != previous
        )
    output.write_text(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--data", required=True, nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--grf-steps", type=int, default=100)
    parser.add_argument("--grf-python", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(1)
    print(json.dumps(evaluate(args.result, args.data, args.output, grf_steps=args.grf_steps,
                              grf_python=args.grf_python, seed=args.seed, device=args.device), indent=2))


if __name__ == "__main__":
    main()
