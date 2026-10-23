"""Train and evaluate the single-team behavioural-cloning experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from thesis_experiments.policies.team import load_shared_actor_policy
from thesis_experiments.training.team_bc import evaluate_team_bc, train_team_bc
from thesis_experiments.transitions.schema import JointTransitionDataset


METERS_PER_GRF = np.asarray([52.5, 68.0 / 0.84], dtype=np.float32)


def _paths(manifest: dict, split: str) -> list[Path]:
    return [Path(row["path"]) for row in manifest["splits"][split]]


def _match_ids(manifest: dict, split: str) -> list[str]:
    return sorted({str(row["match_id"]) for row in manifest["splits"][split]})


def _shape_metrics(predicted: np.ndarray, target: np.ndarray) -> dict[str, float]:
    predicted_m = predicted * METERS_PER_GRF
    target_m = target * METERS_PER_GRF
    errors = np.linalg.norm(predicted_m - target_m, axis=-1)
    predicted_centroid = predicted_m.mean(axis=1)
    target_centroid = target_m.mean(axis=1)
    centroid_error = np.linalg.norm(predicted_centroid - target_centroid, axis=-1)
    predicted_depth = np.ptp(predicted_m[..., 0], axis=1)
    target_depth = np.ptp(target_m[..., 0], axis=1)
    predicted_width = np.ptp(predicted_m[..., 1], axis=1)
    target_width = np.ptp(target_m[..., 1], axis=1)
    return {
        "steps": int(len(predicted)),
        "ade_meters": float(errors.mean()),
        "fde_meters": float(errors[-1].mean()),
        "centroid_error_meters": float(centroid_error.mean()),
        "team_depth_mae_meters": float(np.abs(predicted_depth - target_depth).mean()),
        "team_width_mae_meters": float(np.abs(predicted_width - target_width).mean()),
    }


def conditional_reconstruction(
    checkpoint: Path,
    data_path: Path,
    output: Path,
    *,
    team_id: str,
    steps: int,
    grf_python: Path | None,
    seed: int,
    device: str,
    focal_controller: str = "bc_policy",
    warmup_steps: int = 0,
    evaluation_start_transition: int | None = None,
) -> dict[str, object]:
    """Roll out the focal policy while replaying recorded opponent actions."""

    from thesis_experiments.simulation.gymnasium_grf import FootballGRFEnv

    dataset = JointTransitionDataset.load(data_path)
    selected = np.flatnonzero(dataset.team_ids.astype(str) == team_id)
    if len(selected) != 11:
        raise ValueError(f"{data_path}: team {team_id!r} does not occupy 11 slots.")
    if focal_controller not in {"bc_policy", "recorded_expert_actions", "previous_action"}:
        raise ValueError(f"Unknown focal controller: {focal_controller}")
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    evaluation_start = (
        warmup_steps
        if evaluation_start_transition is None
        else evaluation_start_transition
    )
    reset_index = evaluation_start - warmup_steps
    if reset_index < 0:
        raise ValueError("evaluation_start_transition must be at least warmup_steps")
    if evaluation_start >= len(dataset.actions):
        raise ValueError("evaluation_start_transition is outside the chunk")
    evaluation_stop = min(evaluation_start + steps, len(dataset.actions))
    environment_steps = evaluation_stop - reset_index
    policy = (
        load_shared_actor_policy(checkpoint, device=device)
        if focal_controller == "bc_policy"
        else None
    )
    env = FootballGRFEnv(
        [data_path], grf_python=grf_python, max_episode_steps=environment_steps
    )
    predicted_positions = []
    applied_actions = []
    persistent_actions = (
        dataset.actions[evaluation_start - 1, selected].copy()
        if evaluation_start > 0
        else np.zeros(len(selected), dtype=np.int64)
    )
    try:
        observations, info = env.reset(
            seed=seed, options={"transition_index": reset_index}
        )
        initial_positions = info["positions"].copy()
        control_initial_positions = None
        for index in range(reset_index, evaluation_stop):
            joint_actions = dataset.actions[index].copy()
            if index >= evaluation_start:
                if control_initial_positions is None:
                    control_initial_positions = info["positions"].copy()
                if focal_controller == "bc_policy":
                    joint_actions[selected] = policy.act(observations[selected])
                elif focal_controller == "previous_action":
                    joint_actions[selected] = persistent_actions
            observations, _, terminated, truncated, info = env.step(joint_actions)
            if index >= evaluation_start:
                predicted_positions.append(info["positions"][selected].copy())
                applied_actions.append(joint_actions.copy())
            if terminated or truncated:
                break
    finally:
        env.close()
    predicted = np.asarray(predicted_positions, dtype=np.float32)
    target_all = dataset.next_global_states[
        evaluation_start : evaluation_start + len(predicted), :44
    ].reshape(-1, 22, 2)
    target = target_all[:, selected]
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        predicted_positions=predicted,
        target_positions=target,
        applied_joint_actions=np.asarray(applied_actions),
        initial_positions=initial_positions,
        control_initial_positions=control_initial_positions,
        focal_player_indices=selected,
        focal_team_id=team_id,
        focal_controller=focal_controller,
    )
    return {
        "match_id": dataset.match_id,
        "source": dataset.source,
        "chunk": str(data_path),
        "team_id": team_id,
        "team_side": "left" if bool(np.all(selected < 11)) else "right",
        "opponent_control": "recorded_open_loop_actions",
        "focal_control": focal_controller,
        "recorded_warmup_steps": warmup_steps,
        "reset_transition": reset_index,
        "evaluation_start_transition": evaluation_start,
        "previous_action_initialization": (
            "recorded_action_at_t_minus_1"
            if focal_controller == "previous_action" and evaluation_start > 0
            else None
        ),
        "metrics": _shape_metrics(predicted, target),
        "artifact": str(output),
        "scope": (
            "Conditional reconstruction: the opponent uses held-out recorded actions. "
            "The recorded-expert focal controller estimates simulator mismatch; this is "
            "not a fully autonomous match."
        ),
    }


def _write_report(path: Path, result: dict) -> None:
    train = result["training"]
    validation = result["evaluations"]["validation"]["overall"]
    test = result["evaluations"]["test"]["overall"]
    lines = [
        "# Single-team behavioural-cloning run",
        "",
        f"- Team: `{result['team_id']}`",
        f"- Train matches: {', '.join(result['split_matches']['train'])}",
        f"- Validation matches: {', '.join(result['split_matches']['validation'])}",
        f"- Test matches: {', '.join(result['split_matches']['test'])}",
        f"- Observation: `{train['observation_variant']}` ({train['retained_features']} / 61 features)",
        f"- Training samples: {train['samples']:,}",
        f"- Validation-selected epoch: {train['selected_epoch']} of {train['epochs_run']} run",
        "",
        "## Offline reconstruction",
        "",
        "| Split | Policy accuracy | Balanced accuracy | Majority baseline | Previous-action baseline | Action-change accuracy |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, metrics in (("Validation", validation), ("Test", test)):
        policy = metrics["policy"]
        previous = metrics["previous_action_baseline"]
        majority = metrics["training_majority_baseline"]
        change = metrics["action_change_policy"]
        lines.append(
            f"| {name} | {policy['action_accuracy']:.4f} | "
            f"{policy['balanced_action_accuracy']:.4f} | "
            f"{majority['action_accuracy']:.4f} | "
            f"{previous['action_accuracy']:.4f} | "
            f"{change['action_accuracy']:.4f} |" if change else
            f"| {name} | {policy['action_accuracy']:.4f} | "
            f"{policy['balanced_action_accuracy']:.4f} | "
            f"{majority['action_accuracy']:.4f} | "
            f"{previous['action_accuracy']:.4f} | n/a |"
        )
    gate = result["pilot_gate"]
    lines.extend(
        [
            "",
            "## Pilot gate",
            "",
            f"**{'PASS' if gate['passed'] else 'NO-GO'}**: " + "; ".join(gate["reasons"]),
        ]
    )
    if result["conditional_reconstructions"]:
        lines.extend(
            [
                "",
                "## Conditional GRF reconstruction",
                "",
                "| Focal controller | ADE (m) | FDE (m) | Centroid error (m) | Width MAE (m) | Depth MAE (m) |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for reconstruction in result["conditional_reconstructions"]:
            metrics = reconstruction["metrics"]
            lines.append(
                f"| {reconstruction['focal_control']} | {metrics['ade_meters']:.3f} | "
                f"{metrics['fde_meters']:.3f} | {metrics['centroid_error_meters']:.3f} | "
                f"{metrics['team_width_mae_meters']:.3f} | "
                f"{metrics['team_depth_mae_meters']:.3f} |"
            )
    lines.extend(
        [
            "",
            "Offline metrics use recorded observations (teacher forcing). Conditional GRF "
            "rollouts, when requested, let the learned focal team act while replaying the "
            "held-out opponent actions.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument(
        "--observation-variant",
        choices=["spatial", "temporal_context", "temporal_context_no_slot", "full"],
        default="temporal_context_no_slot",
    )
    parser.add_argument(
        "--class-weighting", choices=["none", "inverse_frequency"], default="none"
    )
    parser.add_argument("--max-class-weight", type=float, default=20.0)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--grf-steps", type=int, default=0)
    parser.add_argument("--grf-warmup-steps", type=int, default=0)
    parser.add_argument("--grf-evaluation-start", type=int)
    parser.add_argument("--grf-python", type=Path)
    parser.add_argument("--rollouts-per-test-match", type=int, default=1)
    args = parser.parse_args()
    if args.grf_steps < 0 or args.grf_warmup_steps < 0 or args.rollouts_per_test_match < 1:
        parser.error("GRF steps must be non-negative and rollouts per match positive.")
    torch.set_num_threads(args.torch_threads)
    manifest = json.loads(args.manifest.read_text())
    team_id = str(manifest["team_id"])
    train_paths = _paths(manifest, "train")
    validation_paths = _paths(manifest, "validation")
    test_paths = _paths(manifest, "test")
    for split, paths in (
        ("train", train_paths),
        ("validation", validation_paths),
        ("test", test_paths),
    ):
        if not paths:
            raise ValueError(f"Manifest split {split!r} is empty.")
    if manifest["split_strategy"]["type"] == "match_disjoint":
        split_sets = [set(_match_ids(manifest, name)) for name in ("train", "validation", "test")]
        if any(split_sets[i] & split_sets[j] for i in range(3) for j in range(i + 1, 3)):
            raise ValueError("Match-disjoint manifest contains overlap.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "single_team_bc.pt"
    training = train_team_bc(
        train_paths,
        checkpoint,
        team_id=team_id,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hidden_size=args.hidden_size,
        seed=args.seed,
        device=args.device,
        observation_variant=args.observation_variant,
        class_weighting=args.class_weighting,
        max_class_weight=args.max_class_weight,
        validation_paths=validation_paths,
        early_stopping_patience=args.early_stopping_patience,
    )
    majority = int(np.argmax(training["class_counts"]))
    evaluations = {}
    for split, paths in (
        ("train", train_paths),
        ("validation", validation_paths),
        ("test", test_paths),
    ):
        evaluation = evaluate_team_bc(
            checkpoint,
            paths,
            team_id=team_id,
            training_majority_action=majority,
            device=args.device,
        )
        evaluation_path = args.output_dir / f"{split}_evaluation.json"
        evaluation_path.write_text(json.dumps(evaluation, indent=2))
        evaluations[split] = evaluation

    reconstructions = []
    if args.grf_steps:
        by_match: dict[str, list[Path]] = {}
        for row in manifest["splits"]["test"]:
            by_match.setdefault(str(row["match_id"]), []).append(Path(row["path"]))
        for match_id, paths in sorted(by_match.items()):
            for index, path in enumerate(paths[: args.rollouts_per_test_match]):
                artifact = args.output_dir / "reconstructions" / f"{match_id}_{index:02d}.npz"
                for controller in ("bc_policy", "previous_action", "recorded_expert_actions"):
                    controller_artifact = artifact.with_name(
                        f"{artifact.stem}_{controller}{artifact.suffix}"
                    )
                    reconstructions.append(
                        conditional_reconstruction(
                            checkpoint,
                            path,
                            controller_artifact,
                            team_id=team_id,
                            steps=args.grf_steps,
                            grf_python=args.grf_python,
                            seed=args.seed + index,
                            device=args.device,
                            focal_controller=controller,
                            warmup_steps=args.grf_warmup_steps,
                            evaluation_start_transition=args.grf_evaluation_start,
                        )
                    )
        (args.output_dir / "conditional_reconstructions.json").write_text(
            json.dumps(reconstructions, indent=2)
        )

    config = {
        "manifest": str(args.manifest),
        "team_id": team_id,
        "split_strategy": manifest["split_strategy"],
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "hidden_size": args.hidden_size,
            "seed": args.seed,
            "observation_variant": args.observation_variant,
            "class_weighting": args.class_weighting,
            "max_class_weight": args.max_class_weight,
            "early_stopping_patience": args.early_stopping_patience,
        },
        "data_paths": {name: [str(path) for path in paths] for name, paths in (
            ("train", train_paths), ("validation", validation_paths), ("test", test_paths)
        )},
    }
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2))
    validation_overall = evaluations["validation"]["overall"]
    validation_accuracy_pass = (
        validation_overall["policy"]["action_accuracy"]
        > validation_overall["training_majority_baseline"]["action_accuracy"]
    )
    validation_change_pass = (
        validation_overall["action_change_policy"] is not None
        and validation_overall["action_change_policy"]["action_accuracy"]
        > validation_overall["action_change_training_majority_baseline"]["action_accuracy"]
    )
    gate_reasons = [
        (
            "validation policy beats the training-majority baseline"
            if validation_accuracy_pass
            else "validation policy does not beat the training-majority baseline"
        ),
        (
            "validation action-change policy beats the change-point majority baseline"
            if validation_change_pass
            else "validation action-change policy does not beat the change-point majority baseline"
        ),
    ]
    result = {
        "method": "bc",
        "experiment": "single_team_reconstruction",
        "status": "completed",
        "team_id": team_id,
        "split_matches": {
            split: _match_ids(manifest, split) for split in ("train", "validation", "test")
        },
        "training": training,
        "evaluations": evaluations,
        "pilot_gate": {
            "passed": bool(validation_accuracy_pass and validation_change_pass),
            "uses_test_for_decision": False,
            "reasons": gate_reasons,
        },
        "conditional_reconstructions": reconstructions,
        "artifacts": {
            "checkpoint": str(checkpoint),
            "config": str(args.output_dir / "config.json"),
        },
    }
    result_path = args.output_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2))
    _write_report(args.output_dir / "RUN_REPORT.md", result)
    print(json.dumps({"result": str(result_path), "report": str(args.output_dir / "RUN_REPORT.md")}, indent=2))


if __name__ == "__main__":
    main()
