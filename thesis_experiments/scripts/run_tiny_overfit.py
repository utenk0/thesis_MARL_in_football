"""Run local tiny-data learning diagnostics for BC, CQL, DT, and Ghosting.

The purpose is not to create useful final models.  A small contiguous clip is
reused many times to establish whether each implementation can optimize its
own objective before expensive full-data experiments are attempted.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from thesis_experiments.methods import MethodConfig, build_method
from thesis_experiments.policies.loading import load_method_policy
from thesis_experiments.policies.team import load_shared_actor_policy
from thesis_experiments.training.bc import train_bc
from thesis_experiments.training.decision_transformer import (
    DecisionTransformer,
    DecisionTransformerPolicy,
    TrajectoryWindows,
    train_decision_transformer,
)
from thesis_experiments.training.ghosting import (
    ghosting_actions,
    load_ghosting_policy,
    train_ghosting,
)
from thesis_experiments.transitions.schema import JointTransitionDataset


def _slice_dataset(source: JointTransitionDataset, frames: int) -> JointTransitionDataset:
    if frames <= 1:
        raise ValueError("The tiny diagnostic requires at least two frames.")
    if frames > len(source.actions):
        raise ValueError(f"Requested {frames} frames, but the source contains {len(source.actions)}.")
    time_fields = {
        "local_observations",
        "global_states",
        "actions",
        "team_rewards",
        "next_local_observations",
        "next_global_states",
        "dones",
        "frames",
    }
    values: dict[str, Any] = {}
    for descriptor in fields(JointTransitionDataset):
        value = getattr(source, descriptor.name)
        values[descriptor.name] = value[:frames].copy() if descriptor.name in time_fields else value
    values["dones"][:] = False
    values["dones"][-1] = True
    values["match_id"] = f"{source.match_id}_tiny_{frames}"
    result = JointTransitionDataset(**values)
    result.validate()
    return result


def _accuracy(predictions: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean(np.asarray(predictions).reshape(-1) == np.asarray(labels).reshape(-1)))


def _loss_trend(path: Path) -> dict[str, Any]:
    with path.open(newline="", encoding="utf-8") as stream:
        values = [float(row[-1]) for row in csv.reader(stream) if row]
    return {
        "updates_logged": len(values),
        "initial": values[0],
        "final": values[-1],
        "decreased": bool(values[-1] < values[0]),
    }


@torch.no_grad()
def _dt_teacher_forced_accuracy(path: Path, checkpoint: Path, context_length: int) -> float:
    windows = TrajectoryWindows([path], context_length=context_length)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = DecisionTransformer(**saved["config"])
    model.load_state_dict(saved["model_state_dict"])
    model.eval()
    predictions, labels = [], []
    for observations, actions, returns, timesteps, valid in DataLoader(windows, batch_size=64):
        logits = model(observations, actions, returns, timesteps)
        predictions.append(logits[valid].argmax(-1).numpy())
        labels.append(actions[valid].numpy())
    return _accuracy(np.concatenate(predictions), np.concatenate(labels))


def _dt_autoregressive_accuracy(
    dataset: JointTransitionDataset,
    checkpoint: Path,
    *,
    recorded_initial_return: bool = False,
) -> float:
    policy = DecisionTransformerPolicy(checkpoint)
    if recorded_initial_return:
        returns_to_go = np.cumsum(dataset.team_rewards[::-1], axis=0)[::-1]
        policy.remaining = np.concatenate(
            [
                np.repeat(returns_to_go[0, 0], 11),
                np.repeat(returns_to_go[0, 1], 11),
            ]
        ).astype(np.float32)
    predictions = []
    for index, observations in enumerate(dataset.local_observations):
        predictions.append(policy.act(observations))
        policy.observe_rewards(np.repeat(dataset.team_rewards[index], 11))
        if dataset.dones[index]:
            policy.reset()
    return _accuracy(np.asarray(predictions), dataset.actions)


def _run_bc(
    path: Path,
    output_dir: Path,
    dataset: JointTransitionDataset,
    args: argparse.Namespace,
) -> dict[str, Any]:
    print("[1/4] Training tiny BC diagnostic", flush=True)
    checkpoint = output_dir / "bc" / "shared_actor_bc.pt"
    training = train_bc(
        [path],
        checkpoint,
        epochs=args.bc_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hidden_size=args.hidden_size,
        seed=args.seed,
    )
    policy = load_shared_actor_policy(checkpoint)
    observations = dataset.local_observations.reshape(-1, dataset.local_observations.shape[-1])
    accuracy = _accuracy(policy.act(observations), dataset.actions)
    return {
        "objective": "expert-action cross-entropy",
        "training_metrics": training,
        "tiny_training_accuracy": accuracy,
        "memorization_threshold": 0.98,
        "memorization_passed": accuracy >= 0.98,
        "checkpoint": str(checkpoint),
    }


def _run_cql(
    path: Path,
    output_dir: Path,
    dataset: JointTransitionDataset,
    args: argparse.Namespace,
) -> dict[str, Any]:
    print("[2/4] Training tiny CQL diagnostic", flush=True)
    method_root = output_dir / "cql_method"
    config = MethodConfig(
        method="cql",
        data_paths=[path],
        output_dir=method_root,
        seed=args.seed,
        total_timesteps=args.cql_steps,
        batch_size=min(args.batch_size, len(dataset.actions) * 22),
        learning_rate=args.learning_rate,
        hidden_size=args.hidden_size,
        device="cpu",
    )
    result = build_method(config).train()
    method_dir = method_root / "cql"
    policy = load_method_policy(method_dir / "result.json")
    observations = dataset.local_observations.reshape(-1, dataset.local_observations.shape[-1])
    accuracy = _accuracy(policy.act(observations), dataset.actions)
    majority = float(np.bincount(dataset.actions.reshape(-1), minlength=19).max() / dataset.actions.size)
    loss = _loss_trend(method_dir / "d3rlpy_logs" / "cql" / "loss.csv")
    return {
        "objective": "conservative offline Q-learning",
        "training_metrics": result.metrics,
        "tiny_training_accuracy": accuracy,
        "tiny_majority_baseline": majority,
        "gain_over_majority": accuracy - majority,
        "loss": loss,
        "optimization_passed": loss["decreased"],
        "interpretation": (
            "CQL is not supervised classification, so near-100% action memorization is not required; "
            "the primary diagnostic is a finite decreasing offline-RL loss plus improvement over a trivial baseline."
        ),
        "checkpoint": str(method_dir / "cql_policy.d3"),
    }


def _run_dt(
    path: Path,
    output_dir: Path,
    dataset: JointTransitionDataset,
    args: argparse.Namespace,
) -> dict[str, Any]:
    print("[3/4] Training tiny Decision Transformer diagnostic", flush=True)
    checkpoint = output_dir / "dt" / "decision_transformer.pt"
    training = train_decision_transformer(
        [path],
        checkpoint,
        epochs=args.epochs,
        batch_size=min(64, args.batch_size),
        learning_rate=args.learning_rate,
        hidden_size=args.dt_hidden_size,
        seed=args.seed,
        context_length=min(args.context_length, len(dataset.actions)),
        device="cpu",
    )
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    teacher = _dt_teacher_forced_accuracy(path, checkpoint, saved["config"]["context_length"])
    autoregressive = _dt_autoregressive_accuracy(dataset, checkpoint)
    oracle_return_autoregressive = _dt_autoregressive_accuracy(
        dataset,
        checkpoint,
        recorded_initial_return=True,
    )
    return {
        "objective": "causal sequence action prediction",
        "training_metrics": training,
        "loss_history": [float(value) for value in saved["loss_history"]],
        "teacher_forced_training_accuracy": teacher,
        "autoregressive_training_accuracy": autoregressive,
        "recorded_return_autoregressive_accuracy": oracle_return_autoregressive,
        "teacher_forced_memorization_threshold": 0.95,
        "teacher_forced_memorization_passed": teacher >= 0.95,
        "exposure_gap": teacher - autoregressive,
        "target_return_gap": oracle_return_autoregressive - autoregressive,
        "interpretation": (
            "Teacher-forced accuracy checks whether the model can learn the recorded sequence. "
            "Autoregressive accuracy also includes compounding errors from the model's own earlier predictions. "
            "The recorded-return variant is diagnostic only: it uses the trajectory's future return, which is not available at deployment."
        ),
        "checkpoint": str(checkpoint),
    }


def _run_ghosting(
    path: Path,
    output_dir: Path,
    dataset: JointTransitionDataset,
    args: argparse.Namespace,
) -> dict[str, Any]:
    print("[4/4] Training tiny Ghosting diagnostic", flush=True)
    checkpoint = output_dir / "ghosting" / "ghosting_baseline.pt"
    training = train_ghosting(
        [path],
        checkpoint,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hidden_size=args.hidden_size,
        seed=args.seed,
    )
    observations = dataset.local_observations.reshape(-1, dataset.local_observations.shape[-1])
    targets = dataset.next_local_observations.reshape(-1, dataset.next_local_observations.shape[-1])[:, :2]
    model = load_ghosting_policy(checkpoint)
    with torch.no_grad():
        positions = model(torch.from_numpy(observations).float()).numpy()
    model_mse = float(np.mean((positions - targets) ** 2))
    persistence_mse = float(np.mean((observations[:, :2] - targets) ** 2))
    action_accuracy = _accuracy(ghosting_actions(observations, positions), dataset.actions)
    improvement = 100.0 * (1.0 - model_mse / persistence_mse) if persistence_mse > 0 else None
    return {
        "objective": "next-player-position mean squared error",
        "training_metrics": training,
        "tiny_position_mse": model_mse,
        "no_movement_position_mse": persistence_mse,
        "improvement_over_no_movement_percent": improvement,
        "position_learning_passed": bool(improvement is not None and improvement > 0),
        "derived_movement_action_accuracy": action_accuracy,
        "interpretation": "Position error, not 19-action accuracy, is Ghosting's primary diagnostic.",
        "checkpoint": str(checkpoint),
    }


def _render_markdown(report: dict[str, Any]) -> str:
    bc, cql, dt, ghosting = (report["methods"][name] for name in ("bc", "cql", "dt", "ghosting"))
    lines = [
        "# Tiny overfitting diagnostics",
        "",
        "These are local implementation diagnostics on one deliberately tiny contiguous clip. "
        "They are not thesis results and the checkpoints must not be used as final models.",
        "",
        f"- Source: `{report['source_data']}`",
        f"- Tiny data: `{report['tiny_data']}`",
        f"- Frames: {report['frames']} ({report['player_samples']} player transitions)",
        f"- Observed actions: {report['action_counts']}",
        "",
        "| Method | Primary diagnostic | Result | Gate |",
        "|---|---|---:|---|",
        f"| BC | Tiny training action accuracy | {100 * bc['tiny_training_accuracy']:.2f}% | {'PASS' if bc['memorization_passed'] else 'FAIL'} (target ≥98%) |",
        f"| CQL | Offline loss | {cql['loss']['initial']:.4f} → {cql['loss']['final']:.4f} | {'PASS' if cql['optimization_passed'] else 'FAIL'} |",
        f"| DT | Teacher-forced training accuracy | {100 * dt['teacher_forced_training_accuracy']:.2f}% | {'PASS' if dt['teacher_forced_memorization_passed'] else 'FAIL'} (target ≥95%) |",
        f"| Ghosting | Improvement over no movement | {ghosting['improvement_over_no_movement_percent']:.2f}% | {'PASS' if ghosting['position_learning_passed'] else 'FAIL'} |",
        "",
        "## Secondary findings",
        "",
        f"- CQL action reconstruction: {100 * cql['tiny_training_accuracy']:.2f}% versus a {100 * cql['tiny_majority_baseline']:.2f}% majority baseline.",
        f"- DT autoregressive accuracy: {100 * dt['autoregressive_training_accuracy']:.2f}%; exposure gap: {100 * dt['exposure_gap']:.2f} percentage points.",
        f"- DT autoregressive accuracy with the recorded initial return: {100 * dt['recorded_return_autoregressive_accuracy']:.2f}% (diagnostic oracle, not deployable).",
        f"- Ghosting-derived movement-action accuracy: {100 * ghosting['derived_movement_action_accuracy']:.2f}%.",
        "",
        "## Interpretation",
        "",
        "- A BC or teacher-forced DT failure means the supervised data/model path cannot even memorize a small sample and should be repaired before scaling.",
        "- A large DT teacher-forced/autoregressive gap isolates sequence rollout or exposure-bias problems rather than basic optimization failure.",
        "- CQL is judged primarily through its offline-RL optimization trace; it is not expected to behave exactly like a supervised classifier.",
        "- Ghosting is judged against the no-movement position baseline because it predicts positions, not general football actions.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--bc-epochs", type=int, default=3_000)
    parser.add_argument("--cql-steps", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--dt-hidden-size", type=int, default=64)
    parser.add_argument("--context-length", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error(f"{args.output_dir} is not empty; choose a new diagnostic output directory")
    if args.epochs <= 0 or args.bc_epochs <= 0 or args.cql_steps <= 0 or args.batch_size <= 0:
        parser.error("epochs, bc-epochs, cql-steps, and batch-size must be positive")
    if args.dt_hidden_size % 4:
        parser.error("--dt-hidden-size must be divisible by 4")

    torch.set_num_threads(1)
    np.random.seed(args.seed)
    source_path = args.data.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = _slice_dataset(JointTransitionDataset.load(source_path), args.frames)
    tiny_path = output_dir / "tiny_dataset.npz"
    dataset.save(tiny_path)
    counts = np.bincount(dataset.actions.reshape(-1), minlength=19)
    report = {
        "purpose": "implementation diagnostic only; not a final experiment",
        "source_data": str(source_path),
        "tiny_data": str(tiny_path),
        "frames": len(dataset.actions),
        "player_samples": int(dataset.actions.size),
        "action_counts": {str(index): int(value) for index, value in enumerate(counts) if value},
        "configuration": {
            "epochs": args.epochs,
            "bc_epochs": args.bc_epochs,
            "cql_steps": args.cql_steps,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "hidden_size": args.hidden_size,
            "dt_hidden_size": args.dt_hidden_size,
            "context_length": min(args.context_length, len(dataset.actions)),
            "seed": args.seed,
        },
        "methods": {},
    }
    report["methods"]["bc"] = _run_bc(tiny_path, output_dir, dataset, args)
    report["methods"]["cql"] = _run_cql(tiny_path, output_dir, dataset, args)
    report["methods"]["dt"] = _run_dt(tiny_path, output_dir, dataset, args)
    report["methods"]["ghosting"] = _run_ghosting(tiny_path, output_dir, dataset, args)
    report["all_primary_gates_passed"] = bool(
        report["methods"]["bc"]["memorization_passed"]
        and report["methods"]["cql"]["optimization_passed"]
        and report["methods"]["dt"]["teacher_forced_memorization_passed"]
        and report["methods"]["ghosting"]["position_learning_passed"]
    )
    (output_dir / "tiny_overfit_results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output_dir / "TINY_OVERFIT_REPORT.md").write_text(_render_markdown(report), encoding="utf-8")
    print(json.dumps({
        "all_primary_gates_passed": report["all_primary_gates_passed"],
        "report": str(output_dir / "TINY_OVERFIT_REPORT.md"),
        "results": str(output_dir / "tiny_overfit_results.json"),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
