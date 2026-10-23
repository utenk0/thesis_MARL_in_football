"""Audit whether a completed method run trained, reloads, and behaves as claimed.

This is intentionally stricter than checking ``status == "completed"``.  It
combines orchestration evidence, checkpoint reload/inference, optimization
evidence specific to each method, and held-out evaluation diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np

from thesis_experiments.policies.loading import load_method_policy
from thesis_experiments.transitions.schema import JointTransitionDataset


METHODS = ("bc", "gail", "ppo", "cql", "dt", "ghosting", "ctde")

METHOD_DESCRIPTIONS = {
    "bc": {
        "family": "supervised imitation learning",
        "training_input": "61-feature local player observation and inferred expert action",
        "objective": "cross-entropy classification of the expert action",
        "policy_output": "one shared 19-action policy used by every player",
        "implementation": "project PyTorch MLP (the trial explicitly selected backend=torch)",
    },
    "gail": {
        "family": "adversarial imitation learning with online policy optimization",
        "training_input": "offline expert transitions plus live 22-player GRF rollouts",
        "objective": "discriminator-derived reward optimized by a shared PPO generator",
        "policy_output": "one shared 19-action PPO policy used by every player",
        "implementation": "imitation GAIL + Stable-Baselines3 PPO",
    },
    "ppo": {
        "family": "online reinforcement learning",
        "training_input": "61-feature local observations and rewards from live GRF rollouts",
        "objective": "clipped PPO policy/value objective",
        "policy_output": "one shared 19-action policy used by every player",
        "implementation": "Stable-Baselines3 PPO over 22 coupled player streams",
    },
    "cql": {
        "family": "offline reinforcement learning",
        "training_input": "recorded observation, action, reward, next-observation transitions",
        "objective": "discrete conservative Q-learning",
        "policy_output": "the action with the highest learned conservative Q-value",
        "implementation": "d3rlpy DiscreteCQL",
    },
    "dt": {
        "family": "offline sequence modelling",
        "training_input": "return-to-go, observation and previous-action sequences",
        "objective": "causal next-action cross-entropy",
        "policy_output": "a return-conditioned 19-action sequence policy",
        "implementation": "project PyTorch Decision Transformer",
    },
    "ghosting": {
        "family": "supervised trajectory prediction baseline",
        "training_input": "local player observation and next player position",
        "objective": "next-position mean squared error",
        "policy_output": "movement action inferred from predicted displacement",
        "implementation": "project PyTorch displacement MLP",
    },
    "ctde": {
        "family": "centralized training with decentralized execution",
        "training_input": "global team state/reward for the critic and local observations/actions for the actor",
        "objective": "centralized TD critic plus advantage-weighted BC actor fine-tuning",
        "policy_output": "a BC-initialized shared 19-action actor used without the critic at execution",
        "implementation": "project PyTorch centralized critic and shared actor",
    },
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _artifact_path(method_dir: Path, result: dict[str, Any]) -> Path:
    artifact = result.get("artifacts", {}).get("checkpoint")
    if artifact is None:
        artifact = result.get("artifacts", {}).get("policy")
    if artifact is None:
        raise ValueError(f"{method_dir}: result has no checkpoint/policy artifact")
    # Server-side absolute/relative prefixes are deliberately ignored.  A run
    # directory remains portable after vm/fetch_results.sh downloads it.
    return method_dir / Path(artifact).name


def _resolve_local_path(value: str, *, project_dir: Path) -> Path | None:
    raw = Path(value)
    for candidate in (raw, project_dir / raw):
        if candidate.exists():
            return candidate.resolve()
    return None


def _infer_sample_data(config: dict[str, Any], *, project_dir: Path) -> Path:
    paths = config.get("data_paths", [])
    if not paths:
        raise ValueError("Method config does not contain data_paths")
    training = _resolve_local_path(paths[0], project_dir=project_dir)
    if training is None:
        raise FileNotFoundError(paths[0])
    dataset_root = training.parent.parent
    candidates = sorted((dataset_root / "test").glob("*.npz"))
    if not candidates:
        candidates = sorted((dataset_root / "validation").glob("*.npz"))
    if not candidates:
        raise FileNotFoundError(f"No held-out .npz file below {dataset_root}")
    return candidates[0]


def _predict(
    policy: Any,
    method: str,
    observations: np.ndarray,
    team_rewards: np.ndarray | None = None,
) -> np.ndarray:
    if hasattr(policy, "reset"):
        policy.reset()
    if method == "dt":
        predictions = []
        for index, frame in enumerate(observations):
            predictions.append(policy.act(frame))
            if team_rewards is not None and hasattr(policy, "observe_rewards"):
                policy.observe_rewards(np.repeat(team_rewards[index], 11))
        return np.asarray(predictions, dtype=np.int64)
    flat = observations.reshape(-1, observations.shape[-1])
    chunks = [
        np.asarray(policy.act(flat[start : start + 4096]), dtype=np.int64)
        for start in range(0, len(flat), 4096)
    ]
    return np.concatenate(chunks).reshape(observations.shape[:2])


def _prediction_summary(predictions: np.ndarray) -> dict[str, Any]:
    flat = np.asarray(predictions, dtype=np.int64).reshape(-1)
    counts = np.bincount(flat, minlength=19)
    probabilities = counts[counts > 0] / len(flat)
    return {
        "samples": int(len(flat)),
        "valid_action_range": bool(np.all((0 <= flat) & (flat < 19))),
        "unique_actions": int(np.count_nonzero(counts)),
        "dominant_action": int(counts.argmax()),
        "dominant_action_share": float(counts.max() / len(flat)),
        "entropy_nats": float(-(probabilities * np.log(probabilities)).sum()),
        "action_counts": counts.tolist(),
        "consecutive_change_rate": float(np.mean(predictions[1:] != predictions[:-1]))
        if len(predictions) > 1
        else None,
    }


def _relative_state_delta(left: dict[str, Any], right: dict[str, Any]) -> float:
    import torch

    squared_difference = torch.tensor(0.0)
    squared_reference = torch.tensor(0.0)
    for name, tensor in left.items():
        comparison = right[name].detach().cpu()
        reference = tensor.detach().cpu()
        squared_difference += ((reference - comparison) ** 2).sum()
        squared_reference += (reference**2).sum()
    return float(torch.sqrt(squared_difference) / torch.sqrt(squared_reference).clamp_min(1e-12))


def _tensor_state_norm(state: dict[str, Any]) -> float:
    import torch

    squared = torch.tensor(0.0)
    for value in state.values():
        if torch.is_tensor(value):
            squared += (value.detach().cpu().float() ** 2).sum()
    return float(torch.sqrt(squared))


def _csv_trend(path: Path) -> dict[str, Any]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.reader(stream))
    values = [float(row[-1]) for row in rows if row]
    if not values:
        raise ValueError(f"No metric rows in {path}")
    return {
        "updates_logged": len(values),
        "initial": values[0],
        "final": values[-1],
        "finite": bool(np.isfinite(values).all()),
        "decreased": bool(values[-1] < values[0]),
    }


def _gail_log_evidence(log_path: Path) -> dict[str, Any]:
    text = log_path.read_text(encoding="utf-8", errors="replace")

    def values(name: str) -> list[float]:
        pattern = rf"{re.escape(name)}\s*\|\s*([-+0-9.eE]+)"
        return [float(value) for value in re.findall(pattern, text)]

    expert = values("disc/disc_acc_expert")
    generated = values("disc/disc_acc_gen")
    total = values("disc/disc_acc")
    loss = values("disc/disc_loss")
    return {
        "discriminator_updates_logged": len(loss),
        "final_discriminator_accuracy": total[-1] if total else None,
        "final_expert_accuracy": expert[-1] if expert else None,
        "final_generator_accuracy": generated[-1] if generated else None,
        "initial_discriminator_loss": loss[0] if loss else None,
        "final_discriminator_loss": loss[-1] if loss else None,
    }


def _optimization_evidence(
    method: str,
    policy: Any,
    checkpoint: Path,
    method_dir: Path,
    config: dict[str, Any],
    sample: JointTransitionDataset,
    run_dir: Path,
) -> tuple[dict[str, Any], list[str]]:
    import torch

    evidence: dict[str, Any] = {}
    warnings: list[str] = []
    if method == "bc":
        from thesis_experiments.policies.networks import SharedActor

        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        torch.manual_seed(int(config["seed"]))
        initial = SharedActor(
            int(saved["observation_size"]), int(saved["action_size"]), int(saved["hidden_size"])
        )
        delta = _relative_state_delta(initial.state_dict(), saved["actor_state_dict"])
        evidence.update(actor_relative_change_from_initialization=delta, actor_weights_changed=delta > 1e-8)
    elif method in {"ppo", "gail"}:
        evidence.update(
            saved_environment_steps=int(policy.model.num_timesteps),
            saved_optimizer_updates=int(getattr(policy.model, "_n_updates", 0)),
        )
        if method == "gail":
            reward_path = method_dir / "gail_reward_state.pt"
            reward_state = torch.load(reward_path, map_location="cpu", weights_only=True)
            evidence["reward_network_state_norm"] = _tensor_state_norm(reward_state)
            evidence.update(_gail_log_evidence(run_dir / "logs" / f"seed_{config['seed']}_gail.log"))
            if evidence.get("final_expert_accuracy") is not None and evidence["final_expert_accuracy"] < 0.1:
                warnings.append(
                    "The final GAIL discriminator labels almost every expert sample incorrectly; "
                    "the adversarial game has not reached a useful balance in this short trial."
                )
    elif method == "cql":
        evidence["loss"] = _csv_trend(method_dir / "d3rlpy_logs" / "cql" / "loss.csv")
        evidence["td_loss"] = _csv_trend(method_dir / "d3rlpy_logs" / "cql" / "td_loss.csv")
        evidence["conservative_loss"] = _csv_trend(
            method_dir / "d3rlpy_logs" / "cql" / "conservative_loss.csv"
        )
    elif method == "dt":
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        history = [float(value) for value in saved.get("loss_history", [])]
        evidence.update(
            epochs_logged=len(history),
            loss_history=history,
            loss_finite=bool(history and np.isfinite(history).all()),
            loss_decreased=bool(history and history[-1] < history[0]),
        )
    elif method == "ghosting":
        observations = sample.local_observations.reshape(-1, sample.local_observations.shape[-1])
        targets = sample.next_local_observations.reshape(-1, sample.next_local_observations.shape[-1])[:, :2]
        with torch.no_grad():
            predicted = policy.model(torch.from_numpy(observations).float()).numpy()
        model_mse = float(np.mean((predicted - targets) ** 2))
        persistence_mse = float(np.mean((observations[:, :2] - targets) ** 2))
        evidence.update(
            held_out_position_mse=model_mse,
            held_out_no_movement_mse=persistence_mse,
            improvement_over_no_movement_percent=(
                100.0 * (1.0 - model_mse / persistence_mse) if persistence_mse > 0 else None
            ),
        )
        if persistence_mse > 0 and model_mse >= persistence_mse:
            warnings.append("Ghosting does not beat the no-movement position baseline on the sampled clip.")
    elif method == "ctde":
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        bc_saved = torch.load(method_dir.parent / "bc" / "shared_actor_bc.pt", map_location="cpu", weights_only=True)
        delta = _relative_state_delta(bc_saved["actor_state_dict"], saved["actor_state_dict"])
        evidence.update(
            actor_relative_change_from_bc=delta,
            actor_changed_from_bc=delta > 1e-8,
            centralized_critic_state_norm=_tensor_state_norm(saved["critic_state_dict"]),
        )
        if abs(float(_read_json(method_dir / "result.json")["metrics"]["mean_advantage"])) < 1e-4:
            warnings.append(
                "CTDE's mean advantage is close to zero, so this trial provides weak evidence that "
                "the centralized critic supplied a useful fine-tuning signal."
            )
    return evidence, warnings


def _evaluation_summary(method_dir: Path, evaluation_dir: Path | None = None) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    evaluation_dir = evaluation_dir or method_dir
    for split in ("validation", "test", "external_test"):
        path = evaluation_dir / f"{split}_evaluation.json"
        if not path.exists():
            continue
        report = _read_json(path)
        offline = report["offline"]
        change = report.get("action_change_points") or {}
        previous = report.get("previous_action_baseline") or {}
        summary[split] = {
            "checkpoint_reloaded": bool(report.get("checkpoint_reloaded")),
            "action_accuracy": offline.get("action_accuracy"),
            "balanced_action_accuracy": offline.get("balanced_action_accuracy"),
            "rare_action_accuracy": offline.get("rare_action_accuracy"),
            "predicted_rare_actions": offline.get("predicted_rare_actions"),
            "previous_action_accuracy": previous.get("action_accuracy"),
            "action_change_accuracy": change.get("action_accuracy"),
            "action_change_balanced_accuracy": change.get("balanced_action_accuracy"),
        }
    return summary


def verify_run(
    run_dir: Path,
    *,
    project_dir: Path,
    sample_data: Path | None = None,
    sample_frames: int = 128,
    evaluation_overrides: dict[str, Path] | None = None,
) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    project_dir = project_dir.resolve()
    manifest = _read_json(run_dir / "training_manifest.json")
    manifest_by_method = {row["method"]: row for row in manifest}
    methods_root = run_dir / "seed_0" / "methods"
    first_config = _read_json(methods_root / METHODS[0] / "config.json")
    sample_path = sample_data.resolve() if sample_data else _infer_sample_data(first_config, project_dir=project_dir)
    sample = JointTransitionDataset.load(sample_path)
    frame_count = min(sample_frames, len(sample.actions))
    observations = sample.local_observations[:frame_count]
    expert_actions = sample.actions[:frame_count]
    method_reports: dict[str, Any] = {}
    predictions_by_method: dict[str, np.ndarray] = {}
    evaluation_overrides = evaluation_overrides or {}

    for method in METHODS:
        method_dir = methods_root / method
        result = _read_json(method_dir / "result.json")
        config = _read_json(method_dir / "config.json")
        row = manifest_by_method.get(method, {})
        checkpoint = _artifact_path(method_dir, result)
        execution = {
            "manifest_completed": row.get("status") == "completed",
            "return_code_zero": row.get("returncode") == 0,
            "result_trained": result.get("status") == "trained",
            "checkpoint_exists": checkpoint.is_file(),
            "checkpoint_bytes": checkpoint.stat().st_size if checkpoint.is_file() else 0,
        }
        warnings: list[str] = []
        reload_error = None
        inference: dict[str, Any] = {}
        optimization: dict[str, Any] = {}
        try:
            policy = load_method_policy(method_dir / "result.json", device="cpu")
            rewards = sample.team_rewards[:frame_count]
            predictions = _predict(policy, method, observations, rewards)
            repeated = _predict(policy, method, observations, rewards)
            predictions_by_method[method] = predictions
            inference = _prediction_summary(predictions)
            inference.update(
                checkpoint_reloaded=True,
                deterministic_repeat=bool(np.array_equal(predictions, repeated)),
                sampled_action_accuracy=float(np.mean(predictions == expert_actions)),
                expert_consecutive_change_rate=float(np.mean(expert_actions[1:] != expert_actions[:-1]))
                if frame_count > 1
                else None,
            )
            optimization, method_warnings = _optimization_evidence(
                method, policy, checkpoint, method_dir, config, sample, run_dir
            )
            warnings.extend(method_warnings)
        except Exception as exc:  # Report all methods even if one backend is broken.
            reload_error = f"{type(exc).__name__}: {exc}"
            inference = {"checkpoint_reloaded": False}

        evaluation_dir = evaluation_overrides.get(method)
        evaluation = _evaluation_summary(method_dir, evaluation_dir)
        test = evaluation.get("test", {})
        if test.get("previous_action_accuracy") is not None and test.get("action_accuracy") is not None:
            test["gain_over_previous_action_baseline"] = (
                test["action_accuracy"] - test["previous_action_accuracy"]
            )
        if (test.get("rare_action_accuracy") or 0.0) < 0.05:
            warnings.append("Held-out rare-action accuracy is below 5%; movement accuracy dominates the total score.")
        if method == "ghosting":
            warnings.append("Ghosting is a next-position baseline, not a general 19-action policy learner.")
        if method == "dt":
            warnings.append("This is the project's compact DT implementation, not an external maintained DT backend.")
        if method == "ppo":
            warnings.append(
                "The 22 players are exposed as coupled shared-policy streams; describe this as parameter-sharing PPO/IPPO-style training."
            )
        method_reports[method] = {
            **METHOD_DESCRIPTIONS[method],
            "backend": result.get("backend"),
            "execution": execution,
            "inference": inference,
            "reload_error": reload_error,
            "optimization_evidence": optimization,
            "evaluation": evaluation,
            "evaluation_source": str((evaluation_dir or method_dir).resolve()),
            "warnings": warnings,
        }

    if "bc" in predictions_by_method and "ctde" in predictions_by_method:
        method_reports["ctde"]["optimization_evidence"]["sampled_action_disagreement_with_bc"] = float(
            np.mean(predictions_by_method["bc"] != predictions_by_method["ctde"])
        )

    operational = all(
        all(report["execution"].values())
        and report["inference"].get("checkpoint_reloaded")
        and report["inference"].get("valid_action_range")
        and report["inference"].get("deterministic_repeat")
        for report in method_reports.values()
    )
    return {
        "run_dir": str(run_dir),
        "sample_data": str(sample_path),
        "sample_frames": frame_count,
        "methods_expected": list(METHODS),
        "operational_verification_passed": operational,
        "scientific_validation_complete": False,
        "interpretation": (
            "All seven methods are operationally verified only if the first flag is true. "
            "This does not establish that they learned good football behaviour or that the final comparison is valid."
        ),
        "methods": method_reports,
        "evaluation_overrides": {
            method: str(path.resolve()) for method, path in evaluation_overrides.items()
        },
        "global_limitations": [
            "This is one seed and a short trial budget, not a final experiment.",
            "Raw action accuracy is strongly inflated by action persistence; action-change and balanced metrics remain essential.",
            "Rare football events have very low support and cannot yet be judged reliably.",
            "The methods use different learning objectives, so an epoch or training step is not directly comparable across all seven.",
        ],
    }


def _percent(value: Any) -> str:
    return "—" if value is None else f"{100.0 * float(value):.2f}%"


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Method verification report",
        "",
        f"Run: `{report['run_dir']}`  ",
        f"Held-out sample: `{report['sample_data']}` ({report['sample_frames']} frames)",
        "",
        "## Verdict",
        "",
        (
            "**Operational verification: PASS.** Every method completed, produced a non-empty checkpoint, "
            "reloaded independently, and returned deterministic actions in the valid GRF range."
            if report["operational_verification_passed"]
            else "**Operational verification: FAIL.** At least one method did not satisfy the artifact/reload checks."
        ),
        "",
        "**Scientific validation: NOT COMPLETE.** The trial proves wiring and execution, not good football behaviour.",
        "",
        "| Method | What is actually trained | Reload | Optimization evidence | Test accuracy | Test change accuracy |",
        "|---|---|---:|---|---:|---:|",
    ]
    for method in METHODS:
        row = report["methods"][method]
        reload_ok = "PASS" if row["inference"].get("checkpoint_reloaded") else "FAIL"
        evidence = row["optimization_evidence"]
        if method == "bc":
            opt = f"weights changed: {evidence.get('actor_weights_changed', False)}"
        elif method in {"ppo", "gail"}:
            opt = f"{evidence.get('saved_optimizer_updates', 0)} PPO updates"
        elif method == "cql":
            loss = evidence.get("loss", {})
            opt = f"loss {loss.get('initial', math.nan):.3f} → {loss.get('final', math.nan):.3f}"
        elif method == "dt":
            history = evidence.get("loss_history", [])
            opt = f"loss {history[0]:.3f} → {history[-1]:.3f}" if history else "no loss history"
        elif method == "ghosting":
            opt = f"{evidence.get('improvement_over_no_movement_percent', math.nan):.2f}% vs persistence"
        else:
            opt = f"actor changed from BC: {evidence.get('actor_changed_from_bc', False)}"
        test = row["evaluation"].get("test", {})
        lines.append(
            f"| {method.upper()} | {row['objective']} | {reload_ok} | {opt} | "
            f"{_percent(test.get('action_accuracy'))} | {_percent(test.get('action_change_accuracy'))} |"
        )

    lines.extend(["", "## What each method means in this repository", ""])
    for method in METHODS:
        row = report["methods"][method]
        lines.extend(
            [
                f"### {method.upper()}",
                "",
                f"- Family: {row['family']}.",
                f"- Input: {row['training_input']}.",
                f"- Objective: {row['objective']}.",
                f"- Output: {row['policy_output']}.",
                f"- Code/backend: {row['implementation']}.",
            ]
        )
        if row["warnings"]:
            lines.append("- Current cautions: " + " ".join(row["warnings"]))
        lines.append("")

    lines.extend(["## Global limitations", ""])
    lines.extend(f"- {item}" for item in report["global_limitations"])
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "The software path is ready for method-level diagnostic experiments. It is not yet ready for the final assessment run. "
            "The next gate is to inspect learning curves and method-specific behaviour, repair the GAIL discriminator issue, and "
            "decide how rare actions and the persistence baseline will be handled before running multiple seeds.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--project-dir", type=Path, default=Path.cwd())
    parser.add_argument("--sample-data", type=Path)
    parser.add_argument("--sample-frames", type=int, default=128)
    parser.add_argument(
        "--evaluation-override",
        action="append",
        default=[],
        metavar="METHOD=DIR",
        help="Use repaired evaluation JSON files from DIR for one method.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.sample_frames <= 0:
        parser.error("--sample-frames must be positive")
    overrides: dict[str, Path] = {}
    for value in args.evaluation_override:
        if "=" not in value:
            parser.error("--evaluation-override must have the form METHOD=DIR")
        method, directory = value.split("=", 1)
        if method not in METHODS:
            parser.error(f"Unknown evaluation override method: {method}")
        overrides[method] = Path(directory)
    report = verify_run(
        args.run_dir,
        project_dir=args.project_dir,
        sample_data=args.sample_data,
        sample_frames=args.sample_frames,
        evaluation_overrides=overrides,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "method_verification.json"
    markdown_path = args.output_dir / "METHOD_VERIFICATION.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({
        "operational_verification_passed": report["operational_verification_passed"],
        "scientific_validation_complete": report["scientific_validation_complete"],
        "json": str(json_path),
        "markdown": str(markdown_path),
    }, indent=2))


if __name__ == "__main__":
    main()
