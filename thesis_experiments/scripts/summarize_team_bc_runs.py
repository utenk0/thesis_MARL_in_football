"""Aggregate repeated-seed single-team BC results and make the two-part gate decision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "min": float(array.min()),
        "max": float(array.max()),
    }


def summarize(
    paths: list[Path], rollout_evaluation_paths: list[Path] | None = None
) -> dict[str, object]:
    runs = [json.loads(path.read_text()) for path in paths]
    if len(runs) < 2:
        raise ValueError("At least two repeated-seed results are required.")
    teams = {run["team_id"] for run in runs}
    variants = {run["training"]["observation_variant"] for run in runs}
    split_matches = {json.dumps(run["split_matches"], sort_keys=True) for run in runs}
    if len(teams) != 1 or len(variants) != 1 or len(split_matches) != 1:
        raise ValueError("All runs must use the same team, representation and splits.")

    offline = {}
    for split in ("train", "validation", "test"):
        rows = [run["evaluations"][split]["overall"] for run in runs]
        offline[split] = {
            "action_accuracy": _stats([row["policy"]["action_accuracy"] for row in rows]),
            "balanced_action_accuracy": _stats(
                [row["policy"]["balanced_action_accuracy"] for row in rows]
            ),
            "action_change_accuracy": _stats(
                [row["action_change_policy"]["action_accuracy"] for row in rows]
            ),
            "majority_baseline_accuracy": rows[0]["training_majority_baseline"]["action_accuracy"],
            "change_majority_baseline_accuracy": rows[0]["action_change_training_majority_baseline"]["action_accuracy"],
            "previous_action_baseline_accuracy": rows[0]["previous_action_baseline"]["action_accuracy"],
            "samples": rows[0]["policy"]["samples"],
            "action_change_samples": rows[0]["action_change_samples"],
        }

    if rollout_evaluation_paths is not None:
        if len(rollout_evaluation_paths) != len(runs):
            raise ValueError("Provide one rollout evaluation per repeated-seed result.")
        rollout_runs = [json.loads(path.read_text()) for path in rollout_evaluation_paths]
    else:
        rollout_runs = [run["conditional_reconstructions"] for run in runs]
    controllers = sorted({row["focal_control"] for rows in rollout_runs for row in rows})
    trajectory = {}
    for controller in controllers:
        per_seed = []
        for rows in rollout_runs:
            selected = [row for row in rows if row["focal_control"] == controller]
            if not selected:
                raise ValueError(f"Missing {controller!r} rollout for one seed.")
            per_seed.append(selected)
        trajectory[controller] = {
            metric: _stats(
                [
                    float(np.mean([row["metrics"][metric] for row in rows]))
                    for rows in per_seed
                ]
            )
            for metric in (
                "ade_meters",
                "fde_meters",
                "centroid_error_meters",
                "team_width_mae_meters",
                "team_depth_mae_meters",
            )
        }

    validation = offline["validation"]
    action_gate = all(run["pilot_gate"]["passed"] for run in runs)
    trajectory_gate = (
        "bc_policy" in trajectory
        and "previous_action" in trajectory
        and trajectory["bc_policy"]["ade_meters"]["mean"]
        < trajectory["previous_action"]["ade_meters"]["mean"]
    )
    return {
        "team_id": next(iter(teams)),
        "observation_variant": next(iter(variants)),
        "seeds": [int(run["training"].get("seed", index)) for index, run in enumerate(runs)],
        "result_paths": [str(path) for path in paths],
        "rollout_evaluation_paths": (
            [str(path) for path in rollout_evaluation_paths]
            if rollout_evaluation_paths is not None
            else None
        ),
        "trajectory_chunks_per_seed": len(rollout_runs[0]) // max(1, len(controllers)),
        "selected_epochs": [run["training"]["selected_epoch"] for run in runs],
        "offline": offline,
        "conditional_trajectory": trajectory,
        "gates": {
            "action_imitation_passed": action_gate,
            "trajectory_reconstruction_passed": trajectory_gate,
            "validation_used_for_model_selection": True,
            "test_used_for_model_selection": False,
            "validation_policy_beats_majority": (
                validation["action_accuracy"]["mean"]
                > validation["majority_baseline_accuracy"]
            ),
            "validation_change_policy_beats_majority": (
                validation["action_change_accuracy"]["mean"]
                > validation["change_majority_baseline_accuracy"]
            ),
        },
        "decision": (
            "BC is a defensible inferred-action imitation baseline, but it does not pass "
            "the trajectory-reconstruction gate. Do not claim game reconstruction from "
            "the discrete BC rollouts. Add a direct displacement/trajectory target (or a "
            "dynamics-aware policy) before treating trajectory quality as solved."
        ),
    }


def write_markdown(result: dict[str, object], path: Path) -> None:
    offline = result["offline"]
    lines = [
        "# Full single-team BC result",
        "",
        f"- Team: `{result['team_id']}`",
        f"- Observation: `{result['observation_variant']}`",
        f"- Validation-selected epochs: {', '.join(map(str, result['selected_epochs']))}",
        f"- Held-out trajectory chunks per seed: {result['trajectory_chunks_per_seed']}",
        "",
        "## Offline inferred-action imitation",
        "",
        "| Split | Accuracy | Balanced | Action-change | Majority | Change majority | Previous action |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split in ("train", "validation", "test"):
        row = offline[split]
        lines.append(
            f"| {split} | {row['action_accuracy']['mean']:.4f} ± {row['action_accuracy']['std']:.4f} | "
            f"{row['balanced_action_accuracy']['mean']:.4f} ± {row['balanced_action_accuracy']['std']:.4f} | "
            f"{row['action_change_accuracy']['mean']:.4f} ± {row['action_change_accuracy']['std']:.4f} | "
            f"{row['majority_baseline_accuracy']:.4f} | {row['change_majority_baseline_accuracy']:.4f} | "
            f"{row['previous_action_baseline_accuracy']:.4f} |"
        )
    lines.extend([
        "",
        "## Conditional 100-step GRF reconstruction",
        "",
        "| Focal controller | ADE (m) | FDE (m) | Centroid error (m) | Width MAE (m) | Depth MAE (m) |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for controller, metrics in result["conditional_trajectory"].items():
        lines.append(
            f"| {controller} | {metrics['ade_meters']['mean']:.3f} ± {metrics['ade_meters']['std']:.3f} | "
            f"{metrics['fde_meters']['mean']:.3f} ± {metrics['fde_meters']['std']:.3f} | "
            f"{metrics['centroid_error_meters']['mean']:.3f} ± {metrics['centroid_error_meters']['std']:.3f} | "
            f"{metrics['team_width_mae_meters']['mean']:.3f} ± {metrics['team_width_mae_meters']['std']:.3f} | "
            f"{metrics['team_depth_mae_meters']['mean']:.3f} ± {metrics['team_depth_mae_meters']['std']:.3f} |"
        )
    gates = result["gates"]
    lines.extend([
        "",
        "## Gate decision",
        "",
        f"- {'PASS' if gates['action_imitation_passed'] else 'FAIL'} — inferred-action imitation",
        f"- {'PASS' if gates['trajectory_reconstruction_passed'] else 'FAIL'} — trajectory reconstruction versus persistence",
        "",
        result["decision"],
    ])
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, nargs="+", type=Path)
    parser.add_argument("--rollout-evaluations", nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    result = summarize(args.results, args.rollout_evaluations)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "summary.json"
    report_path = args.output_dir / "FULL_BC_REPORT.md"
    json_path.write_text(json.dumps(result, indent=2))
    write_markdown(result, report_path)
    print(json.dumps({"summary": str(json_path), "report": str(report_path)}, indent=2))


if __name__ == "__main__":
    main()
