"""Aggregate repeated-seed direct-displacement BC experiments."""

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


def summarize(paths: list[Path]) -> dict[str, object]:
    runs = [json.loads(Path(path).read_text()) for path in paths]
    if len(runs) < 2:
        raise ValueError("At least two repeated-seed results are required.")
    teams = {run["team_id"] for run in runs}
    splits = {json.dumps(run["split_matches"], sort_keys=True) for run in runs}
    variants = {run["training"]["observation_variant"] for run in runs}
    targets = {run["training"]["target"] for run in runs}
    if any(len(values) != 1 for values in (teams, splits, variants, targets)):
        raise ValueError("All runs must share team, split, observation, and target contracts.")

    one_step = {}
    for split in ("train", "validation", "test"):
        rows = [run["evaluations"][split]["overall"] for run in runs]
        one_step[split] = {
            predictor: {
                metric: _stats([float(row[predictor][metric]) for row in rows])
                for metric in (
                    "mean_displacement_error_meters",
                    "root_mean_squared_displacement_error_meters",
                )
            }
            for predictor in (
                "policy",
                "stationary_baseline",
                "previous_velocity_baseline",
            )
        }
        one_step[split]["samples"] = rows[0]["policy"]["samples"]

    cumulative = {}
    test_rows = [
        run["evaluations"]["test"]["teacher_forced_cumulative"] for run in runs
    ]
    for predictor in ("policy", "stationary", "previous_velocity"):
        cumulative[predictor] = {
            metric: _stats([float(row[predictor][metric]) for row in test_rows])
            for metric in ("ade_meters", "fde_meters")
        }
        cumulative[predictor]["windows"] = test_rows[0][predictor]["windows"]
        cumulative[predictor]["horizon_steps"] = test_rows[0][predictor][
            "horizon_steps"
        ]

    validation = one_step["validation"]
    test = one_step["test"]
    gates = {
        "finite": all(
            np.isfinite(run["evaluations"][split]["overall"]["policy"]["mean_displacement_error_meters"])
            for run in runs
            for split in ("train", "validation", "test")
        ),
        "validation_beats_stationary": (
            validation["policy"]["mean_displacement_error_meters"]["mean"]
            < validation["stationary_baseline"]["mean_displacement_error_meters"]["mean"]
        ),
        "validation_beats_previous_velocity": (
            validation["policy"]["mean_displacement_error_meters"]["mean"]
            < validation["previous_velocity_baseline"]["mean_displacement_error_meters"]["mean"]
        ),
        "test_beats_stationary_report_only": (
            test["policy"]["mean_displacement_error_meters"]["mean"]
            < test["stationary_baseline"]["mean_displacement_error_meters"]["mean"]
        ),
        "test_beats_previous_velocity_report_only": (
            test["policy"]["mean_displacement_error_meters"]["mean"]
            < test["previous_velocity_baseline"]["mean_displacement_error_meters"]["mean"]
        ),
        "test_used_for_model_selection": False,
    }
    return {
        "method": "direct_displacement_bc",
        "team_id": next(iter(teams)),
        "observation_variant": next(iter(variants)),
        "target": next(iter(targets)),
        "seeds": [int(run["training"]["seed"]) for run in runs],
        "selected_epochs": [int(run["training"]["selected_epoch"]) for run in runs],
        "selection_metric": "validation_mean_displacement_error_meters",
        "result_paths": [str(path) for path in paths],
        "one_step": one_step,
        "test_teacher_forced_cumulative": cumulative,
        "gates": gates,
        "decision": (
            "Direct displacement BC is a functional supervised trajectory target. "
            "It is accepted as a reconstruction model only if it improves on both "
            "stationary and previous-velocity baselines on validation; test remains "
            "report-only. Teacher-forced cumulative metrics do not establish closed-loop "
            "game reconstruction."
        ),
    }


def write_report(result: dict[str, object], path: Path) -> None:
    lines = [
        "# Direct displacement BC result",
        "",
        f"- Team: `{result['team_id']}`",
        f"- Target: `{result['target']}`",
        f"- Observation: `{result['observation_variant']}`",
        f"- Seeds: {', '.join(map(str, result['seeds']))}",
        f"- Validation-selected epochs: {', '.join(map(str, result['selected_epochs']))}",
        "- Test data: report-only",
        "",
        "## One-step displacement error",
        "",
        "| Split | Policy (m) | Stationary (m) | Previous velocity (m) | Policy RMSE (m) |",
        "|---|---:|---:|---:|---:|",
    ]
    for split in ("train", "validation", "test"):
        row = result["one_step"][split]
        policy = row["policy"]["mean_displacement_error_meters"]
        stationary = row["stationary_baseline"]["mean_displacement_error_meters"]
        velocity = row["previous_velocity_baseline"]["mean_displacement_error_meters"]
        rmse = row["policy"]["root_mean_squared_displacement_error_meters"]
        lines.append(
            f"| {split} | {policy['mean']:.4f} ± {policy['std']:.4f} | "
            f"{stationary['mean']:.4f} ± {stationary['std']:.4f} | "
            f"{velocity['mean']:.4f} ± {velocity['std']:.4f} | "
            f"{rmse['mean']:.4f} ± {rmse['std']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Test 100-step teacher-forced cumulative displacement",
            "",
            "| Predictor | ADE (m) | FDE (m) | Windows |",
            "|---|---:|---:|---:|",
        ]
    )
    for predictor, row in result["test_teacher_forced_cumulative"].items():
        lines.append(
            f"| {predictor} | {row['ade_meters']['mean']:.4f} ± {row['ade_meters']['std']:.4f} | "
            f"{row['fde_meters']['mean']:.4f} ± {row['fde_meters']['std']:.4f} | "
            f"{row['windows']} |"
        )
    gates = result["gates"]
    lines.extend(
        [
            "",
            "## Gate decision",
            "",
            f"- {'PASS' if gates['finite'] else 'FAIL'} — finite predictions",
            f"- {'PASS' if gates['validation_beats_stationary'] else 'FAIL'} — validation improvement over stationary",
            f"- {'PASS' if gates['validation_beats_previous_velocity'] else 'FAIL'} — validation improvement over previous velocity",
            "",
            result["decision"],
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    result = summarize(args.results)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2))
    write_report(result, args.output_dir / "DISPLACEMENT_BC_REPORT.md")
    print(
        json.dumps(
            {
                "summary": str(args.output_dir / "summary.json"),
                "report": str(args.output_dir / "DISPLACEMENT_BC_REPORT.md"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
