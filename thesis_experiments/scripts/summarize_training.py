"""Create a comparable CSV/JSON table from completed training-suite artifacts."""

import argparse
import csv
import json
from pathlib import Path


def summarize(output_dir):
    output_dir = Path(output_dir)
    manifest = json.loads((output_dir / "training_manifest.json").read_text())
    rows = []
    for run in manifest:
        directory = output_dir / f"seed_{run['seed']}" / "methods" / run["method"]
        row = {"seed": run["seed"], "method": run["method"], "status": run["status"]}
        for split in ("validation", "test", "external_test"):
            path = directory / f"{split}_evaluation.json"
            if path.exists():
                result = json.loads(path.read_text())
                offline = result["offline"]
                row[f"{split}_action_accuracy"] = offline["action_accuracy"]
                row[f"{split}_balanced_accuracy"] = offline["balanced_action_accuracy"]
                row[f"{split}_common_action_accuracy"] = offline.get("common_action_accuracy")
                row[f"{split}_common_balanced_accuracy"] = offline.get("common_balanced_action_accuracy")
                row[f"{split}_movement_accuracy"] = offline.get("movement_action_accuracy")
                row[f"{split}_rare_accuracy"] = offline.get("rare_action_accuracy")
                row[f"{split}_rare_samples"] = offline.get("rare_action_samples")
                row[f"{split}_majority_baseline"] = result["majority_baseline_accuracy"]
                majority = result.get("majority_baseline", {})
                row[f"{split}_majority_balanced_baseline"] = majority.get("balanced_action_accuracy")
                row[f"{split}_majority_common_balanced_baseline"] = majority.get(
                    "common_balanced_action_accuracy"
                )
                if "previous_action_baseline" in result:
                    persistence = result["previous_action_baseline"]
                    row[f"{split}_previous_action_baseline"] = persistence["action_accuracy"]
                    row[f"{split}_previous_action_balanced_baseline"] = persistence["balanced_action_accuracy"]
                    row[f"{split}_previous_action_common_balanced_baseline"] = persistence.get(
                        "common_balanced_action_accuracy"
                    )
                if "action_change_points" in result:
                    changes = result["action_change_points"]
                    row[f"{split}_action_change_accuracy"] = changes["action_accuracy"]
                    row[f"{split}_action_change_balanced_accuracy"] = changes["balanced_action_accuracy"]
                    row[f"{split}_action_change_common_balanced_accuracy"] = changes.get(
                        "common_balanced_action_accuracy"
                    )
                    row[f"{split}_action_change_samples"] = changes["samples"]
                for source, source_metrics in result.get("sources", {}).items():
                    prefix = f"{split}_{source}"
                    row[f"{prefix}_action_accuracy"] = source_metrics["action_accuracy"]
                    row[f"{prefix}_balanced_accuracy"] = source_metrics["balanced_action_accuracy"]
                    row[f"{prefix}_common_balanced_accuracy"] = source_metrics.get(
                        "common_balanced_action_accuracy"
                    )
                    source_changes = source_metrics.get("action_change_points", {})
                    row[f"{prefix}_action_change_accuracy"] = source_changes.get("action_accuracy")
                rollouts = [d["grf"] for d in result["datasets"] if "grf" in d]
                if rollouts:
                    row["grf_joint_steps"] = sum(d["steps"] for d in rollouts)
                    row["grf_mean_position_error"] = sum(d["steps"] * d["mean_position_error_grf"] for d in rollouts) / row["grf_joint_steps"]
                    row["grf_all_finite"] = all(d["finite"] for d in rollouts)
        rows.append(row)
    (output_dir / "comparison.json").write_text(json.dumps(rows, indent=2))
    with (output_dir / "comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
