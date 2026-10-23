#!/usr/bin/env bash
# Train validation-selected trajectory CQL without opening the held-out test split.
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
METHODS_PYTHON="${METHODS_PYTHON:-$PROJECT_DIR/.venv-methods/bin/python}"
CQL_MANIFEST="${CQL_MANIFEST:-data/processed/final_stage_v1/team_00000P_manifest.json}"
CQL_CONTRACT="${CQL_CONTRACT:-thesis_plan/trajectory_evaluation_contract_v1.json}"
CQL_OUTPUT_ROOT="${CQL_OUTPUT_ROOT:-artifacts/vm_runs}"
CQL_RUN_NAME="${CQL_RUN_NAME:-trajectory_cql_pretest_v1}"
CQL_SEEDS="${CQL_SEEDS:-0}"
CQL_DEVICE="${CQL_DEVICE:-cpu}"
CQL_TORCH_THREADS="${CQL_TORCH_THREADS:-4}"
CQL_EPOCHS="${CQL_EPOCHS:-10}"
CQL_BATCH_SIZE="${CQL_BATCH_SIZE:-2048}"
CQL_LEARNING_RATE="${CQL_LEARNING_RATE:-3e-4}"
CQL_HIDDEN_SIZE="${CQL_HIDDEN_SIZE:-128}"
CQL_EARLY_STOPPING_PATIENCE="${CQL_EARLY_STOPPING_PATIENCE:-3}"
CQL_MINIMUM_EPOCHS="${CQL_MINIMUM_EPOCHS:-5}"
CQL_ACTION_COUNT="${CQL_ACTION_COUNT:-32}"
CQL_ALPHA="${CQL_ALPHA:-1.0}"
CQL_GAMMA="${CQL_GAMMA:-0.99}"
CQL_TARGET_UPDATE_INTERVAL="${CQL_TARGET_UPDATE_INTERVAL:-500}"
CQL_PROTOTYPE_SAMPLES="${CQL_PROTOTYPE_SAMPLES:-100000}"
CQL_KMEANS_ITERATIONS="${CQL_KMEANS_ITERATIONS:-15}"

if [[ ! -x "$METHODS_PYTHON" ]]; then
  echo "Trajectory Python not found: $METHODS_PYTHON" >&2
  exit 1
fi

if [[ "$CQL_DEVICE" == cuda* ]]; then
  "$METHODS_PYTHON" -c '
import sys
import torch

device = sys.argv[1]
if not torch.cuda.is_available():
    raise SystemExit(
        "CQL_DEVICE requests CUDA, but PyTorch cannot access an NVIDIA GPU. "
        "Check the NVIDIA driver and Docker GPU runtime."
    )
try:
    probe = torch.empty(1, device=device)
except Exception as exc:
    raise SystemExit(f"CUDA device probe failed for {device}: {exc}") from exc
print(f"CUDA ready for trajectory CQL: {torch.cuda.get_device_name(probe.device)} ({device})")
' "$CQL_DEVICE"
fi

cd "$PROJECT_DIR"
for seed in $CQL_SEEDS; do
  output_dir="$CQL_OUTPUT_ROOT/$CQL_RUN_NAME/seed_$seed"
  "$METHODS_PYTHON" -m thesis_experiments.scripts.run_trajectory_method \
    --method cql \
    --manifest "$CQL_MANIFEST" \
    --contract "$CQL_CONTRACT" \
    --device "$CQL_DEVICE" \
    --torch-threads "$CQL_TORCH_THREADS" \
    train \
    --output-dir "$output_dir" \
    --seed "$seed" \
    --epochs "$CQL_EPOCHS" \
    --batch-size "$CQL_BATCH_SIZE" \
    --learning-rate "$CQL_LEARNING_RATE" \
    --hidden-size "$CQL_HIDDEN_SIZE" \
    --early-stopping-patience "$CQL_EARLY_STOPPING_PATIENCE" \
    --minimum-epochs "$CQL_MINIMUM_EPOCHS" \
    --cql-action-count "$CQL_ACTION_COUNT" \
    --cql-alpha "$CQL_ALPHA" \
    --cql-gamma "$CQL_GAMMA" \
    --cql-target-update-interval "$CQL_TARGET_UPDATE_INTERVAL" \
    --cql-prototype-samples "$CQL_PROTOTYPE_SAMPLES" \
    --cql-kmeans-iterations "$CQL_KMEANS_ITERATIONS"
done

echo "Trajectory CQL pretest training completed: $CQL_OUTPUT_ROOT/$CQL_RUN_NAME"
