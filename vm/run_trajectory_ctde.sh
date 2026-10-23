#!/usr/bin/env bash
# Train validation-selected trajectory CTDE from frozen BC initializations.
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
METHODS_PYTHON="${METHODS_PYTHON:-$PROJECT_DIR/.venv-methods/bin/python}"
CTDE_MANIFEST="${CTDE_MANIFEST:-data/processed/final_stage_v1/team_00000P_manifest.json}"
CTDE_CONTRACT="${CTDE_CONTRACT:-thesis_plan/trajectory_evaluation_contract_v1.json}"
CTDE_BC_ROOT="${CTDE_BC_ROOT:-artifacts/single_team_multi_horizon_bc_full}"
CTDE_OUTPUT_ROOT="${CTDE_OUTPUT_ROOT:-artifacts/vm_runs}"
CTDE_RUN_NAME="${CTDE_RUN_NAME:-trajectory_ctde_pretest_v1}"
CTDE_SEEDS="${CTDE_SEEDS:-0}"
CTDE_DEVICE="${CTDE_DEVICE:-cpu}"
CTDE_TORCH_THREADS="${CTDE_TORCH_THREADS:-4}"
CTDE_ACTOR_EPOCHS="${CTDE_ACTOR_EPOCHS:-10}"
CTDE_CRITIC_EPOCHS="${CTDE_CRITIC_EPOCHS:-20}"
CTDE_BATCH_SIZE="${CTDE_BATCH_SIZE:-2048}"
CTDE_LEARNING_RATE="${CTDE_LEARNING_RATE:-3e-4}"
CTDE_ACTOR_HIDDEN_SIZE="${CTDE_ACTOR_HIDDEN_SIZE:-128}"
CTDE_CRITIC_HIDDEN_SIZE="${CTDE_CRITIC_HIDDEN_SIZE:-256}"
CTDE_EARLY_STOPPING_PATIENCE="${CTDE_EARLY_STOPPING_PATIENCE:-3}"
CTDE_MINIMUM_EPOCHS="${CTDE_MINIMUM_EPOCHS:-5}"
CTDE_GAMMA="${CTDE_GAMMA:-0.99}"
CTDE_TEMPERATURE="${CTDE_TEMPERATURE:-1.0}"
CTDE_MAX_WEIGHT="${CTDE_MAX_WEIGHT:-20.0}"
CTDE_BC_COEFFICIENT="${CTDE_BC_COEFFICIENT:-0.5}"

if [[ ! -x "$METHODS_PYTHON" ]]; then
  echo "Trajectory Python not found: $METHODS_PYTHON" >&2
  exit 1
fi

if [[ "$CTDE_DEVICE" == cuda* ]]; then
  "$METHODS_PYTHON" -c '
import sys
import torch

device = sys.argv[1]
if not torch.cuda.is_available():
    raise SystemExit(
        "CTDE_DEVICE requests CUDA, but PyTorch cannot access an NVIDIA GPU. "
        "Check the NVIDIA driver and Docker GPU runtime."
    )
try:
    probe = torch.empty(1, device=device)
except Exception as exc:
    raise SystemExit(f"CUDA device probe failed for {device}: {exc}") from exc
print(f"CUDA ready for trajectory CTDE: {torch.cuda.get_device_name(probe.device)} ({device})")
' "$CTDE_DEVICE"
fi

cd "$PROJECT_DIR"
for seed in $CTDE_SEEDS; do
  bc_checkpoint="$CTDE_BC_ROOT/seed_$seed/single_team_multi_horizon_bc.pt"
  if [[ ! -f "$bc_checkpoint" ]]; then
    echo "Frozen BC initialization not found: $bc_checkpoint" >&2
    exit 1
  fi
  output_dir="$CTDE_OUTPUT_ROOT/$CTDE_RUN_NAME/seed_$seed"
  "$METHODS_PYTHON" -m thesis_experiments.scripts.run_trajectory_method \
    --method ctde \
    --manifest "$CTDE_MANIFEST" \
    --contract "$CTDE_CONTRACT" \
    --device "$CTDE_DEVICE" \
    --torch-threads "$CTDE_TORCH_THREADS" \
    train \
    --output-dir "$output_dir" \
    --seed "$seed" \
    --epochs "$CTDE_ACTOR_EPOCHS" \
    --ctde-critic-epochs "$CTDE_CRITIC_EPOCHS" \
    --batch-size "$CTDE_BATCH_SIZE" \
    --learning-rate "$CTDE_LEARNING_RATE" \
    --hidden-size "$CTDE_ACTOR_HIDDEN_SIZE" \
    --ctde-critic-hidden-size "$CTDE_CRITIC_HIDDEN_SIZE" \
    --early-stopping-patience "$CTDE_EARLY_STOPPING_PATIENCE" \
    --minimum-epochs "$CTDE_MINIMUM_EPOCHS" \
    --ctde-bc-checkpoint "$bc_checkpoint" \
    --ctde-gamma "$CTDE_GAMMA" \
    --ctde-temperature "$CTDE_TEMPERATURE" \
    --ctde-max-weight "$CTDE_MAX_WEIGHT" \
    --ctde-bc-coefficient "$CTDE_BC_COEFFICIENT"
done

echo "Trajectory CTDE pretest training completed: $CTDE_OUTPUT_ROOT/$CTDE_RUN_NAME"
