#!/usr/bin/env bash
# Prepare a larger match-disjoint dataset and run all methods with three seeds.
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN_NAME="${RUN_NAME:-vm_training_20260924}"
FRAMES="${FRAMES:-3000}"
EPOCHS="${EPOCHS:-50}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-250000}"
SEEDS="${SEEDS:-0 1 2}"
BATCH_SIZE="${BATCH_SIZE:-512}"
HIDDEN_SIZE="${HIDDEN_SIZE:-256}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-256}"
EPISODE_STEPS="${EPISODE_STEPS:-512}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-32}"
EVAL_GRF_STEPS="${EVAL_GRF_STEPS:-500}"
DATA_DIR="${DATA_DIR:-data/processed/vm_training_${FRAMES}}"
SPLIT_MANIFEST="${SPLIT_MANIFEST:-split_manifest.json}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/vm_runs/${RUN_NAME}}"
MIN_FREE_GB="${MIN_FREE_GB:-5}"

cd "$PROJECT_DIR"

if [[ ! -x .venv-methods/bin/python || ! -x .venv-grf/bin/python ]]; then
  echo "Run vm/setup_ubuntu.sh before starting training." >&2
  exit 1
fi
export SDL_VIDEODRIVER=dummy
export SDL_AUDIODRIVER=dummy
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/thesis-matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/thesis-cache}"
export TORCH_HOME="${TORCH_HOME:-$XDG_CACHE_HOME/torch}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$TORCH_HOME" "${HOME:-/tmp/thesis-home}"

MANIFEST_PATH="$DATA_DIR/$SPLIT_MANIFEST"
if [[ ! -f "$MANIFEST_PATH" ]]; then
  echo "Training split not found: $MANIFEST_PATH" >&2
  echo "Mount or copy a prepared dataset under data/processed, or set DATA_DIR." >&2
  exit 1
fi

echo "Validating training data: $MANIFEST_PATH"
.venv-methods/bin/python -m thesis_experiments.scripts.validate_split_manifest \
  "$MANIFEST_PATH" --project-dir "$PROJECT_DIR"

mkdir -p "$OUTPUT_DIR"
if ! [[ "$MIN_FREE_GB" =~ ^[0-9]+$ ]]; then
  echo "MIN_FREE_GB must be a non-negative integer, got: $MIN_FREE_GB" >&2
  exit 2
fi
AVAILABLE_KB="$(df -Pk "$OUTPUT_DIR" | awk 'NR == 2 {print $4}')"
REQUIRED_KB="$((MIN_FREE_GB * 1024 * 1024))"
if (( AVAILABLE_KB < REQUIRED_KB )); then
  AVAILABLE_GB="$((AVAILABLE_KB / 1024 / 1024))"
  echo "Insufficient output space: ${AVAILABLE_GB} GiB available; ${MIN_FREE_GB} GiB required." >&2
  echo "Output directory: $OUTPUT_DIR" >&2
  exit 1
fi

DEVICE="${DEVICE:-auto}"
if [[ "$DEVICE" == "auto" ]]; then
  DEVICE="$(.venv-methods/bin/python -c 'import torch; print("cuda" if torch.cuda.is_available() else "cpu")')"
elif [[ "$DEVICE" == "cuda" ]]; then
  if ! .venv-methods/bin/python -c 'import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)'; then
    echo "DEVICE=cuda was requested, but PyTorch cannot access CUDA inside the container." >&2
    echo "Verify NVIDIA Container Toolkit/CDI, or run with TRAINING_DEVICE=cpu." >&2
    exit 1
  fi
fi
echo "Training device: $DEVICE"

.venv-methods/bin/python -m thesis_experiments.scripts.run_training_suite \
  --split-manifest "$MANIFEST_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --methods all \
  --seeds $SEEDS \
  --epochs "$EPOCHS" \
  --actor-epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --hidden-size "$HIDDEN_SIZE" \
  --total-timesteps "$TOTAL_TIMESTEPS" \
  --rollout-steps "$ROLLOUT_STEPS" \
  --episode-steps "$EPISODE_STEPS" \
  --context-length "$CONTEXT_LENGTH" \
  --bc-backend torch \
  --env-backend grf \
  --eval-grf-steps "$EVAL_GRF_STEPS" \
  --device "$DEVICE" \
  --grf-python "$PROJECT_DIR/.venv-grf/bin/python" \
  --continue-on-error \
  --resume

.venv-methods/bin/python -m thesis_experiments.scripts.summarize_training \
  --output-dir "$OUTPUT_DIR"

echo "Training complete: $OUTPUT_DIR"
