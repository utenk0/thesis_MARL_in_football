#!/usr/bin/env bash
# Verify hashes and reproduce frozen trajectory-BC predictions on this machine.
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN_NAME="${RUN_NAME:-trajectory_bc_portability_v1}"
FREEZE="${TRAJECTORY_BC_FREEZE:-thesis_plan/trajectory_bc_vm_freeze_v1.json}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/vm_runs/$RUN_NAME}"
PYTHON="${METHODS_PYTHON:-$PROJECT_DIR/.venv-methods/bin/python}"
TORCH_THREADS="${TORCH_THREADS:-1}"

cd "$PROJECT_DIR"
if [[ ! -x "$PYTHON" ]]; then
  echo "Methods environment not found: $PYTHON" >&2
  echo "Run bash vm/setup_ubuntu.sh first, or set METHODS_PYTHON." >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
"$PYTHON" -m thesis_experiments.scripts.verify_trajectory_bc_portability \
  --freeze "$FREEZE" \
  --project-dir "$PROJECT_DIR" \
  --output "$OUTPUT_DIR/portability_report.json" \
  --device cpu \
  --torch-threads "$TORCH_THREADS" \
  | tee "$OUTPUT_DIR/console.log"

echo "Trajectory BC portability check passed: $OUTPUT_DIR/portability_report.json"
