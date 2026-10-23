#!/usr/bin/env bash
# Start the VM training job in the background and preserve a PID and console log.
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN_NAME="${RUN_NAME:-team_00000P_closed_loop_full}"
STATE_DIR="$PROJECT_DIR/artifacts/vm_runs/$RUN_NAME"
PID_FILE="$STATE_DIR/training.pid"
CONSOLE_LOG="$STATE_DIR/console.log"

mkdir -p "$STATE_DIR"
if [[ -f "$PID_FILE" ]]; then
  EXISTING_PID="$(tr -cd '0-9' < "$PID_FILE")"
  if [[ -n "$EXISTING_PID" ]] && kill -0 "$EXISTING_PID" 2>/dev/null; then
    echo "Training is already running with PID $EXISTING_PID"
    exit 0
  fi
fi

cd "$PROJECT_DIR"
nohup env PROJECT_DIR="$PROJECT_DIR" RUN_NAME="$RUN_NAME" \
  bash vm/run_training.sh >> "$CONSOLE_LOG" 2>&1 &
TRAINING_PID=$!
printf '%s\n' "$TRAINING_PID" > "$PID_FILE"
echo "Started PID $TRAINING_PID"
echo "Console: $CONSOLE_LOG"
echo "Status:  bash vm/status.sh $RUN_NAME"
