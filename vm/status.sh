#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN_NAME="${1:-${RUN_NAME:-team_00000P_closed_loop_full}}"
STATE_DIR="$PROJECT_DIR/artifacts/vm_runs/$RUN_NAME"

if [[ -f "$STATE_DIR/training.pid" ]]; then
  TRAINING_PID="$(tr -cd '0-9' < "$STATE_DIR/training.pid")"
  if [[ -n "$TRAINING_PID" ]] && kill -0 "$TRAINING_PID" 2>/dev/null; then
    echo "running (PID $TRAINING_PID)"
  else
    echo "not running (last PID ${TRAINING_PID:-unknown})"
  fi
else
  echo "not started"
fi

if [[ -f "$STATE_DIR/training_manifest.json" ]]; then
  "$PROJECT_DIR/.venv-methods/bin/python" - "$STATE_DIR/training_manifest.json" <<'PY'
import json, sys
from collections import Counter
items = json.load(open(sys.argv[1]))
print("manifest:", dict(Counter(item["status"] for item in items)))
for item in items:
    print(f"  seed={item['seed']} method={item['method']} status={item['status']}")
PY
fi

if [[ -f "$STATE_DIR/console.log" ]]; then
  echo "last console lines:"
  tail -n 12 "$STATE_DIR/console.log"
fi
