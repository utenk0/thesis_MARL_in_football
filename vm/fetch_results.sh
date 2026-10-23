#!/usr/bin/env bash
# Fetch training artifacts without modifying or deleting the VM copy.
set -euo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
  echo "Usage: $0 USER@HOST [REMOTE_PROJECT_DIR] [RUN_NAME]" >&2
  exit 2
fi

REMOTE="$1"
REMOTE_DIR="${2:-thesis_MARL_in_football}"
RUN_NAME="${3:-team_00000P_closed_loop_full}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCAL_DIR="$PROJECT_DIR/artifacts/vm_runs/$RUN_NAME"

mkdir -p "$LOCAL_DIR"
rsync -az --progress --partial \
  "$REMOTE:$REMOTE_DIR/artifacts/vm_runs/$RUN_NAME/" "$LOCAL_DIR/"
echo "Fetched results to $LOCAL_DIR"
