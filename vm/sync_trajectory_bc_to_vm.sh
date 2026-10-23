#!/usr/bin/env bash
# Copy the frozen trajectory BC, one validation chunk, and source to a VM.
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 USER@HOST [REMOTE_PROJECT_DIR]" >&2
  exit 2
fi

REMOTE="$1"
REMOTE_DIR="${2:-thesis_MARL_in_football}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FREEZE_RELATIVE="thesis_plan/trajectory_bc_vm_freeze_v1.json"
FREEZE_PATH="$PROJECT_DIR/$FREEZE_RELATIVE"
FILE_LIST="$(mktemp)"
trap 'rm -f "$FILE_LIST"' EXIT

python3 "$PROJECT_DIR/thesis_experiments/scripts/list_frozen_bundle_files.py" \
  "$FREEZE_PATH" --project-dir "$PROJECT_DIR" > "$FILE_LIST"

ssh "$REMOTE" "mkdir -p '$REMOTE_DIR'"
RSYNC_EXCLUDES=(
  '--exclude=.git/'
  # Preserve machine-specific environments whether they are directories or
  # symlinks (a trailing slash only matches directories in older rsync).
  '--exclude=.venv*'
  '--exclude=.pytest_cache/'
  '--exclude=__pycache__/'
  '--exclude=.DS_Store'
  '--exclude=artifacts/'
  '--exclude=data/processed/'
  '--exclude=data/raw/'
)
rsync -az --progress --partial \
  "${RSYNC_EXCLUDES[@]}" \
  "$PROJECT_DIR/" "$REMOTE:$REMOTE_DIR/"

rsync -az --progress --partial --files-from="$FILE_LIST" \
  "$PROJECT_DIR/" "$REMOTE:$REMOTE_DIR/"

echo "Synced the frozen trajectory BC portability bundle."
echo "Destination: $REMOTE:$REMOTE_DIR"
echo "Next: ssh $REMOTE \"cd '$REMOTE_DIR' && bash vm/run_trajectory_bc_portability.sh\""
