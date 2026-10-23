#!/usr/bin/env bash
# Copy source and one selected prepared dataset to an SSH-accessible machine.
# Set SYNC_METHOD=ctde to include the frozen BC initialization bundle.
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 USER@HOST [REMOTE_PROJECT_DIR]" >&2
  exit 2
fi

REMOTE="$1"
REMOTE_DIR="${2:-thesis_MARL_in_football}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROCESSED_DATASET="${SYNC_PROCESSED_DATASET:-final_stage_v1}"
SPLIT_MANIFEST="${SYNC_SPLIT_MANIFEST:-team_00000P_manifest.json}"
SYNC_METHOD="${SYNC_METHOD:-cql}"
MANIFEST_PATH="$PROJECT_DIR/data/processed/$PROCESSED_DATASET/$SPLIT_MANIFEST"

case "$SYNC_METHOD" in
  cql|ctde|legacy) ;;
  *)
    echo "SYNC_METHOD must be one of: cql, ctde, legacy (got: $SYNC_METHOD)" >&2
    exit 2
    ;;
esac

if [[ ! -f "$MANIFEST_PATH" ]]; then
  echo "Prepared manifest not found: data/processed/$PROCESSED_DATASET/$SPLIT_MANIFEST" >&2
  exit 1
fi
FILE_LIST="$(mktemp)"
trap 'rm -f "$FILE_LIST"' EXIT
python3 "$PROJECT_DIR/thesis_experiments/scripts/list_manifest_files.py" \
  "$MANIFEST_PATH" --project-dir "$PROJECT_DIR" > "$FILE_LIST"

if [[ "$SYNC_METHOD" == "ctde" ]]; then
  FREEZE_PATH="$PROJECT_DIR/thesis_plan/trajectory_bc_vm_freeze_v1.json"
  if [[ ! -f "$FREEZE_PATH" ]]; then
    echo "Frozen BC manifest not found: $FREEZE_PATH" >&2
    exit 1
  fi
  python3 "$PROJECT_DIR/thesis_experiments/scripts/list_frozen_bundle_files.py" \
    "$FREEZE_PATH" --project-dir "$PROJECT_DIR" >> "$FILE_LIST"
  sort -u "$FILE_LIST" -o "$FILE_LIST"
fi

ssh "$REMOTE" "mkdir -p '$REMOTE_DIR'"
RSYNC_EXCLUDES=(
  '--exclude=.git/'
  '--exclude=.venv*/'
  '--exclude=.pytest_cache/'
  '--exclude=__pycache__/'
  '--exclude=.DS_Store'
  '--exclude=artifacts/'
  '--exclude=data/processed/'
)
if [[ "${SYNC_RAW_DATA:-0}" != "1" ]]; then
  RSYNC_EXCLUDES+=('--exclude=data/raw/')
fi
rsync -az --progress --partial \
  "${RSYNC_EXCLUDES[@]}" \
  "$PROJECT_DIR/" "$REMOTE:$REMOTE_DIR/"

rsync -az --progress --partial --files-from="$FILE_LIST" \
  "$PROJECT_DIR/" "$REMOTE:$REMOTE_DIR/"

echo "Synced project and files referenced by data/processed/$PROCESSED_DATASET/$SPLIT_MANIFEST"
echo "Deployment method: $SYNC_METHOD"
if [[ "$SYNC_METHOD" == "ctde" ]]; then
  echo "Included the hash-frozen BC checkpoints and metadata required by CTDE."
fi
echo "Destination: $REMOTE:$REMOTE_DIR"
if [[ "${SYNC_RAW_DATA:-0}" != "1" ]]; then
  echo "Raw datasets were omitted; set SYNC_RAW_DATA=1 to include them."
fi
echo "Next: configure docker/server.env on $REMOTE and validate the Compose configuration."
