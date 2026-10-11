#!/usr/bin/env bash
set -u -o pipefail
ROOT=/workspace/architectai-pretraining-v2
OUTPUT="$ROOT/outputs/dapt-v3-1epoch"
STATE=/workspace/architectai-hf-backup-state.txt
LOG=/workspace/architectai-hf-backup.log
REPO_ID=babkindo/architectai-qwen3-8b-dapt-backups
mkdir -p "$(dirname "$STATE")"
touch "$STATE"
printf '%s watcher started pid=%s\n' "$(date -Is)" "$$" >>"$LOG"
while true; do
  for checkpoint in "$OUTPUT"/checkpoint-*; do
    [ -d "$checkpoint" ] || continue
    name=$(basename "$checkpoint")
    if grep -Fxq "$name" "$STATE"; then
      continue
    fi
    if /venv/main/bin/python "$ROOT/scripts/publish_hf_backup.py" --root "$ROOT" --checkpoint "$checkpoint" --repo-id "$REPO_ID" >>"$LOG" 2>&1; then
      printf '%s\n' "$name" >>"$STATE"
      printf '%s uploaded %s\n' "$(date -Is)" "$name" >>"$LOG"
    else
      printf '%s upload failed %s; will retry\n' "$(date -Is)" "$name" >>"$LOG"
    fi
  done
  sleep 60
done
