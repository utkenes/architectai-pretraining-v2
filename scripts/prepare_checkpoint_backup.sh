#!/usr/bin/env bash
# Build a self-verifying checkpoint archive on the Vast instance.
set -euo pipefail

checkpoint="${1:?usage: prepare_checkpoint_backup.sh CHECKPOINT_DIR BACKUP_DIR}"
backup_dir="${2:?usage: prepare_checkpoint_backup.sh CHECKPOINT_DIR BACKUP_DIR}"

if [[ ! -d "$checkpoint" ]]; then
  echo "Checkpoint directory does not exist: $checkpoint" >&2
  exit 2
fi

checkpoint="$(cd "$checkpoint" && pwd)"
checkpoint_name="$(basename "$checkpoint")"
if [[ ! "$checkpoint_name" =~ ^checkpoint-([0-9]+)$ ]]; then
  echo "Checkpoint directory must be named checkpoint-N: $checkpoint" >&2
  exit 2
fi
expected_step="${BASH_REMATCH[1]}"

required=(checkpoint_metadata.json trainer_state.json optimizer.pt scheduler.pt model/adapter_config.json)
for relative_path in "${required[@]}"; do
  if [[ ! -f "$checkpoint/$relative_path" ]]; then
    echo "Required checkpoint file is missing: $relative_path" >&2
    exit 3
  fi
done
if [[ ! -f "$checkpoint/model/adapter_model.safetensors" && ! -f "$checkpoint/model/adapter_model.bin" ]]; then
  echo "Adapter weights are missing from $checkpoint/model" >&2
  exit 3
fi

metadata_step="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["global_step"])' "$checkpoint/checkpoint_metadata.json")"
trainer_step="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["global_step"])' "$checkpoint/trainer_state.json")"
if [[ "$metadata_step" != "$expected_step" || "$trainer_step" != "$expected_step" ]]; then
  echo "Checkpoint step mismatch: directory=$expected_step metadata=$metadata_step trainer=$trainer_step" >&2
  exit 4
fi

mkdir -p "$backup_dir"
archive="$backup_dir/$checkpoint_name.tar.gz"
file_manifest="$backup_dir/$checkpoint_name.files.sha256"
archive_manifest="$backup_dir/$checkpoint_name.archive.sha256"

(
  cd "$checkpoint"
  find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum
) > "$file_manifest"
tar -C "$(dirname "$checkpoint")" -czf "$archive" "$checkpoint_name"
sha256sum "$archive" > "$archive_manifest"

printf 'archive=%s\n' "$archive"
printf 'archive_sha256=%s\n' "$(cut -d ' ' -f1 "$archive_manifest")"
printf 'file_manifest=%s\n' "$file_manifest"
printf 'global_step=%s\n' "$expected_step"
