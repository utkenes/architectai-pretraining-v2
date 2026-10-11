#!/usr/bin/env bash
set -u -o pipefail
ROOT=/workspace/architectai-pretraining-v2
RESULTS="$ROOT/data/benchmark/results/dapt_v3_2epoch_final"
STATE=/workspace/architectai-hf-gate-results-uploaded
LOG=/workspace/architectai-hf-gate-results.log
if [ -f "$STATE" ]; then exit 0; fi
while true; do
  if [ -f "$RESULTS/quality_gate.json" ]; then
    if /venv/main/bin/python "$ROOT/scripts/publish_hf_gate_results.py" --results "$RESULTS" >>"$LOG" 2>&1; then
      date -Is > "$STATE"
      exit 0
    fi
  fi
  sleep 60
done
