#!/usr/bin/env bash
# Start ~16h wall-clock 2-car MARL on Prime (see config/prime_marl_2car_budget_fast.yaml: 8M stream steps).
# Run from repo root on the GPU instance after venv + deps are installed.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

ARTIFACT_ROOT="${PRIME_16H_ARTIFACT_ROOT:-artifacts/prime_marl_2car_budget_fast}"
LOG="${PRIME_16H_LOG:-$ARTIFACT_ROOT/nohup_train.log}"

mkdir -p "$ARTIFACT_ROOT"

if [[ -f "${PRIME_16H_PID_FILE:-$ARTIFACT_ROOT/train.pid}" ]] && [[ -z "${PRIME_16H_FORCE:-}" ]]; then
  echo "Refusing to start: ${PRIME_16H_PID_FILE:-$ARTIFACT_ROOT/train.pid} exists. Remove it or set PRIME_16H_FORCE=1." >&2
  exit 1
fi

PID_FILE="${PRIME_16H_PID_FILE:-$ARTIFACT_ROOT/train.pid}"
rm -f "$PID_FILE"

echo "Logging to: $LOG"
nohup bash scripts/prime_train_budget_fast.sh "$@" >>"$LOG" 2>&1 &
echo $! >"$PID_FILE"
echo "Started PID $(cat "$PID_FILE") (stream budget: 8_000_000 in YAML)."
echo "Monitor: tail -f \"$LOG\""
echo "Process: ps -p \$(cat \"$PID_FILE\") -o pid,etime,cmd"
