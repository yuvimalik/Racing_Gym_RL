#!/usr/bin/env bash
# Launch 10-car 16M policy-only resume run under nohup.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

ARTIFACT_ROOT="${PRIME_10CAR_ARTIFACT_ROOT:-artifacts/prime_marl_10car_16m}"
LOG="${PRIME_10CAR_LOG:-$ARTIFACT_ROOT/nohup_train.log}"
PID_FILE="${PRIME_10CAR_PID_FILE:-$ARTIFACT_ROOT/train.pid}"

mkdir -p "$ARTIFACT_ROOT"

if [[ -f "$PID_FILE" ]] && [[ -z "${PRIME_10CAR_FORCE:-}" ]]; then
  echo "Refusing to start: $PID_FILE exists. Remove it or set PRIME_10CAR_FORCE=1." >&2
  exit 1
fi

CHECKPOINT="${PRIME_10CAR_RESUME_CHECKPOINT:-artifacts_prime_intellect_pull/prime_marl_2car_budget_fast/models/final_model_torch.pt}"
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Resume checkpoint not found: $CHECKPOINT" >&2
  echo "Set PRIME_10CAR_RESUME_CHECKPOINT to an existing .pt file." >&2
  exit 1
fi

rm -f "$PID_FILE"
echo "Logging to: $LOG"
nohup env PRIME_10CAR_RESUME_CHECKPOINT="$CHECKPOINT" bash scripts/prime_train_10car_16m_policy_resume.sh "$@" >>"$LOG" 2>&1 &
echo $! >"$PID_FILE"
echo "Started PID $(cat "$PID_FILE") using checkpoint: $CHECKPOINT"
echo "Monitor: tail -f \"$LOG\""
echo "Process: ps -p $(cat "$PID_FILE") -o pid,etime,cmd"
