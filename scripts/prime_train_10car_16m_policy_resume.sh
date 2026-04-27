#!/usr/bin/env bash
# 10-car MARL run (10M target) with policy-only warm start.
# Usage:
#   PRIME_10CAR_RESUME_CHECKPOINT=/path/to/best_model_torch.pt bash scripts/prime_train_10car_16m_policy_resume.sh
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export RACING_HEADLESS_PYGLET="${RACING_HEADLESS_PYGLET:-1}"
# shellcheck source=/dev/null
source venv/bin/activate

CHECKPOINT="${PRIME_10CAR_RESUME_CHECKPOINT:-artifacts_prime_intellect_pull/prime_marl_2car_budget_fast/models/final_model_torch.pt}"
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Resume checkpoint not found: $CHECKPOINT" >&2
  echo "Set PRIME_10CAR_RESUME_CHECKPOINT to your best .pt file before launching." >&2
  exit 1
fi

if [[ -z "${DISPLAY:-}" ]] && command -v xvfb-run >/dev/null 2>&1; then
  exec xvfb-run -a python train.py \
    --config config/prime_marl_10car_16m.yaml \
    --seed "${PRIME_SEED:-42}" \
    --resume "$CHECKPOINT" \
    --resume_mode policy_only \
    "$@"
fi
exec python train.py \
  --config config/prime_marl_10car_16m.yaml \
  --seed "${PRIME_SEED:-42}" \
  --resume "$CHECKPOINT" \
  --resume_mode policy_only \
  "$@"
