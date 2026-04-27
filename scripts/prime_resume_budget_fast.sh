#!/usr/bin/env bash
# Resume 2-car budget_fast training and add stream timesteps (for retargeting wall clock).
# Example:
#   PRIME_RESUME_CHECKPOINT=artifacts/prime_marl_2car_budget_fast/models/best_model_torch.pt \
#   PRIME_TIMESTEPS_ADD=2000000 \
#   bash scripts/prime_resume_budget_fast.sh
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export RACING_HEADLESS_PYGLET="${RACING_HEADLESS_PYGLET:-1}"
# shellcheck source=/dev/null
source venv/bin/activate

CHECKPOINT="${PRIME_RESUME_CHECKPOINT:-artifacts/prime_marl_2car_budget_fast/models/best_model_torch.pt}"
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Checkpoint not found: $CHECKPOINT" >&2
  echo "Set PRIME_RESUME_CHECKPOINT to a .pt file under artifacts/prime_marl_2car_budget_fast/models/." >&2
  exit 1
fi

if [[ -z "${PRIME_TIMESTEPS_ADD:-}" ]]; then
  echo "Set PRIME_TIMESTEPS_ADD to positive stream timesteps to add (e.g. 2000000)." >&2
  exit 1
fi

if [[ -z "${DISPLAY:-}" ]] && command -v xvfb-run >/dev/null 2>&1; then
  exec xvfb-run -a python train.py \
    --config config/prime_marl_2car_budget_fast.yaml \
    --seed "${PRIME_SEED:-42}" \
    --resume "$CHECKPOINT" \
    --timesteps_add "$PRIME_TIMESTEPS_ADD" \
    "$@"
fi
exec python train.py \
  --config config/prime_marl_2car_budget_fast.yaml \
  --seed "${PRIME_SEED:-42}" \
  --resume "$CHECKPOINT" \
  --timesteps_add "$PRIME_TIMESTEPS_ADD" \
  "$@"
