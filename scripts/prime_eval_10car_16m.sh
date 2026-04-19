#!/usr/bin/env bash
# Evaluate the 10-car 16M run checkpoint.
# Default: no video (faster). Set PRIME_10CAR_EVAL_WITH_VIDEO=1 to keep video.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export RACING_HEADLESS_PYGLET="${RACING_HEADLESS_PYGLET:-1}"
# shellcheck source=/dev/null
source venv/bin/activate

MODEL_PATH="${PRIME_10CAR_EVAL_MODEL:-artifacts/prime_marl_10car_16m/models/best_model_torch.pt}"
if [[ ! -f "$MODEL_PATH" ]]; then
  echo "Model not found: $MODEL_PATH" >&2
  echo "Set PRIME_10CAR_EVAL_MODEL to a .pt checkpoint path." >&2
  exit 1
fi

ARGS=(
  --model "$MODEL_PATH"
  --config config/prime_marl_10car_16m.yaml
  --episodes "${PRIME_10CAR_EVAL_EPISODES:-5}"
  --seed "${PRIME_SEED:-42}"
)
if [[ "${PRIME_10CAR_EVAL_WITH_VIDEO:-0}" != "1" ]]; then
  ARGS+=(--no-video)
fi

if [[ -z "${DISPLAY:-}" ]] && command -v xvfb-run >/dev/null 2>&1; then
  exec xvfb-run -a python evaluate.py "${ARGS[@]}"
fi
exec python evaluate.py "${ARGS[@]}"
