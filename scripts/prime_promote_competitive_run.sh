#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# shellcheck source=/dev/null
source venv/bin/activate
export RACING_HEADLESS_PYGLET="${RACING_HEADLESS_PYGLET:-1}"

WINNER_PATH="artifacts/competitive_tuning/winner.json"
PROMOTE_SEED="${PRIME_PROMOTE_SEED:-42}"
PROMOTE_EVAL_EPISODES="${PRIME_PROMOTE_EVAL_EPISODES:-8}"

if [[ ! -f "$WINNER_PATH" ]]; then
  echo "Missing winner file: $WINNER_PATH" >&2
  echo "Run scripts/prime_select_competitive_winner.sh first." >&2
  exit 1
fi

WINNER_CONFIG="$(python - "$WINNER_PATH" <<'PY'
import json
import sys
with open(sys.argv[1], "r") as f:
    data = json.load(f)
print(data["winner"]["base_config"])
PY
)"

run_python() {
  if [[ -z "${DISPLAY:-}" ]] && command -v xvfb-run >/dev/null 2>&1; then
    xvfb-run -a python "$@"
  else
    python "$@"
  fi
}

echo "Promoting config: ${WINNER_CONFIG}"
run_python train.py --config "$WINNER_CONFIG" --seed "$PROMOTE_SEED"

RESULTS_DIR="$(python - "$WINNER_CONFIG" <<'PY'
import sys
import yaml
cfg = yaml.safe_load(open(sys.argv[1], "r"))
print(cfg["paths"]["results_dir"])
PY
)"
MODEL_DIR="$(python - "$WINNER_CONFIG" <<'PY'
import sys
import yaml
cfg = yaml.safe_load(open(sys.argv[1], "r"))
print(cfg["paths"]["model_dir"])
PY
)"

mkdir -p "$RESULTS_DIR"
MODEL_PATH="${MODEL_DIR}/best_model_torch.pt"
if [[ ! -f "$MODEL_PATH" ]]; then
  MODEL_PATH="${MODEL_DIR}/final_model_torch.pt"
fi

run_python evaluate.py \
  --model "$MODEL_PATH" \
  --config "$WINNER_CONFIG" \
  --episodes "$PROMOTE_EVAL_EPISODES" \
  --seed "$PROMOTE_SEED" \
  --output-json "${RESULTS_DIR}/promoted_final_eval_seed${PROMOTE_SEED}.json" \
  --output-video "${RESULTS_DIR}/promoted_final_eval_seed${PROMOTE_SEED}.mp4"

echo "Promoted run complete."
