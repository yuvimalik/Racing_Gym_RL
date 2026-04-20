#!/usr/bin/env bash
# Export two 5-loop race videos from a trained 10-car checkpoint:
#  1) tiled all-cars view
#  2) broadcast reference-car view
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export RACING_HEADLESS_PYGLET="${RACING_HEADLESS_PYGLET:-1}"
# shellcheck source=/dev/null
source venv/bin/activate

MODEL_PATH="${PRIME_10CAR_EVAL_MODEL:-artifacts/prime_marl_10car_16m/models/best_model_torch.pt}"
CONFIG_PATH="${PRIME_10CAR_EVAL_CONFIG:-config/prime_marl_10car_16m.yaml}"
LOOPS="${PRIME_10CAR_VIDEO_LOOPS:-5}"
MAX_STEPS="${PRIME_10CAR_VIDEO_MAX_STEPS:-120000}"
REF_CAR="${PRIME_10CAR_BROADCAST_REF_CAR:-0}"
SEED="${PRIME_SEED:-42}"
OUT_DIR="${PRIME_10CAR_VIDEO_OUT_DIR:-artifacts/prime_marl_10car_16m/results/5loop_videos}"

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "Model not found: $MODEL_PATH" >&2
  echo "Set PRIME_10CAR_EVAL_MODEL to a valid .pt checkpoint path." >&2
  exit 1
fi
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config not found: $CONFIG_PATH" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

run_eval() {
  local mode="$1"
  local out_video="$2"
  local out_json="$3"
  if [[ -z "${DISPLAY:-}" ]] && command -v xvfb-run >/dev/null 2>&1; then
    xvfb-run -a python evaluate.py       --model "$MODEL_PATH"       --config "$CONFIG_PATH"       --seed "$SEED"       --target-loops "$LOOPS"       --max-steps "$MAX_STEPS"       --camera-mode "$mode"       --reference-car "$REF_CAR"       --output-video "$out_video"       --output-json "$out_json"
  else
    python evaluate.py       --model "$MODEL_PATH"       --config "$CONFIG_PATH"       --seed "$SEED"       --target-loops "$LOOPS"       --max-steps "$MAX_STEPS"       --camera-mode "$mode"       --reference-car "$REF_CAR"       --output-video "$out_video"       --output-json "$out_json"
  fi
}

echo "Exporting tiled 5-loop video..."
run_eval   "tiled"   "$OUT_DIR/ten_car_5loops_tiled_seed${SEED}.mp4"   "$OUT_DIR/ten_car_5loops_tiled_seed${SEED}.json"

echo "Exporting broadcast 5-loop video (reference car $REF_CAR)..."
run_eval   "broadcast"   "$OUT_DIR/ten_car_5loops_broadcast_seed${SEED}_car${REF_CAR}.mp4"   "$OUT_DIR/ten_car_5loops_broadcast_seed${SEED}_car${REF_CAR}.json"

echo "Done. Videos + JSON summaries saved under: $OUT_DIR"
