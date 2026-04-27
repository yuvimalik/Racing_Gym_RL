#!/usr/bin/env bash
# Generate all evaluation video styles supported by evaluate.py for a 10-car Torch checkpoint
# (multi-episode rollouts + optional target-loops rollouts; tiled + broadcast per reference car).
#
# Usage (from repo root, venv active):
#   PRIME_10CAR_EVAL_MODEL=/path/to/best_model_torch.pt bash scripts/export_prime_10car_all_videos.sh
#
# On Linux headless, set RACING_HEADLESS_PYGLET=1 (default below). On macOS, omit it for local display.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

UNAME_S="$(uname -s 2>/dev/null || echo unknown)"
if [[ "$UNAME_S" == Darwin* ]]; then
  export RACING_HEADLESS_PYGLET="${RACING_HEADLESS_PYGLET:-0}"
else
  export RACING_HEADLESS_PYGLET="${RACING_HEADLESS_PYGLET:-1}"
fi

# shellcheck source=/dev/null
source venv/bin/activate

MODEL_PATH="${PRIME_10CAR_EVAL_MODEL:-artifacts/prime_marl_10car_16m/models/best_model_torch.pt}"
CONFIG_PATH="${PRIME_10CAR_EVAL_CONFIG:-config/prime_marl_10car_16m.yaml}"
SEED="${PRIME_SEED:-42}"
EPISODES="${PRIME_10CAR_ALLVID_EPISODES:-5}"
TARGET_LOOPS="${PRIME_10CAR_ALLVID_TARGET_LOOPS:-5}"
MAX_STEPS="${PRIME_10CAR_ALLVID_MAX_STEPS:-120000}"
OUT_DIR="${PRIME_10CAR_ALLVID_OUT_DIR:-artifacts/prime_marl_10car_16m/results/all_eval_videos}"
FIRST_CAR="${PRIME_10CAR_BROADCAST_FIRST_CAR:-0}"
LAST_CAR="${PRIME_10CAR_BROADCAST_LAST_CAR:-9}"
RUN_EPISODE_VIDEOS="${PRIME_10CAR_ALLVID_RUN_EPISODES:-1}"
RUN_LOOP_VIDEOS="${PRIME_10CAR_ALLVID_RUN_LOOPS:-1}"

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "Model not found: $MODEL_PATH" >&2
  echo "Set PRIME_10CAR_EVAL_MODEL to your saved best_model_torch.pt (or final_model_torch.pt)." >&2
  exit 1
fi
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config not found: $CONFIG_PATH" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

py_eval() {
  if [[ -z "${DISPLAY:-}" ]] && command -v xvfb-run >/dev/null 2>&1; then
    xvfb-run -a python evaluate.py "$@"
  else
    python evaluate.py "$@"
  fi
}

run_episode_mode() {
  local mode="$1"
  local ref="$2"
  local tag="$3"
  py_eval \
    --model "$MODEL_PATH" \
    --config "$CONFIG_PATH" \
    --seed "$SEED" \
    --episodes "$EPISODES" \
    --camera-mode "$mode" \
    --reference-car "$ref" \
    --output-video "$OUT_DIR/allvids_ep${EPISODES}_seed${SEED}_${tag}.mp4" \
    --output-json "$OUT_DIR/allvids_ep${EPISODES}_seed${SEED}_${tag}.json"
}

run_loop_mode() {
  local mode="$1"
  local ref="$2"
  local tag="$3"
  py_eval \
    --model "$MODEL_PATH" \
    --config "$CONFIG_PATH" \
    --seed "$SEED" \
    --target-loops "$TARGET_LOOPS" \
    --max-steps "$MAX_STEPS" \
    --camera-mode "$mode" \
    --reference-car "$ref" \
    --output-video "$OUT_DIR/allvids_loops${TARGET_LOOPS}_seed${SEED}_${tag}.mp4" \
    --output-json "$OUT_DIR/allvids_loops${TARGET_LOOPS}_seed${SEED}_${tag}.json"
}

echo "Output directory: $OUT_DIR"
echo "Model: $MODEL_PATH"

if [[ "$RUN_EPISODE_VIDEOS" == "1" ]]; then
  echo "=== Multi-episode (${EPISODES} eps), tiled ==="
  run_episode_mode "tiled" 0 "tiled"
  for car in $(seq "$FIRST_CAR" "$LAST_CAR"); do
    echo "=== Multi-episode (${EPISODES} eps), broadcast car ${car} ==="
    run_episode_mode "broadcast" "$car" "broadcast_car${car}"
  done
fi

if [[ "$RUN_LOOP_VIDEOS" == "1" ]]; then
  echo "=== Target-loops (${TARGET_LOOPS}), tiled ==="
  run_loop_mode "tiled" 0 "tiled"
  for car in $(seq "$FIRST_CAR" "$LAST_CAR"); do
    echo "=== Target-loops (${TARGET_LOOPS}), broadcast car ${car} ==="
    run_loop_mode "broadcast" "$car" "broadcast_car${car}"
  done
fi

echo "Done. MP4 + JSON files under: $OUT_DIR"
