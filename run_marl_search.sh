#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-venv/bin/python}"
BASE_CONFIG="${BASE_CONFIG:-config/multi_car_marl_config.yaml}"
WARM_START_CKPT="${WARM_START_CKPT:-models/v3_marl_control_bootstrap/best_model_torch.pt}"
MODEL="${MODEL:-gpt-4o-mini}"
GENERATIONS="${GENERATIONS:-5}"
CANDIDATES_PER_BATCH="${CANDIDATES_PER_BATCH:-4}"
TIMEOUT="${TIMEOUT:-7200}"
SEED="${SEED:-42}"
SCREEN_TIMESTEPS="${SCREEN_TIMESTEPS:-100000}"
SCREEN_STAGE="${SCREEN_STAGE:-smoke_control}"
CONFIRM_TOP="${CONFIRM_TOP:-0}"
RESULTS_SUBDIR="${RESULTS_SUBDIR:-research_branches/marl_search_$(date +%Y%m%d_%H%M%S)}"
LOG_PATH="${LOG_PATH:-autoresearch/results/${RESULTS_SUBDIR}/launch.log}"

if [ ! -x "$PYTHON_BIN" ]; then
  echo "Missing Python interpreter: $PYTHON_BIN" >&2
  exit 1
fi

if [ ! -f "$BASE_CONFIG" ]; then
  echo "Base config not found: $BASE_CONFIG" >&2
  exit 1
fi

if [ ! -f "$WARM_START_CKPT" ]; then
  echo "Warm-start checkpoint not found: $WARM_START_CKPT" >&2
  exit 1
fi

if [ -z "${OPENAI_API_KEY:-}" ] && [ ! -f ".env" ] && [ ! -f "autoresearch/.env" ] && [ ! -f "autoresearch/results/.env" ]; then
  echo "OPENAI_API_KEY is not set and no .env found at repo root, autoresearch/.env, or autoresearch/results/.env" >&2
  exit 1
fi

mkdir -p "$(dirname "$LOG_PATH")"

nohup "$PYTHON_BIN" -m autoresearch.marl_search_loop \
  --provider openai \
  --model "$MODEL" \
  --warm-start-ckpt "$WARM_START_CKPT" \
  --base-config "$BASE_CONFIG" \
  --results-subdir "$RESULTS_SUBDIR" \
  --generations "$GENERATIONS" \
  --candidates-per-batch "$CANDIDATES_PER_BATCH" \
  --timeout "$TIMEOUT" \
  --seed "$SEED" \
  --screen-timesteps "$SCREEN_TIMESTEPS" \
  --screen-stage "$SCREEN_STAGE" \
  --confirm-top "$CONFIRM_TOP" \
  > "$LOG_PATH" 2>&1 &

PID="$!"

echo "Started marl_search_loop (fixed warm-start MARL search)."
echo "pid=$PID"
echo "results_subdir=$RESULTS_SUBDIR"
echo "log_path=$LOG_PATH"
echo "Promoted bundle: autoresearch/results/${RESULTS_SUBDIR}/promoted/"
