#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-venv/bin/python}"
BASE_CONFIG="${BASE_CONFIG:-config/multi_car_marl_anti_contact_4m_config.yaml}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-models/v4_marl_anti_contact_4m/best_model_torch.pt}"
MODEL="${MODEL:-gpt-4o-mini}"
GENERATIONS="${GENERATIONS:-4}"
CANDIDATES_PER_BATCH="${CANDIDATES_PER_BATCH:-3}"
TIMEOUT="${TIMEOUT:-7200}"
SEED="${SEED:-42}"
MODE="${MODE:-fully_autonomous}"
RESULTS_SUBDIR="${RESULTS_SUBDIR:-research_branches/v4_autoresearch_gpt4omini_$(date +%Y%m%d_%H%M%S)}"
LOG_PATH="${LOG_PATH:-autoresearch/results/${RESULTS_SUBDIR}/launch.log}"

if [ ! -x "$PYTHON_BIN" ]; then
  echo "Missing Python interpreter: $PYTHON_BIN" >&2
  exit 1
fi

if [ ! -f "$BASE_CONFIG" ]; then
  echo "Base config not found: $BASE_CONFIG" >&2
  exit 1
fi

if [ ! -f "$BASE_CHECKPOINT" ]; then
  echo "Base checkpoint not found: $BASE_CHECKPOINT" >&2
  exit 1
fi

if [ -z "${OPENAI_API_KEY:-}" ] && [ ! -f ".env" ] && [ ! -f "autoresearch/.env" ] && [ ! -f "autoresearch/results/.env" ]; then
  echo "OPENAI_API_KEY is not set and no .env found at repo root, autoresearch/.env, or autoresearch/results/.env" >&2
  exit 1
fi

mkdir -p "$(dirname "$LOG_PATH")"

nohup "$PYTHON_BIN" -m autoresearch.run_marl_recursive \
  --provider openai \
  --model "$MODEL" \
  --base-config "$BASE_CONFIG" \
  --base-checkpoint "$BASE_CHECKPOINT" \
  --results-subdir "$RESULTS_SUBDIR" \
  --generations "$GENERATIONS" \
  --candidates-per-batch "$CANDIDATES_PER_BATCH" \
  --timeout "$TIMEOUT" \
  --seed "$SEED" \
  --mode "$MODE" \
  > "$LOG_PATH" 2>&1 &

PID="$!"

echo "Started MARL autoresearch."
echo "pid=$PID"
echo "results_subdir=$RESULTS_SUBDIR"
echo "log_path=$LOG_PATH"
echo "After completion: ./autoresearch/postrun_marl_branch.sh autoresearch/results/${RESULTS_SUBDIR}"
