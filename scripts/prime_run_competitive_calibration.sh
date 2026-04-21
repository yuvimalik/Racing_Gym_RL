#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# shellcheck source=/dev/null
source venv/bin/activate
export RACING_HEADLESS_PYGLET="${RACING_HEADLESS_PYGLET:-1}"

CALIB_TIMESTEPS="${PRIME_CALIB_TIMESTEPS:-600000}"
CALIB_EVAL_EPISODES="${PRIME_CALIB_EVAL_EPISODES:-5}"
CALIB_SEED="${PRIME_CALIB_SEED:-42}"

GEN_DIR="artifacts/competitive_tuning/generated_configs"
mkdir -p "$GEN_DIR"

run_python() {
  if [[ -z "${DISPLAY:-}" ]] && command -v xvfb-run >/dev/null 2>&1; then
    xvfb-run -a python "$@"
  else
    python "$@"
  fi
}

make_calib_config() {
  local src_config="$1"
  local out_config="$2"
  python - "$src_config" "$out_config" "$CALIB_TIMESTEPS" <<'PY'
import copy
import sys
from pathlib import Path
import yaml

src = Path(sys.argv[1])
out = Path(sys.argv[2])
steps = int(sys.argv[3])

cfg = yaml.safe_load(src.read_text())
cfg = copy.deepcopy(cfg)
cfg["training"]["total_timesteps"] = steps
cfg["training"]["eval_freq"] = max(50000, steps // 3)
cfg["training"]["save_freq"] = max(50000, steps // 3)
cfg["training"]["n_eval_episodes"] = 1
cfg["training"]["visual_eval"]["enabled"] = False
cfg["training"]["visual_eval"]["freq"] = max(50000, steps // 3)

for key in ("model_dir", "log_dir", "results_dir"):
    cfg["paths"][key] = str(Path(cfg["paths"][key]) / "calibration")

out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(yaml.safe_dump(cfg, sort_keys=False))
print(out)
PY
}

run_variant() {
  local variant_name="$1"
  local base_config="$2"
  local calib_config="$GEN_DIR/${variant_name}_calibration.yaml"

  echo ""
  echo "=== ${variant_name}: build calibration config ==="
  make_calib_config "$base_config" "$calib_config"

  echo "=== ${variant_name}: short training (${CALIB_TIMESTEPS} steps) ==="
  run_python train.py --config "$calib_config" --seed "$CALIB_SEED"

  local model_dir
  model_dir="$(python - "$calib_config" <<'PY'
import sys
import yaml
cfg = yaml.safe_load(open(sys.argv[1], "r"))
print(cfg["paths"]["model_dir"])
PY
)"
  local results_dir
  results_dir="$(python - "$calib_config" <<'PY'
import sys
import yaml
cfg = yaml.safe_load(open(sys.argv[1], "r"))
print(cfg["paths"]["results_dir"])
PY
)"

  mkdir -p "$results_dir"

  local model_path="${model_dir}/best_model_torch.pt"
  if [[ ! -f "$model_path" ]]; then
    model_path="${model_dir}/final_model_torch.pt"
  fi
  if [[ ! -f "$model_path" ]]; then
    echo "No model checkpoint found for ${variant_name} in ${model_dir}" >&2
    exit 1
  fi

  echo "=== ${variant_name}: fixed-seed evaluation ==="
  run_python evaluate.py \
    --model "$model_path" \
    --config "$calib_config" \
    --episodes "$CALIB_EVAL_EPISODES" \
    --seed "$CALIB_SEED" \
    --output-json "${results_dir}/calibration_eval_seed${CALIB_SEED}.json" \
    --output-video "${results_dir}/calibration_eval_seed${CALIB_SEED}.mp4"
}

run_variant "pace" "config/prime_marl_2car_compete_pace.yaml"
run_variant "overtake" "config/prime_marl_2car_compete_overtake.yaml"
run_variant "combined" "config/prime_marl_2car_compete_combined.yaml"

echo ""
echo "Calibration sweep completed for all three variants."
echo "Next: run scripts/prime_select_competitive_winner.sh"
