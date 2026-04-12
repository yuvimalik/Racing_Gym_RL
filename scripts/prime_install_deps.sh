#!/usr/bin/env bash
# Run from repository ROOT on Prime (after git clone). Creates venv and installs deps.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ ! -f train.py ]] || [[ ! -f requirements.txt ]]; then
  echo "ERROR: Run this script from inside the Racing_Gym_RL repo (train.py missing)."
  exit 1
fi

if [[ ! -d venv ]]; then
  python3 -m venv venv
fi
# shellcheck source=/dev/null
source venv/bin/activate

pip install -U pip
pip install -r requirements.txt
if [[ -f requirements_sb3.txt ]]; then
  pip install -r requirements_sb3.txt --no-deps
else
  echo "WARNING: requirements_sb3.txt missing; install stable-baselines3[extra]==1.8.0 --no-deps manually."
fi

echo ""
echo "Re-checking CUDA after pip installs..."
python3 <<'PY'
import torch
if not torch.cuda.is_available():
    print("WARNING: CUDA not available. Reinstall PyTorch with CUDA 12.4 wheels, e.g.:")
    print("  pip install torch --index-url https://download.pytorch.org/whl/cu124")
    raise SystemExit(1)
print("CUDA OK:", torch.cuda.get_device_name(0))
PY

pip install git+https://github.com/igilitschenski/multi_car_racing.git --no-deps

echo ""
echo "Optional: if rendering fails at runtime, on Ubuntu install OpenGL libs, e.g.:"
echo "  sudo apt-get update && sudo apt-get install -y libgl1 libglib2.0-0"

echo "OK: venv + requirements + multi_car_racing installed. Activate with: source venv/bin/activate"
