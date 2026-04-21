#!/usr/bin/env bash
# Run on your Prime / Linux GPU box after SSH. Exits non-zero if PyTorch cannot see CUDA.
set -euo pipefail

echo "=== nvidia-smi (if available) ==="
if command -v nvidia-smi &>/dev/null; then
  nvidia-smi -L || true
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
  echo "WARNING: nvidia-smi not in PATH (driver tools may be missing)."
fi

echo ""
echo "=== PyTorch CUDA check ==="
python3 <<'PY'
import torch
print("torch.__version__", torch.__version__)
ok = torch.cuda.is_available()
print("torch.cuda.is_available()", ok)
if ok:
    print("device[0]", torch.cuda.get_device_name(0))
else:
    raise SystemExit("ERROR: CUDA not available to PyTorch. Fix torch install (cu124) before training.")
PY

echo "OK: GPU visible to PyTorch."
