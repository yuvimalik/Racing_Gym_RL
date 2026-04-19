#!/usr/bin/env bash
# 10-car MARL preflight run (~1M stream steps) for Prime.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export RACING_HEADLESS_PYGLET="${RACING_HEADLESS_PYGLET:-1}"
# shellcheck source=/dev/null
source venv/bin/activate
if [[ -z "${DISPLAY:-}" ]] && command -v xvfb-run >/dev/null 2>&1; then
  exec xvfb-run -a python train.py --config config/prime_marl_10car_preflight.yaml --seed "${PRIME_SEED:-42}" "$@"
fi
exec python train.py --config config/prime_marl_10car_preflight.yaml --seed "${PRIME_SEED:-42}" "$@"
