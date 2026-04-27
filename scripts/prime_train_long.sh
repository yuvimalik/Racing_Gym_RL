#!/usr/bin/env bash
# Long 2-car MARL run (see config/prime_marl_2car_long.yaml). Run from repo root on Linux with venv active.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export RACING_HEADLESS_PYGLET="${RACING_HEADLESS_PYGLET:-1}"
# shellcheck source=/dev/null
source venv/bin/activate
if [[ -z "${DISPLAY:-}" ]] && command -v xvfb-run >/dev/null 2>&1; then
  exec xvfb-run -a python train.py --config config/prime_marl_2car_long.yaml --seed "${PRIME_SEED:-42}" "$@"
fi
exec python train.py --config config/prime_marl_2car_long.yaml --seed "${PRIME_SEED:-42}" "$@"
