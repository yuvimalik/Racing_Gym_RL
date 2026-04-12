#!/usr/bin/env bash
# Budget 2-car MARL run (see config/prime_marl_2car_budget.yaml). Run from repo root on Linux with venv active.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export RACING_HEADLESS_PYGLET="${RACING_HEADLESS_PYGLET:-1}"
# shellcheck source=/dev/null
source venv/bin/activate
exec python train.py --config config/prime_marl_2car_budget.yaml --seed "${PRIME_SEED:-42}" "$@"
