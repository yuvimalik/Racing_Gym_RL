#!/usr/bin/env bash
# Summarize a MARL autoresearch branch after (or during) run_marl_recursive.
# Usage: ./autoresearch/postrun_marl_branch.sh autoresearch/results/research_branches/<branch_name>
set -euo pipefail

BRANCH_DIR="${1:?Usage: $0 <path-to-branch-directory>}"

if [[ ! -d "$BRANCH_DIR" ]]; then
  echo "Not a directory: $BRANCH_DIR" >&2
  exit 1
fi

echo "=== Branch: $BRANCH_DIR ==="
echo ""

if [[ -f "$BRANCH_DIR/branch_state.json" ]]; then
  echo "--- branch_state.json (parent checkpoint path) ---"
  command -v python3 >/dev/null 2>&1 && python3 -c "import json,sys; d=json.load(open(sys.argv[1])); p=d.get('parent',{}); print('config_path:', p.get('config_path')); print('surface_path:', p.get('surface_path')); print('checkpoint_path:', p.get('checkpoint_path'))" "$BRANCH_DIR/branch_state.json" || cat "$BRANCH_DIR/branch_state.json"
  echo ""
else
  echo "No branch_state.json yet."
  echo ""
fi

if [[ -d "$BRANCH_DIR/promoted" ]] && [[ -n "$(ls -A "$BRANCH_DIR/promoted" 2>/dev/null || true)" ]]; then
  echo "--- promoted/ (use for long train.py) ---"
  ls -la "$BRANCH_DIR/promoted"
  echo ""
else
  echo "--- promoted/ --- (empty or missing until a candidate passes the full ladder)"
  echo ""
fi

if [[ -f "$BRANCH_DIR/generations.jsonl" ]]; then
  echo "--- generations.jsonl (last line) ---"
  last=$(tail -n 1 "$BRANCH_DIR/generations.jsonl")
  if command -v python3 >/dev/null 2>&1; then
    echo "$last" | python3 -m json.tool 2>/dev/null || echo "$last"
  else
    echo "$last"
  fi
  echo ""
fi

echo "--- generation_review.txt files ---"
find "$BRANCH_DIR" -maxdepth 4 -name generation_review.txt 2>/dev/null | sort || true
echo ""

REVIEW=$(find "$BRANCH_DIR" -maxdepth 4 -name generation_review.txt 2>/dev/null | sort | tail -n 1 || true)
if [[ -n "$REVIEW" ]]; then
  echo "--- Latest: $REVIEW ---"
  cat "$REVIEW"
  echo ""
fi

echo "--- Suggested commands (after you pick checkpoint + config) ---"
echo "# Heavier eval than the autoresearch ladder:"
echo "cd \"$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)\" && \\"
echo "  python evaluate.py --model <CHECKPOINT.pt> --config <CONFIG.yaml> --episodes 15 --no-video --seed 42"
echo ""
echo "# Long training (example: resume promoted checkpoint, add steps):"
echo "  python train.py --config <CONFIG.yaml> --resume <CHECKPOINT.pt> --timesteps_add 5000000 --seed 42"
echo ""
echo "If promoted/config.yaml still points torch_policy_variant_source at a generation folder, copy promoted/surface.py"
echo "to a stable path and set training.torch_policy_variant_source in the YAML you pass to train.py."
echo ""
