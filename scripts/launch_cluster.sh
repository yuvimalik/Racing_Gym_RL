#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-config/world_model_cluster_e1_baseline.yaml}"
WANDB_PROJECT="${WANDB_PROJECT:-racing-world-model}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-29500}"

echo "Launching world model training"
echo "Config: ${CONFIG}"
echo "GPUs per node: ${NPROC_PER_NODE}"
echo "W&B project: ${WANDB_PROJECT}"
echo "Additional args: $*"

if [ "${NPROC_PER_NODE}" -gt 1 ]; then
  torchrun \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    world_model_train.py \
      --config "${CONFIG}" \
      --distributed \
      --wandb-project "${WANDB_PROJECT}" \
      "$@"
else
  python world_model_train.py \
    --config "${CONFIG}" \
    --wandb-project "${WANDB_PROJECT}" \
    "$@"
fi
