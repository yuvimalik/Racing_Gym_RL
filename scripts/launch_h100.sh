#!/usr/bin/env bash
# Launch distributed RSSM training on H100 nodes via Prime Intellect.
#
# How this works:
#   torchrun is PyTorch's built-in distributed launcher. It starts N identical
#   processes (one per GPU) and sets environment variables (LOCAL_RANK, RANK,
#   WORLD_SIZE) so the training script knows which GPU it is and how many GPUs
#   there are. The processes communicate via NCCL (NVIDIA's GPU-to-GPU comms lib).
#
# Usage (single node, 4 GPUs):
#   bash scripts/launch_h100.sh --epochs 20 --run-name h100_sweep_run01
#
# Usage (multi-node, 2 nodes × 4 GPUs each = 8 GPUs total):
#   # On node 0 (master):
#   MASTER_ADDR=<node0_ip> MASTER_PORT=29500 \
#     torchrun --nnodes=2 --node_rank=0 --nproc_per_node=4 \
#       world_model_train.py --config config/world_model_h100.yaml --distributed "$@"
#   # On node 1:
#   MASTER_ADDR=<node0_ip> MASTER_PORT=29500 \
#     torchrun --nnodes=2 --node_rank=1 --nproc_per_node=4 \
#       world_model_train.py --config config/world_model_h100.yaml --distributed "$@"
#
# KL annealing example (recommended for new training runs):
#   bash scripts/launch_h100.sh --epochs 30 \
#     --free-nats-start 2.0 --free-nats-target 0.3 --free-nats-warmup 10
#
# W&B sweep agent (run on each node to participate in the sweep):
#   wandb agent <sweep_id>

set -euo pipefail

# Number of GPUs on this node — change to 1 for single-GPU debugging
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"

# Master node address for multi-node training (leave as localhost for single node)
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-29500}"

# W&B project name (passed through to the training script)
WANDB_PROJECT="${WANDB_PROJECT:-racing-world-model}"

echo "Launching RSSM training on ${NPROC_PER_NODE} GPUs"
echo "Master: ${MASTER_ADDR}:${MASTER_PORT}"
echo "W&B project: ${WANDB_PROJECT}"
echo "Additional args: $*"

torchrun \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  world_model_train.py \
    --config config/world_model_h100.yaml \
    --distributed \
    --wandb-project "${WANDB_PROJECT}" \
    "$@"
