"""
Distributed multi-GPU PPO entry point for Prime Intellect (and any torchrun-compatible cluster).

Usage:
    # Single node, all GPUs:
    torchrun --nproc_per_node=NUM_GPUS distributed_train.py --config config/multi_car_config.yaml

    # Multi-node (Prime Intellect sets MASTER_ADDR / MASTER_PORT / WORLD_SIZE / RANK):
    torchrun --nproc_per_node=NUM_GPUS --nnodes=NUM_NODES distributed_train.py --config config/multi_car_config.yaml

Environment variables expected (set automatically by torchrun):
    LOCAL_RANK  — GPU index on this node
    WORLD_SIZE  — total number of processes
    RANK        — global rank (0 = master)
    MASTER_ADDR / MASTER_PORT — rendezvous endpoint
"""

import os
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecTransposeImage
from stable_baselines3.common.utils import set_random_seed

from train import (
    load_config,
    TorchPPOTrainer,
    make_env,
    create_env,
    get_device,
)


def main():
    parser = argparse.ArgumentParser(description="Distributed PPO training on Multi-Car Racing")
    parser.add_argument("--config", type=str, default="config/multi_car_config.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # Distributed init (torchrun sets these env vars automatically)
    # -------------------------------------------------------------------------
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    global_rank = int(os.environ.get("RANK", 0))

    if world_size > 1:
        dist.init_process_group(
            backend="nccl",
            init_method="env://",  # uses MASTER_ADDR + MASTER_PORT
        )

    torch.cuda.set_device(local_rank)
    is_master = (global_rank == 0)

    # -------------------------------------------------------------------------
    # Config + reproducibility
    # -------------------------------------------------------------------------
    config = load_config(args.config)
    set_random_seed(args.seed + global_rank)  # different seed per rank

    training_config = config.get("training", {})
    obs_config = config.get("observation", {})

    model_dir = Path(config["paths"]["model_dir"]).resolve()
    log_dir = Path(config["paths"]["log_dir"])
    results_dir = Path(config["paths"]["results_dir"])

    if is_master:
        model_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)

    if world_size > 1:
        dist.barrier()  # wait for master to create directories

    # -------------------------------------------------------------------------
    # Performance knobs
    # -------------------------------------------------------------------------
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    # -------------------------------------------------------------------------
    # Environments — each rank gets its own slice of envs
    # -------------------------------------------------------------------------
    total_num_envs = int(training_config.get("num_envs", 8))
    # Divide envs evenly across ranks; remainder goes to last rank
    envs_per_rank = total_num_envs // world_size
    if global_rank == world_size - 1:
        envs_per_rank += total_num_envs % world_size  # pick up any remainder

    env_seed = args.seed + global_rank * 1000

    if envs_per_rank > 1:
        env = SubprocVecEnv([
            make_env(config, rank=i, seed=env_seed)
            for i in range(envs_per_rank)
        ])
    else:
        env = DummyVecEnv([lambda: create_env(config, rank=0, seed=env_seed)])

    if not obs_config.get("enabled", False):
        env = VecTransposeImage(env)

    # Eval env only on master (to avoid redundant evaluation on workers)
    eval_env = None
    if is_master:
        eval_env = DummyVecEnv([lambda: create_env(config, rank=1, seed=env_seed + 1000)])
        if not obs_config.get("enabled", False):
            eval_env = VecTransposeImage(eval_env)
    else:
        # Workers need an eval_env stub — create a single env; evals only run on master
        eval_env = DummyVecEnv([lambda: create_env(config, rank=1, seed=env_seed + 1000)])
        if not obs_config.get("enabled", False):
            eval_env = VecTransposeImage(eval_env)

    if is_master:
        print(f"[Rank {global_rank}] World size: {world_size} | Local rank: {local_rank} | "
              f"Envs this rank: {envs_per_rank} | Total envs: {total_num_envs}")

    # -------------------------------------------------------------------------
    # Trainer — pass local_rank + world_size for DDP wrapping
    # -------------------------------------------------------------------------
    trainer = TorchPPOTrainer(
        env=env,
        eval_env=eval_env,
        config=config,
        device=f"cuda:{local_rank}",
        model_dir=model_dir,
        log_dir=log_dir,
        local_rank=local_rank,
        world_size=world_size,
    )

    if args.resume:
        resume_path = Path(args.resume).resolve()
        if not resume_path.is_file():
            resume_path = model_dir / Path(args.resume).name
        if is_master:
            print(f"Loading checkpoint: {resume_path}")
        trainer.load(resume_path)

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------
    total_timesteps = int(training_config["total_timesteps"])

    if is_master:
        print(f"Starting distributed training: {total_timesteps:,} steps across {world_size} GPU(s).")

    trainer.learn(
        total_timesteps=total_timesteps,
        eval_freq=int(training_config["eval_freq"]),
        n_eval_episodes=int(training_config["n_eval_episodes"]),
        save_freq=int(training_config["save_freq"]),
        log_interval=int(training_config.get("log_interval", 10)),
        success_gate=training_config.get("success_gate", {}),
        visual_eval_cfg=training_config.get("visual_eval", {}),
    )

    if is_master:
        final_path = model_dir / "final_model_torch.pt"
        trainer.save(final_path)
        print(f"Final model saved: {final_path}")

    # -------------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------------
    env.close()
    eval_env.close()

    if world_size > 1:
        dist.destroy_process_group()

    if is_master:
        print("Distributed training complete.")


if __name__ == "__main__":
    main()
