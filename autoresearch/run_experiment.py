"""
LOCKED - Single experiment runner (subprocess isolation).

Called by run_loop.py as a subprocess. Trains for a fixed budget,
evaluates, and prints metrics as JSON to stdout.

Usage:
    python -m autoresearch.run_experiment --config config/multi_car_config.yaml \
        --timesteps 500000 --eval-episodes 20 --seed 42 --checkpoint-dir autoresearch/results/run_001
"""

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from autoresearch.prepare import create_training_envs, evaluate, load_config
from autoresearch.train_ppo import HYPERPARAMS, PPOTrainer


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds >= 3600:
        return f"{seconds / 3600:.1f}h"
    if seconds >= 60:
        return f"{seconds / 60:.1f}m"
    return f"{seconds:.0f}s"


def merge_hyperparams(config: dict) -> dict:
    hp = dict(HYPERPARAMS)
    hp.update(config.get("ppo", {}) or {})
    return hp


def summarize_hyperparams(hp: dict) -> str:
    ordered_keys = [
        "learning_rate",
        "n_steps",
        "batch_size",
        "n_epochs",
        "gamma",
        "gae_lambda",
        "clip_range",
        "ent_coef",
        "vf_coef",
        "max_grad_norm",
        "min_log_std",
        "max_log_std",
        "steer_min_log_std",
        "steer_max_log_std",
    ]
    parts = [f"{key}={hp[key]}" for key in ordered_keys if key in hp]
    return ", ".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Run a single autoresearch experiment")
    parser.add_argument("--config", type=str, default="config/multi_car_config.yaml")
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-dir", type=str, default="autoresearch/results/latest")
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from (loads policy weights)")
    parser.add_argument("--experiment-id", type=int, default=None,
                        help="Optional experiment id for logging")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        log(f"[experiment] CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        log("[experiment] WARNING: running on CPU; training will be slow")

    log("[experiment] Loading config...")
    config = load_config(args.config)
    effective_hp = merge_hyperparams(config)

    log(f"[experiment] Creating {args.num_envs} environments...")
    env_t0 = time.time()
    env = create_training_envs(config, n_envs=args.num_envs, seed=args.seed, use_subproc=False)
    env_time = time.time() - env_t0
    log(f"[experiment] Envs created in {env_time:.1f}s")

    obs_shape = tuple(env.observation_space.shape)
    action_dim = int(np.prod(env.action_space.shape))

    trainer = PPOTrainer(obs_shape, action_dim, device=device, hp=effective_hp)
    param_count = sum(p.numel() for p in trainer.policy.parameters())
    checkpoint_dir = Path(args.checkpoint_dir)

    experiment_label = f"#{args.experiment_id}" if args.experiment_id is not None else "(ad hoc)"
    log(f"[experiment] Starting experiment {experiment_label}")
    log(
        "[experiment] Setup | "
        f"device={device} | obs_shape={obs_shape} | action_dim={action_dim} | "
        f"timesteps={args.timesteps:,} | num_envs={args.num_envs} | seed={args.seed} | "
        f"params={param_count:,}"
    )

    n_steps = int(effective_hp.get("n_steps", 128))
    n_envs = args.num_envs
    steps_per_rollout = n_steps * n_envs
    n_iterations = max(1, args.timesteps // max(steps_per_rollout, 1))
    log(
        "[experiment] Plan | "
        f"steps_per_rollout={steps_per_rollout:,} ({n_steps} x {n_envs}) | "
        f"planned_iterations={n_iterations:,}"
    )
    log(f"[experiment] Hyperparams | {summarize_hyperparams(effective_hp)}")

    if args.resume and Path(args.resume).is_file():
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        trainer.policy.load_state_dict(ckpt["policy_state_dict"])
        if "optimizer_state_dict" in ckpt:
            try:
                trainer.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except Exception as exc:
                log(f"[experiment] Resume optimizer state skipped: {exc}")
        log(f"[experiment] Resumed from {args.resume}")

    t0 = time.time()
    train_time = 0.0
    num_timesteps = 0

    try:
        obs = env.reset()
        log_every = max(1, n_iterations // 20)
        interval_timesteps = 0
        interval_t0 = time.time()

        for iteration in range(1, n_iterations + 1):
            obs, buf = trainer._collect_rollout(env, obs, n_steps)
            buf = trainer._compute_gae(buf)
            metrics = trainer._ppo_update(buf)
            num_timesteps += steps_per_rollout
            interval_timesteps += steps_per_rollout

            if iteration == 1 or iteration == n_iterations or iteration % log_every == 0:
                now = time.time()
                elapsed = now - t0
                interval_elapsed = max(now - interval_t0, 1e-6)
                overall_sps = num_timesteps / max(elapsed, 1e-6)
                recent_sps = interval_timesteps / interval_elapsed
                progress_pct = 100.0 * num_timesteps / max(args.timesteps, 1)
                remaining_steps = max(0, args.timesteps - num_timesteps)
                eta_seconds = remaining_steps / max(overall_sps, 1e-6)
                rollout_reward = float(buf["rewards"].sum(axis=0).mean())
                log(
                    "[experiment] Progress | "
                    f"iter={iteration:,}/{n_iterations:,} | "
                    f"{progress_pct:5.1f}% | "
                    f"steps={num_timesteps:,}/{args.timesteps:,} | "
                    f"sps={overall_sps:.1f} | recent_sps={recent_sps:.1f} | "
                    f"elapsed={format_duration(elapsed)} | eta={format_duration(eta_seconds)} | "
                    f"rollout_reward={rollout_reward:.2f} | "
                    f"pg={metrics['pg_loss']:.4f} vf={metrics['vf_loss']:.4f} ent={metrics['entropy']:.4f}"
                )
                interval_t0 = now
                interval_timesteps = 0

        train_time = time.time() - t0

        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = checkpoint_dir / "final.pt"
        torch.save({
            "policy_state_dict": trainer.policy.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "hyperparams": effective_hp,
            "num_timesteps": num_timesteps,
            "experiment_id": args.experiment_id,
            "config_path": str(Path(args.config)),
        }, ckpt_path)

        log(
            "[experiment] Train done | "
            f"train_seconds={train_time:.1f} | mean_sps={num_timesteps / max(train_time, 1e-6):.1f} | "
            f"checkpoint={ckpt_path}"
        )
        log(f"[experiment] Evaluating {args.eval_episodes} episodes...")

        trainer.policy.eval()
        eval_t0 = time.time()
        eval_metrics = evaluate(
            trainer.policy,
            device,
            config,
            n_episodes=args.eval_episodes,
            seed=args.seed + 10000,
        )
        eval_seconds = time.time() - eval_t0

        eval_metrics["train_wall_clock_seconds"] = train_time
        eval_metrics["eval_wall_clock_seconds"] = eval_seconds
        eval_metrics["total_timesteps"] = num_timesteps
        eval_metrics["steps_per_second"] = num_timesteps / max(train_time, 1e-6)
        eval_metrics["effective_hyperparams"] = effective_hp
        eval_metrics["checkpoint_path"] = str(ckpt_path)
        if args.experiment_id is not None:
            eval_metrics["experiment_id"] = args.experiment_id

        log(
            "[experiment] Summary | "
            f"reward={eval_metrics.get('mean_reward', float('nan')):.2f} | "
            f"progress={eval_metrics.get('mean_progress', float('nan')):.4f} | "
            f"speed={eval_metrics.get('mean_speed', float('nan')):.2f} | "
            f"offtrack_rate={eval_metrics.get('offtrack_rate', float('nan')):.4f} | "
            f"train_s={train_time:.1f} | eval_s={eval_seconds:.1f} | "
            f"mean_sps={eval_metrics['steps_per_second']:.1f}"
        )
        print(json.dumps(eval_metrics))

    except Exception as exc:
        error_result = {
            "mean_reward": -999.0,
            "error": str(exc),
            "train_wall_clock_seconds": time.time() - t0,
            "total_timesteps": num_timesteps,
            "effective_hyperparams": effective_hp,
        }
        if args.experiment_id is not None:
            error_result["experiment_id"] = args.experiment_id
        print(json.dumps(error_result))
        log(f"[experiment] FAILED: {exc}")
        import traceback
        traceback.print_exc(file=sys.stderr)

    finally:
        env.close()


if __name__ == "__main__":
    main()
