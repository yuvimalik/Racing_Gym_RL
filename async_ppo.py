"""
Asynchronous PPO (APPO) — IMPALA-style actor-learner architecture.

Architecture:
  N actor workers (CPU processes, each with own env + policy copy)
    → trajectory queue (mp.Queue)
      → GPU learner (drains queue, PPO update, pushes new weights)
        → param store (shared memory, workers pull latest weights)

The GPU never idles — it always has queued trajectories to process.
Workers never idle — they immediately start next rollout after queuing.

Usage:
    python async_ppo.py --config config/multi_car_config.yaml --num-workers 16
"""

import os
import time
import queue
import argparse
import signal
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.multiprocessing as mp
import yaml

# Import env factory and model from train.py
from train import (
    CnnActorCritic,
    load_config,
    create_env,
    make_env,
    get_device,
)


# ---------------------------------------------------------------------------
# Shared-memory parameter store
# ---------------------------------------------------------------------------

class ParamStore:
    """Lock-free shared memory for policy weights.

    Learner pushes new weights after each update (~0.5ms for small CNN).
    Workers pull latest weights after each trajectory (~0.5ms).
    """

    def __init__(self, policy: CnnActorCritic):
        # Flatten all params into a single 1D array in shared memory
        flat = torch.cat([p.data.cpu().flatten() for p in policy.parameters()])
        self._size = flat.numel()
        self._shared = mp.Array('f', self._size, lock=False)
        # Store shapes for reconstruction
        self._shapes = [p.shape for p in policy.parameters()]
        self._numels = [p.numel() for p in policy.parameters()]
        self.version = mp.Value('i', 0, lock=False)
        # Initial push
        self._write(flat.numpy())

    def _write(self, flat_np: np.ndarray):
        # Direct memcpy into shared memory — no lock needed (atomic-ish for floats)
        np.frombuffer(self._shared, dtype=np.float32)[:] = flat_np

    def push(self, policy: torch.nn.Module):
        """Learner publishes new weights."""
        flat = torch.cat([p.data.cpu().flatten() for p in policy.parameters()])
        self._write(flat.numpy())
        self.version.value += 1

    def pull(self, policy: torch.nn.Module) -> int:
        """Worker loads latest weights. Returns current version."""
        flat = np.frombuffer(self._shared, dtype=np.float32).copy()
        flat_t = torch.from_numpy(flat)
        offset = 0
        for p, n in zip(policy.parameters(), self._numels):
            p.data.copy_(flat_t[offset:offset + n].view(p.shape))
            offset += n
        return self.version.value


# ---------------------------------------------------------------------------
# Trajectory collection (runs inside actor worker process)
# ---------------------------------------------------------------------------

def collect_trajectory(env, policy, obs, n_steps: int, device: torch.device) -> dict:
    """Collect one trajectory of n_steps from a single env."""
    obs_buf = []
    actions_buf = []
    rewards_buf = []
    dones_buf = []
    log_probs_buf = []
    values_buf = []

    for _ in range(n_steps):
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0) / 255.0
        with torch.inference_mode():
            raw_action, log_prob, value = policy.act(obs_tensor, deterministic=False)

        raw_action_np = raw_action.cpu().numpy().squeeze(0)
        # Map raw action to env action space
        env_action = _raw_to_env_action(raw_action_np)

        obs_buf.append(obs.copy() if isinstance(obs, np.ndarray) else np.array(obs))
        actions_buf.append(raw_action_np)
        log_probs_buf.append(float(log_prob.cpu().item()))
        values_buf.append(float(value.cpu().item()))

        obs, reward, done, info = env.step(env_action)
        rewards_buf.append(float(reward))
        dones_buf.append(float(done))

        if done:
            obs = env.reset()

    return {
        "obs": np.array(obs_buf, dtype=np.uint8),  # uint8 saves pickle bandwidth
        "actions": np.array(actions_buf, dtype=np.float32),
        "rewards": np.array(rewards_buf, dtype=np.float32),
        "dones": np.array(dones_buf, dtype=np.float32),
        "old_log_probs": np.array(log_probs_buf, dtype=np.float32),
        "old_values": np.array(values_buf, dtype=np.float32),
        "final_obs": obs.copy() if isinstance(obs, np.ndarray) else np.array(obs),
    }


def _raw_to_env_action(raw: np.ndarray) -> np.ndarray:
    """Map unconstrained policy output to env action ranges."""
    out = raw.copy()
    if out.size >= 1:
        out[0] = np.tanh(out[0])                         # steer: [-1, 1]
    if out.size >= 2:
        out[1] = 1.0 / (1.0 + np.exp(-out[1]))           # throttle: [0, 1]
    if out.size >= 3:
        out[2] = 1.0 / (1.0 + np.exp(-out[2]))           # brake: [0, 1]
    return np.clip(out, -1.0, 1.0)


def actor_worker(
    worker_id: int,
    config: dict,
    param_store: ParamStore,
    trajectory_queue: mp.Queue,
    stop_event: mp.Event,
    obs_shape: tuple,
    action_dim: int,
    n_steps: int,
    seed: int,
    ppo_cfg: dict,
):
    """Actor process: own env + policy copy, produce trajectories continuously."""
    # Suppress SIGINT in workers — main process handles shutdown
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    env = create_env(config, rank=worker_id, seed=seed + worker_id)
    # Set training mode on RewardShapingWrapper if it exists
    if hasattr(env, 'set_training_mode'):
        env.set_training_mode(True)
    else:
        # Walk wrapper chain
        e = env
        while hasattr(e, 'env'):
            if hasattr(e, 'set_training_mode'):
                e.set_training_mode(True)
                break
            e = e.env

    device = torch.device("cpu")  # actors run on CPU
    policy = CnnActorCritic(
        obs_shape, action_dim,
        min_log_std=float(ppo_cfg.get("min_log_std", -1.5)),
        max_log_std=float(ppo_cfg.get("max_log_std", 1.0)),
        steer_min_log_std=ppo_cfg.get("steer_min_log_std"),
        steer_max_log_std=ppo_cfg.get("steer_max_log_std"),
    ).to(device)
    policy.eval()

    policy_version = param_store.pull(policy)
    obs = env.reset()

    while not stop_event.is_set():
        traj = collect_trajectory(env, policy, obs, n_steps, device)
        traj["worker_id"] = worker_id
        traj["policy_version"] = policy_version

        # Push to queue (blocks if full — backpressure is correct)
        try:
            trajectory_queue.put(traj, timeout=5.0)
        except queue.Full:
            if stop_event.is_set():
                break
            continue

        obs = traj["final_obs"]

        # Pull latest weights from shared memory
        policy_version = param_store.pull(policy)

    env.close()


# ---------------------------------------------------------------------------
# GAE computation
# ---------------------------------------------------------------------------

def compute_gae(values, rewards, dones, gamma=0.99, gae_lambda=0.95):
    """Compute returns and advantages using GAE for a single trajectory."""
    n = len(rewards)
    advantages = np.zeros(n, dtype=np.float32)
    last_gae = 0.0

    for t in reversed(range(n)):
        if t == n - 1:
            next_value = 0.0  # bootstrap = 0 (trajectory boundary)
        else:
            next_value = values[t + 1]
        non_terminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * non_terminal - values[t]
        last_gae = delta + gamma * gae_lambda * non_terminal * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    return returns, advantages


# ---------------------------------------------------------------------------
# GPU Learner
# ---------------------------------------------------------------------------

class APPOLearner:
    """GPU learner that drains trajectory queue and runs PPO updates."""

    def __init__(
        self,
        policy: CnnActorCritic,
        config: dict,
        device: torch.device,
        param_store: ParamStore,
        trajectory_queue: mp.Queue,
        model_dir: Path,
        log_dir: Path,
    ):
        self.policy = policy.to(device)
        self.device = device
        self.config = config
        self.param_store = param_store
        self.trajectory_queue = trajectory_queue
        self.model_dir = Path(model_dir)
        self.log_dir = Path(log_dir)

        ppo_cfg = config["ppo"]
        self.learning_rate = float(ppo_cfg["learning_rate"])
        self.clip_range = float(ppo_cfg["clip_range"])
        self.ent_coef = float(ppo_cfg["ent_coef"])
        self.vf_coef = float(ppo_cfg["vf_coef"])
        self.max_grad_norm = float(ppo_cfg["max_grad_norm"])
        self.n_epochs = int(ppo_cfg["n_epochs"])
        self.batch_size = int(ppo_cfg["batch_size"])
        self.gamma = float(ppo_cfg["gamma"])
        self.gae_lambda = float(ppo_cfg["gae_lambda"])

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.learning_rate)
        self.policy_version = 0
        self.num_timesteps = 0

        # Phase 4A: AMP (mixed precision) — significant speedup on A100/H100 with BF16
        self.use_amp = device.type == "cuda"
        self.amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.grad_scaler = torch.amp.GradScaler("cuda", enabled=(self.use_amp and self.amp_dtype == torch.float16))

        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def train(
        self,
        total_timesteps: int,
        target_batch: int = 8,
        max_staleness: int = 5,
        save_freq: int = 25000,
        eval_freq: int = 0,
        log_interval: int = 1,
    ):
        """Main learner loop. Drains queue, updates policy, pushes new weights."""
        print(f"[APPO Learner] Starting. Target: {total_timesteps:,} steps, "
              f"batch={target_batch} trajectories, max_staleness={max_staleness}",
              flush=True)

        start_time = time.time()
        last_log_step = 0
        last_save_step = 0
        best_reward = -np.inf
        update_count = 0

        while self.num_timesteps < total_timesteps:
            # 1. Drain queue — collect target_batch trajectories
            batch = []
            stalenesses = []
            drain_start = time.time()

            while len(batch) < target_batch:
                try:
                    traj = self.trajectory_queue.get(timeout=1.0)
                except queue.Empty:
                    if len(batch) > 0:
                        break  # process what we have
                    continue

                staleness = self.policy_version - traj["policy_version"]
                if staleness > max_staleness:
                    continue  # drop very stale data
                batch.append(traj)
                stalenesses.append(staleness)

            if not batch:
                continue

            # 2. Prepare batch — compute GAE per trajectory, stack into tensors
            all_obs = []
            all_actions = []
            all_old_log_probs = []
            all_advantages = []
            all_returns = []
            batch_steps = 0

            for traj in batch:
                returns, advantages = compute_gae(
                    traj["old_values"], traj["rewards"], traj["dones"],
                    self.gamma, self.gae_lambda,
                )
                all_obs.append(traj["obs"])
                all_actions.append(traj["actions"])
                all_old_log_probs.append(traj["old_log_probs"])
                all_advantages.append(advantages)
                all_returns.append(returns)
                batch_steps += len(traj["rewards"])

            obs_batch = torch.as_tensor(
                np.concatenate(all_obs), dtype=torch.float32, device=self.device
            ) / 255.0
            actions_batch = torch.as_tensor(
                np.concatenate(all_actions), dtype=torch.float32, device=self.device
            )
            old_lp_batch = torch.as_tensor(
                np.concatenate(all_old_log_probs), dtype=torch.float32, device=self.device
            )
            adv_batch = torch.as_tensor(
                np.concatenate(all_advantages), dtype=torch.float32, device=self.device
            )
            ret_batch = torch.as_tensor(
                np.concatenate(all_returns), dtype=torch.float32, device=self.device
            )

            # Normalize advantages globally
            adv_batch = (adv_batch - adv_batch.mean()) / (adv_batch.std() + 1e-8)

            # 3. PPO update
            n_samples = obs_batch.shape[0]
            metrics = self._ppo_update(
                obs_batch, actions_batch, old_lp_batch, adv_batch, ret_batch, n_samples
            )

            # 4. Push new weights
            self.policy_version += 1
            self.param_store.push(self.policy)
            self.num_timesteps += batch_steps
            update_count += 1

            # 5. Logging
            now = time.time()
            elapsed = now - start_time
            steps_per_sec = self.num_timesteps / elapsed if elapsed > 0 else 0
            mean_staleness = np.mean(stalenesses) if stalenesses else 0
            queue_size = self.trajectory_queue.qsize()
            pct = 100.0 * self.num_timesteps / total_timesteps

            if update_count % log_interval == 0:
                eta_sec = (total_timesteps - self.num_timesteps) / steps_per_sec if steps_per_sec > 0 else 0
                eta_str = f"{eta_sec/3600:.1f}h" if eta_sec >= 3600 else (
                    f"{eta_sec/60:.1f}m" if eta_sec >= 60 else f"{eta_sec:.0f}s")
                gpu_mb = ""
                if self.device.type == "cuda":
                    gpu_mb = f" | {torch.cuda.memory_allocated(self.device) / 1024**2:.0f}MB"
                print(
                    f"[APPO] {pct:.1f}% | {self.num_timesteps:,}/{total_timesteps:,} | "
                    f"{steps_per_sec:.0f} sps | v{self.policy_version} | "
                    f"stale={mean_staleness:.1f} | q={queue_size} | "
                    f"pg={metrics['policy_loss']:.4f} | vf={metrics['value_loss']:.4f} | "
                    f"ent={metrics['entropy']:.4f} | clip={metrics['clip_fraction']:.3f} | "
                    f"ETA {eta_str}{gpu_mb}",
                    flush=True,
                )

            # 6. Checkpoint
            if save_freq > 0 and (self.num_timesteps - last_save_step) >= save_freq:
                ckpt_path = self.model_dir / f"appo_step_{self.num_timesteps}.pt"
                self.save(ckpt_path)
                last_save_step = self.num_timesteps
                print(f"[APPO] Checkpoint: {ckpt_path}", flush=True)

        # Final save
        final_path = self.model_dir / "appo_final.pt"
        self.save(final_path)
        elapsed = time.time() - start_time
        print(f"[APPO] Done. {self.num_timesteps:,} steps in {elapsed:.0f}s "
              f"({self.num_timesteps/elapsed:.0f} sps). Saved: {final_path}", flush=True)

    def _ppo_update(self, obs, actions, old_log_probs, advantages, returns, n_samples):
        """Standard PPO clipped update on a batch."""
        policy_losses = []
        value_losses = []
        entropies = []
        clip_fracs = []

        self.policy.train()
        for _ in range(self.n_epochs):
            indices = torch.randperm(n_samples, device=self.device)
            for start in range(0, n_samples, self.batch_size):
                end = min(start + self.batch_size, n_samples)
                idx = indices[start:end]

                # Phase 4A: AMP autocast for forward pass
                with torch.amp.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                    values, new_log_probs, entropy = self.policy.evaluate_actions(
                        obs[idx], actions[idx]
                    )

                    # IS ratio — naturally handles staleness
                    ratio = torch.exp(new_log_probs - old_log_probs[idx])

                    # PPO clipped surrogate
                    surr1 = ratio * advantages[idx]
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * advantages[idx]
                    policy_loss = -torch.min(surr1, surr2).mean()

                    value_loss = 0.5 * (values - returns[idx]).pow(2).mean()
                    entropy_loss = -entropy.mean()

                    loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss

                self.optimizer.zero_grad()
                self.grad_scaler.scale(loss).backward()
                self.grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()

                with torch.no_grad():
                    clip_frac = (torch.abs(ratio - 1.0) > self.clip_range).float().mean()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(-entropy_loss.item())
                clip_fracs.append(clip_frac.item())

        self.policy.eval()
        return {
            "policy_loss": np.mean(policy_losses),
            "value_loss": np.mean(value_losses),
            "entropy": np.mean(entropies),
            "clip_fraction": np.mean(clip_fracs),
        }

    def save(self, path: Path):
        torch.save({
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "num_timesteps": self.num_timesteps,
            "policy_version": self.policy_version,
            "config": self.config,
        }, str(path))

    def load(self, path: Path):
        payload = torch.load(str(path), map_location=self.device)
        self.policy.load_state_dict(payload["policy_state_dict"])
        self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        self.num_timesteps = int(payload.get("num_timesteps", 0))
        self.policy_version = int(payload.get("policy_version", 0))
        self.param_store.push(self.policy)
        print(f"[APPO] Loaded checkpoint: step={self.num_timesteps}, v={self.policy_version}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(description="Async PPO training")
    parser.add_argument("--config", type=str, default="config/multi_car_config.yaml")
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--n-steps", type=int, default=128,
                        help="Steps per trajectory per worker")
    parser.add_argument("--target-batch", type=int, default=8,
                        help="Number of trajectories to batch before each learner update")
    parser.add_argument("--max-staleness", type=int, default=5,
                        help="Max policy versions behind before dropping a trajectory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    ppo_cfg = config["ppo"]
    training_cfg = config.get("training", {})

    total_timesteps = int(training_cfg.get("total_timesteps", 2_000_000))
    save_freq = int(training_cfg.get("save_freq", 25000))

    model_dir = Path(config["paths"]["model_dir"]).resolve()
    log_dir = Path(config["paths"]["log_dir"]).resolve()
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    device_str = get_device(config)
    device = torch.device(device_str)

    # GPU performance knobs
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

    # Determine obs_shape and action_dim from a temporary env
    tmp_env = create_env(config, rank=999, seed=args.seed + 999)
    obs = tmp_env.reset()
    # Observation is (H, W, C) from env; after VecTransposeImage it becomes (C, H, W)
    # But we're using raw envs in workers, so obs_shape = transposed
    obs_shape = (obs.shape[2], obs.shape[0], obs.shape[1])  # (C, H, W)
    action_dim = int(np.prod(tmp_env.action_space.shape))
    tmp_env.close()

    print(f"\n{'='*70}")
    print("ASYNC PPO (APPO) TRAINING")
    print(f"{'='*70}")
    print(f"Workers: {args.num_workers}")
    print(f"Steps per trajectory: {args.n_steps}")
    print(f"Target batch: {args.target_batch} trajectories")
    print(f"Max staleness: {args.max_staleness} versions")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Device: {device}")
    print(f"Obs shape: {obs_shape}, Action dim: {action_dim}")
    print(f"Model dir: {model_dir}")
    print(f"{'='*70}\n")

    # Create learner policy on GPU
    policy = CnnActorCritic(
        obs_shape, action_dim,
        min_log_std=float(ppo_cfg.get("min_log_std", -1.5)),
        max_log_std=float(ppo_cfg.get("max_log_std", 1.0)),
        steer_min_log_std=ppo_cfg.get("steer_min_log_std"),
        steer_max_log_std=ppo_cfg.get("steer_max_log_std"),
    ).to(device)
    policy.eval()

    # Phase 4B: torch.compile() — fuses CNN+MLP kernels, ~1.2-1.5x speedup
    if hasattr(torch, "compile") and device.type == "cuda":
        try:
            policy = torch.compile(policy, mode="reduce-overhead")
            print("[APPO] torch.compile() enabled (reduce-overhead mode)", flush=True)
        except Exception as e:
            print(f"[APPO] torch.compile() failed, continuing without it: {e}", flush=True)

    # Shared memory param store
    param_store = ParamStore(policy)

    # Trajectory queue
    trajectory_queue = mp.Queue(maxsize=64)
    stop_event = mp.Event()

    # Learner
    learner = APPOLearner(
        policy=policy,
        config=config,
        device=device,
        param_store=param_store,
        trajectory_queue=trajectory_queue,
        model_dir=model_dir,
        log_dir=log_dir,
    )

    if args.resume:
        learner.load(Path(args.resume).resolve())

    # Launch actor workers
    workers = []
    for i in range(args.num_workers):
        p = mp.Process(
            target=actor_worker,
            args=(
                i, config, param_store, trajectory_queue, stop_event,
                obs_shape, action_dim, args.n_steps, args.seed, ppo_cfg,
            ),
            daemon=True,
        )
        p.start()
        workers.append(p)

    print(f"[APPO] Launched {args.num_workers} actor workers.", flush=True)

    # Train
    try:
        learner.train(
            total_timesteps=total_timesteps,
            target_batch=args.target_batch,
            max_staleness=args.max_staleness,
            save_freq=save_freq,
        )
    except KeyboardInterrupt:
        print("\n[APPO] Interrupted. Saving checkpoint...", flush=True)
        learner.save(model_dir / f"appo_interrupted_{learner.num_timesteps}.pt")
    finally:
        stop_event.set()
        # Give workers time to finish, then terminate stragglers
        for w in workers:
            w.join(timeout=5.0)
            if w.is_alive():
                w.terminate()

    print("[APPO] All workers stopped. Training complete.", flush=True)


if __name__ == "__main__":
    main()
