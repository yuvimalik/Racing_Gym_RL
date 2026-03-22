"""
Training script for Multi-Car Racing with selectable PPO backend.

Backends:
- stable-baselines3 PPO (existing)
- local PyTorch PPO trainer (in this file)
"""

import os
import json
import yaml
import argparse
from pathlib import Path
from typing import Dict, Iterable
import gym
import gym_multi_car_racing
import numpy as np
from gym_multi_car_racing import multi_car_racing as mcr
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import threading
import time
from datetime import datetime, timedelta
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import cv2


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


class CnnActorCritic(nn.Module):
    """Actor-critic network for image observations (N, C, H, W)."""

    def __init__(self, obs_shape, action_dim: int, min_log_std: float = -1.5, max_log_std: float = 1.0,
                 steer_min_log_std: float = None, steer_max_log_std: float = None):
        super().__init__()
        c, h, w = obs_shape
        self.features = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            n_flatten = self.features(torch.zeros(1, c, h, w)).shape[1]

        # Separate MLP heads for policy and value (prevents value gradient from
        # corrupting policy features — this is how SB3's CnnPolicy works).
        self.policy_mlp = nn.Sequential(
            nn.Linear(n_flatten, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.policy_mean = nn.Linear(128, action_dim)

        self.value_mlp = nn.Sequential(
            nn.Linear(n_flatten, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.value_head = nn.Linear(128, 1)
        # Per-dimension log_std: steering needs more exploration than throttle/brake
        log_std_init = torch.full((action_dim,), -0.5)
        if action_dim >= 1:
            log_std_init[0] = 0.0    # steer: std=1.0 (high exploration through tanh)
        if action_dim >= 3:
            log_std_init[2] = -1.0   # brake: std=0.37 (low exploration, mostly off)
        self.log_std = nn.Parameter(log_std_init)
        self.min_log_std = float(min_log_std)
        self.max_log_std = float(max_log_std)
        # Steer-specific log_std bounds — allows tighter control over steering exploration
        # independently of throttle/brake.  None = fall back to global min/max.
        self.steer_min_log_std = float(steer_min_log_std) if steer_min_log_std is not None else self.min_log_std
        self.steer_max_log_std = float(steer_max_log_std) if steer_max_log_std is not None else self.max_log_std

        # Bias exploration toward moving forward initially: throttle high, brake low.
        if action_dim >= 3:
            nn.init.constant_(self.policy_mean.bias[1], 2.0)   # throttle ~= sigmoid(2) = 0.88 (was 3.0=0.95, too fast)
            nn.init.constant_(self.policy_mean.bias[2], -3.0)  # brake ~= sigmoid(-3) = 0.05

    def _features(self, obs: torch.Tensor) -> torch.Tensor:
        """Shared CNN feature extraction."""
        return self.features(obs)

    def get_dist_and_value(self, obs: torch.Tensor):
        shared = self._features(obs)
        # Separate paths for policy and value (prevents value gradient from
        # corrupting policy features via the MLP heads).
        policy_latent = self.policy_mlp(shared)
        mean = self.policy_mean(policy_latent)
        # Per-dimension log_std clamping: steer uses its own bounds so its exploration
        # range can be tuned independently of throttle/brake.
        # Use torch.cat (NOT in-place indexing) to avoid breaking autograd.
        steer_ls = torch.clamp(self.log_std[0:1], self.steer_min_log_std, self.steer_max_log_std)
        if self.log_std.shape[0] > 1:
            other_ls = torch.clamp(self.log_std[1:], self.min_log_std, self.max_log_std)
            log_std = torch.cat([steer_ls, other_ls])
        else:
            log_std = steer_ls
        std = log_std.exp().expand_as(mean)
        dist = Normal(mean, std)
        value_latent = self.value_mlp(shared)
        value = self.value_head(value_latent).squeeze(-1)
        return dist, value

    @staticmethod
    def raw_to_env_action(raw_action: torch.Tensor) -> torch.Tensor:
        """Map unconstrained policy action to env action ranges."""
        out = raw_action.clone()
        if out.shape[-1] >= 1:
            out[..., 0] = torch.tanh(out[..., 0])      # steer in [-1, 1]
        if out.shape[-1] >= 2:
            out[..., 1] = torch.sigmoid(out[..., 1])   # throttle in [0, 1]
        if out.shape[-1] >= 3:
            out[..., 2] = torch.sigmoid(out[..., 2])   # brake in [0, 1]
        return out

    def act(self, obs: torch.Tensor, deterministic: bool = False):
        dist, value = self.get_dist_and_value(obs)
        if deterministic:
            action = dist.mean
        else:
            action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob, value

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        dist, value = self.get_dist_and_value(obs)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return value, log_prob, entropy

    def forward(self, obs: torch.Tensor, actions: torch.Tensor):
        """DDP-compatible entry point for training updates (evaluate_actions path)."""
        return self.evaluate_actions(obs, actions)


class RolloutBuffer:
    """Rollout storage for PPO."""

    def __init__(self, n_steps, n_envs, obs_shape, action_dim):
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.obs = np.zeros((n_steps, n_envs, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((n_steps, n_envs, action_dim), dtype=np.float32)
        self.rewards = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.dones = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.values = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.log_probs = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.advantages = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.returns = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.pos = 0

    def add(self, obs, action, reward, done, value, log_prob):
        self.obs[self.pos] = obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        self.values[self.pos] = value
        self.log_probs[self.pos] = log_prob
        self.pos += 1

    def compute_returns_advantages(self, last_values, last_dones, gamma, gae_lambda):
        last_gae = np.zeros(self.n_envs, dtype=np.float32)
        for step in reversed(range(self.n_steps)):
            if step == self.n_steps - 1:
                next_non_terminal = 1.0 - last_dones.astype(np.float32)
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.dones[step]
                next_values = self.values[step + 1]
            delta = self.rewards[step] + gamma * next_values * next_non_terminal - self.values[step]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            self.advantages[step] = last_gae
        self.returns = self.advantages + self.values

    def batches(self, batch_size, device, normalize_advantage=True,
                distributed=False) -> Iterable[Dict[str, torch.Tensor]]:
        n_samples = self.n_steps * self.n_envs
        obs = self.obs.reshape(n_samples, *self.obs.shape[2:])
        actions = self.actions.reshape(n_samples, self.actions.shape[-1])
        old_log_probs = self.log_probs.reshape(n_samples)
        advantages = self.advantages.reshape(n_samples)
        returns = self.returns.reshape(n_samples)
        old_values = self.values.reshape(n_samples)

        if normalize_advantage and advantages.size > 1:
            if distributed and dist.is_available() and dist.is_initialized():
                # Global advantage normalization across all ranks — ensures consistent
                # gradient scaling when each GPU holds a different subset of rollout data.
                adv_t = torch.as_tensor(advantages, dtype=torch.float32, device=device)
                adv_mean = adv_t.mean()
                adv_sq_mean = (adv_t ** 2).mean()
                dist.all_reduce(adv_mean, op=dist.ReduceOp.AVG)
                dist.all_reduce(adv_sq_mean, op=dist.ReduceOp.AVG)
                adv_std = (adv_sq_mean - adv_mean ** 2).clamp(min=0).sqrt()
                advantages = ((adv_t - adv_mean) / (adv_std + 1e-8)).cpu().numpy()
            else:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        for start in range(0, n_samples, batch_size):
            batch_idx = indices[start:start + batch_size]
            yield {
                "obs": torch.as_tensor(obs[batch_idx], dtype=torch.float32, device=device) / 255.0,
                "actions": torch.as_tensor(actions[batch_idx], dtype=torch.float32, device=device),
                "old_log_probs": torch.as_tensor(old_log_probs[batch_idx], dtype=torch.float32, device=device),
                "advantages": torch.as_tensor(advantages[batch_idx], dtype=torch.float32, device=device),
                "returns": torch.as_tensor(returns[batch_idx], dtype=torch.float32, device=device),
                "old_values": torch.as_tensor(old_values[batch_idx], dtype=torch.float32, device=device),
            }


class TorchPPOTrainer:
    """PPO training loop implemented locally in PyTorch."""

    def __init__(self, env, eval_env, config, device, model_dir: Path, log_dir: Path,
                 local_rank: int = 0, world_size: int = 1):
        self.env = env
        self.eval_env = eval_env
        self.config = config
        self.local_rank = int(local_rank)
        self.world_size = int(world_size)
        self.rank = local_rank  # within-node rank; for multi-node use dist.get_rank()
        self.distributed = world_size > 1
        # Use the device resolved by get_device(); fall back to CPU only if nothing was provided.
        # (The old hardcoded CUDA check here was ignoring the 'device' argument entirely.)
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model_dir = Path(model_dir)
        self.log_dir = Path(log_dir)

        if not hasattr(env.observation_space, "shape"):
            raise ValueError("Torch backend currently supports Box image observations only.")
        if not hasattr(env.action_space, "shape"):
            raise ValueError("Torch backend currently supports Box action space only.")

        self.obs_shape = tuple(env.observation_space.shape)
        self.action_dim = int(np.prod(env.action_space.shape))
        self.env_action_low = np.asarray(env.action_space.low, dtype=np.float32)
        self.env_action_high = np.asarray(env.action_space.high, dtype=np.float32)

        ppo_cfg = config["ppo"]
        self.learning_rate = float(ppo_cfg["learning_rate"])
        self.n_steps = int(ppo_cfg["n_steps"])
        self.batch_size = int(ppo_cfg["batch_size"])
        self.n_epochs = int(ppo_cfg["n_epochs"])
        self.gamma = float(ppo_cfg["gamma"])
        self.gae_lambda = float(ppo_cfg["gae_lambda"])
        self.clip_range = float(ppo_cfg["clip_range"])
        self.ent_coef = float(ppo_cfg["ent_coef"])
        self.vf_coef = float(ppo_cfg["vf_coef"])
        self.max_grad_norm = float(ppo_cfg["max_grad_norm"])
        # Value function clip range: limits how far V can move from V_old per update.
        # Prevents the huge value-loss MSE (proportional to return scale) from
        # dominating gradients and blowing up Adam's moment estimates early in training.
        self.vf_clip_range = float(ppo_cfg.get("vf_clip_range", 10.0))
        min_log_std = float(ppo_cfg.get("min_log_std", -1.5))
        max_log_std = float(ppo_cfg.get("max_log_std", 1.0))
        steer_min_log_std = ppo_cfg.get("steer_min_log_std", None)
        steer_max_log_std = ppo_cfg.get("steer_max_log_std", None)
        steer_min_log_std = float(steer_min_log_std) if steer_min_log_std is not None else None
        steer_max_log_std = float(steer_max_log_std) if steer_max_log_std is not None else None

        self.policy = CnnActorCritic(
            self.obs_shape, self.action_dim,
            min_log_std=min_log_std, max_log_std=max_log_std,
            steer_min_log_std=steer_min_log_std, steer_max_log_std=steer_max_log_std,
        ).to(self.device)
        if self.distributed:
            self.policy = DDP(self.policy, device_ids=[self.local_rank])
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.learning_rate)
        self.num_timesteps = 0
        self.eval_history_path = self.log_dir / "torch_eval_history.jsonl"

    @property
    def _policy(self) -> CnnActorCritic:
        """Return the underlying CnnActorCritic, unwrapping DDP if present."""
        return self.policy.module if isinstance(self.policy, DDP) else self.policy

    def _raw_to_env_action_np(self, raw_action_np: np.ndarray) -> np.ndarray:
        raw_t = torch.as_tensor(raw_action_np, dtype=torch.float32)
        env_t = CnnActorCritic.raw_to_env_action(raw_t)
        env_np = env_t.numpy()
        return np.clip(env_np, self.env_action_low, self.env_action_high)

    def _collect_rollout(self, obs):
        n_envs = self.env.num_envs
        buffer = RolloutBuffer(self.n_steps, n_envs, self.obs_shape, self.action_dim)
        last_dones = np.zeros(n_envs, dtype=np.float32)

        for _ in range(self.n_steps):
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device) / 255.0
            with torch.no_grad():
                raw_action, log_prob, value = self._policy.act(obs_tensor, deterministic=False)

            raw_action_np = raw_action.cpu().numpy()
            env_actions = self._raw_to_env_action_np(raw_action_np)

            # Guard against NaN/Inf from an exploded policy — Box2D crashes on non-finite actions
            if not np.isfinite(env_actions).all():
                print("[WARN] NaN/Inf in actions (policy may have diverged). Substituting safe no-op action.", flush=True)
                safe = np.array([[0.0, 0.3, 0.0]], dtype=np.float32)  # steer=0, gentle throttle, no brake
                env_actions = np.where(np.isfinite(env_actions), env_actions, np.broadcast_to(safe, env_actions.shape))
                env_actions = np.clip(env_actions, self.env_action_low, self.env_action_high)

            next_obs, rewards, dones, infos = self.env.step(env_actions)

            buffer.add(
                obs=obs.astype(np.float32),
                action=raw_action_np.astype(np.float32),
                reward=np.asarray(rewards, dtype=np.float32),
                done=np.asarray(dones, dtype=np.float32),
                value=value.cpu().numpy().astype(np.float32),
                log_prob=log_prob.cpu().numpy().astype(np.float32),
            )

            self.num_timesteps += n_envs
            obs = next_obs
            last_dones = np.asarray(dones, dtype=np.float32)

            for info in infos:
                ep = info.get("episode") if isinstance(info, dict) else None
                if ep is not None and "r" in ep:
                    # Throttle: only print every 5th episode so progress lines stay visible
                    if getattr(self, "_episode_print_count", 0) % 5 == 0:
                        print(f"Episode reward: {ep['r']:.2f} | length: {ep.get('l', -1)}", flush=True)
                    self._episode_print_count = getattr(self, "_episode_print_count", 0) + 1

        with torch.no_grad():
            last_obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device) / 255.0
            _, last_values = self._policy.get_dist_and_value(last_obs_tensor)
        buffer.compute_returns_advantages(
            last_values=last_values.cpu().numpy().astype(np.float32),
            last_dones=last_dones,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )
        return obs, buffer

    def _update(self, buffer: RolloutBuffer):
        policy_losses = []
        value_losses = []
        entropy_losses = []
        clip_fractions = []
        approx_kls = []

        for _ in range(self.n_epochs):
            for batch in buffer.batches(self.batch_size, self.device, normalize_advantage=True,
                                        distributed=self.distributed):
                if self.distributed:
                    # Call through DDP so it intercepts forward and syncs gradients via AllReduce
                    values, new_log_probs, entropy = self.policy(batch["obs"], batch["actions"])
                else:
                    values, new_log_probs, entropy = self._policy.evaluate_actions(batch["obs"], batch["actions"])
                ratio = torch.exp(new_log_probs - batch["old_log_probs"])

                policy_loss_1 = batch["advantages"] * ratio
                policy_loss_2 = batch["advantages"] * torch.clamp(
                    ratio, 1.0 - self.clip_range, 1.0 + self.clip_range
                )
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                # Value function loss with clipping (standard PPO-clip for value function).
                # Limits how far V can move from V_old each update, preventing the value
                # MSE from exploding when returns are large (e.g. 3000+) early in training.
                value_loss_unclipped = F.mse_loss(values, batch["returns"])
                if self.vf_clip_range > 0:
                    values_clipped = batch["old_values"] + torch.clamp(
                        values - batch["old_values"], -self.vf_clip_range, self.vf_clip_range
                    )
                    value_loss_clipped = F.mse_loss(values_clipped, batch["returns"])
                    value_loss = torch.max(value_loss_unclipped, value_loss_clipped)
                else:
                    value_loss = value_loss_unclipped
                entropy_loss = -entropy.mean()
                loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)

                # Skip update if gradients contain NaN/Inf — applying them would corrupt weights
                if not torch.isfinite(grad_norm):
                    print("[WARN] Non-finite gradient norm — skipping optimizer step to protect weights.", flush=True)
                    self.optimizer.zero_grad()
                    grad_norm = torch.tensor(0.0)
                else:
                    self.optimizer.step()

                with torch.no_grad():
                    log_ratio = new_log_probs - batch["old_log_probs"]
                    approx_kl = torch.mean((torch.exp(log_ratio) - 1.0) - log_ratio)
                    clip_fraction = torch.mean((torch.abs(ratio - 1.0) > self.clip_range).float())

                policy_losses.append(float(policy_loss.item()))
                value_losses.append(float(value_loss.item()))
                entropy_losses.append(float(entropy_loss.item()))
                clip_fractions.append(float(clip_fraction.item()))
                approx_kls.append(float(approx_kl.item()))

        return {
            "policy_loss": float(np.mean(policy_losses)),
            "value_loss": float(np.mean(value_losses)),
            "entropy_loss": float(np.mean(entropy_losses)),
            "clip_fraction": float(np.mean(clip_fractions)),
            "approx_kl": float(np.mean(approx_kls)),
            "grad_norm": float(grad_norm.item()) if torch.is_tensor(grad_norm) else float(grad_norm),
        }

    def _set_learning_rate(self, total_timesteps: int) -> float:
        total = max(1, int(total_timesteps))
        frac_remaining = max(0.0, 1.0 - (self.num_timesteps / float(total)))
        lr_now = self.learning_rate * frac_remaining
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr_now
        return float(lr_now)

    def _get_eval_base_env(self):
        env = self.eval_env
        # Unwrap VecTransposeImage/DummyVecEnv wrappers until we reach the actual gym env.
        for attr in ("venv", "env"):
            while hasattr(env, attr):
                env = getattr(env, attr)
        if hasattr(env, "envs") and len(env.envs) > 0:
            return env.envs[0]
        return None

    def evaluate_visual(self, n_episodes: int = 1):
        base_env = self._get_eval_base_env()
        if base_env is None:
            print("[TorchPPO VisualEval] Skipped: could not access base eval env.", flush=True)
            return

        print(f"[TorchPPO VisualEval] Running {n_episodes} episode(s) with live render...", flush=True)
        rewards = []
        progresses = []
        for ep in range(n_episodes):
            obs = base_env.reset()
            done = False
            ep_reward = 0.0
            final_progress = 0.0
            while not done:
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device) / 255.0
                obs_t = obs_t.permute(2, 0, 1).unsqueeze(0)
                with torch.no_grad():
                    raw_action, _, _ = self._policy.act(obs_t, deterministic=True)
                env_action_np = self._raw_to_env_action_np(raw_action.cpu().numpy().squeeze(0))
                obs, reward, done, info = base_env.step(env_action_np)
                ep_reward += float(reward)
                if isinstance(info, dict):
                    final_progress = float(info.get("progress", final_progress))
                frame = base_env.render(mode="rgb_array")
                if frame is not None:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    cv2.imshow("Torch PPO Visual Eval", frame_bgr)
                    cv2.waitKey(1)
            rewards.append(ep_reward)
            progresses.append(final_progress)
            print(
                f"[TorchPPO VisualEval] Episode {ep + 1}/{n_episodes}: "
                f"reward={ep_reward:.2f}, progress={final_progress:.2%}",
                flush=True,
            )
        print(
            f"[TorchPPO VisualEval] Summary: mean_reward={np.mean(rewards):.2f}, "
            f"mean_progress={np.mean(progresses):.2%}",
            flush=True,
        )

    def evaluate(self, n_episodes: int = 5):
        rewards = []
        progresses = []
        offtrack_events = []
        steer_variances = []
        mean_speeds = []
        mean_throttles = []
        mean_brakes = []
        lengths = []
        for _ in range(n_episodes):
            obs = self.eval_env.reset()
            done = np.array([False])
            total_reward = 0.0
            final_progress = 0.0
            episode_offtrack = 0
            steer_values = []
            speed_values = []
            throttle_values = []
            brake_values = []
            episode_len = 0
            while not bool(done[0]):
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device) / 255.0
                with torch.no_grad():
                    raw_action, _, _ = self._policy.act(obs_tensor, deterministic=True)
                raw_action_np = raw_action.cpu().numpy()
                env_actions = self._raw_to_env_action_np(raw_action_np)
                obs, reward, done, info = self.eval_env.step(env_actions)
                total_reward += float(np.asarray(reward).reshape(-1)[0])
                episode_len += 1
                flat_env_actions = env_actions.reshape(-1, self.action_dim)
                steer_values.append(float(flat_env_actions[0, 0]))
                if self.action_dim >= 2:
                    throttle_values.append(float(flat_env_actions[0, 1]))
                if self.action_dim >= 3:
                    brake_values.append(float(flat_env_actions[0, 2]))
                info0 = info[0] if isinstance(info, (list, tuple)) and len(info) > 0 else {}
                if isinstance(info0, dict):
                    final_progress = float(info0.get("progress", final_progress))
                    episode_offtrack += int(info0.get("events/offtrack", 0) > 0)
                    speed_values.append(float(info0.get("telemetry/speed", 0.0)))
            rewards.append(total_reward)
            progresses.append(final_progress)
            offtrack_events.append(int(episode_offtrack > 0))
            lengths.append(episode_len)
            steer_variances.append(float(np.var(steer_values)) if len(steer_values) > 1 else 0.0)
            mean_speeds.append(float(np.mean(speed_values)) if speed_values else 0.0)
            mean_throttles.append(float(np.mean(throttle_values)) if throttle_values else 0.0)
            mean_brakes.append(float(np.mean(brake_values)) if brake_values else 0.0)
        return {
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "mean_progress": float(np.mean(progresses)),
            "std_progress": float(np.std(progresses)),
            "offtrack_rate": float(np.mean(offtrack_events)),
            "mean_steer_variance": float(np.mean(steer_variances)),
            "mean_speed": float(np.mean(mean_speeds)),
            "mean_throttle": float(np.mean(mean_throttles)),
            "mean_brake": float(np.mean(mean_brakes)),
            "mean_length": float(np.mean(lengths)),
            "episode_rewards": rewards,
            "episode_progress": progresses,
            "episode_offtrack": offtrack_events,
            "episode_steer_variance": steer_variances,
        }

    def save(self, path: Path):
        # In distributed mode only rank 0 writes — all ranks hold identical weights
        # (DDP guarantees this) so one checkpoint is sufficient.
        if self.distributed and self.rank != 0:
            return
        cuda_rng_state_all = None
        if torch.cuda.is_available():
            cuda_rng_state_all = torch.cuda.get_rng_state_all()
        # DDP wraps the module under .module; unwrap for portable checkpoints.
        raw_policy = self.policy.module if isinstance(self.policy, DDP) else self.policy
        payload = {
            "checkpoint_format_version": 2,
            "policy_state_dict": raw_policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "num_timesteps": self.num_timesteps,
            "config": self.config,
            "rng_state": {
                "numpy": np.random.get_state(),
                "torch_cpu": torch.get_rng_state(),
                "torch_cuda_all": cuda_rng_state_all,
            },
        }
        torch.save(payload, str(path))

    def load(self, path: Path):
        payload = torch.load(str(path), map_location=self.device)
        raw_policy = self.policy.module if isinstance(self.policy, DDP) else self.policy
        raw_policy.load_state_dict(payload["policy_state_dict"])
        self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        self.num_timesteps = int(payload.get("num_timesteps", 0))
        rng_state = payload.get("rng_state")
        if isinstance(rng_state, dict):
            np_state = rng_state.get("numpy")
            if np_state is not None:
                np.random.set_state(np_state)
            torch_cpu_state = rng_state.get("torch_cpu")
            if torch_cpu_state is not None:
                try:
                    torch.set_rng_state(torch_cpu_state)
                except (TypeError, RuntimeError):
                    pass  # RNG state format mismatch (e.g. saved on different device); non-fatal
            torch_cuda_all = rng_state.get("torch_cuda_all")
            if torch_cuda_all is not None and torch.cuda.is_available():
                try:
                    torch.cuda.set_rng_state_all(torch_cuda_all)
                except Exception:
                    # Older checkpoints or device-count mismatch: continue without hard failure.
                    pass

    def learn(
        self,
        total_timesteps: int,
        eval_freq: int,
        n_eval_episodes: int,
        save_freq: int,
        log_interval: int = 10,
        success_gate: dict = None,
        visual_eval_cfg: dict = None,
    ):
        obs = self.env.reset()
        update_idx = 0
        best_eval_reward = -np.inf
        start_time = time.time()
        last_log_step = self.num_timesteps
        last_log_update_idx = 0
        last_log_time = start_time
        last_checkpoint_step = self.num_timesteps
        last_eval_step = self.num_timesteps

        # Progress summary (especially clear when resuming)
        remaining = max(0, total_timesteps - self.num_timesteps)
        if self.num_timesteps > 0:
            print(
                f"[TorchPPO] Resumed from step {self.num_timesteps:,}. "
                f"Training {self.num_timesteps:,} -> {total_timesteps:,} ({remaining:,} steps remaining)."
            )
        else:
            print(f"[TorchPPO] Starting from scratch. Target: {total_timesteps:,} steps.")
        print(
            f"[TorchPPO] Progress: {100.0 * self.num_timesteps / total_timesteps:.1f}% "
            f"({self.num_timesteps:,} / {total_timesteps:,})"
        )
        print(
            "[TorchPPO] Log columns: % | steps | steps/s | iters/s | ETA | policy_loss | value_loss | entropy | kl | clip_frac | grad_norm | GPU_MB",
            flush=True,
        )
        gate_cfg = success_gate or {}
        gate_enabled = bool(gate_cfg.get("enabled", False))
        gate_reward = float(gate_cfg.get("mean_reward_threshold", np.inf))
        gate_progress = float(gate_cfg.get("mean_progress_threshold", 0.95))
        gate_offtrack = float(gate_cfg.get("max_offtrack_rate", 0.05))
        gate_min_episodes = int(gate_cfg.get("min_eval_episodes", 10))
        fail_fast_cfg = (self.config.get("training", {}) or {}).get("fail_fast", {}) or {}
        fail_fast_enabled = bool(fail_fast_cfg.get("enabled", True))
        fail_fast_min_steps = int(fail_fast_cfg.get("min_timesteps_before_check", 20000))
        fail_fast_min_progress = float(fail_fast_cfg.get("min_mean_progress", 0.10))
        fail_fast_min_speed = float(fail_fast_cfg.get("min_mean_speed", 2.0))
        fail_fast_patience = int(fail_fast_cfg.get("patience_evals", 3))
        fail_fast_bad_evals = 0
        curriculum_cfg = (self.config.get("reward_shaping", {}) or {}).get("curriculum", {}) or {}
        curriculum_enabled = bool(curriculum_cfg.get("enabled", False))
        curriculum_stage = int(curriculum_cfg.get("start_stage", 1))
        promote_progress = float(curriculum_cfg.get("promote_progress_threshold", 0.35))
        promote_speed = float(curriculum_cfg.get("promote_speed_threshold", 8.0))
        visual_cfg = visual_eval_cfg or {}
        visual_enabled = bool(visual_cfg.get("enabled", True))
        visual_freq = int(visual_cfg.get("freq", 50000))
        visual_episodes = int(visual_cfg.get("n_episodes", 1))
        last_visual_eval_step = self.num_timesteps

        # Sequential collect → update (macOS requires all pyglet/env ops on the main thread;
        # the async double-buffer pattern crashes with NSInternalInconsistencyException on Apple Silicon).
        current_buffer = None
        obs, current_buffer = self._collect_rollout(obs)

        while self.num_timesteps < total_timesteps:
            lr_now = self._set_learning_rate(total_timesteps)

            # Update policy on the current rollout buffer
            train_metrics = self._update(current_buffer)

            # Collect the next rollout on the main thread
            obs, current_buffer = self._collect_rollout(obs)

            update_idx += 1

            # Log progress every update (steps, iters/s, GPU) so progress is always visible
            now = time.time()
            pct = 100.0 * min(self.num_timesteps, total_timesteps) / float(total_timesteps)
            steps_since_last = self.num_timesteps - last_log_step
            updates_since_last = update_idx - last_log_update_idx
            elapsed_since_last = now - last_log_time
            steps_per_sec = steps_since_last / elapsed_since_last if elapsed_since_last > 0 else 0.0
            iters_per_sec = updates_since_last / elapsed_since_last if elapsed_since_last > 0 else 0.0
            remaining = max(0, total_timesteps - self.num_timesteps)
            eta_sec = remaining / steps_per_sec if steps_per_sec > 0 else 0.0
            if eta_sec >= 3600:
                eta_str = f"{eta_sec / 3600:.1f}h"
            elif eta_sec >= 60:
                eta_str = f"{eta_sec / 60:.1f}m"
            else:
                eta_str = f"{eta_sec:.0f}s"
            last_log_step = self.num_timesteps
            last_log_update_idx = update_idx
            last_log_time = now
            gpu_mb = ""
            if self.device.type == "cuda":
                gpu_mb = f" | {torch.cuda.memory_allocated(self.device) / 1024**2:.0f} MB alloc"
            elif self.device.type == "mps":
                try:
                    gpu_mb = f" | {torch.mps.current_allocated_memory() / 1024**2:.0f} MB alloc"
                except Exception:
                    pass
            print(
                f"[TorchPPO] {pct:.1f}% | {self.num_timesteps:,}/{total_timesteps:,} | "
                f"{steps_per_sec:.0f} steps/s | {iters_per_sec:.2f} iters/s | ETA {eta_str} | "
                f"pg={train_metrics['policy_loss']:.4f} | vf={train_metrics['value_loss']:.4f} | "
                f"ent={train_metrics['entropy_loss']:.4f} | kl={train_metrics['approx_kl']:.6f} | "
                f"clip={train_metrics['clip_fraction']:.3f} | grad={train_metrics['grad_norm']:.3f} | lr={lr_now:.6f}"
                f"{gpu_mb}",
                flush=True,
            )

            # Save checkpoint every save_freq steps (boundary-based so we never miss)
            if save_freq > 0 and (self.num_timesteps - last_checkpoint_step) >= save_freq:
                ckpt_path = self.model_dir / f"torch_ppo_step_{self.num_timesteps}.pt"
                self.save(ckpt_path)
                last_checkpoint_step = self.num_timesteps
                print(f"[TorchPPO] Checkpoint saved: {ckpt_path.resolve()}", flush=True)

            if visual_enabled and visual_freq > 0 and (self.num_timesteps - last_visual_eval_step) >= visual_freq:
                self.evaluate_visual(n_episodes=max(1, visual_episodes))
                last_visual_eval_step = self.num_timesteps

            # Eval every eval_freq steps (boundary-based so we never miss)
            if eval_freq > 0 and (self.num_timesteps - last_eval_step) >= eval_freq:
                eval_stats = self.evaluate(n_episodes=n_eval_episodes)
                eval_record = {
                    "step": int(self.num_timesteps),
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    **{k: float(v) for k, v in eval_stats.items() if isinstance(v, (int, float, np.integer, np.floating))},
                }
                with open(self.eval_history_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(eval_record) + "\n")
                mean_rew = eval_stats["mean_reward"]
                std_rew = eval_stats["std_reward"]
                is_first_eval = (best_eval_reward == -np.inf)
                is_new_best = mean_rew > best_eval_reward
                print(
                    f"[TorchPPO Eval] step={self.num_timesteps:,} | "
                    f"mean_reward={mean_rew:.2f} +/- {std_rew:.2f} | "
                    f"progress={eval_stats['mean_progress']:.2%} | "
                    f"offtrack_rate={eval_stats['offtrack_rate']:.2%} | "
                    f"steer_var={eval_stats['mean_steer_variance']:.5f} | "
                    f"mean_speed={eval_stats['mean_speed']:.2f} | "
                    f"throttle={eval_stats['mean_throttle']:.2f} | "
                    f"brake={eval_stats['mean_brake']:.2f} | "
                    f"stage={curriculum_stage}"
                    f"{' (first eval)' if is_first_eval else (' (new best!)' if is_new_best else '')}",
                    flush=True,
                )
                if is_first_eval or is_new_best:
                    best_eval_reward = mean_rew
                    best_path = self.model_dir / "best_model_torch.pt"
                    self.save(best_path)
                    step_path = self.model_dir / f"best_model_torch_step_{self.num_timesteps}.pt"
                    self.save(step_path)
                    print(
                        f"[TorchPPO] Best model saved (step {self.num_timesteps:,}, reward {mean_rew:.2f}):",
                        flush=True,
                    )
                    print(f"  -> {best_path.resolve()}", flush=True)
                    print(f"  -> (copy) {step_path.resolve()}", flush=True)
                if gate_enabled and n_eval_episodes >= gate_min_episodes:
                    if (
                        mean_rew >= gate_reward
                        and eval_stats["mean_progress"] >= gate_progress
                        and eval_stats["offtrack_rate"] <= gate_offtrack
                    ):
                        solved_path = self.model_dir / f"solved_model_torch_step_{self.num_timesteps}.pt"
                        self.save(solved_path)
                        print(
                            f"[TorchPPO] Success gate reached at step {self.num_timesteps:,}: "
                            f"reward={mean_rew:.2f}, progress={eval_stats['mean_progress']:.2%}, "
                            f"offtrack={eval_stats['offtrack_rate']:.2%}.",
                            flush=True,
                        )
                        print(f"[TorchPPO] Solved checkpoint: {solved_path.resolve()}", flush=True)
                        break
                if curriculum_enabled and curriculum_stage == 1:
                    if eval_stats["mean_progress"] >= promote_progress and eval_stats["mean_speed"] >= promote_speed:
                        self.env.env_method("set_curriculum_stage", 2)
                        self.eval_env.env_method("set_curriculum_stage", 2)
                        curriculum_stage = 2
                        print(
                            f"[TorchPPO Curriculum] Promoted to stage 2 at step {self.num_timesteps:,} "
                            f"(progress={eval_stats['mean_progress']:.2%}, speed={eval_stats['mean_speed']:.2f}).",
                            flush=True,
                        )
                if fail_fast_enabled and self.num_timesteps >= fail_fast_min_steps:
                    is_bad_eval = (
                        eval_stats["mean_progress"] < fail_fast_min_progress
                        and eval_stats["mean_speed"] < fail_fast_min_speed
                        and eval_stats.get("mean_throttle", 0.0) < 0.25
                    )
                    if is_bad_eval:
                        fail_fast_bad_evals += 1
                        print(
                            f"[TorchPPO FailFast] idle-pattern eval {fail_fast_bad_evals}/{fail_fast_patience} "
                            f"(progress={eval_stats['mean_progress']:.2%}, speed={eval_stats['mean_speed']:.2f}).",
                            flush=True,
                        )
                    else:
                        fail_fast_bad_evals = 0
                    if fail_fast_bad_evals >= fail_fast_patience:
                        fail_path = self.model_dir / f"failed_idle_model_torch_step_{self.num_timesteps}.pt"
                        self.save(fail_path)
                        print(
                            f"[TorchPPO FailFast] Stopping early due to persistent idling. "
                            f"Saved checkpoint: {fail_path.resolve()}",
                            flush=True,
                        )
                        break
                last_eval_step = self.num_timesteps


class SingleAgentWrapper(gym.Wrapper):
    """Wrap MultiCarRacing to expose a single-agent view."""

    def __init__(self, env):
        super().__init__(env)
        obs_space = env.observation_space
        act_space = env.action_space

        if len(obs_space.shape) == 4 and obs_space.shape[0] == 1:
            self.observation_space = gym.spaces.Box(
                low=obs_space.low[0],
                high=obs_space.high[0],
                shape=obs_space.shape[1:],
                dtype=obs_space.dtype
            )
        if len(act_space.shape) == 2 and act_space.shape[0] == 1:
            self.action_space = gym.spaces.Box(
                low=act_space.low[0],
                high=act_space.high[0],
                shape=act_space.shape[1:],
                dtype=act_space.dtype
            )

    def reset(self, **kwargs):
        # Gym 0.17.3 reset() returns just obs, not (obs, info)
        obs = self.env.reset(**kwargs)
        # Extract single agent observation if multi-agent format (num_agents, H, W, C)
        if hasattr(obs, "shape") and len(obs.shape) == 4 and obs.shape[0] == 1:
            obs = obs[0]  # Remove first dimension: (1, H, W, C) -> (H, W, C)
        elif isinstance(obs, (list, tuple)) and len(obs) == 1:
            obs = obs[0]
        # Gym 0.17.3: return just obs (not tuple)
        return obs

    def render(self, mode='human', **kwargs):
        out = self.env.render(mode=mode, **kwargs)
        # Extract single agent frame if multi-agent format (num_agents, H, W, C)
        if hasattr(out, "shape") and len(out.shape) == 4 and out.shape[0] == 1:
            out = out[0]
        return out

    def step(self, action):
        if hasattr(self.env.action_space, "shape") and len(self.env.action_space.shape) == 2:
            action = action.reshape(1, -1)
        obs, reward, done, info = self.env.step(action)
        # Extract single agent observation if multi-agent format (num_agents, H, W, C)
        if hasattr(obs, "shape") and len(obs.shape) == 4 and obs.shape[0] == 1:
            obs = obs[0]  # Remove first dimension: (1, H, W, C) -> (H, W, C)
        elif isinstance(obs, (list, tuple)) and len(obs) == 1:
            obs = obs[0]
        # Extract single agent reward if multi-agent format
        if isinstance(reward, (list, tuple)) or (hasattr(reward, "shape") and len(reward.shape) > 0 and reward.shape[0] == 1):
            reward = float(reward[0] if isinstance(reward, (list, tuple)) else reward[0])
        return obs, reward, done, info


class RewardShapingWrapper(gym.Wrapper):
    """Minimal reward shaping focused on speed, cornering, and stability."""

    def __init__(self, env, reward_config):
        super().__init__(env)
        reward_config = reward_config or {}
        self.enabled = bool(reward_config.get('enabled', True))
        self.use_custom_reward = bool(reward_config.get('use_custom_reward', True))

        self.forward_progress_scale = float(reward_config.get('forward_progress_scale', 1.0))
        # Dense alignment reward: velocity · track_dir. Gives step-by-step directional feedback
        # so the policy learns WHICH WAY to steer before it ever sees a tile progress signal.
        # Safe with off_track_mode=terminate since grass contact ends episode immediately.
        self.track_alignment_scale = float(reward_config.get('track_alignment_scale', 0.0))
        self.straight_speed_scale = float(reward_config.get('straight_speed_scale', 0.05))
        self.sharp_turn_threshold = float(reward_config.get('sharp_turn_threshold', 0.35))
        self.sharp_turn_lookahead = int(reward_config.get('sharp_turn_lookahead', 6))
        self.corner_target_speed = float(reward_config.get('corner_target_speed', 8.0))
        self.corner_overspeed_penalty_scale = float(
            reward_config.get('corner_overspeed_penalty_scale', 0.6)
        )
        self.apex_decel_reward_scale = float(reward_config.get('apex_decel_reward_scale', 0.4))
        self.apex_decel_reward_cap = float(reward_config.get('apex_decel_reward_cap', 1.0))
        self.time_penalty = float(reward_config.get('time_penalty', -0.1))
        self.steer_smoothness_penalty = float(reward_config.get('steer_smoothness_penalty', 0.05))
        self.steer_delta_cap = float(reward_config.get('steer_delta_cap', 0.5))
        # Lateral velocity penalty: penalises sliding/spinning perpendicular to track direction.
        # At speed=29 with steering, the car slides hard — this directly penalises that physics.
        self.lateral_velocity_penalty = float(reward_config.get('lateral_velocity_penalty', 0.0))
        # Magnitude penalty: penalises |steer|² to discourage full-lock steering / donuts.
        # Squared so small adjustments are cheap, extreme angles are costly.
        self.steer_magnitude_penalty = float(reward_config.get('steer_magnitude_penalty', 0.0))
        self.idle_speed_threshold = float(reward_config.get('idle_speed_threshold', 1.5))
        self.idle_penalty = float(reward_config.get('idle_penalty', -0.4))
        self.throttle_bonus_scale = float(reward_config.get('throttle_bonus_scale', 0.0))
        self.brake_penalty_scale = float(reward_config.get('brake_penalty_scale', 0.0))
        self.launch_boost_steps = int(reward_config.get('launch_boost_steps', 0))
        self.launch_speed_target = float(reward_config.get('launch_speed_target', 4.0))
        self.launch_bonus_scale = float(reward_config.get('launch_bonus_scale', 0.0))
        self.stuck_speed_threshold = float(reward_config.get('stuck_speed_threshold', 1.2))
        self.stuck_progress_epsilon = float(reward_config.get('stuck_progress_epsilon', 1e-3))
        self.stuck_max_steps = int(reward_config.get('stuck_max_steps', 120))
        self.stuck_terminal_penalty = float(reward_config.get('stuck_terminal_penalty', -50.0))
        # No-progress termination: catches donuts/spinning at high speed (speed check passes
        # but progress stays zero — normal stuck detector misses this).
        self.no_progress_max_steps = int(reward_config.get('no_progress_max_steps', 200))
        self.no_progress_terminal_penalty = float(
            reward_config.get('no_progress_terminal_penalty', -15.0)
        )
        # Yaw-rate penalty: directly taxes spinning/donuts by penalising angular velocity.
        self.yaw_rate_penalty = float(reward_config.get('yaw_rate_penalty', 0.0))

        # ── Local-geometry reward (new primary signal) ────────────────────────
        # Master switch. When True the local-geometry path is used and tile-progress
        # becomes purely auxiliary (set forward_progress_scale: 0.0 to disable it).
        self.use_local_geometry_reward = bool(reward_config.get('use_local_geometry_reward', False))
        # Scale for the forward-motion term: forward_local_scale * max(0, v_parallel)
        self.forward_local_scale = float(reward_config.get('forward_local_scale', 2.0))
        # Scale for the controlled-speed bonus (gated by center and alignment)
        self.controlled_speed_scale = float(reward_config.get('controlled_speed_scale', 1.0))
        # Base scale for lateral-deviation penalty (softened in curves by curvature_relief_scale)
        self.center_penalty_scale = float(reward_config.get('center_penalty_scale', 0.5))
        # Normalised lateral offset (fraction of road half-width) inside which no penalty applies
        self.center_free_corridor = float(reward_config.get('center_free_corridor', 0.15))
        # Normalised offset at which center_gate reaches 0 (for speed-bonus gating)
        self.center_distance_tolerance = float(reward_config.get('center_distance_tolerance', 0.4))
        # Divides center_penalty_scale in curves: effective_scale = base / (1 + scale * |curvature|)
        self.curvature_relief_scale = float(reward_config.get('curvature_relief_scale', 5.0))
        # Number of image rows ahead of the car to sample for the local centerline
        self.local_lookahead_rows = int(reward_config.get('local_lookahead_rows', 30))
        # Vertical position of the car in the image as a fraction of H (0=top, 1=bottom)
        self.local_car_row_frac = float(reward_config.get('local_car_row_frac', 0.65))
        # Minimum road pixels in a row for it to be included in the centerline fit
        self.local_min_road_px = int(reward_config.get('local_min_road_px', 5))
        # When True, logs all local-geometry fields into info dict each step (slow — debug only)
        self.local_geometry_debug = bool(reward_config.get('local_geometry_debug', False))

        self.off_track_mode = str(reward_config.get('off_track_mode', 'terminate')).strip().lower()
        self.off_track_terminal_penalty = float(
            reward_config.get('off_track_terminal_penalty', -100.0)
        )
        self.off_track_step_penalty = float(reward_config.get('off_track_step_penalty', -25.0))
        self.curriculum = reward_config.get('curriculum', {}) or {}
        self.curriculum_enabled = bool(self.curriculum.get('enabled', False))
        self.curriculum_stage = int(self.curriculum.get('start_stage', 1))
        self._stage2_defaults = {
            "time_penalty": self.time_penalty,
            "idle_penalty": self.idle_penalty,
            "off_track_mode": self.off_track_mode,
            "off_track_terminal_penalty": self.off_track_terminal_penalty,
            "off_track_step_penalty": self.off_track_step_penalty,
            "corner_overspeed_penalty_scale": self.corner_overspeed_penalty_scale,
        }
        if self.curriculum_enabled:
            self._apply_curriculum_stage(self.curriculum_stage)

        self._prev_steer = None
        self._prev_speed = None
        self._lap_count = 0
        self._prev_progress = None
        self._lap_completion_latched = False
        self._stuck_steps = 0
        self._no_progress_steps = 0
        self._last_stuck_progress = None
        self._episode_steps = 0

    def reset(self, **kwargs):
        self._prev_steer = None
        self._prev_speed = None
        self._lap_count = 0
        self._prev_progress = None
        self._lap_completion_latched = False
        self._stuck_steps = 0
        self._no_progress_steps = 0
        self._last_stuck_progress = None
        self._episode_steps = 0
        return self.env.reset(**kwargs)

    def _apply_curriculum_stage(self, stage: int):
        stage = int(stage)
        if stage <= 1:
            self.time_penalty = float(self.curriculum.get("stage1_time_penalty", -0.05))
            self.idle_penalty = float(self.curriculum.get("stage1_idle_penalty", -0.8))
            self.off_track_mode = str(self.curriculum.get("stage1_off_track_mode", "penalty")).strip().lower()
            self.off_track_terminal_penalty = float(
                self.curriculum.get("stage1_off_track_terminal_penalty", -40.0)
            )
            self.off_track_step_penalty = float(self.curriculum.get("stage1_off_track_step_penalty", -8.0))
            self.corner_overspeed_penalty_scale = float(
                self.curriculum.get("stage1_corner_overspeed_penalty_scale", 0.3)
            )
        else:
            self.time_penalty = float(self._stage2_defaults["time_penalty"])
            self.idle_penalty = float(self._stage2_defaults["idle_penalty"])
            self.off_track_mode = str(self._stage2_defaults["off_track_mode"])
            self.off_track_terminal_penalty = float(self._stage2_defaults["off_track_terminal_penalty"])
            self.off_track_step_penalty = float(self._stage2_defaults["off_track_step_penalty"])
            self.corner_overspeed_penalty_scale = float(
                self._stage2_defaults["corner_overspeed_penalty_scale"]
            )
        self.curriculum_stage = stage

    def set_curriculum_stage(self, stage: int):
        if not self.curriculum_enabled:
            return False
        prev = int(self.curriculum_stage)
        self._apply_curriculum_stage(stage)
        return prev != int(self.curriculum_stage)

    @staticmethod
    def _normalize(vec):
        norm = float(np.linalg.norm(vec))
        if norm <= 1e-8:
            return np.zeros_like(vec), 0.0
        return vec / norm, norm

    def _get_track_context(self, base_env, car):
        if not (hasattr(base_env, "track") and base_env.track):
            return np.zeros(2, dtype=np.float32), False, 0.0

        track_xy = np.array(base_env.track)[:, 2:]
        if len(track_xy) < 2:
            return np.zeros(2, dtype=np.float32), False, 0.0

        car_pos = np.array(car.hull.position).reshape((1, 2))
        track_index = int(np.argmin(np.linalg.norm(car_pos - track_xy, ord=2, axis=1)))
        next_index = (track_index + 1) % len(track_xy)
        track_vec = track_xy[next_index] - track_xy[track_index]
        track_dir, _ = self._normalize(track_vec.astype(np.float32))

        lookahead_index = (track_index + self.sharp_turn_lookahead) % len(base_env.track)
        beta_now = float(base_env.track[track_index][1])
        beta_next = float(base_env.track[lookahead_index][1])
        angle_diff = abs(beta_next - beta_now)
        if angle_diff > np.pi:
            angle_diff = abs(angle_diff - 2 * np.pi)
        is_sharp_turn = angle_diff >= self.sharp_turn_threshold
        return track_dir, is_sharp_turn, float(angle_diff)

    def _update_lap_count(self, info):
        # info["progress"] is written by step() before this is called on subsequent steps,
        # but on the very first call it may be absent — safe to skip if missing.
        progress = info.get("progress")
        if progress is None:
            return
        try:
            progress = float(progress)
        except (TypeError, ValueError):
            return
        if self._prev_progress is not None and progress < (self._prev_progress - 0.5):
            self._lap_count += 1
        if progress >= 0.999 and not self._lap_completion_latched:
            self._lap_count += 1
            self._lap_completion_latched = True
        if progress < 0.2:
            self._lap_completion_latched = False
        self._prev_progress = progress

    # ── Local road-geometry helpers ───────────────────────────────────────────

    @staticmethod
    def _extract_road_mask(obs: np.ndarray) -> np.ndarray:
        """Return a boolean mask (H, W) of pixels that look like drivable road.

        Road is grayish (R≈G≈B, mid-range intensity).
        Grass is green (G >> R, G >> B).
        Curbs alternate red/white — excluded by the grayness test.

        Uses simple RGB thresholds with no HSV conversion.
        Threshold values may need tuning if the renderer or track skin changes.
        """
        obs_i = np.asarray(obs, dtype=np.int32)   # signed ints so diffs cannot wrap
        R, G, B = obs_i[:, :, 0], obs_i[:, :, 1], obs_i[:, :, 2]
        is_gray = (np.abs(R - G) < 25) & (np.abs(R - B) < 25) & (np.abs(G - B) < 25)
        intensity = R + G + B
        # Exclude very dark pixels (road markings, car shadow) and very bright (white curb stripe)
        in_range = (intensity > 150) & (intensity < 480)
        # Explicitly exclude clearly green pixels (grass) regardless of grayness
        is_green = (G - R > 20) & (G - B > 20)
        road_mask = is_gray & in_range & (~is_green)
        return road_mask.astype(bool)

    def _estimate_local_centerline(self, road_mask: np.ndarray) -> list:
        """Sample local road-centre points from the visible crop.

        Returns a list of (row, col) tuples where col is the horizontal centroid
        of road pixels at that image row.  Rows are sampled from just ahead of the
        car up to local_lookahead_rows further ahead (lower row index = further ahead
        because row 0 is at the top of the ego-centred image).
        """
        H, W = road_mask.shape
        car_row = int(H * self.local_car_row_frac)
        # Leave a 5-row buffer below car_row so the car body does not bias the centroid.
        start_row = max(car_row - 5, 0)
        end_row = max(car_row - 5 - self.local_lookahead_rows, 0)
        pts = []
        for r in range(start_row, end_row - 1, -1):
            road_cols = np.where(road_mask[r])[0]
            if len(road_cols) < self.local_min_road_px:
                continue
            col_center = float(np.mean(road_cols))
            pts.append((float(r), col_center))
        return pts

    def _estimate_local_tangent_curvature(self, centerline_pts: list) -> tuple:
        """Fit local road tangent slope and curvature from sampled centreline points.

        Linear fit:    col = a*row + b          → tangent slope a = d(col)/d(row)
        Quadratic fit: col = a*row² + b*row + c → curvature ≈ 2*a (2nd derivative)

        Returns (tangent_slope, curvature_scalar).
        Falls back to (0.0, 0.0) — i.e. straight-ahead — if too few points.
        """
        if len(centerline_pts) < 2:
            return 0.0, 0.0
        rows = np.array([p[0] for p in centerline_pts], dtype=np.float64)
        cols = np.array([p[1] for p in centerline_pts], dtype=np.float64)
        try:
            linear_coeffs = np.polyfit(rows, cols, 1)
            tangent_slope = float(linear_coeffs[0])
        except (np.linalg.LinAlgError, ValueError):
            tangent_slope = 0.0
        curvature = 0.0
        if len(centerline_pts) >= 3:
            try:
                quad_coeffs = np.polyfit(rows, cols, 2)
                curvature = float(2.0 * quad_coeffs[0])   # 2nd derivative
            except (np.linalg.LinAlgError, ValueError):
                curvature = 0.0
        return tangent_slope, curvature

    def _compute_lateral_offset(self, road_mask: np.ndarray) -> tuple:
        """Compute the car's lateral offset from the local road centre.

        Returns (normalised_offset, road_half_width_px) where
            normalised_offset = (road_centre_col - car_col) / road_half_width.
        Positive = road centre is to the right of the car (car is left of centre).
        Negative = road centre is to the left of the car (car is right of centre).
        Returns (0.0, 1.0) when road is not visible near the car's row.
        """
        H, W = road_mask.shape
        car_row = int(H * self.local_car_row_frac)
        car_col = W // 2   # car is horizontally centred in the ego-view
        window = 4
        r0 = max(car_row - window, 0)
        r1 = min(car_row + window, H - 1)
        road_cols = np.where(road_mask[r0:r1 + 1].any(axis=0))[0]
        if len(road_cols) < self.local_min_road_px:
            return 0.0, 1.0
        road_centre = float(np.mean(road_cols))
        road_half_width = max(float(np.max(road_cols) - np.min(road_cols)) / 2.0, 1.0)
        normalised_offset = (road_centre - car_col) / road_half_width
        return float(normalised_offset), road_half_width

    def _compute_local_geometry(self, obs: np.ndarray) -> dict:
        """Orchestrate local road geometry extraction from the current obs crop.

        Returns a dict with keys:
            road_mask       — (H, W) bool array
            centerline_pts  — list of (row, col) tuples from lookahead scan
            tangent_slope   — d(col)/d(row) of fitted local centreline
            curvature       — 2nd derivative of fitted centreline (approx)
            lateral_offset  — normalised lateral displacement of car from road centre
            road_half_width — road half-width in pixels at the car's row
        """
        road_mask = self._extract_road_mask(obs)
        centerline_pts = self._estimate_local_centerline(road_mask)
        tangent_slope, curvature = self._estimate_local_tangent_curvature(centerline_pts)
        lateral_offset, road_half_width = self._compute_lateral_offset(road_mask)
        return {
            "road_mask": road_mask,
            "centerline_pts": centerline_pts,
            "tangent_slope": tangent_slope,
            "curvature": curvature,
            "lateral_offset": lateral_offset,
            "road_half_width": road_half_width,
        }

    @staticmethod
    def _tangent_slope_to_world_vec(tangent_slope: float, car_angle: float) -> np.ndarray:
        """Convert pixel-space tangent slope to a normalised world-space direction vector.

        Image convention (ego-centred, camera rotates with car):
          - row 0 is at the top (far ahead), row H-1 at the bottom (behind the car)
          - col increases to the right
          - "forward in image" = decreasing row = car-local forward direction
          - "right in image"   = increasing col = car-local right direction

        Given slope a = d(col)/d(row), the road tangent in image space points toward
        lower row numbers (forward), so the image-space direction is (col=a, row=-1).
        Mapping to car-local frame (x=right, y=forward):
            car_local_x = a   (col → right)
            car_local_y = 1   (−row → forward)

        Box2D world convention: x=right, y=up; car.hull.angle = rotation from +x axis.
        ASSUMPTION: angle=0 means the car points in the +y (world-up) direction, which
        is standard for CarRacing-style Box2D environments.  If angle=0 actually points
        along +x, swap sin and cos in the rotation below.

        Car-local → world rotation:
            car forward (local y) in world = (−sin(angle), cos(angle))
            car right   (local x) in world = ( cos(angle), sin(angle))

        Returns normalised (wx, wy) float32 array.
        """
        local_x = float(tangent_slope)
        local_y = 1.0
        norm = np.sqrt(local_x ** 2 + local_y ** 2)
        if norm < 1e-8:
            norm = 1.0
        local_x /= norm
        local_y /= norm
        ca, sa = np.cos(car_angle), np.sin(car_angle)
        wx = local_x * ca + local_y * (-sa)
        wy = local_x * sa + local_y * ca
        result = np.array([wx, wy], dtype=np.float32)
        n = float(np.linalg.norm(result))
        if n > 1e-8:
            result /= n
        return result

    def _compute_local_forward_term(self, v_parallel: float) -> float:
        """Reward forward motion along the locally estimated road direction.

        Only positive v_parallel is rewarded — backward motion clips to 0.
        The time_penalty and idle_penalty already discourage standing still.
        """
        return self.forward_local_scale * max(0.0, v_parallel)

    def _compute_center_penalty(self, lateral_offset: float, curvature: float) -> float:
        """Soft quadratic penalty for lateral deviation from the local road centre.

        A free corridor [0, center_free_corridor] has zero penalty.
        Beyond the corridor the penalty grows quadratically.
        In sharp curves the penalty is reduced via curvature_relief_scale so the car
        can take a faster racing line without being forced to hug the exact centre.
        """
        effective_scale = self.center_penalty_scale / (
            1.0 + self.curvature_relief_scale * abs(curvature)
        )
        excess = max(0.0, abs(lateral_offset) - self.center_free_corridor)
        denom = max(1.0 - self.center_free_corridor, 1e-6)
        normalised_excess = excess / denom
        return effective_scale * (normalised_excess ** 2)

    def _compute_controlled_speed_bonus(
        self, v_parallel: float, lateral_offset: float, alignment: float
    ) -> float:
        """Reward speed only when it is aligned with the road and within the drivable corridor.

        center_gate    → 0 when lateral offset exceeds center_distance_tolerance
        alignment_gate → 0 when velocity direction is not aligned with road tangent
        Both gates multiply the speed bonus so all three conditions must hold.
        This lets the car carry speed through a curve that deviates slightly from centre,
        but prevents rewarding fast driving that is not actually along the road.
        """
        center_gate = max(
            0.0, 1.0 - abs(lateral_offset) / max(self.center_distance_tolerance, 1e-6)
        )
        alignment_gate = max(0.0, alignment)
        return self.controlled_speed_scale * max(0.0, v_parallel) * center_gate * alignment_gate

    def _check_off_track_terminal(self, is_offtrack: bool) -> tuple:
        """Handle off-track detection according to off_track_mode config.

        Returns (is_terminal: bool, reward_adjustment: float).
        terminate mode: episode ends immediately with a large negative reward.
        penalty mode:   a per-step additive penalty; episode continues.
        """
        if not is_offtrack:
            return False, 0.0
        if self.off_track_mode == "terminate":
            return True, float(self.off_track_terminal_penalty)
        return False, float(self.off_track_step_penalty)

    def _build_reward_debug_info(
        self,
        geo: dict,
        tangent_world: np.ndarray,
        v_parallel: float,
        alignment: float,
        center_gate: float,
        alignment_gate: float,
        comp_forward_local: float,
        comp_center_penalty: float,
        comp_speed_bonus: float,
        comp_time: float,
        comp_idle: float,
        comp_yaw: float,
    ) -> dict:
        """Build a structured debug dict for local-geometry reward fields.

        Only called when local_geometry_debug=True.  Adds lg/ prefixed fields to
        info so they can be inspected in logs or tensorboard without changing any
        normal monitoring code that reads the standard reward keys.
        """
        return {
            "lg/tangent_slope": float(geo["tangent_slope"]),
            "lg/curvature": float(geo["curvature"]),
            "lg/lateral_offset": float(geo["lateral_offset"]),
            "lg/road_half_width_px": float(geo["road_half_width"]),
            "lg/centerline_n_pts": int(len(geo["centerline_pts"])),
            "lg/tangent_world_x": float(tangent_world[0]),
            "lg/tangent_world_y": float(tangent_world[1]),
            "lg/v_parallel": float(v_parallel),
            "lg/alignment": float(alignment),
            "lg/center_gate": float(center_gate),
            "lg/alignment_gate": float(alignment_gate),
            "lg/comp_forward_local": float(comp_forward_local),
            "lg/comp_center_penalty": float(comp_center_penalty),
            "lg/comp_speed_bonus": float(comp_speed_bonus),
            "lg/comp_time": float(comp_time),
            "lg/comp_idle": float(comp_idle),
            "lg/comp_yaw": float(comp_yaw),
            "lg/road_mask_sum": int(geo["road_mask"].sum()),
        }

    # ─────────────────────────────────────────────────────────────────────────

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if not isinstance(info, dict):
            info = {}
        self._episode_steps += 1

        # Compute and inject progress BEFORE _update_lap_count so it has correct data.
        # The base env always returns info={} — we read tile_visited_count directly.
        base_env = self.env.unwrapped
        if (hasattr(base_env, "tile_visited_count") and hasattr(base_env, "track")
                and base_env.track and len(base_env.track) > 0):
            info["progress"] = float(base_env.tile_visited_count[0]) / float(len(base_env.track))

        self._update_lap_count(info)

        speed = 0.0
        track_dir = np.zeros(2, dtype=np.float32)
        is_sharp_turn = False
        corner_angle = 0.0
        velocity_vec = np.zeros(2, dtype=np.float32)
        # car_angle: Box2D hull angle in radians; 0.0 = safe default when car unavailable.
        # Used by _tangent_slope_to_world_vec to convert pixel-space road tangent to world space.
        car_angle = 0.0
        if hasattr(base_env, "cars") and base_env.cars:
            car = base_env.cars[0]
            vel = car.hull.linearVelocity
            velocity_vec = np.array([vel[0], vel[1]], dtype=np.float32)
            speed = float(np.linalg.norm(velocity_vec))
            car_angle = float(car.hull.angle)
            track_dir, is_sharp_turn, corner_angle = self._get_track_context(base_env, car)

        # Angular velocity (yaw rate) — high when doing donuts.
        yaw_rate = 0.0
        if hasattr(base_env, "cars") and base_env.cars:
            car = base_env.cars[0]
            if hasattr(car, "hull"):
                yaw_rate = abs(float(car.hull.angularVelocity))

        action_arr = np.asarray(action).reshape(-1) if action is not None else np.zeros(3, dtype=np.float32)
        steer_value = float(np.clip(action_arr[0], -1.0, 1.0)) if action_arr.size >= 1 else 0.0
        throttle_value = float(np.clip(action_arr[1], 0.0, 1.0)) if action_arr.size >= 2 else 0.0
        brake_value = float(np.clip(action_arr[2], 0.0, 1.0)) if action_arr.size >= 3 else 0.0

        # Read progress from info (written at top of step from tile_visited_count).
        progress_now = info.get("progress")  # float or None

        progress_delta = 0.0
        if progress_now is not None:
            if self._last_stuck_progress is not None:
                progress_delta = max(0.0, progress_now - self._last_stuck_progress)
            self._last_stuck_progress = progress_now

        # Detect off-track early so we can gate the speed reward.
        driving_on_grass = getattr(base_env, "driving_on_grass", None)
        is_offtrack = bool(
            driving_on_grass is not None
            and len(driving_on_grass) > 0
            and bool(driving_on_grass[0])
        )

        # ── Reward computation ────────────────────────────────────────────────
        # Two paths: local-geometry-based (new) or tile-progress-based (legacy).
        # Switch via use_local_geometry_reward in config. Old code is fully preserved.
        already_terminated = False

        if self.use_local_geometry_reward and self.use_custom_reward:
            # ── NEW PATH: local-geometry-based reward ─────────────────────────
            # Off-track is checked first. In 'terminate' mode it ends the episode
            # immediately with a crash penalty; in 'penalty' mode a per-step
            # adjustment is returned and the episode continues.
            is_terminal_offtrack, offtrack_adj = self._check_off_track_terminal(is_offtrack)

            if is_terminal_offtrack:
                # Fatal crash: skip all reward shaping, set done and terminate.
                total_reward = offtrack_adj
                done = True
                already_terminated = True
                # Zero-fill local-geometry fields so info logging is consistent.
                geo = {
                    "tangent_slope": 0.0, "curvature": 0.0,
                    "lateral_offset": 0.0, "road_half_width": 1.0,
                    "centerline_pts": [],
                    "road_mask": np.zeros((1, 1), dtype=bool),
                }
                tangent_world = np.array([0.0, 1.0], dtype=np.float32)
                v_parallel = 0.0
                alignment = 0.0
                center_gate = 0.0
                alignment_gate = 0.0
                comp_forward_local = 0.0
                comp_center_penalty = 0.0
                comp_speed_bonus = 0.0
                comp_time_lg = 0.0
                comp_idle_lg = 0.0
                comp_yaw_lg = 0.0
            else:
                # Extract local road geometry from the current ego-centred observation.
                geo = self._compute_local_geometry(obs)
                tangent_world = self._tangent_slope_to_world_vec(
                    geo["tangent_slope"], car_angle
                )
                # Project world velocity onto the locally estimated road tangent.
                v_parallel = float(np.dot(velocity_vec, tangent_world))
                vel_len = float(np.linalg.norm(velocity_vec))
                vel_norm = (
                    velocity_vec / vel_len if vel_len > 1e-6
                    else np.zeros(2, dtype=np.float32)
                )
                # alignment ∈ [-1, 1]: 1 = perfectly aligned with road, -1 = going backward
                alignment = float(np.dot(vel_norm, tangent_world))

                # Gates (also stored for debug logging).
                center_gate = max(
                    0.0,
                    1.0 - abs(geo["lateral_offset"]) / max(self.center_distance_tolerance, 1e-6),
                )
                alignment_gate = max(0.0, alignment)

                # ── Individual reward components ──────────────────────────────
                # 1. Forward-motion term: reward movement along local road direction.
                comp_forward_local = self._compute_local_forward_term(v_parallel)

                # 2. Soft centre-deviation penalty (weakened in curves).
                comp_center_penalty = self._compute_center_penalty(
                    geo["lateral_offset"], geo["curvature"]
                )

                # 3. Controlled-speed bonus: speed rewarded only when aligned and centred.
                comp_speed_bonus = (
                    self.controlled_speed_scale
                    * max(0.0, v_parallel)
                    * center_gate
                    * alignment_gate
                )

                comp_time_lg = float(self.time_penalty)
                comp_idle_lg = float(self.idle_penalty) if speed < self.idle_speed_threshold else 0.0
                comp_yaw_lg = -self.yaw_rate_penalty * yaw_rate

                # 4. Throttle bonus / brake penalty: directly incentivise forward action.
                # These were stored in self but never applied in this path (bug fix).
                comp_throttle_lg = self.throttle_bonus_scale * throttle_value
                comp_brake_pen_lg = -self.brake_penalty_scale * brake_value

                # ── Assemble total reward ─────────────────────────────────────
                # reward = forward_term + speed_bonus - center_penalty
                #        + time_penalty + idle_penalty + yaw_penalty
                #        + throttle_bonus + brake_penalty
                #        + offtrack_adj  (0 in terminate mode; step penalty in penalty mode)
                total_reward = (
                    comp_forward_local
                    + comp_speed_bonus
                    - comp_center_penalty
                    + comp_time_lg
                    + comp_idle_lg
                    + comp_yaw_lg
                    + comp_throttle_lg
                    + comp_brake_pen_lg
                    + offtrack_adj
                )

            # Stuck and no-progress detectors remain active in the new path.
            # They use tile-count progress_delta as a termination guard (not a reward
            # signal), so they still work even without the tile-progress reward.
            if not already_terminated:
                if speed < self.stuck_speed_threshold and progress_delta < self.stuck_progress_epsilon:
                    self._stuck_steps += 1
                else:
                    self._stuck_steps = 0
                is_stuck = self._stuck_steps >= self.stuck_max_steps
                if is_stuck:
                    total_reward = float(self.stuck_terminal_penalty)
                    done = True
                    already_terminated = True
            else:
                self._stuck_steps = 0
                is_stuck = False

            if not already_terminated:
                if progress_delta < self.stuck_progress_epsilon:
                    self._no_progress_steps += 1
                else:
                    self._no_progress_steps = 0
                is_no_progress = (
                    (not is_stuck)
                    and self._no_progress_steps >= self.no_progress_max_steps
                )
                if is_no_progress:
                    total_reward = float(self.no_progress_terminal_penalty)
                    done = True
            else:
                is_no_progress = False

            self._prev_steer = steer_value
            self._prev_speed = speed

            # ── Info: telemetry and reward decomposition ──────────────────────
            info["events/offtrack"] = int(is_offtrack)
            info["events/stuck"] = int(is_stuck)
            info["events/no_progress"] = int(is_no_progress)
            info["telemetry/yaw_rate"] = float(yaw_rate)
            info["telemetry/curriculum_stage"] = int(self.curriculum_stage)
            info["telemetry/speed"] = float(speed)
            info["telemetry/is_corner"] = 0        # not computed in local-geometry path
            info["telemetry/corner_angle"] = 0.0   # not computed in local-geometry path
            info["telemetry/steer"] = float(steer_value)
            info["telemetry/throttle"] = float(throttle_value)
            info["telemetry/brake"] = float(brake_value)
            # Legacy reward keys zeroed so downstream monitors do not crash.
            info["rewards/forward_progress"] = 0.0
            info["rewards/alignment"] = 0.0
            info["rewards/straight_speed"] = 0.0
            info["rewards/corner_overspeed"] = 0.0
            info["rewards/apex_decel"] = 0.0
            info["rewards/steer_smoothness"] = 0.0
            info["rewards/steer_magnitude"] = 0.0
            info["rewards/lateral"] = 0.0
            info["rewards/time"] = float(comp_time_lg)
            info["rewards/idle"] = float(comp_idle_lg)
            info["rewards/throttle"] = float(comp_throttle_lg) if not already_terminated else 0.0
            info["rewards/brake"] = float(comp_brake_pen_lg) if not already_terminated else 0.0
            info["rewards/launch"] = 0.0
            info["rewards/yaw"] = float(comp_yaw_lg)
            info["rewards/total"] = float(total_reward)
            # New local-geometry reward keys.
            info["rewards/lg_forward"] = float(comp_forward_local)
            info["rewards/lg_speed_bonus"] = float(comp_speed_bonus)
            info["rewards/lg_center_penalty"] = float(comp_center_penalty)
            info["lap_count"] = int(self._lap_count)

            if self.local_geometry_debug:
                debug_dict = self._build_reward_debug_info(
                    geo, tangent_world, v_parallel, alignment,
                    center_gate, alignment_gate,
                    comp_forward_local, comp_center_penalty, comp_speed_bonus,
                    comp_time_lg, comp_idle_lg, comp_yaw_lg,
                )
                info.update(debug_dict)

        else:
            # ── LEGACY PATH: tile-progress-based reward (fully preserved) ─────
            # Primary reward: actual tile progress (0→1 per lap). Scaled up so one full
            # lap ≈ forward_progress_scale total reward.
            comp_forward = self.forward_progress_scale * progress_delta

            # Dense directional guidance: velocity · track_dir. Tells the policy which way
            # to steer every step — without this, progress_delta=0 for "wrong direction"
            # AND "not moving", so the policy gets no steering correction signal.
            # Safe because off_track_mode=terminate prevents off-track exploitation.
            comp_alignment = self.track_alignment_scale * float(np.dot(velocity_vec, track_dir))

            # Speed bonus only while ON track — prevents gaming it by driving on grass.
            comp_straight_speed = self.straight_speed_scale * speed * (0.0 if is_offtrack else 1.0) if not is_sharp_turn else 0.0

            # Lateral velocity penalty: penalises sliding perpendicular to the track.
            # When car goes fast and oversteers (speed=29), it slides sideways — this directly
            # penalises the loss-of-control physics that causes tailspins and off-track.
            lateral_dir = np.array([-track_dir[1], track_dir[0]], dtype=np.float32)
            lateral_speed = float(np.dot(velocity_vec, lateral_dir))
            comp_lateral = -self.lateral_velocity_penalty * abs(lateral_speed)
            corner_overspeed = max(0.0, speed - self.corner_target_speed) if is_sharp_turn else 0.0
            comp_corner_overspeed = -self.corner_overspeed_penalty_scale * corner_overspeed

            speed_delta = 0.0
            if self._prev_speed is not None:
                speed_delta = self._prev_speed - speed
            comp_apex_decel = 0.0
            if is_sharp_turn and speed_delta > 0.0:
                comp_apex_decel = min(self.apex_decel_reward_scale * speed_delta, self.apex_decel_reward_cap)

            comp_steer_smooth = 0.0
            if self._prev_steer is not None:
                steer_delta = abs(steer_value - self._prev_steer)
                if self.steer_delta_cap > 0.0:
                    steer_delta = min(steer_delta, self.steer_delta_cap)
                comp_steer_smooth = -self.steer_smoothness_penalty * steer_delta
            # Magnitude penalty: |steer|² — makes extreme lock expensive, gentle curves cheap.
            comp_steer_mag = -self.steer_magnitude_penalty * (steer_value ** 2)
            self._prev_steer = steer_value
            self._prev_speed = speed

            comp_time = float(self.time_penalty)
            comp_idle = float(self.idle_penalty) if speed < self.idle_speed_threshold else 0.0
            comp_throttle = float(self.throttle_bonus_scale * throttle_value)
            comp_brake = 0.0
            if not is_sharp_turn:
                comp_brake = -float(self.brake_penalty_scale * brake_value)
            comp_launch = 0.0
            if self._episode_steps <= self.launch_boost_steps and speed < self.launch_speed_target:
                comp_launch = float(self.launch_bonus_scale * throttle_value)
            # Yaw-rate penalty: penalise spinning/donuts directly via angular velocity.
            comp_yaw = -self.yaw_rate_penalty * yaw_rate
            shaped_reward = (
                comp_forward
                + comp_alignment
                + comp_straight_speed
                + comp_lateral
                + comp_corner_overspeed
                + comp_apex_decel
                + comp_steer_smooth
                + comp_steer_mag
                + comp_time
                + comp_idle
                + comp_throttle
                + comp_brake
                + comp_launch
                + comp_yaw
            )

            total_reward = shaped_reward if self.use_custom_reward else float(reward)
            if is_offtrack:
                if self.off_track_mode == "terminate":
                    total_reward = float(self.off_track_terminal_penalty)
                    done = True
                    already_terminated = True
                else:
                    total_reward += float(self.off_track_step_penalty)

            # Stuck detection: speed-based (slow AND no progress).
            if not already_terminated:
                if speed < self.stuck_speed_threshold and progress_delta < self.stuck_progress_epsilon:
                    self._stuck_steps += 1
                else:
                    self._stuck_steps = 0
                is_stuck = self._stuck_steps >= self.stuck_max_steps
                if is_stuck:
                    total_reward = float(self.stuck_terminal_penalty)
                    done = True
                    already_terminated = True
            else:
                self._stuck_steps = 0
                is_stuck = False

            # No-progress detection: catches donuts — fast but zero tile advancement.
            # Separate from speed-stuck so spinning at high speed still terminates.
            if not already_terminated:
                if progress_delta < self.stuck_progress_epsilon:
                    self._no_progress_steps += 1
                else:
                    self._no_progress_steps = 0
                is_no_progress = (not is_stuck) and self._no_progress_steps >= self.no_progress_max_steps
                if is_no_progress:
                    total_reward = float(self.no_progress_terminal_penalty)
                    done = True
            else:
                is_no_progress = False

            info["events/offtrack"] = int(is_offtrack)
            info["events/stuck"] = int(is_stuck)
            info["events/no_progress"] = int(is_no_progress)
            info["telemetry/yaw_rate"] = float(yaw_rate)
            info["telemetry/curriculum_stage"] = int(self.curriculum_stage)
            info["telemetry/speed"] = float(speed)
            info["telemetry/is_corner"] = int(is_sharp_turn)
            info["telemetry/corner_angle"] = float(corner_angle)
            info["telemetry/steer"] = float(steer_value)
            info["telemetry/throttle"] = float(throttle_value)
            info["telemetry/brake"] = float(brake_value)
            info["rewards/forward_progress"] = float(comp_forward)
            info["rewards/alignment"] = float(comp_alignment)
            info["rewards/straight_speed"] = float(comp_straight_speed)
            info["rewards/corner_overspeed"] = float(comp_corner_overspeed)
            info["rewards/apex_decel"] = float(comp_apex_decel)
            info["rewards/steer_smoothness"] = float(comp_steer_smooth)
            info["rewards/steer_magnitude"] = float(comp_steer_mag)
            info["rewards/lateral"] = float(comp_lateral)
            info["rewards/time"] = float(comp_time)
            info["rewards/idle"] = float(comp_idle)
            info["rewards/throttle"] = float(comp_throttle)
            info["rewards/brake"] = float(comp_brake)
            info["rewards/launch"] = float(comp_launch)
            info["rewards/yaw"] = float(comp_yaw)
            info["rewards/total"] = float(total_reward)
            # New local-geometry keys zeroed so downstream monitors never KeyError.
            info["rewards/lg_forward"] = 0.0
            info["rewards/lg_speed_bonus"] = 0.0
            info["rewards/lg_center_penalty"] = 0.0
            info["lap_count"] = int(self._lap_count)

        return obs, float(total_reward), done, info


class SafetyGovernorWrapper(gym.Wrapper):
    """Optional speed cap to keep the agent below a target velocity."""

    def __init__(self, env, governor_config):
        super().__init__(env)
        governor_config = governor_config or {}
        self.enabled = bool(
            governor_config.get('enabled', governor_config.get('speed_cap_enabled', False))
        )
        self.speed_cap_ratio = float(governor_config.get('speed_cap_ratio', 0.5))
        self.speed_cap_top_speed = float(governor_config.get('speed_cap_top_speed', 30.0))
        self.speed_cap_brake = float(governor_config.get('speed_cap_brake', 0.2))

    def step(self, action):
        if self.enabled and action is not None:
            base_env = self.env.unwrapped
            if hasattr(base_env, "cars") and base_env.cars:
                car = base_env.cars[0]
                vel = car.hull.linearVelocity
                speed = float(np.linalg.norm([vel[0], vel[1]]))
                speed_cap = self.speed_cap_ratio * self.speed_cap_top_speed
                if speed_cap > 0.0 and speed > speed_cap:
                    action_arr = np.asarray(action).copy()
                    orig_shape = action_arr.shape
                    action_arr = action_arr.reshape(-1)
                    if action_arr.size >= 3:
                        action_arr[1] = 0.0
                        action_arr[2] = max(float(action_arr[2]), self.speed_cap_brake)
                    action = action_arr.reshape(orig_shape)
        return self.env.step(action)


class ObservationAugmentWrapper(gym.Wrapper):
    """Augment observations with angular velocity, centerline distance, and look-ahead angles."""

    def __init__(self, env, obs_config):
        super().__init__(env)
        obs_config = obs_config or {}
        self.enabled = bool(obs_config.get('enabled', False))
        if not self.enabled:
            return

        image_space = env.observation_space
        if len(image_space.shape) != 3:
            raise ValueError("Expected image observations of shape (H, W, C)")

        c, h, w = image_space.shape[2], image_space.shape[0], image_space.shape[1]
        self.observation_space = gym.spaces.Dict({
            "image": gym.spaces.Box(
                low=0,
                high=255,
                shape=(c, h, w),
                dtype=image_space.dtype
            ),
            "state": gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(4,),
                dtype=np.float32
            )
        })

    def _compute_state(self):
        base_env = self.env.unwrapped
        ang_vel = 0.0
        dist_norm = 0.0
        beta10 = 0.0
        beta20 = 0.0

        if hasattr(base_env, "cars") and base_env.cars:
            car = base_env.cars[0]
            ang_vel = float(car.hull.angularVelocity)
            if hasattr(base_env, "track") and base_env.track:
                car_pos = np.array(car.hull.position).reshape((1, 2))
                track_xy = np.array(base_env.track)[:, 2:]
                distances = np.linalg.norm(car_pos - track_xy, ord=2, axis=1)
                track_index = int(np.argmin(distances))
                lane_half_width = float(mcr.TRACK_WIDTH) / 2.0
                if lane_half_width > 0.0:
                    dist_norm = float(distances[track_index]) / lane_half_width

                offset_10 = int(round(10.0 / float(mcr.TRACK_DETAIL_STEP)))
                offset_20 = int(round(20.0 / float(mcr.TRACK_DETAIL_STEP)))
                idx_10 = (track_index + offset_10) % len(base_env.track)
                idx_20 = (track_index + offset_20) % len(base_env.track)
                beta10 = float(base_env.track[idx_10][1])
                beta20 = float(base_env.track[idx_20][1])

        return np.array([ang_vel, dist_norm, beta10, beta20], dtype=np.float32)

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        if not self.enabled:
            return obs
        image = np.transpose(obs, (2, 0, 1))
        state = self._compute_state()
        return {"image": image, "state": state}

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if not self.enabled:
            return obs, reward, done, info
        image = np.transpose(obs, (2, 0, 1))
        state = self._compute_state()
        return {"image": image, "state": state}, reward, done, info


def create_env(config, rank=0, seed=0):
    """Create and wrap the multi_car_racing environment."""
    env_config = config['environment']

    env_id = env_config.get('env_id', 'MultiCarRacing-v0')
    env = gym.make(
        env_id,
        num_agents=env_config.get('num_agents', 1),
        direction=env_config.get('direction', 'CCW'),
        use_random_direction=env_config.get('use_random_direction', True),
        backwards_flag=env_config.get('backwards_flag', True),
        h_ratio=env_config.get('h_ratio', 0.25),
        use_ego_color=env_config.get('use_ego_color', False)
    )

    if env_config.get('num_agents', 1) == 1:
        env = SingleAgentWrapper(env)

    # Safety governor (optional)
    governor_config = config.get('safety_governor', {})
    if governor_config.get('enabled', False):
        env = SafetyGovernorWrapper(env, governor_config)

    # Reward shaping wrapper (optional)
    reward_config = config.get('reward_shaping', {})
    if reward_config.get('enabled', False):
        env = RewardShapingWrapper(env, reward_config)

    # Observation augmentation (optional)
    obs_config = config.get('observation', {})
    if obs_config.get('enabled', False):
        env = ObservationAugmentWrapper(env, obs_config)

    # Wrap with Monitor for logging
    log_dir = config['paths']['log_dir']
    os.makedirs(log_dir, exist_ok=True)
    env = Monitor(env, filename=os.path.join(log_dir, f'monitor_{rank}'))
    
    # Set seed — gym 0.17.3 does not accept seed= kwarg in reset(); use env.seed() instead.
    try:
        env.reset(seed=seed)
    except TypeError:
        if hasattr(env, "seed"):
            env.seed(seed)
        env.reset()

    # #region agent log — create_env success (debug session 31f00d)
    import time as _t; import os as _o
    _o.makedirs("/Users/epablo/Documents/STAT4830/Mar20_Work/Racing_Gym_RL/.cursor", exist_ok=True)
    open("/Users/epablo/Documents/STAT4830/Mar20_Work/Racing_Gym_RL/.cursor/debug-31f00d.log", "a").write(
        '{"sessionId":"31f00d","runId":"run7","hypothesisId":"cocoa_main_thread","location":"train.py:create_env","message":"env_reset_ok","data":{"rank":%d},"timestamp":%d}\n' % (rank, int(_t.time()*1000))
    )
    # #endregion agent log

    return env


def make_env(config, rank, seed):
    def _init():
        env = create_env(config, rank=rank, seed=seed + rank)
        return env
    return _init


def get_device(config):
    """Determine the device to use for training.

    Priority order for 'auto': CUDA > MPS (Apple Silicon) > CPU.
    Requesting 'cuda' on an M-series Mac gracefully falls back to MPS.
    """
    device_config = config.get('device', 'auto')

    if device_config == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    elif device_config in ('cuda', 'gpu'):
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            print("WARNING: CUDA requested but not available. Falling back to MPS (Apple Silicon GPU).")
            device = 'mps'
        else:
            print("WARNING: No GPU available. Falling back to CPU.")
            device = 'cpu'
    elif device_config == 'mps':
        if torch.backends.mps.is_available():
            device = 'mps'
        else:
            print("WARNING: MPS requested but not available. Falling back to CPU.")
            device = 'cpu'
    else:
        device = device_config

    # Print device info
    if device == 'cuda':
        print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    elif device == 'mps':
        print("Using Apple Silicon GPU (MPS) for training")
    else:
        print("Using CPU for training")

    return device


class ProgressCallback(BaseCallback):
    """Custom callback to display training progress with percentage and ETA."""
    
    def __init__(self, total_timesteps, eval_freq, verbose=1):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.eval_freq = eval_freq
        self.start_time = None
        self.last_log_time = None
        self.last_log_timestep = 0
        
    def _on_training_start(self) -> None:
        """Called when training starts."""
        self.start_time = time.time()
        self.last_log_time = time.time()
        print("\n" + "="*70)
        print("TRAINING STARTED")
        print("="*70)
        print(f"Total timesteps: {self.total_timesteps:,}")
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70 + "\n")
        
    def _on_step(self) -> bool:
        """Called at each step."""
        # Log progress periodically
        if self.num_timesteps % self.eval_freq == 0:
            self._log_progress()
        return True
    
    def _log_progress(self):
        """Log training progress with percentage and ETA."""
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        elapsed_since_last = current_time - self.last_log_time
        
        # Calculate progress
        progress_pct = (self.num_timesteps / self.total_timesteps) * 100
        timesteps_since_last = self.num_timesteps - self.last_log_timestep
        
        # Calculate speed
        if elapsed_since_last > 0:
            steps_per_sec = timesteps_since_last / elapsed_since_last
        else:
            steps_per_sec = 0
        
        # Calculate ETA
        remaining_timesteps = self.total_timesteps - self.num_timesteps
        if steps_per_sec > 0:
            eta_seconds = remaining_timesteps / steps_per_sec
            eta = timedelta(seconds=int(eta_seconds))
        else:
            eta = timedelta(seconds=0)
        
        # Format elapsed time
        elapsed = timedelta(seconds=int(elapsed_time))
        
        print("\n" + "-"*70)
        print(f"Progress: {progress_pct:.2f}% ({self.num_timesteps:,} / {self.total_timesteps:,} timesteps)")
        print(f"Elapsed: {str(elapsed)} | ETA: {str(eta)}")
        print(f"Speed: {steps_per_sec:.1f} steps/sec")
        print("-"*70)
        
        self.last_log_time = current_time
        self.last_log_timestep = self.num_timesteps
    
    def _on_training_end(self) -> None:
        """Called when training ends."""
        total_time = time.time() - self.start_time
        total_elapsed = timedelta(seconds=int(total_time))
        print("\n" + "="*70)
        print("TRAINING COMPLETED")
        print("="*70)
        print(f"Total time: {str(total_elapsed)}")
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70 + "\n")


class TelemetryCallback(BaseCallback):
    """Log per-step telemetry/reward components from env info to TensorBoard."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.metric_keys = [
            "telemetry/speed",
            "telemetry/is_corner",
            "telemetry/corner_angle",
            "telemetry/steer",
            "telemetry/throttle",
            "telemetry/brake",
            "telemetry/curriculum_stage",
            "events/offtrack",
            "events/stuck",
            "rewards/forward_progress",
            "rewards/straight_speed",
            "rewards/corner_overspeed",
            "rewards/apex_decel",
            "rewards/steer_smoothness",
            "rewards/time",
            "rewards/idle",
            "rewards/total",
            "lap_count",
        ]

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        if not infos:
            return True

        for key in self.metric_keys:
            values = []
            for info in infos:
                if not isinstance(info, dict):
                    continue
                value = info.get(key)
                if isinstance(value, (int, float, np.integer, np.floating)):
                    values.append(float(value))
            if values:
                self.logger.record(key, float(np.mean(values)))

        return True


def create_model(config, env, device):
    """Create PPO model for image-based observations."""
    ppo_config = config['ppo']
    policy_config = config['policy']

    policy_type = policy_config.get('policy_type', 'CnnPolicy')
    policy_kwargs = None
    if policy_type != 'CnnPolicy':
        activation_fn_map = {
            'tanh': torch.nn.Tanh,
            'relu': torch.nn.ReLU,
            'elu': torch.nn.ELU
        }
        activation_fn = activation_fn_map.get(
            policy_config.get('activation_fn', 'tanh'),
            torch.nn.Tanh
        )
        policy_kwargs = dict(
            net_arch=policy_config.get('net_arch', [256, 256]),
            activation_fn=activation_fn
        )

    model = PPO(
        policy=policy_type,
        env=env,
        learning_rate=ppo_config['learning_rate'],
        n_steps=ppo_config['n_steps'],
        batch_size=ppo_config['batch_size'],
        n_epochs=ppo_config['n_epochs'],
        gamma=ppo_config['gamma'],
        gae_lambda=ppo_config['gae_lambda'],
        clip_range=ppo_config['clip_range'],
        ent_coef=ppo_config['ent_coef'],
        vf_coef=ppo_config['vf_coef'],
        max_grad_norm=ppo_config['max_grad_norm'],
        use_sde=ppo_config.get('use_sde', False),
        sde_sample_freq=ppo_config.get('sde_sample_freq', -1),
        policy_kwargs=policy_kwargs,
        device=device,
        verbose=1,
        tensorboard_log=config['paths']['log_dir']
    )
    
    return model


def main():
    parser = argparse.ArgumentParser(description='Train PPO agent on Multi-Car Racing')
    parser.add_argument(
        '--config',
        type=str,
        default='config/multi_car_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume training from'
    )
    parser.add_argument(
        '--trainer_backend',
        type=str,
        default=None,
        choices=['sb3', 'torch'],
        help='Training backend override (sb3 or torch). Defaults to config.training.trainer_backend or sb3'
    )
    parser.add_argument(
        '--timesteps_add',
        type=int,
        default=None,
        help='Relative timesteps to add on top of the loaded checkpoint step (torch backend).'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    training_config = config.get('training', {})
    trainer_backend = (
        args.trainer_backend
        if args.trainer_backend is not None
        else training_config.get('trainer_backend', 'sb3')
    )
    trainer_backend = str(trainer_backend).strip().lower()
    if trainer_backend not in {'sb3', 'torch'}:
        raise ValueError(f"Unknown trainer backend: {trainer_backend}")
    
    # Set random seeds
    set_random_seed(args.seed)
    
    # Create directories (resolve model_dir to absolute so save/resume use same path)
    model_dir = Path(config['paths']['model_dir']).resolve()
    log_dir = Path(config['paths']['log_dir'])
    results_dir = Path(config['paths']['results_dir'])
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Get device
    device = get_device(config)
    print("\n" + "="*70)
    print("TRAINING CONFIGURATION")
    print("="*70)
    print(f"Config file: {args.config}")
    print(f"Device: {device}")
    print(f"Seed: {args.seed}")
    print(f"Trainer backend: {trainer_backend}")
    if args.resume:
        print(f"Resuming from: {args.resume}")
    if args.timesteps_add is not None:
        print(f"Timesteps add mode: +{args.timesteps_add:,}")
    print("="*70 + "\n")

    # GPU performance knobs
    if device == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
    
    # Create environment
    print("Creating environment...")
    env_id = config['environment'].get('env_id', 'MultiCarRacing-v0')
    print(f"Environment ID: {env_id}")

    # Vectorized environments for higher throughput
    num_envs = int(config.get('training', {}).get('num_envs', 1))
    if num_envs < 1:
        num_envs = 1
    obs_config = config.get('observation', {})
    if num_envs > 1:
        env = SubprocVecEnv([make_env(config, rank=i, seed=args.seed) for i in range(num_envs)])
        print(f"Using SubprocVecEnv with {num_envs} parallel environments")
    else:
        env = DummyVecEnv([lambda: create_env(config, rank=0, seed=args.seed)])
        print("Using DummyVecEnv with 1 environment")
    if not obs_config.get('enabled', False):
        env = VecTransposeImage(env)
    
    # Create evaluation environment
    eval_env = DummyVecEnv([lambda: create_env(config, rank=1, seed=args.seed + 1000)])
    if not obs_config.get('enabled', False):
        eval_env = VecTransposeImage(eval_env)
    print("Environment created successfully!\n")

    if trainer_backend == 'torch':
        if obs_config.get('enabled', False):
            raise ValueError(
                "Torch backend currently supports Box image observations only; set observation.enabled=false."
            )

        ppo_config = config['ppo']
        print("\n" + "-"*70)
        print("PPO HYPERPARAMETERS")
        print("-"*70)
        print(f"Learning rate: {ppo_config['learning_rate']}")
        print(f"Steps per update: {ppo_config['n_steps']}")
        print(f"Batch size: {ppo_config['batch_size']}")
        print(f"Epochs per update: {ppo_config['n_epochs']}")
        print(f"Gamma (discount): {ppo_config['gamma']}")
        print(f"GAE lambda: {ppo_config['gae_lambda']}")
        print(f"Clip range: {ppo_config['clip_range']}")
        print(f"Entropy coefficient: {ppo_config['ent_coef']}")
        print(f"Value function coefficient: {ppo_config['vf_coef']}")
        print(f"Max gradient norm: {ppo_config['max_grad_norm']}")
        print("-"*70 + "\n")

        # If resuming, resolve path (try cwd then model_dir) and check file exists
        resume_path = None
        if args.resume:
            raw = args.resume.strip()
            # Shorthand: "best" -> best model in model_dir
            if raw.lower() == "best":
                resume_path = model_dir / "best_model_torch.pt"
            else:
                candidate = Path(raw).resolve()
                if candidate.is_file():
                    resume_path = candidate
                else:
                    # Fallback: look in model_dir (same place we save)
                    fallback = model_dir / Path(raw).name
                    if fallback.is_file():
                        resume_path = fallback
                    else:
                        resume_path = candidate  # use for error message
            if resume_path is not None and not resume_path.is_file():
                step_glob = list(model_dir.glob("torch_ppo_step_*.pt"))
                msg = (
                    f"Resume path not found: {resume_path}\n"
                    f"  Best model path: {(model_dir / 'best_model_torch.pt').resolve()}\n"
                    f"  (Created on first eval, then updated when an evaluation beats the previous best.)\n"
                )
                if step_glob:
                    latest = max(step_glob, key=lambda p: int(p.stem.rsplit("_", 1)[-1]))
                    msg += f"  You can resume from a step checkpoint, e.g.: --resume {latest}"
                else:
                    msg += "  No step checkpoints found. Run without --resume to start from scratch."
                raise FileNotFoundError(msg)

        print("Creating PPO model (local torch backend)...")
        trainer = TorchPPOTrainer(
            env=env,
            eval_env=eval_env,
            config=config,
            device=device,
            model_dir=model_dir,
            log_dir=log_dir,
        )
        if args.resume:
            print(f"Loading torch checkpoint from {resume_path}")
            trainer.load(resume_path)
        if args.timesteps_add is not None and args.timesteps_add <= 0:
            raise ValueError("--timesteps_add must be a positive integer.")

        configured_total = int(training_config['total_timesteps'])
        if args.timesteps_add is not None:
            target_total_timesteps = int(trainer.num_timesteps + args.timesteps_add)
        else:
            target_total_timesteps = configured_total

        print("\n" + "-"*70)
        print("MODEL INFORMATION")
        print("-"*70)
        print(f"Observation space: {env.observation_space}")
        print(f"Action space: {env.action_space}")
        print(f"Policy: {trainer.policy}")

        print("-"*70)
        print("TRAINING SETTINGS")
        print("-"*70)
        print(f"Configured total timesteps: {configured_total:,}")
        if args.timesteps_add is not None:
            print(f"Resume +N target timesteps: {target_total_timesteps:,}")
        else:
            print(f"Run target timesteps: {target_total_timesteps:,}")
        print(f"Evaluation frequency: {training_config['eval_freq']:,} steps")
        print(f"Evaluation episodes: {training_config['n_eval_episodes']}")
        print(f"Checkpoint frequency: {training_config['save_freq']:,} steps")
        visual_cfg = training_config.get('visual_eval', {})
        print(
            f"Visual eval: enabled={bool(visual_cfg.get('enabled', True))}, "
            f"freq={int(visual_cfg.get('freq', 50000)):,}, "
            f"episodes={int(visual_cfg.get('n_episodes', 1))}"
        )
        reward_cfg = config.get("reward_shaping", {}) or {}
        curr_cfg = reward_cfg.get("curriculum", {}) or {}
        print(
            f"PPO-v2 action mapping: steer=tanh, throttle=sigmoid, brake=sigmoid | "
            f"log_std=[{ppo_config.get('min_log_std', -1.5)}, {ppo_config.get('max_log_std', 1.0)}]"
        )
        print(
            f"Curriculum: enabled={bool(curr_cfg.get('enabled', False))}, "
            f"stage1->2 gate: progress>={float(curr_cfg.get('promote_progress_threshold', 0.35)):.2f}, "
            f"speed>={float(curr_cfg.get('promote_speed_threshold', 8.0)):.2f}"
        )
        print(f"Model directory: {model_dir.resolve()}")
        print(f"  Best model (updated every eval that beats previous): {model_dir.resolve() / 'best_model_torch.pt'}")
        print(f"  Best copies (per step): {model_dir.resolve() / 'best_model_torch_step_<step>.pt'}")
        print(f"  Checkpoints: {model_dir.resolve() / 'torch_ppo_step_<step>.pt'}")
        print(f"Log directory: {log_dir}")
        print(f"Eval history log: {log_dir / 'torch_eval_history.jsonl'}")
        print("-"*70 + "\n")

        trainer.learn(
            total_timesteps=target_total_timesteps,
            eval_freq=training_config['eval_freq'],
            n_eval_episodes=training_config['n_eval_episodes'],
            save_freq=training_config['save_freq'],
            log_interval=training_config.get('log_interval', 10),
            success_gate=training_config.get('success_gate', {}),
            visual_eval_cfg=training_config.get('visual_eval', {}),
        )
        final_model_path = model_dir / 'final_model_torch.pt'
        trainer.save(final_model_path)

        print("\n" + "="*70)
        print("TRAINING SUMMARY")
        print("="*70)
        print(f"Final model saved to: {final_model_path}")
        print(f"Best model saved to: {model_dir / 'best_model_torch.pt'}")
        print(f"Checkpoints saved to: {model_dir}")
        print(f"TensorBoard logs: {log_dir}")
        print("="*70 + "\n")

        env.close()
        eval_env.close()
        print("Environments closed. Training complete!")
        return
    
    # Create model
    print("Creating PPO model...")
    if args.resume:
        print(f"Loading model from {args.resume}")
        model = PPO.load(args.resume, env=env, device=device)
    else:
        model = create_model(config, env, device)
    
    # Print model info
    print("\n" + "-"*70)
    print("MODEL INFORMATION")
    print("-"*70)
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Policy: {model.policy}")
    
    # Print hyperparameters
    ppo_config = config['ppo']
    print("\n" + "-"*70)
    print("PPO HYPERPARAMETERS")
    print("-"*70)
    print(f"Learning rate: {ppo_config['learning_rate']}")
    print(f"Steps per update: {ppo_config['n_steps']}")
    print(f"Batch size: {ppo_config['batch_size']}")
    print(f"Epochs per update: {ppo_config['n_epochs']}")
    print(f"Gamma (discount): {ppo_config['gamma']}")
    print(f"GAE lambda: {ppo_config['gae_lambda']}")
    print(f"Clip range: {ppo_config['clip_range']}")
    print(f"Entropy coefficient: {ppo_config['ent_coef']}")
    print(f"Value function coefficient: {ppo_config['vf_coef']}")
    print(f"Max gradient norm: {ppo_config['max_grad_norm']}")
    print("-"*70 + "\n")
    
    # Setup callbacks
    training_config = config['training']
    
    # Progress callback for percentage and ETA
    progress_callback = ProgressCallback(
        total_timesteps=training_config['total_timesteps'],
        eval_freq=training_config['eval_freq']
    )
    telemetry_callback = TelemetryCallback()
    
    # Evaluation callback with custom logging
    class LoggingEvalCallback(EvalCallback):
        """Extended EvalCallback with better logging."""
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.eval_count = 0
            
        def _on_step(self) -> bool:
            if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
                self.eval_count += 1
                print(f"\n{'='*70}")
                print(f"EVALUATION #{self.eval_count} (Step {self.num_timesteps:,})")
                print(f"{'='*70}")
            return super()._on_step()
        
        def _on_evaluation_end(self, locals_, globals_):
            """Log evaluation results."""
            if 'mean_reward' in locals_:
                mean_rew = locals_['mean_reward']
                std_rew = locals_.get('std_reward', 0)
                print(f"Mean reward: {mean_rew:.2f} ± {std_rew:.2f}")
                print(f"{'='*70}\n")
            return super()._on_evaluation_end(locals_, globals_)
    
    eval_callback = LoggingEvalCallback(
        eval_env,
        best_model_save_path=str(model_dir / 'best_model'),
        log_path=str(log_dir),
        eval_freq=training_config['eval_freq'],
        n_eval_episodes=training_config['n_eval_episodes'],
        deterministic=True,
        render=False
    )
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=training_config['save_freq'],
        save_path=str(model_dir),
        name_prefix='ppo_racecar'
    )
    
    # Combine callbacks
    callbacks = CallbackList([progress_callback, telemetry_callback, eval_callback, checkpoint_callback])
    
    # Print training settings
    print("-"*70)
    print("TRAINING SETTINGS")
    print("-"*70)
    print(f"Total timesteps: {training_config['total_timesteps']:,}")
    print(f"Evaluation frequency: {training_config['eval_freq']:,} steps")
    print(f"Evaluation episodes: {training_config['n_eval_episodes']}")
    print(f"Checkpoint frequency: {training_config['save_freq']:,} steps")
    print(f"Model directory: {model_dir}")
    print(f"Log directory: {log_dir}")
    print(f"TensorBoard logs: tensorboard --logdir {log_dir}")
    print("-"*70 + "\n")
    
    # Train model
    model.learn(
        total_timesteps=training_config['total_timesteps'],
        callback=callbacks,
        log_interval=training_config.get('log_interval', 10),
        progress_bar=True
    )
    
    # Save final model
    final_model_path = model_dir / 'final_model'
    model.save(str(final_model_path))
    
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    print(f"Final model saved to: {final_model_path}")
    print(f"Best model saved to: {model_dir / 'best_model' / 'best_model.zip'}")
    print(f"Checkpoints saved to: {model_dir}")
    print(f"TensorBoard logs: {log_dir}")
    print("="*70 + "\n")
    
    # Close environments
    env.close()
    eval_env.close()
    print("Environments closed. Training complete!")


if __name__ == '__main__':
    main()
