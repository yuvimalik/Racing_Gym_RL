"""
EDITABLE — This is the ONLY file the autoresearch agent modifies.

Contains:
  - CnnActorCritic:  network architecture (can change layers, distribution, etc.)
  - HYPERPARAMS:     all tunable hyperparameters
  - PPOTrainer:      training logic (can change optimizer, schedule, aux losses, etc.)

Contract (enforced by run_experiment.py):
  - PPOTrainer.train(env, eval_env, device, total_timesteps, checkpoint_dir) -> dict
    Must return metrics dict with at least 'mean_reward' key.
  - PPOTrainer.make_policy(checkpoint_path, device) -> policy
    Policy must have .act(obs_tensor, deterministic=True) -> (action, _, _)
    and a static method raw_to_env_action(raw) -> env_action tensor.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from pathlib import Path

# === TUNABLE: Hyperparameters ===
HYPERPARAMS = {
    "learning_rate": 2.5e-4,
    "n_steps": 128,
    "batch_size": 256,
    "n_epochs": 4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.05,
    "vf_coef": 0.25,
    "max_grad_norm": 0.5,
    "min_log_std": -1.0,
    "max_log_std": 0.5,
    "steer_min_log_std": -0.5,
    "steer_max_log_std": 0.0,
}


# === TUNABLE: Network Architecture ===

class CnnActorCritic(nn.Module):
    """Actor-critic for image observations (N, C, H, W)."""

    def __init__(self, obs_shape, action_dim: int, hp: dict = None):
        super().__init__()
        hp = hp or HYPERPARAMS
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

        self.policy_mlp = nn.Sequential(
            nn.Linear(n_flatten, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
        )
        self.policy_mean = nn.Linear(128, action_dim)

        self.value_mlp = nn.Sequential(
            nn.Linear(n_flatten, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
        )
        self.value_head = nn.Linear(128, 1)

        log_std_init = torch.full((action_dim,), -0.5)
        if action_dim >= 1:
            log_std_init[0] = 0.0
        if action_dim >= 3:
            log_std_init[2] = -1.0
        self.log_std = nn.Parameter(log_std_init)

        self.min_log_std = float(hp.get("min_log_std", -1.0))
        self.max_log_std = float(hp.get("max_log_std", 0.5))
        self.steer_min_log_std = float(hp.get("steer_min_log_std", self.min_log_std))
        self.steer_max_log_std = float(hp.get("steer_max_log_std", self.max_log_std))

        if action_dim >= 3:
            nn.init.constant_(self.policy_mean.bias[1], 2.0)
            nn.init.constant_(self.policy_mean.bias[2], -3.0)

    def _features(self, obs: torch.Tensor) -> torch.Tensor:
        return self.features(obs)

    def get_dist_and_value(self, obs: torch.Tensor):
        shared = self._features(obs)
        policy_latent = self.policy_mlp(shared)
        mean = self.policy_mean(policy_latent)
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
        out = raw_action.clone()
        if out.shape[-1] >= 1:
            out[..., 0] = torch.tanh(out[..., 0])
        if out.shape[-1] >= 2:
            out[..., 1] = torch.sigmoid(out[..., 1])
        if out.shape[-1] >= 3:
            out[..., 2] = torch.sigmoid(out[..., 2])
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


# === TUNABLE: Training Logic ===

class PPOTrainer:
    """Synchronous PPO trainer for autoresearch experiments.

    Kept simple and self-contained so the autoresearch agent can freely
    modify the training loop, optimizer, schedule, etc.
    """

    def __init__(self, obs_shape, action_dim: int, device, hp: dict = None):
        self.hp = hp or HYPERPARAMS
        self.device = device
        self.obs_shape = obs_shape
        self.action_dim = action_dim

        self.policy = CnnActorCritic(obs_shape, action_dim, hp=self.hp).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.hp["learning_rate"])

    def _raw_to_env_action_np(self, raw_action_np: np.ndarray) -> np.ndarray:
        raw_t = torch.as_tensor(raw_action_np, dtype=torch.float32)
        env_t = CnnActorCritic.raw_to_env_action(raw_t)
        return env_t.numpy()

    def _collect_rollout(self, env, obs, n_steps: int):
        """Collect n_steps of experience from the vectorized env."""
        n_envs = env.num_envs
        obs_buf = np.zeros((n_steps, n_envs, *self.obs_shape), dtype=np.uint8)
        act_buf = np.zeros((n_steps, n_envs, self.action_dim), dtype=np.float32)
        rew_buf = np.zeros((n_steps, n_envs), dtype=np.float32)
        done_buf = np.zeros((n_steps, n_envs), dtype=np.float32)
        val_buf = np.zeros((n_steps, n_envs), dtype=np.float32)
        logp_buf = np.zeros((n_steps, n_envs), dtype=np.float32)

        for step in range(n_steps):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device) / 255.0
            with torch.inference_mode():
                action, log_prob, value = self.policy.act(obs_t)
            raw_np = action.cpu().numpy()
            env_np = self._raw_to_env_action_np(raw_np)

            obs_buf[step] = (obs * 255).astype(np.uint8) if obs.max() <= 1.0 else obs
            act_buf[step] = raw_np
            val_buf[step] = value.cpu().numpy()
            logp_buf[step] = log_prob.cpu().numpy()

            obs, rewards, dones, infos = env.step(env_np)
            rew_buf[step] = rewards
            done_buf[step] = dones

        # Bootstrap last value
        with torch.inference_mode():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device) / 255.0
            _, _, last_value = self.policy.act(obs_t)
        last_values = last_value.cpu().numpy()

        return obs, {
            "obs": obs_buf, "actions": act_buf, "rewards": rew_buf,
            "dones": done_buf, "values": val_buf, "log_probs": logp_buf,
            "last_values": last_values, "last_dones": dones,
        }

    def _compute_gae(self, buf: dict) -> dict:
        """Compute GAE advantages and returns."""
        gamma = self.hp["gamma"]
        lam = self.hp["gae_lambda"]
        n_steps = buf["rewards"].shape[0]
        n_envs = buf["rewards"].shape[1]
        advantages = np.zeros((n_steps, n_envs), dtype=np.float32)
        last_gae = np.zeros(n_envs, dtype=np.float32)

        for step in reversed(range(n_steps)):
            if step == n_steps - 1:
                next_non_terminal = 1.0 - buf["last_dones"].astype(np.float32)
                next_values = buf["last_values"]
            else:
                next_non_terminal = 1.0 - buf["dones"][step]
                next_values = buf["values"][step + 1]
            delta = buf["rewards"][step] + gamma * next_values * next_non_terminal - buf["values"][step]
            last_gae = delta + gamma * lam * next_non_terminal * last_gae
            advantages[step] = last_gae

        buf["advantages"] = advantages
        buf["returns"] = advantages + buf["values"]
        return buf

    def _ppo_update(self, buf: dict) -> dict:
        """Run PPO update epochs on the rollout buffer."""
        n_steps, n_envs = buf["rewards"].shape
        n_samples = n_steps * n_envs
        batch_size = self.hp["batch_size"]

        obs_flat = buf["obs"].reshape(n_samples, *self.obs_shape)
        act_flat = buf["actions"].reshape(n_samples, self.action_dim)
        logp_flat = buf["log_probs"].reshape(n_samples)
        adv_flat = buf["advantages"].reshape(n_samples)
        ret_flat = buf["returns"].reshape(n_samples)

        adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)

        total_pg_loss = 0.0
        total_vf_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for epoch in range(self.hp["n_epochs"]):
            indices = np.random.permutation(n_samples)
            for start in range(0, n_samples, batch_size):
                batch_idx = indices[start:start + batch_size]

                obs_b = torch.as_tensor(obs_flat[batch_idx], dtype=torch.float32, device=self.device) / 255.0
                act_b = torch.as_tensor(act_flat[batch_idx], dtype=torch.float32, device=self.device)
                old_logp_b = torch.as_tensor(logp_flat[batch_idx], dtype=torch.float32, device=self.device)
                adv_b = torch.as_tensor(adv_flat[batch_idx], dtype=torch.float32, device=self.device)
                ret_b = torch.as_tensor(ret_flat[batch_idx], dtype=torch.float32, device=self.device)

                values, new_logp, entropy = self.policy.evaluate_actions(obs_b, act_b)
                ratio = torch.exp(new_logp - old_logp_b)

                surr1 = ratio * adv_b
                surr2 = torch.clamp(ratio, 1 - self.hp["clip_range"], 1 + self.hp["clip_range"]) * adv_b
                pg_loss = -torch.min(surr1, surr2).mean()
                vf_loss = F.mse_loss(values, ret_b)
                entropy_loss = -entropy.mean()

                loss = pg_loss + self.hp["vf_coef"] * vf_loss + self.hp["ent_coef"] * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.hp["max_grad_norm"])
                self.optimizer.step()

                total_pg_loss += pg_loss.item()
                total_vf_loss += vf_loss.item()
                total_entropy += entropy.mean().item()
                n_updates += 1

        return {
            "pg_loss": total_pg_loss / max(n_updates, 1),
            "vf_loss": total_vf_loss / max(n_updates, 1),
            "entropy": total_entropy / max(n_updates, 1),
        }

    def train(self, env, eval_env, device, total_timesteps: int, checkpoint_dir: Path) -> dict:
        """Main training loop. Returns metrics dict with 'mean_reward' key."""
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        n_steps = self.hp["n_steps"]
        n_envs = env.num_envs
        steps_per_rollout = n_steps * n_envs
        n_iterations = total_timesteps // steps_per_rollout

        obs = env.reset()
        best_reward = -float("inf")
        num_timesteps = 0

        for iteration in range(1, n_iterations + 1):
            obs, buf = self._collect_rollout(env, obs, n_steps)
            buf = self._compute_gae(buf)
            metrics = self._ppo_update(buf)
            num_timesteps += steps_per_rollout

            if iteration % 10 == 0:
                print(f"  [{num_timesteps:>8,}/{total_timesteps:,}] "
                      f"pg={metrics['pg_loss']:.4f} vf={metrics['vf_loss']:.4f} "
                      f"ent={metrics['entropy']:.4f}")

        # Final eval
        from autoresearch.prepare import evaluate
        eval_metrics = evaluate(self.policy, device, None, n_episodes=10)

        # Save final checkpoint
        ckpt_path = checkpoint_dir / "final.pt"
        torch.save({
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "hyperparams": self.hp,
        }, ckpt_path)

        return eval_metrics

    def make_policy(self, checkpoint_path, device):
        """Load and return policy for evaluation."""
        ckpt = torch.load(checkpoint_path, map_location=device)
        self.policy.load_state_dict(ckpt["policy_state_dict"])
        self.policy.eval()
        return self.policy
