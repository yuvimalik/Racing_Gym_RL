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
    "min_log_std": -1.0, # General min log_std for throttle/brake raw actions
    "max_log_std": 0.5,  # General max log_std for throttle/brake raw actions
    "steer_min_log_std": -0.5, # Specific min log_std for steer raw action
    "steer_max_log_std": 0.0, # Specific max log_std for steer raw action
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
            log_std_init[0] = 0.0 # Initial std for steer
        if action_dim >= 3:
            log_std_init[1] = -0.5 # Initial std for throttle
            log_std_init[2] = -1.0 # Initial std for brake
        self.log_std = nn.Parameter(log_std_init)

        self.min_log_std = float(hp.get("min_log_std", -1.0))
        self.max_log_std = float(hp.get("max_log_std", 0.5))
        self.steer_min_log_std = float(hp.get("steer_min_log_std", self.min_log_std))
        self.steer_max_log_std = float(hp.get("steer_max_log_std", self.max_log_std))

        # --- START OF CHANGE ---
        # Removing strong initial biases for throttle (positive) and brake (negative).
        # By default, biases are initialized to zero, leading to a neutral starting point
        # (tanh(0)=0, which maps to 0.5 for throttle/brake). This encourages the agent
        # to learn initial behaviors rather than being forced into aggressive acceleration.
        # if action_dim >= 3:
        #     nn.init.constant_(self.policy_mean.bias[1], 2.0)
        #     nn.init.constant_(self.policy_mean.bias[2], -3.0)
        # --- END OF CHANGE ---

    def _features(self, obs: torch.Tensor) -> torch.Tensor:
        return self.features(obs)

    def get_raw_dist_and_value(self, obs: torch.Tensor):
        """Returns the untransformed (Normal) distribution and value."""
        shared = self._features(obs)
        policy_latent = self.policy_mlp(shared)
        mean = self.policy_mean(policy_latent)

        # Apply specific clamping for steer log_std if action_dim >= 1
        steer_ls = torch.clamp(self.log_std[0:1], self.steer_min_log_std, self.steer_max_log_std)
        
        # Apply general clamping for other log_std values
        if self.log_std.shape[0] > 1:
            other_ls = torch.clamp(self.log_std[1:], self.min_log_std, self.max_log_std)
            log_std = torch.cat([steer_ls, other_ls])
        else: # only steer action
            log_std = steer_ls
        
        std = log_std.exp()
        dist = Normal(mean, std)
        
        value_latent = self.value_mlp(shared)
        value = self.value_head(value_latent).squeeze(-1)
        return dist, value

    @staticmethod
    def raw_to_env_action(raw_action: torch.Tensor) -> torch.Tensor:
        """
        raw_action is the squashed action (output of tanh), range (-1, 1).
        Maps to environment actions: steer (-1, 1), throttle (0, 1), brake (0, 1).
        """
        out = raw_action.clone()
        # Steer is already in (-1, 1)
        # Throttle (index 1) and Brake (index 2) need to be mapped from (-1, 1) to (0, 1)
        if out.shape[-1] >= 2:
            out[..., 1] = (out[..., 1] + 1.0) / 2.0  # Map (-1, 1) to (0, 1)
        if out.shape[-1] >= 3:
            out[..., 2] = (out[..., 2] + 1.0) / 2.0  # Map (-1, 1) to (0, 1)
        return out

    def act(self, obs: torch.Tensor, deterministic: bool = False):
        """
        Returns squashed action and its corrected log_prob.
        """
        raw_dist, value = self.get_raw_dist_and_value(obs)
        
        if deterministic:
            action_raw = raw_dist.mean
        else:
            action_raw = raw_dist.rsample()
        
        action_squashed = torch.tanh(action_raw)
        
        # Compute log_prob with tanh correction
        log_prob_raw = raw_dist.log_prob(action_raw).sum(dim=-1)
        
        # Correction for tanh squashing: log(1 - tanh(x)^2)
        # Add epsilon for numerical stability
        log_prob_correction = torch.sum(torch.log(1 - action_squashed.pow(2) + 1e-6), dim=-1)
        
        log_prob = log_prob_raw - log_prob_correction
        
        return action_squashed, log_prob, value

    def evaluate_actions(self, obs: torch.Tensor, actions_squashed: torch.Tensor):
        """
        actions_squashed are the actions sampled from the squashed distribution (range -1 to 1).
        """
        raw_dist, value = self.get_raw_dist_and_value(obs)
        
        # To calculate log_prob of actions_squashed, we need to convert them back to raw actions
        # Clamp to prevent NaNs in atanh for values very close to -1 or 1
        actions_raw = torch.atanh(actions_squashed.clamp(-0.999999, 0.999999))
        
        log_prob_raw = raw_dist.log_prob(actions_raw).sum(dim=-1)
        
        # The log_prob correction term uses the squashed actions
        log_prob_correction = torch.sum(torch.log(1 - actions_squashed.pow(2) + 1e-6), dim=-1)
        
        log_prob_squashed = log_prob_raw - log_prob_correction
        
        # For entropy, we typically use the entropy of the un-squashed Normal distribution
        # as a proxy for exploration, or a more complex analytical form.
        # Sticking to the un-squashed entropy for now as is common practice for simplicity.
        entropy = raw_dist.entropy().sum(dim=-1)
        
        return value, log_prob_squashed, entropy


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
        # act_buf stores the SQUASHED actions, which are passed to evaluate_actions later
        act_buf = np.zeros((n_steps, n_envs, self.action_dim), dtype=np.float32) 
        rew_buf = np.zeros((n_steps, n_envs), dtype=np.float32)
        done_buf = np.zeros((n_steps, n_envs), dtype=np.float32)
        val_buf = np.zeros((n_steps, n_envs), dtype=np.float32)
        logp_buf = np.zeros((n_steps, n_envs), dtype=np.float32)

        for step in range(n_steps):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device) / 255.0
            with torch.inference_mode():
                # self.policy.act now returns squashed action and corrected log_prob
                action_squashed, log_prob, value = self.policy.act(obs_t)
            
            # Store the squashed actions for PPO update
            act_np_squashed = action_squashed.cpu().numpy()
            
            # Convert squashed actions to env actions for stepping
            env_np = self._raw_to_env_action_np(act_np_squashed)

            obs_buf[step] = (obs * 255).astype(np.uint8) if obs.max() <= 1.0 else obs
            act_buf[step] = act_np_squashed # Store squashed actions
            val_buf[step] = value.cpu().numpy()
            logp_buf[step] = log_prob.cpu().numpy()

            obs, rewards, dones, infos = env.step(env_np)
            rew_buf[step] = rewards
            done_buf[step] = dones

        # Bootstrap last value
        with torch.inference_mode():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device) / 255.0
            _, _, last_value = self.policy.act(obs_t) # Policy.act needs to be consistent, returns squashed action, corrected log_prob
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
        # act_flat now contains squashed actions
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
                # act_b now contains squashed actions
                act_b = torch.as_tensor(act_flat[batch_idx], dtype=torch.float32, device=self.device) 
                old_logp_b = torch.as_tensor(logp_flat[batch_idx], dtype=torch.float32, device=self.device)
                adv_b = torch.as_tensor(adv_flat[batch_idx], dtype=torch.float32, device=self.device)
                ret_b = torch.as_tensor(ret_flat[batch_idx], dtype=torch.float32, device=self.device)

                # evaluate_actions now expects squashed actions
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