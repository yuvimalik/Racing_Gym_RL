import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Beta # Import Beta distribution
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
    # The min/max_log_std are now primarily for the steer action
    "min_log_std": -1.0, # (General, now effectively unused for Beta actions)
    "max_log_std": 0.5,  # (General, now effectively unused for Beta actions)
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
        
        # Policy output for steer (1 mean) and throttle/brake (2 alphas, 2 betas)
        # Total output dimensions: 1 (steer_mean) + 2 (throttle_alpha/beta) + 2 (brake_alpha/beta) = 5
        # This assumes action_dim=3. If action_dim=1, it would be just 1 output.
        # We explicitly support action_dim=3 for MultiCarRacing, with steer + throttle + brake.
        assert action_dim == 3, "This Beta distribution setup is designed for action_dim=3 (steer, throttle, brake)."
        self.policy_mean = nn.Linear(128, 5) # 1 for steer_mean, 2 for throttle_alpha/beta, 2 for brake_alpha/beta

        self.value_mlp = nn.Sequential(
            nn.Linear(n_flatten, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
        )
        self.value_head = nn.Linear(128, 1)

        # log_std is now only for the steer action (action_dim=1)
        # The other actions (throttle, brake) will use Beta distribution parameters (alpha, beta)
        self.log_std = nn.Parameter(torch.full((1,), -0.5)) # Initial log_std for steer

        self.steer_min_log_std = float(hp.get("steer_min_log_std", -0.5))
        self.steer_max_log_std = float(hp.get("steer_max_log_std", 0.0))

        # Initial biases for policy_mean:
        # steer_mean (index 0)
        # throttle_alpha_raw (index 1), throttle_beta_raw (index 2)
        # brake_alpha_raw (index 3), brake_beta_raw (index 4)
        nn.init.constant_(self.policy_mean.bias[1], 2.0)  # Throttle alpha bias (encourage higher throttle)
        nn.init.constant_(self.policy_mean.bias[2], -1.0) # Throttle beta bias (encourage higher throttle)
        nn.init.constant_(self.policy_mean.bias[3], -1.0) # Brake alpha bias (encourage higher brake for stop)
        nn.init.constant_(self.policy_mean.bias[4], 2.0)   # Brake beta bias (encourage higher brake for stop)

    def _features(self, obs: torch.Tensor) -> torch.Tensor:
        return self.features(obs)

    def get_raw_dist_and_value(self, obs: torch.Tensor):
        """Returns the untransformed (Normal/Beta) distributions and value."""
        shared = self._features(obs)
        policy_latent = self.policy_mlp(shared)
        mean_out = self.policy_mean(policy_latent) # Output (batch_size, 5)

        # Steer distribution (Normal, then tanh squashed)
        steer_mean = mean_out[..., 0:1] # (batch_size, 1)
        steer_log_std = torch.clamp(self.log_std, self.steer_min_log_std, self.steer_max_log_std)
        steer_dist = Normal(steer_mean, steer_log_std.exp())

        # Throttle and Brake distributions (Beta)
        throttle_alpha_raw = mean_out[..., 1:2]
        throttle_beta_raw = mean_out[..., 2:3]
        brake_alpha_raw = mean_out[..., 3:4]
        brake_beta_raw = mean_out[..., 4:5]

        # Apply softplus to ensure positive alpha/beta values for Beta distribution
        # Add epsilon for numerical stability and to prevent alpha/beta from being exactly 0
        throttle_alpha = F.softplus(throttle_alpha_raw) + 1e-6
        throttle_beta = F.softplus(throttle_beta_raw) + 1e-6
        throttle_dist = Beta(throttle_alpha, throttle_beta)

        brake_alpha = F.softplus(brake_alpha_raw) + 1e-6
        brake_beta = F.softplus(brake_beta_raw) + 1e-6
        brake_dist = Beta(brake_alpha, brake_beta)
        
        value_latent = self.value_mlp(shared)
        value = self.value_head(value_latent).squeeze(-1)
        
        return (steer_dist, throttle_dist, brake_dist), value

    @staticmethod
    def raw_to_env_action(raw_action: torch.Tensor) -> torch.Tensor:
        """
        This method is kept for compatibility with the evaluation interface,
        but act() now directly returns environment actions. It simply returns
        its input as it is already in the environment action space.
        """
        return raw_action

    def act(self, obs: torch.Tensor, deterministic: bool = False):
        """
        Returns environment actions (steer in [-1,1], throttle in [0,1], brake in [0,1])
        and their combined log_prob.
        """
        (steer_dist, throttle_dist, brake_dist), value = self.get_raw_dist_and_value(obs)
        
        if deterministic:
            steer_raw = steer_dist.mean
            throttle_action = throttle_dist.mean
            brake_action = brake_dist.mean
        else:
            steer_raw = steer_dist.rsample() # Sample from Normal for steer
            throttle_action = throttle_dist.sample() # Sample from Beta for throttle
            brake_action = brake_dist.sample() # Sample from Beta for brake
        
        # Steer action is tanh-squashed to be in [-1, 1]
        steer_action = torch.tanh(steer_raw)
        
        # Combine all environment actions
        action_env = torch.cat([steer_action, throttle_action, brake_action], dim=-1) # (batch_size, 3)
        
        # Compute log_prob for steer (Normal with tanh correction)
        log_prob_steer_raw = steer_dist.log_prob(steer_raw).sum(dim=-1) # (batch_size,)
        log_prob_steer_correction = torch.sum(torch.log(1 - steer_action.pow(2) + 1e-6), dim=-1)
        log_prob_steer = log_prob_steer_raw - log_prob_steer_correction
        
        # Compute log_prob for throttle and brake (Beta distribution directly)
        log_prob_throttle = throttle_dist.log_prob(throttle_action).sum(dim=-1) # (batch_size,)
        log_prob_brake = brake_dist.log_prob(brake_action).sum(dim=-1) # (batch_size,)
        
        # Total log_prob is the sum of independent action log_probs
        total_log_prob = log_prob_steer + log_prob_throttle + log_prob_brake
        
        return action_env, total_log_prob, value

    def evaluate_actions(self, obs: torch.Tensor, actions_env: torch.Tensor):
        """
        actions_env are the actions taken in the environment (steer in [-1,1], throttle in [0,1], brake in [0,1]).
        """
        (steer_dist, throttle_dist, brake_dist), value = self.get_raw_dist_and_value(obs)
        
        # Split environment actions
        steer_action = actions_env[..., 0:1]
        throttle_action = actions_env[..., 1:2]
        brake_action = actions_env[..., 2:3]

        # Reconstruct raw steer action from squashed steer_action for log_prob calculation
        # Clamp to prevent NaNs in atanh for values very close to -1 or 1
        steer_raw = torch.atanh(steer_action.clamp(-0.999999, 0.999999))
        
        # Compute log_prob for steer (Normal with tanh correction)
        log_prob_steer_raw = steer_dist.log_prob(steer_raw).sum(dim=-1)
        log_prob_steer_correction = torch.sum(torch.log(1 - steer_action.pow(2) + 1e-6), dim=-1)
        log_prob_steer = log_prob_steer_raw - log_prob_steer_correction
        
        # Compute log_prob for throttle and brake (Beta distribution directly)
        log_prob_throttle = throttle_dist.log_prob(throttle_action).sum(dim=-1)
        log_prob_brake = brake_dist.log_prob(brake_action).sum(dim=-1)
        
        # Total log_prob
        total_log_prob = log_prob_steer + log_prob_throttle + log_prob_brake
        
        # Compute entropy
        # For Normal, use its entropy. For Beta, use its entropy.
        entropy_steer = steer_dist.entropy().sum(dim=-1)
        entropy_throttle = throttle_dist.entropy().sum(dim=-1)
        entropy_brake = brake_dist.entropy().sum(dim=-1)
        
        total_entropy = entropy_steer + entropy_throttle + entropy_brake
        
        return value, total_log_prob, total_entropy


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

    # Removed _raw_to_env_action_np as act() now returns environment actions directly

    def _collect_rollout(self, env, obs, n_steps: int):
        """Collect n_steps of experience from the vectorized env."""
        n_envs = env.num_envs
        obs_buf = np.zeros((n_steps, n_envs, *self.obs_shape), dtype=np.uint8)
        # act_buf now stores the ENV actions directly
        act_buf = np.zeros((n_steps, n_envs, self.action_dim), dtype=np.float32) 
        rew_buf = np.zeros((n_steps, n_envs), dtype=np.float32)
        done_buf = np.zeros((n_steps, n_envs), dtype=np.float32)
        val_buf = np.zeros((n_steps, n_envs), dtype=np.float32)
        logp_buf = np.zeros((n_steps, n_envs), dtype=np.float32)

        for step in range(n_steps):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device) / 255.0
            with torch.inference_mode():
                # self.policy.act now returns env_action and corrected log_prob
                action_env, log_prob, value = self.policy.act(obs_t)
            
            # Store the environment actions for PPO update
            act_np_env = action_env.cpu().numpy()
            
            obs_buf[step] = (obs * 255).astype(np.uint8) if obs.max() <= 1.0 else obs
            act_buf[step] = act_np_env # Store environment actions
            val_buf[step] = value.cpu().numpy()
            logp_buf[step] = log_prob.cpu().numpy()

            # Env.step directly uses the env_action
            obs, rewards, dones, infos = env.step(act_np_env)
            rew_buf[step] = rewards
            done_buf[step] = dones

        # Bootstrap last value
        with torch.inference_mode():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device) / 255.0
            # Policy.act needs to be consistent, returns env_action, corrected log_prob
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
        # act_flat now contains environment actions
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
                # act_b now contains environment actions
                act_b = torch.as_tensor(act_flat[batch_idx], dtype=torch.float32, device=self.device) 
                old_logp_b = torch.as_tensor(logp_flat[batch_idx], dtype=torch.float32, device=self.device)
                adv_b = torch.as_tensor(adv_flat[batch_idx], dtype=torch.float32, device=self.device)
                ret_b = torch.as_tensor(ret_flat[batch_idx], dtype=torch.float32, device=self.device)

                # evaluate_actions now expects environment actions
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