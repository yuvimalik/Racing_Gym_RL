import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Beta
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
    # Steer uses squashed Gaussian, so it has log_std bounds
    "steer_min_log_std": -0.5, # Specific min log_std for steer raw action
    "steer_max_log_std": 0.0, # Specific max log_std for steer raw action
    # For Beta distribution, parameters alpha/beta must be > 0.
    # We use F.softplus(param) + BETA_EPS for this.
    "beta_min_val_epsilon": 1e-6,
}


# === TUNABLE: Network Architecture ===

class CnnActorCritic(nn.Module):
    """Actor-critic for image observations (N, C, H, W).
    Mixed action space: Steer is squashed Gaussian, Throttle/Brake are Beta distributions.
    """

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
        
        # Policy output:
        # Action 0 (steer): 1 mean output (for Normal distribution)
        # Action 1 (throttle): 2 outputs (alpha_param, beta_param for Beta distribution)
        # Action 2 (brake): 2 outputs (alpha_param, beta_param for Beta distribution)
        # Total outputs: 1 + (action_dim - 1) * 2 = 1 + 2*2 = 5 for action_dim=3
        self.policy_output_dim = 1 + (action_dim - 1) * 2 if action_dim > 1 else 1 # If action_dim=1 (only steer), then 1.
        self.policy_mean_head = nn.Linear(128, self.policy_output_dim)

        self.value_mlp = nn.Sequential(
            nn.Linear(n_flatten, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
        )
        self.value_head = nn.Linear(128, 1)

        # Log standard deviation for steer action (Normal distribution)
        # This parameter is only for the steer action.
        self.log_std_steer = nn.Parameter(torch.full((1,), 0.0)) # Initial std for steer
        
        self.steer_min_log_std = float(hp.get("steer_min_log_std", -0.5))
        self.steer_max_log_std = float(hp.get("steer_max_log_std", 0.0))
        self.beta_min_val_epsilon = float(hp.get("beta_min_val_epsilon", 1e-6))


        # Initial biases for throttle/brake Beta parameters
        # For throttle (index 1 in env action): encourage alpha > beta for initial forward movement
        # For a 3-dim action space (steer, throttle, brake), the policy_mean_head output is structured as:
        # [0]: steer_mean
        # [1]: throttle_alpha_param
        # [2]: throttle_beta_param
        # [3]: brake_alpha_param
        # [4]: brake_beta_param
        if action_dim >= 3:
            # Throttle: initial bias for alpha_param (positive) and beta_param (negative)
            # This pushes the Beta distribution towards higher values (more throttle).
            nn.init.constant_(self.policy_mean_head.bias[1], 2.0) # throttle_alpha_param
            nn.init.constant_(self.policy_mean_head.bias[2], -3.0) # throttle_beta_param
            # Brake: initial bias for alpha_param (negative) and beta_param (positive)
            # This pushes the Beta distribution towards lower values (less brake).
            nn.init.constant_(self.policy_mean_head.bias[3], -3.0) # brake_alpha_param
            nn.init.constant_(self.policy_mean_head.bias[4], 2.0) # brake_beta_param


    def _features(self, obs: torch.Tensor) -> torch.Tensor:
        return self.features(obs)

    def get_raw_dist_and_value(self, obs: torch.Tensor):
        """Returns the distributions for each action and value."""
        shared = self._features(obs)
        policy_latent = self.policy_mlp(shared)
        policy_output = self.policy_mean_head(policy_latent)

        # Steer action (index 0): Normal distribution
        steer_mean = policy_output[..., 0:1]
        steer_log_std = torch.clamp(self.log_std_steer, self.steer_min_log_std, self.steer_max_log_std)
        steer_std = steer_log_std.exp()
        steer_dist = Normal(steer_mean, steer_std)

        throttle_dist = None
        brake_dist = None

        # Throttle (index 1) and Brake (index 2) actions: Beta distribution
        if self.policy_output_dim >= 5: # action_dim >= 3 for both throttle and brake
            throttle_alpha_param = policy_output[..., 1:2]
            throttle_beta_param = policy_output[..., 2:3]
            brake_alpha_param = policy_output[..., 3:4]
            brake_beta_param = policy_output[..., 4:5]

            # Ensure alpha and beta are positive for Beta distribution using softplus
            throttle_alpha = F.softplus(throttle_alpha_param) + self.beta_min_val_epsilon
            throttle_beta = F.softplus(throttle_beta_param) + self.beta_min_val_epsilon
            brake_alpha = F.softplus(brake_alpha_param) + self.beta_min_val_epsilon
            brake_beta = F.softplus(brake_beta_param) + self.beta_min_val_epsilon

            throttle_dist = Beta(throttle_alpha, throttle_beta)
            brake_dist = Beta(brake_alpha, brake_beta)
        elif self.policy_output_dim >= 3: # action_dim == 2 for only throttle
             throttle_alpha_param = policy_output[..., 1:2]
             throttle_beta_param = policy_output[..., 2:3]
             throttle_alpha = F.softplus(throttle_alpha_param) + self.beta_min_val_epsilon
             throttle_beta = F.softplus(throttle_beta_param) + self.beta_min_val_epsilon
             throttle_dist = Beta(throttle_alpha, throttle_beta)

        value_latent = self.value_mlp(shared)
        value = self.value_head(value_latent).squeeze(-1)
        
        return steer_dist, throttle_dist, brake_dist, value

    @staticmethod
    def raw_to_env_action(combined_action: torch.Tensor) -> torch.Tensor:
        """
        combined_action: 
            - Action 0 (steer) is squashed Normal output, range (-1, 1).
            - Action 1 (throttle) is Beta output, range (0, 1).
            - Action 2 (brake) is Beta output, range (0, 1).
        This function already receives actions in their final env-ready ranges, 
        so it just returns the tensor.
        """
        # Steer is already in (-1, 1)
        # Throttle and Brake are already in (0, 1) from Beta distribution
        return combined_action.clamp(-1.0, 1.0) # Clamp to be safe, especially for steer

    def act(self, obs: torch.Tensor, deterministic: bool = False):
        """
        Returns combined action (steer_squashed, throttle, brake) and its total log_prob.
        """
        steer_dist, throttle_dist, brake_dist, value = self.get_raw_dist_and_value(obs)
        
        # Steer: Squashed Gaussian (-1 to 1)
        if deterministic:
            steer_raw = steer_dist.mean
        else:
            steer_raw = steer_dist.rsample()
        steer_squashed = torch.tanh(steer_raw)
        
        # Calculate log_prob for steer with tanh correction
        log_prob_steer_raw = steer_dist.log_prob(steer_raw).sum(dim=-1)
        log_prob_steer_correction = torch.sum(torch.log(1 - steer_squashed.pow(2) + 1e-6), dim=-1)
        log_prob_steer = log_prob_steer_raw - log_prob_steer_correction
        
        # Combine action parts and total log_prob
        action_parts = [steer_squashed]
        total_log_prob = log_prob_steer
        
        # Throttle and Brake: Beta distribution (0 to 1)
        if throttle_dist is not None:
            throttle_action = throttle_dist.mean if deterministic else throttle_dist.sample()
            action_parts.append(throttle_action)
            total_log_prob += throttle_dist.log_prob(throttle_action).sum(dim=-1)
        
        if brake_dist is not None:
            brake_action = brake_dist.mean if deterministic else brake_dist.sample()
            action_parts.append(brake_action)
            total_log_prob += brake_dist.log_prob(brake_action).sum(dim=-1)
        
        combined_action = torch.cat(action_parts, dim=-1)
        
        return combined_action, total_log_prob, value

    def evaluate_actions(self, obs: torch.Tensor, actions_combined: torch.Tensor):
        """
        actions_combined are the actions sampled from the mixed distribution:
        steer in (-1, 1), throttle in (0, 1), brake in (0, 1).
        """
        steer_dist, throttle_dist, brake_dist, value = self.get_raw_dist_and_value(obs)
        
        # Split combined actions
        steer_squashed = actions_combined[..., 0:1]
        
        log_prob_steer = torch.zeros_like(value)
        entropy_steer = torch.zeros_like(value)

        # Steer: Convert back to raw action to evaluate Normal dist log_prob
        # Clamp to prevent NaNs in atanh for values very close to -1 or 1
        steer_raw = torch.atanh(steer_squashed.clamp(-0.999999, 0.999999))
        log_prob_steer_raw = steer_dist.log_prob(steer_raw).sum(dim=-1)
        # Log_prob correction for tanh squashing
        log_prob_steer_correction = torch.sum(torch.log(1 - steer_squashed.pow(2) + 1e-6), dim=-1)
        log_prob_steer = log_prob_steer_raw - log_prob_steer_correction
        entropy_steer = steer_dist.entropy().sum(dim=-1) # Entropy of the raw Normal distribution
        
        total_log_prob = log_prob_steer
        total_entropy = entropy_steer
        
        # Throttle and Brake: Evaluate Beta distribution log_prob and entropy
        if throttle_dist is not None:
            throttle_action = actions_combined[..., 1:2]
            # Clamp throttle action to avoid issues with log_prob at boundaries if it somehow goes out of (0,1)
            throttle_action = throttle_action.clamp(self.beta_min_val_epsilon, 1.0 - self.beta_min_val_epsilon)
            total_log_prob += throttle_dist.log_prob(throttle_action).sum(dim=-1)
            total_entropy += throttle_dist.entropy().sum(dim=-1)
        
        if brake_dist is not None:
            brake_action = actions_combined[..., 2:3]
            # Clamp brake action
            brake_action = brake_action.clamp(self.beta_min_val_epsilon, 1.0 - self.beta_min_val_epsilon)
            total_log_prob += brake_dist.log_prob(brake_action).sum(dim=-1)
            total_entropy += brake_dist.entropy().sum(dim=-1)
            
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

    def _raw_to_env_action_np(self, raw_action_np: np.ndarray) -> np.ndarray:
        raw_t = torch.as_tensor(raw_action_np, dtype=torch.float32)
        env_t = CnnActorCritic.raw_to_env_action(raw_t)
        return env_t.numpy()

    def _collect_rollout(self, env, obs, n_steps: int):
        """Collect n_steps of experience from the vectorized env."""
        n_envs = env.num_envs
        obs_buf = np.zeros((n_steps, n_envs, *self.obs_shape), dtype=np.uint8)
        # act_buf stores the combined actions (steer:(-1,1), throttle:(0,1), brake:(0,1))
        act_buf = np.zeros((n_steps, n_envs, self.action_dim), dtype=np.float32) 
        rew_buf = np.zeros((n_steps, n_envs), dtype=np.float32)
        done_buf = np.zeros((n_steps, n_envs), dtype=np.float32)
        val_buf = np.zeros((n_steps, n_envs), dtype=np.float32)
        logp_buf = np.zeros((n_steps, n_envs), dtype=np.float32)

        for step in range(n_steps):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device) / 255.0
            with torch.inference_mode():
                # self.policy.act now returns combined action and corrected total log_prob
                action_combined, log_prob, value = self.policy.act(obs_t)
            
            # Store the combined actions for PPO update
            act_np_combined = action_combined.cpu().numpy()
            
            # Convert combined actions to env actions for stepping (no op now)
            env_np = self._raw_to_env_action_np(act_np_combined)

            obs_buf[step] = (obs * 255).astype(np.uint8) if obs.max() <= 1.0 else obs
            act_buf[step] = act_np_combined # Store combined actions
            val_buf[step] = value.cpu().numpy()
            logp_buf[step] = log_prob.cpu().numpy()

            obs, rewards, dones, infos = env.step(env_np)
            rew_buf[step] = rewards
            done_buf[step] = dones

        # Bootstrap last value
        with torch.inference_mode():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device) / 255.0
            # Policy.act needs to be consistent, returns combined action, total log_prob
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
        # act_flat now contains combined actions
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
                # act_b now contains combined actions
                act_b = torch.as_tensor(act_flat[batch_idx], dtype=torch.float32, device=self.device) 
                old_logp_b = torch.as_tensor(logp_flat[batch_idx], dtype=torch.float32, device=self.device)
                adv_b = torch.as_tensor(adv_flat[batch_idx], dtype=torch.float32, device=self.device)
                ret_b = torch.as_tensor(ret_flat[batch_idx], dtype=torch.float32, device=self.device)

                # evaluate_actions now expects combined actions
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