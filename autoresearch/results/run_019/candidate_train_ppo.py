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
    "min_log_std": -1.0, # General min log_std for throttle/brake raw actions (not used for Beta)
    "max_log_std": 0.5,  # General max log_std for throttle/brake raw actions (not used for Beta)
    "steer_min_log_std": -0.5, # Specific min log_std for steer raw action
    "steer_max_log_std": 0.0, # Specific max log_std for steer raw action
}


# === TUNABLE: Network Architecture ===

class CnnActorCritic(nn.Module):
    """Actor-critic for image observations (N, C, H, W) with mixed action distributions."""

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
        
        # Output: 1 for steer mean, 2 for throttle (alpha, beta raw), 2 for brake (alpha, beta raw)
        # Total 5 outputs for 3 environment actions
        self.policy_mean = nn.Linear(128, 5) # Updated output dimension for mixed distribution

        self.value_mlp = nn.Sequential(
            nn.Linear(n_flatten, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
        )
        self.value_head = nn.Linear(128, 1)

        # self.log_std now only for the steer action (1 parameter)
        # Initial std for steer
        self.log_std = nn.Parameter(torch.full((1,), float(hp.get("steer_min_log_std", -0.5))))

        self.steer_min_log_std = float(hp.get("steer_min_log_std", -0.5))
        self.steer_max_log_std = float(hp.get("steer_max_log_std", 0.0))
        
        # No initial biases for Beta distribution raw parameters for now,
        # let the network learn based on Softplus + 1 activation.
        # Original biases removed as they don't apply to Beta raw parameters directly.

    def _features(self, obs: torch.Tensor) -> torch.Tensor:
        return self.features(obs)

    def get_raw_dist_and_value(self, obs: torch.Tensor):
        """Returns the untransformed (Normal for steer, Beta for throttle/brake) distributions and value."""
        shared = self._features(obs)
        policy_latent = self.policy_mlp(shared)
        raw_policy_output = self.policy_mean(policy_latent)

        # Steer distribution (Gaussian, then tanh squashed)
        steer_mean = raw_policy_output[..., 0:1] # Shape (N, 1)
        steer_log_std = torch.clamp(self.log_std, self.steer_min_log_std, self.steer_max_log_std)
        steer_dist = Normal(steer_mean, steer_log_std.exp())

        # Throttle distribution (Beta)
        throttle_alpha_param = raw_policy_output[..., 1:2] # Shape (N, 1)
        throttle_beta_param = raw_policy_output[..., 2:3]  # Shape (N, 1)
        throttle_alpha = F.softplus(throttle_alpha_param) + 1.0
        throttle_beta = F.softplus(throttle_beta_param) + 1.0
        throttle_dist = Beta(throttle_alpha, throttle_beta)

        # Brake distribution (Beta)
        brake_alpha_param = raw_policy_output[..., 3:4] # Shape (N, 1)
        brake_beta_param = raw_policy_output[..., 4:5]  # Shape (N, 1)
        brake_alpha = F.softplus(brake_alpha_param) + 1.0
        brake_beta = F.softplus(brake_beta_param) + 1.0
        brake_dist = Beta(brake_alpha, brake_beta)
        
        value_latent = self.value_mlp(shared)
        value = self.value_head(value_latent).squeeze(-1)
        
        return (steer_dist, throttle_dist, brake_dist), value

    @staticmethod
    def raw_to_env_action(canonical_action: torch.Tensor) -> torch.Tensor:
        """
        canonical_action is a tensor of samples from the policy's canonical distributions:
        - steer: tanh-squashed Normal sample, in (-1, 1)
        - throttle: Beta sample, in (0, 1)
        - brake: Beta sample, in (0, 1)
        
        This function ensures the actions are within the environment's expected range.
        For this mixed distribution, it's largely an identity mapping but includes clamping
        for numerical stability.
        """
        out = canonical_action.clone()
        if out.shape[-1] >= 1:
            out[..., 0] = torch.clamp(out[..., 0], -1.0, 1.0) # Steer
        if out.shape[-1] >= 2:
            out[..., 1] = torch.clamp(out[..., 1], 0.0, 1.0) # Throttle
        if out.shape[-1] >= 3:
            out[..., 2] = torch.clamp(out[..., 2], 0.0, 1.0) # Brake
        return out

    def act(self, obs: torch.Tensor, deterministic: bool = False):
        """
        Returns a tensor of environment actions and the combined log_prob.
        """
        (steer_dist, throttle_dist, brake_dist), value = self.get_raw_dist_and_value(obs)
        
        # Steer action: Sample from Normal, then tanh squashed
        if deterministic:
            steer_raw = steer_dist.mean
        else:
            steer_raw = steer_dist.rsample()
        steer_squashed = torch.tanh(steer_raw)
        
        # Throttle and Brake actions: Sample directly from Beta
        if deterministic:
            # For Beta, deterministic usually means mean, but for exploration sample is better
            throttle_sample = throttle_dist.mean
            brake_sample = brake_dist.mean
        else:
            throttle_sample = throttle_dist.sample()
            brake_sample = brake_dist.sample()
        
        # Concatenate actions into a single tensor
        action_canonical = torch.cat([steer_squashed, throttle_sample, brake_sample], dim=-1)
        
        # Compute log_prob for steer with tanh correction
        log_prob_steer_raw = steer_dist.log_prob(steer_raw).sum(dim=-1)
        log_prob_steer_correction = torch.log(1 - steer_squashed.pow(2) + 1e-6).sum(dim=-1)
        log_prob_steer_squashed = log_prob_steer_raw - log_prob_steer_correction
        
        # Compute log_prob for throttle and brake (Beta distributions)
        log_prob_throttle = throttle_dist.log_prob(throttle_sample).sum(dim=-1)
        log_prob_brake = brake_dist.log_prob(brake_sample).sum(dim=-1)
        
        # Combined log_prob
        log_prob = log_prob_steer_squashed + log_prob_throttle + log_prob_brake
        
        return action_canonical, log_prob, value

    def evaluate_actions(self, obs: torch.Tensor, actions_canonical: torch.Tensor):
        """
        actions_canonical are the actions sampled from the canonical distributions:
        (steer_squashed, throttle_beta_sample, brake_beta_sample).
        """
        (steer_dist, throttle_dist, brake_dist), value = self.get_raw_dist_and_value(obs)
        
        # Separate canonical actions
        steer_actions_squashed = actions_canonical[..., 0:1] # Shape (N, 1)
        throttle_actions = actions_canonical[..., 1:2]       # Shape (N, 1)
        brake_actions = actions_canonical[..., 2:3]          # Shape (N, 1)
        
        # Compute log_prob for steer (revert tanh, then Normal log_prob, then apply tanh correction)
        steer_actions_raw = torch.atanh(steer_actions_squashed.clamp(-0.999999, 0.999999))
        log_prob_steer_raw = steer_dist.log_prob(steer_actions_raw).sum(dim=-1)
        log_prob_steer_correction = torch.log(1 - steer_actions_squashed.pow(2) + 1e-6).sum(dim=-1)
        log_prob_steer_squashed = log_prob_steer_raw - log_prob_steer_correction
        
        # Compute log_prob for throttle and brake (Beta distributions)
        log_prob_throttle = throttle_dist.log_prob(throttle_actions).sum(dim=-1)
        log_prob_brake = brake_dist.log_prob(brake_actions).sum(dim=-1)
        
        # Combined log_prob
        log_prob_canonical = log_prob_steer_squashed + log_prob_throttle + log_prob_brake
        
        # Entropy for each distribution
        entropy_steer = steer_dist.entropy().sum(dim=-1)
        entropy_throttle = throttle_dist.entropy().sum(dim=-1)
        entropy_brake = brake_dist.entropy().sum(dim=-1)
        
        # Combined entropy
        entropy = entropy_steer + entropy_throttle + entropy_brake
        
        return value, log_prob_canonical, entropy


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

    def _raw_to_env_action_np(self, canonical_action_np: np.ndarray) -> np.ndarray:
        canonical_t = torch.as_tensor(canonical_action_np, dtype=torch.float32)
        env_t = CnnActorCritic.raw_to_env_action(canonical_t)
        return env_t.numpy()

    def _collect_rollout(self, env, obs, n_steps: int):
        """Collect n_steps of experience from the vectorized env."""
        n_envs = env.num_envs
        obs_buf = np.zeros((n_steps, n_envs, *self.obs_shape), dtype=np.uint8)
        # act_buf stores the CANONICAL actions, which are passed to evaluate_actions later
        act_buf = np.zeros((n_steps, n_envs, self.action_dim), dtype=np.float32) 
        rew_buf = np.zeros((n_steps, n_envs), dtype=np.float32)
        done_buf = np.zeros((n_steps, n_envs), dtype=np.float32)
        val_buf = np.zeros((n_steps, n_envs), dtype=np.float32)
        logp_buf = np.zeros((n_steps, n_envs), dtype=np.float32)

        for step in range(n_steps):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device) / 255.0
            with torch.inference_mode():
                # self.policy.act returns canonical actions and combined log_prob
                action_canonical, log_prob, value = self.policy.act(obs_t)
            
            # Store the canonical actions for PPO update
            act_np_canonical = action_canonical.cpu().numpy()
            
            # Convert canonical actions to env actions for stepping (may be identity here)
            env_np = self._raw_to_env_action_np(act_np_canonical)

            obs_buf[step] = (obs * 255).astype(np.uint8) if obs.max() <= 1.0 else obs
            act_buf[step] = act_np_canonical # Store canonical actions
            val_buf[step] = value.cpu().numpy()
            logp_buf[step] = log_prob.cpu().numpy()

            obs, rewards, dones, infos = env.step(env_np)
            rew_buf[step] = rewards
            done_buf[step] = dones

        # Bootstrap last value
        with torch.inference_mode():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device) / 255.0
            _, _, last_value = self.policy.act(obs_t) # Policy.act needs to be consistent, returns canonical action, corrected log_prob
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
        # act_flat now contains canonical actions
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
                # act_b now contains canonical actions
                act_b = torch.as_tensor(act_flat[batch_idx], dtype=torch.float32, device=self.device) 
                old_logp_b = torch.as_tensor(logp_flat[batch_idx], dtype=torch.float32, device=self.device)
                adv_b = torch.as_tensor(adv_flat[batch_idx], dtype=torch.float32, device=self.device)
                ret_b = torch.as_tensor(ret_flat[batch_idx], dtype=torch.float32, device=self.device)

                # evaluate_actions now expects canonical actions
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