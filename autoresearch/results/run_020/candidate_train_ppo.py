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
    "min_log_std": -1.0, # General min log_std (now only for non-steer, unused with Beta)
    "max_log_std": 0.5,  # General max log_std (now only for non-steer, unused with Beta)
    "steer_min_log_std": -0.5, # Specific min log_std for steer raw action
    "steer_max_log_std": 0.0, # Specific max log_std for steer raw action
}


# === TUNABLE: Network Architecture ===

class CnnActorCritic(nn.Module):
    """Actor-critic for image observations (N, C, H, W).
    Uses Squashed Gaussian for steer, Beta distribution for throttle/brake.
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

        # Policy head outputs: 1 for steer_mean, 2 for throttle_alpha/beta, 2 for brake_alpha/beta
        # Total: 1 + 2 + 2 = 5 outputs
        self.policy_mlp = nn.Sequential(
            nn.Linear(n_flatten, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
        )
        self.policy_output_head = nn.Linear(128, 5) # Outputs: steer_mean, th_alpha_raw, th_beta_raw, br_alpha_raw, br_beta_raw

        self.value_mlp = nn.Sequential(
            nn.Linear(n_flatten, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
        )
        self.value_head = nn.Linear(128, 1)

        # Log_std for steer only. Throttle/Brake use Beta distribution.
        self.log_std = nn.Parameter(torch.full((1,), -0.5)) # Initial log_std for steer

        self.steer_min_log_std = float(hp.get("steer_min_log_std", -0.5))
        self.steer_max_log_std = float(hp.get("steer_max_log_std", 0.0))

        # Initial biases are removed for throttle/brake as Beta distribution handles [0,1]
        # and its parameters (alpha, beta) determine the mean and shape.
        nn.init.constant_(self.policy_output_head.bias[0], 0.0) # steer mean initial
        # For Beta params, initializing with a positive value like 2.0 encourages more uniform-like starting distributions.
        nn.init.constant_(self.policy_output_head.bias[1], 2.0) # th_alpha_raw
        nn.init.constant_(self.policy_output_head.bias[2], 2.0) # th_beta_raw
        nn.init.constant_(self.policy_output_head.bias[3], 2.0) # br_alpha_raw
        nn.init.constant_(self.policy_output_head.bias[4], 2.0) # br_beta_raw

    def _features(self, obs: torch.Tensor) -> torch.Tensor:
        return self.features(obs)

    def get_raw_dist_and_value(self, obs: torch.Tensor):
        """Returns the distributions for steer, throttle, brake and value."""
        shared = self._features(obs)
        policy_latent = self.policy_mlp(shared)
        policy_output = self.policy_output_head(policy_latent) # shape (N, 5)

        # Steer distribution (Normal)
        steer_mean = policy_output[..., 0]
        steer_log_std = torch.clamp(self.log_std, self.steer_min_log_std, self.steer_max_log_std)
        steer_std = steer_log_std.exp()
        steer_dist = Normal(steer_mean, steer_std)

        # Throttle and Brake distributions (Beta)
        # Beta parameters (alpha, beta) must be > 0.
        # F.softplus(x) + 1.0 ensures they are >= 1 for numerical stability and a well-defined PDF.
        th_alpha_raw = policy_output[..., 1]
        th_beta_raw = policy_output[..., 2]
        th_alpha = F.softplus(th_alpha_raw) + 1.0
        th_beta = F.softplus(th_beta_raw) + 1.0
        throttle_dist = Beta(th_alpha, th_beta)

        br_alpha_raw = policy_output[..., 3]
        br_beta_raw = policy_output[..., 4]
        br_alpha = F.softplus(br_alpha_raw) + 1.0
        br_beta = F.softplus(br_beta_raw) + 1.0
        brake_dist = Beta(br_alpha, br_beta)
        
        value_latent = self.value_mlp(shared)
        value = self.value_head(value_latent).squeeze(-1)
        
        return steer_dist, throttle_dist, brake_dist, value

    @staticmethod
    def raw_to_env_action(action_mixed: torch.Tensor) -> torch.Tensor:
        """
        action_mixed: steer in (-1,1), throttle in (0,1), brake in (0,1).
        Maps to environment actions: steer (-1, 1), throttle (0, 1), brake (0, 1).
        Throttle and Brake are already in [0,1] from Beta sampling, so no further transformation.
        """
        # Steer (index 0) is already in (-1, 1) due to tanh squashing.
        # Throttle (index 1) and Brake (index 2) are already in (0, 1) from Beta sampling.
        return action_mixed

    def act(self, obs: torch.Tensor, deterministic: bool = False):
        """
        Returns mixed action (steer squashed, th/br beta sampled) and its total log_prob.
        """
        steer_dist, throttle_dist, brake_dist, value = self.get_raw_dist_and_value(obs)
        
        # Steer action (squashed Gaussian)
        if deterministic:
            action_raw_steer = steer_dist.mean
        else:
            action_raw_steer = steer_dist.rsample()
        action_squashed_steer = torch.tanh(action_raw_steer)
        
        # Throttle and Brake actions (Beta distribution, naturally bounded [0,1])
        if deterministic:
            action_throttle = throttle_dist.mean # Mean of Beta distribution
            action_brake = brake_dist.mean # Mean of Beta distribution
        else:
            action_throttle = throttle_dist.sample()
            action_brake = brake_dist.sample()
        
        # Concatenate actions: steer (-1,1), throttle (0,1), brake (0,1)
        action_mixed = torch.stack([action_squashed_steer, action_throttle, action_brake], dim=-1)
        
        # Compute log_prob for steer (with tanh correction)
        log_prob_raw_steer = steer_dist.log_prob(action_raw_steer)
        log_prob_correction_steer = torch.log(1 - action_squashed_steer.pow(2) + 1e-6)
        log_prob_steer = log_prob_raw_steer - log_prob_correction_steer
        
        # Compute log_prob for throttle and brake (direct from Beta distribution)
        log_prob_throttle = throttle_dist.log_prob(action_throttle)
        log_prob_brake = brake_dist.log_prob(action_brake)
        
        # Total log_prob is the sum of log_probs for independent action components
        log_prob_total = log_prob_steer + log_prob_throttle + log_prob_brake
        
        return action_mixed, log_prob_total, value

    def evaluate_actions(self, obs: torch.Tensor, actions_mixed: torch.Tensor):
        """
        actions_mixed: steer in (-1,1), throttle in (0,1), brake in (0,1).
        """
        steer_dist, throttle_dist, brake_dist, value = self.get_raw_dist_and_value(obs)
        
        # Split actions_mixed tensor into components
        actions_squashed_steer = actions_mixed[..., 0]
        actions_throttle = actions_mixed[..., 1]
        actions_brake = actions_mixed[..., 2]

        # Steer log_prob: convert squashed action back to raw, then apply tanh correction
        actions_raw_steer = torch.atanh(actions_squashed_steer.clamp(-0.999999, 0.999999))
        log_prob_raw_steer = steer_dist.log_prob(actions_raw_steer)
        log_prob_correction_steer = torch.log(1 - actions_squashed_steer.pow(2) + 1e-6)
        log_prob_steer = log_prob_raw_steer - log_prob_correction_steer
        
        # Throttle and Brake log_prob (direct from Beta distribution)
        log_prob_throttle = throttle_dist.log_prob(actions_throttle)
        log_prob_brake = brake_dist.log_prob(actions_brake)
        
        # Total log_prob
        log_prob_total = log_prob_steer + log_prob_throttle + log_prob_brake
        
        # Total entropy is the sum of entropies for independent action components
        entropy_total = steer_dist.entropy() + throttle_dist.entropy() + brake_dist.entropy()
        
        return value, log_prob_total, entropy_total


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

    def _raw_to_env_action_np(self, action_mixed_np: np.ndarray) -> np.ndarray:
        # Action mixed comes from policy.act(): steer in (-1,1), th/br in (0,1)
        action_mixed_t = torch.as_tensor(action_mixed_np, dtype=torch.float32)
        env_t = CnnActorCritic.raw_to_env_action(action_mixed_t)
        return env_t.numpy()

    def _collect_rollout(self, env, obs, n_steps: int):
        """Collect n_steps of experience from the vectorized env."""
        n_envs = env.num_envs
        obs_buf = np.zeros((n_steps, n_envs, *self.obs_shape), dtype=np.uint8)
        # act_buf stores the mixed actions from policy.act (steer squashed, th/br beta sampled)
        act_buf = np.zeros((n_steps, n_envs, self.action_dim), dtype=np.float32) 
        rew_buf = np.zeros((n_steps, n_envs), dtype=np.float32)
        done_buf = np.zeros((n_steps, n_envs), dtype=np.float32)
        val_buf = np.zeros((n_steps, n_envs), dtype=np.float32)
        logp_buf = np.zeros((n_steps, n_envs), dtype=np.float32)

        for step in range(n_steps):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device) / 255.0
            with torch.inference_mode():
                # self.policy.act now returns mixed action and total log_prob
                action_mixed, log_prob, value = self.policy.act(obs_t)
            
            act_np_mixed = action_mixed.cpu().numpy()
            
            # Convert mixed actions to env actions for stepping (no change needed by raw_to_env_action)
            env_np = self._raw_to_env_action_np(act_np_mixed)

            obs_buf[step] = (obs * 255).astype(np.uint8) if obs.max() <= 1.0 else obs
            act_buf[step] = act_np_mixed # Store mixed actions
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
        # act_flat now contains mixed actions (steer squashed, th/br beta sampled)
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
                # act_b now contains mixed actions
                act_b = torch.as_tensor(act_flat[batch_idx], dtype=torch.float32, device=self.device) 
                old_logp_b = torch.as_tensor(logp_flat[batch_idx], dtype=torch.float32, device=self.device)
                adv_b = torch.as_tensor(adv_flat[batch_idx], dtype=torch.float32, device=self.device)
                ret_b = torch.as_tensor(ret_flat[batch_idx], dtype=torch.float32, device=self.device)

                # evaluate_actions now expects mixed actions
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