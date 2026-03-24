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
    # min_log_std, max_log_std are no longer used for throttle/brake with Beta distribution
    "min_log_std": -1.0, 
    "max_log_std": 0.5, 
    "steer_min_log_std": -0.5, # Specific min log_std for steer raw action
    "steer_max_log_std": 0.0, # Specific max log_std for steer raw action
}


# === TUNABLE: Network Architecture ===

class CnnActorCritic(nn.Module):
    """Actor-critic for image observations (N, C, H, W).
    Uses a mixed action distribution:
    - Steer: Tanh-squashed Gaussian (range -1 to 1)
    - Throttle: Beta distribution (range 0 to 1)
    - Brake: Beta distribution (range 0 to 1)
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
        # Policy head output:
        # 0: steer_mean (for Normal distribution)
        # 1: throttle_log_alpha (for Beta distribution)
        # 2: throttle_log_beta (for Beta distribution)
        # 3: brake_log_alpha (for Beta distribution)
        # 4: brake_log_beta (for Beta distribution)
        # Total output size: 1 (steer) + 2 (throttle) + 2 (brake) = 5
        self.policy_head = nn.Linear(128, 5)

        self.value_mlp = nn.Sequential(
            nn.Linear(n_flatten, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
        )
        self.value_head = nn.Linear(128, 1)

        # Only one log_std parameter for the steer action
        self.log_std_steer = nn.Parameter(torch.tensor([-0.5], dtype=torch.float32))

        # Steering log_std limits
        self.steer_min_log_std = float(hp.get("steer_min_log_std", -0.5))
        self.steer_max_log_std = float(hp.get("steer_max_log_std", 0.0))

        # Initial biases for Beta distribution parameters (log_alpha, log_beta)
        # These are added to the raw outputs of the policy_head for throttle/brake parameters.
        # For throttle: bias towards higher values (e.g., Beta(2,1) -> mean 2/3)
        self.register_buffer("throttle_log_alpha_bias", torch.tensor(np.log(2.0), dtype=torch.float32))
        self.register_buffer("throttle_log_beta_bias", torch.tensor(np.log(1.0), dtype=torch.float32))
        # For brake: bias towards lower values (e.g., Beta(1,2) -> mean 1/3)
        self.register_buffer("brake_log_alpha_bias", torch.tensor(np.log(1.0), dtype=torch.float32))
        self.register_buffer("brake_log_beta_bias", torch.tensor(np.log(2.0), dtype=torch.float32))


    def _features(self, obs: torch.Tensor) -> torch.Tensor:
        return self.features(obs)

    def get_all_dists_and_value(self, obs: torch.Tensor):
        """Returns a list of distributions (Normal for steer, Beta for throttle/brake) and value."""
        shared = self._features(obs)
        policy_latent = self.policy_mlp(shared)
        raw_policy_output = self.policy_head(policy_latent) # Shape: (N, 5)

        # 1. Steer distribution (Normal, tanh-squashed)
        steer_mean = raw_policy_output[:, 0]
        log_std_steer_clamped = torch.clamp(self.log_std_steer, self.steer_min_log_std, self.steer_max_log_std)
        steer_std = log_std_steer_clamped.exp()
        steer_dist = Normal(steer_mean, steer_std)

        # 2. Throttle distribution (Beta)
        # Ensure alpha, beta are positive by exponentiating log_alpha/log_beta
        log_alpha_throttle = raw_policy_output[:, 1] + self.throttle_log_alpha_bias
        log_beta_throttle = raw_policy_output[:, 2] + self.throttle_log_beta_bias
        throttle_alpha = log_alpha_throttle.exp() + 1e-6 # Add small epsilon to prevent 0 or issues
        throttle_beta = log_beta_throttle.exp() + 1e-6
        throttle_dist = Beta(throttle_alpha, throttle_beta)

        # 3. Brake distribution (Beta)
        log_alpha_brake = raw_policy_output[:, 3] + self.brake_log_alpha_bias
        log_beta_brake = raw_policy_output[:, 4] + self.brake_log_beta_bias
        brake_alpha = log_alpha_brake.exp() + 1e-6
        brake_beta = log_beta_brake.exp() + 1e-6
        brake_dist = Beta(brake_alpha, brake_beta)
        
        value_latent = self.value_mlp(shared)
        value = self.value_head(value_latent).squeeze(-1)

        return [steer_dist, throttle_dist, brake_dist], value

    @staticmethod
    def raw_to_env_action(action_from_act: torch.Tensor) -> torch.Tensor:
        """
        action_from_act is the action tensor produced by self.act(),
        which should already be in the environment's expected ranges:
        steer (-1, 1), throttle (0, 1), brake (0, 1).
        No further transformation is needed.
        """
        return action_from_act.clone()

    def act(self, obs: torch.Tensor, deterministic: bool = False):
        """
        Returns combined action [steer, throttle, brake] and its total log_prob.
        Steer is tanh-squashed Gaussian. Throttle/Brake are Beta.
        """
        dists, value = self.get_all_dists_and_value(obs)
        steer_dist, throttle_dist, brake_dist = dists

        # 1. Steer (Normal)
        if deterministic:
            steer_raw = steer_dist.mean
        else:
            steer_raw = steer_dist.rsample()
        steer_squashed = torch.tanh(steer_raw)

        # 2. Throttle (Beta)
        if deterministic:
            throttle_action = throttle_dist.mean # Beta mean alpha / (alpha + beta)
        else:
            throttle_action = throttle_dist.sample()

        # 3. Brake (Beta)
        if deterministic:
            brake_action = brake_dist.mean
        else:
            brake_action = brake_dist.sample()

        # Combine actions into a single tensor
        action = torch.stack([steer_squashed, throttle_action, brake_action], dim=-1)

        # Calculate total log_prob
        # Log_prob for steer (with tanh correction)
        log_prob_steer_raw = steer_dist.log_prob(steer_raw).sum(dim=-1)
        log_prob_steer_correction = torch.sum(torch.log(1 - steer_squashed.pow(2) + 1e-6), dim=-1)
        log_prob_steer = log_prob_steer_raw - log_prob_steer_correction

        # Log_prob for throttle (Beta distribution directly)
        log_prob_throttle = throttle_dist.log_prob(throttle_action)

        # Log_prob for brake (Beta distribution directly)
        log_prob_brake = brake_dist.log_prob(brake_action)
        
        total_log_prob = log_prob_steer + log_prob_throttle + log_prob_brake

        return action, total_log_prob, value

    def evaluate_actions(self, obs: torch.Tensor, actions_taken: torch.Tensor):
        """
        actions_taken are the actions from the environment (steer -1 to 1, throttle 0 to 1, brake 0 to 1).
        """
        dists, value = self.get_all_dists_and_value(obs)
        steer_dist, throttle_dist, brake_dist = dists

        # Extract individual actions
        steer_squashed = actions_taken[..., 0]
        throttle_action = actions_taken[..., 1]
        brake_action = actions_taken[..., 2]

        # 1. Steer log_prob and entropy (tanh correction)
        steer_raw = torch.atanh(steer_squashed.clamp(-0.999999, 0.999999))
        log_prob_steer_raw = steer_dist.log_prob(steer_raw).sum(dim=-1)
        log_prob_steer_correction = torch.sum(torch.log(1 - steer_squashed.pow(2) + 1e-6), dim=-1)
        log_prob_steer = log_prob_steer_raw - log_prob_steer_correction
        entropy_steer = steer_dist.entropy().sum(dim=-1)

        # 2. Throttle log_prob and entropy (Beta distribution directly)
        log_prob_throttle = throttle_dist.log_prob(throttle_action)
        entropy_throttle = throttle_dist.entropy()

        # 3. Brake log_prob and entropy (Beta distribution directly)
        log_prob_brake = brake_dist.log_prob(brake_action)
        entropy_brake = brake_dist.entropy()

        total_log_prob = log_prob_steer + log_prob_throttle + log_prob_brake
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

    def _raw_to_env_action_np(self, raw_action_np: np.ndarray) -> np.ndarray:
        # With the new mixed distribution, policy.act directly outputs environment actions.
        # This function simply converts to tensor, calls the static method, and converts back.
        raw_t = torch.as_tensor(raw_action_np, dtype=torch.float32)
        env_t = CnnActorCritic.raw_to_env_action(raw_t)
        return env_t.numpy()

    def _collect_rollout(self, env, obs, n_steps: int):
        """Collect n_steps of experience from the vectorized env."""
        n_envs = env.num_envs
        obs_buf = np.zeros((n_steps, n_envs, *self.obs_shape), dtype=np.uint8)
        # act_buf stores the environment-ready actions
        act_buf = np.zeros((n_steps, n_envs, self.action_dim), dtype=np.float32) 
        rew_buf = np.zeros((n_steps, n_envs), dtype=np.float32)
        done_buf = np.zeros((n_steps, n_envs), dtype=np.float32)
        val_buf = np.zeros((n_steps, n_envs), dtype=np.float32)
        logp_buf = np.zeros((n_steps, n_envs), dtype=np.float32)

        for step in range(n_steps):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device) / 255.0
            with torch.inference_mode():
                # self.policy.act now returns env-ready action and total log_prob
                action, log_prob, value = self.policy.act(obs_t)
            
            # Store the env-ready actions for PPO update
            act_np = action.cpu().numpy()
            
            obs_buf[step] = (obs * 255).astype(np.uint8) if obs.max() <= 1.0 else obs
            act_buf[step] = act_np # Store env-ready actions
            val_buf[step] = value.cpu().numpy()
            logp_buf[step] = log_prob.cpu().numpy()

            obs, rewards, dones, infos = env.step(act_np) # Directly use act_np
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
        # act_flat now contains env-ready actions
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
                # act_b now contains env-ready actions
                act_b = torch.as_tensor(act_flat[batch_idx], dtype=torch.float32, device=self.device) 
                old_logp_b = torch.as_tensor(logp_flat[batch_idx], dtype=torch.float32, device=self.device)
                adv_b = torch.as_tensor(adv_flat[batch_idx], dtype=torch.float32, device=self.device)
                ret_b = torch.as_tensor(ret_flat[batch_idx], dtype=torch.float32, device=self.device)

                # evaluate_actions now expects env-ready actions
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