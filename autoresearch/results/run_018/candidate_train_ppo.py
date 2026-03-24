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
    # These log_std params now apply ONLY to the steer action (Gaussian)
    "min_log_std": -1.0, 
    "max_log_std": 0.5,  
    "steer_min_log_std": -0.5, 
    "steer_max_log_std": 0.0, 
}


# === TUNABLE: Network Architecture ===

class CnnActorCritic(nn.Module):
    """Actor-critic for image observations (N, C, H, W).
    Uses a hybrid action distribution: Gaussian (tanh-squashed) for steer,
    and Beta distributions for throttle and brake.
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
        # For steer (1 action): 1 mean.
        # For throttle (1 action) and brake (1 action): each needs 2 parameters (alpha, beta).
        # Total output = 1 (steer_mean) + 2 (throttle_alpha/beta) + 2 (brake_alpha/beta) = 5
        assert action_dim == 3, "Expected 3 actions (steer, throttle, brake) for hybrid distribution"
        self.policy_mean = nn.Linear(128, 1 + 2 * 2) 

        self.value_mlp = nn.Sequential(
            nn.Linear(n_flatten, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
        )
        self.value_head = nn.Linear(128, 1)

        # log_std only for steer (action_dim = 1 for steer component)
        log_std_init = torch.full((1,), -0.5) 
        self.log_std = nn.Parameter(log_std_init)

        self.min_log_std = float(hp.get("min_log_std", -1.0))
        self.max_log_std = float(hp.get("max_log_std", 0.5))
        self.steer_min_log_std = float(hp.get("steer_min_log_std", self.min_log_std))
        self.steer_max_log_std = float(hp.get("steer_max_log_std", self.max_log_std))

        # Initialize biases for Beta distribution parameters (indices 1 to 4) to encourage alpha=beta=1
        # F.softplus(0) + 1.0 approx 0.693 + 1.0 = 1.693, resulting in mean 0.5 for uniform-like Beta.
        nn.init.constant_(self.policy_mean.bias[1:], 0.0)

    def _features(self, obs: torch.Tensor) -> torch.Tensor:
        return self.features(obs)

    def get_raw_dist_and_value(self, obs: torch.Tensor):
        """Returns the untransformed distributions (Normal for steer, Beta for throttle/brake) and value."""
        shared = self._features(obs)
        policy_latent = self.policy_mlp(shared)
        raw_policy_output = self.policy_mean(policy_latent) # shape (N, 5)

        # Steer distribution (Gaussian)
        steer_mean = raw_policy_output[..., 0] # shape (N,)
        steer_log_std = torch.clamp(self.log_std, self.steer_min_log_std, self.steer_max_log_std)
        steer_std = steer_log_std.exp()
        dist_steer = Normal(steer_mean, steer_std)

        # Throttle distribution (Beta)
        # Using F.softplus(x) + 1.0 to ensure alpha, beta > 1.0
        throttle_alpha_param = F.softplus(raw_policy_output[..., 1]) + 1.0
        throttle_beta_param = F.softplus(raw_policy_output[..., 2]) + 1.0
        dist_throttle = Beta(throttle_alpha_param, throttle_beta_param)

        # Brake distribution (Beta)
        brake_alpha_param = F.softplus(raw_policy_output[..., 3]) + 1.0
        brake_beta_param = F.softplus(raw_policy_output[..., 4]) + 1.0
        dist_brake = Beta(brake_alpha_param, brake_beta_param)
        
        value_latent = self.value_mlp(shared)
        value = self.value_head(value_latent).squeeze(-1)
        
        return dist_steer, dist_throttle, dist_brake, value

    @staticmethod
    def raw_to_env_action(raw_action: torch.Tensor) -> torch.Tensor:
        """
        raw_action is the combined action tensor (steer_squashed, throttle_beta, brake_beta).
        Steer is tanh-squashed, in (-1, 1). Throttle/brake are from Beta dist, in (0, 1).
        All actions are already in the correct range for the environment.
        """
        return raw_action.clone()

    def act(self, obs: torch.Tensor, deterministic: bool = False):
        """
        Returns combined action (steer is tanh-squashed, throttle/brake are Beta outputs)
        and its combined log_prob.
        """
        dist_steer, dist_throttle, dist_brake, value = self.get_raw_dist_and_value(obs)
        
        # Steer (Gaussian with tanh squashing)
        if deterministic:
            raw_steer_action = dist_steer.mean
        else:
            raw_steer_action = dist_steer.rsample()
        squashed_steer_action = torch.tanh(raw_steer_action)
        
        # Throttle (Beta distribution)
        if deterministic:
            throttle_action = dist_throttle.mean
        else:
            # Beta.rsample() is not available, use Beta.sample()
            throttle_action = dist_throttle.sample() 
        
        # Brake (Beta distribution)
        if deterministic:
            brake_action = dist_brake.mean
        else:
            # Beta.rsample() is not available, use Beta.sample()
            brake_action = dist_brake.sample() 

        # Combine actions into a single tensor
        action_combined = torch.stack([squashed_steer_action, throttle_action, brake_action], dim=-1)
        
        # Compute log_prob with tanh correction for steer
        log_prob_steer_raw = dist_steer.log_prob(raw_steer_action)
        # Add epsilon for numerical stability in log(1 - x^2)
        log_prob_steer_correction = torch.log(1 - squashed_steer_action.pow(2) + 1e-6)
        log_prob_steer = log_prob_steer_raw - log_prob_steer_correction
        
        # Log_probs for Beta actions (no squashing correction needed)
        log_prob_throttle = dist_throttle.log_prob(throttle_action)
        log_prob_brake = dist_brake.log_prob(brake_action)
        
        # Sum all log_probs for the total log_prob
        log_prob_total = log_prob_steer + log_prob_throttle + log_prob_brake
        
        return action_combined, log_prob_total, value

    def evaluate_actions(self, obs: torch.Tensor, actions_combined: torch.Tensor):
        """
        Evaluates log_prob and entropy for the given combined actions.
        actions_combined are (steer_squashed, throttle_beta, brake_beta).
        """
        dist_steer, dist_throttle, dist_brake, value = self.get_raw_dist_and_value(obs)
        
        # Split actions
        steer_squashed_action = actions_combined[..., 0]
        throttle_action = actions_combined[..., 1]
        brake_action = actions_combined[..., 2]
        
        # Steer log_prob and entropy (Gaussian with tanh squashing)
        # Clamp steer_squashed_action to prevent NaNs in atanh for values very close to -1 or 1
        raw_steer_action = torch.atanh(steer_squashed_action.clamp(-0.999999, 0.999999))
        log_prob_steer_raw = dist_steer.log_prob(raw_steer_action)
        log_prob_steer_correction = torch.log(1 - steer_squashed_action.pow(2) + 1e-6)
        log_prob_steer = log_prob_steer_raw - log_prob_steer_correction
        entropy_steer = dist_steer.entropy() # Standard Normal entropy (un-squashed)
        
        # Throttle log_prob and entropy (Beta distribution)
        # Clamp Beta actions to (epsilon, 1-epsilon) for log_prob numerical stability
        throttle_action_clamped = throttle_action.clamp(1e-6, 1.0 - 1e-6)
        log_prob_throttle = dist_throttle.log_prob(throttle_action_clamped)
        entropy_throttle = dist_throttle.entropy()
        
        # Brake log_prob and entropy (Beta distribution)
        brake_action_clamped = brake_action.clamp(1e-6, 1.0 - 1e-6)
        log_prob_brake = dist_brake.log_prob(brake_action_clamped)
        entropy_brake = dist_brake.entropy()
        
        # Sum all log_probs and entropies
        log_prob_total = log_prob_steer + log_prob_throttle + log_prob_brake
        entropy_total = entropy_steer + entropy_throttle + entropy_brake
        
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

    def _raw_to_env_action_np(self, raw_action_np: np.ndarray) -> np.ndarray:
        raw_t = torch.as_tensor(raw_action_np, dtype=torch.float32)
        env_t = CnnActorCritic.raw_to_env_action(raw_t)
        return env_t.numpy()

    def _collect_rollout(self, env, obs, n_steps: int):
        """Collect n_steps of experience from the vectorized env."""
        n_envs = env.num_envs
        obs_buf = np.zeros((n_steps, n_envs, *self.obs_shape), dtype=np.uint8)
        # act_buf stores the combined actions (steer_squashed, throttle_beta, brake_beta)
        act_buf = np.zeros((n_steps, n_envs, self.action_dim), dtype=np.float32) 
        rew_buf = np.zeros((n_steps, n_envs), dtype=np.float32)
        done_buf = np.zeros((n_steps, n_envs), dtype=np.float32)
        val_buf = np.zeros((n_steps, n_envs), dtype=np.float32)
        logp_buf = np.zeros((n_steps, n_envs), dtype=np.float32)

        for step in range(n_steps):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device) / 255.0
            with torch.inference_mode():
                # self.policy.act returns combined action and combined log_prob
                action_combined, log_prob, value = self.policy.act(obs_t)
            
            # Store the combined actions for PPO update
            act_np_combined = action_combined.cpu().numpy()
            
            # Convert combined actions to env actions for stepping (no further mapping needed)
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
            # Policy.act needs to be consistent, returns combined action, combined log_prob
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