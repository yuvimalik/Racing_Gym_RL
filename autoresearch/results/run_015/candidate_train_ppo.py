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
    "min_steer_log_std": -0.5, # Specific min log_std for steer
    "max_steer_log_std": 0.0,  # Specific max log_std for steer
    "initial_steer_log_std": 0.0, # Initial log_std for steer
}


# === TUNABLE: Network Architecture ===

class CnnActorCritic(nn.Module):
    """Actor-critic for image observations (N, C, H, W).
    Uses Normal distribution with tanh squashing for steer,
    and Beta distribution for throttle and brake.
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
        
        # Policy head for steer (Normal distribution mean)
        self.steer_mean_head = nn.Linear(128, 1)
        self.steer_log_std = nn.Parameter(torch.full((1,), float(hp.get("initial_steer_log_std", 0.0))))

        # Policy heads for throttle and brake (Beta distribution alpha and beta parameters)
        # Beta distribution parameters (alpha, beta) must be > 0.
        # We use F.softplus to ensure positivity.
        self.tb_alpha_head = nn.Linear(128, 2) # For throttle and brake
        self.tb_beta_head = nn.Linear(128, 2)   # For throttle and brake

        self.min_steer_log_std = float(hp.get("min_steer_log_std", -0.5))
        self.max_steer_log_std = float(hp.get("max_steer_log_std", 0.0))

        self.value_mlp = nn.Sequential(
            nn.Linear(n_flatten, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
        )
        self.value_head = nn.Linear(128, 1)

        # Initial biases
        nn.init.constant_(self.steer_mean_head.bias[0], 0.0) # Steer mean around 0
        # Initialize alpha/beta heads to encourage values > 1 for smoother distributions
        # F.softplus(1.31) is approx 2.0. So alpha=2, beta=2 is a good starting point.
        nn.init.constant_(self.tb_alpha_head.bias, 1.31) 
        nn.init.constant_(self.tb_beta_head.bias, 1.31) 

    def _features(self, obs: torch.Tensor) -> torch.Tensor:
        return self.features(obs)

    def get_dists_and_value(self, obs: torch.Tensor):
        """Returns the steer (Normal) and throttle/brake (Beta) distributions and value."""
        shared = self._features(obs)
        policy_latent = self.policy_mlp(shared)
        
        # Steer distribution (Normal with tanh squashing)
        steer_mean = self.steer_mean_head(policy_latent)
        steer_log_std = torch.clamp(self.steer_log_std, self.min_steer_log_std, self.max_steer_log_std)
        steer_std = steer_log_std.exp()
        steer_dist = Normal(steer_mean, steer_std)
        
        # Throttle/Brake distribution (Beta)
        # Add a small epsilon to softplus output to ensure alpha/beta > 0
        tb_alpha = F.softplus(self.tb_alpha_head(policy_latent)) + 1e-5
        tb_beta = F.softplus(self.tb_beta_head(policy_latent)) + 1e-5
        tb_dist = Beta(tb_alpha, tb_beta)
        
        value_latent = self.value_mlp(shared)
        value = self.value_head(value_latent).squeeze(-1)
        
        return steer_dist, tb_dist, value

    @staticmethod
    def raw_to_env_action(raw_action: torch.Tensor) -> torch.Tensor:
        """
        raw_action is the action output by CnnActorCritic.act (steer [-1,1], throttle/brake [0,1]).
        This method acts as an identity function for actions already in the environment's range.
        It's kept for API consistency but performs no transformation.
        """
        return raw_action

    def act(self, obs: torch.Tensor, deterministic: bool = False):
        """
        Returns environment-ready action (steer [-1,1], throttle/brake [0,1])
        and its corrected log_prob.
        """
        steer_dist, tb_dist, value = self.get_dists_and_value(obs)
        
        if deterministic:
            steer_raw = steer_dist.mean
            tb_action = tb_dist.mean # Mean of Beta distribution
        else:
            steer_raw = steer_dist.rsample()
            tb_action = tb_dist.rsample() # rsample() for Beta supports reparameterization trick
        
        # Steer action is squashed with tanh
        steer_squashed = torch.tanh(steer_raw)
        
        # Combine steer and throttle/brake actions
        action_env = torch.cat([steer_squashed, tb_action], dim=-1)
        
        # Compute log_prob for steer with tanh correction
        log_prob_steer_raw = steer_dist.log_prob(steer_raw).sum(dim=-1)
        log_prob_steer_correction = torch.sum(torch.log(1 - steer_squashed.pow(2) + 1e-6), dim=-1)
        log_prob_steer = log_prob_steer_raw - log_prob_steer_correction
        
        # Compute log_prob for throttle/brake from Beta distribution
        log_prob_tb = tb_dist.log_prob(tb_action).sum(dim=-1)
        
        # Total log_prob is the sum for independent distributions
        log_prob = log_prob_steer + log_prob_tb
        
        return action_env, log_prob, value

    def evaluate_actions(self, obs: torch.Tensor, actions_env: torch.Tensor):
        """
        actions_env are the actions sampled from the combined distribution
        (steer [-1,1], throttle/brake [0,1]).
        """
        steer_dist, tb_dist, value = self.get_dists_and_value(obs)
        
        # Separate steer and throttle/brake actions
        steer_actions_env = actions_env[:, 0:1]
        tb_actions_env = actions_env[:, 1:]
        
        # Convert squashed steer action back to raw steer for Normal distribution log_prob
        # Clamp to prevent NaNs in atanh for values very close to -1 or 1
        steer_raw_actions = torch.atanh(steer_actions_env.clamp(-0.999999, 0.999999))
        
        # Log_prob for steer (Normal + tanh correction)
        log_prob_steer_raw = steer_dist.log_prob(steer_raw_actions).sum(dim=-1)
        log_prob_steer_correction = torch.sum(torch.log(1 - steer_actions_env.pow(2) + 1e-6), dim=-1)
        log_prob_steer = log_prob_steer_raw - log_prob_steer_correction
        
        # Log_prob for throttle/brake (Beta)
        log_prob_tb = tb_dist.log_prob(tb_actions_env).sum(dim=-1)
        
        log_prob_combined = log_prob_steer + log_prob_tb
        
        # Entropy for steer (Normal distribution)
        entropy_steer = steer_dist.entropy().sum(dim=-1)
        # Entropy for throttle/brake (Beta distribution)
        entropy_tb = tb_dist.entropy().sum(dim=-1)
        
        entropy_combined = entropy_steer + entropy_tb
        
        return value, log_prob_combined, entropy_combined


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
        # With the new CnnActorCritic.act, the output is already env-ready.
        # This function becomes an identity mapping to satisfy the API.
        return raw_action_np

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
                # self.policy.act now returns env-ready action and corrected log_prob
                action_env, log_prob, value = self.policy.act(obs_t)
            
            # Store the env-ready actions for PPO update
            act_np_env = action_env.cpu().numpy()
            
            # Convert action_env (already env-ready) to env_np for stepping
            env_np = self._raw_to_env_action_np(act_np_env)

            obs_buf[step] = (obs * 255).astype(np.uint8) if obs.max() <= 1.0 else obs
            act_buf[step] = act_np_env # Store env-ready actions
            val_buf[step] = value.cpu().numpy()
            logp_buf[step] = log_prob.cpu().numpy()

            obs, rewards, dones, infos = env.step(env_np)
            rew_buf[step] = rewards
            done_buf[step] = dones

        # Bootstrap last value
        with torch.inference_mode():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device) / 255.0
            _, _, last_value = self.policy.act(obs_t) # Policy.act needs to be consistent, returns env-ready action, corrected log_prob
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