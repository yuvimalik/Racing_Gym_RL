"""
Editable MARL policy surface for autoresearch experiments.

Candidate experiments should prefer mutating this API surface instead of
rewriting train.py. The main trainer can optionally load policy variants and
optimizer hooks from a surface module path.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal


class MarlAutoresearchBaselinePolicy(nn.Module):
    """Baseline editable surface matching the promoted autoresearch policy."""

    def __init__(
        self,
        obs_shape,
        action_dim: int,
        min_log_std: float = -1.5,
        max_log_std: float = 1.0,
        steer_min_log_std: float | None = None,
        steer_max_log_std: float | None = None,
    ):
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

        log_std_init = torch.full((action_dim,), -0.5)
        if action_dim >= 1:
            log_std_init[0] = 0.0
        if action_dim >= 3:
            log_std_init[1] = -0.5
            log_std_init[2] = -1.0
        self.log_std = nn.Parameter(log_std_init)
        self.min_log_std = float(min_log_std)
        self.max_log_std = float(max_log_std)
        self.steer_min_log_std = (
            float(steer_min_log_std) if steer_min_log_std is not None else self.min_log_std
        )
        self.steer_max_log_std = (
            float(steer_max_log_std) if steer_max_log_std is not None else self.max_log_std
        )

        if action_dim >= 3:
            nn.init.constant_(self.policy_mean.bias[1], 2.0)
            nn.init.constant_(self.policy_mean.bias[2], -3.0)

    def _features(self, obs: torch.Tensor) -> torch.Tensor:
        return self.features(obs)

    def get_raw_dist_and_value(self, obs: torch.Tensor):
        shared = self._features(obs)
        policy_latent = self.policy_mlp(shared)
        mean = self.policy_mean(policy_latent)
        steer_ls = torch.clamp(self.log_std[0:1], self.steer_min_log_std, self.steer_max_log_std)
        if self.log_std.shape[0] > 1:
            other_ls = torch.clamp(self.log_std[1:], self.min_log_std, self.max_log_std)
            log_std = torch.cat([steer_ls, other_ls])
        else:
            log_std = steer_ls
        std = log_std.exp()
        dist = Normal(mean, std)
        value_latent = self.value_mlp(shared)
        value = self.value_head(value_latent).squeeze(-1)
        return dist, value

    @staticmethod
    def raw_to_env_action(raw_action: torch.Tensor) -> torch.Tensor:
        out = raw_action.clone()
        if out.shape[-1] >= 2:
            out[..., 1] = (out[..., 1] + 1.0) / 2.0
        if out.shape[-1] >= 3:
            out[..., 2] = (out[..., 2] + 1.0) / 2.0
        return out

    def act(self, obs: torch.Tensor, deterministic: bool = False):
        raw_dist, value = self.get_raw_dist_and_value(obs)
        action_raw = raw_dist.mean if deterministic else raw_dist.rsample()
        action_squashed = torch.tanh(action_raw)
        log_prob_raw = raw_dist.log_prob(action_raw).sum(dim=-1)
        log_prob_correction = torch.sum(torch.log(1 - action_squashed.pow(2) + 1e-6), dim=-1)
        log_prob = log_prob_raw - log_prob_correction
        return action_squashed, log_prob, value

    def evaluate_actions(self, obs: torch.Tensor, actions_squashed: torch.Tensor):
        raw_dist, value = self.get_raw_dist_and_value(obs)
        actions_raw = torch.atanh(actions_squashed.clamp(-0.999999, 0.999999))
        log_prob_raw = raw_dist.log_prob(actions_raw).sum(dim=-1)
        log_prob_correction = torch.sum(torch.log(1 - actions_squashed.pow(2) + 1e-6), dim=-1)
        log_prob = log_prob_raw - log_prob_correction
        entropy = raw_dist.entropy().sum(dim=-1)
        return value, log_prob, entropy

    def forward(self, obs: torch.Tensor, actions: torch.Tensor):
        return self.evaluate_actions(obs, actions)


def get_policy_variants():
    """Return policy variants exposed by this editable surface."""
    return {
        "marl_surface_baseline": MarlAutoresearchBaselinePolicy,
    }


def build_optimizer(policy: nn.Module, learning_rate: float):
    """Default optimizer hook for editable surfaces."""
    return torch.optim.Adam(policy.parameters(), lr=float(learning_rate))


def surface_metadata():
    return {
        "surface_name": "marl_surface_baseline",
        "api_version": 1,
        "default_variant": "marl_surface_baseline",
    }
