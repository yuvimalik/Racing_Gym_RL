from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.distributions import Independent, Normal, kl_divergence


def reconstruction_loss(reconstruction: torch.Tensor, target_images: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(reconstruction, target_images)


def reward_loss(predicted_reward: torch.Tensor, target_reward: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(predicted_reward, target_reward)


def free_bits_kl(
    posterior_mean: torch.Tensor,
    posterior_std: torch.Tensor,
    prior_mean: torch.Tensor,
    prior_std: torch.Tensor,
    free_nats: float = 1.0,
) -> torch.Tensor:
    posterior = Independent(Normal(posterior_mean, posterior_std), 1)
    prior = Independent(Normal(prior_mean, prior_std), 1)
    kl_values = kl_divergence(posterior, prior)
    return torch.clamp(kl_values, min=float(free_nats)).mean()
