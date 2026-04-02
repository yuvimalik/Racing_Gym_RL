from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Independent, Normal


@dataclass
class RSSMState:
    deterministic: torch.Tensor
    stochastic: torch.Tensor


@dataclass
class LatentStats:
    mean: torch.Tensor
    std: torch.Tensor
    dist: Independent


@dataclass
class RSSMStepOutput:
    state: RSSMState
    prior: LatentStats
    posterior: Optional[LatentStats]
    reward: torch.Tensor
    reconstruction: torch.Tensor


@dataclass
class RSSMSequenceOutput:
    deterministic: torch.Tensor
    stochastic: torch.Tensor
    prior_mean: torch.Tensor
    prior_std: torch.Tensor
    posterior_mean: torch.Tensor
    posterior_std: torch.Tensor
    reward: torch.Tensor
    reconstruction: torch.Tensor


class Encoder(nn.Module):
    """Compresses 96x96 RGB images into a 512-wide embedding."""

    def __init__(self, embedding_dim: int = 512):
        super().__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.Flatten(),
        )
        self.projection = nn.Linear(256 * 6 * 6, embedding_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.projection(self.conv_stack(images))


class Decoder(nn.Module):
    """Decodes latent features back to 96x96 RGB reconstructions."""

    def __init__(self, input_dim: int = 512):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.SiLU(),
            nn.Linear(1024, 256 * 6 * 6),
            nn.SiLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, latent_features: torch.Tensor) -> torch.Tensor:
        hidden = self.fc(latent_features).view(latent_features.shape[0], 256, 6, 6)
        return self.deconv(hidden)


class SequenceModel(nn.Module):
    """GRU-based deterministic latent transition."""

    def __init__(self, stochastic_dim: int = 32, action_dim: int = 3, hidden_dim: int = 512):
        super().__init__()
        self.input_dim = stochastic_dim + action_dim
        self.hidden_dim = hidden_dim
        self.input_projection = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.SiLU(),
        )
        self.gru_cell = nn.GRUCell(hidden_dim, hidden_dim)

    def forward(
        self,
        prev_stochastic: torch.Tensor,
        prev_action: torch.Tensor,
        prev_hidden: torch.Tensor,
    ) -> torch.Tensor:
        gru_input = torch.cat([prev_stochastic, prev_action], dim=-1)
        return self.gru_cell(self.input_projection(gru_input), prev_hidden)


class _LatentHead(nn.Module):
    def __init__(self, input_dim: int, stochastic_dim: int = 32, hidden_dim: int = 512):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.mean = nn.Linear(hidden_dim, stochastic_dim)
        self.std_raw = nn.Linear(hidden_dim, stochastic_dim)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone(inputs)
        return self.mean(hidden), self.std_raw(hidden)


class DynamicsModel(nn.Module):
    def __init__(self, hidden_dim: int = 512, stochastic_dim: int = 32):
        super().__init__()
        self.head = _LatentHead(hidden_dim, stochastic_dim=stochastic_dim, hidden_dim=hidden_dim)

    def forward(self, deterministic: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.head(deterministic)


class PosteriorModel(nn.Module):
    def __init__(self, hidden_dim: int = 512, embedding_dim: int = 512, stochastic_dim: int = 32):
        super().__init__()
        self.head = _LatentHead(hidden_dim + embedding_dim, stochastic_dim=stochastic_dim, hidden_dim=hidden_dim)

    def forward(self, deterministic: torch.Tensor, embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.head(torch.cat([deterministic, embedding], dim=-1))


class RewardPredictor(nn.Module):
    def __init__(self, hidden_dim: int = 512, stochastic_dim: int = 32):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(hidden_dim + stochastic_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Linear(256, 1),
        )

    def forward(self, deterministic: torch.Tensor, stochastic: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat([deterministic, stochastic], dim=-1))


class RSSMCell(nn.Module):
    """Single-step observe and imagine logic."""

    def __init__(
        self,
        action_dim: int = 3,
        hidden_dim: int = 512,
        stochastic_dim: int = 32,
        embedding_dim: int = 512,
        min_std: float = 0.1,
        max_std: float = 2.0,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.stochastic_dim = stochastic_dim
        self.min_std = float(min_std)
        self.max_std = float(max_std)

        self.encoder = Encoder(embedding_dim=embedding_dim)
        self.sequence_model = SequenceModel(stochastic_dim=stochastic_dim, action_dim=action_dim, hidden_dim=hidden_dim)
        self.dynamics_model = DynamicsModel(hidden_dim=hidden_dim, stochastic_dim=stochastic_dim)
        self.posterior_model = PosteriorModel(
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            stochastic_dim=stochastic_dim,
        )
        self.reward_predictor = RewardPredictor(hidden_dim=hidden_dim, stochastic_dim=stochastic_dim)
        self.decoder = Decoder(input_dim=hidden_dim + stochastic_dim)

    def initial_state(self, batch_size: int, device: torch.device | str) -> RSSMState:
        return RSSMState(
            deterministic=torch.zeros(batch_size, self.hidden_dim, device=device),
            stochastic=torch.zeros(batch_size, self.stochastic_dim, device=device),
        )

    def _build_stats(self, mean: torch.Tensor, raw_std: torch.Tensor) -> LatentStats:
        std = F.softplus(raw_std) + self.min_std
        std = torch.clamp(std, max=self.max_std)
        return LatentStats(mean=mean, std=std, dist=Independent(Normal(mean, std), 1))

    def _decode_latent(self, deterministic: torch.Tensor, stochastic: torch.Tensor) -> torch.Tensor:
        return self.decoder(torch.cat([deterministic, stochastic], dim=-1))

    def observe_step(self, prev_state: RSSMState, prev_action: torch.Tensor, image_t: torch.Tensor) -> RSSMStepOutput:
        deterministic = self.sequence_model(prev_state.stochastic, prev_action, prev_state.deterministic)
        prior = self._build_stats(*self.dynamics_model(deterministic))
        embedding = self.encoder(image_t)
        posterior = self._build_stats(*self.posterior_model(deterministic, embedding))
        stochastic = posterior.dist.rsample()
        reward = self.reward_predictor(deterministic, stochastic)
        reconstruction = self._decode_latent(deterministic, stochastic)
        return RSSMStepOutput(
            state=RSSMState(deterministic=deterministic, stochastic=stochastic),
            prior=prior,
            posterior=posterior,
            reward=reward,
            reconstruction=reconstruction,
        )

    def imagine_step(self, prev_state: RSSMState, prev_action: torch.Tensor) -> RSSMStepOutput:
        deterministic = self.sequence_model(prev_state.stochastic, prev_action, prev_state.deterministic)
        prior = self._build_stats(*self.dynamics_model(deterministic))
        stochastic = prior.dist.rsample()
        reward = self.reward_predictor(deterministic, stochastic)
        reconstruction = self._decode_latent(deterministic, stochastic)
        return RSSMStepOutput(
            state=RSSMState(deterministic=deterministic, stochastic=stochastic),
            prior=prior,
            posterior=None,
            reward=reward,
            reconstruction=reconstruction,
        )

    def forward(self, prev_state: RSSMState, prev_action: torch.Tensor, image_t: torch.Tensor) -> RSSMStepOutput:
        return self.observe_step(prev_state, prev_action, image_t)


class RSSMSequence(nn.Module):
    """Batch-major recurrent wrapper that returns stacked outputs."""

    def __init__(
        self,
        action_dim: int = 3,
        hidden_dim: int = 512,
        stochastic_dim: int = 32,
        embedding_dim: int = 512,
        min_std: float = 0.1,
        max_std: float = 2.0,
    ):
        super().__init__()
        self.cell = RSSMCell(
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            stochastic_dim=stochastic_dim,
            embedding_dim=embedding_dim,
            min_std=min_std,
            max_std=max_std,
        )
        self.action_dim = action_dim

    def initial_state(self, batch_size: int, device: torch.device | str) -> RSSMState:
        return self.cell.initial_state(batch_size=batch_size, device=device)

    def forward(
        self,
        images: torch.Tensor,
        actions: torch.Tensor,
        initial_state: Optional[RSSMState] = None,
        is_first: Optional[torch.Tensor] = None,
    ) -> RSSMSequenceOutput:
        if images.ndim != 5:
            raise ValueError(f"Expected [B, T, C, H, W], got {tuple(images.shape)}")
        if actions.ndim != 3:
            raise ValueError(f"Expected [B, T, A], got {tuple(actions.shape)}")

        batch_size, time_steps = images.shape[:2]
        state = initial_state or self.initial_state(batch_size=batch_size, device=images.device)
        prev_action = torch.zeros(batch_size, self.action_dim, device=images.device, dtype=actions.dtype)

        deterministic_steps = []
        stochastic_steps = []
        prior_means = []
        prior_stds = []
        post_means = []
        post_stds = []
        rewards = []
        reconstructions = []

        for time_index in range(time_steps):
            if is_first is not None:
                reset_mask = is_first[:, time_index].to(dtype=torch.bool)
                if torch.any(reset_mask):
                    keep = (~reset_mask).to(dtype=state.deterministic.dtype).unsqueeze(-1)
                    state = RSSMState(
                        deterministic=state.deterministic * keep,
                        stochastic=state.stochastic * keep,
                    )
                    prev_action = prev_action * (~reset_mask).to(dtype=prev_action.dtype).unsqueeze(-1)

            step = self.cell.observe_step(state, prev_action, images[:, time_index])
            state = step.state
            prev_action = actions[:, time_index]

            deterministic_steps.append(state.deterministic)
            stochastic_steps.append(state.stochastic)
            prior_means.append(step.prior.mean)
            prior_stds.append(step.prior.std)
            post_means.append(step.posterior.mean)
            post_stds.append(step.posterior.std)
            rewards.append(step.reward)
            reconstructions.append(step.reconstruction)

        return RSSMSequenceOutput(
            deterministic=torch.stack(deterministic_steps, dim=1),
            stochastic=torch.stack(stochastic_steps, dim=1),
            prior_mean=torch.stack(prior_means, dim=1),
            prior_std=torch.stack(prior_stds, dim=1),
            posterior_mean=torch.stack(post_means, dim=1),
            posterior_std=torch.stack(post_stds, dim=1),
            reward=torch.stack(rewards, dim=1),
            reconstruction=torch.stack(reconstructions, dim=1),
        )
