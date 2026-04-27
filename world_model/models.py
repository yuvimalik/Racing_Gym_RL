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
    telemetry: dict[str, torch.Tensor]
    ego_motion: dict[str, torch.Tensor]


@dataclass
class RSSMSequenceOutput:
    deterministic: torch.Tensor
    stochastic: torch.Tensor
    stochastic_world: torch.Tensor
    stochastic_motion: torch.Tensor
    prior_mean: torch.Tensor
    prior_std: torch.Tensor
    posterior_mean: torch.Tensor
    posterior_std: torch.Tensor
    reward: torch.Tensor
    reconstruction: torch.Tensor
    telemetry: dict[str, torch.Tensor]
    ego_motion: dict[str, torch.Tensor]


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
    """GRU-based deterministic latent transition.

    LayerNorm on GRU output:
    Without normalisation, the GRU hidden state h_t can grow in magnitude over
    long rollouts. During backpropagation through many time steps (BPTT), large
    hidden state values produce large gradients that either explode (NaN/Inf) or
    — after gradient clipping — vanish, preventing the model from learning from
    distant past frames.

    LayerNorm rescales h_t to mean=0, std=1 before it is passed to the prior and
    posterior heads. This keeps the hidden state magnitude bounded regardless of
    rollout length. DreamerV3 (Hafner et al. 2023) uses the same technique and
    attributes much of its long-horizon stability to it.

    Benefit: stable gradients when sequence_length > 100; hallucinations stay
    coherent at 150–200 steps instead of drifting after 50.
    """

    def __init__(self, stochastic_dim: int = 32, action_dim: int = 3, hidden_dim: int = 512):
        super().__init__()
        self.input_dim = stochastic_dim + action_dim
        self.hidden_dim = hidden_dim
        self.input_projection = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.SiLU(),
        )
        self.gru_cell = nn.GRUCell(hidden_dim, hidden_dim)
        # LayerNorm applied to h_t after every GRU step.
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        prev_stochastic: torch.Tensor,
        prev_action: torch.Tensor,
        prev_hidden: torch.Tensor,
    ) -> torch.Tensor:
        gru_input = torch.cat([prev_stochastic, prev_action], dim=-1)
        h_t = self.gru_cell(self.input_projection(gru_input), prev_hidden)
        return self.layer_norm(h_t)


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
    def __init__(self, hidden_dim: int = 512, stochastic_world_dim: int = 32, stochastic_motion_dim: int = 0):
        super().__init__()
        self.world_head = _LatentHead(hidden_dim, stochastic_dim=stochastic_world_dim, hidden_dim=hidden_dim)
        self.motion_head = (
            _LatentHead(hidden_dim, stochastic_dim=stochastic_motion_dim, hidden_dim=hidden_dim)
            if stochastic_motion_dim > 0 else None
        )

    def forward(self, deterministic: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        world_mean, world_std_raw = self.world_head(deterministic)
        if self.motion_head is None:
            return world_mean, world_std_raw
        motion_mean, motion_std_raw = self.motion_head(deterministic)
        return (
            torch.cat([world_mean, motion_mean], dim=-1),
            torch.cat([world_std_raw, motion_std_raw], dim=-1),
        )


class PosteriorModel(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 512,
        embedding_dim: int = 512,
        stochastic_world_dim: int = 32,
        stochastic_motion_dim: int = 0,
    ):
        super().__init__()
        input_dim = hidden_dim + embedding_dim
        self.world_head = _LatentHead(input_dim, stochastic_dim=stochastic_world_dim, hidden_dim=hidden_dim)
        self.motion_head = (
            _LatentHead(input_dim, stochastic_dim=stochastic_motion_dim, hidden_dim=hidden_dim)
            if stochastic_motion_dim > 0 else None
        )

    def forward(self, deterministic: torch.Tensor, embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = torch.cat([deterministic, embedding], dim=-1)
        world_mean, world_std_raw = self.world_head(inputs)
        if self.motion_head is None:
            return world_mean, world_std_raw
        motion_mean, motion_std_raw = self.motion_head(inputs)
        return (
            torch.cat([world_mean, motion_mean], dim=-1),
            torch.cat([world_std_raw, motion_std_raw], dim=-1),
        )


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


class TelemetryPredictor(nn.Module):
    def __init__(self, hidden_dim: int = 512, stochastic_dim: int = 32):
        super().__init__()
        input_dim = hidden_dim + stochastic_dim
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 256),
            nn.SiLU(),
        )
        self.speed_head = nn.Linear(256, 1)
        self.progress_delta_head = nn.Linear(256, 1)
        self.steer_head = nn.Linear(256, 1)
        self.corner_angle_head = nn.Linear(256, 1)
        self.offtrack_head = nn.Linear(256, 1)

    def forward(self, deterministic: torch.Tensor, stochastic: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.trunk(torch.cat([deterministic, stochastic], dim=-1))
        return {
            "speed": self.speed_head(hidden),
            "progress_delta": self.progress_delta_head(hidden),
            "steer": self.steer_head(hidden),
            "corner_angle": self.corner_angle_head(hidden),
            "offtrack_logits": self.offtrack_head(hidden),
        }


class EgoMotionHead(nn.Module):
    def __init__(self, hidden_dim: int = 512, motion_dim: int = 0):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(hidden_dim + motion_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
        )
        self.speed_head = nn.Linear(128, 1)
        self.steer_head = nn.Linear(128, 1)
        self.progress_delta_head = nn.Linear(128, 1)

    def forward(self, deterministic: torch.Tensor, motion_latent: Optional[torch.Tensor] = None) -> dict[str, torch.Tensor]:
        inputs = deterministic if motion_latent is None else torch.cat([deterministic, motion_latent], dim=-1)
        hidden = self.trunk(inputs)
        return {
            "speed": self.speed_head(hidden),
            "steer": self.steer_head(hidden),
            "progress_delta": self.progress_delta_head(hidden),
        }


class RSSMCell(nn.Module):
    """Single-step observe and imagine logic."""

    def __init__(
        self,
        action_dim: int = 3,
        hidden_dim: int = 512,
        stochastic_dim: int = 32,
        stochastic_world_dim: Optional[int] = None,
        stochastic_motion_dim: int = 0,
        embedding_dim: int = 512,
        min_std: float = 0.1,
        max_std: float = 2.0,
        ego_motion_head_enabled: bool = False,
        ego_motion_decoder_conditioning: bool = False,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.stochastic_world_dim = int(stochastic_world_dim if stochastic_world_dim is not None else stochastic_dim)
        self.stochastic_motion_dim = int(stochastic_motion_dim)
        self.stochastic_dim = self.stochastic_world_dim + self.stochastic_motion_dim
        self.min_std = float(min_std)
        self.max_std = float(max_std)
        self.ego_motion_head_enabled = bool(ego_motion_head_enabled)
        self.ego_motion_decoder_conditioning = bool(ego_motion_decoder_conditioning and ego_motion_head_enabled)
        self.ego_motion_dim = 3 if self.ego_motion_head_enabled else 0

        self.encoder = Encoder(embedding_dim=embedding_dim)
        self.sequence_model = SequenceModel(stochastic_dim=self.stochastic_dim, action_dim=action_dim, hidden_dim=hidden_dim)
        self.dynamics_model = DynamicsModel(
            hidden_dim=hidden_dim,
            stochastic_world_dim=self.stochastic_world_dim,
            stochastic_motion_dim=self.stochastic_motion_dim,
        )
        self.posterior_model = PosteriorModel(
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            stochastic_world_dim=self.stochastic_world_dim,
            stochastic_motion_dim=self.stochastic_motion_dim,
        )
        self.reward_predictor = RewardPredictor(hidden_dim=hidden_dim, stochastic_dim=self.stochastic_dim)
        self.telemetry_predictor = TelemetryPredictor(hidden_dim=hidden_dim, stochastic_dim=self.stochastic_dim)
        self.ego_motion_head = (
            EgoMotionHead(hidden_dim=hidden_dim, motion_dim=self.stochastic_motion_dim) if self.ego_motion_head_enabled else None
        )
        decoder_input_dim = hidden_dim + self.stochastic_dim + (self.ego_motion_dim if self.ego_motion_decoder_conditioning else 0)
        self.decoder = Decoder(input_dim=decoder_input_dim)

    def initial_state(self, batch_size: int, device: torch.device | str) -> RSSMState:
        return RSSMState(
            deterministic=torch.zeros(batch_size, self.hidden_dim, device=device),
            stochastic=torch.zeros(batch_size, self.stochastic_dim, device=device),
        )

    def _build_stats(self, mean: torch.Tensor, raw_std: torch.Tensor) -> LatentStats:
        std = F.softplus(raw_std) + self.min_std
        std = torch.clamp(std, max=self.max_std)
        return LatentStats(mean=mean, std=std, dist=Independent(Normal(mean, std), 1))

    def _decode_latent(
        self,
        deterministic: torch.Tensor,
        stochastic: torch.Tensor,
        stochastic_world: Optional[torch.Tensor] = None,
        stochastic_motion: Optional[torch.Tensor] = None,
        ego_motion: Optional[dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        if stochastic_world is None or stochastic_motion is None:
            latent_features = [deterministic, stochastic]
        else:
            latent_features = [deterministic, stochastic_world, stochastic_motion]
        if self.ego_motion_decoder_conditioning:
            if ego_motion is None:
                raise ValueError("ego_motion must be provided when ego-motion decoder conditioning is enabled.")
            latent_features.extend([ego_motion["speed"], ego_motion["steer"], ego_motion["progress_delta"]])
        return self.decoder(torch.cat(latent_features, dim=-1))

    def _split_stochastic(self, stochastic: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        stochastic_world = stochastic[..., : self.stochastic_world_dim]
        stochastic_motion = stochastic[..., self.stochastic_world_dim :]
        return stochastic_world, stochastic_motion

    def _predict_ego_motion(self, deterministic: torch.Tensor, motion_latent: Optional[torch.Tensor] = None) -> dict[str, torch.Tensor]:
        if self.ego_motion_head is None:
            zero = deterministic.new_zeros(deterministic.shape[0], 1)
            return {
                "speed": zero,
                "steer": zero,
                "progress_delta": zero,
            }
        return self.ego_motion_head(deterministic, motion_latent=motion_latent)

    def observe_step(self, prev_state: RSSMState, prev_action: torch.Tensor, image_t: torch.Tensor) -> RSSMStepOutput:
        deterministic = self.sequence_model(prev_state.stochastic, prev_action, prev_state.deterministic)
        prior = self._build_stats(*self.dynamics_model(deterministic))
        embedding = self.encoder(image_t)
        posterior = self._build_stats(*self.posterior_model(deterministic, embedding))
        stochastic = posterior.dist.rsample()
        stochastic_world, stochastic_motion = self._split_stochastic(stochastic)
        reward = self.reward_predictor(deterministic, stochastic)
        telemetry = self.telemetry_predictor(deterministic, stochastic)
        ego_motion = self._predict_ego_motion(deterministic, motion_latent=stochastic_motion)
        reconstruction = self._decode_latent(
            deterministic,
            stochastic,
            stochastic_world=stochastic_world,
            stochastic_motion=stochastic_motion,
            ego_motion=ego_motion,
        )
        return RSSMStepOutput(
            state=RSSMState(deterministic=deterministic, stochastic=stochastic),
            prior=prior,
            posterior=posterior,
            reward=reward,
            reconstruction=reconstruction,
            telemetry=telemetry,
            ego_motion=ego_motion,
        )

    def imagine_step(self, prev_state: RSSMState, prev_action: torch.Tensor) -> RSSMStepOutput:
        deterministic = self.sequence_model(prev_state.stochastic, prev_action, prev_state.deterministic)
        prior = self._build_stats(*self.dynamics_model(deterministic))
        stochastic = prior.dist.rsample()
        stochastic_world, stochastic_motion = self._split_stochastic(stochastic)
        reward = self.reward_predictor(deterministic, stochastic)
        telemetry = self.telemetry_predictor(deterministic, stochastic)
        ego_motion = self._predict_ego_motion(deterministic, motion_latent=stochastic_motion)
        reconstruction = self._decode_latent(
            deterministic,
            stochastic,
            stochastic_world=stochastic_world,
            stochastic_motion=stochastic_motion,
            ego_motion=ego_motion,
        )
        return RSSMStepOutput(
            state=RSSMState(deterministic=deterministic, stochastic=stochastic),
            prior=prior,
            posterior=None,
            reward=reward,
            reconstruction=reconstruction,
            telemetry=telemetry,
            ego_motion=ego_motion,
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
        stochastic_world_dim: Optional[int] = None,
        stochastic_motion_dim: int = 0,
        embedding_dim: int = 512,
        min_std: float = 0.1,
        max_std: float = 2.0,
        ego_motion_head_enabled: bool = False,
        ego_motion_decoder_conditioning: bool = False,
    ):
        super().__init__()
        self.cell = RSSMCell(
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            stochastic_dim=stochastic_dim,
            stochastic_world_dim=stochastic_world_dim,
            stochastic_motion_dim=stochastic_motion_dim,
            embedding_dim=embedding_dim,
            min_std=min_std,
            max_std=max_std,
            ego_motion_head_enabled=ego_motion_head_enabled,
            ego_motion_decoder_conditioning=ego_motion_decoder_conditioning,
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
        stochastic_world_steps = []
        stochastic_motion_steps = []
        prior_means = []
        prior_stds = []
        post_means = []
        post_stds = []
        rewards = []
        reconstructions = []
        telemetry_steps = {
            "speed": [],
            "progress_delta": [],
            "steer": [],
            "corner_angle": [],
            "offtrack_logits": [],
        }
        ego_motion_steps = {
            "speed": [],
            "steer": [],
            "progress_delta": [],
        }

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
            stochastic_world, stochastic_motion = self.cell._split_stochastic(state.stochastic)
            stochastic_world_steps.append(stochastic_world)
            stochastic_motion_steps.append(stochastic_motion)
            prior_means.append(step.prior.mean)
            prior_stds.append(step.prior.std)
            post_means.append(step.posterior.mean)
            post_stds.append(step.posterior.std)
            rewards.append(step.reward)
            reconstructions.append(step.reconstruction)
            for key in telemetry_steps:
                telemetry_steps[key].append(step.telemetry[key])
            for key in ego_motion_steps:
                ego_motion_steps[key].append(step.ego_motion[key])

        return RSSMSequenceOutput(
            deterministic=torch.stack(deterministic_steps, dim=1),
            stochastic=torch.stack(stochastic_steps, dim=1),
            stochastic_world=torch.stack(stochastic_world_steps, dim=1),
            stochastic_motion=torch.stack(stochastic_motion_steps, dim=1),
            prior_mean=torch.stack(prior_means, dim=1),
            prior_std=torch.stack(prior_stds, dim=1),
            posterior_mean=torch.stack(post_means, dim=1),
            posterior_std=torch.stack(post_stds, dim=1),
            reward=torch.stack(rewards, dim=1),
            reconstruction=torch.stack(reconstructions, dim=1),
            telemetry={key: torch.stack(values, dim=1) for key, values in telemetry_steps.items()},
            ego_motion={key: torch.stack(values, dim=1) for key, values in ego_motion_steps.items()},
        )
