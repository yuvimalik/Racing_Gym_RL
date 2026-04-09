from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from world_model.models import RSSMSequence, RSSMState


@dataclass
class ImaginedRollout:
    features: list[torch.Tensor]
    actions: list[torch.Tensor]
    rewards: list[torch.Tensor]
    final_state: RSSMState
    final_feature: torch.Tensor


class FrozenWorldModel(nn.Module):
    """Thin control-facing wrapper around a frozen RSSM."""

    def __init__(self, model: RSSMSequence):
        super().__init__()
        self.model = model
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @property
    def action_dim(self) -> int:
        return int(self.model.action_dim)

    @property
    def latent_dim(self) -> int:
        return int(self.model.cell.hidden_dim + self.model.cell.stochastic_dim)

    def initial_state(self, batch_size: int, device: torch.device | str) -> RSSMState:
        return self.model.initial_state(batch_size=batch_size, device=device)

    @staticmethod
    def flatten_state(state: RSSMState) -> torch.Tensor:
        return torch.cat([state.deterministic, state.stochastic], dim=-1)

    def encode_context(
        self,
        images: torch.Tensor,
        actions: torch.Tensor,
        is_first: torch.Tensor | None = None,
        initial_state: RSSMState | None = None,
    ) -> RSSMState:
        if images.ndim != 5:
            raise ValueError(f"Expected images with shape [B, T, C, H, W], got {tuple(images.shape)}")
        if actions.ndim != 3:
            raise ValueError(f"Expected actions with shape [B, T, A], got {tuple(actions.shape)}")
        if images.shape[:2] != actions.shape[:2]:
            raise ValueError("Images and actions must share batch/time dimensions.")

        batch_size, time_steps = images.shape[:2]
        state = initial_state or self.initial_state(batch_size=batch_size, device=images.device)
        prev_action = torch.zeros(batch_size, self.action_dim, device=images.device, dtype=actions.dtype)

        with torch.no_grad():
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

                step = self.model.cell.observe_step(
                    prev_state=state,
                    prev_action=prev_action,
                    image_t=images[:, time_index],
                )
                state = step.state
                prev_action = actions[:, time_index]

        return state

    def imagine_step(self, state: RSSMState, action: torch.Tensor):
        return self.model.cell.imagine_step(prev_state=state, prev_action=action)


class LatentActor(nn.Module):
    """Simple deterministic policy over RSSM latent state."""

    def __init__(self, latent_dim: int, action_dim: int = 3, hidden_dim: int = 256):
        super().__init__()
        self.action_dim = int(action_dim)
        self.trunk = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.action_head = nn.Linear(hidden_dim, action_dim)
        if self.action_dim >= 3:
            nn.init.constant_(self.action_head.bias[1], 1.0)
            nn.init.constant_(self.action_head.bias[2], -2.0)

    @staticmethod
    def raw_to_env_action(raw_action: torch.Tensor) -> torch.Tensor:
        out = raw_action.clone()
        if out.shape[-1] >= 1:
            out[..., 0] = torch.tanh(out[..., 0])
        if out.shape[-1] >= 2:
            out[..., 1] = torch.sigmoid(out[..., 1])
        if out.shape[-1] >= 3:
            out[..., 2] = torch.sigmoid(out[..., 2])
        return out

    def forward(self, latent_features: torch.Tensor) -> torch.Tensor:
        raw_action = self.action_head(self.trunk(latent_features))
        return self.raw_to_env_action(raw_action)


class LatentCritic(nn.Module):
    """Value head over flattened RSSM latent state."""

    def __init__(self, latent_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, latent_features: torch.Tensor) -> torch.Tensor:
        return self.network(latent_features)


def imagine_with_actor(
    world_model: FrozenWorldModel,
    actor: LatentActor,
    initial_state: RSSMState,
    horizon: int,
) -> ImaginedRollout:
    state = initial_state
    features: list[torch.Tensor] = []
    actions: list[torch.Tensor] = []
    rewards: list[torch.Tensor] = []

    for _ in range(int(horizon)):
        feature = world_model.flatten_state(state)
        action = actor(feature)
        step = world_model.imagine_step(state, action)
        features.append(feature)
        actions.append(action)
        rewards.append(step.reward.squeeze(-1))
        state = step.state

    final_feature = world_model.flatten_state(state)
    return ImaginedRollout(
        features=features,
        actions=actions,
        rewards=rewards,
        final_state=state,
        final_feature=final_feature,
    )


def discounted_bootstrap_returns(
    rewards_bt: torch.Tensor,
    bootstrap_b: torch.Tensor,
    discount: float,
) -> torch.Tensor:
    if rewards_bt.ndim != 2:
        raise ValueError(f"Expected rewards with shape [B, T], got {tuple(rewards_bt.shape)}")
    returns = torch.zeros_like(rewards_bt)
    next_return = bootstrap_b
    for time_index in range(rewards_bt.shape[1] - 1, -1, -1):
        next_return = rewards_bt[:, time_index] + (float(discount) * next_return)
        returns[:, time_index] = next_return
    return returns
