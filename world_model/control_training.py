from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from train import create_env, load_config
from world_model.control import (
    FrozenWorldModel,
    LatentActor,
    LatentCritic,
    discounted_bootstrap_returns,
    imagine_with_actor,
)


def train_latent_actor_critic_epoch(
    world_model: FrozenWorldModel,
    actor: LatentActor,
    critic: LatentCritic,
    loader: DataLoader,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    device: str | torch.device,
    context_length: int,
    imagination_horizon: int,
    discount: float = 0.99,
    action_l2_scale: float = 0.0,
    log_every: int = 0,
) -> dict[str, float]:
    device = torch.device(device)
    non_blocking = device.type == "cuda"
    actor.train()
    critic.train()
    totals = {
        "actor_loss": 0.0,
        "critic_loss": 0.0,
        "imagined_reward_mean": 0.0,
        "imagined_return_mean": 0.0,
        "action_abs_mean": 0.0,
    }
    num_batches = 0

    for batch_index, batch in enumerate(loader, start=1):
        images = batch["images"][:, :context_length].to(device, non_blocking=non_blocking)
        actions = batch["actions"][:, :context_length].to(device, non_blocking=non_blocking)
        is_first = batch["is_first"][:, :context_length].to(device, non_blocking=non_blocking)

        initial_state = world_model.encode_context(images=images, actions=actions, is_first=is_first)
        rollout = imagine_with_actor(
            world_model=world_model,
            actor=actor,
            initial_state=initial_state,
            horizon=imagination_horizon,
        )

        rewards = torch.stack(rollout.rewards, dim=1)
        action_tensor = torch.stack(rollout.actions, dim=1)

        with torch.no_grad():
            critic_bootstrap = critic(rollout.final_feature.detach()).squeeze(-1)
            critic_targets = discounted_bootstrap_returns(rewards.detach(), critic_bootstrap, discount=discount)

        critic_values = torch.stack([critic(features.detach()).squeeze(-1) for features in rollout.features], dim=1)
        critic_loss = F.mse_loss(critic_values, critic_targets)

        actor_bootstrap = critic_bootstrap.detach()
        actor_returns = discounted_bootstrap_returns(rewards, actor_bootstrap, discount=discount)
        actor_loss = -actor_returns.mean()
        if float(action_l2_scale) > 0.0:
            actor_loss = actor_loss + (float(action_l2_scale) * action_tensor.square().mean())

        actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_optimizer.step()

        critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_optimizer.step()

        totals["actor_loss"] += float(actor_loss.item())
        totals["critic_loss"] += float(critic_loss.item())
        totals["imagined_reward_mean"] += float(rewards.mean().item())
        totals["imagined_return_mean"] += float(actor_returns.mean().item())
        totals["action_abs_mean"] += float(action_tensor.abs().mean().item())
        num_batches += 1

        if log_every > 0 and (batch_index == 1 or batch_index % log_every == 0):
            print(
                "[LatentControl][batch "
                f"{batch_index}/{len(loader)}] "
                f"actor_loss={actor_loss.item():.6f} "
                f"critic_loss={critic_loss.item():.6f} "
                f"imagined_reward={rewards.mean().item():.6f} "
                f"imagined_return={actor_returns.mean().item():.6f}",
                flush=True,
            )

    if num_batches == 0:
        raise ValueError("Latent-control replay loader produced zero batches.")
    return {key: value / num_batches for key, value in totals.items()}


def _obs_to_image_tensor(observation: np.ndarray, device: torch.device) -> torch.Tensor:
    if observation.ndim != 3:
        raise ValueError(f"Expected HWC image observation, got shape {tuple(observation.shape)}")
    return torch.from_numpy(observation).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0


def _unwrap_reset(reset_output):
    if isinstance(reset_output, tuple):
        return reset_output[0]
    return reset_output


def _unwrap_step(step_output):
    if len(step_output) == 5:
        next_obs, reward, terminated, truncated, info = step_output
        done = bool(terminated or truncated)
        if not isinstance(info, dict):
            info = {}
        info.setdefault("TimeLimit.truncated", bool(truncated))
        return next_obs, float(reward), done, info
    if len(step_output) == 4:
        next_obs, reward, done, info = step_output
        if not isinstance(info, dict):
            info = {}
        return next_obs, float(reward), bool(done), info
    raise ValueError(f"Unexpected env.step output length: {len(step_output)}")


def evaluate_latent_actor(
    world_model: FrozenWorldModel,
    actor: LatentActor,
    base_config_path: str | Path,
    device: str | torch.device,
    episodes: int = 3,
    max_steps: int = 1000,
    seed: int = 0,
) -> dict[str, float]:
    device = torch.device(device)
    config = load_config(str(base_config_path))
    env = create_env(config, rank=0, seed=int(seed))
    actor.eval()
    episode_rewards: list[float] = []
    episode_lengths: list[int] = []
    episode_progress: list[float] = []
    episode_offtrack: list[int] = []

    try:
        for episode_index in range(int(episodes)):
            observation = _unwrap_reset(env.reset())
            state = world_model.initial_state(batch_size=1, device=device)
            prev_action = torch.zeros(1, world_model.action_dim, device=device)
            done = False
            total_reward = 0.0
            episode_length = 0
            offtrack_seen = False
            final_info: dict[str, float] = {}

            while not done and episode_length < int(max_steps):
                image_t = _obs_to_image_tensor(np.asarray(observation, dtype=np.uint8), device=device)
                with torch.no_grad():
                    posterior_step = world_model.model.cell.observe_step(
                        prev_state=state,
                        prev_action=prev_action,
                        image_t=image_t,
                    )
                    state = posterior_step.state
                    action_tensor = actor(world_model.flatten_state(state))
                next_observation, reward, done, info = _unwrap_step(env.step(action_tensor[0].cpu().numpy().astype(np.float32)))
                total_reward += float(reward)
                episode_length += 1
                if int(info.get("events/offtrack", 0)) > 0:
                    offtrack_seen = True
                final_info = info
                prev_action = action_tensor.detach()
                observation = next_observation

            episode_rewards.append(total_reward)
            episode_lengths.append(episode_length)
            episode_progress.append(float(final_info.get("progress", 0.0)))
            episode_offtrack.append(int(offtrack_seen))
    finally:
        env.close()

    return {
        "mean_reward": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
        "mean_length": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
        "mean_progress": float(np.mean(episode_progress)) if episode_progress else 0.0,
        "offtrack_rate": float(np.mean(episode_offtrack)) if episode_offtrack else 0.0,
        "episodes": int(episodes),
    }


def save_control_metrics(path: str | Path, payload: dict[str, object]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
