from __future__ import annotations

import json
from pathlib import Path

import cv2
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
from world_model.training import save_video


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


def _render_overlay(frame_rgb: np.ndarray, lines: list[str]) -> np.ndarray:
    overlay = frame_rgb.copy()
    y = 18
    for line in lines:
        cv2.putText(
            overlay,
            line,
            (8, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            line,
            (8, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
        y += 18
    return overlay


def _build_compare_frames(
    world_model: FrozenWorldModel,
    observations_uint8: np.ndarray,
    actions: np.ndarray,
    device: torch.device,
    context_length: int,
    horizon: int,
) -> np.ndarray:
    total_length = int(context_length) + int(horizon)
    if observations_uint8.shape[0] < total_length or actions.shape[0] < total_length:
        raise ValueError("Need at least context_length + horizon real steps to build compare frames.")

    images = torch.from_numpy(observations_uint8[:total_length]).permute(0, 3, 1, 2).float().unsqueeze(0).to(device) / 255.0
    action_tensor = torch.from_numpy(actions[:total_length]).float().unsqueeze(0).to(device)
    imagined_frames = []
    state = world_model.encode_context(images=images[:, :context_length], actions=action_tensor[:, :context_length])
    for time_index in range(context_length, total_length):
        step = world_model.imagine_step(state, action_tensor[:, time_index])
        imagined_frames.append(step.reconstruction[0].detach().cpu().permute(1, 2, 0).numpy())
        state = step.state

    real_frames = images[0, context_length:total_length].detach().cpu().permute(0, 2, 3, 1).numpy()
    real_uint8 = np.clip(real_frames * 255.0, 0.0, 255.0).astype(np.uint8)
    imagined_uint8 = np.clip(np.stack(imagined_frames, axis=0) * 255.0, 0.0, 255.0).astype(np.uint8)
    return np.concatenate([real_uint8, imagined_uint8], axis=2)


def save_latent_actor_compare_video(
    world_model: FrozenWorldModel,
    observations_uint8: np.ndarray,
    actions: np.ndarray,
    output_path: str | Path,
    device: str | torch.device,
    context_length: int,
    horizon: int,
    fps: float = 10.0,
) -> Path:
    compare_frames = _build_compare_frames(
        world_model=world_model,
        observations_uint8=observations_uint8,
        actions=actions,
        device=torch.device(device),
        context_length=context_length,
        horizon=horizon,
    )
    return save_video(compare_frames, output_path, fps=fps)


def evaluate_latent_actor(
    world_model: FrozenWorldModel,
    actor: LatentActor,
    base_config_path: str | Path,
    device: str | torch.device,
    episodes: int = 3,
    max_steps: int = 1000,
    seed: int = 0,
    record_video: bool = False,
    video_path: str | Path | None = None,
    record_compare_video: bool = False,
    compare_video_path: str | Path | None = None,
    video_fps: float = 10.0,
    compare_context_length: int | None = None,
    compare_horizon: int = 50,
    overlay: bool = True,
) -> dict[str, object]:
    device = torch.device(device)
    config = load_config(str(base_config_path))
    env = create_env(config, rank=0, seed=int(seed))
    actor.eval()
    episode_rewards: list[float] = []
    episode_lengths: list[int] = []
    episode_progress: list[float] = []
    episode_offtrack: list[int] = []
    saved_video_path: Path | None = Path(video_path) if record_video and video_path is not None else None
    saved_compare_path: Path | None = Path(compare_video_path) if record_compare_video and compare_video_path is not None else None
    video_frames: list[np.ndarray] = []
    compare_observations: list[np.ndarray] = []
    compare_actions: list[np.ndarray] = []
    compare_length = int(compare_context_length or 0) + int(compare_horizon)

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
                observation_uint8 = np.asarray(observation, dtype=np.uint8)
                image_t = _obs_to_image_tensor(np.asarray(observation, dtype=np.uint8), device=device)
                with torch.no_grad():
                    posterior_step = world_model.model.cell.observe_step(
                        prev_state=state,
                        prev_action=prev_action,
                        image_t=image_t,
                    )
                    state = posterior_step.state
                    action_tensor = actor(world_model.flatten_state(state))
                action_np = action_tensor[0].cpu().numpy().astype(np.float32)
                next_observation, reward, done, info = _unwrap_step(env.step(action_np))
                total_reward += float(reward)
                episode_length += 1
                if int(info.get("events/offtrack", 0)) > 0:
                    offtrack_seen = True
                final_info = info
                prev_action = action_tensor.detach()
                if record_compare_video and episode_index == 0 and len(compare_observations) < compare_length:
                    compare_observations.append(observation_uint8.copy())
                    compare_actions.append(action_np.copy())
                if record_video:
                    frame = env.render(mode="rgb_array")
                    if frame is not None:
                        frame_uint8 = np.asarray(frame, dtype=np.uint8)
                        if overlay:
                            frame_uint8 = _render_overlay(
                                frame_uint8,
                                [
                                    f"episode={episode_index + 1}/{int(episodes)} step={episode_length}",
                                    f"reward={reward:.3f} total={total_reward:.2f}",
                                    f"progress={float(info.get('progress', 0.0)):.2%} offtrack={int(offtrack_seen)}",
                                    f"action=[{action_np[0]:+.2f}, {action_np[1]:.2f}, {action_np[2]:.2f}]",
                                ],
                            )
                        video_frames.append(frame_uint8)
                observation = next_observation

            episode_rewards.append(total_reward)
            episode_lengths.append(episode_length)
            episode_progress.append(float(final_info.get("progress", 0.0)))
            episode_offtrack.append(int(offtrack_seen))
    finally:
        env.close()

    if record_video and saved_video_path is not None and video_frames:
        save_video(np.stack(video_frames, axis=0), saved_video_path, fps=video_fps)

    if (
        record_compare_video
        and saved_compare_path is not None
        and compare_context_length is not None
        and len(compare_observations) >= (int(compare_context_length) + int(compare_horizon))
        and len(compare_actions) >= (int(compare_context_length) + int(compare_horizon))
    ):
        save_latent_actor_compare_video(
            world_model=world_model,
            observations_uint8=np.stack(compare_observations, axis=0),
            actions=np.stack(compare_actions, axis=0),
            output_path=saved_compare_path,
            device=device,
            context_length=int(compare_context_length),
            horizon=int(compare_horizon),
            fps=video_fps,
        )

    return {
        "mean_reward": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
        "mean_length": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
        "mean_progress": float(np.mean(episode_progress)) if episode_progress else 0.0,
        "offtrack_rate": float(np.mean(episode_offtrack)) if episode_offtrack else 0.0,
        "episodes": int(episodes),
        "artifact_paths": {
            "real_video": str(saved_video_path) if record_video and saved_video_path is not None and saved_video_path.exists() else None,
            "compare_video": str(saved_compare_path)
            if record_compare_video and saved_compare_path is not None and saved_compare_path.exists()
            else None,
        },
    }


def save_control_metrics(path: str | Path, payload: dict[str, object]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
