from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from train import TORCH_POLICY_VARIANTS, CnnActorCritic, create_env, load_config
from world_model.replay import EpisodeReplay, ReplayWriter


@dataclass
class DrunkExpertConfig:
    steer_std: float = 0.60
    throttle_std: float = 0.35
    brake_std: float = 0.15

    def as_array(self) -> np.ndarray:
        return np.asarray([self.steer_std, self.throttle_std, self.brake_std], dtype=np.float32)


def apply_drunk_expert_noise(
    action_mean_raw: np.ndarray,
    noise_std: np.ndarray,
    rng: np.random.Generator,
    action_low: np.ndarray,
    action_high: np.ndarray,
    raw_to_env_action,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    noise = rng.normal(loc=0.0, scale=noise_std, size=action_mean_raw.shape).astype(np.float32)
    noisy_raw = action_mean_raw.astype(np.float32) + noise
    env_action = raw_to_env_action(torch.as_tensor(noisy_raw, dtype=torch.float32)).cpu().numpy()
    env_action = np.clip(env_action, action_low, action_high).astype(np.float32)
    return noisy_raw.astype(np.float32), noise, env_action


class TorchPolicyAdapter:
    """Loads the current torch PPO policy and exposes deterministic raw means."""

    def __init__(self, config: dict[str, Any], checkpoint_path: str | Path, device: str | torch.device):
        self.config = config
        self.device = torch.device(device)
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        probe_env = create_env(config, rank=0, seed=0)
        obs_shape = tuple(probe_env.observation_space.shape)
        obs_shape = (obs_shape[2], obs_shape[0], obs_shape[1])
        action_dim = int(np.prod(probe_env.action_space.shape))
        self.action_low = np.array(probe_env.action_space.low, dtype=np.float32)
        self.action_high = np.array(probe_env.action_space.high, dtype=np.float32)
        probe_env.close()

        training_cfg = config.get("training", {}) or {}
        policy_variant = str(training_cfg.get("torch_policy_variant", "legacy")).strip().lower()
        policy_cls = TORCH_POLICY_VARIANTS.get(policy_variant, CnnActorCritic)
        ppo_cfg = config.get("ppo", {}) or {}
        self.policy = policy_cls(
            obs_shape,
            action_dim,
            min_log_std=float(ppo_cfg.get("min_log_std", -1.5)),
            max_log_std=float(ppo_cfg.get("max_log_std", 1.0)),
            steer_min_log_std=float(ppo_cfg["steer_min_log_std"]) if ppo_cfg.get("steer_min_log_std") is not None else None,
            steer_max_log_std=float(ppo_cfg["steer_max_log_std"]) if ppo_cfg.get("steer_max_log_std") is not None else None,
        )
        payload = torch.load(str(self.checkpoint_path), map_location="cpu")
        self.policy.load_state_dict(payload["policy_state_dict"])
        self.policy.to(self.device)
        self.policy.eval()
        self.policy_cls = policy_cls

    def raw_action_mean(self, observation_hwc_uint8: np.ndarray) -> np.ndarray:
        obs_tensor = torch.as_tensor(observation_hwc_uint8, dtype=torch.float32, device=self.device) / 255.0
        obs_tensor = obs_tensor.permute(2, 0, 1).unsqueeze(0)
        with torch.no_grad():
            if hasattr(self.policy, "get_raw_dist_and_value"):
                dist, _ = self.policy.get_raw_dist_and_value(obs_tensor)
            else:
                dist, _ = self.policy.get_dist_and_value(obs_tensor)
        return dist.mean.squeeze(0).detach().cpu().numpy().astype(np.float32)


def collect_drunk_expert_dataset(
    base_config_path: str | Path,
    output_dir: str | Path,
    checkpoint_path: str | Path,
    split: str = "train",
    target_frames: int = 10000,
    seed: int = 42,
    noise_cfg: DrunkExpertConfig | None = None,
) -> list[Path]:
    config = load_config(base_config_path)
    rng = np.random.default_rng(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = TorchPolicyAdapter(config=config, checkpoint_path=checkpoint_path, device=device)
    env = create_env(config, rank=0, seed=seed)
    writer = ReplayWriter(output_dir, split=split)
    noise_cfg = noise_cfg or DrunkExpertConfig()
    noise_std = noise_cfg.as_array()

    saved_paths: list[Path] = []
    frames_collected = 0
    episode_id = 0
    try:
        while frames_collected < target_frames:
            obs = env.reset()
            done = False
            observations = []
            actions = []
            rewards = []
            dones = []
            truncated = []
            action_mean_raw = []
            action_noisy_raw = []
            noise_values = []

            while not done and frames_collected < target_frames:
                mean_raw = policy.raw_action_mean(obs)
                noisy_raw, noise, env_action = apply_drunk_expert_noise(
                    action_mean_raw=mean_raw,
                    noise_std=noise_std,
                    rng=rng,
                    action_low=policy.action_low,
                    action_high=policy.action_high,
                    raw_to_env_action=policy.policy_cls.raw_to_env_action,
                )
                next_obs, reward, done, info = env.step(env_action)

                info = info if isinstance(info, dict) else {}
                observations.append(obs.astype(np.uint8))
                actions.append(env_action.astype(np.float32))
                rewards.append(np.float32(reward))
                dones.append(bool(done))
                truncated.append(bool(info.get("TimeLimit.truncated", False)))
                action_mean_raw.append(mean_raw)
                action_noisy_raw.append(noisy_raw)
                noise_values.append(noise)
                frames_collected += 1
                obs = next_obs

            episode = EpisodeReplay(
                observations_uint8=np.asarray(observations, dtype=np.uint8),
                actions=np.asarray(actions, dtype=np.float32),
                rewards=np.asarray(rewards, dtype=np.float32),
                dones=np.asarray(dones, dtype=np.bool_),
                truncated=np.asarray(truncated, dtype=np.bool_),
                action_mean_raw=np.asarray(action_mean_raw, dtype=np.float32),
                action_noisy_raw=np.asarray(action_noisy_raw, dtype=np.float32),
                noise=np.asarray(noise_values, dtype=np.float32),
                metadata={
                    "episode_id": episode_id,
                    "seed": seed,
                    "base_config_path": str(base_config_path),
                    "checkpoint_path": str(checkpoint_path),
                    "noise_std": noise_std.tolist(),
                    "num_steps": len(observations),
                },
            )
            saved_paths.append(writer.save_episode(episode=episode, episode_id=episode_id))
            episode_id += 1
    finally:
        env.close()

    return saved_paths


def save_collection_manifest(output_dir: str | Path, split: str, episode_paths: list[Path]) -> Path:
    manifest_path = Path(output_dir) / f"{split}_manifest.json"
    manifest_path.write_text(
        json.dumps({"split": split, "episodes": [str(path) for path in episode_paths]}, indent=2),
        encoding="utf-8",
    )
    return manifest_path
