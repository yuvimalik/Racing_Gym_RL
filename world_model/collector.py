from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from train import create_env, load_config
from world_model.replay import EpisodeReplay, ReplayWriter


def _load_torch_policy(checkpoint_path: str | Path, policy_variant: str, obs_shape: tuple, action_dim: int, device: str = "cpu"):
    """Load a custom PyTorch PPO policy from a .pt checkpoint.

    How it works: the .pt file stores a payload dict with 'policy_state_dict'.
    We instantiate the correct policy class (based on policy_variant) and load
    the weights. The policy is returned in eval mode so dropout/batchnorm are
    frozen and actions are deterministic at inference time.
    """
    import torch
    from train import TORCH_POLICY_VARIANTS

    policy_cls = TORCH_POLICY_VARIANTS.get(policy_variant)
    if policy_cls is None:
        raise ValueError(f"Unknown policy_variant '{policy_variant}'. Choose from: {list(TORCH_POLICY_VARIANTS.keys())}")

    policy = policy_cls(obs_shape=obs_shape, action_dim=action_dim)
    payload = torch.load(str(checkpoint_path), map_location=device)
    state_dict = payload.get("policy_state_dict", payload)  # fallback if saved as raw state dict
    policy.load_state_dict(state_dict)
    policy.to(device)
    policy.eval()
    return policy


def _load_sb3_policy(checkpoint_path: str | Path):
    """Load a Stable-Baselines3 PPO policy from a .zip checkpoint.

    How it works: SB3 packages the policy + config inside a zip archive.
    We load the full SB3 model then use its .predict() method which handles
    observation preprocessing internally.
    """
    try:
        from stable_baselines3 import PPO as SB3PPO
    except ImportError as exc:
        raise RuntimeError("stable-baselines3 is required for SB3 checkpoints. Install with: pip install stable-baselines3") from exc
    model = SB3PPO.load(str(checkpoint_path))
    return model


def _obs_to_tensor(obs: np.ndarray, device: str = "cpu"):
    """Convert (H, W, C) uint8 observation to (1, C, H, W) float tensor in [0, 1]."""
    import torch
    obs_t = torch.from_numpy(obs).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    return obs_t.to(device)


def collect_ppo_policy_dataset(
    base_config_path: str | Path,
    output_dir: str | Path,
    split: str,
    target_frames: int,
    seed: int,
    policy_checkpoint: str | Path,
    policy_variant: str = "autoresearch_run_008",
    deterministic: bool = True,
    render: bool = False,
    record_video: bool = False,
    video_fps: float = 30.0,
    config_overrides: dict[str, Any] | None = None,
    device: str = "cpu",
) -> CollectionResult:
    """Collect replay data by running a trained PPO policy in the environment.

    Why use a PPO policy instead of scripted maneuvers?
    - The scripted heuristic covers pre-defined motion families but misses the
      natural action correlations a trained agent produces (e.g. smoothly
      accelerating out of a corner apex with appropriate steering).
    - A mid-training PPO checkpoint (e.g. 300k steps) drives well enough to
      stay on track most of the time, but still makes mistakes — this gives the
      world model diverse recovery and correction sequences to learn from.
    - Expert laps (v2_progress) provide clean, consistent geometry the model
      can anchor its predictions on.

    Supports both custom .pt checkpoints and SB3 .zip checkpoints.

    Args:
        policy_checkpoint: Path to .pt (custom) or .zip (SB3) checkpoint.
        policy_variant: For .pt files — 'legacy' or 'autoresearch_run_008'.
        deterministic: If True, use policy mean (no sampling noise). If False,
            sample from the policy distribution for more diverse trajectories.
    """
    import torch

    config = load_config(base_config_path)
    if config_overrides:
        config = _deep_update(copy.deepcopy(config), config_overrides)

    env = create_env(config, rank=0, seed=seed)
    action_low = np.array(env.action_space.low, dtype=np.float32)
    action_high = np.array(env.action_space.high, dtype=np.float32)
    obs_shape = (3, 96, 96)  # (C, H, W) — consistent with training setup
    action_dim = int(env.action_space.shape[0])
    direction = str(config["environment"].get("direction", "CCW")).upper()

    checkpoint_path = Path(policy_checkpoint)
    use_sb3 = checkpoint_path.suffix == ".zip"

    if use_sb3:
        sb3_model = _load_sb3_policy(checkpoint_path)
        torch_policy = None
    else:
        torch_policy = _load_torch_policy(checkpoint_path, policy_variant, obs_shape, action_dim, device)
        sb3_model = None

    writer = ReplayWriter(output_dir, split=split)
    saved_paths: list[Path] = []
    episode_rewards: list[float] = []
    episode_lengths: list[int] = []
    episode_progress: list[float] = []
    episode_offtrack: list[int] = []
    frames_collected = 0
    episode_id = 0
    video_writer: cv2.VideoWriter | None = None
    video_path: Path | None = None

    try:
        while frames_collected < target_frames:
            obs = env.reset()
            done = False
            observations = []
            actions = []
            rewards = []
            dones = []
            truncated = []
            episode_reward = 0.0
            offtrack_seen = False
            info: dict = {}

            while not done and frames_collected < target_frames:
                if use_sb3:
                    action_np, _ = sb3_model.predict(obs, deterministic=deterministic)
                    action_np = action_np.astype(np.float32)
                else:
                    with torch.no_grad():
                        obs_t = _obs_to_tensor(obs, device)
                        raw_action, _, _ = torch_policy.act(obs_t, deterministic=deterministic)
                        # Apply env-space mapping (tanh for steer, sigmoid for throttle/brake)
                        env_action_t = torch_policy.raw_to_env_action(raw_action)
                        action_np = env_action_t.squeeze(0).cpu().numpy().astype(np.float32)

                env_action = _clip_action(action_np, action_low, action_high)
                next_obs, reward, done, info = env.step(env_action)
                info = info if isinstance(info, dict) else {}

                observations.append(obs.astype(np.uint8))
                actions.append(env_action)
                rewards.append(float(reward))
                dones.append(bool(done))
                truncated.append(bool(info.get("TimeLimit.truncated", False)))
                episode_reward += float(reward)
                if int(info.get("events/offtrack", 0)) > 0:
                    offtrack_seen = True
                frames_collected += 1

                if render or record_video:
                    frame = env.render(mode="rgb_array")
                    if frame is not None:
                        overlay = _render_overlay(
                            frame,
                            [
                                f"source=ppo_policy split={split}",
                                f"checkpoint={checkpoint_path.name} direction={direction}",
                                f"action=[{env_action[0]:+.2f}, {env_action[1]:.2f}, {env_action[2]:.2f}]",
                                f"frames={frames_collected}/{target_frames}",
                            ],
                        )
                        frame_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
                        if record_video:
                            if video_writer is None:
                                video_writer, video_path = _init_video_writer(
                                    Path(output_dir) / "videos", split=split, frame_shape=overlay.shape, fps=video_fps
                                )
                            video_writer.write(frame_bgr)
                        if render:
                            cv2.imshow(f"PPO Collector: {split}", frame_bgr)
                            cv2.waitKey(1)

                obs = next_obs

            if not observations:
                continue

            final_info = info if isinstance(info, dict) else {}
            metadata = {
                "collection_source": "ppo_policy",
                "policy_checkpoint": str(checkpoint_path),
                "policy_variant": policy_variant if not use_sb3 else "sb3",
                "deterministic": deterministic,
                "direction": direction,
                "seed": seed + episode_id,
                "config_overrides": config_overrides or {},
                "num_steps": len(observations),
            }
            saved_paths.append(
                _finalize_episode(
                    writer=writer,
                    split=split,
                    episode_id=episode_id,
                    observations=observations,
                    actions=actions,
                    rewards=rewards,
                    dones=dones,
                    truncated=truncated,
                    metadata=metadata,
                )
            )
            episode_rewards.append(episode_reward)
            episode_lengths.append(len(observations))
            episode_progress.append(float(final_info.get("progress", 0.0)))
            episode_offtrack.append(int(offtrack_seen))
            print(
                f"[PPOCollector:{split}] episode={episode_id} steps={len(observations)} "
                f"reward={episode_reward:.2f} progress={float(final_info.get('progress', 0.0)):.2%} "
                f"offtrack={int(offtrack_seen)} frames_total={frames_collected}/{target_frames}",
                flush=True,
            )
            episode_id += 1
    finally:
        if video_writer is not None:
            video_writer.release()
            print(f"[PPOCollector:{split}] Video saved to {video_path}", flush=True)
        env.close()
        cv2.destroyAllWindows()

    summary = _collection_summary(saved_paths, episode_rewards, episode_lengths, episode_progress, episode_offtrack, frames_collected)
    print(f"[PPOCollector:{split}] Summary: {summary}", flush=True)
    return CollectionResult(episode_paths=saved_paths, summary=summary)


LEFT_STEER = -1.0
RIGHT_STEER = 1.0
NEUTRAL_STEER = 0.0
MANUAL_MAX_STEER = 0.55
MANUAL_STEER_STEP = 0.12
MANUAL_STEER_DECAY = 0.18
MANUAL_THROTTLE_RISE = 0.22
MANUAL_THROTTLE_DECAY = 0.32
MANUAL_THROTTLE_BRAKE_DECAY = 0.05
MANUAL_HALF_THROTTLE_TARGET = 0.50
MANUAL_BRAKE_RISE = 0.10
MANUAL_BRAKE_DECAY = 0.18


@dataclass(frozen=True)
class ManualControlState:
    steer: float = 0.0
    throttle: float = 0.0
    brake: float = 0.0
    recovery_focus: bool = False


@dataclass
class CollectionResult:
    episode_paths: list[Path]
    summary: dict[str, float]


@dataclass
class ManeuverEpisode:
    family: str
    actions: np.ndarray
    recovery_focused: bool


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _clip_action(action: np.ndarray, action_low: np.ndarray, action_high: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(action, dtype=np.float32), action_low, action_high).astype(np.float32)


def _decay_toward_zero(value: float, amount: float) -> float:
    if value > 0.0:
        return max(0.0, value - amount)
    if value < 0.0:
        return min(0.0, value + amount)
    return 0.0


def _interpolate_action(start: np.ndarray, end: np.ndarray, steps: int) -> np.ndarray:
    if steps <= 1:
        return np.asarray([end], dtype=np.float32)
    weights = np.linspace(0.0, 1.0, steps, dtype=np.float32).reshape(-1, 1)
    return ((1.0 - weights) * start.reshape(1, -1) + weights * end.reshape(1, -1)).astype(np.float32)


def _append_phase(
    phases: list[np.ndarray],
    last_action: np.ndarray,
    target_action: np.ndarray,
    hold_steps: int,
    transition_steps: int,
) -> np.ndarray:
    pieces = []
    if transition_steps > 0:
        pieces.append(_interpolate_action(last_action, target_action, transition_steps))
    if hold_steps > 0:
        pieces.append(np.repeat(target_action.reshape(1, -1), hold_steps, axis=0).astype(np.float32))
    phase = np.concatenate(pieces, axis=0) if pieces else np.empty((0, last_action.shape[0]), dtype=np.float32)
    phases.append(phase)
    return target_action


def _build_maneuver_actions(family: str, rng: np.random.Generator, regime: dict[str, Any]) -> ManeuverEpisode:
    throttle_scale = float(regime.get("throttle_scale", 1.0))
    steering_scale = float(regime.get("steering_scale", 1.0))
    brake_scale = float(regime.get("brake_scale", 1.0))
    transition_steps = int(regime.get("transition_steps", 4))

    medium_throttle = np.clip(0.55 * throttle_scale, 0.0, 1.0)
    strong_throttle = np.clip(0.85 * throttle_scale, 0.0, 1.0)
    full_throttle = np.clip(1.00 * throttle_scale, 0.0, 1.0)
    light_brake = np.clip(0.35 * brake_scale, 0.0, 1.0)
    strong_brake = np.clip(0.85 * brake_scale, 0.0, 1.0)
    gentle_turn = np.clip(0.35 * steering_scale, 0.0, 1.0)
    hard_turn = np.clip(0.75 * steering_scale, 0.0, 1.0)

    phases: list[np.ndarray] = []
    last_action = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
    recovery_focused = False

    if family == "straight_accel_hold":
        last_action = _append_phase(phases, last_action, np.asarray([NEUTRAL_STEER, full_throttle, 0.0], dtype=np.float32), rng.integers(24, 40), transition_steps)
    elif family == "partial_throttle_cruise_hold":
        last_action = _append_phase(phases, last_action, np.asarray([NEUTRAL_STEER, medium_throttle, 0.0], dtype=np.float32), rng.integers(30, 48), transition_steps)
    elif family == "hard_brake_pulse":
        last_action = _append_phase(phases, last_action, np.asarray([NEUTRAL_STEER, full_throttle, 0.0], dtype=np.float32), rng.integers(16, 26), transition_steps)
        last_action = _append_phase(phases, last_action, np.asarray([NEUTRAL_STEER, 0.0, strong_brake], dtype=np.float32), rng.integers(8, 14), transition_steps)
        last_action = _append_phase(phases, last_action, np.asarray([NEUTRAL_STEER, medium_throttle, 0.0], dtype=np.float32), rng.integers(10, 18), transition_steps)
    elif family == "left_constant_radius_turn_hold":
        last_action = _append_phase(phases, last_action, np.asarray([LEFT_STEER * gentle_turn, strong_throttle, 0.0], dtype=np.float32), rng.integers(24, 40), transition_steps)
    elif family == "right_constant_radius_turn_hold":
        last_action = _append_phase(phases, last_action, np.asarray([RIGHT_STEER * gentle_turn, strong_throttle, 0.0], dtype=np.float32), rng.integers(24, 40), transition_steps)
    elif family == "steer_then_countersteer_recovery":
        recovery_focused = True
        steer_dir = LEFT_STEER if rng.random() < 0.5 else RIGHT_STEER
        last_action = _append_phase(phases, last_action, np.asarray([steer_dir * hard_turn, strong_throttle, 0.0], dtype=np.float32), rng.integers(10, 18), transition_steps)
        last_action = _append_phase(phases, last_action, np.asarray([-steer_dir * gentle_turn, medium_throttle, 0.0], dtype=np.float32), rng.integers(12, 20), transition_steps)
        last_action = _append_phase(phases, last_action, np.asarray([NEUTRAL_STEER, medium_throttle, 0.0], dtype=np.float32), rng.integers(8, 14), transition_steps)
    elif family == "throttle_lift_corner_entry":
        steer_dir = LEFT_STEER if rng.random() < 0.5 else RIGHT_STEER
        last_action = _append_phase(phases, last_action, np.asarray([NEUTRAL_STEER, full_throttle, 0.0], dtype=np.float32), rng.integers(14, 22), transition_steps)
        last_action = _append_phase(phases, last_action, np.asarray([steer_dir * gentle_turn, 0.25 * throttle_scale, 0.0], dtype=np.float32), rng.integers(12, 18), transition_steps)
    elif family == "brake_while_turning_segment":
        recovery_focused = True
        steer_dir = LEFT_STEER if rng.random() < 0.5 else RIGHT_STEER
        last_action = _append_phase(phases, last_action, np.asarray([NEUTRAL_STEER, strong_throttle, 0.0], dtype=np.float32), rng.integers(10, 18), transition_steps)
        last_action = _append_phase(phases, last_action, np.asarray([steer_dir * gentle_turn, 0.15 * throttle_scale, light_brake], dtype=np.float32), rng.integers(12, 18), transition_steps)
        last_action = _append_phase(phases, last_action, np.asarray([-steer_dir * 0.20 * steering_scale, medium_throttle, 0.0], dtype=np.float32), rng.integers(8, 14), transition_steps)
    elif family == "drift_inducing_oversteer_segment":
        recovery_focused = True
        steer_dir = LEFT_STEER if rng.random() < 0.5 else RIGHT_STEER
        last_action = _append_phase(phases, last_action, np.asarray([NEUTRAL_STEER, full_throttle, 0.0], dtype=np.float32), rng.integers(14, 20), transition_steps)
        last_action = _append_phase(phases, last_action, np.asarray([steer_dir * hard_turn, full_throttle, 0.0], dtype=np.float32), rng.integers(12, 18), transition_steps)
        last_action = _append_phase(phases, last_action, np.asarray([-steer_dir * 0.40 * steering_scale, 0.30 * throttle_scale, 0.0], dtype=np.float32), rng.integers(10, 16), transition_steps)
    elif family == "off_track_recovery_segment":
        recovery_focused = True
        steer_dir = LEFT_STEER if rng.random() < 0.5 else RIGHT_STEER
        last_action = _append_phase(phases, last_action, np.asarray([steer_dir * hard_turn, full_throttle, 0.0], dtype=np.float32), rng.integers(16, 24), transition_steps)
        last_action = _append_phase(phases, last_action, np.asarray([-steer_dir * gentle_turn, 0.20 * throttle_scale, light_brake], dtype=np.float32), rng.integers(16, 24), transition_steps)
        last_action = _append_phase(phases, last_action, np.asarray([NEUTRAL_STEER, medium_throttle, 0.0], dtype=np.float32), rng.integers(10, 16), transition_steps)
    else:
        raise ValueError(f"Unknown maneuver family: {family}")

    actions = np.concatenate(phases, axis=0).astype(np.float32)
    return ManeuverEpisode(family=family, actions=actions, recovery_focused=recovery_focused)


def _automatic_maneuver_families() -> list[str]:
    return [
        "straight_accel_hold",
        "partial_throttle_cruise_hold",
        "hard_brake_pulse",
        "left_constant_radius_turn_hold",
        "right_constant_radius_turn_hold",
        "steer_then_countersteer_recovery",
        "throttle_lift_corner_entry",
        "brake_while_turning_segment",
        "drift_inducing_oversteer_segment",
        "off_track_recovery_segment",
    ]


def _init_video_writer(video_root: Path, split: str, frame_shape: tuple[int, ...], fps: float) -> tuple[cv2.VideoWriter, Path]:
    video_root.mkdir(parents=True, exist_ok=True)
    video_path = video_root / f"{split}.mp4"
    height, width = frame_shape[:2]
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open collector video writer for {video_path}")
    return writer, video_path


def _render_overlay(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    canvas = frame.copy()
    y = 20
    for line in lines:
        cv2.putText(canvas, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1, cv2.LINE_AA)
        y += 20
    return canvas


def _finalize_episode(
    writer: ReplayWriter,
    split: str,
    episode_id: int,
    observations: list[np.ndarray],
    actions: list[np.ndarray],
    rewards: list[float],
    dones: list[bool],
    truncated: list[bool],
    metadata: dict[str, Any],
) -> Path:
    action_array = np.asarray(actions, dtype=np.float32)
    zero_noise = np.zeros_like(action_array, dtype=np.float32)
    episode = EpisodeReplay(
        observations_uint8=np.asarray(observations, dtype=np.uint8),
        actions=action_array,
        rewards=np.asarray(rewards, dtype=np.float32),
        dones=np.asarray(dones, dtype=np.bool_),
        truncated=np.asarray(truncated, dtype=np.bool_),
        action_mean_raw=action_array.copy(),
        action_noisy_raw=action_array.copy(),
        noise=zero_noise,
        metadata=metadata,
    )
    return writer.save_episode(episode=episode, episode_id=episode_id)


def _collection_summary(saved_paths: list[Path], episode_rewards: list[float], episode_lengths: list[int], episode_progress: list[float], episode_offtrack: list[int], frames_collected: int) -> dict[str, float]:
    return {
        "num_episodes": float(len(saved_paths)),
        "frames_collected": float(frames_collected),
        "mean_reward": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
        "mean_length": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
        "mean_progress": float(np.mean(episode_progress)) if episode_progress else 0.0,
        "offtrack_rate": float(np.mean(episode_offtrack)) if episode_offtrack else 0.0,
    }


def collect_automatic_maneuver_dataset(
    base_config_path: str | Path,
    output_dir: str | Path,
    split: str,
    target_frames: int,
    seed: int,
    render: bool = False,
    record_video: bool = False,
    video_fps: float = 30.0,
    config_overrides: dict[str, Any] | None = None,
    regime_name: str = "medium",
) -> CollectionResult:
    config = load_config(base_config_path)
    if config_overrides:
        config = _deep_update(copy.deepcopy(config), config_overrides)
    rng = np.random.default_rng(seed)
    env = create_env(config, rank=0, seed=seed)
    writer = ReplayWriter(output_dir, split=split)
    action_low = np.array(env.action_space.low, dtype=np.float32)
    action_high = np.array(env.action_space.high, dtype=np.float32)
    direction = str(config["environment"].get("direction", "CCW")).upper()
    maneuver_regime = ((config_overrides or {}).get("collector_runtime") or {}).get("regime", {})

    saved_paths: list[Path] = []
    episode_rewards: list[float] = []
    episode_lengths: list[int] = []
    episode_progress: list[float] = []
    episode_offtrack: list[int] = []
    frames_collected = 0
    episode_id = 0
    families = _automatic_maneuver_families()
    video_writer: cv2.VideoWriter | None = None
    video_path: Path | None = None

    try:
        while frames_collected < target_frames:
            family = families[episode_id % len(families)]
            maneuver = _build_maneuver_actions(family=family, rng=rng, regime=maneuver_regime)
            obs = env.reset()
            done = False
            observations = []
            actions = []
            rewards = []
            dones = []
            truncated = []
            episode_reward = 0.0
            offtrack_seen = False

            for action in maneuver.actions:
                if done or frames_collected >= target_frames:
                    break
                env_action = _clip_action(action, action_low, action_high)
                next_obs, reward, done, info = env.step(env_action)
                info = info if isinstance(info, dict) else {}

                observations.append(obs.astype(np.uint8))
                actions.append(env_action)
                rewards.append(float(reward))
                dones.append(bool(done))
                truncated.append(bool(info.get("TimeLimit.truncated", False)))
                episode_reward += float(reward)
                if int(info.get("events/offtrack", 0)) > 0:
                    offtrack_seen = True
                frames_collected += 1

                if render or record_video:
                    frame = env.render(mode="rgb_array")
                    if frame is not None:
                        overlay = _render_overlay(
                            frame,
                            [
                                f"source=automatic split={split}",
                                f"family={family} regime={regime_name} direction={direction}",
                                f"action=[{env_action[0]:+.2f}, {env_action[1]:.2f}, {env_action[2]:.2f}]",
                            ],
                        )
                        frame_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
                        if record_video:
                            if video_writer is None:
                                video_writer, video_path = _init_video_writer(Path(output_dir) / "videos", split=split, frame_shape=overlay.shape, fps=video_fps)
                            video_writer.write(frame_bgr)
                        if render:
                            cv2.imshow(f"World Model Collector: {split}", frame_bgr)
                            cv2.waitKey(1)
                obs = next_obs

            if not observations:
                continue

            final_info = info if isinstance(info, dict) else {}
            metadata = {
                "collection_source": "automatic",
                "maneuver_family": family,
                "speed_regime": regime_name,
                "direction": direction,
                "seed": seed,
                "config_overrides": config_overrides or {},
                "recovery_focused": bool(maneuver.recovery_focused),
                "num_steps": len(observations),
            }
            saved_paths.append(
                _finalize_episode(
                    writer=writer,
                    split=split,
                    episode_id=episode_id,
                    observations=observations,
                    actions=actions,
                    rewards=rewards,
                    dones=dones,
                    truncated=truncated,
                    metadata=metadata,
                )
            )
            episode_rewards.append(episode_reward)
            episode_lengths.append(len(observations))
            episode_progress.append(float(final_info.get("progress", 0.0)))
            episode_offtrack.append(int(offtrack_seen))
            print(
                f"[Collector:{split}] episode={episode_id} family={family} steps={len(observations)} "
                f"reward={episode_reward:.2f} progress={float(final_info.get('progress', 0.0)):.2%} "
                f"offtrack={int(offtrack_seen)} frames_total={frames_collected}/{target_frames}",
                flush=True,
            )
            episode_id += 1
    finally:
        if video_writer is not None:
            video_writer.release()
            print(f"[Collector:{split}] Video saved to {video_path}", flush=True)
        env.close()
        cv2.destroyAllWindows()

    summary = _collection_summary(saved_paths, episode_rewards, episode_lengths, episode_progress, episode_offtrack, frames_collected)
    print(f"[Collector:{split}] Summary: {summary}", flush=True)
    return CollectionResult(episode_paths=saved_paths, summary=summary)


def _poll_manual_controls(
    pygame_module,
    state: ManualControlState,
    markers: list[int],
    current_step: int,
) -> tuple[ManualControlState, bool, bool]:
    quit_requested = False
    reset_requested = False
    next_recovery_focus = bool(state.recovery_focus)

    for event in pygame_module.event.get():
        if event.type == pygame_module.QUIT:
            quit_requested = True
        elif event.type == pygame_module.KEYDOWN:
            if event.key == pygame_module.K_q:
                quit_requested = True
            elif event.key == pygame_module.K_r:
                reset_requested = True
            elif event.key == pygame_module.K_f:
                next_recovery_focus = not next_recovery_focus
            elif event.key == pygame_module.K_m:
                markers.append(int(current_step))

    keys = pygame_module.key.get_pressed()
    steer = float(state.steer)
    if keys[pygame_module.K_LEFT] and not keys[pygame_module.K_RIGHT]:
        steer = max(-MANUAL_MAX_STEER, steer - MANUAL_STEER_STEP)
    elif keys[pygame_module.K_RIGHT] and not keys[pygame_module.K_LEFT]:
        steer = min(MANUAL_MAX_STEER, steer + MANUAL_STEER_STEP)
    else:
        steer = _decay_toward_zero(steer, MANUAL_STEER_DECAY)

    brake_pressed = bool(keys[pygame_module.K_DOWN])
    throttle = float(state.throttle)
    if brake_pressed:
        throttle = max(0.0, throttle - MANUAL_THROTTLE_BRAKE_DECAY)
    elif keys[pygame_module.K_UP]:
        throttle = min(1.0, throttle + MANUAL_THROTTLE_RISE)
    elif keys[pygame_module.K_x]:
        throttle = min(MANUAL_HALF_THROTTLE_TARGET, throttle + MANUAL_THROTTLE_RISE)
    else:
        throttle = max(0.0, throttle - MANUAL_THROTTLE_DECAY)

    brake = float(state.brake)
    if brake_pressed:
        brake = min(1.0, brake + MANUAL_BRAKE_RISE)
    else:
        brake = max(0.0, brake - MANUAL_BRAKE_DECAY)

    if keys[pygame_module.K_SPACE] or keys[pygame_module.K_c]:
        steer = 0.0
        throttle = 0.0
        brake = 0.0

    if keys[pygame_module.K_e]:
        steer = 0.0

    return ManualControlState(
        steer=steer,
        throttle=throttle,
        brake=brake,
        recovery_focus=next_recovery_focus,
    ), quit_requested, reset_requested


def _init_pygame_window(frame_shape: tuple[int, ...]):
    try:
        import pygame
    except ImportError as exc:
        raise RuntimeError(
            "pygame is required for manual teleoperation. Install it with 'pip install pygame'."
        ) from exc

    pygame.init()
    height, width = frame_shape[:2]
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("World Model Manual Collector")
    return pygame, screen


def _draw_pygame_frame(pygame_module, screen, overlay: np.ndarray) -> None:
    surface = pygame_module.surfarray.make_surface(np.transpose(overlay, (1, 0, 2)))
    screen.blit(surface, (0, 0))
    pygame_module.display.flip()


def collect_manual_keyboard_dataset(
    base_config_path: str | Path,
    output_dir: str | Path,
    split: str,
    seed: int,
    target_frames: int,
    render: bool = True,
    record_video: bool = False,
    video_fps: float = 30.0,
    config_overrides: dict[str, Any] | None = None,
    speed_regime: str = "medium",
) -> CollectionResult:
    config = load_config(base_config_path)
    if config_overrides:
        config = _deep_update(copy.deepcopy(config), config_overrides)
    env = create_env(config, rank=0, seed=seed)
    writer = ReplayWriter(output_dir, split=split)
    action_low = np.array(env.action_space.low, dtype=np.float32)
    action_high = np.array(env.action_space.high, dtype=np.float32)
    direction = str(config["environment"].get("direction", "CCW")).upper()

    saved_paths: list[Path] = []
    episode_rewards: list[float] = []
    episode_lengths: list[int] = []
    episode_progress: list[float] = []
    episode_offtrack: list[int] = []
    frames_collected = 0
    episode_id = 0
    video_writer: cv2.VideoWriter | None = None
    video_path: Path | None = None
    pygame_module = None
    screen = None
    quit_requested = False

    try:
        while frames_collected < target_frames and not quit_requested:
            obs = env.reset()
            done = False
            state = ManualControlState()
            segment_markers: list[int] = []
            observations = []
            actions = []
            rewards = []
            dones = []
            truncated = []
            episode_reward = 0.0
            offtrack_seen = False

            while not done and frames_collected < target_frames and not quit_requested:
                env_action = _clip_action(np.asarray([state.steer, state.throttle, state.brake], dtype=np.float32), action_low, action_high)
                next_obs, reward, done, info = env.step(env_action)
                info = info if isinstance(info, dict) else {}

                observations.append(obs.astype(np.uint8))
                actions.append(env_action)
                rewards.append(float(reward))
                dones.append(bool(done))
                truncated.append(bool(info.get("TimeLimit.truncated", False)))
                episode_reward += float(reward)
                if int(info.get("events/offtrack", 0)) > 0:
                    offtrack_seen = True
                frames_collected += 1

                frame = env.render(mode="rgb_array")
                if frame is not None:
                    overlay = _render_overlay(
                        frame,
                        [
                            f"source=manual split={split}",
                            f"direction={direction} regime={speed_regime} recovery_focus={int(state.recovery_focus)}",
                            f"action=[{env_action[0]:+.2f}, {env_action[1]:.2f}, {env_action[2]:.2f}]",
                            f"keys: Left/Right ramp steer to +/-{MANUAL_MAX_STEER:.2f}, Up ramps throttle, Down ramps brake",
                            "keys: release = fast throttle/brake decay, Down adds brake and rapidly bleeds throttle",
                            "keys: f recovery, m marker, r reset, q quit",
                        ],
                    )
                    if render and pygame_module is None:
                        pygame_module, screen = _init_pygame_window(overlay.shape)
                    if render and pygame_module is not None and screen is not None:
                        _draw_pygame_frame(pygame_module, screen, overlay)
                    if record_video:
                        if video_writer is None:
                            video_writer, video_path = _init_video_writer(Path(output_dir) / "videos", split=split, frame_shape=overlay.shape, fps=video_fps)
                        video_writer.write(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

                if pygame_module is not None:
                    state, quit_requested, reset_requested = _poll_manual_controls(
                        pygame_module,
                        state,
                        segment_markers,
                        len(observations),
                    )
                else:
                    state = ManualControlState(recovery_focus=state.recovery_focus)
                    quit_requested = False
                    reset_requested = False
                if reset_requested:
                    done = True
                obs = next_obs

            if not observations:
                continue

            final_info = info if isinstance(info, dict) else {}
            metadata = {
                "collection_source": "manual",
                "maneuver_family": None,
                "speed_regime": speed_regime,
                "direction": direction,
                "seed": seed,
                "config_overrides": config_overrides or {},
                "recovery_focused": bool(state.recovery_focus),
                "segment_markers": segment_markers,
                "num_steps": len(observations),
            }
            saved_paths.append(
                _finalize_episode(
                    writer=writer,
                    split=split,
                    episode_id=episode_id,
                    observations=observations,
                    actions=actions,
                    rewards=rewards,
                    dones=dones,
                    truncated=truncated,
                    metadata=metadata,
                )
            )
            episode_rewards.append(episode_reward)
            episode_lengths.append(len(observations))
            episode_progress.append(float(final_info.get("progress", 0.0)))
            episode_offtrack.append(int(offtrack_seen))
            print(
                f"[Collector:{split}] episode={episode_id} source=manual steps={len(observations)} "
                f"reward={episode_reward:.2f} progress={float(final_info.get('progress', 0.0)):.2%} "
                f"offtrack={int(offtrack_seen)} frames_total={frames_collected}/{target_frames}",
                flush=True,
            )
            episode_id += 1
    finally:
        if video_writer is not None:
            video_writer.release()
            print(f"[Collector:{split}] Video saved to {video_path}", flush=True)
        if pygame_module is not None:
            pygame_module.quit()
        env.close()
        cv2.destroyAllWindows()

    summary = _collection_summary(saved_paths, episode_rewards, episode_lengths, episode_progress, episode_offtrack, frames_collected)
    print(f"[Collector:{split}] Summary: {summary}", flush=True)
    return CollectionResult(episode_paths=saved_paths, summary=summary)


def save_collection_manifest(
    output_dir: str | Path,
    split: str,
    episode_paths: list[Path],
    summary: dict[str, float] | None = None,
) -> Path:
    manifest_path = Path(output_dir) / f"{split}_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "split": split,
                "episodes": [str(path) for path in episode_paths],
                "summary": summary or {},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return manifest_path
