"""
Evaluation script for trained Multi-Car Racing PPO models.

This script loads a trained model and evaluates its performance,
generating metrics and optionally recording videos.
"""

import os
import yaml
import argparse
import json
import numpy as np
from pathlib import Path
import gym
import gym_multi_car_racing
from gym_multi_car_racing import multi_car_racing as mcr
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
import cv2
from datetime import datetime
import torch

# Import Torch policy and wrappers from training module for consistency
try:
    from train import (
        CnnActorCritic,
        MultiAgentSpaceWrapper,
        RewardShapingWrapper,
        TORCH_POLICY_VARIANTS,
        mps_is_available,
        validate_agent_space_contract,
    )
except ImportError:
    CnnActorCritic = None
    MultiAgentSpaceWrapper = None
    RewardShapingWrapper = None
    TORCH_POLICY_VARIANTS = {}
    validate_agent_space_contract = None

    def mps_is_available():
        return False


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def reward_to_scalar(reward):
    """Collapse scalar or per-agent reward arrays into one comparable metric."""
    reward_arr = np.asarray(reward, dtype=np.float32)
    if reward_arr.ndim == 0:
        return float(reward_arr)
    return float(np.mean(reward_arr))


def done_to_bool(done):
    done_arr = np.asarray(done)
    if done_arr.ndim == 0:
        return bool(done_arr)
    return bool(done_arr.reshape(-1).any())


def first_agent_info(info):
    if isinstance(info, dict):
        return info
    if isinstance(info, (list, tuple)) and len(info) > 0:
        first = info[0]
        if isinstance(first, dict):
            return first
    return {}


def infer_image_space_layout(space_shape):
    shape = tuple(space_shape)
    if len(shape) == 3:
        if shape[-1] in (1, 3, 4):
            h, w, c = shape
            return 1, (c, h, w), "hwc"
        c, h, w = shape
        return 1, (c, h, w), "chw"
    if len(shape) == 4 and shape[-1] in (1, 3, 4):
        n_agents, h, w, c = shape
        return int(n_agents), (c, h, w), "agent_hwc"
    if len(shape) == 4:
        n_agents, c, h, w = shape
        return int(n_agents), (c, h, w), "agent_chw"
    raise ValueError(f"Unsupported observation space shape: {shape}")


def obs_to_policy_batch(obs, obs_layout):
    obs_arr = np.asarray(obs, dtype=np.float32)
    if obs_layout == "chw":
        if obs_arr.ndim == 3:
            return obs_arr[None, ...]
        return obs_arr.reshape(-1, *obs_arr.shape[-3:])
    if obs_layout == "hwc":
        if obs_arr.ndim == 3:
            obs_arr = obs_arr[None, ...]
        obs_arr = obs_arr.reshape(-1, *obs_arr.shape[-3:])
        return np.transpose(obs_arr, (0, 3, 1, 2))
    if obs_layout == "agent_hwc":
        if obs_arr.ndim == 4:
            obs_arr = obs_arr[None, ...]
        obs_arr = obs_arr.reshape(-1, *obs_arr.shape[-3:])
        return np.transpose(obs_arr, (0, 3, 1, 2))
    if obs_layout == "agent_chw":
        if obs_arr.ndim == 4:
            obs_arr = obs_arr[None, ...]
        return obs_arr.reshape(-1, *obs_arr.shape[-3:])
    raise ValueError(f"Unknown observation layout: {obs_layout}")


def action_batch_to_env(action_batch, n_envs, n_agents):
    action_batch = np.asarray(action_batch, dtype=np.float64)
    if n_agents <= 1:
        return action_batch.reshape(n_envs, -1)
    return action_batch.reshape(n_envs, n_agents, -1)


def compose_render_frame(frame):
    """Convert single-agent or stacked multi-agent render output into one RGB frame."""
    if frame is None:
        return None
    frame_arr = np.asarray(frame)
    if frame_arr.ndim == 3:
        return frame_arr
    if frame_arr.ndim == 4:
        # MultiCarRacing returns one frame per car: (num_agents, H, W, C).
        # Tile them side-by-side so OpenCV receives a standard HWC image.
        return np.concatenate(list(frame_arr), axis=1)
    raise ValueError(f"Unsupported render frame shape: {frame_arr.shape}")


def seed_environment(env, seed):
    """Seed the environment and spaces for reproducible evaluation rollouts."""
    seed = int(seed)
    if hasattr(env, "seed"):
        env.seed(seed)
    elif hasattr(env, "unwrapped") and hasattr(env.unwrapped, "seed"):
        env.unwrapped.seed(seed)
    if hasattr(env, "action_space") and hasattr(env.action_space, "seed"):
        env.action_space.seed(seed)
    if hasattr(env, "observation_space") and hasattr(env.observation_space, "seed"):
        env.observation_space.seed(seed)


def resolve_output_path(explicit_path, results_dir: Path, model_path, seed: int, suffix: str, extension: str) -> Path:
    """Resolve explicit or deterministic default output artifact paths."""
    if explicit_path:
        path = Path(explicit_path)
    else:
        model_stem = Path(model_path).stem.replace(" ", "_")
        path = results_dir / f"{model_stem}_seed{int(seed)}_{suffix}.{extension}"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


class SingleAgentWrapper(gym.Wrapper):
    """Wrap MultiCarRacing to expose a single-agent view."""

    def __init__(self, env):
        super().__init__(env)
        obs_space = env.observation_space
        act_space = env.action_space

        if len(obs_space.shape) == 4 and obs_space.shape[0] == 1:
            self.observation_space = gym.spaces.Box(
                low=obs_space.low[0],
                high=obs_space.high[0],
                shape=obs_space.shape[1:],
                dtype=obs_space.dtype
            )
        if len(act_space.shape) == 2 and act_space.shape[0] == 1:
            self.action_space = gym.spaces.Box(
                low=act_space.low[0],
                high=act_space.high[0],
                shape=act_space.shape[1:],
                dtype=act_space.dtype
            )

    def reset(self, **kwargs):
        # Gym 0.17.3 reset() returns just obs, not (obs, info)
        obs = self.env.reset(**kwargs)
        # Extract single agent observation if multi-agent format (num_agents, H, W, C)
        if hasattr(obs, "shape") and len(obs.shape) == 4 and obs.shape[0] == 1:
            obs = obs[0]  # Remove first dimension: (1, H, W, C) -> (H, W, C)
        elif isinstance(obs, (list, tuple)) and len(obs) == 1:
            obs = obs[0]
        # Gym 0.17.3: return just obs (not tuple)
        return obs

    def render(self, mode='human', **kwargs):
        out = self.env.render(mode=mode, **kwargs)
        # Extract single agent frame if multi-agent format (num_agents, H, W, C)
        if hasattr(out, "shape") and len(out.shape) == 4 and out.shape[0] == 1:
            out = out[0]
        return out

    def step(self, action):
        if hasattr(self.env.action_space, "shape") and len(self.env.action_space.shape) == 2:
            action = np.asarray(action, dtype=np.float64).reshape(1, -1)
        elif action is not None:
            action = np.asarray(action, dtype=np.float64)
        obs, reward, done, info = self.env.step(action)
        # Extract single agent observation if multi-agent format (num_agents, H, W, C)
        if hasattr(obs, "shape") and len(obs.shape) == 4 and obs.shape[0] == 1:
            obs = obs[0]  # Remove first dimension: (1, H, W, C) -> (H, W, C)
        elif isinstance(obs, (list, tuple)) and len(obs) == 1:
            obs = obs[0]
        # Extract single agent reward if multi-agent format
        if isinstance(reward, (list, tuple)) or (hasattr(reward, "shape") and len(reward.shape) > 0 and reward.shape[0] == 1):
            reward = float(reward[0] if isinstance(reward, (list, tuple)) else reward[0])
        # Extract single agent info
        if isinstance(info, (list, tuple)) and len(info) == 1:
            info = info[0]
        return obs, reward, done, info


class SafetyGovernorWrapper(gym.Wrapper):
    """Optional speed cap to keep the agent below a target velocity."""

    def __init__(self, env, governor_config):
        super().__init__(env)
        governor_config = governor_config or {}
        self.enabled = bool(
            governor_config.get('enabled', governor_config.get('speed_cap_enabled', False))
        )
        self.speed_cap_ratio = float(governor_config.get('speed_cap_ratio', 0.5))
        self.speed_cap_top_speed = float(governor_config.get('speed_cap_top_speed', 30.0))
        self.speed_cap_brake = float(governor_config.get('speed_cap_brake', 0.2))

    def step(self, action):
        if self.enabled and action is not None:
            base_env = self.env.unwrapped
            if hasattr(base_env, "cars") and base_env.cars:
                speed_cap = self.speed_cap_ratio * self.speed_cap_top_speed
                if speed_cap > 0.0:
                    action_arr = np.asarray(action, dtype=np.float64).copy()
                    orig_shape = action_arr.shape
                    if action_arr.ndim == 1:
                        action_matrix = action_arr.reshape(1, -1)
                    else:
                        action_matrix = action_arr.reshape(action_arr.shape[0], -1)
                    for idx, car in enumerate(base_env.cars[:action_matrix.shape[0]]):
                        vel = car.hull.linearVelocity
                        speed = float(np.linalg.norm([vel[0], vel[1]]))
                        if speed <= speed_cap or action_matrix.shape[1] < 3:
                            continue
                        action_matrix[idx, 1] = 0.0
                        action_matrix[idx, 2] = max(float(action_matrix[idx, 2]), self.speed_cap_brake)
                    action = action_matrix.reshape(orig_shape)
        return self.env.step(action)


class ObservationAugmentWrapper(gym.Wrapper):
    """Augment observations with angular velocity, centerline distance, and look-ahead angles."""

    def __init__(self, env, obs_config):
        super().__init__(env)
        obs_config = obs_config or {}
        self.enabled = bool(obs_config.get('enabled', False))
        if not self.enabled:
            return

        image_space = env.observation_space
        if len(image_space.shape) != 3:
            raise ValueError("Expected image observations of shape (H, W, C)")

        c, h, w = image_space.shape[2], image_space.shape[0], image_space.shape[1]
        self.observation_space = gym.spaces.Dict({
            "image": gym.spaces.Box(
                low=0,
                high=255,
                shape=(c, h, w),
                dtype=image_space.dtype
            ),
            "state": gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(4,),
                dtype=np.float32
            )
        })

    def _compute_state(self):
        base_env = self.env.unwrapped
        ang_vel = 0.0
        dist_norm = 0.0
        beta10 = 0.0
        beta20 = 0.0

        if hasattr(base_env, "cars") and base_env.cars:
            car = base_env.cars[0]
            ang_vel = float(car.hull.angularVelocity)
            if hasattr(base_env, "track") and base_env.track:
                car_pos = np.array(car.hull.position).reshape((1, 2))
                track_xy = np.array(base_env.track)[:, 2:]
                distances = np.linalg.norm(car_pos - track_xy, ord=2, axis=1)
                track_index = int(np.argmin(distances))
                lane_half_width = float(mcr.TRACK_WIDTH) / 2.0
                if lane_half_width > 0.0:
                    dist_norm = float(distances[track_index]) / lane_half_width

                offset_10 = int(round(10.0 / float(mcr.TRACK_DETAIL_STEP)))
                offset_20 = int(round(20.0 / float(mcr.TRACK_DETAIL_STEP)))
                idx_10 = (track_index + offset_10) % len(base_env.track)
                idx_20 = (track_index + offset_20) % len(base_env.track)
                beta10 = float(base_env.track[idx_10][1])
                beta20 = float(base_env.track[idx_20][1])

        return np.array([ang_vel, dist_norm, beta10, beta20], dtype=np.float32)

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        if not self.enabled:
            return obs
        image = np.transpose(obs, (2, 0, 1))
        state = self._compute_state()
        return {"image": image, "state": state}

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if not self.enabled:
            return obs, reward, done, info
        image = np.transpose(obs, (2, 0, 1))
        state = self._compute_state()
        return {"image": image, "state": state}, reward, done, info


def create_env(config, render_mode='rgb_array', seed=None):
    """Create evaluation environment with rendering."""
    env_config = config['environment']
    env_id = env_config.get('env_id', 'MultiCarRacing-v0')
    
    # Note: Gym 0.17.3 doesn't support render_mode in gym.make()
    # Rendering is handled via env.render() calls instead
    env = gym.make(
        env_id,
        num_agents=env_config.get('num_agents', 1),
        direction=env_config.get('direction', 'CCW'),
        use_random_direction=env_config.get('use_random_direction', True),
        backwards_flag=env_config.get('backwards_flag', True),
        h_ratio=env_config.get('h_ratio', 0.25),
        use_ego_color=env_config.get('use_ego_color', False)
    )
    max_episode_steps = env_config.get('max_episode_steps', None)
    if max_episode_steps is not None:
        env = gym.wrappers.TimeLimit(env.unwrapped, max_episode_steps=int(max_episode_steps))

    if env_config.get('num_agents', 1) == 1:
        env = SingleAgentWrapper(env)
    elif env_config.get('num_agents', 1) > 1:
        if MultiAgentSpaceWrapper is None:
            raise RuntimeError("MultiAgentSpaceWrapper unavailable. Ensure train.py is importable.")
        env = MultiAgentSpaceWrapper(env, env_config.get('num_agents', 1))
    if validate_agent_space_contract is not None:
        validate_agent_space_contract(env, int(env_config.get('num_agents', 1)), "evaluate.create_env")

    governor_config = config.get('safety_governor', {})
    if governor_config.get('enabled', False):
        env = SafetyGovernorWrapper(env, governor_config)

    reward_config = config.get('reward_shaping', {})
    if reward_config.get('enabled', False):
        if RewardShapingWrapper is None:
            raise RuntimeError("RewardShapingWrapper unavailable. Ensure train.py is importable.")
        env = RewardShapingWrapper(env, reward_config)

    obs_config = config.get('observation', {})
    if obs_config.get('enabled', False):
        if int(env_config.get('num_agents', 1)) > 1:
            raise ValueError("Observation augmentation evaluation is not supported for multi-agent image observations.")
        env = ObservationAugmentWrapper(env, obs_config)
    if seed is not None:
        seed_environment(env, seed)
    
    return env


def evaluate_torch_model(model_path, config, n_episodes=10, record_video=True, seed=42,
                         show_window=False, video_path=None):
    """Evaluate a Torch .pt checkpoint with optional video and optional live window."""
    if CnnActorCritic is None:
        raise RuntimeError("Cannot load Torch model: train.py CnnActorCritic not available")
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Torch checkpoint not found: {model_path}")
    print(f"Loading Torch checkpoint from {model_path}")
    payload = torch.load(str(model_path), map_location="cpu", weights_only=False)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif mps_is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    np.random.seed(seed)
    torch.manual_seed(int(seed))
    base_env = create_env(config, seed=seed)
    num_agents, obs_shape, obs_layout = infer_image_space_layout(base_env.observation_space.shape)
    action_shape = tuple(base_env.action_space.shape)
    action_dim = int(action_shape[-1] if len(action_shape) >= 2 else action_shape[0])
    action_low = np.array(base_env.action_space.low, dtype=np.float32)
    action_high = np.array(base_env.action_space.high, dtype=np.float32)
    training_cfg = config.get("training", {}) or {}
    policy_variant = str(training_cfg.get("torch_policy_variant", "legacy")).strip().lower()
    policy_cls = TORCH_POLICY_VARIANTS.get(policy_variant, CnnActorCritic)
    print(f"Using torch policy variant: {policy_variant}")

    ppo_cfg = config.get("ppo", {}) or {}
    min_log_std = float(ppo_cfg.get("min_log_std", -1.5))
    max_log_std = float(ppo_cfg.get("max_log_std", 1.0))
    steer_min_log_std = ppo_cfg.get("steer_min_log_std", None)
    steer_max_log_std = ppo_cfg.get("steer_max_log_std", None)
    steer_min_log_std = float(steer_min_log_std) if steer_min_log_std is not None else None
    steer_max_log_std = float(steer_max_log_std) if steer_max_log_std is not None else None

    policy = policy_cls(
        obs_shape,
        action_dim,
        min_log_std=min_log_std,
        max_log_std=max_log_std,
        steer_min_log_std=steer_min_log_std,
        steer_max_log_std=steer_max_log_std,
    )
    policy.load_state_dict(payload["policy_state_dict"])
    policy.to(device)
    policy.eval()

    episode_rewards = []
    episode_lengths = []
    episode_progress = []
    episode_times = []
    episode_offtrack = []
    episode_steer_variance = []
    episode_mean_rank = []
    episode_collision = []
    episode_mean_speed = []
    episode_mean_throttle = []
    episode_mean_brake = []
    episode_mean_overtakes = []
    results_dir = Path(config["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    video_writer = None
    if record_video:
        video_path = resolve_output_path(video_path, results_dir, model_path, seed, "evaluation", "mp4")
    else:
        video_path = None

    print(
        f"Evaluating Torch model for {n_episodes} episodes "
        f"({'video' if record_video else 'no video'}, {'live window' if show_window else 'headless window'})..."
    )
    for episode in range(n_episodes):
        obs = base_env.reset()
        done = False
        episode_reward = 0.0
        episode_length = 0
        start_time = None
        offtrack_seen = False
        steer_values = []
        rank_values = []
        collision_seen = False
        speed_values = []
        throttle_values = []
        brake_values = []
        overtake_count = 0
        while not done_to_bool(done):
            obs_policy = obs_to_policy_batch(obs, obs_layout)
            obs_t = torch.as_tensor(obs_policy, dtype=torch.float32, device=device) / 255.0
            with torch.no_grad():
                raw_action, _, _ = policy.act(obs_t, deterministic=True)
            env_action = policy_cls.raw_to_env_action(raw_action).cpu().numpy()
            env_action = action_batch_to_env(env_action, n_envs=1, n_agents=num_agents)
            env_action = np.clip(env_action, action_low, action_high)
            if num_agents <= 1:
                env_action = env_action.reshape(-1)
            else:
                env_action = env_action.reshape(num_agents, action_dim)
            obs, reward, done, info = base_env.step(env_action)
            episode_reward += reward_to_scalar(reward)
            episode_length += 1
            flat_actions = np.asarray(env_action).reshape(-1, action_dim)
            steer_values.extend(flat_actions[:, 0].astype(np.float32).tolist())
            if action_dim >= 2:
                throttle_values.extend(flat_actions[:, 1].astype(np.float32).tolist())
            if action_dim >= 3:
                brake_values.extend(flat_actions[:, 2].astype(np.float32).tolist())
            agent_infos = info if isinstance(info, (list, tuple)) else [info]
            info0 = first_agent_info(info)
            for agent_info in agent_infos:
                if not isinstance(agent_info, dict):
                    continue
                offtrack_seen = offtrack_seen or int(agent_info.get("events/offtrack", 0)) > 0
                collision_seen = collision_seen or int(agent_info.get("events/collision", 0)) > 0
                speed_values.append(float(agent_info.get("telemetry/speed", 0.0)))
                overtake_count += int(agent_info.get("events/overtake", 0))
                if "telemetry/rank" in agent_info:
                    rank_values.append(float(agent_info.get("telemetry/rank", 0.0)))
            if start_time is None and info0.get("time") is not None:
                start_time = info0["time"]
            frame = base_env.render(mode="rgb_array")
            if frame is not None:
                frame = compose_render_frame(frame)
                if record_video:
                    if video_writer is None:
                        h, w = frame.shape[:2]
                        video_writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h))
                    video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                if show_window:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    cv2.imshow("Racecar Gym (Torch policy)", frame_bgr)
                    cv2.waitKey(1)
        progress_values = [
            float(agent_info.get("progress", 0.0))
            for agent_info in (info if isinstance(info, (list, tuple)) else [info])
            if isinstance(agent_info, dict)
        ]
        progress = float(np.mean(progress_values)) if progress_values else 0.0
        final_time = info0.get("time", 0.0)
        episode_time = final_time - start_time if start_time is not None else 0.0
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_progress.append(progress)
        episode_times.append(episode_time)
        episode_offtrack.append(int(offtrack_seen))
        episode_steer_variance.append(float(np.var(steer_values)) if len(steer_values) > 1 else 0.0)
        episode_mean_rank.append(float(np.mean(rank_values)) if rank_values else 1.0)
        episode_collision.append(int(collision_seen))
        episode_mean_speed.append(float(np.mean(speed_values)) if speed_values else 0.0)
        episode_mean_throttle.append(float(np.mean(throttle_values)) if throttle_values else 0.0)
        episode_mean_brake.append(float(np.mean(brake_values)) if brake_values else 0.0)
        episode_mean_overtakes.append(int(overtake_count))
        print(
            f"Episode {episode + 1}/{n_episodes}: Reward={episode_reward:.2f}, "
            f"Length={episode_length}, Progress={progress:.2%}, Time={episode_time:.2f}s, "
            f"MeanRank={np.mean(rank_values) if rank_values else 1.0:.2f}, Collision={int(collision_seen)}"
        )
    if video_writer is not None:
        video_writer.release()
        print(f"Video saved to {video_path}")
    base_env.close()
    if show_window:
        cv2.destroyAllWindows()
    stats = {
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "min_reward": float(np.min(episode_rewards)),
        "max_reward": float(np.max(episode_rewards)),
        "mean_length": float(np.mean(episode_lengths)),
        "std_length": float(np.std(episode_lengths)),
        "mean_progress": float(np.mean(episode_progress)),
        "std_progress": float(np.std(episode_progress)),
        "mean_time": float(np.mean(episode_times)),
        "std_time": float(np.std(episode_times)),
        "offtrack_rate": float(np.mean(episode_offtrack)),
        "mean_steer_variance": float(np.mean(episode_steer_variance)),
        "mean_rank": float(np.mean(episode_mean_rank)),
        "collision_rate": float(np.mean(episode_collision)),
        "mean_speed": float(np.mean(episode_mean_speed)),
        "mean_throttle": float(np.mean(episode_mean_throttle)),
        "mean_brake": float(np.mean(episode_mean_brake)),
        "mean_overtakes": float(np.mean(episode_mean_overtakes)),
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "episode_progress": episode_progress,
        "episode_times": episode_times,
        "episode_offtrack": episode_offtrack,
        "episode_steer_variance": episode_steer_variance,
        "episode_mean_rank": episode_mean_rank,
        "episode_collision": episode_collision,
        "episode_mean_speed": episode_mean_speed,
        "episode_mean_throttle": episode_mean_throttle,
        "episode_mean_brake": episode_mean_brake,
        "episode_mean_overtakes": episode_mean_overtakes,
    }
    return stats, str(video_path) if video_path else None


def evaluate_model(model_path, config, n_episodes=10, record_video=True, seed=42,
                   show_window=False, video_path=None):
    """Evaluate a trained SB3 model."""
    if int(config.get('environment', {}).get('num_agents', 1)) > 1:
        raise ValueError("SB3 evaluation is not supported for multi-agent configs in this repo.")
    print(f"Loading model from {model_path}")
    model = PPO.load(model_path)
    np.random.seed(seed)
    torch.manual_seed(int(seed))
    
    # Create environment
    # Note: render_mode parameter is ignored in create_env for Gym 0.17.3 compatibility
    # Rendering is handled via env.render() calls
    base_env = create_env(config, render_mode='rgb_array' if record_video else None, seed=seed)
    env = DummyVecEnv([lambda: base_env])
    if not config.get('observation', {}).get('enabled', False):
        env = VecTransposeImage(env)
    
    # Evaluation metrics
    episode_rewards = []
    episode_lengths = []
    episode_progress = []
    episode_times = []
    episode_offtrack = []
    episode_steer_variance = []
    
    results_dir = Path(config['paths']['results_dir'])
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Create video writer lazily after first frame
    video_writer = None
    if record_video:
        video_path = resolve_output_path(video_path, results_dir, model_path, seed, "evaluation", "mp4")
    else:
        video_path = None
    
    print(f"Evaluating model for {n_episodes} episodes...")
    
    for episode in range(n_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        start_time = None
        offtrack_seen = False
        steer_values = []
        
        while not done_to_bool(done):
            action, _ = model.predict(obs, deterministic=True)
            steer_values.append(float(np.asarray(action).reshape(-1)[0]))
            
            obs, reward, done, info = env.step(action)
            
            episode_reward += reward_to_scalar(reward)
            episode_length += 1
            
            # Record first frame time
            info_env = info[0] if isinstance(info, (list, tuple)) else info
            info0 = first_agent_info(info_env)
            if start_time is None and isinstance(info0, dict) and 'time' in info0:
                start_time = info0['time']
            if isinstance(info0, dict) and int(info0.get("events/offtrack", 0)) > 0:
                offtrack_seen = True
            
            # Get frame for video or live view
            frame = compose_render_frame(base_env.render(mode='rgb_array'))
            
            # Record video frame
            if record_video:
                if frame is not None:
                    if video_writer is None:
                        height, width = frame.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        video_writer = cv2.VideoWriter(str(video_path), fourcc, 30.0, (width, height))
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    video_writer.write(frame_bgr)
            
            # Show live window
            if frame is not None and show_window:
                # Convert if not already done
                if 'frame_bgr' not in locals():
                     frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.imshow("Racecar Gym Evaluation", frame_bgr)
                cv2.waitKey(1)
        
        # Extract final metrics from info
        final_info = info0 if isinstance(info0, dict) else {}
        progress = final_info.get('progress', 0.0)
        final_time = final_info.get('time', 0.0)
        episode_time = final_time - start_time if start_time is not None else 0.0
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_progress.append(progress)
        episode_times.append(episode_time)
        episode_offtrack.append(int(offtrack_seen))
        episode_steer_variance.append(float(np.var(steer_values)) if len(steer_values) > 1 else 0.0)
        
        print(f"Episode {episode + 1}/{n_episodes}: "
              f"Reward={episode_reward:.2f}, "
              f"Length={episode_length}, "
              f"Progress={progress:.2%}, "
              f"Time={episode_time:.2f}s")
    
    # Close video writer
    if video_writer is not None:
        video_writer.release()
        print(f"Video saved to {video_path}")
    
    env.close()
    base_env.close()
    if show_window:
        cv2.destroyAllWindows()
    
    # Calculate statistics
    stats = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'min_reward': np.min(episode_rewards),
        'max_reward': np.max(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'std_length': np.std(episode_lengths),
        'mean_progress': np.mean(episode_progress),
        'std_progress': np.std(episode_progress),
        'mean_time': np.mean(episode_times),
        'std_time': np.std(episode_times),
        'offtrack_rate': np.mean(episode_offtrack),
        'mean_steer_variance': np.mean(episode_steer_variance),
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'episode_progress': episode_progress,
        'episode_times': episode_times,
        'episode_offtrack': episode_offtrack,
        'episode_steer_variance': episode_steer_variance,
    }
    
    return stats, video_path


def print_stats(stats):
    """Print evaluation statistics."""
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Mean Reward: {stats['mean_reward']:.2f} +/- {stats['std_reward']:.2f}")
    print(f"Reward Range: [{stats['min_reward']:.2f}, {stats['max_reward']:.2f}]")
    print(f"Mean Episode Length: {stats['mean_length']:.1f} +/- {stats['std_length']:.1f}")
    print(f"Mean Progress: {stats['mean_progress']:.2%} +/- {stats['std_progress']:.2%}")
    print(f"Mean Episode Time: {stats['mean_time']:.2f}s +/- {stats['std_time']:.2f}s")
    if 'offtrack_rate' in stats:
        print(f"Off-track Rate: {stats['offtrack_rate']:.2%}")
    if 'mean_steer_variance' in stats:
        print(f"Mean Steering Variance: {stats['mean_steer_variance']:.5f}")
    if 'mean_rank' in stats:
        print(f"Mean Rank: {stats['mean_rank']:.2f}")
    if 'collision_rate' in stats:
        print(f"Collision Rate: {stats['collision_rate']:.2%}")
    if 'mean_speed' in stats:
        print(f"Mean Speed: {stats['mean_speed']:.2f}")
    if 'mean_throttle' in stats:
        print(f"Mean Throttle: {stats['mean_throttle']:.2f}")
    if 'mean_brake' in stats:
        print(f"Mean Brake: {stats['mean_brake']:.2f}")
    if 'mean_overtakes' in stats:
        print(f"Mean Overtakes: {stats['mean_overtakes']:.2f}")
    print("="*50)


def save_stats(stats, output_path):
    """Save statistics to file."""
    # Convert numpy arrays to lists for JSON serialization
    stats_json = {
        'mean_reward': float(stats['mean_reward']),
        'std_reward': float(stats['std_reward']),
        'min_reward': float(stats['min_reward']),
        'max_reward': float(stats['max_reward']),
        'mean_length': float(stats['mean_length']),
        'std_length': float(stats['std_length']),
        'mean_progress': float(stats['mean_progress']),
        'std_progress': float(stats['std_progress']),
        'mean_time': float(stats['mean_time']),
        'std_time': float(stats['std_time']),
        'offtrack_rate': float(stats.get('offtrack_rate', 0.0)),
        'mean_steer_variance': float(stats.get('mean_steer_variance', 0.0)),
        'mean_rank': float(stats.get('mean_rank', 1.0)),
        'collision_rate': float(stats.get('collision_rate', 0.0)),
        'episode_rewards': [float(r) for r in stats['episode_rewards']],
        'episode_lengths': [int(l) for l in stats['episode_lengths']],
        'episode_progress': [float(p) for p in stats['episode_progress']],
        'episode_times': [float(t) for t in stats['episode_times']],
        'episode_offtrack': [int(v) for v in stats.get('episode_offtrack', [])],
        'episode_steer_variance': [float(v) for v in stats.get('episode_steer_variance', [])],
        'episode_mean_rank': [float(v) for v in stats.get('episode_mean_rank', [])],
        'episode_collision': [int(v) for v in stats.get('episode_collision', [])],
        'mean_speed': float(stats.get('mean_speed', 0.0)),
        'mean_throttle': float(stats.get('mean_throttle', 0.0)),
        'mean_brake': float(stats.get('mean_brake', 0.0)),
        'mean_overtakes': float(stats.get('mean_overtakes', 0.0)),
        'episode_mean_speed': [float(v) for v in stats.get('episode_mean_speed', [])],
        'episode_mean_throttle': [float(v) for v in stats.get('episode_mean_throttle', [])],
        'episode_mean_brake': [float(v) for v in stats.get('episode_mean_brake', [])],
        'episode_mean_overtakes': [int(v) for v in stats.get('episode_mean_overtakes', [])],
        'model_path': stats.get('model_path'),
        'config_path': stats.get('config_path'),
        'seed': int(stats.get('seed', 0)),
        'episodes': int(stats.get('episodes', len(stats.get('episode_rewards', [])))),
        'video_path': stats.get('video_path'),
        'evaluated_at': stats.get('evaluated_at'),
    }
    
    with open(output_path, 'w') as f:
        json.dump(stats_json, f, indent=2)
    
    print(f"Statistics saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained PPO model')
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model checkpoint'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/multi_car_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=10,
        help='Number of evaluation episodes'
    )
    parser.add_argument(
        '--no-video',
        action='store_true',
        help='Disable video recording'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for evaluation'
    )
    parser.add_argument(
        '--output-json',
        type=str,
        default=None,
        help='Optional explicit path for the saved JSON summary'
    )
    parser.add_argument(
        '--output-video',
        type=str,
        default=None,
        help='Optional explicit path for the saved evaluation video'
    )
    parser.add_argument(
        '--show-window',
        action='store_true',
        help='Display the OpenCV live window during evaluation'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Evaluate model (Torch .pt or SB3 .zip)
    model_path = Path(args.model)
    if model_path.suffix.lower() == ".pt":
        stats, video_path = evaluate_torch_model(
            model_path,
            config,
            n_episodes=args.episodes,
            record_video=not args.no_video,
            seed=args.seed,
            show_window=args.show_window,
            video_path=args.output_video,
        )
    else:
        stats, video_path = evaluate_model(
            args.model,
            config,
            n_episodes=args.episodes,
            record_video=not args.no_video,
            seed=args.seed,
            show_window=args.show_window,
            video_path=args.output_video,
        )
    stats["model_path"] = str(model_path.resolve())
    stats["config_path"] = str(Path(args.config).resolve())
    stats["seed"] = int(args.seed)
    stats["episodes"] = int(args.episodes)
    stats["video_path"] = str(video_path) if video_path else None
    stats["evaluated_at"] = datetime.now().isoformat()
    
    # Print statistics
    print_stats(stats)
    
    # Save statistics
    if args.output_json:
        stats_path = Path(args.output_json)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        results_dir = Path(config['paths']['results_dir'])
        stats_path = resolve_output_path(None, results_dir, model_path, args.seed, "evaluation_stats", "json")
    save_stats(stats, stats_path)


if __name__ == '__main__':
    main()

