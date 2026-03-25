"""
LOCKED — Environment factory + evaluation harness for autoresearch.

DO NOT MODIFY THIS FILE. The autoresearch agent edits only train_ppo.py.
This module provides:
  - create_env()          — builds a single wrapped env instance
  - create_training_envs() — returns SubprocVecEnv for training
  - create_eval_env()     — returns DummyVecEnv for evaluation
  - evaluate()            — deterministic evaluation harness

Contract: policy must have .act(obs_tensor, deterministic=True) -> (action, _, _)
          and a static method raw_to_env_action(raw) -> env_action tensor.
"""

import os
import platform
import time
from pathlib import Path

import gym
import gym_multi_car_racing
import numpy as np
import torch
import yaml
from gym_multi_car_racing import multi_car_racing as mcr
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecTransposeImage


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Wrappers (extracted from train.py — Phase 1 optimizations included)
# ---------------------------------------------------------------------------

class SingleAgentWrapper(gym.Wrapper):
    """Wrap MultiCarRacing to expose a single-agent view."""

    def __init__(self, env):
        super().__init__(env)
        obs_space = env.observation_space
        act_space = env.action_space

        if len(obs_space.shape) == 4 and obs_space.shape[0] == 1:
            self.observation_space = gym.spaces.Box(
                low=obs_space.low[0], high=obs_space.high[0],
                shape=obs_space.shape[1:], dtype=obs_space.dtype,
            )
        if len(act_space.shape) == 2 and act_space.shape[0] == 1:
            self.action_space = gym.spaces.Box(
                low=act_space.low[0], high=act_space.high[0],
                shape=act_space.shape[1:], dtype=act_space.dtype,
            )

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        if hasattr(obs, "shape") and len(obs.shape) == 4 and obs.shape[0] == 1:
            obs = obs[0]
        elif isinstance(obs, (list, tuple)) and len(obs) == 1:
            obs = obs[0]
        return obs

    def render(self, mode="human", **kwargs):
        out = self.env.render(mode=mode, **kwargs)
        if hasattr(out, "shape") and len(out.shape) == 4 and out.shape[0] == 1:
            out = out[0]
        return out

    def step(self, action):
        if hasattr(self.env.action_space, "shape") and len(self.env.action_space.shape) == 2:
            action = action.reshape(1, -1)
        obs, reward, done, info = self.env.step(action)
        if hasattr(obs, "shape") and len(obs.shape) == 4 and obs.shape[0] == 1:
            obs = obs[0]
        elif isinstance(obs, (list, tuple)) and len(obs) == 1:
            obs = obs[0]
        if isinstance(reward, (list, tuple)) or (
            hasattr(reward, "shape") and len(reward.shape) > 0 and reward.shape[0] == 1
        ):
            reward = float(reward[0] if isinstance(reward, (list, tuple)) else reward[0])
        return obs, reward, done, info


class RewardShapingWrapper(gym.Wrapper):
    """Minimal reward shaping — Phase 1 optimized (cached track, gated rewards)."""

    def __init__(self, env, reward_config):
        super().__init__(env)
        rc = reward_config or {}
        self.enabled = bool(rc.get("enabled", True))
        self.use_custom_reward = bool(rc.get("use_custom_reward", True))

        self.forward_progress_scale = float(rc.get("forward_progress_scale", 1.0))
        self.track_alignment_scale = float(rc.get("track_alignment_scale", 0.0))
        self.straight_speed_scale = float(rc.get("straight_speed_scale", 0.05))
        self.sharp_turn_threshold = float(rc.get("sharp_turn_threshold", 0.35))
        self.sharp_turn_lookahead = int(rc.get("sharp_turn_lookahead", 6))
        self.corner_target_speed = float(rc.get("corner_target_speed", 8.0))
        self.corner_overspeed_penalty_scale = float(rc.get("corner_overspeed_penalty_scale", 0.6))
        self.apex_decel_reward_scale = float(rc.get("apex_decel_reward_scale", 0.4))
        self.apex_decel_reward_cap = float(rc.get("apex_decel_reward_cap", 1.0))
        self.time_penalty = float(rc.get("time_penalty", -0.1))
        self.steer_smoothness_penalty = float(rc.get("steer_smoothness_penalty", 0.05))
        self.steer_delta_cap = float(rc.get("steer_delta_cap", 0.5))
        self.lateral_velocity_penalty = float(rc.get("lateral_velocity_penalty", 0.0))
        self.steer_magnitude_penalty = float(rc.get("steer_magnitude_penalty", 0.0))
        self.idle_speed_threshold = float(rc.get("idle_speed_threshold", 1.5))
        self.idle_penalty = float(rc.get("idle_penalty", -0.4))
        self.throttle_bonus_scale = float(rc.get("throttle_bonus_scale", 0.0))
        self.brake_penalty_scale = float(rc.get("brake_penalty_scale", 0.0))
        self.launch_boost_steps = int(rc.get("launch_boost_steps", 0))
        self.launch_speed_target = float(rc.get("launch_speed_target", 4.0))
        self.launch_bonus_scale = float(rc.get("launch_bonus_scale", 0.0))
        self.stuck_speed_threshold = float(rc.get("stuck_speed_threshold", 1.2))
        self.stuck_progress_epsilon = float(rc.get("stuck_progress_epsilon", 1e-3))
        self.stuck_max_steps = int(rc.get("stuck_max_steps", 120))
        self.stuck_terminal_penalty = float(rc.get("stuck_terminal_penalty", -50.0))
        self.no_progress_max_steps = int(rc.get("no_progress_max_steps", 200))
        self.no_progress_terminal_penalty = float(rc.get("no_progress_terminal_penalty", -15.0))
        self.yaw_rate_penalty = float(rc.get("yaw_rate_penalty", 0.0))
        self.off_track_mode = str(rc.get("off_track_mode", "terminate")).strip().lower()
        self.off_track_terminal_penalty = float(rc.get("off_track_terminal_penalty", -100.0))
        self.off_track_step_penalty = float(rc.get("off_track_step_penalty", -25.0))
        self.curriculum = rc.get("curriculum", {}) or {}
        self.curriculum_enabled = bool(self.curriculum.get("enabled", False))
        self.curriculum_stage = int(self.curriculum.get("start_stage", 1))

        self._prev_steer = None
        self._prev_speed = None
        self._lap_count = 0
        self._prev_progress = None
        self._lap_completion_latched = False
        self._stuck_steps = 0
        self._no_progress_steps = 0
        self._last_stuck_progress = None
        self._episode_steps = 0
        self._training_mode = True

        # Phase 1A: cached track geometry
        self._track_xy = None
        self._track_betas = None
        self._track_dirs = None
        self._n_track = 0
        self._last_track_index = 0

        # Phase 1B: precompute whether we need track context
        self._needs_track_context = (
            self.track_alignment_scale != 0.0
            or self.straight_speed_scale != 0.0
            or self.corner_overspeed_penalty_scale != 0.0
            or self.apex_decel_reward_scale != 0.0
            or self.lateral_velocity_penalty != 0.0
        )

    def set_training_mode(self, mode: bool):
        self._training_mode = bool(mode)

    def reset(self, **kwargs):
        self._prev_steer = None
        self._prev_speed = None
        self._lap_count = 0
        self._prev_progress = None
        self._lap_completion_latched = False
        self._stuck_steps = 0
        self._no_progress_steps = 0
        self._last_stuck_progress = None
        self._episode_steps = 0
        self._last_track_index = 0
        obs = self.env.reset(**kwargs)

        # Cache track geometry once per episode
        base_env = self.env.unwrapped
        if hasattr(base_env, "track") and base_env.track and len(base_env.track) >= 2:
            raw = base_env.track
            self._n_track = len(raw)
            self._track_xy = np.array([t[2:] for t in raw], dtype=np.float32)
            self._track_betas = np.array([t[1] for t in raw], dtype=np.float32)
            next_xy = np.roll(self._track_xy, -1, axis=0)
            diff = next_xy - self._track_xy
            norms = np.linalg.norm(diff, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-8)
            self._track_dirs = (diff / norms).astype(np.float32)
        else:
            self._track_xy = None
            self._track_betas = None
            self._track_dirs = None
            self._n_track = 0
        return obs

    def _get_track_context(self, car_pos):
        if self._track_xy is None or self._n_track < 2:
            return np.zeros(2, dtype=np.float32), False, 0.0
        search_radius = 30
        center = self._last_track_index
        n = self._n_track
        indices = np.arange(center - search_radius, center + search_radius + 1) % n
        local_xy = self._track_xy[indices]
        dists = (local_xy[:, 0] - car_pos[0]) ** 2 + (local_xy[:, 1] - car_pos[1]) ** 2
        best_local = int(np.argmin(dists))
        track_index = int(indices[best_local])
        self._last_track_index = track_index
        track_dir = self._track_dirs[track_index]
        lookahead_index = (track_index + self.sharp_turn_lookahead) % n
        beta_now = float(self._track_betas[track_index])
        beta_next = float(self._track_betas[lookahead_index])
        angle_diff = abs(beta_next - beta_now)
        if angle_diff > np.pi:
            angle_diff = abs(angle_diff - 2 * np.pi)
        is_sharp_turn = angle_diff >= self.sharp_turn_threshold
        return track_dir, is_sharp_turn, float(angle_diff)

    def _update_lap_count(self, info):
        progress = info.get("progress")
        if progress is None:
            return
        try:
            progress = float(progress)
        except (TypeError, ValueError):
            return
        if self._prev_progress is not None and progress < (self._prev_progress - 0.5):
            self._lap_count += 1
        if progress >= 0.999 and not self._lap_completion_latched:
            self._lap_count += 1
            self._lap_completion_latched = True
        if progress < 0.2:
            self._lap_completion_latched = False
        self._prev_progress = progress

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if not isinstance(info, dict):
            info = {}
        self._episode_steps += 1

        base_env = self.env.unwrapped
        if (hasattr(base_env, "tile_visited_count") and hasattr(base_env, "track")
                and base_env.track and len(base_env.track) > 0):
            info["progress"] = float(base_env.tile_visited_count[0]) / float(len(base_env.track))

        self._update_lap_count(info)

        # Consolidated base_env access
        speed = 0.0
        track_dir = np.zeros(2, dtype=np.float32)
        is_sharp_turn = False
        corner_angle = 0.0
        velocity_vec = np.zeros(2, dtype=np.float32)
        yaw_rate = 0.0
        if hasattr(base_env, "cars") and base_env.cars:
            car = base_env.cars[0]
            vel = car.hull.linearVelocity
            velocity_vec = np.array([vel[0], vel[1]], dtype=np.float32)
            speed = float(np.linalg.norm(velocity_vec))
            yaw_rate = abs(float(car.hull.angularVelocity))
            if self._needs_track_context:
                car_pos = np.array([car.hull.position[0], car.hull.position[1]], dtype=np.float32)
                track_dir, is_sharp_turn, corner_angle = self._get_track_context(car_pos)

        action_arr = np.asarray(action).reshape(-1) if action is not None else np.zeros(3, dtype=np.float32)
        steer_value = float(np.clip(action_arr[0], -1.0, 1.0)) if action_arr.size >= 1 else 0.0
        throttle_value = float(np.clip(action_arr[1], 0.0, 1.0)) if action_arr.size >= 2 else 0.0
        brake_value = float(np.clip(action_arr[2], 0.0, 1.0)) if action_arr.size >= 3 else 0.0

        progress_now = info.get("progress")
        progress_delta = 0.0
        if progress_now is not None:
            if self._last_stuck_progress is not None:
                progress_delta = max(0.0, progress_now - self._last_stuck_progress)
            self._last_stuck_progress = progress_now

        driving_on_grass = getattr(base_env, "driving_on_grass", None)
        is_offtrack = bool(
            driving_on_grass is not None
            and len(driving_on_grass) > 0
            and bool(driving_on_grass[0])
        )

        comp_forward = self.forward_progress_scale * progress_delta

        comp_alignment = 0.0
        if self.track_alignment_scale != 0.0:
            comp_alignment = self.track_alignment_scale * float(np.dot(velocity_vec, track_dir))

        comp_straight_speed = 0.0
        if self.straight_speed_scale != 0.0 and not is_sharp_turn:
            comp_straight_speed = self.straight_speed_scale * speed * (0.0 if is_offtrack else 1.0)

        comp_lateral = 0.0
        if self.lateral_velocity_penalty != 0.0:
            lateral_dir = np.array([-track_dir[1], track_dir[0]], dtype=np.float32)
            lateral_speed = float(np.dot(velocity_vec, lateral_dir))
            comp_lateral = -self.lateral_velocity_penalty * abs(lateral_speed)

        comp_corner_overspeed = 0.0
        if self.corner_overspeed_penalty_scale != 0.0 and is_sharp_turn:
            corner_overspeed = max(0.0, speed - self.corner_target_speed)
            comp_corner_overspeed = -self.corner_overspeed_penalty_scale * corner_overspeed

        comp_apex_decel = 0.0
        if self.apex_decel_reward_scale != 0.0 and is_sharp_turn:
            speed_delta = (self._prev_speed - speed) if self._prev_speed is not None else 0.0
            if speed_delta > 0.0:
                comp_apex_decel = min(self.apex_decel_reward_scale * speed_delta, self.apex_decel_reward_cap)

        comp_steer_smooth = 0.0
        if self.steer_smoothness_penalty != 0.0 and self._prev_steer is not None:
            steer_delta = abs(steer_value - self._prev_steer)
            if self.steer_delta_cap > 0.0:
                steer_delta = min(steer_delta, self.steer_delta_cap)
            comp_steer_smooth = -self.steer_smoothness_penalty * steer_delta

        comp_steer_mag = 0.0
        if self.steer_magnitude_penalty != 0.0:
            comp_steer_mag = -self.steer_magnitude_penalty * (steer_value ** 2)
        self._prev_steer = steer_value
        self._prev_speed = speed

        comp_time = float(self.time_penalty)
        comp_idle = float(self.idle_penalty) if speed < self.idle_speed_threshold else 0.0
        comp_throttle = float(self.throttle_bonus_scale * throttle_value) if self.throttle_bonus_scale != 0.0 else 0.0
        comp_brake = 0.0
        if self.brake_penalty_scale != 0.0 and not is_sharp_turn:
            comp_brake = -float(self.brake_penalty_scale * brake_value)
        comp_launch = 0.0
        if self.launch_bonus_scale != 0.0 and self._episode_steps <= self.launch_boost_steps and speed < self.launch_speed_target:
            comp_launch = float(self.launch_bonus_scale * throttle_value)
        comp_yaw = -self.yaw_rate_penalty * yaw_rate

        shaped_reward = (
            comp_forward + comp_alignment + comp_straight_speed + comp_lateral
            + comp_corner_overspeed + comp_apex_decel + comp_steer_smooth
            + comp_steer_mag + comp_time + comp_idle + comp_throttle
            + comp_brake + comp_launch + comp_yaw
        )

        total_reward = shaped_reward if self.use_custom_reward else float(reward)
        if is_offtrack:
            if self.off_track_mode == "terminate":
                total_reward = float(self.off_track_terminal_penalty)
                done = True
            else:
                total_reward += float(self.off_track_step_penalty)

        if speed < self.stuck_speed_threshold and progress_delta < self.stuck_progress_epsilon:
            self._stuck_steps += 1
        else:
            self._stuck_steps = 0
        is_stuck = self._stuck_steps >= self.stuck_max_steps
        if is_stuck:
            total_reward = float(self.stuck_terminal_penalty)
            done = True

        if progress_delta < self.stuck_progress_epsilon:
            self._no_progress_steps += 1
        else:
            self._no_progress_steps = 0
        is_no_progress = (not is_stuck) and self._no_progress_steps >= self.no_progress_max_steps
        if is_no_progress:
            total_reward = float(self.no_progress_terminal_penalty)
            done = True

        info["_track_index"] = self._last_track_index
        info["events/offtrack"] = int(is_offtrack)
        info["events/stuck"] = int(is_stuck)
        info["telemetry/speed"] = float(speed)
        if not self._training_mode:
            info["events/no_progress"] = int(is_no_progress)
            info["telemetry/yaw_rate"] = float(yaw_rate)
            info["rewards/total"] = float(total_reward)
            info["lap_count"] = int(self._lap_count)
        return obs, float(total_reward), done, info


class SafetyGovernorWrapper(gym.Wrapper):
    """Optional speed cap to keep the agent below a target velocity."""

    def __init__(self, env, governor_config):
        super().__init__(env)
        governor_config = governor_config or {}
        self.enabled = bool(governor_config.get("enabled", False))
        self.speed_cap_ratio = float(governor_config.get("speed_cap_ratio", 0.5))
        self.speed_cap_top_speed = float(governor_config.get("speed_cap_top_speed", 30.0))
        self.speed_cap_brake = float(governor_config.get("speed_cap_brake", 0.2))

    def step(self, action):
        if self.enabled and action is not None:
            base_env = self.env.unwrapped
            if hasattr(base_env, "cars") and base_env.cars:
                car = base_env.cars[0]
                vel = car.hull.linearVelocity
                speed = float(np.linalg.norm([vel[0], vel[1]]))
                speed_cap = self.speed_cap_ratio * self.speed_cap_top_speed
                if speed_cap > 0.0 and speed > speed_cap:
                    action_arr = np.asarray(action).copy()
                    orig_shape = action_arr.shape
                    action_arr = action_arr.reshape(-1)
                    if action_arr.size >= 3:
                        action_arr[1] = 0.0
                        action_arr[2] = max(float(action_arr[2]), self.speed_cap_brake)
                    action = action_arr.reshape(orig_shape)
        return self.env.step(action)


# ---------------------------------------------------------------------------
# Environment factories
# ---------------------------------------------------------------------------

def create_env(config: dict, rank: int = 0, seed: int = 0, training_mode: bool = True):
    """Create and wrap a single multi_car_racing environment."""
    env_config = config["environment"]

    env = gym.make(
        env_config.get("env_id", "MultiCarRacing-v0"),
        num_agents=env_config.get("num_agents", 1),
        direction=env_config.get("direction", "CCW"),
        use_random_direction=env_config.get("use_random_direction", True),
        backwards_flag=env_config.get("backwards_flag", True),
        h_ratio=env_config.get("h_ratio", 0.25),
        use_ego_color=env_config.get("use_ego_color", False),
    )
    max_episode_steps = env_config.get("max_episode_steps", None)
    if max_episode_steps is not None:
        env = gym.wrappers.TimeLimit(env.unwrapped, max_episode_steps=int(max_episode_steps))

    if env_config.get("num_agents", 1) == 1:
        env = SingleAgentWrapper(env)

    governor_config = config.get("safety_governor", {})
    if governor_config.get("enabled", False):
        env = SafetyGovernorWrapper(env, governor_config)

    reward_config = config.get("reward_shaping", {})
    if reward_config.get("enabled", False):
        env = RewardShapingWrapper(env, reward_config)
        if training_mode:
            env.set_training_mode(True)
        else:
            env.set_training_mode(False)

    log_dir = config["paths"]["log_dir"]
    os.makedirs(log_dir, exist_ok=True)
    env = Monitor(env, filename=os.path.join(log_dir, f"monitor_{rank}"))

    try:
        env.reset(seed=seed)
    except TypeError:
        if hasattr(env, "seed"):
            env.seed(seed)
        env.reset()

    return env


def _make_env(config: dict, rank: int, seed: int, training_mode: bool = True):
    def _init():
        return create_env(config, rank=rank, seed=seed + rank, training_mode=training_mode)
    return _init


def create_training_envs(config: dict, n_envs: int, seed: int = 42,
                         use_subproc: bool = False) -> VecTransposeImage:
    """Return a VecTransposeImage-wrapped VecEnv for training.

    Args:
        use_subproc: If True, use SubprocVecEnv (faster but can deadlock on Windows).
                     Default False = DummyVecEnv (safe, single-process).
    """
    import sys as _sys
    env_fns = [_make_env(config, rank=i, seed=seed, training_mode=True) for i in range(n_envs)]
    is_windows = platform.system().lower().startswith("win")
    if use_subproc and is_windows:
        print("[prepare] SubprocVecEnv requested on Windows; forcing DummyVecEnv for safety.", file=_sys.stderr, flush=True)
        use_subproc = False
    if use_subproc and n_envs > 1:
        print(f"[prepare] Creating SubprocVecEnv with {n_envs} workers...", file=_sys.stderr, flush=True)
        env = SubprocVecEnv(env_fns)
    else:
        if n_envs > 1:
            print(
                f"[prepare] Creating DummyVecEnv with {n_envs} envs (serial stepping; Windows-safe but CPU-bound).",
                file=_sys.stderr,
                flush=True,
            )
        else:
            print(f"[prepare] Creating DummyVecEnv with {n_envs} envs...", file=_sys.stderr, flush=True)
        env = DummyVecEnv(env_fns)
    return VecTransposeImage(env)


def create_eval_env(config: dict, seed: int = 42) -> VecTransposeImage:
    """Return a single DummyVecEnv for deterministic evaluation."""
    env = DummyVecEnv([_make_env(config, rank=0, seed=seed, training_mode=False)])
    return VecTransposeImage(env)


# ---------------------------------------------------------------------------
# Evaluation harness
# ---------------------------------------------------------------------------

def evaluate(policy, device, config: dict, n_episodes: int = 20, seed: int = 42) -> dict:
    """Run deterministic evaluation and return metrics.

    Args:
        policy: Must have .act(obs_tensor, deterministic=True) -> (action, _, _)
                and a static method raw_to_env_action(raw) -> env_action tensor.
        device: torch device for inference.
        config: Full config dict.
        n_episodes: Number of evaluation episodes.
        seed: Random seed for reproducibility.

    Returns:
        dict with: mean_reward, std_reward, mean_progress, offtrack_rate,
                   mean_speed, mean_episode_length, wall_clock_seconds.
    """
    import sys as _sys

    print(f"[eval] Creating eval env...", file=_sys.stderr, flush=True)
    eval_env = create_eval_env(config, seed=seed)
    print(f"[eval] Running {n_episodes} eval episodes (deterministic)...", file=_sys.stderr, flush=True)

    episode_rewards = []
    episode_lengths = []
    episode_progresses = []
    offtrack_counts = []
    speed_sums = []
    steer_variances = []
    throttle_means = []
    brake_means = []

    t0 = time.time()

    for ep in range(n_episodes):
        obs = eval_env.reset()
        ep_reward = 0.0
        ep_len = 0
        ep_offtrack = 0
        ep_speed_sum = 0.0
        steer_values = []
        throttle_values = []
        brake_values = []
        done = False

        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device) / 255.0
            with torch.inference_mode():
                raw_action, _, _ = policy.act(obs_t.unsqueeze(0) if obs_t.dim() == 3 else obs_t,
                                              deterministic=True)
            env_action = policy.raw_to_env_action(raw_action)
            action_np = env_action.cpu().numpy().reshape(-1)
            if action_np.size >= 1:
                steer_values.append(float(action_np[0]))
            if action_np.size >= 2:
                throttle_values.append(float(action_np[1]))
            if action_np.size >= 3:
                brake_values.append(float(action_np[2]))

            obs, reward, done_arr, infos = eval_env.step(np.array([action_np]))
            done = bool(done_arr[0])
            info = infos[0] if isinstance(infos, (list, tuple)) else infos

            ep_reward += float(reward[0]) if hasattr(reward, "__len__") else float(reward)
            ep_len += 1
            ep_offtrack += int(info.get("events/offtrack", 0))
            ep_speed_sum += float(info.get("telemetry/speed", 0.0))

        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_len)
        progress = info.get("progress", 0.0)
        episode_progresses.append(float(progress) if progress is not None else 0.0)
        offtrack_counts.append(ep_offtrack)
        speed_sums.append(ep_speed_sum / max(ep_len, 1))
        steer_variances.append(float(np.var(steer_values)) if len(steer_values) > 1 else 0.0)
        throttle_means.append(float(np.mean(throttle_values)) if throttle_values else 0.0)
        brake_means.append(float(np.mean(brake_values)) if brake_values else 0.0)

        print(f"[eval] Episode {ep+1}/{n_episodes}: reward={ep_reward:.1f} len={ep_len} "
              f"progress={episode_progresses[-1]:.3f}",
              file=_sys.stderr, flush=True)

    eval_env.close()
    wall_clock = time.time() - t0
    print(f"[eval] Done in {wall_clock:.1f}s | mean_reward={np.mean(episode_rewards):.2f}",
          file=_sys.stderr, flush=True)

    return {
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "mean_progress": float(np.mean(episode_progresses)),
        "offtrack_rate": float(np.mean([c / max(l, 1) for c, l in zip(offtrack_counts, episode_lengths)])),
        "mean_speed": float(np.mean(speed_sums)),
        "mean_episode_length": float(np.mean(episode_lengths)),
        "mean_steer_variance": float(np.mean(steer_variances)),
        "mean_throttle": float(np.mean(throttle_means)),
        "mean_brake": float(np.mean(brake_means)),
        "std_episode_length": float(np.std(episode_lengths)),
        "wall_clock_seconds": wall_clock,
    }
