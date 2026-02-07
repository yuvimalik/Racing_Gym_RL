"""
Evaluation script for trained Multi-Car Racing PPO models.

This script loads a trained model and evaluates its performance,
generating metrics and optionally recording videos.
"""

import os
import yaml
import argparse
import numpy as np
from pathlib import Path
import gym
import gym_multi_car_racing
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
import cv2
from datetime import datetime


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


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
            action = action.reshape(1, -1)
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


def create_env(config, render_mode='rgb_array'):
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

    if env_config.get('num_agents', 1) == 1:
        env = SingleAgentWrapper(env)
    
    return env


def evaluate_model(model_path, config, n_episodes=10, record_video=True, seed=42):
    """Evaluate a trained model."""
    print(f"Loading model from {model_path}")
    model = PPO.load(model_path)
    
    # Create environment
    # Note: render_mode parameter is ignored in create_env for Gym 0.17.3 compatibility
    # Rendering is handled via env.render() calls
    base_env = create_env(config, render_mode='rgb_array' if record_video else None)
    env = DummyVecEnv([lambda: base_env])
    env = VecTransposeImage(env)
    
    # Evaluation metrics
    episode_rewards = []
    episode_lengths = []
    episode_progress = []
    episode_times = []
    
    results_dir = Path(config['paths']['results_dir'])
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Create video writer lazily after first frame
    video_writer = None
    video_path = None
    if record_video:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path = results_dir / f'evaluation_{timestamp}.mp4'
    
    print(f"Evaluating model for {n_episodes} episodes...")
    
    for episode in range(n_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        start_time = None
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            
            obs, reward, done, info = env.step(action)
            
            episode_reward += float(reward[0])
            episode_length += 1
            
            # Record first frame time
            info0 = info[0] if isinstance(info, (list, tuple)) else info
            if episode == 0 and episode_length == 1:
                print(f"DEBUG info type: {type(info)}, info0 type: {type(info0)}", flush=True)
                if isinstance(info0, dict):
                    print(f"DEBUG info0 keys: {list(info0.keys())}", flush=True)
                else:
                    print(f"DEBUG info0 content: {info0}", flush=True)
            if start_time is None and isinstance(info0, dict) and 'time' in info0:
                start_time = info0['time']
            
            # Get frame for video or live view
            frame = base_env.render(mode='rgb_array')
            
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
            if frame is not None:
                # Convert if not already done
                if 'frame_bgr' not in locals():
                     frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # Debug frame stats on first frame
                if episode == 0 and episode_length == 1:
                    print(f"DEBUG Frame stats: dtype={frame.dtype}, min={np.min(frame)}, max={np.max(frame)}", flush=True)
                
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
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'episode_progress': episode_progress,
        'episode_times': episode_times
    }
    
    return stats, video_path


def print_stats(stats):
    """Print evaluation statistics."""
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Mean Reward: {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
    print(f"Reward Range: [{stats['min_reward']:.2f}, {stats['max_reward']:.2f}]")
    print(f"Mean Episode Length: {stats['mean_length']:.1f} ± {stats['std_length']:.1f}")
    print(f"Mean Progress: {stats['mean_progress']:.2%} ± {stats['std_progress']:.2%}")
    print(f"Mean Episode Time: {stats['mean_time']:.2f}s ± {stats['std_time']:.2f}s")
    print("="*50)


def save_stats(stats, output_path):
    """Save statistics to file."""
    import json
    
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
        'episode_rewards': [float(r) for r in stats['episode_rewards']],
        'episode_lengths': [int(l) for l in stats['episode_lengths']],
        'episode_progress': [float(p) for p in stats['episode_progress']],
        'episode_times': [float(t) for t in stats['episode_times']]
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
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Evaluate model
    stats, video_path = evaluate_model(
        args.model,
        config,
        n_episodes=args.episodes,
        record_video=not args.no_video,
        seed=args.seed
    )
    
    # Print statistics
    print_stats(stats)
    
    # Save statistics
    results_dir = Path(config['paths']['results_dir'])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stats_path = results_dir / f'evaluation_stats_{timestamp}.json'
    save_stats(stats, stats_path)


if __name__ == '__main__':
    main()
