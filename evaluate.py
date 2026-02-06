"""
Evaluation script for trained Racecar Gym PPO models.

This script loads a trained model and evaluates its performance,
generating metrics and optionally recording videos.
"""

import os
import yaml
import argparse
import numpy as np
from pathlib import Path
import gymnasium as gym
import racecar_gym.envs.gym_api
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import cv2
from datetime import datetime


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_env(config, render_mode='rgb_array'):
    """Create evaluation environment with rendering."""
    env_config = config['environment']
    track = env_config['track']
    env_id = f'SingleAgent{track.capitalize()}-v0'
    
    env = gym.make(
        env_id,
        render_mode=render_mode,
        render_options=env_config.get('render_options', {})
    )
    
    return env


def evaluate_model(model_path, config, n_episodes=10, record_video=True, seed=42):
    """Evaluate a trained model."""
    print(f"Loading model from {model_path}")
    model = PPO.load(model_path)
    
    # Create environment
    env = create_env(config, render_mode='rgb_array' if record_video else None)
    
    # Evaluation metrics
    episode_rewards = []
    episode_lengths = []
    episode_progress = []
    episode_times = []
    
    results_dir = Path(config['paths']['results_dir'])
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Create video writer if recording
    video_writer = None
    video_path = None
    if record_video:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path = results_dir / f'evaluation_{timestamp}.mp4'
        render_options = config['environment'].get('render_options', {})
        width = render_options.get('width', 320)
        height = render_options.get('height', 240)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(str(video_path), fourcc, 30.0, (width, height))
    
    print(f"Evaluating model for {n_episodes} episodes...")
    
    for episode in range(n_episodes):
        obs, info = env.reset(seed=seed + episode)
        done = False
        episode_reward = 0
        episode_length = 0
        start_time = None
        
        while not done:
            # Get action from model
            if isinstance(obs, dict):
                # Handle Dict observation
                action, _ = model.predict(obs, deterministic=True)
            else:
                action, _ = model.predict(obs, deterministic=True)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_length += 1
            
            # Record first frame time
            if start_time is None and 'time' in info:
                start_time = info['time']
            
            # Record video frame
            if record_video and video_writer is not None:
                frame = env.render()
                if frame is not None:
                    # Convert RGB to BGR for OpenCV
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    video_writer.write(frame_bgr)
        
        # Extract final metrics from info
        final_info = info
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
        default='config/circle_config.yaml',
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
