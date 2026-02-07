"""
Training script for Multi-Car Racing using PPO from Stable-Baselines3.

This script handles single-agent training on multi_car_racing with
image observations and continuous actions.
"""

import os
import yaml
import argparse
from pathlib import Path
import gym
import gym_multi_car_racing
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.utils import set_random_seed
import torch
import time
from datetime import datetime, timedelta


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
        return obs, reward, done, info


def create_env(config, rank=0, seed=0):
    """Create and wrap the multi_car_racing environment."""
    env_config = config['environment']

    env_id = env_config.get('env_id', 'MultiCarRacing-v0')
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
    
    # Wrap with Monitor for logging
    log_dir = config['paths']['log_dir']
    os.makedirs(log_dir, exist_ok=True)
    env = Monitor(env, filename=os.path.join(log_dir, f'monitor_{rank}'))
    
    # Set seed
    try:
        env.reset(seed=seed)
    except TypeError:
        if hasattr(env, "seed"):
            env.seed(seed)
        env.reset()
    
    return env


def get_device(config):
    """Determine the device to use for training."""
    device_config = config.get('device', 'auto')
    
    if device_config == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
    elif device_config == 'cuda':
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            print("WARNING: CUDA requested but not available. Falling back to CPU.")
            device = 'cpu'
    else:
        device = device_config
    
    # Print device info
    if device == 'cuda':
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    else:
        print("Using CPU for training")
    
    return device


class ProgressCallback(BaseCallback):
    """Custom callback to display training progress with percentage and ETA."""
    
    def __init__(self, total_timesteps, eval_freq, verbose=1):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.eval_freq = eval_freq
        self.start_time = None
        self.last_log_time = None
        self.last_log_timestep = 0
        
    def _on_training_start(self) -> None:
        """Called when training starts."""
        self.start_time = time.time()
        self.last_log_time = time.time()
        print("\n" + "="*70)
        print("TRAINING STARTED")
        print("="*70)
        print(f"Total timesteps: {self.total_timesteps:,}")
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70 + "\n")
        
    def _on_step(self) -> bool:
        """Called at each step."""
        # Log progress periodically
        if self.num_timesteps % self.eval_freq == 0:
            self._log_progress()
        return True
    
    def _log_progress(self):
        """Log training progress with percentage and ETA."""
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        elapsed_since_last = current_time - self.last_log_time
        
        # Calculate progress
        progress_pct = (self.num_timesteps / self.total_timesteps) * 100
        timesteps_since_last = self.num_timesteps - self.last_log_timestep
        
        # Calculate speed
        if elapsed_since_last > 0:
            steps_per_sec = timesteps_since_last / elapsed_since_last
        else:
            steps_per_sec = 0
        
        # Calculate ETA
        remaining_timesteps = self.total_timesteps - self.num_timesteps
        if steps_per_sec > 0:
            eta_seconds = remaining_timesteps / steps_per_sec
            eta = timedelta(seconds=int(eta_seconds))
        else:
            eta = timedelta(seconds=0)
        
        # Format elapsed time
        elapsed = timedelta(seconds=int(elapsed_time))
        
        print("\n" + "-"*70)
        print(f"Progress: {progress_pct:.2f}% ({self.num_timesteps:,} / {self.total_timesteps:,} timesteps)")
        print(f"Elapsed: {str(elapsed)} | ETA: {str(eta)}")
        print(f"Speed: {steps_per_sec:.1f} steps/sec")
        print("-"*70)
        
        self.last_log_time = current_time
        self.last_log_timestep = self.num_timesteps
    
    def _on_training_end(self) -> None:
        """Called when training ends."""
        total_time = time.time() - self.start_time
        total_elapsed = timedelta(seconds=int(total_time))
        print("\n" + "="*70)
        print("TRAINING COMPLETED")
        print("="*70)
        print(f"Total time: {str(total_elapsed)}")
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70 + "\n")


def create_model(config, env, device):
    """Create PPO model for image-based observations."""
    ppo_config = config['ppo']
    policy_config = config['policy']

    policy_type = policy_config.get('policy_type', 'CnnPolicy')
    policy_kwargs = None
    if policy_type != 'CnnPolicy':
        activation_fn_map = {
            'tanh': torch.nn.Tanh,
            'relu': torch.nn.ReLU,
            'elu': torch.nn.ELU
        }
        activation_fn = activation_fn_map.get(
            policy_config.get('activation_fn', 'tanh'),
            torch.nn.Tanh
        )
        policy_kwargs = dict(
            net_arch=policy_config.get('net_arch', [256, 256]),
            activation_fn=activation_fn
        )

    model = PPO(
        policy=policy_type,
        env=env,
        learning_rate=ppo_config['learning_rate'],
        n_steps=ppo_config['n_steps'],
        batch_size=ppo_config['batch_size'],
        n_epochs=ppo_config['n_epochs'],
        gamma=ppo_config['gamma'],
        gae_lambda=ppo_config['gae_lambda'],
        clip_range=ppo_config['clip_range'],
        ent_coef=ppo_config['ent_coef'],
        vf_coef=ppo_config['vf_coef'],
        max_grad_norm=ppo_config['max_grad_norm'],
        use_sde=ppo_config.get('use_sde', False),
        sde_sample_freq=ppo_config.get('sde_sample_freq', -1),
        policy_kwargs=policy_kwargs,
        device=device,
        verbose=1,
        tensorboard_log=config['paths']['log_dir']
    )
    
    return model


def main():
    parser = argparse.ArgumentParser(description='Train PPO agent on Multi-Car Racing')
    parser.add_argument(
        '--config',
        type=str,
        default='config/multi_car_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume training from'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set random seeds
    set_random_seed(args.seed)
    
    # Create directories
    model_dir = Path(config['paths']['model_dir'])
    log_dir = Path(config['paths']['log_dir'])
    results_dir = Path(config['paths']['results_dir'])
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Get device
    device = get_device(config)
    print("\n" + "="*70)
    print("TRAINING CONFIGURATION")
    print("="*70)
    print(f"Config file: {args.config}")
    print(f"Device: {device}")
    print(f"Seed: {args.seed}")
    if args.resume:
        print(f"Resuming from: {args.resume}")
    print("="*70 + "\n")
    
    # Create environment
    print("Creating environment...")
    env_id = config['environment'].get('env_id', 'MultiCarRacing-v0')
    print(f"Environment ID: {env_id}")
    
    # Wrap in vectorized environment (required for SB3)
    env = DummyVecEnv([lambda: create_env(config, rank=0, seed=args.seed)])
    env = VecTransposeImage(env)
    
    # Create evaluation environment
    eval_env = DummyVecEnv([lambda: create_env(config, rank=1, seed=args.seed + 1000)])
    eval_env = VecTransposeImage(eval_env)
    print("Environment created successfully!\n")
    
    # Create model
    print("Creating PPO model...")
    if args.resume:
        print(f"Loading model from {args.resume}")
        model = PPO.load(args.resume, env=env, device=device)
    else:
        model = create_model(config, env.envs[0], device)
    
    # Print model info
    print("\n" + "-"*70)
    print("MODEL INFORMATION")
    print("-"*70)
    print(f"Observation space: {env.envs[0].observation_space}")
    print(f"Action space: {env.envs[0].action_space}")
    print(f"Policy: {model.policy}")
    
    # Print hyperparameters
    ppo_config = config['ppo']
    print("\n" + "-"*70)
    print("PPO HYPERPARAMETERS")
    print("-"*70)
    print(f"Learning rate: {ppo_config['learning_rate']}")
    print(f"Steps per update: {ppo_config['n_steps']}")
    print(f"Batch size: {ppo_config['batch_size']}")
    print(f"Epochs per update: {ppo_config['n_epochs']}")
    print(f"Gamma (discount): {ppo_config['gamma']}")
    print(f"GAE lambda: {ppo_config['gae_lambda']}")
    print(f"Clip range: {ppo_config['clip_range']}")
    print(f"Entropy coefficient: {ppo_config['ent_coef']}")
    print(f"Value function coefficient: {ppo_config['vf_coef']}")
    print(f"Max gradient norm: {ppo_config['max_grad_norm']}")
    print("-"*70 + "\n")
    
    # Setup callbacks
    training_config = config['training']
    
    # Progress callback for percentage and ETA
    progress_callback = ProgressCallback(
        total_timesteps=training_config['total_timesteps'],
        eval_freq=training_config['eval_freq']
    )
    
    # Evaluation callback with custom logging
    class LoggingEvalCallback(EvalCallback):
        """Extended EvalCallback with better logging."""
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.eval_count = 0
            
        def _on_step(self) -> bool:
            if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
                self.eval_count += 1
                print(f"\n{'='*70}")
                print(f"EVALUATION #{self.eval_count} (Step {self.num_timesteps:,})")
                print(f"{'='*70}")
            return super()._on_step()
        
        def _on_evaluation_end(self, locals_, globals_):
            """Log evaluation results."""
            if 'mean_reward' in locals_:
                mean_rew = locals_['mean_reward']
                std_rew = locals_.get('std_reward', 0)
                print(f"Mean reward: {mean_rew:.2f} ± {std_rew:.2f}")
                print(f"{'='*70}\n")
            return super()._on_evaluation_end(locals_, globals_)
    
    eval_callback = LoggingEvalCallback(
        eval_env,
        best_model_save_path=str(model_dir / 'best_model'),
        log_path=str(log_dir),
        eval_freq=training_config['eval_freq'],
        n_eval_episodes=training_config['n_eval_episodes'],
        deterministic=True,
        render=False
    )
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=training_config['save_freq'],
        save_path=str(model_dir),
        name_prefix='ppo_racecar'
    )
    
    # Combine callbacks
    callbacks = CallbackList([progress_callback, eval_callback, checkpoint_callback])
    
    # Print training settings
    print("-"*70)
    print("TRAINING SETTINGS")
    print("-"*70)
    print(f"Total timesteps: {training_config['total_timesteps']:,}")
    print(f"Evaluation frequency: {training_config['eval_freq']:,} steps")
    print(f"Evaluation episodes: {training_config['n_eval_episodes']}")
    print(f"Checkpoint frequency: {training_config['save_freq']:,} steps")
    print(f"Model directory: {model_dir}")
    print(f"Log directory: {log_dir}")
    print(f"TensorBoard logs: tensorboard --logdir {log_dir}")
    print("-"*70 + "\n")
    
    # Train model
    model.learn(
        total_timesteps=training_config['total_timesteps'],
        callback=callbacks,
        log_interval=training_config.get('log_interval', 10),
        progress_bar=True
    )
    
    # Save final model
    final_model_path = model_dir / 'final_model'
    model.save(str(final_model_path))
    
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    print(f"Final model saved to: {final_model_path}")
    print(f"Best model saved to: {model_dir / 'best_model' / 'best_model.zip'}")
    print(f"Checkpoints saved to: {model_dir}")
    print(f"TensorBoard logs: {log_dir}")
    print("="*70 + "\n")
    
    # Close environments
    env.close()
    eval_env.close()
    print("Environments closed. Training complete!")


if __name__ == '__main__':
    main()
