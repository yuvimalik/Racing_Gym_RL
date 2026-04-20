# Racing Gym RL - PPO Training Project

A reinforcement learning project for training a Proximal Policy Optimization (PPO) agent to drive a car in the `multi_car_racing` environment using Stable-Baselines3.

Google Colab:[https://colab.research.google.com/github/yuvimalik/Racing_Gym_RL/blob/main/colab_training.ipynb]

## Project Overview

This project implements a complete training pipeline for a single-agent car racing setup using:
- **Environment**: `multi_car_racing` (Gym-based multi-agent car racing)
- **Algorithm**: Proximal Policy Optimization (PPO) from Stable-Baselines3
- **Observation Space**: 96x96 RGB image
- **Action Space**: Continuous controls (steer, gas, brake)

Current torch training ground truth:
- **Autoresearch run 008** is the promoted torch policy variant for long training.
- It uses a tanh-squashed Gaussian policy with tanh log-prob correction.
- The legacy torch policy remains available for comparison and regression testing.

## Project Structure

```
Racing_Gym_RL/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── train.py                       # Main training script
├── evaluate.py                    # Model evaluation script
├── colab_training.ipynb           # Google Colab notebook for training
├── config/
│   ├── circle_config.yaml         # Legacy config (racecar_gym)
│   └── multi_car_config.yaml      # Training configuration for multi_car_racing
├── models/                        # Saved model checkpoints (gitignored)
├── logs/                          # Training logs and TensorBoard data (gitignored)
└── results/                       # Evaluation results and videos (gitignored)
```

## Installation

### Local Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Racing_Gym_RL
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install `multi_car_racing` (no-deps to avoid shapely/box2d build issues):
```bash
pip install git+https://github.com/igilitschenski/multi_car_racing.git --no-deps
```

Note: `multi_car_racing` is installed from GitHub and registers the environment during import.

### Google Colab

The project includes a complete Colab notebook (`colab_training.ipynb`) that handles all setup automatically. Simply:

1. Upload the notebook to Google Colab
2. Enable GPU: Runtime -> Change runtime type -> GPU
3. Run all cells sequentially

The notebook will install all dependencies and create necessary directories automatically.

## Configuration

Training parameters are configured in `config/multi_car_config.yaml`. Key settings include:

- **Environment**: Track selection, rendering options
- **PPO Hyperparameters**: Learning rate, batch size, number of epochs, etc.
- **Policy Network**: Architecture and activation functions
- **Training**: Total timesteps, evaluation frequency, checkpoint saving

### Environment Options

Key options in `config/multi_car_config.yaml`:
- `num_agents`: Set to 1 for single-agent training
- `direction`: Track direction (`CW` or `CCW`)
- `use_random_direction`: Randomize direction
- `backwards_flag`, `h_ratio`, `use_ego_color`: Rendering/visual options

## Usage

### Training

Train a new model:
```bash
python train.py --config config/multi_car_config.yaml --seed 42
```

Resume training from a checkpoint:
```bash
python train.py --config config/multi_car_config.yaml --resume models/ppo_racecar_50000_steps.zip
```

Train with the promoted autoresearch policy variant explicitly:
```bash
python train.py --config config/multi_car_config.yaml --trainer_backend torch --torch_policy_variant autoresearch_run_008 --seed 42
```

Resume a long torch run from the promoted autoresearch checkpoint:
```bash
python train.py --config config/multi_car_config.yaml --trainer_backend torch --torch_policy_variant autoresearch_run_008 --resume autoresearch/results/best/final.pt --timesteps_add 500000
```

Run the legacy torch policy for comparison:
```bash
python train.py --config config/multi_car_config.yaml --trainer_backend torch --torch_policy_variant legacy --seed 42
```

Training will:
- Save checkpoints periodically (default: every 50,000 steps)
- Save the best model based on evaluation performance
- Log training metrics to TensorBoard
- Save final model upon completion

### Evaluation

Evaluate a trained model:
```bash
python evaluate.py --model models/best_model/best_model.zip --episodes 10
```

Options:
- `--model`: Path to model checkpoint
- `--config`: Configuration file (default: `config/multi_car_config.yaml`)
- `--episodes`: Number of evaluation episodes (default: 10)
- `--no-video`: Disable video recording
- `--seed`: Random seed for evaluation

Evaluation generates:
- Performance statistics (mean reward, episode length, progress, etc.)
- Video recording of agent performance (if enabled)
- JSON file with detailed metrics

### TensorBoard Visualization

View training progress:
```bash
tensorboard --logdir logs
```

Then open `http://localhost:6006` in your browser.

## World Model Status

The repository now also contains an offline PyTorch world-model pipeline under `world_model/`.

Current scope:
- Recurrent State Space Model (RSSM)
- frozen-world-model latent actor-critic baseline
- replay-based imagined rollout training plus real-environment actor evaluation
- telemetry-supervised continuation path for physics-aware latent modeling

Implemented pieces:
- vision encoder and decoder
- deterministic GRU sequence model
- stochastic prior and posterior latent models
- reward predictor
- offline replay loader and sequence sampling
- manual and automatic replay collection scripts
- offline RSSM training with checkpoint and hallucination video saving
- frozen RSSM control wrapper with latent-state flattening
- latent actor and critic MLP heads
- imagined latent-rollout actor-critic training script
- real-environment actor evaluation on top of the frozen RSSM
- reward and physics-faithfulness evaluation on held-out replay

Useful entry points:
```bash
python world_model_autoencoder.py --config config/world_model_config.yaml
python world_model_prepare_dataset.py --config config/world_model_config.yaml
python world_model_train.py --config config/world_model_config.yaml --epochs 10
python world_model_train_control.py --config config/world_model_config.yaml --epochs 10
python scripts/evaluate_reward_faithfulness.py --config config/world_model_config.yaml --manifest results/world_model/replay/d4_pilot_val_manifest.json --world-model-checkpoint models/world_model/rssm_sequence.pt
python world_model_collect_replay.py --config config/world_model_config.yaml --split-prefix d4pilot
python world_model_collect_replay.py --config config/world_model_config.yaml --manual --manual_target_frames 12000 --manual_target_episodes 5 --manual_direction CCW --manual_regime harsh_turns_fast --manual_split d5_manual_hard_turns_ccw_fast --render --record_video
```

Current qualitative status:
- straight-line hallucinations are broadly coherent
- car, grass, road corridor, and bottom HUD-like structure are usually preserved
- corner consistency is the main current weakness
- latent-control training now assumes the RSSM is frozen and uses it as a differentiable simulator
- `E3` improved geometry persistence over `E2`, but the dominant remaining failure mode is world/ego inconsistency
- the actor-critic diagnostic showed that imagined short-horizon optimization can look healthy while still failing to transfer in the real environment
- `P5` is the current best stable checkpoint
- later experiments showed that adding too much noisy manual hard-turn data at once can visibly regress decoder quality
- curated manual turn data is useful diagnostically, but data mixing alone did not solve sharp-turn hallucination
- the next planned intervention is a perceptual-loss fine-tune from `P5` with corrected progress supervision
- actor-critic is paused until the world model earns better sharp-turn geometry

Current training defaults for the local RTX 4070 Laptop GPU:
- world-model training uses replay windowing, CUDA AMP, and worker-based loading
- latent-control training uses a short real context plus imagined rollouts from the frozen RSSM
- telemetry-supervised world-model training adds auxiliary heads for speed, progress delta, steer, corner angle, and offtrack probability
- for targeted turn adaptation, prefer `config/world_model_local_p5_finetune.yaml` over a full continuation config
- for the next intervention stage, the mainline is objective-level correction rather than a larger fixed-objective hero run

Artifacts:
- checkpoints: `models/world_model/<run_name>/`
- latest checkpoint: `models/world_model/rssm_sequence.pt`
- hallucinations: `results/world_model/artifacts/hallucination/<run_name>/`
- latest hallucination: `results/world_model/artifacts/hallucination/hallucination.mp4`
- latent-control checkpoints: `models/world_model_control/<run_name>/`
- latest latent-control checkpoint: `models/world_model_control/latent_actor_critic.pt`
- latent-control metrics: `results/world_model/control/<run_name>/`
- physics-faithfulness summaries: `results/world_model/control/*.json`

See `WORLD_MODEL_PROGRESS.md` for a concise progress log and next-step roadmap.
See `WORLD_MODEL_INTERVENTION_1.md` for the active execution handoff for the next world-model intervention.

## Observation and Action Spaces

### Observation Space

The agent receives a 96x96 RGB image observation:
- **shape**: `(96, 96, 3)`
- **type**: uint8 image

### Action Space

The agent controls the car via continuous actions:
- **steering**: left/right steering
- **gas**: acceleration
- **brake**: braking force

## Model Architecture

The PPO model uses:
- **Policy**: CnnPolicy (for image observations)
- **Device**: Automatically detects GPU/CPU availability

### Torch Policy Variants

`train.py` supports two local torch policy variants:
- `autoresearch_run_008`: current ground-truth policy discovered by autoresearch; uses a tanh-squashed Gaussian with corrected PPO log-prob
- `legacy`: previous torch policy using unconstrained Gaussian outputs with `tanh/sigmoid` environment mapping

The main config `config/multi_car_config.yaml` now defaults to:
- `training.trainer_backend: torch`
- `training.torch_policy_variant: autoresearch_run_008`

## Training Details

### PPO Hyperparameters (Default)

- Learning Rate: 3.0e-4
- Steps per Update: 2048
- Batch Size: 64
- Epochs per Update: 10
- Discount Factor (gamma): 0.99
- GAE Lambda: 0.95
- Clip Range: 0.2
- Entropy Coefficient: 0.01
- Value Function Coefficient: 0.5

### Training Process

1. Environment creation with `MultiCarRacing-v0` and `num_agents=1`
2. Model initialization with `CnnPolicy`
3. Training loop with periodic evaluation
4. Automatic checkpointing and best model saving
5. TensorBoard logging for monitoring

## Results Interpretation

### Evaluation Metrics

- **Mean Reward**: Average cumulative reward per episode
- **Episode Length**: Average number of steps per episode
- **Progress**: Average track progress (0.0 to 1.0, where 1.0 = one complete lap)
- **Episode Time**: Average simulation time per episode

### Performance Indicators

- Higher reward indicates better performance
- Longer episodes suggest the agent stays on track longer
- Progress approaching 1.0 indicates successful lap completion
- Consistent performance across episodes shows stable learning

## Troubleshooting

### Common Issues

**Import Error for multi_car_racing**
- Ensure it is installed: `pip install git+https://github.com/igilitschenski/multi_car_racing.git`
- Make sure `gym` is installed (not just gymnasium)

**CUDA/GPU Issues**
- Set `device: cpu` in config file to force CPU usage
- Check PyTorch CUDA installation: `python -c "import torch; print(torch.cuda.is_available())"`

**Image Observation Issues**
- Ensure `CnnPolicy` is used (default in config)
- Ensure `VecTransposeImage` is applied (handled in `train.py`)

**Memory Issues**
- Reduce `batch_size` or `n_steps` in config
- Use CPU instead of GPU if GPU memory is limited
- Reduce number of evaluation episodes

**Environment Not Found**
- Ensure `gym_multi_car_racing` is imported before `gym.make`

## Google Colab Specific Notes

- GPU is recommended for faster training
- Training progress is displayed in real-time
- Results can be downloaded as a zip file
- TensorBoard is integrated for visualization
- Configuration is created automatically if not present

## Fine-tuning and Optimization

After initial training, consider:

1. **Hyperparameter Tuning**: Adjust learning rate, batch size, network architecture
2. **Reward Shaping**: Modify reward function in environment (requires custom scenario)
3. **Observation Space**: Experiment with different sensor combinations
4. **Training Duration**: Increase total timesteps for better performance
5. **Different Tracks**: Test generalization across different track layouts

## Dependencies

- numpy >= 1.22.0,<1.23.0
- gym == 0.17.3
- stable-baselines3[extra] == 1.8.0
- matplotlib >= 3.7.0
- opencv-python >= 4.8.0
- tensorboard >= 2.13.0
- pyyaml >= 6.0
- pyglet == 1.5.27
- torch >= 2.0.0
- multi_car_racing (from GitHub)

## References

- Multi-Car Racing: https://github.com/igilitschenski/multi_car_racing
- Stable-Baselines3: https://github.com/DLR-RM/stable-baselines3
- Gymnasium: https://gymnasium.farama.org/

## License

This project is for educational purposes as part of a reinforcement learning course.

## Changelog

### Initial Setup
- Created training pipeline with PPO from Stable-Baselines3
- Implemented support for Dict observation/action spaces
- Added evaluation script with video recording
- Created Google Colab notebook for cloud training
- Configured Circle track for initial training
- Set up comprehensive documentation
