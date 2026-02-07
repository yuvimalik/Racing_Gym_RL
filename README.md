# Racing Gym RL - PPO Training Project

A reinforcement learning project for training a Proximal Policy Optimization (PPO) agent to drive a car in the `multi_car_racing` environment using Stable-Baselines3.

Google Colab: https://colab.research.google.com/drive/18B9vCAqM9xmiXearYYa3bdq415TytHJg?usp=sharing

## Project Overview

This project implements a complete training pipeline for a single-agent car racing setup using:
- **Environment**: `multi_car_racing` (Gym-based multi-agent car racing)
- **Algorithm**: Proximal Policy Optimization (PPO) from Stable-Baselines3
- **Observation Space**: 96x96 RGB image
- **Action Space**: Continuous controls (steer, gas, brake)

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
