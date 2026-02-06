# Racing Gym RL - PPO Training Project

A reinforcement learning project for training a Proximal Policy Optimization (PPO) agent to drive a racecar optimally on various tracks using the racecar_gym environment and Stable-Baselines3.

Google Colab: https://colab.research.google.com/drive/18B9vCAqM9xmiXearYYa3bdq415TytHJg?usp=sharing

## Project Overview

This project implements a complete training pipeline for a single-agent racecar using:
- **Environment**: racecar_gym (Gymnasium-compatible environment using PyBullet physics)
- **Algorithm**: Proximal Policy Optimization (PPO) from Stable-Baselines3
- **Track**: Circle track (configurable to other tracks)
- **Observation Space**: Dict containing lidar, pose, velocity, and acceleration sensors
- **Action Space**: Dict containing motor and steering actuators

## Project Structure

```
Racing_Gym_RL/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── train.py                  # Main training script
├── evaluate.py               # Model evaluation script
├── colab_training.ipynb      # Google Colab notebook for training
├── config/
│   └── circle_config.yaml    # Training configuration file
├── models/                   # Saved model checkpoints (gitignored)
├── logs/                     # Training logs and TensorBoard data (gitignored)
└── results/                  # Evaluation results and videos (gitignored)
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

Note: On first use, racecar_gym will automatically download track files. This may take a few minutes.

### Google Colab

The project includes a complete Colab notebook (`colab_training.ipynb`) that handles all setup automatically. Simply:

1. Upload the notebook to Google Colab
2. Enable GPU: Runtime -> Change runtime type -> GPU
3. Run all cells sequentially

The notebook will install all dependencies and create necessary directories automatically.

## Configuration

Training parameters are configured in `config/circle_config.yaml`. Key settings include:

- **Environment**: Track selection, rendering options
- **PPO Hyperparameters**: Learning rate, batch size, number of epochs, etc.
- **Policy Network**: Architecture and activation functions
- **Training**: Total timesteps, evaluation frequency, checkpoint saving

### Changing Tracks

To train on a different track, modify the `track` field in `config/circle_config.yaml`:
- `circle` - Simple circular track (good for initial testing)
- `austria` - Austria Formula 1 track
- `berlin` - Berlin track
- `montreal` - Montreal track
- `torino` - Torino track
- `plechaty` - Plechaty track

The environment ID will automatically be set to `SingleAgent{Track}-v0`.

## Usage

### Training

Train a new model:
```bash
python train.py --config config/circle_config.yaml --seed 42
```

Resume training from a checkpoint:
```bash
python train.py --config config/circle_config.yaml --resume models/ppo_racecar_50000_steps.zip
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
- `--config`: Configuration file (default: `config/circle_config.yaml`)
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

### Observation Space (Dict)

The agent receives observations from multiple sensors:

- **lidar**: Box(1080,) - LiDAR range scans (1080 points)
- **pose**: Box(6,) - Position (x, y, z) and orientation (roll, pitch, yaw)
- **velocity**: Box(6,) - Translational and rotational velocity components
- **acceleration**: Box(6,) - Translational and rotational acceleration components
- **time**: float - Current simulation time

### Action Space (Dict)

The agent controls the racecar through:

- **motor**: Box(low=-1, high=1, shape=(1,)) - Throttle command (-1 to 1)
- **steering**: Box(low=-1, high=1, shape=(1,)) - Steering angle (-1 to 1)

## Model Architecture

The PPO model uses:
- **Policy**: MultiInputPolicy (for Dict observation spaces)
- **Network Architecture**: Two hidden layers with 256 units each (configurable)
- **Activation Function**: Tanh (configurable)
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

1. Environment creation with specified track and sensors
2. Model initialization with MultiInputPolicy for Dict observations
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

**Import Error for racecar_gym**
- Ensure racecar_gym is installed: `pip install git+https://github.com/axelbr/racecar_gym.git`
- Tracks are downloaded automatically on first use

**CUDA/GPU Issues**
- Set `device: cpu` in config file to force CPU usage
- Check PyTorch CUDA installation: `python -c "import torch; print(torch.cuda.is_available())"`

**Dict Observation Space Errors**
- Ensure using MultiInputPolicy (automatically selected in train.py)
- Check that all required sensors are specified in config

**Memory Issues**
- Reduce `batch_size` or `n_steps` in config
- Use CPU instead of GPU if GPU memory is limited
- Reduce number of evaluation episodes

**Track Download Issues**
- Tracks download automatically, but if issues occur, manually download from:
  https://github.com/axelbr/racecar_gym/releases/download/tracks-v1.0.0/all.zip
- Extract to `~/.racecar_gym/scenes/` (or equivalent)

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

- gymnasium >= 0.29.0
- stable-baselines3[extra] >= 2.0.0
- numpy >= 1.24.0
- pybullet >= 3.2.0
- matplotlib >= 3.7.0
- opencv-python >= 4.8.0
- tensorboard >= 2.13.0
- pyyaml >= 6.0
- torch >= 2.0.0
- racecar_gym (from GitHub)

## References

- Racecar Gym: https://github.com/axelbr/racecar_gym
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
