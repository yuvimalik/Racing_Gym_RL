# Racing Gym RL - PPO Training Project

A reinforcement learning project for training policies in the `multi_car_racing` environment. The repo supports the legacy single-agent SB3 flow and a maintained local torch path for single-agent and multi-agent training.

Google Colab:
- PPO training notebook: [https://colab.research.google.com/github/yuvimalik/Racing_Gym_RL/blob/main/colab_training.ipynb]
- Executable PPO + world-model demo: [https://colab.research.google.com/github/yuvimalik/Racing_Gym_RL/blob/main/colab_ppo_world_model_demo.ipynb]

## Project Overview

This project implements a complete training pipeline for car racing using:
- **Environment**: `multi_car_racing` (Gym-based multi-agent car racing)
- **Algorithms**:
  - Stable-Baselines3 PPO for the legacy single-agent path
  - Local torch PPO for the maintained single-agent and multi-agent paths
- **Observation Space**: 96x96 RGB image
- **Action Space**: Continuous controls (steer, gas, brake)

Current torch training ground truth:
- **Autoresearch run 008** is the promoted torch policy variant for long training.
- It uses a tanh-squashed Gaussian policy with tanh log-prob correction.
- The legacy torch policy remains available for comparison and regression testing.
- Multi-agent training uses `training.trainer_backend: torch`, `training.marl_paradigm: shared_policy_ippo`, and `config/multi_car_marl_config.yaml`.

## Project Structure

```
Racing_Gym_RL/
├── README.md                      # This file
├── requirements.txt               # Python dependencies (gym 0.17, env stack)
├── requirements_sb3.txt           # Stable-Baselines3 (install with --no-deps after requirements.txt)
├── train.py                       # Main training script
├── evaluate.py                    # Model evaluation script
├── distributed_train.py           # CUDA-only distributed torch entry point
├── colab_training.ipynb           # Google Colab notebook for training
├── config/
│   ├── circle_config.yaml         # Legacy config (racecar_gym)
│   ├── multi_car_config.yaml      # Single-agent torch config
│   └── multi_car_marl_config.yaml # Shared-policy multi-agent torch config
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

3. Install dependencies (two steps so `pip` can keep **gym==0.17.3** for `multi_car_racing` while still installing Stable-Baselines3):
```bash
pip install -r requirements.txt
pip install -r requirements_sb3.txt --no-deps
```

4. Install `multi_car_racing` (no-deps to avoid shapely/box2d build issues):
```bash
pip install git+https://github.com/igilitschenski/multi_car_racing.git --no-deps
```

Note: `multi_car_racing` is installed from GitHub and registers the environment during import.

5. If you want to use the autoresearch loops, add an API key in `.env`:
```bash
# Gemini
GOOGLE_API_KEY=...

# OpenAI / Codex-compatible provider path
OPENAI_API_KEY=...
```

### Google Colab

The project includes:

- `colab_training.ipynb` for the main PPO/autoresearch Colab flow
- `colab_ppo_world_model_demo.ipynb` for a smaller, executable PPO + world-model demo that:
  - evaluates the tracked PPO checkpoint
  - summarizes tracked world-model benchmark results
  - collects a tiny replay and trains a one-epoch world-model smoke run in Colab

For either notebook:

1. Upload the notebook to Google Colab
2. Enable GPU: Runtime -> Change runtime type -> GPU
3. Run all cells sequentially

The notebooks install dependencies and create their own output directories automatically.

## Configuration

Training parameters are configured in YAML files under `config/`.

Recommended entry points:
- `config/multi_car_config.yaml`: single-agent torch training
- `config/multi_car_marl_config.yaml`: shared-policy multi-agent torch training

Key settings include:

- **Environment**: Track selection, rendering options
- **PPO Hyperparameters**: Learning rate, batch size, number of epochs, etc.
- **Policy Network**: Architecture and activation functions
- **Training**: Total timesteps, evaluation frequency, checkpoint saving

### Environment Options

Key options in the multi-car configs:
- `num_agents`: `1` for single-agent training, `>1` for shared-policy multi-agent training
- `direction`: Track direction (`CW` or `CCW`)
- `use_random_direction`: Randomize direction
- `backwards_flag`, `h_ratio`, `use_ego_color`: Rendering/visual options

## Usage

### Training

Train a new single-agent torch model:
```bash
python train.py --config config/multi_car_config.yaml --seed 42
```

Train a multi-agent torch model:
```bash
python train.py --config config/multi_car_marl_config.yaml --seed 42
```

Run a short MARL smoke test before any long run:
```bash
python train.py --config config/multi_car_marl_smoke_config.yaml --seed 42
```

Cloud GPU (e.g. Prime Intellect): long-run, budget (~$10–20 window when calibrated), and smoke configs plus Docker/runbook are in [docs/PRIME_INTELLECT_GPU.md](docs/PRIME_INTELLECT_GPU.md) (`config/prime_marl_2car_long.yaml`, `config/prime_marl_2car_budget.yaml`, `config/prime_marl_2car_smoke.yaml`).

Resume a torch checkpoint:
```bash
python train.py --config config/multi_car_marl_config.yaml --resume models/v3_marl_control_bootstrap/best_model_torch.pt --seed 42
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

### Autoresearch Providers

The autoresearch entrypoints now support both Gemini and OpenAI/Codex-style APIs through a shared provider adapter.

Examples:

Run the classic single-agent autoresearch loop with Gemini:
```bash
python -m autoresearch.run_loop --provider gemini --model gemini-2.5-flash
```

Run the classic single-agent autoresearch loop with OpenAI:
```bash
python -m autoresearch.run_loop --provider openai --model gpt-4.1-mini
```

Run the MARL recursive loop with OpenAI:
```bash
python -m autoresearch.run_marl_recursive --provider openai --model gpt-4.1-mini --base-config config/multi_car_marl_config.yaml --results-subdir research_branches/control_recovery
```

Fixed warm-start MARL search (every candidate fine-tunes the same checkpoint; promoted bundle under `autoresearch/results/.../promoted/`):
```bash
python -m autoresearch.marl_search_loop --provider openai --model gpt-4o-mini --warm-start-ckpt models/v3_marl_control_bootstrap/best_model_torch.pt --base-config config/multi_car_marl_config.yaml --results-subdir research_branches/my_search
```
Long-running helper: `./run_marl_search.sh` (set `WARM_START_CKPT`, `BASE_CONFIG`, `OPENAI_API_KEY`).

If you pass `--provider codex`, the loop uses the OpenAI backend and expects `OPENAI_API_KEY`.

Distributed multi-GPU torch training is available through `distributed_train.py`, but it is currently CUDA-only and keeps `MultiCarRacing-v0` at one `DummyVecEnv` per rank:
```bash
torchrun --nproc_per_node=NUM_GPUS distributed_train.py --config config/multi_car_marl_config.yaml --seed 42
```

Training will:
- Save checkpoints periodically (default: every 50,000 steps)
- Save the best model based on evaluation performance
- Log torch training metrics to TensorBoard and JSONL files
- Save final model upon completion
- Write a run manifest and summary under `results/.../torch_<timestamp>_seed<seed>/`

### Recommended MARL Training Ladder

The current `models/v2_marl_reward/final_model_torch.pt` checkpoint is not a good resume point for racing behavior. It can be visualized, but it should be treated as a failed control branch rather than a strong multi-agent baseline.

Recommended sequence:

1. Smoke-test the control-first MARL config:
```bash
python train.py --config config/multi_car_marl_smoke_config.yaml --seed 42
```

2. Evaluate the smoke run's best checkpoint:
```bash
python evaluate.py --model models/v3_marl_control_bootstrap_smoke/best_model_torch.pt --config config/multi_car_marl_smoke_config.yaml --episodes 5 --no-video --seed 42
```

Inspect the anti-contact metrics in that eval before promoting the policy:
- `contact_rate`: how often an episode had real overlap/contact.
- `hook_contact_rate`: how often cars got stuck in wheel-hooking style contact.
- `contact_termination_rate`: how often the stricter anti-contact cutoff ended the interaction.
- `mean_max_contact_steps`: how long the worst contact streak lasted per episode.

3. If the smoke run passes, launch the longer control-first MARL run:
```bash
python train.py --config config/multi_car_marl_config.yaml --seed 42
```

4. Resume from the best checkpoint, not the final checkpoint:
```bash
python train.py --config config/multi_car_marl_config.yaml --resume models/v3_marl_control_bootstrap/best_model_torch.pt --seed 42 --timesteps_add 500000
```

Go / no-go criteria after the smoke run:
- Stop and retune if eval still shows near-zero steering variance, `100%` off-track, or progress stuck near zero.
- Stop and retune if `contact_rate` or `hook_contact_rate` stays materially above zero, or if `mean_max_contact_steps` shows that cars are still leaning on each other for multiple steps.
- Continue if progress becomes clearly non-zero, steering variance stays alive, and off-track rate trends down from the current failure case.
- Prefer `best_model_torch.pt` over `final_model_torch.pt` whenever periodic eval is enabled.

### Evaluation

Evaluate a trained SB3 `.zip` model:
```bash
python evaluate.py --model models/best_model/best_model.zip --episodes 10
```

Evaluate a torch `.pt` checkpoint, including multi-agent checkpoints:
```bash
python evaluate.py --model models/v3_marl_control_bootstrap/best_model_torch.pt --config config/multi_car_marl_config.yaml --episodes 10
```

Watch a multi-agent checkpoint live in a window without writing a video:
```bash
python evaluate.py --model models/v3_marl_control_bootstrap/best_model_torch.pt --config config/multi_car_marl_config.yaml --episodes 1 --show-window --no-video --seed 42
```

Save a multi-agent evaluation video to an explicit path:
```bash
python evaluate.py --model models/v3_marl_control_bootstrap/best_model_torch.pt --config config/multi_car_marl_config.yaml --episodes 5 --seed 42 --output-video results/v3_marl_control_bootstrap/best_model_torch_seed42_evaluation.mp4
```

Save a headless JSON summary to a specific path:
```bash
python evaluate.py --model models/v3_marl_control_bootstrap/best_model_torch.pt --config config/multi_car_marl_config.yaml --episodes 5 --no-video --output-json results/v3_marl_control_bootstrap/manual_eval_best.json
```

Options:
- `--model`: Path to model checkpoint
- `--config`: Configuration file (default: `config/multi_car_config.yaml`)
- `--episodes`: Number of evaluation episodes (default: 10)
- `--no-video`: Disable video recording
- `--output-video`: Explicit path for the saved evaluation video
- `--output-json`: Explicit path for the saved evaluation summary
- `--seed`: Random seed for evaluation
- `--show-window`: Show the OpenCV evaluation window during evaluation

If `--output-video` or `--output-json` is omitted, `evaluate.py` now uses deterministic default names under the configured `results_dir` based on the checkpoint stem and seed. Re-running the same command with the same seed overwrites the same artifacts, which makes comparison easier.

Evaluation generates:
- Performance statistics (mean reward, episode length, progress, and MARL metrics such as rank/collision/contact when present)
- Video recording of agent performance (if enabled)
- JSON file with detailed metrics
- Metadata including checkpoint path, config path, seed, and video path when present

For `MultiCarRacing-v0` multi-agent evaluation, the renderer returns one frame per car. `evaluate.py` now tiles those per-car views side-by-side into a single RGB frame so both live viewing and MP4 export work reliably.

Interpretation note for showcase runs:
- A successful video export does not mean the policy is good.
- If eval shows near-zero progress, `100%` off-track, or near-zero steering variance, the checkpoint is not showcase-ready even if the viewer works.
- For strict anti-contact runs, treat non-trivial `contact_rate`, `hook_contact_rate`, or repeated contact terminations as a sign that the cars are still trying to interfere with each other instead of race cleanly.
- The current `models/v2_marl_reward/final_model_torch.pt` checkpoint should be treated as a debugging artifact rather than a good racing demo unless later evals improve materially.

### TensorBoard Visualization

View training progress:
```bash
tensorboard --logdir logs
```

Then open `http://localhost:6006` in your browser.

Generate saved figures from a completed run:
```bash
python plot_marl_results.py --run-dir results/v2_marl_reward/torch_<timestamp>_seed42
```

## World Model Status

The repository now also contains an offline PyTorch world-model pipeline under `world_model/`.

Current scope:
- Recurrent State Space Model (RSSM)
- frozen-world-model latent actor-critic baseline
- replay-based imagined rollout training plus real-environment actor evaluation

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

Useful entry points:
```bash
python world_model_autoencoder.py --config config/world_model_config.yaml
python world_model_prepare_dataset.py --config config/world_model_config.yaml
python world_model_train.py --config config/world_model_config.yaml --epochs 10
python world_model_train_control.py --config config/world_model_config.yaml --epochs 10
```

Current qualitative status:
- straight-line hallucinations are broadly coherent
- car, grass, road corridor, and bottom HUD-like structure are usually preserved
- corner consistency is the main current weakness
- latent-control training now assumes the RSSM is frozen and uses it as a differentiable simulator

Current training defaults for the local RTX 4070 Laptop GPU:
- world-model training uses replay windowing, CUDA AMP, and worker-based loading
- latent-control training uses a short real context plus imagined rollouts from the frozen RSSM

Artifacts:
- checkpoints: `models/world_model/<run_name>/`
- latest checkpoint: `models/world_model/rssm_sequence.pt`
- hallucinations: `results/world_model/artifacts/hallucination/<run_name>/`
- latest hallucination: `results/world_model/artifacts/hallucination/hallucination.mp4`
- latent-control checkpoints: `models/world_model_control/<run_name>/`
- latest latent-control checkpoint: `models/world_model_control/latent_actor_critic.pt`
- latent-control metrics: `results/world_model/control/<run_name>/`

See `WORLD_MODEL_PROGRESS.md` for a concise progress log and next-step roadmap.
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

The multi-agent config `config/multi_car_marl_config.yaml` additionally sets:
- `environment.num_agents: 2`
- `training.marl_paradigm: shared_policy_ippo`

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

1. Environment creation with `MultiCarRacing-v0` and the configured `num_agents`
2. Space wrapping so the torch trainer sees consistent per-agent image/action tensors
3. Shared-policy PPO rollout collection and updates across all agent streams
4. Periodic evaluation, checkpointing, and best-model saving
5. TensorBoard plus JSONL logging for monitoring and later visualization

### Multi-Agent Notes

- Multi-agent training uses one shared actor-critic policy across all cars.
- Trainer timestep accounting is per agent stream. With `num_envs=1` and `num_agents=2`, each simulator step advances the trainer by `2` timesteps.
- `MultiCarRacing-v0` is currently supported only with a single `DummyVecEnv` in `train.py`.
- Multi-agent eval can run through a subprocess (`training.eval_subprocess: true`) so evaluation writes saved JSON artifacts without reusing the training process viewer state.
- On macOS, requesting `device: cuda` falls back to `mps` or `cpu` when CUDA is unavailable.
- `distributed_train.py` is CUDA-only and uses one `DummyVecEnv` per rank for `MultiCarRacing-v0`.

### Run Artifacts

Each torch MARL run now writes a self-contained results directory such as `results/v2_marl_reward/torch_20260402_153000_seed42/` with:
- `run_manifest.json`: resolved config and run metadata
- `run_summary.json`: final summary with key artifact paths
- `training_metrics.jsonl`: per-update losses, KL, clip fraction, learning rate, throughput, and step semantics
- `episode_summaries.jsonl`: episodic reward and length records when available
- `torch_eval_history.jsonl`: scalar eval history
- `evaluations/*.json`: saved checkpoint evaluation summaries
- `plots/*.png`: figures generated by `plot_marl_results.py`

## Results Interpretation

### Evaluation Metrics

- **Mean Reward**: Average cumulative reward per episode
- **Episode Length**: Average number of steps per episode
- **Progress**: Average track progress (0.0 to 1.0, where 1.0 = one complete lap)
- **Episode Time**: Average simulation time per episode
- **Mean Rank / Collision Rate**: Multi-agent metrics reported by torch evaluation when `num_agents > 1`

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
- On macOS, `device: cuda` in this repo falls back to `mps` or `cpu`

**Image Observation Issues**
- Ensure `CnnPolicy` is used (default in config)
- Ensure `VecTransposeImage` is applied only for single-agent image training (handled in `train.py`)

**Memory Issues**
- Reduce `batch_size` or `n_steps` in config
- Use CPU instead of GPU if GPU memory is limited
- Reduce number of evaluation episodes

**Environment Not Found**
- Ensure `gym_multi_car_racing` is imported before `gym.make`

**Multi-Agent Launch Issues**
- Use `training.trainer_backend: torch`; SB3 multi-agent training/evaluation is not supported in this repo
- Use `config/multi_car_marl_config.yaml` for multiple cars
- Evaluate multi-agent checkpoints with torch `.pt` files, not SB3 `.zip` files
- Keep `training.num_envs: 1` for `MultiCarRacing-v0` in the main trainer

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
