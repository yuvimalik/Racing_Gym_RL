# Racing Gym RL - Final Project

Racing Gym RL is a reinforcement learning final project for training and analyzing racing agents in the `multi_car_racing` environment. The repository includes a maintained PyTorch PPO training pipeline, a legacy Stable-Baselines3 path, shared-policy multi-agent racing experiments, an offline RSSM world-model branch, final report materials, and an executable Colab demo.

Team: Yuv Malik, Pablo Echevarria, Matthew Lobo

## Final Submission Contents

- Final written report: [`docs/index.html`](docs/index.html), viewable through GitHub Pages when published.
- Final presentation slides: [`slides/final_project_slides.pdf`](slides/final_project_slides.pdf).
- Executable Colab demo: [`notebooks/colab_ppo_world_model_demo.ipynb`](notebooks/colab_ppo_world_model_demo.ipynb).
- Main training notebook: [`notebooks/colab_training.ipynb`](notebooks/colab_training.ipynb).
- Main PPO checkpoint for quick evaluation: [`models/best_model_torch.pt`](models/best_model_torch.pt).
- World-model benchmark summary: [`demo_assets/world_model_benchmark_summary.json`](demo_assets/world_model_benchmark_summary.json).
- Supporting histories: [`TRAINING_HISTORY.md`](TRAINING_HISTORY.md), [`WORLD_MODEL_PROGRESS.md`](WORLD_MODEL_PROGRESS.md), and [`WORLD_MODEL_HANDOFF.md`](WORLD_MODEL_HANDOFF.md).

Open the Colab demo directly:

- [PPO + world-model demo](https://colab.research.google.com/github/yuvimalik/Racing_Gym_RL/blob/main/notebooks/colab_ppo_world_model_demo.ipynb)
- [Training notebook](https://colab.research.google.com/github/yuvimalik/Racing_Gym_RL/blob/main/notebooks/colab_training.ipynb)

## Motivation

The project studies how far image-based PPO can be pushed in a racing simulator, and what failure modes appear when the task expands from single-car control to multi-agent racing and then to learned world-model prediction. Rather than presenting a single champion policy, the repository documents a progression: stabilize PPO control, diagnose reward and action-distribution failures, add multi-agent safety/evaluation infrastructure, and use an RSSM world model to identify sharp-turn pose evolution as a remaining bottleneck.

## High-Level Approach

The work has three connected parts:

1. Single-agent PPO control from `96 x 96` RGB observations with continuous steering, throttle, and brake actions.
2. Shared-policy multi-agent PPO/IPPO where one visual policy is trained over per-agent observation streams and evaluated with progress, rank, overtake, contact, and spacing metrics.
3. Offline world-model experiments using a recurrent state-space model (RSSM) trained on replay data to reconstruct observations, predict reward/telemetry, and generate short imagined rollouts.

The repository also includes AutoResearch experiment loops. These are preserved as project methodology and experiment history, but the standard reproduction path uses the explicit training, evaluation, and notebook commands below.

## Repository Structure

```text
Racing_Gym_RL/
  README.md                         # Final-submission guide
  requirements.txt                  # Core environment dependencies
  requirements_sb3.txt              # SB3 install layer, installed with --no-deps
  train.py                          # Main PPO training entry point
  evaluate.py                       # Evaluation, metrics, and video export
  distributed_train.py              # CUDA-only distributed training entry point
  config/                           # PPO, MARL, Prime, and world-model configs
  world_model/                      # RSSM/world-model implementation
  scripts/                          # Utility scripts for demos, datasets, metrics, plots
  autoresearch/                     # Automated experiment-loop code and selected results
  notebooks/                        # Colab notebooks
  docs/                             # GitHub Pages report and supporting docs
  slides/                           # Final presentation slides
  demo_assets/                      # Small tracked benchmark/demo summaries
  models/                           # Tracked key checkpoint plus local generated checkpoints
  tests/                            # Lightweight unit/smoke tests
  archive/                          # Historical PDFs, dev notes, and large Prime run history
```

Generated training outputs are normally written to `models/`, `logs/`, and `results/`. Most new files in those directories should stay local and are ignored by git unless they are intentionally promoted as a small final artifact.

## Fresh Clone Setup

Create and activate a virtual environment:

```bash
git clone https://github.com/yuvimalik/Racing_Gym_RL.git
cd Racing_Gym_RL
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install dependencies in this order:

```bash
pip install -r requirements.txt
pip install -r requirements_sb3.txt --no-deps
pip install git+https://github.com/igilitschenski/multi_car_racing.git --no-deps
```

Why two requirements files: `multi_car_racing` works with `gym==0.17.3`, while Stable-Baselines3 1.8.0 advertises a different gym pin. Installing SB3 with `--no-deps` avoids the resolver replacing the environment stack.

For headless Linux evaluation or training, use `xvfb-run` or set `RACING_HEADLESS_PYGLET=1` if EGL support is available:

```bash
xvfb-run -a python evaluate.py --model models/best_model_torch.pt --episodes 2 --no-video
```

On macOS, a normal display is usually simpler than pyglet EGL headless mode.

## Quick Demo

Evaluate the tracked PPO checkpoint without writing video:

```bash
python evaluate.py --model models/best_model_torch.pt --config config/multi_car_config.yaml --episodes 3 --no-video --seed 42
```

Save an evaluation video and JSON summary:

```bash
python evaluate.py \
  --model models/best_model_torch.pt \
  --config config/multi_car_config.yaml \
  --episodes 3 \
  --seed 42 \
  --output-video results/demo/best_model_eval.mp4 \
  --output-json results/demo/best_model_eval.json
```

For the shortest end-to-end hosted demo, run [`notebooks/colab_ppo_world_model_demo.ipynb`](notebooks/colab_ppo_world_model_demo.ipynb) in Google Colab. It evaluates the tracked PPO checkpoint, displays the tracked world-model benchmark summary, collects a tiny replay, and runs a small world-model smoke training path.

## Training Commands

Single-agent PyTorch PPO:

```bash
python train.py --config config/multi_car_config.yaml --seed 42
```

Shared-policy multi-agent smoke test:

```bash
python train.py --config config/multi_car_marl_smoke_config.yaml --seed 42
```

Longer shared-policy multi-agent run:

```bash
python train.py --config config/multi_car_marl_config.yaml --seed 42
```

Resume from a checkpoint:

```bash
python train.py \
  --config config/multi_car_marl_config.yaml \
  --resume models/v3_marl_control_bootstrap/best_model_torch.pt \
  --timesteps_add 500000 \
  --seed 42
```

Distributed training is available for CUDA systems only:

```bash
torchrun --nproc_per_node=NUM_GPUS distributed_train.py --config config/multi_car_marl_config.yaml --seed 42
```

Do not run long training jobs as a smoke test. They are included for reproducibility and continuation, but final training can take many GPU hours depending on the config.

## World-Model Commands

Collect a small replay dataset:

```bash
python world_model_collect_replay.py --config config/world_model_colab_demo.yaml --device cpu
```

Train a small RSSM world-model smoke run:

```bash
python world_model_train.py --config config/world_model_colab_demo.yaml --epochs 1 --no-wandb
```

Evaluate reward and telemetry faithfulness:

```bash
python scripts/evaluate_reward_faithfulness.py \
  --config config/world_model_colab_demo.yaml \
  --manifest results/world_model/demo_replay/val_manifest.json \
  --world-model-checkpoint models/world_model_demo/rssm_sequence.pt \
  --context-length 10 \
  --horizon 8 \
  --batch-size 4 \
  --num-batches 4 \
  --output results/world_model/demo_artifacts/faithfulness.json
```

The larger world-model configs in `config/world_model_*.yaml` reference replay manifests and checkpoints that are intentionally not tracked because they are large. See [`WORLD_MODEL_PROGRESS.md`](WORLD_MODEL_PROGRESS.md) and [`WORLD_MODEL_HANDOFF.md`](WORLD_MODEL_HANDOFF.md) for the full experiment chronology.

## Important Config Files

See [`config/README.md`](config/README.md) for a concise map. The most important configs are:

- `config/multi_car_config.yaml`: maintained single-agent PyTorch PPO setup.
- `config/multi_car_marl_smoke_config.yaml`: short multi-agent smoke test.
- `config/multi_car_marl_config.yaml`: shared-policy IPPO multi-agent training.
- `config/world_model_colab_demo.yaml`: small world-model demo configuration.
- `config/world_model_config.yaml`: fuller local world-model configuration.
- `config/prime_marl_*.yaml`: cloud/Prime Intellect training variants.

## Results and Outputs

Typical outputs:

- `models/`: checkpoints such as `best_model_torch.pt` and generated training checkpoints.
- `logs/`: TensorBoard event files.
- `results/`: evaluation JSON, training summaries, videos, plots, replay manifests, and world-model artifacts.
- `demo_assets/`: small tracked summaries used by the report and demo.

Most large outputs are ignored by git. The repository intentionally tracks only a small PPO checkpoint and lightweight benchmark summaries needed for a final-project demo.

## Testing and Verification

Lightweight checks for a fresh environment:

```bash
python check_setup.py
python -m unittest tests.test_world_model
python -m compileall train.py evaluate.py world_model scripts autoresearch
```

If `check_setup.py` fails at `multi_car_racing`, reinstall it with:

```bash
pip install git+https://github.com/igilitschenski/multi_car_racing.git --no-deps
```

## Known Limitations

- Full PPO and MARL training is compute-intensive and should not be treated as a quick unit test.
- Exact determinism is not guaranteed across CUDA, MPS, CPU, and rendering backends.
- The world-model branch is limited by sharp-turn future pose evolution; telemetry prediction improved substantially, but robust imagination through hard turns remains unresolved.
- Several historical world-model configs require replay data and checkpoints that are not tracked in git.
- `multi_car_racing` depends on an older Gym stack, so dependency installation order matters.

## Archive

Historical PDFs, development prompts, Prime run pulls, and backup artifacts were moved to [`archive/`](archive/). They are preserved for project history but are not part of the primary reproduction path.

## Credits

This repository was developed as a STAT 4830 reinforcement learning final project by Yuv Malik, Pablo Echevarria, and Matthew Lobo.
