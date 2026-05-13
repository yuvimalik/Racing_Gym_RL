# Racing Gym RL

This repository contains the full project history for a racing-agent research effort that moved through three major phases:

1. Single-car PPO stabilization
2. Multi-agent and shared-policy racing infrastructure
3. Offline world modeling with RSSM-style latent dynamics

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yuvimalik/Racing_Gym_RL/blob/yuv-dev/colab_ppo_world_model_demo.ipynb)

The `docs/` directory is the final report surface. This README is separate from that report. Its job is to help a reader run the project, understand the chronology, and navigate the codebase.

## What This Repo Contains

- `Direct control`: PPO-based racing agents trained from `96 x 96` RGB observations.
- `Agentic search`: AutoResearch loops that mutate reward shaping and PPO settings, then promote or reject candidates from measured results.
- `World modeling`: An offline RSSM pipeline for latent dynamics, replay-based training, hallucination evaluation, and telemetry-faithfulness analysis.

## Quick Start

### Zero-setup: Colab

The fastest way to see something running is Colab — no local install needed.

| Notebook | What it does |
|---|---|
| [`colab_ppo_world_model_demo.ipynb`](colab_ppo_world_model_demo.ipynb) | **No install required.** Loads pre-committed benchmark JSONs, plots the AutoResearch experiment chart, displays inline agent and hallucination videos, and prints the RSSM architecture. CPU runtime, under 2 minutes. |
| [`colab_training.ipynb`](colab_training.ipynb) | Full environment install, train from scratch, record an evaluation video. GPU recommended. |

Open the demo notebook directly from the badge above, or open `colab_training.ipynb` in Colab for the full training path.

### Local setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements_sb3.txt --no-deps
pip install git+https://github.com/igilitschenski/multi_car_racing.git --no-deps
```

If you want to use the AutoResearch loops, create a `.env` file:

```bash
OPENAI_API_KEY=...
GOOGLE_API_KEY=...
```

### Main commands

Train the single-car torch PPO path:

```bash
python train.py --config config/multi_car_config.yaml --seed 42
```

Train the shared-policy multi-agent path:

```bash
python train.py --config config/multi_car_marl_config.yaml --seed 42
```

Run a short multi-agent smoke test:

```bash
python train.py --config config/multi_car_marl_smoke_config.yaml --seed 42
```

Evaluate a checkpoint:

```bash
python evaluate.py --model models/v2_progress/best_model_torch.pt --config config/multi_car_config.yaml --episodes 10
```

Run the classic AutoResearch PPO loop:

```bash
python -m autoresearch.run_loop --provider openai --model gpt-4.1-mini
```

Run a recursive control search:

```bash
python -m autoresearch.run_recursive
```

Train the offline world model:

```bash
python world_model_train.py --config config/world_model_config.yaml --epochs 1
```

Run the world-model control diagnostic:

```bash
python world_model_train_control.py --config config/world_model_config.yaml
```

## How To Read The Project

There are two different reading modes for this repository.

- If you want to run code without installing anything: open `colab_ppo_world_model_demo.ipynb` in Colab.
- If you want to run code locally: start with `train.py`, `evaluate.py`, and the commands above.
- If you want to understand the research history: read `TRAINING_HISTORY.md`, then `WORLD_MODEL_PROGRESS.md`, then the intervention memos in the repo root.

The repo is not a single clean implementation written in one pass. It is a research record. That means the useful way to understand it is by phase.

## Project Chronology

### Phase 1: PPO baseline and stabilization

The earliest working control reference in the repo is the historical SB3 baseline documented in `TRAINING_HISTORY.md`. That branch showed the task was learnable before the custom torch backend was mature.

The torch PPO stack then went through several failure modes:

- GAE off-by-one instability
- Steering collapse from shared action-scale exploration
- Policy collapse from a shared actor-critic head
- Off-track reward hacking
- Donut spinning and no-progress circling
- Tailspin overspeed from overly aggressive throttle incentives

Those repairs turned the torch path into a usable baseline. The first clear AutoResearch promotion was `autoresearch/results/run_008/metrics.json`, which reached:

- reward `205.46`
- progress `0.5551`
- off-track `0.0360`
- speed `28.32`

Later recursive search improved braking behavior more explicitly. The clearest promoted branch was `autoresearch/results/recursive_cap100/promoted/metrics.json`, corresponding to `g003_c03`, with:

- reward `232.58`
- progress `0.7376`
- off-track `0.2678`
- speed `44.52`

### Phase 2: Multi-agent and racing infrastructure

Once single-car PPO was stable enough to operate, the project expanded into multi-agent racing, shared-policy training, evaluation tooling, anti-contact logic, and cloud execution scripts.

This phase matters mainly as a systems bridge:

- shared-policy PPO configs were added
- evaluation became race-aware rather than reward-only
- anti-hooking and anti-contact logic became necessary
- cloud and distributed launch scripts were added

The repo contains working assets for this phase, but the cleanest benchmark chronology remains stronger for single-car PPO and the world-model branch than for the multi-agent branch.

### Phase 3: Offline world modeling

The world-model branch began as a fixed-architecture RSSM ladder and later became a targeted investigation into sharp-turn failures.

High-level chronology:

1. `E1` established a credible visual baseline.
2. `E2` and `E3` expanded replay diversity and horizon.
3. Latent-control transfer failed, which showed that a plausible latent model was not yet control-faithful.
4. The `D4` telemetry recollection pivot changed the branch from pure visual continuation to physics-aware modeling.
5. `P3 -> P4 -> P5` became the strongest positive sequence.
6. Broad hard-turn injection `D6` regressed quality.
7. Curated supplement `D6b` was cleaner but still not a breakthrough.
8. Objective-only interventions and small structural interventions failed to unlock sharp-turn pose evolution.
9. `2D` is the documented next architecture hypothesis, not a completed checked-in result.

The strongest stable world-model checkpoint is:

- `models/world_model/P5_d4_main_telemetry_a100_bs128_e15/rssm_sequence_epoch_015.pt`

Its benchmark summary is reflected in `demo_assets/world_model_benchmark_summary.json`, where telemetry supervision materially improves reward, speed, steer, and off-track faithfulness, while `progress_delta` remains weak.

## How To Understand The Codebase

### Top-level execution files

- `train.py`: main PPO training entry point, environment wrapping, reward shaping, evaluation hooks, and trainer selection.
- `evaluate.py`: checkpoint evaluation, metrics, and video/export logic.
- `distributed_train.py`: distributed torch training path.
- `world_model_train.py`: offline world-model training entry point.
- `world_model_train_control.py`: latent-control diagnostic path.
- `world_model_collect_replay.py`: replay collection for world-model datasets.

### PPO and control stack

Start here if you want to understand direct control:

- `train.py`
- `autoresearch/train_ppo.py`
- `TRAINING_HISTORY.md`
- `config/multi_car_config.yaml`
- `config/multi_car_marl_config.yaml`

Important concepts in this part of the repo:

- actor-critic architecture
- rollout buffer and GAE
- reward shaping and term weighting
- off-track and no-progress termination
- steering, throttle, and brake exploration
- safety governor logic

### AutoResearch stack

Start here if you want to understand the agentic search workflow:

- `autoresearch/run_loop.py`
- `autoresearch/run_experiment.py`
- `autoresearch/run_recursive.py`
- `autoresearch/marl_search_loop.py`
- `autoresearch/results/`

What this subsystem does:

- mutates PPO code and configuration surfaces
- launches short experiments
- screens outputs with gates and failure patterns
- promotes only candidates that outperform the current parent branch

### World-model stack

Start here if you want to understand offline latent dynamics:

- `world_model/models.py`
- `world_model/training.py`
- `world_model/losses.py`
- `world_model/control.py`
- `world_model/control_training.py`
- `WORLD_MODEL_PROGRESS.md`

Important concepts in this part of the repo:

- encoder, GRU sequence model, stochastic prior and posterior
- reward and telemetry prediction
- hallucinated rollout evaluation
- progress-delta supervision
- structural interventions like ego-motion heads and latent factorization

### Configs and experiment surfaces

- `config/`: main training and world-model configs
- `config/world_model_*`: world-model ladders and intervention configs
- `config/prime_*`: longer cloud-run configs
- `sweeps/`: sweep definitions

### Artifacts and evidence

- `autoresearch/results/`: PPO search artifacts and promoted candidates
- `results/`: evaluation JSONs, manifests, and generated outputs
- `wandb/`: exported world-model run summaries
- `docs/assets/`: local videos used by the final report
- `demo_assets/world_model_benchmark_summary.json`: compact benchmark summary for the strongest world-model branch

## Key Evidence Files

If you only want the shortest route to the research story, read these in order:

1. `TRAINING_HISTORY.md`
2. `autoresearch/results/best_metrics.json`
3. `autoresearch/results/recursive_cap100/promoted/metrics.json`
4. `WORLD_MODEL_PROGRESS.md`
5. `WORLD_MODEL_HANDOFF.md`
6. `WORLD_MODEL_INTERVENTION_1.md`
7. `WORLD_MODEL_INTERVENTION_2C.md`
8. `WORLD_MODEL_INTERVENTION_2D.md`
9. `demo_assets/world_model_benchmark_summary.json`

## Benchmark Anchors

These are the main anchored results that are clearly evidenced in the workspace.

### PPO anchors

- Historical SB3 baseline: reward `282+` over `500k` steps
- AutoResearch `run_008`: reward `205.46`, progress `0.5551`, off-track `0.0360`
- Recursive `g003_c03`: reward `232.58`, progress `0.7376`, off-track `0.2678`
- Whole-track reference `v2_progress`: reward `382.86`, progress `0.9993`, off-track `0.6000`

### World-model anchors

- `P2_d4_pilot_telemetry_smoke`: telemetry path works but correlations are weak
- `P3_d4_main_telemetry_warmstart_fast`: first strong local-physics gain
- `P4_d4_main_telemetry_horizon`: stronger horizon-aware telemetry branch
- `P5_d4_main_telemetry_a100_bs128_e15`: best stable checkpoint

## Final Report

The final report is intentionally separate from this README.

- Main report page: `docs/index.html`
- Report styling: `docs/styles.css`
- Report media: `docs/assets/`

That report is the polished research narrative. This README is the practical repo guide and chronology map that helps a reader run the code and understand how the repository is organized.
