# Configuration Guide

This folder contains the experiment configurations used by the PPO, multi-agent PPO, Prime/cloud, and world-model branches.

## Recommended Entry Points

- `multi_car_config.yaml`: maintained single-agent PyTorch PPO configuration.
- `multi_car_marl_smoke_config.yaml`: short shared-policy multi-agent smoke test.
- `multi_car_marl_config.yaml`: main shared-policy IPPO multi-agent configuration.
- `world_model_colab_demo.yaml`: small world-model demo path used by the Colab notebook.
- `world_model_config.yaml`: fuller local world-model training configuration.

## Experiment Families

- `multi_car_config_*.yaml`: single-agent PPO reward and control variants.
- `multi_car_marl_*.yaml`: multi-agent PPO/IPPO reward, safety, and evaluation variants.
- `prime_marl_*.yaml`: cloud/Prime Intellect training variants.
- `world_model_*.yaml`: RSSM world-model data, training, intervention, and cluster variants.
- `experiments/*.yaml` and `sweeps/*.yaml`: smaller sweeps and safety pilots.

Many world-model configs reference replay manifests and checkpoints under `results/world_model/` and `models/world_model/`. Those large artifacts are intentionally not tracked in git; use `world_model_colab_demo.yaml` for the lightweight reproducible demo.
