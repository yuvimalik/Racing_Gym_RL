# World Model Progress

## Sprint Framing

### Macro Question

Can a learned world model provide a more holistic representation of racing than direct PPO control, by capturing track geometry, dynamics, and rollout structure in a reusable latent state rather than only optimizing immediate driving behavior?

### Micro Question

Does increasing replay diversity and training horizon improve the world model's ability to preserve track geometry through turns over longer imagined rollouts, with the architecture held fixed?

### Fixed-Architecture Rule

For the final sprint:
- keep the RSSM architecture fixed
- do not change encoder, decoder, latent heads, or control architecture
- treat data diversity and temporal training horizon as the two active levers
- do not spend time on latent-control headline results until the `E1 -> E3` ladder is complete

## Current State

The repository now has a working offline Recurrent State Space Model pipeline for the racing simulator, plus the core infrastructure needed to scale the final sprint.

Implemented:
- PPO replay collection for world-model data
- replay manifest preparation and loading
- RSSM training with:
  - encoder
  - GRU sequence model
  - prior and posterior latent heads
  - reward predictor
  - decoder
- LayerNorm after GRU steps for long-horizon stability
- W&B logging, KL annealing flags, and distributed training hooks
- Prime Intellect / H100 training configs and launcher
- checkpoint saving and hallucination video generation
- side-by-side demo video generation
- latent-space visualization
- frozen-world-model latent actor-critic baseline

Current qualitative status:
- straight-line rollouts are broadly coherent
- the ego car is consistently preserved
- grass and road corridor separation is usually preserved
- turns are partially learned but still degrade too early
- long-horizon road-shape persistence is the main open weakness

## Dataset Ledger

### D1: PPO 300k, CCW + CW

- dataset_id: `D1_ppo300k_ccw_cw`
- source checkpoint: `models/ppo_racecar_300000_steps.zip`
- collection command:

```bash
python world_model_collect_replay.py \
  --policy_checkpoint models/ppo_racecar_300000_steps.zip \
  --train_frames 5000 --val_frames 1000 --directions CCW CW
```

- status: collected
- purpose: first PPO-driven diverse replay baseline for the sprint
- expected manifests:
  - `results/world_model/replay/ppo_ppo_racecar_300000_steps_ccw_train_manifest.json`
  - `results/world_model/replay/ppo_ppo_racecar_300000_steps_cw_train_manifest.json`
  - `results/world_model/replay/ppo_ppo_racecar_300000_steps_ccw_val_manifest.json`
  - `results/world_model/replay/ppo_ppo_racecar_300000_steps_cw_val_manifest.json`
- quality notes:
  - confirm both directions are present
  - inspect summary stats with `scripts/manage_world_model_dataset.py summarize ...`
  - visually inspect several episodes for turns, recoveries, and non-repetitive trajectories

### D2: Complementary PPO / torch policy dataset

- dataset_id: `D2_complementary_policy`
- preferred source checkpoint: `models/v2_progress/best_model_torch.pt`
- fallback 1: `models/ppo_racecar_500000_steps.zip`
- fallback 2: `models/ppo_racecar_400000_steps.zip`
- status: pending collection
- purpose: add cleaner or behaviorally different driving traces so the replay mix is not dominated by one policy regime
- target scale:
  - train: enough to bring the merged train set into the `12k -> 18k` frame range
  - val: enough to bring the merged val set into the `2k -> 3k` frame range

### D3: Merged diverse sprint dataset

- dataset_id: `D3_diverse`
- definition: `D1 + D2`
- status: pending merge and validation
- purpose: shared training dataset for `E2` and `E3`
- manifests to create:
  - `results/world_model/replay/d3_diverse_train_manifest.json`
  - `results/world_model/replay/d3_diverse_val_manifest.json`

### Dataset Validation Checklist

Before training on any dataset:
- confirm manifests load cleanly
- confirm both `CCW` and `CW` are represented
- inspect summary stats:
  - mean episode reward
  - mean progress
  - offtrack rate
  - average episode length
- visually inspect several episodes for:
  - turns
  - recoveries
  - non-repetitive trajectories

Useful command:

```bash
python scripts/manage_world_model_dataset.py summarize <manifest paths...>
```

## Experiment Ladder

### E1: Baseline Narrow

- run_name: `E1_baseline_narrow`
- config: `config/world_model_h100_e1_baseline.yaml`
- dataset: current baseline-style narrow manifest pair
- sequence length: `50`
- epochs: `20`
- status: pending
- purpose: anchor the pre-diversity result
- decision criterion:
  - establish turn-fidelity baseline on the curated clips

### E2: Diverse Data

- run_name: `E2_diverse_data`
- config: `config/world_model_h100_e2_diverse.yaml`
- dataset: `D3_diverse`
- sequence length: `50`
- epochs: `20`
- status: pending
- purpose: isolate the effect of replay diversity
- decision criterion:
  - improved turn clips and slower degradation relative to `E1`

### E3: Diverse Data Plus Horizon

- run_name: `E3_diverse_plus_horizon`
- config: `config/world_model_h100_e3_diverse_horizon.yaml`
- dataset: `D3_diverse`
- sequence length: `100`
- epochs: `20`
- status: pending
- purpose: isolate the effect of longer temporal training after data expansion
- decision criterion:
  - materially longer stable rollout depth than `E2`

### Experiment Rules

- `E1 -> E2` isolates replay diversity
- `E2 -> E3` isolates temporal horizon
- `E1 -> E3` is the headline comparison
- do not run the sweep before `E1`, `E2`, and `E3` are complete
- do not use latent-control transfer as the headline for this sprint

### Training Commands

Baseline narrow:

```bash
bash scripts/launch_h100.sh \
  --config config/world_model_h100_e1_baseline.yaml \
  --run-name E1_baseline_narrow \
  --epochs 20
```

Diverse data:

```bash
bash scripts/launch_h100.sh \
  --config config/world_model_h100_e2_diverse.yaml \
  --train-manifest results/world_model/replay/d3_diverse_train_manifest.json \
  --val-manifest results/world_model/replay/d3_diverse_val_manifest.json \
  --run-name E2_diverse_data \
  --epochs 20
```

Diverse data plus horizon:

```bash
bash scripts/launch_h100.sh \
  --config config/world_model_h100_e3_diverse_horizon.yaml \
  --train-manifest results/world_model/replay/d3_diverse_train_manifest.json \
  --val-manifest results/world_model/replay/d3_diverse_val_manifest.json \
  --run-name E3_diverse_plus_horizon \
  --epochs 20
```

## Evaluation Bundle

Run the same evaluation bundle after each experiment:

- hallucination SSIM
- hallucination MSE
- per-step rollout degradation curve
- curated clip metrics for:
  - `sharp_turn`
  - `gentle_turn`
  - `recovery`

Interpretation rule:
- the sprint only counts as successful if the best run shows materially improved turn geometry and longer stable rollout depth on the same curated clips
- lower training loss alone does not count as success

## Presentation Assets

Required primary assets:
- baseline vs best side-by-side rollout video
- rollout-quality-vs-step figure
- latent visualization figure

Final chosen artifacts:
- comparison video: pending
- rollout curve: pending
- latent figure: pending

## Concrete Next Steps

1. Register `D1` summaries in this file after validating the collected manifests.
2. Collect `D2` from `models/v2_progress/best_model_torch.pt` if loading works cleanly, else fall back to `ppo_racecar_500000_steps.zip`.
3. Merge `D1` and `D2` into `D3_diverse` with `scripts/manage_world_model_dataset.py merge`.
4. Validate `D3_diverse` with the dataset summary tool and visual inspection.
5. Launch `E1`.
6. Launch `E2`.
7. Launch `E3`.
8. Record metrics and artifact choices here after each run.

## Implementation Notes

Important repo state:
- `PerceptualLoss` exists in `world_model/losses.py`, but it is not yet wired into the active training loop in `world_model/training.py`
- do not base the sprint narrative on perceptual-loss results until that integration is complete and tested
- `world_model_train.py` now supports explicit `--train-manifest` and `--val-manifest` overrides for sprint datasets
- `scripts/manage_world_model_dataset.py` is the canonical utility for dataset summarization and merged manifest creation

## Success Definition

The final sprint succeeds if:
- `D1`, `D2`, and `D3_diverse` are real, validated datasets
- the `E1 -> E3` run ladder is completed
- the best run clearly improves fixed-architecture world-model fidelity through:
  - more diverse replay data
  - longer temporal training
- the presentation can show one clean causal story rather than multiple scattered claims
