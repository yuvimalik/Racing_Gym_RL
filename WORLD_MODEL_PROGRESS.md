# World Model Progress

## Current State

The repository now has a working offline Recurrent State Space Model pipeline for the racing simulator.

Implemented:
- autoencoder sanity check for the vision stack
- replay collection for manual and automatic world-model data
- replay manifest preparation for train/validation splits
- RSSM training with:
  - encoder
  - GRU sequence model
  - prior and posterior latent heads
  - reward predictor
  - decoder
- checkpoint saving and hallucination video generation
- local GPU optimizations for the RTX 4070 Laptop GPU
- frozen-world-model latent actor-critic baseline
- real-environment actor evaluation from RSSM latent state

## What Is Working

Current hallucination quality is meaningful:
- straight-line rollouts are broadly coherent
- the ego car is consistently preserved
- grass and road corridor separation is usually preserved
- the bottom HUD-like structure is often reconstructed
- turns are partially learned and usually move in the correct broad direction

This means the model is no longer failing at basic world representation. The current weakness is turn consistency and long-horizon road-shape persistence.

## Recent Improvements

Recent training/runtime improvements:
- unique run-specific output directories for checkpoints and hallucinations
- epoch-by-epoch checkpoint saving
- epoch-by-epoch hallucination saving
- CUDA AMP enabled
- pinned-memory and worker-based DataLoader path
- non-blocking CUDA transfers
- batch size increased to `16`
- replay `window_stride` increased to `4`
- curated multi-clip hallucination evaluation
- frozen RSSM wrapper for latent control
- latent actor and critic training entry point: `world_model_train_control.py`

These changes reduced iteration cost while giving the RSSM longer temporal context for corners.

## Key Code Files

Core world-model architecture:
- `world_model/models.py`
  - RSSM state dataclasses
  - encoder / decoder
  - deterministic GRU transition model
  - stochastic prior and posterior heads
  - reward predictor
  - `RSSMCell` and `RSSMSequence`

Offline world-model training:
- `world_model/training.py`
  - replay loader construction
  - RSSM training epoch
  - hallucination rollout and video saving
  - curated side-by-side validation clips
- `world_model_train.py`
  - top-level offline RSSM training entry point
  - run directories, checkpointing, timing, hallucination artifacts

Replay collection and dataset preparation:
- `world_model/collector.py`
  - automatic replay collection
  - environment interaction and episode serialization helpers
- `world_model_collect_manual.py`
  - manual keyboard collection entry point
- `world_model_collect_replay.py`
  - automatic collection entry point
- `world_model_prepare_dataset.py`
  - manifest preparation for train/validation splits
- `world_model/replay.py`
  - `EpisodeReplay`
  - `ReplayWriter`
  - `SequenceReplayDataset`

Latent-control stack:
- `world_model/control.py`
  - frozen RSSM wrapper
  - latent-state flattening
  - actor and critic heads
  - imagined latent rollout helper
- `world_model/control_training.py`
  - imagined actor-critic training epoch
  - discounted bootstrapped return targets
  - real-environment actor evaluation
- `world_model_train_control.py`
  - top-level latent-control training entry point
  - checkpointing and evaluation metric saving

Configuration and tests:
- `config/world_model_config.yaml`
  - RSSM architecture
  - offline training
  - curated evaluation clips
  - latent-control hyperparameters
- `tests/test_world_model.py`
  - RSSM shape tests
  - replay and video smoke tests
  - frozen-control interface and gradient-path tests

## Architecture Breakdown

### 1. Offline World Model

The world model is an RSSM with two latent parts:
- deterministic state `h_t`
- stochastic state `z_t`

The per-step training path is:

```text
previous latent state + previous action
    -> SequenceModel (GRU)
    -> deterministic state h_t
    -> DynamicsModel
    -> prior p(z_t | h_t)

current image x_t
    -> Encoder
    -> image embedding e_t

h_t + e_t
    -> PosteriorModel
    -> posterior q(z_t | h_t, x_t)
    -> sample z_t

[h_t, z_t]
    -> RewardPredictor -> immediate reward
    -> Decoder -> reconstructed image
```

The training losses are:
- reconstruction loss on decoded images
- reward prediction loss
- KL loss between posterior and prior with free bits

### 2. Hallucination / Validation Path

Curated validation clips are split into:
- context frames: real observations used to infer latent state
- future frames: imagined rollout horizon

The hallucination path is:

```text
real context frames
    -> observe_step(...)
    -> posterior-updated latent state

then no more images
    -> imagine_step(...)
    -> prior-only latent rollout
    -> decoded imagined future frames
```

Current side-by-side evaluation compares:
- real future on the left
- imagined future on the right

### 3. Frozen-World-Model Latent Control

The control phase freezes RSSM weights and treats the world model as a differentiable simulator.

The control loop is:

```text
real replay context
    -> frozen RSSM posterior update
    -> initial latent state

flatten [h_t, z_t]
    -> Actor -> action
    -> frozen RSSM imagine_step
    -> next latent state + predicted immediate reward

flatten [h_t, z_t]
    -> Critic -> value estimate
```

Important separation:
- world model predicts observation dynamics and immediate reward
- actor chooses actions in latent space
- critic predicts long-horizon return from latent state

### 4. Real-Environment Actor Evaluation

The actor is not only trained in imagination. It is also evaluated online in the actual simulator:

```text
reset env
    -> observe real image with frozen RSSM
    -> actor chooses action from latent state
    -> env.step(action)
    -> observe next real image
    -> repeat
```

Reported metrics:
- mean reward
- mean progress
- off-track rate

## Current Baseline Interpretation

The world model is now good enough to be judged on targeted failure modes rather than on whether it works at all.

Strong points:
- stable ego-car rendering
- stable broad road geometry
- coherent short-horizon scene evolution

Weak points:
- corner geometry is still patchy in some clips
- turn consistency degrades deeper into imagined rollout
- visual sharpness remains low, especially at longer horizons

## Tangible Next Steps

1. Train the first frozen-world-model latent actor-critic baseline.
2. Measure real-environment transfer with:
   - mean reward
   - mean progress
   - off-track rate
3. Keep the curated hallucination clips as a regression gate on the frozen RSSM.
4. If latent control improves in imagination but transfers poorly, revisit world-model fidelity in the failing turn regimes.
5. If transfer is acceptable, expand the latent-control loop before revisiting RSSM capacity.

## What Success Looks Like Next

The next meaningful improvement should be:
- more consistent road curvature in turns
- less abrupt change of turn direction
- longer persistence of road edges after context ends
- more stable geometry across multiple validation clips

At this stage, success is not photorealism. Success is a more reliable and geometrically consistent rollout.
