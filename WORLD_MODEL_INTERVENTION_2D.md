# World Model Intervention 2D

## Rationale

The previous branches converged on a narrower failure mode:

- `P5` remains the best stable checkpoint
- objective-only tuning did not materially improve `sharp_turn`
- `2A` ego-motion head was non-transformative
- `2B` larger stochastic capacity degraded both sharp-turn and general hallucination quality
- `2C` factorized world-vs-motion latents were at best more geometrically stable, but still did not produce meaningful turning

The active problem now looks like **missing pose evolution**, not just missing supervision, not just missing capacity, and not just missing latent organization.

## Active Hypothesis

The model needs an explicit state transform for viewpoint / pose change.

Current behavior suggests the model can:
- preserve a plausible road corridor
- maintain broad geometry for short horizons

But it still struggles to:
- move the camera/car convincingly through a hard turn
- express turn evolution as a transformation of a stable world

The hypothesis for `2D` is:
- keep a stable world-content latent
- predict a compact pose delta each step
- condition reconstruction on world state plus pose state

This should force the model to represent turning as:
- motion through the world

rather than:
- another latent texture / geometry continuation

## Intervention 2D Design

### Core idea

Introduce an explicit pose branch with two roles:
- represent compact viewpoint / turn evolution
- condition decoding on that pose representation

### Minimal v1 structure

Keep:
- deterministic GRU
- world-content latent branch

Add:
- `pose_delta_head`
  - predicts a compact pose delta vector from the deterministic state
- `pose_state`
  - accumulated or recurrent pose representation over time
- decoder conditioning on:
  - deterministic state
  - world latent
  - pose state

### Pose representation

For the first test, use a compact learned pose vector of size `3`, aligned to existing signals:
- forward motion proxy
- turn-rate / heading proxy
- lateral / curvature proxy

Do **not** try to recover a full physical pose yet.
The goal is to force an explicit motion state into the reconstruction path.

### Supervision

Supervise pose-related outputs against existing telemetry targets:
- `speed`
- `steer`
- `progress_delta`

The supervision is not meant to define full pose exactly.
It is meant to anchor the pose branch to meaningful motion evolution.

## Code Targets

- `world_model/models.py`
  - add a pose branch or pose state update module
  - condition decoding on world latent plus pose state
  - keep world-content and pose-content clearly separated
- `world_model/training.py`
  - add a `pose_loss` term
  - keep old telemetry supervision for compatibility
- `world_model_train.py`
  - support partial warm-start from `P5`
  - report missing keys clearly
- `config/`
  - add one local `2D` config

## Warm-Start Policy

Reuse from `P5` only where defensible:
- encoder
- deterministic GRU / sequence model
- decoder weights where the original input slice still aligns
- reward / telemetry layers where input slices still align

Initialize fresh:
- pose branch
- pose-conditioned decoder slice
- any new pose-state update logic

Do not force strict compatibility beyond that.

## First Local Test

### Defaults

- dataset:
  - `D4_main`
- eval:
  - `sharp_turn` only
- training:
  - `2` local epochs
  - low batch size / laptop-safe settings
- losses:
  - no perceptual loss
  - no special normalized `progress_delta` trick
  - raw MSE for pose-supervised outputs

### Success criteria

- `sharp_turn` finally shows more interpretable turn evolution
- general hallucination does not regress below the broad `P5` baseline
- geometry remains stable while the rollout actually turns

### Failure criteria

- still no meaningful turning
- general hallucinations regress again
- pose branch behaves like another non-transformative auxiliary path

## Decision Gate

If `2D` is promising:
- continue this branch
- then consider a bounded remote run

If `2D` is weak:
- conclude that this RSSM family likely needs a more substantial redesign than small branch refactors
- stop local patch-by-patch iteration and move to a larger architecture rethink
