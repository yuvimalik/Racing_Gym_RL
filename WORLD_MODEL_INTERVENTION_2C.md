# World Model Intervention 2C

## Rationale

The previous branches converged on the same conclusion:

- `P5` remains the best stable checkpoint
- objective-only tuning did not materially improve `sharp_turn`
- `2A` explicit ego-motion head was non-transformative
- `2B` larger stochastic capacity (`32 -> 64`) degraded both `sharp_turn` and general hallucination quality

The remaining issue now looks like a **factorization problem**, not just a supervision problem and not just a capacity problem.

## Active Hypothesis

The model needs an explicit separation between:
- **world-content state**
  - road geometry
  - scene layout
  - stable corridor structure
- **ego-motion state**
  - forward advancement
  - turning / steering evolution
  - viewpoint change through the world

Without that separation, the decoder can continue to learn:
- plausible road continuation

without truly learning:
- ego moving through a stable road during a hard turn

## Intervention 2C Design

### Core idea

Factorize the stochastic latent into two branches:
- `z_world`
- `z_motion`

Keep the deterministic GRU state, but stop treating the entire stochastic state as a single undifferentiated vector.

### Minimal v1 structure

Use:
- `stochastic_world_dim = 32`
- `stochastic_motion_dim = 16`

Total stochastic capacity:
- `48`

This is larger than the original `32`, but smaller and more disciplined than the failed `64`-dim single-blob expansion.

### Decoder conditioning

Decode from:
- deterministic state
- `z_world`
- `z_motion`

Do not collapse the two stochastic branches back together before decoding.

The point of `2C` is that the decoder must see:
- stable world content separately from motion content

### Supervision

Attach a dedicated ego-motion head to:
- deterministic state plus `z_motion`

Predict:
- `speed`
- `steer`
- `progress_delta`

Do **not** supervise `z_world` directly with these targets.

Keep the existing telemetry head for compatibility, but treat the new motion-supervised branch as the main structural signal.

## Code Targets

- `world_model/models.py`
  - split stochastic outputs into world and motion branches
  - add separate prior / posterior heads or split the existing outputs cleanly
  - add decoder path using `deterministic + z_world + z_motion`
  - add motion head on `deterministic + z_motion`
- `world_model/training.py`
  - add separate `motion_branch_loss`
  - keep old telemetry loss path for comparison
- `world_model_train.py`
  - support partial warm-start where possible
  - report missing keys clearly
- `config/`
  - add one local `2C` config

## Warm-Start Policy

`2C` is not fully shape-compatible with `P5`.

Use partial transplant only:
- reuse:
  - encoder
  - deterministic GRU / sequence model where shapes match
  - decoder/reward/telemetry layers only where input slices can be copied safely
- reinitialize:
  - split stochastic prior / posterior output layers
  - any new motion-branch-specific modules

Do not force a fake full warm-start.

## First Local Test

### Config defaults

- dataset:
  - `D4_main`
- eval:
  - `sharp_turn` only
- training:
  - `2` local epochs
  - low batch size / laptop-safe loader settings
- losses:
  - no perceptual loss
  - no special normalized `progress_delta` trick
  - raw MSE for the motion branch in v1

### Success criteria

- `sharp_turn` becomes more interpretable than `P5`
- general hallucination does not regress below the broad `P5` baseline
- no obvious smear / collapse / random artifacts

### Failure criteria

- still no meaningful sharp-turn gain
- general hallucinations regress again
- factorized branch behaves no better than `2A`

## Decision Gate

If `2C` is promising:
- continue this branch
- then consider a bounded remote run

If `2C` is weak:
- stop local incremental experimentation
- conclude that this RSSM family likely needs a more substantial redesign than small branch refactors
