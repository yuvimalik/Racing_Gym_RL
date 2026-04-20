# World Model Intervention 1

## Rationale

`P5` is the best stable checkpoint under the current objective:
- pixel-MSE reconstruction
- reward loss
- KL loss
- telemetry supervision

The latest data-mixing experiments showed that more turn-heavy data alone does not break the sharp-turn plateau. Intervention 1 tested whether objective-only changes could unlock the failure mode before escalating to structural work.

## Hypothesis

The first combined Intervention 1 run should be treated as a formulation failure, not a clean negative result.

The updated read after the ablations is:
- perceptual-only did not give a credible positive signal on the `sharp_turn` clip
- safer progress-only kept general hallucinations decent but still did not materially improve hard-turn future dynamics
- objective-only tuning therefore looks largely exhausted for this branch
- the next step is structural intervention, not a larger run on the same objective family

## Current Failure Read

Why `progress_delta` likely blew up:
- the target is extremely small in magnitude
- many windows are effectively zero
- per-batch normalization makes scale unstable
- high weighting amplifies rare nonzero errors
- the model then optimizes a badly conditioned target at the expense of geometry

Observed `D4_main` scale:
- mean abs around `0.00040`
- median abs `0.0`
- p90 abs around `0.00217`

## Code Targets

- `world_model/training.py`
  - wire perceptual loss conditionally from config
  - make progress loss configurable
- `world_model/losses.py`
  - add fixed-scale progress normalization
- `config/`
  - add explicit ablation configs

## Chosen Defaults

- warm-start checkpoint:
  - `models/world_model/P5_d4_main_telemetry_a100_bs128_e15/rssm_sequence_epoch_015.pt`
- dataset:
  - keep the current stable training baseline unless the intervention config explicitly changes it
- perceptual loss:
  - enabled
  - `perceptual_loss_scale = 0.05`
- progress supervision:
  - safer default is fixed-scale normalized MSE
  - use a dataset-level constant scale
  - lower `progress_delta` telemetry weight to `2.0`
- evaluation:
  - `sharp_turn` only

## Ablation Ladder

### I1B-1: Perceptual-Only

- warm-start from `P5`
- keep `progress_delta` on raw MSE with weight `1.0`
- goal:
  - isolate whether perceptual loss helps geometry without the unstable progress branch
- outcome:
  - neutral-to-worse on `sharp_turn`
  - not enough evidence to justify a larger perceptual-only continuation

### I1B-2: Safer Progress-Only

- warm-start from `P5`
- no perceptual loss
- use fixed-scale normalized MSE
- use `progress_delta` weight `2.0`
- goal:
  - test whether progress supervision can help without degrading image quality
- outcome:
  - general hallucinations remained baseline decent
  - sharp-turn-specific future dynamics still looked weak
  - not enough evidence to justify a larger progress-only continuation

### I1B-3: Combined-Safe

- only if one of the first two is non-destructive
- combine perceptual loss with the safer progress formulation
- goal:
  - test whether the two are complementary rather than destructive
- status:
  - deprioritized
  - the branch should not be the active frontier until structural work has been evaluated

## Validation Stages

### 1. Static Implementation Check

- confirm `PerceptualLoss` is wired conditionally from config
- confirm progress loss is configurable and does not silently alter unrelated telemetry heads
- confirm `P5` checkpoint remains warm-start compatible

### 2. Local Ablation Runs

- warm-start from `P5`
- run `2` local epochs first, optionally `3` if stable
- use the existing sharp-turn-only eval
- judge:
  - does training remain numerically stable?
  - does the sharp-turn footage get sharper or structurally more stable?
  - are there new artifacts?

### 3. Bounded Remote Run

- only if local sanity is stable
- use a bounded A100 continuation, not a hero run
- compare against `P5` on:
  - sharp-turn side-by-side clip
  - hallucination SSIM / MSE
  - held-out reward / telemetry metrics, especially `progress_delta`

## Success Criteria

- sharper or more stable sharp-turn geometry than `P5`
- no obvious decoder artifact regression
- at least modest improvement in forward-advancement behavior

## Failure Criteria

- crispness drops
- geometry destabilizes
- progress remains flat with no qualitative gain

## Decision Gate

- if one objective-only ablation is clearly positive:
  - continue objective-level refinement
- actual result:
  - no objective-only ablation showed a meaningful `sharp_turn` gain
  - objective-only tuning should be treated as closed for now
  - promote structural work to the active frontier

## Secondary Structural Hypothesis

- concern:
  - `stochastic_dim=32` may be too small for the world uncertainty the model must carry
- likely effect:
  - too much burden shifts into the 512-dim GRU
  - the GRU entangles ego motion with world geometry
  - the decoder learns "road shape that explains sequence" instead of "ego moving through a stable road"
- current status:
  - plausible and important
  - objective-only ablations were too weak to disconfirm it
  - but the first structural branch should still be explicit ego-motion modeling before changing latent dimensionality

## Next Branch

Intervention 1 is now a completed diagnostic branch.

The active successor is:
- **Structural Intervention 2A**
  - add an ego-motion head
  - condition the decoder on predicted ego motion
  - keep `stochastic_dim=32` initially so the first structural test stays attributable
