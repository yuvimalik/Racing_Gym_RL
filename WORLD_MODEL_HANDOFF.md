# World Model Handoff

## Purpose

This file is a critical handoff for the next planning pass.

It summarizes:
- the current world-model architecture
- the strongest positive results so far
- the repeated failure modes
- the dataset / training interventions already tried
- the current belief that objective-only tuning has now largely exhausted itself and the active frontier is structural

## Current Recommendation

The next disciplined step is **Structural Intervention 2D**, not another generic continuation run, not more objective-only tuning, and not another latent-only variant.

Lead claim:
- the current issue now looks like a **pose-evolution bottleneck first**
- objective-only tuning was worth testing, but it did not materially improve the `sharp_turn` failure mode
- `2A`, `2B`, and `2C` were all informative but insufficient

Explicit code facts already in the repo:
- `PerceptualLoss` is now wired conditionally into the training loop
- the first combined Intervention 1 formulation used normalized `progress_delta` with high weight and destabilized training
- `progress_delta` is a tiny, zero-heavy target and is easy to mis-scale
- there is still no explicit ego-motion pathway from latent dynamics into image decoding
- `stochastic_dim` is still `32`
- the first explicit ego-motion head branch was non-transformative
- the `stochastic_dim=64` branch degraded both sharp-turn and general hallucination quality
- the factorized `2C` branch was more geometrically stable at best, but still did not produce meaningful turning

This is intended to be read by a fresh agent or reviewer who should challenge the current approach and propose a stronger plan.

## Current Architecture

The active world model is a fixed-architecture RSSM-style latent dynamics model.

Core components:
- vision encoder
- deterministic GRU sequence model
- stochastic posterior / prior latent heads
- image decoder
- reward predictor
- telemetry predictor heads for:
  - speed
  - progress delta
  - steer
  - corner angle
  - offtrack probability

Training objective:
- reconstruction loss
- reward loss
- KL loss
- telemetry supervision loss

Important constraints:
- the encoder / GRU / latent core / decoder architecture has been held fixed through the main sprint
- improvements have mainly come from:
  - replay diversity
  - longer sequence training
  - telemetry supervision
  - warm-start continuation from stronger checkpoints

## Best Checkpoint

The strongest overall checkpoint so far is:

- `models/world_model/P5_d4_main_telemetry_a100_bs128_e15/rssm_sequence_epoch_015.pt`

Why `P5` is still the best overall:
- best balance between visual stability and short-horizon physics faithfulness
- strong reward / speed / steer / offtrack prediction
- clean hallucination quality
- later turn-focused runs did not clearly beat it overall

## Strongest Positive Results

### 1. Geometry persistence improved materially over the sprint

`E1 -> E3` showed real gains:
- more stable road corridor
- better centerline continuity
- longer medium-horizon geometry persistence

This was not noise. Diversity and longer temporal training both helped.

### 2. Telemetry supervision helped local dynamics

Adding auxiliary supervision improved:
- short-horizon reward faithfulness
- speed prediction
- steer prediction
- offtrack / viability prediction

This is one of the clearest real technical improvements in the project.

### 4. General hallucinations are now baseline decent

Recent objective ablations suggest:
- general hallucinations remain broadly serviceable
- the unresolved failure is not generic decoder collapse
- the unresolved failure is specifically hard-turn future dynamics

### 3. `P5` produced genuinely better hallucinations

The model reached a point where:
- track geometry evolved in a semi-natural way
- geometry stayed more consistent
- the world no longer collapsed immediately into generic blur

This was meaningful progress.

## Main Failure Modes

### 1. Ego motion is still under-modeled

The clearest persistent issue is:
- the car often stays too anchored in image space
- the world geometry evolves, but the ego/camera motion is not represented strongly enough

This is a narrower and more informative failure than the early runs.

### 2. Progress / forward advancement is still weak

Across telemetry-faithfulness runs:
- `progress_delta` remained the weakest continuous signal
- reward, speed, steer, and offtrack improved much more than progress delta

This strongly suggests the model is not fully capturing forward ego-world advancement.

Additional lesson:
- a badly conditioned `progress_delta` loss can dominate telemetry optimization and hurt geometry rather than improve forward-motion modeling

### 2b. Objective-only tuning did not fix the hard-turn failure

Recent local ablations showed:
- perceptual-only was neutral-to-worse on the curated `sharp_turn` clip
- safer progress-only kept general hallucination quality decent but still did not materially improve hard-turn future dynamics

This is the key reason the active frontier should now move to structural changes.

### 2c. First structural branches were not enough

Recent structural results showed:
- `2A` ego-motion-head conditioning looked broadly in-family with prior runs
- `2B` larger stochastic capacity made both `sharp_turn` and general hallucinations worse
- `2C` factorized world-vs-motion latents still failed to produce meaningful turning

This means the remaining problem is likely not missing supervision alone and not missing capacity alone.

### 3. Hard-turn specialization can damage the visual prior

When large amounts of manual hard-turn data were injected:
- turn coverage improved in principle
- but reconstruction and side-by-side hallucinations often regressed
- artifacts appeared, including scattered dots / poorer geometry consistency

So the current model is sensitive to distribution shifts in the replay mix.

### 4. Actor-critic did not transfer

The frozen-world-model latent actor-critic diagnostic showed:
- imagined optimization could look healthy
- real-environment transfer still failed badly

That means the current world model is not yet a reliable simulator for policy learning, even when internal training metrics look good.

## Datasets Tried

### `D4_main`

This was the strongest stable telemetry-enabled replay dataset.

It provided:
- a good general visual prior
- strong overall hallucination quality
- the best base for `P5`

### `D5`

Manual hard-turn data was collected specifically to expose real turns.

Positive:
- finally introduced genuine turn-heavy data
- improved evaluation realism

Negative:
- large portions were noisy / offtrack-heavy
- too much of it in training degraded the decoder prior

### `D6`

`D4_main + D5` in a broad merge.

Result:
- too much manual hard-turn data at once
- likely overwhelmed the stable visual distribution
- produced worse geometry / more artifacts

### `D6b`

Curated small manual supplement merged into `D4_main`.

Result:
- better than `D6`
- cleaner than the broad hard-turn merge
- still did not clearly unlock robust future turning behavior

## Key Experimental Lessons

### Lesson 1

More training on the same architecture does improve things, but it now looks like it mostly provides incremental gains, not a qualitative fix.

### Lesson 2

Manual turn data is useful for evaluation and targeted augmentation, but not as a dominant training distribution in the current setup.

### Lesson 3

The remaining problem no longer looks like generic blur or lack of optimization.

It now looks more like:
- a hard-turn-specific ego/world representation issue
- with objective-conditioning mistakes having been ruled in as secondary contributors, not the main unlock

### Lesson 4

`P5` is likely close to the ceiling of the current architecture under the current objective family.

This is why another very large “hero run” is not the recommended next use of credits.

### Lesson 5

The first combined Intervention 1 run was not a clean test of perceptual loss.

It more likely showed:
- unstable `progress_delta` scaling
- excessive `progress_delta` weight
- a formulation failure rather than evidence that only more compute is needed

### Lesson 6

Even after fixing the obvious objective mistake, the sharp-turn clip remained weak.

That means:
- objective-only tuning is likely close to exhausted
- the next serious intervention should be structural

### Lesson 7

The first two structural nudges were not enough.

That means:
- an extra ego-motion head is too weak a change by itself
- a bigger stochastic latent is too blunt a change by itself
- lightweight factorization is still too weak if it does not explicitly model pose evolution
- the next intervention should explicitly model viewpoint / pose change

## Why The Current Approach May Be Structurally Limited

The evidence suggests the model can learn:
- road appearance
- short-horizon continuity
- some local dynamics

But it still struggles to represent:
- actual future ego pose change through a turn
- clean separation between world geometry and camera/car movement
- strong forward advancement consistency

That suggests the decoder may be asked to solve too much from a latent that does not explicitly structure:
- pose
- heading
- ego/world factorization

Possible structural issues:
- latent state entangles world geometry with ego pose
- decoder can preserve a plausible corridor without preserving a stable world
- reward + telemetry supervision still do not force the right factorization
- the model may need an explicit inductive bias around pose / motion / viewpoint change
- simple capacity increases are insufficient without better separation of roles inside the latent state
- even factorized latent branches may be insufficient if the decoder is never forced to apply an explicit state transform for turning

## What The Next Agent Should Critique

The next agent should not assume that:
- more epochs
- more replay
- or a larger compute budget

will solve the remaining issue.

The next agent should work on **pose-conditioned structural changes first**, not another head-only, latent-only, or capacity-only branch.

Questions to challenge:
1. what is the smallest pose-conditioned change that directly targets turn evolution?
2. how should the model represent pose delta or viewpoint change at each step?
3. which parts of `P5` can still be reused safely in a pose-conditioned branch?

## Recommended Baseline For Any Future Plan

Future experiments should treat this as the stable baseline:

- checkpoint: `P5_d4_main_telemetry_a100_bs128_e15`

Any new plan should justify why it is better than:
- continuing to fine-tune `P5`
- or using `P5` as the best stable final model

## Intervention 1 Baseline

Treat the following as the active starting point:

- warm-start checkpoint:
  - `models/world_model/P5_d4_main_telemetry_a100_bs128_e15/rssm_sequence_epoch_015.pt`
- objective changes under consideration:
  - perceptual-only
  - safer progress-only
  - combined-safe only if one isolated ablation is non-destructive
- evaluation gate:
  - `sharp_turn` only

## Bottom-Line Current Conclusion

The project made real progress:
- geometry persistence improved
- telemetry supervision improved local physics faithfulness
- hallucinations became substantially more coherent than the baseline

But the main unsolved issue remains:

- the model still does not robustly represent future ego motion through turns

At this point, that issue looks more likely to require either:
- an objective correction first
- or, if that fails, a structural change rather than a simple increase in training time
