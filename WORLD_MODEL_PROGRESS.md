# World Model Progress

## Active Frontier

### Official Reframe: Pose Evolution Bottleneck

The active question is no longer whether more data, more horizon, or another objective-only tweak will solve turning.

The active question is:

- **Macro question:** can explicit pose evolution improve sharp-turn future dynamics while preserving the clean `P5` decoder prior?
- **Primary experiment question:** can a pose-conditioned branch make the curated `sharp_turn` clip actually turn, rather than just remain geometrically stable?

The older diversity / horizon ladder remains important as completed context, but it is no longer the active frontier.

### Current Architecture Ceiling

- best stable checkpoint under pixel-MSE + telemetry supervision:
  - `models/world_model/P5_d4_main_telemetry_a100_bs128_e15/rssm_sequence_epoch_015.pt`
- what `P5` represents:
  - best overall tradeoff between visual stability and short-horizon physics faithfulness
  - strongest current baseline to beat for any follow-up intervention

### Current Bottleneck

- objective-only tuning result:
  - exhausted for now
- current failure read:
  - general hallucinations are best at `P5`
  - `2C` may be geometrically steadier, but still does not produce meaningful turning
  - sharp-turn-specific future dynamics remain weak across all tested interventions
- discarded structural sub-hypotheses:
  - explicit ego-motion head alone was non-transformative
  - stochastic capacity increase to `64` degraded both sharp-turn and general hallucination quality
  - lightweight latent factorization improved stability at best, but not turn behavior
- active structural hypothesis:
  - the model needs an explicit pose / viewpoint evolution mechanism, not just better latent organization

### Active Intervention

- active mainline:
  - **Structural Intervention 2D**
- immediate implementation targets:
  - keep a stable world-content latent
  - predict a compact pose / viewpoint delta each step
  - decode from world state plus pose representation
  - use `P5` only as a partial transplant where defensible
  - evaluate only the curated `sharp_turn` clip
  - treat further latent scaling as secondary unless pose-conditioned decoding is clean but weak

## Objective Ablation Outcome

### I1B-1: Perceptual-Only

- outcome:
  - neutral-to-worse on the `sharp_turn` clip
  - no credible positive signal that perceptual loss alone fixes the failure mode

### I1B-2: Safer Progress-Only

- outcome:
  - general hallucinations remained decent
  - sharp-turn-specific future dynamics remained weak
  - objective-only progress supervision did not unlock the failure mode

### I1B-3: Combined Objective

- outcome:
  - failed due to unstable `progress_delta` scaling
  - not a valid positive test of the combined objective

### Current conclusion

- objective-only tuning did not materially improve sharp-turn modeling
- the unresolved issue is:
  - hard-turn ego/world evolution
  - not generic hallucination quality
- the next branch is structural, not another longer run on the same objective family

## Structural Intervention Ladder

### 2A: Ego-Motion Head + Decoder Conditioning

- goal:
  - make ego motion explicit in the latent-to-image path
- structure:
  - predict `speed`, `steer`, and `progress_delta` from the deterministic state
  - condition the decoder on those predicted values
- evaluation:
  - `P5` baseline
  - `sharp_turn` only

### 2B: Latent Capacity Expansion

- trigger:
  - only if `2A` is weak, neutral, or destabilizing
- structure:
  - increase `stochastic_dim` from `32 -> 64`
  - optionally retain the ego-motion pathway if `2A` is non-destructive
- note:
  - this is not a clean warm-start-compatible tweak and should be treated as a new branch

### 2C: Factorized World-State vs Ego-Motion Latents

- trigger:
  - `2A` was non-transformative and `2B` was a negative result
- structure:
  - split latent decoding inputs into:
    - a world-content latent branch
    - an ego-motion latent branch
  - supervise the ego-motion branch directly on `speed`, `steer`, and `progress_delta`
  - keep the decoder aware of both branches explicitly
- note:
  - this is the first intervention that directly targets ego/world disentanglement rather than adding capacity or another side head

### 2D: Pose-Conditioned World Model

- trigger:
  - `2C` was at best geometrically steadier but still did not produce real turning
- structure:
  - maintain a world-content latent
  - predict an explicit pose / viewpoint delta each step
  - decode from world state plus pose representation
- note:
  - this is the first branch that tries to model turn evolution as a state transform rather than as latent content alone

## Reclassification Of Recent Results

- `P5`:
  - baseline to beat
- `D6`:
  - over-injection failure
- `D6b`, `P6b`, `P8`:
  - diagnostic evidence that curated data alone does not solve future-turn modeling
- `I1B_p5_perc_progress_local_e5`:
  - combined-objective formulation failure
  - not yet a valid negative result on perceptual loss
- `I1B1_p5_perceptual_only`:
  - objective-only ablation with no meaningful sharp-turn improvement
- `I1B2_p5_progress_only`:
  - objective-only ablation with decent general hallucination but weak sharp-turn future dynamics
- `S2A_p5_ego_motion_local`:
  - structural intervention with explicit ego-motion head
  - non-transformative relative to prior runs
- `S2B_p5_stoch64_local`:
  - latent-capacity expansion branch
  - negative result with weaker sharp-turn and weaker general hallucination quality
- `S2C_p5_factorized_local`:
  - factorized world-vs-motion branch
  - geometrically steadier at best, but still no meaningful turning
- current implication:
  - another fixed-objective hero run is not the right next experiment
  - another capacity-only or auxiliary-head-only run is also not justified
  - the next branch should explicitly model pose / viewpoint evolution

## Why `progress_delta` Can Blow Up

- `D4_main` scale:
  - mean abs around `0.00040`
  - median abs `0.0`
  - p90 abs around `0.00217`
- why this matters:
  - many windows are effectively zero
  - per-batch normalization makes scale unstable
  - high weighting amplifies rare nonzero errors and noisy batches
  - the model can optimize a numerically dominant but poorly conditioned surrogate
- likely result:
  - telemetry loss dominates
  - geometry worsens even while total loss falls

## Sprint Framing

Historical context only: the sections below document the completed fixed-architecture diversity / horizon ladder and the experiments that led to `P5`. They are no longer the active mainline.

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

## Run Results

### E1: Baseline Narrow

- run_name: `E1_baseline_narrow`
- dataset: `manual_sprint_train_manifest.json` / `manual_sprint_val_manifest.json`
- summary:
  - baseline manual-data RSSM preserves local road corridor and ego presence
  - turns remain plausible only at short horizon and degrade too early once imagination takes over
  - this is the reference point for all later comparisons
- final train metrics:
  - `recon_loss`: `0.005997`
  - `reward_loss`: `0.175471`
  - `kl_loss`: `1.001981`
  - `total_loss`: `1.183450`
- final validation metrics:
  - `hallucination_mse`: `0.007542`
  - `hallucination_ssim`: `0.9351`
  - `sharp_turn_ssim`: `0.9429`
  - `gentle_turn_ssim`: `0.9397`
  - `recovery_ssim`: `0.9461`
- best artifact paths:
  - checkpoint: `models/world_model/E1_baseline_narrow/rssm_sequence_epoch_020.pt`
  - hallucination dir: `results/world_model/artifacts/hallucination/E1_baseline_narrow/`
- conclusion:
  - `E1` is a credible baseline, but it does not sustain turn geometry deeply enough to support the final story on its own

### E2: Diverse Data

- run_name: `E2_diverse_data`
- dataset: `d3_diverse_train_manifest.json` / `d3_diverse_val_manifest.json`
- summary:
  - diverse replay improves structural plausibility and gives the model broader road-shape priors
  - post-context geometry is still underconstrained and the rollout can keep “re-deciding” the track
  - qualitative gains are real, but the world frame remains unstable
- final train metrics:
  - refer to the W&B run as the source of truth for the complete metric dump
- final validation metrics:
  - `gentle_turn_ssim`: `0.9393`
  - `recovery_ssim`: `0.9347`
  - other final metrics should be copied from W&B into this file before presentation lock
- best artifact paths:
  - checkpoint dir: `models/world_model/E2_diverse_data/`
  - hallucination dir: `results/world_model/artifacts/hallucination/E2_diverse_data/`
- conclusion:
  - `E2` improves visual continuation quality relative to `E1`, but geometry after the context window still shifts too freely to count as world-consistent dynamics

### E3: Diverse Data Plus Horizon

- run_name: `E3_diverse_plus_horizon`
- dataset: `d3_diverse_train_manifest.json` / `d3_diverse_val_manifest.json`
- summary:
  - longer temporal training improves medium-horizon geometric persistence and centerline continuity
  - the strongest effect is that the rollout keeps meaningful track structure for longer before drifting
  - the remaining failure mode is egocentric inconsistency rather than simple reconstruction collapse
- final train metrics:
  - refer to the W&B run as the source of truth for the complete metric dump
- final validation metrics:
  - copy the final W&B values here once the metric table is locked
- best artifact paths:
  - checkpoint dir: `models/world_model/E3_diverse_plus_horizon/`
  - hallucination dir: `results/world_model/artifacts/hallucination/E3_diverse_plus_horizon/`
- conclusion:
  - `E3` is the best current world-model result and the correct frozen checkpoint to use for diagnostic downstream experiments

## Hallucination Review

### E2 Epoch 15

- visual continuation quality:
  - relatively strong hallucination while still benefiting from the initial 50-frame context
  - underlying road corridor and centerline are often plausible
- weaknesses:
  - when free hallucination begins, track geometry is only average and keeps changing with limited structural commitment
  - the model often preserves “some road” without preserving one stable road
- verdict:
  - `E2` is visually competent, but not yet world-consistent

### E3 Epoch 15

- visual continuation quality:
  - stronger geometric structure persists for longer than `E2`
  - centerline stays comparatively stable deeper into rollout
- weaknesses:
  - after the stronger middle portion, future generation becomes more squiggly
  - the model still explains ego movement partly by deforming the track rather than changing pose within a stable world
- verdict:
  - `E3` improves medium-horizon geometry persistence, but does not resolve the egocentric/world-frame mismatch

### Explicit Failure Mode

- `car moves the track` failure:
  - the model often preserves an approximate road corridor while allowing the world geometry to bend with the ego car
  - this means visual continuation quality can look better than true world-consistent dynamics quality

## Decision Log

- what we now believe:
  - data diversity and longer temporal training both matter
  - `E3` is likely the best current frozen world model
  - the dominant remaining issue is world/ego consistency, not generic blur alone
  - the latent actor-critic diagnostic confirms that imagined short-horizon optimization can still exploit model errors and fail real transfer
  - the `P5` checkpoint is still the cleanest visual prior among the physics-aware runs
  - manual hard-turn data is valuable, but only in a limited curated proportion
- what we have ruled out:
  - the baseline failure is not just lack of optimization time on the original narrow dataset
  - more replay diversity alone does not fully solve long-horizon geometry drift
  - adding a large noisy manual hard-turn block directly into training does not improve the model by default and can visibly regress reconstruction quality
- what remains uncertain:
  - whether short-horizon imagined dynamics are still control-faithful despite the egocentric visual artifact
  - whether predicted reward stays aligned enough to support actor-critic training
  - whether one larger-scale run is justified after the next diagnostic checks
  - whether a very small `P5` fine-tune on curated turn-heavy data can improve sharp-turn hallucinations without destabilizing the decoder

## Budget Ledger

- Prime Intellect credits remaining: `~$95`
- policy:
  - do not spend remaining credits on sweeps
  - do not start larger remote runs until the diagnostic control experiment and reward-faithfulness check are interpretable
- approved next spends only:
  - one short-horizon latent actor-critic diagnostic
  - one reward-faithfulness evaluation pass
  - one larger `A100 80GB` hero run only if the diagnostic is promising

## Physics Modeling Continuation

- current hypothesis:
  - the model preserves enough visual structure to look plausible, but not enough explicit physical state to remain world-consistent under ego motion
- why offtrack matters:
  - offtrack probability is a direct viability target
  - it distinguishes a visually plausible corridor from a physically invalid rollout that has already left the drivable manifold
  - it is the cleanest supervision signal for the current “car moves the track” failure mode
- chosen continuation:
  - add telemetry-supervised auxiliary heads for:
    - speed
    - progress delta
    - steer
    - corner angle
    - offtrack probability
  - recollect a telemetry-enabled `D4` dataset incrementally rather than replacing `D3`

## Next Experiment Queue

### Q1: Frozen E3 Short-Horizon Latent Actor-Critic Diagnostic

- purpose:
  - test whether the frozen `E3` model preserves enough short-horizon physics for useful control
- planned setup:
  - frozen checkpoint: `models/world_model/E3_diverse_plus_horizon/rssm_sequence_epoch_020.pt`
  - train manifest: `results/world_model/replay/d3_diverse_train_manifest.json`
  - context length: `25`
  - imagination horizon: `12`
  - small initial epoch budget with real-environment evaluation every epoch
- cost estimate:
  - local preferred first, remote only if needed
- go/no-go criterion:
  - actor trained on imagined rollouts must perform meaningfully above weak baselines in the real environment
- status:
  - completed
- result:
  - run_name: `latent_control_e3_short_horizon`
  - frozen checkpoint: `models/world_model/E3_diverse_plus_horizon/rssm_sequence_epoch_020.pt`
  - final train metrics:
    - `actor_loss`: `-43.2737`
    - `critic_loss`: `5.6433`
    - `imagined_reward_mean`: `0.4358`
    - `imagined_return_mean`: `43.2737`
    - `action_abs_mean`: `0.6051`
  - final real-environment eval:
    - `mean_reward`: `-350.4400`
    - `mean_length`: `973.67`
    - `mean_progress`: `0.0211`
    - `offtrack_rate`: `1.0`
    - `episodes`: `3`
  - artifacts:
    - compare video: `results/world_model/control/latent_control_e3_short_horizon/eval_epoch_005_compare.mp4`
    - real video: `results/world_model/control/latent_control_e3_short_horizon/eval_epoch_005_real.mp4`
    - checkpoint dir: `models/world_model_control/latent_control_e3_short_horizon/`
  - conclusion:
    - the imagined control objective was learnable inside the frozen model, but this did not transfer to the real environment
    - this is strong evidence that short-horizon visual plausibility is not the same as short-horizon control-faithful dynamics
    - the current frozen `E3` world model is not yet reliable enough to justify a larger actor-critic spend

### Q2: Reward-Faithfulness Check

- purpose:
  - determine whether predicted reward remains aligned with recorded reward over the first imagined steps
- planned setup:
  - frozen checkpoint: `E3`
  - held-out replay windows
  - context length: `25` to `50`
  - horizon: `10` to `20`
- cost estimate:
  - lightweight local evaluation
- go/no-go criterion:
  - predicted reward should track the sign, trend, and rough magnitude of held-out reward over short horizons
- status:
  - completed on legacy `D3`
- result:
  - output: `results/world_model/control/reward_faithfulness_e3_d3.json`
  - frozen checkpoint: `models/world_model/E3_diverse_plus_horizon/rssm_sequence_epoch_020.pt`
  - reward metrics:
    - `mean_mse`: `0.0155`
    - `mean_mae`: `0.0623`
    - `mean_corr`: `0.9760`
    - `mean_sign_match`: `0.9908`
    - `mean_bias`: `0.0368`
  - telemetry metrics:
    - all `NaN` on `D3`, as expected, because legacy replay does not contain telemetry targets
  - conclusion:
    - frozen `E3` predicts short-horizon reward well on held-out `D3`
    - reward alignment alone is therefore not enough to explain the actor-critic transfer failure
    - the missing piece remains local physics / state-faithfulness rather than pure reward modeling

### Q2b: Physics-Faithfulness Check

- purpose:
  - determine whether the world model predicts local physical telemetry faithfully over short horizons
- planned setup:
  - held-out telemetry-enabled replay windows from `D4`
  - compare predicted vs actual:
    - speed
    - progress delta
    - steer
    - corner angle
    - offtrack probability
- go/no-go criterion:
  - telemetry alignment must improve without catastrophic regression in hallucination quality
- status:
  - completed on `D4_pilot`
- result:
  - output: `results/world_model/control/reward_faithfulness_d4_pilot_telemetry_smoke.json`
  - checkpoint: `models/world_model/P2_d4_pilot_telemetry_smoke/rssm_sequence_epoch_003.pt`
  - reward metrics:
    - `mean_mse`: `0.5540`
    - `mean_mae`: `0.5119`
    - `mean_corr`: `0.0086`
    - `mean_sign_match`: `0.1633`
  - telemetry metrics:
    - `speed`: `mean_mae=2.5414`, `mean_corr=-0.0306`
    - `progress_delta`: `mean_mae=0.0232`, `mean_corr=-0.0100`
    - `steer`: `mean_mae=0.1025`, `mean_corr=0.0265`
    - `corner_angle`: `mean_mae=0.0245`, correlation mostly undefined / non-informative on this pilot slice
    - `offtrack`: `mean_bce=0.1351`, `mean_accuracy=0.9700`
- conclusion:
    - the telemetry-supervised smoke run learned a highly useful offtrack / viability signal very quickly
    - the continuous short-horizon telemetry heads are not yet predictive in a strong correlational sense
    - the next step should improve coverage and distribution in `D4_main`, not jump immediately to a remote hero run
- follow-up result on `D4_main`:
  - output: `results/world_model/control/reward_faithfulness_d4_main_telemetry_warmstart_fast.json`
  - checkpoint: `models/world_model/P3_d4_main_telemetry_warmstart_fast/rssm_sequence_epoch_005.pt`
  - reward metrics:
    - `mean_mse`: `0.1604`
    - `mean_mae`: `0.1804`
    - `mean_corr`: `0.7728`
    - `mean_sign_match`: `0.7833`
    - `mean_bias`: `0.0290`
  - telemetry metrics:
    - `speed`: `mean_mae=1.1242`, `mean_corr=0.4966`
    - `progress_delta`: `mean_mae=0.0280`, `mean_corr=0.0772`
    - `steer`: `mean_mae=0.0856`, `mean_corr=0.4068`
    - `corner_angle`: `mean_mae=0.0201`, correlation still mostly undefined / low-signal
    - `offtrack`: `mean_bce=0.0173`, `mean_accuracy=1.0000`
  - training snapshot:
    - run: `P3_d4_main_telemetry_warmstart_fast`
    - warm start: `models/world_model/E3_diverse_plus_horizon/rssm_sequence_epoch_020.pt`
    - final epoch losses:
      - `recon_loss`: `0.0041`
      - `reward_loss`: `0.1543`
      - `telemetry_loss`: `1.5725`
      - `speed_loss`: `1.3706`
      - `progress_delta_loss`: `0.0007`
      - `steer_loss`: `0.0150`
      - `corner_angle_loss`: `0.0009`
      - `offtrack_loss`: `0.0967`
      - `total_loss`: `1.6016`
    - hallucination metrics:
      - `mean_mse=0.006166`
      - `mean_ssim=0.9470`
      - `sharp_turn_ssim=0.9257`
      - `gentle_turn_ssim=0.9392`
      - `recovery_ssim=0.9285`
  - conclusion:
    - warm-started telemetry supervision on `D4_main` materially improved short-horizon reward faithfulness and made `speed`, `steer`, and `offtrack` meaningfully predictive
    - `progress_delta` remains the weakest continuous target and is the main remaining local-physics gap
    - this is the first physics-aware run strong enough to justify a bounded horizon follow-up before any remote spend

### Q3: One Budgeted Hero Run

- purpose:
  - test whether one larger fixed-architecture run is justified after the diagnostics
- planned setup:
  - larger diverse dataset
  - `seq_len=150`
  - single `A100 80GB`
- cost estimate:
  - moderate, one run only
- go/no-go criterion:
  - only approved if `E3` is clearly best and the control + reward diagnostics are at least somewhat promising
- status:
  - not yet approved
- current view:
  - still do not spend Prime Intellect credits yet
  - `D4_main` telemetry warm-start was promising enough to justify one bounded local horizon run
  - reconsider remote spend only after the `P4_d4_main_telemetry_horizon` held-out evaluation is in hand

### Q4: D4 Incremental Recollection

- purpose:
  - build a telemetry-enabled continuation dataset for physics-aware training
- rollout:
  - `D4_pilot` for schema validation and a local smoke run
  - `D4_main` only after the pilot passes
- source mix:
  - manual recovery / wide-line / cornering episodes
  - PPO policy replay in both directions
- priority scenarios:
  - sustained turns
  - turn exits
  - steering reversals
  - throttle-lift corner entry
  - braking while turning
  - off-track recovery
  - near-boundary driving
- status:
  - `D4_pilot` and `D4_main` collected and merged
- result:
  - manifests:
    - `results/world_model/replay/d4_pilot_train_manifest.json`
    - `results/world_model/replay/d4_pilot_val_manifest.json`
  - dataset summary:
    - train: `72` episodes / `7000` frames
    - val: `20` episodes / `1400` frames
    - directions:
      - train: `CCW 35`, `CW 37`
      - val: `CCW 10`, `CW 10`
    - sources:
      - train: `automatic 69`, `ppo_policy 3`
      - val: `automatic 18`, `ppo_policy 2`
  - schema validation:
    - replay episodes now store per-step telemetry arrays:
      - `progress`
      - `progress_delta`
      - `speed`
      - `steer`
      - `corner_angle`
      - `offtrack`
      - `track_index`
      - `telemetry_valid`
    - sample `D4_pilot` episodes show `telemetry_valid=1.0`
  - smoke train:
    - run: `P2_d4_pilot_telemetry_smoke`
    - epochs: `3`
    - final train losses:
      - `recon_loss`: `0.0080`
      - `reward_loss`: `0.5261`
      - `telemetry_loss`: `20.6149`
      - `speed_loss`: `20.3648`
      - `offtrack_loss`: `0.1174`
      - `total_loss`: `6.6879`
    - hallucination metrics:
      - epoch 1: `mse=0.011109`, `ssim=0.8995`
      - epoch 2: `mse=0.009405`, `ssim=0.9158`
      - epoch 3: `mse=0.009232`, `ssim=0.9194`
    - curated clip SSIM at epoch 3:
      - `sharp_turn`: `0.9305`
      - `gentle_turn`: `0.9294`
      - `recovery`: `0.9330`
  - conclusion:
    - the telemetry path is stable enough to train locally
    - auxiliary supervision does not catastrophically damage hallucination quality
    - `offtrack` looks immediately useful; continuous telemetry still needs better data coverage and likely more training
- `D4_main` follow-up:
  - purpose:
    - scale the telemetry-enabled continuation dataset after the pilot passed
  - warm-start run:
    - `P3_d4_main_telemetry_warmstart_fast`
    - initialized from `E3`
    - local 5-epoch run using `d4_main_train_manifest.json` / `d4_main_val_manifest.json`
  - takeaway:
    - `D4_main` is large enough to move reward and local telemetry faithfulness substantially without harming hallucination structure
    - the next question is now horizon robustness, not whether telemetry supervision works at all
- next:
  - evaluate `P4_d4_main_telemetry_horizon` on held-out `D4_main`
  - if `progress_delta` and horizon stability improve again, then approve one remote `A100 80GB` run

### Q5: D4 Main Telemetry Horizon Follow-Up

- purpose:
  - test whether the improved local physics from `P3` survives a longer temporal horizon
- setup:
  - config: `config/world_model_local_e3_diverse_horizon.yaml`
  - init checkpoint: `models/world_model/P3_d4_main_telemetry_warmstart_fast/rssm_sequence_epoch_005.pt`
  - train/val: `d4_main_train_manifest.json` / `d4_main_val_manifest.json`
  - epochs: `5`
- status:
  - training completed
- result:
  - run: `P4_d4_main_telemetry_horizon`
  - checkpoint: `models/world_model/P4_d4_main_telemetry_horizon/rssm_sequence_epoch_005.pt`
  - final train losses:
    - `recon_loss`: `0.00337`
    - `reward_loss`: `0.03148`
    - `kl_loss`: `1.00928`
    - `telemetry_loss`: `0.86074`
    - `speed_loss`: `0.82395`
    - `progress_delta_loss`: `0.000437`
    - `steer_loss`: `0.00840`
    - `corner_angle_loss`: `0.000323`
    - `offtrack_loss`: `0.015995`
    - `total_loss`: `1.25932`
  - hallucination metrics:
    - `mean_mse=0.003912`
    - `mean_ssim=0.9665`
    - `sharp_turn_ssim=0.9241`
    - `gentle_turn_ssim=0.9373`
    - `recovery_ssim=0.9153`
  - throughput:
    - `epoch_time=1076.3s`
    - `elapsed=58.8m`
    - `batches_per_sec=0.22`
    - `windows_per_sec=1.75`
  - immediate read:
    - the longer-horizon follow-up improved reconstruction and telemetry losses further
    - hallucination quality remained strong overall
    - the remaining gate is held-out reward / telemetry faithfulness, especially `progress_delta`
- remote A100 continuation:
  - purpose:
    - test whether a larger-throughput continuation on `A100 80GB` can strengthen short-horizon physics faithfulness without giving up hallucination quality
  - run:
    - `P5_d4_main_telemetry_a100_bs128_e15`
  - hardware / setup:
    - `1x A100 80GB`
    - warm start from `models/world_model/P4_d4_main_telemetry_horizon/rssm_sequence_epoch_005.pt`
    - checkpoint-compatible horizon config with throughput scaled up:
      - `batch_size=128`
      - `num_workers=24`
      - `prefetch_factor=6`
      - `use_amp=true`
  - final train / validation snapshot:
    - checkpoint: `models/world_model/P5_d4_main_telemetry_a100_bs128_e15/rssm_sequence_epoch_015.pt`
    - hallucination metrics:
      - `mean_mse=0.004041`
      - `mean_ssim=0.9651`
      - `sharp_turn_ssim=0.9159`
      - `gentle_turn_ssim=0.9307`
      - `recovery_ssim=0.9119`
    - W&B run:
      - `https://wandb.ai/yuv05malik-university-of-pennsylvania/racing-world-model/runs/nv6lx85v`
  - held-out evaluation:
    - output: `results/world_model/control/reward_faithfulness_p5_d4_main_telemetry_a100_bs128_e15.json`
    - reward metrics:
      - `mean_mse`: `0.1007`
      - `mean_mae`: `0.0965`
      - `mean_corr`: `0.8387`
      - `mean_sign_match`: `0.8921`
      - `mean_bias`: `0.0090`
    - telemetry metrics:
      - `speed`: `mean_mse=2.8807`, `mean_mae=0.8186`, `mean_corr=0.7750`
      - `progress_delta`: `mean_mse=0.001131`, `mean_mae=0.01933`, `mean_corr=0.0713`
      - `steer`: `mean_mse=0.006319`, `mean_mae=0.06235`, `mean_corr=0.6478`
      - `corner_angle`: `mean_mse=0.000556`, `mean_mae=0.01735`, correlation still low-signal / undefined
      - `offtrack`: `mean_bce=0.000830`, `mean_accuracy=1.0000`
  - conclusion:
    - the A100 continuation improved `speed` substantially and kept `reward`, `steer`, and `offtrack` strong
    - hallucination quality remained high, so the physics-aware continuation did not damage the visual result
    - `progress_delta` improved in absolute error but still did not become a strongly predictive signal
    - this is strong enough to support the final project story, with `progress_delta` / ego-world advancement consistency remaining the clearest unresolved physics gap

### Q6: Manual Hard-Turn Data Injection

- purpose:
  - test whether explicitly collected manual sharp-turn driving can improve the sharp-turn hallucination regime that the earlier replay sets underrepresented
- collected datasets:
  - `results/world_model/replay/d5_manual_hard_turns_ccw_manifest.json`
    - `9` manual episodes
    - `15000` frames
    - `mean_progress=0.7727`
    - `offtrack_rate=0.7778`
  - `results/world_model/replay/d5_manual_hard_turns_ccw_fast_manifest.json`
    - `5` manual episodes
    - `10000` frames
    - `mean_progress=0.9321`
    - `offtrack_rate=1.0000`
- initial merged training set:
  - `results/world_model/replay/d6_d4_plus_d5_hard_turns_train_manifest.json`
  - `results/world_model/replay/d6_d4_plus_d5_hard_turns_val_manifest.json`
- result:
  - `D6` added the right kind of turn coverage, but in too large and too noisy a proportion
  - roughly `25000 / 39000` train frames were manual hard-turn data
  - the model became more specialized toward that regime but visual stability regressed
  - the side-by-side sharp-turn hallucinations showed scattered dot artifacts and worse geometry consistency
- conclusion:
  - the issue was not that manual turn data is useless
  - the issue was the merge ratio and noise level
  - manual turn data should be treated as targeted augmentation, not the dominant training distribution

### Q7: Curated Manual Turn Supplement

- purpose:
  - preserve genuine turn signal from manual collection without overwhelming the stable `D4_main` visual prior
- curated manual subset:
  - manifest: `results/world_model/replay/d5_manual_hard_turns_curated_train_manifest.json`
  - selected episodes:
    - `d5_manual_hard_turns_ccw/episode_000000.npz`
    - `d5_manual_hard_turns_ccw/episode_000003.npz`
    - `d5_manual_hard_turns_ccw/episode_000008.npz`
  - selected summary:
    - `3` manual episodes
    - `5579` frames
    - `mean_progress=0.9664`
    - `offtrack_rate=0.6667`
- balanced mixed dataset:
  - train: `results/world_model/replay/d6b_d4_plus_d5_curated_train_manifest.json`
  - val: `results/world_model/replay/d6b_d4_plus_d5_curated_val_manifest.json`
  - train summary:
    - `146` episodes
    - `19579` frames
    - `sources={'automatic': 138, 'ppo_policy': 5, 'manual': 3}`
    - manual contribution now about `28.5%` of train frames instead of the much larger `D6` shift
- quick continuation run:
  - run: `P6b_d6b_curated_quick`
  - warm start: `models/world_model/P5_d4_main_telemetry_a100_bs128_e15/rssm_sequence_epoch_015.pt`
  - config: `config/world_model_local_e3_diverse_horizon.yaml`
  - epochs: `2`
  - sharp-turn-only evaluation is now the intended decision criterion
- `P6b` result:
  - epoch 1:
    - `hallucination_ssim=0.9567`
    - `sharp_turn_ssim=0.9148`
  - epoch 2:
    - `hallucination_ssim=0.9565`
    - `sharp_turn_ssim=0.9088`
  - train losses at epoch 2:
    - `recon_loss=0.00442`
    - `reward_loss=0.07009`
    - `telemetry_loss=1.07708`
    - `speed_loss=0.96706`
    - `progress_delta_loss=0.000240`
    - `steer_loss=0.00832`
    - `offtrack_loss=0.05270`
    - `total_loss=1.35930`
- conclusion:
  - `D6b` is a better training set than `D6`
  - the footage stayed much crisper than the broader hard-turn injection run
  - the next disciplined experiment is a low-learning-rate `P5` fine-tune on `D6b`, not another broad continuation

### Q8: Planned P5 Fine-Tune

- purpose:
  - test whether `P5` can absorb a small amount of curated manual turn data without sacrificing the stable visual prior
- setup:
  - init checkpoint: `models/world_model/P5_d4_main_telemetry_a100_bs128_e15/rssm_sequence_epoch_015.pt`
  - train manifest: `results/world_model/replay/d6b_d4_plus_d5_curated_train_manifest.json`
  - val manifest: `results/world_model/replay/d6b_d4_plus_d5_curated_val_manifest.json`
  - config: `config/world_model_local_p5_finetune.yaml`
  - learning rate: `1e-4`
  - batch size: `8`
  - curated evaluation clips:
    - `sharp_turn` only
- status:
  - completed
- go / no-go criterion:
  - sharp-turn hallucination should improve or at least remain stable without the decoder regressions seen in `D6/P7`
- result:
  - run family: `P8_p5_finetune_d6b`
  - setup:
    - init checkpoint: `models/world_model/P5_d4_main_telemetry_a100_bs128_e15/rssm_sequence_epoch_015.pt`
    - train manifest: `results/world_model/replay/d6b_d4_plus_d5_curated_train_manifest.json`
    - val manifest: `results/world_model/replay/d6b_d4_plus_d5_curated_val_manifest.json`
    - fine-tune config: `config/world_model_local_p5_finetune.yaml`
    - sharp-turn-only evaluation
    - low learning rate, small local fine-tune
  - qualitative read:
    - results were average rather than clearly positive or clearly negative
    - future track modeling remained limited
    - it was still difficult to tell whether the ego car was truly turning through the imagined sharp turn
    - the run did not produce a decisive breakthrough, but it also did not look like a total dead end
- conclusion:
  - a very small `P5` fine-tune on curated manual turn data is not enough by itself to unlock robust future turn modeling
  - the remaining bottleneck still looks more like coverage plus representation than pure optimization time
  - `P5` remains the cleanest overall checkpoint, while `P6b/P8` are useful for clarifying the remaining gap rather than replacing it

## Intervention 1 Decision Log

- active baseline:
  - `P5`
- active claim:
  - the next disciplined test is objective-level correction, not more generic continuation training
- immediate implementation changes to test:
  - perceptual-only
  - safer progress-only
  - combined-safe only after one isolated ablation is non-destructive
- explicit non-goals for this phase:
  - no actor-critic follow-up
  - no new hero run under the old pixel-MSE objective
  - no more `D6`-style broad hard-turn merges
