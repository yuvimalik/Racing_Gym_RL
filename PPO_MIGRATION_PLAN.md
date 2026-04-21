# PPO Migration Plan (SB3 -> Editable PyTorch)

## Goal
- Move from `stable_baselines3.PPO` to a local PyTorch PPO implementation that is fully readable and editable.
- Keep current environment/wrappers/reward shaping behavior unchanged during migration.
- Validate behavior against current baseline before full cutover.

## Current Boundary
- Training orchestration is local in `train.py`.
- Gradient/update logic is in third-party SB3:
  - `train.py` imports SB3 at `train.py:16`.
  - PPO object creation at `train.py:677`.
  - Training call at `train.py:884`.
  - Actual gradient step is in `.venv310/lib/site-packages/stable_baselines3/ppo/ppo.py` (`PPO.train`, `loss.backward()`, `optimizer.step()`).

## Migration Strategy
1. Freeze baseline and interfaces.
2. Extract project code into reusable modules (env, config, logging) independent of SB3.
3. Implement a minimal local PPO in PyTorch for continuous actions.
4. Add parity/diagnostic checks against SB3 on identical rollouts.
5. Switch default training backend to local PPO once stable.

## Phase 1: Freeze Baseline (No Algorithm Changes)
- Record baseline metrics from current SB3 run:
  - mean eval reward, lap count, off-track rate, FPS/steps-sec.
- Save one known-good config and seed for reproducibility.
- Define acceptance criteria for migration:
  - training runs end-to-end,
  - no NaNs,
  - reward trend within acceptable delta of baseline after N steps.

## Phase 2: Refactor by Responsibility
- Split `train.py` into:
  - `src/envs/factory.py` for environment + wrappers,
  - `src/utils/config.py` for config loading/validation,
  - `src/logging/callbacks.py` for progress/telemetry.
- Keep existing behavior byte-for-byte where possible.
- Add a trainer backend selector in config/CLI:
  - `trainer_backend: sb3 | torch`.

## Phase 3: Implement Local PPO Core
- New modules:
  - `src/algorithms/ppo/model.py`: Actor-Critic network (image + optional state dict observations).
  - `src/algorithms/ppo/distributions.py`: Gaussian policy head for Box actions.
  - `src/algorithms/ppo/rollout_buffer.py`: store obs/actions/rewards/dones/log_probs/values.
  - `src/algorithms/ppo/gae.py`: GAE-lambda return/advantage computation.
  - `src/algorithms/ppo/losses.py`: clipped surrogate, value loss, entropy.
  - `src/algorithms/ppo/trainer.py`: collect rollouts, minibatch epochs, backward/step, gradient clipping, checkpointing.
- Initial scope:
  - continuous Box action space only,
  - single observation mode first (image),
  - then add `MultiInputPolicy` equivalent if needed.

## Phase 4: Instrumentation and Explainability
- Log every training-loss component each update:
  - `policy_loss`, `value_loss`, `entropy_loss`, `approx_kl`, `clip_fraction`, `grad_norm`.
- Add debug mode to print tensor shapes and one minibatch example path end-to-end.
- Add optional gradient hooks for selected layers.

## Phase 5: Parity Validation
- Fixed-seed short run (e.g., 50k-100k steps) comparing SB3 vs local PPO:
  - rollout stats (reward mean/std, episode length),
  - loss magnitudes,
  - action distribution stats.
- Unit checks:
  - GAE correctness on toy trajectory,
  - PPO ratio/clipping math,
  - deterministic batching with fixed RNG.

## Phase 6: Cutover
- Make local trainer default, keep SB3 as fallback for one release cycle.
- Update docs with architecture diagrams and data flow.
- Track known differences from SB3 explicitly (if any).

## Porting Numpy -> PyTorch Rules
- Keep data as tensors on device as early as possible.
- Replace NumPy math with torch equivalents:
  - `np.mean` -> `torch.mean`,
  - `np.clip` -> `torch.clamp`,
  - `np.exp` -> `torch.exp`,
  - manual loops -> vectorized tensor ops when practical.
- Be explicit about dtypes:
  - observations `float32`,
  - actions `float32`,
  - dones/masks `float32`,
  - indices `long`.

## First Execution Slice (Recommended Next Work Item)
- Implement a tiny `TorchPPOTrainer` that can:
  - run rollout collection for `n_steps * n_envs`,
  - compute GAE,
  - run one PPO update epoch with printed losses.
- Wire it behind `--trainer_backend torch`.
- Keep existing wrappers and callbacks to minimize moving parts.

## Risks and Mitigations
- Risk: regressions caused by refactor and algorithm rewrite together.
  - Mitigation: separate refactor phases, preserve SB3 backend until parity.
- Risk: mismatch for dict observations (`MultiInputPolicy` behavior).
  - Mitigation: start with image-only parity, then incrementally add dict support.
- Risk: unstable training due to small implementation differences.
  - Mitigation: strict logging + short-seed parity tests before long runs.
