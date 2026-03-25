# Racing PPO Research Program - Cap100 Branch

## Objective
Maximize `mean_reward` while preserving the fast `cap100` driving regime and improving sharp-corner behavior.

## Starting Point
- Warm-start from the current `cap100` checkpoint.
- Environment and reward shaping are fixed by config and should not be changed by this search.
- The current useful behavior is:
  - much faster whole-track driving than the old constrained baseline
  - failures are concentrated at the sharpest corners and hairpins

## Search Priorities

### 1. Policy distribution and action parameterization
- Improve steering/throttle/brake coordination without changing environment-side reward shaping
- Prefer focused action-distribution changes over broad architecture rewrites
- Preserve fast launch and high-speed pace

### 2. Network / optimization stability
- Small architecture changes that improve corner handling are allowed
- Schedules or optimization changes that improve control at speed are allowed
- Avoid sweeping multi-change experiments

### 3. Hairpin recovery and high-speed control
- Favor changes that help the policy brake, rotate, and recover at very high speed
- Do not bias toward globally slower driving

## Constraints
- DO NOT change environment settings, safety governor, or reward shaping in code
- DO NOT change evaluation protocol
- DO NOT import from `train.py`
- Keep the same public interfaces in `autoresearch/train_ppo.py`
- Make one focused change per experiment
- If an experiment improves sharp-corner handling without losing pace, push further in that direction
