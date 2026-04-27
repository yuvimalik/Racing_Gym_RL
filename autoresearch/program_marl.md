# Multi-Agent Racing Research Program

## Objective
Maximize balanced racing quality for `MultiCarRacing-v0` with two agents.

Primary promotion goals:
- Increase `mean_progress` and `mean_reward`.
- Reduce `offtrack_rate`, `contact_rate`, `hook_contact_rate`, and `contact_termination_rate`.
- Preserve or improve clean interaction quality measured by `mean_overtakes`.

## Current Baseline
- Maintained torch MARL stack in `train.py`.
- Shared-policy IPPO (`training.marl_paradigm: shared_policy_ippo`).
- Promoted single-agent-inspired policy variant `autoresearch_run_008`.
- Current smoke failure pattern: throttle saturation, almost no brake, low steering variance, high off-track.

## Search Priorities

### 1. Recover control before aggression
- Fix throttle-without-brake collapse.
- Improve corner entry and earlier braking.
- Increase steering variance only when it helps stay on track.
- Avoid global slowing that kills progress.

### 2. Improve balanced race interaction
- Reward clean overtakes without promoting blocking.
- Penalize persistent or hook-style contact.
- Prefer passing behavior that still keeps both cars on track.

### 3. Expand policy surface carefully
- Try policy architecture changes on the editable MARL surface first.
- Prefer small, local changes to action distribution, exploration, or shared heads.
- Keep the API contract intact: `get_policy_variants()` must return the candidate variant, and any optimizer hook must remain callable as `build_optimizer(policy, learning_rate)`.

### 4. Trainer logic is phase-two search
- Only explore deeper trainer logic after control is stable.
- Do not rewrite `train.py` directly from the loop.
- Use config mutations and the editable surface module as the default mutation surfaces.

## Mutation Rules
- Make 1-2 focused changes per candidate.
- Prefer config overrides before code changes.
- If the parent already improved a direction, push that direction incrementally.
- If a candidate regressed badly, revert and try a different axis.
- If the last run crashed, fix the crash before making a new behavioral change.

## Hard Constraints
- Do not change environment registration, wrapper wiring, or evaluation schema.
- Do not change artifact contracts expected by `run_marl_experiment.py`.
- Do not remove metrics used by promotion gates.
- Keep candidate surface modules self-contained and import-safe.
- Do not optimize for reward alone if contact or off-track rises sharply.

## Failure Patterns To Reject
- `offtrack_rate` near 1.0
- Near-zero `mean_progress`
- Very low `mean_steer_variance` with high throttle
- `mean_throttle` near 1.0 with `mean_brake` near 0
- High hook-contact or repeated contact termination
- Reward spikes caused by chaotic racing or contact farming

## Success Pattern
A strong candidate should:
- Drive farther while staying on track more often.
- Show meaningful but controlled steering.
- Brake earlier into hard turns instead of saturating throttle.
- Produce cleaner overtakes with low sustained contact.
- Beat the parent on the balanced-racing score, not just one metric.
