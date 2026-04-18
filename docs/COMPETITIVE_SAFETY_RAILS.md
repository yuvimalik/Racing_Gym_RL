# Competitive MARL Safety Rails (Phase 1 Frozen)

This document freezes the non-negotiable safety settings for the initial competitiveness sweep.
All variant configs must keep these values unchanged.

## Frozen contact and termination rails

From `config/prime_marl_2car_budget_fast.yaml`:

- `reward_shaping.multi_agent.contact_termination_mode: both`
- `reward_shaping.multi_agent.contact_terminate_steps: 1`
- `reward_shaping.multi_agent.terminate_on_hook_contact: true`
- `reward_shaping.multi_agent.contact_terminal_penalty: -75.0`
- `reward_shaping.multi_agent.contact_penalty: -2.5`
- `reward_shaping.multi_agent.sustained_contact_steps: 2`
- `reward_shaping.multi_agent.sustained_contact_penalty: -7.0`
- `reward_shaping.multi_agent.hook_contact_steps: 3`
- `reward_shaping.multi_agent.hook_contact_speed_threshold: 1.75`
- `reward_shaping.multi_agent.hook_contact_penalty: -20.0`
- `reward_shaping.multi_agent.collision_low_penalty: -3.5`
- `reward_shaping.multi_agent.collision_medium_penalty: -10.0`
- `reward_shaping.multi_agent.collision_high_penalty: -22.0`
- `reward_shaping.multi_agent.overtake_requires_clean_contact: true`

## Sweep-only tunable themes

Only tune competitiveness terms in Phase 1:

- Pace variability: front-follow penalties, open-space acceleration, governor cap.
- Overtake competitiveness: overtake bonus, relative velocity shaping, safe overtake spacing and dampening.

## Evaluation gates (used after calibration)

Safety gate vs baseline (`best_seed42_eval.json`) with small tolerance:

- `contact_rate <= baseline + 0.01`
- `hook_contact_rate <= baseline + 0.01`
- `contact_termination_rate <= baseline + 0.01`
- `offtrack_rate <= baseline + 0.02`

Competitiveness gate:

- `mean_overtakes > baseline`
- `mean_speed_std > baseline`
- `mean_progress >= baseline - 0.02`
- `mean_reward >= baseline - 5.0`
