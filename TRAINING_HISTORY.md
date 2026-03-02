# Training History & Debug Log

Tracks every training run, problem encountered, root-cause diagnosis, and code/config change made.

---

## Architecture Timeline

### Phase 0 — Stable-Baselines3 (SB3) Backend (historical)
- Used SB3 `PPO` with `CnnPolicy` directly.
- **Result**: Car successfully drove laps. Reward improved from −61.5 → 282+ over 500k steps.
- **Problem**: Teacher required switching to custom PyTorch backend.

---

### Phase 1 — Custom PyTorch PPO (first attempt)
**Backend**: `TorchPPOTrainer` in `train.py`
**Architecture**: `CnnActorCritic` with shared CNN + single `self.shared` MLP (512-dim) feeding both policy head and value head.

#### Run 1 — Donut / No Movement
- **Observed**: Car either did donut-like behaviour off-track or did not move at all.
- **Cause**: Multiple compounding bugs (see below).

---

## Bug Log

### Bug 1 — GAE Off-by-One
**File**: `train.py` → `RolloutBuffer.compute_returns_advantages()`
**Symptom**: Policy instability, rewards not improving.
**Root cause**: `self.dones[step + 1]` used instead of `self.dones[step]`, causing value bootstraps to leak across episode boundaries.
**Fix**: Changed to `self.dones[step]`.

---

### Bug 2 — Steering Collapse (shared log_std)
**Observed** (step 57,344 eval):
```
throttle=1.00, brake=0.00, speed=11.28
steer_var=0.00000, progress=0.00%, offtrack_rate=100%
```
**Root cause**: Single shared `log_std` parameter — throttle saturated quickly, pulling all dimensions toward low std. Steering collapsed to one fixed direction.
**Fix**:
- Per-dimension `log_std` init: steer=0.0, throttle=−0.5, brake=−1.0
- Raised `min_log_std` −1.5 → −1.0
- Raised `ent_coef` 0.01 → 0.03

---

### Bug 3 — Policy Collapse (shared MLP head)
**Observed** (step 73,728 eval):
```
throttle=0.00, brake=0.19, speed=0.00
steer_var=0.00000, progress=0.00%
FailFast triggered
```
**Root cause**: Value function gradient flowing back through the single shared MLP (`self.shared`) corrupted policy features. SB3's `CnnPolicy` avoids this with separate MLP heads for actor and critic — our custom net did not.
**Fix**: Replaced `self.shared → Linear(n_flatten, 512)` with separate:
- `self.policy_mlp`: `n_flatten → 256 → 128` (feeds `policy_mean`)
- `self.value_mlp`: `n_flatten → 256 → 128` (feeds `value_head`)

`_latent()` removed; `get_dist_and_value()` routes through separate paths.
**Also**: `n_epochs` 10 → 4, `vf_coef` 0.5 → 0.25.

---

### Bug 4 — Reward Hacking via Velocity·Track (off-track driving)
**Observed** (eval of `best_model_torch.pt`):
```
Mean Reward: 6861.91, Progress: 0.00%, Off-track Rate: 100%
Episode Length: 953 steps, Steer Variance: 0.05178
```
**Root cause**: `comp_forward = forward_progress_scale × dot(velocity, track_dir)`
This rewards moving fast in the approximate track direction regardless of whether the car is on the track. The car learned to drive at full throttle on grass in the track direction, earning ~12 reward/step vs. only −3/step off-track penalty. Result: 6861 reward, 0% actual lap progress.

**Fix**:
- Changed `comp_forward` to use **actual tile progress delta**: `forward_progress_scale × progress_delta`
- `forward_progress_scale`: 1.5 → **300.0** (scales progress 0→1 to meaningful reward per lap)
- Gated `comp_straight_speed` to **on-track only** (`× 0 if is_offtrack`)
- `throttle_bonus_scale`: 0.3 → **0.0** (was also gamed off-track)
- `off_track_mode`: penalty → **terminate** (instant episode end on grass)
- `idle_penalty`: −0.2 → **−0.5**

Progress_delta computation moved to top of `step()` so it is available for both reward and stuck detection.

---

### Bug 5 — Donut Behaviour (spinning on track)
**Observed**: Car moves forward briefly then spins in circles on the track for the rest of the episode.
**Root cause**: Stuck detection only fires when `speed < 0.8 AND progress < 0.001`. A car doing donuts has high speed (passes the speed check) but zero tile progress — counter never increments, car spins forever with no termination.

**Fixes**:
1. **No-progress termination** (new `_no_progress_steps` counter):
   Increments every step `progress_delta < epsilon`, regardless of speed.
   Terminates after `no_progress_max_steps` (200) steps. Catches fast-spinning that old stuck detector missed.
   - `no_progress_max_steps: 200`, `no_progress_terminal_penalty: −15.0`

2. **Yaw-rate penalty** (`comp_yaw = −yaw_rate_penalty × |angular_velocity|`):
   Directly taxes spinning via Box2D `hull.angularVelocity`. Normal cornering has mild yaw; sustained donuts have very high yaw.
   - `yaw_rate_penalty: 0.1`

3. **Higher entropy**: `ent_coef` 0.03 → **0.05** to prevent steering from collapsing to a fixed angle.

---

## Current Configuration (as of latest changes)

### PPO Hyperparameters
| Param | Value |
|---|---|
| learning_rate | 2.5e-4 |
| n_steps | 1024 |
| batch_size | 256 |
| n_epochs | 4 |
| gamma | 0.99 |
| gae_lambda | 0.95 |
| clip_range | 0.2 |
| ent_coef | 0.05 |
| vf_coef | 0.25 |
| max_grad_norm | 0.5 |
| min_log_std | −1.0 |
| max_log_std | 0.5 |

### Reward Components (per step)
| Component | Formula | Purpose |
|---|---|---|
| `comp_forward` | `300 × progress_delta` | Actual lap tile progress (primary signal) |
| `comp_straight_speed` | `0.05 × speed × (on_track)` | Speed bonus, gated to track only |
| `comp_corner_overspeed` | `−0.2 × max(0, speed−8)` (corners) | Slow down for turns |
| `comp_apex_decel` | `+0.2 × speed_delta` (corners) | Reward braking into turns |
| `comp_steer_smooth` | `−0.02 × |steer_delta|` | Smooth steering |
| `comp_yaw` | `−0.1 × |yaw_rate|` | Penalise spinning/donuts |
| `comp_time` | `−0.02` | Time pressure |
| `comp_idle` | `−0.5` (if speed < 1.0) | Discourage stopping |
| `comp_launch` | `+0.8 × throttle` (first 150 steps) | Launch boost |
| `comp_brake` | `−0.05 × brake` (straights) | Discourage braking off-corners |

### Termination Events
| Event | Condition | Penalty |
|---|---|---|
| Off-track | `driving_on_grass[0] == True` | −10 + done |
| Stuck | `speed < 0.8 AND progress < 0.001` for 150 steps | −30 + done |
| No-progress | `progress < 0.001` for 200 steps (any speed) | −15 + done |

### Architecture
- **Shared CNN**: Conv(32,8,4) → Conv(64,4,2) → Conv(64,3,1) → Flatten
- **Policy MLP**: n_flatten → 256 → 128 → `policy_mean(3)`
- **Value MLP**: n_flatten → 256 → 128 → `value_head(1)`
- **log_std**: Per-dimension learnable parameter (steer=0.0, throttle=−0.5, brake=−1.0 init)

---

## Benchmarks

| Run | Steps | Mean Reward | Progress | Off-track | Notes |
|---|---|---|---|---|---|
| SB3 baseline | 500k | 282+ | ~100% | low | Working reference |
| Torch run 1 (GAE bug) | 57k | −1872 | 0% | 100% | Steering collapsed |
| Torch run 2 (shared head) | 73k | −64 | 0% | 0% | Policy collapsed, no movement |
| Torch run 3 (best_model_torch) | ~933k | 6861 | 0% | 100% | Velocity reward exploit, donuts |
| **Target** | 500k+ | >500 | >50% | <20% | Car stays on track and makes laps |

---

## Known Remaining Issues / Watchlist
- Steering still tends toward low variance after enough training — monitor `steer_var` in eval logs
- `no_progress_max_steps=200` may be too tight or too loose — adjust if car terminates too early on slow corners
- Need more total training steps (currently capped at 500k) once basic track-following works
- Visual eval (50k interval) shows raw behaviour — key signal is `progress > 0` in early evals
