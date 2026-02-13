# Training History and Iterations

This document tracks the evolution of the racing RL agent, including reward function revisions, hyperparameter changes, and lessons learned throughout the training process.

## Overview

The project trains a PPO agent to drive a car in the `multi_car_racing` environment. The training process involved multiple iterations focused on addressing specific behavioral issues through reward shaping and hyperparameter tuning.

---

## Initial Training Phase

### Baseline Configuration
- **Total Timesteps**: 500,000
- **n_steps**: 2048
- **batch_size**: 64
- **n_epochs**: 10
- **Learning Rate**: 3.0e-4
- **Policy**: CnnPolicy (image-based)

### Initial Observations
- Agent learned basic driving behavior
- Successfully completed laps
- Mean reward improved from -61.5 to 282+ over training
- Episodes consistently reached max length (1000 steps)

---

## Issue 1: Speed Overweighting

### Problem Description
The agent was prioritizing high speeds over safe driving, leading to:
- Excessive speed on straight sections
- Inability to slow down appropriately for turns
- Frequent crashes when encountering sharp corners
- Poor cornering performance

### Root Cause Analysis
The base reward function from `multi_car_racing` environment:
- Rewards track progress (touching new tiles)
- Small time penalty (-0.1 per step)
- No explicit speed regulation
- Agent learned that faster = more tiles touched = higher reward

### Initial Attempts
1. **Speed Penalty**: Added negative reward proportional to speed
   - Result: Agent learned to drive very slowly everywhere
   - Issue: Global speed reduction, not selective braking

2. **Speed Cap**: Implemented safety governor to cap maximum speed
   - Result: Reduced crashes but didn't solve cornering issue
   - Issue: Still no incentive for optimal speed management

---

## Issue 2: Sharp Turn Handling

### Problem Description
Agent struggled specifically with sharp turns:
- Would maintain high speed entering corners
- Would crash or go off-track on tight turns
- No understanding of when to brake vs. accelerate

### Reward Shaping Approach

#### Sharp Turn Detection
Implemented curvature-based detection:
```python
# Calculate track curvature ahead
curvature = measure_track_curvature(track_points_ahead)
is_sharp_turn = curvature > sharp_turn_threshold  # 0.35 radians
```

#### Braking Reward for Sharp Turns
Added positive reward for braking when approaching sharp turns:
```python
if is_sharp_turn and speed > brake_min_speed:
    brake_reward = brake_action * brake_reward_scale
    # Reward scale: 0.4
    # Only applies when speed > 5.0
    # Capped at 0.5 per step
```

**Parameters**:
- `sharp_turn_threshold`: 0.35 radians (very sharp turns only)
- `sharp_turn_lookahead`: 6 track points ahead
- `brake_reward_scale`: 0.4
- `brake_min_speed`: 5.0 (only reward braking when moving fast)
- `brake_max_reward`: 0.5 (cap to avoid overpowering)

### Results
- Some improvement in cornering behavior
- Agent began to slow down before sharp turns
- Still struggled with optimal speed management

---

## Issue 3: Off-Track Behavior

### Problem Description
Agent would frequently go off-track:
- Would cut corners too aggressively
- Would drive on grass to "shortcut"
- Episodes would continue even after going off-track
- No strong disincentive for leaving the track

### Solution: Terminal Off-Track Penalty

#### Implementation
Made going off-track a terminal event with large penalty:
```python
if car_on_grass:
    reward += off_track_penalty  # -1.0 per step
    if terminal:
        reward += off_track_terminal_penalty  # -100.0
        done = True  # End episode immediately
```

**Parameters**:
- `off_track_penalty`: 1.0 (per-step penalty while on grass)
- `off_track_terminal_penalty`: -100.0 (large penalty when episode terminates)
- Episode terminates immediately when going off-track

### Results
- Strong disincentive for leaving track
- Agent learned to stay on track more consistently
- Reduced corner-cutting behavior
- Episodes ended immediately on off-track events

---

## Issue 4: Speed Management Optimization

### Problem Description
After implementing speed penalties, agent exhibited:
- **Global low speeds**: Consistently slow everywhere, not just in corners
- **No optimized braking**: Would brake unnecessarily on straights
- **Poor lap times**: Too conservative overall
- **Lack of speed variation**: No understanding of when to speed up vs. slow down

### Training Run: 1,000,000 Steps with 1024 Batch Size

#### Configuration Changes
- **Total Timesteps**: Increased to 1,000,000
- **batch_size**: Increased to 1024 (from 64)
- **n_steps**: Reduced to 1024 (from 2048)
- **num_envs**: 8 parallel environments
- **gamma**: Increased to 0.999 (from 0.99) for longer-term planning

#### Reward Shaping Attempts

**Attempt 1: Speed-Based Penalties**
```python
# Penalty proportional to speed
speed_penalty = -speed * speed_penalty_coefficient
```
- **Result**: Agent learned to minimize speed globally
- **Issue**: No differentiation between appropriate speeds for different track sections

**Attempt 2: Conditional Speed Penalties**
```python
# Only penalize high speed on sharp turns
if is_sharp_turn and speed > threshold:
    speed_penalty = -(speed - threshold) * penalty_coefficient
```
- **Result**: Some improvement but still too conservative
- **Issue**: Threshold tuning was difficult

**Attempt 3: Braking Reward (Current)**
```python
# Reward braking specifically on sharp turns
if is_sharp_turn and speed > brake_min_speed:
    brake_reward = brake_action * brake_reward_scale
```
- **Result**: Agent learned to brake for turns but still maintains low speeds globally
- **Issue**: No incentive to speed up on straights

### Current State
The agent demonstrates:
- ✅ Proper braking behavior on sharp turns
- ✅ Staying on track (terminal penalty working)
- ❌ Consistently low speeds everywhere
- ❌ No speed optimization (slow on straights, slow in corners)
- ❌ Poor lap time performance

---

## Mathematical Formulation

### Base Reward Function
```
R_base = track_progress_reward - time_penalty - backward_penalty
```

Where:
- `track_progress_reward`: +1000.0 / track_length per new tile touched
- `time_penalty`: -0.1 per step
- `backward_penalty`: -K_BACKWARD * angle_diff if driving wrong direction

### Enhanced Reward Function (Current)
```
R_total = R_base + R_shaping
```

Where `R_shaping` includes:

#### Off-Track Penalties
```
R_off_track = {
    -1.0 * steps_on_grass,           if on grass
    -100.0,                           if episode terminates off-track
    0,                                otherwise
}
```

#### Sharp Turn Braking Reward
```
R_brake = {
    brake_action * 0.4,               if is_sharp_turn AND speed > 5.0
    0,                                otherwise
}
```

Capped at 0.5 per step.

#### Steering Smoothness Penalty
```
R_steer = -0.05 * min(|steer_delta|, 0.5)
```

Penalizes large steering changes to reduce jitter.

#### Cornering and Centerline Rewards
```
R_cornering = cornering_lambda * cornering_score
R_centerline = {
    +0.5,                             if near inner edge (10% of track width)
    +0.1,                             if near outer edge (50% of track width)
    0,                                otherwise
}
```

---

## Key Lessons Learned

### 1. Reward Shaping Balance
- **Too much penalty**: Agent becomes overly conservative (global low speeds)
- **Too little penalty**: Agent takes risks and crashes frequently
- **Solution needed**: Context-aware rewards that vary by track conditions

### 2. Speed Management
- **Problem**: Speed is a global behavior, hard to condition on local track geometry
- **Challenge**: Agent needs to learn speed profiles (fast on straights, slow in corners)
- **Current limitation**: Reward function doesn't explicitly encode this pattern

### 3. Terminal Events
- **Success**: Making off-track terminal with large penalty worked well
- **Benefit**: Clear signal that going off-track is catastrophic
- **Trade-off**: Episodes end immediately, reducing data collection

### 4. Batch Size and Training Efficiency
- **1024 batch size**: Improved GPU utilization
- **8 parallel envs**: Increased sample throughput
- **Trade-off**: Larger batches may require more samples to converge

---

## Current Configuration Summary

### Hyperparameters
- **total_timesteps**: 1,000,000
- **n_steps**: 1024
- **batch_size**: 1024
- **n_epochs**: 10
- **learning_rate**: 3.0e-4
- **gamma**: 0.999
- **num_envs**: 8

### Reward Shaping Parameters
- **off_track_penalty**: 1.0 per step
- **off_track_terminal_penalty**: -100.0
- **brake_reward_scale**: 0.4
- **sharp_turn_threshold**: 0.35 radians
- **brake_min_speed**: 5.0
- **steer_smoothness_penalty**: 0.05

### Policy Architecture
- **Policy Type**: MultiInputPolicy (supports multiple observation types)
- **Observation**: 96x96 RGB image
- **Action Space**: Continuous [steering, gas, brake] ∈ [-1, 1]³

---

## Future Directions

### Potential Solutions for Speed Optimization

1. **Speed Profile Learning**
   - Add reward for maintaining optimal speed for current track section
   - Use track curvature to define speed targets
   - Reward agent for matching target speed profile

2. **Differential Rewards**
   - High reward for fast speeds on low-curvature sections
   - Low/no penalty for slow speeds on high-curvature sections
   - Explicit speed targets based on track geometry

3. **Curriculum Learning**
   - Start with simple tracks (few sharp turns)
   - Gradually increase track complexity
   - Allow agent to learn speed management incrementally

4. **Multi-Objective Optimization**
   - Separate rewards for speed and safety
   - Use Pareto optimization to balance objectives
   - Allow agent to learn trade-offs explicitly

5. **Velocity-Based Rewards**
   - Reward forward velocity (not just speed)
   - Penalize lateral velocity (sliding)
   - Encourage smooth, controlled cornering

---

## Training Iterations Timeline

1. **Initial Training** (500k steps)
   - Baseline PPO with standard hyperparameters
   - No reward shaping
   - Result: Basic driving, speed overweighting

2. **Speed Penalty Phase**
   - Added speed-based penalties
   - Result: Global low speeds

3. **Sharp Turn Detection**
   - Implemented curvature-based turn detection
   - Added braking rewards for sharp turns
   - Result: Improved cornering, but still conservative

4. **Terminal Off-Track Penalty**
   - Made off-track episodes terminal
   - Large penalty (-100) for going off-track
   - Result: Agent stays on track consistently

5. **Large-Scale Training** (1M steps, 1024 batch)
   - Increased training scale
   - Attempted speed optimization
   - Result: Still global low speeds, no optimized braking

---

## Technical Notes

### Reward Function Implementation
The reward shaping is implemented in a custom wrapper that:
- Monitors car position relative to track
- Calculates track curvature ahead
- Detects off-track conditions
- Applies penalties and bonuses accordingly

### Observation Space
- **Format**: 96x96x3 RGB image (uint8)
- **Preprocessing**: Normalized to [0, 1] float32
- **Augmentation**: Currently disabled (can be enabled for robustness)

### Action Space
- **Steering**: [-1, 1] (left to right)
- **Gas**: [0, 1] (no acceleration to full acceleration)
- **Brake**: [0, 1] (no braking to full braking)

---

## Conclusion

The training process has successfully addressed several key issues:
- ✅ Off-track behavior (terminal penalty)
- ✅ Sharp turn handling (braking rewards)
- ✅ Basic driving competence

However, speed optimization remains a challenge:
- ❌ Global speed management (too conservative)
- ❌ Lack of speed variation (no fast/slow differentiation)
- ❌ No optimized braking (brakes everywhere, not just corners)

Future work should focus on context-aware speed rewards that encourage appropriate speed profiles for different track sections rather than global speed penalties.

---

*Last Updated: [Current Date]*
*Training Status: Ongoing - Speed optimization phase*
