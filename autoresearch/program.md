# Racing PPO Research Program

## Objective
Maximize `mean_reward` over 20-episode deterministic evaluation on MultiCarRacing-v0.

## Current Baseline
- CNN Actor-Critic with separate policy/value MLPs
- Gaussian action distribution (tanh steer, sigmoid throttle/brake)
- Standard PPO with clipped objective
- Adam optimizer, fixed LR

## Search Priorities (try in this order)

### 1. Action Distribution
- **Beta distribution** for throttle/brake (naturally bounded [0,1], no sigmoid squashing)
- Squashed Gaussian alternative (tanh + log_prob correction)
- Compare entropy behavior vs current Gaussian+sigmoid

### 2. Network Architecture
- LayerNorm after conv layers (stabilize training)
- Deeper value head (3 layers instead of 2 — value estimation is harder)
- Skip/residual connections in MLP heads
- Smaller conv filters for fine-grained steering

### 3. Learning Rate Schedule
- Cosine annealing with warmup (1000 steps warmup → cosine to 0)
- Linear decay as alternative
- Per-parameter LR (lower for conv, higher for policy head)

### 4. Entropy Schedule
- Start high (0.1) for exploration → decay to 0.01 by end
- Adaptive entropy (target entropy threshold)

### 5. GAE Lambda
- Try 0.9 (less bias, more variance) vs 0.97 (baseline-like)
- Try 0.92 as compromise

### 6. Value Loss
- Huber loss instead of MSE (robust to outliers from terminal penalties)
- Value function clipping

### 7. Observation Processing
- Running mean/std normalization on rewards
- Frame stacking (2 frames for velocity estimation)
- Grayscale conversion (reduce input dimensionality)

## Constraints
- **Budget**: 500k timesteps per experiment, <10 min wall-clock
- **DO NOT** change the environment, reward shaping, or evaluation protocol
- **DO NOT** import from train.py — only use autoresearch.prepare
- Make ONE focused change per experiment
- If last experiment crashed, fix the bug first
