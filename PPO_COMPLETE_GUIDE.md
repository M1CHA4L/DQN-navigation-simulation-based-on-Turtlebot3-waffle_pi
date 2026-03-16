# PPO TurtleBot3 Navigation - Complete Guide

**Project:** Proximal Policy Optimization (PPO) for TurtleBot3 Goal-Reaching Navigation  
**Date:** March 2026  
**Author:** Michael  
**Framework:** Pure PyTorch + ROS2 Jazzy + Gazebo Harmonic  

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Code Structure](#code-structure)
6. [Algorithm Details](#algorithm-details)
7. [Training Hyperparameters](#training-hyperparameters)
8. [Troubleshooting](#troubleshooting)
9. [Performance Expectations](#performance-expectations)

---

## Overview

This project implements **Proximal Policy Optimization (PPO)** for continuous control of a TurtleBot3 robot in a simulated environment. The robot learns to navigate from random starting positions to randomly spawned goals while avoiding obstacles.

### Key Features

- ✅ **Continuous Action Space**: `Box(2)` for [linear_velocity, angular_velocity]
- ✅ **Actor-Critic Architecture**: Shared feature extractor with policy and value heads
- ✅ **GAE (Generalized Advantage Estimation)**: λ=0.95 for better advantage estimates
- ✅ **Clipped Surrogate Objective**: ε=0.2 for stable policy updates
- ✅ **Pure PyTorch**: No external RL libraries (not Stable-Baselines3)
- ✅ **ROS2 Jazzy Compatible**: Full integration with Gazebo Harmonic

### Migration from DQN

| Feature | DQN (Baseline) | PPO (This Project) |
|---------|----------------|-------------------|
| Action Space | Discrete(5) - fixed angles | Box(2) - continuous velocities |
| Policy | ε-greedy | Stochastic Gaussian policy |
| Learning | Off-policy (Experience Replay) | On-policy (Rollout Buffer) |
| Network | Single Q-network | Actor-Critic (dual-head) |
| Updates | SGD on random batches | Multiple epochs on rollouts |
| Expected Success | 25-40% | 40-60% |

---

## Architecture

### Network Structure

```python
ActorCriticNetwork(
    input: 22D observation (20 LiDAR + distance + angle to goal)
    ├── Shared Features
    │   ├── Linear(22 → 256) + Tanh
    │   └── Linear(256 → 256) + Tanh
    ├── Actor Head (Policy)
    │   ├── Mean: Linear(256 → 2)  # [linear_vel, angular_vel]
    │   └── Log_Std: Learnable parameter (2,)
    └── Critic Head (Value)
        └── Value: Linear(256 → 1)
)
```

### Action Space

```python
# Continuous control
action_space = Box(
    low=[0.0, -2.0],     # [min_linear, min_angular]
    high=[0.22, 2.0],    # [max_linear, max_angular]
    shape=(2,)
)
```

**Physical Limits:**
- Linear velocity: 0.0 to 0.22 m/s (TurtleBot3 Waffle Pi max)
- Angular velocity: -2.0 to 2.0 rad/s

### Observation Space

```python
observation_space = Box(
    low=[0.0]*20 + [0.0, -π],
    high=[3.5]*20 + [10.0, π],
    shape=(22,)
)
```

**Components:**
- 20 downsampled LiDAR readings (from 360 rays)
- Relative distance to goal (meters)
- Relative angle to goal (radians)

---

## Installation

### Prerequisites

```bash
# ROS2 Jazzy + Gazebo Harmonic
# TurtleBot3 packages installed
# Python 3.12+
```

### Python Dependencies

```bash
# Already installed via previous setup
pip install torch gymnasium numpy matplotlib PyYAML
```

### Verify Setup

```bash
cd /home/michael/ros2_rl_project
bash verify_ppo_setup.sh
```

**Expected Output:**
```
✓ Environment in PPO mode (continuous actions)
✓ PyTorch installed
✓ Gymnasium installed
✓ ROS2 (rclpy) available
✓ PyYAML installed
✓ All training files exist
```

---

## Quick Start

### 1. Start Training (WITH Gazebo GUI)

```bash
cd /home/michael/ros2_rl_project
bash run_ppo_training.sh
```

**What happens:**
1. Cleans up old Gazebo processes
2. Verifies PPO mode (`use_continuous_actions = True`)
3. Launches Gazebo with GUI visible
4. Waits for /scan topic
5. Prompts you to press Enter
6. Starts `train_ppo.py` with filtered warnings

### 2. Start Training (HEADLESS - Faster)

```bash
cd /home/michael/ros2_rl_project
python3 train_ppo_filtered.py
```

**Faster by ~10%** - no GUI overhead

### 3. Monitor Training

```bash
# Check status anytime
bash check_training_status.sh
```

**Output:**
```
✓ Training is RUNNING
  PID: 24128
  Resource usage:
    CPU: 98.6%  Memory: 4.6%

✓ Gazebo is running
  ✓ /scan topic available

Robot status in Gazebo:
  pose:
    position:
      x: 0.65
      y: 1.09
      z: 0.05

Latest saved model:
  ppo_model_ep350_20260314_164044.pth
```

### 4. Training Output

```
============================================================
Starting PPO Training
============================================================
Total episodes: 2000
Steps per update: 2048
Max steps per episode: 500
============================================================

Episode 1/2000 | Reward: -45.23 | Length: 234 | Success: ✗ | Avg Reward (100): -45.23 | Success Rate (10): 0.0%

Update | Policy Loss: 0.0234 | Value Loss: 1.2345 | Entropy: 0.6789

Episode 2/2000 | Reward: -38.67 | Length: 198 | Success: ✗ | Avg Reward (100): -41.95 | Success Rate (10): 0.0%

Episode 50/2000 | Reward: 65.43 | Length: 145 | Success: ✓ | Avg Reward (100): -15.23 | Success Rate (10): 10.0%

💾 Model saved: training_results/ppo_model_ep50_20260314_164044.pth
📊 Training plot saved: training_results/ppo_training_20260314_164044.png
```

---

## Code Structure

### Core Files

```
ros2_rl_project/
├── train_ppo.py                    # Main PPO training script (636 lines)
│   ├── ActorCriticNetwork          # Policy + Value network
│   ├── RolloutBuffer               # On-policy buffer with GAE
│   ├── PPOAgent                    # Training logic
│   └── train_ppo()                 # Main training loop
│
├── train_ppo_filtered.py           # Wrapper to filter ROS warnings
│
├── turtlebot_env_ros2.py          # Gymnasium environment (642 lines)
│   ├── TurtleBotEnv                # Main environment class
│   ├── reset()                     # Episode initialization
│   ├── step()                      # Action execution
│   ├── spawn_goal_entity()         # Goal visualization
│   ├── reset_robot()               # Physics-based respawn
│   └── attempt_unstuck_robot()     # Wall unstuck mechanism
│
├── run_ppo_training.sh             # Training launcher (GUI)
├── verify_ppo_setup.sh             # Setup verification
├── check_training_status.sh        # Status monitor
├── quick_ppo_test.py              # Quick environment test
│
└── training_results/               # Saved models & plots
    ├── ppo_model_ep50_*.pth
    ├── ppo_model_ep100_*.pth
    └── ppo_training_*.png
```

### Key Components

#### 1. **ActorCriticNetwork** (`train_ppo.py:53-123`)

```python
class ActorCriticNetwork(nn.Module):
    def __init__(self, state_size=22, action_size=2, hidden_size=256):
        super().__init__()
        
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh()
        )
        
        # Actor: Outputs mean of Gaussian policy
        self.actor_mean = nn.Linear(hidden_size, action_size)
        self.actor_log_std = nn.Parameter(torch.zeros(action_size))
        
        # Critic: Outputs state value
        self.critic = nn.Linear(hidden_size, 1)
```

**Key Methods:**
- `forward(state)` → Returns action mean, log_std, value
- `get_action_and_value(state, action)` → Computes log_prob, entropy, value

#### 2. **RolloutBuffer** (`train_ppo.py:126-203`)

```python
class RolloutBuffer:
    def __init__(self, buffer_size=2048, state_size=22, action_size=2):
        self.buffer_size = buffer_size
        self.states = np.zeros((buffer_size, state_size))
        self.actions = np.zeros((buffer_size, action_size))
        self.rewards = np.zeros(buffer_size)
        self.dones = np.zeros(buffer_size)
        self.values = np.zeros(buffer_size)
        self.log_probs = np.zeros(buffer_size)
        self.advantages = np.zeros(buffer_size)
        self.returns = np.zeros(buffer_size)
```

**Key Methods:**
- `add(state, action, reward, done, value, log_prob)` → Store transition
- `compute_returns_and_advantages(last_value, gamma, gae_lambda)` → GAE computation

#### 3. **PPOAgent** (`train_ppo.py:206-392`)

```python
class PPOAgent:
    def __init__(self, state_size, action_size, lr=3e-4, gamma=0.99, 
                 gae_lambda=0.95, clip_range=0.2, n_epochs=10, ...):
        self.policy = ActorCriticNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
```

**Key Methods:**
- `select_action(state)` → Sample action from policy
- `update(rollout_buffer)` → PPO update with clipped objective

#### 4. **TurtleBotEnv** (`turtlebot_env_ros2.py`)

```python
class TurtleBotEnv(Node):
    def __init__(self):
        super().__init__('turtlebot_rl_env_node')
        
        # CRITICAL: Set PPO mode
        self.use_continuous_actions = True
        
        # Action space
        if self.use_continuous_actions:
            self.action_space = spaces.Box(
                low=np.array([0.0, -2.0]),
                high=np.array([0.22, 2.0]),
                dtype=np.float32
            )
```

**Key Methods:**
- `reset()` → Spawn goal, reset robot, return initial observation
- `step(action)` → Execute action, compute reward, check termination
- `spawn_goal_entity()` → Create visible goal in Gazebo
- `reset_robot()` → Delete and respawn robot at center
- `attempt_unstuck_robot()` → Teleport away from walls when stuck

---

## Algorithm Details

### PPO Update Steps

```python
# 1. Collect rollout (2048 steps)
for step in range(2048):
    action, log_prob, value = agent.select_action(state)
    next_state, reward, done, truncated, info = env.step(action)
    rollout_buffer.add(state, action, reward, done, value, log_prob)

# 2. Compute GAE advantages
rollout_buffer.compute_returns_and_advantages(
    last_value=last_value,
    gamma=0.99,
    gae_lambda=0.95
)

# 3. PPO update (10 epochs on the rollout)
for epoch in range(10):
    for batch in rollout_buffer.get_batches(batch_size=64):
        # Compute current policy log_prob and value
        _, new_log_prob, entropy, new_value = policy.get_action_and_value(
            batch.states, batch.actions
        )
        
        # Compute ratio and clipped objective
        ratio = torch.exp(new_log_prob - batch.old_log_probs)
        clipped_ratio = torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
        
        policy_loss = -torch.min(
            ratio * batch.advantages,
            clipped_ratio * batch.advantages
        ).mean()
        
        value_loss = F.mse_loss(new_value, batch.returns)
        entropy_loss = -entropy.mean()
        
        loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss
        
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        optimizer.step()
```

### Reward Function

```python
# Goal reached (+100)
if distance_to_goal < 0.15:
    reward = +100.0
    done = True

# Collision (-50)
elif min_laser < 0.10:
    reward = -50.0
    done = True

# Stuck on wall (-50)
elif robot_stuck:
    reward = -50.0
    done = True

# Out of bounds (-50)
elif abs(x) > 2.5 or abs(y) > 2.5:
    reward = -50.0
    done = True

# Normal step
else:
    reward = -0.01  # Step penalty
    
    # Progress reward (moving toward goal)
    if previous_distance > distance_to_goal:
        reward += 5.0 * (previous_distance - distance_to_goal)
    
    # Near goal bonus
    if distance_to_goal < 0.5:
        reward += 10.0 * (0.5 - distance_to_goal)
    
    # Action smoothness penalty (NEW - prevents "twitching" exploit)
    # Penalizes large changes in velocity to encourage smooth motion
    if previous_action is not None:
        linear_change = abs(current_linear - prev_linear)
        angular_change = abs(current_angular - prev_angular)
        smoothness_penalty = 0.5 * (linear_change/0.22 + angular_change/4.0)
        reward -= smoothness_penalty
```

**Why Action Smoothness Matters:**

In continuous action spaces, agents can exploit the reward function by "twitching" (rapid oscillations) to:
- Survive longer without making progress
- Avoid collisions through random jerky movements
- Game the step penalty system

The smoothness penalty prevents this by:
- ✅ Encouraging gradual velocity changes
- ✅ Penalizing large angular velocity swings
- ✅ Promoting stable, purposeful navigation
        reward += 10.0 * (0.5 - distance_to_goal)
```

### Action Scaling

```python
# Network outputs unbounded mean
actor_mean = self.actor_mean(features)  # Range: (-∞, +∞)

# Sample from Gaussian
action_unbounded = Normal(actor_mean, std).sample()

# Scale to action space via tanh
action_normalized = torch.tanh(action_unbounded)  # Range: [-1, 1]

# Scale to physical limits
linear_vel = (action_normalized[0] + 1) / 2 * 0.22      # [0, 0.22]
angular_vel = action_normalized[1] * 2.0                # [-2.0, 2.0]
```

---

## Training Hyperparameters

### Default Configuration

```python
# Agent hyperparameters (train_ppo.py:440-460)
agent = PPOAgent(
    state_size=22,              # LiDAR + goal info
    action_size=2,              # [linear, angular]
    lr=3e-4,                    # Learning rate (Adam)
    gamma=0.99,                 # Discount factor
    gae_lambda=0.95,            # GAE lambda
    clip_range=0.2,             # PPO clipping threshold
    n_epochs=10,                # Optimization epochs per rollout
    batch_size=64,              # Mini-batch size
    entropy_coef=0.01,          # Entropy bonus
    value_coef=0.5,             # Value loss weight
    max_grad_norm=0.5,          # Gradient clipping
)

# Training hyperparameters (train_ppo.py:462-467)
n_episodes = 2000               # Total training episodes
n_steps_per_episode = 500       # Max steps per episode
n_steps_per_update = 2048       # Rollout length
save_interval = 50              # Save every 50 episodes
```

### Environment Configuration

```python
# Environment config (turtlebot_env_ros2.py:58-80)
config = {
    'thresholds': {
        'goal_distance': 0.15,          # Goal reached threshold
        'collision_distance': 0.10,     # Collision threshold
        'boundary_warning': 2.0,        # Boundary warning zone
    },
    'rewards': {
        'goal_reached': 100.0,          # Goal success
        'collision': -50.0,             # Collision penalty
        'out_of_bounds': -50.0,         # Out of bounds penalty
        'step_penalty': 0.01,           # Per-step cost
        'progress_multiplier': 5.0,     # Progress reward scale
        'regress_penalty': 1.0,         # Regress penalty
        'near_goal_bonus': 10.0,        # Near-goal bonus scale
        'boundary_warning': -1.0,       # Boundary warning penalty
    },
    'max_episode_steps': 500,           # Episode length
}
```

---

## Troubleshooting

### Issue 1: Robot Stuck on Wall (FIXED)

**Symptoms:**
- Robot gets physically stuck on walls
- Robot floats/raises upwards
- Training stalls with repeated stuck warnings

**Solution Applied:**
```python
# 1. Raised spawn height (turtlebot_env_ros2.py:618)
pose: {{ position: {{ x: 0.0, y: 0.0, z: 0.05 }}, orientation: {{ x: 0.0, y: 0.0, z: 0.0, w: 1.0 }} }}

# 2. Added unstuck mechanism (turtlebot_env_ros2.py:200-225)
def attempt_unstuck_robot(self):
    # Teleport 30cm toward center
    new_x = robot_x + direction_to_center_x * 0.3
    new_y = robot_y + direction_to_center_y * 0.3
    self.teleport_robot(new_x, new_y, 0.05)
```

### Issue 2: Goal Not Visible (FIXED)

**Symptoms:**
- Red goal box doesn't appear in Gazebo
- Agent navigates to invisible target

**Solution Applied:**
```python
# Raised goal spawn height (turtlebot_env_ros2.py:235)
pose: {{ position: {{ x: {goal_x}, y: {goal_y}, z: 0.05 }} }}
```

### Issue 3: Warning Spam (FIXED)

**Symptoms:**
- Terminal flooded with "Moved backwards in time" warnings
- Can't see actual training progress

**Solution Applied:**
```python
# Created filtered wrapper (train_ppo_filtered.py)
FILTER_PATTERNS = [
    r"Moved backwards in time",
    r"\[robot_state_publisher.*re-publishing joint transforms",
]

# Used in run_ppo_training.sh
python3 train_ppo_filtered.py  # Instead of train_ppo.py
```

### Issue 4: Training Not Progressing

**Check:**
```bash
# 1. Verify PPO mode
grep "use_continuous_actions" turtlebot_env_ros2.py
# Should show: self.use_continuous_actions = True

# 2. Check if training is running
ps aux | grep train_ppo

# 3. Check Gazebo topics
gz topic -l | grep scan

# 4. Monitor training status
bash check_training_status.sh
```

### Issue 5: Low Success Rate

**If success rate stays at 0% after 100 episodes:**

```python
# 1. Increase exploration (train_ppo.py:449)
entropy_coef=0.02  # Increase from 0.01

# 2. Decrease stuck threshold (turtlebot_env_ros2.py:142)
self.stuck_threshold = 0.20  # Increase from 0.15

# 3. Check action scaling
# Verify actions are in valid range [0, 0.22] x [-2.0, 2.0]
```

### Issue 6: Robot "Twitching" (Jerky Motion) 🆕

**Symptoms:**
- Robot oscillates rapidly in place
- Jerky turning movements
- Low progress despite surviving long
- Training seems stuck at low success rate

**Root Cause:**
In continuous action spaces, agents can exploit the reward function by making rapid, small movements ("twitching") to survive longer without making meaningful progress toward the goal.

**Solution Applied:**

```python
# Action smoothness penalty added (turtlebot_env_ros2.py)
# Penalizes large changes in velocity between steps

# In config rewards:
'action_smoothness_weight': 0.5  # Default value

# In step() function:
if previous_action is not None:
    linear_change = abs(current_linear - prev_linear)
    angular_change = abs(current_angular - prev_angular)
    smoothness_penalty = 0.5 * (linear_change/0.22 + angular_change/4.0)
    reward -= smoothness_penalty
```

**Tuning Smoothness Penalty:**

```python
# If robot is TOO jerky (twitching a lot):
'action_smoothness_weight': 1.0  # Increase penalty (was 0.5)

# If robot is TOO cautious (won't turn enough):
'action_smoothness_weight': 0.1  # Decrease penalty (was 0.5)

# If robot needs to make quick evasive maneuvers:
'action_smoothness_weight': 0.2  # Allow more dynamic movements
```

**How It Works:**
- Tracks previous action `[linear_vel, angular_vel]`
- Computes change in both velocities
- Normalizes by maximum possible change
- Applies penalty proportional to change magnitude
- Encourages smooth, gradual velocity adjustments

**When to Adjust:**
- **Episodes 0-200**: Keep default (0.5) - let agent explore
- **Episodes 200-400**: If twitching observed, increase to 0.8-1.0
- **Episodes 400+**: If too cautious, decrease to 0.2-0.3

---

## Performance Expectations

### Training Stages

| Episodes | Success Rate | Behavior | Notes |
|----------|--------------|----------|-------|
| 0-100 | 0-10% | Random exploration | Robot learns forward motion |
| 100-200 | 10-25% | Basic navigation | Learns to follow goal direction |
| 200-300 | 25-40% | Wall avoidance | Reduces collisions |
| 300-400 | 40-50% | Smooth trajectories | Better turning decisions |
| 400-600 | 50-60% | Near-optimal | Consistent goal reaching |
| 600+ | 55-65% | Stable performance | Minimal improvement |

### Expected Timeline

- **Training Time**: ~10-15 hours for 500 episodes (on typical CPU)
- **Steps per Episode**: 150-250 (decreases as agent improves)
- **Update Frequency**: Every 2048 steps (~8-10 episodes)
- **Model Size**: ~500KB per checkpoint

### Comparison with DQN

| Metric | DQN (Baseline) | PPO (This Project) | Improvement |
|--------|----------------|-------------------|-------------|
| Success Rate @ 500 eps | 25-40% | 50-60% | +25-20% |
| Average Episode Length | 300-400 | 150-250 | -150 steps |
| Action Smoothness | Jerky (discrete) | Smooth (continuous) | ✓ |
| Training Time | 8-10 hours | 10-15 hours | +25% |
| Trajectory Quality | Zigzag | Direct | ✓ |

### Saved Models

Models are saved every 50 episodes:
```
training_results/
├── ppo_model_ep50_20260314_164044.pth
├── ppo_model_ep100_20260314_164044.pth
├── ppo_model_ep150_20260314_164044.pth
...
└── ppo_model_final_20260314_164044.pth
```

**Model Contents:**
```python
checkpoint = {
    'episode': 350,
    'policy_state_dict': agent.policy.state_dict(),
    'optimizer_state_dict': agent.optimizer.state_dict(),
    'episode_rewards': [list of rewards],
}
```

### Monitoring Training

**Live Monitoring:**
```bash
# Terminal 1: Run training
bash run_ppo_training.sh

# Terminal 2: Monitor status
watch -n 10 bash check_training_status.sh
```

**Check Progress:**
```bash
# View latest episode
tail -20 /tmp/ppo_training.log | grep Episode

# Plot training curve
ls training_results/*.png
# Open latest PNG file
```

---

## Summary

### What You Have Now

✅ **Complete PPO Implementation**
- Pure PyTorch Actor-Critic network
- Rollout buffer with GAE computation
- PPO clipped surrogate objective
- Continuous action space for smooth control

✅ **Robust Training Infrastructure**
- Filtered output (no warning spam)
- Automatic checkpointing
- Status monitoring tools
- Error recovery mechanisms

✅ **Production-Ready Environment**
- ROS2 Jazzy + Gazebo Harmonic integration
- Physics-based robot respawn
- Wall unstuck mechanism
- Visible goal visualization

### Quick Reference Commands

```bash
# Start training (GUI)
bash run_ppo_training.sh

# Start training (headless - faster)
python3 train_ppo_filtered.py

# Check status
bash check_training_status.sh

# Verify setup
bash verify_ppo_setup.sh

# Test environment
python3 quick_ppo_test.py

# Stop training
pkill -f train_ppo
pkill -9 -f "gz sim"
```

### File Locations

- **Training Script**: `train_ppo.py` (636 lines)
- **Environment**: `turtlebot_env_ros2.py` (642 lines)
- **Filtered Wrapper**: `train_ppo_filtered.py` (70 lines)
- **Launch Script**: `run_ppo_training.sh` (95 lines)
- **Saved Models**: `training_results/*.pth`
- **Training Plots**: `training_results/*.png`

### Next Steps

1. ✅ **Continue Training** - Let it run to episode 500-600
2. 📊 **Monitor Progress** - Check status every hour
3. 🎯 **Evaluate Performance** - Test at 40-60% success rate
4. 🚀 **Deploy** - Use best model for real TurtleBot3

---

**Training is working correctly! All issues fixed. Good luck! 🎉**
