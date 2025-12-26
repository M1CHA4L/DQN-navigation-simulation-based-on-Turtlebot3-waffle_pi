# TurtleBot3 DQN Reinforcement Learning

## 🚀 Quick Start

### **Option 1: Training with Live Visualization** ⭐ RECOMMENDED
```bash
./run_with_visualization.sh
```
Opens: **Gazebo GUI** + **Live Graphs** (reward/loss/epsilon/steps) + **Training**

### **Option 2: Headless Training** (Faster)
```bash
./run_training.sh
```
Training only, no GUI - for faster performance

### **Option 3: Test Trained Model** 🎯 NEW!
```bash
./run_testing.sh
```
Test your trained model and see performance statistics (requires trained model)

### **Option 4: Test Environment First**
```bash
python3 test_environment.py
```
Verify everything works before training (recommended first time)

---

## 🧪 Testing Your Trained Model

After training, test your model's performance:

```bash
./run_testing.sh
```

### What the Test Script Does:
1. **Loads trained model** from `tb3_dqn_models_pytorch/dqn_model.pth`
2. **Runs multiple episodes** (you choose how many, default: 10)
3. **No exploration** - uses only learned policy (no random actions)
4. **Shows statistics**:
   - Success rate (% episodes reaching goal)
   - Average reward
   - Average steps per episode
   - Detailed results for each episode

### Sample Test Output:
```
Episode 1
Goal: (1.23, -0.87)
✅ Episode 1 RESULT: REACHED goal in 142 steps!
   Total Reward: 456.32

Episode 2
Goal: (-1.45, 0.92)
❌ Episode 2 RESULT: DID NOT reach goal (final dist: 0.35m, steps: 500)
   Total Reward: -42.18

TEST SUMMARY
Total Episodes: 10
Successful: 7 (70.0%)
Failed: 3 (30.0%)
Average Reward: 234.56
Average Steps: 287.3
Average Steps (successful episodes): 201.5
```

---

## 📊 What You'll See

**Terminal Output:**
```
Ep   1/200 | Steps: 125 | Reward:  -28.45 | Loss: 0.000 | ε: 0.995 | Mem: 125
Ep   1 RESULT: DID NOT reach goal (final dist 1.45m)
Ep   2/200 | Steps:  98 | Reward:  -22.31 | Loss: 0.156 | ε: 0.990 | Mem: 223
  → New best: -22.31 (saved)
Ep   2 RESULT: REACHED goal at (-0.52,1.23)
```

**Episode Outcome Messages:**
- After each episode, you'll see whether the robot reached the goal or not
- **REACHED goal at (x,y)** - Robot successfully navigated to target! 🎉
- **DID NOT reach goal (final dist Xm)** - Robot failed to reach target

**Live Graphs Window:** 4 plots updating every 2 seconds
- 📈 **Episode Reward** (blue) - trending up ↗️
- 📉 **Training Loss** (red) - decreasing ↘️
- 🎯 **Epsilon Decay** (green) - 1.0 → 0.01
- 📊 **Steps per Episode** (orange bars) - increasing

**Gazebo Window:** 
- Watch robot learn to navigate and avoid obstacles in real-time
- 🎯 **Red goal box** visible at goal position (actual SDF model, not marker!)
- Goal position changes randomly each episode for better generalization
- Uses official ROBOTIS goal_box model from turtlebot3_gazebo package

---

## 📁 Project Structure

```
ros2_rl_project/
├── train_pytorch.py                    # Main DQN training script
├── turtlebot_env_ros2.py               # Gymnasium RL environment
├── test_trained_model.py               # 🆕 Test trained model
├── visualize_training.py               # Live matplotlib graphs
├── test_environment.py                 # Verification script
├── run_training.sh                     # Headless launcher
├── run_with_visualization.sh           # GUI + graphs launcher
├── run_testing.sh                      # 🆕 Model testing launcher
├── launch/                             # ROS2 launch files
│   ├── turtlebot3_dqn_stage1_headless.launch.py
│   └── turtlebot3_dqn_stage1_with_gui.launch.py
├── tb3_dqn_models_pytorch/             # Saved models
└── turtlebot3_dqn_stage1_modified.world
```

---

## 🎓 For Your Report

### Screenshots to Take:
1. **Training graphs** after 200 episodes (all 4 plots)
2. **Gazebo** showing robot navigating successfully  
3. **Before/after comparison** (Episode 10 vs Episode 190)

### Graph Interpretation:
- **Reward**: Should improve from -150 to -25 (more positive = better navigation)
- **Loss**: Stabilizes around 0.15 (indicates learning convergence)
- **Epsilon**: Decays from 1.0 to 0.01 (exploration → exploitation transition)
- **Steps**: Increases over time (robot survives longer, navigates better)

---

## ⚙️ Technical Details

- **Framework**: ROS2 Jazzy + Gazebo Harmonic + PyTorch
- **Algorithm**: DQN with experience replay and target network
- **State Space**: 22 dimensions (20 LiDAR + distance to goal + angle to goal)
- **Action Space**: 3 discrete actions (forward, turn left, turn right)
- **Training**: 200 episodes, max 500 steps per episode
- **Neural Network**: 3 hidden layers (512 → 256 → 64 neurons)
- **Replay Buffer**: 50,000 transitions
- **Epsilon Decay**: 0.99 per episode (1.0 → 0.01)
- **Goal Generation**: Random positions in [-1.8m, 1.8m] square, avoiding obstacles
- **Goal Visualization**: Official ROBOTIS goal_box model (red cylinder entity)

---

## 🔧 Troubleshooting

### If Training Seems Broken:

**Stop everything and restart:**
```bash
# Kill all processes
pkill -9 gz
pkill -9 python3
sleep 2

# Restart training
./run_with_visualization.sh
# OR
./run_training.sh
```

---

## 💡 Important Notes

- ⚠️ **Gazebo must start BEFORE training** (15-20 second wait required)
- 📁 Training logs: `/tmp/training_log.txt`
- 💾 Best model: Auto-saved to `tb3_dqn_models_pytorch/dqn_model.pth`
- 🛑 Stop training/testing: Press `Ctrl+C` (saves automatically)
- 🎯 **Reset sequence**: Goal spawned → Robot reset (prevents conflicts)

---

## 📞 Quick Commands

```bash
# Train with visualization (EASIEST)
./run_with_visualization.sh

# Train headless (faster)
./run_training.sh

# Test trained model
./run_testing.sh

# Test environment first
python3 test_environment.py

# View training log
tail -f /tmp/training_log.txt

# Show graphs only
python3 visualize_training.py
```

