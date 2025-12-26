# Training Outputs Reference

## 🎯 What Gets Saved After Training

### 1. Model File
- **Location**: `./tb3_dqn_models_pytorch/dqn_model.pth`
- **Description**: Trained neural network weights (best performing episode)
- **Used by**: `test_trained_model.py` for testing

### 2. Training Graphs (NEW!)
- **Location**: `./training_results/`
- **Files**:
  - `training_results_latest.png` - Always the most recent training run
  - `training_results_YYYYMMDD_HHMMSS.png` - Timestamped backup copies
  - `training_data_YYYYMMDD_HHMMSS.npz` - Raw numpy data

### 3. Graph Contents
The saved PNG contains 4 subplots:

1. **Top Left - Episode Rewards**
   - Blue line: Raw episode rewards
   - Red line: 10-episode moving average
   - Shows: Learning progress (should trend upward)

2. **Top Right - Training Loss**
   - Green line: Raw loss values
   - Orange line: 10-episode moving average
   - Shows: Network learning stability (should stabilize/decrease)

3. **Bottom Left - Steps per Episode**
   - Magenta line: Steps taken each episode
   - Dark red line: 10-episode moving average
   - Shows: Efficiency (successful episodes should use fewer steps)

4. **Bottom Right - Success Rate**
   - Cyan line: Rolling 20-episode success rate
   - Red dashed line: 50% target
   - Shows: Overall performance (target: 20-40% for this task)

### 4. Terminal Logs
- **Location**: `/tmp/training_log.txt`
- **Description**: Real-time training status (used by `visualize_training.py`)
- **Contents**: Episode stats, rewards, loss, epsilon, memory usage

## 📊 Viewing Results

### During Training
```bash
# Live visualization in matplotlib window
./run_with_visualization.sh
```

### After Training
```bash
# View saved graphs
eog ./training_results/training_results_latest.png

# Or use any image viewer
firefox ./training_results/training_results_latest.png
```

### Load Raw Data for Analysis
```python
import numpy as np

# Load data
data = np.load('./training_results/training_data_YYYYMMDD_HHMMSS.npz')

# Access arrays
rewards = data['rewards']      # Episode rewards
losses = data['losses']        # Episode losses  
steps = data['steps']          # Steps per episode
successes = data['successes']  # 1 if goal reached, 0 otherwise

# Calculate statistics
success_rate = np.mean(successes) * 100
avg_reward = np.mean(rewards)
print(f"Success Rate: {success_rate:.1f}%")
print(f"Average Reward: {avg_reward:.2f}")
```

## 🎓 Expected Results (ROBOTIS Stage 1)

After 200 episodes with proper configuration:

- **Success Rate**: 20-40% (rolling 20-episode window)
- **Final Loss**: 5-20 (stable, not exploding)
- **Episode Rewards**: Trending upward, occasional spikes to +100
- **Best Reward**: Should exceed 50-100 in later episodes
- **Epsilon**: Decays from 1.0 → 0.05 over 200 episodes

## 🐛 Troubleshooting

### No graphs saved
- Check `./training_results/` directory exists (auto-created)
- Verify matplotlib installed: `pip list | grep matplotlib`
- Check terminal for save errors

### Graphs look wrong
- **Loss exploding**: Network/reward configuration issue
- **Rewards always negative**: Robot never reaching goal (check environment)
- **No learning trend**: Epsilon too low, learning rate wrong, or replay buffer issue

### Can't open saved graphs
```bash
# Install image viewer if needed
sudo apt install eog

# Or use Python
python3 -c "from PIL import Image; Image.open('./training_results/training_results_latest.png').show()"
```

## 📁 Directory Structure
```
ros2_rl_project/
├── tb3_dqn_models_pytorch/
│   └── dqn_model.pth                    # Best model weights
├── training_results/                     # NEW! Auto-created
│   ├── training_results_latest.png      # Most recent graphs
│   ├── training_results_20251116_*.png  # Timestamped backups
│   └── training_data_20251116_*.npz     # Raw data arrays
└── /tmp/training_log.txt                # Live training log
```

## 🔄 Running Multiple Training Sessions

Each training run:
1. Overwrites `dqn_model.pth` (only best model kept)
2. Creates NEW timestamped graph files (no overwriting)
3. Updates `training_results_latest.png` (easy access to most recent)

To keep old models, manually backup before training:
```bash
cp ./tb3_dqn_models_pytorch/dqn_model.pth ./tb3_dqn_models_pytorch/dqn_model_backup.pth
```
