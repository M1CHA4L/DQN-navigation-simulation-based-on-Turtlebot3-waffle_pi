#!/bin/bash
# PPO Training Launcher with Gazebo GUI
# Use this when you want to SEE the robot training

set -e

echo "=================================="
echo "TurtleBot3 PPO Training (with GUI)"
echo "=================================="
echo ""

# Kill old processes
echo "Cleaning up old processes..."
pkill -9 gz 2>/dev/null || true
pkill -9 gzclient 2>/dev/null || true
pkill -9 gzserver 2>/dev/null || true
pkill -9 ruby 2>/dev/null || true
sleep 2

# Setup environment
export TURTLEBOT3_MODEL=waffle_pi
source /opt/ros/jazzy/setup.bash 2>/dev/null || source /opt/ros/humble/setup.bash
source /home/michael/turtlebot3_ws/install/setup.bash

# Suppress robot_state_publisher warnings about time
export RCUTILS_CONSOLE_OUTPUT_FORMAT="[{severity}] [{name}]: {message}"
export RCUTILS_LOGGING_USE_STDOUT=1

# Ensure PPO mode is enabled
echo "Checking environment configuration..."
if grep -q "self.use_continuous_actions = True" turtlebot_env_ros2.py; then
    echo "✓ Environment configured for PPO (continuous actions)"
else
    echo "⚠ WARNING: Environment may be in DQN mode (discrete actions)"
    echo "  For PPO, ensure turtlebot_env_ros2.py line 72 has:"
    echo "  self.use_continuous_actions = True"
    echo ""
fi

# Start Gazebo WITH GUI
echo "Starting Gazebo WITH GUI..."
cd /home/michael/ros2_rl_project

# Use the custom launch file that shows GUI
ros2 launch launch/turtlebot3_dqn_custom.launch.py &
GAZEBO_PID=$!

echo "Waiting 20 seconds for Gazebo GUI to load..."
sleep 20

# Check if Gazebo is running (check for "gz sim" process)
if ! pgrep -f "gz sim" > /dev/null; then
    echo "ERROR: Gazebo failed to start!"
    echo "Check logs: tail /tmp/gazebo_ppo.log"
    echo "Or run manually: gz topic -l"
    exit 1
fi

# Check topics (with retries)
echo "Checking ROS2 topics..."
MAX_RETRIES=5
RETRY=0
while [ $RETRY -lt $MAX_RETRIES ]; do
    if gz topic -l 2>/dev/null | grep -q "/scan"; then
        echo "✓ /scan topic found"
        break
    fi
    RETRY=$((RETRY+1))
    if [ $RETRY -lt $MAX_RETRIES ]; then
        echo "  Attempt $RETRY/$MAX_RETRIES - Waiting for /scan topic..."
        sleep 3
    else
        echo "ERROR: /scan topic not found after $MAX_RETRIES attempts!"
        echo "Available topics:"
        gz topic -l 2>/dev/null || echo "  (none - Gazebo may not be ready)"
        kill $GAZEBO_PID 2>/dev/null || true
        pkill -9 -f "gz sim" 2>/dev/null || true
        exit 1
    fi
done

echo "✓ Gazebo GUI ready"
echo "✓ Topics available"
echo ""
echo "=================================="
echo "Ready to start PPO training!"
echo "=================================="
echo ""
echo "Gazebo is now visible on your screen."
echo "The robot will start moving when training begins."
echo ""
echo "Press Enter to start PPO training..."
read

# Start PPO training
echo "Starting PPO training..."
echo "=================================="
echo ""
echo "NOTE: ROS2 'Moved backwards in time' warnings are NORMAL"
echo "      They occur on every episode reset - training IS working!"
echo ""
echo "🚀 PPO TRAINING STARTED - Watch Gazebo window!"
echo "========================================"
echo ""

# Use filtered training script to hide annoying ROS warnings
# Check if a model argument was passed (e.g. bash run_ppo_training.sh ./training_results/ppo_model_ep300.pth)
if [ -n "$1" ] && [ -f "$1" ]; then
    echo "Resuming training from checkpoint: $1"
    python3 train_ppo_filtered.py "$1"
else
    # Regular start
    python3 train_ppo_filtered.py
fi

# Cleanup on exit
echo ""
echo "Training finished. Cleaning up..."
pkill -9 gz 2>/dev/null || true
