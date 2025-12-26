#!/bin/bash
# Training with Gazebo GUI and Live Visualization

set -e

echo "============================================"
echo "TurtleBot3 DQN Training with Visualization"
echo "============================================"
echo ""

# Setup environment
export TURTLEBOT3_MODEL=waffle_pi
export LD_PRELOAD=/lib/x86_64-linux-gnu/libc.so.6  # Fix snap library issue
source /opt/ros/jazzy/setup.bash 2>/dev/null || source /opt/ros/humble/setup.bash
source /home/michael/turtlebot3_ws/install/setup.bash

cd /home/michael/ros2_rl_project

echo "Step 1: Starting Gazebo with GUI..."
echo "(This lets you watch the robot move!)"
echo "🌍 Using DIRECT launch file (no model override)"
ros2 launch launch/turtlebot3_dqn_custom.launch.py > /tmp/gazebo.log 2>&1 &
GAZEBO_PID=$!

echo "Waiting 20 seconds for Gazebo GUI..."
sleep 20

# Check if Gazebo is running
if ! ps -p $GAZEBO_PID > /dev/null; then
    echo "⚠ Gazebo GUI may have failed (snap library issue)"
    echo "Trying headless mode instead..."
    ros2 launch launch/turtlebot3_dqn_stage1_headless.launch.py > /tmp/gazebo.log 2>&1 &
    GAZEBO_PID=$!
    sleep 15
fi

# Check topics
if ! ros2 topic list | grep -q "/scan"; then
    echo "ERROR: Topics not available!"
    kill $GAZEBO_PID 2>/dev/null || true
    exit 1
fi

echo "✓ Gazebo ready"
echo ""
echo "Step 2: Starting visualization window..."
python3 visualize_training.py > /tmp/visualizer.log 2>&1 &
VIZ_PID=$!
sleep 2

echo "✓ Visualization started"
echo ""
echo "Step 3: Starting training..."
echo "============================================"
echo ""
echo "Watch:"
echo "  - Gazebo window: See robot moving"
echo "  - Visualization: See training progress"
echo ""
echo "Press Ctrl+C to stop"
echo "============================================"
echo ""

# Start training
python3 train_pytorch.py

# Cleanup
echo ""
echo "Cleaning up..."
kill $GAZEBO_PID 2>/dev/null || true
kill $VIZ_PID 2>/dev/null || true
pkill -9 gz 2>/dev/null || true
echo "Done!"
