#!/bin/bash
# Simple training launcher

set -e

echo "=================================="
echo "TurtleBot3 DQN Training"
echo "=================================="
echo ""

# Kill old processes
pkill -9 gz 2>/dev/null || true
sleep 1

# Setup environment
export TURTLEBOT3_MODEL=waffle_pi
source /opt/ros/jazzy/setup.bash 2>/dev/null || source /opt/ros/humble/setup.bash
source /home/michael/turtlebot3_ws/install/setup.bash

# Start Gazebo in background
echo "Starting Gazebo..."
cd /home/michael/ros2_rl_project
ros2 launch launch/turtlebot3_dqn_stage1_headless.launch.py > /tmp/gazebo.log 2>&1 &
GAZEBO_PID=$!

echo "Waiting 15 seconds for Gazebo..."
sleep 15

# Check if Gazebo is running
if ! ps -p $GAZEBO_PID > /dev/null; then
    echo "ERROR: Gazebo failed! Check /tmp/gazebo.log"
    exit 1
fi

# Check topics
if ! ros2 topic list | grep -q "/scan"; then
    echo "ERROR: /scan topic not found!"
    kill $GAZEBO_PID 2>/dev/null || true
    exit 1
fi

echo "✓ Gazebo ready"
echo "✓ Topics available"
echo ""
echo "Starting training..."
echo "=================================="
echo ""

# Start training
python3 train_pytorch.py

# Cleanup
echo ""
echo "Cleaning up..."
kill $GAZEBO_PID 2>/dev/null || true
pkill -9 gz 2>/dev/null || true
echo "Done!"
