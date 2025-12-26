#!/bin/bash
# Test trained model with Gazebo GUI

set -e

echo "============================================"
echo "Testing Trained TurtleBot3 DQN Model"
echo "============================================"
echo ""

# Check if model exists
if [ ! -f "./tb3_dqn_models_pytorch/dqn_model.pth" ]; then
    echo "❌ ERROR: Trained model not found!"
    echo "Please train the model first using:"
    echo "  ./run_training.sh  OR  ./run_with_visualization.sh"
    exit 1
fi

# Setup environment
export TURTLEBOT3_MODEL=waffle_pi
export LD_PRELOAD=/lib/x86_64-linux-gnu/libc.so.6
source /opt/ros/jazzy/setup.bash 2>/dev/null || source /opt/ros/humble/setup.bash
source /home/michael/turtlebot3_ws/install/setup.bash

cd /home/michael/ros2_rl_project

echo "Step 1: Starting Gazebo with GUI..."
echo "🌍 Using DIRECT launch file (no model override)"
ros2 launch launch/turtlebot3_dqn_custom.launch.py > /tmp/gazebo.log 2>&1 &
GAZEBO_PID=$!

echo "Waiting 20 seconds for Gazebo GUI..."
sleep 20

# Check if Gazebo is running
if ! ps -p $GAZEBO_PID > /dev/null; then
    echo "⚠ Gazebo GUI failed (snap library issue)"
    echo "Trying headless mode instead..."
    ros2 launch launch/turtlebot3_dqn_stage1_headless.launch.py > /tmp/gazebo.log 2>&1 &
    GAZEBO_PID=$!
    sleep 15
fi

# Check topics
echo "Checking ROS2 topics..."
if ! ros2 topic list | grep -q "/scan"; then
    echo "ERROR: /scan topic not available!"
    echo "Gazebo may not be running properly."
    echo "Check log: cat /tmp/gazebo.log"
    kill $GAZEBO_PID 2>/dev/null || true
    exit 1
fi

if ! ros2 topic list | grep -q "/odom"; then
    echo "ERROR: /odom topic not available!"
    echo "Robot may not be spawned."
    echo "Check log: cat /tmp/gazebo.log"
    kill $GAZEBO_PID 2>/dev/null || true
    exit 1
fi

# Verify topics are actually publishing data
echo "Verifying topic data..."
if ! timeout 5 ros2 topic echo /scan --once > /dev/null 2>&1; then
    echo "WARNING: /scan topic not publishing data yet, waiting..."
    sleep 5
fi

echo "✓ Gazebo ready"
echo "✓ Topics available: /scan, /odom"
echo ""
echo "Step 2: Starting model testing..."
echo "============================================"
echo ""
echo "Watch the robot navigate using the trained model!"
echo ""
echo "Press Ctrl+C to stop"
echo "============================================"
echo ""

# Start testing
python3 test_trained_model.py

# Cleanup
echo ""
echo "Cleaning up..."
kill $GAZEBO_PID 2>/dev/null || true
pkill -9 gz 2>/dev/null || true
echo "Done!"
