#!/bin/bash
# Clean everything and start fresh training

echo "╔════════════════════════════════════════════════════════════╗"
echo "║         Clean Gazebo & Start Fresh Training                ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

echo "Step 1: Killing ALL Gazebo processes..."
# Kill all Gazebo-related processes INCLUDING ROS2-Gazebo bridges
pkill -9 gz 2>/dev/null || true
pkill -9 gzserver 2>/dev/null || true
pkill -9 gzclient 2>/dev/null || true
pkill -9 ruby 2>/dev/null || true
pkill -9 parameter_bridge 2>/dev/null || true
pkill -9 image_bridge 2>/dev/null || true
sleep 3

# Verify killed - retry if needed
for i in {1..3}; do
    if pgrep -f "gz|bridge" > /dev/null; then
        echo "  Attempt $i: Still running, killing again..."
        pkill -9 gz 2>/dev/null || true
        pkill -9 parameter_bridge 2>/dev/null || true
        pkill -9 image_bridge 2>/dev/null || true
        sleep 2
    else
        echo "✓ All Gazebo processes killed"
        break
    fi
done

# Final check - use sudo if needed
if pgrep -f "gz|bridge" > /dev/null; then
    echo "  ⚠️  Regular kill failed, trying with sudo..."
    sudo killall -9 parameter_bridge image_bridge gz gzserver gzclient ruby 2>/dev/null || true
    sleep 2
    if pgrep -f "gz|bridge" > /dev/null; then
        echo "❌ ERROR: Cannot kill processes even with sudo!"
        echo "  Please reboot your computer."
        exit 1
    else
        echo "✓ All processes killed with sudo"
    fi
fi

echo ""
echo "Step 2: Clearing Gazebo cache (optional)..."
read -p "Clear Gazebo cache? This removes cached worlds/models (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm -rf ~/.gz/sim/* 2>/dev/null || true
    rm -rf ~/.gazebo/models/* 2>/dev/null || true
    echo "✓ Gazebo cache cleared"
else
    echo "  Skipped cache clearing"
fi

echo ""
echo "Step 3: Clearing old trained model..."
if [ -f "./tb3_dqn_models_pytorch/dqn_model.pth" ]; then
    rm -rf tb3_dqn_models_pytorch/dqn_model.pth
    echo "✓ Old model deleted"
else
    echo "  No old model found"
fi

echo ""
echo "Step 4: Verifying world file..."
if grep -q "turtlebot3_dqn_world" turtlebot3_dqn_stage1_modified.world; then
    echo "✓ World file uses official ROBOTIS model"
else
    echo "❌ ERROR: World file may be incorrect!"
    echo "   Should contain: <uri>model://turtlebot3_dqn_world</uri>"
fi

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "✅ Cleanup complete! Ready to start training."
echo "══════════════════════════════════════════════════════════════"
echo ""
echo "Start training with:"
echo "  ./run_with_visualization.sh"
echo ""
echo "Expected environment:"
echo "  • 5m × 5m square arena (walls only, no pillars)"
echo "  • Robot spawns at center (0, 0)"
echo "  • Random goals within ±2.0m"
echo "  • ROBOTIS official velocities (0.15 m/s, 5 actions)"
echo ""
