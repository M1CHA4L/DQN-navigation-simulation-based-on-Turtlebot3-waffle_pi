#!/bin/bash
# Setup script for PPO implementation
# This script automates the migration from DQN to PPO

set -e  # Exit on error

echo "============================================================"
echo "TurtleBot3 PPO Setup Script"
echo "============================================================"
echo ""

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to print colored output
print_success() {
    echo -e "\033[0;32m✓ $1\033[0m"
}

print_error() {
    echo -e "\033[0;31m✗ $1\033[0m"
}

print_warning() {
    echo -e "\033[0;33m⚠ $1\033[0m"
}

print_info() {
    echo -e "\033[0;36mℹ $1\033[0m"
}

# Step 1: Check prerequisites
echo "Step 1: Checking prerequisites..."
echo "-----------------------------------------------------------"

# Check Python
if command_exists python3; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    print_success "Python3 found: $PYTHON_VERSION"
else
    print_error "Python3 not found. Please install Python 3.8+"
    exit 1
fi

# Check ROS2
if [ -z "$ROS_DISTRO" ]; then
    print_warning "ROS_DISTRO not set. Attempting to source..."
    if [ -f "/opt/ros/jazzy/setup.bash" ]; then
        source /opt/ros/jazzy/setup.bash
        print_success "ROS2 Jazzy sourced"
    else
        print_error "ROS2 Jazzy not found. Please install ROS2 Jazzy."
        exit 1
    fi
else
    print_success "ROS2 $ROS_DISTRO detected"
fi

# Check PyTorch
if python3 -c "import torch" 2>/dev/null; then
    TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)")
    print_success "PyTorch found: $TORCH_VERSION"
else
    print_error "PyTorch not found. Install with: pip3 install torch"
    exit 1
fi

# Check Gymnasium
if python3 -c "import gymnasium" 2>/dev/null; then
    print_success "Gymnasium found"
else
    print_warning "Gymnasium not found. Install with: pip3 install gymnasium"
fi

echo ""

# Step 2: Backup existing files
echo "Step 2: Creating backup..."
echo "-----------------------------------------------------------"

BACKUP_DIR="./backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

if [ -f "turtlebot_env_ros2.py" ]; then
    cp turtlebot_env_ros2.py "$BACKUP_DIR/"
    print_success "Backed up turtlebot_env_ros2.py"
fi

if [ -f "train_pytorch.py" ]; then
    cp train_pytorch.py "$BACKUP_DIR/"
    print_success "Backed up train_pytorch.py"
fi

print_info "Backup created in: $BACKUP_DIR"
echo ""

# Step 3: Check environment configuration
echo "Step 3: Checking environment configuration..."
echo "-----------------------------------------------------------"

if grep -q "self.use_continuous_actions = True" turtlebot_env_ros2.py; then
    print_success "Environment configured for PPO (continuous actions)"
elif grep -q "use_continuous_actions" turtlebot_env_ros2.py; then
    print_warning "Environment has use_continuous_actions flag but set to False"
    echo ""
    read -p "Switch to PPO mode (continuous actions)? [y/N] " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        sed -i 's/self.use_continuous_actions = False/self.use_continuous_actions = True/' turtlebot_env_ros2.py
        print_success "Switched to PPO mode"
    else
        print_info "Keeping DQN mode (discrete actions)"
    fi
else
    print_error "Environment missing use_continuous_actions flag"
    print_info "Please apply the modifications from PPO_IMPLEMENTATION_GUIDE.md"
    exit 1
fi

echo ""

# Step 4: Verify files exist
echo "Step 4: Verifying PPO files..."
echo "-----------------------------------------------------------"

REQUIRED_FILES=(
    "train_ppo.py"
    "test_ppo.py"
    "visualize_ppo_vs_dqn.py"
    "PPO_IMPLEMENTATION_GUIDE.md"
    "QUICK_START_PPO.md"
)

ALL_EXIST=true
for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        print_success "$file exists"
    else
        print_error "$file not found"
        ALL_EXIST=false
    fi
done

if [ "$ALL_EXIST" = false ]; then
    print_error "Some required files are missing"
    exit 1
fi

echo ""

# Step 5: Run tests
echo "Step 5: Running verification tests..."
echo "-----------------------------------------------------------"

print_info "Running: python3 test_ppo.py"
echo ""

if python3 test_ppo.py; then
    print_success "All tests passed!"
else
    print_error "Tests failed. Please check the output above."
    exit 1
fi

echo ""

# Step 6: Generate visualizations
echo "Step 6: Generating comparison visualizations..."
echo "-----------------------------------------------------------"

read -p "Generate DQN vs PPO comparison plots? [y/N] " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if python3 visualize_ppo_vs_dqn.py; then
        print_success "Visualizations generated"
        print_info "Check: dqn_vs_ppo_trajectories.png"
        print_info "Check: dqn_vs_ppo_actions.png"
        print_info "Check: dqn_vs_ppo_velocities.png"
    else
        print_warning "Visualization generation failed (non-critical)"
    fi
else
    print_info "Skipping visualizations"
fi

echo ""

# Step 7: Check Gazebo
echo "Step 7: Checking Gazebo status..."
echo "-----------------------------------------------------------"

if pgrep -x "gz" > /dev/null; then
    print_success "Gazebo is running"
    
    # Check if topics are available
    if gz topic -l 2>/dev/null | grep -q "/scan"; then
        print_success "Gazebo topics available (/scan found)"
    else
        print_warning "Gazebo running but topics not available"
        print_info "You may need to launch the TurtleBot3 world"
    fi
else
    print_warning "Gazebo is not running"
    print_info "Start Gazebo with: bash run_training.sh"
    print_info "Or: bash clean_and_start.sh"
fi

echo ""

# Step 8: Summary and next steps
echo "============================================================"
echo "Setup Complete!"
echo "============================================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Start Gazebo (if not running):"
echo "   $ bash run_training.sh"
echo ""
echo "2. In a new terminal, start PPO training:"
echo "   $ python3 train_ppo.py"
echo ""
echo "3. Monitor training progress:"
echo "   - Episode rewards (should increase)"
echo "   - Success rate (target: 40-60% by episode 500)"
echo "   - Policy/Value losses (should decrease)"
echo ""
echo "Expected training time: 10-15 hours"
echo "Expected results: 40-60% success rate (vs 25-40% for DQN)"
echo ""
echo "============================================================"
echo "Documentation:"
echo "============================================================"
echo ""
echo "  📘 Complete Guide:    PPO_IMPLEMENTATION_GUIDE.md"
echo "  🚀 Quick Start:       QUICK_START_PPO.md"
echo "  📊 Summary:           PPO_SUMMARY.md"
echo "  🔄 Comparison:        DQN_VS_PPO_COMPARISON.md"
echo ""
echo "============================================================"
echo "Troubleshooting:"
echo "============================================================"
echo ""
echo "  Issue: Tests fail"
echo "  → Check: python3 test_ppo.py"
echo ""
echo "  Issue: Training hangs"
echo "  → Check: Gazebo is running (gz topic -l)"
echo ""
echo "  Issue: Success rate stuck at 0%"
echo "  → Solution: Increase entropy_coef to 0.02 in train_ppo.py"
echo ""
echo "  Issue: Robot only spins"
echo "  → Solution: Initialize log_std to -0.5 in ActorCriticNetwork"
echo ""
echo "============================================================"
echo ""

# Optional: Ask to start training immediately
read -p "Start PPO training now? [y/N] " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if pgrep -x "gz" > /dev/null; then
        print_info "Starting PPO training..."
        echo ""
        python3 train_ppo.py
    else
        print_error "Cannot start training: Gazebo not running"
        print_info "Start Gazebo first with: bash run_training.sh"
        exit 1
    fi
else
    print_info "Setup complete. Start training when ready with: python3 train_ppo.py"
fi

echo ""
echo "✅ Setup script finished successfully!"
