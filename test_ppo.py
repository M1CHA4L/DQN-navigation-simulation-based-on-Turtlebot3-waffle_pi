#!/usr/bin/env python3
"""
Quick test script to verify PPO implementation.
Tests:
1. Action space is continuous
2. Network forward pass works
3. Action sampling produces valid velocities
4. Environment step accepts continuous actions
"""

import sys
import os
import numpy as np
import torch

# Add ROS2 paths
import subprocess
result = subprocess.run(
    ['bash', '-c', 'source /opt/ros/jazzy/setup.bash && source /home/michael/turtlebot3_ws/install/setup.bash && python3 -c "import sys; print(sys.path)"'],
    capture_output=True,
    text=True
)
for line in result.stdout.strip().split('\n'):
    if line.startswith('['):
        import ast
        paths = ast.literal_eval(line)
        for p in paths:
            if 'ros' in p.lower() and p not in sys.path:
                sys.path.insert(0, p)

import rclpy
from turtlebot_env_ros2 import TurtleBotEnv
from train_ppo import ActorCriticNetwork, PPOAgent


def test_environment():
    """Test that environment uses continuous actions."""
    print("\n" + "="*60)
    print("TEST 1: Environment Action Space")
    print("="*60)
    
    if not rclpy.ok():
        rclpy.init()
    
    env = TurtleBotEnv()
    
    # Check action space
    print(f"Action space type: {type(env.action_space)}")
    print(f"Action space: {env.action_space}")
    print(f"Action shape: {env.action_space.shape}")
    print(f"Action bounds: low={env.action_space.low}, high={env.action_space.high}")
    
    assert hasattr(env.action_space, 'shape'), "❌ Action space should be Box (continuous)"
    assert env.action_space.shape == (2,), "❌ Action space should be 2D (linear, angular)"
    
    print("✅ Environment correctly configured for continuous actions")
    
    env.close()
    return True


def test_network():
    """Test Actor-Critic network."""
    print("\n" + "="*60)
    print("TEST 2: Actor-Critic Network")
    print("="*60)
    
    state_size = 22  # 20 LiDAR + 2 goal info
    action_size = 2  # linear, angular
    
    network = ActorCriticNetwork(state_size, action_size)
    
    # Create dummy state
    dummy_state = torch.randn(1, state_size)
    
    # Forward pass
    action_mean, action_std, value = network.forward(dummy_state)
    
    print(f"State shape: {dummy_state.shape}")
    print(f"Action mean shape: {action_mean.shape}")
    print(f"Action std shape: {action_std.shape}")
    print(f"Value shape: {value.shape}")
    print(f"\nAction mean: {action_mean}")
    print(f"Action std: {action_std}")
    print(f"Value: {value}")
    
    assert action_mean.shape == (1, 2), "❌ Action mean should be (1, 2)"
    assert action_std.shape == (2,), "❌ Action std should be (2,)"
    assert value.shape == (1, 1), "❌ Value should be (1, 1)"
    assert torch.all(action_std > 0), "❌ Action std should be positive"
    
    print("✅ Network forward pass successful")
    return True


def test_action_sampling():
    """Test action sampling and scaling."""
    print("\n" + "="*60)
    print("TEST 3: Action Sampling and Scaling")
    print("="*60)
    
    state_size = 22
    action_size = 2
    
    agent = PPOAgent(state_size, action_size)
    
    # Create dummy state
    dummy_state = np.random.randn(state_size).astype(np.float32)
    
    # Sample 10 actions
    actions = []
    for i in range(10):
        action, log_prob, value = agent.select_action(dummy_state)
        actions.append(action)
        print(f"Action {i+1}: linear={action[0]:.4f} m/s, angular={action[1]:.4f} rad/s")
    
    actions = np.array(actions)
    
    # Check bounds
    linear_vels = actions[:, 0]
    angular_vels = actions[:, 1]
    
    print(f"\nLinear velocity range: [{linear_vels.min():.4f}, {linear_vels.max():.4f}]")
    print(f"Angular velocity range: [{angular_vels.min():.4f}, {angular_vels.max():.4f}]")
    
    assert np.all(linear_vels >= 0.0) and np.all(linear_vels <= 0.22), \
        "❌ Linear velocity outside bounds [0.0, 0.22]"
    assert np.all(angular_vels >= -2.0) and np.all(angular_vels <= 2.0), \
        "❌ Angular velocity outside bounds [-2.0, 2.0]"
    
    print("✅ All sampled actions within valid bounds")
    return True


def test_environment_step():
    """Test environment step with continuous actions."""
    print("\n" + "="*60)
    print("TEST 4: Environment Step Execution")
    print("="*60)
    
    if not rclpy.ok():
        rclpy.init()
    
    env = TurtleBotEnv()
    
    # Reset environment
    print("Resetting environment...")
    state, _ = env.reset()
    print(f"Initial state shape: {state.shape}")
    
    # Test valid action
    test_actions = [
        np.array([0.15, 0.0], dtype=np.float32),   # Straight forward
        np.array([0.10, 1.0], dtype=np.float32),   # Gentle left turn
        np.array([0.05, -1.5], dtype=np.float32),  # Sharp right turn (slow)
        np.array([0.0, 2.0], dtype=np.float32),    # Spin in place
    ]
    
    for i, action in enumerate(test_actions):
        print(f"\nStep {i+1}: linear={action[0]:.2f}, angular={action[1]:.2f}")
        
        try:
            next_state, reward, done, truncated, info = env.step(action)
            print(f"  Reward: {reward:.2f}")
            print(f"  Done: {done}")
            print(f"  Next state shape: {next_state.shape}")
            
            if done:
                print("  Episode ended, resetting...")
                state, _ = env.reset()
        except Exception as e:
            print(f"❌ Step failed with error: {e}")
            env.close()
            return False
    
    print("✅ Environment step execution successful")
    
    env.close()
    return True


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("PPO IMPLEMENTATION VERIFICATION")
    print("="*60)
    
    tests = [
        ("Environment Action Space", test_environment),
        ("Actor-Critic Network", test_network),
        ("Action Sampling", test_action_sampling),
        ("Environment Step", test_environment_step),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n❌ Test '{test_name}' failed with exception:")
            print(f"   {type(e).__name__}: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(success for _, success in results)
    
    print("="*60)
    if all_passed:
        print("🎉 ALL TESTS PASSED! PPO implementation is ready.")
        print("\nYou can now run: python3 train_ppo.py")
    else:
        print("⚠️  SOME TESTS FAILED. Please fix issues before training.")
    print("="*60)
    
    # Cleanup
    if rclpy.ok():
        rclpy.shutdown()
    
    return all_passed


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
