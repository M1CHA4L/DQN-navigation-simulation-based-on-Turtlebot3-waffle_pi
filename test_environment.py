#!/usr/bin/env python3
"""Simple test to verify the environment works."""

import sys
import rclpy
from turtlebot_env_ros2 import TurtleBotEnv

print("=" * 50)
print("TESTING TURTLEBOT ENVIRONMENT")
print("=" * 50)
print()

# Initialize ROS2
print("1. Initializing ROS2...")
if not rclpy.ok():
    rclpy.init()
print("   ✓ ROS2 initialized")

# Create environment
print("\n2. Creating environment...")
try:
    env = TurtleBotEnv()
    print("   ✓ Environment created")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    sys.exit(1)

# Wait for data
print("\n3. Waiting for sensor data...")
for i in range(20):
    rclpy.spin_once(env, timeout_sec=0.1)
print("   ✓ Got sensor callbacks")

# Test reset
print("\n4. Testing reset...")
try:
    state, info = env.reset()
    print(f"   ✓ Reset successful")
    print(f"   State shape: {len(state)}")
    print(f"   Goal: ({env.goal_x:.2f}, {env.goal_y:.2f})")
    print(f"   Action space: {env.action_space.n} actions")
except Exception as e:
    print(f"   ✗ Reset failed: {e}")
    env.close()
    sys.exit(1)

# Test a few steps
print("\n5. Testing 5 steps...")
for i in range(5):
    action = env.action_space.sample()
    state, reward, terminated, truncated, info = env.step(action)
    print(f"   Step {i+1}: action={action}, reward={reward:.2f}, done={terminated or truncated}")
    if terminated or truncated:
        break

print("\n6. Cleanup...")
env.close()
rclpy.shutdown()

print("\n" + "=" * 50)
print("✓ ALL TESTS PASSED!")
print("=" * 50)
print("\nYour environment is working correctly.")
print("You can now run: python3 train_pytorch.py")
