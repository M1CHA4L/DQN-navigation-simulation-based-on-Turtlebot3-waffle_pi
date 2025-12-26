#!/usr/bin/env python3
"""Test trained DQN model and visualize performance."""

import sys
import os
import ast
import subprocess
import time
import numpy as np

# Critical: Source ROS2 environment BEFORE importing rclpy
print("Sourcing ROS2 environment...")
result = subprocess.run(
    ['bash', '-c', 'source /opt/ros/jazzy/setup.bash && source /home/michael/turtlebot3_ws/install/setup.bash && python3 -c "import sys; print(sys.path)"'],
    capture_output=True,
    text=True
)
for line in result.stdout.strip().split('\n'):
    if line.startswith('['):
        paths = ast.literal_eval(line)
        for p in paths:
            if p not in sys.path:
                sys.path.insert(0, p)

import rclpy
import torch
import torch.nn as nn
from turtlebot_env_ros2 import TurtleBotEnv

class DQNNetwork(nn.Module):
    """Same network architecture as training."""
    def __init__(self, state_size, action_size):
        super(DQNNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )
    
    def forward(self, x):
        return self.net(x)

class ModelTester:
    def __init__(self, model_path):
        self.model_path = model_path
        self.state_size = 22  # 20 LiDAR + distance + angle
        self.action_size = 5  # ROBOTIS official: 5 discrete actions (sharp right, gentle right, straight, gentle left, sharp left)
        
        # Load model
        self.model = DQNNetwork(self.state_size, self.action_size)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()  # Set to evaluation mode
        print(f"✓ Model loaded from {model_path}")
    
    def select_action(self, state):
        """Select action using trained model (no exploration)."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.model(state_tensor)
            action = q_values.argmax().item()
        return action
    
    def test_episode(self, env, episode_num, max_steps=500, render=True):
        """Run one test episode."""
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        goal_reached = False
        
        print(f"\n{'='*60}")
        print(f"Episode {episode_num}")
        print(f"{'='*60}")
        print(f"Goal: ({env.goal_x:.2f}, {env.goal_y:.2f})")
        
        for step in range(max_steps):
            # Select action using trained model
            action = self.select_action(state)
            
            # Take action
            next_state, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            steps += 1
            
            # Print progress every 50 steps
            if render and (step + 1) % 50 == 0:
                if env.odom_data:
                    robot_x = env.odom_data.pose.pose.position.x
                    robot_y = env.odom_data.pose.pose.position.y
                    dist_to_goal = np.sqrt((robot_x - env.goal_x)**2 + (robot_y - env.goal_y)**2)
                    min_laser = np.min(next_state[:20]) if len(next_state) >= 20 else 0.0
                    print(f"  Step {step+1:3d} | Pos: ({robot_x:5.2f}, {robot_y:5.2f}) | "
                          f"Dist: {dist_to_goal:5.2f}m | MinLaser: {min_laser:.2f}m | "
                          f"Reward: {total_reward:7.2f} | StepRew: {reward:6.2f}")
            
            state = next_state
            
            if terminated or truncated:
                # Check if goal was reached
                if env.odom_data:
                    robot_x = env.odom_data.pose.pose.position.x
                    robot_y = env.odom_data.pose.pose.position.y
                    dist_to_goal = np.sqrt((robot_x - env.goal_x)**2 + (robot_y - env.goal_y)**2)
                    goal_reached = dist_to_goal <= env.config['thresholds']['goal_distance']
                break
        
        return {
            'steps': steps,
            'total_reward': total_reward,
            'goal_reached': goal_reached,
            'final_distance': dist_to_goal if env.odom_data else None
        }
    
    def run_tests(self, env, num_episodes=10):
        """Run multiple test episodes and collect statistics."""
        print("\n" + "="*60)
        print("TESTING TRAINED MODEL")
        print("="*60)
        print(f"Model: {self.model_path}")
        print(f"Episodes: {num_episodes}")
        print("="*60)
        
        results = []
        success_count = 0
        
        for episode in range(1, num_episodes + 1):
            result = self.test_episode(env, episode)
            results.append(result)
            
            if result['goal_reached']:
                success_count += 1
                print(f"✅ Episode {episode} RESULT: REACHED goal in {result['steps']} steps!")
            else:
                print(f"❌ Episode {episode} RESULT: DID NOT reach goal "
                      f"(final dist: {result['final_distance']:.2f}m, steps: {result['steps']})")
            
            print(f"   Total Reward: {result['total_reward']:.2f}")
            
            # Brief pause between episodes
            time.sleep(1.0)
        
        # Print summary statistics
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        print(f"Total Episodes: {num_episodes}")
        print(f"Successful: {success_count} ({success_count/num_episodes*100:.1f}%)")
        print(f"Failed: {num_episodes - success_count} ({(num_episodes-success_count)/num_episodes*100:.1f}%)")
        print()
        
        avg_reward = np.mean([r['total_reward'] for r in results])
        avg_steps = np.mean([r['steps'] for r in results])
        successful_steps = [r['steps'] for r in results if r['goal_reached']]
        
        print(f"Average Reward: {avg_reward:.2f}")
        print(f"Average Steps: {avg_steps:.1f}")
        if successful_steps:
            print(f"Average Steps (successful episodes): {np.mean(successful_steps):.1f}")
        
        print()
        print("Detailed Results:")
        for i, result in enumerate(results, 1):
            status = "✅ SUCCESS" if result['goal_reached'] else "❌ FAILED"
            print(f"  Ep {i:2d}: {status} | Steps: {result['steps']:3d} | "
                  f"Reward: {result['total_reward']:7.2f} | "
                  f"Final Dist: {result['final_distance']:.2f}m")
        
        print("="*60)
        
        return results

def main():
    # Check if model exists
    model_path = "./tb3_dqn_models_pytorch/dqn_model.pth"
    if not os.path.exists(model_path):
        print(f"❌ ERROR: Model not found at {model_path}")
        print("Please train the model first using: python3 train_pytorch.py")
        sys.exit(1)
    
    # Initialize ROS2
    if not rclpy.ok():
        print("Initializing ROS2...")
        rclpy.init()
    
    env = None
    
    try:
        # Wait for Gazebo
        print("Waiting for Gazebo to fully initialize...")
        time.sleep(5)
        
        # Create environment
        print("Creating environment...")
        env = TurtleBotEnv()
        
        # Wait for sensor data with verification
        print("Waiting for sensor data...")
        max_wait_time = 30  # 30 seconds max
        start_time = time.time()
        
        while (time.time() - start_time) < max_wait_time:
            rclpy.spin_once(env, timeout_sec=0.1)
            
            # Check if we have both laser and odom data
            if env.laser_data is not None and env.odom_data is not None:
                print("✓ Sensor data received!")
                print(f"  - LaserScan: {len(env.laser_data)} points")
                print(f"  - Odometry: Position ({env.odom_data.pose.pose.position.x:.2f}, "
                      f"{env.odom_data.pose.pose.position.y:.2f})")
                break
            
            # Progress indicator every 5 seconds
            elapsed = time.time() - start_time
            if int(elapsed) % 5 == 0 and int(elapsed) > 0:
                laser_ok = "✓" if env.laser_data is not None else "✗"
                odom_ok = "✓" if env.odom_data is not None else "✗"
                print(f"  Waiting... ({int(elapsed)}s) [Laser:{laser_ok} Odom:{odom_ok}]")
                time.sleep(1)  # Avoid multiple prints in same second
        
        # Final check
        if env.laser_data is None or env.odom_data is None:
            print("\n❌ ERROR: Failed to receive sensor data!")
            print("  Possible issues:")
            print("  - Gazebo not running properly")
            print("  - Topics not publishing (/scan, /odom)")
            print("  - ROS2 bridge issues")
            print("\nTry:")
            print("  1. Check Gazebo is running: ps aux | grep gz")
            print("  2. Check topics: ros2 topic list")
            print("  3. Check topic data: ros2 topic echo /scan --once")
            sys.exit(1)
        
        # Additional spins to stabilize
        print("Stabilizing sensor readings...")
        for _ in range(20):
            rclpy.spin_once(env, timeout_sec=0.1)
        
        # Create tester
        tester = ModelTester(model_path)
        
        # Ask user how many episodes to test
        print("\nHow many episodes would you like to test? (default: 10)")
        try:
            num_episodes = input("Enter number (1-50): ").strip()
            num_episodes = int(num_episodes) if num_episodes else 10
            num_episodes = max(1, min(50, num_episodes))  # Clamp to 1-50
        except:
            num_episodes = 10
            print(f"Using default: {num_episodes} episodes")
        
        # Run tests
        results = tester.run_tests(env, num_episodes=num_episodes)
        
        print("\n✓ Testing complete!")
        
    except KeyboardInterrupt:
        print("\n\nTesting interrupted by user.")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if env:
            env.close()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
