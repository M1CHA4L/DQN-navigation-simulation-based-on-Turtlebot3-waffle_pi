#!/usr/bin/env python3
"""DQN Training using PyTorch - Fixed Version."""

import sys
import os

# Critical: Source ROS2 environment BEFORE importing rclpy
import subprocess
result = subprocess.run(
    ['bash', '-c', 'source /opt/ros/jazzy/setup.bash && source /home/michael/turtlebot3_ws/install/setup.bash && python3 -c "import sys; print(sys.path)"'],
    capture_output=True,
    text=True
)
# Add ROS2 paths to Python path
for line in result.stdout.strip().split('\n'):
    if line.startswith('['):
        import ast
        paths = ast.literal_eval(line)
        for p in paths:
            if 'ros' in p.lower() and p not in sys.path:
                sys.path.insert(0, p)

import rclpy
from turtlebot_env_ros2 import TurtleBotEnv
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import time
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
from datetime import datetime

class DQNNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 256),  
            nn.ReLU(),
            nn.Linear(256, 256),  
            nn.ReLU(),
            nn.Linear(256, 128),  # Added extra layer
            nn.ReLU(),
            nn.Linear(128, action_size)
        )
    
    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        self.memory = deque(maxlen=50000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05  #  maintain some exploration
        self.epsilon_decay = 0.9985  # Slower decay for more exploration
        self.learning_rate = 0.0005  # For more stable learning
        self.warmup_episodes = 10  # Collect random experiences first
        
        self.q_network = DQNNetwork(state_size, action_size)
        self.target_network = DQNNetwork(state_size, action_size)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        
        self.update_counter = 0
    
    def act(self, state):
        if np.random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return q_values.argmax().item()
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return 0.0
        
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)
        
        # Normalize rewards for stable training (clip to reasonable range)
        rewards = torch.clamp(rewards, -100, 100)
        
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()
        
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        loss = self.criterion(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)  # Increased from 1.0
        self.optimizer.step()
        
        self.update_counter += 1
        if self.update_counter % 200 == 0:  # Increased from 100 for more stability
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        return loss.item()
    

def save_training_graphs(episode_rewards, episode_losses, episode_steps, success_history, save_dir='./training_results'):
    """Save training metrics as graphs after training completes."""
    
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create figure with 4 subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Training Results - {timestamp}', fontsize=16, fontweight='bold')
    
    episodes = list(range(1, len(episode_rewards) + 1))
    
    # 1. Episode Rewards
    axs[0, 0].plot(episodes, episode_rewards, 'b-', alpha=0.6, label='Episode Reward')
    # Add moving average (window=10)
    if len(episode_rewards) >= 10:
        moving_avg = np.convolve(episode_rewards, np.ones(10)/10, mode='valid')
        axs[0, 0].plot(range(10, len(episode_rewards) + 1), moving_avg, 'r-', linewidth=2, label='Moving Avg (10)')
    axs[0, 0].set_xlabel('Episode')
    axs[0, 0].set_ylabel('Total Reward')
    axs[0, 0].set_title('Episode Rewards')
    axs[0, 0].legend()
    axs[0, 0].grid(True, alpha=0.3)
    
    # 2. Training Loss
    if episode_losses:
        axs[0, 1].plot(episodes[:len(episode_losses)], episode_losses, 'g-', alpha=0.6, label='Loss')
        # Add moving average
        if len(episode_losses) >= 10:
            loss_moving_avg = np.convolve(episode_losses, np.ones(10)/10, mode='valid')
            axs[0, 1].plot(range(10, len(episode_losses) + 1), loss_moving_avg, 'orange', linewidth=2, label='Moving Avg (10)')
        axs[0, 1].set_xlabel('Episode')
        axs[0, 1].set_ylabel('Loss')
        axs[0, 1].set_title('Training Loss')
        axs[0, 1].legend()
        axs[0, 1].grid(True, alpha=0.3)
    
    # 3. Episode Steps
    axs[1, 0].plot(episodes, episode_steps, 'm-', alpha=0.6, label='Steps')
    if len(episode_steps) >= 10:
        steps_moving_avg = np.convolve(episode_steps, np.ones(10)/10, mode='valid')
        axs[1, 0].plot(range(10, len(episode_steps) + 1), steps_moving_avg, 'darkred', linewidth=2, label='Moving Avg (10)')
    axs[1, 0].set_xlabel('Episode')
    axs[1, 0].set_ylabel('Steps')
    axs[1, 0].set_title('Steps per Episode')
    axs[1, 0].legend()
    axs[1, 0].grid(True, alpha=0.3)
    
    # 4. Success Rate (rolling window of 20 episodes)
    if success_history:
        window = 20
        success_rates = []
        for i in range(len(success_history)):
            start_idx = max(0, i - window + 1)
            window_successes = success_history[start_idx:i+1]
            success_rate = sum(window_successes) / len(window_successes) * 100
            success_rates.append(success_rate)
        
        axs[1, 1].plot(episodes[:len(success_rates)], success_rates, 'c-', linewidth=2, label=f'Success Rate (last {window})')
        axs[1, 1].axhline(y=50, color='r', linestyle='--', alpha=0.5, label='50% Target')
        axs[1, 1].set_xlabel('Episode')
        axs[1, 1].set_ylabel('Success Rate (%)')
        axs[1, 1].set_title(f'Success Rate (Rolling {window} Episodes)')
        axs[1, 1].set_ylim([0, 105])
        axs[1, 1].legend()
        axs[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    filepath = os.path.join(save_dir, f'training_results_{timestamp}.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"\n✅ Training graphs saved to: {filepath}")
    
    # Also save as 'latest' for easy access
    latest_filepath = os.path.join(save_dir, 'training_results_latest.png')
    plt.savefig(latest_filepath, dpi=150, bbox_inches='tight')
    print(f"✅ Latest results saved to: {latest_filepath}")
    
    plt.close()
    
    # Save raw data as numpy arrays
    data_file = os.path.join(save_dir, f'training_data_{timestamp}.npz')
    np.savez(data_file,
             rewards=episode_rewards,
             losses=episode_losses,
             steps=episode_steps,
             successes=success_history)
    print(f"✅ Training data saved to: {data_file}")


class DQNAgentMethods:
    """Additional methods for DQNAgent - these should be part of DQNAgent class above."""
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save(self, path):
        torch.save(self.q_network.state_dict(), path)

# Add the methods to DQNAgent
DQNAgent.decay_epsilon = DQNAgentMethods.decay_epsilon
DQNAgent.save = DQNAgentMethods.save


def main():
    print("\n" + "="*60)
    print("STARTING DQN TRAINING")
    print("="*60 + "\n")
    
    # Setup logging to file for visualization
    log_file = open('/tmp/training_log.txt', 'w')
    log_file.write("")  # Clear file
    log_file.close()
    
    # Initialize ROS2 FIRST
    if not rclpy.ok():
        print("Initializing ROS2...")
        rclpy.init()
    
    env = None
    
    try:
        # Wait a bit to ensure Gazebo is fully started
        print("Waiting for Gazebo to fully initialize...")
        time.sleep(5)
        
        print("Creating environment...")
        env = TurtleBotEnv()
        
        # Force some ROS2 spins to get initial data
        print("Waiting for sensor data...")
        for _ in range(30):
            rclpy.spin_once(env, timeout_sec=0.1)
            time.sleep(0.05)
        
        print("Resetting environment...")
        state, _ = env.reset()
        
        state_size = len(state)
        action_size = env.action_space.n
        
        print(f"\nEnvironment ready:")
        print(f"  State size: {state_size}")
        print(f"  Action size: {action_size}")
        print(f"  Initial state: {state[:5]}... (showing first 5)")
        if hasattr(env, 'goal_x'):
            print(f"  Initial goal: ({env.goal_x:.2f}, {env.goal_y:.2f})")
        
        agent = DQNAgent(state_size=state_size, action_size=action_size)
        
        save_dir = "./tb3_dqn_models_pytorch/"
        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(save_dir, "dqn_model.pth")
        
        episodes = 200
        batch_size = 32
        best_reward = -float('inf')
        
        # Track metrics for graphing
        all_episode_rewards = []
        all_episode_losses = []
        all_episode_steps = []
        all_successes = []
        
        print(f"\nStarting training for {episodes} episodes...")
        print("="*60 + "\n")
        
        for episode in range(episodes):
            state, _ = env.reset()
            
            # Give environment time to settle after reset
            time.sleep(0.3)
            for _ in range(5):
                rclpy.spin_once(env, timeout_sec=0.05)
            
            episode_reward = 0
            episode_loss = 0
            loss_count = 0
            steps = 0
            
            for step in range(500):
                # Ensure we have fresh sensor data
                rclpy.spin_once(env, timeout_sec=0.01)
                
                action = agent.act(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                agent.remember(state, action, reward, next_state, done)
                
                # Only train after warmup period and if we have enough samples
                if episode >= agent.warmup_episodes and len(agent.memory) >= batch_size:
                    loss = agent.replay(batch_size)
                    episode_loss += loss
                    loss_count += 1
                
                episode_reward += reward
                state = next_state
                steps += 1
                
                if done:
                    break
            
            # Only decay epsilon after warmup
            if episode >= agent.warmup_episodes:
                agent.decay_epsilon()
            
            avg_loss = episode_loss / loss_count if loss_count > 0 else 0.0
            
            # Print progress
            status = f"Ep {episode+1:3d}/{episodes}"
            status += f" | Steps: {steps:3d}"
            status += f" | Reward: {episode_reward:7.2f}"
            status += f" | Loss: {avg_loss:6.3f}"
            status += f" | ε: {agent.epsilon:.3f}"
            status += f" | Mem: {len(agent.memory)}"
            
            print(status)
            
            # Log to file for visualization
            with open('/tmp/training_log.txt', 'a') as f:
                f.write(status + '\n')
            
            # Save best model
            if episode_reward > best_reward:
                best_reward = episode_reward
                agent.save(model_path)
                print(f"  → New best: {episode_reward:.2f} (saved)")
            
            # At the end of the episode report whether the robot actually reached the goal
            try:
                reached = False
                if hasattr(env, 'odom_data') and env.odom_data is not None:
                    rx = env.odom_data.pose.pose.position.x
                    ry = env.odom_data.pose.pose.position.y
                    gx = getattr(env, 'goal_x', None)
                    gy = getattr(env, 'goal_y', None)
                    if gx is not None and gy is not None:
                        dist_to_goal = np.hypot(rx - gx, ry - gy)
                        threshold = env.config['thresholds'].get('goal_distance', 0.2)
                        if dist_to_goal <= threshold:
                            reached = True
                if reached:
                    result_msg = f"Ep {episode+1:3d} RESULT: REACHED goal at ({gx:.2f},{gy:.2f})"
                else:
                    # If we couldn't compute distance, be conservative and say not reached
                    result_msg = f"Ep {episode+1:3d} RESULT: DID NOT reach goal (final dist {dist_to_goal:.2f}m)"
            except Exception:
                result_msg = f"Ep {episode+1:3d} RESULT: Unknown (no odom)"

            print(result_msg)
            with open('/tmp/training_log.txt', 'a') as f:
                f.write(result_msg + '\n')
            
            # Track metrics
            all_episode_rewards.append(episode_reward)
            all_episode_losses.append(avg_loss)
            all_episode_steps.append(steps)
            all_successes.append(1 if reached else 0)

            # Periodic status
            if (episode + 1) % 10 == 0:
                print(f"\n--- Episode {episode+1} Summary ---")
                print(f"Best reward so far: {best_reward:.2f}")
                print(f"Memory size: {len(agent.memory)}/{agent.memory.maxlen}")
                recent_success_rate = sum(all_successes[-10:]) / min(10, len(all_successes)) * 100
                print(f"Recent success rate (last 10): {recent_success_rate:.1f}%")
                print("")
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE")
        print("="*60)
        print(f"Best reward achieved: {best_reward:.2f}")
        print(f"Model saved to: {model_path}")
        print(f"Total successes: {sum(all_successes)}/{len(all_successes)} ({sum(all_successes)/len(all_successes)*100:.1f}%)")
        
        # Save training graphs
        print("\nSaving training graphs...")
        save_training_graphs(all_episode_rewards, all_episode_losses, all_episode_steps, all_successes)
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        if env is not None:
            print("\nCleaning up...")
            env.close()
        if rclpy.ok():
            rclpy.shutdown()
        print("Done.")

if __name__ == '__main__':
    main()
