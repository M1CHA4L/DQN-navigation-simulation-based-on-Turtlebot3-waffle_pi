#!/usr/bin/env python3
"""
PPO Training for TurtleBot3 - Pure PyTorch Implementation
Proximal Policy Optimization with continuous action space (linear + angular velocity)

Key Differences from DQN:
- On-policy (no replay buffer, uses rollout buffer)
- Actor-Critic architecture (policy + value function)
- Continuous action space: Box(2,) for [linear_vel, angular_vel]
- Clipped surrogate objective for stable policy updates
- Generalized Advantage Estimation (GAE) for variance reduction
"""

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
from torch.distributions import Normal
import numpy as np
from collections import deque
import time
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
from datetime import datetime


class ActorCriticNetwork(nn.Module):
    """
    Actor-Critic architecture with shared feature extractor.
    
    Architecture:
    - Shared layers: Extract features from state (LiDAR + goal info)
    - Actor head: Outputs mean and log_std for Gaussian policy
    - Critic head: Outputs state value V(s)
    """
    def __init__(self, state_size, action_size, hidden_dim=256):
        super(ActorCriticNetwork, self).__init__()
        
        # Shared feature extractor (processes LiDAR + goal information)
        self.shared_net = nn.Sequential(
            nn.Linear(state_size, hidden_dim),
            nn.Tanh(),  # Tanh is more stable than ReLU for PPO
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        
        # Actor head (policy): outputs mean for each action dimension
        self.actor_mean = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.Tanh(),
            nn.Linear(128, action_size),
            nn.Tanh()  # Bounded to [-1, 1] to prevent action divergence
        )
        
        # Actor log_std: learnable parameter (not state-dependent)
        # This is a common design choice for stability
        self.actor_log_std = nn.Parameter(torch.zeros(action_size))
        
        # Critic head (value function): outputs V(s)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )
    
    def forward(self, state):
        """
        Forward pass through network.
        
        Returns:
            action_mean: Mean of action distribution [linear_vel, angular_vel]
            action_std: Std of action distribution (from log_std)
            value: State value V(s)
        """
        features = self.shared_net(state)
        
        # Actor outputs
        action_mean = self.actor_mean(features)
        action_std = torch.exp(self.actor_log_std)  # Ensure positive std
        
        # Critic output
        value = self.critic(features)
        
        return action_mean, action_std, value
    
    def get_action_and_value(self, state, action=None):
        """
        Get action from policy and compute log_prob + value.
        Used during rollout collection and training.
        
        Args:
            state: Current state
            action: If provided, compute log_prob for this action (for training)
                   If None, sample a new action (for rollout)
        
        Returns:
            action: Sampled or provided action
            log_prob: Log probability of the action
            entropy: Entropy of the policy (for exploration bonus)
            value: State value V(s)
        """
        action_mean, action_std, value = self.forward(state)
        
        # Create Gaussian distribution for continuous actions
        dist = Normal(action_mean, action_std)
        
        if action is None:
            # Sample action during rollout
            action = dist.sample()
        
        # Compute log probability
        log_prob = dist.log_prob(action).sum(dim=-1)  # Sum over action dimensions
        
        # Compute entropy (measure of policy randomness)
        entropy = dist.entropy().sum(dim=-1)
        
        return action, log_prob, entropy, value.squeeze(-1)


class RolloutBuffer:
    """
    On-policy rollout buffer for PPO.
    Stores transitions collected during one epoch of interaction.
    
    Unlike DQN's replay buffer:
    - Fixed size (one epoch of data)
    - Used once then discarded
    - Stores advantages and returns (computed after collection)
    """
    def __init__(self, buffer_size, state_size, action_size):
        self.buffer_size = buffer_size
        self.state_size = state_size
        self.action_size = action_size
        
        # Storage
        self.states = np.zeros((buffer_size, state_size), dtype=np.float32)
        self.actions = np.zeros((buffer_size, action_size), dtype=np.float32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        
        # Computed after collection
        self.advantages = np.zeros(buffer_size, dtype=np.float32)
        self.returns = np.zeros(buffer_size, dtype=np.float32)
        
        self.ptr = 0  # Current position in buffer
        self.full = False
    
    def add(self, state, action, reward, done, value, log_prob):
        """Add one transition to buffer."""
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        
        self.ptr += 1
        if self.ptr >= self.buffer_size:
            self.full = True
    
    def compute_returns_and_advantages(self, last_value, gamma=0.99, gae_lambda=0.95):
        """
        Compute returns and advantages using Generalized Advantage Estimation (GAE).
        
        GAE balances bias-variance tradeoff:
        - lambda=0: High bias, low variance (1-step TD)
        - lambda=1: Low bias, high variance (Monte Carlo)
        - lambda=0.95: Good middle ground
        
        Args:
            last_value: Value of the final state (for bootstrapping)
            gamma: Discount factor (0.99 = value future rewards highly)
            gae_lambda: GAE lambda parameter (0.95 = balanced)
        """
        # We need to handle episode boundaries correctly
        advantages = np.zeros_like(self.rewards)
        last_gae_lambda = 0
        
        # Compute advantages backwards from end of buffer
        for t in reversed(range(self.ptr)):
            if t == self.ptr - 1:
                next_value = last_value
            else:
                next_value = self.values[t + 1]
            
            # TD error: δ_t = r_t + γ*V(s_{t+1}) - V(s_t)
            delta = self.rewards[t] + gamma * next_value * (1 - self.dones[t]) - self.values[t]
            
            # GAE: A_t = δ_t + (γλ)*δ_{t+1} + (γλ)^2*δ_{t+2} + ...
            advantages[t] = last_gae_lambda = delta + gamma * gae_lambda * (1 - self.dones[t]) * last_gae_lambda
        
        # Returns: R_t = A_t + V(s_t)
        self.returns = advantages + self.values[:self.ptr]
        self.advantages = advantages
    
    def get(self):
        """Get all data from buffer."""
        assert self.ptr > 0, "Buffer is empty"
        
        # Normalize advantages (improves stability)
        advantages = self.advantages[:self.ptr]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return {
            'states': self.states[:self.ptr],
            'actions': self.actions[:self.ptr],
            'log_probs': self.log_probs[:self.ptr],
            'returns': self.returns[:self.ptr],
            'advantages': advantages,
            'values': self.values[:self.ptr],
        }
    
    def reset(self):
        """Clear buffer for next epoch."""
        self.ptr = 0
        self.full = False


class PPOAgent:
    """
    Proximal Policy Optimization agent.
    
    Key hyperparameters:
    - clip_range: Clipping threshold for policy ratio (0.2 is standard)
    - n_epochs: Number of optimization epochs per rollout (10 is typical)
    - batch_size: Mini-batch size for SGD (64 is typical)
    - entropy_coef: Weight for entropy bonus (0.01 encourages exploration)
    - value_coef: Weight for value loss (0.5 is standard)
    - max_grad_norm: Gradient clipping threshold (0.5 prevents exploding gradients)
    """
    def __init__(self, state_size, action_size, 
                 lr=3e-4, gamma=0.99, gae_lambda=0.95,
                 clip_range=0.2, n_epochs=10, batch_size=64,
                 entropy_coef=0.03, value_coef=0.5, max_grad_norm=0.5):
        
        self.state_size = state_size
        self.action_size = action_size
        
        # Hyperparameters
        self.gamma = gamma  # Discount factor (0.99 = value long-term rewards)
        self.gae_lambda = gae_lambda  # GAE lambda (0.95 = balanced bias-variance)
        self.clip_range = clip_range  # PPO clip threshold (0.2 = standard)
        self.n_epochs = n_epochs  # Optimization epochs per rollout (10 = standard)
        self.batch_size = batch_size  # Mini-batch size (64 = standard)
        self.entropy_coef = entropy_coef  # Entropy bonus weight (0.03 = mild exploration)
        self.value_coef = value_coef  # Value loss weight (0.5 = standard)
        self.max_grad_norm = max_grad_norm  # Gradient clipping (0.5 = prevents explosion)
        
        # Network
        self.policy = ActorCriticNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # Action scaling (to match TurtleBot3 physical limits)
        # Linear velocity: [0.0, 0.22] m/s (ROBOTIS official max for Waffle Pi)
        # Angular velocity: [-2.0, 2.0] rad/s (ROBOTIS official max)
        self.action_low = np.array([0.0, -2.0], dtype=np.float32)
        self.action_high = np.array([0.22, 2.0], dtype=np.float32)
        
        print("="*60)
        print("PPO Agent Initialized")
        print("="*60)
        print(f"State size: {state_size}")
        print(f"Action size: {action_size}")
        print(f"Learning rate: {lr}")
        print(f"Clip range: {clip_range}")
        print(f"Epochs per update: {n_epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Action bounds: linear=[{self.action_low[0]:.2f}, {self.action_high[0]:.2f}], "
              f"angular=[{self.action_low[1]:.2f}, {self.action_high[1]:.2f}]")
        print("="*60)
    
    def load(self, path):
        """Load model weights."""
        checkpoint = torch.load(path, weights_only=False)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # Ensure episode is parsed as integer cleanly
        ep_val = int(checkpoint.get('episode', 0))
        print(f"Loaded PPO checkpoint from {path} (Episode {ep_val})")
        return ep_val, checkpoint.get('episode_rewards', []), checkpoint.get('episode_lengths', []), checkpoint.get('episode_losses', [])

    def scale_action(self, action):
        """
        Scale action from network output (unbounded) to TurtleBot3 limits.
        Uses tanh activation to bound output to [-1, 1], then scales to physical limits.
        """
        # Apply tanh to bound to [-1, 1]
        action_tanh = torch.tanh(action)
        
        # Scale to [action_low, action_high]
        action_scaled = self.action_low + (action_tanh.cpu().numpy() + 1.0) * 0.5 * (self.action_high - self.action_low)
        
        return action_scaled
    
    def select_action(self, state, deterministic=False):
        """
        Select action for current state.
        
        Args:
            state: Current observation
            deterministic: If True, use mean of policy (no sampling). Used for evaluation.
        
        Returns:
            action: Scaled action [linear_vel, angular_vel]
            log_prob: Log probability of action (for training)
            value: Value estimate V(s)
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action, log_prob, entropy, value = self.policy.get_action_and_value(state_tensor)
            
            if deterministic:
                # Use mean of policy (no randomness)
                action_mean, _, _ = self.policy.forward(state_tensor)
                action = action_mean
            
            # Scale action to TurtleBot3 limits
            action_scaled = self.scale_action(action)
        
        return action_scaled[0], log_prob.item(), value.item()
    
    def update(self, rollout_buffer):
        """
        Update policy using PPO objective.
        
        PPO Loss = L_CLIP + c1 * L_VF - c2 * S
        where:
        - L_CLIP: Clipped surrogate objective (policy loss)
        - L_VF: Value function loss (MSE)
        - S: Entropy bonus (encourages exploration)
        """
        # Get data from buffer
        data = rollout_buffer.get()
        
        states = torch.FloatTensor(data['states'])
        actions = torch.FloatTensor(data['actions'])
        old_log_probs = torch.FloatTensor(data['log_probs'])
        returns = torch.FloatTensor(data['returns'])
        advantages = torch.FloatTensor(data['advantages'])
        old_values = torch.FloatTensor(data['values'])
        
        # Training statistics
        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0
        n_updates = 0
        
        # Multiple epochs of optimization
        for epoch in range(self.n_epochs):
            # Generate random mini-batches
            indices = np.arange(len(states))
            np.random.shuffle(indices)
            
            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                # Get batch
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                
                # Evaluate actions with current policy
                _, new_log_probs, entropy, new_values = self.policy.get_action_and_value(
                    batch_states, batch_actions
                )
                
                # Policy loss (PPO clipped objective)
                # ratio = π_new(a|s) / π_old(a|s)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # Clipped surrogate objective
                policy_loss_1 = advantages[batch_indices] * ratio
                policy_loss_2 = advantages[batch_indices] * torch.clamp(
                    ratio, 1.0 - self.clip_range, 1.0 + self.clip_range
                )
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
                
                # Value loss (MSE between predicted and actual returns)
                value_loss = ((new_values - batch_returns) ** 2).mean()
                
                # Entropy loss (negative because we want to maximize entropy)
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                
                # Optimization step
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping (prevents exploding gradients)
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                
                self.optimizer.step()
                
                # Track statistics
                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy_loss.item()
                n_updates += 1
        
        # Return average losses
        return {
            'total_loss': total_loss / n_updates,
            'policy_loss': total_policy_loss / n_updates,
            'value_loss': total_value_loss / n_updates,
            'entropy_loss': total_entropy_loss / n_updates,
        }


def train_ppo():
    """Main training loop for PPO."""
    
    # Initialize ROS2
    if not rclpy.ok():
        rclpy.init()
    
    # Create environment
    print("Initializing TurtleBot3 environment...")
    env = TurtleBotEnv()
    
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]  # Now 2 (linear, angular)
    
    # Create agent
    agent = PPOAgent(
        state_size=state_size,
        action_size=action_size,
        lr=3e-4,  # Learning rate (3e-4 is PPO standard)
        gamma=0.99,  # Discount factor
        gae_lambda=0.95,  # GAE lambda
        clip_range=0.2,  # PPO clip threshold
        n_epochs=10,  # Optimization epochs per rollout
        batch_size=64,  # Mini-batch size
        entropy_coef=0.03,  # Entropy bonus (increased from 0.01 for better exploration)
        value_coef=0.5,  # Value loss weight
        max_grad_norm=0.5,  # Gradient clipping
    )
    
    # Training hyperparameters
    n_episodes = 2000  # Total episodes
    n_steps_per_episode = 500  # Max steps per episode
    n_steps_per_update = 2048  # Rollout length (standard for PPO)
    save_interval = 50  # Save model every 50 episodes
    
    # Rollout buffer
    rollout_buffer = RolloutBuffer(
        buffer_size=n_steps_per_update,
        state_size=state_size,
        action_size=action_size
    )
    
    # Tracking
    episode_rewards = []
    episode_lengths = []
    episode_losses = []
    all_successes = []
    success_history = deque(maxlen=10)  # Track success rate over last 10 episodes
    last_loss = 0.0
    start_episode = 0

    # Optional: Resume from checkpoint
    import sys
    if len(sys.argv) > 1 and sys.argv[1].endswith('.pth'):
        checkpoint_path = sys.argv[1]
        try:
            start_episode, ep_rews, ep_lens, ep_loss = agent.load(checkpoint_path)
            # Only restore histories if they look somewhat valid
            if len(ep_rews) > 0:
                episode_rewards = ep_rews
            if type(ep_lens) is list and len(ep_lens) > 0:
                episode_lengths = ep_lens
            if type(ep_loss) is list and len(ep_loss) > 0:
                episode_losses = ep_loss
            print("Resuming training with loaded statistics.")
        except Exception as e:
            print(f"Warning: Failed to load checkpoint {checkpoint_path}: {e}")
    
    # Create results directory
    results_dir = './training_results'
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    print("\n" + "="*60)
    print("Starting PPO Training")
    print("="*60)
    print(f"Total episodes: {n_episodes}")
    print(f"Steps per update: {n_steps_per_update}")
    print(f"Max steps per episode: {n_steps_per_episode}")
    print("="*60 + "\n", flush=True)  # Force flush to see output immediately
    
    # Training loop
    global_step = 0
    episode = start_episode
    
    while episode < n_episodes:
        # Reset environment
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        episode_success = False
        
        for step in range(n_steps_per_episode):
            # Select action
            action, log_prob, value = agent.select_action(state)
            
            # Execute action in environment
            next_state, reward, done, truncated, info = env.step(action)
            
            # Store transition in rollout buffer
            rollout_buffer.add(
                state=state,
                action=action,
                reward=reward,
                done=done,
                value=value,
                log_prob=log_prob
            )
            
            # Update tracking
            episode_reward += reward
            episode_length += 1
            global_step += 1
            state = next_state
            
            # Check if episode is done
            if done or truncated:
                if reward > 50:  # Goal reached (large positive reward)
                    episode_success = True
                break
            
            # Update policy when buffer is full
            if rollout_buffer.ptr >= n_steps_per_update:
                # Get final value for bootstrapping
                with torch.no_grad():
                    _, _, _, last_value = agent.policy.get_action_and_value(
                        torch.FloatTensor(next_state).unsqueeze(0)
                    )
                    last_value = last_value.item()
                
                # Compute returns and advantages
                rollout_buffer.compute_returns_and_advantages(
                    last_value=last_value,
                    gamma=agent.gamma,
                    gae_lambda=agent.gae_lambda
                )
                
                # Update policy
                losses = agent.update(rollout_buffer)
                last_loss = losses['total_loss']
                
                # Reset buffer
                rollout_buffer.reset()
                
                print(f"Update | Policy Loss: {losses['policy_loss']:.4f} | "
                      f"Value Loss: {losses['value_loss']:.4f} | "
                      f"Entropy: {losses['entropy_loss']:.4f}", flush=True)  # Force output
        
        # Episode finished
        episode += 1
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        success_history.append(1 if episode_success else 0)
        all_successes.append(1 if episode_success else 0)
        episode_losses.append(last_loss)
        
        # Calculate statistics
        avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
        success_rate = np.mean(success_history) * 100
        
        print(f"\nEpisode {episode}/{n_episodes} | "
              f"Reward: {episode_reward:.2f} | "
              f"Length: {episode_length} | "
              f"Success: {'✓' if episode_success else '✗'} | "
              f"Avg Reward (100): {avg_reward:.2f} | "
              f"Success Rate (10): {success_rate:.1f}%", flush=True)  # Force immediate output
        
        # Save model periodically
        if episode % save_interval == 0:
            model_path = os.path.join(results_dir, f'ppo_model_ep{episode}_{timestamp}.pth')
            torch.save({
                'episode': episode,
                'policy_state_dict': agent.policy.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'episode_rewards': episode_rewards,
            }, model_path)
            print(f"💾 Model saved: {model_path}")
            
            # Save training plot
            plot_training_progress(episode_rewards, episode_lengths, episode_losses, all_successes, results_dir, timestamp)
    
    # Save final model
    final_model_path = os.path.join(results_dir, f'ppo_model_final_{timestamp}.pth')
    torch.save({
        'episode': episode,
        'policy_state_dict': agent.policy.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'episode_rewards': episode_rewards,
        'episode_losses': episode_losses,
    }, final_model_path)
    print(f"\n✅ Training complete! Final model saved: {final_model_path}")
    
    # Save final training plot
    plot_training_progress(episode_rewards, episode_lengths, episode_losses, all_successes, results_dir, timestamp)
    
    # Cleanup
    env.close()
    rclpy.shutdown()


def plot_training_progress(rewards, lengths, losses, successes, save_dir, timestamp):
    """Plot and save training progress matching the 2x2 grid style."""
    fig, axs = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f'Training Results - {timestamp}', fontsize=16, fontweight='bold')
    
    episodes = range(1, len(rewards) + 1)
    
    # 1. Plot episode rewards
    axs[0, 0].plot(episodes, rewards, color='blue', alpha=0.5, label='Episode Reward')
    if len(rewards) >= 10:
        avg_rewards = np.convolve(rewards, np.ones(10)/10, mode='valid')
        axs[0, 0].plot(range(10, len(rewards) + 1), avg_rewards, color='red', linewidth=2, label='Moving Avg (10)')
    axs[0, 0].set_xlabel('Episode')
    axs[0, 0].set_ylabel('Total Reward')
    axs[0, 0].set_title('Episode Rewards')
    axs[0, 0].grid(True, alpha=0.3)
    axs[0, 0].legend()

    # 2. Plot Training Loss
    axs[0, 1].plot(episodes[:len(losses)], losses, color='green', alpha=0.5, label='Loss')
    if len(losses) >= 10:
        avg_losses = np.convolve(losses, np.ones(10)/10, mode='valid')
        axs[0, 1].plot(range(10, len(losses) + 1), avg_losses, color='orange', linewidth=2, label='Moving Avg (10)')
    axs[0, 1].set_xlabel('Episode')
    axs[0, 1].set_ylabel('Loss')
    axs[0, 1].set_title('Training Loss')
    axs[0, 1].grid(True, alpha=0.3)
    axs[0, 1].legend()
    
    # 3. Steps per Episode
    axs[1, 0].plot(episodes[:len(lengths)], lengths, color='orchid', alpha=0.8, label='Steps')
    if len(lengths) >= 10:
        avg_steps = np.convolve(lengths, np.ones(10)/10, mode='valid')
        axs[1, 0].plot(range(10, len(lengths) + 1), avg_steps, color='darkred', linewidth=2, label='Moving Avg (10)')
    axs[1, 0].set_xlabel('Episode')
    axs[1, 0].set_ylabel('Steps')
    axs[1, 0].set_title('Steps per Episode')
    axs[1, 0].grid(True, alpha=0.3)
    axs[1, 0].legend()
    
    # 4. Success Rate (Rolling 20 Episodes)
    axs[1, 1].axhline(y=50, color='lightcoral', linestyle='--', label='50% Target')
    if len(successes) > 0:
        rolling_success = []
        for i in range(len(successes)):
            start_idx = max(0, i - 19)
            success_rate = (sum(successes[start_idx:i+1]) / (i - start_idx + 1)) * 100
            rolling_success.append(success_rate)
        axs[1, 1].plot(episodes[:len(rolling_success)], rolling_success, color='c', linewidth=2, label='Success Rate (last 20)')
        
    axs[1, 1].set_xlabel('Episode')
    axs[1, 1].set_ylabel('Success Rate (%)')
    axs[1, 1].set_title('Success Rate (Rolling 20 Episodes)')
    axs[1, 1].set_ylim(0, 105)
    axs[1, 1].grid(True, alpha=0.3)
    axs[1, 1].legend()

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)  # Adjust for suptitle
    
    plot_path = os.path.join(save_dir, f'ppo_training_{timestamp}.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Training plot saved: {plot_path}")


if __name__ == '__main__':
    train_ppo()
