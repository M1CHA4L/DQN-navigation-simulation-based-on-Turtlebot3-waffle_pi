#!/usr/bin/env python3
"""
Visualize and compare DQN vs PPO trajectories.
Shows the difference between discrete and continuous control.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.animation import FuncAnimation


def plot_trajectory_comparison():
    """
    Generate synthetic trajectories to illustrate DQN vs PPO behavior.
    
    DQN: Discrete actions → zigzag pattern
    PPO: Continuous actions → smooth curves
    """
    
    # Simulate DQN trajectory (discrete turns)
    dqn_x = [0]
    dqn_y = [0]
    dqn_theta = 0
    dt = 0.1
    
    # DQN actions: [sharp_right, gentle_right, straight, gentle_left, sharp_left]
    dqn_actions = [-1.5, -0.75, 0.0, 0.75, 1.5]
    dqn_sequence = [2, 2, 2, 3, 3, 2, 2, 1, 1, 2, 2, 4, 4, 2, 2, 3, 2, 2]  # Hand-crafted
    
    for action_idx in dqn_sequence:
        angular_vel = dqn_actions[action_idx]
        linear_vel = 0.15
        
        for _ in range(5):  # 5 steps per action
            dqn_theta += angular_vel * dt
            dqn_x.append(dqn_x[-1] + linear_vel * np.cos(dqn_theta) * dt)
            dqn_y.append(dqn_y[-1] + linear_vel * np.sin(dqn_theta) * dt)
    
    # Simulate PPO trajectory (smooth control)
    ppo_x = [0]
    ppo_y = [0]
    ppo_theta = 0
    
    # PPO learns to smoothly adjust velocities
    goal = np.array([1.5, 1.2])
    
    for _ in range(len(dqn_x)):
        # Simplified PPO policy: proportional control
        dx = goal[0] - ppo_x[-1]
        dy = goal[1] - ppo_y[-1]
        
        angle_to_goal = np.arctan2(dy, dx)
        angle_error = angle_to_goal - ppo_theta
        angle_error = np.arctan2(np.sin(angle_error), np.cos(angle_error))  # Normalize
        
        # Smooth control
        angular_vel = np.clip(2.0 * angle_error, -1.5, 1.5)  # Proportional control
        distance = np.sqrt(dx**2 + dy**2)
        linear_vel = np.clip(0.15 * min(1.0, distance / 0.5), 0.05, 0.22)  # Slow near goal
        
        ppo_theta += angular_vel * dt
        ppo_x.append(ppo_x[-1] + linear_vel * np.cos(ppo_theta) * dt)
        ppo_y.append(ppo_y[-1] + linear_vel * np.sin(ppo_theta) * dt)
    
    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # DQN trajectory
    ax1.plot(dqn_x, dqn_y, 'b-', linewidth=2, label='DQN Path')
    ax1.plot(dqn_x[0], dqn_y[0], 'go', markersize=12, label='Start')
    ax1.plot(goal[0], goal[1], 'r*', markersize=20, label='Goal')
    ax1.arrow(dqn_x[-1], dqn_y[-1], 0.1*np.cos(dqn_theta), 0.1*np.sin(dqn_theta),
              head_width=0.05, head_length=0.05, fc='blue', ec='blue')
    ax1.set_xlabel('X Position (m)', fontsize=12)
    ax1.set_ylabel('Y Position (m)', fontsize=12)
    ax1.set_title('DQN Trajectory\n(Discrete Actions → Zigzag)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    ax1.set_xlim(-0.2, 1.8)
    ax1.set_ylim(-0.2, 1.5)
    
    # PPO trajectory
    ax2.plot(ppo_x, ppo_y, 'r-', linewidth=2, label='PPO Path')
    ax2.plot(ppo_x[0], ppo_y[0], 'go', markersize=12, label='Start')
    ax2.plot(goal[0], goal[1], 'r*', markersize=20, label='Goal')
    ax2.arrow(ppo_x[-1], ppo_y[-1], 0.1*np.cos(ppo_theta), 0.1*np.sin(ppo_theta),
              head_width=0.05, head_length=0.05, fc='red', ec='red')
    ax2.set_xlabel('X Position (m)', fontsize=12)
    ax2.set_ylabel('Y Position (m)', fontsize=12)
    ax2.set_title('PPO Trajectory\n(Continuous Actions → Smooth)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    ax2.set_xlim(-0.2, 1.8)
    ax2.set_ylim(-0.2, 1.5)
    
    plt.tight_layout()
    plt.savefig('dqn_vs_ppo_trajectories.png', dpi=150, bbox_inches='tight')
    print("📊 Trajectory comparison saved: dqn_vs_ppo_trajectories.png")
    plt.close()


def plot_action_distributions():
    """
    Plot action distributions for DQN vs PPO.
    Shows discrete vs continuous nature.
    """
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # DQN: Discrete angular velocities
    dqn_angular = [-1.5, -0.75, 0.0, 0.75, 1.5]
    dqn_counts = [15, 25, 30, 25, 15]  # Example distribution
    dqn_linear = [0.15] * 5  # Always constant
    
    ax1.bar(dqn_angular, dqn_counts, width=0.3, color='blue', alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Angular Velocity (rad/s)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('DQN Action Distribution\n(5 Discrete Actions)', fontsize=14, fontweight='bold')
    ax1.grid(True, axis='y', alpha=0.3)
    ax1.set_xticks(dqn_angular)
    ax1.axhline(y=0, color='black', linewidth=0.5)
    ax1.text(0, max(dqn_counts)*1.1, 'Linear Vel: 0.15 m/s (constant)', 
             ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # PPO: Continuous angular velocities (Gaussian distribution)
    ppo_angular = np.linspace(-2.0, 2.0, 100)
    ppo_density = 100 * np.exp(-((ppo_angular - 0.2) ** 2) / (2 * 0.6 ** 2))  # Example Gaussian
    
    ax2.fill_between(ppo_angular, ppo_density, alpha=0.5, color='red', label='Angular Vel')
    ax2.plot(ppo_angular, ppo_density, 'r-', linewidth=2)
    ax2.set_xlabel('Angular Velocity (rad/s)', fontsize=12)
    ax2.set_ylabel('Probability Density', fontsize=12)
    ax2.set_title('PPO Action Distribution\n(Continuous Gaussian Policy)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.axvline(x=0.2, color='red', linestyle='--', linewidth=2, label='Mean')
    ax2.text(0, max(ppo_density)*1.1, 'Linear Vel: 0.0-0.22 m/s (learned)', 
             ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('dqn_vs_ppo_actions.png', dpi=150, bbox_inches='tight')
    print("📊 Action distribution comparison saved: dqn_vs_ppo_actions.png")
    plt.close()


def plot_velocity_profiles():
    """
    Plot velocity profiles over time for DQN vs PPO.
    Shows adaptive speed control in PPO.
    """
    
    time = np.linspace(0, 10, 100)
    
    # DQN: Constant linear velocity, discrete angular changes
    dqn_linear = np.ones_like(time) * 0.15
    dqn_angular = np.zeros_like(time)
    # Add discrete jumps
    dqn_angular[20:30] = 0.75
    dqn_angular[40:50] = -1.5
    dqn_angular[60:70] = 1.5
    
    # PPO: Adaptive velocities
    ppo_linear = 0.15 * (1 - np.exp(-time / 3))  # Start slow, speed up
    ppo_angular = 1.0 * np.sin(time / 2) * np.exp(-time / 5)  # Smooth adjustments
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Linear velocity
    ax1.plot(time, dqn_linear, 'b-', linewidth=2, label='DQN', drawstyle='steps-post')
    ax1.plot(time, ppo_linear, 'r-', linewidth=2, label='PPO')
    ax1.set_ylabel('Linear Velocity (m/s)', fontsize=12)
    ax1.set_title('Linear Velocity Over Time', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 0.25)
    ax1.axhline(y=0.22, color='gray', linestyle='--', alpha=0.5, label='Max')
    
    # Angular velocity
    ax2.plot(time, dqn_angular, 'b-', linewidth=2, label='DQN', drawstyle='steps-post')
    ax2.plot(time, ppo_angular, 'r-', linewidth=2, label='PPO')
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Angular Velocity (rad/s)', fontsize=12)
    ax2.set_title('Angular Velocity Over Time', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-2.5, 2.5)
    ax2.axhline(y=0, color='black', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig('dqn_vs_ppo_velocities.png', dpi=150, bbox_inches='tight')
    print("📊 Velocity profile comparison saved: dqn_vs_ppo_velocities.png")
    plt.close()


def create_all_visualizations():
    """Generate all comparison visualizations."""
    print("\n" + "="*60)
    print("Generating DQN vs PPO Comparison Visualizations")
    print("="*60 + "\n")
    
    plot_trajectory_comparison()
    plot_action_distributions()
    plot_velocity_profiles()
    
    print("\n" + "="*60)
    print("✅ All visualizations generated!")
    print("="*60)
    print("\nGenerated files:")
    print("  1. dqn_vs_ppo_trajectories.png - Path comparison")
    print("  2. dqn_vs_ppo_actions.png - Action space comparison")
    print("  3. dqn_vs_ppo_velocities.png - Velocity profiles")
    print("\nThese show why PPO produces smoother, more natural motion.")
    print("="*60 + "\n")


if __name__ == '__main__':
    create_all_visualizations()
