#!/usr/bin/env python3
"""Real-time training visualization with matplotlib."""

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import os
import time

class TrainingVisualizer:
    def __init__(self):
        self.episodes = []
        self.rewards = []
        self.losses = []
        self.epsilons = []
        self.steps = []
        
        # Create figure with subplots
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 8))
        self.fig.suptitle('TurtleBot3 DQN Training - Live Metrics', fontsize=14, fontweight='bold')
        
        # Configure subplots
        self.ax_reward = self.axes[0, 0]
        self.ax_loss = self.axes[0, 1]
        self.ax_epsilon = self.axes[1, 0]
        self.ax_steps = self.axes[1, 1]
        
        # Labels
        self.ax_reward.set_title('Episode Reward')
        self.ax_reward.set_xlabel('Episode')
        self.ax_reward.set_ylabel('Total Reward')
        self.ax_reward.grid(True, alpha=0.3)
        
        self.ax_loss.set_title('Training Loss')
        self.ax_loss.set_xlabel('Episode')
        self.ax_loss.set_ylabel('Average Loss')
        self.ax_loss.grid(True, alpha=0.3)
        
        self.ax_epsilon.set_title('Exploration Rate (ε)')
        self.ax_epsilon.set_xlabel('Episode')
        self.ax_epsilon.set_ylabel('Epsilon')
        self.ax_epsilon.grid(True, alpha=0.3)
        
        self.ax_steps.set_title('Steps per Episode')
        self.ax_steps.set_xlabel('Episode')
        self.ax_steps.set_ylabel('Steps')
        self.ax_steps.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Log file path
        self.log_file = '/tmp/training_log.txt'
    
    def parse_log_line(self, line):
        """Parse a training log line."""
        try:
            # Example: "Ep   1/200 | Steps: 125 | Reward:  -28.45 | Loss: 0.000 | ε: 0.995 | Mem: 125"
            parts = line.split('|')
            
            episode = int(parts[0].split()[1].split('/')[0])
            steps = int(parts[1].split(':')[1].strip())
            reward = float(parts[2].split(':')[1].strip())
            loss = float(parts[3].split(':')[1].strip())
            epsilon = float(parts[4].split(':')[1].strip())
            
            return episode, steps, reward, loss, epsilon
        except:
            return None
    
    def update_plot(self, frame):
        """Update plot with new data from log file."""
        if not os.path.exists(self.log_file):
            return
        
        try:
            with open(self.log_file, 'r') as f:
                lines = f.readlines()
            
            # Clear old data
            self.episodes = []
            self.rewards = []
            self.losses = []
            self.epsilons = []
            self.steps = []
            
            # Parse all lines
            for line in lines:
                if 'Ep' in line and 'Steps:' in line:
                    data = self.parse_log_line(line.strip())
                    if data:
                        ep, st, rew, loss, eps = data
                        self.episodes.append(ep)
                        self.steps.append(st)
                        self.rewards.append(rew)
                        self.losses.append(loss)
                        self.epsilons.append(eps)
            
            if len(self.episodes) > 0:
                # Clear axes
                self.ax_reward.clear()
                self.ax_loss.clear()
                self.ax_epsilon.clear()
                self.ax_steps.clear()
                
                # Plot reward
                self.ax_reward.plot(self.episodes, self.rewards, 'b-', linewidth=2)
                self.ax_reward.fill_between(self.episodes, self.rewards, alpha=0.3)
                self.ax_reward.set_title(f'Episode Reward (Latest: {self.rewards[-1]:.2f})')
                self.ax_reward.set_xlabel('Episode')
                self.ax_reward.set_ylabel('Total Reward')
                self.ax_reward.grid(True, alpha=0.3)
                
                # Plot loss
                self.ax_loss.plot(self.episodes, self.losses, 'r-', linewidth=2)
                self.ax_loss.set_title(f'Training Loss (Latest: {self.losses[-1]:.4f})')
                self.ax_loss.set_xlabel('Episode')
                self.ax_loss.set_ylabel('Average Loss')
                self.ax_loss.grid(True, alpha=0.3)
                
                # Plot epsilon
                self.ax_epsilon.plot(self.episodes, self.epsilons, 'g-', linewidth=2)
                self.ax_epsilon.set_title(f'Exploration Rate (ε={self.epsilons[-1]:.3f})')
                self.ax_epsilon.set_xlabel('Episode')
                self.ax_epsilon.set_ylabel('Epsilon')
                self.ax_epsilon.set_ylim([0, 1.05])
                self.ax_epsilon.grid(True, alpha=0.3)
                
                # Plot steps
                self.ax_steps.bar(self.episodes, self.steps, color='orange', alpha=0.7)
                self.ax_steps.set_title(f'Steps per Episode (Latest: {self.steps[-1]})')
                self.ax_steps.set_xlabel('Episode')
                self.ax_steps.set_ylabel('Steps')
                self.ax_steps.grid(True, alpha=0.3)
                
                plt.tight_layout()
        
        except Exception as e:
            print(f"Error updating plot: {e}")
    
    def run(self):
        """Start the visualization."""
        print("=" * 60)
        print("Training Visualizer Started")
        print("=" * 60)
        print(f"Monitoring log file: {self.log_file}")
        print("Waiting for training data...")
        print("")
        print("Episode outcomes will be shown below:")
        print("Close this window to stop visualization.")
        print("=" * 60)
        
        # Monitor log file and print RESULT lines in real-time
        import threading
        def monitor_results():
            last_line_count = 0
            while True:
                try:
                    if os.path.exists(self.log_file):
                        with open(self.log_file, 'r') as f:
                            lines = f.readlines()
                        if len(lines) > last_line_count:
                            for line in lines[last_line_count:]:
                                if 'RESULT:' in line:
                                    print(line.strip())
                            last_line_count = len(lines)
                except:
                    pass
                time.sleep(1)
        
        result_thread = threading.Thread(target=monitor_results, daemon=True)
        result_thread.start()
        
        # Animate
        anim = FuncAnimation(self.fig, self.update_plot, interval=2000)  # Update every 2 seconds
        plt.show()

if __name__ == '__main__':
    visualizer = TrainingVisualizer()
    visualizer.run()
