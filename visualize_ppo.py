import torch
from torchviz import make_dot
from torchinfo import summary

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train_ppo import ActorCriticNetwork

def main():
    state_size = 22  # 20 laser points + distance + angle
    action_size = 2  # Linear velocity, Angular velocity

    print("Initializing PPO ActorCritic Network...")
    model = ActorCriticNetwork(state_size, action_size)
    
    print("\n--- PPO Model Architecture Summary ---")
    summary(model, input_size=(1, state_size))

    device = next(model.parameters()).device
    x = torch.randn(1, state_size).to(device)
    
    # Process the state to get action mean and state value
    action_mean, action_std, value = model(x)

    # 1. Plot the Actor Network Architecture
    print("Generating PPO_Actor_Architecture.png...")
    dot_actor = make_dot(action_mean, params=dict(list(model.named_parameters())), show_attrs=True, show_saved=True)
    dot_actor.format = 'png'
    dot_actor.render('PPO_Actor_Architecture')
    
    # 2. Plot the Critic Network Architecture
    print("Generating PPO_Critic_Architecture.png...")
    dot_critic = make_dot(value, params=dict(list(model.named_parameters())), show_attrs=True, show_saved=True)
    dot_critic.format = 'png'
    dot_critic.render('PPO_Critic_Architecture')
    
    print("Done! Check 'PPO_Actor_Architecture.png' and 'PPO_Critic_Architecture.png'.")

if __name__ == "__main__":
    main()