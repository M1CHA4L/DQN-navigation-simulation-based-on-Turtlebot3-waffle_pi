import torch
from torchviz import make_dot
from torchinfo import summary

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train_pytorch import DQNNetwork

def main():
    state_size = 22  # 20 laser points + distance + angle
    action_size = 5  # 5 discrete actions

    print("Initializing DQN Network...")
    model = DQNNetwork(state_size, action_size)
    
    print("\n--- DQN Model Architecture Summary ---")
    summary(model, input_size=(1, state_size))

    device = next(model.parameters()).device
    x = torch.randn(1, state_size).to(device)
    
    # Process the state to get Q-values for the 5 actions
    q_values = model(x)

    print("Generating DQN_Architecture.png...")
    dot_dqn = make_dot(q_values, params=dict(list(model.named_parameters())), show_attrs=True, show_saved=True)
    dot_dqn.format = 'png'
    dot_dqn.render('DQN_Architecture')
    
    print("Done! Check 'DQN_Architecture.png'.")

if __name__ == "__main__":
    main()