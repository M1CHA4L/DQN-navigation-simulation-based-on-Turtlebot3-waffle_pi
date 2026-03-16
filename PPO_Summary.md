1. Algorithm & Reward Tuning (The "Suicide Exploit" & Circling)
Problem: The robot was exhibiting "circling" behavior at spawn, and later, suffering from "flat rewards" where it would charge directly into walls after exactly 11 steps.
Diagnosis: The PPO agent found a "suicide exploit". Because it was receiving cumulative penalties for time and chaotic movements (smoothness penalty), the algorithm calculated that intentionally crashing into a wall to end the episode quickly resulted in a mathematically higher reward than exploring safely.
Solution:
Applied nn.Tanh() bounding to the Actor network to strictly constrain continuous actions to [-1, 1], preventing erratic gradient spikes.
Re-balanced the reward structure: Heavily penalized collisions (-100.0) and reduced the smoothness penalty weight (0.05) so survival became mathematically preferable.
Increased the PPO entropy coefficient (0.03) to force the robot to explore outside of its circling sub-optimal policy early in training.
2. Gazebo Longevity & Coordinates (The 200-Episode Crash)
Problem: Gazebo would permanently freeze or the robot would turn invisible after ~200-350 episodes. Even when it didn't crash, the robot would immediately trigger an "Out of Bounds" failure upon reset.
Diagnosis: Gazebo's C++ back-end suffers from a memory leak when using ROS 2 /remove and /create services repeatedly. Additionally, the robot's internal Odometry plugin does not reset its coordinates when the robot is physically moved.
Solution:
Teleportation over Deletion: Replaced the delete/spawn functions with move_goal_entity() and teleport_robot_to_center() utilizing the /set_pose service. This recycles the memory pointers rather than creating new ones, allowing infinite episodes without memory crashing.
Odometry Offsetting: Built a mathematical get_robot_pose() function using a 2D rotation matrix. It captures the robot's raw odometry exactly at the moment of teleportation, and mathematically translates the frame back to (0, 0, 0.0) so the neural network gets accurate initial coordinates.
3. Checkpoint Recovery (PyTorch 2.6 Security & Array Mismatches)
Problem: Unable to resume training from previous .pth checkpoints. PyTorch threw an UnpicklingError, and Matplotlib threw ValueError shape mismatches when plotting historical data.
Diagnosis: PyTorch 2.6 defaults to weights_only=True which blocks legacy checkpoint loading. Matplotlib crashed because the loaded history arrays (Rewards vs Episodes) varied in length if training was interrupted.
Solution:
Passed weights_only=False into the torch.load() function to bypass the security block securely.
Modified train_ppo.py, train_ppo_filtered.py, and run_ppo_training.sh to properly accept and parse terminal arguments (sys.argv[1]) so you can launch directly from a saved .pth file.
Applied dynamic array slicing in Matplotlib (episodes[:len(array)]) to ensure X and Y axes always perfectly aligned regardless of when the checkpoint was saved.
4. Infrastructure & Visualization
ROS 2 TypeErrors: Fixed a ROS 2 command velocity bug where the script passed a Twist message instead of a TwistStamped message when trying to send emergency stop commands. Enforced post-reset zero-velocity holding periods to kill residual angular momentum.
Data Visualization: Constructed a custom 2x2 Matplotlib grid (Rewards, Loss, Steps, Success Rate) that generates automatically using np.convolve for smooth 10-period moving average trendlines.
Network Diagrams: Created automated graphviz python scripts to generate clean, academic-style block diagrams of both the DQN and the PPO Actor-Critic neural network architectures for your final report.