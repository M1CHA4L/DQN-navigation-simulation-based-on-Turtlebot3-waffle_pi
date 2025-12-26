#!/usr/bin/env python3
"""Launch TurtleBot3 DQN with Gazebo GUI for visualization."""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    
    launch_file_dir = os.path.join(get_package_share_directory('turtlebot3_gazebo'), 'launch')
    
    # Your custom world
    world = os.path.join(
        os.path.dirname(__file__),
        '..',
        'turtlebot3_dqn_stage1_modified.world'
    )
    
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    
    # Use the standard turtlebot3_world launch (includes GUI)
    world_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(launch_file_dir, 'turtlebot3_world.launch.py')
        ),
        launch_arguments={
            'world': world,
            'x_pose': '0.0',
            'y_pose': '0.0'
        }.items()
    )
    
    return LaunchDescription([
        world_cmd
    ])
