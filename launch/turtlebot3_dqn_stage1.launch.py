#!/usr/bin/env python3
"""
ROS 2 Launch file for TurtleBot3 DQN Stage 1 simulation in Gazebo
Uses the turtlebot3_world launch as base and overrides the world file
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    """Generate launch description for TurtleBot3 DQN training in Gazebo."""
    
    # Get the turtlebot3_gazebo package directory
    turtlebot3_gazebo_dir = get_package_share_directory('turtlebot3_gazebo')
    launch_file_dir = os.path.join(turtlebot3_gazebo_dir, 'launch')
    
    # Get our custom world file
    world_file = os.path.join(
        os.path.dirname(__file__),
        '..',
        'turtlebot3_dqn_stage1_modified.world'
    )
    
    # Include the standard turtlebot3_world launch but override the world
    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(launch_file_dir, 'turtlebot3_world.launch.py')
            ),
            launch_arguments={
                'world': world_file,
                'x_pose': '0.0',
                'y_pose': '0.0'
            }.items()
        )
    ])
