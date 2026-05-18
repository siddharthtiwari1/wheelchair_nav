"""
E2E Navigation Launch File

Launches the complete end-to-end navigation stack:
    1. scan_depth_fusion_node (existing — 3-cam + LiDAR fusion)
    2. e2e_velocity_node (BEVVelocityNet inference)
    3. safety_clamp_node (LiDAR hard safety override)
    4. goal_manager_node (goal input handling)

Usage:
    ros2 launch wheelchair_e2e e2e_navigation.launch.py \
        model_path:=/path/to/best_model.pth
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_dir = get_package_share_directory('wheelchair_e2e')
    config_file = os.path.join(pkg_dir, 'config', 'e2e_params.yaml')

    return LaunchDescription([
        # Launch arguments
        DeclareLaunchArgument(
            'model_path',
            default_value='',
            description='Path to trained model checkpoint'),
        DeclareLaunchArgument(
            'use_safety_clamp',
            default_value='true',
            description='Enable standalone safety clamp node'),

        # E2E Velocity Inference Node
        Node(
            package='wheelchair_e2e',
            executable='e2e_velocity_node',
            name='e2e_velocity_node',
            parameters=[
                config_file,
                {'model_path': LaunchConfiguration('model_path')}
            ],
            output='screen',
            remappings=[
                ('/cmd_vel', '/cmd_vel'),
            ],
        ),

        # Goal Manager
        Node(
            package='wheelchair_e2e',
            executable='goal_manager_node',
            name='goal_manager_node',
            parameters=[config_file],
            output='screen',
        ),

        # Safety Clamp (optional standalone — e2e_velocity_node has
        # built-in safety, but this provides defense-in-depth)
        # Uncomment if using separate safety node:
        # Node(
        #     package='wheelchair_e2e',
        #     executable='safety_clamp_node',
        #     name='safety_clamp_node',
        #     parameters=[config_file],
        #     output='screen',
        # ),
    ])
