#!/usr/bin/env python3
"""Launch the Depth Anything V2 vs RealSense evaluation node.

Layers on top of any running wheelchair stack (localization, SLAM, or nav).
Requires: front camera RGB + aligned depth + LiDAR already running.

Usage:
    ros2 launch wheelchair_mono_depth depth_eval.launch.py
    ros2 launch wheelchair_mono_depth depth_eval.launch.py eval_hz:=1.0
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_dir = get_package_share_directory('wheelchair_mono_depth')
    config_file = os.path.join(pkg_dir, 'config', 'depth_eval.yaml')

    declare_eval_hz = DeclareLaunchArgument(
        'eval_hz', default_value='2.0',
        description='Evaluation rate in Hz',
    )
    declare_output_dir = DeclareLaunchArgument(
        'output_dir',
        default_value='/home/sidd/wheelchair_nav/eval_output/depth_eval',
        description='Directory for CSV metrics and visualization PNGs',
    )

    eval_node = Node(
        package='wheelchair_mono_depth',
        executable='depth_eval_node',
        name='depth_eval_node',
        output='screen',
        parameters=[
            config_file,
            {
                'eval_hz': LaunchConfiguration('eval_hz'),
                'output_dir': LaunchConfiguration('output_dir'),
            },
        ],
    )

    return LaunchDescription([
        declare_eval_hz,
        declare_output_dir,
        LogInfo(msg='Launching DA V2 depth evaluation node...'),
        eval_node,
    ])
