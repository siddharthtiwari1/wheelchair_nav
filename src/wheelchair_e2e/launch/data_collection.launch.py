"""
Data Collection Launch File — 4-Topic Recording

Records what KinoFlow training needs:
    /scan_fused            → BEV grid input (occupancy channels 0, 1)
    /odometry/filtered     → velocity labels (twist), odom history, ego trail,
                             goal computation (pose at 5s stops)
    /goal_pose             → explicit goal positions (Nav2 or manual publish)
    /plan                  → Nav2 global plan for BEV channel 4 (route)

Velocity labels come from actual executed velocity in odom twist, NOT /cmd_vel.
Goals: auto-detected from 5s stops, OR from /goal_pose if available.

~3 MB/min storage.

Prerequisites:
    1. Sensor stack running (LiDAR + cameras + scan_depth_fusion + EKF)
    2. Teleop controller (joystick or keyboard)
    3. Optional: Nav2 running for /goal_pose and /plan topics

Usage:
    # Manual teleop (goals auto-detected from stops):
    ros2 launch wheelchair_e2e data_collection.launch.py

    # With Nav2 (explicit goals + route):
    ros2 launch wheelchair_e2e data_collection.launch.py

    # Custom output name:
    ros2 launch wheelchair_e2e data_collection.launch.py output_bag:=corridor_run_01
"""

from datetime import datetime
from launch import LaunchDescription
from launch.actions import ExecuteProcess, DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    default_output = f'e2e_training_{timestamp}'

    return LaunchDescription([
        DeclareLaunchArgument(
            'output_bag',
            default_value=default_output,
            description='Output bag name'),

        ExecuteProcess(
            cmd=[
                'ros2', 'bag', 'record',
                # BEV occupancy input (channels 0, 1)
                '/scan_fused',
                # Velocity labels (twist), odom history, ego trail, goal detect
                '/odometry/filtered',
                # Explicit goal positions (from Nav2 or manual publish)
                '/goal_pose',
                # Nav2 global plan for BEV channel 4 (route rendering)
                '/plan',
                # Output
                '-o', LaunchConfiguration('output_bag'),
            ],
            output='screen',
        ),
    ])
