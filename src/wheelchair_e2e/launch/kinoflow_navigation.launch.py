"""
KinoFlow Navigation Launch File

Launches the KinoFlow learned controller as a replacement for Nav2's
RegulatedPurePursuit controller. Works alongside Nav2's planner, AMCL,
and behavior tree.

Usage:
    ros2 launch wheelchair_e2e kinoflow_navigation.launch.py \
        model_path:=/path/to/best_kinoflow.pth

What runs:
    1. kinoflow_node - Learned kinodynamic trajectory controller
       Subscribes: /scan_fused, /odometry/filtered, /plan, /goal_pose
       Publishes: /cmd_vel, /kinoflow/trajectories, /kinoflow/best_trajectory

What should already be running (via wheelchair_fusion_nav.launch.py):
    - Nav2 planner_server (SmacPlanner2D) -> /plan
    - AMCL -> localization on metric map
    - map_server -> /map
    - scan_depth_fusion_node -> /scan_fused
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'model_path',
            default_value='',
            description='Path to trained KinoFlow checkpoint'),
        DeclareLaunchArgument(
            'v_max', default_value='0.25',
            description='Maximum linear velocity (m/s)'),
        DeclareLaunchArgument(
            'w_max', default_value='1.0',
            description='Maximum angular velocity (rad/s)'),
        DeclareLaunchArgument(
            'inference_hz', default_value='15.0',
            description='Inference rate (Hz)'),
        DeclareLaunchArgument(
            'n_samples', default_value='8',
            description='Number of trajectory candidates (K)'),
        DeclareLaunchArgument(
            'visualize', default_value='true',
            description='Publish trajectory visualization'),

        Node(
            package='wheelchair_e2e',
            executable='kinoflow_node',
            name='kinoflow_node',
            parameters=[{
                'model_path': LaunchConfiguration('model_path'),
                'v_max': LaunchConfiguration('v_max'),
                'w_max': LaunchConfiguration('w_max'),
                'inference_hz': LaunchConfiguration('inference_hz'),
                'n_samples': LaunchConfiguration('n_samples'),
                'horizon': 10,
                'n_euler_steps': 3,
                'dt': 0.1,
                'safety_min_range': 0.4,
                'safety_slow_range': 0.8,
                'max_accel': 0.5,
                'max_alpha': 2.0,
                'ema_alpha': 0.3,
                'goal_tolerance': 0.15,
                'visualize_trajectories': LaunchConfiguration('visualize'),
            }],
            output='screen',
        ),
    ])
