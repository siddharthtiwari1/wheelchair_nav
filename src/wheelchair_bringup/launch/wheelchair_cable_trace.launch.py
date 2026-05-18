#!/usr/bin/env python3

"""
WHEELCHAIR CABLE TRACING - MINIMAL DEPLOYMENT
==============================================
Runs ONLY what's needed for cable-following with CNN velocity inference.

Velocity chain:
    cable_trace_node → /cmd_vel_nav → velocity_smoother → /cmd_vel
        → twist_stamped_teleop → /wc_control/cmd_vel → DiffDriveController → Motors

Hardware used:
    - Arduino encoders (via ros2_control DiffDriveController)
    - Logitech C270 USB webcam (RGB for CNN inference)

Hardware NOT used:
    - RPLidar S3
    - RealSense cameras (no depth, no IMU)
    - AMCL / map_server / Nav2 stack
    - ZUPT / EKF odometry

Usage:
    ros2 launch wheelchair_bringup wheelchair_cable_trace.launch.py

    Without velocity smoother (direct control):
    ros2 launch wheelchair_bringup wheelchair_cable_trace.launch.py use_smoother:=false
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    TimerAction,
    LogInfo,
)
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Package directories
    wc_control_dir = get_package_share_directory('wc_control')

    # Workspace root
    ws_root = '/home/sidd/wheelchair_nav'
    cable_trace_dir = os.path.join(ws_root, 'cable_trace_deploy')

    # Default checkpoint
    default_checkpoint = os.path.join(cable_trace_dir, 'weights.zip')

    # ========================================================================
    # LAUNCH ARGUMENTS
    # ========================================================================

    declare_checkpoint = DeclareLaunchArgument(
        'checkpoint_path', default_value=default_checkpoint,
        description='Path to WireCNN checkpoint (.zip)'
    )
    declare_camera_index = DeclareLaunchArgument(
        'camera_index', default_value='10',
        description='USB camera index (10 = Logitech C270, -1 = auto-detect)'
    )
    declare_max_linear = DeclareLaunchArgument(
        'max_linear_vel', default_value='0.25',
        description='Max forward velocity (m/s)'
    )
    declare_max_angular = DeclareLaunchArgument(
        'max_angular_vel', default_value='0.60',
        description='Max angular velocity (rad/s)'
    )
    declare_use_smoother = DeclareLaunchArgument(
        'use_smoother', default_value='true',
        description='Route through velocity_smoother for acceleration limiting'
    )
    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time', default_value='false',
        description='Use simulation time'
    )
    declare_show_gui = DeclareLaunchArgument(
        'show_gui', default_value='true',
        description='Show camera feed with velocity overlay'
    )
    declare_record_video = DeclareLaunchArgument(
        'record_video', default_value='true',
        description='Record video + CSV logs'
    )

    use_smoother = LaunchConfiguration('use_smoother')
    use_sim_time = LaunchConfiguration('use_sim_time')

    # ========================================================================
    # STAGE 1: HARDWARE (0s) — ros2_control + DiffDriveController + encoders
    # ========================================================================
    # unified_wheelchair.launch.py provides:
    #   - robot_state_publisher (URDF → TF)
    #   - controller_manager (ros2_control)
    #   - DiffDriveController (subscribes /wc_control/cmd_vel TwistStamped)
    #   - joint_state_broadcaster

    # No-limit controller config for cable tracing
    cable_trace_config = os.path.join(
        get_package_share_directory('wc_control'),
        'config', 'wc_control_cable_trace.yaml'
    )

    unified_wheelchair_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(wc_control_dir, 'launch', 'unified_wheelchair.launch.py')
        ),
        launch_arguments={
            'is_sim': 'false',
            'port': '/dev/ttyACM0',
            'controller_config': cable_trace_config,
        }.items(),
    )

    # ========================================================================
    # STAGE 2: VELOCITY BRIDGE (0s) — Twist → TwistStamped for DiffDriveController
    # ========================================================================
    # DiffDriveController requires TwistStamped (use_stamped_vel: true).
    # This bridge converts /cmd_vel (Twist) → /wc_control/cmd_vel (TwistStamped).

    twist_bridge = Node(
        package='scripts',
        executable='twist_stamped_teleop',
        name='cmd_vel_bridge',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
        remappings=[
            ('cmd_vel_in', '/cmd_vel'),
            ('cmd_vel_out', '/wc_control/cmd_vel'),
        ],
    )

    # ========================================================================
    # STAGE 3: VELOCITY SMOOTHER (optional, 3s) — acceleration limiting
    # ========================================================================
    # When enabled:  cable_trace → /cmd_vel_nav → smoother → /cmd_vel → bridge
    # When disabled: cable_trace → /cmd_vel → bridge (direct, no accel limits)

    nav2_params_path = os.path.join(
        get_package_share_directory('wheelchair_navigation'),
        'config', 'nav2_params_3cam_v29.yaml'
    )

    velocity_smoother_node = TimerAction(
        period=3.0,
        actions=[
            LogInfo(msg='[CABLE-TRACE] Starting velocity smoother...'),
            Node(
                package='nav2_velocity_smoother',
                executable='velocity_smoother',
                name='velocity_smoother',
                output='screen',
                parameters=[nav2_params_path, {'use_sim_time': use_sim_time}],
                remappings=[
                    ('cmd_vel', 'cmd_vel_nav'),
                    ('cmd_vel_smoothed', 'cmd_vel'),
                ],
            ),
        ],
        condition=IfCondition(use_smoother),
    )

    # Lifecycle manager to transition velocity_smoother to active state.
    # Without this, the smoother stays in 'unconfigured' and blocks all velocity.
    smoother_lifecycle_manager = TimerAction(
        period=5.0,  # After smoother spawns at 3s
        actions=[
            LogInfo(msg='[CABLE-TRACE] Activating velocity smoother...'),
            Node(
                package='nav2_lifecycle_manager',
                executable='lifecycle_manager',
                name='lifecycle_manager_smoother',
                output='screen',
                parameters=[{
                    'use_sim_time': use_sim_time,
                    'autostart': True,
                    'node_names': ['velocity_smoother'],
                    'bond_timeout': 4.0,
                }],
            ),
        ],
        condition=IfCondition(use_smoother),
    )

    # ========================================================================
    # STAGE 4: CABLE TRACE CNN NODE (8s) — Logitech camera → inference → velocity
    # ========================================================================

    # With smoother: cable_trace → /cmd_vel_nav
    cable_trace_with_smoother = TimerAction(
        period=8.0,
        actions=[
            LogInfo(msg='[CABLE-TRACE] Starting CNN (Logitech → velocity through smoother)...'),
            Node(
                package='cable_trace_deploy',
                executable='cable_trace_node',
                name='cable_trace_node',
                output='screen',
                parameters=[{
                    'checkpoint_path': LaunchConfiguration('checkpoint_path'),
                    'camera_index': LaunchConfiguration('camera_index'),
                    'max_linear_vel': LaunchConfiguration('max_linear_vel'),
                    'max_angular_vel': LaunchConfiguration('max_angular_vel'),
                    'use_cuda': True,
                    'show_gui': LaunchConfiguration('show_gui'),
                    'record_video': LaunchConfiguration('record_video'),
                }],
                remappings=[
                    ('cmd_vel', 'cmd_vel_nav'),  # Route through smoother
                ],
            ),
        ],
        condition=IfCondition(use_smoother),
    )

    # Without smoother: cable_trace → /cmd_vel directly
    cable_trace_direct = TimerAction(
        period=5.0,  # No smoother to wait for
        actions=[
            LogInfo(msg='[CABLE-TRACE] Starting CNN (Logitech → velocity direct)...'),
            Node(
                package='cable_trace_deploy',
                executable='cable_trace_node',
                name='cable_trace_node',
                output='screen',
                parameters=[{
                    'checkpoint_path': LaunchConfiguration('checkpoint_path'),
                    'camera_index': LaunchConfiguration('camera_index'),
                    'max_linear_vel': LaunchConfiguration('max_linear_vel'),
                    'max_angular_vel': LaunchConfiguration('max_angular_vel'),
                    'use_cuda': True,
                    'show_gui': LaunchConfiguration('show_gui'),
                    'record_video': LaunchConfiguration('record_video'),
                }],
                # No remap — publishes directly to /cmd_vel
            ),
        ],
        condition=UnlessCondition(use_smoother),
    )

    # ========================================================================
    # READY MESSAGE
    # ========================================================================

    ready_message = TimerAction(
        period=10.0,
        actions=[
            LogInfo(msg='=' * 60),
            LogInfo(msg='  CABLE TRACE SYSTEM READY'),
            LogInfo(msg='=' * 60),
            LogInfo(msg='  Camera: Logitech C270 (/dev/video10)'),
            LogInfo(msg='  Model: WireCNN → [velx, velw]'),
            LogInfo(msg='  Press Q in camera window to stop'),
            LogInfo(msg='=' * 60),
        ],
    )

    # ========================================================================
    # LAUNCH
    # ========================================================================

    return LaunchDescription([
        # Arguments
        declare_checkpoint,
        declare_camera_index,
        declare_max_linear,
        declare_max_angular,
        declare_use_smoother,
        declare_use_sim_time,
        declare_show_gui,
        declare_record_video,

        # Stage 1: Hardware — Arduino + DiffDriveController (0s)
        unified_wheelchair_launch,

        # Stage 2: Velocity bridge (0s)
        twist_bridge,

        # Stage 3: Smoother + lifecycle (3s + 5s, optional)
        velocity_smoother_node,
        smoother_lifecycle_manager,

        # Stage 4: Cable trace CNN (8s with smoother, 5s without)
        cable_trace_with_smoother,
        cable_trace_direct,

        # Info
        ready_message,
    ])
