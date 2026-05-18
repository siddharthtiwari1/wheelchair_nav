#!/usr/bin/env python3

"""
WHEELCHAIR FUSION NAVIGATION - COMPLETE AUTONOMOUS SYSTEM
==========================================================
Usage:
    ros2 launch wheelchair_bringup wheelchair_fusion_nav.launch.py

    With custom map:
    ros2 launch wheelchair_bringup wheelchair_fusion_nav.launch.py map_name:=/path/to/map.yaml
"""

import os
from datetime import datetime

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    TimerAction,
    LogInfo,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Package directories
    wheelchair_bringup_dir = get_package_share_directory('wheelchair_bringup')
    wheelchair_navigation_dir = get_package_share_directory('wheelchair_navigation')
    wheelchair_description_dir = get_package_share_directory('wheelchair_description')

    # Workspace root - use hardcoded path for reliability
    ws_root = '/home/sidd/wheelchair_nav'

    # Default configurations - use package paths where possible
    default_map_file = os.path.join(ws_root, 'maps', 'h2.yaml')
    default_nav2_params = os.path.join(wheelchair_navigation_dir, 'config', 'nav2_params_3cam_v29.yaml')
    default_bt_xml = os.path.join(wheelchair_navigation_dir, 'behavior_tree', 'wheelchair_robust_nav_v3.xml')
    default_rviz_config = os.path.join(wheelchair_description_dir, 'rviz', 'fusion_nav_lite.rviz')

    # ========================================================================
    # LAUNCH ARGUMENTS
    # ========================================================================

    declare_map_name = DeclareLaunchArgument(
        'map_name', default_value=default_map_file,
        description='Full path to map YAML file'
    )
    declare_nav2_params = DeclareLaunchArgument(
        'nav2_params', default_value=default_nav2_params,
        description='Full path to Nav2 parameters file'
    )
    declare_bt_xml = DeclareLaunchArgument(
        'bt_xml', default_value=default_bt_xml,
        description='Full path to behavior tree XML file'
    )
    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time', default_value='false',
        description='Use simulation time'
    )
    declare_autostart = DeclareLaunchArgument(
        'autostart', default_value='true',
        description='Automatically start Nav2 lifecycle nodes'
    )
    declare_use_rviz = DeclareLaunchArgument(
        'use_rviz', default_value='true',
        description='Launch RViz visualization'
    )
    declare_rviz_config = DeclareLaunchArgument(
        'rviz_config', default_value=default_rviz_config,
        description='Full path to RViz config file'
    )
    declare_use_collision_monitor = DeclareLaunchArgument(
        'use_collision_monitor', default_value='false',
        description='Enable collision monitor safety layer'
    )
    declare_use_imu_diagnostic = DeclareLaunchArgument(
        'use_imu_diagnostic', default_value='true',
        description='Run IMU diagnostic at startup'
    )
    declare_record_bag = DeclareLaunchArgument(
        'record_bag', default_value='true',
        description='Record diagnostic rosbag (~42 KB/s, no scans). Disable with record_bag:=false'
    )

    # Launch configurations
    map_name = LaunchConfiguration('map_name')
    nav2_params = LaunchConfiguration('nav2_params')
    bt_xml = LaunchConfiguration('bt_xml')
    use_sim_time = LaunchConfiguration('use_sim_time')
    autostart = LaunchConfiguration('autostart')
    use_rviz = LaunchConfiguration('use_rviz')
    rviz_config = LaunchConfiguration('rviz_config')
    use_collision_monitor = LaunchConfiguration('use_collision_monitor')
    use_imu_diagnostic = LaunchConfiguration('use_imu_diagnostic')
    record_bag = LaunchConfiguration('record_bag')

    # ========================================================================
    # LOCALIZATION SYSTEM
    # ========================================================================

    localization_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(wheelchair_bringup_dir, 'launch', 'wheelchair_fusion_localization.launch.py')
        ),
        launch_arguments={
            'map_name': map_name,
            'use_sim_time': use_sim_time,
            'rviz': 'false',
        }.items()
    )

    # ========================================================================
    # NAV2 NAVIGATION STACK
    # ========================================================================

    nav2_lifecycle_nodes = [
        'controller_server',
        'planner_server',
        'behavior_server',
        'bt_navigator',
        'smoother_server',
        'velocity_smoother',
        'waypoint_follower',
    ]

    controller_server = Node(
        package='nav2_controller',
        executable='controller_server',
        name='controller_server',
        output='screen',
        parameters=[nav2_params, {'use_sim_time': use_sim_time}],
        remappings=[('cmd_vel', 'cmd_vel_nav')],
    )

    planner_server = Node(
        package='nav2_planner',
        executable='planner_server',
        name='planner_server',
        output='screen',
        parameters=[nav2_params, {'use_sim_time': use_sim_time}]
    )

    behavior_server = Node(
        package='nav2_behaviors',
        executable='behavior_server',
        name='behavior_server',
        output='screen',
        parameters=[nav2_params, {'use_sim_time': use_sim_time}],
        # SAFETY FIX: Route recovery commands through velocity_smoother.
        # Without this, BackUp/Spin publish directly to /cmd_vel, bypassing
        # the smoother's acceleration limits and velocity caps.
        remappings=[('cmd_vel', 'cmd_vel_nav')],
    )

    bt_navigator = Node(
        package='nav2_bt_navigator',
        executable='bt_navigator',
        name='bt_navigator',
        output='screen',
        parameters=[
            nav2_params,
            {'use_sim_time': use_sim_time},
            {'default_nav_to_pose_bt_xml': bt_xml},
            {'default_nav_through_poses_bt_xml': bt_xml},
        ]
    )

    smoother_server = Node(
        package='nav2_smoother',
        executable='smoother_server',
        name='smoother_server',
        output='screen',
        parameters=[nav2_params, {'use_sim_time': use_sim_time}]
    )

    velocity_smoother = Node(
        package='nav2_velocity_smoother',
        executable='velocity_smoother',
        name='velocity_smoother',
        output='screen',
        parameters=[nav2_params, {'use_sim_time': use_sim_time}],
        remappings=[
            ('cmd_vel', 'cmd_vel_nav'),
            ('cmd_vel_smoothed', 'cmd_vel'),  # RESTORED: direct to /cmd_vel for DiffDriveController
        ]
    )

    waypoint_follower = Node(
        package='nav2_waypoint_follower',
        executable='waypoint_follower',
        name='waypoint_follower',
        output='screen',
        parameters=[nav2_params, {'use_sim_time': use_sim_time}]
    )

    collision_monitor = Node(
        package='nav2_collision_monitor',
        executable='collision_monitor',
        name='collision_monitor',
        output='screen',
        parameters=[nav2_params, {'use_sim_time': use_sim_time}],
        condition=IfCondition(use_collision_monitor),  # RESTORED: v5 params wire cmd_vel→cmd_vel_safe, not inline
    )

    topic_data_logger = Node(
        package='scripts',
        executable='topic_data_logger',
        name='topic_data_logger',
        output='screen',
        parameters=[
            {'log_frequency_hz': 10.0},
            {'file_prefix': 'wheelchair_fusion_log'},
        ],
    )

    imu_diagnostic_node = Node(
        package='scripts',
        executable='imu_diagnostic',
        name='imu_diagnostic',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
        condition=IfCondition(use_imu_diagnostic),
    )

    nav2_lifecycle_manager = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_navigation',
        output='screen',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'autostart': autostart},
            {'node_names': nav2_lifecycle_nodes},
            {'bond_timeout': 4.0},
        ]
    )

    # ========================================================================
    # RVIZ
    # ========================================================================

    # RViz - delayed to wait for localization + 3-camera fusion
    rviz_node = TimerAction(
        period=30.0,  # After localization is fully ready (~25s)
        actions=[
            LogInfo(msg='[RVIZ] Starting RViz with Nav2 + 3-camera fusion display...'),
            Node(
                package='rviz2',
                executable='rviz2',
                name='rviz2',
                output='screen',
                arguments=['-d', rviz_config],
                parameters=[{'use_sim_time': use_sim_time}],
            )
        ],
        condition=IfCondition(use_rviz),
    )

    # NOTE: odom_velocity_corrector is already launched by wheelchair_fusion_localization.launch.py
    # Removing duplicate to avoid conflicts

    # ========================================================================
    # STAGED STARTUP - ADJUSTED FOR 3-CAMERA FUSION TIMING
    # ========================================================================
    # Localization launch (with 3 cameras) is ready at ~25s
    # Nav2 stack starts after localization is ready

    nav2_startup = TimerAction(
        period=28.0,  # After localization is ready (~25s)
        actions=[
            LogInfo(msg='[NAV2] Starting navigation stack...'),
            controller_server,
            planner_server,
            behavior_server,
            bt_navigator,
            smoother_server,
            velocity_smoother,
            waypoint_follower,
        ]
    )

    collision_monitor_startup = TimerAction(
        period=28.0,  # FIX: Start with Nav2 stack (was 32.0) — bridge depends on /cmd_vel_safe
        actions=[collision_monitor],
    )

    # CSV data logger DISABLED — use rosbag + analysis scripts instead
    # data_logger_startup = TimerAction(
    #     period=10.0,
    #     actions=[topic_data_logger],
    # )

    imu_diagnostic_startup = TimerAction(
        period=45.0,
        actions=[
            LogInfo(msg=''),
            LogInfo(msg='=' * 70),
            LogInfo(msg='  IMU DIAGNOSTIC - KEEP ROBOT STATIONARY FOR 30 SECONDS!'),
            LogInfo(msg='=' * 70),
            imu_diagnostic_node,
        ],
    )

    nav2_lifecycle_startup = TimerAction(
        period=35.0,  # After Nav2 nodes are spawned
        actions=[
            LogInfo(msg='[NAV2] Starting lifecycle manager...'),
            nav2_lifecycle_manager,
        ]
    )

    ready_message = TimerAction(
        period=40.0,
        actions=[
            LogInfo(msg='=' * 70),
            LogInfo(msg='  WHEELCHAIR FUSION NAVIGATION + 3-CAMERA - SYSTEM READY'),
            LogInfo(msg='=' * 70),
            LogInfo(msg='  Sensors: RPLidar S3 + 3x RealSense (D455/D455/D435i)'),
            LogInfo(msg='  AMCL: /scan_fused (LiDAR + 3 cameras)'),
            LogInfo(msg='  Send goals via RViz "2D Goal Pose" button'),
            LogInfo(msg='=' * 70),
        ]
    )

    # ========================================================================
    # AUTOMATIC ROSBAG RECORDING — starts at 30s, captures all nav data
    # ========================================================================
    # Records key diagnostic topics for post-session analysis.
    # Stops cleanly on Ctrl+C (SIGINT) — data is always preserved.
    # Disable with: record_bag:=false
    # Bags saved to: /opt/nvidia/rosbags/nav_<timestamp>/
    # Using /opt/nvidia partition (106GB free) to avoid filling root disk

    bag_output_dir = os.path.join(
        '/opt/nvidia', 'rosbags', 'nav_' + datetime.now().strftime('%Y%m%d_%H%M%S')
    )

    rosbag_recorder = TimerAction(
        period=30.0,
        actions=[
            LogInfo(msg='[CONFIG] Nav2 params: ' + default_nav2_params),
            LogInfo(msg='[CONFIG] BT XML: ' + default_bt_xml),
            LogInfo(msg='[CONFIG] Map: ' + default_map_file),
            LogInfo(msg='[ROSBAG] Recording nav diagnostics (~42 KB/s) to ' + bag_output_dir),
            ExecuteProcess(
                cmd=[
                    'ros2', 'bag', 'record',
                    '-o', bag_output_dir,
                    '--max-bag-duration', '300',
                    '--topics',
                    # Velocity chain (0.4 MB/5min)
                    '/cmd_vel', '/cmd_vel_nav',
                    # Localization (11 MB/5min — largest, but essential)
                    '/odometry/filtered', '/amcl_pose',
                    # Navigation goals + paths + action results
                    '/goal_pose', '/plan',
                    '/navigate_to_pose/_action/status',
                    '/navigate_to_pose/_action/feedback',
                    # ZUPT diagnostics (1 MB/5min)
                    '/zupt/diagnostics', '/zupt/slip_detected',
                    # Navigation action results (goal success/failure tracking)
                    '/navigate_to_pose/_action/result',
                    # Scan data for paper analysis (Tables III-V)
                    '/scan_filtered', '/scan_fused',
                    # NOTE: /tf REMOVED — 100+ Hz, GBs of disk
                    # NOTE: /local_plan REMOVED — 12 MB/5min at 20Hz
                ],
                output='log',  # 'log' not 'screen' — prevents terminal flooding
            ),
        ],
        condition=IfCondition(record_bag),
    )

    # ========================================================================
    # LAUNCH DESCRIPTION
    # ========================================================================

    return LaunchDescription([
        declare_map_name,
        declare_nav2_params,
        declare_bt_xml,
        declare_use_sim_time,
        declare_autostart,
        declare_use_rviz,
        declare_rviz_config,
        declare_use_collision_monitor,
        declare_use_imu_diagnostic,
        declare_record_bag,

        localization_launch,
        # NOTE: odom_velocity_corrector comes from localization_launch
        rviz_node,
        nav2_startup,
        collision_monitor_startup,
        nav2_lifecycle_startup,
        # data_logger_startup,  # DISABLED: use rosbag + analysis scripts
        imu_diagnostic_startup,
        rosbag_recorder,
        ready_message,
    ])
