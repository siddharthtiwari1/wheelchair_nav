#!/usr/bin/env python3

"""
WHEELCHAIR KINOFLOW NAVIGATION - E2E LEARNED CONTROLLER
=========================================================
Replaces Nav2's RegulatedPurePursuit (RPP) controller with KinoFlow v2.

Architecture:
    Nav2 planner_server → /plan → kinoflow_node → /cmd_vel_nav → velocity_smoother → /cmd_vel
    Nav2 behavior_server handles recovery actions (backup, spin, clear costmaps)

vs wheelchair_fusion_nav.launch.py:
    - REMOVED: controller_server (RPP) — kinoflow_node replaces it
    - ADDED: kinoflow_node (15Hz Python + PyTorch)
    - CHANGED: BT uses wheelchair_kinoflow_nav.xml (no FollowPath action)
    - CHANGED: nav2_params_kinoflow.yaml (no controller_server section)
    - CHANGED: lifecycle manager does NOT manage controller_server

Usage:
    ros2 launch wheelchair_bringup wheelchair_kinoflow_nav.launch.py

    With trained model:
    ros2 launch wheelchair_bringup wheelchair_kinoflow_nav.launch.py \
        model_path:=/path/to/best_model.pth

    Dummy mode (no model — zero velocity, pipeline test):
    ros2 launch wheelchair_bringup wheelchair_kinoflow_nav.launch.py

Created: 2026-02-28
Pairs with: nav2_params_kinoflow.yaml, wheelchair_kinoflow_nav.xml, modular_e2e_params.yaml
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

    # Workspace root
    ws_root = '/home/sidd/wheelchair_nav'

    # Default configurations
    default_map_file = os.path.join(ws_root, 'maps', 'fused_map.yaml')
    default_nav2_params = os.path.join(
        wheelchair_navigation_dir, 'config', 'nav2_params_kinoflow.yaml')
    default_bt_xml = os.path.join(
        wheelchair_navigation_dir, 'behavior_tree', 'wheelchair_kinoflow_nav.xml')
    default_rviz_config = os.path.join(
        wheelchair_description_dir, 'rviz', 'fusion_navigation.rviz')
    default_kinoflow_params = os.path.join(
        ws_root, 'src', 'wheelchair_e2e', 'config', 'modular_e2e_params.yaml')

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
    declare_model_path = DeclareLaunchArgument(
        'model_path', default_value='',
        description='Path to KinoFlow model checkpoint (empty = dummy mode)'
    )
    declare_record_bag = DeclareLaunchArgument(
        'record_bag', default_value='true',
        description='Automatically record rosbag for analysis'
    )

    # Launch configurations
    map_name = LaunchConfiguration('map_name')
    nav2_params = LaunchConfiguration('nav2_params')
    bt_xml = LaunchConfiguration('bt_xml')
    use_sim_time = LaunchConfiguration('use_sim_time')
    autostart = LaunchConfiguration('autostart')
    use_rviz = LaunchConfiguration('use_rviz')
    rviz_config = LaunchConfiguration('rviz_config')
    model_path = LaunchConfiguration('model_path')
    record_bag = LaunchConfiguration('record_bag')

    # ========================================================================
    # LOCALIZATION SYSTEM (unchanged from wheelchair_fusion_nav.launch.py)
    # ========================================================================

    localization_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(wheelchair_bringup_dir, 'launch',
                         'wheelchair_fusion_localization.launch.py')
        ),
        launch_arguments={
            'map_name': map_name,
            'use_sim_time': use_sim_time,
            'rviz': 'false',
        }.items()
    )

    # ========================================================================
    # KINOFLOW NODE — replaces controller_server (RPP)
    # ========================================================================
    # Subscribes to: /scan_fused, /odometry/filtered, /goal_pose, /plan
    # Publishes to: /cmd_vel (remapped to /cmd_vel_nav for velocity_smoother)

    kinoflow_node = Node(
        package='wheelchair_e2e',
        executable='kinoflow_node',
        name='kinoflow_node',
        output='screen',
        parameters=[
            default_kinoflow_params,
            {
                'use_sim_time': use_sim_time,
                'model_path': model_path,
            },
        ],
        remappings=[('cmd_vel', 'cmd_vel_nav')],
    )

    # ========================================================================
    # NAV2 NAVIGATION STACK (no controller_server)
    # ========================================================================

    # Lifecycle nodes — NO controller_server (kinoflow_node is not lifecycle-managed)
    nav2_lifecycle_nodes = [
        'planner_server',
        'behavior_server',
        'bt_navigator',
        'smoother_server',
        'velocity_smoother',
        'waypoint_follower',
    ]

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
            ('cmd_vel_smoothed', 'cmd_vel'),
        ]
    )

    waypoint_follower = Node(
        package='nav2_waypoint_follower',
        executable='waypoint_follower',
        name='waypoint_follower',
        output='screen',
        parameters=[nav2_params, {'use_sim_time': use_sim_time}]
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

    rviz_node = TimerAction(
        period=30.0,
        actions=[
            LogInfo(msg='[RVIZ] Starting RViz with KinoFlow + Nav2 display...'),
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

    # ========================================================================
    # STAGED STARTUP
    # ========================================================================
    # 0s:   Localization (sensors, SLAM/AMCL, fusion)
    # 25s:  KinoFlow node (needs scan_fused + odom)
    # 28s:  Nav2 stack (planner, behavior, bt_navigator)
    # 35s:  Lifecycle manager
    # 40s:  Ready

    kinoflow_startup = TimerAction(
        period=25.0,
        actions=[
            LogInfo(msg='[KINOFLOW] Starting KinoFlow v2 controller node...'),
            kinoflow_node,
        ]
    )

    nav2_startup = TimerAction(
        period=28.0,
        actions=[
            LogInfo(msg='[NAV2] Starting navigation stack (planner + behavior + BT)...'),
            planner_server,
            behavior_server,
            bt_navigator,
            smoother_server,
            velocity_smoother,
            waypoint_follower,
        ]
    )

    nav2_lifecycle_startup = TimerAction(
        period=35.0,
        actions=[
            LogInfo(msg='[NAV2] Starting lifecycle manager...'),
            nav2_lifecycle_manager,
        ]
    )

    ready_message = TimerAction(
        period=40.0,
        actions=[
            LogInfo(msg='=' * 70),
            LogInfo(msg='  WHEELCHAIR KINOFLOW NAVIGATION - SYSTEM READY'),
            LogInfo(msg='=' * 70),
            LogInfo(msg='  Controller: KinoFlow v2 (replaces RPP)'),
            LogInfo(msg='  Sensors: RPLidar S3 + 3x RealSense (D455/D455/D435i)'),
            LogInfo(msg='  AMCL: /scan_fused (LiDAR + 3 cameras)'),
            LogInfo(msg='  Send goals via RViz "2D Goal Pose" button'),
            LogInfo(msg='=' * 70),
        ]
    )

    # ========================================================================
    # ROSBAG RECORDING
    # ========================================================================

    bag_output_dir = os.path.join(
        ws_root, 'rosbags',
        'kinoflow_' + datetime.now().strftime('%Y%m%d_%H%M%S')
    )

    rosbag_recorder = TimerAction(
        period=30.0,
        actions=[
            LogInfo(msg='[ROSBAG] Recording KinoFlow nav data to ' + bag_output_dir),
            ExecuteProcess(
                cmd=[
                    'ros2', 'bag', 'record',
                    '-o', bag_output_dir,
                    '--topics',
                    '/cmd_vel', '/cmd_vel_nav',
                    '/odometry/filtered', '/amcl_pose',
                    '/goal_pose', '/plan',
                    '/scan_fused',
                    '/kinoflow/best_trajectory',
                    '/kinoflow/trajectories',
                    '/zupt/diagnostics', '/zupt/slip_detected',
                ],
                output='screen',
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
        declare_model_path,
        declare_record_bag,

        localization_launch,
        rviz_node,
        kinoflow_startup,
        nav2_startup,
        nav2_lifecycle_startup,
        rosbag_recorder,
        ready_message,
    ])
