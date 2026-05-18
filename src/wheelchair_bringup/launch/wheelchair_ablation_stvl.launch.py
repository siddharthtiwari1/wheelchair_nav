#!/usr/bin/env python3

"""
ABLATION: STVL 3-CAMERA NAVIGATION (no PRISM scan fusion)
==========================================================
STVL baseline for PRISM paper ablation study.

- AMCL localizes on /scan_lidar (footprint-filtered LiDAR only — no PRISM fusion)
- Local costmap: stvl_layer (3 cameras) + obstacle_layer + inflation_layer
- Map: lidar-only SLAM map (no PRISM-enhanced map)
- scan_lidar node applies wheelchair footprint filter to /scan_filtered
- Cameras contribute ONLY through STVL costmap plugin (runtime obstacle avoidance)
- No pre-SLAM scan fusion — tests cameras-via-STVL-only approach

Usage:
    ros2 launch wheelchair_bringup wheelchair_ablation_stvl.launch.py

    With custom map:
    ros2 launch wheelchair_bringup wheelchair_ablation_stvl.launch.py \
        map_name:=/path/to/map.yaml
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

    # ABLATION DEFAULTS — LiDAR-only map + STVL nav2 params
    default_map_file = os.path.join(ws_root, 'maps', '2026-03-06_lidar_only.yaml')
    default_nav2_params = os.path.join(wheelchair_navigation_dir, 'config', 'nav2_params_ablation_stvl_v2.yaml')
    default_bt_xml = os.path.join(wheelchair_navigation_dir, 'behavior_tree', 'wheelchair_robust_nav.xml')
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
    use_collision_monitor = LaunchConfiguration('use_collision_monitor')
    record_bag = LaunchConfiguration('record_bag')

    # ========================================================================
    # SCAN LIDAR — footprint-filtered /scan_filtered -> /scan_lidar
    # ========================================================================

    scan_lidar_node = Node(
        package='wheelchair_localization',
        executable='scan_lidar',
        name='scan_lidar',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
    )

    # ========================================================================
    # LOCALIZATION SYSTEM (cameras launch — needed for STVL costmap)
    # ========================================================================

    localization_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(wheelchair_bringup_dir, 'launch', 'wheelchair_fusion_localization.launch.py')
        ),
        launch_arguments={
            'map_name': map_name,
            'use_sim_time': use_sim_time,
            'use_fused_scan': 'false',   # ABLATION: AMCL uses /scan_filtered (no PRISM)
            'rviz': 'false',
        }.items()
    )

    # ========================================================================
    # NAV2 NAVIGATION STACK (identical structure to fusion_nav)
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

    collision_monitor = Node(
        package='nav2_collision_monitor',
        executable='collision_monitor',
        name='collision_monitor',
        output='screen',
        parameters=[nav2_params, {'use_sim_time': use_sim_time}],
        condition=IfCondition(use_collision_monitor),
    )

    topic_data_logger = Node(
        package='scripts',
        executable='topic_data_logger',
        name='topic_data_logger',
        output='screen',
        parameters=[
            {'log_frequency_hz': 10.0},
            {'file_prefix': 'ablation_stvl_log'},
        ],
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
            LogInfo(msg='[RVIZ] Starting RViz (ABLATION: STVL 3-camera)...'),
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

    nav2_startup = TimerAction(
        period=28.0,
        actions=[
            LogInfo(msg='[NAV2] Starting navigation stack (ABLATION: STVL 3-camera)...'),
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
        period=28.0,
        actions=[collision_monitor],
    )

    data_logger_startup = TimerAction(
        period=10.0,
        actions=[topic_data_logger],
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
            LogInfo(msg='  ABLATION: STVL 3-CAMERA NAVIGATION - SYSTEM READY'),
            LogInfo(msg='=' * 70),
            LogInfo(msg='  Config: STVL 3-camera costmap, NO PRISM scan fusion'),
            LogInfo(msg='  AMCL: /scan_lidar (footprint-filtered LiDAR — no PRISM)'),
            LogInfo(msg='  Local costmap: stvl_layer + obstacle_layer + inflation'),
            LogInfo(msg='  STVL sources: front D455 + left D455 + right D435i'),
            LogInfo(msg='  Map: lidar-only SLAM map (no PRISM-enhanced map)'),
            LogInfo(msg='  Send goals via RViz "2D Goal Pose" button'),
            LogInfo(msg='=' * 70),
        ]
    )

    # ========================================================================
    # ROSBAG RECORDING
    # ========================================================================

    bag_output_dir = os.path.join(
        ws_root, 'rosbags', 'ablation_stvl_' + datetime.now().strftime('%Y%m%d_%H%M%S')
    )

    rosbag_recorder = TimerAction(
        period=30.0,
        actions=[
            LogInfo(msg='[ROSBAG] Recording ablation data to ' + bag_output_dir),
            ExecuteProcess(
                cmd=[
                    'ros2', 'bag', 'record',
                    '-o', bag_output_dir,
                    '--topics',
                    '/cmd_vel', '/cmd_vel_nav',
                    '/odometry/filtered', '/amcl_pose',
                    '/goal_pose', '/local_plan', '/plan',
                    '/scan_filtered', '/scan_lidar', '/scan_fused',
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
        declare_use_collision_monitor,
        declare_record_bag,

        localization_launch,
        scan_lidar_node,
        rviz_node,
        nav2_startup,
        collision_monitor_startup,
        nav2_lifecycle_startup,
        data_logger_startup,
        rosbag_recorder,
        ready_message,
    ])
