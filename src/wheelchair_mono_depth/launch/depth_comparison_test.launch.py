#!/usr/bin/env python3
"""Stereo vs DA3 Metric Large Depth-to-LaserScan comparison.

Two pipelines on the front D455:
  A)  Stereo depth → PointCloud2 → LaserScan  (/scan_stereo)         GREEN
  B)  RGB → DA3 Metric Large → PC2 → LaserScan (/scan_da3)           ORANGE

Timeline:
  t=0s   URDF + TF + RPLidar + laser_filter
  t=3s   Front D455 camera
  t=8s   Stereo pc2ls
  t=10s  DA3 depth node (no compile, FP16, cudnn.benchmark)
  t=14s  DA3 pc2ls (raw)
  t=18s  Benchmark
  t=20s  RViz
  t=23s  Ready

Usage:
    ros2 launch wheelchair_mono_depth depth_comparison_test.launch.py
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
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


# ============================================================================
# CONSTANTS
# ============================================================================
FRONT_SERIAL = "'337122300107'"   # Front D455
LIDAR_SERIAL_PORT = '/dev/ttyUSB0'

PC2LS_PARAMS = {
    'target_frame': 'laser',
    'transform_tolerance': 0.1,
    'min_height': 0.10,
    'max_height': 1.80,
    'angle_min': -3.14159,
    'angle_max': 3.14159,
    'angle_increment': 0.00872665,  # ~0.5deg, 720 bins
    'scan_time': 0.1667,
    'range_min': 0.30,
    'range_max': 5.0,
    'use_inf': True,
}


def generate_launch_description():
    mono_depth_dir = get_package_share_directory('wheelchair_mono_depth')
    description_dir = get_package_share_directory('wheelchair_description')
    localization_dir = get_package_share_directory('wheelchair_localization')
    rs_launch_path = os.path.join(
        get_package_share_directory('realsense2_camera'),
        'launch', 'rs_launch.py')

    benchmark_config = os.path.join(
        mono_depth_dir, 'config', 'depth_comparison_benchmark_v5.yaml')
    laser_filter_config = os.path.join(
        localization_dir, 'config', 'laser_filter_robust.yaml')
    rviz_config = os.path.join(
        description_dir, 'rviz', 'depth_comparison_test_v4.rviz')
    xacro_path = os.path.join(
        description_dir, 'urdf', 'wheelchair_description.urdf.xacro')

    declare_use_rviz = DeclareLaunchArgument(
        'use_rviz', default_value='true')

    # URDF
    robot_description = Command([
        'xacro ', xacro_path, ' is_sim:=false port:=/dev/ttyACM0'])

    robot_state_pub = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': ParameterValue(robot_description, value_type=str),
            'use_sim_time': False,
        }],
    )

    joint_state_pub = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        parameters=[{
            'use_sim_time': False,
            'robot_description': ParameterValue(robot_description, value_type=str),
        }],
    )

    static_tf_base_imu = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=[
            '--x', '0', '--y', '0', '--z', '0',
            '--qx', '0', '--qy', '0', '--qz', '0', '--qw', '1',
            '--frame-id', 'base_link', '--child-frame-id', 'imu',
        ],
    )

    # RPLidar S3 — matches wheelchair_slam_mapping.launch.py exactly
    rplidar_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(
                get_package_share_directory('rplidar_ros'),
                'launch', 'rplidar_s3_launch.py',
            )
        ]),
        launch_arguments={'inverted': 'true'}.items(),
    )

    # Laser filter chain: /scan → /scan_filtered
    laser_filter = Node(
        package='laser_filters',
        executable='scan_to_scan_filter_chain',
        name='laser_filter',
        output='screen',
        parameters=[laser_filter_config],
        remappings=[
            ('scan', '/scan'),
            ('scan_filtered', '/scan_filtered'),
        ],
    )

    # t=3s: Camera — D455 front
    # NOTE: 640x480x6 was rejected by this D455 (falls back to 1280x720x30).
    # Using 424x240x6 for stereo depth, 1280x720x6 for RGB.
    front_camera = TimerAction(
        period=3.0,
        actions=[
            LogInfo(msg='[CAMERA] Starting front D455...'),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(rs_launch_path),
                launch_arguments={
                    'camera_name': 'camera',
                    'camera_namespace': '',
                    'serial_no': FRONT_SERIAL,
                    'publish_tf': 'false',
                    'enable_gyro': 'false',
                    'enable_accel': 'false',
                    'enable_color': 'true',
                    'depth_module.profile': '424x240x6',
                    'rgb_camera.color_profile': '1280x720x6',
                    'pointcloud.enable': 'true',
                    'pointcloud.stream_filter': '0',
                    'pointcloud.stream_index_filter': '0',
                }.items(),
            ),
        ],
    )

    # t=8s: Stereo pc2ls
    stereo_pc2ls = TimerAction(
        period=8.0,
        actions=[
            LogInfo(msg='[STEREO] Starting stereo pc2ls...'),
            Node(
                package='pointcloud_to_laserscan',
                executable='pointcloud_to_laserscan_node',
                name='pc2ls_stereo',
                output='screen',
                parameters=[PC2LS_PARAMS],
                remappings=[
                    ('cloud_in', '/camera/depth/color/points'),
                    ('scan', '/scan_stereo'),
                ],
            ),
        ],
    )

    # t=10s: DA3 Metric Large — no torch.compile, FP16, cudnn.benchmark
    da3_node = TimerAction(
        period=10.0,
        actions=[
            LogInfo(msg='[DA3] Starting DA3 Metric Large (no compile)...'),
            Node(
                package='wheelchair_mono_depth',
                executable='da3_depth_node',
                name='da3_depth',
                output='screen',
                parameters=[{
                    'model_name': 'da3-metric-large',
                    'rgb_topic': '/camera/color/image_raw',
                    'camera_info_topic': '/camera/color/camera_info',
                    'max_depth': 8.0,
                    'depth_correction': 1.079,  # recalibrated 2026-03-04 (was 1.142)
                    'process_width': 504,
                    'inference_hz': 15.0,
                    'compile_model': False,
                    'temporal_alpha': 0.0,
                    'output_depth_topic': '/camera/mono_da3/image_raw',
                    'output_info_topic': '/camera/mono_da3/camera_info',
                    'pointcloud_topic': '/camera/mono_da3/points',
                }],
            ),
        ],
    )

    # t=14s: DA3 raw pc2ls
    pc2ls_da3 = TimerAction(
        period=14.0,
        actions=[
            LogInfo(msg='[PC2LS] Starting DA3 scan conversion...'),
            Node(
                package='pointcloud_to_laserscan',
                executable='pointcloud_to_laserscan_node',
                name='pc2ls_da3',
                output='screen',
                parameters=[PC2LS_PARAMS],
                remappings=[
                    ('cloud_in', '/camera/mono_da3/points'),
                    ('scan', '/scan_da3'),
                ],
            ),
        ],
    )

    # t=18s: Benchmark
    benchmark = TimerAction(
        period=18.0,
        actions=[
            LogInfo(msg='[BENCHMARK] Starting comparison...'),
            Node(
                package='wheelchair_mono_depth',
                executable='depth_scan_benchmark_node',
                name='depth_scan_benchmark',
                output='screen',
                parameters=[benchmark_config],
            ),
        ],
    )

    # t=20s: RViz
    rviz = TimerAction(
        period=20.0,
        actions=[
            LogInfo(msg='[RVIZ] Launching...'),
            Node(
                package='rviz2',
                executable='rviz2',
                name='rviz2',
                output='screen',
                arguments=['-d', rviz_config],
            ),
        ],
    )

    # t=23s: Ready
    ready_msg = TimerAction(
        period=23.0,
        actions=[
            LogInfo(msg='=' * 60),
            LogInfo(msg='  DEPTH COMPARISON — 2 SCANS (no anchor)'),
            LogInfo(msg='  RED:    /scan_filtered  — LiDAR 360°'),
            LogInfo(msg='  GREEN:  /scan_stereo    — Stereo metric depth'),
            LogInfo(msg='  ORANGE: /scan_da3       — DA3 Metric Large (1.142x)'),
            LogInfo(msg='=' * 60),
        ],
    )

    return LaunchDescription([
        declare_use_rviz,
        robot_state_pub,
        joint_state_pub,
        static_tf_base_imu,
        rplidar_launch,
        laser_filter,
        front_camera,
        stereo_pc2ls,
        da3_node,
        pc2ls_da3,
        benchmark,
        rviz,
        ready_msg,
    ])
