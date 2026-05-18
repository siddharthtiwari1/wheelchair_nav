#!/usr/bin/env python3
"""Launch all 3 RealSense cameras with color enabled + data collection node.

Unlike multi_camera.launch.py (which disables color on side cameras to save
USB bandwidth), this launch enables color on ALL cameras for collecting
paired RGB-depth training data. Side cameras use reduced FPS to compensate.

Usage:
    ros2 launch wheelchair_mono_depth data_collection.launch.py
    ros2 launch wheelchair_mono_depth data_collection.launch.py save_rate_hz:=5.0
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction, LogInfo
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Camera serial numbers (must match multi_camera.launch.py)
    FRONT_SERIAL = '_337122300107'   # D455
    LEFT_SERIAL = '_146222253403'    # D455
    RIGHT_SERIAL = '_207522077542'   # D435i

    pkg_dir = get_package_share_directory('wheelchair_mono_depth')
    realsense_launch = os.path.join(
        get_package_share_directory('realsense2_camera'),
        'launch', 'rs_launch.py',
    )
    config_file = os.path.join(pkg_dir, 'config', 'data_collection.yaml')

    # Launch arguments
    declare_save_rate = DeclareLaunchArgument(
        'save_rate_hz', default_value='3.0',
        description='Rate to save image pairs (Hz)',
    )
    declare_output_dir = DeclareLaunchArgument(
        'output_dir', default_value='/home/sidd/wheelchair_nav/mono_depth_data',
        description='Root directory for collected data',
    )
    declare_session_id = DeclareLaunchArgument(
        'session_id', default_value='',
        description='Session identifier (auto-generated if empty)',
    )

    # Front camera: D455 with color + depth + IMU
    front_camera = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(realsense_launch),
        launch_arguments={
            'camera_name': 'camera',
            'camera_namespace': '',
            'serial_no': FRONT_SERIAL,
            'enable_gyro': 'true',
            'enable_accel': 'true',
            'unite_imu_method': '2',
            'enable_color': 'true',
            'align_depth.enable': 'true',
            'enable_sync': 'true',
            'rgb_camera.profile': '640x480x15',
            'depth_module.profile': '640x480x15',
            'pointcloud.enable': 'false',  # Not needed for collection
            'decimation_filter.enable': 'false',
            'spatial_filter.enable': 'false',
            'temporal_filter.enable': 'false',
            'hole_filling_filter.enable': 'false',
        }.items(),
    )

    # Left camera: D455 with COLOR ENABLED (override from multi_camera)
    left_camera = TimerAction(
        period=5.0,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(realsense_launch),
                launch_arguments={
                    'camera_name': 'mapping_camera',
                    'camera_namespace': '',
                    'serial_no': LEFT_SERIAL,
                    'enable_gyro': 'false',
                    'enable_accel': 'false',
                    'enable_color': 'true',   # ENABLED for data collection
                    'align_depth.enable': 'true',
                    'enable_sync': 'true',
                    'rgb_camera.profile': '424x240x6',  # Low FPS for USB bandwidth
                    'depth_module.profile': '424x240x6',
                    'pointcloud.enable': 'false',
                    'decimation_filter.enable': 'false',
                    'spatial_filter.enable': 'false',
                    'temporal_filter.enable': 'false',
                    'hole_filling_filter.enable': 'false',
                }.items(),
            ),
        ],
    )

    # Right camera: D435i with COLOR ENABLED (override from multi_camera)
    right_camera = TimerAction(
        period=10.0,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(realsense_launch),
                launch_arguments={
                    'camera_name': 'right_camera',
                    'camera_namespace': '',
                    'serial_no': RIGHT_SERIAL,
                    'enable_gyro': 'false',
                    'enable_accel': 'false',
                    'enable_color': 'true',   # ENABLED for data collection
                    'align_depth.enable': 'true',
                    'enable_sync': 'true',
                    'rgb_camera.profile': '424x240x6',
                    'depth_module.profile': '424x240x6',
                    'pointcloud.enable': 'false',
                    'decimation_filter.enable': 'false',
                    'spatial_filter.enable': 'false',
                    'temporal_filter.enable': 'false',
                    'hole_filling_filter.enable': 'false',
                }.items(),
            ),
        ],
    )

    # Data collection node (delay to let all cameras initialize)
    collection_node = TimerAction(
        period=18.0,
        actions=[
            LogInfo(msg='Starting depth data collection node...'),
            Node(
                package='wheelchair_mono_depth',
                executable='data_collection_node',
                name='depth_data_collection_node',
                output='screen',
                parameters=[
                    config_file,
                    {
                        'save_rate_hz': LaunchConfiguration('save_rate_hz'),
                        'output_dir': LaunchConfiguration('output_dir'),
                        'session_id': LaunchConfiguration('session_id'),
                    },
                ],
            ),
        ],
    )

    return LaunchDescription([
        declare_save_rate,
        declare_output_dir,
        declare_session_id,
        LogInfo(msg='Launching 3 cameras for depth data collection (color ON all)...'),
        front_camera,
        left_camera,
        right_camera,
        collection_node,
    ])
