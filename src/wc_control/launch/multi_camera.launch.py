#!/usr/bin/env python3
"""
Multi-Camera Launch for Wheelchair Navigation
Launches 3 RealSense cameras with height-aware depth fusion

Based on RealSense ROS2 documentation:
https://github.com/IntelRealSense/realsense-ros

Camera Configuration:
- Front D455:  Serial 337122300107, namespace 'camera'         (depth + IMU)
- Left D455:   Serial 146222253403, namespace 'mapping_camera' (depth only)
- Right D435i: Serial 207522077542, namespace 'right_camera'   (depth only)

TF Frames (must match URDF):
- camera_link, camera_depth_optical_frame, camera_color_optical_frame
- mapping_camera_link, mapping_camera_depth_optical_frame
- right_camera_link, right_camera_depth_optical_frame

Usage:
  ros2 launch wc_control multi_camera.launch.py
  ros2 launch wc_control multi_camera.launch.py enable_left:=false  # 2-camera mode

Topics Published:
  /camera/depth/color/points         - Front camera point cloud
  /camera/imu                        - Front camera IMU (for fusion)
  /mapping_camera/depth/color/points - Left camera point cloud
  /right_camera/depth/color/points   - Right camera point cloud
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, GroupAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # ========================================================================
    # CAMERA SERIAL NUMBERS (underscore prefix forces string type)
    # ========================================================================
    FRONT_CAMERA_SERIAL = '_337122300107'  # D455 - Top/Front
    LEFT_CAMERA_SERIAL = '_146222253403'   # D455 - Left (mapping_camera)
    RIGHT_CAMERA_SERIAL = '_207522077542'  # D435i - Right

    realsense_launch_path = os.path.join(
        get_package_share_directory('realsense2_camera'),
        'launch',
        'rs_launch.py'
    )

    # ========================================================================
    # LAUNCH ARGUMENTS
    # ========================================================================
    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time', default_value='false',
        description='Use simulation time'
    )
    declare_enable_front = DeclareLaunchArgument(
        'enable_front', default_value='true',
        description='Enable front D455 camera'
    )
    declare_enable_left = DeclareLaunchArgument(
        'enable_left', default_value='true',
        description='Enable left D455 camera (mapping_camera)'
    )
    declare_enable_right = DeclareLaunchArgument(
        'enable_right', default_value='true',
        description='Enable right D435i camera'
    )
    declare_front_resolution = DeclareLaunchArgument(
        'front_resolution', default_value='640x480x15',
        description='Front camera resolution (WxHxFPS)'
    )
    declare_side_resolution = DeclareLaunchArgument(
        'side_resolution', default_value='424x240x15',
        description='Side cameras resolution (WxHxFPS)'
    )

    # ========================================================================
    # FRONT CAMERA (D455) - Primary with IMU
    # Namespace: /camera
    # Topics: /camera/depth/color/points, /camera/imu
    # TF Frame: camera_link (matches URDF)
    # ========================================================================
    front_camera = GroupAction(
        condition=IfCondition(LaunchConfiguration('enable_front')),
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(realsense_launch_path),
                launch_arguments={
                    'camera_name': 'camera',
                    'camera_namespace': '',
                    'serial_no': FRONT_CAMERA_SERIAL,
                    # IMU enabled only on primary camera
                    'enable_gyro': 'true',
                    'enable_accel': 'true',
                    'unite_imu_method': '2',  # Linear interpolation
                    # Depth settings
                    'depth_module.emitter_enabled': 'true',
                    'align_depth.enable': 'true',
                    'enable_sync': 'true',
                    'rgb_camera.profile': LaunchConfiguration('front_resolution'),
                    'depth_module.profile': LaunchConfiguration('front_resolution'),
                    # Point cloud generation
                    'pointcloud.enable': 'true',
                    'pointcloud.stream_filter': '2',  # RS2_STREAM_DEPTH
                    'pointcloud.stream_index_filter': '0',
                    # Post-processing filters (Intel recommended)
                    'decimation_filter.enable': 'true',
                    'decimation_filter.filter_magnitude': '2',
                    'spatial_filter.enable': 'true',
                    'spatial_filter.filter_magnitude': '2',
                    'spatial_filter.filter_smooth_alpha': '0.5',
                    'spatial_filter.filter_smooth_delta': '8',
                    'temporal_filter.enable': 'true',
                    'temporal_filter.filter_smooth_alpha': '0.4',
                    'temporal_filter.filter_smooth_delta': '20',
                    'hole_filling_filter.enable': 'false',
                    'use_sim_time': LaunchConfiguration('use_sim_time'),
                }.items(),
            ),
        ]
    )

    # ========================================================================
    # LEFT CAMERA (D455) - Depth only, no IMU
    # Namespace: /mapping_camera
    # Topics: /mapping_camera/depth/color/points
    # TF Frame: mapping_camera_link (matches URDF)
    # ========================================================================
    left_camera = GroupAction(
        condition=IfCondition(LaunchConfiguration('enable_left')),
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(realsense_launch_path),
                launch_arguments={
                    'camera_name': 'mapping_camera',
                    'camera_namespace': '',
                    'serial_no': LEFT_CAMERA_SERIAL,
                    # NO IMU on side cameras
                    'enable_gyro': 'false',
                    'enable_accel': 'false',
                    # Depth settings
                    'depth_module.emitter_enabled': 'true',
                    'align_depth.enable': 'true',
                    'enable_sync': 'true',
                    'enable_color': 'false',  # Depth only - saves USB bandwidth
                    'depth_module.profile': LaunchConfiguration('side_resolution'),
                    # Point cloud generation
                    'pointcloud.enable': 'true',
                    'pointcloud.stream_filter': '2',
                    'pointcloud.stream_index_filter': '0',
                    # Post-processing filters (lighter for side cameras)
                    'decimation_filter.enable': 'true',
                    'decimation_filter.filter_magnitude': '2',
                    'spatial_filter.enable': 'true',
                    'spatial_filter.filter_magnitude': '2',
                    'spatial_filter.filter_smooth_alpha': '0.5',
                    'spatial_filter.filter_smooth_delta': '8',
                    'temporal_filter.enable': 'false',  # Disabled for bandwidth
                    'hole_filling_filter.enable': 'false',
                    'use_sim_time': LaunchConfiguration('use_sim_time'),
                }.items(),
            ),
        ]
    )

    # ========================================================================
    # RIGHT CAMERA (D435i) - Depth only, no IMU
    # Namespace: /right_camera
    # Topics: /right_camera/depth/color/points
    # TF Frame: right_camera_link (matches URDF)
    # ========================================================================
    right_camera = GroupAction(
        condition=IfCondition(LaunchConfiguration('enable_right')),
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(realsense_launch_path),
                launch_arguments={
                    'camera_name': 'right_camera',
                    'camera_namespace': '',
                    'serial_no': RIGHT_CAMERA_SERIAL,
                    # NO IMU on side cameras
                    'enable_gyro': 'false',
                    'enable_accel': 'false',
                    # Depth settings
                    'depth_module.emitter_enabled': 'true',
                    'align_depth.enable': 'true',
                    'enable_sync': 'true',
                    'enable_color': 'false',  # Depth only - saves USB bandwidth
                    'depth_module.profile': LaunchConfiguration('side_resolution'),
                    # Point cloud generation
                    'pointcloud.enable': 'true',
                    'pointcloud.stream_filter': '2',
                    'pointcloud.stream_index_filter': '0',
                    # Post-processing filters (lighter for side cameras)
                    'decimation_filter.enable': 'true',
                    'decimation_filter.filter_magnitude': '2',
                    'spatial_filter.enable': 'true',
                    'spatial_filter.filter_magnitude': '2',
                    'spatial_filter.filter_smooth_alpha': '0.5',
                    'spatial_filter.filter_smooth_delta': '8',
                    'temporal_filter.enable': 'false',  # Disabled for bandwidth
                    'hole_filling_filter.enable': 'false',
                    'use_sim_time': LaunchConfiguration('use_sim_time'),
                }.items(),
            ),
        ]
    )

    return LaunchDescription([
        declare_use_sim_time,
        declare_enable_front,
        declare_enable_left,
        declare_enable_right,
        declare_front_resolution,
        declare_side_resolution,
        front_camera,
        left_camera,
        right_camera,
    ])
