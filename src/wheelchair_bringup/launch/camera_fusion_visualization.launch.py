#!/usr/bin/env python3
"""
Camera Fusion Visualization Launch
===================================
Complete visualization of 3-camera + LiDAR scan fusion system.

Launches:
- Robot state publisher (URDF with 3 cameras)
- RPLidar S3 driver
- 3 RealSense cameras (Front D455, Left D455, Right D435i)
- Multi-camera SOTA scan fusion node
- RViz with full visualization

Usage:
    ros2 launch wheelchair_bringup camera_fusion_visualization.launch.py

Author: Siddharth Tiwari
Date: 2026-02-02
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    # =========================================================================
    # PACKAGE DIRECTORIES
    # =========================================================================
    pkg_description = get_package_share_directory('wheelchair_description')
    pkg_localization = get_package_share_directory('wheelchair_localization')

    # =========================================================================
    # PATHS
    # =========================================================================
    urdf_path = os.path.join(pkg_description, 'urdf', 'wheelchair_description.urdf.xacro')
    default_rviz_config = os.path.join(pkg_description, 'rviz', 'fusion_navigation.rviz')
    laser_filter_config = os.path.join(pkg_localization, 'config', 'laser_filter_robust.yaml')

    # =========================================================================
    # CAMERA SERIAL NUMBERS (underscore prefix forces string type)
    # =========================================================================
    FRONT_CAMERA_SERIAL = '_337122300107'  # D455 - Top/Front
    LEFT_CAMERA_SERIAL = '_146222253403'   # D455 - Left
    RIGHT_CAMERA_SERIAL = '_207522077542'  # D435i - Right

    # =========================================================================
    # LAUNCH ARGUMENTS
    # =========================================================================
    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time', default_value='false',
        description='Use simulation time'
    )

    declare_rviz_config = DeclareLaunchArgument(
        'rviz_config', default_value=default_rviz_config,
        description='RViz configuration file'
    )

    # =========================================================================
    # ROBOT DESCRIPTION
    # =========================================================================
    robot_description = ParameterValue(
        Command(['xacro ', urdf_path, ' is_sim:=false is_ignition:=false']),
        value_type=str
    )

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_description,
            'use_sim_time': False,
        }]
    )

    # Static transform: base_link -> laser
    static_tf_base_laser = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='base_to_laser_tf',
        arguments=['0.475', '0.12', '0.07', '0', '0', '0', 'base_link', 'laser'],
    )

    # =========================================================================
    # RPLIDAR S3 DRIVER
    # =========================================================================
    rplidar_node = Node(
        package='rplidar_ros',
        executable='rplidar_node',
        name='rplidar_node',
        output='screen',
        parameters=[{
            'serial_port': '/dev/ttyUSB0',
            'serial_baudrate': 1000000,
            'frame_id': 'laser',
            'angle_compensate': True,
            'scan_mode': 'DenseBoost',
        }],
        remappings=[
            ('scan', '/scan_raw'),
        ]
    )

    # =========================================================================
    # LASER FILTER
    # =========================================================================
    laser_filter_node = Node(
        package='laser_filters',
        executable='scan_to_scan_filter_chain',
        name='laser_filter',
        output='screen',
        parameters=[laser_filter_config],
        remappings=[
            ('scan', '/scan_raw'),
            ('scan_filtered', '/scan_filtered'),
        ]
    )

    # =========================================================================
    # REALSENSE CAMERAS
    # =========================================================================
    realsense_launch_path = os.path.join(
        get_package_share_directory('realsense2_camera'),
        'launch', 'rs_launch.py'
    )

    # Front Camera (D455) - with IMU
    front_camera = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(realsense_launch_path),
        launch_arguments={
            'camera_name': 'camera',
            'camera_namespace': '',
            'serial_no': FRONT_CAMERA_SERIAL,
            'publish_tf': 'false',  # Use URDF TFs, not RealSense TFs
            'enable_gyro': 'true',
            'enable_accel': 'true',
            'unite_imu_method': '2',
            'depth_module.emitter_enabled': 'true',
            'align_depth.enable': 'true',
            'enable_sync': 'true',
            'rgb_camera.profile': '640x480x15',
            'depth_module.profile': '640x480x15',
            'pointcloud.enable': 'true',
            'pointcloud.stream_filter': '2',
            'pointcloud.stream_index_filter': '0',
            'decimation_filter.enable': 'true',
            'decimation_filter.filter_magnitude': '2',
            'spatial_filter.enable': 'true',
            'temporal_filter.enable': 'true',
        }.items(),
    )

    # Left Camera (D455) - depth only
    left_camera = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(realsense_launch_path),
        launch_arguments={
            'camera_name': 'mapping_camera',
            'camera_namespace': '',
            'serial_no': LEFT_CAMERA_SERIAL,
            'publish_tf': 'false',  # Use URDF TFs, not RealSense TFs
            'enable_gyro': 'false',
            'enable_accel': 'false',
            'depth_module.emitter_enabled': 'true',
            'align_depth.enable': 'true',
            'enable_sync': 'true',
            'enable_color': 'false',
            'depth_module.profile': '424x240x15',
            'pointcloud.enable': 'true',
            'pointcloud.stream_filter': '2',
            'pointcloud.stream_index_filter': '0',
            'decimation_filter.enable': 'true',
            'decimation_filter.filter_magnitude': '2',
            'spatial_filter.enable': 'true',
            'temporal_filter.enable': 'false',
        }.items(),
    )

    # Right Camera (D435i) - depth only
    right_camera = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(realsense_launch_path),
        launch_arguments={
            'camera_name': 'right_camera',
            'camera_namespace': '',
            'serial_no': RIGHT_CAMERA_SERIAL,
            'publish_tf': 'false',  # Use URDF TFs, not RealSense TFs
            'enable_gyro': 'false',
            'enable_accel': 'false',
            'depth_module.emitter_enabled': 'true',
            'align_depth.enable': 'true',
            'enable_sync': 'true',
            'enable_color': 'false',
            'depth_module.profile': '424x240x15',
            'pointcloud.enable': 'true',
            'pointcloud.stream_filter': '2',
            'pointcloud.stream_index_filter': '0',
            'decimation_filter.enable': 'true',
            'decimation_filter.filter_magnitude': '2',
            'spatial_filter.enable': 'true',
            'temporal_filter.enable': 'false',
        }.items(),
    )

    # =========================================================================
    # SCAN FUSION NODE (delayed start)
    # =========================================================================
    scan_fusion_node = TimerAction(
        period=8.0,  # Wait 8 seconds for cameras to initialize
        actions=[
            Node(
                package='wheelchair_localization',
                executable='multi_camera_scan_fusion_sota',
                name='multi_camera_scan_fusion',
                output='screen',
                parameters=[{
                    'scan_topic': '/scan_filtered',
                    'output_topic': '/scan_fused',
                    'laser_frame': 'laser',
                    # Front camera
                    'front_camera.enabled': True,
                    'front_camera.topic': '/camera/depth/color/points',
                    'front_camera.frame': 'camera_depth_optical_frame',
                    'front_camera.min_depth': 0.30,
                    'front_camera.max_depth': 5.0,
                    'front_camera.downsample': 2,
                    # Left camera
                    'left_camera.enabled': True,
                    'left_camera.topic': '/mapping_camera/depth/color/points',
                    'left_camera.frame': 'mapping_camera_depth_optical_frame',
                    'left_camera.min_depth': 0.30,
                    'left_camera.max_depth': 5.0,
                    'left_camera.downsample': 2,
                    # Right camera
                    'right_camera.enabled': True,
                    'right_camera.topic': '/right_camera/depth/color/points',
                    'right_camera.frame': 'right_camera_depth_optical_frame',
                    'right_camera.min_depth': 0.30,
                    'right_camera.max_depth': 5.0,
                    'right_camera.downsample': 2,
                    # Fusion parameters
                    'fusion.min_height': 0.10,
                    'fusion.max_height': 1.80,
                    'fusion.sigma_base': 0.02,
                    'fusion.sigma_scale': 0.001,
                    'fusion.temporal_window': 9,
                    'fusion.use_soft_association': False,
                    'fusion.verbose': True,
                    # Sync - generous for 4 sensors (LiDAR + 3 cameras)
                    'sync_slop': 0.2,
                    'sync_queue_size': 15,
                }],
            ),
        ]
    )

    # =========================================================================
    # RVIZ
    # =========================================================================
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', LaunchConfiguration('rviz_config')],
    )

    # =========================================================================
    # RETURN LAUNCH DESCRIPTION
    # =========================================================================
    return LaunchDescription([
        # Arguments
        declare_use_sim_time,
        declare_rviz_config,

        # Robot description
        robot_state_publisher,
        static_tf_base_laser,

        # Sensors
        rplidar_node,
        laser_filter_node,
        front_camera,
        left_camera,
        right_camera,

        # Fusion (delayed)
        scan_fusion_node,

        # Visualization
        rviz_node,
    ])
