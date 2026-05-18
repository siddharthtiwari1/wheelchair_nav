#!/usr/bin/env python3
"""
3-CAMERA FUSION TEST - OPTIMIZED FOR CPU
=========================================
Based on PDF: "Height-Aware 3-Camera Fusion for Autonomous Wheelchair Navigation"

Key specs from report (Section 2.1):
- Front D455: 640x480@15Hz (depth + IMU) - but we use 424x240 for fusion
- Left D455: 424x240@15Hz (depth only)
- Right D435i: 424x240@15Hz (depth only)

CPU Optimization:
- All cameras at 424x240x6 for reduced bandwidth
- Staggered startup to avoid USB saturation
- Fusion node with aggressive downsampling (8x)
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import TimerAction, LogInfo, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg_description = get_package_share_directory('wheelchair_description')
    pkg_localization = get_package_share_directory('wheelchair_localization')

    urdf_path = os.path.join(pkg_description, 'urdf', 'wheelchair_description.urdf.xacro')
    rviz_config = os.path.join(pkg_description, 'rviz', 'camera_test_minimal.rviz')
    laser_filter_config = os.path.join(pkg_localization, 'config', 'laser_filter_robust.yaml')

    rs_launch_path = os.path.join(
        get_package_share_directory('realsense2_camera'),
        'launch', 'rs_launch.py'
    )

    robot_description = ParameterValue(
        Command(['xacro ', urdf_path, ' is_sim:=false is_ignition:=false']),
        value_type=str
    )

    # Camera serials (from your hardware)
    FRONT_SERIAL = "'337122300107'"
    LEFT_SERIAL = "'146222253403'"
    RIGHT_SERIAL = "'207522077542'"

    # ========================================================================
    # ROBOT STATE PUBLISHER
    # ========================================================================
    robot_state_pub = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_description}]
    )

    joint_state_pub = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        parameters=[{'rate': 30}]  # Reduced from 50
    )

    # ========================================================================
    # STATIC TFs
    # ========================================================================
    # lidar -> laser: 180° rotation (scan direction correction)
    static_tf_lidar_laser = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='lidar_to_laser_tf',
        arguments=['--x', '0', '--y', '0', '--z', '0',
                   '--qx', '0', '--qy', '0', '--qz', '1', '--qw', '0',
                   '--frame-id', 'lidar', '--child-frame-id', 'laser'],
    )

    # base_link -> imu
    static_tf_imu = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='base_to_imu_tf',
        arguments=['--x', '0', '--y', '0', '--z', '0',
                   '--qx', '0', '--qy', '0', '--qz', '0', '--qw', '1',
                   '--frame-id', 'base_link', '--child-frame-id', 'imu'],
    )

    # ========================================================================
    # RPLIDAR S3
    # ========================================================================
    rplidar_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('rplidar_ros'),
                         'launch', 'rplidar_s3_launch.py')
        ),
        launch_arguments={'inverted': 'true'}.items(),
    )

    # Laser filter
    laser_filter = Node(
        package='laser_filters',
        executable='scan_to_scan_filter_chain',
        name='laser_filter',
        output='screen',
        parameters=[laser_filter_config],
        remappings=[
            ('scan', '/scan'),
            ('scan_filtered', '/scan_filtered'),
        ]
    )

    # ========================================================================
    # CAMERAS - LOW RESOLUTION FOR CPU EFFICIENCY
    # All cameras at 424x240x6 (6fps reduces load significantly)
    # ========================================================================

    # Front Camera (D455) - WITH IMU
    front_cam = TimerAction(
        period=3.0,
        actions=[
            LogInfo(msg='[CAMERA] Starting FRONT D455 (424x240@6Hz + IMU)...'),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(rs_launch_path),
                launch_arguments={
                    'camera_name': 'camera',
                    'camera_namespace': '',
                    'serial_no': FRONT_SERIAL,
                    'publish_tf': 'false',
                    'enable_gyro': 'true',
                    'enable_accel': 'true',
                    'unite_imu_method': '2',
                    'enable_color': 'false',  # No color - save bandwidth
                    'depth_module.profile': '424x240x6',  # LOW resolution + framerate
                    'pointcloud.enable': 'true',
                    'pointcloud.stream_filter': '0',  # Depth only
                    'pointcloud.stream_index_filter': '0',
                }.items(),
            ),
        ]
    )

    # Left Camera (D455) - DEPTH ONLY
    left_cam = TimerAction(
        period=8.0,  # Staggered to avoid USB bandwidth collision
        actions=[
            LogInfo(msg='[CAMERA] Starting LEFT D455 (424x240@6Hz depth only)...'),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(rs_launch_path),
                launch_arguments={
                    'camera_name': 'mapping_camera',
                    'camera_namespace': '',
                    'serial_no': LEFT_SERIAL,
                    'publish_tf': 'false',
                    'enable_gyro': 'false',
                    'enable_accel': 'false',
                    'enable_color': 'false',
                    'depth_module.profile': '424x240x6',
                    'pointcloud.enable': 'true',
                    'pointcloud.stream_filter': '0',
                    'pointcloud.stream_index_filter': '0',
                }.items(),
            ),
        ]
    )

    # Right Camera (D435i) - DEPTH ONLY
    right_cam = TimerAction(
        period=13.0,  # More delay to spread USB load
        actions=[
            LogInfo(msg='[CAMERA] Starting RIGHT D435i (424x240@6Hz depth only)...'),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(rs_launch_path),
                launch_arguments={
                    'camera_name': 'right_camera',
                    'camera_namespace': '',
                    'serial_no': RIGHT_SERIAL,
                    'publish_tf': 'false',
                    'enable_gyro': 'false',
                    'enable_accel': 'false',
                    'enable_color': 'false',
                    'depth_module.profile': '424x240x6',
                    'pointcloud.enable': 'true',
                    'pointcloud.stream_filter': '0',
                    'pointcloud.stream_index_filter': '0',
                }.items(),
            ),
        ]
    )

    # ========================================================================
    # FUSION NODE - OPTIMIZED
    # ========================================================================
    fusion_node = TimerAction(
        period=18.0,  # Wait for all cameras to initialize
        actions=[
            LogInfo(msg='=' * 60),
            LogInfo(msg='[FUSION] Starting 3-camera + LiDAR fusion'),
            LogInfo(msg='=' * 60),
            Node(
                package='wheelchair_localization',
                executable='multi_camera_scan_fusion_simple',
                name='scan_fusion',
                output='screen',
                parameters=[{
                    'scan_topic': '/scan_filtered',
                    'output_topic': '/scan_fused',
                    'laser_frame': 'laser',
                    'min_height': 0.10,  # Report Section 4.4
                    'max_height': 1.80,
                    # Front camera
                    'front_camera.enabled': True,
                    'front_camera.topic': '/camera/depth/color/points',
                    'front_camera.frame': 'camera_depth_optical_frame',
                    'front_camera.min_depth': 0.30,
                    'front_camera.max_depth': 5.0,
                    # Left camera
                    'left_camera.enabled': True,
                    'left_camera.topic': '/mapping_camera/depth/color/points',
                    'left_camera.frame': 'mapping_camera_depth_optical_frame',
                    'left_camera.min_depth': 0.30,
                    'left_camera.max_depth': 5.0,
                    # Right camera
                    'right_camera.enabled': True,
                    'right_camera.topic': '/right_camera/depth/color/points',
                    'right_camera.frame': 'right_camera_depth_optical_frame',
                    'right_camera.min_depth': 0.30,
                    'right_camera.max_depth': 5.0,
                    'verbose': False,  # Reduce logging overhead
                }],
            ),
        ]
    )

    # ========================================================================
    # RVIZ (delayed start)
    # ========================================================================
    rviz = TimerAction(
        period=5.0,
        actions=[
            Node(
                package='rviz2',
                executable='rviz2',
                name='rviz2',
                output='screen',
                arguments=['-d', rviz_config],
            )
        ]
    )

    # Ready message
    ready_msg = TimerAction(
        period=22.0,
        actions=[
            LogInfo(msg='=' * 60),
            LogInfo(msg='  3-CAMERA FUSION TEST READY'),
            LogInfo(msg='  Topics:'),
            LogInfo(msg='    /scan_filtered - LiDAR only'),
            LogInfo(msg='    /scan_fused    - LiDAR + 3 cameras'),
            LogInfo(msg='=' * 60),
        ]
    )

    return LaunchDescription([
        LogInfo(msg='=' * 60),
        LogInfo(msg='  3-CAMERA FUSION TEST - CPU OPTIMIZED'),
        LogInfo(msg='  Resolution: 424x240@6Hz per camera'),
        LogInfo(msg='=' * 60),
        robot_state_pub,
        joint_state_pub,
        static_tf_lidar_laser,
        static_tf_imu,
        rplidar_launch,
        laser_filter,
        front_cam,
        left_cam,
        right_cam,
        fusion_node,
        rviz,
        ready_msg,
    ])
