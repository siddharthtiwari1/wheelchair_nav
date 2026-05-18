#!/usr/bin/env python3
"""
CAMERA + LIDAR FUSION TEST LAUNCH
==================================
Wheelchair URDF + 3 cameras + LiDAR + Scan Fusion + RViz

Depth images are converted to laser scans, then all 4 scans are fused.

Usage:
    ros2 launch wheelchair_bringup camera_test_minimal.launch.py
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg_description = get_package_share_directory('wheelchair_description')
    pkg_localization = get_package_share_directory('wheelchair_localization')

    # URDF
    urdf_path = os.path.join(pkg_description, 'urdf', 'wheelchair_description.urdf.xacro')
    rviz_config = os.path.join(pkg_description, 'rviz', 'camera_test_minimal.rviz')
    laser_filter_config = os.path.join(pkg_localization, 'config', 'laser_filter_robust.yaml')

    # Camera serials (must be quoted string format)
    FRONT_SERIAL = "'337122300107'"
    LEFT_SERIAL = "'146222253403'"
    RIGHT_SERIAL = "'207522077542'"

    robot_description = ParameterValue(
        Command(['xacro ', urdf_path, ' is_sim:=false is_ignition:=false']),
        value_type=str
    )

    # Robot state publisher
    robot_state_pub = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_description}]
    )

    # Joint state publisher (publishes wheel joint states)
    joint_state_pub = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        output='screen',
        parameters=[{'rate': 50}]
    )

    # Static TF: lidar -> laser (180° rotation ONLY, position comes from URDF)
    # URDF defines: base_link -> wheelchair_main -> lidar (position)
    # This TF adds: lidar -> laser (180° yaw rotation for scan alignment)
    # Using quaternion: qz=1, qw=0 = 180° around Z axis
    static_tf_lidar_to_laser = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='lidar_to_laser_tf',
        arguments=['--x', '0', '--y', '0', '--z', '0',
                   '--qx', '0', '--qy', '0', '--qz', '1', '--qw', '0',
                   '--frame-id', 'lidar', '--child-frame-id', 'laser'],
    )

    # RPLidar S3 (physically inverted - requires inverted=true)
    rplidar = Node(
        package='rplidar_ros',
        executable='rplidar_node',
        name='rplidar_node',
        output='screen',
        parameters=[{
            'serial_port': '/dev/ttyUSB0',
            'serial_baudrate': 1000000,
            'frame_id': 'laser',
            'inverted': True,  # CRITICAL: LiDAR is physically inverted
            'angle_compensate': True,
            'scan_mode': 'DenseBoost',
        }],
        remappings=[('scan', '/scan_raw')]
    )

    # Laser filter for raw scan
    laser_filter = Node(
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

    # RealSense launch path
    rs_launch = os.path.join(
        get_package_share_directory('realsense2_camera'),
        'launch', 'rs_launch.py'
    )

    # Front Camera (D455) - facing backward
    # publish_tf=false: URDF provides camera frames with correct positions
    front_cam = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(rs_launch),
        launch_arguments={
            'camera_name': 'camera',
            'camera_namespace': '',
            'serial_no': FRONT_SERIAL,
            'publish_tf': 'false',
            'enable_gyro': 'false',
            'enable_accel': 'false',
            'depth_module.profile': '424x240x15',
            'pointcloud.enable': 'true',
            'pointcloud.stream_filter': '2',
        }.items(),
    )

    # Left Camera (D455) - facing left (+90° yaw in URDF)
    # publish_tf=false: URDF provides camera frames with correct positions
    left_cam = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(rs_launch),
        launch_arguments={
            'camera_name': 'mapping_camera',
            'camera_namespace': '',
            'serial_no': LEFT_SERIAL,
            'publish_tf': 'false',
            'enable_gyro': 'false',
            'enable_accel': 'false',
            'enable_color': 'false',
            'depth_module.profile': '424x240x15',
            'pointcloud.enable': 'true',
            'pointcloud.stream_filter': '2',
        }.items(),
    )

    # Right Camera (D435i) - facing right (-90° yaw in URDF)
    # publish_tf=false: URDF provides camera frames with correct positions
    right_cam = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(rs_launch),
        launch_arguments={
            'camera_name': 'right_camera',
            'camera_namespace': '',
            'serial_no': RIGHT_SERIAL,
            'publish_tf': 'false',
            'enable_gyro': 'false',
            'enable_accel': 'false',
            'enable_color': 'false',
            'depth_module.profile': '424x240x15',
            'pointcloud.enable': 'true',
            'pointcloud.stream_filter': '2',
        }.items(),
    )

    # Depth to LaserScan converters (for visualization only - fusion uses point clouds)
    # Output in camera optical frames (TF transforms to laser frame for visualization)
    depth_to_scan_front = Node(
        package='depthimage_to_laserscan',
        executable='depthimage_to_laserscan_node',
        name='depth_to_scan_front',
        remappings=[
            ('depth', '/camera/depth/image_rect_raw'),
            ('depth_camera_info', '/camera/depth/camera_info'),
            ('scan', '/camera/scan'),
        ],
        parameters=[{
            'scan_height': 100,  # Larger to capture vertical obstacles
            'range_min': 0.30,
            'range_max': 5.0,
            'output_frame_id': 'camera_depth_optical_frame',
        }]
    )

    depth_to_scan_left = Node(
        package='depthimage_to_laserscan',
        executable='depthimage_to_laserscan_node',
        name='depth_to_scan_left',
        remappings=[
            ('depth', '/mapping_camera/depth/image_rect_raw'),
            ('depth_camera_info', '/mapping_camera/depth/camera_info'),
            ('scan', '/mapping_camera/scan'),
        ],
        parameters=[{
            'scan_height': 100,
            'range_min': 0.30,
            'range_max': 5.0,
            'output_frame_id': 'mapping_camera_depth_optical_frame',
        }]
    )

    depth_to_scan_right = Node(
        package='depthimage_to_laserscan',
        executable='depthimage_to_laserscan_node',
        name='depth_to_scan_right',
        remappings=[
            ('depth', '/right_camera/depth/image_rect_raw'),
            ('depth_camera_info', '/right_camera/depth/camera_info'),
            ('scan', '/right_camera/scan'),
        ],
        parameters=[{
            'scan_height': 100,
            'range_min': 0.30,
            'range_max': 5.0,
            'output_frame_id': 'right_camera_depth_optical_frame',
        }]
    )

    # Simple Fusion Node (3 cameras + LiDAR) - No time sync required
    # Uses independent subscriptions - fuses on each LiDAR scan
    fusion_node = TimerAction(
        period=6.0,
        actions=[
            Node(
                package='wheelchair_localization',
                executable='multi_camera_scan_fusion_simple',
                name='multi_camera_scan_fusion',
                output='screen',
                parameters=[{
                    # Input/Output topics
                    'scan_topic': '/scan_filtered',
                    'output_topic': '/scan_fused',
                    'laser_frame': 'laser',

                    # Height filtering
                    'min_height': 0.10,
                    'max_height': 1.80,

                    # Front camera (D455)
                    'front_camera.enabled': True,
                    'front_camera.topic': '/camera/depth/color/points',
                    'front_camera.frame': 'camera_depth_optical_frame',
                    'front_camera.min_depth': 0.30,
                    'front_camera.max_depth': 5.0,

                    # Left camera (D455)
                    'left_camera.enabled': True,
                    'left_camera.topic': '/mapping_camera/depth/color/points',
                    'left_camera.frame': 'mapping_camera_depth_optical_frame',
                    'left_camera.min_depth': 0.30,
                    'left_camera.max_depth': 5.0,

                    # Right camera (D435i)
                    'right_camera.enabled': True,
                    'right_camera.topic': '/right_camera/depth/color/points',
                    'right_camera.frame': 'right_camera_depth_optical_frame',
                    'right_camera.min_depth': 0.30,
                    'right_camera.max_depth': 5.0,

                    'verbose': True,
                }],
            ),
        ]
    )

    # RViz
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config],
    )

    return LaunchDescription([
        robot_state_pub,
        joint_state_pub,
        static_tf_lidar_to_laser,
        rplidar,
        laser_filter,
        front_cam,
        left_cam,
        right_cam,
        depth_to_scan_front,
        depth_to_scan_left,
        depth_to_scan_right,
        fusion_node,
        rviz,
    ])
