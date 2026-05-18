#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, SetParameter
from ament_index_python.packages import get_package_share_directory
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    """
    Simple, clean wheelchair launch that uses intern's existing localization work.
    Perfect for daily use and testing.
    """

    # Launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time')
    is_sim = LaunchConfiguration('is_sim')
    port = LaunchConfiguration('port')
    use_camera = LaunchConfiguration('use_camera')

    declare_args = [
        DeclareLaunchArgument(
            'use_sim_time', default_value='false',
            description='Use simulation time'
        ),
        DeclareLaunchArgument(
            'is_sim', default_value='false',
            description='Whether running in simulation'
        ),
        DeclareLaunchArgument(
            'port', default_value='/dev/ttyACM0',
            description='Serial port for Arduino'
        ),
        DeclareLaunchArgument(
            'use_camera', default_value='true',
            description='Enable RealSense camera'
        )
    ]

    # ==================== CORE SYSTEM ====================

    # Your professional hardware interface
    hardware_interface = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("wheelchair_firmware"),
                "launch",
                "ros2_control_hardware_interface.launch.py"
            )
        ),
        launch_arguments={
            'is_sim': is_sim,
            'port': port,
            'use_sim_time': use_sim_time
        }.items(),
        condition=UnlessCondition(is_sim)
    )

    # Intern's proven localization system (EKF + Custom Kalman)
    localization_system = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("wheelchair_localization"),
                "launch",
                "localization.launch.py"
            )
        ),
        launch_arguments={
            'use_sim_time': use_sim_time
        }.items()
    )

    # ==================== OPTIONAL CAMERA ====================

    # RealSense camera (for vision, mapping, etc.)
    realsense_camera = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(get_package_share_directory('realsense2_camera'), 'launch'),
            '/rs_launch.py'
        ]),
        launch_arguments={
            'camera_namespace': '',
            'enable_gyro': 'true',
            'enable_accel': 'true',
            'unite_imu_method': '2',
            'align_depth.enable': 'true',
            'enable_sync': 'true',
            'rgb_camera.profile': '640x360x30',
            'use_sim_time': use_sim_time
        }.items(),
        condition=IfCondition(use_camera)
    )

    # Camera IMU processing (separate from localization IMU)
    camera_imu_filter = Node(
        package='imu_filter_madgwick',
        executable='imu_filter_madgwick_node',
        name='camera_imu_filter',
        output='screen',
        parameters=[{
            'use_mag': False,
            'world_frame': 'enu',
            'publish_tf': False,
            'use_sim_time': use_sim_time
        }],
        remappings=[
            ('imu/data_raw', '/camera/imu'),
            ('imu/data', '/camera/imu/filtered')
        ],
        condition=IfCondition(use_camera)
    )

    return LaunchDescription([
        # Arguments
        *declare_args,

        # Core system
        hardware_interface,
        localization_system,

        # Optional camera
        realsense_camera,
        camera_imu_filter
    ])