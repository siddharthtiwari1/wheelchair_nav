#!/usr/bin/env python3
"""
Wheelchair Visualization Launch
Displays robot URDF with all 3 cameras in RViz for setup verification.

Usage:
    ros2 launch wheelchair_description display.launch.py

With cameras:
    ros2 launch wheelchair_description display.launch.py launch_cameras:=true
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg_description = get_package_share_directory('wheelchair_description')

    # Paths
    urdf_path = os.path.join(pkg_description, 'urdf', 'wheelchair_description.urdf.xacro')
    rviz_config = os.path.join(pkg_description, 'rviz', 'fusion_navigation.rviz')

    # Launch arguments
    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time', default_value='false',
        description='Use simulation time'
    )

    declare_rviz_config = DeclareLaunchArgument(
        'rviz_config', default_value=rviz_config,
        description='RViz configuration file'
    )

    declare_launch_cameras = DeclareLaunchArgument(
        'launch_cameras', default_value='false',
        description='Launch RealSense cameras'
    )

    # Robot description from xacro
    robot_description = ParameterValue(
        Command(['xacro ', urdf_path, ' is_sim:=false is_ignition:=false']),
        value_type=str
    )

    # Robot state publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_description,
            'use_sim_time': LaunchConfiguration('use_sim_time'),
        }]
    )

    # Joint state publisher GUI (for wheel visualization)
    joint_state_publisher = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        name='joint_state_publisher_gui',
        output='screen',
    )

    # RViz
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', LaunchConfiguration('rviz_config')],
    )

    # Optionally launch cameras
    camera_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(
                get_package_share_directory('wc_control'),
                'launch',
                'multi_camera.launch.py'
            )
        ]),
        condition=IfCondition(LaunchConfiguration('launch_cameras')),
    )

    return LaunchDescription([
        declare_use_sim_time,
        declare_rviz_config,
        declare_launch_cameras,
        robot_state_publisher,
        joint_state_publisher,
        rviz,
        camera_launch,
    ])
