#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare
import xacro


def generate_launch_description():
    # Get package directories
    wheelchair_description_pkg = FindPackageShare('wheelchair_description')
    wheelchair_firmware_pkg = FindPackageShare('wheelchair_firmware')

    # Launch configuration variables
    use_sim_time = LaunchConfiguration('use_sim_time')
    is_sim = LaunchConfiguration('is_sim')
    is_ignition = LaunchConfiguration('is_ignition')
    port = LaunchConfiguration('port')

    # Declare launch arguments
    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation (Gazebo) clock if true'
    )

    declare_is_sim_cmd = DeclareLaunchArgument(
        'is_sim',
        default_value='false',
        description='Whether to run in simulation mode'
    )

    declare_is_ignition_cmd = DeclareLaunchArgument(
        'is_ignition',
        default_value='true',
        description='Whether to use Ignition Gazebo (true) or classic Gazebo (false)'
    )

    declare_port_cmd = DeclareLaunchArgument(
        'port',
        default_value='/dev/ttyACM0',
        description='Serial port for Arduino communication'
    )

    # Process URDF with xacro
    robot_description_content = Command([
        'xacro ',
        os.path.join(get_package_share_directory('wheelchair_description'), 'urdf', 'wheelchair_description.urdf.xacro'),
        ' is_sim:=', is_sim,
        ' is_ignition:=', is_ignition,
        ' port:=', port
    ])

    robot_description = {'robot_description': ParameterValue(robot_description_content, value_type=str)}

    # Robot State Publisher
    robot_state_publisher_cmd = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[robot_description, {'use_sim_time': use_sim_time}]
    )

    # Controller Manager (only for real robot)
    controller_manager = Node(
        package='controller_manager',
        executable='ros2_control_node',
        parameters=[
            robot_description,
            os.path.join(get_package_share_directory('wc_control'), 'config', 'wc_control.yaml'),
            {'use_sim_time': use_sim_time}
        ],
        output='screen',
        condition=UnlessCondition(is_sim)
    )

    # Joint State Broadcaster (for publishing joint states to /joint_states)
    joint_state_broadcaster_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster'],
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    # WC Control Differential Drive Controller
    diff_drive_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['wc_control'],
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    # IMU Sensor Broadcaster removed - IMU data already available at /imu/data topic

    # Arduino communication is handled by the hardware interface
    # No separate arduino monitor needed

    # Create launch description and populate
    ld = LaunchDescription()

    # Add launch arguments
    ld.add_action(declare_use_sim_time_cmd)
    ld.add_action(declare_is_sim_cmd)
    ld.add_action(declare_is_ignition_cmd)
    ld.add_action(declare_port_cmd)

    # Add nodes
    ld.add_action(robot_state_publisher_cmd)
    ld.add_action(controller_manager)
    ld.add_action(joint_state_broadcaster_spawner)
    ld.add_action(diff_drive_spawner)
    # ld.add_action(imu_sensor_broadcaster_spawner)  # Removed - no IMU hardware interface
    # ld.add_action(arduino_monitor)  # Removed - handled by hardware interface

    return ld


def get_package_share_directory(package_name):
    from ament_index_python.packages import get_package_share_directory as get_pkg_share_dir
    return get_pkg_share_dir(package_name)