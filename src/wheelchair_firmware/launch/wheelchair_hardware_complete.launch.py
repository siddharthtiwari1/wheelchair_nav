#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Get package directories
    wheelchair_description_pkg = FindPackageShare('wheelchair_description')
    wc_control_pkg = FindPackageShare('wc_control')

    # Launch configuration variables
    use_sim_time = LaunchConfiguration('use_sim_time')
    port = LaunchConfiguration('port')

    # Declare launch arguments
    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation clock if true'
    )

    declare_port_cmd = DeclareLaunchArgument(
        'port',
        default_value='/dev/ttyACM1',
        description='Serial port for Arduino communication'
    )

    # Process URDF with xacro - REAL HARDWARE MODE
    robot_description_content = Command([
        'xacro ',
        os.path.join(get_package_share_directory('wheelchair_description'), 'urdf', 'wheelchair_description.urdf.xacro'),
        ' is_sim:=false',  # REAL HARDWARE
        ' is_ignition:=false',
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

    # Controller Manager with Hardware Interface
    controller_manager = Node(
        package='controller_manager',
        executable='ros2_control_node',
        parameters=[
            robot_description,
            os.path.join(get_package_share_directory('wc_control'), 'config', 'wc_control.yaml'),
            {'use_sim_time': use_sim_time}
        ],
        output='screen'
    )

    # Wait for controller manager to start, then spawn controllers
    joint_state_broadcaster_spawner = TimerAction(
        period=3.0,  # Wait for hardware interface to initialize
        actions=[
            Node(
                package='controller_manager',
                executable='spawner',
                arguments=['joint_state_broadcaster'],
                parameters=[{'use_sim_time': use_sim_time}],
                output='screen'
            )
        ]
    )

    # WC Control Differential Drive Controller - Main controller for movement and odometry
    wc_control_spawner = TimerAction(
        period=4.0,  # Wait for joint state broadcaster
        actions=[
            Node(
                package='controller_manager',
                executable='spawner',
                arguments=['wc_control'],
                parameters=[{'use_sim_time': use_sim_time}],
                output='screen'
            )
        ]
    )

    # Twist to TwistStamped converter (if needed for compatibility)
    twist_to_stamped = TimerAction(
        period=6.0,  # Wait for controllers to start
        actions=[
            Node(
                package='scripts',
                executable='twist_stamped_teleop',
                name='twist_to_twist_stamped',
                output='screen',
                parameters=[{'use_sim_time': use_sim_time}],
                remappings=[
                    ('cmd_vel_in', '/cmd_vel'),
                    ('cmd_vel_out', '/wc_control/cmd_vel'),
                ],
                condition=IfCondition('false')  # Disable by default, enable if needed
            )
        ]
    )

    # Optional: Teleop keyboard for testing
    teleop_keyboard = TimerAction(
        period=7.0,  # Wait a bit more
        actions=[
            Node(
                package='teleop_twist_keyboard',
                executable='teleop_twist_keyboard',
                name='teleop_twist_keyboard',
                output='screen',
                prefix='xterm -e',
                parameters=[{'use_sim_time': use_sim_time}],
                remappings=[
                    ('cmd_vel', '/wc_control/cmd_vel'),  # Direct to controller
                ],
                condition=IfCondition('false')  # Disable by default, enable for testing
            )
        ]
    )

    # Arduino Status Monitor (separate from hardware interface for debugging)
    arduino_status_monitor = TimerAction(
        period=2.0,
        actions=[
            Node(
                package='wheelchair_firmware',
                executable='arduino_receiver_updated.py',
                name='arduino_status_monitor',
                output='screen',
                parameters=[{
                    'port': '/dev/ttyACM0',  # Use different port for monitoring
                    'use_sim_time': use_sim_time,
                    'publish_odom': False  # Don't conflict with hardware interface
                }],
                condition=IfCondition('false')  # Disable by default to avoid port conflicts
            )
        ]
    )

    # Create launch description and populate
    ld = LaunchDescription()

    # Add launch arguments
    ld.add_action(declare_use_sim_time_cmd)
    ld.add_action(declare_port_cmd)

    # Add nodes in startup order
    ld.add_action(robot_state_publisher_cmd)
    ld.add_action(controller_manager)
    ld.add_action(joint_state_broadcaster_spawner)
    ld.add_action(wc_control_spawner)
    ld.add_action(twist_to_stamped)
    ld.add_action(teleop_keyboard)
    ld.add_action(arduino_status_monitor)

    return ld


def get_package_share_directory(package_name):
    """Helper function to get package share directory"""
    from ament_index_python.packages import get_package_share_directory as get_pkg_share_dir
    return get_pkg_share_dir(package_name)