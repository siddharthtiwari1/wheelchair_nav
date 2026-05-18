#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    
    # Declare launch arguments
    serial_port_arg = DeclareLaunchArgument(
        'serial_port',
        default_value='/dev/ttyACM1',
        description='Serial port for Arduino connection'
    )
    
    baud_rate_arg = DeclareLaunchArgument(
        'baud_rate',
        default_value='115200',
        description='Baud rate for serial communication'
    )
    
    wheel_base_arg = DeclareLaunchArgument(
        'wheel_base',
        default_value='0.565',  # FIXED 2026-01-08: Standardized physical measurement
        description='Distance between wheels in meters'
    )
    
    wheel_radius_arg = DeclareLaunchArgument(
        'wheel_radius',
        default_value='0.1524',
        description='Wheel radius in meters (6 inches)'
    )
    
    publish_odom_arg = DeclareLaunchArgument(
        'publish_odom',
        default_value='true',
        description='Whether to publish odometry data'
    )
    
    # Arduino Transmitter Node (sends cmd_vel to Arduino)
    arduino_transmitter_node = Node(
        package='wheelchair_firmware',
        executable='arduino_transmitter.py',
        name='arduino_transmitter',
        output='screen',
        parameters=[{
            'serial_port': LaunchConfiguration('serial_port'),
            'baud_rate': LaunchConfiguration('baud_rate'),
            'wheel_base': LaunchConfiguration('wheel_base'),
            'wheel_radius': LaunchConfiguration('wheel_radius'),
        }],
        remappings=[
            ('cmd_vel', '/cmd_vel'),
            ('serial_transmitter', '/serial_transmitter')
        ]
    )
    
    # Arduino Receiver Node (receives wheel velocities and publishes odometry)
    arduino_receiver_node = Node(
        package='wheelchair_firmware',
        executable='arduino_receiver_updated.py',
        name='arduino_receiver',
        output='screen',
        parameters=[{
            'serial_port': LaunchConfiguration('serial_port'),
            'baud_rate': LaunchConfiguration('baud_rate'),
            'wheel_base': LaunchConfiguration('wheel_base'),
            'wheel_radius': LaunchConfiguration('wheel_radius'),
            'publish_odom': LaunchConfiguration('publish_odom'),
            'odom_frame': 'odom',
            'base_frame': 'base_link',
            'reconnect_timeout': 5.0
        }],
        remappings=[
            ('wheel_velocities', '/wheel_velocities'),
            ('arduino_odom', '/arduino_odom'),
            ('arduino_twist', '/arduino_twist'),
            ('odom', '/odom')
        ]
    )
    
    return LaunchDescription([
        # Launch arguments
        serial_port_arg,
        baud_rate_arg,
        wheel_base_arg,
        wheel_radius_arg,
        publish_odom_arg,
        
        # Nodes
        arduino_transmitter_node,
        arduino_receiver_node,
    ])