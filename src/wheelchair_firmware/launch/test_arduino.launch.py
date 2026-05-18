#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    
    # Declare launch arguments
    serial_port_arg = DeclareLaunchArgument(
        'serial_port',
        default_value='/dev/ttyACM1',
        description='Serial port for Arduino connection'
    )
    
    # Arduino Receiver Node (receives wheel velocities and publishes odometry)
    arduino_receiver_node = Node(
        package='wheelchair_firmware',
        executable='arduino_receiver_updated.py',
        name='arduino_receiver',
        output='screen',
        parameters=[{
            'serial_port': LaunchConfiguration('serial_port'),
            'baud_rate': 115200,
            'wheel_base': 0.565,  # FIXED 2026-01-08: Standardized
            'wheel_radius': 0.1524,
            'publish_odom': True,
            'odom_frame': 'odom',
            'base_frame': 'base_link',
            'reconnect_timeout': 5.0
        }]
    )
    
    return LaunchDescription([
        # Launch arguments
        serial_port_arg,
        
        # Node
        arduino_receiver_node,
    ])