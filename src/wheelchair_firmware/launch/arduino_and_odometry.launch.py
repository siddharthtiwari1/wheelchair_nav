#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='wheelchair_firmware',
            executable='arduino_receiver_updated.py',
            name='arduino_receiver',
            output='screen',
            parameters=[
                {'serial_port': '/dev/ttyACM0'},
                {'baud_rate': 115200},
                {'wheel_base': 0.565},  # FIXED 2026-01-08: Standardized
                {'wheel_radius': 0.1524},
                {'publish_odom': True}
            ]
        ),
        Node(
            package='wheelchair_firmware',
            executable='odometry_calculator.py',
            name='odometry_calculator',
            output='screen'
        )
    ])