#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    
    # Launch arguments
    serial_port_arg = DeclareLaunchArgument(
        'serial_port',
        default_value='/dev/ttyACM1',
        description='Serial port for Arduino connection'
    )
    
    # Hardware Interface (Arduino communication)
    hardware_interface = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(
                get_package_share_directory("wheelchair_firmware"),
                "launch",
                "hardware_interface.launch.py"
            )
        ]),
        launch_arguments={
            'serial_port': LaunchConfiguration('serial_port'),
        }.items(),
    )
    
    # Wheelchair Controller (without differential drive controller)
    controller = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(
                get_package_share_directory("wc_control"),
                "launch",
                "wheelchair_controller_no_odom.py"
            )
        ])
    )

    return LaunchDescription([
        # Launch arguments
        serial_port_arg,
        
        # Core components
        hardware_interface,
        controller,
    ])