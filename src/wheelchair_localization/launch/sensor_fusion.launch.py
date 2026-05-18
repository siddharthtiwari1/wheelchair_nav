import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    pkg_dir = get_package_share_directory('wheelchair_localization')

    return LaunchDescription([
        # Parameters
        DeclareLaunchArgument('use_sim_time', default_value='false', 
                              description='Use simulation time'),
        DeclareLaunchArgument('verbose', default_value='false',
                              description='Enable verbose logging'),

        # Camera-LiDAR Fusion Node
        Node(
            package='wheelchair_localization',
            executable='camera_lidar_fusion',
            name='camera_lidar_fusion',
            output='screen',
            parameters=[{
                'use_sim_time': LaunchConfiguration('use_sim_time'),
                'verbose': LaunchConfiguration('verbose'),
                'enable_motion_compensation': True,
                'enable_footprint_filter': True,
                'sync_timeout': 0.5,
                
                # Default Camera Topics (Adjust as needed)
                'front_camera.topic': '/camera/depth/color/points',
                'left_camera.topic': '/mapping_camera/depth/color/points',
                'right_camera.topic': '/right_camera/depth/color/points',
            }]
        )
    ])
