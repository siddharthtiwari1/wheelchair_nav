#!/usr/bin/env python3
"""Launch 3 monocular depth inference nodes as drop-in replacement for RealSense depth.

Each node subscribes to an RGB camera topic and publishes depth images + point clouds
with the same topic names and formats as the RealSense cameras. This enables
scan_depth_fusion_node.py to work unchanged.

Usage:
    ros2 launch wheelchair_mono_depth mono_depth_inference.launch.py
    ros2 launch wheelchair_mono_depth mono_depth_inference.launch.py \
        model_path:=/path/to/model.engine use_tensorrt:=true
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_dir = get_package_share_directory('wheelchair_mono_depth')
    config_file = os.path.join(pkg_dir, 'config', 'inference.yaml')

    declare_model_path = DeclareLaunchArgument(
        'model_path',
        default_value='/home/sidd/wheelchair_nav/models/mono_depth_vits_fp16.engine',
        description='Path to model (.pth or .engine)',
    )
    declare_use_tensorrt = DeclareLaunchArgument(
        'use_tensorrt', default_value='true',
        description='Use TensorRT backend',
    )
    declare_encoder = DeclareLaunchArgument(
        'encoder', default_value='vits',
        description='ViT encoder size',
    )

    # Front camera inference
    front_node = Node(
        package='wheelchair_mono_depth',
        executable='mono_depth_node',
        name='mono_depth_front',
        output='screen',
        parameters=[
            config_file,
            {
                'model_path': LaunchConfiguration('model_path'),
                'use_tensorrt': LaunchConfiguration('use_tensorrt'),
                'encoder': LaunchConfiguration('encoder'),
                'camera_name': 'front',
                'rgb_topic': '/camera/color/image_raw',
                'output_depth_topic': '/camera/depth/image_rect_raw',
                'output_info_topic': '/camera/depth/camera_info',
                'publish_pointcloud': True,
                'pointcloud_topic': '/camera/depth/color/points',
            },
        ],
    )

    # Left camera inference (delayed to stagger GPU load)
    left_node = TimerAction(
        period=2.0,
        actions=[
            Node(
                package='wheelchair_mono_depth',
                executable='mono_depth_node',
                name='mono_depth_left',
                output='screen',
                parameters=[
                    config_file,
                    {
                        'model_path': LaunchConfiguration('model_path'),
                        'use_tensorrt': LaunchConfiguration('use_tensorrt'),
                        'encoder': LaunchConfiguration('encoder'),
                        'camera_name': 'left',
                        'rgb_topic': '/mapping_camera/color/image_raw',
                        'output_depth_topic': '/mapping_camera/depth/image_rect_raw',
                        'output_info_topic': '/mapping_camera/depth/camera_info',
                        'publish_pointcloud': True,
                        'pointcloud_topic': '/mapping_camera/depth/color/points',
                    },
                ],
            ),
        ],
    )

    # Right camera inference
    right_node = TimerAction(
        period=4.0,
        actions=[
            Node(
                package='wheelchair_mono_depth',
                executable='mono_depth_node',
                name='mono_depth_right',
                output='screen',
                parameters=[
                    config_file,
                    {
                        'model_path': LaunchConfiguration('model_path'),
                        'use_tensorrt': LaunchConfiguration('use_tensorrt'),
                        'encoder': LaunchConfiguration('encoder'),
                        'camera_name': 'right',
                        'rgb_topic': '/right_camera/color/image_raw',
                        'output_depth_topic': '/right_camera/depth/image_rect_raw',
                        'output_info_topic': '/right_camera/depth/camera_info',
                        'publish_pointcloud': True,
                        'pointcloud_topic': '/right_camera/depth/color/points',
                    },
                ],
            ),
        ],
    )

    return LaunchDescription([
        declare_model_path,
        declare_use_tensorrt,
        declare_encoder,
        LogInfo(msg='Launching monocular depth inference (3 cameras)...'),
        front_node,
        left_node,
        right_node,
    ])
