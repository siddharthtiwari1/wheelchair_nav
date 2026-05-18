#!/usr/bin/env python3
"""3-Camera DA3 Mono Depth + Scan Fusion Test.

Runs DA3 Metric Large on all 3 cameras (single model, round-robin),
fuses with LiDAR via scan_fusion_v9, and benchmarks against stereo.

Pipeline:
  Front D455 RGB  ─┐
  Left D455 RGB   ─┼─→ da3_multi_depth_node ─→ 3x PointCloud2
  Right D435i RGB ─┘    (one model, ~126ms cycle)

  /scan_filtered (LiDAR 10Hz) ────┐
  3x DA3 PointCloud2 ────────────┼─→ scan_fusion_v9 ─→ /scan_mono_fused

Timeline:
  t=0s   URDF + TF + RPLidar + laser_filter
  t=3s   Front D455 (depth 424x240x6 + RGB 1280x720x6 + pointcloud)
  t=6s   Left D455 (depth 424x240x6 + RGB 640x480x6)
  t=10s  Right D435i (depth 424x240x6 + RGB 640x480x6)
  t=15s  DA3 multi-camera node (single model, 3 cameras)
  t=20s  3x stereo pc2ls (for comparison)
  t=22s  scan_fusion_v9 (DA3 pointclouds)
  t=25s  Benchmark
  t=28s  RViz
  t=32s  Ready

Usage:
    ros2 launch wheelchair_mono_depth mono_depth_3cam_test.launch.py
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    TimerAction,
    LogInfo,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


# ============================================================================
# CONSTANTS
# ============================================================================
FRONT_SERIAL = "'337122300107'"   # Front D455
LEFT_SERIAL = "'146222253403'"    # Left D455
RIGHT_SERIAL = "'207522077542'"   # Right D435i
LIDAR_SERIAL_PORT = '/dev/ttyUSB0'

# pc2ls params for stereo comparison scans
PC2LS_PARAMS = {
    'target_frame': 'laser',
    'transform_tolerance': 0.1,
    'min_height': 0.10,
    'max_height': 1.80,
    'angle_min': -3.14159,
    'angle_max': 3.14159,
    'angle_increment': 0.00872665,  # ~0.5deg, 720 bins
    'scan_time': 0.1667,
    'range_min': 0.30,
    'range_max': 5.0,
    'use_inf': True,
}


def generate_launch_description():
    description_dir = get_package_share_directory('wheelchair_description')
    localization_dir = get_package_share_directory('wheelchair_localization')
    rs_launch_path = os.path.join(
        get_package_share_directory('realsense2_camera'),
        'launch', 'rs_launch.py')

    laser_filter_config = os.path.join(
        localization_dir, 'config', 'laser_filter_robust.yaml')
    rviz_config = os.path.join(
        description_dir, 'rviz', 'mono_depth_3cam_test.rviz')
    xacro_path = os.path.join(
        description_dir, 'urdf', 'wheelchair_description.urdf.xacro')

    declare_use_rviz = DeclareLaunchArgument(
        'use_rviz', default_value='true')

    # ==================================================================
    # URDF + TF
    # ==================================================================
    robot_description = Command([
        'xacro ', xacro_path, ' is_sim:=false port:=/dev/ttyACM0'])

    robot_state_pub = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': ParameterValue(
                robot_description, value_type=str),
            'use_sim_time': False,
        }],
    )

    joint_state_pub = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        parameters=[{
            'use_sim_time': False,
            'robot_description': ParameterValue(
                robot_description, value_type=str),
        }],
    )

    static_tf_base_imu = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=[
            '--x', '0', '--y', '0', '--z', '0',
            '--qx', '0', '--qy', '0', '--qz', '0', '--qw', '1',
            '--frame-id', 'base_link', '--child-frame-id', 'imu',
        ],
    )

    # ==================================================================
    # RPLidar S3
    # ==================================================================
    rplidar_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(
                get_package_share_directory('rplidar_ros'),
                'launch', 'rplidar_s3_launch.py',
            )
        ]),
        launch_arguments={'inverted': 'true'}.items(),
    )

    # Laser filter chain: /scan -> /scan_filtered
    laser_filter = Node(
        package='laser_filters',
        executable='scan_to_scan_filter_chain',
        name='laser_filter',
        output='screen',
        parameters=[laser_filter_config],
        remappings=[
            ('scan', '/scan'),
            ('scan_filtered', '/scan_filtered'),
        ],
    )

    # ==================================================================
    # t=3s: Front D455 (depth + RGB + stereo pointcloud)
    # ==================================================================
    front_camera = TimerAction(
        period=3.0,
        actions=[
            LogInfo(msg='[CAMERA] Starting front D455...'),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(rs_launch_path),
                launch_arguments={
                    'camera_name': 'camera',
                    'camera_namespace': '',
                    'serial_no': FRONT_SERIAL,
                    'publish_tf': 'true',
                    'enable_gyro': 'false',
                    'enable_accel': 'false',
                    'enable_color': 'true',
                    'depth_module.profile': '424x240x6',
                    'rgb_camera.color_profile': '1280x720x6',
                    'pointcloud.enable': 'true',
                    'pointcloud.stream_filter': '0',
                    'pointcloud.stream_index_filter': '0',
                }.items(),
            ),
        ],
    )

    # ==================================================================
    # t=6s: Left D455 (depth + RGB, staggered for USB bandwidth)
    # ==================================================================
    left_camera = TimerAction(
        period=6.0,
        actions=[
            LogInfo(msg='[CAMERA] Starting left D455...'),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(rs_launch_path),
                launch_arguments={
                    'camera_name': 'mapping_camera',
                    'camera_namespace': '',
                    'serial_no': LEFT_SERIAL,
                    'publish_tf': 'true',
                    'enable_gyro': 'false',
                    'enable_accel': 'false',
                    'enable_color': 'true',
                    'depth_module.profile': '424x240x6',
                    'rgb_camera.color_profile': '640x480x6',
                    'pointcloud.enable': 'true',
                    'pointcloud.stream_filter': '0',
                    'pointcloud.stream_index_filter': '0',
                }.items(),
            ),
        ],
    )

    # ==================================================================
    # t=10s: Right D435i (depth + RGB, staggered)
    # ==================================================================
    right_camera = TimerAction(
        period=10.0,
        actions=[
            LogInfo(msg='[CAMERA] Starting right D435i...'),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(rs_launch_path),
                launch_arguments={
                    'camera_name': 'right_camera',
                    'camera_namespace': '',
                    'serial_no': RIGHT_SERIAL,
                    'publish_tf': 'true',
                    'enable_gyro': 'false',
                    'enable_accel': 'false',
                    'enable_color': 'true',
                    'depth_module.profile': '424x240x6',
                    'rgb_camera.color_profile': '640x480x6',
                    'pointcloud.enable': 'true',
                    'pointcloud.stream_filter': '0',
                    'pointcloud.stream_index_filter': '0',
                }.items(),
            ),
        ],
    )

    # ==================================================================
    # t=15s: DA3 Multi-Camera Node (single model, 3 cameras)
    # ==================================================================
    da3_multi = TimerAction(
        period=15.0,
        actions=[
            LogInfo(msg='[DA3] Starting multi-camera DA3 (3 cameras)...'),
            Node(
                package='wheelchair_mono_depth',
                executable='da3_multi_depth_node',
                name='da3_multi_depth',
                output='screen',
                parameters=[{
                    'model_name': 'da3-metric-large',
                    'max_depth': 8.0,
                    'depth_correction': 1.0,  # raw — correction TBD from data
                    'compile_model': False,
                    'temporal_alpha': 0.0,
                    'camera_names': ['front', 'left', 'right'],
                    # Front D455: 1280x720 RGB -> 504px process
                    'cameras.front.rgb_topic': '/camera/color/image_raw',
                    'cameras.front.info_topic': '/camera/color/camera_info',
                    'cameras.front.output_prefix': '/camera/mono_da3',
                    'cameras.front.process_width': 504,
                    # Left D455: 640x480 RGB -> 364px process
                    'cameras.left.rgb_topic':
                        '/mapping_camera/color/image_raw',
                    'cameras.left.info_topic':
                        '/mapping_camera/color/camera_info',
                    'cameras.left.output_prefix': '/mapping_camera/mono_da3',
                    'cameras.left.process_width': 364,
                    # Right D435i: 640x480 RGB -> 364px process
                    'cameras.right.rgb_topic':
                        '/right_camera/color/image_raw',
                    'cameras.right.info_topic':
                        '/right_camera/color/camera_info',
                    'cameras.right.output_prefix': '/right_camera/mono_da3',
                    'cameras.right.process_width': 364,
                }],
            ),
        ],
    )

    # ==================================================================
    # t=20s: 3x Stereo pc2ls (for comparison scans)
    # ==================================================================
    stereo_pc2ls_nodes = TimerAction(
        period=20.0,
        actions=[
            LogInfo(msg='[PC2LS] Starting 3x stereo scan conversions...'),
            # Front stereo -> /scan_stereo_front
            Node(
                package='pointcloud_to_laserscan',
                executable='pointcloud_to_laserscan_node',
                name='pc2ls_stereo_front',
                output='screen',
                parameters=[PC2LS_PARAMS],
                remappings=[
                    ('cloud_in', '/camera/depth/color/points'),
                    ('scan', '/scan_stereo_front'),
                ],
            ),
            # Left stereo -> /scan_stereo_left
            Node(
                package='pointcloud_to_laserscan',
                executable='pointcloud_to_laserscan_node',
                name='pc2ls_stereo_left',
                output='screen',
                parameters=[PC2LS_PARAMS],
                remappings=[
                    ('cloud_in', '/mapping_camera/depth/color/points'),
                    ('scan', '/scan_stereo_left'),
                ],
            ),
            # Right stereo -> /scan_stereo_right
            Node(
                package='pointcloud_to_laserscan',
                executable='pointcloud_to_laserscan_node',
                name='pc2ls_stereo_right',
                output='screen',
                parameters=[PC2LS_PARAMS],
                remappings=[
                    ('cloud_in', '/right_camera/depth/color/points'),
                    ('scan', '/scan_stereo_right'),
                ],
            ),
        ],
    )

    # ==================================================================
    # t=22s: Scan Fusion v9 with DA3 mono depth pointclouds
    # ==================================================================
    scan_fusion = TimerAction(
        period=22.0,
        actions=[
            LogInfo(msg='[FUSION] Starting scan_fusion_v9 with DA3 clouds...'),
            Node(
                package='wheelchair_localization',
                executable='scan_fusion_v9',
                name='scan_fusion_mono',
                output='screen',
                parameters=[{
                    'scan_topic': '/scan_filtered',
                    'output_topic': '/scan_mono_fused',
                    'laser_frame': 'laser',
                    'min_height': 0.10,
                    'max_height': 1.80,
                    'max_camera_age_ms': 300.0,
                    'camera_warmup_sec': 3.0,
                    'downsample_stride': 4,
                    'min_camera_points_per_bin': 2,
                    'enable_footprint': True,
                    'publish_lidar_only': True,
                    # Front DA3 pointcloud
                    'front_camera.enabled': True,
                    'front_camera.topic': '/camera/mono_da3/points',
                    'front_camera.frame': 'camera_color_optical_frame',
                    'front_camera.min_depth': 0.40,
                    'front_camera.max_depth': 6.0,
                    # Left DA3 pointcloud
                    'left_camera.enabled': True,
                    'left_camera.topic': '/mapping_camera/mono_da3/points',
                    'left_camera.frame':
                        'mapping_camera_color_optical_frame',
                    'left_camera.min_depth': 0.40,
                    'left_camera.max_depth': 6.0,
                    # Right DA3 pointcloud
                    'right_camera.enabled': True,
                    'right_camera.topic': '/right_camera/mono_da3/points',
                    'right_camera.frame':
                        'right_camera_color_optical_frame',
                    'right_camera.min_depth': 0.40,
                    'right_camera.max_depth': 6.0,
                }],
            ),
        ],
    )

    # ==================================================================
    # t=25s: Benchmark (LiDAR vs DA3 front scan)
    # ==================================================================
    benchmark = TimerAction(
        period=25.0,
        actions=[
            LogInfo(msg='[BENCHMARK] Starting depth comparison...'),
            Node(
                package='wheelchair_mono_depth',
                executable='depth_scan_benchmark_node',
                name='depth_scan_benchmark',
                output='screen',
                parameters=[{
                    'stereo_scan_topic': '/scan_filtered',
                    'mono_scan_topic': '/scan_mono_fused',
                    'output_dir':
                        '/home/sidd/wheelchair_nav/eval_output/'
                        'mono_3cam_comparison',
                    'sync_slop': 0.5,
                    'fp_threshold_m': 0.5,
                    'summary_interval': 25,
                }],
            ),
        ],
    )

    # ==================================================================
    # t=28s: RViz
    # ==================================================================
    rviz = TimerAction(
        period=28.0,
        actions=[
            LogInfo(msg='[RVIZ] Launching...'),
            Node(
                package='rviz2',
                executable='rviz2',
                name='rviz2',
                output='screen',
                arguments=['-d', rviz_config],
            ),
        ],
    )

    # ==================================================================
    # t=32s: Ready message
    # ==================================================================
    ready_msg = TimerAction(
        period=32.0,
        actions=[
            LogInfo(msg='=' * 60),
            LogInfo(msg='  3-CAMERA DA3 MONO DEPTH + SCAN FUSION'),
            LogInfo(msg='  RED:    /scan_filtered     — LiDAR 360deg'),
            LogInfo(msg='  GREEN:  /scan_stereo_*     — Stereo depth (3 cams)'),
            LogInfo(msg='  BLUE:   /scan_mono_fused   — DA3 + LiDAR fused'),
            LogInfo(msg='  '),
            LogInfo(msg='  DA3 pointclouds:'),
            LogInfo(msg='    /camera/mono_da3/points         (front 504px)'),
            LogInfo(msg='    /mapping_camera/mono_da3/points  (left 364px)'),
            LogInfo(msg='    /right_camera/mono_da3/points    (right 364px)'),
            LogInfo(msg='=' * 60),
        ],
    )

    return LaunchDescription([
        declare_use_rviz,
        robot_state_pub,
        joint_state_pub,
        static_tf_base_imu,
        rplidar_launch,
        laser_filter,
        front_camera,
        left_camera,
        right_camera,
        da3_multi,
        stereo_pc2ls_nodes,
        scan_fusion,
        benchmark,
        rviz,
        ready_msg,
    ])
