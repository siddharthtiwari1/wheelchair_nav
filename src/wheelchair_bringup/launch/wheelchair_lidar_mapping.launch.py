#!/usr/bin/env python3

"""
LIDAR-ONLY SLAM MAPPING (no cameras, no scan fusion)
=====================================================
Ablation SLAM mapping launch for PRISM paper.

- NO cameras launched (saves USB bandwidth + CPU)
- /scan -> laser_filter -> /scan_filtered -> scan_lidar -> /scan_lidar
- SLAM Toolbox uses /scan_lidar (footprint-filtered, clean)
- EKF/ZUPT odometry from wheel encoders + IMU (front camera IMU still needed)

Usage:
    ros2 launch wheelchair_bringup wheelchair_lidar_mapping.launch.py
"""

import os
import subprocess

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
    TimerAction,
    LogInfo,
)
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    # Package directories
    wheelchair_description_dir = get_package_share_directory('wheelchair_description')
    wheelchair_localization_dir = get_package_share_directory('wheelchair_localization')
    wc_control_dir = get_package_share_directory('wc_control')

    ws_root = '/home/sidd/wheelchair_nav'

    default_model_path = os.path.join(
        wheelchair_description_dir, 'urdf', 'wheelchair_description.urdf.xacro'
    )
    default_rviz_config = os.path.join(
        wheelchair_description_dir, 'rviz', 'slam_mapping.rviz'
    )
    default_slam_config = os.path.join(
        wheelchair_localization_dir, 'config', 'slam_toolbox_motion_compensated_v2.yaml'
    )
    default_laser_filter_config = os.path.join(
        wheelchair_localization_dir, 'config', 'laser_filter_robust.yaml'
    )

    # Front camera serial (needed for IMU only — no depth)
    FRONT_SERIAL = "'337122300107'"
    rs_launch_path = os.path.join(
        get_package_share_directory('realsense2_camera'),
        'launch', 'rs_launch.py'
    )

    is_sim_value = 'false'
    port_value = '/dev/ttyACM0'
    lidar_port_value = '/dev/ttyUSB0'

    # ========================================================================
    # LAUNCH ARGUMENTS
    # ========================================================================

    declare_is_sim = DeclareLaunchArgument('is_sim', default_value=is_sim_value)
    declare_port = DeclareLaunchArgument('port', default_value=port_value)
    declare_model = DeclareLaunchArgument('model', default_value=default_model_path)
    declare_use_sim_time = DeclareLaunchArgument('use_sim_time', default_value='false')
    declare_world = DeclareLaunchArgument('world', default_value='empty.sdf')
    declare_lidar_port = DeclareLaunchArgument('lidar_port', default_value=lidar_port_value)
    declare_sudo_password = DeclareLaunchArgument('sudo_password', default_value='12345')
    declare_rviz = DeclareLaunchArgument('rviz', default_value='true')
    declare_rviz_config = DeclareLaunchArgument('rviz_config', default_value=default_rviz_config)
    declare_slam_config = DeclareLaunchArgument('slam_config', default_value=default_slam_config)

    is_sim = LaunchConfiguration('is_sim')

    # ========================================================================
    # USB PERMISSIONS
    # ========================================================================

    def _grant_usb_permissions(context, *_args, **_kwargs):
        lidar_port = LaunchConfiguration('lidar_port').perform(context)
        wheelchair_port = LaunchConfiguration('port').perform(context)
        password = LaunchConfiguration('sudo_password').perform(context)
        ports = [wheelchair_port]
        if os.path.exists(lidar_port):
            ports.append(lidar_port)
        for port in ports:
            try:
                subprocess.run(
                    ['sudo', '-S', 'chmod', '666', port],
                    input=f'{password}\n', text=True, check=True, capture_output=True,
                )
            except subprocess.CalledProcessError:
                pass
        return []

    permission_setup = OpaqueFunction(
        function=_grant_usb_permissions,
        condition=UnlessCondition(is_sim)
    )

    # ========================================================================
    # HARDWARE (ros2_control + motors)
    # ========================================================================

    unified_wheelchair_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(wc_control_dir, 'launch', 'unified_wheelchair.launch.py')
        ),
        launch_arguments={
            'is_sim': LaunchConfiguration('is_sim'),
            'port': LaunchConfiguration('port'),
            'world': LaunchConfiguration('world'),
        }.items(),
    )

    # ========================================================================
    # FRONT CAMERA — IMU ONLY (no depth, no left/right cameras)
    # ========================================================================

    front_camera_imu_only = TimerAction(
        period=3.0,
        actions=[
            LogInfo(msg='[CAMERA] Starting FRONT D455 (IMU only — no depth for lidar mapping)...'),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(rs_launch_path),
                launch_arguments={
                    'camera_name': 'camera',
                    'camera_namespace': '',
                    'serial_no': FRONT_SERIAL,
                    'publish_tf': 'false',
                    'enable_gyro': 'true',
                    'enable_accel': 'true',
                    'unite_imu_method': '2',
                    'enable_color': 'false',
                    'enable_depth': 'false',
                    'pointcloud.enable': 'false',
                }.items(),
            ),
        ],
        condition=UnlessCondition(is_sim)
    )

    # ========================================================================
    # IMU PIPELINE (bias correction + madgwick + republish)
    # ========================================================================

    imu_startup_calibrator = TimerAction(
        period=5.0,
        actions=[
            Node(
                package='wc_control',
                executable='imu_startup_calibrator.py',
                name='imu_startup_calibrator',
                output='screen',
                parameters=[{
                    'use_sim_time': LaunchConfiguration('use_sim_time'),
                    'input_topic': '/camera/imu',
                    'bias_topic': '/imu/calibrated_bias',
                    'status_topic': '/imu/calibration_status',
                    'calibration_duration': 3.0,
                    'min_samples': 100,
                    'max_motion_threshold': 0.05,
                    'default_gyro_x_bias': -0.004302,
                    'default_gyro_y_bias': 0.000787,
                    'default_gyro_z_bias': 0.000948,
                }],
            ),
        ],
        condition=UnlessCondition(is_sim)
    )

    imu_bias_corrector = TimerAction(
        period=5.0,
        actions=[
            Node(
                package='wc_control',
                executable='imu_bias_corrector.py',
                name='imu_bias_corrector',
                output='screen',
                parameters=[{
                    'use_sim_time': LaunchConfiguration('use_sim_time'),
                    'input_topic': '/camera/imu',
                    'output_topic': '/camera/imu_corrected',
                    'calibrated_bias_topic': '/imu/calibrated_bias',
                    'gyro_x_bias': -0.004302,
                    'gyro_y_bias': 0.000787,
                    'gyro_z_bias': 0.000948,
                }],
            ),
        ],
        condition=UnlessCondition(is_sim)
    )

    imu_filter = TimerAction(
        period=6.0,
        actions=[
            Node(
                package='imu_filter_madgwick',
                executable='imu_filter_madgwick_node',
                name='imu_filter_madgwick',
                output='screen',
                parameters=[{
                    'use_mag': False,
                    'world_frame': 'enu',
                    'publish_tf': False,
                    'gain': 0.041,
                    'use_sim_time': LaunchConfiguration('use_sim_time'),
                }],
                remappings=[('imu/data_raw', '/camera/imu_corrected')],
            ),
        ],
        condition=UnlessCondition(is_sim)
    )

    imu_wheelchair_republisher = TimerAction(
        period=7.0,
        actions=[
            Node(
                package='wc_control',
                executable='imu_wheelchair_republisher.py',
                name='imu_wheelchair_republisher',
                output='screen',
                parameters=[{
                    'use_sim_time': LaunchConfiguration('use_sim_time'),
                    'input_topic': '/imu/data',
                    'output_topic': '/imu',
                    'output_frame': 'imu',
                    'zero_on_start': True,
                    'orientation_quaternion': [-0.5, 0.5, -0.5, 0.5],
                    'vector_quaternion': [-0.5, 0.5, -0.5, 0.5],
                }],
            ),
        ],
        condition=UnlessCondition(is_sim)
    )

    # ========================================================================
    # RPLIDAR + LASER FILTER + SCAN_LIDAR
    # ========================================================================

    rplidar_s3_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('rplidar_ros'),
                'launch', 'rplidar_s3_launch.py'
            )
        ),
        launch_arguments={'inverted': 'true'}.items(),
        condition=UnlessCondition(is_sim)
    )

    laser_filter_node = Node(
        package='laser_filters',
        executable='scan_to_scan_filter_chain',
        name='laser_filter',
        output='screen',
        parameters=[default_laser_filter_config],
        remappings=[
            ('scan', '/scan'),
            ('scan_filtered', '/scan_filtered'),
        ],
        condition=UnlessCondition(is_sim)
    )

    # Footprint-filtered lidar: /scan_filtered -> /scan_lidar
    scan_lidar_node = Node(
        package='wheelchair_localization',
        executable='scan_lidar',
        name='scan_lidar',
        output='screen',
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
    )

    # ========================================================================
    # STATIC TF + JOINT STATE PUBLISHER
    # ========================================================================

    static_transform_imu = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['--x', '0', '--y', '0', '--z', '0',
                   '--qx', '0', '--qy', '0', '--qz', '0', '--qw', '1',
                   '--frame-id', 'base_link', '--child-frame-id', 'imu']
    )

    joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        parameters=[{
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'robot_description': ParameterValue(Command([
                'xacro ',
                os.path.join(wheelchair_description_dir, 'urdf', 'wheelchair_description.urdf.xacro'),
                ' is_sim:=false port:=/dev/ttyACM0'
            ]), value_type=str),
        }],
        condition=UnlessCondition(is_sim)
    )

    # ========================================================================
    # EKF + ZUPT ODOMETRY
    # ========================================================================

    robust_ekf_zupt_node = TimerAction(
        period=10.0,
        actions=[
            LogInfo(msg='[EKF+ZUPT] Starting Robust EKF + ZUPT Sensor Fusion...'),
            Node(
                package='wheelchair_zupt',
                executable='robust_ekf_zupt_node',
                name='robust_ekf_zupt',
                output='screen',
                parameters=[{
                    'imu_topic': '/imu',
                    'odom_topic': '/wc_control/odom',
                    'odom_frame': 'odom',
                    'base_frame': 'base_link',
                    'publish_tf': True,
                    'stationary_v_thresh': 0.005,
                    'stationary_omega_thresh': 0.005,
                    'accel_gravity_tolerance': 0.3,
                    'accel_xy_thresh': 0.15,
                    'min_stationary_samples': 15,
                    'hysteresis_samples': 8,
                    'initial_gyro_bias': 0.0,
                    'bias_alpha_stationary': 0.02,
                    'bias_alpha_moving': 0.0005,
                    'bias_min_samples': 20,
                    'sigma_v': 0.05,
                    'sigma_omega': 0.02,
                    'sigma_bias': 0.0001,
                    'sigma_enc_v': 0.02,
                    'sigma_enc_omega': 0.015,
                    'sigma_gyro': 0.005,
                    'sigma_zupt_v': 0.001,
                    'sigma_zupt_omega': 0.001,
                    'sigma_imu_yaw': 0.03,
                    'use_imu_orientation': True,
                    'orientation_update_rate': 2.0,
                    'mahalanobis_threshold': 5.0,
                    'initial_x': 0.0,
                    'initial_y': 0.0,
                    'initial_theta': 0.0,
                }]
            )
        ],
        condition=UnlessCondition(is_sim)
    )

    # ========================================================================
    # SLAM TOOLBOX — LIDAR ONLY with /scan_lidar
    # ========================================================================

    slam_lifecycle_nodes = ['map_saver_server', 'slam_toolbox']

    slam_toolbox_node = TimerAction(
        period=15.0,
        actions=[
            LogInfo(msg='[SLAM] Starting SLAM Toolbox with /scan_lidar (footprint-filtered)...'),
            Node(
                package='slam_toolbox',
                executable='sync_slam_toolbox_node',
                name='slam_toolbox',
                output='screen',
                parameters=[
                    LaunchConfiguration('slam_config'),
                    {'use_sim_time': is_sim,
                     'scan_topic': '/scan_lidar',
                     'use_lifecycle_manager': True}
                ],
            ),
        ],
        condition=UnlessCondition(is_sim)
    )

    map_saver_server = TimerAction(
        period=17.0,
        actions=[
            Node(
                package='nav2_map_server',
                executable='map_saver_server',
                name='map_saver_server',
                output='screen',
                parameters=[
                    {'save_map_timeout': 5.0},
                    {'use_sim_time': is_sim},
                    {'free_thresh_default': 0.196},
                    {'occupied_thresh_default': 0.65},
                ],
            ),
        ],
        condition=UnlessCondition(is_sim)
    )

    slam_lifecycle_manager = TimerAction(
        period=25.0,
        actions=[
            Node(
                package='nav2_lifecycle_manager',
                executable='lifecycle_manager',
                name='lifecycle_manager_slam',
                output='screen',
                parameters=[
                    {'node_names': slam_lifecycle_nodes},
                    {'use_sim_time': is_sim},
                    {'autostart': True},
                    {'bond_timeout': 15.0},
                ],
            ),
        ],
        condition=UnlessCondition(is_sim)
    )

    # Session manager (auto-save map on Ctrl+C)
    slam_session_manager = TimerAction(
        period=30.0,
        actions=[
            LogInfo(msg='[SESSION] Starting SLAM session manager (auto-save on Ctrl+C)...'),
            Node(
                package='scripts',
                executable='slam_session_manager',
                name='slam_session_manager',
                output='screen',
            ),
        ],
        condition=UnlessCondition(is_sim)
    )

    # ========================================================================
    # DATA LOGGER
    # ========================================================================

    topic_data_logger = Node(
        package='scripts',
        executable='topic_data_logger',
        name='topic_data_logger',
        output='screen',
        parameters=[{
            'imu_topic': '/imu',
            'raw_odom_topic': '/wc_control/odom',
            'filtered_odom_topic': '/odometry/filtered',
            'log_frequency_hz': 10.0,
            'file_prefix': 'lidar_mapping_log'
        }]
    )

    # ========================================================================
    # RVIZ
    # ========================================================================

    rviz_node = TimerAction(
        period=30.0,
        actions=[
            LogInfo(msg='[RVIZ] Starting RViz (LIDAR-ONLY mapping)...'),
            Node(
                package='rviz2',
                executable='rviz2',
                name='rviz2',
                output='screen',
                arguments=['-d', LaunchConfiguration('rviz_config')],
                parameters=[{'use_sim_time': is_sim}],
            )
        ],
        condition=IfCondition(LaunchConfiguration('rviz')),
    )

    ready_message = TimerAction(
        period=35.0,
        actions=[
            LogInfo(msg='=' * 60),
            LogInfo(msg='  LIDAR-ONLY SLAM MAPPING READY'),
            LogInfo(msg='=' * 60),
            LogInfo(msg='  Odometry: Robust EKF + ZUPT (Encoder + IMU)'),
            LogInfo(msg='  Sensors: RPLidar S3 only (NO cameras for mapping)'),
            LogInfo(msg='  SLAM Input: /scan_lidar (footprint-filtered)'),
            LogInfo(msg='  NO cameras, NO scan_fusion, NO STVL'),
            LogInfo(msg='  Use slam_toolbox panel in RViz to save map'),
            LogInfo(msg='=' * 60),
        ]
    )

    # ========================================================================
    # LAUNCH DESCRIPTION
    # ========================================================================

    return LaunchDescription([
        declare_is_sim,
        declare_port,
        declare_model,
        declare_use_sim_time,
        declare_world,
        declare_lidar_port,
        declare_sudo_password,
        declare_rviz,
        declare_rviz_config,
        declare_slam_config,

        # USB permissions
        permission_setup,

        # Hardware
        unified_wheelchair_launch,

        # Front camera (IMU only)
        front_camera_imu_only,

        # IMU pipeline
        imu_startup_calibrator,
        imu_bias_corrector,
        imu_filter,
        imu_wheelchair_republisher,

        # LiDAR + filter + footprint
        rplidar_s3_launch,
        laser_filter_node,
        scan_lidar_node,

        # TF + joint states
        static_transform_imu,
        joint_state_publisher,

        # Odometry
        robust_ekf_zupt_node,

        # SLAM
        slam_toolbox_node,
        map_saver_server,
        slam_lifecycle_manager,
        slam_session_manager,

        # Logging + visualization
        topic_data_logger,
        rviz_node,
        ready_message,
    ])
