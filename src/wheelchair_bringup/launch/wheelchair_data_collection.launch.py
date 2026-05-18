#!/usr/bin/env python3

"""
WHEELCHAIR DATA COLLECTION — RGB + Velocity + Trajectory
=========================================================
Lightweight launch for collecting timestamped image-velocity pairs.

Launches:
    - ros2_control + DiffDriveController (motor interface + wheel odom)
    - IMU pipeline (calibrator → bias corrector → madgwick → republisher)
    - Robust EKF + ZUPT (odom → /odometry/filtered)
    - Logitech C270 USB camera (→ /logitech/image_raw)
    - RGB+Velocity recorder (saves timestamped JPG + CSV)

NO lidar, NO SLAM, NO Nav2, NO RealSense cameras — minimal for data collection.
RViz shows camera feed + odometry trail for visual validation.

Usage:
    # Default: 10 fps, auto-timestamped output, RViz enabled
    ros2 launch wheelchair_bringup wheelchair_data_collection.launch.py

    # Custom FPS and output directory
    ros2 launch wheelchair_bringup wheelchair_data_collection.launch.py \
        save_fps:=15.0 output_dir:=/home/sidd/wheelchair_nav/data/hallway_01

    # No RViz (headless)
    ros2 launch wheelchair_bringup wheelchair_data_collection.launch.py rviz:=false

    # With teleop (separate terminal):
    ros2 run scripts twist_stamped_teleop
"""

import os
import subprocess

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    LogInfo,
    OpaqueFunction,
    TimerAction,
)
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    wheelchair_description_dir = get_package_share_directory('wheelchair_description')
    wc_control_dir = get_package_share_directory('wc_control')

    # ========================================================================
    # LAUNCH ARGUMENTS
    # ========================================================================

    declare_is_sim = DeclareLaunchArgument(
        'is_sim', default_value='false',
        description='True for simulation, false for real hardware.')

    declare_port = DeclareLaunchArgument(
        'port', default_value='/dev/ttyACM0',
        description='Serial port for wheelchair hardware interface.')

    declare_sudo_password = DeclareLaunchArgument(
        'sudo_password', default_value='12345',
        description='Sudo password for USB permissions.')

    declare_video_device = DeclareLaunchArgument(
        'video_device', default_value='/dev/video4',
        description='Logitech C270 video device path.')

    declare_save_fps = DeclareLaunchArgument(
        'save_fps', default_value='10.0',
        description='Frames per second to save (throttles image saving).')

    declare_output_dir = DeclareLaunchArgument(
        'output_dir', default_value='',
        description='Output directory (auto-generated if empty).')

    declare_image_quality = DeclareLaunchArgument(
        'image_quality', default_value='90',
        description='JPEG quality (1-100).')

    declare_rviz = DeclareLaunchArgument(
        'rviz', default_value='true',
        description='Launch RViz with camera feed + odom trail.')

    declare_use_improved_ekf = DeclareLaunchArgument(
        'use_improved_ekf', default_value='false',
        description='Use improved EKF with ZUPT + heading constraints (vs robust_ekf_zupt).')

    is_sim = LaunchConfiguration('is_sim')

    # ========================================================================
    # USB PERMISSIONS
    # ========================================================================

    def _grant_usb_permissions(context, *_args, **_kwargs):
        wheelchair_port = LaunchConfiguration('port').perform(context)
        password = LaunchConfiguration('sudo_password').perform(context)
        for port in [wheelchair_port]:
            if os.path.exists(port):
                try:
                    subprocess.run(
                        ['sudo', '-S', 'chmod', '666', port],
                        input=f'{password}\n', text=True, check=True)
                    print(f'  Updated permissions for {port}')
                except subprocess.CalledProcessError as exc:
                    print(f'  Failed to set permissions for {port}: {exc}')
        return []

    permission_setup = OpaqueFunction(
        function=_grant_usb_permissions,
        condition=UnlessCondition(is_sim))

    # ========================================================================
    # MOTION STACK (ros2_control + DiffDriveController)
    # ========================================================================

    unified_wheelchair_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(wc_control_dir, 'launch', 'unified_wheelchair.launch.py')
        ]),
        launch_arguments={
            'is_sim': LaunchConfiguration('is_sim'),
            'port': LaunchConfiguration('port'),
        }.items(),
    )

    # ========================================================================
    # IMU PIPELINE (same as slam_mapping — needed for EKF)
    # ========================================================================
    # /camera/imu → calibrator → bias_corrector → madgwick → republisher → /imu
    # NOTE: Front RealSense IS needed for IMU. Launches depth-disabled for bandwidth.

    # Front D455 — IMU only (no depth, no color)
    rs_launch_path = os.path.join(
        get_package_share_directory('realsense2_camera'),
        'launch', 'rs_launch.py')

    front_camera_imu = TimerAction(
        period=3.0,
        actions=[
            LogInfo(msg='[CAMERA] Starting front D455 for IMU only...'),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(rs_launch_path),
                launch_arguments={
                    'camera_name': 'camera',
                    'camera_namespace': '',
                    'serial_no': "'337122300107'",
                    'publish_tf': 'false',
                    'enable_gyro': 'true',
                    'enable_accel': 'true',
                    'unite_imu_method': '2',
                    'enable_color': 'false',
                    'enable_depth': 'false',
                    'enable_infra1': 'false',
                    'enable_infra2': 'false',
                }.items(),
            ),
        ],
        condition=UnlessCondition(is_sim))

    imu_startup_calibrator = TimerAction(
        period=5.0,
        actions=[
            Node(
                package='wc_control',
                executable='imu_startup_calibrator.py',
                name='imu_startup_calibrator',
                output='screen',
                parameters=[{
                    'input_topic': '/camera/imu',
                    'bias_topic': '/imu/calibrated_bias',
                    'calibration_duration': 3.0,
                    'min_samples': 100,
                    'default_gyro_x_bias': -0.004302,
                    'default_gyro_y_bias': 0.000787,
                    'default_gyro_z_bias': 0.000948,
                }],
            ),
        ],
        condition=UnlessCondition(is_sim))

    imu_bias_corrector = TimerAction(
        period=5.0,
        actions=[
            Node(
                package='wc_control',
                executable='imu_bias_corrector.py',
                name='imu_bias_corrector',
                output='screen',
                parameters=[{
                    'input_topic': '/camera/imu',
                    'output_topic': '/camera/imu_corrected',
                    'calibrated_bias_topic': '/imu/calibrated_bias',
                    'gyro_x_bias': -0.004302,
                    'gyro_y_bias': 0.000787,
                    'gyro_z_bias': 0.000948,
                }],
            ),
        ],
        condition=UnlessCondition(is_sim))

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
                }],
                remappings=[('imu/data_raw', '/camera/imu_corrected')],
            ),
        ],
        condition=UnlessCondition(is_sim))

    imu_republisher = TimerAction(
        period=7.0,
        actions=[
            Node(
                package='wc_control',
                executable='imu_wheelchair_republisher.py',
                name='imu_wheelchair_republisher',
                output='screen',
                parameters=[{
                    'input_topic': '/imu/data',
                    'output_topic': '/imu',
                    'output_frame': 'imu',
                    'zero_on_start': True,
                    'orientation_quaternion': [-0.5, 0.5, -0.5, 0.5],
                    'vector_quaternion': [-0.5, 0.5, -0.5, 0.5],
                }],
            ),
        ],
        condition=UnlessCondition(is_sim))

    # ========================================================================
    # STATIC TRANSFORMS + JOINT STATE PUBLISHER
    # ========================================================================

    static_tf_base_to_imu = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['--x', '0', '--y', '0', '--z', '0',
                   '--qx', '0', '--qy', '0', '--qz', '0', '--qw', '1',
                   '--frame-id', 'base_link', '--child-frame-id', 'imu'])

    robot_description_for_jsp = Command([
        'xacro ',
        os.path.join(wheelchair_description_dir, 'urdf',
                     'wheelchair_description.urdf.xacro'),
        ' is_sim:=false port:=/dev/ttyACM0'])

    joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        parameters=[{
            'robot_description': ParameterValue(
                robot_description_for_jsp, value_type=str),
        }],
        condition=UnlessCondition(is_sim))

    # ========================================================================
    # EKF + ZUPT (produces /odometry/filtered)
    # ========================================================================

    use_improved_ekf = LaunchConfiguration('use_improved_ekf')

    # Original EKF (when use_improved_ekf:=false)
    robust_ekf_zupt_node = TimerAction(
        period=10.0,
        actions=[
            LogInfo(msg='[EKF] Starting robust_ekf_zupt (original)...'),
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
                }],
            ),
        ],
        condition=UnlessCondition(PythonExpression([
            "'", use_improved_ekf, "' == 'true'"
        ])))

    # Improved EKF (when use_improved_ekf:=true)
    improved_ekf_node = TimerAction(
        period=10.0,
        actions=[
            LogInfo(msg='[EKF] Starting improved_ekf (ZUPT + heading constraints)...'),
            Node(
                package='wheelchair_zupt',
                executable='improved_ekf_node',
                name='improved_ekf',
                output='screen',
                parameters=[{
                    'imu_topic': '/imu',
                    'odom_topic': '/wc_control/odom',
                    'odom_frame': 'odom',
                    'base_frame': 'base_link',
                    'publish_tf': True,
                    'enable_zupt': True,
                    'enable_bias_calibration': True,
                    'enable_heading_constraint': True,
                    'enable_straight_omega_constraint': True,
                    'enable_gyro_health_monitor': True,
                }],
            ),
        ],
        condition=IfCondition(PythonExpression([
            "'", use_improved_ekf, "' == 'true'"
        ])))

    # ========================================================================
    # LOGITECH C270 USB CAMERA
    # ========================================================================

    logitech_camera = TimerAction(
        period=2.0,
        actions=[
            LogInfo(msg='[CAMERA] Starting Logitech C270 USB camera...'),
            Node(
                package='v4l2_camera',
                executable='v4l2_camera_node',
                name='logitech_camera',
                namespace='logitech',
                output='screen',
                parameters=[{
                    'video_device': LaunchConfiguration('video_device'),
                    'image_size': [640, 480],
                    'time_per_frame': [1, 15],  # 15 fps
                    'camera_frame_id': 'logitech_frame',
                }],
            ),
        ])

    # ========================================================================
    # RGB + VELOCITY RECORDER
    # ========================================================================

    rgb_velocity_recorder = TimerAction(
        period=15.0,  # Wait for EKF + camera to be ready
        actions=[
            LogInfo(msg='=' * 60),
            LogInfo(msg='  DATA COLLECTION: RGB + Velocity recorder starting'),
            LogInfo(msg='=' * 60),
            Node(
                package='scripts',
                executable='rgb_velocity_recorder',
                name='rgb_velocity_recorder',
                output='screen',
                parameters=[{
                    'output_dir': LaunchConfiguration('output_dir'),
                    'save_fps': LaunchConfiguration('save_fps'),
                    'image_quality': LaunchConfiguration('image_quality'),
                    'image_topic': '/logitech/image_raw',
                    'odom_topic': '/odometry/filtered',
                    'cmd_vel_topic': '/cmd_vel',
                }],
            ),
        ])

    # ========================================================================
    # RVIZ — camera feed + odometry trail for validation
    # ========================================================================

    rviz_config = os.path.join(
        wheelchair_description_dir, 'rviz', 'data_collection.rviz')

    rviz_node = TimerAction(
        period=16.0,
        actions=[
            LogInfo(msg='[RVIZ] Starting RViz (camera + odom trail)...'),
            Node(
                package='rviz2',
                executable='rviz2',
                name='rviz2',
                output='screen',
                arguments=['-d', rviz_config],
            ),
        ],
        condition=IfCondition(LaunchConfiguration('rviz')))

    # ========================================================================
    # READY MESSAGE
    # ========================================================================

    ready_message = TimerAction(
        period=18.0,
        actions=[
            LogInfo(msg='=' * 60),
            LogInfo(msg='  WHEELCHAIR DATA COLLECTION READY'),
            LogInfo(msg='=' * 60),
            LogInfo(msg='  Controller: DiffDrive active'),
            LogInfo(msg='  Camera:     Logitech C270 @ /logitech/image_raw'),
            LogInfo(msg='  Odom:       EKF+ZUPT @ /odometry/filtered'),
            LogInfo(msg='  Recorder:   Saving timestamped RGB + velocity pairs'),
            LogInfo(msg=''),
            LogInfo(msg='  Drive with: ros2 run scripts twist_stamped_teleop'),
            LogInfo(msg='  Stop with:  Ctrl+C (saves metadata.json)'),
            LogInfo(msg='=' * 60),
        ])

    # ========================================================================
    # LAUNCH DESCRIPTION
    # ========================================================================

    return LaunchDescription([
        # Arguments
        declare_is_sim,
        declare_port,
        declare_sudo_password,
        declare_video_device,
        declare_save_fps,
        declare_output_dir,
        declare_image_quality,
        declare_rviz,
        declare_use_improved_ekf,

        # USB permissions
        permission_setup,

        # Motion stack (ros2_control + DiffDriveController)
        unified_wheelchair_launch,

        # Front RealSense for IMU only
        front_camera_imu,

        # IMU pipeline
        imu_startup_calibrator,
        imu_bias_corrector,
        imu_filter,
        imu_republisher,

        # TF
        static_tf_base_to_imu,
        joint_state_publisher,

        # EKF + ZUPT (one or the other, selected by use_improved_ekf arg)
        robust_ekf_zupt_node,
        improved_ekf_node,

        # Logitech camera
        logitech_camera,

        # Data recorder
        rgb_velocity_recorder,

        # RViz (camera feed + odom trail)
        rviz_node,

        # Ready
        ready_message,
    ])
