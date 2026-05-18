#!/usr/bin/env python3

"""
ALL-IN-ONE WHEELCHAIR SLAM MAPPING WITH 3-CAMERA FUSION
========================================================
Launches complete system for SLAM mapping (hardware + sensors + 3-camera fusion + SLAM + RViz)

Based on working camera_fusion_test.launch.py pattern.

Key Features:
    - 3 RealSense cameras (front D455, left D455, right D435i) with staggered startup
    - LiDAR + 3-camera scan fusion available (tightened params to reduce noise)
    - SLAM Toolbox defaults to lidar-only for clean maps
    - EKF sensor fusion (wheel odom + IMU)

Usage:
    # Default: lidar-only SLAM (clean, proven maps)
    ros2 launch wheelchair_bringup wheelchair_slam_mapping.launch.py

    # Experimental: fused SLAM (lidar + 3 cameras, tightened params)
    ros2 launch wheelchair_bringup wheelchair_slam_mapping.launch.py use_fused_slam:=true
"""

import os
import subprocess

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction, TimerAction, LogInfo
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression, Command
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    # Get package directories
    wheelchair_description_dir = get_package_share_directory('wheelchair_description')
    wheelchair_localization_dir = get_package_share_directory('wheelchair_localization')
    wc_control_dir = get_package_share_directory('wc_control')

    # Default paths
    default_model_path = os.path.join(
        wheelchair_description_dir,
        'urdf',
        'wheelchair_description.urdf.xacro',
    )
    # Default RViz config - SLAM mapping with 3-camera fusion
    default_rviz_config = os.path.join(
        wheelchair_description_dir,
        'rviz',
        'slam_mapping.rviz',
    )

    # Camera serials (from hardware - same as camera_fusion_test.launch.py)
    FRONT_SERIAL = "'337122300107'"   # Front D455
    LEFT_SERIAL = "'146222253403'"    # Left D455
    RIGHT_SERIAL = "'207522077542'"   # Right D435i

    # RealSense launch path
    rs_launch_path = os.path.join(
        get_package_share_directory('realsense2_camera'),
        'launch', 'rs_launch.py'
    )
    # SLAM configuration options:
    # - slam_toolbox_v14r7_hector_hybrid.yaml: HECTOR HYBRID (0.3m search, 2mm res, 0.75/0.75) [NEW!]
    # - slam_toolbox_v14_pro.yaml: Maximum quality Hector-style (3.4° + 2cm res, ~65% CPU)
    # - slam_toolbox_v14r6.yaml: Symmetric 0.75/0.75 (0.6m search, no leaks fix)
    # - slam_toolbox_v14r5.yaml: Maximum rotation trust (corners good, but scan leaks)
    # - slam_toolbox_v14r4.yaml: Stable odometry-led (60% both, but rotation still reorients)
    # - slam_toolbox_v14r3.yaml: Rotation-aware (poor results - position instability)
    # - slam_toolbox_v14r2.yaml: Geometry-first (sharp L-corners, 5mm precision, but rotation issues)
    # - slam_toolbox_v14r1.yaml: Optimized for excellent odometry (3.4°, 2cm, loop closure)
    # - slam_toolbox_v14.yaml: Balanced config (5° threshold, ~35% CPU, good results)
    # - slam_toolbox_v2.yaml: Legacy config (poor results, not recommended)
    default_slam_config = os.path.join(
        wheelchair_localization_dir,
        'config',
        'slam_toolbox_motion_compensated_v2.yaml',  # Jetson-proven correlation (0.35/0.005/0.03)
    )
    default_slam_config_fused = os.path.join(
        wheelchair_localization_dir,
        'config',
        'slam_toolbox_fused_v21.yaml',  # v21: proven v11 base (smear=0.03, range=8.0) + camera occupancy
    )
    # Hospital configs — 25m range, wider loop closure for spacious corridors
    hospital_slam_config = os.path.join(
        wheelchair_localization_dir,
        'config',
        'slam_toolbox_hospital_lidar_v4.yaml',  # 30m range (full corridor), 30m loop search
    )
    hospital_slam_config_fused = os.path.join(
        wheelchair_localization_dir,
        'config',
        'slam_toolbox_hospital_fused_v2.yaml',  # 15m range (rotation-safe), 30m loop search, camera occupancy
    )
    hospital_laser_filter_config = os.path.join(
        wheelchair_localization_dir,
        'config',
        'laser_filter_hospital_v2.yaml',  # 30m range filter (vs 12m in robust)
    )
    # Detect ROS distro for compatibility
    ros_distro = os.environ.get('ROS_DISTRO', 'jazzy')
    default_is_ignition = 'true' if ros_distro == 'humble' else 'false'

    # Hardcoded values for real robot
    is_sim_value = 'false'
    port_value = '/dev/ttyACM0'
    lidar_port_value = '/dev/ttyUSB0'
    enable_plotting_value = 'false'  # No plotting during mapping

    # ========================================================================
    # LAUNCH ARGUMENTS
    # ========================================================================

    declare_is_sim = DeclareLaunchArgument(
        'is_sim',
        default_value=is_sim_value,
        description='True for simulation, false for real hardware.',
    )
    declare_port = DeclareLaunchArgument(
        'port',
        default_value=port_value,
        description='Serial port for wheelchair hardware interface.',
    )
    declare_model = DeclareLaunchArgument(
        'model',
        default_value=default_model_path,
        description='Absolute path to the wheelchair URDF/xacro file.',
    )
    declare_is_ignition = DeclareLaunchArgument(
        'is_ignition',
        default_value=default_is_ignition,
        description='Set to true when running on Humble/ros_ign combination.',
    )
    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation time if true.',
    )
    declare_world = DeclareLaunchArgument(
        'world',
        default_value='empty.sdf',
        description='Gazebo world file (simulation only).',
    )
    declare_unite_imu = DeclareLaunchArgument(
        'unite_imu_method',
        default_value='2',
        description='RealSense IMU synchronization (0=None, 1=copy, 2=linear interpolation).',
    )
    declare_lidar_port = DeclareLaunchArgument(
        'lidar_port',
        default_value=lidar_port_value,
        description='Serial port for RPLidar (hardware only).',
    )
    declare_sudo_password = DeclareLaunchArgument(
        'sudo_password',
        default_value='12345',
        description='Sudo password for USB permissions (hardware only).',
    )
    declare_rviz = DeclareLaunchArgument(
        'rviz',
        default_value='true',
        description='Launch RViz for visualization.',
    )
    declare_rviz_config = DeclareLaunchArgument(
        'rviz_config',
        default_value=default_rviz_config,
        description='RViz configuration file.',
    )
    declare_enable_plotting = DeclareLaunchArgument(
        'enable_plotting',
        default_value=enable_plotting_value,
        description='Enable square path EKF plotting.',
    )
    declare_test_type = DeclareLaunchArgument(
        'test_type',
        default_value='square',
        description='Test type: square or lshape',
    )
    declare_slam_config = DeclareLaunchArgument(
        'slam_config',
        default_value=default_slam_config,
        description='SLAM Toolbox configuration file.',
    )
    declare_use_fused_slam = DeclareLaunchArgument(
        'use_fused_slam', default_value='true',
        description='Use fused scan for SLAM (true=/scan_fused, false=/scan_filtered). '
                    'Fused SLAM: map includes camera obstacles (tables/shelves).')
    declare_dataset_mode = DeclareLaunchArgument(
        'dataset_mode', default_value='false',
        description='Enable RGB+depth capture (turns on color stream + depth alignment).')
    declare_hospital_mode = DeclareLaunchArgument(
        'hospital_mode', default_value='false',
        description='Hospital mapping: 25m lidar range + wider loop closure for spacious corridors. '
                    'Uses laser_filter_hospital.yaml + slam_toolbox_hospital_*.yaml configs.')

    # Get launch configurations
    is_sim = LaunchConfiguration('is_sim')
    use_fused_slam = LaunchConfiguration('use_fused_slam')
    dataset_mode = LaunchConfiguration('dataset_mode')

    # ========================================================================
    # USB PERMISSIONS SETUP (only for real hardware)
    # ========================================================================

    def _grant_usb_permissions(context, *_args, **_kwargs):
        """Ensure serial devices are writable before nodes start."""
        lidar_port = LaunchConfiguration('lidar_port').perform(context)
        wheelchair_port = LaunchConfiguration('port').perform(context)
        password = LaunchConfiguration('sudo_password').perform(context)

        ports = [wheelchair_port]
        if os.path.exists(lidar_port):
            ports.append(lidar_port)
        else:
            print(f'ℹ Skipping permission change for {lidar_port}: device not present')

        for port in ports:
            try:
                subprocess.run(
                    ['sudo', '-S', 'chmod', '666', port],
                    input=f'{password}\n',
                    text=True,
                    check=True,
                )
                print(f'✓ Updated permissions for {port}')
            except subprocess.CalledProcessError as exc:
                print(f'⚠ Failed to set permissions for {port}: {exc}')
        return []

    permission_setup = OpaqueFunction(
        function=_grant_usb_permissions,
        condition=UnlessCondition(is_sim)
    )

    # Unified wheelchair hardware/control stack (ros2_control, Gazebo, teleop, sim helpers)
    unified_wheelchair_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(
                wc_control_dir,
                'launch',
                'unified_wheelchair.launch.py',
            )
        ]),
        launch_arguments={
            'is_sim': LaunchConfiguration('is_sim'),
            'port': LaunchConfiguration('port'),
            'world': LaunchConfiguration('world'),
        }.items(),
    )

    # ========================================================================
    # IMU PROCESSING PIPELINE (from wheelchair_sensors.launch.py)
    # ========================================================================
    # Since we're launching cameras directly for proper 3-camera fusion,
    # we add IMU processing here instead of using wheelchair_sensors_launch.
    # Pipeline: /camera/imu -> bias correction -> madgwick -> republish -> /imu

    # Step 0: Startup calibration - measures current gyro bias
    imu_startup_calibrator = TimerAction(
        period=5.0,  # After front camera starts
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

    # Step 1: Apply gyro bias correction
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

    # Step 2: Madgwick filter - fuses accel+gyro into orientation
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

    # Step 3: Transform to base_link frame
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

    # RPLidar S3
    rplidar_s3_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(
                get_package_share_directory('rplidar_ros'),
                'launch',
                'rplidar_s3_launch.py',
            )
        ]),
        launch_arguments={'inverted': 'true'}.items(),  # CRITICAL FIX: S3 requires inverted=true
        condition=UnlessCondition(is_sim)
    )

    # ========================================================================
    # LASER FILTER - REMOVES CLUTTER/NOISE BEFORE SLAM
    # ========================================================================
    # Pipeline: /scan (raw) -> laser_filter -> /scan_filtered -> SLAM
    # hospital_mode uses 25m range filter; default uses 12m

    def _make_laser_filter(context, *_args, **_kwargs):
        hosp = LaunchConfiguration('hospital_mode').perform(context) == 'true'
        sim = LaunchConfiguration('is_sim').perform(context) == 'true'
        if sim:
            return []
        config = hospital_laser_filter_config if hosp else os.path.join(
            wheelchair_localization_dir, 'config', 'laser_filter_robust.yaml')
        label = '30m hospital' if hosp else '12m default'
        return [
            LogInfo(msg=f'[LASER FILTER] Using {label} range filter'),
            Node(
                package='laser_filters',
                executable='scan_to_scan_filter_chain',
                name='laser_filter',
                output='screen',
                parameters=[config],
                remappings=[
                    ('scan', '/scan'),
                    ('scan_filtered', '/scan_filtered'),
                ],
            ),
        ]

    laser_filter_setup = OpaqueFunction(function=_make_laser_filter)

    # ========================================================================
    # 3-CAMERA SYSTEM - STAGGERED STARTUP FOR USB STABILITY
    # ========================================================================
    # Pattern from working camera_fusion_test.launch.py
    # All cameras at 424x240x6 for CPU efficiency
    # When dataset_mode=true, color + depth alignment enabled for data capture

    def _make_cameras(context, *args, **kwargs):
        """Generate camera launches; adds color+alignment when dataset_mode=true."""
        ds_mode = LaunchConfiguration('dataset_mode').perform(context) == 'true'
        sim = LaunchConfiguration('is_sim').perform(context) == 'true'
        if sim:
            return []

        # Camera args match wc_ws/wheelchair_slam_mapping.launch.py EXACTLY
        # Do NOT add extra params (align_depth, enable_infra) — RealSense
        # defaults work; overriding can break depth/IMU pipeline.
        front_args = {
            'camera_name': 'camera',
            'camera_namespace': '',
            'serial_no': FRONT_SERIAL,
            'publish_tf': 'false',
            'enable_gyro': 'true',
            'enable_accel': 'true',
            'unite_imu_method': '2',
            'enable_color': 'true' if ds_mode else 'false',
            'depth_module.profile': '424x240x6',
            'pointcloud.enable': 'true',
            'pointcloud.stream_filter': '0',
            'pointcloud.stream_index_filter': '0',
        }
        if ds_mode:
            front_args['rgb_camera.profile'] = '424x240x6'

        left_args = {
            'camera_name': 'mapping_camera',
            'camera_namespace': '',
            'serial_no': LEFT_SERIAL,
            'publish_tf': 'false',
            'enable_gyro': 'false',
            'enable_accel': 'false',
            'enable_color': 'true' if ds_mode else 'false',
            'depth_module.profile': '424x240x6',
            'pointcloud.enable': 'true',
            'pointcloud.stream_filter': '0',
            'pointcloud.stream_index_filter': '0',
        }
        if ds_mode:
            left_args['rgb_camera.profile'] = '424x240x6'

        right_args = {
            'camera_name': 'right_camera',
            'camera_namespace': '',
            'serial_no': RIGHT_SERIAL,
            'publish_tf': 'false',
            'enable_gyro': 'false',
            'enable_accel': 'false',
            'enable_color': 'true' if ds_mode else 'false',
            'depth_module.profile': '424x240x6',
            'pointcloud.enable': 'true',
            'pointcloud.stream_filter': '0',
            'pointcloud.stream_index_filter': '0',
        }
        if ds_mode:
            right_args['rgb_camera.profile'] = '424x240x6'

        mode_label = '+ RGB capture' if ds_mode else 'depth only'
        return [
            TimerAction(period=3.0, actions=[
                LogInfo(msg=f'[CAMERA] Starting FRONT D455 (424x240@6Hz + IMU {mode_label})...'),
                IncludeLaunchDescription(
                    PythonLaunchDescriptionSource(rs_launch_path),
                    launch_arguments=front_args.items(),
                ),
            ]),
            TimerAction(period=8.0, actions=[
                LogInfo(msg=f'[CAMERA] Starting LEFT D455 (424x240@6Hz {mode_label})...'),
                IncludeLaunchDescription(
                    PythonLaunchDescriptionSource(rs_launch_path),
                    launch_arguments=left_args.items(),
                ),
            ]),
            TimerAction(period=13.0, actions=[
                LogInfo(msg=f'[CAMERA] Starting RIGHT D435i (424x240@6Hz {mode_label})...'),
                IncludeLaunchDescription(
                    PythonLaunchDescriptionSource(rs_launch_path),
                    launch_arguments=right_args.items(),
                ),
            ]),
        ]

    camera_setup = OpaqueFunction(function=_make_cameras)

    # ========================================================================
    # SCAN FUSION NODE - LIDAR + 3 CAMERAS
    # ========================================================================
    # Fuses filtered 2D LiDAR with 3 depth camera point clouds for robust
    # height-aware SLAM that detects obstacles at ALL heights
    # Input: /scan_filtered (2D LiDAR) + 3x /*/depth/color/points (3D depth)
    # Output: /scan_fused (height-augmented 2D scan for SLAM)

    # USING scan_fusion_v9 — SIMPLE "CAMERA WINS IF CLOSER" FUSION.
    # v9 logic: for each angular bin, if camera distance < lidar distance, use camera.
    # No delta threshold, no rate limiting. Simpler than v7's MIN+delta+cap logic.
    # Both side cameras: 3.0m max depth. Front camera: 3.5m (better D455 baseline).
    scan_fusion_node = TimerAction(
        period=18.0,  # Wait for all 3 cameras to initialize
        actions=[
            LogInfo(msg='=' * 60),
            LogInfo(msg='[FUSION] Starting scan_fusion_v9 (camera wins if closer)'),
            LogInfo(msg='=' * 60),
            Node(
                package='wheelchair_localization',
                executable='scan_fusion_v9',
                name='scan_fusion',
                output='screen',
                parameters=[{
                    'scan_topic': '/scan_filtered',
                    'output_topic': '/scan_fused',
                    'laser_frame': 'laser',
                    # Height filter
                    'min_height': 0.10,
                    'max_height': 1.80,
                    # Staleness + warmup
                    'max_camera_age_ms': 500.0,
                    'camera_warmup_sec': 3.0,
                    'downsample_stride': 4,
                    'min_camera_points_per_bin': 2,
                    # Footprint filter on lidar
                    'enable_footprint': True,
                    'rear_crop_deg': 180.0,
                    # Front camera D455
                    'front_camera.enabled': True,
                    'front_camera.topic': '/camera/depth/color/points',
                    'front_camera.frame': 'camera_depth_optical_frame',
                    'front_camera.max_depth': 3.5,
                    'front_camera.min_depth': 0.30,
                    # Left camera D455 — 3.0m depth range
                    'left_camera.enabled': True,
                    'left_camera.topic': '/mapping_camera/depth/color/points',
                    'left_camera.frame': 'mapping_camera_depth_optical_frame',
                    'left_camera.max_depth': 3.0,
                    'left_camera.min_depth': 0.30,
                    # Right camera D435i — 3.0m depth range
                    'right_camera.enabled': True,
                    'right_camera.topic': '/right_camera/depth/color/points',
                    'right_camera.frame': 'right_camera_depth_optical_frame',
                    'right_camera.max_depth': 3.0,
                    'right_camera.min_depth': 0.30,
                }],
            ),
        ],
        condition=UnlessCondition(is_sim)
    )

    # ========================================================================
    # EKF LOCALIZATION (both simulation and hardware)
    # ========================================================================
    # NOTE: SLAM mapping is now in separate launch file: wheelchair_slam.launch.py

    static_transform_publisher = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=["--x", "0", "--y", "0", "--z", "0",
                   "--qx", "0", "--qy", "0", "--qz", "0", "--qw", "1",
                   "--frame-id", "base_link",
                   "--child-frame-id", "imu"]
    )

    # NOTE: laser frame now defined in URDF (wheelchair_description.urdf.xacro)
    # as a fixed joint under lidar with 180° Z rotation — no runtime TF publisher needed

    # Joint state publisher — publishes default (zero) wheel joint states so
    # robot_state_publisher can compute full TF tree even if joint_state_broadcaster
    # from ros2_control hasn't started yet (e.g., Arduino not connected).
    # Without this, RViz shows no robot model when hardware is unavailable.
    # NOTE: Pass robot_description directly via URDF xacro to avoid QoS mismatch
    # with robot_state_publisher's transient_local /robot_description topic.
    robot_description_for_jsp = Command([
        'xacro ',
        os.path.join(wheelchair_description_dir, 'urdf', 'wheelchair_description.urdf.xacro'),
        ' is_sim:=false port:=/dev/ttyACM0'
    ])
    joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        parameters=[{
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'robot_description': ParameterValue(robot_description_for_jsp, value_type=str),
        }],
        condition=UnlessCondition(is_sim)
    )

    # ========================================================================
    # ROBUST EKF + ZUPT SENSOR FUSION
    # ========================================================================
    # Complete sensor fusion with:
    # - EKF: Continuous probabilistic encoder + gyro fusion
    # - ZUPT: Zero Velocity Update when stationary (encoder=0 AND accel=gravity)
    # - Continuous gyro bias recalibration during stationary periods
    # - Adaptive gyro weighting based on motion state
    # - Outlier rejection for sensor failures
    # Publishes odom->base_link transform and /odometry/filtered
    robust_ekf_zupt_node = TimerAction(
        period=10.0,  # Wait for IMU pipeline to be ready (~7s)
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
                    # Stationary detection (multi-sensor)
                    'stationary_v_thresh': 0.005,       # Encoder velocity threshold
                    'stationary_omega_thresh': 0.005,  # Encoder omega threshold
                    'accel_gravity_tolerance': 0.3,    # Accel magnitude tolerance
                    'accel_xy_thresh': 0.15,           # Horizontal accel threshold
                    'min_stationary_samples': 15,      # Samples to confirm stationary
                    'hysteresis_samples': 8,           # Samples to exit stationary
                    # Gyro bias calibration
                    'initial_gyro_bias': 0.0,
                    'bias_alpha_stationary': 0.02,     # Fast learning when still
                    'bias_alpha_moving': 0.0005,       # Very slow when moving
                    'bias_min_samples': 20,            # Samples before publishing
                    # EKF process noise
                    'sigma_v': 0.05,
                    'sigma_omega': 0.02,
                    'sigma_bias': 0.0001,
                    # Encoder measurement noise
                    'sigma_enc_v': 0.02,
                    'sigma_enc_omega': 0.015,
                    # Gyro measurement noise
                    'sigma_gyro': 0.005,
                    # ZUPT noise (very low - high confidence)
                    'sigma_zupt_v': 0.001,
                    'sigma_zupt_omega': 0.001,
                    # IMU orientation (drift correction)
                    'sigma_imu_yaw': 0.03,
                    'use_imu_orientation': True,
                    'orientation_update_rate': 2.0,
                    # Outlier rejection
                    'mahalanobis_threshold': 5.0,
                    # Initial pose
                    'initial_x': 0.0,
                    'initial_y': 0.0,
                    'initial_theta': 0.0,
                }]
            )
        ],
        condition=UnlessCondition(is_sim)
    )

    # ========================================================================
    # PATH TESTING/PLOTTING (SQUARE or L-SHAPE)
    # ========================================================================

    # Square path plotter (default)
    square_path_plotter = TimerAction(
        period=7.0,  # Wait for EKF to start
        actions=[
            Node(
                package='scripts',
                executable='square_path_ekf_tester',
                name='square_path_ekf_tester',
                output='screen',
                parameters=[{
                    'use_sim_time': is_sim,
                }],
            )
        ],
        condition=IfCondition(PythonExpression([
            "'", LaunchConfiguration('enable_plotting'), "' == 'true' and '",
            LaunchConfiguration('test_type'), "' == 'square'"
        ]))
    )

    # L-shape path plotter (for testing 6m forward + 90° turn + 4m forward)
    lshape_path_plotter = TimerAction(
        period=7.0,  # Wait for EKF to start
        actions=[
            Node(
                package='scripts',
                executable='l_shape_odometry_test.py',
                name='l_shape_odometry_test',
                output='screen',
                parameters=[{
                    'use_sim_time': is_sim,
                }],
            )
        ],
        condition=IfCondition(PythonExpression([
            "'", LaunchConfiguration('enable_plotting'), "' == 'true' and '",
            LaunchConfiguration('test_type'), "' == 'lshape'"
        ]))
    )

    # ========================================================================
    # DATA LOGGERS
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
            'file_prefix': 'wheelchair_data_log'
        }]
    )

    # Scan data logger for SLAM debugging and fusion metrics
    # Logs all scan sources for comparing LiDAR vs camera contributions
    scan_data_logger = TimerAction(
        period=20.0,  # Wait for scan fusion to be ready
        actions=[
            LogInfo(msg='[LOGGER] Starting scan data logger for fusion metrics...'),
            Node(
                package='scripts',
                executable='scan_data_logger',
                name='scan_data_logger',
                output='screen',
                parameters=[{
                    'log_frequency_hz': 10.0,
                    'log_ranges': False,  # Set True for full ranges (larger files)
                    # LiDAR sources
                    'log_raw': True,           # /scan - raw RPLidar
                    'log_filtered': True,      # /scan_filtered - after laser filter
                    'log_lidar_only': True,    # /scan_lidar_only - LiDAR with footprint filter
                    # Camera sources (for fusion comparison)
                    'log_front_cam': True,     # /scan_front_camera - front D455
                    'log_left_cam': True,      # /scan_left_camera - left D455
                    'log_right_cam': True,     # /scan_right_camera - right D435i
                    # Fused output
                    'log_fused': True,         # /scan_fused - final fusion
                }]
            ),
        ],
        condition=UnlessCondition(is_sim)
    )

    # ========================================================================
    # SLAM TOOLBOX
    # ========================================================================

    # Nav2 lifecycle-managed nodes. slam_toolbox doesn't implement Nav2 bond
    # protocol so bond check will fail (cosmetic) — SLAM keeps running regardless.
    slam_lifecycle_nodes = ["map_saver_server", "slam_toolbox"]

    # SLAM nodes - two variants controlled by use_fused_slam argument
    # Default (false): LIDAR-ONLY SLAM — proven clean maps, no camera noise
    #   After 5 iterations (v0-v4), depth camera noise fundamentally degrades
    #   scan matching quality → localization drift → map corruption.
    #   TurtleBot4 uses the same architecture: lidar SLAM + depth costmap.
    # Optional (true): FUSED SLAM — experimental, uses slam_toolbox_fused_v4.yaml
    # NOTE: Cameras STILL run in both modes for Nav2 costmap obstacle detection.
    # SLAM nodes — hospital_mode selects 25m configs; default uses proven home configs
    def _make_slam_toolbox(context, *_args, **_kwargs):
        fused = LaunchConfiguration('use_fused_slam').perform(context) == 'true'
        hosp = LaunchConfiguration('hospital_mode').perform(context) == 'true'
        sim = LaunchConfiguration('is_sim').perform(context) == 'true'
        if sim:
            return []

        if fused:
            config = hospital_slam_config_fused if hosp else default_slam_config_fused
            scan_topic_val = '/scan_fused'
            label = f"FUSED {'(hospital 25m)' if hosp else '(v21)'}"
            delay = 28.0
        else:
            config = hospital_slam_config if hosp else LaunchConfiguration('slam_config').perform(context)
            scan_topic_val = '/scan_filtered'
            label = f"LIDAR-ONLY {'(hospital 25m)' if hosp else '(default)'}"
            delay = 20.0

        return [
            TimerAction(
                period=delay,
                actions=[
                    LogInfo(msg=f'[SLAM] Starting SLAM Toolbox: {label}'),
                    LogInfo(msg=f'[SLAM] Config: {config}'),
                    Node(
                        package='slam_toolbox',
                        executable='sync_slam_toolbox_node',
                        name='slam_toolbox',
                        output='screen',
                        parameters=[
                            config,
                            {'use_sim_time': False,
                             'scan_topic': scan_topic_val,
                             'use_lifecycle_manager': True}
                        ],
                    ),
                ],
            ),
        ]

    slam_toolbox_setup = OpaqueFunction(function=_make_slam_toolbox)

    # map_saver_server delayed to start AFTER SLAM Toolbox (20s)
    map_saver_server = TimerAction(
        period=22.0,
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
        period=38.0,  # After SLAM Toolbox (28s fused / 20s lidar) and map_saver (22s)
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
                    {'bond_timeout': 15.0},  # slam_toolbox bond will fail (cosmetic)
                ],
            ),
        ],
        condition=UnlessCondition(is_sim)
    )

    # ========================================================================
    # SESSION MANAGER — AUTO-SAVES MAP + ROSBAG ON CTRL+C
    # ========================================================================
    slam_session_manager = TimerAction(
        period=42.0,  # After SLAM lifecycle is fully active
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
    # DATASET RECORDER (only when dataset_mode:=true)
    # ========================================================================
    # Records ALL sensor streams (6 images + scans + odom + IMU + TF) to MCAP
    # rosbag at native rates. Prints live topic rate table every 10s.
    # Offline sync extraction: python3 bag_extract_synced.py <bag_path>
    dataset_recorder_node = TimerAction(
        period=22.0,  # Wait for all 3 cameras to initialize (front@3s, left@8s, right@13s)
        actions=[
            LogInfo(msg='=' * 60),
            LogInfo(msg='[DATASET] Starting dataset recorder (MCAP + rate monitor)'),
            LogInfo(msg='[DATASET] Recording 6 image + 8 sensor streams'),
            LogInfo(msg='[DATASET] ~18 GB/hr compressed'),
            LogInfo(msg='=' * 60),
            Node(
                package='scripts',
                executable='dataset_recorder',
                name='dataset_recorder',
                output='screen',
                parameters=[{
                    'report_interval': 10.0,
                }],
            ),
        ],
        condition=IfCondition(dataset_mode)
    )

    # ========================================================================
    # RVIZ VISUALIZATION
    # ========================================================================

    # RViz launches with a delay to ensure topics and transforms are available
    rviz_node = TimerAction(
        period=42.0,  # After SLAM lifecycle fully activated + map frame available
        actions=[
            LogInfo(msg='[RVIZ] Starting RViz with SLAM + 3-camera fusion display...'),
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

    # Ready message
    ready_message_lidar = TimerAction(
        period=45.0,
        actions=[
            LogInfo(msg='=' * 60),
            LogInfo(msg='  WHEELCHAIR SLAM MAPPING + EKF + ZUPT READY'),
            LogInfo(msg='=' * 60),
            LogInfo(msg='  Odometry: Robust EKF + ZUPT (Encoder + IMU + Accel)'),
            LogInfo(msg='    - ZUPT: Zero velocity when stationary'),
            LogInfo(msg='    - Auto gyro recalibration when still'),
            LogInfo(msg='  Sensors: RPLidar S3 + 3x RealSense (D455/D455/D435i)'),
            LogInfo(msg='  SLAM Input: /scan_filtered (LIDAR-ONLY — clean maps)'),
            LogInfo(msg='  Cameras: ACTIVE for Nav2 costmap (/scan_fused available)'),
            LogInfo(msg='  To use fused SLAM: ros2 launch ... use_fused_slam:=true'),
            LogInfo(msg='  Use slam_toolbox panel in RViz to save map'),
            LogInfo(msg='=' * 60),
        ],
        condition=UnlessCondition(use_fused_slam),
    )

    ready_message_fused = TimerAction(
        period=45.0,
        actions=[
            LogInfo(msg='=' * 60),
            LogInfo(msg='  WHEELCHAIR SLAM MAPPING + EKF + ZUPT READY'),
            LogInfo(msg='=' * 60),
            LogInfo(msg='  Odometry: Robust EKF + ZUPT (Encoder + IMU + Accel)'),
            LogInfo(msg='    - ZUPT: Zero velocity when stationary'),
            LogInfo(msg='    - Auto gyro recalibration when still'),
            LogInfo(msg='  Sensors: RPLidar S3 + 3x RealSense (D455/D455/D435i)'),
            LogInfo(msg='  SLAM Input: /scan_fused (LIDAR + 3 CAMERAS — experimental)'),
            LogInfo(msg='  Use slam_toolbox panel in RViz to save map'),
            LogInfo(msg='=' * 60),
        ],
        condition=IfCondition(use_fused_slam),
    )

    # ========================================================================
    # LAUNCH DESCRIPTION
    # ========================================================================

    return LaunchDescription([
        # Launch arguments
        declare_is_sim,
        declare_port,
        declare_model,
        declare_is_ignition,
        declare_use_sim_time,
        declare_world,
        declare_unite_imu,
        declare_lidar_port,
        declare_sudo_password,
        declare_rviz,
        declare_rviz_config,
        declare_enable_plotting,
        declare_test_type,
        declare_slam_config,
        declare_use_fused_slam,
        declare_dataset_mode,
        declare_hospital_mode,

        # USB permissions (hardware only)
        permission_setup,

        # Unified motion stack (ros2_control, teleop)
        unified_wheelchair_launch,

        # RPLidar S3 + Laser filter (hospital_mode uses 25m range)
        rplidar_s3_launch,
        laser_filter_setup,  # Filter /scan -> /scan_filtered

        # 3-Camera System (staggered startup, dataset_mode adds color+alignment)
        camera_setup,

        # IMU Processing Pipeline
        imu_startup_calibrator,    # 5s
        imu_bias_corrector,        # 5s
        imu_filter,                # 6s
        imu_wheelchair_republisher, # 7s

        # Scan Fusion (LiDAR + 3 cameras)
        scan_fusion_node,  # 18s - v9: camera wins if closer -> /scan_fused

        # Static Transforms + Joint State Publisher
        static_transform_publisher,  # base_link -> imu
        joint_state_publisher,       # Fallback joint states for robot model
        # laser frame now in URDF — no runtime TF needed

        # Robust EKF + ZUPT Sensor Fusion
        robust_ekf_zupt_node,  # 10s - publishes odom->base_link TF

        # SLAM Toolbox (hospital_mode selects 25m configs)
        slam_toolbox_setup,  # OpaqueFunction: selects config based on hospital_mode + use_fused_slam
        map_saver_server,        # 22s
        slam_lifecycle_manager,  # 25s - configures+activates both (bond warning is cosmetic)

        # Session manager (auto-save map + rosbag on Ctrl+C)
        slam_session_manager,

        # Dataset recorder (only when dataset_mode:=true)
        dataset_recorder_node,

        # Visualization and plotting
        square_path_plotter,
        lshape_path_plotter,
        topic_data_logger,
        scan_data_logger,
        rviz_node,              # 32s
        ready_message_lidar,    # 35s (lidar-only mode)
        ready_message_fused,    # 35s (fused mode)
    ])
