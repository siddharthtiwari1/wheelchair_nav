#!/usr/bin/env python3

"""
WHEELCHAIR FUSION LOCALIZATION - 3-CAMERA VISUAL-LIDAR INTEGRATION
===================================================================
Publication-quality global localization with 3-camera sensor fusion.

Based on working camera_fusion_test.launch.py pattern.

Launches:
    - Hardware interface (motors, ros2_control)
    - RPLidar S3 with robust filtering
    - 3 RealSense cameras (front D455 + left D455 + right D435i)
    - 3-camera + LiDAR scan fusion for height-aware localization
    - EKF sensor fusion (wheel odom + IMU)
    - Enhanced AMCL (map-based localization)
    - Map server

Key Features:
    - Robust laser filtering (6-stage chain)
    - 3 depth cameras for 270° obstacle detection at ALL heights
    - High-precision AMCL (12k particles) with /scan_fused input
    - Real-time pose estimation

Usage:
    ros2 launch wheelchair_bringup wheelchair_fusion_localization.launch.py

    With custom map:
    ros2 launch wheelchair_bringup wheelchair_fusion_localization.launch.py \\
        map_name:=/path/to/map.yaml

Author: Siddharth Tiwari (s24035@students.iitmandi.ac.in)
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
    GroupAction,
    LogInfo,
)
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression, Command
from launch_ros.actions import Node, SetRemap
from launch_ros.descriptions import ParameterFile
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    # Get package directories
    wheelchair_description_dir = get_package_share_directory('wheelchair_description')
    wheelchair_localization_dir = get_package_share_directory('wheelchair_localization')
    wheelchair_bringup_dir = get_package_share_directory('wheelchair_bringup')
    wc_control_dir = get_package_share_directory('wc_control')

    # Workspace root - use hardcoded path for reliability
    ws_root = '/home/sidd/wheelchair_nav'

    # Default paths
    default_model_path = os.path.join(
        wheelchair_description_dir, 'urdf', 'wheelchair_description.urdf.xacro'
    )
    default_rviz_config = os.path.join(
        wheelchair_description_dir, 'rviz', 'fusion_localization.rviz'
    )
    default_map_file = os.path.join(ws_root, 'maps', 'my_map_lidar.yaml')

    # Configuration files
    default_amcl_config = os.path.join(
        wheelchair_localization_dir, 'config', 'amcl_fusion.yaml'
    )
    default_ekf_config = os.path.join(
        wheelchair_localization_dir, 'config', 'ekf.yaml'
    )
    default_laser_filter_config = os.path.join(
        wheelchair_localization_dir, 'config', 'laser_filter_robust.yaml'
    )

    # ROS distro detection
    ros_distro = os.environ.get('ROS_DISTRO', 'jazzy')
    default_is_ignition = 'true' if ros_distro == 'humble' else 'false'

    # Hardware defaults
    is_sim_value = 'false'
    port_value = '/dev/ttyACM0'
    lidar_port_value = '/dev/ttyUSB0'

    # Camera serials (from hardware - same as camera_fusion_test.launch.py)
    # NOTE: Front camera comes from wheelchair_sensors_launch (includes IMU processing)
    LEFT_SERIAL = "'146222253403'"    # Left D455
    RIGHT_SERIAL = "'207522077542'"   # Right D435i

    # RealSense launch path
    rs_launch_path = os.path.join(
        get_package_share_directory('realsense2_camera'),
        'launch', 'rs_launch.py'
    )

    # ========================================================================
    # LAUNCH ARGUMENTS
    # ========================================================================

    declare_map_name = DeclareLaunchArgument(
        'map_name', default_value=default_map_file,
        description='Full path to map YAML file.'
    )
    declare_is_sim = DeclareLaunchArgument(
        'is_sim', default_value=is_sim_value,
        description='True for simulation, false for real hardware.'
    )
    declare_port = DeclareLaunchArgument(
        'port', default_value=port_value,
        description='Serial port for wheelchair hardware interface.'
    )
    declare_model = DeclareLaunchArgument(
        'model', default_value=default_model_path,
        description='Absolute path to the wheelchair URDF/xacro file.'
    )
    declare_is_ignition = DeclareLaunchArgument(
        'is_ignition', default_value=default_is_ignition,
        description='Set to true when running on Humble/ros_ign combination.'
    )
    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time', default_value='false',
        description='Use simulation time if true.'
    )
    declare_world = DeclareLaunchArgument(
        'world', default_value='empty.sdf',
        description='Gazebo world file (simulation only).'
    )
    declare_unite_imu = DeclareLaunchArgument(
        'unite_imu_method', default_value='2',
        description='RealSense IMU synchronization (0=None, 1=copy, 2=linear interpolation).'
    )
    declare_lidar_port = DeclareLaunchArgument(
        'lidar_port', default_value=lidar_port_value,
        description='Serial port for RPLidar (hardware only).'
    )
    declare_sudo_password = DeclareLaunchArgument(
        'sudo_password', default_value='12345',
        description='Sudo password for USB permissions (hardware only).'
    )
    declare_rviz = DeclareLaunchArgument(
        'rviz', default_value='true',
        description='Launch RViz for visualization.'
    )
    declare_rviz_config = DeclareLaunchArgument(
        'rviz_config', default_value=default_rviz_config,
        description='RViz configuration file.'
    )
    declare_amcl_config = DeclareLaunchArgument(
        'amcl_config', default_value=default_amcl_config,
        description='Full path to AMCL yaml configuration file.'
    )
    declare_use_depth = DeclareLaunchArgument(
        'use_depth', default_value='true',
        description='Enable depth camera for obstacle detection.'
    )
    declare_use_fused_scan = DeclareLaunchArgument(
        'use_fused_scan', default_value='false',
        description='Use fused scan for AMCL. Default false: AMCL uses /scan_filtered to match lidar-built maps. Fused scan available on /scan_fused for Nav2 costmaps.'
    )
    declare_use_custom_fusion = DeclareLaunchArgument(
        'use_custom_fusion', default_value='false',
        description='Use custom AMF (Adaptive Multi-sensor Fusion) instead of robot_localization EKF.'
    )
    declare_use_zupt = DeclareLaunchArgument(
        'use_zupt', default_value='true',
        description='Use ZUPT-Enhanced Odometry (encoder + gyro bias calibration) - BEST ACCURACY.'
    )

    # Get launch configurations
    is_sim = LaunchConfiguration('is_sim')
    map_name = LaunchConfiguration('map_name')
    amcl_config = LaunchConfiguration('amcl_config')
    use_rviz = LaunchConfiguration('rviz')
    use_depth = LaunchConfiguration('use_depth')
    use_fused_scan = LaunchConfiguration('use_fused_scan')
    use_custom_fusion = LaunchConfiguration('use_custom_fusion')
    use_zupt = LaunchConfiguration('use_zupt')

    # ========================================================================
    # USB PERMISSIONS SETUP
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
            print(f'[INFO] Skipping permission change for {lidar_port}: device not present')

        for port in ports:
            try:
                subprocess.run(
                    ['sudo', '-S', 'chmod', '666', port],
                    input=f'{password}\n',
                    text=True,
                    check=True,
                    capture_output=True,
                )
                print(f'[OK] Updated permissions for {port}')
            except subprocess.CalledProcessError as exc:
                print(f'[WARN] Failed to set permissions for {port}: {exc}')
        return []

    permission_setup = OpaqueFunction(
        function=_grant_usb_permissions,
        condition=UnlessCondition(is_sim)
    )

    # ========================================================================
    # HARDWARE & SENSORS
    # ========================================================================

    # Unified wheelchair hardware/control stack
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

    # Hardware sensor pipeline (RealSense + IMU filtering)
    # FIXED 2026-01-08: Set standalone=false to prevent double robot_state_publisher
    # (unified_wheelchair.launch.py already launches robot_state_publisher)
    wheelchair_sensors_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(wc_control_dir, 'launch', 'wheelchair_sensors.launch.py')
        ),
        launch_arguments={
            'model': LaunchConfiguration('model'),
            'is_ignition': LaunchConfiguration('is_ignition'),
            'is_sim': LaunchConfiguration('is_sim'),
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'unite_imu_method': LaunchConfiguration('unite_imu_method'),
            'rviz': 'false',
            'rviz_config': LaunchConfiguration('rviz_config'),
            'standalone': 'false',  # CRITICAL: unified_wheelchair provides robot_state_publisher
        }.items(),
        condition=UnlessCondition(is_sim)
    )

    # RPLidar S3
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

    # ========================================================================
    # ADDITIONAL CAMERAS FOR 3-CAMERA FUSION
    # ========================================================================
    # Front camera (D455 + IMU) comes from wheelchair_sensors_launch
    # Here we add left and right cameras with staggered startup

    # Left Camera (D455) - DEPTH ONLY
    left_cam = TimerAction(
        period=5.0,  # After front camera from wheelchair_sensors_launch (~3s)
        actions=[
            LogInfo(msg='[CAMERA] Starting LEFT D455 (424x240@6Hz depth only)...'),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(rs_launch_path),
                launch_arguments={
                    'camera_name': 'mapping_camera',
                    'camera_namespace': '',
                    'serial_no': LEFT_SERIAL,
                    'publish_tf': 'false',
                    'enable_gyro': 'false',
                    'enable_accel': 'false',
                    'enable_color': 'false',
                    'depth_module.profile': '424x240x6',
                    'pointcloud.enable': 'true',
                    'pointcloud.stream_filter': '0',
                    'pointcloud.stream_index_filter': '0',
                }.items(),
            ),
        ],
        condition=UnlessCondition(is_sim)
    )

    # Right Camera (D435i) - DEPTH ONLY
    right_cam = TimerAction(
        period=10.0,  # Staggered to avoid USB bandwidth collision
        actions=[
            LogInfo(msg='[CAMERA] Starting RIGHT D435i (424x240@6Hz depth only)...'),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(rs_launch_path),
                launch_arguments={
                    'camera_name': 'right_camera',
                    'camera_namespace': '',
                    'serial_no': RIGHT_SERIAL,
                    'publish_tf': 'false',
                    'enable_gyro': 'false',
                    'enable_accel': 'false',
                    'enable_color': 'false',
                    'depth_module.profile': '424x240x6',
                    'pointcloud.enable': 'true',
                    'pointcloud.stream_filter': '0',
                    'pointcloud.stream_index_filter': '0',
                }.items(),
            ),
        ],
        condition=UnlessCondition(is_sim)
    )

    # ========================================================================
    # ROBUST LASER FILTER (6-stage chain)
    # ========================================================================

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

    # ========================================================================
    # SCAN-DEPTH FUSION NODE (Height-Aware 2D Scan)
    # ========================================================================
    # Fuses filtered 2D LiDAR with depth camera point cloud for robust
    # localization that detects obstacles at ALL heights (tables, shelves, etc.)
    # Input: /scan_filtered (2D LiDAR) + /camera/depth/color/points (3D depth)
    # Output: /scan_fused (height-augmented 2D scan for AMCL)

    # ========================================================================
    # SCAN FUSION V7 — HIGH PERFORMANCE 3-CAMERA + LIDAR
    # ========================================================================
    # v7 fixes: dedup guard, final zero cleanup, min_camera_points_per_bin=3.
    # Camera data only fills lidar gaps (with point agreement) or detects
    # genuinely closer obstacles (15cm margin over lidar).
    scan_depth_fusion_node = TimerAction(
        period=15.0,  # Wait for all 3 cameras to be ready (right_cam at 10s + 5s buffer)
        actions=[
            LogInfo(msg='=' * 60),
            LogInfo(msg='[FUSION] Starting SCAN FUSION V9 (camera wins if closer) for localization'),
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
                    'max_camera_age_ms': 2000.0,   # 2s — never drop camera frames
                    'camera_warmup_sec': 3.0,
                    # Turn suppression DISABLED — cameras always active
                    'turn_suppress_wz': 999.0,  # effectively disabled — cameras fuse at all times
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
                    # Left camera D455
                    'left_camera.enabled': True,
                    'left_camera.topic': '/mapping_camera/depth/color/points',
                    'left_camera.frame': 'mapping_camera_depth_optical_frame',
                    'left_camera.max_depth': 3.0,
                    'left_camera.min_depth': 0.30,
                    # Right camera D435i
                    'right_camera.enabled': True,
                    'right_camera.topic': '/right_camera/depth/color/points',
                    'right_camera.frame': 'right_camera_depth_optical_frame',
                    'right_camera.max_depth': 3.0,
                    'right_camera.min_depth': 0.30,
                }],
                condition=UnlessCondition(is_sim)
            )
        ]
    )

    # ========================================================================
    # TF TRANSFORMS
    # ========================================================================

    # Static TF: base_link -> imu
    static_transform_imu = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['--x', '0', '--y', '0', '--z', '0',
                   '--qx', '0', '--qy', '0', '--qz', '0', '--qw', '1',
                   '--frame-id', 'base_link', '--child-frame-id', 'imu']
    )

    # NOTE: laser frame now defined in URDF (wheelchair_description.urdf.xacro)
    # as a fixed joint under lidar with 180° Z rotation — no runtime TF publisher needed

    # ========================================================================
    # ODOMETRY VELOCITY CORRECTOR
    # ========================================================================
    # Computes correct velocity from position changes (fixes 20% underestimation)
    # Publishes /wheel_velocity for EKF to fuse with IMU yaw

    odom_velocity_corrector = TimerAction(
        period=4.0,  # Start before EKF
        actions=[
            Node(
                package='wheelchair_localization',
                executable='odom_velocity_corrector',
                name='odom_velocity_corrector',
                output='screen',
                parameters=[{
                    'input_topic': '/wc_control/odom',
                    'output_topic': '/wheel_velocity',
                    'velocity_covariance': 0.01,
                    'use_sim_time': LaunchConfiguration('use_sim_time')
                }]
            )
        ]
    )

    # ========================================================================
    # LOCALIZATION FUSION (Encoder + IMU)
    # ========================================================================
    # Two options:
    # 1. Custom AMF (Adaptive Multi-sensor Fusion) - Novel approach
    # 2. robot_localization EKF - Standard ROS2 approach

    # OPTION 1: Custom Adaptive Multi-sensor Fusion (AMF)
    # Novel encoder + IMU fusion with:
    # - Adaptive weighting based on motion state
    # - Automatic gyro bias estimation
    # - Outlier rejection
    # - Sensor failure detection
    custom_fusion_node = TimerAction(
        period=5.0,
        actions=[
            LogInfo(msg='[LOCALIZATION] Starting Custom AMF (Encoder + IMU Fusion)...'),
            Node(
                package='wheelchair_localization',
                executable='custom_localization_fusion',
                name='custom_localization_fusion',
                output='screen',
                parameters=[{
                    'frequency': 50.0,
                    'odom_topic': '/wc_control/odom',
                    'imu_topic': '/imu',
                    'visual_odom_topic': '/camera/odom/sample',
                    'output_topic': '/odometry/filtered',
                    'status_topic': '/localization/status',
                    'odom_frame': 'odom',
                    'base_frame': 'base_link',
                    'wheel_position_trust': 0.95,
                    'imu_orientation_trust': 0.90,
                    'imu_angular_vel_trust': 0.85,
                    'wheel_linear_vel_trust': 0.80,
                    'stationary_linear_threshold': 0.01,
                    'stationary_angular_threshold': 0.02,
                    'max_linear_velocity': 0.5,
                    'max_angular_velocity': 1.0,
                    'max_position_jump': 0.1,
                    'max_orientation_jump': 0.2,
                    'use_sim_time': LaunchConfiguration('use_sim_time')
                }]
            )
        ],
        condition=IfCondition(PythonExpression([
            "'", use_custom_fusion, "' == 'true' and '", use_zupt, "' == 'false'"
        ]))
    )

    # OPTION 2: Standard robot_localization EKF (fallback)
    ekf_local_node = TimerAction(
        period=5.0,
        actions=[
            LogInfo(msg='[LOCALIZATION] Starting robot_localization EKF...'),
            Node(
                package='robot_localization',
                executable='ekf_node',
                name='ekf_filter_node',
                output='screen',
                parameters=[
                    default_ekf_config,
                    {'use_sim_time': LaunchConfiguration('use_sim_time')}
                ]
            )
        ],
        condition=IfCondition(PythonExpression([
            "'", use_custom_fusion, "' == 'false' and '", use_zupt, "' == 'false'"
        ]))
    )

    # OPTION 3: ZUPT-Enhanced Odometry - BEST ACCURACY
    # Uses stationary periods to calibrate gyro bias, slip detection with bias-corrected gyro
    zupt_node = TimerAction(
        period=5.0,
        actions=[
            LogInfo(msg='[LOCALIZATION] Starting ZUPT-Enhanced Odometry (Encoder + Gyro Bias Calibration)...'),
            Node(
                package='wheelchair_zupt',
                executable='zupt_node',
                name='zupt_odometry',
                output='screen',
                parameters=[{
                    'imu_topic': '/imu',
                    'odom_topic': '/wc_control/odom',
                    'odom_frame': 'odom',
                    'base_frame': 'base_link',
                    'publish_tf': True,
                    # Wheel calibration
                    'wheel_radius_L': 0.1524,
                    'wheel_radius_R': 0.1524,
                    'wheel_baseline': 0.565,
                    # ZUPT parameters
                    'stationary_threshold': 0.015,  # above encoder noise (~0.005), below min real velocity (0.05)
                    'stationary_omega_threshold': 0.05,
                    'bias_adaptation_rate': 0.01,
                    'slip_threshold': 0.08,  # diagnostic-only — slip detection published but doesn't swap source
                    'gyro_blend_alpha': 0.30,  # complementary filter: 30% encoder + 70% gyro, prevents heading drift
                    'use_sim_time': LaunchConfiguration('use_sim_time')
                }]
            )
        ],
        condition=IfCondition(use_zupt)
    )

    # ========================================================================
    # GLOBAL LOCALIZATION - MAP SERVER + AMCL
    # ========================================================================

    lifecycle_nodes = ['map_server', 'amcl']

    # Map Server
    nav2_map_server = TimerAction(
        period=16.0,  # After scan fusion is ready
        actions=[
            LogInfo(msg='[NAV2] Starting map server...'),
            Node(
                package='nav2_map_server',
                executable='map_server',
                name='map_server',
                output='screen',
                parameters=[
                    {'yaml_filename': map_name},
                    {'use_sim_time': LaunchConfiguration('use_sim_time')}
                ],
            )
        ]
    )

    # AMCL with fused scan support
    # Uses /scan_fused (height-aware) when use_fused_scan=true, otherwise /scan_filtered
    nav2_amcl_fused = TimerAction(
        period=17.0,  # After map server
        actions=[
            LogInfo(msg='[NAV2] Starting AMCL with /scan_fused (LiDAR + 3 cameras)...'),
            Node(
                package='nav2_amcl',
                executable='amcl',
                name='amcl',
                output='screen',
                emulate_tty=True,
                parameters=[
                    amcl_config,
                    {'use_sim_time': LaunchConfiguration('use_sim_time')},
                    {'scan_topic': '/scan_fused'},  # Override to use fused scan
                ],
            )
        ],
        condition=IfCondition(use_fused_scan)
    )

    # AMCL with filtered scan (fallback when depth fusion not available)
    nav2_amcl_filtered = TimerAction(
        period=17.0,
        actions=[
            LogInfo(msg='[NAV2] Starting AMCL with /scan_filtered (LiDAR only)...'),
            Node(
                package='nav2_amcl',
                executable='amcl',
                name='amcl',
                output='screen',
                emulate_tty=True,
                parameters=[
                    amcl_config,
                    {'use_sim_time': LaunchConfiguration('use_sim_time')},
                    {'scan_topic': '/scan_filtered'},  # Use filtered scan
                ],
            )
        ],
        condition=UnlessCondition(use_fused_scan)
    )

    # Lifecycle Manager for localization
    nav2_lifecycle_manager = TimerAction(
        period=19.0,  # After AMCL
        actions=[
            LogInfo(msg='[NAV2] Starting localization lifecycle manager...'),
            Node(
                package='nav2_lifecycle_manager',
                executable='lifecycle_manager',
                name='lifecycle_manager_localization',
                output='screen',
                parameters=[
                    {'node_names': lifecycle_nodes},
                    {'use_sim_time': LaunchConfiguration('use_sim_time')},
                    {'autostart': True},
                    {'bond_timeout': 0.0},
                ],
            )
        ]
    )

    # ========================================================================
    # VELOCITY COMMAND BRIDGE
    # ========================================================================

    # Converts Twist to TwistStamped for ros2_control
    # Chain: velocity_smoother -> /cmd_vel -> bridge -> /wc_control/cmd_vel -> DiffDriveController
    # Safety layers: velocity_smoother (acceleration), wc_control_safe.yaml (hard limits), twist bridge (clamping)
    # TODO: Wire collision_monitor inline with separate lifecycle manager
    twist_to_stamped_converter = Node(
        package='scripts',
        executable='twist_stamped_teleop',
        name='cmd_vel_bridge',
        output='screen',
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
        remappings=[
            ('cmd_vel_in', '/cmd_vel'),
            ('cmd_vel_out', '/wc_control/cmd_vel'),
        ],
    )

    # ========================================================================
    # DATA LOGGER - REMOVED (consolidated into navigation_debug_logger in nav launch)
    # Use navigation_debug_logger for comprehensive logging to /home/sidd/wheelchair_nav/data_logs/
    # ========================================================================

    # ========================================================================
    # RVIZ VISUALIZATION
    # ========================================================================

    rviz_node = TimerAction(
        period=22.0,  # After localization is ready
        actions=[
            LogInfo(msg='[RVIZ] Starting RViz with 3-camera fusion display...'),
            Node(
                package='rviz2',
                executable='rviz2',
                name='rviz2',
                output='screen',
                arguments=['-d', LaunchConfiguration('rviz_config')],
                parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
                condition=IfCondition(use_rviz),
            )
        ],
    )

    # ========================================================================
    # STARTUP MESSAGES
    # ========================================================================

    startup_message = TimerAction(
        period=25.0,
        actions=[
            LogInfo(msg='=' * 60),
            LogInfo(msg='  WHEELCHAIR FUSION LOCALIZATION + 3-CAMERA READY'),
            LogInfo(msg='=' * 60),
            LogInfo(msg='  Sensors: RPLidar S3 + 3x RealSense (D455/D455/D435i)'),
            LogInfo(msg='  Filter:  6-stage robust laser filter'),
            LogInfo(msg='  AMCL:    /scan_filtered (lidar-only, matches lidar-built map)'),
            LogInfo(msg='  Costmap: /scan_fused available (lidar + 3 cameras)'),
            LogInfo(msg='  Fusion:  scan_fusion_v9 (camera wins if closer, turn compensation)'),
            LogInfo(msg='=' * 60),
        ]
    )

    # ========================================================================
    # LAUNCH DESCRIPTION
    # ========================================================================

    return LaunchDescription([
        # Launch arguments
        declare_map_name,
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
        declare_amcl_config,
        declare_use_depth,
        declare_use_fused_scan,
        declare_use_custom_fusion,
        declare_use_zupt,

        # USB permissions (hardware only)
        permission_setup,

        # Hardware and sensors
        unified_wheelchair_launch,
        wheelchair_sensors_launch,  # Front camera + IMU processing
        rplidar_s3_launch,

        # Additional cameras for 3-camera fusion
        left_cam,   # 5s  - Left D455
        right_cam,  # 10s - Right D435i

        # Laser filtering
        laser_filter_node,

        # Scan-depth fusion for robust localization (height-aware)
        scan_depth_fusion_node,  # 15s - /scan_filtered + 3 cameras -> /scan_fused

        # TF transforms + joint state fallback
        static_transform_imu,
        # laser frame now in URDF — no runtime TF needed
        # Joint state publisher for robot model (fallback if joint_state_broadcaster hasn't started)
        # Pass URDF directly to avoid QoS mismatch with robot_state_publisher topic
        Node(
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
        ),

        # Odometry velocity corrector (before fusion)
        odom_velocity_corrector,

        # Localization fusion (ZUPT / Custom AMF / robot_localization EKF)
        zupt_node,            # Uses ZUPT-Enhanced Odometry when use_zupt=true (DEFAULT - BEST)
        custom_fusion_node,   # Uses custom AMF when use_custom_fusion=true and use_zupt=false
        ekf_local_node,       # Uses robot_localization EKF when both are false

        # Global localization (AMCL) - uses fused or filtered scan based on argument
        nav2_map_server,
        nav2_amcl_fused,      # Uses /scan_fused when use_fused_scan=true
        nav2_amcl_filtered,   # Uses /scan_filtered when use_fused_scan=false
        nav2_lifecycle_manager,

        # Velocity command bridge
        twist_to_stamped_converter,

        # Visualization (logging moved to navigation_debug_logger in nav launch)
        rviz_node,
        startup_message,
    ])
