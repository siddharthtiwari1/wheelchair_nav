#!/usr/bin/env python3

"""
WHEELCHAIR RTAB-MAP HOSPITAL MAPPING — 3-CAMERA VISUAL + 25m LIDAR
===================================================================
RTAB-Map graph-based SLAM for spacious hospital environments.

Architecture:
  RPLidar S3 (25m) ──→ /scan_filtered ──────────────┐
                                                      ├→ RTAB-Map → map + visual loop closure
  Front D455  → rgbd_sync → /front_rgbd_image  ──────┤
  Left  D455  → rgbd_sync → /left_rgbd_image   ──────┤
  Right D435i → rgbd_sync → /right_rgbd_image  ──────┘

  Lidar: ICP registration (25m walls/corridors)
  Cameras: Visual features for loop closure (room numbers, signs, floor patterns)
  → Solves corridor drift (corridors look identical to lidar, but distinct to cameras)

Usage:
    # Full 3-camera visual + lidar (recommended for hospital)
    ros2 launch wheelchair_bringup wheelchair_rtabmap_mapping.launch.py

    # Fused scan (lidar + camera obstacles in SLAM scan)
    ros2 launch wheelchair_bringup wheelchair_rtabmap_mapping.launch.py use_fused_slam:=true

    # Lidar-only (no camera visual features — will drift in corridors!)
    ros2 launch wheelchair_bringup wheelchair_rtabmap_mapping.launch.py use_cameras:=false

Save map:
    ros2 run nav2_map_server map_saver_cli -f ~/wheelchair_nav/maps/hospital_map
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

    default_model_path = os.path.join(
        wheelchair_description_dir, 'urdf', 'wheelchair_description.urdf.xacro',
    )
    default_rviz_config = os.path.join(
        wheelchair_description_dir, 'rviz', 'slam_mapping.rviz',
    )

    # Camera serials
    FRONT_SERIAL = "'337122300107'"   # Front D455
    LEFT_SERIAL = "'146222253403'"    # Left D455
    RIGHT_SERIAL = "'207522077542'"   # Right D435i

    rs_launch_path = os.path.join(
        get_package_share_directory('realsense2_camera'), 'launch', 'rs_launch.py'
    )

    # HOSPITAL laser filter — 25m range
    laser_filter_config = os.path.join(
        wheelchair_localization_dir, 'config', 'laser_filter_hospital.yaml',
    )

    ros_distro = os.environ.get('ROS_DISTRO', 'jazzy')
    default_is_ignition = 'true' if ros_distro == 'humble' else 'false'

    is_sim_value = 'false'
    port_value = '/dev/ttyACM0'
    lidar_port_value = '/dev/ttyUSB0'

    # ========================================================================
    # LAUNCH ARGUMENTS
    # ========================================================================

    declare_is_sim = DeclareLaunchArgument('is_sim', default_value=is_sim_value)
    declare_port = DeclareLaunchArgument('port', default_value=port_value)
    declare_model = DeclareLaunchArgument('model', default_value=default_model_path)
    declare_is_ignition = DeclareLaunchArgument('is_ignition', default_value=default_is_ignition)
    declare_use_sim_time = DeclareLaunchArgument('use_sim_time', default_value='false')
    declare_world = DeclareLaunchArgument('world', default_value='empty.sdf')
    declare_unite_imu = DeclareLaunchArgument('unite_imu_method', default_value='2')
    declare_lidar_port = DeclareLaunchArgument('lidar_port', default_value=lidar_port_value)
    declare_sudo_password = DeclareLaunchArgument('sudo_password', default_value='12345')
    declare_rviz = DeclareLaunchArgument('rviz', default_value='true')
    declare_rviz_config = DeclareLaunchArgument('rviz_config', default_value=default_rviz_config)
    declare_use_fused_slam = DeclareLaunchArgument(
        'use_fused_slam', default_value='false',
        description='false=/scan_filtered (lidar-only), true=/scan_fused (lidar+camera obstacles in scan)')
    declare_use_cameras = DeclareLaunchArgument(
        'use_cameras', default_value='true',
        description='Enable 3-camera RGB for RTAB-Map visual loop closure (recommended for hospital).')
    declare_database_path = DeclareLaunchArgument(
        'database_path',
        default_value=os.path.join(os.path.expanduser('~'), 'wheelchair_nav', 'maps', 'rtabmap_hospital.db'),
        description='RTAB-Map database file path.')
    declare_delete_db = DeclareLaunchArgument(
        'delete_db_on_start', default_value='true',
        description='Delete existing DB before mapping (fresh map each run).')

    is_sim = LaunchConfiguration('is_sim')
    use_fused_slam = LaunchConfiguration('use_fused_slam')
    use_cameras = LaunchConfiguration('use_cameras')

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
                    input=f'{password}\n', text=True, check=True,
                )
                print(f'  Updated permissions for {port}')
            except subprocess.CalledProcessError as exc:
                print(f'  Failed to set permissions for {port}: {exc}')
        return []

    permission_setup = OpaqueFunction(
        function=_grant_usb_permissions, condition=UnlessCondition(is_sim)
    )

    # Unified wheelchair hardware/control stack
    unified_wheelchair_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(wc_control_dir, 'launch', 'unified_wheelchair.launch.py')
        ]),
        launch_arguments={
            'is_sim': LaunchConfiguration('is_sim'),
            'port': LaunchConfiguration('port'),
            'world': LaunchConfiguration('world'),
        }.items(),
    )

    # ========================================================================
    # IMU PROCESSING PIPELINE
    # ========================================================================

    imu_startup_calibrator = TimerAction(
        period=5.0,
        actions=[Node(
            package='wc_control', executable='imu_startup_calibrator.py',
            name='imu_startup_calibrator', output='screen',
            parameters=[{
                'use_sim_time': LaunchConfiguration('use_sim_time'),
                'input_topic': '/camera/imu',
                'bias_topic': '/imu/calibrated_bias',
                'status_topic': '/imu/calibration_status',
                'calibration_duration': 3.0, 'min_samples': 100,
                'max_motion_threshold': 0.05,
                'default_gyro_x_bias': -0.004302,
                'default_gyro_y_bias': 0.000787,
                'default_gyro_z_bias': 0.000948,
            }],
        )],
        condition=UnlessCondition(is_sim)
    )

    imu_bias_corrector = TimerAction(
        period=5.0,
        actions=[Node(
            package='wc_control', executable='imu_bias_corrector.py',
            name='imu_bias_corrector', output='screen',
            parameters=[{
                'use_sim_time': LaunchConfiguration('use_sim_time'),
                'input_topic': '/camera/imu',
                'output_topic': '/camera/imu_corrected',
                'calibrated_bias_topic': '/imu/calibrated_bias',
                'gyro_x_bias': -0.004302,
                'gyro_y_bias': 0.000787,
                'gyro_z_bias': 0.000948,
            }],
        )],
        condition=UnlessCondition(is_sim)
    )

    imu_filter = TimerAction(
        period=6.0,
        actions=[Node(
            package='imu_filter_madgwick', executable='imu_filter_madgwick_node',
            name='imu_filter_madgwick', output='screen',
            parameters=[{
                'use_mag': False, 'world_frame': 'enu',
                'publish_tf': False, 'gain': 0.041,
                'use_sim_time': LaunchConfiguration('use_sim_time'),
            }],
            remappings=[('imu/data_raw', '/camera/imu_corrected')],
        )],
        condition=UnlessCondition(is_sim)
    )

    imu_wheelchair_republisher = TimerAction(
        period=7.0,
        actions=[Node(
            package='wc_control', executable='imu_wheelchair_republisher.py',
            name='imu_wheelchair_republisher', output='screen',
            parameters=[{
                'use_sim_time': LaunchConfiguration('use_sim_time'),
                'input_topic': '/imu/data', 'output_topic': '/imu',
                'output_frame': 'imu', 'zero_on_start': True,
                'orientation_quaternion': [-0.5, 0.5, -0.5, 0.5],
                'vector_quaternion': [-0.5, 0.5, -0.5, 0.5],
            }],
        )],
        condition=UnlessCondition(is_sim)
    )

    # ========================================================================
    # RPLIDAR S3 + HOSPITAL LASER FILTER (25m range)
    # ========================================================================

    rplidar_s3_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(get_package_share_directory('rplidar_ros'), 'launch', 'rplidar_s3_launch.py')
        ]),
        launch_arguments={'inverted': 'true'}.items(),
        condition=UnlessCondition(is_sim)
    )

    laser_filter_node = Node(
        package='laser_filters', executable='scan_to_scan_filter_chain',
        name='laser_filter', output='screen',
        parameters=[laser_filter_config],
        remappings=[('scan', '/scan'), ('scan_filtered', '/scan_filtered')],
        condition=UnlessCondition(is_sim)
    )

    # ========================================================================
    # 3-CAMERA SYSTEM — RGB + DEPTH + POINTCLOUD FOR RTAB-MAP
    # ========================================================================
    # All cameras run with COLOR ENABLED for RTAB-Map visual loop closure.
    # align_depth aligns depth to color frame so rgbd_sync can bundle them.
    # Pointclouds kept for scan_fusion_v9 (fused scan mode).

    def _make_cameras(context, *args, **kwargs):
        sim = LaunchConfiguration('is_sim').perform(context) == 'true'
        if sim:
            return []

        # Common: color + depth + alignment + pointcloud
        front_args = {
            'camera_name': 'camera', 'camera_namespace': '',
            'serial_no': FRONT_SERIAL, 'publish_tf': 'false',
            'enable_gyro': 'true', 'enable_accel': 'true',
            'unite_imu_method': '2',
            'enable_color': 'true',                # RGB for RTAB-Map visual features
            'rgb_camera.profile': '424x240x6',
            'depth_module.profile': '424x240x6',
            'align_depth.enable': 'true',          # Align depth to color frame
            'pointcloud.enable': 'true',           # For scan_fusion
            'pointcloud.stream_filter': '0', 'pointcloud.stream_index_filter': '0',
        }
        left_args = {
            'camera_name': 'mapping_camera', 'camera_namespace': '',
            'serial_no': LEFT_SERIAL, 'publish_tf': 'false',
            'enable_gyro': 'false', 'enable_accel': 'false',
            'enable_color': 'true',
            'rgb_camera.profile': '424x240x6',
            'depth_module.profile': '424x240x6',
            'align_depth.enable': 'true',
            'pointcloud.enable': 'true',
            'pointcloud.stream_filter': '0', 'pointcloud.stream_index_filter': '0',
        }
        right_args = {
            'camera_name': 'right_camera', 'camera_namespace': '',
            'serial_no': RIGHT_SERIAL, 'publish_tf': 'false',
            'enable_gyro': 'false', 'enable_accel': 'false',
            'enable_color': 'true',
            'rgb_camera.profile': '424x240x6',
            'depth_module.profile': '424x240x6',
            'align_depth.enable': 'true',
            'pointcloud.enable': 'true',
            'pointcloud.stream_filter': '0', 'pointcloud.stream_index_filter': '0',
        }
        return [
            TimerAction(period=3.0, actions=[
                LogInfo(msg='[CAMERA] Starting FRONT D455 (RGB + depth + IMU)...'),
                IncludeLaunchDescription(
                    PythonLaunchDescriptionSource(rs_launch_path),
                    launch_arguments=front_args.items(),
                ),
            ]),
            TimerAction(period=8.0, actions=[
                LogInfo(msg='[CAMERA] Starting LEFT D455 (RGB + depth)...'),
                IncludeLaunchDescription(
                    PythonLaunchDescriptionSource(rs_launch_path),
                    launch_arguments=left_args.items(),
                ),
            ]),
            TimerAction(period=13.0, actions=[
                LogInfo(msg='[CAMERA] Starting RIGHT D435i (RGB + depth)...'),
                IncludeLaunchDescription(
                    PythonLaunchDescriptionSource(rs_launch_path),
                    launch_arguments=right_args.items(),
                ),
            ]),
        ]

    camera_setup = OpaqueFunction(function=_make_cameras)

    # ========================================================================
    # RGBD SYNC NODES — PREPROCESSING LAYER FOR RTAB-MAP
    # ========================================================================
    # Each rgbd_sync bundles one camera's RGB + aligned depth into a single
    # RGBDImage message. This is the "preprocessing layer" that feeds
    # visual features from all 3 cameras into RTAB-Map.
    #
    # Pipeline per camera:
    #   /camera/color/image_raw + /camera/aligned_depth_to_color/image_raw
    #   → rgbd_sync → /front_rgbd_image (RGBDImage msg)
    #   → RTAB-Map extracts ORB features → bag-of-words loop closure

    front_rgbd_sync = TimerAction(
        period=16.0,  # After front camera is ready (3s + warmup)
        actions=[
            LogInfo(msg='[RGBD_SYNC] Starting front camera sync...'),
            Node(
                package='rtabmap_sync', executable='rgbd_sync',
                name='front_rgbd_sync', output='screen',
                parameters=[{
                    'approx_sync': True,
                    'approx_sync_max_interval': 0.1,
                    'topic_queue_size': 10,
                    'sync_queue_size': 10,
                }],
                remappings=[
                    ('rgb/image', '/camera/color/image_raw'),
                    ('depth/image', '/camera/aligned_depth_to_color/image_raw'),
                    ('rgb/camera_info', '/camera/color/camera_info'),
                    ('rgbd_image', '/front_rgbd_image'),
                ],
            ),
        ],
        condition=IfCondition(use_cameras)
    )

    left_rgbd_sync = TimerAction(
        period=18.0,  # After left camera is ready (8s + warmup)
        actions=[
            LogInfo(msg='[RGBD_SYNC] Starting left camera sync...'),
            Node(
                package='rtabmap_sync', executable='rgbd_sync',
                name='left_rgbd_sync', output='screen',
                parameters=[{
                    'approx_sync': True,
                    'approx_sync_max_interval': 0.1,
                    'topic_queue_size': 10,
                    'sync_queue_size': 10,
                }],
                remappings=[
                    ('rgb/image', '/mapping_camera/color/image_raw'),
                    ('depth/image', '/mapping_camera/aligned_depth_to_color/image_raw'),
                    ('rgb/camera_info', '/mapping_camera/color/camera_info'),
                    ('rgbd_image', '/left_rgbd_image'),
                ],
            ),
        ],
        condition=IfCondition(use_cameras)
    )

    right_rgbd_sync = TimerAction(
        period=20.0,  # After right camera is ready (13s + warmup)
        actions=[
            LogInfo(msg='[RGBD_SYNC] Starting right camera sync...'),
            Node(
                package='rtabmap_sync', executable='rgbd_sync',
                name='right_rgbd_sync', output='screen',
                parameters=[{
                    'approx_sync': True,
                    'approx_sync_max_interval': 0.1,
                    'topic_queue_size': 10,
                    'sync_queue_size': 10,
                }],
                remappings=[
                    ('rgb/image', '/right_camera/color/image_raw'),
                    ('depth/image', '/right_camera/aligned_depth_to_color/image_raw'),
                    ('rgb/camera_info', '/right_camera/color/camera_info'),
                    ('rgbd_image', '/right_rgbd_image'),
                ],
            ),
        ],
        condition=IfCondition(use_cameras)
    )

    # ========================================================================
    # SCAN FUSION (lidar + 3 cameras → /scan_fused)
    # ========================================================================

    scan_fusion_node = TimerAction(
        period=22.0,
        actions=[
            LogInfo(msg='[FUSION] Starting scan_fusion_v9 (camera wins if closer)'),
            Node(
                package='wheelchair_localization', executable='scan_fusion_v9',
                name='scan_fusion', output='screen',
                parameters=[{
                    'scan_topic': '/scan_filtered',
                    'output_topic': '/scan_fused',
                    'laser_frame': 'laser',
                    'min_height': 0.10, 'max_height': 1.80,
                    'max_camera_age_ms': 500.0, 'camera_warmup_sec': 3.0,
                    'downsample_stride': 4, 'min_camera_points_per_bin': 2,
                    'enable_footprint': True, 'rear_crop_deg': 180.0,
                    'front_camera.enabled': True,
                    'front_camera.topic': '/camera/depth/color/points',
                    'front_camera.frame': 'camera_depth_optical_frame',
                    'front_camera.max_depth': 3.5, 'front_camera.min_depth': 0.30,
                    'left_camera.enabled': True,
                    'left_camera.topic': '/mapping_camera/depth/color/points',
                    'left_camera.frame': 'mapping_camera_depth_optical_frame',
                    'left_camera.max_depth': 3.0, 'left_camera.min_depth': 0.30,
                    'right_camera.enabled': True,
                    'right_camera.topic': '/right_camera/depth/color/points',
                    'right_camera.frame': 'right_camera_depth_optical_frame',
                    'right_camera.max_depth': 3.0, 'right_camera.min_depth': 0.30,
                }],
            ),
        ],
        condition=UnlessCondition(is_sim)
    )

    # ========================================================================
    # STATIC TRANSFORMS + JOINT STATE PUBLISHER
    # ========================================================================

    static_transform_publisher = Node(
        package='tf2_ros', executable='static_transform_publisher',
        arguments=["--x", "0", "--y", "0", "--z", "0",
                   "--qx", "0", "--qy", "0", "--qz", "0", "--qw", "1",
                   "--frame-id", "base_link", "--child-frame-id", "imu"]
    )

    robot_description_for_jsp = Command([
        'xacro ',
        os.path.join(wheelchair_description_dir, 'urdf', 'wheelchair_description.urdf.xacro'),
        ' is_sim:=false port:=/dev/ttyACM0'
    ])
    joint_state_publisher = Node(
        package='joint_state_publisher', executable='joint_state_publisher',
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

    robust_ekf_zupt_node = TimerAction(
        period=10.0,
        actions=[
            LogInfo(msg='[EKF+ZUPT] Starting Robust EKF + ZUPT Sensor Fusion...'),
            Node(
                package='wheelchair_zupt', executable='robust_ekf_zupt_node',
                name='robust_ekf_zupt', output='screen',
                parameters=[{
                    'imu_topic': '/imu', 'odom_topic': '/wc_control/odom',
                    'odom_frame': 'odom', 'base_frame': 'base_link',
                    'publish_tf': True,
                    'stationary_v_thresh': 0.005, 'stationary_omega_thresh': 0.005,
                    'accel_gravity_tolerance': 0.3, 'accel_xy_thresh': 0.15,
                    'min_stationary_samples': 15, 'hysteresis_samples': 8,
                    'initial_gyro_bias': 0.0,
                    'bias_alpha_stationary': 0.02, 'bias_alpha_moving': 0.0005,
                    'bias_min_samples': 20,
                    'sigma_v': 0.05, 'sigma_omega': 0.02, 'sigma_bias': 0.0001,
                    'sigma_enc_v': 0.02, 'sigma_enc_omega': 0.015,
                    'sigma_gyro': 0.005,
                    'sigma_zupt_v': 0.001, 'sigma_zupt_omega': 0.001,
                    'sigma_imu_yaw': 0.03,
                    'use_imu_orientation': True, 'orientation_update_rate': 2.0,
                    'mahalanobis_threshold': 5.0,
                    'initial_x': 0.0, 'initial_y': 0.0, 'initial_theta': 0.0,
                }]
            )
        ],
        condition=UnlessCondition(is_sim)
    )

    # ========================================================================
    # RTAB-MAP SLAM — HOSPITAL: 3-CAMERA VISUAL + 25m LIDAR
    # ========================================================================
    #
    # Architecture:
    #   Lidar (/scan_filtered) → ICP registration (scan-to-scan matching)
    #   3 Cameras (rgbd_image0/1/2) → Visual features → bag-of-words loop closure
    #
    # Why both: Lidar gives precise 25m geometry. Cameras see room numbers,
    # signs, floor patterns — unique visual features that solve corridor drift.
    #
    # RTAB-Map params use "Section/Param" naming (passed as dict, NOT yaml).

    RTABMAP_HOSPITAL_PARAMS = {
        # --- Registration: ICP for scan matching (lidar) ---
        'Reg/Strategy': '1',           # 1=ICP (uses lidar scans)
        'Reg/Force3DoF': 'true',       # 2D mode (x, y, yaw)

        # --- Graph node creation ---
        'RGBD/LinearUpdate': '0.15',   # 15cm travel → new node
        'RGBD/AngularUpdate': '0.1',   # ~5.7° → new node
        'RGBD/CreateOccupancyGrid': 'true',
        'RGBD/OptimizeFromGraphEnd': 'false',

        # --- Loop closure (proximity + visual) ---
        'RGBD/ProximityBySpace': 'true',       # ICP proximity loop closure
        'RGBD/ProximityMaxGraphDepth': '0',    # Unlimited search
        'RGBD/ProximityPathMaxNeighbors': '10',
        'RGBD/NeighborLinkRefining': 'true',
        'RGBD/MaxLocalRetrieved': '3',
        'RGBD/ProximityOdomGuess': 'true',

        # --- Visual features (from 3 cameras) ---
        'Vis/FeatureType': '6',        # ORB (fast, CPU-efficient for 3 cameras)
        'Vis/MaxFeatures': '1000',     # Features per image
        'Vis/MinInliers': '15',        # Min matches for loop closure acceptance

        # --- ICP scan matching (lidar) ---
        'Icp/VoxelSize': '0.05',
        'Icp/MaxCorrespondenceDistance': '0.15',
        'Icp/PointToPlane': 'false',   # 2D point-to-point
        'Icp/Iterations': '30',
        'Icp/Epsilon': '0.001',
        'Icp/MaxTranslation': '0.5',   # Reject jumps > 50cm
        'Icp/RangeMax': '25.0',        # *** FULL 25m LIDAR ***
        'Icp/RangeMin': '0.15',
        'Icp/CorrespondenceRatio': '0.1',

        # --- Occupancy grid (from lidar scan) ---
        'Grid/FromDepth': 'false',     # Grid from SCAN, not depth
        'Grid/RangeMax': '25.0',       # Full 25m lidar range
        'Grid/RangeMin': '0.15',
        'Grid/CellSize': '0.05',       # 5cm resolution
        'Grid/MaxObstacleHeight': '2.0',
        'Grid/MinClusterSize': '3',

        # --- Optimizer ---
        'Optimizer/Strategy': '1',     # g2o
        'Optimizer/Iterations': '20',
        'Optimizer/GravitySigma': '0.0',

        # --- Memory ---
        'Mem/STMSize': '30',
        'Mem/LaserScanNormalK': '0',
    }

    def _make_rtabmap(context, *_args, **_kwargs):
        """Create RTAB-Map node: lidar ICP + optional 3-camera visual loop closure."""
        fused = LaunchConfiguration('use_fused_slam').perform(context) == 'true'
        cameras = LaunchConfiguration('use_cameras').perform(context) == 'true'
        sim = LaunchConfiguration('is_sim').perform(context) == 'true'
        if sim:
            return []

        scan_topic = '/scan_fused' if fused else '/scan_filtered'
        db_path = LaunchConfiguration('database_path').perform(context)
        delete_db = LaunchConfiguration('delete_db_on_start').perform(context) == 'true'

        rtabmap_args = []
        if delete_db:
            rtabmap_args.append('--delete_db_on_start')

        # Base params
        node_params = {
            'subscribe_scan': True,
            'frame_id': 'base_link',
            'map_frame_id': 'map',
            'odom_frame_id': 'odom',       # Get odom from TF (EKF publishes odom→base_link)
            'publish_tf': True,
            'wait_for_transform': 0.3,
            'database_path': db_path,
            'approx_sync': True,
            'use_sim_time': False,
            'Mem/IncrementalMemory': 'true',
            'Mem/InitWMWithAllNodes': 'false',
        }

        # Remappings
        remaps = [
            ('scan', scan_topic),
            ('odom', '/odometry/filtered'),
            ('map', '/map'),
        ]

        if cameras:
            # 3-camera visual features for loop closure
            node_params['subscribe_rgbd'] = True
            node_params['rgbd_cameras'] = 3
            node_params['subscribe_depth'] = False
            node_params['subscribe_rgb'] = False
            remaps.extend([
                ('rgbd_image0', '/front_rgbd_image'),
                ('rgbd_image1', '/left_rgbd_image'),
                ('rgbd_image2', '/right_rgbd_image'),
            ])
            mode_label = 'LIDAR ICP + 3-CAMERA VISUAL LOOP CLOSURE'
        else:
            # Lidar-only (no camera visual features)
            node_params['subscribe_rgbd'] = False
            node_params['subscribe_depth'] = False
            node_params['subscribe_rgb'] = False
            mode_label = 'LIDAR-ONLY ICP (no visual loop closure)'

        node_params.update(RTABMAP_HOSPITAL_PARAMS)

        scan_label = '/scan_fused' if fused else '/scan_filtered'

        return [
            LogInfo(msg='=' * 60),
            LogInfo(msg=f'[RTAB-MAP] {mode_label}'),
            LogInfo(msg=f'[RTAB-MAP] Scan: {scan_label} | Range: 25m | Grid: 5cm'),
            LogInfo(msg=f'[RTAB-MAP] Database: {db_path}'),
            LogInfo(msg='=' * 60),
            Node(
                package='rtabmap_slam',
                executable='rtabmap',
                name='rtabmap',
                output='screen',
                parameters=[node_params],
                remappings=remaps,
                arguments=rtabmap_args,
            ),
        ]

    rtabmap_node = TimerAction(
        period=28.0,  # After all cameras + rgbd_sync + fusion ready
        actions=[OpaqueFunction(function=_make_rtabmap)],
        condition=UnlessCondition(is_sim)
    )

    # Map saver server
    map_saver_server = TimerAction(
        period=30.0,
        actions=[Node(
            package='nav2_map_server', executable='map_saver_server',
            name='map_saver_server', output='screen',
            parameters=[{
                'save_map_timeout': 5.0, 'use_sim_time': is_sim,
                'free_thresh_default': 0.196, 'occupied_thresh_default': 0.65,
            }],
        )],
        condition=UnlessCondition(is_sim)
    )

    map_saver_lifecycle = TimerAction(
        period=34.0,
        actions=[Node(
            package='nav2_lifecycle_manager', executable='lifecycle_manager',
            name='lifecycle_manager_map_saver', output='screen',
            parameters=[{
                'node_names': ['map_saver_server'],
                'use_sim_time': is_sim,
                'autostart': True, 'bond_timeout': 10.0,
            }],
        )],
        condition=UnlessCondition(is_sim)
    )

    # ========================================================================
    # DATA LOGGERS
    # ========================================================================

    topic_data_logger = Node(
        package='scripts', executable='topic_data_logger',
        name='topic_data_logger', output='screen',
        parameters=[{
            'imu_topic': '/imu', 'raw_odom_topic': '/wc_control/odom',
            'filtered_odom_topic': '/odometry/filtered',
            'log_frequency_hz': 10.0, 'file_prefix': 'rtabmap_hospital_log'
        }]
    )

    scan_data_logger = TimerAction(
        period=24.0,
        actions=[Node(
            package='scripts', executable='scan_data_logger',
            name='scan_data_logger', output='screen',
            parameters=[{
                'log_frequency_hz': 10.0, 'log_ranges': False,
                'log_raw': True, 'log_filtered': True, 'log_lidar_only': True,
                'log_front_cam': True, 'log_left_cam': True, 'log_right_cam': True,
                'log_fused': True,
            }]
        )],
        condition=UnlessCondition(is_sim)
    )

    # ========================================================================
    # RVIZ
    # ========================================================================

    rviz_node = TimerAction(
        period=38.0,
        actions=[
            LogInfo(msg='[RVIZ] Starting RViz...'),
            Node(
                package='rviz2', executable='rviz2', name='rviz2', output='screen',
                arguments=['-d', LaunchConfiguration('rviz_config')],
                parameters=[{'use_sim_time': is_sim}],
            )
        ],
        condition=IfCondition(LaunchConfiguration('rviz')),
    )

    ready_message = TimerAction(
        period=40.0,
        actions=[
            LogInfo(msg='=' * 60),
            LogInfo(msg='  RTAB-MAP HOSPITAL MAPPING READY'),
            LogInfo(msg='=' * 60),
            LogInfo(msg='  SLAM: RTAB-Map (ICP + 3-camera visual loop closure)'),
            LogInfo(msg='  Lidar: RPLidar S3 at 25m (ICP registration)'),
            LogInfo(msg='  Cameras: 3x RealSense RGB→ORB features→loop closure'),
            LogInfo(msg='  Grid: 5cm resolution, full 25m range'),
            LogInfo(msg=''),
            LogInfo(msg='  SAVE MAP:'),
            LogInfo(msg='    ros2 run nav2_map_server map_saver_cli \\'),
            LogInfo(msg='      -f ~/wheelchair_nav/maps/hospital_map'),
            LogInfo(msg=''),
            LogInfo(msg='  DB VIEWER: rtabmap-databaseViewer <db_path>'),
            LogInfo(msg='=' * 60),
        ],
    )

    # ========================================================================
    # LAUNCH DESCRIPTION
    # ========================================================================

    return LaunchDescription([
        # Arguments
        declare_is_sim, declare_port, declare_model, declare_is_ignition,
        declare_use_sim_time, declare_world, declare_unite_imu,
        declare_lidar_port, declare_sudo_password,
        declare_rviz, declare_rviz_config,
        declare_use_fused_slam, declare_use_cameras,
        declare_database_path, declare_delete_db,

        # USB permissions
        permission_setup,

        # Motion stack
        unified_wheelchair_launch,

        # RPLidar S3 + hospital laser filter (25m)
        rplidar_s3_launch,
        laser_filter_node,

        # 3 cameras (RGB + depth + pointcloud, staggered)
        camera_setup,

        # IMU pipeline
        imu_startup_calibrator, imu_bias_corrector,
        imu_filter, imu_wheelchair_republisher,

        # RGBD Sync preprocessing (bundles RGB+depth per camera for RTAB-Map)
        front_rgbd_sync,    # 16s
        left_rgbd_sync,     # 18s
        right_rgbd_sync,    # 20s

        # Scan fusion (lidar + cameras → /scan_fused)
        scan_fusion_node,   # 22s

        # Static TFs + joint state publisher
        static_transform_publisher,
        joint_state_publisher,

        # EKF + ZUPT
        robust_ekf_zupt_node,  # 10s

        # RTAB-Map SLAM (28s — after all preprocessing ready)
        rtabmap_node,
        map_saver_server,
        map_saver_lifecycle,

        # Loggers
        topic_data_logger,
        scan_data_logger,

        # Visualization
        rviz_node,
        ready_message,
    ])
