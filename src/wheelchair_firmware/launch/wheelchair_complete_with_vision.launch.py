#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch.actions import (
    IncludeLaunchDescription,
    DeclareLaunchArgument,
    TimerAction,
    GroupAction
)
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node, SetParameter
from ament_index_python.packages import get_package_share_directory
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():

    # ==================== LAUNCH ARGUMENTS ====================
    use_slam = LaunchConfiguration('use_slam')
    use_nav = LaunchConfiguration('use_nav')
    use_rviz = LaunchConfiguration('use_rviz')
    use_sim_time = LaunchConfiguration('use_sim_time')
    is_sim = LaunchConfiguration('is_sim')
    port = LaunchConfiguration('port')
    unite_imu_method = LaunchConfiguration('unite_imu_method')
    enable_sensor_fusion = LaunchConfiguration('enable_sensor_fusion')
    map_name = LaunchConfiguration('map_name')

    declare_args = [
        DeclareLaunchArgument(
            'use_slam', default_value='false',
            description='Whether to run SLAM (true) or localization (false)'
        ),
        DeclareLaunchArgument(
            'use_nav', default_value='true',
            description='Whether to run navigation stack'
        ),
        DeclareLaunchArgument(
            'use_rviz', default_value='true',
            description='Whether to launch RViz'
        ),
        DeclareLaunchArgument(
            'use_sim_time', default_value='false',
            description='Use simulation time'
        ),
        DeclareLaunchArgument(
            'is_sim', default_value='false',
            description='Whether running in simulation'
        ),
        DeclareLaunchArgument(
            'port', default_value='/dev/ttyACM0',
            description='Serial port for Arduino'
        ),
        DeclareLaunchArgument(
            'unite_imu_method', default_value='2',
            description='0-None, 1-copy, 2-linear_interpolation for RealSense IMU'
        ),
        DeclareLaunchArgument(
            'enable_sensor_fusion', default_value='true',
            description='Enable robot_localization EKF sensor fusion'
        ),
        DeclareLaunchArgument(
            'map_name', default_value='wheelchair_map',
            description='Map name for localization'
        )
    ]

    # ==================== RTAB-MAP PARAMETERS ====================
    rtabmap_parameters = [{
        'frame_id': 'camera_link',
        'subscribe_depth': True,
        'subscribe_odom_info': True,
        'approx_sync': False,
        'wait_imu_to_init': True,
        'use_sim_time': use_sim_time,
        # RTAB-Map specific parameters for wheelchair
        'Mem/IncrementalMemory': 'true',
        'Mem/InitWMWithAllNodes': 'false',
        'RGBD/NeighborLinkRefining': 'true',
        'Grid/FromDepth': 'false',  # Use laser scan for occupancy grid
        'RGBD/ProximityBySpace': 'true',
        'RGBD/ProximityMaxGraphDepth': '50',
        'RGBD/ProximityPathMaxNeighbors': '10',
        'Reg/Strategy': '1',  # Visual + ICP registration
        'Vis/EstimationType': '1',  # 3D->2D (PnP)
        'Vis/MaxDepth': '4.0',
        'Optimizer/GravitySigma': '0.3'
    }]

    rtabmap_remappings = [
        ('imu', '/imu/data'),
        ('rgb/image', '/camera/color/image_raw'),
        ('rgb/camera_info', '/camera/color/camera_info'),
        ('depth/image', '/camera/aligned_depth_to_color/image_raw'),
        ('scan', '/scan'),  # Add laser scan
        ('odom', '/odometry/filtered')  # Use filtered odometry from EKF
    ]

    # ==================== HARDWARE INTERFACE ====================
    hardware_interface = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("wheelchair_firmware"),
                "launch",
                "ros2_control_hardware_interface.launch.py"
            )
        ),
        launch_arguments={
            'is_sim': is_sim,
            'port': port,
            'use_sim_time': use_sim_time
        }.items(),
        condition=UnlessCondition(is_sim)
    )

    # ==================== REALSENSE CAMERA ====================
    realsense_camera = GroupAction([
        # Enable IR emitter for better depth quality
        SetParameter(name='depth_module.emitter_enabled', value=1),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                os.path.join(get_package_share_directory('realsense2_camera'), 'launch'),
                '/rs_launch.py'
            ]),
            launch_arguments={
                'camera_namespace': '',
                'enable_gyro': 'true',
                'enable_accel': 'true',
                'unite_imu_method': unite_imu_method,
                'align_depth.enable': 'true',
                'enable_sync': 'true',
                'rgb_camera.profile': '640x360x30',
                'depth_module.profile': '640x360x30',
                'enable_pointcloud': 'true',
                'pointcloud.allow_no_texture_points': 'true',
                'use_sim_time': use_sim_time
            }.items()
        )
    ])

    # ==================== IMU PROCESSING ====================
    # IMU filter for orientation estimation (like bumperbot)
    imu_filter = Node(
        package='imu_filter_madgwick',
        executable='imu_filter_madgwick_node',
        name='imu_filter',
        output='screen',
        parameters=[{
            'use_mag': False,
            'world_frame': 'enu',
            'publish_tf': False,
            'use_sim_time': use_sim_time,
            'stateless': False,
            'constant_dt': 0.0,
            'publish_debug_topics': True
        }],
        remappings=[
            ('imu/data_raw', '/camera/imu'),
            ('imu/data', '/imu/data')
        ]
    )

    # IMU republisher for frame conversion (like bumperbot pattern)
    imu_republisher = Node(
        package='wheelchair_firmware',
        executable='imu_republisher.py',
        name='imu_republisher',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
        remappings=[
            ('imu_in', '/imu/data'),
            ('imu_out', '/imu/data_ekf')
        ]
    )

    # ==================== SENSOR FUSION (EKF) ====================
    ekf_localization = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node',
        output='screen',
        parameters=[
            os.path.join(get_package_share_directory('wheelchair_firmware'), 'config', 'ekf.yaml'),
            {'use_sim_time': use_sim_time}
        ],
        remappings=[
            ('odometry/filtered', '/odometry/filtered'),
            ('/set_pose', '/initialpose')
        ],
        condition=IfCondition(enable_sensor_fusion)
    )

    # ==================== LIDAR ====================
    # Add your LiDAR driver here (replace with your actual LiDAR)
    lidar_driver = Node(
        package='rplidar_ros',
        executable='rplidar_node',
        name='rplidar_node',
        output='screen',
        parameters=[{
            'serial_port': '/dev/ttyUSB0',
            'serial_baudrate': 115200,
            'frame_id': 'lidar',
            'inverted': False,
            'angle_compensate': True,
            'scan_mode': 'Standard',
            'use_sim_time': use_sim_time
        }]
    )

    # ==================== RTAB-MAP NODES ====================
    # RTAB-Map odometry
    rtabmap_odom = Node(
        package='rtabmap_odom',
        executable='rgbd_odometry',
        name='rgbd_odometry',
        output='screen',
        parameters=rtabmap_parameters,
        remappings=rtabmap_remappings,
        condition=IfCondition(use_slam)
    )

    # RTAB-Map SLAM
    rtabmap_slam = Node(
        package='rtabmap_slam',
        executable='rtabmap',
        name='rtabmap',
        output='screen',
        parameters=rtabmap_parameters,
        remappings=rtabmap_remappings,
        arguments=['-d'],  # Delete database on startup
        condition=IfCondition(use_slam)
    )

    # RTAB-Map visualization
    rtabmap_viz = Node(
        package='rtabmap_viz',
        executable='rtabmap_viz',
        name='rtabmap_viz',
        output='screen',
        parameters=rtabmap_parameters,
        remappings=rtabmap_remappings,
        condition=IfCondition(PythonExpression([use_slam, ' and ', use_rviz]))
    )

    # ==================== LOCALIZATION (Alternative to SLAM) ====================
    localization = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("wheelchair_firmware"),
                "launch",
                "localization.launch.py"
            )
        ),
        launch_arguments={
            'map_name': map_name,
            'use_sim_time': use_sim_time
        }.items(),
        condition=UnlessCondition(use_slam)
    )

    # ==================== NAVIGATION ====================
    navigation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("wheelchair_firmware"),
                "launch",
                "navigation.launch.py"
            )
        ),
        launch_arguments={
            'use_sim_time': use_sim_time
        }.items(),
        condition=IfCondition(use_nav)
    )

    # ==================== VISUALIZATION ====================
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', os.path.join(
            get_package_share_directory('wheelchair_firmware'),
            'config',
            'wheelchair_nav.rviz'
        )],
        parameters=[{'use_sim_time': use_sim_time}],
        condition=IfCondition(PythonExpression([use_rviz, ' and not ', use_slam]))
    )

    # ==================== DELAYED STARTUP SEQUENCE ====================
    # Start hardware first, then sensors, then processing
    delayed_realsense = TimerAction(
        period=2.0,
        actions=[realsense_camera]
    )

    delayed_imu_processing = TimerAction(
        period=5.0,
        actions=[imu_filter, imu_republisher]
    )

    delayed_ekf = TimerAction(
        period=8.0,
        actions=[ekf_localization]
    )

    delayed_rtabmap = TimerAction(
        period=10.0,
        actions=[rtabmap_odom, rtabmap_slam, rtabmap_viz]
    )

    delayed_nav = TimerAction(
        period=12.0,
        actions=[localization, navigation, rviz]
    )

    # ==================== LAUNCH DESCRIPTION ====================
    return LaunchDescription([
        # Arguments
        *declare_args,

        # Core hardware (starts immediately)
        hardware_interface,
        lidar_driver,

        # Delayed startup sequence
        delayed_realsense,
        delayed_imu_processing,
        delayed_ekf,
        delayed_rtabmap,
        delayed_nav
    ])