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
    """
    Professional wheelchair launch system integrating with existing localization package.
    Uses intern's sensor fusion work instead of duplicating functionality.
    """

    # ==================== LAUNCH ARGUMENTS ====================
    use_slam = LaunchConfiguration('use_slam')
    use_nav = LaunchConfiguration('use_nav')
    use_rviz = LaunchConfiguration('use_rviz')
    use_sim_time = LaunchConfiguration('use_sim_time')
    is_sim = LaunchConfiguration('is_sim')
    port = LaunchConfiguration('port')
    unite_imu_method = LaunchConfiguration('unite_imu_method')
    use_localization = LaunchConfiguration('use_localization')
    use_vision = LaunchConfiguration('use_vision')

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
            'use_localization', default_value='true',
            description='Enable intern\'s localization package (EKF + Custom Kalman)'
        ),
        DeclareLaunchArgument(
            'use_vision', default_value='true',
            description='Enable RealSense camera and RTAB-Map'
        )
    ]

    # ==================== CORE HARDWARE INTERFACE ====================
    # This is your professional ros2_control hardware interface
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

    # ==================== INTERN'S LOCALIZATION SYSTEM ====================
    # Use the existing, working localization package
    localization_system = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("wheelchair_localization"),
                "launch",
                "localization.launch.py"
            )
        ),
        launch_arguments={
            'use_sim_time': use_sim_time
        }.items(),
        condition=IfCondition(use_localization)
    )

    # ==================== REALSENSE CAMERA (Optional) ====================
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
                'enable_pointcloud': 'false',  # Disabled for performance
                'use_sim_time': use_sim_time
            }.items()
        )
    ], condition=IfCondition(use_vision))

    # ==================== IMU PROCESSING FOR REALSENSE ====================
    # IMU filter for RealSense camera IMU (separate from wheel-based localization)
    realsense_imu_filter = Node(
        package='imu_filter_madgwick',
        executable='imu_filter_madgwick_node',
        name='realsense_imu_filter',
        output='screen',
        parameters=[{
            'use_mag': False,
            'world_frame': 'enu',
            'publish_tf': False,
            'use_sim_time': use_sim_time,
            'stateless': False,
            'constant_dt': 0.0
        }],
        remappings=[
            ('imu/data_raw', '/camera/imu'),
            ('imu/data', '/camera/imu/filtered')
        ],
        condition=IfCondition(use_vision)
    )

    # ==================== LIDAR DRIVER ====================
    # Add your LiDAR driver here
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
        }],
        condition=UnlessCondition(is_sim)
    )

    # ==================== RTAB-MAP (Optional for SLAM) ====================
    rtabmap_parameters = [{
        'frame_id': 'camera_link',
        'subscribe_depth': True,
        'subscribe_scan': True,
        'subscribe_odom_info': True,
        'approx_sync': False,
        'wait_imu_to_init': True,
        'use_sim_time': use_sim_time,
        # Optimized for wheelchair
        'Mem/IncrementalMemory': 'true',
        'Mem/InitWMWithAllNodes': 'false',
        'RGBD/NeighborLinkRefining': 'true',
        'Grid/FromDepth': 'false',  # Use laser for occupancy
        'Reg/Strategy': '1',  # Visual + ICP
        'Vis/EstimationType': '1',  # 3D->2D (PnP)
        'Vis/MaxDepth': '4.0',
        'Optimizer/GravitySigma': '0.3'
    }]

    rtabmap_remappings = [
        ('scan', '/scan'),
        ('rgb/image', '/camera/color/image_raw'),
        ('rgb/camera_info', '/camera/color/camera_info'),
        ('depth/image', '/camera/aligned_depth_to_color/image_raw'),
        ('odom', '/odometry/filtered')  # Use intern's fused odometry!
    ]

    # RTAB-Map SLAM (only if explicitly requested)
    rtabmap_slam = Node(
        package='rtabmap_slam',
        executable='rtabmap',
        name='rtabmap',
        output='screen',
        parameters=rtabmap_parameters,
        remappings=rtabmap_remappings,
        arguments=['-d'],  # Delete database on startup
        condition=IfCondition(PythonExpression([use_slam, ' and ', use_vision]))
    )

    # RTAB-Map visualization
    rtabmap_viz = Node(
        package='rtabmap_viz',
        executable='rtabmap_viz',
        name='rtabmap_viz',
        output='screen',
        parameters=rtabmap_parameters,
        remappings=rtabmap_remappings,
        condition=IfCondition(PythonExpression([use_slam, ' and ', use_vision, ' and ', use_rviz]))
    )

    # ==================== NAVIGATION STACK ====================
    # Your navigation system (if you have one)
    navigation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("wc_control"),  # Or wherever your nav is
                "launch",
                "wheelchair_controller.py"
            )
        ),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'use_simple_controller': 'False',
            'use_python': 'False'
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
            'wheelchair_integrated.rviz'
        )],
        parameters=[{'use_sim_time': use_sim_time}],
        condition=IfCondition(PythonExpression([use_rviz, ' and not ', use_slam]))
    )

    # ==================== STAGED STARTUP SEQUENCE ====================
    # Professional startup sequence to avoid conflicts

    # Stage 1: Core hardware
    stage1_hardware = GroupAction([
        hardware_interface,
        lidar_driver
    ])

    # Stage 2: Sensors (3 seconds later)
    stage2_sensors = TimerAction(
        period=3.0,
        actions=[realsense_camera, realsense_imu_filter]
    )

    # Stage 3: Localization (5 seconds later)
    stage3_localization = TimerAction(
        period=5.0,
        actions=[localization_system]
    )

    # Stage 4: High-level systems (8 seconds later)
    stage4_highlevel = TimerAction(
        period=8.0,
        actions=[rtabmap_slam, navigation]
    )

    # Stage 5: Visualization (10 seconds later)
    stage5_viz = TimerAction(
        period=10.0,
        actions=[rtabmap_viz, rviz]
    )

    # ==================== LAUNCH DESCRIPTION ====================
    return LaunchDescription([
        # Arguments
        *declare_args,

        # Staged startup
        stage1_hardware,
        stage2_sensors,
        stage3_localization,
        stage4_highlevel,
        stage5_viz
    ])