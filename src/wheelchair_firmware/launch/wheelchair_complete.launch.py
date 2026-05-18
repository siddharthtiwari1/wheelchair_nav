#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch_ros.actions import Node, SetParameter
from ament_index_python.packages import get_package_share_directory
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    
    # RTAB-Map parameters
    rtabmap_parameters = [{
        'frame_id': 'camera_link',
        'subscribe_depth': True,
        'subscribe_odom_info': True,
        'approx_sync': False,
        'wait_imu_to_init': True
    }]

    # RTAB-Map remappings
    rtabmap_remappings = [
        ('imu', '/imu/data'),
        ('rgb/image', '/camera/color/image_raw'),
        ('rgb/camera_info', '/camera/color/camera_info'),
        ('depth/image', '/camera/aligned_depth_to_color/image_raw')
    ]

    # Launch arguments
    unite_imu_method_arg = DeclareLaunchArgument(
        'unite_imu_method', 
        default_value='2',
        description='0-None, 1-copy, 2-linear_interpolation. Use unite_imu_method:="1" if imu topics stop being published.'
    )
    
    serial_port_arg = DeclareLaunchArgument(
        'serial_port',
        default_value='/dev/ttyACM1',
        description='Serial port for Arduino connection'
    )
    
    use_camera_arg = DeclareLaunchArgument(
        'use_camera',
        default_value='true',
        description='Whether to launch RealSense camera'
    )
    
    use_rtabmap_arg = DeclareLaunchArgument(
        'use_rtabmap',
        default_value='true',
        description='Whether to launch RTAB-Map SLAM'
    )
    

    # Hardware Interface (Arduino communication)
    hardware_interface = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(
                get_package_share_directory("wheelchair_firmware"),
                "launch",
                "hardware_interface.launch.py"
            )
        ]),
        launch_arguments={
            'serial_port': LaunchConfiguration('serial_port'),
        }.items(),
    )
    
    # Wheelchair Controller (without differential drive controller - using Arduino odometry)
    controller = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(
                get_package_share_directory("wc_control"),
                "launch",
                "wheelchair_controller_no_odom.py"
            )
        ])
    )

    # RealSense Camera Launch
    camera_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(get_package_share_directory('realsense2_camera'), 'launch'),
            '/rs_launch.py'
        ]),
        launch_arguments={
            'camera_namespace': '',
            'enable_gyro': 'true',
            'enable_accel': 'true',
            'unite_imu_method': LaunchConfiguration('unite_imu_method'),
            'align_depth.enable': 'true',
            'enable_sync': 'true',
            'rgb_camera.profile': '640x360x30'
        }.items(),
        condition=lambda context: LaunchConfiguration('use_camera').perform(context) == 'true'
    )

    # IMU Filter (Madgwick)
    imu_filter_node = Node(
        package='imu_filter_madgwick', 
        executable='imu_filter_madgwick_node', 
        output='screen',
        parameters=[{
            'use_mag': False, 
            'world_frame': 'enu', 
            'publish_tf': False
        }],
        remappings=[('imu/data_raw', '/camera/imu')],
        condition=lambda context: LaunchConfiguration('use_camera').perform(context) == 'true'
    )

    # RTAB-Map Odometry
    rtabmap_odom_node = Node(
        package='rtabmap_odom', 
        executable='rgbd_odometry', 
        output='screen',
        parameters=rtabmap_parameters,
        remappings=rtabmap_remappings,
        condition=lambda context: LaunchConfiguration('use_rtabmap').perform(context) == 'true'
    )

    # RTAB-Map SLAM
    rtabmap_slam_node = Node(
        package='rtabmap_slam', 
        executable='rtabmap', 
        output='screen',
        parameters=rtabmap_parameters,
        remappings=rtabmap_remappings,
        arguments=['-d'],
        condition=lambda context: LaunchConfiguration('use_rtabmap').perform(context) == 'true'
    )

    # RTAB-Map Visualization
    rtabmap_viz_node = Node(
        package='rtabmap_viz', 
        executable='rtabmap_viz', 
        output='screen',
        parameters=rtabmap_parameters,
        remappings=rtabmap_remappings,
        condition=lambda context: LaunchConfiguration('use_rtabmap').perform(context) == 'true'
    )


    return LaunchDescription([
        # Launch arguments
        unite_imu_method_arg,
        serial_port_arg,
        use_camera_arg,
        use_rtabmap_arg,

        # Make sure IR emitter is enabled for RealSense
        SetParameter(name='depth_module.emitter_enabled', value=1),

        # Core wheelchair components
        hardware_interface,
        controller,

        # Camera and perception
        camera_launch,
        imu_filter_node,

        # SLAM and mapping
        rtabmap_odom_node,
        rtabmap_slam_node,
        rtabmap_viz_node,
    ])