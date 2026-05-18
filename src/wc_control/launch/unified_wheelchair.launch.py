#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Launch arguments
    is_sim_arg = DeclareLaunchArgument(
        'is_sim', 
        default_value='true',
        description='Use simulation mode (true) or real hardware (false)'
    )
    
    port_arg = DeclareLaunchArgument(
        'port', 
        default_value='/dev/ttyACM1',
        description='Arduino port for real hardware'
    )
    
    world_arg = DeclareLaunchArgument(
        'world',
        default_value='empty.sdf',
        description='Gazebo world file'
    )

    controller_config_arg = DeclareLaunchArgument(
        'controller_config',
        default_value=os.path.join(get_package_share_directory('wc_control'), 'config', 'wc_control_safe_v2.yaml'),
        description='DiffDriveController config yaml'
    )

    is_sim = LaunchConfiguration('is_sim')
    port = LaunchConfiguration('port')
    world = LaunchConfiguration('world')
    controller_config = LaunchConfiguration('controller_config')

    # Robot description - unified approach
    robot_description_content = Command([
        'xacro ',
        os.path.join(get_package_share_directory('wheelchair_description'), 'urdf', 'wheelchair_description.urdf.xacro'),
        ' is_sim:=', is_sim,
        ' port:=', port
    ])

    robot_description = {'robot_description': ParameterValue(robot_description_content, value_type=str)}

    # Robot State Publisher - unified for both modes
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[robot_description, {'use_sim_time': is_sim}],
        output='screen'
    )

    # Include Gazebo launch (only if simulation)
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(get_package_share_directory('wheelchair_description'), 'launch', 'gazebo_launch.py')
        ]),
        condition=IfCondition(is_sim)
    )

    # Controller Manager + Hardware Interface - runs for both simulation and real hardware
    controller_manager_node = Node(
        package='controller_manager',
        executable='ros2_control_node',
        parameters=[
            robot_description,
            controller_config,
            {'use_sim_time': is_sim}
        ],
        output='screen'
    )

    # Joint State Broadcaster Spawner
    joint_state_broadcaster_spawner_node = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster', '--controller-manager-timeout', '30'],
        parameters=[{'use_sim_time': is_sim}],
        output='screen'
    )

    # WC Control Controller Spawner
    wc_control_spawner_node = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['wc_control', '--controller-manager-timeout', '30'],
        parameters=[{'use_sim_time': is_sim}],
        output='screen'
    )

    # Delay controller bring-up so robot_description is latched before ros2_control starts
    controller_manager = TimerAction(
        period=2.0,
        actions=[controller_manager_node]
    )

    joint_state_broadcaster_spawner = TimerAction(
        period=4.0,
        actions=[joint_state_broadcaster_spawner_node]
    )

    wc_control_spawner = TimerAction(
        period=4.5,
        actions=[wc_control_spawner_node]
    )

    # IMU Sensor Broadcaster Spawner - DISABLED (using direct Gazebo IMU)
    # imu_sensor_broadcaster_spawner = Node(
    #     package='controller_manager',
    #     executable='spawner',
    #     arguments=['imu_sensor_broadcaster', '--controller-manager-timeout', '30'],
    #     parameters=[{'use_sim_time': is_sim}],
    #     output='screen'
    # )

    # Teleop keyboard (only for simulation)
    teleop_keyboard = Node(
        package='teleop_twist_keyboard',
        executable='teleop_twist_keyboard',
        name='teleop_twist_keyboard',
        output='screen',
        prefix='xterm -e',
        parameters=[{'use_sim_time': is_sim}],
        condition=IfCondition(is_sim)  # Only runs in simulation
    )

    # Convert teleop Twist messages to TwistStamped for diff_drive_controller
    teleop_twist_converter = Node(
        package='scripts',
        executable='twist_stamped_teleop',
        name='twist_stamped_teleop',
        output='screen',
        parameters=[{'use_sim_time': is_sim}],
        remappings=[
            ('cmd_vel_in', 'cmd_vel'),
            ('cmd_vel_out', 'wc_control/cmd_vel'),
        ],
        condition=IfCondition(is_sim)
    )

    # Republish Gazebo IMU to /imu for EKF in simulation
    sim_imu_bridge = Node(
        package='scripts',
        executable='imu_out_to_imu',
        name='imu_out_to_imu',
        output='screen',
        parameters=[{'use_sim_time': is_sim}],
        condition=IfCondition(is_sim)
    )

    # Inject bias into simulated odom to highlight EKF corrections
    sim_odom_bias = Node(
        package='scripts',
        executable='sim_odom_bias',
        name='sim_odom_bias',
        output='screen',
        parameters=[
            {'use_sim_time': is_sim},
            {'input_topic': '/wc_control/odom_raw'},
            {'output_topic': '/wc_control/odom'},
            {'drift_per_meter': 0.02},
            {'drift_axis': 'y'}
        ],
        condition=IfCondition(is_sim)
    )

    return LaunchDescription([
        is_sim_arg,
        port_arg,
        world_arg,
        controller_config_arg,
        robot_state_publisher,
        gazebo_launch,  # Only launches if is_sim=true
        controller_manager,
        joint_state_broadcaster_spawner,
        wc_control_spawner,
        # imu_sensor_broadcaster_spawner,  # DISABLED - using direct Gazebo IMU
        teleop_keyboard,  # Only for simulation
        teleop_twist_converter,  # Convert teleop Twist to TwistStamped
        sim_imu_bridge,  # Simulation IMU bridge
        sim_odom_bias,  # Simulation odom drift injector
    ])
