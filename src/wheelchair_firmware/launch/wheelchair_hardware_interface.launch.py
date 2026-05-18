import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch.substitutions import Command
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    # Robot description with wheelchair hardware interface
    robot_description = ParameterValue(
        Command(
            [
                "xacro ",
                os.path.join(
                    get_package_share_directory("wheelchair_description"),
                    "urdf",
                    "wheelchair_description.urdf",
                ),
            ]
        ),
        value_type=str,
    )

    # Robot state publisher
    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[{"robot_description": robot_description}],
        output="screen"
    )

    # Controller manager with wheelchair hardware interface
    controller_manager = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[
            {"robot_description": robot_description,
             "use_sim_time": False},
            os.path.join(
                get_package_share_directory("wc_control"),
                "config",
                "wc_control.yaml",
            ),
        ],
        output="screen",
        remappings=[
            ("/tf", "tf"),
            ("/tf_static", "tf_static"),
        ]
    )

    # Spawn wc_control controller
    spawn_wc_controller = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["wc_control"],
        output="screen"
    )

    # Spawn joint state broadcaster
    spawn_joint_broadcaster = Node(
        package="controller_manager",
        executable="spawner", 
        arguments=["joint_state_broadcaster"],
        output="screen"
    )

    return LaunchDescription(
        [
            robot_state_publisher_node,
            controller_manager,
            spawn_wc_controller,
            spawn_joint_broadcaster,
        ]
    )