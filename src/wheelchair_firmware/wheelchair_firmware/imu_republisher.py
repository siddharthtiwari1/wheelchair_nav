#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
import tf_transformations
import math


class IMURepublisher(Node):
    """
    IMU Republisher Node - Converts RealSense IMU data for EKF fusion

    Based on bumperbot approach but adapted for wheelchair with RealSense camera.
    Handles frame transformations and data conditioning for robot_localization.
    """

    def __init__(self):
        super().__init__('imu_republisher')

        # Parameters
        self.declare_parameter('target_frame', 'base_link')
        self.declare_parameter('filter_orientation', True)
        self.declare_parameter('orientation_stddev', 0.01)
        self.declare_parameter('angular_velocity_stddev', 0.02)
        self.declare_parameter('linear_acceleration_stddev', 0.04)

        self.target_frame = self.get_parameter('target_frame').value
        self.filter_orientation = self.get_parameter('filter_orientation').value
        self.orientation_stddev = self.get_parameter('orientation_stddev').value
        self.angular_velocity_stddev = self.get_parameter('angular_velocity_stddev').value
        self.linear_acceleration_stddev = self.get_parameter('linear_acceleration_stddev').value

        # Publishers and Subscribers
        self.imu_pub = self.create_publisher(Imu, 'imu_out', 10)
        self.imu_sub = self.create_subscription(Imu, 'imu_in', self.imu_callback, 10)

        # State for filtering
        self.last_orientation = None
        self.orientation_alpha = 0.8  # Low-pass filter coefficient

        self.get_logger().info(f'IMU Republisher started, target frame: {self.target_frame}')

    def imu_callback(self, msg):
        """
        Process incoming IMU data and republish with proper frame and covariance
        """
        try:
            # Create output message
            output_msg = Imu()
            output_msg.header = msg.header
            output_msg.header.frame_id = self.target_frame

            # Copy and process orientation
            if self.filter_orientation and self.last_orientation is not None:
                # Apply simple low-pass filter to orientation for stability
                output_msg.orientation = self.filter_orientation_data(
                    msg.orientation, self.last_orientation
                )
            else:
                output_msg.orientation = msg.orientation

            self.last_orientation = output_msg.orientation

            # Copy angular velocity and linear acceleration
            output_msg.angular_velocity = msg.angular_velocity
            output_msg.linear_acceleration = msg.linear_acceleration

            # Set realistic covariance matrices for wheelchair application
            self.set_covariance_matrices(output_msg)

            # Publish processed IMU data
            self.imu_pub.publish(output_msg)

        except Exception as e:
            self.get_logger().error(f'Error processing IMU data: {str(e)}')

    def filter_orientation_data(self, current_orientation, last_orientation):
        """
        Apply low-pass filter to orientation quaternion for stability
        """
        # Convert quaternions to euler angles for filtering
        current_euler = tf_transformations.euler_from_quaternion([
            current_orientation.x, current_orientation.y,
            current_orientation.z, current_orientation.w
        ])

        last_euler = tf_transformations.euler_from_quaternion([
            last_orientation.x, last_orientation.y,
            last_orientation.z, last_orientation.w
        ])

        # Apply low-pass filter to each axis
        filtered_euler = [
            self.orientation_alpha * current_euler[i] +
            (1.0 - self.orientation_alpha) * last_euler[i]
            for i in range(3)
        ]

        # Convert back to quaternion
        filtered_quat = tf_transformations.quaternion_from_euler(
            filtered_euler[0], filtered_euler[1], filtered_euler[2]
        )

        # Create geometry_msgs/Quaternion
        from geometry_msgs.msg import Quaternion
        result = Quaternion()
        result.x = filtered_quat[0]
        result.y = filtered_quat[1]
        result.z = filtered_quat[2]
        result.w = filtered_quat[3]

        return result

    def set_covariance_matrices(self, imu_msg):
        """
        Set appropriate covariance matrices for wheelchair IMU data
        """
        # Orientation covariance (9x9 matrix, row-major)
        orientation_variance = self.orientation_stddev ** 2
        imu_msg.orientation_covariance = [
            orientation_variance, 0.0, 0.0,
            0.0, orientation_variance, 0.0,
            0.0, 0.0, orientation_variance
        ]

        # Angular velocity covariance (3x3 matrix, row-major)
        angular_velocity_variance = self.angular_velocity_stddev ** 2
        imu_msg.angular_velocity_covariance = [
            angular_velocity_variance, 0.0, 0.0,
            0.0, angular_velocity_variance, 0.0,
            0.0, 0.0, angular_velocity_variance
        ]

        # Linear acceleration covariance (3x3 matrix, row-major)
        linear_acceleration_variance = self.linear_acceleration_stddev ** 2
        imu_msg.linear_acceleration_covariance = [
            linear_acceleration_variance, 0.0, 0.0,
            0.0, linear_acceleration_variance, 0.0,
            0.0, 0.0, linear_acceleration_variance
        ]


def main(args=None):
    rclpy.init(args=args)

    try:
        imu_republisher = IMURepublisher()
        rclpy.spin(imu_republisher)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'Error in IMU republisher: {e}')
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()