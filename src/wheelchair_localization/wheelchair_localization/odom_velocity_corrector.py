#!/usr/bin/env python3
"""
Odometry Velocity Corrector Node

PROBLEM:
  - Wheel odometry POSITION is correct (integrates properly)
  - Wheel odometry VELOCITY is 20% underestimated (timing issues)
  - Position has YAW DRIFT baked in (wheel yaw, not IMU yaw)

SOLUTION:
  - Compute SPEED from position changes: speed = sqrt(dx² + dy²) / dt
  - This gives correct magnitude regardless of yaw drift
  - Publish as TwistWithCovarianceStamped in body frame
  - EKF fuses this speed with IMU yaw → correct trajectory!

This gives BOTH:
  1. Correct distance (from position-derived speed)
  2. Correct trajectory shape (from IMU yaw)
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TwistWithCovarianceStamped
import math


class OdomVelocityCorrector(Node):
    def __init__(self):
        super().__init__('odom_velocity_corrector')

        # Parameters
        self.declare_parameter('input_topic', '/wc_control/odom')
        self.declare_parameter('output_topic', '/wheel_velocity')
        self.declare_parameter('min_dt', 0.01)  # Minimum dt to avoid division by zero
        self.declare_parameter('max_dt', 0.5)   # Maximum dt to detect stale data
        self.declare_parameter('velocity_covariance', 0.01)  # Covariance for velocity

        input_topic = self.get_parameter('input_topic').value
        output_topic = self.get_parameter('output_topic').value
        self.min_dt = self.get_parameter('min_dt').value
        self.max_dt = self.get_parameter('max_dt').value
        self.velocity_covariance = self.get_parameter('velocity_covariance').value

        # State
        self.last_x = None
        self.last_y = None
        self.last_time = None
        self.last_yaw = None  # For angular velocity

        # Publisher
        self.velocity_pub = self.create_publisher(
            TwistWithCovarianceStamped,
            output_topic,
            10
        )

        # Subscriber
        self.odom_sub = self.create_subscription(
            Odometry,
            input_topic,
            self.odom_callback,
            10
        )

        self.get_logger().info(
            f'Odom Velocity Corrector started: {input_topic} -> {output_topic}'
        )
        self.get_logger().info(
            'Computing velocity from position changes (drift-free magnitude)'
        )

    def quat_to_yaw(self, qz, qw):
        """Extract yaw from quaternion (2D assumption)."""
        return 2.0 * math.atan2(qz, qw)

    def odom_callback(self, msg: Odometry):
        """Process odometry and publish corrected velocity."""

        current_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        current_x = msg.pose.pose.position.x
        current_y = msg.pose.pose.position.y
        current_yaw = self.quat_to_yaw(
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        )

        # Initialize on first message
        if self.last_time is None:
            self.last_x = current_x
            self.last_y = current_y
            self.last_time = current_time
            self.last_yaw = current_yaw
            return

        # Compute dt
        dt = current_time - self.last_time

        # Skip if dt is too small or too large
        if dt < self.min_dt or dt > self.max_dt:
            self.last_x = current_x
            self.last_y = current_y
            self.last_time = current_time
            self.last_yaw = current_yaw
            return

        # Compute position change
        dx = current_x - self.last_x
        dy = current_y - self.last_y

        # Compute SPEED (magnitude) - this is drift-free!
        # sqrt(dx² + dy²) gives distance traveled regardless of coordinate frame
        distance = math.sqrt(dx * dx + dy * dy)
        speed = distance / dt

        # Determine direction (forward/backward) from the odom twist sign
        # If odom says we're going backward, make speed negative
        if msg.twist.twist.linear.x < -0.01:
            speed = -speed

        # Compute angular velocity from yaw change
        dyaw = current_yaw - self.last_yaw
        # Handle wrap-around
        while dyaw > math.pi:
            dyaw -= 2 * math.pi
        while dyaw < -math.pi:
            dyaw += 2 * math.pi
        angular_velocity = dyaw / dt

        # Create output message
        twist_msg = TwistWithCovarianceStamped()
        twist_msg.header = msg.header
        twist_msg.header.frame_id = 'base_link'  # Body frame!

        # Linear velocity (in body frame)
        twist_msg.twist.twist.linear.x = speed
        twist_msg.twist.twist.linear.y = 0.0  # Differential drive, no lateral motion
        twist_msg.twist.twist.linear.z = 0.0

        # Angular velocity
        twist_msg.twist.twist.angular.x = 0.0
        twist_msg.twist.twist.angular.y = 0.0
        twist_msg.twist.twist.angular.z = angular_velocity

        # Covariance (6x6 matrix, row-major)
        # [vx, vy, vz, wx, wy, wz]
        cov = [0.0] * 36
        cov[0] = self.velocity_covariance      # vx variance
        cov[7] = 1e6                            # vy variance (not measured, high)
        cov[14] = 1e6                           # vz variance (not measured, high)
        cov[21] = 1e6                           # wx variance (not measured, high)
        cov[28] = 1e6                           # wy variance (not measured, high)
        cov[35] = self.velocity_covariance     # wz variance
        twist_msg.twist.covariance = cov

        # Publish
        self.velocity_pub.publish(twist_msg)

        # Update state
        self.last_x = current_x
        self.last_y = current_y
        self.last_time = current_time
        self.last_yaw = current_yaw


def main(args=None):
    rclpy.init(args=args)
    node = OdomVelocityCorrector()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
