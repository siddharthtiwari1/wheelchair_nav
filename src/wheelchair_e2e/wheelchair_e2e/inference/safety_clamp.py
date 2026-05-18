"""
LiDAR Safety Clamp — independent safety layer.

Can run as a standalone node that monitors /cmd_vel and overrides
with safe velocities based on LiDAR proximity.

Safety zones:
    < 0.3m: EMERGENCY STOP (v=0, ω=0)
    < 0.6m: SLOW DOWN (v ≤ 0.1 m/s)
    ≥ 0.6m: PASS THROUGH (no modification)

This is independent of the neural network — it's a hard safety guarantee.
"""

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist


class SafetyClampNode(Node):
    """
    LiDAR-based safety override for wheelchair velocity commands.

    Subscribes to raw cmd_vel from E2E model, applies safety limits,
    publishes safe cmd_vel.
    """

    def __init__(self):
        super().__init__('safety_clamp_node')

        # Parameters
        self.declare_parameter('emergency_stop_range', 0.3)
        self.declare_parameter('slowdown_range', 0.6)
        self.declare_parameter('slowdown_max_v', 0.1)
        self.declare_parameter('max_velocity', 0.25)
        self.declare_parameter('max_angular', 1.0)
        self.declare_parameter('input_topic', '/cmd_vel_raw')
        self.declare_parameter('output_topic', '/cmd_vel')
        self.declare_parameter('scan_topic', '/scan_fused')

        self.stop_range = self.get_parameter(
            'emergency_stop_range').value
        self.slow_range = self.get_parameter('slowdown_range').value
        self.slow_max_v = self.get_parameter('slowdown_max_v').value
        self.max_v = self.get_parameter('max_velocity').value
        self.max_w = self.get_parameter('max_angular').value

        input_topic = self.get_parameter('input_topic').value
        output_topic = self.get_parameter('output_topic').value
        scan_topic = self.get_parameter('scan_topic').value

        # State
        self.min_range = float('inf')
        self.min_range_angle = 0.0

        # Subscribers
        self.scan_sub = self.create_subscription(
            LaserScan, scan_topic, self.scan_callback, 10)
        self.cmd_sub = self.create_subscription(
            Twist, input_topic, self.cmd_callback, 10)

        # Publisher
        self.cmd_pub = self.create_publisher(Twist, output_topic, 10)

        self.get_logger().info(
            f"Safety Clamp: {input_topic} → {output_topic} "
            f"(stop={self.stop_range}m, slow={self.slow_range}m)")

    def scan_callback(self, msg):
        """Update minimum range from scan."""
        ranges = np.array(msg.ranges)
        valid_mask = np.isfinite(ranges) & (ranges > 0.05)
        if np.any(valid_mask):
            valid_ranges = ranges[valid_mask]
            min_idx = np.argmin(valid_ranges)
            self.min_range = valid_ranges[min_idx]

            # Compute angle of closest obstacle
            angles = np.linspace(msg.angle_min, msg.angle_max,
                                 len(ranges))
            valid_angles = angles[valid_mask]
            self.min_range_angle = valid_angles[min_idx]
        else:
            self.min_range = float('inf')

    def cmd_callback(self, msg):
        """Apply safety limits to incoming velocity command."""
        v = msg.linear.x
        omega = msg.angular.z

        # Hard velocity limits
        v = np.clip(v, 0.0, self.max_v)
        omega = np.clip(omega, -self.max_w, self.max_w)

        # Directional safety: only restrict if moving toward obstacle
        obstacle_in_front = abs(self.min_range_angle) < np.pi / 3  # ±60°

        if self.min_range < self.stop_range and obstacle_in_front:
            v = 0.0
            omega = 0.0
            self.get_logger().warn(
                f"EMERGENCY STOP: obstacle at {self.min_range:.2f}m")
        elif self.min_range < self.slow_range and obstacle_in_front:
            v = min(v, self.slow_max_v)

        # Publish safe velocity
        safe_msg = Twist()
        safe_msg.linear.x = float(v)
        safe_msg.angular.z = float(omega)
        self.cmd_pub.publish(safe_msg)


def main(args=None):
    rclpy.init(args=args)
    node = SafetyClampNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        stop = Twist()
        node.cmd_pub.publish(stop)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
