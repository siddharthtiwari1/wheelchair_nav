#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from geometry_msgs.msg import Twist, TwistStamped

class TwistToTwistStamped(Node):
    def __init__(self):
        super().__init__('twist_to_twist_stamped')

        # QoS profile matching wc_control expectations (BEST_EFFORT)
        cmd_vel_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE
        )

        self.subscription = self.create_subscription(
            Twist,
            'cmd_vel_in',
            self.twist_callback,
            10
        )

        # Use BEST_EFFORT QoS to match wc_control subscriber
        self.publisher = self.create_publisher(
            TwistStamped,
            'cmd_vel_out',
            cmd_vel_qos
        )
        
        self.get_logger().info('Twist to TwistStamped converter started')
    
    # Safety limits — last software defense before Arduino motors
    MAX_LINEAR = 0.30    # m/s forward
    MIN_LINEAR = -0.15   # m/s reverse
    MAX_ANGULAR = 0.50   # rad/s

    def twist_callback(self, msg):
        twist_stamped = TwistStamped()
        twist_stamped.header.stamp = self.get_clock().now().to_msg()
        twist_stamped.header.frame_id = ""  # diff_drive_controller doesn't need frame_id

        # Clamp velocities to safe wheelchair limits
        clamped = Twist()
        clamped.linear.x = max(self.MIN_LINEAR, min(self.MAX_LINEAR, msg.linear.x))
        clamped.angular.z = max(-self.MAX_ANGULAR, min(self.MAX_ANGULAR, msg.angular.z))
        twist_stamped.twist = clamped

        if msg.linear.x != clamped.linear.x or msg.angular.z != clamped.angular.z:
            self.get_logger().warn(
                f'CLAMPED: [{msg.linear.x:.2f}, {msg.angular.z:.2f}] -> [{clamped.linear.x:.2f}, {clamped.angular.z:.2f}]')

        self.publisher.publish(twist_stamped)

        # Debug output
        if clamped.linear.x != 0.0 or clamped.angular.z != 0.0:
            self.get_logger().info(f'Publishing: linear.x={clamped.linear.x:.2f}, angular.z={clamped.angular.z:.2f}')

def main(args=None):
    rclpy.init(args=args)
    node = TwistToTwistStamped()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()