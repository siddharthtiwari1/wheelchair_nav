#!/usr/bin/env python3

import math
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import copy


class SimOdomBias(Node):
    """Inject a small lateral drift into Gazebo odom to highlight EKF correction."""

    def __init__(self):
        super().__init__('sim_odom_bias')

        self.declare_parameter('input_topic', '/wc_control/odom_raw')
        self.declare_parameter('output_topic', '/wc_control/odom')
        self.declare_parameter('drift_per_meter', 0.02)  # meters of lateral drift per meter traveled
        self.declare_parameter('drift_axis', 'y')  # axis to bias: 'x' or 'y'

        self.input_topic = self.get_parameter('input_topic').value
        self.output_topic = self.get_parameter('output_topic').value
        self.drift_per_meter = float(self.get_parameter('drift_per_meter').value)
        self.drift_axis = self.get_parameter('drift_axis').value

        if self.drift_axis not in ('x', 'y'):
            self.get_logger().warn(
                f"Invalid drift_axis '{self.drift_axis}', defaulting to 'y'"
            )
            self.drift_axis = 'y'

        self.publisher_ = self.create_publisher(Odometry, self.output_topic, 10)
        self.subscription = self.create_subscription(Odometry, self.input_topic, self._cb, 10)

        self.prev_position = None
        self.accumulated_distance = 0.0

        self.get_logger().info(
            f"SimOdomBias started: '{self.input_topic}' -> '{self.output_topic}', "
            f"drift {self.drift_per_meter} m per meter along {self.drift_axis}"
        )

    def _cb(self, msg: Odometry):
        current_pos = (msg.pose.pose.position.x, msg.pose.pose.position.y)

        if self.prev_position is not None:
            dx = current_pos[0] - self.prev_position[0]
            dy = current_pos[1] - self.prev_position[1]
            self.accumulated_distance += math.hypot(dx, dy)

        self.prev_position = current_pos

        biased_msg = copy.deepcopy(msg)

        drift_amount = self.accumulated_distance * self.drift_per_meter
        if self.drift_axis == 'y':
            biased_msg.pose.pose.position.y += drift_amount
        else:
            biased_msg.pose.pose.position.x += drift_amount

        self.publisher_.publish(biased_msg)


def main(args=None):
    rclpy.init(args=args)
    node = SimOdomBias()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
