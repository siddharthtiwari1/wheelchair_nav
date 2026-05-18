#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
import copy


class ImuOutToImu(Node):
    """Republish Gazebo /imu/out to /imu for EKF compatibility in simulation."""

    def __init__(self):
        super().__init__('imu_out_to_imu')

        self.declare_parameter('input_topic', '/imu/out')
        self.declare_parameter('output_topic', '/imu')
        self.declare_parameter('frame_id', 'base_link')

        input_topic = self.get_parameter('input_topic').get_parameter_value().string_value
        output_topic = self.get_parameter('output_topic').get_parameter_value().string_value
        frame_id = self.get_parameter('frame_id').get_parameter_value().string_value

        self.publisher_ = self.create_publisher(Imu, output_topic, 10)
        self.subscription = self.create_subscription(Imu, input_topic, self._cb, 10)

        self.output_frame_id = frame_id

        self.get_logger().info(
            f"Republishing IMU: '{input_topic}' -> '{output_topic}' with frame_id '{frame_id}'"
        )

    def _cb(self, msg: Imu):
        repub_msg = copy.deepcopy(msg)
        repub_msg.header.frame_id = self.output_frame_id
        self.publisher_.publish(repub_msg)


def main(args=None):
    rclpy.init(args=args)
    node = ImuOutToImu()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
