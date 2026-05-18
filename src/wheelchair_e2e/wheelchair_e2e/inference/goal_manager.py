"""
Goal Manager Node

Provides goal input for the E2E navigation model.
Accepts goals from:
    1. RViz2 /goal_pose (PoseStamped) — click on map
    2. /e2e_goal topic (PoseStamped) — programmatic
    3. Waypoint list — sequential goal following

Publishes relative goal (dx, dy) in base_link frame for the model.
"""

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, PointStamped
from nav_msgs.msg import Odometry

import tf2_ros
from tf2_geometry_msgs import do_transform_pose_stamped


class GoalManagerNode(Node):
    """Manage navigation goals and publish relative goal position."""

    def __init__(self):
        super().__init__('goal_manager_node')

        self.declare_parameter('goal_reached_threshold', 0.5)
        self.goal_threshold = self.get_parameter(
            'goal_reached_threshold').value

        # TF2 for frame transforms
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(
            self.tf_buffer, self)

        # State
        self.current_goal = None  # PoseStamped in map frame
        self.current_odom = None

        # Subscribers
        self.goal_sub = self.create_subscription(
            PoseStamped, '/goal_pose',
            self.goal_callback, 10)
        self.odom_sub = self.create_subscription(
            Odometry, '/odometry/filtered',
            self.odom_callback, 10)

        # Publisher: relative goal for E2E model
        self.rel_goal_pub = self.create_publisher(
            PointStamped, '/e2e/relative_goal', 10)

        # Timer: publish relative goal at 10Hz
        self.timer = self.create_timer(0.1, self.publish_relative_goal)

        self.get_logger().info("Goal Manager started")

    def goal_callback(self, msg):
        self.current_goal = msg
        self.get_logger().info(
            f"New goal: ({msg.pose.position.x:.2f}, "
            f"{msg.pose.position.y:.2f})")

    def odom_callback(self, msg):
        self.current_odom = msg

    def publish_relative_goal(self):
        """Compute and publish goal relative to base_link."""
        if self.current_goal is None or self.current_odom is None:
            return

        # Transform goal from map to base_link frame
        try:
            transform = self.tf_buffer.lookup_transform(
                'base_link', 'map',
                rclpy.time.Time(), timeout=rclpy.duration.Duration(
                    seconds=0.1))

            goal_base = do_transform_pose_stamped(
                self.current_goal, transform)

            dx = goal_base.pose.position.x
            dy = goal_base.pose.position.y
            dist = np.sqrt(dx**2 + dy**2)

            # Check if goal reached
            if dist < self.goal_threshold:
                self.get_logger().info("Goal reached!")
                self.current_goal = None
                return

            # Publish relative goal
            msg = PointStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'base_link'
            msg.point.x = dx
            msg.point.y = dy
            self.rel_goal_pub.publish(msg)

        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            if hasattr(self, '_tf_err_count'):
                self._tf_err_count += 1
            else:
                self._tf_err_count = 0
            if self._tf_err_count % 50 == 0:
                self.get_logger().warn(f"TF lookup failed: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = GoalManagerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
