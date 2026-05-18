#!/usr/bin/env python3
"""
ROS2 Node: Quantum-Inspired Trajectory Planner for Real-Time Navigation.

Subscribes to live sensor data and publishes velocity commands using
the quantum trajectory planner with safety-clamped outputs.

Topics:
  Subscribed:
    /scan_fused          (LaserScan)  - fused 360 scan
    /odometry/filtered   (Odometry)   - EKF-fused odometry
    /goal_pose           (PoseStamped) - navigation goal

  Published:
    /cmd_vel                          (Twist)     - velocity command
    /quantum_nav/diagnostics          (String)    - phi, mode, time_ms
    /quantum_nav/trajectories         (MarkerArray) - visualization

Parameters:
    planner_type: str  - 'quantum_sup_3q' | 'cost_encoded' | 'phase' |
                         'temporal' | 'knn_mean' | 'mppi' | 'classical'
    max_v: float       - max linear velocity (default 0.25 m/s)
    max_omega: float   - max angular velocity (default 0.35 rad/s)
    rate_hz: float     - planning rate (default 10.0 Hz)
"""

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, Twist
from std_msgs.msg import String
from visualization_msgs.msg import MarkerArray, Marker
import time

from wheelchair_e2e.quantum_nav.quantum_trajectory_planner import (
    QuantumTrajectoryPlanner, AdaptiveQuantumPlanner
)
from wheelchair_e2e.quantum_nav.quantum_planner_v2 import (
    CostEncodedQuantumPlanner, PhaseInterferenceQuantumPlanner,
    TemporalEntanglementPlanner
)
from wheelchair_e2e.quantum_nav.baselines import (
    ClassicalBestCost, KNNMeanScore, MPPIPlanner
)


def get_yaw(q):
    """Extract yaw from quaternion."""
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return np.arctan2(siny, cosy)


class QuantumNavNode(Node):
    def __init__(self):
        super().__init__('quantum_nav_node')

        # Parameters
        self.declare_parameter('planner_type', 'quantum_sup_3q')
        self.declare_parameter('max_v', 0.25)
        self.declare_parameter('max_omega', 0.35)
        self.declare_parameter('rate_hz', 10.0)
        self.declare_parameter('n_candidates', 64)
        self.declare_parameter('safety_clearance', 0.3)
        self.declare_parameter('enabled', True)

        planner_type = self.get_parameter('planner_type').value
        self.max_v = self.get_parameter('max_v').value
        self.max_omega = self.get_parameter('max_omega').value
        rate_hz = self.get_parameter('rate_hz').value
        n_cand = self.get_parameter('n_candidates').value
        self.safety_clearance = self.get_parameter('safety_clearance').value
        self.enabled = self.get_parameter('enabled').value

        # Build planner
        self.planner = self._build_planner(planner_type, n_cand)
        self.planner_type = planner_type
        self.get_logger().info(
            f'Planner: {planner_type}, max_v={self.max_v}, '
            f'max_omega={self.max_omega}, rate={rate_hz}Hz')

        # State
        self.latest_scan = None
        self.latest_odom = None
        self.goal = None
        self.goal_frame = 'map'

        # Subscribers
        scan_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan_fused', self._scan_cb, scan_qos)
        self.odom_sub = self.create_subscription(
            Odometry, '/odometry/filtered', self._odom_cb, 10)
        self.goal_sub = self.create_subscription(
            PoseStamped, '/goal_pose', self._goal_cb, 10)

        # Publishers
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.diag_pub = self.create_publisher(
            String, '/quantum_nav/diagnostics', 10)
        self.marker_pub = self.create_publisher(
            MarkerArray, '/quantum_nav/trajectories', 10)

        # Timer
        self.timer = self.create_timer(1.0 / rate_hz, self._plan_cb)

        # Stats
        self.plan_count = 0
        self.total_time_ms = 0.0

    def _build_planner(self, planner_type, n_cand):
        """Factory for planner types."""
        planners = {
            'quantum_sup_3q': lambda: QuantumTrajectoryPlanner(
                n_candidates=n_cand, nn=3, use_superposition=True),
            'quantum_sup_2q': lambda: QuantumTrajectoryPlanner(
                n_candidates=n_cand, nn=2, use_superposition=True),
            'quantum_adaptive': lambda: AdaptiveQuantumPlanner(
                n_candidates=n_cand, nn=3, use_superposition=True),
            'cost_encoded': lambda: CostEncodedQuantumPlanner(
                n_candidates=n_cand, nn=3, temperature=1.0),
            'phase': lambda: PhaseInterferenceQuantumPlanner(
                n_candidates=n_cand, nn=3),
            'temporal': lambda: TemporalEntanglementPlanner(
                n_candidates=n_cand, nn=3, alpha=0.7),
            'knn_mean': lambda: KNNMeanScore(n_candidates=n_cand, nn=3),
            'mppi': lambda: MPPIPlanner(n_candidates=n_cand, lambda_=1.0),
            'classical': lambda: ClassicalBestCost(n_candidates=n_cand),
        }
        if planner_type not in planners:
            self.get_logger().warn(
                f'Unknown planner_type={planner_type}, using quantum_sup_3q')
            planner_type = 'quantum_sup_3q'
        return planners[planner_type]()

    def _scan_cb(self, msg):
        self.latest_scan = msg

    def _odom_cb(self, msg):
        self.latest_odom = msg

    def _goal_cb(self, msg):
        self.goal = msg
        self.goal_frame = msg.header.frame_id
        self.get_logger().info(
            f'Goal received: ({msg.pose.position.x:.2f}, '
            f'{msg.pose.position.y:.2f}) in {self.goal_frame}')

    def _scan_to_points(self, scan_msg):
        """Convert LaserScan to obstacle points in robot frame."""
        ranges = np.array(scan_msg.ranges, dtype=np.float32)
        n = len(ranges)
        angles = np.linspace(scan_msg.angle_min, scan_msg.angle_max, n)

        valid = (ranges > 0.1) & (ranges < 5.0) & np.isfinite(ranges)
        if not np.any(valid):
            return np.zeros((0, 2))

        x = ranges[valid] * np.cos(angles[valid])
        y = ranges[valid] * np.sin(angles[valid])
        return np.column_stack([x, y])

    def _plan_cb(self):
        """Timer callback: run planner and publish cmd_vel."""
        if not self.enabled:
            return

        if self.latest_scan is None or self.latest_odom is None:
            return

        if self.goal is None:
            return

        t0 = time.time()

        # Obstacle points in robot frame
        obs_pts = self._scan_to_points(self.latest_scan)

        # Goal in robot frame
        rx = self.latest_odom.pose.pose.position.x
        ry = self.latest_odom.pose.pose.position.y
        rtheta = get_yaw(self.latest_odom.pose.pose.orientation)

        gx = self.goal.pose.position.x
        gy = self.goal.pose.position.y

        dx = gx - rx
        dy = gy - ry
        cos_t = np.cos(-rtheta)
        sin_t = np.sin(-rtheta)
        goal_rx = cos_t * dx - sin_t * dy
        goal_ry = sin_t * dx + cos_t * dy

        # Check if at goal
        goal_dist = np.sqrt(goal_rx**2 + goal_ry**2)
        if goal_dist < 0.3:
            self._publish_stop()
            return

        # Run planner
        if isinstance(self.planner, QuantumTrajectoryPlanner):
            if isinstance(self.planner, AdaptiveQuantumPlanner):
                decision = self.planner.select_trajectory_adaptive(
                    0, 0, 0, goal_rx, goal_ry, obs_pts)
            else:
                decision = self.planner.select_trajectory(
                    0, 0, 0, goal_rx, goal_ry, obs_pts, sensor_noise=0.1)

            v_raw = decision.v
            omega_raw = decision.omega
            phi = decision.confidence
            mode = decision.mode
            scores = decision.quantum_scores
        else:
            result = self.planner.plan(0, 0, 0, goal_rx, goal_ry, obs_pts)
            v_raw = result['v']
            omega_raw = result['omega']
            phi = result['confidence']
            mode = result['mode']
            scores = result['scores']

        # Safety clamp (HARD limits from wheelchair hardware)
        v = np.clip(v_raw, -0.05, self.max_v)
        omega = np.clip(omega_raw, -self.max_omega, self.max_omega)

        # Emergency stop if too close to obstacle
        if len(obs_pts) > 0:
            min_dist = np.min(np.linalg.norm(obs_pts, axis=1))
            if min_dist < self.safety_clearance:
                v = 0.0
                omega = 0.0
                mode = 'emergency_stop'

        elapsed_ms = (time.time() - t0) * 1000

        # Publish cmd_vel
        cmd = Twist()
        cmd.linear.x = float(v)
        cmd.angular.z = float(omega)
        self.cmd_pub.publish(cmd)

        # Publish diagnostics
        diag = String()
        diag.data = (f'phi={phi:.3f} mode={mode} v={v:.3f} w={omega:.3f} '
                     f't={elapsed_ms:.1f}ms goal_d={goal_dist:.2f} '
                     f'planner={self.planner_type}')
        self.diag_pub.publish(diag)

        # Stats
        self.plan_count += 1
        self.total_time_ms += elapsed_ms

        if self.plan_count % 50 == 0:
            avg_ms = self.total_time_ms / self.plan_count
            self.get_logger().info(
                f'Plan #{self.plan_count}: phi={phi:.3f} mode={mode} '
                f'v={v:.2f} w={omega:.3f} avg_t={avg_ms:.1f}ms')

        # Publish trajectory markers (every 5th cycle)
        if self.plan_count % 5 == 0 and isinstance(self.planner, QuantumTrajectoryPlanner):
            self._publish_markers(scores)

    def _publish_stop(self):
        """Publish zero velocity."""
        cmd = Twist()
        self.cmd_pub.publish(cmd)

    def _publish_markers(self, scores):
        """Publish trajectory visualization as MarkerArray."""
        if not isinstance(self.planner, QuantumTrajectoryPlanner):
            return

        # Get candidates from the sampler (approximate — uses same seed logic)
        markers = MarkerArray()

        # Delete old markers
        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        delete_marker.header.frame_id = 'base_link'
        delete_marker.header.stamp = self.get_clock().now().to_msg()
        markers.markers.append(delete_marker)

        # Best trajectory marker
        if len(scores) > 0:
            best_idx = np.argmax(scores)
            m = Marker()
            m.header.frame_id = 'base_link'
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = 'quantum_best'
            m.id = 1
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.scale.x = 0.15
            m.scale.y = 0.15
            m.scale.z = 0.15
            m.color.r = 0.0
            m.color.g = 1.0
            m.color.b = 0.0
            m.color.a = 0.8
            m.pose.position.x = 0.5  # approximate
            markers.markers.append(m)

        self.marker_pub.publish(markers)


def main(args=None):
    rclpy.init(args=args)
    node = QuantumNavNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
