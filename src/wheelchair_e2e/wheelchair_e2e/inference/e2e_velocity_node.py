"""
E2E Velocity Inference Node — Learned Nav2 Controller Replacement.

Replaces Nav2's DWB/RegulatedPurePursuit controller with a neural network.
Nav2's planner server (SmacPlanner2D) still runs and provides /plan.
AMCL localization still runs for position on the metric map.

Architecture:
    Nav2 Planner → /plan (global route)
    /scan_fused + /plan + /odometry → BEV (5ch) → BEVVelocityNet → Safety → /cmd_vel

Key difference from Nav2 DWB:
    DWB: sample 1000 (v,ω) → simulate → score vs costmap → pick best
    Ours: BEV (with route) → single forward pass → (v,ω) directly

What we keep from Nav2:  Planner, AMCL, map_server, behavior tree
What we replace:         Controller (DWB), costmap, velocity smoother

Smoothness guaranteed by:
    1. GRU hidden state (temporal continuity in the model)
    2. Jerk loss (trained to produce smooth velocity)
    3. EMA filter (exponential moving average on output)
    4. Acceleration clamp (hard limit on rate of change)
    5. Independent LiDAR safety layer
"""

import os
import time
from collections import deque

import numpy as np
import torch

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry, Path

from wheelchair_e2e.bev_generator import BEVGenerator
from wheelchair_e2e.models.bev_velocity_net import BEVVelocityNet


def get_yaw(q):
    """Extract yaw from quaternion."""
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return np.arctan2(siny, cosy)


class E2EVelocityNode(Node):
    """
    Learned controller that replaces Nav2's DWB/RegulatedPurePursuit.

    Receives global plan from Nav2 planner, renders it as BEV Ch.4,
    and outputs (v, ω) via a single neural network forward pass.
    """

    def __init__(self):
        super().__init__('e2e_velocity_node')

        # --- Parameters (matched to Nav2 defaults) ---
        self.declare_parameter('model_path', '')
        self.declare_parameter('v_max', 0.25)        # Nav2: desired_linear_vel
        self.declare_parameter('w_max', 0.35)         # Nav2: max_rotational_vel
        self.declare_parameter('inference_hz', 15.0)
        self.declare_parameter('safety_min_range', 0.4)   # 40cm e-stop
        self.declare_parameter('safety_slow_range', 0.8)  # 80cm slowdown
        self.declare_parameter('max_accel', 0.5)
        self.declare_parameter('max_alpha', 2.0)
        self.declare_parameter('ema_alpha', 0.3)
        self.declare_parameter('goal_tolerance', 0.15)  # Nav2: xy_goal_tolerance

        model_path = self.get_parameter('model_path').value
        self.v_max = self.get_parameter('v_max').value
        self.w_max = self.get_parameter('w_max').value
        hz = self.get_parameter('inference_hz').value
        self.safety_min = self.get_parameter('safety_min_range').value
        self.safety_slow = self.get_parameter('safety_slow_range').value
        self.max_accel = self.get_parameter('max_accel').value
        self.max_alpha = self.get_parameter('max_alpha').value
        self.ema_alpha = self.get_parameter('ema_alpha').value
        self.goal_tolerance = self.get_parameter('goal_tolerance').value

        assert 0 < self.safety_min < self.safety_slow
        assert hz > 0

        # --- BEV Generator (5 channels with route) ---
        self.bev_generator = BEVGenerator(grid_size=200, resolution=0.05)

        # --- Load Model ---
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f"Device: {self.device}")

        self.model = BEVVelocityNet(
            bev_channels=5, v_max=self.v_max, w_max=self.w_max
        ).to(self.device)

        if model_path and os.path.exists(model_path):
            ckpt = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(ckpt['model_state_dict'])
            self.get_logger().info(f"Loaded model: {model_path}")
        else:
            self.get_logger().warn(
                "No model loaded — will output zero velocity")

        self.model.eval()
        self.hidden = None  # GRU hidden state persists across frames

        # --- State ---
        self.latest_scan = None
        self.latest_odom = None
        self.goal_map = None
        self.latest_plan = None  # nav_msgs/Path from Nav2 planner
        self.odom_buffer = deque(maxlen=10)
        self.odom_trail = deque(maxlen=10)
        self.min_range = float('inf')
        self.min_range_angle = 0.0

        # Smoothing state
        self.ema_v = 0.0
        self.ema_omega = 0.0
        self.prev_v = 0.0
        self.prev_omega = 0.0
        self.prev_time = time.time()
        self.ready = False

        # Stuck detection for recovery
        self.stuck_start = None
        self.stuck_threshold = 0.02
        self.stuck_timeout = 5.0

        # --- Subscribers ---
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan_fused', self.scan_callback, 10)
        self.odom_sub = self.create_subscription(
            Odometry, '/odometry/filtered', self.odom_callback, 10)
        self.goal_sub = self.create_subscription(
            PoseStamped, '/goal_pose', self.goal_callback, 10)
        # Subscribe to Nav2 planner's global plan
        self.plan_sub = self.create_subscription(
            Path, '/plan', self.plan_callback, 10)

        # --- Publisher ---
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # --- Timer ---
        self.timer = self.create_timer(1.0 / hz, self.inference_loop)

        self.get_logger().info(
            f"E2E Controller started at {hz}Hz, "
            f"v_max={self.v_max}, w_max={self.w_max}, "
            f"replaces DWB/RegulatedPurePursuit")

    # ──────────────────────────── Callbacks ────────────────────────────

    def scan_callback(self, msg):
        self.latest_scan = msg
        ranges = np.array(msg.ranges)
        valid_mask = np.isfinite(ranges) & (ranges > 0.05)
        if np.any(valid_mask):
            valid = ranges[valid_mask]
            min_idx = np.argmin(valid)
            self.min_range = float(valid[min_idx])
            angles = np.linspace(
                msg.angle_min, msg.angle_max, len(ranges))
            self.min_range_angle = float(angles[valid_mask][min_idx])
        else:
            self.min_range = float('inf')

    def odom_callback(self, msg):
        self.latest_odom = msg
        v = msg.twist.twist.linear.x
        w = msg.twist.twist.angular.z
        theta = get_yaw(msg.pose.pose.orientation)
        self.odom_buffer.append([v, w, theta])

        self.odom_trail.append((
            msg.pose.pose.position.x,
            msg.pose.pose.position.y))

        if not self.ready and len(self.odom_buffer) >= 3:
            self.ready = True
            self.get_logger().info("Odom buffer ready")

    def goal_callback(self, msg):
        self.goal_map = (msg.pose.position.x, msg.pose.position.y)
        self.hidden = None
        self.stuck_start = None
        self.get_logger().info(
            f"New goal: ({self.goal_map[0]:.2f}, {self.goal_map[1]:.2f})")

    def plan_callback(self, msg):
        """Receive global plan from Nav2 planner server."""
        self.latest_plan = msg

    # ──────────────────────────── Helpers ────────────────────────────

    def _compute_relative_goal(self):
        """Compute goal relative to current wheelchair pose."""
        if self.goal_map is None or self.latest_odom is None:
            return None, None

        odom = self.latest_odom
        x = odom.pose.pose.position.x
        y = odom.pose.pose.position.y
        yaw = get_yaw(odom.pose.pose.orientation)

        dx_world = self.goal_map[0] - x
        dy_world = self.goal_map[1] - y

        dist = np.sqrt(dx_world ** 2 + dy_world ** 2)
        if dist < self.goal_tolerance:
            return None, None

        cos_yaw = np.cos(-yaw)
        sin_yaw = np.sin(-yaw)
        dx_base = cos_yaw * dx_world - sin_yaw * dy_world
        dy_base = sin_yaw * dx_world + cos_yaw * dy_world
        return dx_base, dy_base

    def _build_odom_trail(self):
        """Build ego-motion trail relative to current pose for BEV Ch.3."""
        if self.latest_odom is None or len(self.odom_trail) < 2:
            return []

        curr_x = self.latest_odom.pose.pose.position.x
        curr_y = self.latest_odom.pose.pose.position.y
        yaw = get_yaw(self.latest_odom.pose.pose.orientation)
        cos_yaw = np.cos(-yaw)
        sin_yaw = np.sin(-yaw)

        trail = []
        for (ox, oy) in self.odom_trail:
            dx = ox - curr_x
            dy = oy - curr_y
            trail.append((
                cos_yaw * dx - sin_yaw * dy,
                sin_yaw * dx + cos_yaw * dy))
        return trail

    def _build_route_points(self):
        """
        Transform Nav2 global plan (map frame) to base_link for BEV Ch.4.
        This is the route from SmacPlanner2D rendered in the BEV.
        """
        if self.latest_plan is None or self.latest_odom is None:
            return None

        odom = self.latest_odom
        x = odom.pose.pose.position.x
        y = odom.pose.pose.position.y
        yaw = get_yaw(odom.pose.pose.orientation)
        cos_yaw = np.cos(-yaw)
        sin_yaw = np.sin(-yaw)

        max_range = 5.0  # BEV covers 5m radius
        route_pts = []

        for pose_stamped in self.latest_plan.poses:
            px = pose_stamped.pose.position.x
            py = pose_stamped.pose.position.y

            dx = px - x
            dy = py - y
            bx = cos_yaw * dx - sin_yaw * dy
            by = sin_yaw * dx + cos_yaw * dy

            if abs(bx) <= max_range and abs(by) <= max_range:
                route_pts.append((bx, by))

        return route_pts if len(route_pts) >= 2 else None

    # ──────────────────────────── Main Loop ────────────────────────────

    def inference_loop(self):
        """
        Main inference loop — replaces Nav2 controller_server.
        Every cycle: BEV (with route from planner) → model → (v,ω).
        """
        if not self.ready or self.latest_scan is None:
            return

        goal_dx, goal_dy = self._compute_relative_goal()
        if goal_dx is None:
            if self.goal_map is not None:
                self.get_logger().info("Goal reached!")
                self.goal_map = None
            self._publish_stop()
            return

        t0 = time.time()
        dt = t0 - self.prev_time
        self.prev_time = t0

        # --- Build 5-channel BEV ---
        odom_trail = self._build_odom_trail()
        route_points = self._build_route_points()
        bev = self.bev_generator.scan_msg_to_bev(
            self.latest_scan, goal_dx, goal_dy, odom_trail, route_points)

        # --- Odom history ---
        odom_list = list(self.odom_buffer)
        while len(odom_list) < 10:
            odom_list.insert(0, [0.0, 0.0, 0.0])
        odom_flat = np.array(odom_list, dtype=np.float32).flatten()

        # --- Neural network inference ---
        with torch.no_grad():
            bev_t = torch.from_numpy(bev).unsqueeze(0).float().to(
                self.device)
            odom_t = torch.from_numpy(odom_flat).unsqueeze(0).to(
                self.device)
            vel, self.hidden = self.model(bev_t, odom_t, self.hidden)
            v_raw = vel[0, 0].item()
            w_raw = vel[0, 1].item()

        # --- EMA smoothing ---
        a = self.ema_alpha
        v = a * v_raw + (1 - a) * self.ema_v
        omega = a * w_raw + (1 - a) * self.ema_omega
        self.ema_v = v
        self.ema_omega = omega

        # --- Acceleration clamp ---
        if dt > 0:
            max_dv = self.max_accel * dt
            max_dw = self.max_alpha * dt
            v = float(np.clip(v, self.prev_v - max_dv,
                              self.prev_v + max_dv))
            omega = float(np.clip(omega, self.prev_omega - max_dw,
                                  self.prev_omega + max_dw))

        # --- Independent safety layer (parallel to model) ---
        obstacle_in_front = abs(self.min_range_angle) < np.pi / 3
        if self.min_range < self.safety_min and obstacle_in_front:
            v, omega = 0.0, 0.0
            self.get_logger().warn(
                f"EMERGENCY STOP: obstacle at {self.min_range:.2f}m")
        elif self.min_range < self.safety_slow and obstacle_in_front:
            v = min(v, 0.1)

        # --- Stuck detection + recovery ---
        speed = abs(v) + abs(omega) * 0.1
        if speed < self.stuck_threshold and self.goal_map is not None:
            if self.stuck_start is None:
                self.stuck_start = time.time()
            elif time.time() - self.stuck_start > self.stuck_timeout:
                self.get_logger().warn("STUCK — executing recovery")
                self._execute_recovery()
                self.stuck_start = None
                return
        else:
            self.stuck_start = None

        self.prev_v = v
        self.prev_omega = omega

        # --- Publish ---
        msg = Twist()
        msg.linear.x = float(v)
        msg.angular.z = float(omega)
        self.cmd_pub.publish(msg)

        # Log periodically
        latency_ms = (time.time() - t0) * 1000
        if not hasattr(self, '_log_count'):
            self._log_count = 0
        self._log_count += 1
        if self._log_count % 150 == 0:
            has_plan = self.latest_plan is not None
            self.get_logger().info(
                f"v={v:.3f} w={omega:.3f} "
                f"goal=({goal_dx:.1f},{goal_dy:.1f}) "
                f"min_r={self.min_range:.2f}m "
                f"plan={'yes' if has_plan else 'no'} "
                f"lat={latency_ms:.1f}ms")

    # ──────────────────────────── Recovery ────────────────────────────

    def _execute_recovery(self):
        """
        Simple recovery: backup 0.25m, wait, spin 30°, reset GRU.
        Mirrors Nav2's backup-first recovery strategy.
        """
        cmd = Twist()

        # Backup 0.25m at 0.1 m/s (~2.5s)
        self.get_logger().info("Recovery: backing up")
        cmd.linear.x = -0.1
        for _ in range(25):
            self.cmd_pub.publish(cmd)
            time.sleep(0.1)

        # Stop + wait
        cmd.linear.x = 0.0
        self.cmd_pub.publish(cmd)
        time.sleep(1.0)

        # Spin 30° at 0.25 rad/s
        self.get_logger().info("Recovery: spinning")
        cmd.angular.z = 0.25
        for _ in range(21):
            self.cmd_pub.publish(cmd)
            time.sleep(0.1)

        cmd.angular.z = 0.0
        self.cmd_pub.publish(cmd)

        # Reset GRU (fresh start)
        self.hidden = None
        self.ema_v = 0.0
        self.ema_omega = 0.0
        self.prev_v = 0.0
        self.prev_omega = 0.0
        self.get_logger().info("Recovery complete — GRU reset")

    def _publish_stop(self):
        """Publish zero velocity."""
        msg = Twist()
        self.cmd_pub.publish(msg)
        self.prev_v = 0.0
        self.prev_omega = 0.0
        self.ema_v = 0.0
        self.ema_omega = 0.0


def main(args=None):
    rclpy.init(args=args)
    node = E2EVelocityNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        stop_msg = Twist()
        node.cmd_pub.publish(stop_msg)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
