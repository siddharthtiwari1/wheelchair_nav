#!/usr/bin/env python3
"""
KinoFlow v2 Inference Node — Modular Learned Nav2 Controller Replacement.

Replaces Nav2's RegulatedPurePursuit/DWB controller with KinoFlow v2:
    /scan_fused → E1(scan) + E2(temporal residuals) + E3(goal) + E4(odom)
    → Transformer Fusion → GRU → Trajectory Transformer → K trajectories
    → Score → best (v, ω) → safety layers → /cmd_vel

Supports both v1 (BEV + ResNet-18) and v2 (modular polar encoders) via
the --model_version parameter.

Multi-sample trajectory generation (K=8):
    1. Generate K velocity trajectories from different noise seeds
    2. Forward-integrate to full kinodynamic poses (x, y, θ)
    3. Score: collision + comfort + goal progress
    4. Execute first (v, ω) of best trajectory (receding horizon)
    5. Warm-start next cycle from shifted best trajectory

RViz2 visualization:
    - Best trajectory: green line
    - Other candidates: semi-transparent blue lines
"""

import os
import time
from collections import deque

import numpy as np
import torch

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, PoseStamped, Point
from nav_msgs.msg import Odometry, Path
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA

from wheelchair_e2e.bev_generator import BEVGenerator
from wheelchair_e2e.models.kinoflow_net import ModularKinoFlowNet, KinoFlowNet
from wheelchair_e2e.models.scoring_network import DualSpaceScoringTransformer
from wheelchair_e2e.models.goal_encoder import compute_goal_features


def get_yaw(q):
    """Extract yaw from quaternion."""
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return np.arctan2(siny, cosy)


class KinoFlowNode(Node):
    """KinoFlow v2 controller: modular kinodynamic trajectory generation."""

    def __init__(self):
        super().__init__('kinoflow_node')

        # --- Parameters ---
        self.declare_parameter('model_path', '')
        self.declare_parameter('model_version', 'v2')  # 'v1' or 'v2'
        self.declare_parameter('v_max', 0.25)
        self.declare_parameter('w_max', 1.0)
        self.declare_parameter('inference_hz', 15.0)
        self.declare_parameter('horizon', 10)
        self.declare_parameter('n_samples', 8)
        self.declare_parameter('n_euler_steps', 3)
        self.declare_parameter('dt', 0.1)
        self.declare_parameter('safety_min_range', 0.4)
        self.declare_parameter('safety_slow_range', 0.8)
        self.declare_parameter('max_accel', 0.5)
        self.declare_parameter('max_alpha', 2.0)
        self.declare_parameter('ema_alpha', 0.3)
        self.declare_parameter('goal_tolerance', 0.15)
        self.declare_parameter('visualize_trajectories', True)
        self.declare_parameter('scorer_path', '')
        self.declare_parameter('scan_points', 720)
        self.declare_parameter('temporal_frames', 5)

        model_path = self.get_parameter('model_path').value
        self.model_version = self.get_parameter('model_version').value
        self.v_max = self.get_parameter('v_max').value
        self.w_max = self.get_parameter('w_max').value
        hz = self.get_parameter('inference_hz').value
        horizon = self.get_parameter('horizon').value
        n_samples = self.get_parameter('n_samples').value
        n_euler_steps = self.get_parameter('n_euler_steps').value
        dt = self.get_parameter('dt').value
        self.safety_min = self.get_parameter('safety_min_range').value
        self.safety_slow = self.get_parameter('safety_slow_range').value
        self.max_accel = self.get_parameter('max_accel').value
        self.max_alpha = self.get_parameter('max_alpha').value
        self.ema_alpha = self.get_parameter('ema_alpha').value
        self.goal_tolerance = self.get_parameter('goal_tolerance').value
        self.visualize = self.get_parameter('visualize_trajectories').value
        scorer_path = self.get_parameter('scorer_path').value
        scan_points = self.get_parameter('scan_points').value
        temporal_frames = self.get_parameter('temporal_frames').value

        # --- BEV Generator (for collision scoring + v1 compatibility) ---
        self.bev_generator = BEVGenerator(grid_size=200, resolution=0.05)

        # Initialize scan temporal buffer for v2
        if self.model_version == 'v2':
            self.bev_generator.init_scan_buffer(temporal_frames)

        # --- Load Model ---
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f"Device: {self.device}")

        if self.model_version == 'v2':
            self.model = ModularKinoFlowNet(
                v_max=self.v_max, w_max=self.w_max,
                horizon=horizon, n_euler_steps=n_euler_steps,
                dt=dt, n_samples=n_samples,
                scan_points=scan_points,
                temporal_frames=temporal_frames,
            ).to(self.device)
            backbone_dim = 128
        else:
            self.model = KinoFlowNet(
                bev_channels=5, v_max=self.v_max, w_max=self.w_max,
                horizon=horizon, n_euler_steps=n_euler_steps,
                dt=dt, n_samples=n_samples,
            ).to(self.device)
            backbone_dim = 512

        self.dummy_mode = True
        if model_path and os.path.exists(model_path):
            ckpt = torch.load(model_path, map_location=self.device,
                              weights_only=False)
            self.model.load_state_dict(ckpt['model_state_dict'])
            epoch = ckpt.get('epoch', '?')
            phase = ckpt.get('phase', '?')
            arch = ckpt.get('architecture', self.model_version)
            self.dummy_mode = False
            self.get_logger().info(
                f"Loaded {arch}: {model_path} (epoch={epoch}, phase={phase})")
        else:
            self.get_logger().warn(
                "*** DUMMY MODE *** No model loaded — running control loop "
                "with zero velocity. Set model_path param to enable.")

        # --- Learned Scorer ---
        if scorer_path and os.path.exists(scorer_path):
            scorer_ckpt = torch.load(
                scorer_path, map_location=self.device, weights_only=False)
            scorer = DualSpaceScoringTransformer(
                embed_dim=scorer_ckpt.get('embed_dim', 128),
                n_heads=scorer_ckpt.get('n_heads', 4),
                n_layers=scorer_ckpt.get('n_layers', 3),
                backbone_dim=backbone_dim,
                horizon=horizon,
            ).to(self.device)
            scorer.load_state_dict(scorer_ckpt['model_state_dict'])
            self.model.set_learned_scorer(scorer)
            self.get_logger().info(f"Loaded learned scorer: {scorer_path}")
        else:
            self.get_logger().info("Using hand-crafted scorer")

        self.model.eval()
        self.hidden = None
        self.warm_start = None

        # --- State ---
        self.latest_scan = None
        self.latest_odom = None
        self.goal_map = None
        self.latest_plan = None
        self.odom_buffer = deque(maxlen=10)
        self.odom_trail = deque(maxlen=10)
        self.min_range = float('inf')
        self.min_range_angle = 0.0

        # Smoothing
        self.ema_v = 0.0
        self.ema_omega = 0.0
        self.prev_v = 0.0
        self.prev_omega = 0.0
        self.prev_time = time.time()
        self.ready = False

        # Stuck detection
        self.stuck_start = None
        self.stuck_threshold = 0.02
        self.stuck_timeout = 5.0

        # Log / state counters (avoid hasattr checks in hot loop)
        self._dummy_log_count = 0
        self._ood_warned = False
        self._log_count = 0

        # --- Subscribers ---
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan_fused', self.scan_callback, 10)
        self.odom_sub = self.create_subscription(
            Odometry, '/odometry/filtered', self.odom_callback, 10)
        self.goal_sub = self.create_subscription(
            PoseStamped, '/goal_pose', self.goal_callback, 10)
        self.plan_sub = self.create_subscription(
            Path, '/plan', self.plan_callback, 10)

        # --- Publishers ---
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        if self.visualize:
            self.traj_viz_pub = self.create_publisher(
                MarkerArray, '/kinoflow/trajectories', 10)
            self.best_traj_pub = self.create_publisher(
                Path, '/kinoflow/best_trajectory', 10)

        # --- Timer ---
        self.timer = self.create_timer(1.0 / hz, self.inference_loop)

        pc = self.model.get_param_count()
        self.get_logger().info(
            f"KinoFlow {self.model_version} started: {hz}Hz, "
            f"H={horizon}, K={n_samples}, "
            f"euler_steps={n_euler_steps}, "
            f"params={pc['total']/1e6:.2f}M")

    # ---- Callbacks ----

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

        # Update scan temporal buffer for v2
        if self.model_version == 'v2' and self.latest_odom is not None:
            ego_odom = (
                self.latest_odom.pose.pose.position.x,
                self.latest_odom.pose.pose.position.y,
                get_yaw(self.latest_odom.pose.pose.orientation),
            )
            self.bev_generator.update_scan_buffer(
                msg.ranges, msg.angle_min, msg.angle_max, ego_odom)

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
        self.warm_start = None
        self.stuck_start = None
        self.bev_generator.reset_temporal()
        if self.model_version == 'v2':
            self.bev_generator.init_scan_buffer()
        self.get_logger().info(
            f"New goal: ({self.goal_map[0]:.2f}, {self.goal_map[1]:.2f})")

    def plan_callback(self, msg):
        self.latest_plan = msg

    # ---- Helpers ----

    def _compute_relative_goal(self):
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
            trail.append((cos_yaw * dx - sin_yaw * dy,
                          sin_yaw * dx + cos_yaw * dy))
        return trail

    def _build_route_points(self):
        if self.latest_plan is None or self.latest_odom is None:
            return None
        odom = self.latest_odom
        x = odom.pose.pose.position.x
        y = odom.pose.pose.position.y
        yaw = get_yaw(odom.pose.pose.orientation)
        cos_yaw = np.cos(-yaw)
        sin_yaw = np.sin(-yaw)
        route_pts = []
        for pose_stamped in self.latest_plan.poses:
            px = pose_stamped.pose.position.x
            py = pose_stamped.pose.position.y
            dx = px - x
            dy = py - y
            bx = cos_yaw * dx - sin_yaw * dy
            by = sin_yaw * dx + cos_yaw * dy
            if abs(bx) <= 5.0 and abs(by) <= 5.0:
                route_pts.append((bx, by))
        return route_pts if len(route_pts) >= 2 else None

    # ---- Main Loop ----

    def inference_loop(self):
        if not self.ready or self.latest_scan is None:
            return

        # Dummy mode: publish zero vel, log periodically
        if self.dummy_mode:
            self._publish_stop()
            self._dummy_log_count += 1
            if self._dummy_log_count % 150 == 1:
                self.get_logger().info(
                    "DUMMY MODE: publishing zero vel "
                    "(load a trained model to enable navigation)")
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

        # --- Build odom history ---
        odom_list = list(self.odom_buffer)
        while len(odom_list) < 10:
            odom_list.insert(0, [0.0, 0.0, 0.0])
        odom_flat = np.array(odom_list, dtype=np.float32).flatten()
        odom_t = torch.from_numpy(odom_flat).unsqueeze(0).to(self.device)

        # --- BEV for collision scoring ---
        odom_trail = self._build_odom_trail()
        route_points = self._build_route_points()
        ego_odom = None
        if self.latest_odom is not None:
            ego_odom = (
                self.latest_odom.pose.pose.position.x,
                self.latest_odom.pose.pose.position.y,
                get_yaw(self.latest_odom.pose.pose.orientation),
            )
        bev = self.bev_generator.scan_msg_to_bev(
            self.latest_scan, goal_dx, goal_dy, odom_trail,
            route_points, ego_odom=ego_odom)
        bev_t = torch.from_numpy(bev).unsqueeze(0).float().to(self.device)
        bev_occ = torch.max(bev_t[:, 0], bev_t[:, 1])

        if self.model_version == 'v2':
            # --- Modular v2 inputs ---
            scan_current, scan_residuals, _ = \
                self.bev_generator.get_scan_temporal_data()
            if scan_current is None:
                return

            # Replace inf/nan with 0 for tensor
            scan_current = np.where(
                np.isfinite(scan_current), scan_current, 0.0
            ).astype(np.float32)

            scan_t = torch.from_numpy(scan_current).unsqueeze(0).to(
                self.device)
            res_t = torch.from_numpy(scan_residuals).unsqueeze(0).to(
                self.device)

            # Goal features
            goal_feat = compute_goal_features(goal_dx, goal_dy)
            goal_t = torch.tensor(
                goal_feat, dtype=torch.float32
            ).unsqueeze(0).to(self.device)

            # Multi-sample generation
            with torch.no_grad():
                (best_vel, best_poses, all_vel, all_poses,
                 best_idx, scores, self.hidden) = \
                    self.model.generate_multi_sample(
                        scan_t, res_t, goal_t, odom_t,
                        hidden=self.hidden,
                        bev_occupancy=bev_occ,
                        goal_dx=goal_dx, goal_dy=goal_dy,
                        warm_start=self.warm_start)
        else:
            # --- Legacy v1 BEV input ---
            with torch.no_grad():
                (best_vel, best_poses, all_vel, all_poses,
                 best_idx, scores, self.hidden) = \
                    self.model.generate_multi_sample(
                        bev_t, odom_t, hidden=self.hidden,
                        bev_occupancy=bev_occ,
                        goal_dx=goal_dx, goal_dy=goal_dy,
                        warm_start=self.warm_start)

        # Update warm-start
        self.warm_start = best_vel.view(1, -1)
        v_raw = best_vel[0, 0, 0].item()
        w_raw = best_vel[0, 0, 1].item()

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

        # --- Safety layer ---
        obstacle_in_front = abs(self.min_range_angle) < np.pi / 3
        if self.min_range < self.safety_min and obstacle_in_front:
            v, omega = 0.0, 0.0
            self.get_logger().warn(
                f"E-STOP: obstacle at {self.min_range:.2f}m")
        elif self.min_range < self.safety_slow and obstacle_in_front:
            v = min(v, 0.1)

        # --- OOD Detection (BEFORE publish — reduces speed when uncertain) ---
        if all_poses.shape[0] >= 2:
            endpoints = all_poses[:, -1, :2]
            spread = endpoints.std(dim=0).sum().item()
            if spread > 1.0:
                v = min(v, 0.05)
                if not self._ood_warned:
                    self.get_logger().warn(
                        f"OOD: spread={spread:.2f}m — reducing speed")
                    self._ood_warned = True
            else:
                self._ood_warned = False

        # --- Stuck detection ---
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

        # --- Publish velocity ---
        msg = Twist()
        msg.linear.x = float(v)
        msg.angular.z = float(omega)
        self.cmd_pub.publish(msg)

        # --- Visualize ---
        if self.visualize:
            self._publish_trajectory_viz(all_poses, best_idx, scores)
            self._publish_best_trajectory_path(best_poses)

        # --- Logging ---
        latency_ms = (time.time() - t0) * 1000
        self._log_count += 1
        if self._log_count % 150 == 0:
            self.get_logger().info(
                f"v={v:.3f} w={omega:.3f} "
                f"best={best_idx}/{self.model.n_samples} "
                f"score={scores[best_idx]:.2f} "
                f"goal=({goal_dx:.1f},{goal_dy:.1f}) "
                f"min_r={self.min_range:.2f}m "
                f"lat={latency_ms:.1f}ms")

    # ---- Visualization ----

    def _publish_trajectory_viz(self, all_poses, best_idx, scores):
        K = all_poses.shape[0]
        markers = MarkerArray()
        stamp = self.get_clock().now().to_msg()

        for k in range(K):
            marker = Marker()
            marker.header.frame_id = 'base_link'
            marker.header.stamp = stamp
            marker.ns = 'kinoflow_trajectories'
            marker.id = k
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD

            if k == best_idx:
                marker.scale.x = 0.04
                marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)
            else:
                marker.scale.x = 0.015
                marker.color = ColorRGBA(r=0.3, g=0.5, b=1.0, a=0.4)

            for t in range(all_poses.shape[1]):
                pt = Point()
                pt.x = float(all_poses[k, t, 0].item())
                pt.y = float(all_poses[k, t, 1].item())
                pt.z = 0.05
                marker.points.append(pt)

            marker.lifetime.sec = 0
            marker.lifetime.nanosec = int(0.2 * 1e9)
            markers.markers.append(marker)

        text_marker = Marker()
        text_marker.header.frame_id = 'base_link'
        text_marker.header.stamp = stamp
        text_marker.ns = 'kinoflow_score'
        text_marker.id = K
        text_marker.type = Marker.TEXT_VIEW_FACING
        text_marker.action = Marker.ADD
        text_marker.pose.position.x = float(
            all_poses[best_idx, -1, 0].item())
        text_marker.pose.position.y = float(
            all_poses[best_idx, -1, 1].item())
        text_marker.pose.position.z = 0.2
        text_marker.scale.z = 0.08
        text_marker.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
        text_marker.text = f"S:{scores[best_idx]:.1f}"
        text_marker.lifetime.sec = 0
        text_marker.lifetime.nanosec = int(0.2 * 1e9)
        markers.markers.append(text_marker)

        self.traj_viz_pub.publish(markers)

    def _publish_best_trajectory_path(self, best_poses):
        path = Path()
        path.header.frame_id = 'base_link'
        path.header.stamp = self.get_clock().now().to_msg()
        for t in range(best_poses.shape[1]):
            ps = PoseStamped()
            ps.header = path.header
            ps.pose.position.x = float(best_poses[0, t, 0].item())
            ps.pose.position.y = float(best_poses[0, t, 1].item())
            theta = float(best_poses[0, t, 2].item())
            ps.pose.orientation.z = float(np.sin(theta / 2))
            ps.pose.orientation.w = float(np.cos(theta / 2))
            path.poses.append(ps)
        self.best_traj_pub.publish(path)

    # ---- Recovery ----

    def _execute_recovery(self):
        cmd = Twist()
        self.get_logger().info("Recovery: backing up")
        cmd.linear.x = -0.1
        for _ in range(25):
            self.cmd_pub.publish(cmd)
            time.sleep(0.1)
        cmd.linear.x = 0.0
        self.cmd_pub.publish(cmd)
        time.sleep(1.0)
        self.get_logger().info("Recovery: spinning")
        cmd.angular.z = 0.25
        for _ in range(21):
            self.cmd_pub.publish(cmd)
            time.sleep(0.1)
        cmd.angular.z = 0.0
        self.cmd_pub.publish(cmd)
        self.hidden = None
        self.warm_start = None
        self.ema_v = 0.0
        self.ema_omega = 0.0
        self.prev_v = 0.0
        self.prev_omega = 0.0
        self.get_logger().info("Recovery complete — state reset")

    def _publish_stop(self):
        msg = Twist()
        self.cmd_pub.publish(msg)
        self.prev_v = 0.0
        self.prev_omega = 0.0
        self.ema_v = 0.0
        self.ema_omega = 0.0


def main(args=None):
    rclpy.init(args=args)
    node = KinoFlowNode()
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
