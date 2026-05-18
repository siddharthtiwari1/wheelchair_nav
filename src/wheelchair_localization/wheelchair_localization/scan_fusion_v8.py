#!/usr/bin/env python3
"""
SCAN FUSION V8 — RANGE-ADAPTIVE MIN FUSION
===========================================
Created: 2026-02-25
Based on: scan_fusion_v7.py (MIN fusion + PointCloud2 + async callbacks)

V7 problem: flat override_min_delta (0.4m) treats 1m and 5m camera data
the same. At 1m, 0.4m delta is meaningful (40% error = real obstacle).
At 5m, 0.4m is only 8% — well within camera noise (D455: ~100mm at 5m).
This causes:
  - Thick/blurry walls from camera noise overriding lidar at range
  - Ray streak artifacts from long-range camera depth errors
  - Turn artifacts when side cameras sweep into lidar FOV with parallax

V8 fixes (4 changes, each independently testable):
  1. RANGE-DEPENDENT OVERRIDE DELTA: delta(r) = base + scale * r
     At 1m: 0.15 + 0.08*1 = 0.23m (catches table legs)
     At 3m: 0.15 + 0.08*3 = 0.39m (rejects worst camera noise)
     At 5m: 0.15 + 0.08*5 = 0.55m (hard to override = good)
     Relaxed from 0.25+0.12*r after data showed map identical to lidar-only.

  2. GAP-FILL MAX RANGE: Don't gap-fill with camera data beyond 3.0m.
     Gap-fills are the most dangerous for SLAM (change scan shape).
     Long-range gap-fills are always camera noise, never real obstacles.

  3. ANGULAR VELOCITY GATING: When turning (|omega| > 0.15 rad/s),
     reduce gap-fills to 5/camera. Side cameras sweep into lidar gaps
     during turns, adding 60+ bins at once → SLAM matcher fails.

  4. OVERRIDE RATE-LIMITING: Geometry-derived per-camera limits.
     Computes each camera's angular coverage from TF + HFOV (87°),
     limits overrides to 15% of covered bins. Closest overrides kept.
     ~967 bins/cam × 15% = ~145/cam × 3 = ~435 max total.
     Hard ceiling: max_overrides_per_camera (safety cap).

Unchanged from v7:
  - Async callbacks (no TimeSynchronizer)
  - Footprint filter, rear crop
  - Stride-4 downsampling before transform
  - Per-camera debug scan publishers
  - Dedup guard on scan messages
  - Final zero cleanup before publish

Architecture:
  Camera CB (6Hz per camera, async):
    PointCloud2 -> parse -> downsample -> depth filter -> transform
    -> height filter -> polar -> range filter -> store overlay

  Odom CB (20Hz):
    /odometry/filtered -> extract angular velocity -> store for gating

  Scan CB (10Hz):
    copy /scan_filtered -> NaN->inf -> footprint filter -> rear crop
    -> range-adaptive MIN merge from camera overlays -> publish /scan_fused

Paired with: slam_toolbox_fused_v11.yaml (use_fused_slam:=true)
             or slam_toolbox_motion_compensated_v2.yaml (lidar-only)
DO NOT EDIT — create a new versioned file (scan_fusion_v9.py)
"""

import array as _array
import numpy as np
from typing import Optional, Dict
from dataclasses import dataclass
import time
import threading

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from rclpy.time import Time
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup

from sensor_msgs.msg import LaserScan, PointCloud2
from nav_msgs.msg import Odometry
from sensor_msgs_py import point_cloud2
from tf2_ros import Buffer, TransformListener

SENSOR_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1
)


@dataclass
class CameraOverlay:
    """Per-camera processed polar data: min range per scan bin."""
    min_ranges: np.ndarray  # float32[num_bins], inf = no data
    timestamp: float        # time.monotonic() when processed
    point_count: int        # valid points contributed


class WheelchairFootprintFilter:
    """URDF-calibrated self-detection filter (same as v6/v7)."""

    def __init__(self):
        self.min_valid_range = 0.20
        self.robot_half_width = 0.33
        self.robot_rear = 0.50
        self.robot_front = 0.20

        # (start_deg, end_deg, max_range_m)
        self.exclusion_zones_deg = [
            ( 150,  180, 1.00),
            (-180, -140, 1.00),
            ( 120,  150, 0.50),
            (-140, -100, 0.65),
            (  90,  120, 0.35),
            (-100,  -90, 0.35),
            ( -35,  -23, 0.32),
            (  50,   60, 0.48),
            (  22,   32, 0.45),
        ]
        self.exclusion_zones_rad = [
            (np.radians(a1), np.radians(a2), r)
            for a1, a2, r in self.exclusion_zones_deg
        ]
        self._cached = False

    def cache_geometry(self, n: int, robot_angles: np.ndarray):
        self._angles = robot_angles
        self._cos_a = np.cos(self._angles)
        self._sin_a = np.sin(self._angles)
        self._arc_masks = []
        for a_start, a_end, max_r in self.exclusion_zones_rad:
            if a_start <= a_end:
                mask = (self._angles >= a_start) & (self._angles <= a_end)
            else:
                mask = (self._angles >= a_start) | (self._angles <= a_end)
            self._arc_masks.append((mask, max_r))
        self._cached = True

    def filter_scan(self, ranges: np.ndarray) -> np.ndarray:
        valid = np.isfinite(ranges) & (ranges > 0)
        ranges[valid & (ranges < self.min_valid_range)] = np.inf

        for arc_mask, max_r in self._arc_masks:
            ranges[valid & arc_mask & (ranges < max_r)] = np.inf

        x = np.where(valid, ranges * self._cos_a, 0.0)
        y = np.where(valid, ranges * self._sin_a, 0.0)
        in_box = (
            valid
            & (x >= -self.robot_rear) & (x <= self.robot_front)
            & (y >= -self.robot_half_width) & (y <= self.robot_half_width)
        )
        ranges[in_box] = np.inf
        return ranges


class ScanFusionV8(Node):
    """Range-adaptive MIN fusion: lidar + 3 PointCloud2 cameras + odom gating."""

    def __init__(self):
        super().__init__('scan_fusion')

        # ---- Parameters (v7 inherited) ----
        self.declare_parameter('scan_topic', '/scan_filtered')
        self.declare_parameter('output_topic', '/scan_fused')
        self.declare_parameter('laser_frame', 'laser')

        self.declare_parameter('max_camera_range', 3.5)
        self.declare_parameter('min_camera_range', 0.30)
        self.declare_parameter('min_height', 0.10)
        self.declare_parameter('max_height', 1.80)

        self.declare_parameter('max_camera_age_ms', 500.0)
        self.declare_parameter('camera_warmup_sec', 3.0)

        self.declare_parameter('enable_footprint', True)
        self.declare_parameter('rear_crop_deg', 135.0)
        self.declare_parameter('downsample_stride', 4)
        self.declare_parameter('max_gap_fills_per_camera', 20)
        self.declare_parameter('min_camera_points_per_bin', 5)

        # ---- NEW V8: Range-dependent override delta ----
        # delta(r) = base_delta + range_scale * camera_range
        # Replaces v7's flat override_min_delta=0.4
        self.declare_parameter('override_base_delta', 0.25)
        self.declare_parameter('override_range_scale', 0.12)

        # ---- NEW V8: Gap-fill max range ----
        # Don't gap-fill with camera data beyond this range.
        # Long-range camera data is noisy (error grows as z^2/(f*b)).
        # Gap-fills change scan shape → SLAM matcher confusion.
        self.declare_parameter('gap_fill_max_range', 2.5)

        # ---- NEW V8: Angular velocity gating ----
        # During turns, side cameras sweep into lidar gaps.
        # Reduce gap-fills when turning to prevent abrupt scan shape changes.
        self.declare_parameter('odom_topic', '/odometry/filtered')
        self.declare_parameter('angular_velocity_threshold', 0.15)
        # Gap-fills per camera when turning (reduced from max_gap_fills)
        self.declare_parameter('turning_gap_fills', 5)

        # ---- NEW V8: Override rate-limiting ----
        # Cap overrides per camera to prevent flooding SLAM matcher.
        # Data: 453 overrides/scan with no limit → SLAM can't match.
        # 30/cam × 3 = 90 max → 10% of scan modified → SLAM handles it.
        self.declare_parameter('max_overrides_per_camera', 30)

        # Camera HFOV for angular coverage computation
        self.declare_parameter('camera_hfov_deg', 87.0)
        # Fraction of camera-covered bins allowed as overrides (0.10 = 10%)
        self.declare_parameter('override_fraction', 0.10)

        # Camera configs — PointCloud2 topics
        cameras_config = [
            ('front_camera', '/camera/depth/color/points',
             'camera_depth_optical_frame'),
            ('left_camera', '/mapping_camera/depth/color/points',
             'mapping_camera_depth_optical_frame'),
            ('right_camera', '/right_camera/depth/color/points',
             'right_camera_depth_optical_frame'),
        ]

        for name, pc_topic, frame in cameras_config:
            self.declare_parameter(f'{name}.enabled', True)
            self.declare_parameter(f'{name}.topic', pc_topic)
            self.declare_parameter(f'{name}.frame', frame)
            self.declare_parameter(f'{name}.max_depth', 3.5)
            self.declare_parameter(f'{name}.min_depth', 0.30)

        # ---- Read parameters ----
        self.laser_frame = self.get_parameter('laser_frame').value
        self.max_cam_range = float(self.get_parameter('max_camera_range').value)
        self.min_cam_range = float(self.get_parameter('min_camera_range').value)
        self.min_height = float(self.get_parameter('min_height').value)
        self.max_height = float(self.get_parameter('max_height').value)
        self.max_camera_age = float(self.get_parameter('max_camera_age_ms').value) / 1000.0
        self.camera_warmup = float(self.get_parameter('camera_warmup_sec').value)
        self.rear_crop_deg = float(self.get_parameter('rear_crop_deg').value)
        self.enable_footprint = bool(self.get_parameter('enable_footprint').value)
        self.downsample = int(self.get_parameter('downsample_stride').value)
        self.max_gap_fills = int(self.get_parameter('max_gap_fills_per_camera').value)
        self.min_points_per_bin = int(self.get_parameter('min_camera_points_per_bin').value)

        # V8 params
        self.override_base_delta = float(self.get_parameter('override_base_delta').value)
        self.override_range_scale = float(self.get_parameter('override_range_scale').value)
        self.gap_fill_max_range = float(self.get_parameter('gap_fill_max_range').value)
        self.omega_threshold = float(self.get_parameter('angular_velocity_threshold').value)
        self.turning_gap_fills = int(self.get_parameter('turning_gap_fills').value)
        self.max_overrides = int(self.get_parameter('max_overrides_per_camera').value)
        self.camera_hfov = float(self.get_parameter('camera_hfov_deg').value)
        self.override_fraction = float(self.get_parameter('override_fraction').value)

        # ---- Footprint filter ----
        self.footprint_filter = WheelchairFootprintFilter() if self.enable_footprint else None

        # ---- Callback groups ----
        self._cam_cb_group = ReentrantCallbackGroup()
        self._scan_cb_group = MutuallyExclusiveCallbackGroup()
        self._odom_cb_group = MutuallyExclusiveCallbackGroup()

        # ---- TF ----
        self.tf_buffer = Buffer(cache_time=rclpy.duration.Duration(seconds=10))
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ---- State ----
        self._lock = threading.Lock()
        self._overlays: Dict[str, Optional[CameraOverlay]] = {}
        self._tf_cache: Dict[str, Optional[tuple]] = {}
        self._num_bins = 0
        self._angle_min = 0.0
        self._angle_increment = 0.0
        self._rear_crop_mask = None
        self._initialized = False
        self._start_time = time.monotonic()

        # V8: angular velocity state
        self._current_omega = 0.0

        # Per-camera override limits derived from FOV geometry
        self._cam_max_overrides: Dict[str, int] = {}

        # Stats
        self._frame_count = 0
        self._total_fills = 0
        self._total_overrides = 0
        self._total_latency_us = 0
        self._max_latency_us = 0
        self._turn_gated_count = 0

        # ---- Publishers ----
        self.fused_pub = self.create_publisher(
            LaserScan, self.get_parameter('output_topic').value, 10)
        self.lidar_pub = self.create_publisher(LaserScan, '/scan_lidar_only', 10)

        self._cam_pubs: Dict[str, any] = {}

        # ---- Subscribe to lidar ----
        self.create_subscription(
            LaserScan, self.get_parameter('scan_topic').value,
            self._scan_cb, SENSOR_QOS,
            callback_group=self._scan_cb_group)

        # ---- Subscribe to odometry (V8: angular velocity gating) ----
        self.create_subscription(
            Odometry, self.get_parameter('odom_topic').value,
            self._odom_cb, SENSOR_QOS,
            callback_group=self._odom_cb_group)

        # ---- Subscribe to cameras ----
        self._cam_names = []
        for name, _, _ in cameras_config:
            enabled = self.get_parameter(f'{name}.enabled').value
            if not enabled:
                continue
            topic = self.get_parameter(f'{name}.topic').value
            frame = self.get_parameter(f'{name}.frame').value

            self._cam_names.append(name)
            self._overlays[name] = None

            scan_topic_name = f'/scan_{name}'
            self._cam_pubs[name] = self.create_publisher(LaserScan, scan_topic_name, 10)

            setattr(self, f'_{name}_frame', frame)
            setattr(self, f'_{name}_min_depth',
                    float(self.get_parameter(f'{name}.min_depth').value))
            setattr(self, f'_{name}_max_depth',
                    float(self.get_parameter(f'{name}.max_depth').value))

            self.create_subscription(
                PointCloud2, topic,
                lambda msg, n=name: self._pointcloud_cb(msg, n), SENSOR_QOS,
                callback_group=self._cam_cb_group)

        # ---- Diagnostics timer ----
        self.create_timer(10.0, self._print_stats)

        info = self.get_logger().info
        info('=' * 60)
        info('SCAN FUSION V8 — Range-Adaptive MIN Fusion')
        info('=' * 60)
        info(f'  Cameras: {", ".join(self._cam_names)}')
        info(f'  Fusion: MIN (range-adaptive delta)')
        info(f'  Cam range: {self.min_cam_range:.1f}-{self.max_cam_range:.1f} m')
        info(f'  Height: {self.min_height:.2f}-{self.max_height:.2f} m')
        info(f'  Downsample: stride {self.downsample}')
        info(f'  Override delta: {self.override_base_delta:.2f} + '
             f'{self.override_range_scale:.2f}*r  '
             f'(at 1m={self.override_base_delta + self.override_range_scale:.2f}m, '
             f'at 3m={self.override_base_delta + 3*self.override_range_scale:.2f}m)')
        info(f'  Gap-fill max range: {self.gap_fill_max_range:.1f} m')
        info(f'  Turn gating: |omega| > {self.omega_threshold:.2f} rad/s '
             f'-> gap-fills reduced to {self.turning_gap_fills}/camera')
        info(f'  Override fraction: {self.override_fraction:.0%} of covered bins '
             f'(ceiling: {self.max_overrides}/camera)')
        info(f'  Min points/bin: {self.min_points_per_bin}')
        info(f'  Footprint filter: {self.enable_footprint}')
        info(f'  Rear crop: +/-{self.rear_crop_deg:.0f} deg')

    # ------------------------------------------------------------------
    # Odometry callback — extract angular velocity for turn gating
    # ------------------------------------------------------------------
    def _odom_cb(self, msg: Odometry):
        self._current_omega = abs(msg.twist.twist.angular.z)

    # ------------------------------------------------------------------
    # Get cached TF (camera optical frame -> laser frame)
    # ------------------------------------------------------------------
    def _get_tf(self, cam_frame: str):
        cached = self._tf_cache.get(cam_frame)
        if cached is not None:
            return cached
        try:
            tf = self.tf_buffer.lookup_transform(
                self.laser_frame, cam_frame, Time())
            t = tf.transform.translation
            q = tf.transform.rotation
            x, y, z, w = q.x, q.y, q.z, q.w
            R = np.array([
                [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
                [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
                [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]
            ], dtype=np.float32)
            t_vec = np.array([t.x, t.y, t.z], dtype=np.float32)
            self._tf_cache[cam_frame] = (R, t_vec)
            self.get_logger().info(
                f'TF {cam_frame} -> {self.laser_frame}: '
                f't=[{t.x:.3f}, {t.y:.3f}, {t.z:.3f}]')
            return (R, t_vec)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Compute angular coverage of camera in laser frame from TF rotation
    # ------------------------------------------------------------------
    def _compute_camera_coverage(self, cam_name: str, R: np.ndarray):
        """Compute angular coverage of camera in laser frame from TF rotation.

        Camera optical frame: Z=forward, X=right, Y=down.
        Project FOV edges to laser frame XY plane, count covered bins.
        """
        half_fov = np.radians(self.camera_hfov / 2.0)

        # Camera forward direction in optical frame = [0, 0, 1]
        # Left FOV edge = [-sin(half_fov), 0, cos(half_fov)]
        # Right FOV edge = [sin(half_fov), 0, cos(half_fov)]
        dirs_optical = np.array([
            [-np.sin(half_fov), 0.0, np.cos(half_fov)],  # left edge
            [0.0, 0.0, 1.0],                               # center
            [np.sin(half_fov), 0.0, np.cos(half_fov)],    # right edge
        ], dtype=np.float32)

        # Transform to laser frame (rotation only — direction vectors)
        dirs_laser = dirs_optical @ R.T

        # Project to XY plane and get angles
        angles = np.arctan2(dirs_laser[:, 1], dirs_laser[:, 0])
        left_angle, center_angle, right_angle = angles

        # Count bins in angular span (handle wrapping)
        # Normalize so span goes from right_angle to left_angle CCW
        span = (left_angle - right_angle) % (2 * np.pi)
        n_bins = int(span / self._angle_increment)

        # Derive override limit
        geo_limit = max(1, int(self.override_fraction * n_bins))
        # Apply hard ceiling
        cam_limit = min(geo_limit, self.max_overrides)
        self._cam_max_overrides[cam_name] = cam_limit

        self.get_logger().info(
            f'  {cam_name}: covers {np.degrees(right_angle):.0f}\u00b0 to '
            f'{np.degrees(left_angle):.0f}\u00b0 (center {np.degrees(center_angle):.0f}\u00b0), '
            f'{n_bins} bins, max_overrides={cam_limit} '
            f'({self.override_fraction:.0%} of {n_bins})')

    # ------------------------------------------------------------------
    # PointCloud2 camera callback — process and store as overlay
    # ------------------------------------------------------------------
    def _pointcloud_cb(self, msg: PointCloud2, cam_name: str):
        if not self._initialized:
            return

        cam_frame = getattr(self, f'_{cam_name}_frame')
        tf_data = self._get_tf(cam_frame)
        if tf_data is None:
            return

        R, t_vec = tf_data

        # Compute angular coverage once per camera
        if cam_name not in self._cam_max_overrides:
            self._compute_camera_coverage(cam_name, R)
        min_depth = getattr(self, f'_{cam_name}_min_depth')
        max_depth = getattr(self, f'_{cam_name}_max_depth')

        pts = point_cloud2.read_points_numpy(
            msg, field_names=('x', 'y', 'z'), skip_nans=True)

        if len(pts) == 0:
            return

        if pts.dtype.names is not None:
            x = pts['x']
            y = pts['y']
            z = pts['z']
        elif pts.ndim == 2 and pts.shape[1] >= 3:
            x = pts[:, 0]
            y = pts[:, 1]
            z = pts[:, 2]
        else:
            return

        # Downsample FIRST (before transform — saves CPU)
        if self.downsample > 1:
            x = x[::self.downsample]
            y = y[::self.downsample]
            z = z[::self.downsample]

        # Depth filter in camera optical frame
        depth_mask = (z >= min_depth) & (z <= max_depth)
        if not np.any(depth_mask):
            return
        x, y, z = x[depth_mask], y[depth_mask], z[depth_mask]

        # Transform to laser frame
        pts_cam = np.column_stack([x, y, z]).astype(np.float32)
        pts_laser = pts_cam @ R.T + t_vec

        # Height filter in laser frame
        z_laser = pts_laser[:, 2]
        height_ok = (z_laser >= self.min_height) & (z_laser <= self.max_height)
        pts_laser = pts_laser[height_ok]
        if len(pts_laser) == 0:
            return

        # Project to 2D polar
        x_l = pts_laser[:, 0]
        y_l = pts_laser[:, 1]
        ranges = np.sqrt(x_l * x_l + y_l * y_l)
        angles = np.arctan2(y_l, x_l)

        # Range filter
        range_ok = (ranges >= self.min_cam_range) & (ranges <= self.max_cam_range)
        ranges = ranges[range_ok]
        angles = angles[range_ok]
        if len(ranges) == 0:
            return

        # Bin into overlay — min range per bin
        indices = np.rint(
            (angles - self._angle_min) / self._angle_increment
        ).astype(np.int32)
        in_bounds = (indices >= 0) & (indices < self._num_bins)
        idx = indices[in_bounds]
        rng = ranges[in_bounds]

        overlay_ranges = np.full(self._num_bins, np.inf, dtype=np.float32)
        if len(idx) > 0:
            np.minimum.at(overlay_ranges, idx, rng)
            if self.min_points_per_bin > 1:
                overlay_counts = np.zeros(self._num_bins, dtype=np.int32)
                np.add.at(overlay_counts, idx, 1)
                overlay_ranges[overlay_counts < self.min_points_per_bin] = np.inf

        overlay = CameraOverlay(
            min_ranges=overlay_ranges,
            timestamp=time.monotonic(),
            point_count=len(idx)
        )

        with self._lock:
            self._overlays[cam_name] = overlay

    # ------------------------------------------------------------------
    # Scan callback — lidar + footprint + rear crop + range-adaptive merge
    # ------------------------------------------------------------------
    def _scan_cb(self, scan_msg: LaserScan):
        # Dedup guard
        stamp = (scan_msg.header.stamp.sec, scan_msg.header.stamp.nanosec)
        if hasattr(self, '_last_scan_stamp') and stamp == self._last_scan_stamp:
            return
        self._last_scan_stamp = stamp

        t0 = time.monotonic_ns()

        # Initialize on first scan
        if not self._initialized:
            self._num_bins = len(scan_msg.ranges)
            self._angle_min = scan_msg.angle_min
            self._angle_increment = scan_msg.angle_increment

            scan_angles = self._angle_min + np.arange(
                self._num_bins, dtype=np.float32) * self._angle_increment

            robot_angles = np.arctan2(
                -np.sin(scan_angles), -np.cos(scan_angles)
            ).astype(np.float32)

            if self.rear_crop_deg < 180.0:
                limit_rad = np.radians(self.rear_crop_deg)
                self._rear_crop_mask = (robot_angles > limit_rad) | (robot_angles < -limit_rad)
                self.get_logger().info(
                    f'Rear crop: +/-{self.rear_crop_deg:.0f} deg '
                    f'({int(np.sum(self._rear_crop_mask))} bins cropped)')

            if self.footprint_filter is not None:
                self.footprint_filter.cache_geometry(self._num_bins, robot_angles)

            self._initialized = True
            self.get_logger().info(
                f'Scan: {self._num_bins} bins, '
                f'scan [{np.degrees(scan_angles[0]):.1f}, '
                f'{np.degrees(scan_angles[-1]):.1f}] deg (laser frame)')

        # Step 1: Copy lidar, clean invalid
        fused = np.array(scan_msg.ranges, dtype=np.float32)
        np.nan_to_num(fused, nan=np.inf, copy=False)
        fused[fused <= 0.0] = np.inf

        # Step 2: Footprint filter
        if self.footprint_filter is not None:
            self.footprint_filter.filter_scan(fused)

        # Step 3: Rear crop
        if self._rear_crop_mask is not None:
            fused[self._rear_crop_mask] = np.inf

        # Publish lidar-only debug
        self._publish(fused.copy(), scan_msg, self.lidar_pub)

        # Step 4: Range-adaptive camera MIN fusion
        now = time.monotonic()
        if (now - self._start_time) < self.camera_warmup:
            self._publish(fused, scan_msg, self.fused_pub)
            return

        # V8: Determine gap-fill limit based on angular velocity
        is_turning = self._current_omega > self.omega_threshold
        effective_gap_fills = self.turning_gap_fills if is_turning else self.max_gap_fills

        n_filled = 0
        n_overrides = 0
        with self._lock:
            for cam_name in self._cam_names:
                ov = self._overlays.get(cam_name)
                if ov is None or ov.point_count == 0:
                    continue
                age = now - ov.timestamp
                if age > self.max_camera_age:
                    continue

                cam_valid = np.isfinite(ov.min_ranges)

                # Publish per-camera debug scan
                cam_pub = self._cam_pubs.get(cam_name)
                if cam_pub is not None:
                    self._publish(ov.min_ranges.copy(), scan_msg, cam_pub)

                # Case 1: Gap-fill — lidar has no data, camera does
                # V8: Also enforce gap_fill_max_range — don't fill with
                # noisy long-range camera data.
                lidar_gap = np.isinf(fused)
                fill = (lidar_gap & cam_valid
                        & (ov.min_ranges <= self.gap_fill_max_range))
                n_fill = int(np.sum(fill))
                if n_fill > 0:
                    if n_fill > effective_gap_fills:
                        fill_idx = np.where(fill)[0]
                        fill_ranges = ov.min_ranges[fill_idx]
                        closest = np.argsort(fill_ranges)[:effective_gap_fills]
                        fill[:] = False
                        fill[fill_idx[closest]] = True
                        n_fill = effective_gap_fills
                    fused[fill] = ov.min_ranges[fill]
                    n_filled += n_fill

                # Case 2: MIN override — range-dependent delta
                # V8: delta = base + scale * camera_range (per-bin)
                # Camera must be delta(r) closer than lidar to override.
                # At close range: small delta → camera trusted more
                # At far range: large delta → camera nearly ignored
                lidar_has_data = np.isfinite(fused) & ~lidar_gap
                delta = self.override_base_delta + self.override_range_scale * ov.min_ranges
                closer = (lidar_has_data & cam_valid
                          & (ov.min_ranges < fused - delta))
                n_override = int(np.sum(closer))
                if n_override > 0:
                    cam_limit = self._cam_max_overrides.get(cam_name, self.max_overrides)
                    if n_override > cam_limit:
                        # Keep closest overrides (most safety-critical)
                        override_idx = np.where(closer)[0]
                        override_ranges = ov.min_ranges[override_idx]
                        closest = np.argsort(override_ranges)[:cam_limit]
                        closer[:] = False
                        closer[override_idx[closest]] = True
                        n_override = cam_limit
                    fused[closer] = ov.min_ranges[closer]
                    n_overrides += n_override

        self._total_fills += n_filled
        self._total_overrides += n_overrides
        self._frame_count += 1
        if is_turning:
            self._turn_gated_count += 1

        # Final cleanup: zero/negative/NaN → inf
        bad = ~np.isfinite(fused) | (fused <= 0.0)
        fused[bad] = np.inf

        self._publish(fused, scan_msg, self.fused_pub)

        elapsed_us = (time.monotonic_ns() - t0) // 1000
        self._total_latency_us += elapsed_us
        if elapsed_us > self._max_latency_us:
            self._max_latency_us = elapsed_us

    # ------------------------------------------------------------------
    def _publish(self, ranges: np.ndarray, template: LaserScan, pub):
        msg = LaserScan()
        msg.header.stamp = template.header.stamp
        msg.header.frame_id = template.header.frame_id
        msg.angle_min = template.angle_min
        msg.angle_max = template.angle_max
        msg.angle_increment = template.angle_increment
        msg.time_increment = template.time_increment
        msg.scan_time = template.scan_time
        msg.range_min = template.range_min
        msg.range_max = template.range_max
        msg.ranges = _array.array('f', ranges.astype(np.float32, copy=False).tobytes())
        msg.intensities = []
        pub.publish(msg)

    # ------------------------------------------------------------------
    def _print_stats(self):
        fc = self._frame_count
        if fc == 0:
            return
        avg_us = self._total_latency_us // fc
        avg_fills = self._total_fills // fc
        avg_overrides = self._total_overrides // fc
        turn_pct = (self._turn_gated_count * 100) // fc
        cam_status = []
        now = time.monotonic()
        for name in self._cam_names:
            ov = self._overlays.get(name)
            if ov and ov.point_count > 0:
                age = now - ov.timestamp
                cam_status.append(f'{name}:{ov.point_count}pts({age*1000:.0f}ms)')
            else:
                cam_status.append(f'{name}:none')

        self.get_logger().info(
            f'[V8] {fc} frames | {avg_us}us avg / {self._max_latency_us}us max | '
            f'gap-fills: {avg_fills}/frame | overrides: {avg_overrides}/frame | '
            f'turn-gated: {turn_pct}% | {" | ".join(cam_status)}')

        self._frame_count = 0
        self._total_fills = 0
        self._total_overrides = 0
        self._total_latency_us = 0
        self._max_latency_us = 0
        self._turn_gated_count = 0


def main(args=None):
    rclpy.init(args=args)
    node = ScanFusionV8()
    # 5 threads: 1 scan CB + 3 camera CBs + 1 odom CB
    executor = MultiThreadedExecutor(num_threads=5)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
