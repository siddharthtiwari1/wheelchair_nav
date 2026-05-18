#!/usr/bin/env python3
"""
SCAN FUSION V2 — HIGH PERFORMANCE 3-CAMERA + LIDAR FUSION
==========================================================
Complete rewrite fixing ALL bugs from v4 audit (2026-02-21).

Bug fixes vs robust_scan_fusion.py:
  1. TF buffer: 30s → 5s (prevents memory bloat + slow O(n) lookups)
  2. Static TF cache: permanent (no 60s expiry, no repeated lookups)
  3. Heavy work moved to camera callbacks (off the critical lidar path)
  4. Early downsampling: stride BEFORE any computation (not after full read)
  5. Vectorized binning: sort + reduceat O(N log N) replaces np.minimum.at
  6. Single footprint filter pass AFTER fusion (was 5× before: 1+3+1)
  7. Thread safety: snapshot camera data inside lock, process outside
  8. Camera freshness: 250ms for 6Hz cameras (was 150ms — too tight)
  9. Pre-allocated scan params: zero repeated computation in hot path
  10. float32 throughout: no mixed precision, no unnecessary copies

Architecture:
  Camera callbacks (6Hz each, ~3ms):
    parse → downsample → depth filter → transform → height filter → polar
    → store as immutable ProcessedCamera

  Scan callback (10Hz, target <2ms):
    copy lidar → snapshot cameras → bin each → footprint filter → publish

Performance targets:
  - Scan callback: <2ms on laptop CPU
  - Camera callback: ~3ms each at 6Hz
  - Total CPU: <10% single core (was 60-100%)
  - Memory: constant (no growth over hours of operation)
  - Zero GC pressure in hot path

Author: Rewrite from audit findings
Date: 2026-02-21
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, List
import time
import threading

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from rclpy.time import Time
from rclpy.duration import Duration

from sensor_msgs.msg import LaserScan, PointCloud2
from sensor_msgs_py import point_cloud2
from nav_msgs.msg import Odometry
from std_msgs.msg import Header

import tf2_ros
from tf2_ros import TransformException


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CameraConfig:
    """Per-camera configuration loaded from ROS parameters."""
    name: str
    topic: str
    frame: str
    enabled: bool = True
    min_depth: float = 0.40
    max_depth: float = 3.0
    downsample: int = 8


@dataclass
class OdomPose2D:
    """Lightweight 2D odometry snapshot for motion compensation."""
    x: float
    y: float
    theta: float


@dataclass
class ProcessedCamera:
    """Pre-processed camera data ready for fast scan binning.

    Created in camera callback (heavy work), consumed in scan callback
    (lightweight). Immutable after creation — no race conditions.
    """
    ranges: np.ndarray    # float32 polar ranges in laser frame
    angles: np.ndarray    # float32 polar angles in laser frame
    timestamp: float      # wall-clock receive time
    point_count: int      # valid points (diagnostics)
    odom_pose: Optional[OdomPose2D] = None  # robot pose when captured


# ---------------------------------------------------------------------------
# Wheelchair footprint filter (proven correct from v4, vectorized)
# ---------------------------------------------------------------------------

class WheelchairFootprintFilter:
    """
    URDF-calibrated self-detection filter.

    Removes lidar returns from wheelchair body (wheels, castors, frame).
    Based on exact wheelchair URDF geometry with safety margins.

    Convention (standard ROS LaserScan):
      0°    = FORWARD  (X+)
      +90°  = LEFT     (Y+)
      -90°  = RIGHT    (Y-)
      ±180° = BACKWARD (X-)
    """

    def __init__(self):
        self.min_valid_range = 0.20  # meters

        # Robot bounding box (cartesian backup filter)
        # Lidar is at (0.475, 0.12) from wheelchair_main center
        self.robot_half_width = 0.45   # 90 cm total width
        self.robot_rear = 1.0          # 1 m behind lidar
        self.robot_front = 0.30        # 30 cm in front of lidar (was 15cm — too small)

        # Angular exclusion zones: (start_deg, end_deg, max_range_m)
        # Calibrated from rosbag analysis (session_20260223_102114):
        #   Right front panel: -34° to -24°, 0.25m mean, 100% occupancy
        #   Left front panel:  +51° to +59°, 0.40m mean, 100% occupancy
        #   Left footrest:     +23° to +31°, 0.80m mean, 100% occupancy, std=0.003m
        self.exclusion_zones_deg = [
            ( 150,  180, 1.00),   # Left rear wheel
            (-180, -140, 1.00),   # Right rear wheel
            ( 120,  150, 0.50),   # Left castor
            (-140, -100, 0.65),   # Right castor
            (  90,  120, 0.35),   # Left side frame
            (-100,  -90, 0.35),   # Right side frame
            ( -35,  -23, 0.32),   # Right front panel/armrest (rosbag: 0.25m @ 100%)
            (  50,   60, 0.48),   # Left front panel/armrest (rosbag: 0.40m @ 100%)
            (  22,   32, 0.85),   # Left footrest guard (rosbag: 0.80m, std=0.003m)
        ]
        self.exclusion_zones_rad = [
            (np.radians(a1), np.radians(a2), r)
            for a1, a2, r in self.exclusion_zones_deg
        ]

    def filter_scan(self, ranges: np.ndarray, angle_min: float,
                    angle_increment: float) -> np.ndarray:
        """Remove self-detections from scan. Returns filtered copy."""
        result = ranges.copy()
        n = len(ranges)
        angles = angle_min + np.arange(n, dtype=np.float32) * angle_increment
        valid = np.isfinite(ranges) & (ranges > 0)

        # 1. Minimum range
        result[valid & (ranges < self.min_valid_range)] = np.inf

        # 2. Angular exclusion zones (vectorized)
        for a_start, a_end, max_r in self.exclusion_zones_rad:
            if a_start <= a_end:
                in_arc = (angles >= a_start) & (angles <= a_end)
            else:
                in_arc = (angles >= a_start) | (angles <= a_end)
            result[valid & in_arc & (ranges < max_r)] = np.inf

        # 3. Cartesian bounding box (backup for odd angles)
        cos_a = np.cos(angles)
        sin_a = np.sin(angles)
        x = np.where(valid, ranges * cos_a, 0.0)
        y = np.where(valid, ranges * sin_a, 0.0)
        in_box = (
            valid
            & (x >= -self.robot_rear) & (x <= self.robot_front)
            & (y >= -self.robot_half_width) & (y <= self.robot_half_width)
        )
        result[in_box] = np.inf

        return result


# ---------------------------------------------------------------------------
# Main node
# ---------------------------------------------------------------------------

class ScanFusionV2(Node):
    """
    High-performance 3-camera + LiDAR scan fusion.

    Heavy point-cloud processing happens in camera callbacks (6 Hz each).
    The scan callback only bins pre-processed data + filters + publishes.
    """

    def __init__(self):
        super().__init__('scan_fusion_v2')

        # ==================================================================
        # Parameters
        # ==================================================================
        self.declare_parameter('scan_topic', '/scan_filtered')
        self.declare_parameter('output_topic', '/scan_fused')
        self.declare_parameter('laser_frame', 'laser')
        self.declare_parameter('min_height', 0.05)  # Accept obstacles ≥12cm world (laser at z=0.07m)
        self.declare_parameter('max_height', 1.40)  # Reject ceiling
        self.declare_parameter('max_camera_age_ms', 500.0)  # 500ms: handles USB jitter + dropped frames at 6Hz
        self.declare_parameter('camera_grace_factor', 2.0)  # Grace period = factor × max_age (gap-fill only, no override)
        self.declare_parameter('enable_footprint_filter', True)
        self.declare_parameter('publish_lidar_only', True)
        self.declare_parameter('camera_override_margin', 0.15)  # 8× D455 noise at 3m (18mm)
        self.declare_parameter('min_camera_points_per_bin', 2)  # 2-point agreement filters noise
        self.declare_parameter('max_camera_range', 3.5)  # D455 reliable to 4m
        self.declare_parameter('camera_fill_gaps', True)  # Fill lidar shadows with camera data
        self.declare_parameter('odom_topic', '/odometry/filtered')
        self.declare_parameter('enable_motion_compensation', True)
        self.declare_parameter('motion_comp_max_dt', 0.30)

        for prefix in ('front_camera', 'left_camera', 'right_camera'):
            self.declare_parameter(f'{prefix}.enabled', True)
            self.declare_parameter(f'{prefix}.topic', '')
            self.declare_parameter(f'{prefix}.frame', '')
            self.declare_parameter(f'{prefix}.min_depth', 0.40)
            self.declare_parameter(f'{prefix}.max_depth', 4.0)
            self.declare_parameter(f'{prefix}.downsample', 4)  # Was 8 — more points for bin coverage

        # Load scalar params
        self.laser_frame = self.get_parameter('laser_frame').value
        self.min_height = float(self.get_parameter('min_height').value)
        self.max_height = float(self.get_parameter('max_height').value)
        self.max_camera_age = self.get_parameter('max_camera_age_ms').value / 1000.0
        self.camera_grace_age = self.max_camera_age * float(
            self.get_parameter('camera_grace_factor').value)  # Stale but usable for gap-fill
        self.enable_footprint = self.get_parameter('enable_footprint_filter').value
        self.publish_lidar_only = self.get_parameter('publish_lidar_only').value
        self.camera_override_margin = float(
            self.get_parameter('camera_override_margin').value)
        self.min_cam_points_per_bin = int(
            self.get_parameter('min_camera_points_per_bin').value)
        self.max_camera_range = float(
            self.get_parameter('max_camera_range').value)
        self.camera_fill_gaps = self.get_parameter('camera_fill_gaps').value
        self.odom_topic = self.get_parameter('odom_topic').value
        self.enable_motion_comp = self.get_parameter(
            'enable_motion_compensation').value
        self.motion_comp_max_dt = float(
            self.get_parameter('motion_comp_max_dt').value)

        # Build camera configs
        _defaults = {
            'front_camera': ('/camera/depth/color/points',
                             'camera_depth_optical_frame'),
            'left_camera':  ('/mapping_camera/depth/color/points',
                             'mapping_camera_depth_optical_frame'),
            'right_camera': ('/right_camera/depth/color/points',
                             'right_camera_depth_optical_frame'),
        }
        self.cameras: List[CameraConfig] = []
        for prefix, (def_topic, def_frame) in _defaults.items():
            topic = self.get_parameter(f'{prefix}.topic').value or def_topic
            frame = self.get_parameter(f'{prefix}.frame').value or def_frame
            self.cameras.append(CameraConfig(
                name=prefix.replace('_camera', ''),
                topic=topic,
                frame=frame,
                enabled=self.get_parameter(f'{prefix}.enabled').value,
                min_depth=float(self.get_parameter(f'{prefix}.min_depth').value),
                max_depth=float(self.get_parameter(f'{prefix}.max_depth').value),
                downsample=int(self.get_parameter(f'{prefix}.downsample').value),
            ))

        # ==================================================================
        # TF — small buffer, static transform cache
        # ==================================================================
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=5.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self._static_tfs: Dict[str, np.ndarray] = {}

        # ==================================================================
        # Pre-processed camera data (thread-safe)
        # ==================================================================
        self._cam_lock = threading.Lock()
        self._cam_data: Dict[str, Optional[ProcessedCamera]] = {
            cam.name: None for cam in self.cameras
        }

        # ==================================================================
        # Scan parameters (set on first lidar message)
        # ==================================================================
        self._angle_min = 0.0
        self._angle_max = 0.0
        self._angle_increment = 0.0
        self._num_bins = 0
        self._range_min = 0.0
        self._range_max = 0.0
        self._initialized = False

        # ==================================================================
        # Footprint filter
        # ==================================================================
        self.footprint_filter = WheelchairFootprintFilter()

        # ==================================================================
        # Motion compensation state (thread-safe)
        # ==================================================================
        self._odom_lock = threading.Lock()
        self._latest_odom: Optional[OdomPose2D] = None
        self._odom_received = False

        # ==================================================================
        # Statistics (protected by stats lock)
        # ==================================================================
        self._stats_lock = threading.Lock()
        self._frame_count = 0
        self._cam_fused_count = [0] * len(self.cameras)
        self._cam_stale_count = [0] * len(self.cameras)
        self._cam_missing_count = [0] * len(self.cameras)
        self._total_latency_us = 0
        self._max_latency_us = 0
        self._motion_comp_applied = 0
        self._motion_comp_skipped = 0

        # Per-camera filter pipeline stats (updated in camera callbacks)
        self._cam_name_to_idx = {cam.name: i for i, cam in enumerate(self.cameras)}
        self._cam_total_pts = [0] * len(self.cameras)
        self._cam_depth_pass = [0] * len(self.cameras)
        self._cam_height_pass = [0] * len(self.cameras)
        self._cam_range_pass = [0] * len(self.cameras)
        self._cam_bins_written = [0] * len(self.cameras)

        # ==================================================================
        # QoS — BEST_EFFORT, depth=1 for minimum latency
        # ==================================================================
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ==================================================================
        # Subscribers
        # ==================================================================
        self.create_subscription(
            LaserScan,
            self.get_parameter('scan_topic').value,
            self._scan_callback,
            sensor_qos,
        )

        for cam in self.cameras:
            if cam.enabled:
                # Use default arg trick to capture cam by value
                self.create_subscription(
                    PointCloud2, cam.topic,
                    lambda msg, c=cam: self._camera_callback(msg, c),
                    sensor_qos,
                )

        # Odometry subscription for motion compensation
        if self.enable_motion_comp:
            self.create_subscription(
                Odometry, self.odom_topic,
                self._odom_callback, sensor_qos,
            )

        # ==================================================================
        # Publishers
        # ==================================================================
        self.fused_pub = self.create_publisher(
            LaserScan, self.get_parameter('output_topic').value, 10)

        self.lidar_pub = None
        if self.publish_lidar_only:
            self.lidar_pub = self.create_publisher(
                LaserScan, '/scan_lidar_only', 10)

        # ==================================================================
        # Diagnostics timer
        # ==================================================================
        self.create_timer(10.0, self._print_stats)

        self._log_startup()

    # ------------------------------------------------------------------
    # Startup log
    # ------------------------------------------------------------------

    def _log_startup(self):
        info = self.get_logger().info
        info('=' * 60)
        info('SCAN FUSION V2 — HIGH PERFORMANCE')
        info('=' * 60)
        info(f'Input:       {self.get_parameter("scan_topic").value}')
        info(f'Output:      {self.get_parameter("output_topic").value}')
        info(f'Laser frame: {self.laser_frame}')
        info(f'Height:      [{self.min_height:.2f}, {self.max_height:.2f}] m')
        info(f'Camera age:  {self.max_camera_age * 1000:.0f} ms max '
             f'(grace: {self.camera_grace_age * 1000:.0f} ms gap-fill only)')
        info(f'Footprint:   {"ON" if self.enable_footprint else "OFF"}')
        info(f'Cam margin:  {self.camera_override_margin:.2f} m '
             f'(camera must be this much closer than lidar to override)')
        for cam in self.cameras:
            tag = 'ON' if cam.enabled else 'OFF'
            info(f'  {cam.name:6s}: {tag} | ds={cam.downsample} '
                 f'| depth=[{cam.min_depth:.1f}, {cam.max_depth:.1f}]m '
                 f'| {cam.topic}')
        info(f'Motion comp: {"ON" if self.enable_motion_comp else "OFF"}'
             f' (odom: {self.odom_topic}, max_dt: {self.motion_comp_max_dt:.2f}s)')
        info(f'Gap density: {self.min_cam_points_per_bin} pts/bin min')
        info('=' * 60)

        # Track per-camera TF status and first-callback logging
        self._cam_tf_warned = {cam.name: False for cam in self.cameras}
        self._cam_first_fused = {cam.name: False for cam in self.cameras}
        self._cam_callback_count = {cam.name: 0 for cam in self.cameras}
        self._cam_tf_fail_count = {cam.name: 0 for cam in self.cameras}

    # ------------------------------------------------------------------
    # Static TF caching — looked up once, stored forever
    # ------------------------------------------------------------------

    def _get_static_tf(self, target: str, source: str) -> Optional[np.ndarray]:
        """Return 4×4 homogeneous transform. Cached permanently."""
        key = f'{source}>{target}'
        cached = self._static_tfs.get(key)
        if cached is not None:
            return cached

        try:
            tf_msg = self.tf_buffer.lookup_transform(
                target, source, Time(), Duration(seconds=0.5))
        except TransformException:
            return None

        t = tf_msg.transform.translation
        q = tf_msg.transform.rotation

        # Quaternion to rotation matrix (float64 for accuracy, store once)
        norm = max(np.sqrt(q.x**2 + q.y**2 + q.z**2 + q.w**2), 1e-10)
        qx, qy, qz, qw = q.x / norm, q.y / norm, q.z / norm, q.w / norm

        T = np.eye(4, dtype=np.float64)
        T[0, 0] = 1.0 - 2.0 * (qy * qy + qz * qz)
        T[0, 1] = 2.0 * (qx * qy - qw * qz)
        T[0, 2] = 2.0 * (qx * qz + qw * qy)
        T[1, 0] = 2.0 * (qx * qy + qw * qz)
        T[1, 1] = 1.0 - 2.0 * (qx * qx + qz * qz)
        T[1, 2] = 2.0 * (qy * qz - qw * qx)
        T[2, 0] = 2.0 * (qx * qz - qw * qy)
        T[2, 1] = 2.0 * (qy * qz + qw * qx)
        T[2, 2] = 1.0 - 2.0 * (qx * qx + qy * qy)
        T[0, 3] = t.x
        T[1, 3] = t.y
        T[2, 3] = t.z

        # Pre-compute float32 rotation + translation for fast camera callback
        # Store as (R_f32, t_f32, T_f64) tuple
        R_f32 = T[:3, :3].astype(np.float32)
        t_f32 = T[:3, 3].astype(np.float32)

        self._static_tfs[key] = (R_f32, t_f32)
        self.get_logger().info(f'Cached static TF: {source} → {target}')
        return (R_f32, t_f32)

    # ------------------------------------------------------------------
    # Odometry callback — lightweight pose caching for motion compensation
    # ------------------------------------------------------------------

    def _odom_callback(self, msg: Odometry):
        """Cache latest 2D odometry pose for motion compensation."""
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        theta = float(np.arctan2(siny_cosp, cosy_cosp))

        pose = OdomPose2D(
            x=msg.pose.pose.position.x,
            y=msg.pose.pose.position.y,
            theta=theta,
        )
        with self._odom_lock:
            self._latest_odom = pose
            if not self._odom_received:
                self._odom_received = True
                self.get_logger().info(
                    f'Motion compensation ACTIVE: receiving odom '
                    f'from {self.odom_topic}')

    # ------------------------------------------------------------------
    # Motion compensation — transform camera points for robot motion
    # ------------------------------------------------------------------

    @staticmethod
    def _motion_compensate(
        ranges: np.ndarray,
        angles: np.ndarray,
        cam_odom: OdomPose2D,
        scan_odom: OdomPose2D,
    ) -> tuple:
        """
        Transform camera polar points from cam-time laser frame to
        scan-time laser frame, compensating for robot motion between
        camera capture and lidar scan.

        Math (2D rigid body):
          p_odom = R(cam_theta) @ p_cam + [cam_x, cam_y]
          p_scan = R(-scan_theta) @ (p_odom - [scan_x, scan_y])

        Combined:
          dtheta = cam_theta - scan_theta
          dt = R(-scan_theta) @ [cam_x - scan_x, cam_y - scan_y]
          p_scan = R(dtheta) @ p_cam + dt
        """
        dtheta = cam_odom.theta - scan_odom.theta
        dx_odom = cam_odom.x - scan_odom.x
        dy_odom = cam_odom.y - scan_odom.y

        # Skip if motion is negligible
        if abs(dtheta) < 0.001 and abs(dx_odom) < 0.002 and abs(dy_odom) < 0.002:
            return ranges, angles

        # Polar to cartesian in cam-time laser frame
        x_cam = ranges * np.cos(angles)
        y_cam = ranges * np.sin(angles)

        # Rotation for angular delta
        cos_d = np.float32(np.cos(dtheta))
        sin_d = np.float32(np.sin(dtheta))

        # Translation in scan-time laser frame
        cos_s = np.cos(-scan_odom.theta)
        sin_s = np.sin(-scan_odom.theta)
        dt_x = np.float32(cos_s * dx_odom - sin_s * dy_odom)
        dt_y = np.float32(sin_s * dx_odom + cos_s * dy_odom)

        # Apply rigid transform: p_scan = R(dtheta) @ p_cam + dt
        x_scan = cos_d * x_cam - sin_d * y_cam + dt_x
        y_scan = sin_d * x_cam + cos_d * y_cam + dt_y

        # Back to polar
        new_ranges = np.sqrt(x_scan * x_scan + y_scan * y_scan)
        new_angles = np.arctan2(y_scan, x_scan)

        return new_ranges, new_angles

    # ------------------------------------------------------------------
    # Camera callback — heavy work (runs at 6 Hz per camera)
    # ------------------------------------------------------------------

    def _camera_callback(self, msg: PointCloud2, cam: CameraConfig):
        """
        Pre-process point cloud completely:
          parse → downsample → depth filter → transform → height filter → polar

        Stores immutable ProcessedCamera for the scan callback to consume.
        """
        self._cam_callback_count[cam.name] = self._cam_callback_count.get(cam.name, 0) + 1

        # Get static transform (cached after first successful lookup)
        tf_data = self._get_static_tf(self.laser_frame, cam.frame)
        if tf_data is None:
            cnt = self._cam_tf_fail_count.get(cam.name, 0) + 1
            self._cam_tf_fail_count[cam.name] = cnt
            # Log TF failure periodically (every 30 attempts = ~5s at 6Hz)
            if cnt <= 3 or cnt % 30 == 0:
                self.get_logger().warn(
                    f'[{cam.name}] TF lookup FAILED: {cam.frame} → {self.laser_frame} '
                    f'(attempt {cnt}) — camera data DROPPED')
            return
        R, t_vec = tf_data

        # Log first successful TF + camera callback
        if not self._cam_first_fused.get(cam.name, False):
            self._cam_first_fused[cam.name] = True
            self.get_logger().info(
                f'[{cam.name}] FIRST callback with valid TF — camera fusion ACTIVE')

        # Parse point cloud
        try:
            pts = point_cloud2.read_points_numpy(
                msg, field_names=('x', 'y', 'z'), skip_nans=True)
            if len(pts) == 0:
                return

            # Handle structured vs unstructured arrays
            if pts.dtype.names:
                x = pts['x']
                y = pts['y']
                z = pts['z']
                # Downsample FIRST (before column_stack)
                if cam.downsample > 1:
                    x = x[::cam.downsample]
                    y = y[::cam.downsample]
                    z = z[::cam.downsample]
                xyz = np.column_stack([
                    x.astype(np.float32),
                    y.astype(np.float32),
                    z.astype(np.float32),
                ])
            elif pts.ndim == 2:
                # Downsample FIRST
                if cam.downsample > 1:
                    pts = pts[::cam.downsample]
                xyz = pts[:, :3].astype(np.float32)
            else:
                return
        except Exception as e:
            self.get_logger().warn(
                f'[{cam.name}] point cloud parse error: {e}',
                throttle_duration_sec=5.0)
            return

        if len(xyz) < 10:
            return

        total_pts = len(xyz)

        # Depth filter in camera frame (z = depth for optical frame)
        depths = xyz[:, 2]
        depth_mask = (depths >= cam.min_depth) & (depths <= cam.max_depth)
        xyz = xyz[depth_mask]
        depth_pass = len(xyz)

        if len(xyz) < 5:
            return

        # Transform to laser frame: p_laser = p_cam @ R^T + t
        # Using xyz @ R.T is faster than R @ xyz.T for row-major arrays
        pts_laser = xyz @ R.T + t_vec

        # Height filter in laser frame
        z_laser = pts_laser[:, 2]
        height_mask = (z_laser >= self.min_height) & (z_laser <= self.max_height)
        pts_laser = pts_laser[height_mask]
        height_pass = len(pts_laser)

        if len(pts_laser) < 3:
            return

        # Convert to polar (range, angle)
        x_l = pts_laser[:, 0]
        y_l = pts_laser[:, 1]
        ranges = np.sqrt(x_l * x_l + y_l * y_l)
        angles = np.arctan2(y_l, x_l)

        # Reject very close points (wheelchair self-detection) and
        # clamp max polar range to prevent long-range projection noise
        valid_range = (ranges >= 0.25) & (ranges <= self.max_camera_range)
        ranges = ranges[valid_range]
        angles = angles[valid_range]

        if len(ranges) < 3:
            return

        # Accumulate filter pipeline stats
        cam_idx = self._cam_name_to_idx[cam.name]
        self._cam_total_pts[cam_idx] += total_pts
        self._cam_depth_pass[cam_idx] += depth_pass
        self._cam_height_pass[cam_idx] += height_pass
        self._cam_range_pass[cam_idx] += len(ranges)

        # Snapshot odometry for motion compensation
        odom_snapshot = None
        if self.enable_motion_comp:
            with self._odom_lock:
                odom_snapshot = self._latest_odom

        # Store immutable result
        processed = ProcessedCamera(
            ranges=ranges,
            angles=angles,
            timestamp=time.monotonic(),
            point_count=len(ranges),
            odom_pose=odom_snapshot,
        )
        with self._cam_lock:
            self._cam_data[cam.name] = processed

    # ------------------------------------------------------------------
    # Scan callback — lightweight (runs at 10 Hz lidar rate)
    # ------------------------------------------------------------------

    def _scan_callback(self, scan_msg: LaserScan):
        """
        Fuse lidar with pre-processed camera data.
        Target: <2 ms on laptop CPU.
        """
        t0 = time.monotonic_ns()
        now = time.monotonic()

        # Initialize scan parameters on first message
        if not self._initialized:
            if scan_msg.angle_increment <= 0 or len(scan_msg.ranges) < 10:
                self.get_logger().warn('Invalid first scan message, skipping')
                return
            self._angle_min = scan_msg.angle_min
            self._angle_max = scan_msg.angle_max
            self._angle_increment = scan_msg.angle_increment
            self._num_bins = len(scan_msg.ranges)
            self._range_min = scan_msg.range_min
            self._range_max = scan_msg.range_max
            self._initialized = True
            self.get_logger().info(
                f'Scan template: {self._num_bins} bins, '
                f'[{np.degrees(self._angle_min):.1f}°, '
                f'{np.degrees(self._angle_max):.1f}°], '
                f'range [{self._range_min:.2f}, {self._range_max:.2f}] m')

        # Start with lidar ranges — replace NaN with inf so camera can fill gaps
        # (RPLidar reports invalid as NaN, but np.minimum propagates NaN,
        #  blocking camera data from overwriting gaps)
        fused = np.asarray(scan_msg.ranges, dtype=np.float32)
        np.nan_to_num(fused, nan=np.inf, copy=False)

        # Save lidar-only snapshot for gradient filter (before camera fusion)
        lidar_snapshot = fused.copy()

        # Publish lidar-only debug topic (footprint-filtered)
        if self.lidar_pub is not None:
            if self.enable_footprint:
                lidar_clean = self.footprint_filter.filter_scan(
                    fused, scan_msg.angle_min, scan_msg.angle_increment)
            else:
                lidar_clean = fused
            self._publish_scan(lidar_clean, scan_msg, self.lidar_pub)

        # Camera warmup: for the first 5 seconds of operation, publish
        # lidar-only scans. Ensures SLAM establishes correct map orientation
        # from clean lidar geometry before noisy camera data is added.
        if not hasattr(self, '_first_scan_time'):
            self._first_scan_time = now
        camera_warmup_sec = 5.0
        cameras_ready = (now - self._first_scan_time) >= camera_warmup_sec

        # Snapshot camera data (minimal time under lock)
        with self._cam_lock:
            cam_snapshot = dict(self._cam_data)

        # Snapshot current odom for motion compensation
        scan_odom = None
        if self.enable_motion_comp:
            with self._odom_lock:
                scan_odom = self._latest_odom

        # Fuse each camera's pre-processed polar data (skip during warmup)
        compensated_this_frame = False
        for i, cam in enumerate(self.cameras):
            if not cameras_ready:
                break
            if not cam.enabled:
                continue

            processed = cam_snapshot.get(cam.name)
            if processed is None:
                with self._stats_lock:
                    self._cam_missing_count[i] += 1
                continue

            age = now - processed.timestamp
            if age > self.camera_grace_age:
                # Beyond grace period — truly stale, discard entirely
                with self._stats_lock:
                    self._cam_stale_count[i] += 1
                continue

            # Determine if camera data is fresh (full fusion) or in grace period (gap-fill only)
            grace_only = age > self.max_camera_age  # Stale but within grace → gap-fill, no override

            # Motion compensation: adjust camera points for robot motion
            cam_ranges = processed.ranges
            cam_angles = processed.angles
            if (self.enable_motion_comp
                    and scan_odom is not None
                    and processed.odom_pose is not None
                    and age <= self.motion_comp_max_dt):
                cam_ranges, cam_angles = self._motion_compensate(
                    cam_ranges, cam_angles,
                    processed.odom_pose, scan_odom)
                compensated_this_frame = True

            # Vectorized binning into fused scan (the hot path)
            bins_written = self._bin_polar_min(
                cam_ranges, cam_angles, fused, gap_fill_only=grace_only)
            self._cam_bins_written[i] += bins_written

            with self._stats_lock:
                self._cam_fused_count[i] += 1

        # Track motion compensation stats
        with self._stats_lock:
            if compensated_this_frame:
                self._motion_comp_applied += 1
            else:
                self._motion_comp_skipped += 1

        # Temporal persistence filter: camera data must appear in 2+ consecutive
        # scans in the same bin. Random noise is different each frame; real
        # obstacles persist. This eliminates both single-bin AND multi-bin
        # transient artifacts that the gradient filter can't catch.
        if cameras_ready:
            # Current camera-contribution mask
            cam_mask = ~np.isfinite(lidar_snapshot) & np.isfinite(fused)

            if not hasattr(self, '_prev_cam_mask'):
                self._prev_cam_mask = np.zeros(len(fused), dtype=np.bool_)

            # Only keep camera bins that were also camera-contributed last scan
            transient = cam_mask & ~self._prev_cam_mask
            fused[transient] = np.inf  # Reject first-frame camera data

            # Update persistence mask for next scan
            self._prev_cam_mask = cam_mask.copy()

        # Gradient filter: remove remaining isolated spikes
        if cameras_ready:
            fused = self._gradient_filter(fused, lidar_snapshot)

        # Single footprint filter pass AFTER all fusion
        if self.enable_footprint:
            fused = self.footprint_filter.filter_scan(
                fused, scan_msg.angle_min, scan_msg.angle_increment)

        # Publish fused result
        self._publish_scan(fused, scan_msg, self.fused_pub)

        # Track latency
        elapsed_us = (time.monotonic_ns() - t0) // 1000
        with self._stats_lock:
            self._frame_count += 1
            self._total_latency_us += elapsed_us
            if elapsed_us > self._max_latency_us:
                self._max_latency_us = elapsed_us

    # ------------------------------------------------------------------
    # Gradient filter — removes flying-pixel spikes from camera data
    # ------------------------------------------------------------------

    @staticmethod
    def _gradient_filter(fused: np.ndarray, lidar: np.ndarray,
                         max_jump: float = 0.80) -> np.ndarray:
        """
        Remove camera-contributed bins that show range discontinuities
        with their neighbors. Targets flying-pixel artifacts at depth edges.

        A bin is invalidated if:
          1. It was contributed by camera (lidar was inf at that bin), AND
          2. The range jumps > max_jump from BOTH adjacent bins (isolated spike)

        Legitimate camera detections (e.g., table legs) have at least one
        neighbor at a similar range, so they survive this filter.
        """
        result = fused.copy()
        n = len(fused)
        if n < 3:
            return result

        # Camera-only bins: lidar was inf but fused is finite
        cam_mask = ~np.isfinite(lidar) & np.isfinite(fused)
        if not np.any(cam_mask):
            return result

        # Vectorized: compute left/right jumps
        left_jump = np.full(n, np.inf)
        right_jump = np.full(n, np.inf)

        # Left neighbor differences
        left_valid = np.isfinite(fused[:-1])
        left_jump[1:] = np.where(left_valid, np.abs(fused[1:] - fused[:-1]), np.inf)

        # Right neighbor differences
        right_valid = np.isfinite(fused[1:])
        right_jump[:-1] = np.where(right_valid, np.abs(fused[:-1] - fused[1:]), np.inf)

        # Reject: camera bin AND both neighbors jump too much
        reject = cam_mask & (left_jump > max_jump) & (right_jump > max_jump)
        result[reject] = np.inf

        return result

    # ------------------------------------------------------------------
    # Vectorized binning — O(N log N) via sort + reduceat
    # ------------------------------------------------------------------

    def _bin_polar_min(self, ranges: np.ndarray, angles: np.ndarray,
                       scan_out: np.ndarray, gap_fill_only: bool = False):
        """
        Bin polar (range, angle) points into scan array.

        Camera data is only written when:
          1. Lidar has NO reading (inf) AND >= min_cam_points_per_bin camera
             points agree on that bin (rejects single-point noise), OR
          2. Camera range < lidar range - margin — a genuinely closer obstacle
             (e.g., table leg in front of wall)

        If gap_fill_only=True (grace period), only path 1 is used — stale
        camera data can fill lidar gaps but cannot override lidar readings.
        This prevents flickering when cameras temporarily drop frames.

        This prevents camera depth noise (±2-4 cm) from smearing clean
        lidar wall readings while still detecting elevated obstacles.
        """
        margin = self.camera_override_margin
        min_pts = self.min_cam_points_per_bin
        fill_gaps = self.camera_fill_gaps
        # Plausible range guard: camera can only override lidar when the lidar
        # reading is within 2x camera's max polar range. Blocks multi-path IR
        # cross-talk (e.g., camera reads 1.2m, lidar reads 4m → impossible).
        max_plausible = self.max_camera_range * 2.0

        # Compute bin indices (rint rounds to nearest; truncation creates half-bin bias)
        indices = np.rint(
            (angles - self._angle_min) / self._angle_increment
        ).astype(np.int32)

        # Keep only valid bins + valid ranges
        valid = (
            (indices >= 0)
            & (indices < self._num_bins)
            & (ranges >= self._range_min)
            & (ranges <= self._range_max)
        )
        idx = indices[valid]
        rng = ranges[valid]

        n = len(idx)
        if n == 0:
            return 0

        # Fast path: few points, direct loop is faster than sort overhead
        if n < 32:
            bin_counts = {}
            bin_mins = {}
            for j in range(n):
                b = int(idx[j])
                r = float(rng[j])
                if b not in bin_mins or r < bin_mins[b]:
                    bin_mins[b] = r
                bin_counts[b] = bin_counts.get(b, 0) + 1

            bins_written = 0
            for b, r in bin_mins.items():
                cur = scan_out[b]
                cnt = bin_counts[b]
                if not np.isfinite(cur):
                    # Gap fill: only if enabled + enough points agree
                    if fill_gaps and cnt >= min_pts:
                        scan_out[b] = r
                        bins_written += 1
                elif (not gap_fill_only
                      and r < cur - margin
                      and cnt >= min_pts
                      and cur <= max_plausible):
                    # Genuinely closer obstacle: requires point agreement
                    # AND lidar must be within plausible camera range (blocks cross-talk)
                    # Disabled during grace period to prevent stale overrides
                    scan_out[b] = r
                    bins_written += 1
            return bins_written

        # Sort by bin index
        order = np.argsort(idx, kind='quicksort')
        sorted_idx = idx[order]
        sorted_rng = rng[order]

        # Find segment boundaries (where bin index changes)
        changes = np.empty(n, dtype=np.bool_)
        changes[0] = True
        np.not_equal(sorted_idx[1:], sorted_idx[:-1], out=changes[1:])
        starts = np.flatnonzero(changes)

        # Minimum range per segment via reduceat
        mins = np.minimum.reduceat(sorted_rng, starts)
        bins = sorted_idx[starts]

        # Count points per segment
        counts = np.empty(len(starts), dtype=np.int32)
        counts[:-1] = starts[1:] - starts[:-1]
        counts[-1] = n - starts[-1]

        # Only write camera data where:
        #   - lidar gap AND enough camera points agree → fill gap
        #   - camera is significantly closer → real obstacle in front of wall
        current = scan_out[bins]
        lidar_gap = ~np.isfinite(current)

        # Both paths require minimum point density
        enough_pts = counts >= min_pts

        if gap_fill_only:
            # Grace period: only fill lidar gaps, never override
            if fill_gaps:
                write_mask = lidar_gap & enough_pts
            else:
                return 0  # Nothing to do in grace period without gap-fill
        else:
            genuinely_closer = (
                np.isfinite(current)
                & (mins < (current - margin))
                & (current <= max_plausible)  # Lidar must be in camera's plausible zone
            )
            closer_ok = genuinely_closer & enough_pts
            if fill_gaps:
                gap_fill_ok = lidar_gap & enough_pts
                write_mask = gap_fill_ok | closer_ok
            else:
                write_mask = closer_ok

        if np.any(write_mask):
            write_bins = bins[write_mask]
            scan_out[write_bins] = mins[write_mask]
            return int(len(write_bins))
        return 0

    # ------------------------------------------------------------------
    # Publish helper
    # ------------------------------------------------------------------

    def _publish_scan(self, ranges: np.ndarray, template: LaserScan,
                      publisher):
        """Create and publish LaserScan from numpy array."""
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
        msg.ranges = ranges.tolist()
        msg.intensities = []
        publisher.publish(msg)

    # ------------------------------------------------------------------
    # Diagnostics (every 10 s)
    # ------------------------------------------------------------------

    def _print_stats(self):
        with self._stats_lock:
            fc = self._frame_count
            if fc == 0:
                self.get_logger().warn(
                    '[V2] 0 lidar frames received! Check /scan_filtered topic')
                return

            fused = list(self._cam_fused_count)
            stale = list(self._cam_stale_count)
            missing = list(self._cam_missing_count)
            avg_us = self._total_latency_us // max(fc, 1)
            max_us = self._max_latency_us
            mc_applied = self._motion_comp_applied
            mc_skipped = self._motion_comp_skipped

            # Per-camera filter pipeline stats
            cam_total = list(self._cam_total_pts)
            cam_depth = list(self._cam_depth_pass)
            cam_height = list(self._cam_height_pass)
            cam_range = list(self._cam_range_pass)
            cam_bins = list(self._cam_bins_written)

            # Reset
            self._frame_count = 0
            self._cam_fused_count = [0] * len(self.cameras)
            self._cam_stale_count = [0] * len(self.cameras)
            self._cam_missing_count = [0] * len(self.cameras)
            self._total_latency_us = 0
            self._max_latency_us = 0
            self._motion_comp_applied = 0
            self._motion_comp_skipped = 0
            self._cam_total_pts = [0] * len(self.cameras)
            self._cam_depth_pass = [0] * len(self.cameras)
            self._cam_height_pass = [0] * len(self.cameras)
            self._cam_range_pass = [0] * len(self.cameras)
            self._cam_bins_written = [0] * len(self.cameras)

        # Build per-camera status with detail
        parts = []
        for i, cam in enumerate(self.cameras):
            if not cam.enabled:
                parts.append(f'{cam.name}:OFF')
                continue
            total = fused[i] + stale[i] + missing[i]
            if total > 0:
                pct = 100 * fused[i] / total
                if fused[i] == 0:
                    # Explain WHY it's 0%
                    if missing[i] > 0 and stale[i] == 0:
                        parts.append(f'{cam.name}:0%(no_data)')
                    elif stale[i] > 0:
                        parts.append(f'{cam.name}:0%(stale)')
                    else:
                        parts.append(f'{cam.name}:0%')
                else:
                    avg_bins = cam_bins[i] // max(fused[i], 1)
                    if cam_total[i] > 0:
                        avg_t = cam_total[i] // max(fused[i], 1)
                        avg_d = cam_depth[i] // max(fused[i], 1)
                        avg_h = cam_height[i] // max(fused[i], 1)
                        avg_r = cam_range[i] // max(fused[i], 1)
                        parts.append(
                            f'{cam.name}:{pct:.0f}%({avg_bins}bins, '
                            f'{avg_t}→{avg_d}→{avg_h}→{avg_r}pts)')
                    else:
                        parts.append(f'{cam.name}:{pct:.0f}%({avg_bins}bins)')
            else:
                parts.append(f'{cam.name}:--')

        mc_total = mc_applied + mc_skipped
        mc_str = (f'mcomp: {mc_applied}/{mc_total}'
                  if mc_total > 0 else 'mcomp: --')

        self.get_logger().info(
            f'[V2] {fc} frames | '
            f'scan_cb: {avg_us} μs avg / {max_us} μs max | '
            f'{mc_str} | '
            f'{" | ".join(parts)}')

        if max_us > 10_000:
            self.get_logger().warn(
                f'Scan callback exceeded 10 ms! ({max_us} μs) — '
                f'check CPU load or increase camera downsample')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    node = ScanFusionV2()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
