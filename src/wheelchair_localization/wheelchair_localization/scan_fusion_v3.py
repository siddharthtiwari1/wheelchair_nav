#!/usr/bin/env python3
"""
SCAN FUSION V3 — CONSISTENT GAP-FILLING FUSION FOR SLAM
=========================================================
Based on scan_fusion_v2.py (2026-02-21). Fixes v5 SLAM collapse caused by
over-aggressive temporal filter + disabled gap-fill.

Changes from v2:
  1. REMOVED hard temporal persistence filter (caused data starvation in v5)
  2. ADDED spatial consistency filter: gap-fill bins need >= 1 finite neighbor
     within +/-2 bins. Real obstacles have spatial extent; isolated noise doesn't.
  3. ADDED optional range smoothing (EMA): camera-contributed bins are smoothed
     with previous frame to reduce frame-to-frame jitter for SLAM consistency.
  4. IMPROVED gradient filter: also rejects 2-bin camera clusters with high
     range jumps, not just single-bin spikes.
  5. Camera warmup reduced: 5s -> 3s (less data starvation at startup)

Key principle: Gap-fill MUST be enabled for fused SLAM. AMCL at runtime uses
/scan_fused, so the SLAM map must be built from the same source. Camera data
fills lidar blind spots and detects elevated obstacles.

v5 failure analysis:
  - camera_fill_gaps=False starved SLAM of camera data
  - Hard 2-frame temporal filter rejected ALL first-appearance camera data
  - Tight SLAM params (0.05 angle, 0.40 search) couldn't recover
  - Result: complete map collapse with fan-shaped ray artifacts

v3 approach:
  - Gap-fill ON (cameras fill lidar shadows)
  - Spatial filter catches isolated noise WITHOUT frame-to-frame dependency
  - Range smoothing provides consistency WITHOUT rejecting data
  - Gradient filter catches flying pixels at depth edges
  - SLAM config v6 (balanced between v4 and v5)

Architecture (unchanged from v2):
  Camera callbacks (6Hz each, ~3ms):
    parse -> downsample -> depth filter -> transform -> height filter -> polar
    -> store as immutable ProcessedCamera

  Scan callback (10Hz, target <2ms):
    copy lidar -> snapshot cameras -> bin each -> spatial filter -> gradient
    filter -> range smooth -> footprint filter -> publish

Performance targets (unchanged):
  - Scan callback: <3ms on laptop CPU (slightly more than v2 due to spatial filter)
  - Camera callback: ~3ms each at 6Hz
  - Total CPU: <10% single core
  - Memory: constant

Author: Iteration from v2 audit + v5 failure analysis
Date: 2026-02-24
Paired with: slam_toolbox_fused_v6.yaml
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

class ScanFusionV3(Node):
    """
    Consistent gap-filling 3-camera + LiDAR scan fusion for SLAM.

    Key improvement over V2: spatial consistency filter replaces hard
    temporal persistence, gap-fill always enabled, range smoothing for
    frame-to-frame consistency.
    """

    def __init__(self):
        super().__init__('scan_fusion_v3')

        # ==================================================================
        # Parameters
        # ==================================================================
        self.declare_parameter('scan_topic', '/scan_filtered')
        self.declare_parameter('output_topic', '/scan_fused')
        self.declare_parameter('laser_frame', 'laser')
        self.declare_parameter('min_height', 0.05)
        self.declare_parameter('max_height', 1.40)
        self.declare_parameter('max_camera_age_ms', 500.0)
        self.declare_parameter('camera_grace_factor', 2.0)
        self.declare_parameter('enable_footprint_filter', True)
        self.declare_parameter('publish_lidar_only', True)
        self.declare_parameter('camera_override_margin', 0.15)
        self.declare_parameter('min_camera_points_per_bin', 2)
        self.declare_parameter('max_camera_range', 3.5)
        self.declare_parameter('camera_fill_gaps', True)
        self.declare_parameter('odom_topic', '/odometry/filtered')
        self.declare_parameter('enable_motion_compensation', True)
        self.declare_parameter('motion_comp_max_dt', 0.30)

        # V3 new parameters
        self.declare_parameter('enable_spatial_filter', True)
        self.declare_parameter('spatial_neighbor_radius', 2)  # Check +/- N bins
        self.declare_parameter('enable_range_smoothing', True)
        self.declare_parameter('range_smooth_alpha', 0.7)  # 0.7 = 70% current, 30% previous
        self.declare_parameter('camera_warmup_sec', 3.0)  # Was 5.0 in v2

        for prefix in ('front_camera', 'left_camera', 'right_camera'):
            self.declare_parameter(f'{prefix}.enabled', True)
            self.declare_parameter(f'{prefix}.topic', '')
            self.declare_parameter(f'{prefix}.frame', '')
            self.declare_parameter(f'{prefix}.min_depth', 0.40)
            self.declare_parameter(f'{prefix}.max_depth', 4.0)
            self.declare_parameter(f'{prefix}.downsample', 4)

        # Load scalar params
        self.laser_frame = self.get_parameter('laser_frame').value
        self.min_height = float(self.get_parameter('min_height').value)
        self.max_height = float(self.get_parameter('max_height').value)
        self.max_camera_age = self.get_parameter('max_camera_age_ms').value / 1000.0
        self.camera_grace_age = self.max_camera_age * float(
            self.get_parameter('camera_grace_factor').value)
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

        # V3 params
        self.enable_spatial_filter = self.get_parameter('enable_spatial_filter').value
        self.spatial_neighbor_radius = int(
            self.get_parameter('spatial_neighbor_radius').value)
        self.enable_range_smoothing = self.get_parameter('enable_range_smoothing').value
        self.range_smooth_alpha = float(
            self.get_parameter('range_smooth_alpha').value)
        self.camera_warmup_sec = float(
            self.get_parameter('camera_warmup_sec').value)

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
        # V3: Range smoothing state (previous camera ranges per bin)
        # ==================================================================
        self._prev_cam_ranges: Optional[np.ndarray] = None

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
        self._spatial_rejected = 0
        self._gradient_rejected = 0

        # Per-camera filter pipeline stats
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
                self.create_subscription(
                    PointCloud2, cam.topic,
                    lambda msg, c=cam: self._camera_callback(msg, c),
                    sensor_qos,
                )

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
        info('SCAN FUSION V3 — CONSISTENT GAP-FILLING FOR SLAM')
        info('=' * 60)
        info(f'Input:       {self.get_parameter("scan_topic").value}')
        info(f'Output:      {self.get_parameter("output_topic").value}')
        info(f'Laser frame: {self.laser_frame}')
        info(f'Height:      [{self.min_height:.2f}, {self.max_height:.2f}] m')
        info(f'Camera age:  {self.max_camera_age * 1000:.0f} ms max '
             f'(grace: {self.camera_grace_age * 1000:.0f} ms gap-fill only)')
        info(f'Footprint:   {"ON" if self.enable_footprint else "OFF"}')
        info(f'Cam margin:  {self.camera_override_margin:.2f} m')
        info(f'Gap fill:    {"ON" if self.camera_fill_gaps else "OFF"}')
        info(f'Spatial:     {"ON" if self.enable_spatial_filter else "OFF"}'
             f' (radius={self.spatial_neighbor_radius})')
        info(f'Smoothing:   {"ON" if self.enable_range_smoothing else "OFF"}'
             f' (alpha={self.range_smooth_alpha:.2f})')
        info(f'Warmup:      {self.camera_warmup_sec:.1f}s')
        for cam in self.cameras:
            tag = 'ON' if cam.enabled else 'OFF'
            info(f'  {cam.name:6s}: {tag} | ds={cam.downsample} '
                 f'| depth=[{cam.min_depth:.1f}, {cam.max_depth:.1f}]m '
                 f'| {cam.topic}')
        info(f'Motion comp: {"ON" if self.enable_motion_comp else "OFF"}'
             f' (odom: {self.odom_topic}, max_dt: {self.motion_comp_max_dt:.2f}s)')
        info(f'Gap density: {self.min_cam_points_per_bin} pts/bin min')
        info('=' * 60)

        self._cam_tf_warned = {cam.name: False for cam in self.cameras}
        self._cam_first_fused = {cam.name: False for cam in self.cameras}
        self._cam_callback_count = {cam.name: 0 for cam in self.cameras}
        self._cam_tf_fail_count = {cam.name: 0 for cam in self.cameras}

    # ------------------------------------------------------------------
    # Static TF caching — looked up once, stored forever
    # ------------------------------------------------------------------

    def _get_static_tf(self, target: str, source: str) -> Optional[np.ndarray]:
        """Return 4x4 homogeneous transform. Cached permanently."""
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

        R_f32 = T[:3, :3].astype(np.float32)
        t_f32 = T[:3, 3].astype(np.float32)

        self._static_tfs[key] = (R_f32, t_f32)
        self.get_logger().info(f'Cached static TF: {source} -> {target}')
        return (R_f32, t_f32)

    # ------------------------------------------------------------------
    # Odometry callback
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
    # Motion compensation
    # ------------------------------------------------------------------

    @staticmethod
    def _motion_compensate(
        ranges: np.ndarray,
        angles: np.ndarray,
        cam_odom: OdomPose2D,
        scan_odom: OdomPose2D,
    ) -> tuple:
        """Transform camera polar points for robot motion between capture times."""
        dtheta = cam_odom.theta - scan_odom.theta
        dx_odom = cam_odom.x - scan_odom.x
        dy_odom = cam_odom.y - scan_odom.y

        if abs(dtheta) < 0.001 and abs(dx_odom) < 0.002 and abs(dy_odom) < 0.002:
            return ranges, angles

        x_cam = ranges * np.cos(angles)
        y_cam = ranges * np.sin(angles)

        cos_d = np.float32(np.cos(dtheta))
        sin_d = np.float32(np.sin(dtheta))

        cos_s = np.cos(-scan_odom.theta)
        sin_s = np.sin(-scan_odom.theta)
        dt_x = np.float32(cos_s * dx_odom - sin_s * dy_odom)
        dt_y = np.float32(sin_s * dx_odom + cos_s * dy_odom)

        x_scan = cos_d * x_cam - sin_d * y_cam + dt_x
        y_scan = sin_d * x_cam + cos_d * y_cam + dt_y

        new_ranges = np.sqrt(x_scan * x_scan + y_scan * y_scan)
        new_angles = np.arctan2(y_scan, x_scan)

        return new_ranges, new_angles

    # ------------------------------------------------------------------
    # Camera callback — heavy work (runs at 6 Hz per camera)
    # ------------------------------------------------------------------

    def _camera_callback(self, msg: PointCloud2, cam: CameraConfig):
        """Pre-process point cloud completely. Stores immutable ProcessedCamera."""
        self._cam_callback_count[cam.name] = self._cam_callback_count.get(cam.name, 0) + 1

        tf_data = self._get_static_tf(self.laser_frame, cam.frame)
        if tf_data is None:
            cnt = self._cam_tf_fail_count.get(cam.name, 0) + 1
            self._cam_tf_fail_count[cam.name] = cnt
            if cnt <= 3 or cnt % 30 == 0:
                self.get_logger().warn(
                    f'[{cam.name}] TF lookup FAILED: {cam.frame} -> {self.laser_frame} '
                    f'(attempt {cnt}) — camera data DROPPED')
            return
        R, t_vec = tf_data

        if not self._cam_first_fused.get(cam.name, False):
            self._cam_first_fused[cam.name] = True
            self.get_logger().info(
                f'[{cam.name}] FIRST callback with valid TF — camera fusion ACTIVE')

        try:
            pts = point_cloud2.read_points_numpy(
                msg, field_names=('x', 'y', 'z'), skip_nans=True)
            if len(pts) == 0:
                return

            if pts.dtype.names:
                x = pts['x']
                y = pts['y']
                z = pts['z']
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

        depths = xyz[:, 2]
        depth_mask = (depths >= cam.min_depth) & (depths <= cam.max_depth)
        xyz = xyz[depth_mask]
        depth_pass = len(xyz)

        if len(xyz) < 5:
            return

        pts_laser = xyz @ R.T + t_vec

        z_laser = pts_laser[:, 2]
        height_mask = (z_laser >= self.min_height) & (z_laser <= self.max_height)
        pts_laser = pts_laser[height_mask]
        height_pass = len(pts_laser)

        if len(pts_laser) < 3:
            return

        x_l = pts_laser[:, 0]
        y_l = pts_laser[:, 1]
        ranges = np.sqrt(x_l * x_l + y_l * y_l)
        angles = np.arctan2(y_l, x_l)

        valid_range = (ranges >= 0.25) & (ranges <= self.max_camera_range)
        ranges = ranges[valid_range]
        angles = angles[valid_range]

        if len(ranges) < 3:
            return

        cam_idx = self._cam_name_to_idx[cam.name]
        self._cam_total_pts[cam_idx] += total_pts
        self._cam_depth_pass[cam_idx] += depth_pass
        self._cam_height_pass[cam_idx] += height_pass
        self._cam_range_pass[cam_idx] += len(ranges)

        odom_snapshot = None
        if self.enable_motion_comp:
            with self._odom_lock:
                odom_snapshot = self._latest_odom

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
        """Fuse lidar with pre-processed camera data. Target: <3 ms."""
        t0 = time.monotonic_ns()
        now = time.monotonic()

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
                f'[{np.degrees(self._angle_min):.1f}, '
                f'{np.degrees(self._angle_max):.1f}], '
                f'range [{self._range_min:.2f}, {self._range_max:.2f}] m')

        # Start with lidar ranges — NaN → inf so camera can fill gaps
        fused = np.asarray(scan_msg.ranges, dtype=np.float32)
        np.nan_to_num(fused, nan=np.inf, copy=False)

        # Save lidar-only snapshot (before camera fusion)
        lidar_snapshot = fused.copy()

        # Publish lidar-only debug topic
        if self.lidar_pub is not None:
            if self.enable_footprint:
                lidar_clean = self.footprint_filter.filter_scan(
                    fused, scan_msg.angle_min, scan_msg.angle_increment)
            else:
                lidar_clean = fused
            self._publish_scan(lidar_clean, scan_msg, self.lidar_pub)

        # Camera warmup: lidar-only for first N seconds
        if not hasattr(self, '_first_scan_time'):
            self._first_scan_time = now
        cameras_ready = (now - self._first_scan_time) >= self.camera_warmup_sec

        # Snapshot camera data
        with self._cam_lock:
            cam_snapshot = dict(self._cam_data)

        scan_odom = None
        if self.enable_motion_comp:
            with self._odom_lock:
                scan_odom = self._latest_odom

        # Fuse each camera (skip during warmup)
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
                with self._stats_lock:
                    self._cam_stale_count[i] += 1
                continue

            grace_only = age > self.max_camera_age

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

            bins_written = self._bin_polar_min(
                cam_ranges, cam_angles, fused, gap_fill_only=grace_only)
            self._cam_bins_written[i] += bins_written

            with self._stats_lock:
                self._cam_fused_count[i] += 1

        with self._stats_lock:
            if compensated_this_frame:
                self._motion_comp_applied += 1
            else:
                self._motion_comp_skipped += 1

        # ============================================================
        # V3: SPATIAL CONSISTENCY FILTER (replaces v2 temporal filter)
        # ============================================================
        # Reject camera gap-fill bins that have NO finite neighbor within
        # +/- spatial_neighbor_radius bins. Real obstacles have spatial
        # extent; isolated random noise doesn't.
        # This works within a single frame — no frame-to-frame dependency,
        # so it doesn't starve SLAM of data like the temporal filter did.
        if cameras_ready and self.enable_spatial_filter:
            n_bins = len(fused)
            radius = self.spatial_neighbor_radius

            # Camera-only gap-fill bins
            gap_filled = ~np.isfinite(lidar_snapshot) & np.isfinite(fused)
            n_gap = int(np.sum(gap_filled))

            if n_gap > 0:
                # Check if any neighbor within +/-radius is finite
                has_support = np.zeros(n_bins, dtype=np.bool_)
                for offset in range(-radius, radius + 1):
                    if offset == 0:
                        continue
                    # Shifted finite check
                    if offset > 0:
                        has_support[:-offset] |= np.isfinite(fused[offset:])
                    else:
                        has_support[-offset:] |= np.isfinite(fused[:offset])

                # Reject isolated gap-fills (no finite neighbor nearby)
                isolated = gap_filled & ~has_support
                n_rejected = int(np.sum(isolated))
                fused[isolated] = np.inf

                with self._stats_lock:
                    self._spatial_rejected += n_rejected

        # ============================================================
        # V3: GRADIENT FILTER (enhanced from v2)
        # ============================================================
        # Catches flying-pixel spikes AND small camera clusters (1-2 bins)
        # with large range discontinuities.
        if cameras_ready:
            fused, n_grad_rejected = self._gradient_filter_v3(
                fused, lidar_snapshot)
            with self._stats_lock:
                self._gradient_rejected += n_grad_rejected

        # ============================================================
        # V3: RANGE SMOOTHING (new)
        # ============================================================
        # For camera-contributed bins, EMA smooth with previous frame.
        # Reduces frame-to-frame jitter for SLAM consistency.
        # Only smooths bins that were camera-contributed in BOTH frames.
        if cameras_ready and self.enable_range_smoothing:
            cam_mask = ~np.isfinite(lidar_snapshot) & np.isfinite(fused)

            if self._prev_cam_ranges is not None:
                # Both current and previous are camera-contributed and finite
                both_cam = cam_mask & np.isfinite(self._prev_cam_ranges)
                if np.any(both_cam):
                    alpha = self.range_smooth_alpha
                    fused[both_cam] = (
                        alpha * fused[both_cam]
                        + (1.0 - alpha) * self._prev_cam_ranges[both_cam]
                    )

            # Store current camera ranges for next frame
            cam_ranges_snapshot = np.full(len(fused), np.inf, dtype=np.float32)
            cam_ranges_snapshot[cam_mask] = fused[cam_mask]
            self._prev_cam_ranges = cam_ranges_snapshot

        # Footprint filter (single pass after all fusion)
        if self.enable_footprint:
            fused = self.footprint_filter.filter_scan(
                fused, scan_msg.angle_min, scan_msg.angle_increment)

        # Publish
        self._publish_scan(fused, scan_msg, self.fused_pub)

        # Track latency
        elapsed_us = (time.monotonic_ns() - t0) // 1000
        with self._stats_lock:
            self._frame_count += 1
            self._total_latency_us += elapsed_us
            if elapsed_us > self._max_latency_us:
                self._max_latency_us = elapsed_us

    # ------------------------------------------------------------------
    # V3 Gradient filter — enhanced flying-pixel rejection
    # ------------------------------------------------------------------

    @staticmethod
    def _gradient_filter_v3(fused: np.ndarray, lidar: np.ndarray,
                            max_jump: float = 0.60) -> tuple:
        """
        Remove camera-contributed bins with range discontinuities.

        Enhanced from v2:
          - max_jump reduced from 0.80 to 0.60 (catches more artifacts)
          - Also rejects 2-bin camera clusters: if both bins in a pair are
            camera-only AND both have large jumps to their outer neighbors,
            the pair is rejected as a flying-pixel artifact.

        Returns (filtered_scan, n_rejected).
        """
        result = fused.copy()
        n = len(fused)
        if n < 3:
            return result, 0

        cam_mask = ~np.isfinite(lidar) & np.isfinite(fused)
        if not np.any(cam_mask):
            return result, 0

        # Compute left/right jumps (vectorized)
        left_jump = np.full(n, np.inf)
        right_jump = np.full(n, np.inf)

        left_valid = np.isfinite(fused[:-1])
        left_jump[1:] = np.where(left_valid, np.abs(fused[1:] - fused[:-1]), np.inf)

        right_valid = np.isfinite(fused[1:])
        right_jump[:-1] = np.where(right_valid, np.abs(fused[:-1] - fused[1:]), np.inf)

        # Single-bin spikes: camera bin with both neighbors jumping
        reject_single = cam_mask & (left_jump > max_jump) & (right_jump > max_jump)

        # 2-bin clusters: pair of adjacent camera bins where outer edges jump
        # [... big_jump | cam_bin_A | cam_bin_B | big_jump ...]
        reject_pair = np.zeros(n, dtype=np.bool_)
        if n >= 4:
            for i in range(1, n - 2):
                if (cam_mask[i] and cam_mask[i + 1]
                        and left_jump[i] > max_jump
                        and right_jump[i + 1] > max_jump):
                    reject_pair[i] = True
                    reject_pair[i + 1] = True

        reject = reject_single | reject_pair
        n_rejected = int(np.sum(reject))
        result[reject] = np.inf

        return result, n_rejected

    # ------------------------------------------------------------------
    # Vectorized binning — O(N log N) via sort + reduceat
    # ------------------------------------------------------------------

    def _bin_polar_min(self, ranges: np.ndarray, angles: np.ndarray,
                       scan_out: np.ndarray, gap_fill_only: bool = False):
        """
        Bin polar (range, angle) points into scan array.

        Camera data is only written when:
          1. Lidar has NO reading (inf) AND >= min_cam_points_per_bin camera
             points agree (gap fill), OR
          2. Camera range < lidar range - margin (genuinely closer obstacle)

        If gap_fill_only=True (grace period), only path 1 is used.
        """
        margin = self.camera_override_margin
        min_pts = self.min_cam_points_per_bin
        fill_gaps = self.camera_fill_gaps
        max_plausible = self.max_camera_range * 2.0

        indices = np.rint(
            (angles - self._angle_min) / self._angle_increment
        ).astype(np.int32)

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

        # Fast path for few points
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
                    if fill_gaps and cnt >= min_pts:
                        scan_out[b] = r
                        bins_written += 1
                elif (not gap_fill_only
                      and r < cur - margin
                      and cnt >= min_pts
                      and cur <= max_plausible):
                    scan_out[b] = r
                    bins_written += 1
            return bins_written

        # Sort by bin index
        order = np.argsort(idx, kind='quicksort')
        sorted_idx = idx[order]
        sorted_rng = rng[order]

        changes = np.empty(n, dtype=np.bool_)
        changes[0] = True
        np.not_equal(sorted_idx[1:], sorted_idx[:-1], out=changes[1:])
        starts = np.flatnonzero(changes)

        mins = np.minimum.reduceat(sorted_rng, starts)
        bins = sorted_idx[starts]

        counts = np.empty(len(starts), dtype=np.int32)
        counts[:-1] = starts[1:] - starts[:-1]
        counts[-1] = n - starts[-1]

        current = scan_out[bins]
        lidar_gap = ~np.isfinite(current)
        enough_pts = counts >= min_pts

        if gap_fill_only:
            if fill_gaps:
                write_mask = lidar_gap & enough_pts
            else:
                return 0
        else:
            genuinely_closer = (
                np.isfinite(current)
                & (mins < (current - margin))
                & (current <= max_plausible)
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
                    '[V3] 0 lidar frames received! Check /scan_filtered topic')
                return

            fused = list(self._cam_fused_count)
            stale = list(self._cam_stale_count)
            missing = list(self._cam_missing_count)
            avg_us = self._total_latency_us // max(fc, 1)
            max_us = self._max_latency_us
            mc_applied = self._motion_comp_applied
            mc_skipped = self._motion_comp_skipped
            spatial_rej = self._spatial_rejected
            grad_rej = self._gradient_rejected

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
            self._spatial_rejected = 0
            self._gradient_rejected = 0
            self._cam_total_pts = [0] * len(self.cameras)
            self._cam_depth_pass = [0] * len(self.cameras)
            self._cam_height_pass = [0] * len(self.cameras)
            self._cam_range_pass = [0] * len(self.cameras)
            self._cam_bins_written = [0] * len(self.cameras)

        parts = []
        for i, cam in enumerate(self.cameras):
            if not cam.enabled:
                parts.append(f'{cam.name}:OFF')
                continue
            total = fused[i] + stale[i] + missing[i]
            if total > 0:
                pct = 100 * fused[i] / total
                if fused[i] == 0:
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
                            f'{avg_t}->{avg_d}->{avg_h}->{avg_r}pts)')
                    else:
                        parts.append(f'{cam.name}:{pct:.0f}%({avg_bins}bins)')
            else:
                parts.append(f'{cam.name}:--')

        mc_total = mc_applied + mc_skipped
        mc_str = (f'mcomp: {mc_applied}/{mc_total}'
                  if mc_total > 0 else 'mcomp: --')

        self.get_logger().info(
            f'[V3] {fc} frames | '
            f'scan_cb: {avg_us} us avg / {max_us} us max | '
            f'{mc_str} | '
            f'spatial_rej: {spatial_rej} grad_rej: {grad_rej} | '
            f'{" | ".join(parts)}')

        if max_us > 10_000:
            self.get_logger().warn(
                f'Scan callback exceeded 10 ms! ({max_us} us) — '
                f'check CPU load or increase camera downsample')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    node = ScanFusionV3()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
