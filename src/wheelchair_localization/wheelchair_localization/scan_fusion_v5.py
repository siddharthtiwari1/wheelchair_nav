#!/usr/bin/env python3
"""
SCAN FUSION V5 — FLICKER-FREE OVERLAY FUSION FOR SLAM
=======================================================
Created: 2026-02-24

Fixes 3 critical v4 bugs:
  1. FLICKERING: v4 re-binned camera polar data every lidar frame (10Hz).
     Motion compensation shifted angles each frame → points hopped bins.
     FIX: Pre-bin camera data ONCE per camera callback (6Hz), store as
     persistent overlay array. Scan callback does vectorized array merge
     — no re-binning, deterministic, zero flicker.

  2. CAMERA OVERRIDE NOISE: v4 allowed camera to override lidar when
     camera_range < lidar_range - 0.15m. At 2-4m, RealSense noise is
     ±10-30cm, so random noise constantly overwrote good lidar readings.
     SLAM saw "moving walls" → scan matching confusion → lost localization.
     FIX: Gap-fill ONLY. Camera NEVER overrides lidar. Only writes where
     lidar is inf. For SLAM, lidar is always king.

  3. LONG-RANGE NOISE: v4 used max_camera_range=4.0m. Beyond 2m, RealSense
     depth noise degrades SLAM. FIX: Default 2.0m cap.

Architecture:
  Camera CB (6Hz per camera):
    pointcloud → downsample → depth filter → transform → height filter
    → polar → PRE-BIN into overlay array (once, stable)

  Scan CB (10Hz):
    copy lidar (NaN→inf) → merge overlay arrays (vectorized, no re-binning)
    → footprint filter → publish

  Result: Camera contributions are rock-solid between camera updates.
  No motion comp jitter. No override noise. Clean SLAM maps.

Paired with: slam_toolbox_fused_v7.yaml
"""

import array as _array
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
class CameraOverlay:
    """Pre-binned camera data for flicker-free scan merging.

    Created in camera callback (heavy work). Each camera maintains its own
    overlay. Scan callback merges all overlays with lidar — no re-binning.
    """
    min_ranges: np.ndarray   # float32[num_bins] — min camera range per bin (inf=no data)
    counts: np.ndarray       # int32[num_bins] — camera points per bin
    timestamp: float         # wall-clock time when overlay was computed
    point_count: int         # total valid points (diagnostics)


# ---------------------------------------------------------------------------
# Wheelchair footprint filter (proven correct, vectorized)
# ---------------------------------------------------------------------------

class WheelchairFootprintFilter:
    """URDF-calibrated self-detection filter."""

    def __init__(self):
        self.min_valid_range = 0.20

        self.robot_half_width = 0.45
        self.robot_rear = 1.0
        self.robot_front = 0.30

        self.exclusion_zones_deg = [
            ( 150,  180, 1.00),   # Rear-left: wheelchair back
            (-180, -140, 1.00),   # Rear-right: wheelchair back
            ( 120,  150, 0.50),   # Side-rear left
            (-140, -100, 0.65),   # Side-rear right
            (  90,  120, 0.35),   # Left side
            (-100,  -90, 0.35),   # Right side
            ( -35,  -23, 0.32),   # Right-front: armrest
            (  50,   60, 0.48),   # Left-front: armrest
            (  22,   32, 0.45),   # Forward-left: was 0.85 — too aggressive, killed real obstacles
        ]
        self.exclusion_zones_rad = [
            (np.radians(a1), np.radians(a2), r)
            for a1, a2, r in self.exclusion_zones_deg
        ]
        self._geom_cached = False

    def _cache_geometry(self, n: int, angle_min: float,
                        angle_increment: float):
        """Pre-compute angles, trig, and arc masks (constant per scan geometry)."""
        self._angles = angle_min + np.arange(n, dtype=np.float32) * angle_increment
        self._cos_a = np.cos(self._angles)
        self._sin_a = np.sin(self._angles)
        self._arc_masks = []
        for a_start, a_end, max_r in self.exclusion_zones_rad:
            if a_start <= a_end:
                mask = (self._angles >= a_start) & (self._angles <= a_end)
            else:
                mask = (self._angles >= a_start) | (self._angles <= a_end)
            self._arc_masks.append((mask, max_r))
        self._geom_cached = True

    def filter_scan(self, ranges: np.ndarray, angle_min: float,
                    angle_increment: float) -> np.ndarray:
        result = ranges.copy()
        valid = np.isfinite(ranges) & (ranges > 0)

        result[valid & (ranges < self.min_valid_range)] = np.inf

        if self._geom_cached:
            for arc_mask, max_r in self._arc_masks:
                result[valid & arc_mask & (ranges < max_r)] = np.inf
            cos_a = self._cos_a
            sin_a = self._sin_a
        else:
            n = len(ranges)
            angles = angle_min + np.arange(n, dtype=np.float32) * angle_increment
            for a_start, a_end, max_r in self.exclusion_zones_rad:
                if a_start <= a_end:
                    in_arc = (angles >= a_start) & (angles <= a_end)
                else:
                    in_arc = (angles >= a_start) | (angles <= a_end)
                result[valid & in_arc & (ranges < max_r)] = np.inf
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

class ScanFusionV5(Node):
    """
    Flicker-free 3-camera + LiDAR scan fusion for SLAM.

    Key design: pre-bin camera data in camera callback, merge arrays in
    scan callback. Camera gap-fills ONLY — never overrides lidar.
    """

    def __init__(self):
        super().__init__('scan_fusion_v5')

        # ==================================================================
        # Parameters
        # ==================================================================
        self.declare_parameter('scan_topic', '/scan_filtered')
        self.declare_parameter('output_topic', '/scan_fused')
        self.declare_parameter('laser_frame', 'laser')
        self.declare_parameter('min_height', 0.05)
        self.declare_parameter('max_height', 1.40)
        self.declare_parameter('max_camera_age_ms', 500.0)
        self.declare_parameter('enable_footprint_filter', True)
        self.declare_parameter('publish_lidar_only', True)
        self.declare_parameter('min_camera_points_per_bin', 3)
        self.declare_parameter('max_camera_range', 2.0)
        self.declare_parameter('camera_warmup_sec', 3.0)
        self.declare_parameter('rear_crop_deg', 180.0)
        self.declare_parameter('verbose', False)

        for prefix in ('front_camera', 'left_camera', 'right_camera'):
            self.declare_parameter(f'{prefix}.enabled', True)
            self.declare_parameter(f'{prefix}.topic', '')
            self.declare_parameter(f'{prefix}.frame', '')
            self.declare_parameter(f'{prefix}.min_depth', 0.40)
            self.declare_parameter(f'{prefix}.max_depth', 2.5)
            self.declare_parameter(f'{prefix}.downsample', 4)

        # Load params
        self.laser_frame = self.get_parameter('laser_frame').value
        self.min_height = float(self.get_parameter('min_height').value)
        self.max_height = float(self.get_parameter('max_height').value)
        self.max_camera_age = self.get_parameter('max_camera_age_ms').value / 1000.0
        self.enable_footprint = self.get_parameter('enable_footprint_filter').value
        self.publish_lidar_only = self.get_parameter('publish_lidar_only').value
        self.min_cam_pts = int(self.get_parameter('min_camera_points_per_bin').value)
        self.max_camera_range = float(self.get_parameter('max_camera_range').value)
        self.camera_warmup_sec = float(self.get_parameter('camera_warmup_sec').value)
        self.rear_crop_deg = float(self.get_parameter('rear_crop_deg').value)
        self.verbose = self.get_parameter('verbose').value

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
        self._static_tfs: Dict[str, tuple] = {}

        # ==================================================================
        # Per-camera overlays (thread-safe)
        # Overlay = pre-binned scan-sized arrays, computed in camera CB
        # ==================================================================
        self._overlay_lock = threading.Lock()
        self._cam_overlays: Dict[str, Optional[CameraOverlay]] = {
            cam.name: None for cam in self.cameras
        }

        # Pre-combined camera overlay (recomputed in camera CB)
        self._combined_min: Optional[np.ndarray] = None
        self._combined_counts: Optional[np.ndarray] = None
        self._combined_cam_statuses: List[str] = []

        # Pre-allocated scan buffer (initialized on first lidar message)
        self._fused_buf: Optional[np.ndarray] = None

        # ==================================================================
        # Scan parameters (set on first lidar message)
        # ==================================================================
        self._angle_min = 0.0
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
        # Statistics
        # ==================================================================
        self._stats_lock = threading.Lock()
        self._frame_count = 0
        self._total_latency_us = 0
        self._max_latency_us = 0
        self._cam_used_count = [0] * len(self.cameras)
        self._cam_stale_count = [0] * len(self.cameras)
        self._cam_missing_count = [0] * len(self.cameras)
        self._cam_bins_filled = [0] * len(self.cameras)
        self._total_gap_fills = 0

        # ==================================================================
        # QoS
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
        # Diagnostics
        # ==================================================================
        self.create_timer(10.0, self._print_stats)

        self._cam_tf_fail_count: Dict[str, int] = {}
        self._cam_first_active: Dict[str, bool] = {}
        self._cam_callback_count: Dict[str, int] = {}
        for cam in self.cameras:
            self._cam_tf_fail_count[cam.name] = 0
            self._cam_first_active[cam.name] = False
            self._cam_callback_count[cam.name] = 0

        self._log_startup()

    # ------------------------------------------------------------------
    # Startup log
    # ------------------------------------------------------------------

    def _log_startup(self):
        info = self.get_logger().info
        info('=' * 60)
        info('SCAN FUSION V5 — FLICKER-FREE OVERLAY (GAP-FILL ONLY)')
        info('=' * 60)
        info(f'  Input:       {self.get_parameter("scan_topic").value}')
        info(f'  Output:      {self.get_parameter("output_topic").value}')
        info(f'  Laser frame: {self.laser_frame}')
        info(f'  Height:      [{self.min_height:.2f}, {self.max_height:.2f}] m')
        info(f'  Max cam age: {self.max_camera_age * 1000:.0f} ms')
        info(f'  Max cam rng: {self.max_camera_range:.1f} m')
        info(f'  Min pts/bin: {self.min_cam_pts}')
        info(f'  Footprint:   {"ON" if self.enable_footprint else "OFF"}')
        info(f'  Warmup:      {self.camera_warmup_sec:.1f}s')
        info(f'  Rear crop:   ±{self.rear_crop_deg:.0f}° (rear bins → inf → cameras fill)')
        info(f'  Mode:        GAP-FILL ONLY (camera NEVER overrides lidar)')
        info(f'  Overlay:     Pre-binned in camera CB (no scan-time re-binning)')
        for cam in self.cameras:
            tag = 'ON' if cam.enabled else 'OFF'
            info(f'  {cam.name:6s}: {tag} | ds={cam.downsample} '
                 f'| depth=[{cam.min_depth:.1f}, {cam.max_depth:.1f}]m')
        info('=' * 60)

    # ------------------------------------------------------------------
    # Buffer initialization (called once after first scan)
    # ------------------------------------------------------------------

    def _init_buffers(self):
        """Pre-allocate reusable buffers and cache footprint geometry."""
        n = self._num_bins
        self._fused_buf = np.full(n, np.inf, dtype=np.float32)
        self._combined_min = np.full(n, np.inf, dtype=np.float32)
        self._combined_counts = np.zeros(n, dtype=np.int32)
        self._combined_cam_statuses = ['missing'] * len(self.cameras)
        self.footprint_filter._cache_geometry(
            n, self._angle_min, self._angle_increment)

        # Pre-compute rear crop mask — replaces Jazzy-crashing angular filter
        # Bins outside ±rear_crop_deg from front (0°) get set to inf
        if self.rear_crop_deg < 180.0:
            angles = (self._angle_min
                      + np.arange(n, dtype=np.float32) * self._angle_increment)
            limit_rad = np.radians(self.rear_crop_deg)
            self._rear_crop_mask = (angles > limit_rad) | (angles < -limit_rad)
            n_cropped = int(np.sum(self._rear_crop_mask))
            self.get_logger().info(
                f'Rear crop: ±{self.rear_crop_deg:.0f}° '
                f'({n_cropped}/{n} bins cropped, '
                f'{n - n_cropped} bins kept)')
        else:
            self._rear_crop_mask = None
            self.get_logger().info('Rear crop: DISABLED (180°)')

    # ------------------------------------------------------------------
    # Static TF cache
    # ------------------------------------------------------------------

    def _get_static_tf(self, target: str, source: str) -> Optional[tuple]:
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

        R = T[:3, :3].astype(np.float32)
        tv = T[:3, 3].astype(np.float32)

        self._static_tfs[key] = (R, tv)
        self.get_logger().info(f'Cached static TF: {source} -> {target}')
        return (R, tv)

    # ------------------------------------------------------------------
    # Camera callback — PRE-BINS into overlay (runs at 6 Hz per camera)
    # ------------------------------------------------------------------

    def _camera_callback(self, msg: PointCloud2, cam: CameraConfig):
        """Process point cloud and pre-bin into scan-sized overlay array.

        Heavy work done here (off lidar's critical path). The overlay is
        an immutable snapshot consumed by the scan callback — no re-binning
        needed at 10Hz, so camera bins are perfectly stable.
        """
        self._cam_callback_count[cam.name] = self._cam_callback_count.get(cam.name, 0) + 1

        # Need scan template before we can bin
        if not self._initialized:
            return

        tf_data = self._get_static_tf(self.laser_frame, cam.frame)
        if tf_data is None:
            cnt = self._cam_tf_fail_count.get(cam.name, 0) + 1
            self._cam_tf_fail_count[cam.name] = cnt
            if cnt <= 3 or cnt % 30 == 0:
                self.get_logger().warn(
                    f'[{cam.name}] TF {cam.frame}->{self.laser_frame} FAILED '
                    f'(#{cnt}) — DROPPED')
            return
        R, t_vec = tf_data

        if not self._cam_first_active.get(cam.name, False):
            self._cam_first_active[cam.name] = True
            self.get_logger().info(
                f'[{cam.name}] First valid TF — camera overlay ACTIVE')

        # -- Parse point cloud --
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
                f'[{cam.name}] parse error: {e}',
                throttle_duration_sec=5.0)
            return

        if len(xyz) < 10:
            return

        # -- Depth filter --
        depths = xyz[:, 2]
        xyz = xyz[(depths >= cam.min_depth) & (depths <= cam.max_depth)]
        if len(xyz) < 5:
            return

        # -- Transform to laser frame --
        pts_laser = xyz @ R.T + t_vec

        # -- Height filter --
        z_laser = pts_laser[:, 2]
        pts_laser = pts_laser[(z_laser >= self.min_height) & (z_laser <= self.max_height)]
        if len(pts_laser) < 3:
            return

        # -- Convert to polar --
        x_l = pts_laser[:, 0]
        y_l = pts_laser[:, 1]
        ranges = np.sqrt(x_l * x_l + y_l * y_l)
        angles = np.arctan2(y_l, x_l)

        # -- Range filter --
        valid = (ranges >= 0.25) & (ranges <= self.max_camera_range)
        ranges = ranges[valid]
        angles = angles[valid]

        if len(ranges) < 3:
            return

        # ============================================================
        # PRE-BIN into scan-sized overlay (THE KEY DIFFERENCE from v4)
        # This runs ONCE per camera frame (6Hz), NOT per lidar frame.
        # ============================================================
        overlay_min = np.full(self._num_bins, np.inf, dtype=np.float32)
        overlay_counts = np.zeros(self._num_bins, dtype=np.int32)

        indices = np.rint(
            (angles - self._angle_min) / self._angle_increment
        ).astype(np.int32)

        in_bounds = (indices >= 0) & (indices < self._num_bins)
        idx = indices[in_bounds]
        rng = ranges[in_bounds]

        if len(idx) > 0:
            # Per-bin minimum range and point count
            np.minimum.at(overlay_min, idx, rng)
            np.add.at(overlay_counts, idx, 1)

        overlay = CameraOverlay(
            min_ranges=overlay_min,
            counts=overlay_counts,
            timestamp=time.monotonic(),
            point_count=int(len(idx)),
        )

        with self._overlay_lock:
            self._cam_overlays[cam.name] = overlay

        self._recompute_combined_overlay()

    # ------------------------------------------------------------------
    # Pre-combine camera overlays (runs after each camera update)
    # ------------------------------------------------------------------

    def _recompute_combined_overlay(self):
        """Merge all non-stale camera overlays into one pre-combined array.

        Called from camera callback. Scan callback reads the result — no
        per-camera loop at scan time.
        """
        if not self._initialized:
            return

        now = time.monotonic()
        self._combined_min.fill(np.inf)
        self._combined_counts.fill(0)

        statuses = []
        for cam in self.cameras:
            if not cam.enabled:
                statuses.append('off')
                continue
            ov = self._cam_overlays.get(cam.name)
            if ov is None:
                statuses.append('missing')
                continue
            if (now - ov.timestamp) > self.max_camera_age:
                statuses.append('stale')
                continue
            statuses.append('used')
            better = ov.min_ranges < self._combined_min
            self._combined_min[better] = ov.min_ranges[better]
            self._combined_counts += ov.counts

        self._combined_cam_statuses = statuses

    # ------------------------------------------------------------------
    # Scan callback — FAST array merge (runs at 10 Hz lidar rate)
    # ------------------------------------------------------------------

    def _scan_callback(self, scan_msg: LaserScan):
        """Merge lidar with pre-binned camera overlays. Target: <1 ms."""
        t0 = time.monotonic_ns()
        now = time.monotonic()

        # Initialize scan template on first message
        if not self._initialized:
            if scan_msg.angle_increment <= 0 or len(scan_msg.ranges) < 10:
                self.get_logger().warn('Invalid first scan, skipping')
                return
            self._angle_min = scan_msg.angle_min
            self._angle_increment = scan_msg.angle_increment
            self._num_bins = len(scan_msg.ranges)
            self._range_min = scan_msg.range_min
            self._range_max = scan_msg.range_max
            self._initialized = True
            self._init_buffers()
            self.get_logger().info(
                f'Scan template: {self._num_bins} bins, '
                f'[{np.degrees(scan_msg.angle_min):.1f}, '
                f'{np.degrees(scan_msg.angle_max):.1f}] deg, '
                f'range [{self._range_min:.2f}, {self._range_max:.2f}] m')

        # Step 1: Copy lidar, NaN → inf
        fused = self._fused_buf
        fused[:] = scan_msg.ranges
        np.nan_to_num(fused, nan=np.inf, copy=False)

        # Step 2: Footprint filter — remove wheelchair wheel/body detections
        # Lidar at footrest height sees wheels at specific angles.
        # Must run BEFORE camera merge so wheel bins become inf → cameras can fill them.
        if self.enable_footprint:
            fused = self.footprint_filter.filter_scan(
                fused, scan_msg.angle_min, scan_msg.angle_increment)

        # Step 3: Rear crop — set bins behind robot to inf
        # Replaces angular_bounds filter which crashes on Jazzy (SIGSEGV)
        # Creates inf bins that cameras (especially rear-facing front D455) can fill
        if self._rear_crop_mask is not None:
            fused[self._rear_crop_mask] = np.inf

        # Publish lidar-only debug topic (after footprint + crop, before camera)
        if self.lidar_pub is not None:
            self._publish_scan(fused.copy(), scan_msg, self.lidar_pub)

        # Step 4: Camera gap-fill — fill inf bins with camera data
        # NO filter on camera data — cameras are high-mounted, cannot see wheelchair.
        if not hasattr(self, '_first_scan_time'):
            self._first_scan_time = now
        cameras_ready = (now - self._first_scan_time) >= self.camera_warmup_sec
        n_filled = 0

        if cameras_ready:
            combined_min = self._combined_min
            combined_counts = self._combined_counts

            # Update per-camera stats
            statuses = self._combined_cam_statuses
            with self._stats_lock:
                for i, status in enumerate(statuses):
                    if status == 'used':
                        self._cam_used_count[i] += 1
                    elif status == 'stale':
                        self._cam_stale_count[i] += 1
                    elif status == 'missing':
                        self._cam_missing_count[i] += 1

            # GAP-FILL ONLY: camera writes ONLY where lidar is inf
            # (from footprint filter, rear crop, or natural lidar gaps)
            lidar_gap = np.isinf(fused)
            cam_valid = np.isfinite(combined_min) & (combined_counts >= self.min_cam_pts)
            fill_mask = lidar_gap & cam_valid

            n_filled = int(np.sum(fill_mask))
            if n_filled > 0:
                fused[fill_mask] = combined_min[fill_mask]

            with self._stats_lock:
                self._total_gap_fills += n_filled

        # Step 5: Publish — footprint already applied, camera fills are clean
        self._publish_scan(fused, scan_msg, self.fused_pub)

        # Track latency
        elapsed_us = (time.monotonic_ns() - t0) // 1000
        with self._stats_lock:
            self._frame_count += 1
            self._total_latency_us += elapsed_us
            if elapsed_us > self._max_latency_us:
                self._max_latency_us = elapsed_us

    # ------------------------------------------------------------------
    # Publish helper
    # ------------------------------------------------------------------

    def _publish_scan(self, ranges: np.ndarray, template: LaserScan,
                      publisher):
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
        publisher.publish(msg)

    # ------------------------------------------------------------------
    # Diagnostics (every 10s)
    # ------------------------------------------------------------------

    def _print_stats(self):
        with self._stats_lock:
            fc = self._frame_count
            if fc == 0:
                self.get_logger().warn(
                    '[V5] 0 lidar frames! Check /scan_filtered topic')
                return

            used = list(self._cam_used_count)
            stale = list(self._cam_stale_count)
            missing = list(self._cam_missing_count)
            avg_us = self._total_latency_us // max(fc, 1)
            max_us = self._max_latency_us
            gap_fills = self._total_gap_fills

            # Reset
            self._frame_count = 0
            self._cam_used_count = [0] * len(self.cameras)
            self._cam_stale_count = [0] * len(self.cameras)
            self._cam_missing_count = [0] * len(self.cameras)
            self._total_latency_us = 0
            self._max_latency_us = 0
            self._total_gap_fills = 0

        parts = []
        for i, cam in enumerate(self.cameras):
            if not cam.enabled:
                parts.append(f'{cam.name}:OFF')
                continue
            total = used[i] + stale[i] + missing[i]
            if total > 0:
                pct = 100 * used[i] / total
                cb = self._cam_callback_count.get(cam.name, 0)
                parts.append(f'{cam.name}:{pct:.0f}%(cb={cb})')
            else:
                parts.append(f'{cam.name}:--')

        avg_fills = gap_fills // max(fc, 1)

        self.get_logger().info(
            f'[V5] {fc} frames | '
            f'{avg_us}us avg / {max_us}us max | '
            f'gap-fills: {avg_fills}/frame ({gap_fills} total) | '
            f'{" | ".join(parts)}')

        if max_us > 5_000:
            self.get_logger().warn(
                f'Scan callback exceeded 5ms! ({max_us}us)')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    node = ScanFusionV5()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
