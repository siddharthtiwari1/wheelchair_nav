#!/usr/bin/env python3
"""
SCAN FUSION V7 — MIN FUSION WITH POINTCLOUD2 (BASED ON ORIGINAL)
=================================================================
Created: 2026-02-24

Based on the ORIGINAL scan_depth_fusion_node.py (OG) that WORKED,
but fixes the freezing bug by using v6's two-callback architecture.

OG problem: ApproximateTimeSynchronizer on 4 topics -> unbounded
memory when one sensor lags -> system freeze after minutes.

V7 fix: Separate async callbacks for lidar and each camera.
Camera callbacks store processed polar data. Scan callback
merges them using MIN fusion (closest obstacle wins).

Key differences from v6:
  - PointCloud2 input (like OG) instead of depth images
  - MIN fusion (like OG) instead of gap-fill only
  - Catches elevated obstacles (tables, shelves) in lidar's FOV

Key differences from OG:
  - NO ApproximateTimeSynchronizer (prevents freezing)
  - Footprint filter (from v6) removes wheelchair self-detections
  - Rear crop (from v6) handles rear blind spot
  - Staleness check on camera data
  - Stride-based downsampling BEFORE transform (from v2) for CPU

Architecture:
  Camera CB (6Hz per camera, async):
    PointCloud2 -> parse -> downsample -> depth filter -> transform
    -> height filter -> polar -> range filter -> store overlay

  Scan CB (10Hz):
    copy /scan_filtered -> NaN->inf -> footprint filter -> rear crop
    -> MIN merge from camera overlays -> publish /scan_fused

Paired with: slam_toolbox_fused_v8.yaml (use_fused_slam:=true)
             or slam_toolbox_motion_compensated_v2.yaml (lidar-only)
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
    """URDF-calibrated self-detection filter (same as v6)."""

    def __init__(self):
        self.min_valid_range = 0.20
        self.robot_half_width = 0.33   # Was 0.45 — caught legitimate walls
        self.robot_rear = 0.50         # Was 1.0 — rear crop handles far-back
        self.robot_front = 0.20        # Was 0.30 — tighter

        # (start_deg, end_deg, max_range_m)
        self.exclusion_zones_deg = [
            ( 150,  180, 1.00),   # Rear-left: wheelchair back
            (-180, -140, 1.00),   # Rear-right: wheelchair back
            ( 120,  150, 0.50),   # Side-rear left
            (-140, -100, 0.65),   # Side-rear right
            (  90,  120, 0.35),   # Left side
            (-100,  -90, 0.35),   # Right side
            ( -35,  -23, 0.32),   # Right-front: armrest
            (  50,   60, 0.48),   # Left-front: armrest
            (  22,   32, 0.45),   # Forward-left: footrest
        ]
        self.exclusion_zones_rad = [
            (np.radians(a1), np.radians(a2), r)
            for a1, a2, r in self.exclusion_zones_deg
        ]
        self._cached = False

    def cache_geometry(self, n: int, robot_angles: np.ndarray):
        """Pre-compute arc masks + trig from ROBOT-frame angles.

        robot_angles: array where 0°=forward, +90°=left, -90°=right, ±180°=rear.
        The 180° Z rotation in the laser frame is already accounted for by the caller.
        """
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
        """Filter in-place: set wheelchair self-detections to inf."""
        valid = np.isfinite(ranges) & (ranges > 0)
        ranges[valid & (ranges < self.min_valid_range)] = np.inf

        for arc_mask, max_r in self._arc_masks:
            ranges[valid & arc_mask & (ranges < max_r)] = np.inf

        # Rectangular footprint box
        x = np.where(valid, ranges * self._cos_a, 0.0)
        y = np.where(valid, ranges * self._sin_a, 0.0)
        in_box = (
            valid
            & (x >= -self.robot_rear) & (x <= self.robot_front)
            & (y >= -self.robot_half_width) & (y <= self.robot_half_width)
        )
        ranges[in_box] = np.inf
        return ranges


class ScanFusionV7(Node):
    """MIN fusion: lidar + 3 PointCloud2 cameras. OG logic + v6 architecture."""

    def __init__(self):
        super().__init__('scan_fusion')

        # ---- Parameters ----
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
        # MIN override delta: camera must be at least this much closer than
        # lidar to override. Simple fixed threshold — lidar is trusted for
        # small differences (noise), camera wins only for large differences
        # (real obstacle lidar missed due to height/geometry).
        # lidar=3.5 cam=1.5 → diff=2.0 > 0.4 → camera (real table)
        # lidar=1.7 cam=1.5 → diff=0.2 < 0.4 → lidar (noise)
        self.declare_parameter('override_min_delta', 0.4)
        # Max gap-fill bins per camera per scan. Prevents abrupt scan shape
        # changes when a camera suddenly sees a new object during turns.
        # Closest bins are filled first; rest appear over subsequent scans.
        self.declare_parameter('max_gap_fills_per_camera', 20)
        self.declare_parameter('max_overrides_per_camera', 80)
        self.declare_parameter('min_camera_points_per_bin', 3)

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
        self.override_min_delta = float(self.get_parameter('override_min_delta').value)
        self.max_gap_fills = int(self.get_parameter('max_gap_fills_per_camera').value)
        self.max_overrides = int(self.get_parameter('max_overrides_per_camera').value)
        self.min_points_per_bin = int(self.get_parameter('min_camera_points_per_bin').value)

        # ---- Footprint filter ----
        self.footprint_filter = WheelchairFootprintFilter() if self.enable_footprint else None

        # ---- Callback groups (for MultiThreadedExecutor) ----
        # Cameras: reentrant → all 3 process in parallel
        # Scan: own group → never blocked by camera processing
        self._cam_cb_group = ReentrantCallbackGroup()
        self._scan_cb_group = MutuallyExclusiveCallbackGroup()

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

        # Stats
        self._frame_count = 0
        self._total_fills = 0
        self._total_overrides = 0
        self._total_latency_us = 0
        self._max_latency_us = 0

        # ---- Publishers ----
        self.fused_pub = self.create_publisher(
            LaserScan, self.get_parameter('output_topic').value, 10)
        self.lidar_pub = self.create_publisher(LaserScan, '/scan_lidar_only', 10)

        # Per-camera debug scan publishers (visualize each camera's contribution)
        self._cam_pubs: Dict[str, any] = {}

        # ---- Subscribe to lidar (own callback group — never blocked by cameras) ----
        self.create_subscription(
            LaserScan, self.get_parameter('scan_topic').value,
            self._scan_cb, SENSOR_QOS,
            callback_group=self._scan_cb_group)

        # ---- Subscribe to cameras (async — no time sync = no freezing) ----
        self._cam_names = []
        for name, _, _ in cameras_config:
            enabled = self.get_parameter(f'{name}.enabled').value
            if not enabled:
                continue
            topic = self.get_parameter(f'{name}.topic').value
            frame = self.get_parameter(f'{name}.frame').value

            self._cam_names.append(name)
            self._overlays[name] = None

            # Per-camera debug scan topic: /scan_front_camera, /scan_left_camera, /scan_right_camera
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
        info('SCAN FUSION V7 — MIN Fusion + PointCloud2 (OG-based)')
        info('=' * 60)
        info(f'  Cameras: {", ".join(self._cam_names)}')
        info(f'  Fusion: MIN (closest obstacle wins)')
        info(f'  Input: PointCloud2 (like original node)')
        info(f'  Cam range: {self.min_cam_range:.1f}-{self.max_cam_range:.1f} m')
        info(f'  Height: {self.min_height:.2f}-{self.max_height:.2f} m')
        info(f'  Downsample: stride {self.downsample}')
        info(f'  Override delta: {self.override_min_delta:.2f}m (cam must be this much closer than lidar)')
        info(f'  Max overrides/camera: {self.max_overrides}')
        info(f'  Footprint filter: {self.enable_footprint}')
        info(f'  Rear crop: +/-{self.rear_crop_deg:.0f} deg')

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
        min_depth = getattr(self, f'_{cam_name}_min_depth')
        max_depth = getattr(self, f'_{cam_name}_max_depth')

        # Parse PointCloud2 (same as OG node)
        pts = point_cloud2.read_points_numpy(
            msg, field_names=('x', 'y', 'z'), skip_nans=True)

        if len(pts) == 0:
            return

        # Handle structured vs unstructured arrays (RealSense driver varies)
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

        # Downsample FIRST (before transform — saves CPU, from v2 design)
        if self.downsample > 1:
            x = x[::self.downsample]
            y = y[::self.downsample]
            z = z[::self.downsample]

        # Depth filter in camera optical frame (Z = depth in optical frame)
        depth_mask = (z >= min_depth) & (z <= max_depth)
        if not np.any(depth_mask):
            return
        x, y, z = x[depth_mask], y[depth_mask], z[depth_mask]

        # Transform to laser frame: p_laser = p_cam @ R^T + t
        pts_cam = np.column_stack([x, y, z]).astype(np.float32)
        pts_laser = pts_cam @ R.T + t_vec

        # Height filter in laser frame (Z = height above scan plane)
        z_laser = pts_laser[:, 2]
        height_ok = (z_laser >= self.min_height) & (z_laser <= self.max_height)
        pts_laser = pts_laser[height_ok]
        if len(pts_laser) == 0:
            return

        # Project to 2D polar (in laser frame X-Y plane)
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

        # Bin into overlay — min range per bin, with point count consensus
        indices = np.rint(
            (angles - self._angle_min) / self._angle_increment
        ).astype(np.int32)
        in_bounds = (indices >= 0) & (indices < self._num_bins)
        idx = indices[in_bounds]
        rng = ranges[in_bounds]

        overlay_ranges = np.full(self._num_bins, np.inf, dtype=np.float32)
        if len(idx) > 0:
            np.minimum.at(overlay_ranges, idx, rng)
            # Reject bins with too few points — single-point outliers become inf
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
    # Scan callback — lidar + footprint + rear crop + MIN camera merge
    # ------------------------------------------------------------------
    def _scan_cb(self, scan_msg: LaserScan):
        # Dedup: MultiThreadedExecutor can deliver the same message twice.
        # Skip if we already processed this exact timestamp.
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

            # Scan angles in laser frame (0° = robot REAR due to 180° Z rotation)
            scan_angles = self._angle_min + np.arange(
                self._num_bins, dtype=np.float32) * self._angle_increment

            # Convert to ROBOT-frame angles (0°=forward, +90°=left, ±180°=rear)
            # The laser frame has 180° Z rotation from base_laser, so:
            #   scan 0° = robot rear, scan ±π = robot forward
            #   scan +90° = robot right, scan -90° = robot left
            # atan2(-sin(θ), -cos(θ)) rotates by 180° AND fixes Y-axis flip
            robot_angles = np.arctan2(
                -np.sin(scan_angles), -np.cos(scan_angles)
            ).astype(np.float32)

            # Rear crop in ROBOT frame: crop bins outside ±rear_crop_deg
            if self.rear_crop_deg < 180.0:
                limit_rad = np.radians(self.rear_crop_deg)
                self._rear_crop_mask = (robot_angles > limit_rad) | (robot_angles < -limit_rad)
                self.get_logger().info(
                    f'Rear crop: +/-{self.rear_crop_deg:.0f} deg '
                    f'({int(np.sum(self._rear_crop_mask))} bins cropped) '
                    f'[robot frame: 0°=forward, ±180°=rear]')

            # Footprint filter uses ROBOT-frame angles
            if self.footprint_filter is not None:
                self.footprint_filter.cache_geometry(self._num_bins, robot_angles)

            self._initialized = True
            self.get_logger().info(
                f'Scan: {self._num_bins} bins, '
                f'scan [{np.degrees(scan_angles[0]):.1f}, '
                f'{np.degrees(scan_angles[-1]):.1f}] deg (laser frame), '
                f'robot [{np.degrees(robot_angles[0]):.1f}, '
                f'{np.degrees(robot_angles[-1]):.1f}] deg (robot frame)')

        # Step 1: Copy lidar, clean invalid ranges
        # RPLidar reports NaN AND zero for invalid readings — both must become inf.
        # Zero-range bins confuse SLAM's correlative matcher (point at sensor origin).
        fused = np.array(scan_msg.ranges, dtype=np.float32)
        np.nan_to_num(fused, nan=np.inf, copy=False)
        fused[fused <= 0.0] = np.inf

        # Step 2: Footprint filter — remove wheelchair self-detections
        if self.footprint_filter is not None:
            self.footprint_filter.filter_scan(fused)

        # Step 3: Rear crop — set rear bins to inf
        if self._rear_crop_mask is not None:
            fused[self._rear_crop_mask] = np.inf

        # Publish lidar-only debug (after footprint + crop, before cameras)
        self._publish(fused.copy(), scan_msg, self.lidar_pub)

        # Step 4: Camera MIN fusion
        now = time.monotonic()
        if (now - self._start_time) < self.camera_warmup:
            self._publish(fused, scan_msg, self.fused_pub)
            return

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
                # Rate-limited: only fill closest N bins per camera to prevent
                # abrupt scan shape changes during turns (e.g., camera sweeps
                # onto conference table → 60 bins appear at once → SLAM breaks).
                lidar_gap = np.isinf(fused)
                fill = lidar_gap & cam_valid
                n_fill = int(np.sum(fill))
                if n_fill > 0:
                    if n_fill > self.max_gap_fills:
                        # Apply closest bins first (most safety-critical)
                        fill_idx = np.where(fill)[0]
                        fill_ranges = ov.min_ranges[fill_idx]
                        closest = np.argsort(fill_ranges)[:self.max_gap_fills]
                        fill[:] = False
                        fill[fill_idx[closest]] = True
                        n_fill = self.max_gap_fills
                    fused[fill] = ov.min_ranges[fill]
                    n_filled += n_fill

                # Case 2: MIN override — camera sees SIGNIFICANTLY closer obstacle
                # diff > 0.4m: real obstacle (table, shelf) → take camera
                # diff ≤ 0.4m: noise → trust lidar
                lidar_has_data = np.isfinite(fused) & ~lidar_gap
                closer = (lidar_has_data & cam_valid
                          & (ov.min_ranges < fused - self.override_min_delta))
                n_override = int(np.sum(closer))
                if n_override > 0:
                    if n_override > self.max_overrides:
                        # Cap overrides: keep closest bins (most safety-critical)
                        over_idx = np.where(closer)[0]
                        over_ranges = ov.min_ranges[over_idx]
                        closest = np.argsort(over_ranges)[:self.max_overrides]
                        closer[:] = False
                        closer[over_idx[closest]] = True
                        n_override = self.max_overrides
                    fused[closer] = ov.min_ranges[closer]
                    n_overrides += n_override

        self._total_fills += n_filled
        self._total_overrides += n_overrides
        self._frame_count += 1

        # Final cleanup: zero/negative/NaN ranges → inf BEFORE publishing.
        # Some zeros survive from lidar or get introduced during fusion.
        # SLAM Toolbox treats 0-range as "point at sensor origin" → matcher crash.
        bad = ~np.isfinite(fused) | (fused <= 0.0)
        fused[bad] = np.inf

        # Publish fused
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
            f'[V7] {fc} frames | {avg_us}us avg / {self._max_latency_us}us max | '
            f'gap-fills: {avg_fills}/frame | overrides: {avg_overrides}/frame | '
            f'{" | ".join(cam_status)}')

        self._frame_count = 0
        self._total_fills = 0
        self._total_overrides = 0
        self._total_latency_us = 0
        self._max_latency_us = 0


def main(args=None):
    rclpy.init(args=args)
    node = ScanFusionV7()
    # 4 threads: 1 scan CB + 3 camera CBs running in parallel
    # This prevents PointCloud2 parsing from starving the scan callback
    executor = MultiThreadedExecutor(num_threads=4)
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
