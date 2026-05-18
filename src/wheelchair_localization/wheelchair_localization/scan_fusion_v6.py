#!/usr/bin/env python3
"""
SCAN FUSION V6 — SIMPLE GAP-FILL FUSION (PROVEN APPROACH)
==========================================================
Created: 2026-02-24

Replaces v5's over-engineered pipeline that starved cameras to 0 fills.

Root cause of v5 failure:
  min_camera_points_per_bin=3 + stride=4 downsampling = 0.99 pts/bin avg
  → 99% of bins fail the threshold → cameras contribute nothing.

V6 design (based on what ACTUALLY WORKED in scan_depth_fusion_node.py):
  1. Subscribe to /scan_filtered (lidar, already range+speckle filtered)
  2. Subscribe to 3x depth images (NOT PointCloud2 — lighter)
  3. Use depthimage_to_laserscan logic inline: project depth rows to 2D
  4. NO min_points_per_bin — single point can fill a bin (like the original)
  5. NO stride downsampling on depth — use scan_height rows (like depthimage_to_laserscan)
  6. Footprint filter on lidar FIRST, then rear crop, then camera gap-fill LAST

Architecture:
  Camera CB (6Hz per camera):
    depth_image → select scan_height rows around center → project to polar
    → store as overlay array (min range per bin)

  Scan CB (10Hz):
    copy /scan_filtered → footprint filter → rear crop → gap-fill from overlays → publish

Paired with: slam_toolbox_motion_compensated_v2.yaml (lidar-only SLAM)
             Use /scan_fused for Nav2 costmap + AMCL only.
"""

import array as _array
import numpy as np
from typing import Optional, Dict
import time
import threading

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from rclpy.time import Time

from sensor_msgs.msg import LaserScan, Image, CameraInfo
from tf2_ros import Buffer, TransformListener

SENSOR_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1
)


# ---------------------------------------------------------------------------
# Wheelchair footprint filter — removes self-detections from lidar
# ---------------------------------------------------------------------------
class WheelchairFootprintFilter:
    """URDF-calibrated self-detection filter for wheelchair body/wheels."""

    def __init__(self):
        self.min_valid_range = 0.20

        self.robot_half_width = 0.33   # Was 0.45 — caught legitimate walls
        self.robot_rear = 0.50         # Was 1.0 — rear crop handles far-back
        self.robot_front = 0.20        # Was 0.30 — tighter

        # (start_deg, end_deg, max_range_m) — lidar sees wheelchair parts here
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

    def cache_geometry(self, n: int, angle_min: float, angle_increment: float):
        """Pre-compute arc masks + trig (called once on first scan)."""
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


class CameraOverlay:
    """Per-camera scan overlay — min range per bin."""
    __slots__ = ('min_ranges', 'timestamp', 'point_count')

    def __init__(self, num_bins: int):
        self.min_ranges = np.full(num_bins, np.inf, dtype=np.float32)
        self.timestamp = 0.0
        self.point_count = 0


class ScanFusionV6(Node):
    """Simple gap-fill fusion: lidar + 3 depth cameras."""

    def __init__(self):
        super().__init__('scan_fusion')

        # ---- Parameters ----
        self.declare_parameter('scan_topic', '/scan_filtered')
        self.declare_parameter('output_topic', '/scan_fused')
        self.declare_parameter('laser_frame', 'laser')

        # Camera range limits
        self.declare_parameter('max_camera_range', 3.5)  # Was 2.0 in v5 — too tight
        self.declare_parameter('min_camera_range', 0.30)

        # Height filter (in laser frame Z)
        self.declare_parameter('min_height', 0.05)
        self.declare_parameter('max_height', 1.80)  # Was 1.40 in v5 — too tight

        # Depth image scan_height — rows to aggregate (like depthimage_to_laserscan)
        self.declare_parameter('scan_height', 60)  # 60 rows around center = robust

        # Camera staleness
        self.declare_parameter('max_camera_age_ms', 500.0)
        self.declare_parameter('camera_warmup_sec', 5.0)

        # Footprint filter
        self.declare_parameter('enable_footprint', True)

        # Rear crop
        self.declare_parameter('rear_crop_deg', 135.0)

        # Camera configs
        cameras_config = [
            ('front_camera', '/camera/depth/image_rect_raw',
             '/camera/depth/camera_info', 'camera_depth_optical_frame'),
            ('left_camera', '/mapping_camera/depth/image_rect_raw',
             '/mapping_camera/depth/camera_info', 'mapping_camera_depth_optical_frame'),
            ('right_camera', '/right_camera/depth/image_rect_raw',
             '/right_camera/depth/camera_info', 'right_camera_depth_optical_frame'),
        ]

        for name, depth_topic, info_topic, frame in cameras_config:
            self.declare_parameter(f'{name}.enabled', True)
            self.declare_parameter(f'{name}.depth_topic', depth_topic)
            self.declare_parameter(f'{name}.info_topic', info_topic)
            self.declare_parameter(f'{name}.frame', frame)
            self.declare_parameter(f'{name}.max_depth', 3.5)
            self.declare_parameter(f'{name}.min_depth', 0.30)
            self.declare_parameter(f'{name}.scan_height', -1)           # -1 = use global
            self.declare_parameter(f'{name}.scan_center_offset', 0)     # rows below image center

        # ---- Read parameters ----
        self.laser_frame = self.get_parameter('laser_frame').value
        self.max_cam_range = float(self.get_parameter('max_camera_range').value)
        self.min_cam_range = float(self.get_parameter('min_camera_range').value)
        self.min_height = float(self.get_parameter('min_height').value)
        self.max_height = float(self.get_parameter('max_height').value)
        self.scan_height = int(self.get_parameter('scan_height').value)
        self.max_camera_age = float(self.get_parameter('max_camera_age_ms').value) / 1000.0
        self.camera_warmup = float(self.get_parameter('camera_warmup_sec').value)
        self.rear_crop_deg = float(self.get_parameter('rear_crop_deg').value)
        self.enable_footprint = bool(self.get_parameter('enable_footprint').value)

        # ---- Footprint filter ----
        self.footprint_filter = WheelchairFootprintFilter() if self.enable_footprint else None

        # ---- TF ----
        self.tf_buffer = Buffer(cache_time=rclpy.duration.Duration(seconds=10))
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ---- State ----
        self._lock = threading.Lock()
        self._overlays: Dict[str, CameraOverlay] = {}
        self._cam_infos: Dict[str, CameraInfo] = {}
        self._tf_cache: Dict[str, Optional[tuple]] = {}
        self._scan_template: Optional[LaserScan] = None
        self._num_bins = 0
        self._angle_min = 0.0
        self._angle_increment = 0.0
        self._rear_crop_mask = None
        self._initialized = False
        self._start_time = time.monotonic()

        # Stats
        self._frame_count = 0
        self._total_fills = 0
        self._total_latency_us = 0
        self._max_latency_us = 0

        # ---- Publishers ----
        self.fused_pub = self.create_publisher(LaserScan, self.get_parameter('output_topic').value, 10)
        self.lidar_pub = self.create_publisher(LaserScan, '/scan_lidar_only', 10)

        # ---- Subscribe to lidar ----
        self.create_subscription(
            LaserScan, self.get_parameter('scan_topic').value,
            self._scan_cb, SENSOR_QOS)

        # ---- Subscribe to cameras ----
        self._cam_names = []
        for name, _, _, _ in cameras_config:
            enabled = self.get_parameter(f'{name}.enabled').value
            if not enabled:
                continue
            depth_topic = self.get_parameter(f'{name}.depth_topic').value
            info_topic = self.get_parameter(f'{name}.info_topic').value
            frame = self.get_parameter(f'{name}.frame').value

            self._cam_names.append(name)
            self._overlays[name] = None  # Will be created after first scan

            # Store camera config
            setattr(self, f'_{name}_frame', frame)
            setattr(self, f'_{name}_min_depth',
                    float(self.get_parameter(f'{name}.min_depth').value))
            setattr(self, f'_{name}_max_depth',
                    float(self.get_parameter(f'{name}.max_depth').value))
            cam_sh = int(self.get_parameter(f'{name}.scan_height').value)
            setattr(self, f'_{name}_scan_height', cam_sh if cam_sh > 0 else self.scan_height)
            setattr(self, f'_{name}_scan_center_offset',
                    int(self.get_parameter(f'{name}.scan_center_offset').value))

            # Subscribe to depth image + camera info
            self.create_subscription(
                Image, depth_topic,
                lambda msg, n=name: self._depth_cb(msg, n), SENSOR_QOS)
            self.create_subscription(
                CameraInfo, info_topic,
                lambda msg, n=name: self._info_cb(msg, n), SENSOR_QOS)

        # ---- Diagnostics timer ----
        self.create_timer(10.0, self._print_stats)

        info = self.get_logger().info
        info('=' * 60)
        info('SCAN FUSION V6 — Simple Gap-Fill (no point starvation)')
        info('=' * 60)
        info(f'  Cameras: {", ".join(self._cam_names)}')
        info(f'  Cam range: {self.min_cam_range:.1f}-{self.max_cam_range:.1f} m')
        info(f'  Height: {self.min_height:.2f}-{self.max_height:.2f} m')
        info(f'  Scan height: {self.scan_height} rows')
        info(f'  Footprint filter: {self.enable_footprint}')
        info(f'  Rear crop: ±{self.rear_crop_deg:.0f}°')
        info(f'  Max age: {self.max_camera_age*1000:.0f} ms')

    # ------------------------------------------------------------------
    # Camera info callback
    # ------------------------------------------------------------------
    def _info_cb(self, msg: CameraInfo, cam_name: str):
        if cam_name not in self._cam_infos:
            self._cam_infos[cam_name] = msg
            self.get_logger().info(
                f'[{cam_name}] CameraInfo: {msg.width}x{msg.height}, '
                f'fx={msg.k[0]:.1f}, fy={msg.k[4]:.1f}, '
                f'cx={msg.k[2]:.1f}, cy={msg.k[5]:.1f}')

    # ------------------------------------------------------------------
    # Get cached TF (camera optical frame → laser frame)
    # ------------------------------------------------------------------
    def _get_tf(self, cam_frame: str):
        if cam_frame in self._tf_cache:
            return self._tf_cache[cam_frame]
        try:
            tf = self.tf_buffer.lookup_transform(
                self.laser_frame, cam_frame, Time())
            t = tf.transform.translation
            q = tf.transform.rotation
            # Quaternion to rotation matrix
            x, y, z, w = q.x, q.y, q.z, q.w
            R = np.array([
                [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
                [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
                [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]
            ], dtype=np.float32)
            t_vec = np.array([t.x, t.y, t.z], dtype=np.float32)
            self._tf_cache[cam_frame] = (R, t_vec)
            self.get_logger().info(
                f'TF {cam_frame} → {self.laser_frame}: '
                f't=[{t.x:.3f}, {t.y:.3f}, {t.z:.3f}]')
            return (R, t_vec)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Depth image callback — project to 2D scan overlay
    # ------------------------------------------------------------------
    def _depth_cb(self, msg: Image, cam_name: str):
        if not self._initialized:
            return

        cam_info = self._cam_infos.get(cam_name)
        if cam_info is None:
            return

        cam_frame = getattr(self, f'_{cam_name}_frame')
        tf_data = self._get_tf(cam_frame)
        if tf_data is None:
            return

        R, t_vec = tf_data
        min_depth = getattr(self, f'_{cam_name}_min_depth')
        max_depth = getattr(self, f'_{cam_name}_max_depth')

        # Parse depth image (16UC1 or 32FC1)
        h, w = msg.height, msg.width
        if msg.encoding == '16UC1':
            depth = np.frombuffer(msg.data, dtype=np.uint16).reshape(h, w).astype(np.float32) * 0.001
        elif msg.encoding == '32FC1':
            depth = np.frombuffer(msg.data, dtype=np.float32).reshape(h, w)
        else:
            return

        # Select scan rows — per-camera offset for high-mounted cameras
        cam_scan_height = getattr(self, f'_{cam_name}_scan_height')
        cam_center_offset = getattr(self, f'_{cam_name}_scan_center_offset')
        center = h // 2 + cam_center_offset
        half = cam_scan_height // 2
        row_start = max(0, center - half)
        row_end = min(h, center + half)
        depth_slice = depth[row_start:row_end, :]  # (cam_scan_height, W)

        # Take minimum depth per column (closest obstacle in height band)
        # This is exactly what depthimage_to_laserscan does
        with np.errstate(invalid='ignore'):
            depth_slice = np.where(
                (depth_slice >= min_depth) & (depth_slice <= max_depth),
                depth_slice, np.inf)
        col_depths = np.nanmin(depth_slice, axis=0)  # (W,)

        # Valid columns only
        valid = np.isfinite(col_depths) & (col_depths > 0)
        if np.sum(valid) == 0:
            return

        # Project to 3D in camera optical frame (Z forward, X right, Y down)
        fx = cam_info.k[0]
        cx = cam_info.k[2]
        cols = np.arange(w, dtype=np.float32)
        z_cam = col_depths  # depth = Z in camera frame
        x_cam = (cols - cx) * z_cam / fx  # X in camera frame
        y_cam = np.zeros_like(z_cam)  # Y = 0 for center rows

        # Stack and filter valid
        pts_cam = np.column_stack([x_cam[valid], y_cam[valid], z_cam[valid]])

        # Transform to laser frame
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

        # Bin into scan overlay — NO min_points_per_bin threshold
        indices = np.rint(
            (angles - self._angle_min) / self._angle_increment
        ).astype(np.int32)
        in_bounds = (indices >= 0) & (indices < self._num_bins)
        idx = indices[in_bounds]
        rng = ranges[in_bounds]

        overlay = CameraOverlay(self._num_bins)
        if len(idx) > 0:
            np.minimum.at(overlay.min_ranges, idx, rng)
            overlay.point_count = len(idx)
        overlay.timestamp = time.monotonic()

        with self._lock:
            self._overlays[cam_name] = overlay

    # ------------------------------------------------------------------
    # Scan callback — lidar + footprint + rear crop + camera gap-fill
    # ------------------------------------------------------------------
    def _scan_cb(self, scan_msg: LaserScan):
        t0 = time.monotonic_ns()

        # Initialize on first scan
        if not self._initialized:
            self._num_bins = len(scan_msg.ranges)
            self._angle_min = scan_msg.angle_min
            self._angle_increment = scan_msg.angle_increment
            self._scan_template = scan_msg

            # Rear crop mask
            if self.rear_crop_deg < 180.0:
                angles = self._angle_min + np.arange(self._num_bins, dtype=np.float32) * self._angle_increment
                limit_rad = np.radians(self.rear_crop_deg)
                self._rear_crop_mask = (angles > limit_rad) | (angles < -limit_rad)
                self.get_logger().info(
                    f'Rear crop: ±{self.rear_crop_deg:.0f}° '
                    f'({int(np.sum(self._rear_crop_mask))} bins cropped)')

            # Pre-compute footprint geometry (constant for scan layout)
            if self.footprint_filter is not None:
                self.footprint_filter.cache_geometry(
                    self._num_bins, self._angle_min, self._angle_increment)

            self._initialized = True
            n = self._num_bins
            self.get_logger().info(
                f'Scan: {n} bins, [{np.degrees(scan_msg.angle_min):.1f}, '
                f'{np.degrees(scan_msg.angle_max):.1f}] deg, '
                f'footprint={self.enable_footprint}')

        # Step 1: Copy lidar, NaN → inf
        fused = np.array(scan_msg.ranges, dtype=np.float32)
        np.nan_to_num(fused, nan=np.inf, copy=False)

        # Step 2: Footprint filter — remove wheelchair self-detections from lidar
        # Must run BEFORE camera merge: lidar sees wheels, cameras at 1.34m cannot
        if self.footprint_filter is not None:
            self.footprint_filter.filter_scan(fused)

        # Step 3: Rear crop — set rear bins to inf (cameras fill them)
        if self._rear_crop_mask is not None:
            fused[self._rear_crop_mask] = np.inf

        # Publish lidar-only debug (after footprint + crop, before camera fill)
        self._publish(fused.copy(), scan_msg, self.lidar_pub)

        # Step 4: Camera gap-fill
        now = time.monotonic()
        if (now - self._start_time) < self.camera_warmup:
            self._publish(fused, scan_msg, self.fused_pub)
            return

        n_filled = 0
        with self._lock:
            for cam_name in self._cam_names:
                ov = self._overlays.get(cam_name)
                if ov is None or ov.point_count == 0:
                    continue
                age = now - ov.timestamp
                if age > self.max_camera_age:
                    continue

                # Gap-fill: write ONLY where lidar is inf
                lidar_gap = np.isinf(fused)
                cam_valid = np.isfinite(ov.min_ranges)
                fill = lidar_gap & cam_valid
                n = int(np.sum(fill))
                if n > 0:
                    fused[fill] = ov.min_ranges[fill]
                    n_filled += n

        self._total_fills += n_filled
        self._frame_count += 1

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
            f'[V6] {fc} frames | {avg_us}us avg / {self._max_latency_us}us max | '
            f'gap-fills: {avg_fills}/frame ({self._total_fills} total) | '
            f'{" | ".join(cam_status)}')

        self._frame_count = 0
        self._total_fills = 0
        self._total_latency_us = 0
        self._max_latency_us = 0


def main(args=None):
    rclpy.init(args=args)
    node = ScanFusionV6()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
