#!/usr/bin/env python3
"""
MONO SCAN FUSION — LaserScan from DA3 monocular depth images.
==============================================================
Created: 2026-03-05

Mirrors scan_fusion_v9.py but uses DA3Metric-Large mono depth images
instead of RealSense stereo PointCloud2.

Pipeline per camera:
  depth_image (uint16 mm) + camera_info → back-project to 3D in optical frame
  → transform to laser frame via TF → height filter → project to 2D polar
  → bin into overlay (min range per angular bin)

Scan callback (10Hz, driven by /scan_filtered):
  copy lidar → footprint filter → rear crop
  → for each camera: if cam < lidar, use cam → publish /scan_mono_fused

Subscribes to:
  /camera/mono_da3/image_raw       (front DA3 depth, uint16 mm)
  /camera/mono_da3/camera_info     (front DA3 intrinsics)
  /mapping_camera/mono_da3/...     (left)
  /right_camera/mono_da3/...       (right)
  /scan_filtered                   (lidar base scan)

Publishes:
  /scan_mono_fused                 (fused LaserScan)
  /scan_mono_lidar_only            (debug: lidar-only after footprint filter)
  /scan_mono_{camera_name}         (debug: per-camera scan contribution)

DO NOT EDIT — create a new versioned file instead.
"""

import array as _array
import math
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

from sensor_msgs.msg import LaserScan, Image, CameraInfo
from cv_bridge import CvBridge
from tf2_ros import Buffer, TransformListener


SENSOR_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)


@dataclass
class CameraOverlay:
    """Per-camera processed polar data: min range per scan bin."""
    min_ranges: np.ndarray   # float32[num_bins], inf = no data
    timestamp: float         # time.monotonic() when processed
    point_count: int         # valid bins contributed
    heading: Optional[float] = None


class WheelchairFootprintFilter:
    """URDF-calibrated self-detection filter (identical to scan_fusion_v9)."""

    def __init__(self):
        self.min_valid_range = 0.20
        self.robot_half_width = 0.33
        self.robot_rear = 0.50
        self.robot_front = 0.20

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


class MonoScanFusion(Node):
    """Fuse DA3 mono depth images into a LaserScan, mirroring scan_fusion_v9."""

    def __init__(self):
        super().__init__('mono_scan_fusion')

        # ---- Parameters ----
        self.declare_parameter('scan_topic', '/scan_filtered')
        self.declare_parameter('output_topic', '/scan_mono_fused')
        self.declare_parameter('laser_frame', 'laser')

        self.declare_parameter('min_height', 0.10)
        self.declare_parameter('max_height', 1.80)

        self.declare_parameter('max_camera_age_ms', 350.0)
        self.declare_parameter('camera_warmup_sec', 5.0)

        self.declare_parameter('enable_footprint', True)
        self.declare_parameter('rear_crop_deg', 180.0)
        self.declare_parameter('downsample_stride', 4)
        self.declare_parameter('min_camera_points_per_bin', 2)
        self.declare_parameter('turn_suppress_wz', 0.10)

        # Camera configs: name, depth_topic, info_topic, optical_frame
        cameras_config = [
            ('front_camera',
             '/camera/mono_da3/image_raw',
             '/camera/mono_da3/camera_info',
             'camera_depth_optical_frame'),
            ('left_camera',
             '/mapping_camera/mono_da3/image_raw',
             '/mapping_camera/mono_da3/camera_info',
             'mapping_camera_depth_optical_frame'),
            ('right_camera',
             '/right_camera/mono_da3/image_raw',
             '/right_camera/mono_da3/camera_info',
             'right_camera_depth_optical_frame'),
        ]

        for name, depth_topic, info_topic, frame in cameras_config:
            self.declare_parameter(f'{name}.enabled', True)
            self.declare_parameter(f'{name}.depth_topic', depth_topic)
            self.declare_parameter(f'{name}.info_topic', info_topic)
            self.declare_parameter(f'{name}.frame', frame)
            self.declare_parameter(f'{name}.max_depth', 3.5)
            self.declare_parameter(f'{name}.min_depth', 0.30)

        # ---- Read parameters ----
        self.laser_frame = self.get_parameter('laser_frame').value
        self.min_height = float(self.get_parameter('min_height').value)
        self.max_height = float(self.get_parameter('max_height').value)
        self.max_camera_age = float(self.get_parameter('max_camera_age_ms').value) / 1000.0
        self.camera_warmup = float(self.get_parameter('camera_warmup_sec').value)
        self.rear_crop_deg = float(self.get_parameter('rear_crop_deg').value)
        self.enable_footprint = bool(self.get_parameter('enable_footprint').value)
        self.downsample = int(self.get_parameter('downsample_stride').value)
        self.min_points_per_bin = int(self.get_parameter('min_camera_points_per_bin').value)
        self.turn_suppress_wz = float(self.get_parameter('turn_suppress_wz').value)

        # ---- Footprint filter ----
        self.footprint_filter = WheelchairFootprintFilter() if self.enable_footprint else None

        # ---- Callback groups ----
        self._cam_cb_group = ReentrantCallbackGroup()
        self._scan_cb_group = MutuallyExclusiveCallbackGroup()

        # ---- TF ----
        self.tf_buffer = Buffer(cache_time=rclpy.duration.Duration(seconds=10))
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ---- State ----
        self._lock = threading.Lock()
        self._bridge = CvBridge()
        self._overlays: Dict[str, Optional[CameraOverlay]] = {}
        self._tf_cache: Dict[str, Optional[tuple]] = {}
        self._intrinsics: Dict[str, Optional[dict]] = {}
        self._num_bins = 0
        self._angle_min = 0.0
        self._angle_increment = 0.0
        self._rear_crop_mask = None
        self._initialized = False
        self._start_time = time.monotonic()

        # Turn detection
        self._last_heading = None
        self._last_heading_time = None
        self._turn_skipped = 0

        # Stats
        self._frame_count = 0
        self._total_cam_bins = 0
        self._total_latency_us = 0

        # ---- Publishers ----
        self.fused_pub = self.create_publisher(
            LaserScan, self.get_parameter('output_topic').value, 10)
        self.lidar_pub = self.create_publisher(LaserScan, '/scan_mono_lidar_only', 10)
        self._cam_pubs: Dict[str, any] = {}

        # ---- Subscribe to lidar ----
        self.create_subscription(
            LaserScan, self.get_parameter('scan_topic').value,
            self._scan_cb, SENSOR_QOS,
            callback_group=self._scan_cb_group)

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
            self._overlays[name] = None
            self._intrinsics[name] = None

            scan_topic_name = f'/scan_mono_{name}'
            self._cam_pubs[name] = self.create_publisher(
                LaserScan, scan_topic_name, 10)

            setattr(self, f'_{name}_frame', frame)
            setattr(self, f'_{name}_min_depth',
                    float(self.get_parameter(f'{name}.min_depth').value))
            setattr(self, f'_{name}_max_depth',
                    float(self.get_parameter(f'{name}.max_depth').value))

            # Subscribe to mono depth image
            self.create_subscription(
                Image, depth_topic,
                lambda msg, n=name: self._depth_image_cb(msg, n), SENSOR_QOS,
                callback_group=self._cam_cb_group)

            # Subscribe to camera info (for intrinsics)
            self.create_subscription(
                CameraInfo, info_topic,
                lambda msg, n=name: self._camera_info_cb(msg, n), SENSOR_QOS,
                callback_group=self._cam_cb_group)

        # ---- Diagnostics timer ----
        self.create_timer(10.0, self._print_stats)

        info = self.get_logger().info
        info('=' * 60)
        info('MONO SCAN FUSION — DA3 Depth Image to LaserScan')
        info('=' * 60)
        info(f'  Cameras: {", ".join(self._cam_names)}')
        info(f'  Logic: if cam_dist < lidar_dist -> use camera')
        info(f'  Height: {self.min_height:.2f}-{self.max_height:.2f} m')
        info(f'  Downsample: stride {self.downsample}')
        info(f'  Min points/bin: {self.min_points_per_bin}')
        info(f'  Footprint filter: {self.enable_footprint}')
        info(f'  Rear crop: +/-{self.rear_crop_deg:.0f} deg')
        for name in self._cam_names:
            min_d = getattr(self, f'_{name}_min_depth')
            max_d = getattr(self, f'_{name}_max_depth')
            info(f'  {name}: {min_d:.1f}-{max_d:.1f}m depth')

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
    def _get_heading(self) -> Optional[float]:
        try:
            tf = self.tf_buffer.lookup_transform('odom', 'base_link', Time())
            q = tf.transform.rotation
            siny = 2.0 * (q.w * q.z + q.x * q.y)
            cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            return math.atan2(siny, cosy)
        except Exception:
            return None

    # ------------------------------------------------------------------
    def _camera_info_cb(self, msg: CameraInfo, cam_name: str):
        """Capture mono depth intrinsics per camera (once)."""
        if self._intrinsics.get(cam_name) is not None:
            return
        K = msg.k
        self._intrinsics[cam_name] = {
            'fx': K[0], 'fy': K[4], 'cx': K[2], 'cy': K[5],
            'width': msg.width, 'height': msg.height,
        }
        self.get_logger().info(
            f'[{cam_name}] mono intrinsics: {msg.width}x{msg.height}, '
            f'fx={K[0]:.1f}, fy={K[4]:.1f}')

    # ------------------------------------------------------------------
    def _depth_image_cb(self, msg: Image, cam_name: str):
        """Convert DA3 mono depth image to polar overlay (vectorized)."""
        if not self._initialized:
            return

        intrinsics = self._intrinsics.get(cam_name)
        if intrinsics is None:
            return

        cam_frame = getattr(self, f'_{cam_name}_frame')
        tf_data = self._get_tf(cam_frame)
        if tf_data is None:
            return

        R, t_vec = tf_data
        min_depth = getattr(self, f'_{cam_name}_min_depth')
        max_depth = getattr(self, f'_{cam_name}_max_depth')

        # Parse depth image (uint16 mm -> float32 m)
        try:
            depth_mm = self._bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception:
            return

        depth_m = depth_mm.astype(np.float32) / 1000.0
        h, w = depth_m.shape

        # Downsample for speed
        stride = self.downsample
        if stride > 1:
            depth_m = depth_m[::stride, ::stride]
            h_ds, w_ds = depth_m.shape
        else:
            h_ds, w_ds = h, w

        fx = intrinsics['fx']
        fy = intrinsics['fy']
        cx = intrinsics['cx']
        cy = intrinsics['cy']

        # Adjust intrinsics for downsampling
        if stride > 1:
            fx_ds = fx / stride
            fy_ds = fy / stride
            cx_ds = cx / stride
            cy_ds = cy / stride
        else:
            fx_ds, fy_ds, cx_ds, cy_ds = fx, fy, cx, cy

        # Depth filter
        valid = (depth_m >= min_depth) & (depth_m <= max_depth)
        if not np.any(valid):
            return

        # Back-project to 3D in camera optical frame (vectorized)
        # In optical frame: x=right, y=down, z=forward (depth)
        v_idx, u_idx = np.where(valid)
        z = depth_m[valid]
        x_cam = (u_idx.astype(np.float32) - cx_ds) * z / fx_ds
        y_cam = (v_idx.astype(np.float32) - cy_ds) * z / fy_ds

        # Transform to laser frame: pts_laser = R @ pts_cam + t
        pts_cam = np.column_stack([x_cam, y_cam, z]).astype(np.float32)
        pts_laser = pts_cam @ R.T + t_vec

        # Height filter in laser frame
        z_laser = pts_laser[:, 2]
        height_ok = (z_laser >= self.min_height) & (z_laser <= self.max_height)
        pts_laser = pts_laser[height_ok]
        if len(pts_laser) == 0:
            return

        # Project to 2D polar in laser frame
        x_l = pts_laser[:, 0]
        y_l = pts_laser[:, 1]
        ranges = np.sqrt(x_l * x_l + y_l * y_l)
        angles = np.arctan2(y_l, x_l)

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
            point_count=int(np.sum(np.isfinite(overlay_ranges))),
            heading=self._get_heading(),
        )

        with self._lock:
            self._overlays[cam_name] = overlay

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
                self._rear_crop_mask = (
                    (robot_angles > limit_rad) | (robot_angles < -limit_rad))

            if self.footprint_filter is not None:
                self.footprint_filter.cache_geometry(self._num_bins, robot_angles)

            self._initialized = True
            self.get_logger().info(
                f'Scan initialized: {self._num_bins} bins, '
                f'[{np.degrees(scan_angles[0]):.1f}, '
                f'{np.degrees(scan_angles[-1]):.1f}] deg')

        # Step 1: Copy lidar, clean invalid ranges
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

        # Step 4: Camera fusion
        now = time.monotonic()
        if (now - self._start_time) < self.camera_warmup:
            self._publish(fused, scan_msg, self.fused_pub)
            return

        current_heading = self._get_heading()

        # Turn detection
        is_turning = False
        if (current_heading is not None and self._last_heading is not None
                and self._last_heading_time is not None):
            dt = now - self._last_heading_time
            if dt > 0.01:
                dh = (current_heading - self._last_heading + math.pi) % (2 * math.pi) - math.pi
                wz_est = abs(dh / dt)
                is_turning = wz_est > self.turn_suppress_wz
        if current_heading is not None:
            self._last_heading = current_heading
            self._last_heading_time = now

        if is_turning:
            self._turn_skipped += 1
            self._publish(fused, scan_msg, self.fused_pub)
            return

        cam_bins_used = 0
        with self._lock:
            for cam_name in self._cam_names:
                ov = self._overlays.get(cam_name)
                if ov is None or ov.point_count == 0:
                    continue
                age = now - ov.timestamp
                if age > self.max_camera_age:
                    continue

                cam_ranges = ov.min_ranges

                # Angular compensation for robot rotation since capture
                if ov.heading is not None and current_heading is not None:
                    delta = current_heading - ov.heading
                    delta = (delta + math.pi) % (2 * math.pi) - math.pi
                    shift = int(round(delta / self._angle_increment))
                    if abs(shift) > 0:
                        cam_ranges = np.roll(cam_ranges, shift)
                        if shift > 0:
                            cam_ranges[:shift] = np.inf
                        else:
                            cam_ranges[shift:] = np.inf

                cam_valid = np.isfinite(cam_ranges)

                # Publish per-camera debug scan
                cam_pub = self._cam_pubs.get(cam_name)
                if cam_pub is not None:
                    self._publish(cam_ranges.copy(), scan_msg, cam_pub)

                # CORE LOGIC: if camera < lidar, use camera
                use_camera = cam_valid & (cam_ranges < fused)
                fused[use_camera] = cam_ranges[use_camera]
                cam_bins_used += int(np.sum(use_camera))

        self._total_cam_bins += cam_bins_used
        self._frame_count += 1

        # Final cleanup
        bad = ~np.isfinite(fused) | (fused <= 0.0)
        fused[bad] = np.inf

        self._publish(fused, scan_msg, self.fused_pub)

        elapsed_us = (time.monotonic_ns() - t0) // 1000
        self._total_latency_us += elapsed_us

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
            # Report waiting status
            intrinsics_status = {n: (self._intrinsics.get(n) is not None)
                                 for n in self._cam_names}
            self.get_logger().info(
                f'[MONO] Waiting... intrinsics: {intrinsics_status}, '
                f'initialized: {self._initialized}')
            return
        avg_us = self._total_latency_us // fc
        avg_cam = self._total_cam_bins // fc
        cam_status = []
        now = time.monotonic()
        for name in self._cam_names:
            ov = self._overlays.get(name)
            if ov and ov.point_count > 0:
                age = now - ov.timestamp
                cam_status.append(f'{name}:{ov.point_count}bins({age*1000:.0f}ms)')
            else:
                cam_status.append(f'{name}:none')

        turn_info = f' | turn_skip:{self._turn_skipped}' if self._turn_skipped > 0 else ''
        self.get_logger().info(
            f'[MONO] {fc} frames | {avg_us}us avg | '
            f'cam bins used: {avg_cam}/frame{turn_info} | '
            f'{" | ".join(cam_status)}')

        self._frame_count = 0
        self._total_cam_bins = 0
        self._total_latency_us = 0
        self._turn_skipped = 0


def main(args=None):
    rclpy.init(args=args)
    node = MonoScanFusion()
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
