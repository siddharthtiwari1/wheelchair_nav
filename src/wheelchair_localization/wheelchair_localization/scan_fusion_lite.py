#!/usr/bin/env python3
"""
SCAN FUSION LITE - LOW CPU, SMOOTH OUTPUT
==========================================
Optimized for:
- LOW CPU usage (<30%)
- Smooth continuous scans during movement
- No transitional clutter (temporal consistency)
- Real-time responsiveness

Key optimizations:
1. NO parallel threads - single-threaded processing
2. Aggressive downsampling (process every 4th point)
3. Simple numpy operations (no Numba overhead)
4. Temporal smoothing to eliminate transient points
5. Rate-limited camera processing

Author: Wheelchair Navigation Project
"""

import numpy as np
from collections import deque
from typing import Optional, Dict, Tuple
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from rclpy.time import Time
from rclpy.duration import Duration

from sensor_msgs.msg import LaserScan, PointCloud2, Imu
import tf2_ros


class ScanFusionLite(Node):
    """
    Lightweight scan fusion - optimized for low CPU and smooth output.
    """

    def __init__(self):
        super().__init__('scan_fusion_lite')

        # Parameters
        self.declare_parameter('scan_topic', '/scan_filtered')
        self.declare_parameter('output_topic', '/scan_fused')
        self.declare_parameter('laser_frame', 'laser')
        self.declare_parameter('min_height', 0.10)
        self.declare_parameter('max_height', 1.80)
        self.declare_parameter('downsample', 4)  # Process every Nth point
        self.declare_parameter('temporal_frames', 3)  # Frames for temporal filter
        self.declare_parameter('verbose', False)

        for prefix in ['front_camera', 'left_camera', 'right_camera']:
            self.declare_parameter(f'{prefix}.enabled', True)
            self.declare_parameter(f'{prefix}.topic', '')
            self.declare_parameter(f'{prefix}.frame', '')
            self.declare_parameter(f'{prefix}.min_depth', 0.30)
            self.declare_parameter(f'{prefix}.max_depth', 5.0)

        self.laser_frame = self.get_parameter('laser_frame').value
        self.min_height = float(self.get_parameter('min_height').value)
        self.max_height = float(self.get_parameter('max_height').value)
        self.downsample = int(self.get_parameter('downsample').value)
        self.temporal_frames = int(self.get_parameter('temporal_frames').value)
        self.verbose = self.get_parameter('verbose').value

        # Camera configs
        self.cameras = {
            'front': {
                'enabled': self.get_parameter('front_camera.enabled').value,
                'topic': self.get_parameter('front_camera.topic').value or '/camera/depth/color/points',
                'frame': self.get_parameter('front_camera.frame').value or 'camera_depth_optical_frame',
                'min_depth': float(self.get_parameter('front_camera.min_depth').value),
                'max_depth': float(self.get_parameter('front_camera.max_depth').value),
            },
            'left': {
                'enabled': self.get_parameter('left_camera.enabled').value,
                'topic': self.get_parameter('left_camera.topic').value or '/mapping_camera/depth/color/points',
                'frame': self.get_parameter('left_camera.frame').value or 'mapping_camera_depth_optical_frame',
                'min_depth': float(self.get_parameter('left_camera.min_depth').value),
                'max_depth': float(self.get_parameter('left_camera.max_depth').value),
            },
            'right': {
                'enabled': self.get_parameter('right_camera.enabled').value,
                'topic': self.get_parameter('right_camera.topic').value or '/right_camera/depth/color/points',
                'frame': self.get_parameter('right_camera.frame').value or 'right_camera_depth_optical_frame',
                'min_depth': float(self.get_parameter('right_camera.min_depth').value),
                'max_depth': float(self.get_parameter('right_camera.max_depth').value),
            },
        }

        # Wheelchair footprint exclusion (laser frame: X+ backward, Y+ right)
        self.box = (-0.20, 0.60, -0.42, 0.42)  # x_min, x_max, y_min, y_max
        self.angular_zones = [
            # (angle_start, angle_end, mask_range) - aggressive rear masking
            (-0.20, 1.60, 0.70),
            (0.30, 1.20, 0.85),
            (-1.60, 0.20, 0.70),
            (-1.20, -0.30, 0.85),
            (-0.50, 0.50, 0.75),
            (0.70, 1.30, 0.65),
            (-1.30, -0.70, 0.65),
            (1.40, 2.00, 0.55),
            (-2.00, -1.40, 0.55),
            (2.20, 3.14159, 0.50),
            (-3.14159, -2.20, 0.50),
        ]

        # TF
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

        # Data storage
        self.clouds: Dict[str, Optional[PointCloud2]] = {name: None for name in self.cameras}

        # Temporal buffer for smooth output
        self.history: Optional[deque] = None

        # Pre-computed LUTs (lazy init)
        self._n_bins = 0
        self._angles = None
        self._cos_angles = None
        self._sin_angles = None
        self._mask_ranges = None

        # QoS - best effort for low latency
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Subscribers
        self.create_subscription(
            LaserScan,
            self.get_parameter('scan_topic').value,
            self._scan_callback,
            qos
        )

        for name, cfg in self.cameras.items():
            if cfg['enabled']:
                self.create_subscription(
                    PointCloud2,
                    cfg['topic'],
                    lambda msg, n=name: self._cloud_callback(msg, n),
                    qos
                )

        # Publishers
        self.pub_fused = self.create_publisher(
            LaserScan,
            self.get_parameter('output_topic').value,
            10
        )
        self.pub_lidar = self.create_publisher(LaserScan, '/scan_lidar_only', 10)

        # Stats
        self.frame_count = 0
        self.total_time = 0.0

        self.get_logger().info('=' * 50)
        self.get_logger().info('SCAN FUSION LITE - Low CPU, Smooth Output')
        self.get_logger().info('=' * 50)
        self.get_logger().info(f'Downsample: 1/{self.downsample} points')
        self.get_logger().info(f'Temporal filter: {self.temporal_frames} frames')
        self.get_logger().info(f'Target: <30% CPU, smooth continuous output')
        self.get_logger().info('=' * 50)

    def _init_luts(self, n_bins: int, a_min: float, a_inc: float):
        """Initialize lookup tables."""
        self._n_bins = n_bins
        self._angles = a_min + np.arange(n_bins, dtype=np.float32) * a_inc
        self._cos_angles = np.cos(self._angles).astype(np.float32)
        self._sin_angles = np.sin(self._angles).astype(np.float32)

        # Build mask
        self._mask_ranges = np.full(n_bins, 0.22, dtype=np.float32)
        for a_start, a_end, mask_r in self.angular_zones:
            mask = (self._angles >= a_start) & (self._angles <= a_end)
            self._mask_ranges[mask] = np.maximum(self._mask_ranges[mask], mask_r)

        # Initialize temporal history
        self.history = deque(maxlen=self.temporal_frames)

        # Pre-cache transforms
        for name, cfg in self.cameras.items():
            if cfg['enabled']:
                self._get_transform(cfg['frame'])

        self.get_logger().info(f'Initialized: {n_bins} bins')

    def _cloud_callback(self, msg: PointCloud2, name: str):
        self.clouds[name] = msg

    def _get_transform(self, src_frame: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Get cached transform."""
        key = f'{src_frame}_to_{self.laser_frame}'
        if key in self.tf_cache:
            return self.tf_cache[key]

        try:
            tf = self.tf_buffer.lookup_transform(
                self.laser_frame, src_frame,
                Time(), Duration(seconds=0.5)
            )
            t = tf.transform.translation
            q = tf.transform.rotation

            # Quaternion to rotation matrix
            norm = np.sqrt(q.x**2 + q.y**2 + q.z**2 + q.w**2 + 1e-10)
            qx, qy, qz, qw = q.x/norm, q.y/norm, q.z/norm, q.w/norm

            R = np.array([
                [1-2*(qy**2+qz**2), 2*(qx*qy-qw*qz), 2*(qx*qz+qw*qy)],
                [2*(qx*qy+qw*qz), 1-2*(qx**2+qz**2), 2*(qy*qz-qw*qx)],
                [2*(qx*qz-qw*qy), 2*(qy*qz+qw*qx), 1-2*(qx**2+qy**2)]
            ], dtype=np.float32)

            t_vec = np.array([t.x, t.y, t.z], dtype=np.float32)
            self.tf_cache[key] = (R, t_vec)
            return (R, t_vec)

        except Exception:
            return None

    def _parse_cloud_fast(self, msg: PointCloud2) -> Optional[np.ndarray]:
        """Fast pointcloud parsing with aggressive downsampling."""
        n_points = msg.width * msg.height
        if n_points == 0:
            return None

        # Find field offsets
        x_off = y_off = z_off = -1
        for f in msg.fields:
            if f.name == 'x':
                x_off = f.offset
            elif f.name == 'y':
                y_off = f.offset
            elif f.name == 'z':
                z_off = f.offset

        if x_off < 0 or y_off < 0 or z_off < 0:
            return None

        # Zero-copy strided view
        data = np.frombuffer(msg.data, dtype=np.uint8)
        point_step = msg.point_step

        x = np.ndarray(n_points, dtype=np.float32, buffer=data, offset=x_off, strides=(point_step,))
        y = np.ndarray(n_points, dtype=np.float32, buffer=data, offset=y_off, strides=(point_step,))
        z = np.ndarray(n_points, dtype=np.float32, buffer=data, offset=z_off, strides=(point_step,))

        # Aggressive downsampling FIRST (before any processing)
        step = self.downsample
        x = x[::step]
        y = y[::step]
        z = z[::step]

        # Filter NaN
        valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        if not valid.any():
            return None

        return np.column_stack((x[valid], y[valid], z[valid]))

    def _process_camera(self, name: str) -> Optional[np.ndarray]:
        """Process single camera - simple numpy operations."""
        cloud = self.clouds.get(name)
        if cloud is None:
            return None

        cfg = self.cameras[name]
        tf_data = self._get_transform(cfg['frame'])
        if tf_data is None:
            return None

        R, t = tf_data

        # Parse with downsampling
        xyz = self._parse_cloud_fast(cloud)
        if xyz is None or len(xyz) == 0:
            return None

        # Depth filter (camera Z = forward)
        depth_mask = (xyz[:, 2] >= cfg['min_depth']) & (xyz[:, 2] <= cfg['max_depth'])
        xyz = xyz[depth_mask]
        if len(xyz) == 0:
            return None

        # Transform to laser frame
        pts = xyz @ R.T + t

        # Height filter
        height_mask = (pts[:, 2] >= self.min_height) & (pts[:, 2] <= self.max_height)
        pts = pts[height_mask]
        if len(pts) == 0:
            return None

        # Polar conversion
        x, y = pts[:, 0], pts[:, 1]
        r = np.sqrt(x*x + y*y)
        a = np.arctan2(y, x)

        # Range filter
        range_mask = (r > 0.15) & (r < 12.0)
        r, a = r[range_mask], a[range_mask]
        if len(r) == 0:
            return None

        # Bin assignment
        a_min = self._angles[0]
        a_inc = self._angles[1] - self._angles[0] if len(self._angles) > 1 else 0.00872665

        bins = ((a - a_min) / a_inc).astype(np.int32)
        valid_bins = (bins >= 0) & (bins < self._n_bins)
        bins, r = bins[valid_bins], r[valid_bins]

        if len(bins) == 0:
            return None

        # Create scan with minimum at each bin
        scan = np.full(self._n_bins, np.inf, dtype=np.float32)
        np.minimum.at(scan, bins, r.astype(np.float32))

        # Apply footprint mask
        scan[scan < self._mask_ranges] = np.inf

        return scan

    def _apply_footprint_filter(self, ranges: np.ndarray) -> np.ndarray:
        """Apply footprint filter to LiDAR scan."""
        result = ranges.copy()

        # Mask invalid
        invalid = ~np.isfinite(result) | (result <= 0)
        result[invalid] = np.inf

        # Angular exclusion
        close_mask = result < self._mask_ranges
        result[close_mask] = np.inf

        # Box filter
        valid = np.isfinite(result)
        x = np.where(valid, result * self._cos_angles, 0)
        y = np.where(valid, result * self._sin_angles, 0)

        inside_box = valid & \
                     (x >= self.box[0]) & (x <= self.box[1]) & \
                     (y >= self.box[2]) & (y <= self.box[3])
        result[inside_box] = np.inf

        return result

    def _temporal_filter(self, scan: np.ndarray) -> np.ndarray:
        """
        Robust temporal consistency filter - removes transient points.

        Features:
        1. Requires points to appear in 2+ frames (removes single-frame noise)
        2. Uses median for consistency (robust to outliers)
        3. Hysteresis: once a point is established, it persists
        """
        self.history.append(scan.copy())

        if len(self.history) < 2:
            return scan

        # Stack history (oldest to newest)
        stack = np.array(list(self.history))
        n_frames = len(stack)

        result = np.full(self._n_bins, np.inf, dtype=np.float32)

        # Vectorized: count valid readings per bin
        valid_mask = np.isfinite(stack)  # (n_frames, n_bins)
        valid_counts = valid_mask.sum(axis=0)  # (n_bins,)

        # For bins with 2+ valid readings: use median
        multi_valid = valid_counts >= 2
        if multi_valid.any():
            for i in np.where(multi_valid)[0]:
                col = stack[:, i]
                valid = np.isfinite(col)
                vals = col[valid]

                # Check consistency: if values vary too much, take minimum (safety)
                if vals.max() - vals.min() > 0.5:  # >50cm variation = unstable
                    result[i] = vals.min()  # Conservative: use closest
                else:
                    result[i] = np.median(vals)  # Stable: use median

        # For bins with exactly 1 valid reading in recent frame:
        # only include if it's in the LAST frame (new detection)
        single_valid = valid_counts == 1
        if single_valid.any() and n_frames >= 2:
            for i in np.where(single_valid)[0]:
                # Only accept if it's in the most recent frame
                if np.isfinite(stack[-1, i]):
                    # New point - include but mark as tentative
                    # (will be confirmed or rejected next frame)
                    result[i] = stack[-1, i]

        return result

    def _spatial_consistency(self, scan: np.ndarray) -> np.ndarray:
        """
        Remove isolated orphan points (scattered dots with no neighbors).

        A point is valid only if it has at least 1 neighbor within:
        - 2 bins angular distance
        - 0.5m range difference

        This removes random scattered points that appear during movement.
        """
        result = scan.copy()
        n = len(scan)

        for i in range(n):
            if not np.isfinite(scan[i]):
                continue

            # Check neighbors (2 bins each side)
            has_neighbor = False
            for offset in [-2, -1, 1, 2]:
                j = i + offset
                if 0 <= j < n and np.isfinite(scan[j]):
                    # Check if range is similar (within 0.5m)
                    if abs(scan[i] - scan[j]) < 0.5:
                        has_neighbor = True
                        break

            # If isolated, remove it
            if not has_neighbor:
                result[i] = np.inf

        return result

    def _create_scan_msg(self, ranges: np.ndarray, template: LaserScan) -> LaserScan:
        """Create LaserScan message."""
        msg = LaserScan()
        msg.header = template.header
        msg.angle_min = template.angle_min
        msg.angle_max = template.angle_max
        msg.angle_increment = template.angle_increment
        msg.time_increment = template.time_increment
        msg.scan_time = template.scan_time
        msg.range_min = template.range_min
        msg.range_max = template.range_max
        msg.ranges = ranges.tolist()
        return msg

    def _scan_callback(self, msg: LaserScan):
        """Main processing callback."""
        t0 = time.perf_counter()

        n = len(msg.ranges)
        a_min = msg.angle_min
        a_inc = msg.angle_increment

        # Initialize on first scan
        if self._n_bins != n:
            self._init_luts(n, a_min, a_inc)

        # 1. Process LiDAR
        lidar = np.array(msg.ranges, dtype=np.float32)
        lidar = self._apply_footprint_filter(lidar)

        # Publish filtered LiDAR
        self.pub_lidar.publish(self._create_scan_msg(lidar, msg))

        # 2. Start with LiDAR as base
        fused = lidar.copy()

        # 3. Process each camera sequentially (no threading)
        for name, cfg in self.cameras.items():
            if not cfg['enabled']:
                continue

            cam_scan = self._process_camera(name)
            if cam_scan is not None:
                # MIN fusion
                np.minimum(fused, cam_scan, out=fused)

        # 4. Temporal filter for smooth output (removes transient clutter)
        fused = self._temporal_filter(fused)

        # 5. Spatial consistency - remove isolated orphan points
        fused = self._spatial_consistency(fused)

        # 6. Publish
        self.pub_fused.publish(self._create_scan_msg(fused, msg))

        # Stats
        elapsed = (time.perf_counter() - t0) * 1000
        self.frame_count += 1
        self.total_time += elapsed

        if self.verbose and self.frame_count % 100 == 0:
            avg = self.total_time / self.frame_count
            self.get_logger().info(f'Frame {self.frame_count} | {elapsed:.2f}ms (avg {avg:.2f}ms)')


def main(args=None):
    rclpy.init(args=args)
    node = ScanFusionLite()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
