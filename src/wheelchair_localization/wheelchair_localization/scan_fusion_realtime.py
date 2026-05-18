#!/usr/bin/env python3
"""
ZERO-LATENCY SCAN FUSION WITH IMU MOTION COMPENSATION
======================================================
Designed for maximum responsiveness - NO temporal buffering.

Key Design Principles:
1. ZERO LATENCY: No temporal filters, no ring buffers
2. IMU MOTION COMPENSATION: Undistort scans using angular velocity
3. DIRECT PASSTHROUGH: Footprint filter only, no smoothing
4. SINGLE-THREAD: Avoid thread pool overhead for low latency

IMU Integration:
- Uses /imu topic (filtered, base_link frame)
- Angular velocity (gyro) for scan motion compensation
- Compensates for robot rotation during 360° scan acquisition

Data Flow:
  /scan_filtered ─┬─→ footprint_filter ─┬─→ /scan_lidar_only
                  │                      │
  /imu ───────────┼─→ motion_compensate ─┤
                  │                      │
  cameras ────────┴─→ footprint_filter ─┴─→ min() ─→ /scan_fused
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
import time
import threading

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from rclpy.time import Time
from rclpy.duration import Duration

from sensor_msgs.msg import LaserScan, PointCloud2, Imu
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header

import tf2_ros
from tf2_ros import TransformException


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class CameraConfig:
    name: str
    topic: str
    frame: str
    enabled: bool = True
    min_depth: float = 0.30
    max_depth: float = 5.0
    downsample: int = 2  # Less downsampling for better resolution


# =============================================================================
# LIGHTWEIGHT FOOTPRINT FILTER (No LUT overhead)
# =============================================================================

class FastFootprintFilter:
    """
    Minimal-overhead footprint filter.
    Pre-computes exclusion ranges per angle once, then O(n) filtering.
    """

    def __init__(self):
        self.min_range = 0.22
        self.box_bounds = (-0.20, 0.60, -0.42, 0.42)  # x_min, x_max, y_min, y_max

        # Angular exclusion: (angle_min, angle_max, range_max)
        self.exclusion_zones = [
            (-0.20, 1.60, 0.70),    # Right rear wheel
            (0.30, 1.20, 0.80),     # Right wheel hub
            (-1.60, 0.20, 0.70),    # Left rear wheel
            (-1.20, -0.30, 0.80),   # Left wheel hub
            (-0.50, 0.50, 0.75),    # Direct rear
            (0.70, 1.30, 0.65),     # Right corner
            (-1.30, -0.70, 0.65),   # Left corner
            (1.40, 2.00, 0.55),     # Right side
            (-2.00, -1.40, 0.55),   # Left side
            (2.20, 3.14159, 0.45),  # Right caster
            (-3.14159, -2.20, 0.45), # Left caster
        ]

        # Pre-computed mask (built on first scan)
        self._mask_ranges: Optional[np.ndarray] = None
        self._angles: Optional[np.ndarray] = None
        self._cos: Optional[np.ndarray] = None
        self._sin: Optional[np.ndarray] = None

    def _init_mask(self, num_bins: int, angle_min: float, angle_inc: float):
        """Build mask once, reuse forever."""
        self._angles = angle_min + np.arange(num_bins, dtype=np.float32) * angle_inc
        self._cos = np.cos(self._angles)
        self._sin = np.sin(self._angles)

        # Build exclusion range per bin
        self._mask_ranges = np.full(num_bins, self.min_range, dtype=np.float32)
        for a_min, a_max, r_max in self.exclusion_zones:
            mask = (self._angles >= a_min) & (self._angles <= a_max)
            self._mask_ranges[mask] = np.maximum(self._mask_ranges[mask], r_max)

    def filter(self, ranges: np.ndarray, angle_min: float, angle_inc: float) -> np.ndarray:
        """Filter in single pass - O(n)."""
        n = len(ranges)

        # Initialize on first call
        if self._mask_ranges is None or len(self._mask_ranges) != n:
            self._init_mask(n, angle_min, angle_inc)

        result = ranges.copy()

        # 1. Angular exclusion (vectorized)
        valid = np.isfinite(ranges) & (ranges > 0)
        exclude = valid & (ranges < self._mask_ranges)
        result[exclude] = np.inf

        # 2. Box filter (only remaining valid points)
        still_valid = np.isfinite(result) & (result > 0)
        if np.any(still_valid):
            x = result * self._cos
            y = result * self._sin
            x_min, x_max, y_min, y_max = self.box_bounds
            inside = still_valid & (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)
            result[inside] = np.inf

        return result


# =============================================================================
# IMU MOTION COMPENSATOR
# =============================================================================

# =============================================================================
# GAP FILLING & SPATIAL CONSISTENCY
# =============================================================================

class GapFiller:
    """
    Fast gap filling using linear interpolation.

    For small gaps (< max_gap_size bins), interpolates between valid neighbors.
    This is O(n) and adds minimal latency.
    """

    def __init__(self, max_gap_size: int = 5, max_range_diff: float = 0.5):
        """
        Args:
            max_gap_size: Maximum gap to fill (bins)
            max_range_diff: Max difference between endpoints to interpolate (meters)
        """
        self.max_gap_size = max_gap_size
        self.max_range_diff = max_range_diff

    def fill(self, scan: np.ndarray) -> np.ndarray:
        """
        Fill small gaps with linear interpolation.
        O(n) single pass algorithm.
        """
        result = scan.copy()
        n = len(scan)

        valid = np.isfinite(scan)

        # Find gap starts and ends
        i = 0
        while i < n:
            if not valid[i]:
                # Found start of gap
                gap_start = i

                # Find end of gap
                while i < n and not valid[i]:
                    i += 1
                gap_end = i

                gap_size = gap_end - gap_start

                # Only fill small gaps with valid neighbors
                if gap_size <= self.max_gap_size:
                    # Find left neighbor
                    left_idx = gap_start - 1
                    left_val = scan[left_idx] if left_idx >= 0 and valid[left_idx] else None

                    # Find right neighbor
                    right_idx = gap_end
                    right_val = scan[right_idx] if right_idx < n and valid[right_idx] else None

                    if left_val is not None and right_val is not None:
                        # Both neighbors exist - check range difference
                        if abs(left_val - right_val) <= self.max_range_diff:
                            # Linear interpolation
                            for j in range(gap_start, gap_end):
                                t = (j - left_idx) / (right_idx - left_idx)
                                result[j] = left_val + t * (right_val - left_val)
                    elif left_val is not None:
                        # Only left neighbor - extend (for edge gaps)
                        for j in range(gap_start, min(gap_end, gap_start + 2)):
                            result[j] = left_val
                    elif right_val is not None:
                        # Only right neighbor - extend (for edge gaps)
                        for j in range(max(gap_start, gap_end - 2), gap_end):
                            result[j] = right_val
            else:
                i += 1

        return result


class SpatialConsistencyFilter:
    """
    Enforces spatial consistency between neighboring bins.

    Removes isolated outliers that are inconsistent with neighbors.
    Uses median-based approach for robustness.
    """

    def __init__(self, window_size: int = 3, max_deviation: float = 0.3):
        """
        Args:
            window_size: Number of neighbors to consider
            max_deviation: Max deviation from local median (meters)
        """
        self.window_size = window_size
        self.half_window = window_size // 2
        self.max_deviation = max_deviation

    def filter(self, scan: np.ndarray) -> np.ndarray:
        """
        Remove outliers that deviate too much from local median.
        O(n * window_size) but window is small.
        """
        result = scan.copy()
        n = len(scan)

        for i in range(n):
            if not np.isfinite(scan[i]):
                continue

            # Get window
            start = max(0, i - self.half_window)
            end = min(n, i + self.half_window + 1)

            window = scan[start:end]
            valid_vals = window[np.isfinite(window)]

            if len(valid_vals) >= 2:
                median = np.median(valid_vals)

                # If current value deviates too much, replace with median
                if abs(scan[i] - median) > self.max_deviation:
                    result[i] = median

        return result


class MotionCompensator:
    """
    Compensates for robot rotation during scan acquisition.

    LiDAR scans take ~100ms to complete 360°. During this time, if robot
    is rotating, early rays and late rays see different actual angles.

    Uses IMU angular velocity (gyro Z) to shift ray angles.
    """

    def __init__(self):
        self.last_angular_vel_z = 0.0  # rad/s
        self.scan_time = 0.1  # 10Hz LiDAR = 100ms per scan

    def update_imu(self, angular_vel_z: float):
        """Update current angular velocity from IMU."""
        self.last_angular_vel_z = angular_vel_z

    def compensate(self, ranges: np.ndarray, angle_min: float,
                   angle_inc: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return motion-compensated angles.

        During scan acquisition:
        - First ray (bin 0) was captured at t=0
        - Last ray (bin N) was captured at t=scan_time
        - If rotating at ω rad/s, bin i was captured at angle_actual = angle_nominal - ω*t_i

        Returns:
            ranges: unchanged
            angles: compensated angles
        """
        n = len(ranges)

        # Time offset per bin (assuming uniform acquisition)
        t_per_bin = self.scan_time / n
        time_offsets = np.arange(n, dtype=np.float32) * t_per_bin

        # Nominal angles
        angles = angle_min + np.arange(n, dtype=np.float32) * angle_inc

        # Compensate: during acquisition, robot rotated by ω*t
        # So actual world angle = nominal - ω*t (subtract because robot moved)
        if abs(self.last_angular_vel_z) > 0.01:  # Only if significant rotation
            angle_correction = self.last_angular_vel_z * time_offsets
            angles = angles - angle_correction

        return ranges, angles


# =============================================================================
# FAST POINTCLOUD CONVERTER
# =============================================================================

class FastPCConverter:
    """Minimal-overhead pointcloud to scan conversion."""

    def __init__(self, min_height: float, max_height: float):
        self.min_height = min_height
        self.max_height = max_height

    def convert(self, cloud: PointCloud2, transform: np.ndarray,
                cam: CameraConfig, num_bins: int,
                angle_min: float, angle_inc: float) -> Optional[np.ndarray]:
        """Convert pointcloud to scan - optimized path."""
        try:
            pts = point_cloud2.read_points_numpy(
                cloud, field_names=("x", "y", "z"), skip_nans=True)
            if len(pts) == 0:
                return None

            if pts.dtype.names:
                xyz = np.column_stack([pts['x'], pts['y'], pts['z']])
            else:
                xyz = pts[:, :3] if pts.ndim == 2 else None

            if xyz is None or len(xyz) == 0:
                return None
        except:
            return None

        # Downsample
        if cam.downsample > 1:
            xyz = xyz[::cam.downsample]

        # Depth filter
        mask = (xyz[:, 2] >= cam.min_depth) & (xyz[:, 2] <= cam.max_depth)
        xyz = xyz[mask]
        if len(xyz) == 0:
            return None

        # Transform
        R, t = transform[:3, :3], transform[:3, 3]
        pts = (R @ xyz.T).T + t

        # Height filter
        mask = (pts[:, 2] >= self.min_height) & (pts[:, 2] <= self.max_height)
        pts = pts[mask]
        if len(pts) == 0:
            return None

        # Polar conversion
        x, y = pts[:, 0], pts[:, 1]
        ranges = np.sqrt(x*x + y*y).astype(np.float32)
        angles = np.arctan2(y, x).astype(np.float32)

        # Bounds filter
        angle_max = angle_min + (num_bins - 1) * angle_inc
        mask = (angles >= angle_min) & (angles <= angle_max) & (ranges > 0.1) & (ranges < 12.0)
        ranges, angles = ranges[mask], angles[mask]
        if len(ranges) == 0:
            return None

        # Bin assignment
        scan = np.full(num_bins, np.inf, dtype=np.float32)
        bins = ((angles - angle_min) / angle_inc).astype(np.int32)
        bins = np.clip(bins, 0, num_bins - 1)
        np.minimum.at(scan, bins, ranges)

        return scan


# =============================================================================
# MAIN NODE - ZERO LATENCY
# =============================================================================

class RealtimeScanFusion(Node):
    """
    Zero-latency scan fusion with IMU motion compensation.
    """

    def __init__(self):
        super().__init__('scan_fusion_realtime')

        # Parameters
        self.declare_parameter('scan_topic', '/scan_filtered')
        self.declare_parameter('output_topic', '/scan_fused')
        self.declare_parameter('imu_topic', '/imu')
        self.declare_parameter('laser_frame', 'laser')
        self.declare_parameter('min_height', 0.10)
        self.declare_parameter('max_height', 1.80)
        self.declare_parameter('enable_motion_compensation', True)

        self.declare_parameter('front_camera.enabled', True)
        self.declare_parameter('front_camera.topic', '/camera/depth/color/points')
        self.declare_parameter('front_camera.frame', 'camera_depth_optical_frame')
        self.declare_parameter('front_camera.min_depth', 0.30)
        self.declare_parameter('front_camera.max_depth', 5.0)

        self.declare_parameter('left_camera.enabled', True)
        self.declare_parameter('left_camera.topic', '/mapping_camera/depth/color/points')
        self.declare_parameter('left_camera.frame', 'mapping_camera_depth_optical_frame')
        self.declare_parameter('left_camera.min_depth', 0.30)
        self.declare_parameter('left_camera.max_depth', 5.0)

        self.declare_parameter('right_camera.enabled', True)
        self.declare_parameter('right_camera.topic', '/right_camera/depth/color/points')
        self.declare_parameter('right_camera.frame', 'right_camera_depth_optical_frame')
        self.declare_parameter('right_camera.min_depth', 0.30)
        self.declare_parameter('right_camera.max_depth', 5.0)

        self.declare_parameter('verbose', False)

        # Load params
        self.laser_frame = self.get_parameter('laser_frame').value
        self.min_height = self.get_parameter('min_height').value
        self.max_height = self.get_parameter('max_height').value
        self.enable_motion_comp = self.get_parameter('enable_motion_compensation').value
        self.verbose = self.get_parameter('verbose').value

        # Cameras
        self.cameras = [
            CameraConfig(
                name='front',
                topic=self.get_parameter('front_camera.topic').value,
                frame=self.get_parameter('front_camera.frame').value,
                enabled=self.get_parameter('front_camera.enabled').value,
                min_depth=self.get_parameter('front_camera.min_depth').value,
                max_depth=self.get_parameter('front_camera.max_depth').value,
            ),
            CameraConfig(
                name='left',
                topic=self.get_parameter('left_camera.topic').value,
                frame=self.get_parameter('left_camera.frame').value,
                enabled=self.get_parameter('left_camera.enabled').value,
                min_depth=self.get_parameter('left_camera.min_depth').value,
                max_depth=self.get_parameter('left_camera.max_depth').value,
            ),
            CameraConfig(
                name='right',
                topic=self.get_parameter('right_camera.topic').value,
                frame=self.get_parameter('right_camera.frame').value,
                enabled=self.get_parameter('right_camera.enabled').value,
                min_depth=self.get_parameter('right_camera.min_depth').value,
                max_depth=self.get_parameter('right_camera.max_depth').value,
            ),
        ]

        # TF
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=5.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_cache: Dict[str, np.ndarray] = {}

        # Data storage (lock-free for speed - single writer per topic)
        self.camera_clouds: Dict[str, Optional[PointCloud2]] = {c.name: None for c in self.cameras}
        self.latest_imu: Optional[Imu] = None

        # Processing components
        self.footprint = FastFootprintFilter()
        self.motion_comp = MotionCompensator()
        self.pc_converter = FastPCConverter(self.min_height, self.max_height)
        self.gap_filler = GapFiller(max_gap_size=5, max_range_diff=0.5)
        self.spatial_filter = SpatialConsistencyFilter(window_size=3, max_deviation=0.3)

        # QoS - best effort for sensors
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Subscribers
        self.create_subscription(
            LaserScan, self.get_parameter('scan_topic').value,
            self._scan_cb, sensor_qos)

        self.create_subscription(
            Imu, self.get_parameter('imu_topic').value,
            self._imu_cb, sensor_qos)

        for cam in self.cameras:
            if cam.enabled:
                self.create_subscription(
                    PointCloud2, cam.topic,
                    lambda msg, c=cam: self._camera_cb(msg, c),
                    sensor_qos)

        # Publishers
        self.pub_fused = self.create_publisher(
            LaserScan, self.get_parameter('output_topic').value, 10)
        self.pub_lidar = self.create_publisher(LaserScan, '/scan_lidar_only', 10)
        self.pub_front = self.create_publisher(LaserScan, '/scan_front_camera', 10)
        self.pub_left = self.create_publisher(LaserScan, '/scan_left_camera', 10)
        self.pub_right = self.create_publisher(LaserScan, '/scan_right_camera', 10)

        # Stats
        self.frame_count = 0
        self.total_time = 0.0

        self._log_startup()

    def _log_startup(self):
        self.get_logger().info('=' * 60)
        self.get_logger().info('ZERO-LATENCY SCAN FUSION')
        self.get_logger().info('=' * 60)
        self.get_logger().info(f'Motion compensation: {self.enable_motion_comp}')
        self.get_logger().info(f'IMU topic: {self.get_parameter("imu_topic").value}')
        self.get_logger().info('NO temporal filtering - direct passthrough')
        self.get_logger().info('=' * 60)

    def _imu_cb(self, msg: Imu):
        """Store latest IMU and update motion compensator."""
        self.latest_imu = msg
        # Angular velocity Z in base_link frame
        self.motion_comp.update_imu(msg.angular_velocity.z)

    def _camera_cb(self, msg: PointCloud2, cam: CameraConfig):
        """Store latest camera cloud."""
        self.camera_clouds[cam.name] = msg

    def _get_tf(self, source: str) -> Optional[np.ndarray]:
        """Get cached transform."""
        key = f'{source}_to_{self.laser_frame}'
        if key in self.tf_cache:
            return self.tf_cache[key]

        try:
            tf = self.tf_buffer.lookup_transform(
                self.laser_frame, source, Time(), Duration(seconds=0.5))
            t = tf.transform.translation
            q = tf.transform.rotation

            # Quaternion to matrix
            n = 1.0 / np.sqrt(q.x**2 + q.y**2 + q.z**2 + q.w**2 + 1e-10)
            qx, qy, qz, qw = q.x*n, q.y*n, q.z*n, q.w*n
            R = np.array([
                [1-2*(qy**2+qz**2), 2*(qx*qy-qw*qz), 2*(qx*qz+qw*qy)],
                [2*(qx*qy+qw*qz), 1-2*(qx**2+qz**2), 2*(qy*qz-qw*qx)],
                [2*(qx*qz-qw*qy), 2*(qy*qz+qw*qx), 1-2*(qx**2+qy**2)]
            ], dtype=np.float32)
            T = np.eye(4, dtype=np.float32)
            T[:3, :3] = R
            T[:3, 3] = [t.x, t.y, t.z]
            self.tf_cache[key] = T
            return T
        except TransformException:
            return None

    def _make_scan(self, ranges: np.ndarray, template: LaserScan) -> LaserScan:
        """Create scan message."""
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
        return msg

    def _scan_cb(self, msg: LaserScan):
        """Main callback - ZERO LATENCY path."""
        t0 = time.perf_counter()

        num_bins = len(msg.ranges)
        angle_min = msg.angle_min
        angle_inc = msg.angle_increment

        # =====================================================================
        # 1. LIDAR - Direct passthrough with footprint filter only
        # =====================================================================
        lidar = np.array(msg.ranges, dtype=np.float32)
        lidar = self.footprint.filter(lidar, angle_min, angle_inc)

        # Publish LiDAR immediately
        self.pub_lidar.publish(self._make_scan(lidar, msg))

        # =====================================================================
        # 2. CAMERAS - Direct conversion, no buffering
        # =====================================================================
        cam_pubs = {'front': self.pub_front, 'left': self.pub_left, 'right': self.pub_right}

        # Start with LiDAR
        fused = lidar.copy()

        for cam in self.cameras:
            if not cam.enabled:
                continue

            cloud = self.camera_clouds.get(cam.name)
            if cloud is None:
                continue

            tf = self._get_tf(cam.frame)
            if tf is None:
                continue

            # Convert directly - no temporal filtering
            cam_scan = self.pc_converter.convert(
                cloud, tf, cam, num_bins, angle_min, angle_inc)

            if cam_scan is None:
                continue

            # Apply footprint filter
            cam_scan = self.footprint.filter(cam_scan, angle_min, angle_inc)

            # Publish camera scan immediately
            cam_pubs[cam.name].publish(self._make_scan(cam_scan, msg))

            # Fuse: take minimum (nearest point)
            fused = np.minimum(fused, cam_scan)

        # =====================================================================
        # 3. POST-PROCESSING: Gap filling + Spatial consistency
        # =====================================================================

        # Fill small gaps with linear interpolation
        fused = self.gap_filler.fill(fused)

        # Enforce spatial consistency (remove outliers)
        fused = self.spatial_filter.filter(fused)

        # Publish final fused scan
        self.pub_fused.publish(self._make_scan(fused, msg))

        # Stats
        elapsed = (time.perf_counter() - t0) * 1000
        self.frame_count += 1
        self.total_time += elapsed

        if self.verbose and self.frame_count % 100 == 0:
            avg = self.total_time / self.frame_count
            omega = self.motion_comp.last_angular_vel_z
            self.get_logger().info(
                f'Frame {self.frame_count} | {elapsed:.2f}ms (avg {avg:.2f}ms) | '
                f'ω_z={omega:.3f} rad/s'
            )


def main(args=None):
    rclpy.init(args=args)
    node = RealtimeScanFusion()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
