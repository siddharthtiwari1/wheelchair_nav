#!/usr/bin/env python3
"""
OPTIMIZED MULTI-SENSOR SCAN FUSION - Production Grade
======================================================
Advanced algorithms & data structures for real-time performance.

Optimizations:
1. Pre-computed lookup tables (angles, sin/cos, bin indices)
2. Vectorized NumPy operations (zero Python loops in hot path)
3. Memory pre-allocation (no allocations during fusion)
4. Cache-friendly memory access patterns
5. Aggressive footprint masking with polar + Cartesian filters

Author: Wheelchair Navigation Project
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

from sensor_msgs.msg import LaserScan, PointCloud2
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
    downsample: int = 8  # Increased from 4 to reduce CPU


@dataclass
class FootprintConfig:
    """
    Wheelchair footprint in LASER frame.

    LASER FRAME ORIENTATION (180° rotated from lidar):
    - X+ points BACKWARD (toward rear wheels)
    - Y+ points RIGHT
    - Z+ points UP

    Coordinate system:
                    X+ (backward/rear)
                    ↑
                    |
         Y- (left) ←───→ Y+ (right)
                    |
                    ↓
                    X- (forward/front, where LiDAR is)
    """
    # Minimum valid range - self-reflection cutoff
    min_range: float = 0.18

    # Box filter bounds (meters)
    # LiDAR INVERTED: X- = rear (wheels), X+ = front
    box_x_min: float = -0.70   # Rear edge (wheels)
    box_x_max: float = 0.15    # Front edge (near laser)
    box_y_min: float = -0.45   # Left edge
    box_y_max: float = 0.45    # Right edge

    # Angular exclusion zones: (angle_min, angle_max, range_max)
    # LiDAR INVERTED: 0° = FRONT, ±180° = REAR, +90° = LEFT, -90° = RIGHT
    exclusion_zones: List[Tuple[float, float, float]] = field(default_factory=lambda: [
        # ============================================================
        # REAR AREA (around ±180°) - WHEELS + SEAT BACK
        # ============================================================

        # LEFT REAR WHEEL (135° to 180°)
        (2.35, 3.14159, 1.10),    # Left wheel area

        # RIGHT REAR WHEEL (-180° to -135°)
        (-3.14159, -2.35, 1.10),  # Right wheel area

        # DIRECT REAR - seat back (150° to -150° wrapping through ±180°)
        (2.60, 3.14159, 0.90),    # Seat back left
        (-3.14159, -2.60, 0.90),  # Seat back right

        # ============================================================
        # SIDE ARMRESTS
        # ============================================================

        # LEFT ARMREST (around +90°)
        (1.20, 1.95, 0.55),      # Left armrest

        # RIGHT ARMREST (around -90°)
        (-1.95, -1.20, 0.55),    # Right armrest

        # ============================================================
        # FRONT CASTERS (around 0°) - small zone
        # ============================================================
        (-0.50, 0.50, 0.35),     # Front casters near laser
    ])


# =============================================================================
# PRE-COMPUTED LOOKUP TABLES
# =============================================================================

class ScanGeometryLUT:
    """
    Pre-computed lookup tables for scan geometry.
    Eliminates repeated trigonometry calculations.
    """

    def __init__(self, num_bins: int, angle_min: float, angle_increment: float):
        self.num_bins = num_bins
        self.angle_min = angle_min
        self.angle_increment = angle_increment

        # Pre-compute angles for all bins
        self.angles = angle_min + np.arange(num_bins, dtype=np.float32) * angle_increment

        # Pre-compute sin/cos for polar→Cartesian conversion
        self.cos_angles = np.cos(self.angles).astype(np.float32)
        self.sin_angles = np.sin(self.angles).astype(np.float32)

        # Pre-compute bin indices for angle→bin mapping
        self.angle_max = angle_min + (num_bins - 1) * angle_increment

    def ranges_to_cartesian(self, ranges: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Convert ranges to X,Y using pre-computed sin/cos."""
        x = ranges * self.cos_angles
        y = ranges * self.sin_angles
        return x, y

    def angles_to_bins(self, angles: np.ndarray) -> np.ndarray:
        """Convert angles to bin indices."""
        bins = ((angles - self.angle_min) / self.angle_increment).astype(np.int32)
        return np.clip(bins, 0, self.num_bins - 1)


# =============================================================================
# OPTIMIZED FOOTPRINT FILTER
# =============================================================================

class OptimizedFootprintFilter:
    """
    High-performance footprint filter using pre-computed masks.
    """

    def __init__(self, config: FootprintConfig):
        self.config = config
        self.lut: Optional[ScanGeometryLUT] = None

        # Pre-computed exclusion mask (computed on first scan)
        self._exclusion_mask: Optional[np.ndarray] = None
        self._mask_ranges: Optional[np.ndarray] = None

    def _build_exclusion_mask(self, lut: ScanGeometryLUT):
        """Build pre-computed exclusion mask for angular zones."""
        num_bins = lut.num_bins
        angles = lut.angles

        # For each bin, compute the maximum range that should be excluded
        # Start with minimum valid range
        mask_ranges = np.full(num_bins, self.config.min_range, dtype=np.float32)

        # Apply each exclusion zone
        for angle_min, angle_max, range_max in self.config.exclusion_zones:
            in_zone = (angles >= angle_min) & (angles <= angle_max)
            # Take maximum exclusion range for overlapping zones
            mask_ranges[in_zone] = np.maximum(mask_ranges[in_zone], range_max)

        self._mask_ranges = mask_ranges
        self._exclusion_mask = mask_ranges > self.config.min_range

    def filter(self, ranges: np.ndarray, lut: ScanGeometryLUT) -> np.ndarray:
        """
        Apply footprint filter using vectorized operations.

        Returns filtered ranges with self-detections set to inf.
        """
        # Build mask on first call or if LUT changed
        if self.lut is not lut:
            self.lut = lut
            self._build_exclusion_mask(lut)

        # Start with copy
        result = ranges.copy()

        # Identify valid readings
        valid = np.isfinite(ranges) & (ranges > 0)

        # 1. ANGULAR EXCLUSION (vectorized)
        # Points within exclusion zones AND closer than zone's max range
        angular_exclude = valid & (ranges < self._mask_ranges)
        result[angular_exclude] = np.inf

        # 2. BOX FILTER (Cartesian)
        # Only check remaining valid points
        still_valid = np.isfinite(result) & (result > 0)
        if np.any(still_valid):
            x, y = lut.ranges_to_cartesian(result)

            inside_box = (
                still_valid &
                (x >= self.config.box_x_min) & (x <= self.config.box_x_max) &
                (y >= self.config.box_y_min) & (y <= self.config.box_y_max)
            )
            result[inside_box] = np.inf

        return result


# =============================================================================
# OPTIMIZED TEMPORAL FILTER
# =============================================================================

class OptimizedTemporalFilter:
    """
    Lightweight IIR (Infinite Impulse Response) temporal filter.
    Much faster than median - uses exponential smoothing.
    CPU-friendly: no buffer copies, no median computation.
    """

    def __init__(self, num_bins: int, buffer_size: int = 3):
        self.num_bins = num_bins
        # IIR smoothing factor (0.3 = 30% new, 70% old)
        self.alpha = 0.4

        # Single smoothed output (pre-allocated)
        self._output = np.full(num_bins, np.inf, dtype=np.float32)
        self._initialized = False

    def update(self, measurement: np.ndarray) -> np.ndarray:
        """
        IIR filter: output = alpha * new + (1-alpha) * old
        Handles inf values properly.
        """
        if not self._initialized:
            np.copyto(self._output, measurement)
            self._initialized = True
            return self._output

        # Only smooth where both are finite
        both_valid = np.isfinite(measurement) & np.isfinite(self._output)
        new_only = np.isfinite(measurement) & ~np.isfinite(self._output)

        # IIR update where both valid
        self._output[both_valid] = (
            self.alpha * measurement[both_valid] +
            (1 - self.alpha) * self._output[both_valid]
        )

        # New measurement where only new is valid
        self._output[new_only] = measurement[new_only]

        return self._output


# =============================================================================
# OPTIMIZED POINTCLOUD TO SCAN CONVERTER
# =============================================================================

class PointCloudToScanConverter:
    """
    Efficient pointcloud to 2D scan conversion.
    """

    def __init__(self, min_height: float, max_height: float):
        self.min_height = min_height
        self.max_height = max_height

        # Pre-allocated arrays for intermediate results
        self._ranges = None
        self._angles = None
        self._scan = None

    def convert(self, cloud: PointCloud2, transform: np.ndarray,
                cam_config: CameraConfig, lut: ScanGeometryLUT) -> Optional[np.ndarray]:
        """
        Convert pointcloud to scan using vectorized operations.
        """
        try:
            pts = point_cloud2.read_points_numpy(
                cloud, field_names=("x", "y", "z"), skip_nans=True)
            if len(pts) == 0:
                return None

            # Handle structured array
            if pts.dtype.names:
                xyz = np.column_stack([pts['x'], pts['y'], pts['z']])
            else:
                xyz = pts[:, :3] if pts.ndim == 2 else None

            if xyz is None or len(xyz) == 0:
                return None
        except Exception:
            return None

        # Downsample
        if cam_config.downsample > 1:
            xyz = xyz[::cam_config.downsample]

        # Depth filter (in camera frame, Z is depth)
        depth_mask = (xyz[:, 2] >= cam_config.min_depth) & (xyz[:, 2] <= cam_config.max_depth)
        xyz = xyz[depth_mask]

        if len(xyz) == 0:
            return None

        # Transform to laser frame (vectorized matrix multiplication)
        R = transform[:3, :3]
        t = transform[:3, 3]
        pts_laser = (R @ xyz.T).T + t

        # Height filter
        height_mask = (pts_laser[:, 2] >= self.min_height) & (pts_laser[:, 2] <= self.max_height)
        pts_laser = pts_laser[height_mask]

        if len(pts_laser) == 0:
            return None

        # Convert to polar coordinates
        x, y = pts_laser[:, 0], pts_laser[:, 1]
        ranges = np.sqrt(x*x + y*y).astype(np.float32)
        angles = np.arctan2(y, x).astype(np.float32)

        # Filter by scan bounds
        valid_mask = (
            (angles >= lut.angle_min) & (angles <= lut.angle_max) &
            (ranges > 0.1) & (ranges < 12.0)
        )
        ranges = ranges[valid_mask]
        angles = angles[valid_mask]

        if len(ranges) == 0:
            return None

        # Create scan with minimum range per bin
        scan = np.full(lut.num_bins, np.inf, dtype=np.float32)
        bin_indices = lut.angles_to_bins(angles)

        # Vectorized minimum assignment
        np.minimum.at(scan, bin_indices, ranges)

        return scan


# =============================================================================
# MAIN NODE
# =============================================================================

class OptimizedScanFusion(Node):
    """
    Production-grade multi-sensor scan fusion.
    """

    def __init__(self):
        super().__init__('scan_fusion_optimized')

        # Parameters
        self.declare_parameter('scan_topic', '/scan_filtered')
        self.declare_parameter('output_topic', '/scan_fused')
        self.declare_parameter('laser_frame', 'laser')
        self.declare_parameter('min_height', 0.10)
        self.declare_parameter('max_height', 1.80)

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

        # Load parameters
        self.laser_frame = self.get_parameter('laser_frame').value
        self.min_height = self.get_parameter('min_height').value
        self.max_height = self.get_parameter('max_height').value
        self.verbose = self.get_parameter('verbose').value

        # Camera configs
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

        # TF2
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.transform_cache: Dict[str, np.ndarray] = {}

        # Thread-safe camera data storage
        self.camera_lock = threading.Lock()
        self.camera_clouds: Dict[str, Optional[PointCloud2]] = {
            cam.name: None for cam in self.cameras
        }

        # Processing components (initialized on first scan)
        self.lut: Optional[ScanGeometryLUT] = None
        self.footprint_filter = OptimizedFootprintFilter(FootprintConfig())
        self.pc_converter = PointCloudToScanConverter(self.min_height, self.max_height)

        self.lidar_temporal: Optional[OptimizedTemporalFilter] = None
        self.camera_temporals: Dict[str, OptimizedTemporalFilter] = {}
        self.fused_temporal: Optional[OptimizedTemporalFilter] = None

        # Pre-allocated fusion array
        self._fusion_stack: Optional[np.ndarray] = None
        self._num_bins: Optional[int] = None

        # QoS for sensors
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Subscribers
        self.scan_sub = self.create_subscription(
            LaserScan,
            self.get_parameter('scan_topic').value,
            self._scan_callback,
            sensor_qos
        )

        for cam in self.cameras:
            if cam.enabled:
                self.create_subscription(
                    PointCloud2,
                    cam.topic,
                    lambda msg, c=cam: self._camera_callback(msg, c),
                    sensor_qos
                )

        # Publishers - only fused scan by default (debug pubs disabled for CPU)
        self.fused_pub = self.create_publisher(
            LaserScan, self.get_parameter('output_topic').value, 10)

        # Debug publishers only if verbose (saves significant CPU)
        self.lidar_pub = None
        self.front_pub = None
        self.left_pub = None
        self.right_pub = None
        if self.verbose:
            self.lidar_pub = self.create_publisher(LaserScan, '/scan_lidar_only', 10)
            self.front_pub = self.create_publisher(LaserScan, '/scan_front_camera', 10)
            self.left_pub = self.create_publisher(LaserScan, '/scan_left_camera', 10)
            self.right_pub = self.create_publisher(LaserScan, '/scan_right_camera', 10)

        # Performance tracking
        self.frame_count = 0

        self._log_startup()

    def _log_startup(self):
        self.get_logger().info('=' * 60)
        self.get_logger().info('OPTIMIZED SCAN FUSION - Production Grade')
        self.get_logger().info('=' * 60)
        self.get_logger().info(f'Input:  {self.get_parameter("scan_topic").value}')
        self.get_logger().info(f'Output: {self.get_parameter("output_topic").value}')
        self.get_logger().info('Features:')
        self.get_logger().info('  - Pre-computed LUTs (angles, sin/cos)')
        self.get_logger().info('  - Vectorized NumPy (zero Python loops)')
        self.get_logger().info('  - Ring-buffer temporal filtering')
        self.get_logger().info('  - Aggressive footprint masking')
        self.get_logger().info('=' * 60)

    def _camera_callback(self, msg: PointCloud2, camera: CameraConfig):
        with self.camera_lock:
            self.camera_clouds[camera.name] = msg

    def _get_transform(self, source: str) -> Optional[np.ndarray]:
        """Get cached transform from source frame to laser frame."""
        key = f'{source}_to_{self.laser_frame}'
        if key in self.transform_cache:
            return self.transform_cache[key]

        try:
            tf = self.tf_buffer.lookup_transform(
                self.laser_frame, source, Time(), Duration(seconds=1.0))
            t = tf.transform.translation
            q = tf.transform.rotation

            # Quaternion to rotation matrix
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

            self.transform_cache[key] = T
            return T
        except TransformException:
            return None

    def _create_scan_msg(self, ranges: np.ndarray, template: LaserScan) -> LaserScan:
        """Create LaserScan message."""
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

    def _process_camera(self, cam: CameraConfig, cloud: PointCloud2,
                         scan_msg: LaserScan) -> Tuple[str, Optional[np.ndarray]]:
        """Process single camera - designed for parallel execution."""
        if cloud is None:
            return (cam.name, None)

        transform = self._get_transform(cam.frame)
        if transform is None:
            return (cam.name, None)

        # Convert pointcloud to scan
        cam_scan = self.pc_converter.convert(cloud, transform, cam, self.lut)
        if cam_scan is None:
            return (cam.name, None)

        # Apply footprint filter
        cam_filtered = self.footprint_filter.filter(cam_scan, self.lut)

        # Temporal smoothing
        cam_smoothed = self.camera_temporals[cam.name].update(cam_filtered)

        return (cam.name, cam_smoothed)

    def _process_lidar(self, scan_msg: LaserScan) -> np.ndarray:
        """Process LiDAR - designed for parallel execution."""
        lidar_raw = np.array(scan_msg.ranges, dtype=np.float32)
        lidar_filtered = self.footprint_filter.filter(lidar_raw, self.lut)
        lidar_smoothed = self.lidar_temporal.update(lidar_filtered)
        return lidar_smoothed

    def _scan_callback(self, scan_msg: LaserScan):
        t_start = time.perf_counter()

        num_bins = len(scan_msg.ranges)

        # Initialize on first scan or if scan size changes
        if self._num_bins != num_bins:
            self._num_bins = num_bins
            self.lut = ScanGeometryLUT(
                num_bins, scan_msg.angle_min, scan_msg.angle_increment)

            self.lidar_temporal = OptimizedTemporalFilter(num_bins, buffer_size=3)
            self.fused_temporal = OptimizedTemporalFilter(num_bins, buffer_size=3)

            for cam in self.cameras:
                self.camera_temporals[cam.name] = OptimizedTemporalFilter(
                    num_bins, buffer_size=3)

            # Pre-allocate fusion stack (max 4 sources: lidar + 3 cameras)
            self._fusion_stack = np.full((4, num_bins), np.inf, dtype=np.float32)

            # Pre-cache all transforms at initialization
            for cam in self.cameras:
                if cam.enabled:
                    self._get_transform(cam.frame)

            self.get_logger().info(f'Initialized with {num_bins} bins, transforms cached')

        # =====================================================================
        # GET CAMERA DATA (single lock, minimal critical section)
        # =====================================================================
        with self.camera_lock:
            clouds = {name: cloud for name, cloud in self.camera_clouds.items()}

        # =====================================================================
        # PROCESS LIDAR (simple, no threading)
        # =====================================================================
        lidar_smoothed = self._process_lidar(scan_msg)
        self._fusion_stack[0] = lidar_smoothed

        # =====================================================================
        # PROCESS CAMERAS SEQUENTIALLY (lower CPU than threading)
        # =====================================================================
        stack_idx = 1
        camera_results: Dict[str, np.ndarray] = {}

        for cam in self.cameras:
            if cam.enabled and clouds.get(cam.name) is not None:
                cam_name, cam_smoothed = self._process_camera(cam, clouds[cam.name], scan_msg)
                if cam_smoothed is not None:
                    camera_results[cam_name] = cam_smoothed
                    self._fusion_stack[stack_idx] = cam_smoothed
                    stack_idx += 1

        # =====================================================================
        # FUSE - Single vectorized minimum across ALL sources
        # =====================================================================
        if stack_idx == 1:
            fused = lidar_smoothed
        else:
            fused = np.min(self._fusion_stack[:stack_idx], axis=0)

        # Final temporal smoothing
        fused_smoothed = self.fused_temporal.update(fused)

        # =====================================================================
        # PUBLISH - Only fused scan (debug pubs only in verbose mode)
        # =====================================================================
        self.fused_pub.publish(self._create_scan_msg(fused_smoothed, scan_msg))

        # Debug publishers only if verbose (major CPU savings)
        if self.verbose and self.lidar_pub:
            self.lidar_pub.publish(self._create_scan_msg(lidar_smoothed, scan_msg))
            camera_pubs = {'front': self.front_pub, 'left': self.left_pub, 'right': self.right_pub}
            for cam_name, cam_smoothed in camera_results.items():
                if camera_pubs.get(cam_name):
                    camera_pubs[cam_name].publish(self._create_scan_msg(cam_smoothed, scan_msg))

        # Performance tracking (minimal)
        self.frame_count += 1
        if self.verbose and self.frame_count % 100 == 0:
            elapsed = (time.perf_counter() - t_start) * 1000
            self.get_logger().info(f'Frame {self.frame_count} | {elapsed:.2f}ms')


def main(args=None):
    rclpy.init(args=args)
    node = OptimizedScanFusion()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
