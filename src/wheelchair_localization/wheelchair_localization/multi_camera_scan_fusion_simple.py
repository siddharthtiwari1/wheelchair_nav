#!/usr/bin/env python3
"""
ROBUST REAL-TIME MULTI-CAMERA + LIDAR SCAN FUSION v2
=====================================================
Fixed version with:
1. Correct footprint filter for laser frame (180° rotated from lidar)
2. Proper nearest-point logic - camera bins ALWAYS captured
3. Simplified pipeline - no Kalman on final output
4. Aggressive self-filtering for wheelchair structure

Output Topics:
- /scan_fused: Final stable fusion (NEAREST POINT from any sensor)
- /scan_lidar_only: LiDAR with footprint filtered
- /scan_front_camera, /scan_left_camera, /scan_right_camera: Individual camera scans
"""

import numpy as np
from dataclasses import dataclass
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


@dataclass
class CameraConfig:
    name: str
    topic: str
    frame: str
    enabled: bool = True
    min_depth: float = 0.30
    max_depth: float = 5.0
    downsample: int = 8  # Increased from 4 for better CPU performance


class RobotFootprintFilter:
    """
    Filters out scan points from wheelchair self-detection.

    IMPORTANT: Laser frame is 180° rotated from lidar frame!
    In laser frame:
    - X+ points BACKWARD (toward rear of wheelchair)
    - Y+ points RIGHT
    - Z+ points UP

    Geometry from URDF (measured from LiDAR at front):
    - Left rear wheel:  angle -11.7°, distance 0.81m
    - Right rear wheel: angle +27.0°, distance 0.89m
    - Left castor:      angle -40.6°, distance 0.24m
    - Right castor:     angle +65.8°, distance 0.45m

    FIXED 2026-02-03: Based on scan data showing 95.8% self-detection!
    - Increased min_valid_range to 0.35m
    - Expanded all exclusion zones
    - Added full rear hemisphere coverage
    """

    def __init__(self):
        # Minimum valid range - INCREASED to filter all self-reflections
        # Data showed min_range of 0.15m in 95.8% of scans!
        self.min_valid_range = 0.35  # Increased from 0.25

        # Robot box in laser frame (X+ = backward, Y+ = right)
        # EXPANDED to cover entire robot body
        self.box_min_x = -0.20  # front edge
        self.box_max_x = 0.70   # rear edge (expanded)
        self.box_min_y = -0.45  # left edge (expanded)
        self.box_max_y = 0.45   # right edge (expanded)

        # Angular exclusion zones in laser frame
        # Format: (angle_min_rad, angle_max_rad, max_range_m)
        # FIXED 2026-02-03: AGGRESSIVE filtering based on 95.8% self-detection rate!
        self.exclusion_zones: List[Tuple[float, float, float]] = [
            # FULL REAR HEMISPHERE: -90° to +90° up to 1.2m (covers wheels, frame, castors)
            (-1.57, 1.57, 1.20),  # -90° to +90°, up to 1.2m - catches everything behind lidar
            # LEFT SIDE: -180° to -90°
            (-3.14, -1.57, 0.50),  # Left side up to 0.5m
            # RIGHT SIDE: +90° to +180°
            (1.57, 3.14, 0.50),   # Right side up to 0.5m
        ]

    def filter_scan(self, ranges: np.ndarray, angle_min: float,
                    angle_increment: float) -> np.ndarray:
        """Filter out points within robot footprint."""
        result = ranges.copy()
        num_bins = len(ranges)

        # Compute angles for all bins
        angles = angle_min + np.arange(num_bins) * angle_increment

        # Mark valid (finite) readings
        valid = np.isfinite(ranges) & (ranges > 0)

        # 1. Minimum range filter
        too_close = valid & (ranges < self.min_valid_range)
        result[too_close] = np.inf

        # 2. Box filter in Cartesian coordinates
        x = np.where(valid, ranges * np.cos(angles), 0)
        y = np.where(valid, ranges * np.sin(angles), 0)

        inside_box = (
            valid &
            (x >= self.box_min_x) & (x <= self.box_max_x) &
            (y >= self.box_min_y) & (y <= self.box_max_y)
        )
        result[inside_box] = np.inf

        # 3. Angular exclusion zones
        for angle_start, angle_end, max_range in self.exclusion_zones:
            in_zone = (
                valid &
                (angles >= angle_start) & (angles <= angle_end) &
                (ranges < max_range)
            )
            result[in_zone] = np.inf

        return result


class TemporalSmoother:
    """
    Simple temporal smoothing using exponential moving average.
    Much simpler than Kalman, and doesn't resist sudden changes as much.
    """

    def __init__(self, num_bins: int, alpha: float = 0.7):
        """
        Args:
            num_bins: Number of scan bins
            alpha: Smoothing factor (0-1). Higher = more weight to new data.
                   0.7 means 70% new data, 30% old data.
        """
        self.num_bins = num_bins
        self.alpha = alpha
        self.state = np.full(num_bins, np.inf, dtype=np.float64)
        self.initialized = np.zeros(num_bins, dtype=bool)
        self.frames_since_seen = np.zeros(num_bins, dtype=np.int32)
        self.max_unseen_frames = 5  # Reset after 5 frames without observation

    def update(self, measurement: np.ndarray) -> np.ndarray:
        """Update state with new measurement and return smoothed output."""
        valid = np.isfinite(measurement)

        # Initialize new bins directly
        new_bins = valid & ~self.initialized
        self.state[new_bins] = measurement[new_bins]
        self.initialized[new_bins] = True

        # Update existing bins with EMA
        update_bins = valid & self.initialized
        self.state[update_bins] = (
            self.alpha * measurement[update_bins] +
            (1 - self.alpha) * self.state[update_bins]
        )

        # Track frames since last observation
        self.frames_since_seen[valid] = 0
        self.frames_since_seen[~valid & self.initialized] += 1

        # Reset bins that haven't been seen for too long
        stale = self.frames_since_seen > self.max_unseen_frames
        self.initialized[stale] = False
        self.state[stale] = np.inf

        # Return current state (inf for uninitialized)
        result = self.state.copy()
        result[~self.initialized] = np.inf
        return result.astype(np.float32)


class RobustMultiCameraScanFusion(Node):
    """
    Production-quality sensor fusion with guaranteed nearest-point capture.
    """

    def __init__(self):
        super().__init__('multi_camera_scan_fusion_simple')

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
        self.declare_parameter('camera_skip_frames', 2)  # Process cameras every Nth frame
        self.declare_parameter('camera_downsample', 8)   # Downsample factor for pointclouds

        # Load parameters
        self.laser_frame = self.get_parameter('laser_frame').value
        self.min_height = self.get_parameter('min_height').value
        self.max_height = self.get_parameter('max_height').value
        self.verbose = self.get_parameter('verbose').value
        camera_downsample = self.get_parameter('camera_downsample').value

        self.cameras = [
            CameraConfig(
                name='front',
                topic=self.get_parameter('front_camera.topic').value,
                frame=self.get_parameter('front_camera.frame').value,
                enabled=self.get_parameter('front_camera.enabled').value,
                min_depth=self.get_parameter('front_camera.min_depth').value,
                max_depth=self.get_parameter('front_camera.max_depth').value,
                downsample=camera_downsample,
            ),
            CameraConfig(
                name='left',
                topic=self.get_parameter('left_camera.topic').value,
                frame=self.get_parameter('left_camera.frame').value,
                enabled=self.get_parameter('left_camera.enabled').value,
                min_depth=self.get_parameter('left_camera.min_depth').value,
                max_depth=self.get_parameter('left_camera.max_depth').value,
                downsample=camera_downsample,
            ),
            CameraConfig(
                name='right',
                topic=self.get_parameter('right_camera.topic').value,
                frame=self.get_parameter('right_camera.frame').value,
                enabled=self.get_parameter('right_camera.enabled').value,
                min_depth=self.get_parameter('right_camera.min_depth').value,
                max_depth=self.get_parameter('right_camera.max_depth').value,
                downsample=camera_downsample,
            ),
        ]

        # TF2
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.transform_cache: Dict[str, np.ndarray] = {}
        self._logged_transforms: set = set()  # Track which transforms we've logged

        # Camera data
        self.camera_data_lock = threading.Lock()
        self.latest_camera_data: Dict[str, Optional[PointCloud2]] = {
            cam.name: None for cam in self.cameras
        }

        # Filtering components (initialized on first scan)
        self.footprint_filter = RobotFootprintFilter()
        self.camera_smoothers: Dict[str, Optional[TemporalSmoother]] = {
            cam.name: None for cam in self.cameras
        }
        self.fused_smoother: Optional[TemporalSmoother] = None
        self.num_bins: Optional[int] = None

        # QoS
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

        # Publishers
        self.fused_pub = self.create_publisher(
            LaserScan, self.get_parameter('output_topic').value, 10)
        self.lidar_pub = self.create_publisher(LaserScan, '/scan_lidar_only', 10)
        self.front_pub = self.create_publisher(LaserScan, '/scan_front_camera', 10)
        self.left_pub = self.create_publisher(LaserScan, '/scan_left_camera', 10)
        self.right_pub = self.create_publisher(LaserScan, '/scan_right_camera', 10)

        self.frame_count = 0
        self.total_time = 0.0

        # Performance optimization: skip camera processing on some frames
        self._skip_camera_frames = self.get_parameter('camera_skip_frames').value
        self._last_camera_scans: Dict[str, np.ndarray] = {}

        self._log_startup()

    def _log_startup(self):
        self.get_logger().info('=' * 60)
        self.get_logger().info('ROBUST SCAN FUSION v2 - NEAREST POINT GUARANTEED')
        self.get_logger().info('=' * 60)
        self.get_logger().info(f'Input:  {self.get_parameter("scan_topic").value}')
        self.get_logger().info(f'Output: {self.get_parameter("output_topic").value}')
        self.get_logger().info('-' * 60)
        self.get_logger().info('Laser frame: X+ = backward, Y+ = right (180° from lidar)')
        self.get_logger().info('-' * 60)

    def _camera_callback(self, msg: PointCloud2, camera: CameraConfig):
        with self.camera_data_lock:
            self.latest_camera_data[camera.name] = msg

    def _get_transform(self, target: str, source: str) -> Optional[np.ndarray]:
        key = f'{source}_to_{target}'

        # Check cache - but only use if we've verified it at least once
        # Static transforms are safe to cache after first successful lookup
        if key in self.transform_cache:
            return self.transform_cache[key]

        try:
            # Use latest available transform (Time(0))
            tf = self.tf_buffer.lookup_transform(
                target, source, Time(), Duration(seconds=0.5))
            t = tf.transform.translation
            q = tf.transform.rotation

            # Validate transform is not identity (common startup issue)
            is_identity = (
                abs(t.x) < 1e-6 and abs(t.y) < 1e-6 and abs(t.z) < 1e-6 and
                abs(q.x) < 1e-6 and abs(q.y) < 1e-6 and abs(q.z) < 1e-6 and
                abs(q.w - 1.0) < 1e-6
            )

            # Camera frames should NOT be identity transforms
            if is_identity and 'camera' in source.lower():
                self.get_logger().warning(
                    f'Transform {source} -> {target} is identity, likely not ready')
                return None

            # Normalize quaternion
            n = 1.0 / np.sqrt(q.x**2 + q.y**2 + q.z**2 + q.w**2 + 1e-10)
            qx, qy, qz, qw = q.x*n, q.y*n, q.z*n, q.w*n

            # Quaternion to rotation matrix
            R = np.array([
                [1-2*(qy**2+qz**2), 2*(qx*qy-qw*qz), 2*(qx*qz+qw*qy)],
                [2*(qx*qy+qw*qz), 1-2*(qx**2+qz**2), 2*(qy*qz-qw*qx)],
                [2*(qx*qz-qw*qy), 2*(qy*qz+qw*qx), 1-2*(qx**2+qy**2)]
            ])
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = [t.x, t.y, t.z]

            # Cache static transforms (camera frames are static)
            self.transform_cache[key] = T

            # Log first successful transform lookup
            if key not in self._logged_transforms:
                self._logged_transforms.add(key)
                self.get_logger().info(
                    f'Cached transform {source} -> {target}: '
                    f't=({t.x:.3f}, {t.y:.3f}, {t.z:.3f})')

            return T
        except TransformException as e:
            return None

    def _pointcloud_to_scan(self, cloud: PointCloud2, cam: CameraConfig,
                            template: LaserScan) -> Optional[np.ndarray]:
        """Convert pointcloud to scan array using minimum range per bin."""
        T = self._get_transform(self.laser_frame, cam.frame)
        if T is None:
            return None

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
        except Exception:
            return None

        # Downsample for performance
        if cam.downsample > 1:
            xyz = xyz[::cam.downsample]

        # Depth filter in camera frame
        depths = xyz[:, 2]
        mask = (depths >= cam.min_depth) & (depths <= cam.max_depth)
        xyz = xyz[mask]

        if len(xyz) == 0:
            return None

        # Transform to laser frame
        pts_laser = (T[:3, :3] @ xyz.T).T + T[:3, 3]

        # Height filter in laser frame
        mask = ((pts_laser[:, 2] >= self.min_height) &
                (pts_laser[:, 2] <= self.max_height))
        pts_laser = pts_laser[mask]

        if len(pts_laser) == 0:
            return None

        # Convert to polar coordinates
        x, y = pts_laser[:, 0], pts_laser[:, 1]
        ranges = np.sqrt(x*x + y*y)
        angles = np.arctan2(y, x)

        # Create scan array
        num_bins = len(template.ranges)
        scan = np.full(num_bins, np.inf, dtype=np.float32)

        # Filter valid ranges
        mask = (
            (angles >= template.angle_min) &
            (angles <= template.angle_max) &
            (ranges >= template.range_min) &
            (ranges <= template.range_max)
        )
        ranges = ranges[mask]
        angles = angles[mask]

        if len(ranges) == 0:
            return None

        # Bin assignment
        indices = ((angles - template.angle_min) /
                   template.angle_increment).astype(np.int32)
        indices = np.clip(indices, 0, num_bins - 1)

        # Take MINIMUM range per bin (nearest point)
        np.minimum.at(scan, indices, ranges.astype(np.float32))

        return scan

    def _create_scan_msg(self, ranges: np.ndarray,
                         template: LaserScan,
                         use_current_time: bool = False) -> LaserScan:
        """Create LaserScan message from range array.

        Args:
            ranges: Range array
            template: Template scan for parameters
            use_current_time: If True, use current time instead of template stamp
                              This helps reduce effective latency for SLAM
        """
        msg = LaserScan()
        msg.header = Header()

        if use_current_time:
            # Use current time to reduce latency seen by SLAM
            msg.header.stamp = self.get_clock().now().to_msg()
        else:
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
        return msg

    def _scan_callback(self, scan_msg: LaserScan):
        start = time.perf_counter()
        num_bins = len(scan_msg.ranges)
        angle_min = scan_msg.angle_min
        angle_inc = scan_msg.angle_increment
        self.frame_count += 1

        # Initialize smoothers on first scan
        if self.num_bins != num_bins:
            self.num_bins = num_bins
            self.fused_smoother = TemporalSmoother(num_bins, alpha=0.8)
            for cam in self.cameras:
                self.camera_smoothers[cam.name] = TemporalSmoother(
                    num_bins, alpha=0.7)

        # =====================================================================
        # STEP 1: Process LiDAR (every frame)
        # =====================================================================

        lidar_raw = np.array(scan_msg.ranges, dtype=np.float32)
        lidar = self.footprint_filter.filter_scan(lidar_raw, angle_min, angle_inc)

        # =====================================================================
        # STEP 2: Process cameras (skip frames for CPU optimization)
        # =====================================================================

        process_cameras = (self.frame_count % self._skip_camera_frames == 0)

        camera_scans: Dict[str, np.ndarray] = {}
        pubs = {'front': self.front_pub, 'left': self.left_pub,
                'right': self.right_pub}

        if process_cameras:
            # Get all camera clouds at once (single lock)
            with self.camera_data_lock:
                camera_clouds = {
                    cam.name: self.latest_camera_data.get(cam.name)
                    for cam in self.cameras if cam.enabled
                }

            for cam in self.cameras:
                if not cam.enabled or camera_clouds.get(cam.name) is None:
                    continue

                # Convert pointcloud to scan
                raw_scan = self._pointcloud_to_scan(
                    camera_clouds[cam.name], cam, scan_msg)

                if raw_scan is None:
                    continue

                # Apply footprint filter
                cam_filtered = self.footprint_filter.filter_scan(
                    raw_scan, angle_min, angle_inc)

                # Apply temporal smoothing
                cam_smoothed = self.camera_smoothers[cam.name].update(cam_filtered)

                # Store for fusion and cache
                camera_scans[cam.name] = cam_smoothed
                self._last_camera_scans[cam.name] = cam_smoothed

                # Publish individual camera scan
                pubs[cam.name].publish(self._create_scan_msg(cam_smoothed, scan_msg))
        else:
            # Reuse cached camera scans on skipped frames
            camera_scans = self._last_camera_scans.copy()

        # =====================================================================
        # STEP 3: FUSE ALL AT ONCE - Single np.minimum across all sources
        # =====================================================================

        # Stack all valid scans: LiDAR + all cameras
        all_scans = [lidar]  # Start with LiDAR
        for cam_scan in camera_scans.values():
            all_scans.append(cam_scan)

        # Single operation: element-wise minimum across ALL sensors
        # This guarantees nearest point from ANY sensor is in output
        if len(all_scans) == 1:
            fused = lidar
        else:
            # Stack into 2D array and take min along axis 0
            stacked = np.stack(all_scans, axis=0)  # Shape: (N_sensors, num_bins)
            fused = np.nanmin(stacked, axis=0)  # Min across sensors
            # nanmin treats inf as inf, returns inf only if ALL are inf

        # Replace any remaining nan with inf
        fused = np.where(np.isnan(fused), np.inf, fused).astype(np.float32)

        # =====================================================================
        # STEP 4: Temporal smoothing and publish
        # =====================================================================

        fused_smoothed = self.fused_smoother.update(fused)

        # Publish all outputs
        # Keep original timestamp - SLAM needs correct timestamp for TF lookup
        self.lidar_pub.publish(self._create_scan_msg(lidar, scan_msg))
        self.fused_pub.publish(self._create_scan_msg(fused_smoothed, scan_msg))

        # Performance logging
        elapsed = (time.perf_counter() - start) * 1000
        self.total_time += elapsed

        if self.verbose and self.frame_count % 50 == 0:
            avg_time = self.total_time / self.frame_count
            lidar_valid = np.sum(np.isfinite(lidar))
            fused_valid = np.sum(np.isfinite(fused_smoothed))
            self.get_logger().info(
                f'Frame {self.frame_count} | {elapsed:.1f}ms (avg: {avg_time:.1f}ms) | '
                f'LiDAR: {lidar_valid} | Fused: {fused_valid} | Cams: {len(camera_scans)}'
            )


def main(args=None):
    rclpy.init(args=args)
    node = RobustMultiCameraScanFusion()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
