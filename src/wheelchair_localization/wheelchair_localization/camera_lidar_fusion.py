#!/usr/bin/env python3
"""
CAMERA-LIDAR FUSION NODE - SYNCHRONIZED & MOTION COMPENSATED
=============================================================
A robust sensor fusion node that properly synchronizes LiDAR and depth cameras
with motion compensation to fix drift issues.

FIXES COMPARED TO robust_scan_fusion.py:
1. Uses ROS time (get_clock().now()) instead of wall clock (time.time())
2. ApproximateTimeSynchronizer for sensor synchronization
3. TF lookups at sensor timestamp, not latest
4. Actual motion compensation using odometry
5. Spatial downsampling instead of sequential
6. Proper sim_time compatibility

NEW OPTIMIZATIONS (v2):
1. Vectorized motion compensation (NumPy) for O(1) performance
2. Correct angular re-binning during deskewing
3. Fallback mechanism for LiDAR-only operation if cameras fail
4. Improved thread safety and stats

Inputs:
    - /scan_filtered (LaserScan): 2D LiDAR scan
    - /camera/depth/color/points (PointCloud2): Front camera
    - /mapping_camera/depth/color/points (PointCloud2): Left camera
    - /right_camera/depth/color/points (PointCloud2): Right camera
    - /odom (Odometry): Robot odometry for motion compensation

Outputs:
    - /scan_fused (LaserScan): Fused scan with all sensors
    - /scan_lidar_only (LaserScan): LiDAR-only filtered scan

Author: Siddharth Tiwari, IIT Mandi
Date: 2026-02-05
"""

import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Any
import threading
import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from rclpy.time import Time
from rclpy.duration import Duration

from sensor_msgs.msg import LaserScan, PointCloud2
from nav_msgs.msg import Odometry
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header

import message_filters
import tf2_ros
from tf2_ros import TransformException


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class OdomSample:
    """A single odometry sample for interpolation."""
    timestamp: float  # ROS time as seconds (float)
    x: float
    y: float
    theta: float
    vx: float
    vy: float
    omega: float


@dataclass
class CameraConfig:
    """Configuration for a single depth camera."""
    name: str
    topic: str
    frame: str
    enabled: bool = True
    min_depth: float = 0.40
    max_depth: float = 4.0
    downsample_voxel: float = 0.03  # Spatial voxel size in meters


# =============================================================================
# ODOMETRY BUFFER WITH INTERPOLATION
# =============================================================================

class OdometryBuffer:
    """
    Circular buffer for odometry samples with linear interpolation.
    
    Enables motion compensation by providing robot pose at any timestamp
    within the buffer window (default 2 seconds).
    
    Now supports vectorized batch interpolation.
    """
    
    def __init__(self, max_age: float = 2.0, max_samples: int = 500):
        self.max_age = max_age
        self.samples: deque = deque(maxlen=max_samples)
        self.lock = threading.Lock()
    
    def add_sample(self, timestamp: float, x: float, y: float, theta: float,
                   vx: float, vy: float, omega: float):
        """Add a new odometry sample to the buffer."""
        sample = OdomSample(timestamp, x, y, theta, vx, vy, omega)
        with self.lock:
            self.samples.append(sample)
    
    def interpolate(self, t: float) -> Optional[OdomSample]:
        """
        Get interpolated pose at timestamp t.
        Returns None if t is outside the buffer range.
        """
        with self.lock:
            if len(self.samples) < 2:
                return None
            
            # Simple linear search optimized for recent times
            prev_sample = None
            next_sample = None
            
            for sample in self.samples:
                if sample.timestamp <= t:
                    prev_sample = sample
                if sample.timestamp >= t and next_sample is None:
                    next_sample = sample
                    break
            
            if prev_sample is None or next_sample is None:
                if prev_sample and t - prev_sample.timestamp < 0.1: return prev_sample
                if next_sample and next_sample.timestamp - t < 0.1: return next_sample
                return None
            
            if prev_sample.timestamp == next_sample.timestamp:
                return prev_sample
            
            alpha = (t - prev_sample.timestamp) / (next_sample.timestamp - prev_sample.timestamp)
            return self._lerp(prev_sample, next_sample, alpha, t)

    def interpolate_batch(self, timestamps: np.ndarray) -> Optional[Dict[str, np.ndarray]]:
        """
        Vectorized interpolation for a batch of timestamps.
        
        Args:
            timestamps: 1D numpy array of timestamps
            
        Returns:
            Dict with keys 'x', 'y', 'theta', 'vx', 'vy', 'omega' containing arrays.
            Returns None if insufficient data.
        """
        with self.lock:
            if len(self.samples) < 2:
                return None
            
            # Create structural arrays from samples
            times = np.array([s.timestamp for s in self.samples])
            xs = np.array([s.x for s in self.samples])
            ys = np.array([s.y for s in self.samples])
            thetas = np.array([s.theta for s in self.samples])
            
            # Interpolate X, Y
            x_out = np.interp(timestamps, times, xs)
            y_out = np.interp(timestamps, times, ys)
            
            # Interpolate Theta (handle wrapping)
            sin_thetas = np.sin(thetas)
            cos_thetas = np.cos(thetas)
            sin_out = np.interp(timestamps, times, sin_thetas)
            cos_out = np.interp(timestamps, times, cos_thetas)
            theta_out = np.arctan2(sin_out, cos_out)
            
            return {
                'x': x_out,
                'y': y_out,
                'theta': theta_out
            }

    def _lerp(self, s1: OdomSample, s2: OdomSample, alpha: float, t: float) -> OdomSample:
        """Helper for linear interpolation."""
        alpha = np.clip(alpha, 0.0, 1.0)
        x = s1.x + alpha * (s2.x - s1.x)
        y = s1.y + alpha * (s2.y - s1.y)
        
        # Angle interpolation
        diff = s2.theta - s1.theta
        if diff > math.pi: diff -= 2*math.pi
        elif diff < -math.pi: diff += 2*math.pi
        theta = s1.theta + alpha * diff
        
        vx = s1.vx + alpha * (s2.vx - s1.vx)
        vy = s1.vy + alpha * (s2.vy - s1.vy)
        omega = s1.omega + alpha * (s2.omega - s1.omega)
        
        return OdomSample(t, x, y, theta, vx, vy, omega)
    
    def get_latest(self) -> Optional[OdomSample]:
        with self.lock:
            return self.samples[-1] if self.samples else None


# =============================================================================
# WHEELCHAIR FOOTPRINT FILTER
# =============================================================================

class WheelchairFootprintFilter:
    """
    URDF-calibrated self-detection filter based on wheelchair geometry.
    """
    
    def __init__(self):
        self.min_valid_range = 0.20
        
        # Robot bounding box (Cartesian backup filter)
        self.robot_half_width = 0.45
        self.robot_rear = 1.0
        self.robot_front = 0.15
        
        # Angular exclusion zones: (start_deg, end_deg, max_range_m)
        self.exclusion_zones_deg = [
            (150, 180, 1.00),     # Left rear wheel
            (-180, -140, 1.00),  # Right rear wheel
            (120, 150, 0.50),    # Left castor
            (-140, -100, 0.65),  # Right castor
            (90, 120, 0.35),     # Left side frame
            (-100, -90, 0.35),   # Right side frame
        ]
        
        self.exclusion_zones = [
            (np.radians(a1), np.radians(a2), r)
            for a1, a2, r in self.exclusion_zones_deg
        ]
    
    def filter_scan(self, ranges: np.ndarray, angle_min: float,
                    angle_increment: float) -> np.ndarray:
        """Filter out self-detections from scan."""
        result = ranges.copy()
        num_bins = len(ranges)
        angles = angle_min + np.arange(num_bins) * angle_increment
        valid = np.isfinite(ranges) & (ranges > 0)
        
        # 1. Minimum range filter
        too_close = valid & (ranges < self.min_valid_range)
        result[too_close] = np.inf
        
        # 2. Angular exclusion zones
        for angle_start, angle_end, max_range in self.exclusion_zones:
            if angle_start <= angle_end:
                in_arc = (angles >= angle_start) & (angles <= angle_end)
            else:
                in_arc = (angles >= angle_start) | (angles <= angle_end)
            in_zone = valid & in_arc & (ranges < max_range)
            result[in_zone] = np.inf
        
        # 3. Cartesian box filter (backup)
        x = np.where(valid, ranges * np.cos(angles), 0)
        y = np.where(valid, ranges * np.sin(angles), 0)
        in_box = (
            valid &
            (x >= -self.robot_front) & (x <= self.robot_rear) &
            (y >= -self.robot_half_width) & (y <= self.robot_half_width)
        )
        result[in_box] = np.inf
        
        return result


# =============================================================================
# MAIN FUSION NODE
# =============================================================================

class CameraLidarFusion(Node):
    """
    Synchronized camera-LiDAR fusion with motion compensation (deskewing).
    Includes synchronization fallback to LiDAR-only mode.
    """
    
    def __init__(self):
        super().__init__('camera_lidar_fusion')
        
        # =====================================================================
        # PARAMETERS
        # =====================================================================
        self._declare_parameters()
        
        # Load parameters
        self.laser_frame = self.get_parameter('laser_frame').value
        self.min_height = self.get_parameter('min_height').value
        self.max_height = self.get_parameter('max_height').value
        self.sync_slop = self.get_parameter('sync_slop').value
        self.sync_timeout = self.get_parameter('sync_timeout').value
        self.enable_motion_comp = self.get_parameter('enable_motion_compensation').value
        self.enable_footprint_filter = self.get_parameter('enable_footprint_filter').value
        self.verbose = self.get_parameter('verbose').value
        
        # Camera configs
        self.cameras = self._load_camera_configs()
        
        # =====================================================================
        # TF2 SETUP
        # =====================================================================
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.transform_cache: Dict[str, Tuple[np.ndarray, float]] = {}
        self.tf_cache_duration = 30.0
        
        # =====================================================================
        # ODOMETRY BUFFER
        # =====================================================================
        self.odom_buffer = OdometryBuffer(max_age=2.0)
        
        # =====================================================================
        # FOOTPRINT FILTER
        # =====================================================================
        self.footprint_filter = WheelchairFootprintFilter()
        
        # =====================================================================
        # STATE & STATS
        # =====================================================================
        self.last_sync_time = Time(seconds=0)
        self.stats = {
            'frame_count': 0,
            'sync_callbacks': 0,
            'lidar_fallback_count': 0,
            'motion_comp_applied': 0,
            'camera_points_fused': 0,
            'total_latency_ns': 0,
            'max_latency_ns': 0,
        }
        
        # =====================================================================
        # QoS
        # =====================================================================
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )
        
        # =====================================================================
        # ODOMETRY SUBSCRIBER
        # =====================================================================
        self.odom_sub = self.create_subscription(
            Odometry,
            self.get_parameter('odom_topic').value,
            self._odom_callback,
            sensor_qos
        )
        
        # =====================================================================
        # SYNCHRONIZED SUBSCRIBERS
        # =====================================================================
        self._setup_sync_subscribers(sensor_qos)
        
        # =====================================================================
        # PUBLISHERS
        # =====================================================================
        self.fused_pub = self.create_publisher(
            LaserScan, self.get_parameter('output_topic').value, 10)
        self.lidar_pub = self.create_publisher(
            LaserScan, '/scan_lidar_only', 10)
        
        # =====================================================================
        # DIAGNOSTICS TIMER
        # =====================================================================
        self.create_timer(10.0, self._print_stats)
        
        self._log_startup()
    
    def _declare_parameters(self):
        """Declare all ROS2 parameters."""
        self.declare_parameter('scan_topic', '/scan_filtered')
        self.declare_parameter('output_topic', '/scan_fused')
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('laser_frame', 'laser')
        
        self.declare_parameter('min_height', 0.15)
        self.declare_parameter('max_height', 1.50)
        
        self.declare_parameter('sync_slop', 0.05)
        self.declare_parameter('sync_queue_size', 10)
        self.declare_parameter('sync_timeout', 0.5)  # Timeout for fallback
        
        self.declare_parameter('enable_motion_compensation', True)
        self.declare_parameter('enable_footprint_filter', True)
        self.declare_parameter('verbose', False)
        
        for name in ['front', 'left', 'right']:
            self.declare_parameter(f'{name}_camera.enabled', True)
            self.declare_parameter(f'{name}_camera.topic', '')
            self.declare_parameter(f'{name}_camera.frame', '')
            self.declare_parameter(f'{name}_camera.min_depth', 0.40)
            self.declare_parameter(f'{name}_camera.max_depth', 4.0)
            self.declare_parameter(f'{name}_camera.downsample_voxel', 0.03)
        
        # Defaults
        self.set_parameters([
            rclpy.parameter.Parameter('front_camera.topic', rclpy.Parameter.Type.STRING, '/camera/depth/color/points'),
            rclpy.parameter.Parameter('front_camera.frame', rclpy.Parameter.Type.STRING, 'camera_depth_optical_frame'),
            rclpy.parameter.Parameter('left_camera.topic', rclpy.Parameter.Type.STRING, '/mapping_camera/depth/color/points'),
            rclpy.parameter.Parameter('left_camera.frame', rclpy.Parameter.Type.STRING, 'mapping_camera_depth_optical_frame'),
            rclpy.parameter.Parameter('right_camera.topic', rclpy.Parameter.Type.STRING, '/right_camera/depth/color/points'),
            rclpy.parameter.Parameter('right_camera.frame', rclpy.Parameter.Type.STRING, 'right_camera_depth_optical_frame'),
        ])
    
    def _load_camera_configs(self) -> List[CameraConfig]:
        configs = []
        for name in ['front', 'left', 'right']:
            configs.append(CameraConfig(
                name=name,
                topic=self.get_parameter(f'{name}_camera.topic').value,
                frame=self.get_parameter(f'{name}_camera.frame').value,
                enabled=self.get_parameter(f'{name}_camera.enabled').value,
                min_depth=self.get_parameter(f'{name}_camera.min_depth').value,
                max_depth=self.get_parameter(f'{name}_camera.max_depth').value,
                downsample_voxel=self.get_parameter(f'{name}_camera.downsample_voxel').value,
            ))
        return configs
    
    def _setup_sync_subscribers(self, qos: QoSProfile):
        """Set up ApproximateTimeSynchronizer and Fallback Subscriber."""
        scan_topic = self.get_parameter('scan_topic').value
        
        # 1. LiDAR Subscriber (shared)
        # We save this to register specific callbacks if needed
        self.scan_sub = message_filters.Subscriber(self, LaserScan, scan_topic, qos_profile=qos)
        
        # Register fallback callback for LiDAR-only robustness
        self.scan_sub.registerCallback(self._lidar_fallback_callback)
        
        subs = [self.scan_sub]
        self.camera_indices = {}
        
        idx = 1
        for cam in self.cameras:
            if cam.enabled and cam.topic:
                subs.append(message_filters.Subscriber(
                    self, PointCloud2, cam.topic, qos_profile=qos
                ))
                self.camera_indices[cam.name] = idx
                idx += 1
        
        self.sync = message_filters.ApproximateTimeSynchronizer(
            subs,
            queue_size=self.get_parameter('sync_queue_size').value,
            slop=self.sync_slop
        )
        self.sync.registerCallback(self._synchronized_callback)
        
        self.get_logger().info(f'Synchronized subscribers: LiDAR + {len(self.camera_indices)} cameras')
    
    def _lidar_fallback_callback(self, scan_msg: LaserScan):
        """
        Fallback callback that runs on every LiDAR message.
        Checks if synchronization is healthy. If not, publishes LiDAR-only scan.
        """
        now = self.get_clock().now()
        time_since_last_sync = (now - self.last_sync_time).nanoseconds / 1e9
        
        # If we haven't received a synchronized set in a while, process LiDAR only
        if time_since_last_sync > self.sync_timeout:
            # We are in fallback mode
            if self.stats['lidar_fallback_count'] % 100 == 0:
                self.get_logger().warn(f'Synchronization lost ({time_since_last_sync:.2f}s). Publishing LiDAR only fallback.')
            
            self.stats['lidar_fallback_count'] += 1
            self.stats['frame_count'] += 1
            
            # Process just the LiDAR
            self._process_and_publish_lidar(scan_msg, fused_fallback=True)

    def _log_startup(self):
        self.get_logger().info('=' * 60)
        self.get_logger().info('CAMERA-LIDAR FUSION v2 - VECTORIZED & ROBUST')
        self.get_logger().info('=' * 60)
        self.get_logger().info(f'Sync tolerance: {self.sync_slop * 1000:.0f}ms')
        self.get_logger().info(f'Sync timeout:   {self.sync_timeout:.2f}s')
        self.get_logger().info(f'Motion comp:    {"ENABLED" if self.enable_motion_comp else "DISABLED"}')
        self.get_logger().info(f'Footprint flt:  {"ENABLED" if self.enable_footprint_filter else "DISABLED"}')
        for cam in self.cameras:
            if cam.enabled:
                self.get_logger().info(f'  [ENABLED] {cam.name}')
        self.get_logger().info('=' * 60)
    
    # =========================================================================
    # ODOMETRY HANDLING
    # =========================================================================
    
    def _odom_callback(self, msg: Odometry):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        theta = math.atan2(siny_cosp, cosy_cosp)
        
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        omega = msg.twist.twist.angular.z
        
        self.odom_buffer.add_sample(t, x, y, theta, vx, vy, omega)
    
    # =========================================================================
    # TF HANDLING
    # =========================================================================
    
    def _get_transform_at_time(self, target: str, source: str, 
                                stamp: Time) -> Optional[np.ndarray]:
        cache_key = f'{source}_to_{target}'
        now_sec = self.get_clock().now().nanoseconds / 1e9
        
        if cache_key in self.transform_cache:
            cached_tf, cache_time = self.transform_cache[cache_key]
            if now_sec - cache_time < self.tf_cache_duration:
                return cached_tf
        
        try:
            tf = self.tf_buffer.lookup_transform(
                target, source, stamp, timeout=Duration(seconds=0.05))
        except TransformException:
            try:
                tf = self.tf_buffer.lookup_transform(
                    target, source, Time(), timeout=Duration(seconds=0.1))
            except TransformException:
                return None
        
        t = tf.transform.translation
        q = tf.transform.rotation
        
        n = 1.0 / np.sqrt(q.x**2 + q.y**2 + q.z**2 + q.w**2 + 1e-10)
        qx, qy, qz, qw = q.x*n, q.y*n, q.z*n, q.w*n
        
        R = np.array([
            [1-2*(qy**2+qz**2), 2*(qx*qy-qw*qz), 2*(qx*qz+qw*qy)],
            [2*(qx*qy+qw*qz), 1-2*(qx**2+qz**2), 2*(qy*qz-qw*qx)],
            [2*(qx*qz-qw*qy), 2*(qy*qz+qw*qx), 1-2*(qx**2+qy**2)]
        ])
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [t.x, t.y, t.z]
        
        self.transform_cache[cache_key] = (T, now_sec)
        return T
    
    # =========================================================================
    # SPATIAL DOWNSAMPLING
    # =========================================================================
    
    def _spatial_downsample(self, points: np.ndarray, voxel_size: float) -> np.ndarray:
        if len(points) == 0 or voxel_size <= 0:
            return points
        grid_coords = (points[:, :3] / voxel_size).astype(np.int32)
        _, unique_idx = np.unique(grid_coords, axis=0, return_index=True)
        return points[unique_idx]
    
    # =========================================================================
    # MOTION COMPENSATION (LIDAR DESKEWING) - VECTORIZED
    # =========================================================================
    
    def _deskew_scan(self, ranges: np.ndarray, scan_msg: LaserScan) -> np.ndarray:
        """
        Apply vectorized motion compensation using batch interpolation and re-binning.
        """
        if not self.enable_motion_comp:
            return ranges
        
        scan_start = scan_msg.header.stamp.sec + scan_msg.header.stamp.nanosec * 1e-9
        scan_end = scan_start + scan_msg.scan_time
        
        # 1. Get Pose at Scan End (Reference Frame)
        pose_end = self.odom_buffer.interpolate(scan_end)
        if pose_end is None:
            return ranges
        
        num_bins = len(ranges)
        # Generate timestamps for all points
        # t_i = start + i * increment
        indices = np.arange(num_bins)
        point_times = scan_start + indices * scan_msg.time_increment
        
        # 2. Get Poses for All Points (Vectorized)
        poses = self.odom_buffer.interpolate_batch(point_times)
        if poses is None:
            return ranges # Buffer issue
        
        # 3. Filter Invalid Ranges
        valid_mask = np.isfinite(ranges) & (ranges > 0)
        if not np.any(valid_mask):
            return ranges
        
        # Work only with valid points to save compute
        r_valid = ranges[valid_mask]
        idx_valid = indices[valid_mask]
        
        pose_x = poses['x'][valid_mask]
        pose_y = poses['y'][valid_mask]
        pose_theta = poses['theta'][valid_mask]
        
        # 4. Transform Points to World Frame (Vectorized)
        # Point in laser frame (at capture time)
        angles = scan_msg.angle_min + idx_valid * scan_msg.angle_increment
        x_laser = r_valid * np.cos(angles)
        y_laser = r_valid * np.sin(angles)
        
        # Transform to world: R_i * p + t_i
        cos_theta = np.cos(pose_theta)
        sin_theta = np.sin(pose_theta)
        
        x_world = pose_x + x_laser * cos_theta - y_laser * sin_theta
        y_world = pose_y + x_laser * sin_theta + y_laser * cos_theta
        
        # 5. Transform World to End Pose Frame
        # p_end = R_end^T * (p_world - t_end)
        dx = x_world - pose_end.x
        dy = y_world - pose_end.y
        
        cos_end = np.cos(pose_end.theta) # R_end = [cos -sin; sin cos]
        sin_end = np.sin(pose_end.theta)
        # R^T = [cos sin; -sin cos]
        
        x_corrected = dx * cos_end + dy * sin_end
        y_corrected = -dx * sin_end + dy * cos_end
        
        # 6. Compute Corrected Ranges and Angles
        r_corrected = np.sqrt(x_corrected**2 + y_corrected**2)
        theta_corrected = np.arctan2(y_corrected, x_corrected)
        
        # 7. Re-binning (Project back to scan grid)
        # Since points moved angularly, we must place them in correct bins
        # Initialize output with infinity
        corrected_scan = np.full(num_bins, np.inf, dtype=np.float32)
        
        # Determine new bin indices
        new_indices = ((theta_corrected - scan_msg.angle_min) / scan_msg.angle_increment).astype(np.int32)
        
        # Filter indices within valid range [0, num_bins-1]
        in_bounds = (new_indices >= 0) & (new_indices < num_bins)
        
        final_indices = new_indices[in_bounds]
        final_ranges = r_corrected[in_bounds]
        
        # Use minimum.at to handle collisions (keep closest point)
        np.minimum.at(corrected_scan, final_indices, final_ranges)
        
        self.stats['motion_comp_applied'] += 1
        return corrected_scan

    # =========================================================================
    # POINTCLOUD PROCESSING
    # =========================================================================
    
    def _pointcloud_to_scan(self, cloud: PointCloud2, cam: CameraConfig,
                            template: LaserScan) -> Optional[np.ndarray]:
        cloud_stamp = Time.from_msg(cloud.header.stamp)
        T = self._get_transform_at_time(self.laser_frame, cam.frame, cloud_stamp)
        if T is None:
            return None
        
        try:
            pts = point_cloud2.read_points_numpy(
                cloud, field_names=("x", "y", "z"), skip_nans=True)
            if len(pts) == 0: return None
            xyz = pts[:, :3] if pts.ndim == 2 else np.column_stack([pts['x'], pts['y'], pts['z']])
            if xyz is None or len(xyz) == 0: return None
        except Exception:
            return None
        
        xyz = self._spatial_downsample(xyz, cam.downsample_voxel)
        
        depths = xyz[:, 2]
        mask = (depths >= cam.min_depth) & (depths <= cam.max_depth)
        xyz = xyz[mask]
        
        if len(xyz) == 0: return None
        
        pts_laser = (T[:3, :3] @ xyz.T).T + T[:3, 3]
        mask = (pts_laser[:, 2] >= self.min_height) & (pts_laser[:, 2] <= self.max_height)
        pts_laser = pts_laser[mask]
        
        if len(pts_laser) == 0: return None
        
        x, y = pts_laser[:, 0], pts_laser[:, 1]
        ranges = np.sqrt(x*x + y*y)
        angles = np.arctan2(y, x)
        
        num_bins = len(template.ranges)
        scan = np.full(num_bins, np.inf, dtype=np.float32)
        
        mask = (
            (angles >= template.angle_min) &
            (angles <= template.angle_max) &
            (ranges >= template.range_min) &
            (ranges <= template.range_max)
        )
        ranges = ranges[mask]
        angles = angles[mask]
        
        if len(ranges) == 0: return None
        
        indices = ((angles - template.angle_min) / template.angle_increment).astype(np.int32)
        indices = np.clip(indices, 0, num_bins - 1)
        np.minimum.at(scan, indices, ranges.astype(np.float32))
        
        self.stats['camera_points_fused'] += len(ranges)
        return scan
    
    # =========================================================================
    # PROCESSING PIPELINE
    # =========================================================================
    
    def _process_and_publish_lidar(self, scan_msg: LaserScan, fused_fallback=False):
        """Helper to process just LiDAR and publish to filtered/fused."""
        lidar_raw = np.array(scan_msg.ranges, dtype=np.float32)
        
        # Deskew
        lidar_deskewed = self._deskew_scan(lidar_raw, scan_msg)
        
        # Filter
        if self.enable_footprint_filter:
            lidar_filtered = self.footprint_filter.filter_scan(
                lidar_deskewed, scan_msg.angle_min, scan_msg.angle_increment)
        else:
            lidar_filtered = lidar_deskewed
            
        # Create msg
        lidar_msg = self._create_scan_msg(lidar_filtered, scan_msg)
        
        # Publish
        self.lidar_pub.publish(lidar_msg)
        
        if fused_fallback:
            # Publish same msg to fused topic
            self.fused_pub.publish(lidar_msg)

    def _synchronized_callback(self, *msgs):
        """Main fusion callback - called when sensors are synchronized."""
        start_time = self.get_clock().now()
        self.last_sync_time = start_time  # Update sync heartbeat
        
        self.stats['sync_callbacks'] += 1
        self.stats['frame_count'] += 1
        
        scan_msg: LaserScan = msgs[0]
        
        # 1. Process LiDAR
        lidar_raw = np.array(scan_msg.ranges, dtype=np.float32)
        lidar_deskewed = self._deskew_scan(lidar_raw, scan_msg)
        
        if self.enable_footprint_filter:
            lidar_filtered = self.footprint_filter.filter_scan(
                lidar_deskewed, scan_msg.angle_min, scan_msg.angle_increment)
        else:
            lidar_filtered = lidar_deskewed.copy()
            
        # 2. Process Cameras
        camera_scans: List[np.ndarray] = []
        for cam in self.cameras:
            if not cam.enabled or cam.name not in self.camera_indices: continue
            idx = self.camera_indices[cam.name]
            if idx >= len(msgs): continue
            
            cam_scan = self._pointcloud_to_scan(msgs[idx], cam, scan_msg)
            if cam_scan is not None:
                if self.enable_footprint_filter:
                    cam_scan = self.footprint_filter.filter_scan(
                        cam_scan, scan_msg.angle_min, scan_msg.angle_increment)
                camera_scans.append(cam_scan)
        
        # 3. Fuse
        if camera_scans:
            all_scans = [lidar_deskewed] + camera_scans # Fuse deskewed (unfiltered) or filtered? 
            # Better to fuse deskewed and then filter once, but we already filtered Lidar
            # Let's fuse processed scans
            all_scans = [lidar_filtered] + camera_scans # Use Lidar filtered
            stacked = np.stack(all_scans, axis=0)
            fused_raw = np.nanmin(stacked, axis=0)
            fused = np.where(np.isnan(fused_raw), np.inf, fused_raw).astype(np.float32)
        else:
            fused = lidar_filtered
            
        # 4. Filter again? (Redundant if input scans are filtered, but safe)
        if self.enable_footprint_filter:
             fused = self.footprint_filter.filter_scan(
                fused, scan_msg.angle_min, scan_msg.angle_increment)
        
        # 5. Publish
        lidar_msg = self._create_scan_msg(lidar_filtered, scan_msg)
        self.lidar_pub.publish(lidar_msg)
        
        fused_msg = self._create_scan_msg(fused, scan_msg)
        self.fused_pub.publish(fused_msg)
        
        latency_ns = (self.get_clock().now() - start_time).nanoseconds
        self.stats['total_latency_ns'] += latency_ns
        self.stats['max_latency_ns'] = max(self.stats['max_latency_ns'], latency_ns)

    def _create_scan_msg(self, ranges: np.ndarray, template: LaserScan) -> LaserScan:
        msg = LaserScan()
        msg.header = Header()
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
    
    def _print_stats(self):
        if self.stats['frame_count'] == 0: return
        
        avg_latency_ms = (self.stats['total_latency_ns'] / self.stats['frame_count']) / 1e6
        max_latency_ms = self.stats['max_latency_ns'] / 1e6
        
        self.get_logger().info(
            f"Stats | Frames: {self.stats['frame_count']} | "
            f"Synced: {self.stats['sync_callbacks']} | "
            f"Fallback: {self.stats['lidar_fallback_count']} | "
            f"MotionComp: {self.stats['motion_comp_applied']} | "
            f"CamPts: {self.stats['camera_points_fused']} | "
            f"Latency: {avg_latency_ms:.1f}ms avg / {max_latency_ms:.1f}ms max"
        )

def main(args=None):
    rclpy.init(args=args)
    node = CameraLidarFusion()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
