#!/usr/bin/env python3
"""
MULTI-CAMERA HEIGHT-AWARE SCAN FUSION NODE
==========================================
Fuses 2D LiDAR (RPLidar S3) with 3 depth cameras (RealSense D455/D435i)
to create a height-aware 2D laser scan for wheelchair navigation.

Based on: "Height-Aware 3-Camera Fusion for Autonomous Wheelchair Navigation"
         Siddharth Tiwari, IIT Mandi

Camera Configuration:
- Front D455:  Serial 337122300107, Position (-0.41, 0, 1.54m), facing backward (0°)
- Left D455:   Serial 146222253403, Position (0, 0.22, 0.44m), facing left (+90°)
- Right D435i: Serial 207522077542, Position (0, -0.22, 0.44m), facing right (-90°)

Key Features:
- 260° horizontal FOV coverage (3 cameras × ~90° each)
- Height filtering: 0.10m - 1.80m (wheelchair-specific)
- MIN fusion rule (closest obstacle wins - safety first)
- TF2-based coordinate transformations
- <8ms fusion latency @ 25Hz

Inputs:
    - /scan_filtered (LaserScan): RPLidar S3 filtered scan
    - /camera/depth/color/points (PointCloud2): Front D455
    - /mapping_camera/depth/color/points (PointCloud2): Left D455
    - /right_camera/depth/color/points (PointCloud2): Right D435i

Output:
    - /scan_fused (LaserScan): Height-augmented 2D scan with 3-camera fusion

Author: Siddharth Tiwari
Date: 2026-01-28
"""

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import LaserScan, PointCloud2
from sensor_msgs_py import point_cloud2
import message_filters
import tf2_ros
from tf2_ros import TransformException
from geometry_msgs.msg import TransformStamped
import tf2_geometry_msgs
from rclpy.time import Time


class MultiCameraScanFusion(Node):
    """
    Fuses 2D LiDAR scan with 3 depth camera point clouds for height-aware
    obstacle detection with 260° horizontal FOV coverage.
    """

    def __init__(self):
        super().__init__('multi_camera_scan_fusion')

        # ====================================================================
        # PARAMETERS
        # ====================================================================
        # Topics
        self.declare_parameter('scan_topic', '/scan_filtered')
        self.declare_parameter('front_depth_topic', '/camera/depth/color/points')
        self.declare_parameter('left_depth_topic', '/mapping_camera/depth/color/points')
        self.declare_parameter('right_depth_topic', '/right_camera/depth/color/points')
        self.declare_parameter('output_topic', '/scan_fused')

        # Frame names (must match URDF)
        self.declare_parameter('laser_frame', 'laser')
        self.declare_parameter('front_camera_frame', 'camera_depth_optical_frame')
        self.declare_parameter('left_camera_frame', 'mapping_camera_depth_optical_frame')
        self.declare_parameter('right_camera_frame', 'right_camera_depth_optical_frame')

        # Height filtering (in base_link frame, meters)
        self.declare_parameter('min_obstacle_height', 0.10)  # Above floor noise
        self.declare_parameter('max_obstacle_height', 1.80)  # Below ceiling

        # Depth camera filtering (meters)
        self.declare_parameter('min_depth_range', 0.30)  # Filter camera body
        self.declare_parameter('max_depth_range', 5.0)   # D455 indoor range

        # Synchronization
        self.declare_parameter('sync_slop', 0.1)  # seconds
        self.declare_parameter('sync_queue_size', 10)

        # Camera enable flags (for testing individual cameras)
        self.declare_parameter('enable_front_camera', True)
        self.declare_parameter('enable_left_camera', True)
        self.declare_parameter('enable_right_camera', True)

        # Performance
        self.declare_parameter('downsample_factor', 2)  # Skip every N points
        self.declare_parameter('verbose', True)

        # Get parameters
        self.scan_topic = self.get_parameter('scan_topic').value
        self.front_topic = self.get_parameter('front_depth_topic').value
        self.left_topic = self.get_parameter('left_depth_topic').value
        self.right_topic = self.get_parameter('right_depth_topic').value
        self.output_topic = self.get_parameter('output_topic').value

        self.laser_frame = self.get_parameter('laser_frame').value
        self.front_frame = self.get_parameter('front_camera_frame').value
        self.left_frame = self.get_parameter('left_camera_frame').value
        self.right_frame = self.get_parameter('right_camera_frame').value

        self.min_height = self.get_parameter('min_obstacle_height').value
        self.max_height = self.get_parameter('max_obstacle_height').value
        self.min_depth = self.get_parameter('min_depth_range').value
        self.max_depth = self.get_parameter('max_depth_range').value

        self.sync_slop = self.get_parameter('sync_slop').value
        self.sync_queue_size = self.get_parameter('sync_queue_size').value

        self.enable_front = self.get_parameter('enable_front_camera').value
        self.enable_left = self.get_parameter('enable_left_camera').value
        self.enable_right = self.get_parameter('enable_right_camera').value

        self.downsample = self.get_parameter('downsample_factor').value
        self.verbose = self.get_parameter('verbose').value

        # ====================================================================
        # TF2 BUFFER
        # ====================================================================
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Cache transforms (they're static)
        self.transform_cache = {}

        # ====================================================================
        # SUBSCRIBERS
        # ====================================================================
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )

        # Build subscriber list based on enabled cameras
        subscribers = [message_filters.Subscriber(self, LaserScan, self.scan_topic, qos_profile=qos)]
        self.camera_configs = []  # (topic, frame, index)

        if self.enable_front:
            subscribers.append(message_filters.Subscriber(self, PointCloud2, self.front_topic, qos_profile=qos))
            self.camera_configs.append(('front', self.front_frame, len(subscribers) - 1))

        if self.enable_left:
            subscribers.append(message_filters.Subscriber(self, PointCloud2, self.left_topic, qos_profile=qos))
            self.camera_configs.append(('left', self.left_frame, len(subscribers) - 1))

        if self.enable_right:
            subscribers.append(message_filters.Subscriber(self, PointCloud2, self.right_topic, qos_profile=qos))
            self.camera_configs.append(('right', self.right_frame, len(subscribers) - 1))

        # Time synchronizer
        self.ts = message_filters.ApproximateTimeSynchronizer(
            subscribers,
            queue_size=self.sync_queue_size,
            slop=self.sync_slop
        )
        self.ts.registerCallback(self.fusion_callback)

        # ====================================================================
        # PUBLISHER
        # ====================================================================
        self.fused_scan_pub = self.create_publisher(LaserScan, self.output_topic, 10)

        # ====================================================================
        # STATISTICS
        # ====================================================================
        self.frame_count = 0
        self.last_log_time = self.get_clock().now()
        self.camera_stats = {name: {'points': 0, 'added': 0} for name, _, _ in self.camera_configs}

        # Log startup
        self.get_logger().info('=' * 70)
        self.get_logger().info('MULTI-CAMERA SCAN FUSION NODE INITIALIZED')
        self.get_logger().info('=' * 70)
        self.get_logger().info(f'LiDAR topic:    {self.scan_topic}')
        self.get_logger().info(f'Output topic:   {self.output_topic}')
        self.get_logger().info(f'Height range:   [{self.min_height:.2f}, {self.max_height:.2f}] m')
        self.get_logger().info(f'Depth range:    [{self.min_depth:.2f}, {self.max_depth:.2f}] m')
        self.get_logger().info('Cameras enabled:')
        for name, frame, _ in self.camera_configs:
            self.get_logger().info(f'  - {name}: {frame}')
        self.get_logger().info('=' * 70)

    def get_transform(self, target_frame: str, source_frame: str) -> np.ndarray:
        """
        Get transform matrix from source_frame to target_frame.
        Caches result since camera frames are static.

        Returns 4x4 homogeneous transformation matrix.
        """
        cache_key = f'{source_frame}_to_{target_frame}'

        if cache_key in self.transform_cache:
            return self.transform_cache[cache_key]

        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                Time(),
                timeout=rclpy.duration.Duration(seconds=1.0)
            )

            # Convert to 4x4 matrix
            t = transform.transform.translation
            q = transform.transform.rotation

            # Rotation matrix from quaternion
            qx, qy, qz, qw = q.x, q.y, q.z, q.w
            R = np.array([
                [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
                [2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
                [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)]
            ])

            # Build 4x4 homogeneous matrix
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = [t.x, t.y, t.z]

            self.transform_cache[cache_key] = T
            self.get_logger().info(f'Cached transform: {source_frame} -> {target_frame}')
            return T

        except TransformException as e:
            self.get_logger().warn(f'Transform lookup failed ({source_frame} -> {target_frame}): {e}')
            return None

    def transform_points(self, points: np.ndarray, transform: np.ndarray) -> np.ndarray:
        """
        Apply 4x4 homogeneous transform to Nx3 points array.
        Returns Nx3 transformed points.
        """
        # Add homogeneous coordinate
        ones = np.ones((points.shape[0], 1))
        points_h = np.hstack([points, ones])

        # Apply transform
        transformed = (transform @ points_h.T).T

        return transformed[:, :3]

    def process_pointcloud(self, cloud_msg: PointCloud2, camera_name: str, camera_frame: str) -> tuple:
        """
        Process a single camera's point cloud:
        1. Extract points
        2. Transform to laser frame
        3. Filter by height and depth
        4. Convert to polar coordinates

        Returns (ranges, angles, num_valid_points) or (None, None, 0) on error.
        """
        try:
            # Get transform (camera_optical_frame -> laser)
            T = self.get_transform(self.laser_frame, camera_frame)
            if T is None:
                return None, None, 0

            # Extract points from PointCloud2
            points = point_cloud2.read_points_numpy(
                cloud_msg,
                field_names=("x", "y", "z"),
                skip_nans=True
            )

            if len(points) == 0:
                return None, None, 0

            # Handle structured vs regular arrays
            if points.dtype.names is not None:
                x = points['x']
                y = points['y']
                z = points['z']
            else:
                if points.ndim == 2 and points.shape[1] >= 3:
                    x, y, z = points[:, 0], points[:, 1], points[:, 2]
                else:
                    return None, None, 0

            # Stack as Nx3
            pts_camera = np.column_stack([x, y, z])

            # Downsample for performance
            if self.downsample > 1:
                pts_camera = pts_camera[::self.downsample]

            # Filter by depth (in camera frame, before transform)
            # Camera optical frame: Z = forward (depth)
            depth = pts_camera[:, 2]
            depth_mask = (depth >= self.min_depth) & (depth <= self.max_depth)
            pts_camera = pts_camera[depth_mask]

            if len(pts_camera) == 0:
                return None, None, 0

            # Transform to laser frame
            pts_laser = self.transform_points(pts_camera, T)

            # Filter by height in laser frame (Z = up)
            z_laser = pts_laser[:, 2]
            height_mask = (z_laser >= self.min_height) & (z_laser <= self.max_height)
            pts_laser = pts_laser[height_mask]

            if len(pts_laser) == 0:
                return None, None, 0

            # Convert to polar coordinates (in laser frame XY plane)
            x_laser = pts_laser[:, 0]
            y_laser = pts_laser[:, 1]

            ranges = np.sqrt(x_laser**2 + y_laser**2)
            angles = np.arctan2(y_laser, x_laser)

            return ranges, angles, len(pts_laser)

        except Exception as e:
            self.get_logger().error(f'Error processing {camera_name} cloud: {e}')
            return None, None, 0

    def fusion_callback(self, *msgs):
        """
        Main fusion callback - synchronizes LiDAR scan with all enabled camera point clouds.
        Uses MIN fusion rule: closest obstacle wins (safety first).
        """
        scan_msg = msgs[0]

        # ====================================================================
        # STEP 1: Initialize fused scan from LiDAR
        # ====================================================================
        fused_scan = LaserScan()
        fused_scan.header = scan_msg.header
        fused_scan.angle_min = scan_msg.angle_min
        fused_scan.angle_max = scan_msg.angle_max
        fused_scan.angle_increment = scan_msg.angle_increment
        fused_scan.time_increment = scan_msg.time_increment
        fused_scan.scan_time = scan_msg.scan_time
        fused_scan.range_min = scan_msg.range_min
        fused_scan.range_max = scan_msg.range_max
        fused_scan.ranges = list(scan_msg.ranges)
        fused_scan.intensities = list(scan_msg.intensities) if scan_msg.intensities else []

        num_bins = len(fused_scan.ranges)

        # ====================================================================
        # STEP 2: Process each camera and fuse with MIN rule
        # ====================================================================
        total_points_added = 0

        for camera_name, camera_frame, msg_index in self.camera_configs:
            cloud_msg = msgs[msg_index]

            ranges, angles, num_points = self.process_pointcloud(
                cloud_msg, camera_name, camera_frame
            )

            if ranges is None:
                continue

            # Update stats
            self.camera_stats[camera_name]['points'] = num_points
            points_added = 0

            # Project onto scan bins using MIN fusion
            for r, theta in zip(ranges, angles):
                # Check if angle is within scan FOV
                if theta < fused_scan.angle_min or theta > fused_scan.angle_max:
                    continue

                # Calculate bin index
                bin_idx = int((theta - fused_scan.angle_min) / fused_scan.angle_increment)

                if 0 <= bin_idx < num_bins:
                    current_range = fused_scan.ranges[bin_idx]

                    # MIN fusion: take closest obstacle
                    if np.isinf(current_range) or np.isnan(current_range) or r < current_range:
                        fused_scan.ranges[bin_idx] = float(r)
                        points_added += 1

            self.camera_stats[camera_name]['added'] = points_added
            total_points_added += points_added

        # ====================================================================
        # STEP 3: Publish fused scan
        # ====================================================================
        self.fused_scan_pub.publish(fused_scan)

        # ====================================================================
        # LOGGING
        # ====================================================================
        self.frame_count += 1
        now = self.get_clock().now()
        if self.verbose and (now - self.last_log_time).nanoseconds > 2e9:  # Every 2 sec
            stats_str = ', '.join([
                f'{name}: {s["points"]}/{s["added"]}'
                for name, s in self.camera_stats.items()
            ])
            self.get_logger().info(
                f'Frame {self.frame_count}: {total_points_added} points added | {stats_str}'
            )
            self.last_log_time = now


def main(args=None):
    rclpy.init(args=args)
    node = MultiCameraScanFusion()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
