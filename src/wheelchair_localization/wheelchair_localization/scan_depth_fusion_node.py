#!/usr/bin/env python3

"""
HEIGHT-AUGMENTED 3-CAMERA + LIDAR FUSION NODE
==============================================
Fuses 2D lidar (RPLidar S3) with 3x RealSense depth cameras (D455 + D435i)
to create a height-aware 2D laser scan for wheelchair navigation.

Based on: COMPREHENSIVE_WHEELCHAIR_NAVIGATION_REPORT.pdf
          - Front D455: Position (-0.41, 0, 1.54m) relative to base_link, 0° (backward-facing)
          - Left D455: Position (0, 0.22, 0.44m) relative to base_link, +90° (left-facing)
          - Right D435i: Position (0, -0.22, 0.44m) relative to base_link, -90° (right-facing)

Problem:
    - 2D lidar mounted at footrest height (~0.2m) misses obstacles above scan plane
    - Tables, shelves, and elevated equipment are invisible to the lidar
    - Standard 2D SLAM treats above-plane space as "free" when obstacles exist

Solution:
    - Projects 3x depth camera point clouds onto 2D lidar scan plane
    - Uses TF2 to properly transform from camera optical frames to laser frame
    - Creates virtual laser hits for obstacles at ANY height within camera FOV
    - Maintains LaserScan message format for compatibility with SLAM/localization
    - MIN fusion rule: closest obstacle wins (safety first)

Inputs:
    - /scan_raw (LaserScan): RPLidar S3 2D scan at footrest height
    - /camera/depth/color/points (PointCloud2): Front RealSense D455
    - /mapping_camera/depth/color/points (PointCloud2): Left RealSense D455
    - /right_camera/depth/color/points (PointCloud2): Right RealSense D435i

Output:
    - /scan_fused (LaserScan): Height-augmented 2D scan combining all sensors

Author: Siddharth Tiwari
Date: 2026-02-02
"""

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from sensor_msgs.msg import LaserScan, PointCloud2
from sensor_msgs_py import point_cloud2
from geometry_msgs.msg import PointStamped, TransformStamped
import message_filters
from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException
import tf2_geometry_msgs


class ScanDepthFusionNode(Node):
    """Fuses 2D lidar scan with 3x depth camera point clouds for height-aware obstacle detection."""

    def __init__(self):
        super().__init__('scan_depth_fusion_node')

        # ====================================================================
        # TF2 SETUP
        # ====================================================================
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ====================================================================
        # PARAMETERS
        # ====================================================================
        self.declare_parameter('scan_topic', '/scan_raw')
        self.declare_parameter('front_depth_topic', '/camera/depth/color/points')
        self.declare_parameter('left_depth_topic', '/mapping_camera/depth/color/points')
        self.declare_parameter('right_depth_topic', '/right_camera/depth/color/points')
        self.declare_parameter('output_topic', '/scan_fused')

        # Frame IDs
        self.declare_parameter('laser_frame', 'laser')
        self.declare_parameter('front_camera_optical_frame', 'camera_depth_optical_frame')
        self.declare_parameter('left_camera_optical_frame', 'mapping_camera_depth_optical_frame')
        self.declare_parameter('right_camera_optical_frame', 'right_camera_depth_optical_frame')

        # Height filtering (in laser frame, relative to scan plane)
        self.declare_parameter('min_obstacle_height', 0.10)  # meters - above scan plane
        self.declare_parameter('max_obstacle_height', 1.80)  # meters - below ceiling

        # Depth camera filtering
        self.declare_parameter('min_depth_range', 0.30)  # meters - close to camera spec
        self.declare_parameter('max_depth_range', 4.0)   # meters - D455/D435i effective range

        # Fusion parameters
        self.declare_parameter('use_approximate_sync', True)
        self.declare_parameter('sync_queue_size', 10)
        self.declare_parameter('sync_slop', 0.2)  # seconds - generous for 3 cameras

        # Debugging
        self.declare_parameter('verbose', True)

        # Get parameters
        scan_topic = self.get_parameter('scan_topic').value
        front_topic = self.get_parameter('front_depth_topic').value
        left_topic = self.get_parameter('left_depth_topic').value
        right_topic = self.get_parameter('right_depth_topic').value
        output_topic = self.get_parameter('output_topic').value

        self.laser_frame = self.get_parameter('laser_frame').value
        self.front_optical_frame = self.get_parameter('front_camera_optical_frame').value
        self.left_optical_frame = self.get_parameter('left_camera_optical_frame').value
        self.right_optical_frame = self.get_parameter('right_camera_optical_frame').value

        self.min_height = self.get_parameter('min_obstacle_height').value
        self.max_height = self.get_parameter('max_obstacle_height').value
        self.min_depth = self.get_parameter('min_depth_range').value
        self.max_depth = self.get_parameter('max_depth_range').value
        self.use_approx_sync = self.get_parameter('use_approximate_sync').value
        self.queue_size = self.get_parameter('sync_queue_size').value
        self.sync_slop = self.get_parameter('sync_slop').value
        self.verbose = self.get_parameter('verbose').value

        # ====================================================================
        # SUBSCRIBERS (TIME-SYNCHRONIZED FOR 4 TOPICS)
        # ====================================================================
        self.scan_sub = message_filters.Subscriber(self, LaserScan, scan_topic)
        self.front_depth_sub = message_filters.Subscriber(self, PointCloud2, front_topic)
        self.left_depth_sub = message_filters.Subscriber(self, PointCloud2, left_topic)
        self.right_depth_sub = message_filters.Subscriber(self, PointCloud2, right_topic)

        # Time synchronizer for all 4 topics
        if self.use_approx_sync:
            self.ts = message_filters.ApproximateTimeSynchronizer(
                [self.scan_sub, self.front_depth_sub, self.left_depth_sub, self.right_depth_sub],
                queue_size=self.queue_size,
                slop=self.sync_slop
            )
        else:
            self.ts = message_filters.TimeSynchronizer(
                [self.scan_sub, self.front_depth_sub, self.left_depth_sub, self.right_depth_sub],
                queue_size=self.queue_size
            )

        self.ts.registerCallback(self.fusion_callback)

        # ====================================================================
        # PUBLISHER
        # ====================================================================
        self.fused_scan_pub = self.create_publisher(LaserScan, output_topic, 10)

        # ====================================================================
        # STATE
        # ====================================================================
        self.frame_count = 0
        self.last_log_time = self.get_clock().now()

        self.get_logger().info('=' * 80)
        self.get_logger().info('3-CAMERA + LIDAR FUSION NODE INITIALIZED')
        self.get_logger().info('=' * 80)
        self.get_logger().info(f'Lidar topic:        {scan_topic} (frame: {self.laser_frame})')
        self.get_logger().info(f'Front depth topic:  {front_topic} (frame: {self.front_optical_frame})')
        self.get_logger().info(f'Left depth topic:   {left_topic} (frame: {self.left_optical_frame})')
        self.get_logger().info(f'Right depth topic:  {right_topic} (frame: {self.right_optical_frame})')
        self.get_logger().info(f'Output topic:       {output_topic}')
        self.get_logger().info(f'Height range:       [{self.min_height:.2f}, {self.max_height:.2f}] m')
        self.get_logger().info(f'Depth range:        [{self.min_depth:.2f}, {self.max_depth:.2f}] m')
        self.get_logger().info(f'Sync mode:          {"Approximate" if self.use_approx_sync else "Exact"}')
        self.get_logger().info('=' * 80)

    def fusion_callback(self, scan_msg: LaserScan,
                       front_pc: PointCloud2,
                       left_pc: PointCloud2,
                       right_pc: PointCloud2):
        """Main fusion callback - combines lidar scan with 3x depth camera point clouds."""

        # ====================================================================
        # STEP 1: Copy lidar scan as base
        # ====================================================================
        fused_scan = LaserScan()
        fused_scan.header = scan_msg.header
        fused_scan.header.frame_id = self.laser_frame  # Ensure correct frame
        fused_scan.angle_min = scan_msg.angle_min
        fused_scan.angle_max = scan_msg.angle_max
        fused_scan.angle_increment = scan_msg.angle_increment
        fused_scan.time_increment = scan_msg.time_increment
        fused_scan.scan_time = scan_msg.scan_time
        fused_scan.range_min = scan_msg.range_min
        fused_scan.range_max = scan_msg.range_max
        fused_scan.ranges = list(scan_msg.ranges)  # Copy lidar ranges
        fused_scan.intensities = list(scan_msg.intensities) if scan_msg.intensities else []

        # ====================================================================
        # STEP 2: Process each camera and fuse depth obstacles
        # ====================================================================
        total_points_added = 0

        cameras = [
            (front_pc, self.front_optical_frame, 'Front'),
            (left_pc, self.left_optical_frame, 'Left'),
            (right_pc, self.right_optical_frame, 'Right')
        ]

        for pointcloud_msg, optical_frame, camera_name in cameras:
            points_added = self.fuse_camera_to_scan(
                fused_scan,
                pointcloud_msg,
                optical_frame,
                camera_name
            )
            total_points_added += points_added

        # ====================================================================
        # STEP 3: Publish fused scan
        # ====================================================================
        self.fused_scan_pub.publish(fused_scan)

        # ====================================================================
        # LOGGING
        # ====================================================================
        self.frame_count += 1
        if self.verbose and (self.get_clock().now() - self.last_log_time).nanoseconds > 2e9:  # Every 2 seconds
            self.get_logger().info(
                f'Fusion frame {self.frame_count}: {total_points_added} obstacle points added to scan'
            )
            self.last_log_time = self.get_clock().now()

    def fuse_camera_to_scan(self, fused_scan: LaserScan,
                           pointcloud_msg: PointCloud2,
                           optical_frame: str,
                           camera_name: str) -> int:
        """
        Fuse a single camera's point cloud into the laser scan.

        Returns: Number of points added to scan
        """
        try:
            # ================================================================
            # STEP 1: Get TF transform from camera optical frame to laser frame
            # ================================================================
            try:
                transform: TransformStamped = self.tf_buffer.lookup_transform(
                    self.laser_frame,
                    optical_frame,
                    rclpy.time.Time(),  # Latest available transform
                    timeout=Duration(seconds=0.1)
                )
            except (LookupException, ConnectivityException, ExtrapolationException) as e:
                if self.frame_count % 50 == 0:  # Log every 50 frames to avoid spam
                    self.get_logger().warn(
                        f'{camera_name} camera: TF lookup failed ({optical_frame} -> {self.laser_frame}): {str(e)}'
                    )
                return 0

            # ================================================================
            # STEP 2: Extract and parse point cloud
            # ================================================================
            points = point_cloud2.read_points_numpy(
                pointcloud_msg,
                field_names=("x", "y", "z"),
                skip_nans=True
            )

            if len(points) == 0:
                return 0

            # Extract coordinates - handle both structured and unstructured arrays
            if points.dtype.names is not None:
                x = points['x']
                y = points['y']
                z = points['z']
            else:
                if points.ndim == 2 and points.shape[1] >= 3:
                    x = points[:, 0]
                    y = points[:, 1]
                    z = points[:, 2]
                elif points.ndim == 1 and len(points) >= 3:
                    x = np.array([points[0]])
                    y = np.array([points[1]])
                    z = np.array([points[2]])
                else:
                    return 0

            # ================================================================
            # STEP 3: Transform points from camera optical frame to laser frame
            # ================================================================
            # Extract transform components
            tx = transform.transform.translation.x
            ty = transform.transform.translation.y
            tz = transform.transform.translation.z

            qx = transform.transform.rotation.x
            qy = transform.transform.rotation.y
            qz = transform.transform.rotation.z
            qw = transform.transform.rotation.w

            # Convert quaternion to rotation matrix
            # See: https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
            R = np.array([
                [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
                [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
                [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
            ])

            # Apply transformation: p_laser = R * p_camera + t
            points_camera = np.stack([x, y, z], axis=1)  # Nx3
            points_laser = (R @ points_camera.T).T + np.array([tx, ty, tz])

            laser_x = points_laser[:, 0]
            laser_y = points_laser[:, 1]
            laser_z = points_laser[:, 2]

            # ================================================================
            # STEP 4: Filter by height (Z in laser frame)
            # ================================================================
            height_mask = (laser_z >= self.min_height) & (laser_z <= self.max_height)

            # ================================================================
            # STEP 5: Filter by depth (distance from camera in camera frame)
            # ================================================================
            depth_in_camera_frame = z  # Z coordinate in camera optical frame
            depth_mask = (depth_in_camera_frame >= self.min_depth) & (depth_in_camera_frame <= self.max_depth)

            # Combined filter
            valid_mask = height_mask & depth_mask

            if not np.any(valid_mask):
                return 0

            # Apply filters
            laser_x_filtered = laser_x[valid_mask]
            laser_y_filtered = laser_y[valid_mask]

            # ================================================================
            # STEP 6: Convert to polar coordinates (range, angle)
            # ================================================================
            ranges_depth = np.sqrt(laser_x_filtered**2 + laser_y_filtered**2)
            angles_depth = np.arctan2(laser_y_filtered, laser_x_filtered)

            # ================================================================
            # STEP 7: Project depth obstacles onto lidar scan (MIN fusion)
            # ================================================================
            num_points_added = 0
            for r, theta in zip(ranges_depth, angles_depth):
                # Check if angle is within lidar FOV
                if theta < fused_scan.angle_min or theta > fused_scan.angle_max:
                    continue

                # Calculate scan array index
                angle_index = int((theta - fused_scan.angle_min) / fused_scan.angle_increment)

                if 0 <= angle_index < len(fused_scan.ranges):
                    # MIN fusion strategy: take MINIMUM range (closest obstacle wins)
                    current_range = fused_scan.ranges[angle_index]

                    # Only update if:
                    # 1. Current range is invalid (inf/nan), OR
                    # 2. Camera detects closer obstacle
                    if np.isinf(current_range) or np.isnan(current_range) or r < current_range:
                        fused_scan.ranges[angle_index] = float(r)
                        num_points_added += 1

            return num_points_added

        except Exception as e:
            if self.frame_count % 50 == 0:  # Limit error spam
                self.get_logger().error(f'{camera_name} camera fusion error: {str(e)}')
            return 0


def main(args=None):
    rclpy.init(args=args)
    node = ScanDepthFusionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
