#!/usr/bin/env python3
"""
SCAN LIDAR — FOOTPRINT-FILTERED LIDAR-ONLY SCAN
=================================================
Created: 2026-03-06

Applies wheelchair footprint filtering + rear crop + NaN/zero cleanup
to /scan_filtered, publishing /scan_lidar.

This gives a clean lidar scan with no wheelchair self-hits, identical
to what scan_fusion_v9 produces for the lidar component — but without
any camera fusion. Used for ablation testing (lidar-only baseline).

Pipeline:
    /scan (raw) -> laser_filter -> /scan_filtered -> scan_lidar -> /scan_lidar

DO NOT EDIT — create a new versioned file instead.
"""

import array as _array
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan

SENSOR_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1
)


class WheelchairFootprintFilter:
    """URDF-calibrated self-detection filter (from scan_fusion_v9)."""

    def __init__(self):
        self.min_valid_range = 0.20
        self.robot_half_width = 0.33
        self.robot_rear = 0.50
        self.robot_front = 0.20

        # (start_deg, end_deg, max_range_m)
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


class ScanLidar(Node):
    """Footprint-filtered lidar scan for ablation testing."""

    def __init__(self):
        super().__init__('scan_lidar')

        self.declare_parameter('input_topic', '/scan_filtered')
        self.declare_parameter('output_topic', '/scan_lidar')
        self.declare_parameter('rear_crop_deg', 180.0)
        self.declare_parameter('enable_footprint', True)

        self.rear_crop_deg = float(self.get_parameter('rear_crop_deg').value)
        self.enable_footprint = bool(self.get_parameter('enable_footprint').value)

        self._footprint = WheelchairFootprintFilter() if self.enable_footprint else None
        self._initialized = False
        self._rear_crop_mask = None
        self._frame_count = 0

        input_topic = self.get_parameter('input_topic').value
        output_topic = self.get_parameter('output_topic').value

        self._pub = self.create_publisher(LaserScan, output_topic, 10)
        self.create_subscription(LaserScan, input_topic, self._scan_cb, SENSOR_QOS)

        self.create_timer(10.0, self._print_stats)

        self.get_logger().info(f'scan_lidar: {input_topic} -> {output_topic}')
        self.get_logger().info(f'  Footprint filter: {self.enable_footprint}')
        self.get_logger().info(f'  Rear crop: +/-{self.rear_crop_deg:.0f} deg')

    def _scan_cb(self, scan_msg: LaserScan):
        if not self._initialized:
            num_bins = len(scan_msg.ranges)
            angle_min = scan_msg.angle_min
            angle_inc = scan_msg.angle_increment

            scan_angles = angle_min + np.arange(num_bins, dtype=np.float32) * angle_inc

            # Convert to robot-frame angles (0=forward, +-180=rear)
            robot_angles = np.arctan2(
                -np.sin(scan_angles), -np.cos(scan_angles)
            ).astype(np.float32)

            # Rear crop mask
            if self.rear_crop_deg < 180.0:
                limit_rad = np.radians(self.rear_crop_deg)
                self._rear_crop_mask = (robot_angles > limit_rad) | (robot_angles < -limit_rad)

            # Footprint filter geometry
            if self._footprint is not None:
                self._footprint.cache_geometry(num_bins, robot_angles)

            self._initialized = True
            self.get_logger().info(
                f'Initialized: {num_bins} bins, '
                f'[{np.degrees(scan_angles[0]):.1f}, {np.degrees(scan_angles[-1]):.1f}] deg')

        # Copy + clean
        ranges = np.array(scan_msg.ranges, dtype=np.float32)
        np.nan_to_num(ranges, nan=np.inf, copy=False)
        ranges[ranges <= 0.0] = np.inf

        # Footprint filter
        if self._footprint is not None:
            self._footprint.filter_scan(ranges)

        # Rear crop
        if self._rear_crop_mask is not None:
            ranges[self._rear_crop_mask] = np.inf

        # Final cleanup
        bad = ~np.isfinite(ranges) | (ranges <= 0.0)
        ranges[bad] = np.inf

        # Publish
        out = LaserScan()
        out.header.stamp = scan_msg.header.stamp
        out.header.frame_id = scan_msg.header.frame_id
        out.angle_min = scan_msg.angle_min
        out.angle_max = scan_msg.angle_max
        out.angle_increment = scan_msg.angle_increment
        out.time_increment = scan_msg.time_increment
        out.scan_time = scan_msg.scan_time
        out.range_min = scan_msg.range_min
        out.range_max = scan_msg.range_max
        out.ranges = _array.array('f', ranges.astype(np.float32, copy=False).tobytes())
        out.intensities = []
        self._pub.publish(out)
        self._frame_count += 1

    def _print_stats(self):
        if self._frame_count > 0:
            self.get_logger().info(f'[scan_lidar] {self._frame_count} frames published')
            self._frame_count = 0


def main(args=None):
    rclpy.init(args=args)
    node = ScanLidar()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
