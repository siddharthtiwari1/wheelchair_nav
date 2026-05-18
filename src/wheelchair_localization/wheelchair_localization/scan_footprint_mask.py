#!/usr/bin/env python3
"""
SCAN FOOTPRINT MASK — Remove wheelchair self-detections from lidar scan
========================================================================
Applies URDF-calibrated exclusion zones + rectangular body box to remove
lidar returns from wheelchair wheels, armrests, footrests, and frame.

Ported from: scan_fusion_v9.py WheelchairFootprintFilter (proven in production)

Subscribes: /scan_filtered  (or configurable)
Publishes:  /scan_masked     (clean scan, no wheelchair self-detections)

Use in mapping pipelines where scan_fusion is not needed (lidar-only SLAM).
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


class ScanFootprintMask(Node):
    """Mask wheelchair body from lidar scan using calibrated exclusion zones."""

    def __init__(self):
        super().__init__('scan_footprint_mask')

        self.declare_parameter('input_topic', '/scan_filtered')
        self.declare_parameter('output_topic', '/scan_masked')

        # Wheelchair body dimensions (from URDF / scan_fusion_v9)
        self.declare_parameter('min_valid_range', 0.15)
        self.declare_parameter('robot_half_width', 0.33)
        self.declare_parameter('robot_rear', 0.50)
        self.declare_parameter('robot_front', 0.20)

        self.min_valid_range = float(self.get_parameter('min_valid_range').value)
        self.robot_half_width = float(self.get_parameter('robot_half_width').value)
        self.robot_rear = float(self.get_parameter('robot_rear').value)
        self.robot_front = float(self.get_parameter('robot_front').value)

        # URDF-calibrated angular exclusion zones from scan_fusion_v9
        # (start_deg, end_deg, max_range_m) in robot frame (0=forward)
        self.exclusion_zones_deg = [
            ( 150,  180, 1.00),   # rear-left
            (-180, -140, 1.00),   # rear-right
            ( 120,  150, 0.50),   # left-rear quadrant
            (-140, -100, 0.65),   # right-rear quadrant
            (  90,  120, 0.35),   # left side
            (-100,  -90, 0.35),   # right side
            ( -35,  -23, 0.32),   # front-right (footrest/wheel)
            (  50,   60, 0.48),   # front-left (footrest/wheel)
            (  22,   32, 0.45),   # front (footrest)
        ]

        input_topic = self.get_parameter('input_topic').value
        output_topic = self.get_parameter('output_topic').value

        self.pub = self.create_publisher(LaserScan, output_topic, 10)
        self.create_subscription(LaserScan, input_topic, self._scan_cb, SENSOR_QOS)

        # Cached geometry (computed once on first scan)
        self._initialized = False
        self._arc_masks = []
        self._cos_a = None
        self._sin_a = None

        self.get_logger().info(f'Footprint mask: {input_topic} -> {output_topic}')
        self.get_logger().info(f'  Body box: front={self.robot_front}m, rear={self.robot_rear}m, '
                               f'half_width={self.robot_half_width}m')
        self.get_logger().info(f'  Exclusion zones: {len(self.exclusion_zones_deg)}')
        self.get_logger().info(f'  Min valid range: {self.min_valid_range}m')

    def _init_geometry(self, scan_msg: LaserScan):
        """Cache angle-based masks on first scan (same logic as scan_fusion_v9)."""
        n = len(scan_msg.ranges)
        scan_angles = scan_msg.angle_min + np.arange(n, dtype=np.float32) * scan_msg.angle_increment

        # Convert to robot frame angles (0=forward, same as scan_fusion_v9)
        robot_angles = np.arctan2(
            -np.sin(scan_angles), -np.cos(scan_angles)
        ).astype(np.float32)

        self._cos_a = np.cos(robot_angles)
        self._sin_a = np.sin(robot_angles)

        # Pre-compute arc masks for each exclusion zone
        self._arc_masks = []
        for a1_deg, a2_deg, max_r in self.exclusion_zones_deg:
            a1 = np.radians(a1_deg)
            a2 = np.radians(a2_deg)
            if a1 <= a2:
                mask = (robot_angles >= a1) & (robot_angles <= a2)
            else:
                mask = (robot_angles >= a1) | (robot_angles <= a2)
            self._arc_masks.append((mask, max_r))

        self._initialized = True
        self.get_logger().info(f'Geometry cached: {n} bins, '
                               f'[{np.degrees(scan_angles[0]):.1f}, {np.degrees(scan_angles[-1]):.1f}] deg')

    def _scan_cb(self, scan_msg: LaserScan):
        if not self._initialized:
            self._init_geometry(scan_msg)

        ranges = np.array(scan_msg.ranges, dtype=np.float32)

        # Clean invalid values
        np.nan_to_num(ranges, nan=np.inf, copy=False)
        ranges[ranges <= 0.0] = np.inf

        valid = np.isfinite(ranges) & (ranges > 0)

        # 1) Min range — anything too close is self-reflection
        ranges[valid & (ranges < self.min_valid_range)] = np.inf

        # 2) Angular exclusion zones — wheelchair parts at known angles/ranges
        for arc_mask, max_r in self._arc_masks:
            ranges[valid & arc_mask & (ranges < max_r)] = np.inf

        # 3) Rectangular body box — catch anything inside wheelchair footprint
        x = np.where(valid, ranges * self._cos_a, 0.0)
        y = np.where(valid, ranges * self._sin_a, 0.0)
        in_box = (
            valid
            & (x >= -self.robot_rear) & (x <= self.robot_front)
            & (y >= -self.robot_half_width) & (y <= self.robot_half_width)
        )
        ranges[in_box] = np.inf

        # Publish masked scan
        out = LaserScan()
        out.header = scan_msg.header
        out.angle_min = scan_msg.angle_min
        out.angle_max = scan_msg.angle_max
        out.angle_increment = scan_msg.angle_increment
        out.time_increment = scan_msg.time_increment
        out.scan_time = scan_msg.scan_time
        out.range_min = scan_msg.range_min
        out.range_max = scan_msg.range_max
        out.ranges = _array.array('f', ranges.astype(np.float32, copy=False).tobytes())
        out.intensities = []
        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = ScanFootprintMask()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
