#!/usr/bin/env python3
"""
SCAN DATA LOGGER FOR SLAM MAPPING METRICS
=========================================
Logs all LaserScan data for SLAM debugging and camera fusion analysis.

Topics logged:
  - /scan              : Raw RPLidar S3 scan
  - /scan_filtered     : After laser filter (no casters)
  - /scan_lidar_only   : LiDAR with footprint filter (from fusion node)
  - /scan_front_camera : Front D455 camera scan
  - /scan_left_camera  : Left D455 camera scan
  - /scan_right_camera : Right D435i camera scan
  - /scan_fused        : Final LiDAR + 3-camera fusion output

Metrics computed per scan:
  - valid_points       : Points within valid range
  - coverage_deg       : Angular coverage with valid readings
  - mean_range         : Average distance of valid points
  - min_range          : Closest obstacle detected
  - density            : Valid points per degree

Usage:
  ros2 run scripts scan_data_logger
  ros2 run scripts scan_data_logger --ros-args -p log_frequency_hz:=5.0

Output: ~/wheelchair_nav_logs/scan_log_YYYYMMDD_HHMMSS.csv
"""

import csv
import math
import os
import time
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles
from sensor_msgs.msg import LaserScan


def stamp_to_float(msg_stamp) -> float:
    """Convert ROS2 Time to seconds as float."""
    return msg_stamp.sec + msg_stamp.nanosec * 1e-9


@dataclass
class ScanSnapshot:
    """Full snapshot of a LaserScan message with computed metrics."""
    stamp: float = 0.0
    frame_id: str = ""
    angle_min: float = 0.0
    angle_max: float = 0.0
    angle_increment: float = 0.0
    time_increment: float = 0.0
    scan_time: float = 0.0
    range_min: float = 0.0
    range_max: float = 0.0
    ranges: List[float] = field(default_factory=list)
    intensities: List[float] = field(default_factory=list)
    # Counts
    num_points: int = 0
    valid_points: int = 0
    # Computed metrics
    coverage_deg: float = 0.0      # Angular coverage with valid points
    mean_range: float = 0.0        # Mean distance of valid points
    min_range: float = float('inf')  # Closest point
    max_range_seen: float = 0.0    # Farthest valid point
    density: float = 0.0           # Valid points per degree
    receive_time: float = 0.0


class ScanDataLogger(Node):
    """Logs all scan data for SLAM debugging and fusion metrics."""

    # All scan sources to log
    SCAN_SOURCES = {
        'raw': '/scan',
        'filtered': '/scan_filtered',
        'lidar_only': '/scan_lidar_only',
        'front_cam': '/scan_front_camera',
        'left_cam': '/scan_left_camera',
        'right_cam': '/scan_right_camera',
        'fused': '/scan_fused',
    }

    HEADER = [
        'wall_time',
        'source',
        'stamp',
        'frame_id',
        # Scan parameters
        'angle_min',
        'angle_max',
        'angle_increment',
        'range_min',
        'range_max',
        # Point counts
        'num_points',
        'valid_points',
        # Computed metrics for comparison
        'coverage_deg',
        'mean_range',
        'min_range',
        'max_range_seen',
        'density',  # points per degree
        'latency_ms',
        # Full ranges (optional)
        'ranges',
    ]

    def __init__(self):
        super().__init__('scan_data_logger')

        # Parameters
        self.declare_parameter('log_frequency_hz', 10.0)
        self.declare_parameter('output_dir', os.path.join(os.path.expanduser('~'), 'wheelchair_nav_logs'))
        self.declare_parameter('buffer_size', 50)
        self.declare_parameter('log_ranges', True)  # Set False for smaller files

        # Enable/disable individual sources
        self.declare_parameter('log_raw', True)
        self.declare_parameter('log_filtered', True)
        self.declare_parameter('log_lidar_only', True)
        self.declare_parameter('log_front_cam', True)
        self.declare_parameter('log_left_cam', True)
        self.declare_parameter('log_right_cam', True)
        self.declare_parameter('log_fused', True)

        self._log_freq = self.get_parameter('log_frequency_hz').value
        output_dir = self.get_parameter('output_dir').value
        self._buffer_size = self.get_parameter('buffer_size').value
        self._log_ranges = self.get_parameter('log_ranges').value

        # Which sources to log
        self._enabled_sources = {
            'raw': self.get_parameter('log_raw').value,
            'filtered': self.get_parameter('log_filtered').value,
            'lidar_only': self.get_parameter('log_lidar_only').value,
            'front_cam': self.get_parameter('log_front_cam').value,
            'left_cam': self.get_parameter('log_left_cam').value,
            'right_cam': self.get_parameter('log_right_cam').value,
            'fused': self.get_parameter('log_fused').value,
        }

        # Create output file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_path = Path(output_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        self._csv_path = log_path / f'scan_log_{timestamp}.csv'

        # Data storage - one snapshot per source
        self._scans: Dict[str, Optional[ScanSnapshot]] = {
            source: None for source in self.SCAN_SOURCES.keys()
        }

        # Statistics
        self._rows_written = 0
        self._start_time = time.time()
        self._msg_counts: Dict[str, int] = {source: 0 for source in self.SCAN_SOURCES.keys()}

        # Write buffer
        self._buffer: deque = deque(maxlen=self._buffer_size * 2)

        # QoS for sensor data
        sensor_qos = QoSPresetProfiles.SENSOR_DATA.value

        # Create subscribers for enabled sources
        for source, topic in self.SCAN_SOURCES.items():
            if self._enabled_sources.get(source, False):
                self.create_subscription(
                    LaserScan, topic,
                    lambda msg, s=source: self._scan_cb(msg, s),
                    sensor_qos
                )

        # Open CSV
        self._csv_file = open(self._csv_path, 'w', newline='', buffering=1)
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow(self.HEADER)

        self.get_logger().info('=' * 60)
        self.get_logger().info('SCAN DATA LOGGER - FUSION METRICS')
        self.get_logger().info('=' * 60)
        self.get_logger().info(f'Output: {self._csv_path}')
        self.get_logger().info(f'Frequency: {self._log_freq} Hz')
        self.get_logger().info(f'Log ranges: {self._log_ranges}')
        self.get_logger().info('-' * 60)
        self.get_logger().info('Enabled topics:')
        for source, enabled in self._enabled_sources.items():
            if enabled:
                self.get_logger().info(f'  [{source}] {self.SCAN_SOURCES[source]}')
        self.get_logger().info('=' * 60)

        # Timer for periodic logging
        self._timer = self.create_timer(1.0 / self._log_freq, self._log_scans)

    def _compute_metrics(self, ranges: List[float], angle_min: float,
                         angle_max: float, range_min: float,
                         range_max: float) -> tuple:
        """Compute scan metrics for comparison."""
        ranges_arr = np.array(ranges, dtype=np.float32)

        # Valid points (within sensor range, not inf/nan)
        valid_mask = (ranges_arr > range_min) & (ranges_arr < range_max) & np.isfinite(ranges_arr)
        valid_ranges = ranges_arr[valid_mask]
        valid_points = len(valid_ranges)

        if valid_points == 0:
            return 0, 0.0, 0.0, float('inf'), 0.0, 0.0

        # Angular coverage (degrees with valid readings)
        total_angle_deg = math.degrees(angle_max - angle_min)

        # Find contiguous segments of valid readings
        coverage_deg = (valid_points / len(ranges_arr)) * total_angle_deg

        # Range statistics
        mean_range = float(np.mean(valid_ranges))
        min_range = float(np.min(valid_ranges))
        max_range_seen = float(np.max(valid_ranges))

        # Density: valid points per degree
        density = valid_points / total_angle_deg if total_angle_deg > 0 else 0.0

        return valid_points, coverage_deg, mean_range, min_range, max_range_seen, density

    def _extract_scan(self, msg: LaserScan, source: str) -> ScanSnapshot:
        """Extract all data from a LaserScan message with metrics."""
        now = time.time()
        scan_stamp = stamp_to_float(msg.header.stamp)

        # Compute metrics
        valid_points, coverage_deg, mean_range, min_range, max_range_seen, density = \
            self._compute_metrics(
                msg.ranges, msg.angle_min, msg.angle_max,
                msg.range_min, msg.range_max
            )

        return ScanSnapshot(
            stamp=scan_stamp,
            frame_id=msg.header.frame_id,
            angle_min=msg.angle_min,
            angle_max=msg.angle_max,
            angle_increment=msg.angle_increment,
            time_increment=msg.time_increment,
            scan_time=msg.scan_time,
            range_min=msg.range_min,
            range_max=msg.range_max,
            ranges=list(msg.ranges),
            intensities=list(msg.intensities) if msg.intensities else [],
            num_points=len(msg.ranges),
            valid_points=valid_points,
            coverage_deg=coverage_deg,
            mean_range=mean_range,
            min_range=min_range,
            max_range_seen=max_range_seen,
            density=density,
            receive_time=now,
        )

    def _scan_cb(self, msg: LaserScan, source: str):
        """Callback for any scan topic."""
        self._scans[source] = self._extract_scan(msg, source)
        self._msg_counts[source] += 1

    def _scan_to_row(self, scan: ScanSnapshot, source: str) -> list:
        """Convert scan snapshot to CSV row."""
        now = time.time()
        latency = (now - scan.stamp) * 1000 if scan.stamp > 0 else -1

        # Convert ranges to semicolon-separated string (if enabled)
        ranges_str = ''
        if self._log_ranges:
            ranges_str = ';'.join(f'{r:.4f}' for r in scan.ranges)

        return [
            now,
            source,
            scan.stamp,
            scan.frame_id,
            scan.angle_min,
            scan.angle_max,
            scan.angle_increment,
            scan.range_min,
            scan.range_max,
            scan.num_points,
            scan.valid_points,
            f'{scan.coverage_deg:.2f}',
            f'{scan.mean_range:.3f}',
            f'{scan.min_range:.3f}',
            f'{scan.max_range_seen:.3f}',
            f'{scan.density:.2f}',
            f'{latency:.1f}',
            ranges_str,
        ]

    def _log_scans(self):
        """Log current scan data to buffer."""
        for source in self.SCAN_SOURCES.keys():
            if self._enabled_sources.get(source, False) and self._scans[source] is not None:
                self._buffer.append(self._scan_to_row(self._scans[source], source))
                self._rows_written += 1

        # Flush buffer periodically
        if len(self._buffer) >= self._buffer_size:
            self._flush_buffer()

    def _flush_buffer(self):
        """Write buffered rows to CSV."""
        if self._buffer:
            self._csv_writer.writerows(self._buffer)
            self._csv_file.flush()
            self._buffer.clear()

    def destroy_node(self):
        """Cleanup and print summary."""
        self._flush_buffer()
        self._csv_file.close()

        duration = time.time() - self._start_time

        self.get_logger().info('')
        self.get_logger().info('=' * 60)
        self.get_logger().info('SCAN DATA LOGGER SUMMARY')
        self.get_logger().info('=' * 60)
        self.get_logger().info(f'File: {self._csv_path}')
        self.get_logger().info(f'Duration: {duration:.1f}s')
        self.get_logger().info(f'Total rows: {self._rows_written}')
        self.get_logger().info('-' * 60)
        self.get_logger().info('Messages received per source:')
        for source, count in self._msg_counts.items():
            if self._enabled_sources.get(source, False):
                rate = count / duration if duration > 0 else 0
                self.get_logger().info(f'  {source:12s}: {count:6d} ({rate:.1f} Hz)')
        self.get_logger().info('=' * 60)
        self.get_logger().info('')
        self.get_logger().info('METRICS FOR FUSION ANALYSIS:')
        self.get_logger().info('  - Compare valid_points between sources')
        self.get_logger().info('  - coverage_deg shows angular coverage')
        self.get_logger().info('  - density shows points-per-degree')
        self.get_logger().info('  - Compare cameras vs lidar_only vs fused')
        self.get_logger().info('=' * 60)

        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ScanDataLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
