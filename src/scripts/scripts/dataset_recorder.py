#!/usr/bin/env python3
"""
Dataset Recorder — MCAP Rosbag + Throttled Image Capture.

Records sensor streams (scans, odom, IMU) at native rates into MCAP rosbag.
Saves images to disk at a configurable interval (default: every 2s) for
diverse training data without filling disk with near-duplicate frames.

Designed to run alongside wheelchair_slam_mapping.launch.py with dataset_mode:=true.

Disk usage:
  Sensors (rosbag):  ~100 MB/hr  (scans + odom + IMU + TF)
  Images (JPG files): ~0.5 GB/hr (0.5Hz × 3 RGB streams × ~250KB)
  Total:             ~0.6 GB/hr

Output: /opt/nvidia/wheelchair_datasets/recording_YYYYMMDD_HHMMSS/
  rosbag/                   MCAP rosbag (sensors + depth pointclouds)
  images/front_rgb/         Front D455 color (every 2s)
  images/left_rgb/          Left D455 color
  images/right_rgb/         Right D435i color
  rates_log.csv             topic rate time series

Note: Depth is NOT saved as images — it's already captured real-time as
pointclouds by fusion and recorded in the rosbag. Enabling align_depth
saturates USB bandwidth and starves the fusion pointcloud stream.

Usage:
    ros2 run scripts dataset_recorder
    ros2 run scripts dataset_recorder --ros-args \
        -p output_dir:=/mnt/external/datasets \
        -p image_interval:=2.0 \
        -p max_bag_gb:=50.0 \
        -p min_disk_gb:=5.0
"""

import os
import shutil
import signal
import subprocess
import sys
import time
from collections import OrderedDict, defaultdict
from datetime import datetime

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)

from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CameraInfo, Image, Imu, LaserScan

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

# ========================================================================
# TOPIC CONFIGURATION
# ========================================================================

# Image topics — saved to disk at throttled rate (NOT in rosbag)
# Only RGB — depth is already captured real-time as pointclouds by fusion
# and recorded in the rosbag. Skipping depth images avoids enabling
# align_depth on the RealSense, which saturates USB and starves fusion.
# Each entry: (topic, subfolder_name, is_depth)
IMAGE_STREAMS = [
    ('/camera/color/image_raw',           'front_rgb',   False),
    ('/mapping_camera/color/image_raw',   'left_rgb',    False),
    ('/right_camera/color/image_raw',     'right_rgb',   False),
]

# Camera info — saved once per session for calibration
CAMERA_INFO_TOPICS = [
    '/camera/color/camera_info',
    '/mapping_camera/color/camera_info',
    '/right_camera/color/camera_info',
]

# Sensor topics — recorded in rosbag at native rates (tiny: ~100MB/hr)
SENSOR_TOPICS = [
    '/scan',
    '/scan_filtered',
    '/scan_fused',
    '/odometry/filtered',
    '/wc_control/odom',
    '/imu',
    '/camera/imu',
    '/cmd_vel',
]

EXTRA_RECORD_TOPICS = [
    '/tf', '/tf_static',
    '/goal_pose', '/plan',
]

# Topics to MONITOR for rate reporting
MONITOR_TOPICS = OrderedDict([
    ('/camera/color/image_raw',                    (Image,      'Front RGB',     6)),
    ('/mapping_camera/color/image_raw',             (Image,      'Left RGB',      6)),
    ('/right_camera/color/image_raw',               (Image,      'Right RGB',     6)),
    # Scan topics
    ('/scan',                (LaserScan, 'Scan Raw',       10)),
    ('/scan_filtered',       (LaserScan, 'Scan Filtered',  10)),
    ('/scan_fused',          (LaserScan, 'Scan Fused',     10)),
    # Odometry
    ('/odometry/filtered',   (Odometry,  'Odom EKF',       50)),
    ('/wc_control/odom',     (Odometry,  'Odom Raw',       20)),
    # IMU
    ('/imu',                 (Imu,       'IMU',           100)),
    ('/camera/imu',          (Imu,       'IMU Raw',       200)),
    # Control
    ('/cmd_vel',             (Twist,     'Cmd Vel',        20)),
])


def _get_disk_free_gb(path):
    """Get free disk space in GB for the filesystem containing path."""
    try:
        usage = shutil.disk_usage(path)
        return usage.free / (1024 ** 3)
    except Exception:
        return float('inf')


def _get_dir_size_gb(path):
    """Get total size of directory in GB."""
    total = 0
    try:
        for dirpath, _dirnames, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if os.path.isfile(fp):
                    total += os.path.getsize(fp)
    except Exception:
        pass
    return total / (1024 ** 3)


class DatasetRecorder(Node):
    def __init__(self):
        super().__init__('dataset_recorder')

        self.declare_parameter('output_dir',
                               '/opt/nvidia/wheelchair_datasets')
        self.declare_parameter('report_interval', 10.0)
        self.declare_parameter('min_disk_gb', 5.0)
        self.declare_parameter('max_bag_gb', 50.0)
        self.declare_parameter('warn_disk_gb', 10.0)
        self.declare_parameter('image_interval', 2.0)  # Save images every N seconds

        output_dir = self.get_parameter('output_dir').value
        self._report_interval = self.get_parameter('report_interval').value
        self._min_disk_gb = self.get_parameter('min_disk_gb').value
        self._max_bag_gb = self.get_parameter('max_bag_gb').value
        self._warn_disk_gb = self.get_parameter('warn_disk_gb').value
        self._image_interval = self.get_parameter('image_interval').value

        if not _HAS_CV2:
            self.get_logger().error('cv2 not found! Install: pip install opencv-python')
            rclpy.try_shutdown()
            sys.exit(1)

        self._bridge = CvBridge()

        # ============================================================
        # DISK SPACE CHECK — refuse to start if disk is too full
        # ============================================================
        free_gb = _get_disk_free_gb(output_dir)
        self.get_logger().info(f'Disk free: {free_gb:.1f} GB on {output_dir}')

        if free_gb < self._min_disk_gb:
            self.get_logger().error(
                f'INSUFFICIENT DISK SPACE: {free_gb:.1f} GB free '
                f'(need at least {self._min_disk_gb:.0f} GB). '
                f'Free up space or use: '
                f'--ros-args -p output_dir:=/mnt/external/datasets'
            )
            rclpy.try_shutdown()
            sys.exit(1)

        if free_gb < self._warn_disk_gb:
            self.get_logger().warn(
                f'LOW DISK: Only {free_gb:.1f} GB free. '
                f'Recording will auto-stop at {self._min_disk_gb:.0f} GB.'
            )

        # Session directory
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        self._session_dir = os.path.join(output_dir, f'recording_{ts}')
        os.makedirs(self._session_dir, exist_ok=True)

        # Create image directories
        self._image_dirs = {}
        for _topic, subfolder, _is_depth in IMAGE_STREAMS:
            d = os.path.join(self._session_dir, 'images', subfolder)
            os.makedirs(d, exist_ok=True)
            self._image_dirs[subfolder] = d

        # Image throttle state: last save time per stream
        self._last_image_save = {s[1]: 0.0 for s in IMAGE_STREAMS}
        self._image_save_count = {s[1]: 0 for s in IMAGE_STREAMS}
        self._camera_info_saved = set()

        # Rate counters
        self._counts = defaultdict(int)
        self._total_counts = defaultdict(int)
        self._last_report = time.monotonic()
        self._start_time = time.monotonic()

        # Rates CSV log
        self._rates_csv = os.path.join(self._session_dir, 'rates_log.csv')

        # Rosbag records ONLY sensor topics (no images — those go to disk)
        bag_topics = SENSOR_TOPICS + CAMERA_INFO_TOPICS + EXTRA_RECORD_TOPICS

        # Start rosbag recording (MCAP format)
        self._bag_path = os.path.join(self._session_dir, 'rosbag')
        bag_cmd = [
            'ros2', 'bag', 'record',
            '-s', 'mcap',
            '-o', self._bag_path,
            '--max-bag-duration', '300',
            '--max-cache-size', '50000000',
        ] + bag_topics

        self._bag_proc = subprocess.Popen(
            bag_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

        # QoS: BEST_EFFORT + depth=1 for images (latest only), depth=10 for monitoring
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        image_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,  # Only latest frame — we're sampling at 0.5Hz
        )

        # Subscribe to image topics for throttled saving
        for topic, subfolder, is_depth in IMAGE_STREAMS:
            self.create_subscription(
                Image, topic,
                lambda msg, sf=subfolder, dep=is_depth: self._image_cb(msg, sf, dep),
                image_qos,
            )

        # Subscribe to camera_info for one-time calibration save
        for topic in CAMERA_INFO_TOPICS:
            self.create_subscription(
                CameraInfo, topic,
                lambda msg, t=topic: self._camera_info_cb(msg, t),
                qos,
            )

        # Create subscribers for rate monitoring (non-image topics)
        for topic, (msg_type, _label, _expected) in MONITOR_TOPICS.items():
            # Image topics already subscribed above — just count in _image_cb
            if msg_type == Image:
                continue
            self.create_subscription(
                msg_type, topic,
                lambda _msg, t=topic: self._count_cb(t),
                qos_profile=qos,
            )

        # Rate report timer
        self._timer = self.create_timer(self._report_interval, self._report_rates)

        # Write CSV header
        with open(self._rates_csv, 'w') as f:
            labels = [MONITOR_TOPICS[t][1].replace(' ', '_') + '_hz'
                      for t in MONITOR_TOPICS]
            f.write('wall_time,elapsed_s,disk_free_gb,bag_size_gb,'
                    + ','.join(labels) + '\n')

        # Signal handling
        self._shutting_down = False
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        n_bag = len(bag_topics)
        n_img = len(IMAGE_STREAMS)
        self.get_logger().info(
            f'Dataset recorder started'
        )
        self.get_logger().info(f'  Rosbag: {n_bag} sensor topics (scans+odom+IMU+TF)')
        self.get_logger().info(f'  Images: {n_img} streams → disk every {self._image_interval:.1f}s')
        self.get_logger().info(f'  Est. disk: ~1.6 GB/hr (vs ~18 GB/hr at native rates)')
        self.get_logger().info(f'  Monitoring: {len(MONITOR_TOPICS)} topics')
        self.get_logger().info(f'  Output: {self._session_dir}')
        self.get_logger().info(f'  Rosbag PID: {self._bag_proc.pid}')
        self.get_logger().info(
            f'  Disk: stop<{self._min_disk_gb:.0f}GB, max bag {self._max_bag_gb:.0f}GB'
        )

    def _image_cb(self, msg: Image, subfolder: str, is_depth: bool):
        """Save image to disk if enough time has passed since last save."""
        # Find the original topic for rate counting (reverse lookup)
        for topic, sf, _dep in IMAGE_STREAMS:
            if sf == subfolder:
                self._counts[topic] += 1
                self._total_counts[topic] += 1
                break

        now = time.monotonic()
        if now - self._last_image_save[subfolder] < self._image_interval:
            return

        self._last_image_save[subfolder] = now

        try:
            if is_depth:
                # Depth: 16UC1 (millimeters) — save as 16-bit PNG (lossless)
                cv_img = self._bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
                ext = '.png'
            else:
                # RGB: save as JPEG (good compression, fine for training)
                cv_img = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                ext = '.jpg'

            # Filename: ROS timestamp for sync with rosbag sensor data
            stamp = msg.header.stamp
            ts_str = f'{stamp.sec}_{stamp.nanosec:09d}'
            filepath = os.path.join(self._image_dirs[subfolder], ts_str + ext)

            if is_depth:
                cv2.imwrite(filepath, cv_img)
            else:
                cv2.imwrite(filepath, cv_img, [cv2.IMWRITE_JPEG_QUALITY, 95])

            self._image_save_count[subfolder] += 1
        except Exception as e:
            self.get_logger().warn(
                f'Failed to save {subfolder}: {e}',
                throttle_duration_sec=10.0)

    def _camera_info_cb(self, msg: CameraInfo, topic: str):
        """Save camera intrinsics once per session."""
        if topic in self._camera_info_saved:
            return
        self._camera_info_saved.add(topic)

        # Save as YAML-like text file
        safe_name = topic.strip('/').replace('/', '_')
        filepath = os.path.join(self._session_dir, 'images', f'{safe_name}.txt')
        try:
            with open(filepath, 'w') as f:
                f.write(f'# Camera info for {topic}\n')
                f.write(f'width: {msg.width}\n')
                f.write(f'height: {msg.height}\n')
                f.write(f'distortion_model: {msg.distortion_model}\n')
                f.write(f'D: {list(msg.d)}\n')
                f.write(f'K: {list(msg.k)}\n')
                f.write(f'R: {list(msg.r)}\n')
                f.write(f'P: {list(msg.p)}\n')
                f.write(f'frame_id: {msg.header.frame_id}\n')
            self.get_logger().info(f'Saved camera info: {filepath}')
        except Exception as e:
            self.get_logger().warn(f'Failed to save camera info {topic}: {e}')

    def _count_cb(self, topic):
        self._counts[topic] += 1
        self._total_counts[topic] += 1

    def _report_rates(self):
        now = time.monotonic()
        dt = now - self._last_report
        elapsed = now - self._start_time
        self._last_report = now

        if dt < 0.1:
            return

        # Disk space check every report interval
        free_gb = _get_disk_free_gb(self._session_dir)
        bag_gb = _get_dir_size_gb(self._bag_path)

        if free_gb < self._min_disk_gb:
            self.get_logger().error(
                f'DISK FULL — only {free_gb:.1f} GB free! Auto-stopping.'
            )
            self._shutting_down = True
            self._shutdown()
            return

        if bag_gb >= self._max_bag_gb:
            self.get_logger().warn(
                f'Max bag size ({bag_gb:.1f}/{self._max_bag_gb:.0f} GB). Stopping.'
            )
            self._shutting_down = True
            self._shutdown()
            return

        # Check if rosbag subprocess is still alive
        if self._bag_proc.poll() is not None:
            self.get_logger().error(
                f'Rosbag process DIED (exit {self._bag_proc.returncode})'
            )
            try:
                stderr = self._bag_proc.stderr.read().decode()
                if stderr:
                    self.get_logger().error(f'stderr: {stderr[-500:]}')
            except Exception:
                pass
            self._shutting_down = True
            self._shutdown()
            return

        # Build rate table
        lines = [
            '',
            f'{"=" * 66}',
            f'  TOPIC RATES  (elapsed {elapsed:.0f}s, interval {dt:.1f}s)',
            f'  Disk: {free_gb:.1f} GB free | Bag: {bag_gb:.2f} GB',
            f'{"=" * 66}',
            f'  {"Topic":<18} {"Actual":>8} {"Expected":>8} {"Status":<6} {"Total":>9}',
            f'  {"-" * 55}',
        ]

        csv_values = [
            datetime.now().isoformat(),
            f'{elapsed:.1f}',
            f'{free_gb:.2f}',
            f'{bag_gb:.3f}',
        ]

        for topic, (_msg_type, label, expected_hz) in MONITOR_TOPICS.items():
            count = self._counts.get(topic, 0)
            hz = count / dt
            total = self._total_counts.get(topic, 0)

            if count == 0:
                status = 'NONE'
            elif hz < expected_hz * 0.4:
                status = 'LOW'
            else:
                status = 'OK'

            lines.append(
                f'  {label:<18} {hz:>7.1f}  {expected_hz:>7d}    {status:<6} {total:>9,}'
            )
            csv_values.append(f'{hz:.2f}')

        lines.append(f'  {"-" * 55}')

        if free_gb < self._warn_disk_gb:
            lines.append(
                f'  *** WARNING: Low disk! {free_gb:.1f} GB free '
                f'(stops at {self._min_disk_gb:.0f} GB) ***'
            )

        lines.append('')
        self.get_logger().info('\n'.join(lines))

        # Write CSV row
        with open(self._rates_csv, 'a') as f:
            f.write(','.join(csv_values) + '\n')

        # Reset interval counters
        self._counts.clear()

    def _signal_handler(self, signum, frame):
        if self._shutting_down:
            return
        self._shutting_down = True
        self.get_logger().info('Shutdown signal — stopping dataset recorder...')
        self._shutdown()

    def _shutdown(self):
        elapsed = time.monotonic() - self._start_time

        # Stop rosbag gracefully
        if self._bag_proc and self._bag_proc.poll() is None:
            self._bag_proc.send_signal(signal.SIGINT)
            try:
                self._bag_proc.wait(timeout=15)
                self.get_logger().info('Rosbag stopped cleanly')
            except subprocess.TimeoutExpired:
                self._bag_proc.kill()
                self.get_logger().warn('Rosbag killed after timeout')

        # Print summary
        bag_gb = _get_dir_size_gb(self._bag_path)
        img_gb = _get_dir_size_gb(os.path.join(self._session_dir, 'images'))
        lines = [
            '',
            '=' * 60,
            '  DATASET RECORDING SUMMARY',
            '=' * 60,
            f'  Duration:  {elapsed:.0f}s ({elapsed / 60:.1f} min)',
            f'  Bag size:  {bag_gb:.2f} GB (sensors only)',
            f'  Images:    {img_gb:.2f} GB (throttled to {self._image_interval:.1f}s)',
            f'  Total:     {bag_gb + img_gb:.2f} GB',
            f'  Output:    {self._session_dir}',
            '',
            f'  {"Stream":<18} {"Saved":>8} {"Avg Rate":>10}',
            '  ' + '-' * 42,
        ]
        for _topic, subfolder, _dep in IMAGE_STREAMS:
            count = self._image_save_count.get(subfolder, 0)
            rate = count / elapsed if elapsed > 0 else 0
            lines.append(f'  {subfolder:<18} {count:>8,} {rate:>9.2f} Hz')
        lines.append('')
        lines.append(f'  {"Topic":<18} {"Total Msgs":>12} {"Avg Hz":>8}')
        lines.append('  ' + '-' * 42)
        for topic, (_msg_type, label, _expected) in MONITOR_TOPICS.items():
            total = self._total_counts.get(topic, 0)
            avg_hz = total / elapsed if elapsed > 0 else 0
            lines.append(f'  {label:<18} {total:>12,} {avg_hz:>7.1f}')

        lines.append('  ' + '-' * 42)
        lines.append(f'  Rates log: {self._rates_csv}')
        lines.append('=' * 60)

        self.get_logger().info('\n'.join(lines))

        rclpy.try_shutdown()
        sys.exit(0)


def main(args=None):
    rclpy.init(args=args)
    node = DatasetRecorder()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        if not node._shutting_down:
            node._shutdown()
        node.destroy_node()
