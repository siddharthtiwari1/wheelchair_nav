#!/usr/bin/env python3
"""
RGB + Velocity Recorder — Time-synchronized image-velocity pairs + fusion diagnostics.

Uses message_filters.ApproximateTimeSynchronizer to match RGB and odometry
by header timestamp. Only saves pairs where timestamps are within 30ms (slop).

Subscribes to:
  - /logitech/image_raw (sensor_msgs/Image) — Logitech C270 RGB
  - /odometry/filtered (nav_msgs/Odometry) — EKF-fused pose + velocity
  - /wc_control/odom (nav_msgs/Odometry) — raw encoder odom [if record_diagnostics]
  - /imu (sensor_msgs/Imu) — fused IMU [if record_diagnostics]

Saves:
  output_dir/
    images/              RGB frames as JPEG (NNNNNN_timestamp_ns.jpg)
    velocities.csv       img_ts_ns, odom_ts_ns, offset_ms, v_actual, omega_actual, x, y, theta
    odom_highrate.csv    full-rate EKF odom (timestamp_ns, v, omega, x, y, theta)
    raw_encoder.csv      full-rate raw encoder odom [if record_diagnostics]
    imu.csv              full-rate IMU data [if record_diagnostics]
    metadata.json        recording params, start/end time, frame count, avg sync offset

Sync guarantee: image and odom timestamps differ by at most 30ms (configurable slop).

Usage:
    ros2 run scripts rgb_velocity_recorder --ros-args \
        -p output_dir:=/home/sidd/wheelchair_nav/data/run_01 \
        -p save_fps:=10.0 \
        -p sync_slop:=0.03 \
        -p record_diagnostics:=true
"""

import csv
import json
import math
import os
import time
from datetime import datetime

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

import message_filters
from cv_bridge import CvBridge
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, Imu


def quaternion_to_yaw(q):
    """Extract yaw from quaternion (x, y, z, w)."""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def stamp_to_ns(stamp):
    """Convert ROS2 Time stamp to nanoseconds."""
    return stamp.sec * 10**9 + stamp.nanosec


class RGBVelocityRecorder(Node):
    def __init__(self):
        super().__init__('rgb_velocity_recorder')

        # Parameters
        self.declare_parameter('output_dir', '')
        self.declare_parameter('save_fps', 10.0)
        self.declare_parameter('image_quality', 90)
        self.declare_parameter('image_topic', '/logitech/image_raw')
        self.declare_parameter('odom_topic', '/odometry/filtered')
        self.declare_parameter('sync_slop', 0.03)  # 30ms max offset
        self.declare_parameter('record_diagnostics', False)
        self.declare_parameter('raw_odom_topic', '/wc_control/odom')
        self.declare_parameter('imu_topic', '/imu')

        output_dir = self.get_parameter('output_dir').value
        if not output_dir:
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = os.path.expanduser(
                f'~/wheelchair_nav/data/rgb_vel_{ts}'
            )

        self.output_dir = output_dir
        self.save_fps = self.get_parameter('save_fps').value
        self.image_quality = self.get_parameter('image_quality').value
        image_topic = self.get_parameter('image_topic').value
        odom_topic = self.get_parameter('odom_topic').value
        sync_slop = self.get_parameter('sync_slop').value
        self.record_diagnostics = self.get_parameter('record_diagnostics').value
        raw_odom_topic = self.get_parameter('raw_odom_topic').value
        imu_topic = self.get_parameter('imu_topic').value

        self.save_interval = 1.0 / self.save_fps
        self.last_save_time = 0.0

        # State
        self.bridge = CvBridge()
        self.frame_count = 0
        self.start_time = None
        self.sync_offsets = []  # track sync quality
        self.highrate_odom_count = 0
        self.raw_encoder_count = 0
        self.imu_count = 0

        # Create output directories
        self.image_dir = os.path.join(self.output_dir, 'images')
        os.makedirs(self.image_dir, exist_ok=True)

        # Open CSV with both timestamps + offset
        self.csv_path = os.path.join(self.output_dir, 'velocities.csv')
        self.csv_file = open(self.csv_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            'img_timestamp_ns', 'odom_timestamp_ns', 'offset_ms',
            'frame_id',
            'v_actual', 'omega_actual',
            'x', 'y', 'theta',
        ])

        # High-rate EKF odom CSV (every message, not throttled)
        self.highrate_csv_path = os.path.join(self.output_dir, 'odom_highrate.csv')
        self.highrate_csv_file = open(self.highrate_csv_path, 'w', newline='')
        self.highrate_csv_writer = csv.writer(self.highrate_csv_file)
        self.highrate_csv_writer.writerow([
            'timestamp_ns', 'v', 'omega', 'x', 'y', 'theta',
        ])

        # QoS — BEST_EFFORT for sensor topics
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,  # buffer for sync matching
        )

        # message_filters subscribers (for synced image+odom)
        img_sub = message_filters.Subscriber(
            self, Image, image_topic, qos_profile=sensor_qos)
        odom_sub = message_filters.Subscriber(
            self, Odometry, odom_topic, qos_profile=sensor_qos)

        # ApproximateTimeSynchronizer — matches by header timestamps
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [img_sub, odom_sub],
            queue_size=10,
            slop=sync_slop,
        )
        self.sync.registerCallback(self._synced_cb)

        # High-rate EKF odom subscriber (unsynchronized, every message)
        self.create_subscription(
            Odometry, odom_topic, self._highrate_odom_cb, sensor_qos)

        # Diagnostics: raw encoder odom + IMU at full rate
        if self.record_diagnostics:
            self.raw_csv_path = os.path.join(self.output_dir, 'raw_encoder.csv')
            self.raw_csv_file = open(self.raw_csv_path, 'w', newline='')
            self.raw_csv_writer = csv.writer(self.raw_csv_file)
            self.raw_csv_writer.writerow([
                'timestamp_ns', 'v', 'omega', 'x', 'y', 'theta',
            ])
            self.create_subscription(
                Odometry, raw_odom_topic, self._raw_odom_cb, sensor_qos)

            self.imu_csv_path = os.path.join(self.output_dir, 'imu.csv')
            self.imu_csv_file = open(self.imu_csv_path, 'w', newline='')
            self.imu_csv_writer = csv.writer(self.imu_csv_file)
            self.imu_csv_writer.writerow([
                'timestamp_ns',
                'ax', 'ay', 'az',
                'gx', 'gy', 'gz',
                'qx', 'qy', 'qz', 'qw',
            ])
            self.create_subscription(
                Imu, imu_topic, self._imu_cb, sensor_qos)

        self.get_logger().info(f'RGB+Velocity Recorder (TIME-SYNCED)')
        self.get_logger().info(f'  Output:    {self.output_dir}')
        self.get_logger().info(f'  Image:     {image_topic}')
        self.get_logger().info(f'  Odom:      {odom_topic}')
        self.get_logger().info(f'  Sync slop: {sync_slop*1000:.0f}ms')
        self.get_logger().info(f'  Save at:   {self.save_fps} fps')
        self.get_logger().info(f'  High-rate: odom_highrate.csv (every msg)')
        if self.record_diagnostics:
            self.get_logger().info(f'  Diagnostics: raw_encoder.csv + imu.csv')
            self.get_logger().info(f'    Raw odom: {raw_odom_topic}')
            self.get_logger().info(f'    IMU:      {imu_topic}')
        self.get_logger().info('Waiting for synced data...')

    def _highrate_odom_cb(self, msg):
        """Record every EKF odom message at full rate."""
        ts = stamp_to_ns(msg.header.stamp)
        v = msg.twist.twist.linear.x
        omega = msg.twist.twist.angular.z
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        theta = quaternion_to_yaw(msg.pose.pose.orientation)
        self.highrate_csv_writer.writerow([
            ts, f'{v:.6f}', f'{omega:.6f}',
            f'{x:.6f}', f'{y:.6f}', f'{theta:.6f}',
        ])
        self.highrate_odom_count += 1

    def _raw_odom_cb(self, msg):
        """Record every raw encoder odom message."""
        ts = stamp_to_ns(msg.header.stamp)
        v = msg.twist.twist.linear.x
        omega = msg.twist.twist.angular.z
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        theta = quaternion_to_yaw(msg.pose.pose.orientation)
        self.raw_csv_writer.writerow([
            ts, f'{v:.6f}', f'{omega:.6f}',
            f'{x:.6f}', f'{y:.6f}', f'{theta:.6f}',
        ])
        self.raw_encoder_count += 1

    def _imu_cb(self, msg):
        """Record every IMU message."""
        ts = stamp_to_ns(msg.header.stamp)
        self.imu_csv_writer.writerow([
            ts,
            f'{msg.linear_acceleration.x:.6f}',
            f'{msg.linear_acceleration.y:.6f}',
            f'{msg.linear_acceleration.z:.6f}',
            f'{msg.angular_velocity.x:.6f}',
            f'{msg.angular_velocity.y:.6f}',
            f'{msg.angular_velocity.z:.6f}',
            f'{msg.orientation.x:.6f}',
            f'{msg.orientation.y:.6f}',
            f'{msg.orientation.z:.6f}',
            f'{msg.orientation.w:.6f}',
        ])
        self.imu_count += 1

    def _synced_cb(self, img_msg, odom_msg):
        """Called ONLY when image and odom are matched by timestamp."""
        now = time.monotonic()
        if now - self.last_save_time < self.save_interval:
            return

        self.last_save_time = now

        if self.start_time is None:
            self.start_time = datetime.now().isoformat()
            self.get_logger().info('Recording started (synced pairs)!')

        # Convert image
        try:
            cv_image = self.bridge.imgmsg_to_cv2(img_msg, 'bgr8')
        except Exception as e:
            self.get_logger().warn(f'Image conversion failed: {e}')
            return

        # Timestamps
        img_ts_ns = stamp_to_ns(img_msg.header.stamp)
        odom_ts_ns = stamp_to_ns(odom_msg.header.stamp)
        offset_ms = abs(img_ts_ns - odom_ts_ns) / 1e6
        self.sync_offsets.append(offset_ms)

        frame_id = f'{self.frame_count:06d}'

        # Save image
        img_path = os.path.join(self.image_dir, f'{frame_id}_{img_ts_ns}.jpg')
        cv2.imwrite(img_path, cv_image,
                     [cv2.IMWRITE_JPEG_QUALITY, self.image_quality])

        # Extract odom
        v_actual = odom_msg.twist.twist.linear.x
        omega_actual = odom_msg.twist.twist.angular.z
        x = odom_msg.pose.pose.position.x
        y = odom_msg.pose.pose.position.y
        theta = quaternion_to_yaw(odom_msg.pose.pose.orientation)

        # Write CSV row
        self.csv_writer.writerow([
            img_ts_ns, odom_ts_ns, f'{offset_ms:.2f}',
            frame_id,
            f'{v_actual:.6f}', f'{omega_actual:.6f}',
            f'{x:.6f}', f'{y:.6f}', f'{theta:.6f}',
        ])

        self.frame_count += 1

        if self.frame_count % 100 == 0:
            avg_offset = np.mean(self.sync_offsets[-100:])
            self.get_logger().info(
                f'Saved {self.frame_count} synced frames | '
                f'avg_offset={avg_offset:.1f}ms | '
                f'v={v_actual:.3f} w={omega_actual:.3f} | '
                f'pos=({x:.2f}, {y:.2f}, {math.degrees(theta):.1f}deg)'
            )

    def destroy_node(self):
        # Flush and close all CSV files
        self.csv_file.close()
        self.highrate_csv_file.close()
        if self.record_diagnostics:
            self.raw_csv_file.close()
            self.imu_csv_file.close()

        avg_offset = float(np.mean(self.sync_offsets)) if self.sync_offsets else 0.0
        max_offset = float(np.max(self.sync_offsets)) if self.sync_offsets else 0.0

        metadata = {
            'start_time': self.start_time,
            'end_time': datetime.now().isoformat(),
            'total_frames': self.frame_count,
            'save_fps': self.save_fps,
            'image_quality': self.image_quality,
            'sync_slop_ms': self.get_parameter('sync_slop').value * 1000,
            'avg_sync_offset_ms': round(avg_offset, 2),
            'max_sync_offset_ms': round(max_offset, 2),
            'highrate_odom_samples': self.highrate_odom_count,
            'record_diagnostics': self.record_diagnostics,
            'raw_encoder_samples': self.raw_encoder_count,
            'imu_samples': self.imu_count,
        }
        meta_path = os.path.join(self.output_dir, 'metadata.json')
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        self.get_logger().info(
            f'Recording complete: {self.frame_count} synced frames | '
            f'avg offset: {avg_offset:.1f}ms | max: {max_offset:.1f}ms'
        )
        self.get_logger().info(
            f'  High-rate odom: {self.highrate_odom_count} samples')
        if self.record_diagnostics:
            self.get_logger().info(
                f'  Raw encoder: {self.raw_encoder_count} samples | '
                f'IMU: {self.imu_count} samples')
        self.get_logger().info(f'Saved to {self.output_dir}')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = RGBVelocityRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
