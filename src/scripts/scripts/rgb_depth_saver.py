#!/usr/bin/env python3
"""
RGB+Depth Dataset Capture Node.

Saves timestamped RGB+depth image pairs from all 3 RealSense cameras
at a configurable rate. For training end-to-end navigation models.

Output: ~/wheelchair_nav_datasets/capture_YYYYMMDD_HHMMSS/
  front_camera/rgb/   depth/  timestamps.csv
  left_camera/rgb/    depth/  timestamps.csv
  right_camera/rgb/   depth/  timestamps.csv

Launch: ros2 run scripts rgb_depth_saver --ros-args -p save_rate:=2.0
"""

import os
from datetime import datetime

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image


class CameraCapture:
    """Manages capture for a single camera."""

    def __init__(self, name, rgb_dir, depth_dir, csv_path):
        self.name = name
        self.rgb_dir = rgb_dir
        self.depth_dir = depth_dir
        self.csv_path = csv_path
        self.latest_rgb = None
        self.latest_depth = None
        self.rgb_stamp = None
        self.depth_stamp = None
        self.count = 0

        os.makedirs(rgb_dir, exist_ok=True)
        os.makedirs(depth_dir, exist_ok=True)
        with open(csv_path, 'w') as f:
            f.write('index,timestamp_sec,timestamp_nsec,rgb_file,depth_file\n')


class RgbDepthSaver(Node):
    def __init__(self):
        super().__init__('rgb_depth_saver')

        self.declare_parameter('save_rate', 2.0)
        self.declare_parameter('base_dir', os.path.expanduser('~/wheelchair_nav_datasets'))

        save_rate = self.get_parameter('save_rate').value
        base_dir = self.get_parameter('base_dir').value

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self._session_dir = os.path.join(base_dir, f'capture_{timestamp}')

        self._bridge = CvBridge()

        # Camera configs: (name, rgb_topic, depth_topic)
        cameras = [
            ('front_camera', '/camera/color/image_raw', '/camera/aligned_depth_to_color/image_raw'),
            ('left_camera', '/mapping_camera/color/image_raw', '/mapping_camera/aligned_depth_to_color/image_raw'),
            ('right_camera', '/right_camera/color/image_raw', '/right_camera/aligned_depth_to_color/image_raw'),
        ]

        self._captures = {}
        for cam_name, rgb_topic, depth_topic in cameras:
            cam_dir = os.path.join(self._session_dir, cam_name)
            cap = CameraCapture(
                name=cam_name,
                rgb_dir=os.path.join(cam_dir, 'rgb'),
                depth_dir=os.path.join(cam_dir, 'depth'),
                csv_path=os.path.join(cam_dir, 'timestamps.csv'),
            )
            self._captures[cam_name] = cap

            self.create_subscription(
                Image, rgb_topic,
                lambda msg, c=cap: self._rgb_cb(msg, c),
                10,
            )
            self.create_subscription(
                Image, depth_topic,
                lambda msg, c=cap: self._depth_cb(msg, c),
                10,
            )

        self._timer = self.create_timer(1.0 / save_rate, self._save_tick)
        self._total_saved = 0

        self.get_logger().info(f'RGB+Depth saver started at {save_rate} Hz')
        self.get_logger().info(f'Output: {self._session_dir}')

    def _rgb_cb(self, msg: Image, cap: CameraCapture):
        cap.latest_rgb = msg
        cap.rgb_stamp = msg.header.stamp

    def _depth_cb(self, msg: Image, cap: CameraCapture):
        cap.latest_depth = msg
        cap.depth_stamp = msg.header.stamp

    def _save_tick(self):
        for cap in self._captures.values():
            if cap.latest_rgb is None or cap.latest_depth is None:
                continue

            try:
                rgb_img = self._bridge.imgmsg_to_cv2(cap.latest_rgb, 'bgr8')
                depth_img = self._bridge.imgmsg_to_cv2(cap.latest_depth, 'passthrough')
            except Exception as e:
                self.get_logger().warn(f'{cap.name}: conversion error: {e}')
                continue

            idx = cap.count
            stamp = cap.rgb_stamp

            rgb_file = f'{idx:06d}.jpg'
            depth_file = f'{idx:06d}.png'

            cv2.imwrite(os.path.join(cap.rgb_dir, rgb_file), rgb_img)
            # Save depth as 16-bit PNG (millimeters)
            if depth_img.dtype == np.float32:
                depth_img = (depth_img * 1000).astype(np.uint16)
            cv2.imwrite(os.path.join(cap.depth_dir, depth_file), depth_img)

            with open(cap.csv_path, 'a') as f:
                f.write(f'{idx},{stamp.sec},{stamp.nanosec},{rgb_file},{depth_file}\n')

            cap.count += 1
            self._total_saved += 1

            # Clear to avoid re-saving stale frames
            cap.latest_rgb = None
            cap.latest_depth = None

        if self._total_saved > 0 and self._total_saved % 50 == 0:
            self.get_logger().info(f'Saved {self._total_saved} total frames')


def main(args=None):
    rclpy.init(args=args)
    node = RgbDepthSaver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info(
            f'Capture complete: {node._total_saved} total frames in {node._session_dir}'
        )
        node.destroy_node()
        rclpy.try_shutdown()
