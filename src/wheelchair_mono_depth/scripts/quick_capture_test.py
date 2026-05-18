#!/usr/bin/env python3
"""Quick camera-only capture test — no odom, no full stack needed.

Captures N frames from each RealSense camera (RGB + aligned depth),
saves in the same format as data_collection_node.py. Use this to verify
the cameras work and test the downstream pipeline (DA3 inference, error
analysis) without the full wheelchair stack.

Usage:
    # Source ROS2 first, then:
    # Terminal 1: Launch cameras
    ros2 launch wheelchair_mono_depth data_collection.launch.py

    # Terminal 2: Run this script (cameras must already be streaming)
    python3 quick_capture_test.py --num_frames 20

    # Or standalone (launches own camera node for front only):
    python3 quick_capture_test.py --num_frames 10 --cameras front
"""

import argparse
import json
import os
import sys
import time
import threading
from datetime import datetime

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge


CAMERA_TOPICS = {
    'front': {
        'rgb': '/camera/color/image_raw',
        'depth': '/camera/aligned_depth_to_color/image_raw',
        'info': '/camera/color/camera_info',
    },
    'left': {
        'rgb': '/mapping_camera/color/image_raw',
        'depth': '/mapping_camera/aligned_depth_to_color/image_raw',
        'info': '/mapping_camera/color/camera_info',
    },
    'right': {
        'rgb': '/right_camera/color/image_raw',
        'depth': '/right_camera/aligned_depth_to_color/image_raw',
        'info': '/right_camera/color/camera_info',
    },
}

CAMERA_MAX_DEPTH_MM = {
    'front': 6000,
    'left': 6000,
    'right': 2500,
}


class QuickCaptureNode(Node):
    def __init__(self, cameras, num_frames, output_dir, save_rate_hz):
        super().__init__('quick_capture_test')
        self.bridge = CvBridge()
        self.cameras = cameras
        self.num_frames = num_frames
        self.save_rate_hz = save_rate_hz
        self.save_interval = 1.0 / save_rate_hz

        # Session setup
        session_id = 'test_' + datetime.now().strftime('%Y%m%d_%H%M%S')
        self.session_dir = os.path.join(output_dir, session_id)

        self.cam_dirs = {}
        for cam in cameras:
            cam_dir = os.path.join(self.session_dir, cam)
            os.makedirs(cam_dir, exist_ok=True)
            self.cam_dirs[cam] = cam_dir

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # State per camera
        self.frame_counts = {cam: 0 for cam in cameras}
        self.last_save_time = {cam: 0.0 for cam in cameras}
        self.intrinsics_saved = {cam: False for cam in cameras}
        self.latest_rgb = {cam: None for cam in cameras}
        self.latest_depth = {cam: None for cam in cameras}
        self.done = False

        # Subscribe
        for cam in cameras:
            topics = CAMERA_TOPICS[cam]
            self.create_subscription(
                CameraInfo, topics['info'],
                lambda msg, c=cam: self._info_cb(c, msg),
                sensor_qos)
            self.create_subscription(
                Image, topics['rgb'],
                lambda msg, c=cam: self._rgb_cb(c, msg),
                sensor_qos)
            self.create_subscription(
                Image, topics['depth'],
                lambda msg, c=cam: self._depth_cb(c, msg),
                sensor_qos)

        # Timer to save frames
        self.create_timer(0.05, self._save_timer)  # 20Hz check

        # Save metadata
        metadata = {
            'session_id': session_id,
            'created': datetime.now().isoformat(),
            'save_rate_hz': save_rate_hz,
            'cameras': cameras,
            'test_mode': True,
            'num_frames_target': num_frames,
        }
        with open(os.path.join(self.session_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

        self.get_logger().info(
            f'Quick capture: {session_id} | {cameras} | '
            f'{num_frames} frames @ {save_rate_hz}Hz')

    def _info_cb(self, cam, msg):
        if self.intrinsics_saved[cam]:
            return
        intrinsics = {
            'width': msg.width,
            'height': msg.height,
            'fx': msg.k[0],
            'fy': msg.k[4],
            'cx': msg.k[2],
            'cy': msg.k[5],
            'distortion_model': msg.distortion_model,
            'distortion_coefficients': list(msg.d),
            'frame_id': msg.header.frame_id,
        }
        path = os.path.join(self.cam_dirs[cam], 'intrinsics.json')
        with open(path, 'w') as f:
            json.dump(intrinsics, f, indent=2)
        self.intrinsics_saved[cam] = True
        self.get_logger().info(
            f'[{cam}] intrinsics: {msg.width}x{msg.height}, '
            f'fx={msg.k[0]:.1f}')

    def _rgb_cb(self, cam, msg):
        self.latest_rgb[cam] = msg

    def _depth_cb(self, cam, msg):
        self.latest_depth[cam] = msg

    def _save_timer(self):
        if self.done:
            return

        now = time.monotonic()
        all_done = True

        for cam in self.cameras:
            if self.frame_counts[cam] >= self.num_frames:
                continue
            all_done = False

            if now - self.last_save_time[cam] < self.save_interval:
                continue

            rgb_msg = self.latest_rgb[cam]
            depth_msg = self.latest_depth[cam]
            if rgb_msg is None or depth_msg is None:
                continue

            # Save frame
            frame_id = self.frame_counts[cam]
            cam_dir = self.cam_dirs[cam]

            try:
                rgb = self.bridge.imgmsg_to_cv2(rgb_msg, 'bgr8')
                depth = self.bridge.imgmsg_to_cv2(depth_msg, 'passthrough')
            except Exception as e:
                self.get_logger().warn(f'[{cam}] conversion error: {e}')
                continue

            # Clip depth by camera-specific max
            max_mm = CAMERA_MAX_DEPTH_MM.get(cam, 6000)
            depth = depth.astype(np.uint16)
            depth[depth > max_mm] = 0

            cv2.imwrite(os.path.join(cam_dir, f'{frame_id:06d}_rgb.png'), rgb)
            cv2.imwrite(os.path.join(cam_dir, f'{frame_id:06d}_depth.png'), depth)

            self.frame_counts[cam] = frame_id + 1
            self.last_save_time[cam] = now

            # Clear to avoid re-saving same message
            self.latest_rgb[cam] = None
            self.latest_depth[cam] = None

            if (frame_id + 1) % 5 == 0 or frame_id == 0:
                valid_pct = np.sum(depth > 0) / max(depth.size, 1) * 100
                self.get_logger().info(
                    f'[{cam}] frame {frame_id + 1}/{self.num_frames} | '
                    f'{rgb.shape[1]}x{rgb.shape[0]} | '
                    f'depth valid: {valid_pct:.0f}%')

        if all_done and not self.done:
            self.done = True
            total = sum(self.frame_counts.values())
            self.get_logger().info(
                f'\nCapture complete! {total} frames saved to:\n'
                f'  {self.session_dir}')
            for cam in self.cameras:
                self.get_logger().info(
                    f'  [{cam}] {self.frame_counts[cam]} frames')


def main():
    parser = argparse.ArgumentParser(description='Quick camera capture test')
    parser.add_argument('--num_frames', type=int, default=20,
                        help='Frames per camera')
    parser.add_argument('--cameras', nargs='*',
                        default=['front', 'left', 'right'])
    parser.add_argument('--output_dir',
                        default='/home/sidd/wheelchair_nav/mono_depth_data')
    parser.add_argument('--save_rate_hz', type=float, default=2.0,
                        help='Save rate (Hz)')
    args = parser.parse_args()

    rclpy.init()
    node = QuickCaptureNode(
        args.cameras, args.num_frames, args.output_dir, args.save_rate_hz)

    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass

    total = sum(node.frame_counts.values())
    print(f'\nDone: {total} frames in {node.session_dir}')

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
