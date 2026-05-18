#!/usr/bin/env python3
"""Synchronized RGB + depth data collection from 3 RealSense cameras.

Captures paired RGB-depth images from front/left/right cameras along with
odometry for training monocular depth estimation models. Uses RealSense
hardware depth as ground truth.

Features:
- Odometry-based deduplication: skips frames if robot hasn't moved enough
- Per-camera max depth: D455=6000mm, D435i=2500mm (noise-aware)
- Rate-limited synchronized capture from all 3 cameras

Output structure:
    {output_dir}/{session_id}/{camera_name}/{frame:06d}_rgb.png
    {output_dir}/{session_id}/{camera_name}/{frame:06d}_depth.png
    {output_dir}/{session_id}/{camera_name}/{frame:06d}_odom.json
    {output_dir}/{session_id}/{camera_name}/intrinsics.json
"""

import os
import json
import math
import time
from datetime import datetime

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import message_filters


CAMERAS = ['front', 'left', 'right']

# Per-camera max depth in mm (based on RealSense noise characteristics)
# D455: reliable up to ~6m (36mm error @3m, 49mm @3.5m)
# D435i: reliable up to ~2.5m (68mm error @3m, 92mm @3.5m — 1.9x worse)
CAMERA_MAX_DEPTH_MM = {
    'front': 6000,   # D455
    'left': 6000,    # D455
    'right': 2500,   # D435i
}


def _quaternion_to_yaw(q):
    """Extract yaw from quaternion (x, y, z, w)."""
    siny_cosp = 2.0 * (q['w'] * q['z'] + q['x'] * q['y'])
    cosy_cosp = 1.0 - 2.0 * (q['y'] ** 2 + q['z'] ** 2)
    return math.atan2(siny_cosp, cosy_cosp)


class DepthDataCollectionNode(Node):
    def __init__(self):
        super().__init__('depth_data_collection_node')

        # Parameters
        self.declare_parameter('save_rate_hz', 3.0)
        self.declare_parameter('output_dir', '/home/sidd/wheelchair_nav/mono_depth_data')
        self.declare_parameter('session_id', '')
        self.declare_parameter('max_depth_mm', 10000)
        self.declare_parameter('min_depth_mm', 100)
        self.declare_parameter('sync_queue_size', 10)
        self.declare_parameter('sync_slop', 0.15)
        self.declare_parameter('dedup_min_distance', 0.01)
        self.declare_parameter('dedup_min_angle_deg', 1.0)

        # Camera topics
        self.declare_parameter('front_rgb_topic', '/camera/color/image_raw')
        self.declare_parameter('front_depth_topic', '/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('front_info_topic', '/camera/color/camera_info')
        self.declare_parameter('left_rgb_topic', '/mapping_camera/color/image_raw')
        self.declare_parameter('left_depth_topic', '/mapping_camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('left_info_topic', '/mapping_camera/color/camera_info')
        self.declare_parameter('right_rgb_topic', '/right_camera/color/image_raw')
        self.declare_parameter('right_depth_topic', '/right_camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('right_info_topic', '/right_camera/color/camera_info')
        self.declare_parameter('odom_topic', '/odometry/filtered')

        # Read parameters
        self.save_rate_hz = self.get_parameter('save_rate_hz').value
        self.output_dir = self.get_parameter('output_dir').value
        self.max_depth_mm = self.get_parameter('max_depth_mm').value
        self.min_depth_mm = self.get_parameter('min_depth_mm').value
        sync_queue = self.get_parameter('sync_queue_size').value
        sync_slop = self.get_parameter('sync_slop').value
        self.dedup_min_distance = self.get_parameter('dedup_min_distance').value
        self.dedup_min_angle = math.radians(
            self.get_parameter('dedup_min_angle_deg').value
        )

        session_id = self.get_parameter('session_id').value
        if not session_id:
            session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.session_dir = os.path.join(self.output_dir, session_id)

        # Create output directories
        self.cam_dirs = {}
        for cam in CAMERAS:
            cam_dir = os.path.join(self.session_dir, cam)
            os.makedirs(cam_dir, exist_ok=True)
            self.cam_dirs[cam] = cam_dir

        self.bridge = CvBridge()
        self.frame_count = 0
        self.skipped_dedup = 0
        self.last_save_time = 0.0
        self.save_interval = 1.0 / self.save_rate_hz
        self.intrinsics_saved = {cam: False for cam in CAMERAS}

        # Odometry dedup state
        self.last_saved_pos = None  # (x, y)
        self.last_saved_yaw = None

        # QoS for sensor data
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        # Subscribe to CameraInfo for intrinsics (one-shot per camera)
        info_topics = {
            'front': self.get_parameter('front_info_topic').value,
            'left': self.get_parameter('left_info_topic').value,
            'right': self.get_parameter('right_info_topic').value,
        }
        self.info_subs = {}
        for cam, topic in info_topics.items():
            self.info_subs[cam] = self.create_subscription(
                CameraInfo, topic,
                lambda msg, c=cam: self._save_intrinsics(c, msg),
                sensor_qos,
            )

        # Synchronized subscribers for RGB + depth + odom
        rgb_topics = {
            'front': self.get_parameter('front_rgb_topic').value,
            'left': self.get_parameter('left_rgb_topic').value,
            'right': self.get_parameter('right_rgb_topic').value,
        }
        depth_topics = {
            'front': self.get_parameter('front_depth_topic').value,
            'left': self.get_parameter('left_depth_topic').value,
            'right': self.get_parameter('right_depth_topic').value,
        }
        odom_topic = self.get_parameter('odom_topic').value

        # Create message filter subscribers
        self.front_rgb_sub = message_filters.Subscriber(
            self, Image, rgb_topics['front'], qos_profile=sensor_qos)
        self.front_depth_sub = message_filters.Subscriber(
            self, Image, depth_topics['front'], qos_profile=sensor_qos)
        self.left_rgb_sub = message_filters.Subscriber(
            self, Image, rgb_topics['left'], qos_profile=sensor_qos)
        self.left_depth_sub = message_filters.Subscriber(
            self, Image, depth_topics['left'], qos_profile=sensor_qos)
        self.right_rgb_sub = message_filters.Subscriber(
            self, Image, rgb_topics['right'], qos_profile=sensor_qos)
        self.right_depth_sub = message_filters.Subscriber(
            self, Image, depth_topics['right'], qos_profile=sensor_qos)
        self.odom_sub = message_filters.Subscriber(
            self, Odometry, odom_topic, qos_profile=sensor_qos)

        self.sync = message_filters.ApproximateTimeSynchronizer(
            [
                self.front_rgb_sub, self.front_depth_sub,
                self.left_rgb_sub, self.left_depth_sub,
                self.right_rgb_sub, self.right_depth_sub,
                self.odom_sub,
            ],
            queue_size=sync_queue,
            slop=sync_slop,
        )
        self.sync.registerCallback(self._sync_callback)

        # Save session metadata
        metadata = {
            'session_id': session_id,
            'created': datetime.now().isoformat(),
            'save_rate_hz': self.save_rate_hz,
            'max_depth_mm': self.max_depth_mm,
            'min_depth_mm': self.min_depth_mm,
            'camera_max_depth_mm': CAMERA_MAX_DEPTH_MM,
            'dedup_min_distance': self.dedup_min_distance,
            'dedup_min_angle_deg': math.degrees(self.dedup_min_angle),
            'cameras': CAMERAS,
            'rgb_topics': rgb_topics,
            'depth_topics': depth_topics,
            'odom_topic': odom_topic,
        }
        with open(os.path.join(self.session_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

        self.get_logger().info(
            f'Data collection started: {self.session_dir} @ {self.save_rate_hz} Hz '
            f'(dedup: {self.dedup_min_distance}m / '
            f'{math.degrees(self.dedup_min_angle):.1f}deg)')

    def _save_intrinsics(self, cam_name, msg: CameraInfo):
        """Save camera intrinsics once per session."""
        if self.intrinsics_saved[cam_name]:
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
        path = os.path.join(self.cam_dirs[cam_name], 'intrinsics.json')
        with open(path, 'w') as f:
            json.dump(intrinsics, f, indent=2)

        self.intrinsics_saved[cam_name] = True
        self.get_logger().info(f'Saved {cam_name} intrinsics: {msg.width}x{msg.height}')

        # Unsubscribe after saving
        if all(self.intrinsics_saved.values()):
            for sub in self.info_subs.values():
                self.destroy_subscription(sub)
            self.info_subs.clear()
            self.get_logger().info('All camera intrinsics saved')

    def _check_dedup(self, odom):
        """Check if robot has moved enough since last saved frame.

        Returns True if frame should be saved (enough motion detected).
        First frame is always saved.
        """
        pos = odom.pose.pose.position
        orient = odom.pose.pose.orientation
        current_pos = (pos.x, pos.y)
        current_yaw = _quaternion_to_yaw({
            'x': orient.x, 'y': orient.y,
            'z': orient.z, 'w': orient.w,
        })

        if self.last_saved_pos is None:
            # First frame — always save
            self.last_saved_pos = current_pos
            self.last_saved_yaw = current_yaw
            return True

        dx = current_pos[0] - self.last_saved_pos[0]
        dy = current_pos[1] - self.last_saved_pos[1]
        dist = math.sqrt(dx * dx + dy * dy)

        # Angle difference (handle wraparound)
        angle_diff = abs(current_yaw - self.last_saved_yaw)
        if angle_diff > math.pi:
            angle_diff = 2 * math.pi - angle_diff

        if dist >= self.dedup_min_distance or angle_diff >= self.dedup_min_angle:
            self.last_saved_pos = current_pos
            self.last_saved_yaw = current_yaw
            return True

        return False

    def _sync_callback(self, front_rgb, front_depth, left_rgb, left_depth,
                       right_rgb, right_depth, odom):
        """Handle synchronized messages from all cameras + odometry."""
        now = time.monotonic()
        if now - self.last_save_time < self.save_interval:
            return
        self.last_save_time = now

        # Odometry-based dedup: skip if robot hasn't moved
        if not self._check_dedup(odom):
            self.skipped_dedup += 1
            return

        frame_id = f'{self.frame_count:06d}'

        # Prepare odom data
        odom_data = {
            'timestamp': odom.header.stamp.sec + odom.header.stamp.nanosec * 1e-9,
            'frame_id': odom.header.frame_id,
            'position': {
                'x': odom.pose.pose.position.x,
                'y': odom.pose.pose.position.y,
                'z': odom.pose.pose.position.z,
            },
            'orientation': {
                'x': odom.pose.pose.orientation.x,
                'y': odom.pose.pose.orientation.y,
                'z': odom.pose.pose.orientation.z,
                'w': odom.pose.pose.orientation.w,
            },
            'linear_velocity': {
                'x': odom.twist.twist.linear.x,
                'y': odom.twist.twist.linear.y,
                'z': odom.twist.twist.linear.z,
            },
            'angular_velocity': {
                'x': odom.twist.twist.angular.x,
                'y': odom.twist.twist.angular.y,
                'z': odom.twist.twist.angular.z,
            },
        }

        # Save each camera's data
        cam_data = {
            'front': (front_rgb, front_depth),
            'left': (left_rgb, left_depth),
            'right': (right_rgb, right_depth),
        }

        for cam_name, (rgb_msg, depth_msg) in cam_data.items():
            self._save_frame(cam_name, frame_id, rgb_msg, depth_msg, odom_data)

        self.frame_count += 1
        if self.frame_count % 30 == 0:
            self.get_logger().info(
                f'Collected {self.frame_count} frames '
                f'({self.frame_count * 3} image pairs, '
                f'{self.skipped_dedup} skipped by dedup)')

    def _save_frame(self, cam_name, frame_id, rgb_msg, depth_msg, odom_data):
        """Save one RGB-depth pair + odometry to disk."""
        cam_dir = self.cam_dirs[cam_name]

        # Convert RGB
        try:
            rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f'{cam_name} RGB convert failed: {e}')
            return

        # Convert depth (16-bit uint16, millimeters)
        try:
            depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().warn(f'{cam_name} depth convert failed: {e}')
            return

        # Ensure depth is uint16
        if depth.dtype != np.uint16:
            depth = depth.astype(np.uint16)

        # Per-camera max depth (noise-aware)
        cam_max = CAMERA_MAX_DEPTH_MM.get(cam_name, self.max_depth_mm)

        # Clip depth to valid range
        depth = np.where(depth < self.min_depth_mm, 0, depth)
        depth = np.where(depth > cam_max, 0, depth)

        # Save files
        cv2.imwrite(os.path.join(cam_dir, f'{frame_id}_rgb.png'), rgb)
        cv2.imwrite(os.path.join(cam_dir, f'{frame_id}_depth.png'), depth)

        with open(os.path.join(cam_dir, f'{frame_id}_odom.json'), 'w') as f:
            json.dump(odom_data, f)


def main(args=None):
    rclpy.init(args=args)
    node = DepthDataCollectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info(
            f'Collection complete: {node.frame_count} frames saved to {node.session_dir} '
            f'({node.skipped_dedup} skipped by dedup)')
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
