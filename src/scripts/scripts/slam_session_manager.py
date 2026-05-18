#!/usr/bin/env python3
"""
SLAM Session Manager — auto-saves map + rosbag on Ctrl+C.

Starts rosbag recording at launch. On SIGINT:
  1. Calls slam_toolbox SerializePoseGraph (saves .posegraph + .data)
  2. Calls slam_toolbox SaveMap (saves .pgm + .yaml)
  3. Stops rosbag recording

Rosbag splits every 60s so data is on disk even if force-killed.

Output: ~/wheelchair_nav/maps/session_YYYYMMDD_HHMMSS/
"""

import os
import signal
import subprocess
import sys
from datetime import datetime

import rclpy
from rclpy.node import Node
from slam_toolbox.srv import SerializePoseGraph, SaveMap
from std_msgs.msg import String


class SlamSessionManager(Node):
    def __init__(self):
        super().__init__('slam_session_manager')

        self._timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self._base_dir = os.path.expanduser(
            f'~/wheelchair_nav/maps/session_{self._timestamp}'
        )
        os.makedirs(self._base_dir, exist_ok=True)

        self.get_logger().info(f'Session directory: {self._base_dir}')

        # Service clients
        self._serialize_client = self.create_client(
            SerializePoseGraph, '/slam_toolbox/serialize_map'
        )
        self._save_map_client = self.create_client(
            SaveMap, '/slam_toolbox/save_map'
        )

        # Topics to record — raw + filtered + fused scans + odometry + TF
        topics = [
            '/scan',
            '/scan_filtered',
            '/scan_fused',
            '/odometry/filtered',
            '/tf',
            '/tf_static',
        ]

        # Start rosbag recording with splitting so data persists on force-kill
        bag_output = os.path.join(self._base_dir, 'rosbag')
        bag_cmd = [
            'ros2', 'bag', 'record',
            '-o', bag_output,
            '--max-bag-duration', '60',  # Split every 60s — data on disk even if SIGKILL'd
            '--max-cache-size', '50000000',  # 50MB cache (smaller = more frequent flushes)
        ] + topics
        self._bag_proc = subprocess.Popen(
            bag_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        self.get_logger().info(
            f'Rosbag recording started (PID {self._bag_proc.pid}): '
            f'{len(topics)} topics, splitting every 60s'
        )

        # Handle SIGINT for graceful shutdown
        self._shutting_down = False
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        if self._shutting_down:
            return
        self._shutting_down = True
        self.get_logger().info('Shutdown signal received — saving session...')
        self._save_session()

    def _save_session(self):
        """Save posegraph, occupancy grid map, and stop rosbag."""
        map_path = os.path.join(self._base_dir, 'map')

        # 1. Serialize pose graph
        self.get_logger().info('Serializing pose graph...')
        if self._serialize_client.wait_for_service(timeout_sec=3.0):
            req = SerializePoseGraph.Request()
            req.filename = map_path
            future = self._serialize_client.call_async(req)
            rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
            if future.result() is not None and future.result().result == 0:
                self.get_logger().info(
                    f'Pose graph saved: {map_path}.posegraph + {map_path}.data'
                )
            else:
                self.get_logger().warn('Pose graph serialization failed or timed out')
        else:
            self.get_logger().warn('SerializePoseGraph service not available')

        # 2. Save occupancy grid map (.pgm + .yaml)
        self.get_logger().info('Saving occupancy grid map...')
        if self._save_map_client.wait_for_service(timeout_sec=3.0):
            req = SaveMap.Request()
            req.name = String(data=map_path)
            future = self._save_map_client.call_async(req)
            rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
            if future.result() is not None and future.result().result == 0:
                self.get_logger().info(f'Map saved: {map_path}.pgm + {map_path}.yaml')
            else:
                self.get_logger().warn('Map save failed or timed out')
        else:
            self.get_logger().warn('SaveMap service not available')

        # 3. Stop rosbag
        self.get_logger().info('Stopping rosbag recording...')
        if self._bag_proc and self._bag_proc.poll() is None:
            self._bag_proc.send_signal(signal.SIGINT)
            try:
                self._bag_proc.wait(timeout=10)
                self.get_logger().info('Rosbag stopped cleanly')
            except subprocess.TimeoutExpired:
                self._bag_proc.kill()
                self.get_logger().warn('Rosbag process killed after timeout')

        # List saved files
        self.get_logger().info(f'Session saved to: {self._base_dir}')
        try:
            for f in sorted(os.listdir(self._base_dir)):
                full = os.path.join(self._base_dir, f)
                if os.path.isdir(full):
                    bag_files = os.listdir(full)
                    self.get_logger().info(f'  {f}/ ({len(bag_files)} files)')
                else:
                    size_kb = os.path.getsize(full) / 1024
                    self.get_logger().info(f'  {f} ({size_kb:.1f} KB)')
        except Exception:
            pass

        # Exit
        rclpy.try_shutdown()
        sys.exit(0)


def main(args=None):
    rclpy.init(args=args)
    node = SlamSessionManager()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        if not node._shutting_down:
            node._save_session()
        node.destroy_node()
