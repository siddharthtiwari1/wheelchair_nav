#!/usr/bin/env python3
"""
ROSBAG FRAME EXTRACTOR FOR GROUND TRUTH LABELING
==================================================

Extracts synchronized depth (PointCloud2) and RGB (Image) frames from rosbags
to enable manual labeling of phantom obstacles vs. true obstacles.

Workflow:
1. Select 3 diverse rosbags (different dates/times)
2. Extract ~10 frames per rosbag (start, middle, end)
3. Save synchronized (depth, RGB) pairs for annotation
4. Export to labeled CSV for analysis

Usage:
    python3 rosbag_frame_extractor.py \
        --rosbag /path/to/rosbag.mcap \
        --output-dir /path/to/output \
        --num-frames 10
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional, Tuple, List
import json
import numpy as np
from datetime import datetime
import logging

import rclpy
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message_class
from rosbag2_py import SequentialReader, StorageOptions
from sensor_msgs.msg import PointCloud2, Image, CompressedImage
from sensor_msgs_py import point_cloud2
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
import cv2
from cv_bridge import CvBridge

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RosbagFrameExtractor:
    """Extract and synchronize sensor frames from rosbag."""

    def __init__(self, rosbag_path: str, output_dir: str):
        self.rosbag_path = rosbag_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.bridge = CvBridge()
        self.depth_frames = []
        self.rgb_frames = []
        self.scan_frames = []
        self.metadata = {
            'rosbag': rosbag_path,
            'extracted_at': datetime.now().isoformat(),
            'frames': []
        }

    def read_rosbag(self) -> dict:
        """Read all messages from rosbag, grouped by timestamp."""
        storage_options = StorageOptions(uri=self.rosbag_path, storage_id='mcap')
        reader = SequentialReader()
        reader.open(storage_options)

        topic_types = reader.get_all_topics_and_types()
        type_map = {topic.name: topic.type for topic in topic_types}

        # Group messages by approximate timestamp
        messages_by_time = {}
        tolerance_ns = int(0.1e9)  # 100ms tolerance for synchronization

        for topic, data, ts in reader.read_messages():
            msg_type = type_map.get(topic)
            if msg_type is None:
                continue

            try:
                msg_class = get_message_class(msg_type)
                msg = deserialize_message(data, msg_class)

                # Round timestamp to nearest 100ms bucket
                ts_bucket = (ts // tolerance_ns) * tolerance_ns

                if ts_bucket not in messages_by_time:
                    messages_by_time[ts_bucket] = {}

                if topic not in messages_by_time[ts_bucket]:
                    messages_by_time[ts_bucket][topic] = []

                messages_by_time[ts_bucket][topic].append((msg, ts))

            except Exception as e:
                logger.warning(f"Failed to deserialize {topic}: {e}")
                continue

        return messages_by_time

    def extract_depth_array(self, pc2: PointCloud2) -> Optional[np.ndarray]:
        """Convert PointCloud2 to 2D depth image aligned with LiDAR bins."""
        try:
            # Extract XYZ only
            points = point_cloud2.read_points(pc2, field_names=('x', 'y', 'z'), skip_nans=True)
            points = np.array(list(points))

            if len(points) == 0:
                return None

            # Convert to spherical (range, theta)
            ranges = np.sqrt(points[:, 0]**2 + points[:, 1]**2 + points[:, 2]**2)
            angles = np.arctan2(points[:, 1], points[:, 0])

            # Bin into LiDAR's 3200-bin grid (-π to π)
            num_bins = 3200
            bin_indices = ((angles + np.pi) / (2 * np.pi) * num_bins).astype(int)
            bin_indices = np.clip(bin_indices, 0, num_bins - 1)

            # Per-bin minimum
            depth_image = np.full(num_bins, np.inf, dtype=np.float32)
            for i, r in enumerate(ranges):
                bin_idx = bin_indices[i]
                depth_image[bin_idx] = min(depth_image[bin_idx], r)

            return depth_image

        except Exception as e:
            logger.warning(f"Failed to extract depth: {e}")
            return None

    def extract_rgb_image(self, img_msg: Image) -> Optional[np.ndarray]:
        """Convert ROS Image to CV format."""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
            return cv_image
        except Exception as e:
            logger.warning(f"Failed to extract RGB: {e}")
            return None

    def select_frames(self, messages_by_time: dict, num_frames: int = 10) -> List[dict]:
        """Select uniformly-spaced frames across the rosbag duration."""
        sorted_times = sorted(messages_by_time.keys())

        if len(sorted_times) < num_frames:
            indices = list(range(len(sorted_times)))
        else:
            indices = np.linspace(0, len(sorted_times) - 1, num_frames, dtype=int)

        selected_frames = []
        for idx in indices:
            ts = sorted_times[idx]
            msgs = messages_by_time[ts]

            frame_data = {
                'timestamp': ts / 1e9,  # Convert to seconds
                'rosbag': Path(self.rosbag_path).name,
                'messages': {}
            }

            # Look for relevant topics
            for topic, msg_list in msgs.items():
                if 'depth' in topic.lower() and 'camera' in topic.lower():
                    if isinstance(msg_list[0][0], PointCloud2):
                        frame_data['messages']['depth'] = msg_list[0]

                elif 'image_rect' in topic.lower() or 'rgb' in topic.lower():
                    if isinstance(msg_list[0][0], Image):
                        frame_data['messages']['rgb'] = msg_list[0]

                elif 'scan' in topic.lower():
                    frame_data['messages']['scan'] = msg_list[0]

            if frame_data['messages']:
                selected_frames.append(frame_data)

        return selected_frames

    def save_frame_data(self, frame_data: dict, frame_idx: int):
        """Save frame as numpy arrays and metadata JSON."""
        frame_dir = self.output_dir / f"frame_{frame_idx:03d}"
        frame_dir.mkdir(exist_ok=True)

        frame_meta = {
            'frame_id': frame_idx,
            'timestamp': frame_data['timestamp'],
            'rosbag': frame_data['rosbag'],
            'files': {}
        }

        # Save depth
        if 'depth' in frame_data['messages']:
            depth_msg, _ = frame_data['messages']['depth']
            depth_array = self.extract_depth_array(depth_msg)
            if depth_array is not None:
                depth_path = frame_dir / "depth.npy"
                np.save(depth_path, depth_array)
                frame_meta['files']['depth'] = 'depth.npy'

        # Save RGB
        if 'rgb' in frame_data['messages']:
            rgb_msg, _ = frame_data['messages']['rgb']
            rgb_array = self.extract_rgb_image(rgb_msg)
            if rgb_array is not None:
                rgb_path = frame_dir / "rgb.jpg"
                cv2.imwrite(str(rgb_path), rgb_array)
                frame_meta['files']['rgb'] = 'rgb.jpg'

        # Save metadata
        meta_path = frame_dir / "metadata.json"
        with open(meta_path, 'w') as f:
            json.dump(frame_meta, f, indent=2)

        logger.info(f"Saved frame {frame_idx} to {frame_dir}")
        return frame_meta

    def extract_frames(self, num_frames: int = 10) -> List[dict]:
        """Main extraction pipeline."""
        logger.info(f"Reading rosbag: {self.rosbag_path}")
        messages_by_time = self.read_rosbag()

        logger.info(f"Found {len(messages_by_time)} message bundles")

        selected = self.select_frames(messages_by_time, num_frames)
        logger.info(f"Selected {len(selected)} frames for extraction")

        all_frames_meta = []
        for i, frame_data in enumerate(selected):
            frame_meta = self.save_frame_data(frame_data, i)
            all_frames_meta.append(frame_meta)

        # Save overall metadata
        self.metadata['frames'] = all_frames_meta
        meta_path = self.output_dir / "extraction_metadata.json"
        with open(meta_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)

        logger.info(f"Extraction complete. Results in {self.output_dir}")
        return all_frames_meta


def main():
    parser = argparse.ArgumentParser(description="Extract frames from rosbag for ground truth labeling")
    parser.add_argument("--rosbag", required=True, help="Path to rosbag file")
    parser.add_argument("--output-dir", required=True, help="Output directory for extracted frames")
    parser.add_argument("--num-frames", type=int, default=10, help="Number of frames to extract")

    args = parser.parse_args()

    extractor = RosbagFrameExtractor(args.rosbag, args.output_dir)
    extractor.extract_frames(args.num_frames)


if __name__ == '__main__':
    main()
