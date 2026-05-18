#!/usr/bin/env python3
"""
SIMPLE LABELER — Non-ROS Frame Annotation Tool
================================================

Standalone tool to label frames without requiring ROS runtime.
Allows operator to view depth + RGB side-by-side and annotate bins.

Features:
- Load pre-extracted depth + RGB
- Interactive visualization
- CSV export
- No ROS dependency

Usage:
    python3 simple_labeler.py \
        --frame-dir /path/to/frame_000 \
        --output-csv /path/to/labels.csv
"""

import json
import csv
import sys
from pathlib import Path
from typing import Optional, List, Dict
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FrameAnnotationData:
    """In-memory frame data + annotations."""

    def __init__(self, frame_dir: Path, frame_id: int):
        self.frame_dir = Path(frame_dir)
        self.frame_id = frame_id
        self.metadata = self._load_metadata()
        self.depth = self._load_depth()
        self.rgb_path = self.frame_dir / "rgb.jpg"
        self.annotations = {}

    def _load_metadata(self) -> dict:
        meta_path = self.frame_dir / "metadata.json"
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                return json.load(f)
        return {}

    def _load_depth(self) -> Optional[np.ndarray]:
        depth_path = self.frame_dir / "depth.npy"
        if depth_path.exists():
            return np.load(depth_path)
        return None

    def get_statistics(self) -> Dict:
        """Compute depth statistics for display."""
        if self.depth is None:
            return {}

        valid = np.isfinite(self.depth)
        if not np.any(valid):
            return {}

        ranges = self.depth[valid]
        return {
            'total_bins': len(self.depth),
            'valid_bins': int(np.sum(valid)),
            'min_range': float(np.min(ranges)),
            'max_range': float(np.max(ranges)),
            'mean_range': float(np.mean(ranges)),
            'median_range': float(np.median(ranges)),
            'std_range': float(np.std(ranges))
        }

    def annotate_bin(self, bin_idx: int, label: str, confidence: float, notes: str = ""):
        """Record annotation for a bin."""
        num_bins = len(self.depth) if self.depth is not None else 3200
        angle_deg = (bin_idx / num_bins) * 360.0 - 180.0
        range_m = float(self.depth[bin_idx]) if self.depth is not None else np.nan

        self.annotations[bin_idx] = {
            'bin_idx': bin_idx,
            'angle_deg': angle_deg,
            'range_m': range_m,
            'label': label,
            'confidence': confidence,
            'notes': notes
        }

    def get_bins_by_range(self, max_range: float) -> List[int]:
        """Get all bins with range <= max_range."""
        if self.depth is None:
            return []
        valid = np.isfinite(self.depth) & (self.depth <= max_range) & (self.depth > 0)
        return list(np.where(valid)[0])

    def export_csv(self, csv_path: Path):
        """Export annotations to CSV."""
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'frame_id', 'bin_idx', 'angle_deg', 'range_m',
                'label', 'confidence', 'notes'
            ])

            for bin_idx in sorted(self.annotations.keys()):
                ann = self.annotations[bin_idx]
                writer.writerow([
                    self.frame_id,
                    ann['bin_idx'],
                    f"{ann['angle_deg']:.2f}",
                    f"{ann['range_m']:.3f}",
                    ann['label'],
                    f"{ann['confidence']:.2f}",
                    ann['notes']
                ])


class BatchAnnotator:
    """Manage annotation across multiple frames."""

    def __init__(self, root_dir: Path):
        self.root_dir = Path(root_dir)
        self.frames: Dict[int, FrameAnnotationData] = {}
        self._discover_frames()

    def _discover_frames(self):
        """Find all frame directories."""
        for frame_dir in sorted(self.root_dir.glob("rosbag_*/frame_*")):
            try:
                rosbag_idx = int(frame_dir.parent.name.split('_')[1])
                frame_idx = int(frame_dir.name.split('_')[1])
                global_frame_id = rosbag_idx * 1000 + frame_idx

                self.frames[global_frame_id] = FrameAnnotationData(frame_dir, global_frame_id)
            except (ValueError, IndexError):
                continue

        logger.info(f"Discovered {len(self.frames)} frames")

    def export_all_csv(self, csv_path: Path):
        """Merge all annotations into single CSV."""
        all_annotations = []

        for frame_id in sorted(self.frames.keys()):
            frame = self.frames[frame_id]
            for bin_idx in sorted(frame.annotations.keys()):
                ann = frame.annotations[bin_idx]
                all_annotations.append((frame_id, ann))

        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'frame_id', 'bin_idx', 'angle_deg', 'range_m',
                'label', 'confidence', 'notes'
            ])

            for frame_id, ann in all_annotations:
                writer.writerow([
                    frame_id,
                    ann['bin_idx'],
                    f"{ann['angle_deg']:.2f}",
                    f"{ann['range_m']:.3f}",
                    ann['label'],
                    f"{ann['confidence']:.2f}",
                    ann['notes']
                ])

        logger.info(f"Exported {len(all_annotations)} annotations to {csv_path}")

    def get_statistics_summary(self) -> Dict:
        """Compute aggregate statistics."""
        summary = {
            'num_frames': len(self.frames),
            'frames': {}
        }

        total_annotated = 0
        for frame_id in sorted(self.frames.keys()):
            frame = self.frames[frame_id]
            frame_stats = frame.get_statistics()
            frame_stats['num_annotations'] = len(frame.annotations)
            summary['frames'][frame_id] = frame_stats
            total_annotated += len(frame.annotations)

        summary['total_annotations'] = total_annotated
        return summary


def print_frame_summary(frame: FrameAnnotationData):
    """Print frame data for review."""
    print("\n" + "=" * 70)
    print(f"FRAME {frame.frame_id}")
    print("=" * 70)

    if frame.metadata:
        print(f"Timestamp: {frame.metadata.get('timestamp', 'N/A')}")
        print(f"Rosbag: {frame.metadata.get('rosbag', 'N/A')}")

    stats = frame.get_statistics()
    if stats:
        print(f"\nDepth Statistics:")
        print(f"  Valid bins: {stats['valid_bins']}/{stats['total_bins']}")
        print(f"  Range: {stats['min_range']:.2f}–{stats['max_range']:.2f} m")
        print(f"  Mean: {stats['mean_range']:.2f} m ± {stats['std_range']:.2f} m")

    print(f"\nRGB Image: {frame.rgb_path}")
    if frame.rgb_path.exists():
        print(f"  ✓ Exists ({frame.rgb_path.stat().st_size / 1e6:.1f} MB)")
    else:
        print(f"  ✗ Not found")

    print(f"\nAnnotations: {len(frame.annotations)}")
    if frame.annotations:
        labels = {}
        for ann in frame.annotations.values():
            label = ann['label']
            labels[label] = labels.get(label, 0) + 1
        for label, count in sorted(labels.items()):
            print(f"  {label}: {count}")


def interactive_labeling_prompt():
    """Interactive CLI prompts for labeling."""
    print("\n" + "=" * 70)
    print("INTERACTIVE LABELING INTERFACE")
    print("=" * 70)
    print("\nSteps:")
    print("1. View depth statistics above")
    print("2. Open RGB image: frame_XXX/rgb.jpg")
    print("3. Use analyze_camera_closer_bins() to find bins near objects")
    print("4. Manually review and classify in CSV")
    print("\nDo NOT use this for automated annotation.")
    print("Ground truth requires human judgment at each bin.")
    print("\nFor detailed instructions, see ANNOTATION_PROTOCOL.md")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Simple frame annotation tool (no ROS required)"
    )
    parser.add_argument("--root-dir", required=True, help="Root extraction directory")
    parser.add_argument("--output-csv", required=True, help="Output CSV path")
    parser.add_argument("--summary", action="store_true", help="Print summary only")

    args = parser.parse_args()

    root_dir = Path(args.root_dir)
    if not root_dir.exists():
        logger.error(f"Directory not found: {root_dir}")
        sys.exit(1)

    annotator = BatchAnnotator(root_dir)

    if args.summary:
        summary = annotator.get_statistics_summary()
        print(json.dumps(summary, indent=2, default=str))
        return

    # Print frame summaries
    for frame_id in sorted(annotator.frames.keys()):
        frame = annotator.frames[frame_id]
        print_frame_summary(frame)

    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print("\n1. Manually label each frame:")
    print(f"   - Open rosbag_XX/frame_YYY/rgb.jpg")
    print(f"   - Load rosbag_XX/frame_YYY/depth.npy in Python/NumPy")
    print(f"   - Review ANNOTATION_PROTOCOL.md for classification rules")
    print(f"   - Edit rosbag_XX/frame_YYY/labels.csv with annotations")
    print("\n2. Merge all labels.csv files:")
    print(f"   cat rosbag_*/frame_*/labels.csv > labels_combined.csv")
    print("\n3. Run analysis:")
    print(f"   python3 analysis_tools.py --labels-csv labels_combined.csv")

    interactive_labeling_prompt()


if __name__ == '__main__':
    main()
