#!/usr/bin/env python3
"""
GROUND TRUTH LABELING INTERFACE
================================

Interactive tool to classify phantom candidate bins as:
  - TRUE OBSTACLE: confirmed elevated structure
  - PHANTOM (DEPTH NOISE): RealSense speckle/multipath
  - PHANTOM (REFLECTION): IR bounce, glass penetration
  - PHANTOM (OTHER): artifacts, ceiling, unclassifiable
  - AMBIGUOUS: unclear, review manually

Workflow:
1. Load depth arrays + RGB images
2. Display side-by-side (depth heatmap + RGB video)
3. For each frame, mark camera-closer bins
4. Export labeled CSV

Usage:
    python3 labeling_interface.py \
        --frames-dir /path/to/extracted/frames \
        --output-csv labels.csv
"""

import json
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import logging
from dataclasses import dataclass
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LabelClass(Enum):
    """Bin classification types."""
    TRUE_OBSTACLE = "true_obstacle"
    PHANTOM_DEPTH_NOISE = "phantom_depth_noise"
    PHANTOM_REFLECTION = "phantom_reflection"
    PHANTOM_OTHER = "phantom_other"
    AMBIGUOUS = "ambiguous"
    UNLABELED = "unlabeled"


@dataclass
class BinLabel:
    """Per-bin label record."""
    frame_id: int
    bin_idx: int
    angle_deg: float
    range_m: float
    label: LabelClass
    confidence: float  # 0.0-1.0 (1.0 = certain)
    notes: str = ""


class FrameLabeler:
    """Manage labeling for a single frame."""

    def __init__(self, frame_dir: Path, frame_id: int):
        self.frame_dir = frame_dir
        self.frame_id = frame_id
        self.metadata = self._load_metadata()
        self.depth_array = self._load_depth()
        self.rgb_image = self._load_rgb()
        self.labels: List[BinLabel] = []

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

    def _load_rgb(self) -> Optional[np.ndarray]:
        rgb_path = self.frame_dir / "rgb.jpg"
        if rgb_path.exists():
            import cv2
            return cv2.imread(str(rgb_path))
        return None

    def get_statistics(self) -> dict:
        """Compute basic depth statistics for this frame."""
        if self.depth_array is None:
            return {}

        valid = np.isfinite(self.depth_array)
        if not np.any(valid):
            return {}

        ranges = self.depth_array[valid]
        return {
            'num_bins_valid': int(np.sum(valid)),
            'num_bins_total': len(self.depth_array),
            'range_min': float(np.min(ranges)),
            'range_max': float(np.max(ranges)),
            'range_mean': float(np.mean(ranges)),
            'range_median': float(np.median(ranges)),
            'range_std': float(np.std(ranges))
        }

    def classify_bin(self, bin_idx: int, label: LabelClass, confidence: float, notes: str = ""):
        """Record a label for a specific bin."""
        num_bins = len(self.depth_array) if self.depth_array is not None else 3200
        angle_deg = (bin_idx / num_bins) * 360.0 - 180.0
        range_m = self.depth_array[bin_idx] if self.depth_array is not None else np.nan

        bin_label = BinLabel(
            frame_id=self.frame_id,
            bin_idx=bin_idx,
            angle_deg=angle_deg,
            range_m=float(range_m),
            label=label,
            confidence=confidence,
            notes=notes
        )
        self.labels.append(bin_label)

    def save_labels(self, output_path: Path):
        """Save labels to CSV."""
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'frame_id', 'bin_idx', 'angle_deg', 'range_m',
                'label', 'confidence', 'notes'
            ])
            for label in self.labels:
                writer.writerow([
                    label.frame_id,
                    label.bin_idx,
                    f"{label.angle_deg:.2f}",
                    f"{label.range_m:.3f}",
                    label.label.value,
                    f"{label.confidence:.2f}",
                    label.notes
                ])


class BatchLabeler:
    """Manage labeling across multiple frames."""

    def __init__(self, frames_dir: Path):
        self.frames_dir = frames_dir
        self.frame_labelers: Dict[int, FrameLabeler] = {}
        self._discover_frames()

    def _discover_frames(self):
        """Discover all frame directories."""
        for frame_dir in sorted(self.frames_dir.glob("frame_*")):
            try:
                frame_id = int(frame_dir.name.split('_')[1])
                self.frame_labelers[frame_id] = FrameLabeler(frame_dir, frame_id)
            except (ValueError, IndexError):
                continue

        logger.info(f"Discovered {len(self.frame_labelers)} frames")

    def get_frame(self, frame_id: int) -> Optional[FrameLabeler]:
        return self.frame_labelers.get(frame_id)

    def list_frames(self) -> List[int]:
        return sorted(self.frame_labelers.keys())

    def export_all_labels(self, output_csv: Path):
        """Export all labels from all frames to single CSV."""
        all_labels = []
        for labeler in self.frame_labelers.values():
            all_labels.extend(labeler.labels)

        with open(output_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'frame_id', 'bin_idx', 'angle_deg', 'range_m',
                'label', 'confidence', 'notes'
            ])
            for label in all_labels:
                writer.writerow([
                    label.frame_id,
                    label.bin_idx,
                    f"{label.angle_deg:.2f}",
                    f"{label.range_m:.3f}",
                    label.label.value,
                    f"{label.confidence:.2f}",
                    label.notes
                ])

        logger.info(f"Exported {len(all_labels)} labels to {output_csv}")

    def compute_statistics(self) -> dict:
        """Compute aggregate labeling statistics."""
        all_labels = []
        for labeler in self.frame_labelers.values():
            all_labels.extend(labeler.labels)

        if not all_labels:
            return {}

        label_counts = {}
        for label_class in LabelClass:
            label_counts[label_class.value] = sum(
                1 for lbl in all_labels if lbl.label == label_class
            )

        total = len(all_labels)
        label_percentages = {
            k: (v / total * 100.0) if total > 0 else 0.0
            for k, v in label_counts.items()
        }

        confidence_values = [lbl.confidence for lbl in all_labels]
        avg_confidence = np.mean(confidence_values) if confidence_values else 0.0

        phantom_count = sum(
            1 for lbl in all_labels
            if 'phantom' in lbl.label.value.lower()
        )
        phantom_rate = (phantom_count / total * 100.0) if total > 0 else 0.0

        return {
            'total_labeled_bins': total,
            'label_counts': label_counts,
            'label_percentages': label_percentages,
            'phantom_rate_percent': phantom_rate,
            'avg_confidence': float(avg_confidence),
            'num_frames': len(self.frame_labelers)
        }


def generate_labeling_protocol_doc() -> str:
    """Generate the annotation protocol document."""
    return """
# GROUND TRUTH LABELING PROTOCOL

## Overview
This protocol defines how to manually classify depth bins as true obstacles or phantoms
using synchronized RGB video and RViz depth visualization.

## Classification Definitions

### TRUE OBSTACLE
**Definition**: Confirmed elevated structure at specified depth.

**Criteria**:
- Visible in RGB video at corresponding depth
- Consistent with wheelchair operation (table, shelf, equipment rack, door frame, etc.)
- Range and orientation match physical plausibility
- Surface appears continuous and stable

**Examples**: dining tables, shelving units, equipment racks, door frames, step risers

**Mark When**: Visual confirmation in RGB + consistent range + plausible structure

### PHANTOM (DEPTH NOISE)
**Definition**: RealSense speckle or multipath artifacts.

**Criteria**:
- Isolated point(s) not corresponding to visible object in RGB
- Range appears inconsistent with surroundings
- Disappears frame-to-frame (unstable)
- Typically closer than expected given RGB context

**Examples**: sensor noise dots, single-pixel artifacts, multipath reflections

**Mark When**: No visible confirmation in RGB + inconsistent with scene geometry

### PHANTOM (REFLECTION)
**Definition**: IR bounce from reflective surfaces or glass.

**Criteria**:
- May appear to correspond to object in RGB but at incorrect depth
- Often caused by IR beam bouncing off glass, mirrors, or shiny surfaces
- Range is inconsistent with actual object distance
- May appear layered or doubled

**Examples**: glass doorways, reflective equipment, curved metal surfaces

**Mark When**: RGB shows object but depth doesn't match + reflective surface likely

### PHANTOM (OTHER)
**Definition**: Other artifacts (ceiling, ground bounce, compression artifacts, etc.).

**Criteria**:
- Not noise, not reflection, but clearly false positive
- May be ceiling return, ground multipath, image compression artifact
- Does not match any elevated structure in RGB

**Examples**: ceiling returns, ground bounces, compression artifacts

**Mark When**: Clearly invalid but doesn't fit noise/reflection categories

### AMBIGUOUS
**Definition**: Unclear classification requiring manual review.

**Criteria**:
- Insufficient RGB visibility (camera pointing away)
- Occluded in both depth and RGB
- Operator unfamiliar with location
- Borderline case (could be either true or phantom)

**Mark When**: >50% uncertain after review

## Labeling Procedure

### Per Frame:

1. **Load frame data**:
   - Open depth heatmap (range_m, colors: blue=close, red=far)
   - Open synchronized RGB video (front camera)
   - Read frame metadata (timestamp, rosbag name)

2. **Review depth statistics**:
   - Note min/max/mean range
   - Identify outliers (unusually close bins)
   - Mark suspicious clusters

3. **For each camera-closer bin** (range < 1.5m from front D455 or < 1.0m from D435i):
   - Check RGB frame at corresponding angle
   - Determine if visible structure matches
   - Assign label (TRUE, PHANTOM_NOISE, PHANTOM_REFL, PHANTOM_OTHER, AMBIGUOUS)
   - Rate confidence: 0.5 = uncertain, 1.0 = certain

4. **Document reasoning**:
   - Add brief note if classification is non-obvious
   - Flag any ambiguous cases for secondary review

### Confidence Ratings:

- **0.5-0.6**: Uncertain, borderline, could be either
- **0.7-0.8**: Likely correct, minor doubt
- **0.9-1.0**: Certain, visual confirmation clear

## Quality Control

- **Inter-rater agreement**: Two annotators label same frame, compute κ (kappa)
- **Outlier detection**: Flags bins where confidence < 0.6
- **Spatial continuity**: Checks for isolated bins (likely noise)
- **Temporal consistency**: If same location appears in multiple frames, should have consistent labels

## Acceptance Criteria

- ≥90% of bins labeled (not ambiguous)
- Average confidence ≥0.80 across all labels
- <10% of labels are ambiguous
- Inter-rater κ ≥0.70 for overlap

"""


if __name__ == '__main__':
    # Generate sample protocol
    protocol = generate_labeling_protocol_doc()
    with open('/tmp/labeling_protocol.md', 'w') as f:
        f.write(protocol)
    print("Protocol generated: /tmp/labeling_protocol.md")
