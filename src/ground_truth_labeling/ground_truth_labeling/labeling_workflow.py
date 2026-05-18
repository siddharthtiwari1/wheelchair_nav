#!/usr/bin/env python3
"""
GROUND TRUTH LABELING WORKFLOW
================================

End-to-end workflow to:
1. Select 3 diverse rosbags
2. Extract 10 frames per rosbag
3. Generate interactive labeling interface
4. Analyze and report results

Usage:
    python3 labeling_workflow.py \
        --num-rosbags 3 \
        --frames-per-rosbag 10 \
        --output-dir /home/sidd/wheelchair_nav/ground_truth_labels
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from typing import List, Optional
import random
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


class GroundTruthWorkflow:
    """Manage entire labeling pipeline."""

    def __init__(self, rosbag_archive: str, output_dir: str, num_rosbags: int = 3, frames_per_rosbag: int = 10):
        self.rosbag_archive = Path(rosbag_archive)
        self.output_dir = Path(output_dir)
        self.num_rosbags = num_rosbags
        self.frames_per_rosbag = frames_per_rosbag

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.selected_rosbags = []
        self.extraction_results = {}

    def discover_rosbags(self) -> List[Path]:
        """Find all .mcap rosbags in archive."""
        rosbags = list(self.rosbag_archive.glob("*/*mcap"))
        logger.info(f"Discovered {len(rosbags)} rosbags")
        return sorted(rosbags)

    def select_diverse_rosbags(self) -> List[Path]:
        """Select rosbags from different dates/times for diversity."""
        all_rosbags = self.discover_rosbags()

        if len(all_rosbags) < self.num_rosbags:
            logger.warning(
                f"Only {len(all_rosbags)} rosbags found, "
                f"requesting {self.num_rosbags}"
            )
            self.selected_rosbags = all_rosbags
        else:
            # Group by date
            by_date = {}
            for rosbag in all_rosbags:
                date = rosbag.parent.name[:8]  # e.g., "20260226"
                if date not in by_date:
                    by_date[date] = []
                by_date[date].append(rosbag)

            # Select one from each date if possible
            selected = []
            for date in sorted(by_date.keys()):
                if len(selected) < self.num_rosbags:
                    rosbag = random.choice(by_date[date])
                    selected.append(rosbag)

            # Fill remaining from random selection
            while len(selected) < self.num_rosbags:
                rosbag = random.choice(all_rosbags)
                if rosbag not in selected:
                    selected.append(rosbag)

            self.selected_rosbags = selected

        logger.info(f"Selected {len(self.selected_rosbags)} diverse rosbags:")
        for rosbag in self.selected_rosbags:
            logger.info(f"  {rosbag}")

        return self.selected_rosbags

    def extract_frames(self) -> dict:
        """Extract frames from each selected rosbag."""
        results = {}

        for i, rosbag in enumerate(self.selected_rosbags):
            rosbag_name = rosbag.parent.name
            output_subdir = self.output_dir / f"rosbag_{i:02d}_{rosbag_name}"

            logger.info(f"\nExtracting frames from {rosbag_name}...")

            try:
                # Import here to avoid circular dependency
                from rosbag_frame_extractor import RosbagFrameExtractor

                extractor = RosbagFrameExtractor(str(rosbag), str(output_subdir))
                frame_metas = extractor.extract_frames(self.frames_per_rosbag)

                results[rosbag_name] = {
                    'rosbag_path': str(rosbag),
                    'output_dir': str(output_subdir),
                    'frames_extracted': len(frame_metas),
                    'frame_metadata': frame_metas
                }

                logger.info(f"Extracted {len(frame_metas)} frames to {output_subdir}")

            except Exception as e:
                logger.error(f"Failed to extract frames from {rosbag}: {e}")
                results[rosbag_name] = {
                    'error': str(e),
                    'rosbag_path': str(rosbag)
                }

        self.extraction_results = results
        return results

    def generate_labeling_interface(self) -> Path:
        """Generate HTML/text-based labeling interface."""
        interface_path = self.output_dir / "LABELING_GUIDE.md"

        from labeling_interface import generate_labeling_protocol_doc

        protocol = generate_labeling_protocol_doc()

        interface_content = f"""
# GROUND TRUTH LABELING INTERFACE

**Project**: Wheelchair Navigation Phantom Obstacle Detection
**Date Generated**: {datetime.now().isoformat()}
**Output Directory**: {self.output_dir}

## Quick Start

1. **Read the Protocol** (below)
2. **Label Frames**: For each frame_XXX/ directory:
   - View `rgb.jpg` (RGB video frame)
   - View `depth.npy` (depth array visualization)
   - Mark bins in `labels.csv` using protocol definitions
3. **Export Results**: Run analysis tools

## Frame Structure

Each extracted rosbag has this structure:
```
rosbag_XX_YYYYMMDD_HHMMSS/
├── frame_000/
│   ├── metadata.json      (timestamp, frame_id)
│   ├── depth.npy          (3200-bin depth array)
│   ├── rgb.jpg            (RGB image from front camera)
│   └── labels.csv         (your annotations)
├── frame_001/
...
└── extraction_metadata.json
```

## Frame Selection Rationale

Selected {len(self.selected_rosbags)} rosbags from different dates/times:
"""

        for i, rosbag in enumerate(self.selected_rosbags):
            rosbag_name = rosbag.parent.name
            interface_content += f"\n- **Rosbag {i}**: {rosbag_name}"

        interface_content += f"""

## Annotation Workflow (Per Frame)

1. Load frame metadata (timestamp, rosbag)
2. Visualize depth array as heatmap:
   - Load `depth.npy` with numpy
   - Plot as image (bins × time dimension)
   - Range: blue (close) → red (far)
3. View synchronized RGB:
   - Open `rgb.jpg` in image viewer
   - Cross-reference camera view with depth bins
4. For camera-closer bins (range < 1.5m):
   - Decide: TRUE_OBSTACLE, PHANTOM_DEPTH_NOISE, PHANTOM_REFLECTION, PHANTOM_OTHER, AMBIGUOUS
   - Rate confidence: 0.5–1.0 (0.5=uncertain, 1.0=certain)
   - Write to `labels.csv`

## Annotation Statistics Target

- **Total frames**: 30 (10 per rosbag)
- **Bins per frame**: ~3200 (LiDAR resolution)
- **Camera-closer bins** (to label): ~500–1500 per frame (depth < 1.5m)
- **Total annotations expected**: 15,000–45,000 bins
- **Labeling time**: ~1–2 hours per rosbag (10 frames)

## Output Format

`labels.csv`:
```
frame_id,bin_idx,angle_deg,range_m,label,confidence,notes
0,100,-165.2,1.234,true_obstacle,0.95,"Table edge"
0,101,-165.1,1.245,true_obstacle,0.93,""
0,102,-165.0,0.850,phantom_depth_noise,0.88,"Speckle"
...
```

## Analysis After Labeling

Once all frames labeled, run analysis:
```bash
python3 analysis_tools.py \\
    --labels-csv /path/to/combined_labels.csv \\
    --output-dir /path/to/results/
```

This generates:
- `summary_table.md` — classification counts/percentages
- `analysis_results.json` — detailed statistics
- Phantom rate %, spatial distribution, range distribution

{protocol}
"""

        with open(interface_path, 'w') as f:
            f.write(interface_content)

        logger.info(f"Generated labeling guide: {interface_path}")
        return interface_path

    def generate_template_labels_csv(self):
        """Generate template CSV files in each frame directory."""
        import csv

        for rosbag_dir in self.output_dir.glob("rosbag_*"):
            for frame_dir in sorted(rosbag_dir.glob("frame_*")):
                labels_path = frame_dir / "labels.csv"

                # Only generate if doesn't exist
                if labels_path.exists():
                    continue

                # Create template with header only
                with open(labels_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        'frame_id', 'bin_idx', 'angle_deg', 'range_m',
                        'label', 'confidence', 'notes'
                    ])

        logger.info("Generated template CSV files in all frame directories")

    def generate_workflow_summary(self):
        """Generate summary of extracted rosbags/frames."""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'output_directory': str(self.output_dir),
            'num_rosbags_selected': len(self.selected_rosbags),
            'frames_per_rosbag': self.frames_per_rosbag,
            'extraction_results': self.extraction_results,
            'labeling_guide': 'LABELING_GUIDE.md',
            'next_steps': [
                '1. Read LABELING_GUIDE.md for annotation protocol',
                '2. For each frame_XXX/ directory:',
                '   a. Open rgb.jpg (reference image)',
                '   b. Load depth.npy with numpy.load()',
                '   c. Annotate labels.csv with bin classifications',
                '3. Merge all labels.csv files from all frames into single CSV',
                '4. Run: python3 analysis_tools.py --labels-csv labels_combined.csv'
            ]
        }

        summary_path = self.output_dir / "workflow_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Workflow summary: {summary_path}")
        return summary

    def run(self):
        """Execute full pipeline."""
        logger.info("=" * 70)
        logger.info("GROUND TRUTH LABELING WORKFLOW")
        logger.info("=" * 70)

        self.select_diverse_rosbags()
        logger.info("\n" + "=" * 70)
        logger.info("Extracting frames...")
        logger.info("=" * 70 + "\n")
        self.extract_frames()

        logger.info("\n" + "=" * 70)
        logger.info("Generating labeling interface...")
        logger.info("=" * 70 + "\n")
        self.generate_labeling_interface()
        self.generate_template_labels_csv()
        self.generate_workflow_summary()

        logger.info("\n" + "=" * 70)
        logger.info("WORKFLOW COMPLETE")
        logger.info("=" * 70)
        logger.info(f"\nOutput directory: {self.output_dir}")
        logger.info(f"\nNext steps:")
        logger.info(f"1. Read: {self.output_dir}/LABELING_GUIDE.md")
        logger.info(f"2. Annotate frames in: {self.output_dir}/rosbag_XX_*/frame_YYY/")
        logger.info(f"3. Merge CSVs and run analysis tools")


def main():
    parser = argparse.ArgumentParser(
        description="Ground truth labeling workflow"
    )
    parser.add_argument(
        "--rosbag-archive",
        default="/opt/nvidia/rosbag_archive/rosbags",
        help="Path to rosbag archive"
    )
    parser.add_argument(
        "--output-dir",
        default="/home/sidd/wheelchair_nav/ground_truth_labels",
        help="Output directory for extracted frames and labels"
    )
    parser.add_argument(
        "--num-rosbags",
        type=int,
        default=3,
        help="Number of rosbags to select"
    )
    parser.add_argument(
        "--frames-per-rosbag",
        type=int,
        default=10,
        help="Number of frames to extract per rosbag"
    )

    args = parser.parse_args()

    workflow = GroundTruthWorkflow(
        rosbag_archive=args.rosbag_archive,
        output_dir=args.output_dir,
        num_rosbags=args.num_rosbags,
        frames_per_rosbag=args.frames_per_rosbag
    )
    workflow.run()


if __name__ == '__main__':
    main()
