#!/usr/bin/env python3
"""
FRAME VISUALIZATION TOOL
=========================

Generate visual comparisons of depth + RGB for inspection and labeling.
Creates side-by-side plots and annotated overlays.

Usage:
    python3 visualize_frames.py \
        --frame-dir /path/to/frame_000 \
        --output-dir /path/to/output
"""

import json
import sys
from pathlib import Path
from typing import Optional
import numpy as np
import logging
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def visualize_frame(frame_dir: Path, output_dir: Optional[Path] = None):
    """Create visualization of depth + RGB for a single frame."""
    try:
        import matplotlib.pyplot as plt
        from PIL import Image
    except ImportError:
        logger.error("matplotlib and pillow required. Install with: pip install matplotlib pillow")
        return False

    frame_dir = Path(frame_dir)
    if not frame_dir.exists():
        logger.error(f"Frame directory not found: {frame_dir}")
        return False

    output_dir = Path(output_dir) if output_dir else frame_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    depth_path = frame_dir / "depth.npy"
    rgb_path = frame_dir / "rgb.jpg"
    meta_path = frame_dir / "metadata.json"

    if not depth_path.exists():
        logger.error(f"Depth file not found: {depth_path}")
        return False

    depth = np.load(depth_path)
    logger.info(f"Loaded depth array: shape={depth.shape}, dtype={depth.dtype}")

    rgb = None
    if rgb_path.exists():
        rgb = np.array(Image.open(rgb_path))
        logger.info(f"Loaded RGB image: shape={rgb.shape}")
    else:
        logger.warning(f"RGB image not found: {rgb_path}")

    metadata = {}
    if meta_path.exists():
        with open(meta_path, 'r') as f:
            metadata = json.load(f)

    # Create figure
    if rgb is not None:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12))
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8))

    # Plot 1: Depth scatter
    valid = np.isfinite(depth) & (depth > 0)
    bin_indices = np.where(valid)[0]
    depths = depth[valid]

    ax1.scatter(bin_indices, depths, s=1, alpha=0.6, c=depths, cmap='viridis')
    ax1.set_xlabel("Bin Index (0–3200)")
    ax1.set_ylabel("Range (m)")
    ax1.set_title(f"Depth Array — {metadata.get('timestamp', 'N/A')}")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 3200)

    # Add colorbar
    cbar1 = plt.colorbar(ax1.collections[0], ax=ax1)
    cbar1.set_label("Range (m)")

    # Plot 2: Depth histogram
    ax2.hist(depths, bins=50, edgecolor='black', alpha=0.7)
    ax2.set_xlabel("Range (m)")
    ax2.set_ylabel("Bin Count")
    ax2.set_title("Depth Distribution")
    ax2.axvline(np.median(depths), color='red', linestyle='--', label=f"Median: {np.median(depths):.2f} m")
    ax2.axvline(np.mean(depths), color='orange', linestyle='--', label=f"Mean: {np.mean(depths):.2f} m")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: RGB (if available)
    if rgb is not None:
        ax3.imshow(rgb)
        ax3.set_title("Front Camera RGB")
        ax3.axis('off')

    plt.tight_layout()
    output_path = output_dir / "depth_rgb_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved visualization: {output_path}")
    plt.close()

    # Create statistics summary
    stats = {
        'frame': str(frame_dir),
        'timestamp': metadata.get('timestamp', 'N/A'),
        'rosbag': metadata.get('rosbag', 'N/A'),
        'depth_stats': {
            'total_bins': len(depth),
            'valid_bins': int(np.sum(valid)),
            'valid_percent': float(np.sum(valid) / len(depth) * 100),
            'range_min': float(np.min(depths)),
            'range_max': float(np.max(depths)),
            'range_mean': float(np.mean(depths)),
            'range_median': float(np.median(depths)),
            'range_std': float(np.std(depths)),
            'range_q1': float(np.percentile(depths, 25)),
            'range_q3': float(np.percentile(depths, 75))
        }
    }

    # Camera-closer bins
    for threshold, label in [(1.5, "D455_1.5m"), (1.0, "D435i_1.0m"), (0.5, "Close_0.5m")]:
        camera_close = np.sum(depth <= threshold)
        stats['depth_stats'][f'bins_le_{label}'] = int(camera_close)
        stats['depth_stats'][f'percent_le_{label}'] = float(camera_close / len(depth) * 100)

    # Save statistics
    stats_path = output_dir / "frame_statistics.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Saved statistics: {stats_path}")

    # Print summary
    print(f"\n{'='*70}")
    print(f"FRAME VISUALIZATION SUMMARY")
    print(f"{'='*70}")
    print(f"Frame: {frame_dir}")
    print(f"Timestamp: {stats['timestamp']}")
    print(f"Rosbag: {stats['rosbag']}")
    print(f"\nDepth Statistics:")
    d_stats = stats['depth_stats']
    print(f"  Valid bins: {d_stats['valid_bins']:,} / {d_stats['total_bins']:,} ({d_stats['valid_percent']:.1f}%)")
    print(f"  Range: {d_stats['range_min']:.2f}–{d_stats['range_max']:.2f} m")
    print(f"  Mean: {d_stats['range_mean']:.2f} m ± {d_stats['range_std']:.2f} m")
    print(f"  Median: {d_stats['range_median']:.2f} m (Q1={d_stats['range_q1']:.2f}, Q3={d_stats['range_q3']:.2f})")
    print(f"\nCamera-Closer Bins (candidates for labeling):")
    print(f"  D455 (<1.5m): {d_stats['bins_le_D455_1.5m']:,} ({d_stats['percent_le_D455_1.5m']:.1f}%)")
    print(f"  D435i (<1.0m): {d_stats['bins_le_D435i_1.0m']:,} ({d_stats['percent_le_D435i_1.0m']:.1f}%)")
    print(f"  Close (<0.5m): {d_stats['bins_le_Close_0.5m']:,} ({d_stats['percent_le_Close_0.5m']:.1f}%)")
    print(f"\nVisualizations:")
    print(f"  {output_path.relative_to(output_dir.parent)}")
    print(f"  {stats_path.relative_to(output_dir.parent)}")
    print()

    return True


def batch_visualize(root_dir: Path, output_base: Optional[Path] = None):
    """Create visualizations for all frames."""
    root_dir = Path(root_dir)
    output_base = Path(output_base) if output_base else root_dir

    frame_dirs = sorted(root_dir.glob("rosbag_*/frame_*"))
    if not frame_dirs:
        logger.error(f"No frames found in {root_dir}")
        return False

    logger.info(f"Visualizing {len(frame_dirs)} frames...")

    for frame_dir in frame_dirs:
        output_dir = output_base / frame_dir.relative_to(root_dir)
        visualize_frame(frame_dir, output_dir)

    logger.info(f"Batch visualization complete")
    return True


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Visualize frame data (depth + RGB)"
    )
    parser.add_argument("--frame-dir", help="Single frame directory to visualize")
    parser.add_argument("--root-dir", help="Root extraction directory (batch mode)")
    parser.add_argument("--output-dir", help="Output directory for visualizations")

    args = parser.parse_args()

    if args.frame_dir:
        visualize_frame(Path(args.frame_dir), Path(args.output_dir) if args.output_dir else None)
    elif args.root_dir:
        batch_visualize(Path(args.root_dir), Path(args.output_dir) if args.output_dir else None)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
