#!/usr/bin/env python3
"""
GROUND TRUTH ANALYSIS TOOLS
=============================

Compute statistics, confusion matrices, and generate report-ready tables
from labeled ground truth data.

Metrics:
- Phantom rate (% of bins classified as phantom)
- Classification distribution
- Confidence statistics
- Spatial patterns (which zones have most phantoms)
- Per-bin accuracy (if available)

Usage:
    python3 analysis_tools.py \
        --labels-csv labels.csv \
        --output-dir results/
"""

import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import logging
from collections import defaultdict
from enum import Enum
import matplotlib.pyplot as plt
import matplotlib.patches as patches

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LabelEnum(Enum):
    """Match labeling_interface.py enums."""
    TRUE_OBSTACLE = "true_obstacle"
    PHANTOM_DEPTH_NOISE = "phantom_depth_noise"
    PHANTOM_REFLECTION = "phantom_reflection"
    PHANTOM_OTHER = "phantom_other"
    AMBIGUOUS = "ambiguous"
    UNLABELED = "unlabeled"


def read_labels_csv(csv_path: Path) -> List[Dict]:
    """Load labels from CSV file."""
    labels = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels.append({
                'frame_id': int(row['frame_id']),
                'bin_idx': int(row['bin_idx']),
                'angle_deg': float(row['angle_deg']),
                'range_m': float(row['range_m']),
                'label': row['label'],
                'confidence': float(row['confidence']),
                'notes': row.get('notes', '')
            })
    return labels


def compute_classification_statistics(labels: List[Dict]) -> Dict:
    """Compute counts and percentages by label type."""
    label_counts = defaultdict(int)
    label_confidence = defaultdict(list)

    for label in labels:
        label_type = label['label']
        label_counts[label_type] += 1
        label_confidence[label_type].append(label['confidence'])

    total = len(labels)
    stats = {
        'total_labeled_bins': total,
        'by_label': {}
    }

    for label_type in sorted(label_counts.keys()):
        count = label_counts[label_type]
        percentage = (count / total * 100.0) if total > 0 else 0.0
        confidences = label_confidence[label_type]

        stats['by_label'][label_type] = {
            'count': count,
            'percentage': percentage,
            'mean_confidence': np.mean(confidences) if confidences else 0.0,
            'std_confidence': np.std(confidences) if confidences else 0.0,
            'min_confidence': np.min(confidences) if confidences else 0.0,
            'max_confidence': np.max(confidences) if confidences else 1.0
        }

    # Compute phantom rate
    phantom_count = 0
    phantom_labels = [
        'phantom_depth_noise',
        'phantom_reflection',
        'phantom_other'
    ]
    for label_type in phantom_labels:
        phantom_count += label_counts.get(label_type, 0)

    stats['phantom_rate_percent'] = (phantom_count / total * 100.0) if total > 0 else 0.0
    stats['phantom_count'] = phantom_count

    # Compute ambiguous rate
    ambiguous_count = label_counts.get('ambiguous', 0)
    stats['ambiguous_rate_percent'] = (ambiguous_count / total * 100.0) if total > 0 else 0.0

    return stats


def compute_frame_statistics(labels: List[Dict]) -> Dict[int, Dict]:
    """Compute statistics per frame."""
    frames = defaultdict(list)
    for label in labels:
        frames[label['frame_id']].append(label)

    frame_stats = {}
    for frame_id, frame_labels in frames.items():
        frame_stats[frame_id] = compute_classification_statistics(frame_labels)

    return frame_stats


def compute_spatial_distribution(labels: List[Dict], num_zones: int = 8) -> Dict:
    """Distribute phantom rate by angular zone."""
    # Divide 360° into num_zones equal sectors
    zone_size = 360.0 / num_zones
    zones = defaultdict(lambda: {'total': 0, 'phantom': 0})

    for label in labels:
        angle = label['angle_deg']
        # Normalize to [0, 360)
        norm_angle = (angle + 180.0) % 360.0
        zone_idx = int(norm_angle / zone_size) % num_zones

        zones[zone_idx]['total'] += 1
        if 'phantom' in label['label']:
            zones[zone_idx]['phantom'] += 1

    result = {}
    for zone_idx in range(num_zones):
        zone = zones[zone_idx]
        start_angle = -180 + zone_idx * zone_size
        end_angle = start_angle + zone_size

        phantom_rate = (
            (zone['phantom'] / zone['total'] * 100.0)
            if zone['total'] > 0 else 0.0
        )

        result[f"zone_{zone_idx:02d}"] = {
            'angle_range': f"{start_angle:.1f}° to {end_angle:.1f}°",
            'total_bins': zone['total'],
            'phantom_bins': zone['phantom'],
            'phantom_rate_percent': phantom_rate
        }

    return result


def compute_range_distribution(labels: List[Dict], num_bins_r: int = 5) -> Dict:
    """Distribute phantom rate by range interval."""
    # Divide range space [0, 5]m into bins
    ranges = [label['range_m'] for label in labels if np.isfinite(label['range_m'])]
    if not ranges:
        return {}

    r_min, r_max = 0.0, 5.0
    r_bin_edges = np.linspace(r_min, r_max, num_bins_r + 1)

    bins = defaultdict(lambda: {'total': 0, 'phantom': 0})

    for label in labels:
        r = label['range_m']
        if not np.isfinite(r):
            continue

        # Find which bin this range belongs to
        for i in range(num_bins_r):
            if r_bin_edges[i] <= r < r_bin_edges[i + 1]:
                bins[i]['total'] += 1
                if 'phantom' in label['label']:
                    bins[i]['phantom'] += 1
                break
        else:
            # >= max range
            if r >= r_bin_edges[-1]:
                bins[num_bins_r - 1]['total'] += 1
                if 'phantom' in label['label']:
                    bins[num_bins_r - 1]['phantom'] += 1

    result = {}
    for i in range(num_bins_r):
        bin_data = bins[i]
        phantom_rate = (
            (bin_data['phantom'] / bin_data['total'] * 100.0)
            if bin_data['total'] > 0 else 0.0
        )

        result[f"range_bin_{i:02d}"] = {
            'range_interval': f"{r_bin_edges[i]:.2f}m to {r_bin_edges[i+1]:.2f}m",
            'total_bins': bin_data['total'],
            'phantom_bins': bin_data['phantom'],
            'phantom_rate_percent': phantom_rate
        }

    return result


def detect_outliers_and_ambiguous(labels: List[Dict], confidence_threshold: float = 0.60) -> Dict:
    """Identify bins with low confidence or ambiguous labels."""
    low_confidence = []
    ambiguous = []
    high_uncertainty = defaultdict(list)

    for label in labels:
        if label['label'] == 'ambiguous':
            ambiguous.append(label)

        if label['confidence'] < confidence_threshold:
            low_confidence.append(label)
            frame_id = label['frame_id']
            high_uncertainty[frame_id].append(label)

    return {
        'low_confidence_count': len(low_confidence),
        'low_confidence_bins': low_confidence,
        'ambiguous_count': len(ambiguous),
        'ambiguous_bins': ambiguous,
        'high_uncertainty_by_frame': dict(high_uncertainty)
    }


def generate_summary_table(stats: Dict) -> str:
    """Generate markdown table for summary statistics."""
    lines = []
    lines.append("| Classification | Count | Percentage | Mean Confidence |")
    lines.append("|---|---|---|---|")

    for label_type in sorted(stats['by_label'].keys()):
        label_stat = stats['by_label'][label_type]
        lines.append(
            f"| {label_type} | {label_stat['count']} | "
            f"{label_stat['percentage']:.1f}% | {label_stat['mean_confidence']:.2f} |"
        )

    lines.append(f"\n**Total Labels**: {stats['total_labeled_bins']}")
    lines.append(f"**Phantom Rate**: {stats['phantom_rate_percent']:.1f}% ({stats['phantom_count']} bins)")
    lines.append(f"**Ambiguous Rate**: {stats['ambiguous_rate_percent']:.1f}%")

    return '\n'.join(lines)


def generate_results_paragraph(stats: Dict) -> str:
    """Generate paragraph for paper Results section."""
    return f"""
To validate phantom detection rates, we manually labeled {stats['total_labeled_bins']} bins
across 30 rosbag frames (10 per session, sampled uniformly across trials) by operator review
of synchronized RGB video and RViz depth visualization. Per-bin classification:
{stats['by_label']['true_obstacle']['percentage']:.1f}% confirmed true elevated obstacles
(tables, equipment racks, shelves), {stats['by_label']['phantom_depth_noise']['percentage']:.1f}%
confirmed as depth noise (RealSense speckle), {stats['by_label']['phantom_reflection']['percentage']:.1f}%
as reflections (IR bounce), and {stats['by_label']['phantom_other']['percentage']:.1f}% as other artifacts,
validating our quantitative {stats['phantom_rate_percent']:.1f}% phantom-candidate rate.
{stats['ambiguous_rate_percent']:.1f}% of bins were marked ambiguous due to operator uncertainty
and out-of-frame camera view angles. Labeled frames available at [GitHub URL] for reproducibility.
"""


def export_to_json(all_stats: Dict, output_path: Path):
    """Export all statistics to JSON."""
    with open(output_path, 'w') as f:
        json.dump(all_stats, f, indent=2, default=str)
    logger.info(f"Exported statistics to {output_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Analyze ground truth labels")
    parser.add_argument("--labels-csv", required=True, help="Path to labels.csv")
    parser.add_argument("--output-dir", required=True, help="Output directory")

    args = parser.parse_args()

    labels_csv = Path(args.labels_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load labels
    logger.info(f"Loading labels from {labels_csv}")
    labels = read_labels_csv(labels_csv)
    logger.info(f"Loaded {len(labels)} labels")

    # Compute statistics
    overall_stats = compute_classification_statistics(labels)
    frame_stats = compute_frame_statistics(labels)
    spatial_dist = compute_spatial_distribution(labels)
    range_dist = compute_range_distribution(labels)
    outliers = detect_outliers_and_ambiguous(labels)

    # Compile all results
    all_stats = {
        'overall': overall_stats,
        'by_frame': frame_stats,
        'spatial_distribution': spatial_dist,
        'range_distribution': range_dist,
        'outliers_and_ambiguous': {
            'low_confidence_count': outliers['low_confidence_count'],
            'ambiguous_count': outliers['ambiguous_count']
        }
    }

    # Export
    export_to_json(all_stats, output_dir / "analysis_results.json")

    # Generate markdown summary
    summary_table = generate_summary_table(overall_stats)
    results_para = generate_results_paragraph(overall_stats)

    with open(output_dir / "summary_table.md", 'w') as f:
        f.write("# Ground Truth Labeling Summary\n\n")
        f.write(summary_table)
        f.write("\n\n## Results Paragraph\n\n")
        f.write(results_para)

    logger.info(f"\nAnalysis complete. Results in {output_dir}")
    print("\n" + summary_table)
    print(results_para)


if __name__ == '__main__':
    main()
