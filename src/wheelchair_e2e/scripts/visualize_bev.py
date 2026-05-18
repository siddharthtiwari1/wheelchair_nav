#!/usr/bin/env python3
"""
BEV Grid Visualization Tool

Visualizes 4-channel BEV grids from training data.
Useful for debugging BEV generation quality.

Usage:
    python visualize_bev.py --data_dir /path/to/training_data --idx 0
    python visualize_bev.py --data_dir /path/to/training_data --range 0 10
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def visualize_single(bev, label=None, idx=None, save_path=None):
    """
    Visualize a single 4-channel BEV grid.

    Args:
        bev: (4, 200, 200) numpy array
        label: optional (v, omega) velocity label
        idx: sample index for title
        save_path: optional path to save figure
    """
    fig = plt.figure(figsize=(16, 4))
    gs = GridSpec(1, 5, figure=fig, width_ratios=[1, 1, 1, 1, 0.05])

    channel_names = [
        'Ch0: LiDAR Occupancy',
        'Ch1: Depth Camera Occupancy',
        'Ch2: Goal Direction',
        'Ch3: Odometry Trail'
    ]

    cmaps = ['Reds', 'Oranges', 'Greens', 'Blues']

    for i in range(4):
        ax = fig.add_subplot(gs[0, i])
        im = ax.imshow(bev[i], cmap=cmaps[i], vmin=0, vmax=1,
                        origin='lower')
        ax.set_title(channel_names[i], fontsize=9)
        ax.set_xlabel('x (pixels)')
        if i == 0:
            ax.set_ylabel('y (pixels)')

        # Mark center (wheelchair position)
        center = bev.shape[1] // 2
        ax.plot(center, center, 'k+', markersize=10, markeredgewidth=2)

    # Colorbar
    cax = fig.add_subplot(gs[0, 4])
    plt.colorbar(im, cax=cax)

    # Title
    title = f"BEV Grid"
    if idx is not None:
        title += f" (sample {idx})"
    if label is not None:
        title += f" | v={label[0]:.3f} m/s, ω={label[1]:.3f} rad/s"
    fig.suptitle(title, fontsize=11)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()
    plt.close()


def visualize_combined(bev, label=None, idx=None, save_path=None):
    """Visualize all channels overlaid in a single image."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # Create RGB image from channels
    h, w = bev.shape[1], bev.shape[2]
    rgb = np.zeros((h, w, 3), dtype=np.float32)

    # Red = obstacles (ch0 + ch1)
    rgb[:, :, 0] = np.clip(bev[0] + bev[1], 0, 1)
    # Green = goal direction (ch2)
    rgb[:, :, 1] = bev[2]
    # Blue = odometry trail (ch3)
    rgb[:, :, 2] = bev[3]

    ax.imshow(rgb, origin='lower')

    # Mark center
    center = h // 2
    ax.plot(center, center, 'w+', markersize=15, markeredgewidth=2)
    ax.set_title('Combined BEV (R=obstacles, G=goal, B=odom)')

    if label is not None:
        ax.text(5, h - 10,
                f"v={label[0]:.3f} m/s, ω={label[1]:.3f} rad/s",
                color='white', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize BEV training data')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--idx', type=int, default=None,
                        help='Single sample index')
    parser.add_argument('--range', type=int, nargs=2, default=None,
                        help='Range of samples (start end)')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Save figures to directory')
    parser.add_argument('--combined', action='store_true',
                        help='Show combined overlay view')
    args = parser.parse_args()

    # Load labels
    labels_path = os.path.join(args.data_dir, 'labels.npy')
    labels = np.load(labels_path) if os.path.exists(labels_path) else None

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    # Determine indices to visualize
    if args.idx is not None:
        indices = [args.idx]
    elif args.range is not None:
        indices = list(range(args.range[0], args.range[1]))
    else:
        indices = [0]

    for idx in indices:
        bev_path = os.path.join(args.data_dir, f'bev_{idx:06d}.npy')
        if not os.path.exists(bev_path):
            print(f"Skipping {idx}: file not found")
            continue

        bev = np.load(bev_path)
        label = labels[idx] if labels is not None else None

        save_path = None
        if args.save_dir:
            save_path = os.path.join(args.save_dir, f'bev_{idx:06d}.png')

        if args.combined:
            visualize_combined(bev, label, idx, save_path)
        else:
            visualize_single(bev, label, idx, save_path)

    print(f"Visualized {len(indices)} samples")


if __name__ == '__main__':
    main()
