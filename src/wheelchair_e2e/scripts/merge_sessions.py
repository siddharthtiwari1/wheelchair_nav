#!/usr/bin/env python3
"""
Merge multiple extracted session directories into a single training dataset.

Usage:
    python scripts/merge_sessions.py \
        --input_dirs /path/to/training_data/session_* \
        --output_dir /path/to/training_data/merged
"""

import argparse
import glob
import os
import shutil
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description='Merge session training data')
    parser.add_argument('--input_dirs', type=str, nargs='+', required=True,
                        help='Session directories to merge')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output merged directory')
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='Fraction of data for validation (default 0.1)')
    return parser.parse_args()


def main():
    args = parse_args()

    # Expand globs
    input_dirs = []
    for pattern in args.input_dirs:
        expanded = sorted(glob.glob(pattern))
        input_dirs.extend(expanded)

    # Filter to only dirs that contain labels.npy
    input_dirs = [d for d in input_dirs
                  if os.path.isfile(os.path.join(d, 'labels.npy'))]

    if not input_dirs:
        print("ERROR: No valid session directories found")
        return

    print(f"Merging {len(input_dirs)} sessions:")

    # Collect all data
    all_labels = []
    all_scan_ranges = []
    all_scan_odom = []
    all_goal_relative = []
    all_traj_poses = []
    all_odom_flat = []
    bev_files = []  # Track (src_dir, bev_index, odom_index) for copying

    total_offset = 0
    for d in input_dirs:
        name = os.path.basename(d)
        labels = np.load(os.path.join(d, 'labels.npy'))
        n = len(labels)
        print(f"  {name}: {n} samples")

        all_labels.append(labels)

        # v2 data
        sr_path = os.path.join(d, 'scan_ranges.npy')
        if os.path.exists(sr_path):
            all_scan_ranges.append(np.load(sr_path))
            all_scan_odom.append(np.load(os.path.join(d, 'scan_odom.npy')))
            all_goal_relative.append(
                np.load(os.path.join(d, 'goal_relative.npy')))

        tp_path = os.path.join(d, 'traj_poses.npy')
        if os.path.exists(tp_path):
            all_traj_poses.append(np.load(tp_path))

        # Track per-sample BEV and odom files for copying
        for i in range(n):
            bev_files.append((d, i))

        total_offset += n

    # Merge arrays
    labels = np.concatenate(all_labels, axis=0)
    n_total = len(labels)
    print(f"\nTotal: {n_total} samples")

    # Create output dirs
    train_dir = os.path.join(args.output_dir, 'train')
    val_dir = os.path.join(args.output_dir, 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Shuffle and split
    indices = np.random.RandomState(42).permutation(n_total)
    n_val = max(1, int(n_total * args.val_split))
    val_indices = set(indices[:n_val].tolist())
    train_indices = indices[n_val:]

    print(f"Train: {len(train_indices)}, Val: {n_val}")

    # Save merged arrays for each split
    for split_name, split_dir, split_idx in [
        ('train', train_dir, train_indices),
        ('val', val_dir, np.array(sorted(val_indices))),
    ]:
        idx = np.sort(split_idx)

        np.save(os.path.join(split_dir, 'labels.npy'),
                labels[idx])

        if all_scan_ranges:
            scan_ranges = np.concatenate(all_scan_ranges, axis=0)
            scan_odom = np.concatenate(all_scan_odom, axis=0)
            goal_rel = np.concatenate(all_goal_relative, axis=0)
            np.save(os.path.join(split_dir, 'scan_ranges.npy'),
                    scan_ranges[idx])
            np.save(os.path.join(split_dir, 'scan_odom.npy'),
                    scan_odom[idx])
            np.save(os.path.join(split_dir, 'goal_relative.npy'),
                    goal_rel[idx])

        if all_traj_poses:
            traj_poses = np.concatenate(all_traj_poses, axis=0)
            np.save(os.path.join(split_dir, 'traj_poses.npy'),
                    traj_poses[idx])

        # Copy BEV and odom .npy files with new sequential indexing
        for new_i, old_i in enumerate(idx):
            src_dir, src_idx = bev_files[old_i]

            bev_src = os.path.join(src_dir, f'bev_{src_idx:06d}.npy')
            odom_src = os.path.join(src_dir, f'odom_{src_idx:06d}.npy')

            bev_dst = os.path.join(split_dir, f'bev_{new_i:06d}.npy')
            odom_dst = os.path.join(split_dir, f'odom_{new_i:06d}.npy')

            if os.path.exists(bev_src):
                shutil.copy2(bev_src, bev_dst)
            if os.path.exists(odom_src):
                shutil.copy2(odom_src, odom_dst)

        print(f"  {split_name}: {len(idx)} samples saved to {split_dir}")

    # Stats
    print(f"\nLabel statistics:")
    print(f"  v range:  [{labels[:, 0].min():.3f}, "
          f"{labels[:, 0].max():.3f}] m/s")
    print(f"  w range:  [{labels[:, 1].min():.3f}, "
          f"{labels[:, 1].max():.3f}] rad/s")
    print(f"  v mean:   {labels[:, 0].mean():.3f} +/- "
          f"{labels[:, 0].std():.3f}")
    print(f"  w mean:   {labels[:, 1].mean():.3f} +/- "
          f"{labels[:, 1].std():.3f}")


if __name__ == '__main__':
    main()
