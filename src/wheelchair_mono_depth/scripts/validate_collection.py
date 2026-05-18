#!/usr/bin/env python3
"""Validate collected RGB-depth data quality.

Scans a collection session directory and reports statistics:
- Frame counts per camera
- Depth validity (non-zero pixel percentage, mean/median depth)
- RGB quality (brightness variance check)
- Odometry coverage (distance traveled, duration)
- Missing or corrupted files

Usage:
    python3 validate_collection.py /path/to/mono_depth_data/session_id
    python3 validate_collection.py /path/to/mono_depth_data/session_id --verbose
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np


def validate_session(session_dir, verbose=False):
    if not os.path.isdir(session_dir):
        print(f'Error: {session_dir} is not a directory')
        return False

    # Load metadata
    meta_path = os.path.join(session_dir, 'metadata.json')
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            metadata = json.load(f)
        print(f"Session: {metadata.get('session_id', 'unknown')}")
        print(f"Created: {metadata.get('created', 'unknown')}")
        print(f"Save rate: {metadata.get('save_rate_hz', '?')} Hz")
    else:
        print('Warning: metadata.json not found')

    cameras = ['front', 'left', 'right']
    all_ok = True

    for cam in cameras:
        cam_dir = os.path.join(session_dir, cam)
        if not os.path.isdir(cam_dir):
            print(f'\n  [{cam}] MISSING - directory not found')
            all_ok = False
            continue

        print(f'\n  [{cam}]')

        # Check intrinsics
        intr_path = os.path.join(cam_dir, 'intrinsics.json')
        if os.path.exists(intr_path):
            with open(intr_path) as f:
                intr = json.load(f)
            print(f"    Intrinsics: {intr['width']}x{intr['height']}, "
                  f"fx={intr['fx']:.1f}, fy={intr['fy']:.1f}")
        else:
            print('    Intrinsics: MISSING')
            all_ok = False

        # Count frames
        rgb_files = sorted([f for f in os.listdir(cam_dir) if f.endswith('_rgb.png')])
        depth_files = sorted([f for f in os.listdir(cam_dir) if f.endswith('_depth.png')])
        odom_files = sorted([f for f in os.listdir(cam_dir) if f.endswith('_odom.json')])

        print(f'    Frames: {len(rgb_files)} RGB, {len(depth_files)} depth, {len(odom_files)} odom')

        # Check for mismatches
        rgb_ids = {f[:6] for f in rgb_files}
        depth_ids = {f[:6] for f in depth_files}
        odom_ids = {f[:6] for f in odom_files}
        missing_depth = rgb_ids - depth_ids
        missing_rgb = depth_ids - rgb_ids
        missing_odom = rgb_ids - odom_ids

        if missing_depth:
            print(f'    WARNING: {len(missing_depth)} RGB frames without depth')
            all_ok = False
        if missing_rgb:
            print(f'    WARNING: {len(missing_rgb)} depth frames without RGB')
            all_ok = False
        if missing_odom:
            print(f'    WARNING: {len(missing_odom)} frames without odometry')

        if not rgb_files:
            print('    No frames to analyze')
            continue

        # Sample frames for quality check
        sample_indices = np.linspace(0, len(rgb_files) - 1, min(20, len(rgb_files)), dtype=int)

        depth_valid_pcts = []
        depth_means = []
        rgb_variances = []

        for idx in sample_indices:
            frame_id = rgb_files[idx][:6]

            # Check RGB quality
            rgb_path = os.path.join(cam_dir, f'{frame_id}_rgb.png')
            rgb = cv2.imread(rgb_path)
            if rgb is None:
                print(f'    CORRUPT: {frame_id}_rgb.png')
                all_ok = False
                continue
            gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
            rgb_variances.append(float(gray.var()))

            # Check depth quality
            depth_path = os.path.join(cam_dir, f'{frame_id}_depth.png')
            depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if depth is None:
                print(f'    CORRUPT: {frame_id}_depth.png')
                all_ok = False
                continue
            valid_mask = depth > 0
            valid_pct = valid_mask.sum() / depth.size * 100
            depth_valid_pcts.append(valid_pct)
            if valid_mask.any():
                depth_means.append(float(depth[valid_mask].mean()))

        if depth_valid_pcts:
            mean_valid = np.mean(depth_valid_pcts)
            print(f'    Depth validity: {mean_valid:.1f}% '
                  f'(min={min(depth_valid_pcts):.1f}%, max={max(depth_valid_pcts):.1f}%)')
            if mean_valid < 30:
                print('    WARNING: Low depth validity (<30%)')
                all_ok = False

        if depth_means:
            mean_depth_m = np.mean(depth_means) / 1000.0
            print(f'    Mean depth: {mean_depth_m:.2f} m '
                  f'(range: {min(depth_means)/1000:.2f}-{max(depth_means)/1000:.2f} m)')

        if rgb_variances:
            mean_var = np.mean(rgb_variances)
            print(f'    RGB brightness variance: {mean_var:.1f}')
            if mean_var < 10:
                print('    WARNING: Very low RGB variance (dark/blank images)')
                all_ok = False

        # Odometry coverage
        if odom_files:
            first_odom_path = os.path.join(cam_dir, odom_files[0])
            last_odom_path = os.path.join(cam_dir, odom_files[-1])
            with open(first_odom_path) as f:
                first_odom = json.load(f)
            with open(last_odom_path) as f:
                last_odom = json.load(f)

            duration = last_odom['timestamp'] - first_odom['timestamp']
            dx = last_odom['position']['x'] - first_odom['position']['x']
            dy = last_odom['position']['y'] - first_odom['position']['y']
            displacement = (dx**2 + dy**2)**0.5

            print(f'    Duration: {duration:.1f}s, Displacement: {displacement:.2f}m')

            if verbose:
                # Compute total path length from sequential odom
                total_dist = 0.0
                prev_x = first_odom['position']['x']
                prev_y = first_odom['position']['y']
                for odom_file in odom_files[1::5]:  # Sample every 5th
                    with open(os.path.join(cam_dir, odom_file)) as f:
                        odom = json.load(f)
                    x, y = odom['position']['x'], odom['position']['y']
                    total_dist += ((x - prev_x)**2 + (y - prev_y)**2)**0.5
                    prev_x, prev_y = x, y
                print(f'    Total path (sampled): ~{total_dist:.2f}m')

    print(f'\n{"PASS" if all_ok else "WARNINGS FOUND"} - {session_dir}')
    return all_ok


def main():
    parser = argparse.ArgumentParser(description='Validate mono depth collection data')
    parser.add_argument('session_dir', help='Path to session directory')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show extra details')
    args = parser.parse_args()

    ok = validate_session(args.session_dir, verbose=args.verbose)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
