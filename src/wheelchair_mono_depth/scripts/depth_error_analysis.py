#!/usr/bin/env python3
"""Per-pixel error analysis: stereo GT vs DA3Metric-Large predicted depth.

Computes per-band metrics, per-camera breakdown, error heatmaps, and
distribution plots. Accounts for stereo noise floor (D455: 36mm@3m,
D435i: 68mm@3m).

Run batch_da3_inference.py first to generate XXXXXX_da3_depth.png files.

Usage:
    python3 -m wheelchair_mono_depth.scripts.depth_error_analysis \
        --data_dir /home/sidd/wheelchair_nav/mono_depth_data \
        --output_dir /home/sidd/wheelchair_nav/eval_output/error_analysis

    # Single session:
    python3 -m wheelchair_mono_depth.scripts.depth_error_analysis \
        --data_dir /home/sidd/wheelchair_nav/mono_depth_data \
        --session 20260304_154230
"""

import argparse
import csv
import json
import os
import sys

import cv2
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Distance bands for per-range analysis
DISTANCE_BANDS = [
    ('0-1m', 0.0, 1.0),
    ('1-2m', 1.0, 2.0),
    ('2-3m', 2.0, 3.0),
    ('3-5m', 3.0, 5.0),
]

# Stereo noise floor (mm) at reference distances (from RealSense specs)
# Error ∝ z²/(f×b). D435i is 1.9x worse than D455.
STEREO_NOISE = {
    'front': {'type': 'D455', 'noise_3m_mm': 36.0, 'baseline_mm': 95.0},
    'left':  {'type': 'D455', 'noise_3m_mm': 36.0, 'baseline_mm': 95.0},
    'right': {'type': 'D435i', 'noise_3m_mm': 68.0, 'baseline_mm': 50.0},
}


def stereo_noise_at_depth(camera, depth_m):
    """Estimate stereo noise floor at given depth (meters).

    Noise scales as z²/(f×b), so noise_at_z = noise_at_3m × (z/3)².
    Returns noise in meters.
    """
    info = STEREO_NOISE.get(camera, STEREO_NOISE['front'])
    noise_3m_m = info['noise_3m_mm'] / 1000.0
    return noise_3m_m * (depth_m / 3.0) ** 2


def compute_band_metrics(pred, gt, valid, band_lo, band_hi):
    """Compute metrics for a specific depth band."""
    band_mask = valid & (gt >= band_lo) & (gt < band_hi)
    pred_v = pred[band_mask]
    gt_v = gt[band_mask]

    if len(pred_v) < 10:
        return None

    abs_err = np.abs(pred_v - gt_v)
    return {
        'n_pixels': len(pred_v),
        'mae': float(np.mean(abs_err)),
        'rmse': float(np.sqrt(np.mean(abs_err ** 2))),
        'abs_rel': float(np.mean(abs_err / np.clip(gt_v, 1e-3, None))),
        'delta_1': float(np.mean(
            np.maximum(pred_v / gt_v, gt_v / pred_v) < 1.25)),
        'median_err': float(np.median(abs_err)),
    }


def compute_frame_metrics(pred_m, gt_m, camera, max_depth=6.0):
    """Compute full metrics for one frame."""
    valid = (gt_m > 0.01) & (gt_m <= max_depth) & (pred_m > 0.01)

    pred_v = pred_m[valid]
    gt_v = gt_m[valid]

    if len(pred_v) < 100:
        return None

    abs_err = np.abs(pred_v - gt_v)

    # Noise-corrected error: subtract estimated stereo noise floor
    noise_floor = stereo_noise_at_depth(camera, gt_v)
    corrected_err = np.maximum(0, abs_err - noise_floor)

    result = {
        'camera': camera,
        'n_valid': int(len(pred_v)),
        'valid_ratio': float(np.sum(valid) / valid.size),
        # Raw metrics
        'mae': float(np.mean(abs_err)),
        'rmse': float(np.sqrt(np.mean(abs_err ** 2))),
        'abs_rel': float(np.mean(abs_err / np.clip(gt_v, 1e-3, None))),
        'delta_1': float(np.mean(
            np.maximum(pred_v / gt_v, gt_v / pred_v) < 1.25)),
        'delta_2': float(np.mean(
            np.maximum(pred_v / gt_v, gt_v / pred_v) < 1.25 ** 2)),
        'silog': float(np.sqrt(
            np.mean((np.log(pred_v) - np.log(gt_v)) ** 2) -
            0.5 * np.mean(np.log(pred_v) - np.log(gt_v)) ** 2)),
        'median_err': float(np.median(abs_err)),
        # Noise-corrected
        'mae_corrected': float(np.mean(corrected_err)),
        'rmse_corrected': float(np.sqrt(np.mean(corrected_err ** 2))),
        'mean_noise_floor': float(np.mean(noise_floor)),
    }

    # Per-band metrics
    for band_name, lo, hi in DISTANCE_BANDS:
        band = compute_band_metrics(pred_m, gt_m, valid, lo, hi)
        if band:
            for k, v in band.items():
                result[f'{band_name}_{k}'] = v

    return result


def save_error_heatmap(pred_m, gt_m, save_path, max_depth=6.0, max_error=1.0):
    """Save error heatmap visualization."""
    valid = (gt_m > 0.01) & (gt_m <= max_depth) & (pred_m > 0.01)
    error = np.zeros_like(gt_m)
    error[valid] = np.abs(pred_m[valid] - gt_m[valid])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # GT depth
    gt_vis = np.where(valid, gt_m, np.nan)
    im0 = axes[0].imshow(gt_vis, cmap='turbo', vmin=0, vmax=max_depth)
    axes[0].set_title('Stereo GT Depth (m)')
    axes[0].axis('off')
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    # DA3 predicted depth
    pred_vis = np.where(valid, pred_m, np.nan)
    im1 = axes[1].imshow(pred_vis, cmap='turbo', vmin=0, vmax=max_depth)
    axes[1].set_title('DA3 Predicted Depth (m)')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    # Error heatmap
    error_vis = np.where(valid, error, np.nan)
    im2 = axes[2].imshow(error_vis, cmap='hot', vmin=0, vmax=max_error)
    axes[2].set_title('Absolute Error (m)')
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()


def save_scatter_plot(all_metrics, save_path):
    """Save error vs distance scatter from aggregated per-band data."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    cameras = sorted(set(m['camera'] for m in all_metrics))
    colors = {'front': '#2196F3', 'left': '#4CAF50', 'right': '#FF5722'}

    for camera in cameras:
        cam_metrics = [m for m in all_metrics if m['camera'] == camera]
        band_centers = []
        band_maes = []
        band_absrels = []

        for band_name, lo, hi in DISTANCE_BANDS:
            key_mae = f'{band_name}_mae'
            vals = [m[key_mae] for m in cam_metrics if key_mae in m]
            if vals:
                band_centers.append((lo + hi) / 2)
                band_maes.append(np.mean(vals))

            key_ar = f'{band_name}_abs_rel'
            vals_ar = [m[key_ar] for m in cam_metrics if key_ar in m]
            if vals_ar:
                band_absrels.append(np.mean(vals_ar))

        color = colors.get(camera, '#999999')
        if band_centers:
            axes[0].plot(band_centers, band_maes, 'o-', color=color,
                         label=f'{camera} ({STEREO_NOISE[camera]["type"]})',
                         linewidth=2, markersize=8)
        if band_absrels and len(band_absrels) == len(band_centers):
            axes[1].plot(band_centers, band_absrels, 's-', color=color,
                         label=f'{camera} ({STEREO_NOISE[camera]["type"]})',
                         linewidth=2, markersize=8)

    axes[0].set_xlabel('Depth (m)')
    axes[0].set_ylabel('MAE (m)')
    axes[0].set_title('Mean Absolute Error vs Distance')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Depth (m)')
    axes[1].set_ylabel('AbsRel')
    axes[1].set_title('Absolute Relative Error vs Distance')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_summary_table(all_metrics, save_path):
    """Save per-camera summary as formatted text."""
    cameras = sorted(set(m['camera'] for m in all_metrics))

    lines = []
    lines.append(f'{"Camera":<10} {"Type":<6} {"N":>6} '
                 f'{"MAE":>7} {"RMSE":>7} {"AbsRel":>7} '
                 f'{"d<1.25":>7} {"SiLog":>7} '
                 f'{"MAE_corr":>8} {"NoiseFlr":>8}')
    lines.append('-' * 85)

    for camera in cameras:
        cam_m = [m for m in all_metrics if m['camera'] == camera]
        n = len(cam_m)
        cam_type = STEREO_NOISE[camera]['type']
        mae = np.mean([m['mae'] for m in cam_m])
        rmse = np.mean([m['rmse'] for m in cam_m])
        absrel = np.mean([m['abs_rel'] for m in cam_m])
        d1 = np.mean([m['delta_1'] for m in cam_m])
        silog = np.mean([m['silog'] for m in cam_m])
        mae_c = np.mean([m['mae_corrected'] for m in cam_m])
        nf = np.mean([m['mean_noise_floor'] for m in cam_m])

        lines.append(f'{camera:<10} {cam_type:<6} {n:>6} '
                     f'{mae:>7.4f} {rmse:>7.4f} {absrel:>7.4f} '
                     f'{d1:>7.3f} {silog:>7.4f} '
                     f'{mae_c:>8.4f} {nf:>8.4f}')

    # Overall
    n = len(all_metrics)
    lines.append('-' * 85)
    lines.append(f'{"ALL":<10} {"":6} {n:>6} '
                 f'{np.mean([m["mae"] for m in all_metrics]):>7.4f} '
                 f'{np.mean([m["rmse"] for m in all_metrics]):>7.4f} '
                 f'{np.mean([m["abs_rel"] for m in all_metrics]):>7.4f} '
                 f'{np.mean([m["delta_1"] for m in all_metrics]):>7.3f} '
                 f'{np.mean([m["silog"] for m in all_metrics]):>7.4f} '
                 f'{np.mean([m["mae_corrected"] for m in all_metrics]):>8.4f} '
                 f'{np.mean([m["mean_noise_floor"] for m in all_metrics]):>8.4f}')

    # Per-band summary
    lines.append('')
    lines.append('Per-Band Summary (all cameras):')
    lines.append(f'  {"Band":<8} {"MAE":>7} {"RMSE":>7} {"AbsRel":>7} '
                 f'{"d<1.25":>7} {"N_pixels":>10}')
    lines.append('  ' + '-' * 52)
    for band_name, _, _ in DISTANCE_BANDS:
        key_mae = f'{band_name}_mae'
        vals_mae = [m[key_mae] for m in all_metrics if key_mae in m]
        key_rmse = f'{band_name}_rmse'
        vals_rmse = [m[key_rmse] for m in all_metrics if key_rmse in m]
        key_ar = f'{band_name}_abs_rel'
        vals_ar = [m[key_ar] for m in all_metrics if key_ar in m]
        key_d1 = f'{band_name}_delta_1'
        vals_d1 = [m[key_d1] for m in all_metrics if key_d1 in m]
        key_n = f'{band_name}_n_pixels'
        vals_n = [m[key_n] for m in all_metrics if key_n in m]

        if vals_mae:
            lines.append(
                f'  {band_name:<8} {np.mean(vals_mae):>7.4f} '
                f'{np.mean(vals_rmse):>7.4f} {np.mean(vals_ar):>7.4f} '
                f'{np.mean(vals_d1):>7.3f} {int(np.mean(vals_n)):>10}')

    text = '\n'.join(lines)
    with open(save_path, 'w') as f:
        f.write(text)
    print(text)


def find_sessions(data_dir, session_filter=None):
    """Find valid session directories."""
    sessions = []
    for d in sorted(os.listdir(data_dir)):
        session_path = os.path.join(data_dir, d)
        if not os.path.isdir(session_path):
            continue
        if not os.path.exists(os.path.join(session_path, 'metadata.json')):
            continue
        if session_filter and d not in session_filter:
            continue
        sessions.append(session_path)
    return sessions


def main():
    parser = argparse.ArgumentParser(
        description='Depth error analysis: stereo GT vs DA3 predictions')
    parser.add_argument('--data_dir', required=True,
                        help='Root directory with session folders')
    parser.add_argument('--output_dir',
                        default=None,
                        help='Output directory (default: data_dir/error_analysis)')
    parser.add_argument('--session', nargs='*', default=None,
                        help='Specific session IDs (default: all)')
    parser.add_argument('--cameras', nargs='*',
                        default=['front', 'left', 'right'])
    parser.add_argument('--max_depth', type=float, default=6.0)
    parser.add_argument('--num_heatmaps', type=int, default=10,
                        help='Number of error heatmap visualizations to save')
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(args.data_dir, 'error_analysis')
    os.makedirs(args.output_dir, exist_ok=True)

    sessions = find_sessions(args.data_dir, args.session)
    if not sessions:
        print(f'No valid sessions found in {args.data_dir}')
        sys.exit(1)

    print(f'Analyzing {len(sessions)} session(s), '
          f'cameras: {args.cameras}')

    all_metrics = []
    heatmap_count = 0

    # CSV output
    csv_path = os.path.join(args.output_dir, 'frame_metrics.csv')
    csv_fields = None

    for session_dir in sessions:
        session_name = os.path.basename(session_dir)
        print(f'\nSession: {session_name}')

        for camera in args.cameras:
            cam_dir = os.path.join(session_dir, camera)
            if not os.path.isdir(cam_dir):
                continue

            files = os.listdir(cam_dir)
            da3_files = sorted([f for f in files if f.endswith('_da3_depth.png')])

            if not da3_files:
                print(f'  [{camera}] no DA3 depth files (run batch_da3_inference.py first)')
                continue

            cam_metrics = []
            for da3_file in da3_files:
                frame_id = da3_file[:6]
                gt_path = os.path.join(cam_dir, f'{frame_id}_depth.png')
                da3_path = os.path.join(cam_dir, da3_file)

                if not os.path.exists(gt_path):
                    continue

                # Load depths (uint16 mm -> float32 m)
                gt_mm = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
                da3_mm = cv2.imread(da3_path, cv2.IMREAD_UNCHANGED)

                if gt_mm is None or da3_mm is None:
                    continue

                gt_m = gt_mm.astype(np.float32) / 1000.0
                da3_m = da3_mm.astype(np.float32) / 1000.0

                # Resize DA3 to match GT if needed
                if da3_m.shape != gt_m.shape:
                    da3_m = cv2.resize(da3_m, (gt_m.shape[1], gt_m.shape[0]),
                                       interpolation=cv2.INTER_LINEAR)

                metrics = compute_frame_metrics(
                    da3_m, gt_m, camera, args.max_depth)
                if metrics is None:
                    continue

                metrics['session'] = session_name
                metrics['frame_id'] = frame_id
                cam_metrics.append(metrics)

                # Save heatmaps for a sample of frames
                if heatmap_count < args.num_heatmaps:
                    heatmap_path = os.path.join(
                        args.output_dir,
                        f'heatmap_{session_name}_{camera}_{frame_id}.png')
                    save_error_heatmap(da3_m, gt_m, heatmap_path,
                                       args.max_depth)
                    heatmap_count += 1

            if cam_metrics:
                avg_mae = np.mean([m['mae'] for m in cam_metrics])
                avg_ar = np.mean([m['abs_rel'] for m in cam_metrics])
                avg_d1 = np.mean([m['delta_1'] for m in cam_metrics])
                print(f'  [{camera}] {len(cam_metrics)} frames | '
                      f'MAE={avg_mae:.4f}m  AbsRel={avg_ar:.4f}  '
                      f'd<1.25={avg_d1:.3f}')
                all_metrics.extend(cam_metrics)

    if not all_metrics:
        print('\nNo metrics computed. Check that DA3 depth files exist.')
        sys.exit(1)

    # Write CSV — union all keys since not every frame has every band
    all_keys = set()
    for m in all_metrics:
        all_keys.update(m.keys())
    csv_fields = sorted(all_keys)
    csv_path = os.path.join(args.output_dir, 'frame_metrics.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(all_metrics)
    print(f'\nCSV saved: {csv_path} ({len(all_metrics)} rows)')

    # Summary table
    summary_path = os.path.join(args.output_dir, 'summary.txt')
    print(f'\n{"="*85}')
    save_summary_table(all_metrics, summary_path)
    print(f'\nSummary saved: {summary_path}')

    # Scatter plots
    scatter_path = os.path.join(args.output_dir, 'error_vs_distance.png')
    save_scatter_plot(all_metrics, scatter_path)
    print(f'Scatter plot saved: {scatter_path}')

    print(f'\nAll outputs in: {args.output_dir}')


if __name__ == '__main__':
    main()
