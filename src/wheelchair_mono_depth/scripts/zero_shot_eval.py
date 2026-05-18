#!/usr/bin/env python3
"""Zero-shot evaluation: fine-tuned model on unseen lab/building.

Loads a trained DA V2-Small checkpoint, runs inference on a NEW session
(not used in training), compares predicted vs stereo GT, and optionally
compares side-by-side with DA3Metric-Large baseline (from batch_da3_inference).

Proves: "model trained on Lab A works in Lab B."

Usage:
    # Evaluate fine-tuned model only:
    python3 -m wheelchair_mono_depth.scripts.zero_shot_eval \
        --checkpoint /home/sidd/wheelchair_nav/checkpoints/da2_small_wheelchair/best_model.pth \
        --eval_session /home/sidd/wheelchair_nav/mono_depth_data/20260305_lab_b \
        --encoder vits

    # Compare fine-tuned vs DA3Metric-Large baseline:
    python3 -m wheelchair_mono_depth.scripts.zero_shot_eval \
        --checkpoint /home/sidd/wheelchair_nav/checkpoints/da2_small_wheelchair/best_model.pth \
        --eval_session /home/sidd/wheelchair_nav/mono_depth_data/20260305_lab_b \
        --encoder vits --compare_da3
"""

import argparse
import csv
import json
import os
import sys

import cv2
import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from wheelchair_mono_depth.training.train import load_model, MODEL_CONFIGS
from wheelchair_mono_depth.training.metrics import compute_depth_metrics

# Reuse constants from error analysis
DISTANCE_BANDS = [
    ('0-1m', 0.0, 1.0),
    ('1-2m', 1.0, 2.0),
    ('2-3m', 2.0, 3.0),
    ('3-5m', 3.0, 5.0),
]

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_rgb(rgb_bgr, input_size=518):
    """Preprocess BGR image for DA V2 inference."""
    # BGR -> RGB
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)

    # Resize to model input size (multiple of 14)
    size = ((input_size + 13) // 14) * 14
    rgb_resized = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)

    # Normalize
    tensor = rgb_resized.astype(np.float32) / 255.0
    tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(tensor).permute(2, 0, 1).unsqueeze(0)

    return tensor


def compute_band_metrics(pred, gt, valid, band_lo, band_hi):
    """Compute metrics for a specific depth band."""
    band_mask = valid & (gt >= band_lo) & (gt < band_hi)
    pred_v = pred[band_mask]
    gt_v = gt[band_mask]

    if len(pred_v) < 10:
        return None

    abs_err = np.abs(pred_v - gt_v)
    return {
        'n_pixels': int(len(pred_v)),
        'mae': float(np.mean(abs_err)),
        'rmse': float(np.sqrt(np.mean(abs_err ** 2))),
        'abs_rel': float(np.mean(abs_err / np.clip(gt_v, 1e-3, None))),
        'delta_1': float(np.mean(
            np.maximum(pred_v / gt_v, gt_v / pred_v) < 1.25)),
    }


def evaluate_model(model, session_dir, cameras, device, input_size, max_depth):
    """Run model on all frames, return per-frame metrics."""
    model.eval()
    all_metrics = []

    for camera in cameras:
        cam_dir = os.path.join(session_dir, camera)
        if not os.path.isdir(cam_dir):
            continue

        files = os.listdir(cam_dir)
        rgb_files = sorted([f for f in files if f.endswith('_rgb.png')])

        for rgb_file in rgb_files:
            frame_id = rgb_file[:6]
            gt_path = os.path.join(cam_dir, f'{frame_id}_depth.png')

            if not os.path.exists(gt_path):
                continue

            # Load RGB
            rgb_bgr = cv2.imread(os.path.join(cam_dir, rgb_file),
                                 cv2.IMREAD_COLOR)
            if rgb_bgr is None:
                continue

            # Load GT depth
            gt_mm = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
            if gt_mm is None:
                continue
            gt_m = gt_mm.astype(np.float32) / 1000.0

            # Inference
            tensor = preprocess_rgb(rgb_bgr, input_size).to(device)
            with torch.no_grad():
                pred = model(tensor)
                if pred.dim() == 3:
                    pred = pred.unsqueeze(1)

            pred_m = pred.squeeze().cpu().numpy()

            # Resize prediction to match GT
            if pred_m.shape != gt_m.shape:
                pred_m = cv2.resize(pred_m, (gt_m.shape[1], gt_m.shape[0]),
                                    interpolation=cv2.INTER_LINEAR)

            pred_m = np.clip(pred_m, 0.0, max_depth)

            # Compute metrics
            valid = (gt_m > 0.01) & (gt_m <= max_depth) & (pred_m > 0.01)
            pred_v = pred_m[valid]
            gt_v = gt_m[valid]

            if len(pred_v) < 100:
                continue

            abs_err = np.abs(pred_v - gt_v)
            metrics = {
                'camera': camera,
                'frame_id': frame_id,
                'n_valid': int(len(pred_v)),
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
            }

            # Per-band
            for band_name, lo, hi in DISTANCE_BANDS:
                band = compute_band_metrics(pred_m, gt_m, valid, lo, hi)
                if band:
                    for k, v in band.items():
                        metrics[f'{band_name}_{k}'] = v

            all_metrics.append(metrics)

    return all_metrics


def evaluate_da3_baseline(session_dir, cameras, max_depth):
    """Load pre-computed DA3 depth maps and compute metrics vs GT."""
    all_metrics = []

    for camera in cameras:
        cam_dir = os.path.join(session_dir, camera)
        if not os.path.isdir(cam_dir):
            continue

        files = os.listdir(cam_dir)
        da3_files = sorted([f for f in files if f.endswith('_da3_depth.png')])

        for da3_file in da3_files:
            frame_id = da3_file[:6]
            gt_path = os.path.join(cam_dir, f'{frame_id}_depth.png')

            if not os.path.exists(gt_path):
                continue

            gt_mm = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
            da3_mm = cv2.imread(os.path.join(cam_dir, da3_file),
                                cv2.IMREAD_UNCHANGED)

            if gt_mm is None or da3_mm is None:
                continue

            gt_m = gt_mm.astype(np.float32) / 1000.0
            da3_m = da3_mm.astype(np.float32) / 1000.0

            if da3_m.shape != gt_m.shape:
                da3_m = cv2.resize(da3_m, (gt_m.shape[1], gt_m.shape[0]),
                                   interpolation=cv2.INTER_LINEAR)

            valid = (gt_m > 0.01) & (gt_m <= max_depth) & (da3_m > 0.01)
            pred_v = da3_m[valid]
            gt_v = gt_m[valid]

            if len(pred_v) < 100:
                continue

            abs_err = np.abs(pred_v - gt_v)
            metrics = {
                'camera': camera,
                'frame_id': frame_id,
                'n_valid': int(len(pred_v)),
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
            }

            for band_name, lo, hi in DISTANCE_BANDS:
                band = compute_band_metrics(da3_m, gt_m, valid, lo, hi)
                if band:
                    for k, v in band.items():
                        metrics[f'{band_name}_{k}'] = v

            all_metrics.append(metrics)

    return all_metrics


def print_comparison(finetuned_metrics, da3_metrics, output_dir):
    """Print side-by-side comparison table."""
    def summarize(metrics_list):
        if not metrics_list:
            return {}
        return {
            'mae': np.mean([m['mae'] for m in metrics_list]),
            'rmse': np.mean([m['rmse'] for m in metrics_list]),
            'abs_rel': np.mean([m['abs_rel'] for m in metrics_list]),
            'delta_1': np.mean([m['delta_1'] for m in metrics_list]),
            'silog': np.mean([m['silog'] for m in metrics_list]),
            'n_frames': len(metrics_list),
        }

    ft = summarize(finetuned_metrics)
    da3 = summarize(da3_metrics)

    lines = []
    lines.append('=' * 70)
    lines.append('ZERO-SHOT EVALUATION: Fine-tuned vs DA3Metric-Large')
    lines.append('=' * 70)
    lines.append(f'{"Metric":<12} {"Fine-tuned":>12} {"DA3-Large":>12} {"Diff":>10} {"Winner":>10}')
    lines.append('-' * 70)

    for key, label, lower_better in [
        ('mae', 'MAE (m)', True),
        ('rmse', 'RMSE (m)', True),
        ('abs_rel', 'AbsRel', True),
        ('delta_1', 'd<1.25', False),
        ('silog', 'SiLog', True),
    ]:
        ft_v = ft.get(key, 0)
        da3_v = da3.get(key, 0)
        diff = ft_v - da3_v

        if lower_better:
            winner = 'FT' if ft_v < da3_v else 'DA3'
        else:
            winner = 'FT' if ft_v > da3_v else 'DA3'

        lines.append(f'{label:<12} {ft_v:>12.4f} {da3_v:>12.4f} '
                     f'{diff:>+10.4f} {winner:>10}')

    lines.append('-' * 70)
    lines.append(f'{"N frames":<12} {ft.get("n_frames", 0):>12} '
                 f'{da3.get("n_frames", 0):>12}')

    # Per-band comparison
    lines.append('')
    lines.append('Per-Band AbsRel:')
    lines.append(f'  {"Band":<8} {"Fine-tuned":>12} {"DA3-Large":>12} {"Winner":>10}')

    for band_name, _, _ in DISTANCE_BANDS:
        key = f'{band_name}_abs_rel'
        ft_vals = [m[key] for m in finetuned_metrics if key in m]
        da3_vals = [m[key] for m in da3_metrics if key in m]
        if ft_vals and da3_vals:
            ft_v = np.mean(ft_vals)
            da3_v = np.mean(da3_vals)
            winner = 'FT' if ft_v < da3_v else 'DA3'
            lines.append(f'  {band_name:<8} {ft_v:>12.4f} {da3_v:>12.4f} '
                         f'{winner:>10}')

    text = '\n'.join(lines)
    print(text)

    # Save comparison
    comp_path = os.path.join(output_dir, 'comparison.txt')
    with open(comp_path, 'w') as f:
        f.write(text)
    print(f'\nComparison saved: {comp_path}')

    return ft, da3


def main():
    parser = argparse.ArgumentParser(
        description='Zero-shot evaluation on unseen environment')
    parser.add_argument('--checkpoint', required=True,
                        help='Path to fine-tuned model checkpoint (.pth)')
    parser.add_argument('--eval_session', required=True,
                        help='Path to evaluation session directory')
    parser.add_argument('--encoder', default='vits',
                        choices=['vits', 'vitb', 'vitl'])
    parser.add_argument('--input_size', type=int, default=518)
    parser.add_argument('--max_depth', type=float, default=6.0)
    parser.add_argument('--cameras', nargs='*',
                        default=['front', 'left', 'right'])
    parser.add_argument('--compare_da3', action='store_true',
                        help='Compare with DA3Metric-Large baseline '
                             '(requires batch_da3_inference.py output)')
    parser.add_argument('--output_dir', default=None,
                        help='Output directory (default: eval_session/zero_shot_eval)')
    args = parser.parse_args()

    if not os.path.isdir(args.eval_session):
        print(f'Session not found: {args.eval_session}')
        sys.exit(1)

    if args.output_dir is None:
        args.output_dir = os.path.join(args.eval_session, 'zero_shot_eval')
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'Eval session: {args.eval_session}')

    # Load fine-tuned model
    print(f'Loading checkpoint: {args.checkpoint}')
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    max_depth = ckpt.get('max_depth', args.max_depth)
    encoder = ckpt.get('args', {}).get('encoder', args.encoder)

    model = load_model(encoder, max_depth)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model = model.to(device).eval()

    param_count = sum(p.numel() for p in model.parameters())
    print(f'Model: {encoder}, {param_count / 1e6:.1f}M params, '
          f'max_depth={max_depth}m')

    # Evaluate fine-tuned model
    print(f'\nRunning inference on {args.cameras}...')
    ft_metrics = evaluate_model(
        model, args.eval_session, args.cameras,
        device, args.input_size, max_depth)

    if not ft_metrics:
        print('No valid frames found in evaluation session.')
        sys.exit(1)

    # Print fine-tuned results
    print(f'\nFine-tuned model results ({len(ft_metrics)} frames):')
    mae = np.mean([m['mae'] for m in ft_metrics])
    rmse = np.mean([m['rmse'] for m in ft_metrics])
    absrel = np.mean([m['abs_rel'] for m in ft_metrics])
    d1 = np.mean([m['delta_1'] for m in ft_metrics])
    silog = np.mean([m['silog'] for m in ft_metrics])
    print(f'  MAE={mae:.4f}m  RMSE={rmse:.4f}m  AbsRel={absrel:.4f}  '
          f'd<1.25={d1:.3f}  SiLog={silog:.4f}')

    # Per-camera
    for camera in args.cameras:
        cam_m = [m for m in ft_metrics if m['camera'] == camera]
        if cam_m:
            print(f'  [{camera}] {len(cam_m)} frames | '
                  f'MAE={np.mean([m["mae"] for m in cam_m]):.4f}  '
                  f'AbsRel={np.mean([m["abs_rel"] for m in cam_m]):.4f}  '
                  f'd<1.25={np.mean([m["delta_1"] for m in cam_m]):.3f}')

    # Save fine-tuned CSV
    csv_path = os.path.join(args.output_dir, 'finetuned_metrics.csv')
    fields = sorted(ft_metrics[0].keys())
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(ft_metrics)
    print(f'\nFine-tuned metrics saved: {csv_path}')

    # Compare with DA3Metric-Large
    if args.compare_da3:
        print('\nLoading DA3Metric-Large baseline...')
        da3_metrics = evaluate_da3_baseline(
            args.eval_session, args.cameras, max_depth)

        if not da3_metrics:
            print('No DA3 depth files found. Run batch_da3_inference.py first.')
        else:
            # Save DA3 CSV
            da3_csv = os.path.join(args.output_dir, 'da3_baseline_metrics.csv')
            da3_fields = sorted(da3_metrics[0].keys())
            with open(da3_csv, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=da3_fields)
                writer.writeheader()
                writer.writerows(da3_metrics)

            print_comparison(ft_metrics, da3_metrics, args.output_dir)

    print(f'\nAll outputs in: {args.output_dir}')


if __name__ == '__main__':
    main()
