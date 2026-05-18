#!/usr/bin/env python3
"""
Validate Cable Tracer v3 — offline evaluation with temporal + red mask visualization.

Loads a trained model and data, runs inference on every sample,
and generates scatter plots, trajectory comparison, and sample visualizations.

Usage:
    python3 validate_cable_tracer.py \
        --model models/cable_tracer/cable_tracer.pt \
        --data data/rgb_vel_20260319_162633 data/rgb_vel_20260319_162910 \
        --output models/cable_tracer/validation_v3.png
"""

import argparse
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent))
from train_cable_tracer import CableTracerCNN, CableTracerDataset, extract_red_mask


def validate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load config
    config_path = Path(args.model).parent / 'cable_tracer_config.json'
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        v_max = config['v_max']
        omega_max = config['omega_max']
        in_channels = config.get('in_channels', 12)
    else:
        v_max = args.v_max
        omega_max = args.omega_max
        in_channels = 12

    # Load model
    model = CableTracerCNN(in_channels=in_channels).to(device)
    model.load_state_dict(
        torch.load(args.model, map_location=device, weights_only=True))
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Model: {n_params:,} params, {in_channels}ch input')

    # Load dataset
    dataset = CableTracerDataset(
        data_dirs=args.data, v_max=v_max, omega_max=omega_max)

    # Run inference
    actual_v, actual_w = [], []
    pred_v, pred_w, pred_conf = [], [], []
    actual_cable_vis = []

    with torch.no_grad():
        for i in range(len(dataset)):
            img, vel, cable_vis = dataset[i]
            pred = model(img.unsqueeze(0).to(device)).cpu().numpy()[0]
            actual_v.append(vel[0].item())
            actual_w.append(vel[1].item())
            pred_v.append(pred[0])
            pred_w.append(pred[1])
            pred_conf.append(pred[2] if len(pred) > 2 else 1.0)
            actual_cable_vis.append(float(cable_vis))

    actual_v = np.array(actual_v)
    actual_w = np.array(actual_w)
    pred_v = np.array(pred_v)
    pred_w = np.array(pred_w)
    pred_conf = np.array(pred_conf)
    actual_cable_vis = np.array(actual_cable_vis)

    # Metrics
    mse_v = np.mean((pred_v - actual_v) ** 2)
    mse_w = np.mean((pred_w - actual_w) ** 2)
    mse_total = (mse_v + mse_w) / 2
    mae_v = np.mean(np.abs(pred_v - actual_v))
    mae_w = np.mean(np.abs(pred_w - actual_w))
    max_err_v = np.max(np.abs(pred_v - actual_v))
    max_err_w = np.max(np.abs(pred_w - actual_w))

    # Direction match (exclude near-zero omega to avoid noise)
    significant = np.abs(actual_w) > 0.05
    if significant.sum() > 0:
        dir_match_sig = np.mean(
            np.sign(pred_w[significant]) == np.sign(actual_w[significant])) * 100
    else:
        dir_match_sig = 0.0
    dir_match_all = np.mean(np.sign(pred_w) == np.sign(actual_w)) * 100

    # R-squared
    ss_res_v = np.sum((pred_v - actual_v) ** 2)
    ss_tot_v = np.sum((actual_v - actual_v.mean()) ** 2)
    r2_v = 1 - ss_res_v / max(ss_tot_v, 1e-8)
    ss_res_w = np.sum((pred_w - actual_w) ** 2)
    ss_tot_w = np.sum((actual_w - actual_w.mean()) ** 2)
    r2_w = 1 - ss_res_w / max(ss_tot_w, 1e-8)

    # Confidence metrics
    n_pos = int(actual_cable_vis.sum())
    n_neg = len(actual_cable_vis) - n_pos
    pred_cable = pred_conf > 0.5
    actual_cable_bool = actual_cable_vis > 0.5
    conf_acc = np.mean(pred_cable == actual_cable_bool) * 100 if len(pred_conf) > 0 else 0.0

    # Precision/recall for cable detection
    tp = np.sum(pred_cable & actual_cable_bool)
    fp = np.sum(pred_cable & ~actual_cable_bool)
    fn = np.sum(~pred_cable & actual_cable_bool)
    tn = np.sum(~pred_cable & ~actual_cable_bool)
    precision = tp / max(tp + fp, 1) * 100
    recall = tp / max(tp + fn, 1) * 100

    # Mean confidence by class
    conf_pos = pred_conf[actual_cable_bool].mean() if n_pos > 0 else 0.0
    conf_neg = pred_conf[~actual_cable_bool].mean() if n_neg > 0 else 0.0

    print(f'\nSamples: {len(dataset)} ({n_pos} cable, {n_neg} negative)')
    print(f'MSE   — v: {mse_v:.5f}, omega: {mse_w:.5f}, total: {mse_total:.5f}')
    print(f'MAE   — v: {mae_v:.4f}, omega: {mae_w:.4f}')
    print(f'Max   — v: {max_err_v:.4f}, omega: {max_err_w:.4f}')
    print(f'R²    — v: {r2_v:.4f}, omega: {r2_w:.4f}')
    print(f'Dir match — all: {dir_match_all:.1f}%, '
          f'significant (|w|>0.05): {dir_match_sig:.1f}%')
    print(f'Confidence — accuracy: {conf_acc:.1f}%, '
          f'precision: {precision:.1f}%, recall: {recall:.1f}%')
    print(f'  Mean conf (cable): {conf_pos:.3f}, Mean conf (negative): {conf_neg:.3f}')
    if n_neg > 0:
        neg_zero = np.mean(pred_conf[~actual_cable_bool] < 0.1) * 100
        print(f'  Negatives with conf<0.1: {neg_zero:.1f}%')

    # --- Plots ---
    fig, axes = plt.subplots(3, 3, figsize=(18, 16))

    # Row 1: scatter + trajectory
    # 1) Linear velocity scatter
    ax = axes[0, 0]
    ax.scatter(actual_v, pred_v, alpha=0.4, s=8, c='tab:blue')
    lim = max(abs(actual_v).max(), abs(pred_v).max()) * 1.1
    ax.plot([-lim, lim], [-lim, lim], 'k--', linewidth=0.8)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel('Actual v (norm)'); ax.set_ylabel('Predicted v (norm)')
    ax.set_title(f'Linear Velocity (MSE={mse_v:.4f}, R²={r2_v:.3f})')
    ax.set_aspect('equal'); ax.grid(True, alpha=0.3)

    # 2) Angular velocity scatter
    ax = axes[0, 1]
    ax.scatter(actual_w, pred_w, alpha=0.4, s=8, c='tab:orange')
    lim = max(abs(actual_w).max(), abs(pred_w).max()) * 1.1
    ax.plot([-lim, lim], [-lim, lim], 'k--', linewidth=0.8)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel('Actual omega (norm)'); ax.set_ylabel('Predicted omega (norm)')
    ax.set_title(f'Angular Velocity (MSE={mse_w:.4f}, dir={dir_match_sig:.0f}%)')
    ax.set_aspect('equal'); ax.grid(True, alpha=0.3)

    # 3) Trajectory comparison
    dt = 0.1
    ax = axes[0, 2]
    for label, vv, ww, color in [
        ('Actual', actual_v * v_max, actual_w * omega_max, 'tab:green'),
        ('Predicted', pred_v * v_max, pred_w * omega_max, 'tab:red'),
    ]:
        x, y, theta = 0.0, 0.0, 0.0
        xs, ys = [0.0], [0.0]
        for v, w in zip(vv, ww):
            theta += w * dt
            x += v * np.cos(theta) * dt
            y += v * np.sin(theta) * dt
            xs.append(x); ys.append(y)
        ax.plot(xs, ys, color=color, label=label, linewidth=1.5, alpha=0.8)
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
    ax.set_title('Integrated Trajectory')
    ax.legend(); ax.set_aspect('equal'); ax.grid(True, alpha=0.3)

    # Row 2: time series + error distribution + sample red masks
    # 4) Time series of omega (first 200 frames)
    ax = axes[1, 0]
    n_show = min(200, len(actual_w))
    t = np.arange(n_show) * dt
    ax.plot(t, actual_w[:n_show], 'tab:green', alpha=0.7, label='Actual', linewidth=1)
    ax.plot(t, pred_w[:n_show], 'tab:red', alpha=0.7, label='Predicted', linewidth=1)
    ax.set_xlabel('Time (s)'); ax.set_ylabel('omega (norm)')
    ax.set_title('Angular Velocity Time Series (first 200)')
    ax.legend(); ax.grid(True, alpha=0.3)

    # 5) Error distribution
    ax = axes[1, 1]
    err_w = pred_w - actual_w
    ax.hist(err_w, bins=50, alpha=0.7, color='tab:purple', edgecolor='white')
    ax.axvline(0, color='k', linestyle='--', linewidth=0.8)
    ax.set_xlabel('Omega prediction error'); ax.set_ylabel('Count')
    ax.set_title(f'Error Distribution (mean={err_w.mean():.4f}, std={err_w.std():.4f})')
    ax.grid(True, alpha=0.3)

    # 6) Sample red mask visualization
    ax = axes[1, 2]
    sample_indices = np.linspace(0, len(dataset) - 1, 6, dtype=int)
    vis_imgs = []
    for si in sample_indices:
        gidx = dataset.samples[si][0]
        img_path = dataset.all_image_paths[gidx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (64, 64))
        img_f = img.astype(np.float32) / 255.0
        mask = extract_red_mask(img_f)
        # Overlay: red mask in magenta on original
        overlay = img_f.copy()
        overlay[mask > 0.5] = [1.0, 0.0, 1.0]
        blended = 0.6 * img_f + 0.4 * overlay
        vis_imgs.append(np.clip(blended, 0, 1))
    # 2x3 grid
    row1 = np.concatenate(vis_imgs[:3], axis=1)
    row2 = np.concatenate(vis_imgs[3:], axis=1)
    grid = np.concatenate([row1, row2], axis=0)
    ax.imshow(grid)
    ax.set_title('Red Mask Detection (magenta overlay)')
    ax.axis('off')

    # Row 3: confidence metrics
    # 7) Confidence distribution by class
    ax = axes[2, 0]
    if n_pos > 0:
        ax.hist(pred_conf[actual_cable_bool], bins=30, alpha=0.7,
                color='tab:green', label=f'Cable ({n_pos})', edgecolor='white')
    if n_neg > 0:
        ax.hist(pred_conf[~actual_cable_bool], bins=30, alpha=0.7,
                color='tab:red', label=f'Negative ({n_neg})', edgecolor='white')
    ax.axvline(0.5, color='k', linestyle='--', linewidth=1, label='Threshold')
    ax.set_xlabel('Confidence'); ax.set_ylabel('Count')
    ax.set_title('Confidence Distribution (positive vs negative)')
    ax.legend(); ax.grid(True, alpha=0.3)

    # 8) Confidence vs velocity magnitude (cable samples only)
    ax = axes[2, 1]
    if n_pos > 0:
        vel_mag = np.sqrt(actual_v[actual_cable_bool]**2 + actual_w[actual_cable_bool]**2)
        ax.scatter(vel_mag, pred_conf[actual_cable_bool], alpha=0.4, s=8, c='tab:blue')
        ax.set_xlabel('Velocity magnitude (norm)'); ax.set_ylabel('Confidence')
        ax.set_title('Confidence vs Velocity (cable samples)')
    else:
        ax.text(0.5, 0.5, 'No cable samples', ha='center', va='center', transform=ax.transAxes)
    ax.grid(True, alpha=0.3)

    # 9) Gated vs ungated velocity on negatives
    ax = axes[2, 2]
    if n_neg > 0:
        neg_v = np.abs(pred_v[~actual_cable_bool] * v_max)
        neg_w = np.abs(pred_w[~actual_cable_bool] * omega_max)
        gated_v = neg_v * (pred_conf[~actual_cable_bool] >= 0.3)
        gated_w = neg_w * (pred_conf[~actual_cable_bool] >= 0.3)
        x_idx = np.arange(min(50, n_neg))
        ax.bar(x_idx - 0.2, neg_v[:len(x_idx)], 0.4, alpha=0.6,
               color='tab:red', label='Ungated |v|')
        ax.bar(x_idx + 0.2, gated_v[:len(x_idx)], 0.4, alpha=0.6,
               color='tab:green', label='Gated |v|')
        ax.set_xlabel('Negative sample index'); ax.set_ylabel('|v| (m/s)')
        ax.set_title(f'Gated vs Ungated on Negatives (first {len(x_idx)})')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'No negative samples', ha='center', va='center',
                transform=ax.transAxes)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f'Cable Tracer v4 Validation — {len(dataset)} samples '
        f'({n_pos} cable, {n_neg} neg), '
        f'{sum(p.numel() for p in model.parameters()):,} params',
        fontsize=13)
    fig.tight_layout()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150)
    print(f'\nValidation plot saved to {out_path}')


def main():
    parser = argparse.ArgumentParser(description='Validate Cable Tracer v3')
    parser.add_argument('--model', required=True, help='Path to cable_tracer.pt')
    parser.add_argument('--data', nargs='+', required=True)
    parser.add_argument('--output', default='models/cable_tracer/validation_v3.png')
    parser.add_argument('--v_max', type=float, default=0.50)
    parser.add_argument('--omega_max', type=float, default=0.15)
    args = parser.parse_args()
    validate(args)


if __name__ == '__main__':
    main()
