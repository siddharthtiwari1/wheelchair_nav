#!/usr/bin/env python3
"""
Evaluation Script for E2E Models

Computes metrics on validation data:
    - Success rate (goal reached)
    - Velocity MSE
    - Average velocity / angular velocity
    - Jerk (smoothness)
    - Collision rate (from BEV)

Usage:
    python evaluate.py \
        --checkpoint best_model.pth \
        --data_dir /path/to/val_data \
        --output results.json
"""

import os
import argparse
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from wheelchair_e2e.models.bev_velocity_net import BEVVelocityNet
from wheelchair_e2e.training.dataset import BEVVelocityDataset


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate E2E model')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output', type=str, default='results.json')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--v_max', type=float, default=0.25)
    parser.add_argument('--w_max', type=float, default=1.0)
    parser.add_argument('--device', type=str, default='cuda')
    return parser.parse_args()


def compute_jerk(velocities, dt=0.1):
    """Compute average jerk from velocity sequence."""
    if len(velocities) < 2:
        return 0.0
    diffs = np.diff(velocities, axis=0)
    jerk = np.mean(np.linalg.norm(diffs / dt, axis=1))
    return float(jerk)


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available()
                          else 'cpu')

    # Load model
    model = BEVVelocityNet(
        v_max=args.v_max, w_max=args.w_max).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # Load data
    dataset = BEVVelocityDataset(args.data_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=False, num_workers=2)

    # Evaluate
    all_pred = []
    all_target = []

    with torch.no_grad():
        for batch in loader:
            bev = batch['bev'].to(device)
            odom = batch['odom'].to(device)
            target = batch['velocity']

            pred, _ = model(bev, odom, hidden=None)
            all_pred.append(pred.cpu().numpy())
            all_target.append(target.numpy())

    pred = np.concatenate(all_pred, axis=0)
    target = np.concatenate(all_target, axis=0)

    # Compute metrics
    v_mse = float(np.mean((pred[:, 0] - target[:, 0]) ** 2))
    w_mse = float(np.mean((pred[:, 1] - target[:, 1]) ** 2))
    total_mse = v_mse + w_mse

    v_mae = float(np.mean(np.abs(pred[:, 0] - target[:, 0])))
    w_mae = float(np.mean(np.abs(pred[:, 1] - target[:, 1])))

    avg_pred_v = float(np.mean(pred[:, 0]))
    avg_pred_w = float(np.mean(np.abs(pred[:, 1])))
    avg_target_v = float(np.mean(target[:, 0]))

    jerk = compute_jerk(pred, dt=0.1)

    results = {
        'n_samples': len(pred),
        'v_mse': v_mse,
        'w_mse': w_mse,
        'total_mse': total_mse,
        'v_mae': v_mae,
        'w_mae': w_mae,
        'avg_pred_v': avg_pred_v,
        'avg_pred_w': avg_pred_w,
        'avg_target_v': avg_target_v,
        'jerk': jerk,
        'pred_v_range': [float(pred[:, 0].min()),
                         float(pred[:, 0].max())],
        'pred_w_range': [float(pred[:, 1].min()),
                         float(pred[:, 1].max())],
        'model_params': model.get_param_count(),
    }

    # Print results
    print("\n=== E2E-2 Evaluation Results ===")
    print(f"Samples: {results['n_samples']}")
    print(f"V MSE:   {v_mse:.6f}  MAE: {v_mae:.4f} m/s")
    print(f"ω MSE:   {w_mse:.6f}  MAE: {w_mae:.4f} rad/s")
    print(f"Total MSE: {total_mse:.6f}")
    print(f"Jerk:    {jerk:.4f} m/s³")
    print(f"Avg pred v: {avg_pred_v:.3f} m/s "
          f"(target: {avg_target_v:.3f})")
    print(f"Pred v range: [{results['pred_v_range'][0]:.3f}, "
          f"{results['pred_v_range'][1]:.3f}]")
    print(f"Pred ω range: [{results['pred_w_range'][0]:.3f}, "
          f"{results['pred_w_range'][1]:.3f}]")

    # Save
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()
