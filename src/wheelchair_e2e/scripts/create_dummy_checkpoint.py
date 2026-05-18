#!/usr/bin/env python3
"""
Create a dummy KinoFlow v2 checkpoint for E2E pipeline testing.

Initializes a random ModularKinoFlowNet, runs one forward pass to verify
shapes, and saves a proper checkpoint file. The checkpoint can be loaded
by kinoflow_node.py and test_e2e_inference.py to verify the full pipeline
without any trained weights.

Usage:
    python create_dummy_checkpoint.py
    python create_dummy_checkpoint.py --output /path/to/dummy_model.pth
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from wheelchair_e2e.models.kinoflow_net import ModularKinoFlowNet
from wheelchair_e2e.models.goal_encoder import compute_goal_features


def main():
    parser = argparse.ArgumentParser(
        description='Create dummy KinoFlow v2 checkpoint')
    parser.add_argument(
        '--output', type=str,
        default=os.path.join(os.path.dirname(__file__), '..',
                             '..', '..', 'training_data', 'dummy_model.pth'),
        help='Output checkpoint path')
    args = parser.parse_args()

    # Resolve to absolute path
    output_path = os.path.abspath(args.output)

    # Architecture config (matches modular_e2e_params.yaml)
    config = {
        'scan_points': 720,
        'temporal_frames': 5,
        'horizon': 10,
        'n_samples': 8,
        'n_euler_steps': 3,
        'v_max': 0.25,
        'w_max': 1.0,
        'd_model': 128,
    }

    print("Creating ModularKinoFlowNet v2...")
    model = ModularKinoFlowNet(
        v_max=config['v_max'],
        w_max=config['w_max'],
        horizon=config['horizon'],
        n_euler_steps=config['n_euler_steps'],
        n_samples=config['n_samples'],
        scan_points=config['scan_points'],
        temporal_frames=config['temporal_frames'],
        d_model=config['d_model'],
    )
    model.eval()

    pc = model.get_param_count()
    print(f"Parameters: {pc['total']:,} ({pc['total']/1e6:.2f}M)")
    for name, count in pc.items():
        if name != 'total':
            print(f"  {name}: {count:,}")

    # Verify forward pass with dummy inputs
    print("\nVerifying forward pass...")
    B = 1
    n_residual = config['temporal_frames'] - 1
    scan_current = torch.randn(B, config['scan_points'])
    scan_residuals = torch.randn(B, n_residual, config['scan_points'])
    goal_feat = torch.tensor(
        compute_goal_features(3.0, 0.0), dtype=torch.float32
    ).unsqueeze(0)
    odom_history = torch.randn(B, 30)

    with torch.no_grad():
        cond, hidden = model.encode(
            scan_current, scan_residuals, goal_feat, odom_history)
    print(f"  encode() -> cond: {cond.shape}, hidden: {hidden.shape}")
    assert cond.shape == (B, 256), f"Expected (1, 256), got {cond.shape}"

    # Verify generate
    with torch.no_grad():
        vel_traj, poses, hidden = model.generate(
            scan_current, scan_residuals, goal_feat, odom_history)
    print(f"  generate() -> vel: {vel_traj.shape}, poses: {poses.shape}")
    assert vel_traj.shape == (B, config['horizon'], 2)
    assert poses.shape == (B, config['horizon'], 3)

    # Verify generate_multi_sample
    bev_occ = torch.zeros(1, 200, 200)
    with torch.no_grad():
        (best_vel, best_poses, all_vel, all_poses,
         best_idx, scores, hidden) = model.generate_multi_sample(
            scan_current, scan_residuals, goal_feat, odom_history,
            bev_occupancy=bev_occ, goal_dx=3.0, goal_dy=0.0)
    print(f"  generate_multi_sample() -> all_vel: {all_vel.shape}, "
          f"best_idx: {best_idx}, scores: {scores}")
    assert all_vel.shape == (config['n_samples'], config['horizon'], 2)
    assert all_poses.shape == (config['n_samples'], config['horizon'], 3)

    # Verify velocity bounds
    v_vals = all_vel[:, :, 0]
    w_vals = all_vel[:, :, 1]
    print(f"  v range: [{v_vals.min():.4f}, {v_vals.max():.4f}] "
          f"(max: {config['v_max']})")
    print(f"  w range: [{w_vals.min():.4f}, {w_vals.max():.4f}] "
          f"(max: {config['w_max']})")
    assert v_vals.min() >= 0.0, "v should be non-negative"
    assert v_vals.max() <= config['v_max'] + 1e-6
    assert w_vals.abs().max() <= config['w_max'] + 1e-6

    print("\nAll shape checks passed!")

    # Save checkpoint
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'epoch': 0,
        'phase': 1,
        'architecture': 'v2',
        'config': config,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(checkpoint, output_path)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\nSaved: {output_path} ({size_mb:.1f} MB)")


if __name__ == '__main__':
    main()
