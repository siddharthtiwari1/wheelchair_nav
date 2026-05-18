#!/usr/bin/env python3
"""Visualize monocular depth predictions vs ground truth.

Produces side-by-side comparison images: RGB | GT Depth | Predicted Depth | Error Map.

Usage:
    python3 visualize_predictions.py \
        --checkpoint /path/to/best_model.pth \
        --data_dir /path/to/mono_depth_data \
        --encoder vits \
        --output_dir viz_output/ \
        --num_samples 20
"""

import argparse
import os
import sys

import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from wheelchair_mono_depth.training.train import load_model
from wheelchair_mono_depth.training.dataset import WheelchairDepthDataset, get_session_split
from wheelchair_mono_depth.training.transforms import ValTransform
from wheelchair_mono_depth.training.metrics import compute_depth_metrics


def colorize_depth(depth, vmin=0.0, vmax=10.0):
    """Convert depth map to colorized image using turbo colormap."""
    depth_norm = np.clip((depth - vmin) / (vmax - vmin), 0, 1)
    colored = plt.cm.turbo(depth_norm)[:, :, :3]
    return (colored * 255).astype(np.uint8)


def visualize_sample(rgb, depth_gt, depth_pred, max_depth, save_path):
    """Create 4-panel visualization."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # RGB
    axes[0].imshow(rgb)
    axes[0].set_title('RGB Input')
    axes[0].axis('off')

    # GT Depth
    axes[1].imshow(colorize_depth(depth_gt, vmax=max_depth))
    axes[1].set_title('Ground Truth Depth')
    axes[1].axis('off')

    # Predicted Depth
    axes[2].imshow(colorize_depth(depth_pred, vmax=max_depth))
    axes[2].set_title('Predicted Depth')
    axes[2].axis('off')

    # Error map
    valid = depth_gt > 0
    error = np.zeros_like(depth_gt)
    error[valid] = np.abs(depth_pred[valid] - depth_gt[valid])
    axes[3].imshow(error, cmap='hot', vmin=0, vmax=1.0)
    axes[3].set_title('Absolute Error (m)')
    axes[3].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize depth predictions')
    parser.add_argument('--checkpoint', required=True, help='Path to .pth checkpoint')
    parser.add_argument('--data_dir', required=True, help='Data root directory')
    parser.add_argument('--encoder', default='vits', choices=['vits', 'vitb', 'vitl'])
    parser.add_argument('--max_depth', type=float, default=10.0)
    parser.add_argument('--input_size', type=int, default=518)
    parser.add_argument('--output_dir', default='viz_output')
    parser.add_argument('--num_samples', type=int, default=20)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    max_depth = ckpt.get('max_depth', args.max_depth)
    model = load_model(args.encoder, max_depth)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model = model.to(device).eval()

    # Load validation data
    _, val_sessions = get_session_split(args.data_dir)
    transform = ValTransform(size=args.input_size, max_depth=max_depth)
    dataset = WheelchairDepthDataset(val_sessions, transform=transform, max_depth=max_depth)

    if len(dataset) == 0:
        print('No validation data found')
        return

    # Sample evenly
    indices = np.linspace(0, len(dataset) - 1, min(args.num_samples, len(dataset)), dtype=int)

    print(f'Generating {len(indices)} visualizations...')

    for i, idx in enumerate(indices):
        rgb_tensor, depth_gt_tensor, valid_mask = dataset[idx]

        with torch.no_grad():
            rgb_input = rgb_tensor.unsqueeze(0).to(device)
            depth_pred = model(rgb_input)
            if depth_pred.dim() == 3:
                depth_pred = depth_pred.unsqueeze(1)

        # Compute per-sample metrics
        metrics = compute_depth_metrics(
            depth_pred.cpu(), depth_gt_tensor.unsqueeze(0), valid_mask.unsqueeze(0)
        )

        # Convert for visualization
        from wheelchair_mono_depth.training.transforms import denormalize
        rgb_vis = denormalize(rgb_tensor).permute(1, 2, 0).clamp(0, 1).numpy()
        depth_gt_vis = depth_gt_tensor.squeeze().numpy()
        depth_pred_vis = depth_pred.squeeze().cpu().numpy()

        save_path = os.path.join(args.output_dir, f'sample_{i:03d}.png')
        visualize_sample(rgb_vis, depth_gt_vis, depth_pred_vis, max_depth, save_path)

        print(f'  [{i+1}/{len(indices)}] AbsRel={metrics["abs_rel"]:.4f} '
              f'd<1.25={metrics["delta_1"]:.3f} → {save_path}')

    print(f'Visualizations saved to {args.output_dir}/')


if __name__ == '__main__':
    main()
