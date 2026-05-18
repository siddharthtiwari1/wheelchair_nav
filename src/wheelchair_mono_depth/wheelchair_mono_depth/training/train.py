#!/usr/bin/env python3
"""Fine-tuning script for Depth Anything V2 on wheelchair depth data.

Loads a pretrained Depth Anything V2 metric-indoor model and fine-tunes it
on paired RGB-depth data collected from the wheelchair's RealSense cameras.

Supports edge-aware smoothness loss and camera-aware confidence weighting
(D435i pixels beyond 2m are down-weighted due to worse depth noise).

Usage:
    python -m wheelchair_mono_depth.training.train \
        --data_dir /home/sidd/wheelchair_nav/mono_depth_data \
        --output_dir /home/sidd/wheelchair_nav/checkpoints/mono_depth \
        --encoder vits \
        --max_depth 6.0 \
        --epochs 50 \
        --batch_size 8 \
        --lr 5e-5 \
        --lambda_smooth 0.05

Requirements:
    pip install torch torchvision depth-anything-v2
"""

import argparse
import os
import time
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

from wheelchair_mono_depth.training.dataset import (
    WheelchairDepthDataset, get_session_split
)
from wheelchair_mono_depth.training.transforms import (
    TrainTransform, ValTransform, denormalize
)
from wheelchair_mono_depth.training.losses import DepthEstimationLoss
from wheelchair_mono_depth.training.metrics import compute_depth_metrics, MetricTracker


# Depth Anything V2 model configs (encoder -> features, out_channels)
MODEL_CONFIGS = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
}


def load_model(encoder='vits', max_depth=10.0, pretrained_path=None):
    """Load Depth Anything V2 metric depth model.

    Tries to load from the depth_anything_v2 package. If not installed,
    falls back to loading from a local checkpoint.
    """
    config = MODEL_CONFIGS[encoder]

    try:
        from depth_anything_v2.dpt import DepthAnythingV2
        model = DepthAnythingV2(
            encoder=config['encoder'],
            features=config['features'],
            out_channels=config['out_channels'],
            max_depth=max_depth,
        )
    except ImportError:
        raise ImportError(
            'depth_anything_v2 not installed. Install with:\n'
            '  pip install depth-anything-v2\n'
            'Or clone: https://github.com/DepthAnything/Depth-Anything-V2'
        )

    if pretrained_path and os.path.exists(pretrained_path):
        state_dict = torch.load(pretrained_path, map_location='cpu', weights_only=True)
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        model.load_state_dict(state_dict, strict=False)
        print(f'Loaded pretrained weights from {pretrained_path}')

    return model


def create_optimizer(model, lr=5e-5, encoder_lr_scale=0.1, weight_decay=0.01):
    """Create AdamW optimizer with differentiated LR for encoder vs decoder."""
    encoder_params = []
    decoder_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'pretrained' in name:
            encoder_params.append(param)
        else:
            decoder_params.append(param)

    param_groups = [
        {'params': encoder_params, 'lr': lr * encoder_lr_scale},  # encoder: low LR
        {'params': decoder_params, 'lr': lr},                      # decoder: full LR
    ]

    print(f'Optimizer: encoder LR={lr * encoder_lr_scale:.1e} '
          f'({len(encoder_params)} params), '
          f'decoder LR={lr:.1e} ({len(decoder_params)} params)')

    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)


def train_epoch(model, loader, criterion, optimizer, scaler, device,
                use_smoothness=False):
    model.train()
    tracker = MetricTracker()
    total_loss = 0.0

    for batch_idx, (rgb, depth_gt, valid_mask, camera_weights) in enumerate(loader):
        rgb = rgb.to(device)
        depth_gt = depth_gt.to(device)
        valid_mask = valid_mask.to(device)
        camera_weights = camera_weights.to(device)

        optimizer.zero_grad()

        with autocast(device_type='cuda', enabled=scaler.is_enabled()):
            depth_pred = model(rgb)
            if depth_pred.dim() == 3:
                depth_pred = depth_pred.unsqueeze(1)

            # Pass RGB for edge-aware smoothness, camera_weights for confidence
            loss_kwargs = {'camera_weights': camera_weights}
            if use_smoothness:
                # Denormalize RGB for edge detection (gradients on normalized
                # images still work but raw 0-1 range is better)
                rgb_denorm = denormalize(rgb)
                loss_kwargs['rgb'] = rgb_denorm
            loss = criterion(depth_pred, depth_gt, valid_mask, **loss_kwargs)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        with torch.no_grad():
            metrics = compute_depth_metrics(depth_pred, depth_gt, valid_mask)
            tracker.update(metrics)

        if (batch_idx + 1) % 20 == 0:
            print(f'  Batch {batch_idx + 1}/{len(loader)} | '
                  f'Loss={loss.item():.4f} | {tracker.summary_str()}')

    return total_loss / max(len(loader), 1), tracker.compute()


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    tracker = MetricTracker()
    total_loss = 0.0

    for rgb, depth_gt, valid_mask, camera_weights in loader:
        rgb = rgb.to(device)
        depth_gt = depth_gt.to(device)
        valid_mask = valid_mask.to(device)
        camera_weights = camera_weights.to(device)

        depth_pred = model(rgb)
        if depth_pred.dim() == 3:
            depth_pred = depth_pred.unsqueeze(1)
        loss = criterion(depth_pred, depth_gt, valid_mask,
                         camera_weights=camera_weights)
        total_loss += loss.item()

        metrics = compute_depth_metrics(depth_pred, depth_gt, valid_mask)
        tracker.update(metrics)

    return total_loss / max(len(loader), 1), tracker.compute()


def main():
    parser = argparse.ArgumentParser(description='Fine-tune Depth Anything V2')
    parser.add_argument('--data_dir', required=True,
                        help='Root directory with collected session data')
    parser.add_argument('--output_dir', default='checkpoints/mono_depth',
                        help='Directory to save checkpoints')
    parser.add_argument('--pretrained', default=None,
                        help='Path to pretrained .pth weights')
    parser.add_argument('--encoder', default='vits', choices=['vits', 'vitb', 'vitl'],
                        help='ViT encoder size (default: vits for Jetson)')
    parser.add_argument('--max_depth', type=float, default=6.0,
                        help='Maximum depth in meters (wheelchair range)')
    parser.add_argument('--input_size', type=int, default=518,
                        help='Input resolution (must be multiple of 14)')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--encoder_lr_scale', type=float, default=0.1,
                        help='LR multiplier for pretrained encoder')
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--lambda_silog', type=float, default=1.0)
    parser.add_argument('--lambda_l1', type=float, default=0.1)
    parser.add_argument('--lambda_smooth', type=float, default=0.0,
                        help='Edge-aware smoothness loss weight (0.05 recommended)')
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--use_amp', action='store_true', default=True,
                        help='Use mixed precision training')
    parser.add_argument('--resume', default=None,
                        help='Path to checkpoint to resume from')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Save training config
    with open(os.path.join(args.output_dir, 'train_config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Data
    train_sessions, val_sessions = get_session_split(args.data_dir, args.val_ratio)

    train_transform = TrainTransform(size=args.input_size, max_depth=args.max_depth)
    val_transform = ValTransform(size=args.input_size, max_depth=args.max_depth)

    train_dataset = WheelchairDepthDataset(train_sessions, transform=train_transform,
                                           max_depth=args.max_depth)
    val_dataset = WheelchairDepthDataset(val_sessions, transform=val_transform,
                                         max_depth=args.max_depth)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    print(f'Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples')

    # Model
    model = load_model(args.encoder, args.max_depth, args.pretrained)
    model = model.to(device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model: {args.encoder}, {param_count / 1e6:.1f}M trainable parameters')

    # Optimizer + scheduler
    optimizer = create_optimizer(model, args.lr, args.encoder_lr_scale, args.weight_decay)

    warmup_epochs = max(1, args.epochs // 60)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs - warmup_epochs, eta_min=1e-7
    )

    criterion = DepthEstimationLoss(
        lambda_silog=args.lambda_silog,
        lambda_l1=args.lambda_l1,
        lambda_smooth=args.lambda_smooth,
    )
    use_smoothness = args.lambda_smooth > 0
    if use_smoothness:
        print(f'Edge-aware smoothness loss enabled (lambda={args.lambda_smooth})')

    scaler = GradScaler(enabled=args.use_amp)

    # Resume
    start_epoch = 0
    best_abs_rel = float('inf')

    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_abs_rel = ckpt.get('best_val_absrel', float('inf'))
        print(f'Resumed from epoch {start_epoch}, best AbsRel={best_abs_rel:.4f}')

    # Training loop
    target_lrs = [pg['lr'] for pg in optimizer.param_groups]
    print(f'\nStarting training for {args.epochs} epochs...\n')

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        # Warmup: linear LR ramp from stored targets
        if epoch < warmup_epochs:
            warmup_factor = (epoch + 1) / warmup_epochs
            for pg, target_lr in zip(optimizer.param_groups, target_lrs):
                pg['lr'] = target_lr * warmup_factor

        train_loss, train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            use_smoothness=use_smoothness,
        )

        val_loss, val_metrics = validate(model, val_loader, criterion, device)

        if epoch >= warmup_epochs:
            scheduler.step()

        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[1]['lr']  # decoder LR

        print(f'Epoch {epoch + 1}/{args.epochs} ({elapsed:.0f}s) | '
              f'LR={current_lr:.1e}')
        print(f'  Train: loss={train_loss:.4f} | '
              f"AbsRel={train_metrics.get('abs_rel', 0):.4f} | "
              f"d<1.25={train_metrics.get('delta_1', 0):.3f}")
        print(f'  Val:   loss={val_loss:.4f} | '
              f"AbsRel={val_metrics.get('abs_rel', 0):.4f} | "
              f"d<1.25={val_metrics.get('delta_1', 0):.3f} | "
              f"RMSE={val_metrics.get('rmse', 0):.3f}")

        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_absrel': best_abs_rel,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'args': vars(args),
            'model_config': MODEL_CONFIGS[args.encoder],
            'max_depth': args.max_depth,
        }

        # Save latest
        torch.save(checkpoint, os.path.join(args.output_dir, 'latest.pth'))

        # Save best
        val_abs_rel = val_metrics.get('abs_rel', float('inf'))
        if val_abs_rel < best_abs_rel:
            best_abs_rel = val_abs_rel
            checkpoint['best_val_absrel'] = best_abs_rel
            torch.save(checkpoint, os.path.join(args.output_dir, 'best_model.pth'))
            print(f'  >> New best model! AbsRel={best_abs_rel:.4f}')

        # Save periodic
        if (epoch + 1) % 10 == 0:
            torch.save(checkpoint,
                       os.path.join(args.output_dir, f'epoch_{epoch + 1}.pth'))

        print()

    print(f'Training complete. Best val AbsRel: {best_abs_rel:.4f}')
    print(f'Checkpoints saved to: {args.output_dir}')


if __name__ == '__main__':
    main()
